#!/usr/bin/env python3
"""Static double-buffer execution model for block-major N-outer MoE.

This module deliberately does not import the resident M-outer/four-stage model.
One dynamic group descriptor is consumed by two fixed workers per cluster:

* a DMA producer walks ``(phase, block, expert-sequence-index)``;
* a compute consumer walks the same stream and processes every token tile of
  the current expert while the weight block remains in one ping/pong slot.

The simulator models two global 64 B/cc DMA lanes.  A transfer may occupy one
lane or both lanes, is non-preemptive, and must finish before its compute item.
The two-entry buffer contract is exact: load ``k`` may not overwrite the slot
still consumed by compute ``k-2``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence


LANE_BW_BYTES_PER_CC = 64


class DmaPolicy(str, Enum):
    """Policies implemented by the static DMA grant worker."""

    DEADLINE_AWARE = "deadline_aware"
    SINGLE_ONLY = "single_only"


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    block_count: int
    weight_block_bytes: int
    m4_compute_cc: int
    m2_compute_cc: int

    def __post_init__(self) -> None:
        if self.block_count <= 0:
            raise ValueError(f"{self.name}: block_count must be positive")
        if self.weight_block_bytes <= 0:
            raise ValueError(f"{self.name}: weight_block_bytes must be positive")
        if self.m4_compute_cc <= 0 or self.m2_compute_cc <= 0:
            raise ValueError(f"{self.name}: compute times must be positive")


@dataclass(frozen=True)
class NOuterConfig:
    phases: tuple[PhaseSpec, ...]
    dma_policy: DmaPolicy = DmaPolicy.DEADLINE_AWARE
    force_initial_split: bool = True
    max_group_tokens_per_cluster: int | None = None

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("at least one phase is required")
        if (
            self.max_group_tokens_per_cluster is not None
            and self.max_group_tokens_per_cluster <= 0
        ):
            raise ValueError("max_group_tokens_per_cluster must be positive")


def default_config() -> NOuterConfig:
    """Full 2048->1408->2048 model with eight phase-specific blocks.

    Gate/Up uses 176 output columns per block so 1408 columns form eight
    blocks.  Down uses 128 columns per VersaCore and block, so the two cores
    jointly cover 256 of the 2048 output columns in each of eight blocks.
    INT4 weights are assumed.  M4 and M2 compute rates are the ideal 64 and
    128 B/cc shapes used by the scheduling abstraction.
    """

    gu_block_bytes = 2 * 2048 * 176 // 2
    down_block_bytes = 1408 * 256 // 2
    return NOuterConfig(
        phases=(
            PhaseSpec(
                name="gate_up",
                block_count=8,
                weight_block_bytes=gu_block_bytes,
                m4_compute_cc=math.ceil(gu_block_bytes / 64),
                m2_compute_cc=math.ceil(gu_block_bytes / 128),
            ),
            PhaseSpec(
                name="down",
                block_count=8,
                weight_block_bytes=down_block_bytes,
                m4_compute_cc=math.ceil(down_block_bytes / 64),
                m2_compute_cc=math.ceil(down_block_bytes / 128),
            ),
        )
    )


@dataclass(frozen=True)
class ExpertDescriptor:
    eid: int
    ntokens: int
    token_ref_start: int = 0
    split_token_start: int = 0

    def __post_init__(self) -> None:
        if self.eid < 0:
            raise ValueError("expert id must be non-negative")
        if self.ntokens <= 0:
            raise ValueError("expert token count must be positive")
        if self.token_ref_start < 0 or self.split_token_start < 0:
            raise ValueError("token offsets must be non-negative")

    @property
    def m4_iters(self) -> int:
        quotient, remainder = divmod(self.ntokens, 4)
        return quotient + int(remainder == 3)

    @property
    def m2_iters(self) -> int:
        return int(self.ntokens % 4 in (1, 2))

    @property
    def m4_tail_valid_tokens(self) -> int:
        return 3 if self.ntokens % 4 == 3 else 0

    @property
    def m2_valid_tokens(self) -> int:
        remainder = self.ntokens % 4
        return remainder if remainder in (1, 2) else 0

    def compute_cc(self, phase: PhaseSpec) -> int:
        return self.m4_iters * phase.m4_compute_cc + self.m2_iters * phase.m2_compute_cc


@dataclass(frozen=True)
class GroupDescriptor:
    cluster0: tuple[ExpertDescriptor, ...]
    cluster1: tuple[ExpertDescriptor, ...]
    group_id: int = 0

    def __post_init__(self) -> None:
        if not self.cluster0 and not self.cluster1:
            raise ValueError("a group must contain at least one expert")
        ids = [expert.eid for expert in (*self.cluster0, *self.cluster1)]
        if len(ids) != len(set(ids)):
            raise ValueError("an unsplit group may contain each expert id only once")

    def experts(self, cluster: int) -> tuple[ExpertDescriptor, ...]:
        if cluster == 0:
            return self.cluster0
        if cluster == 1:
            return self.cluster1
        raise ValueError(f"invalid cluster {cluster}")


@dataclass(frozen=True)
class ScheduleCandidate:
    group: GroupDescriptor
    label: str = ""


@dataclass(frozen=True)
class WorkItem:
    cluster: int
    stream_index: int
    phase_index: int
    phase_name: str
    block_id: int
    sequence_index: int
    expert: ExpertDescriptor
    weight_bytes: int
    compute_cc: int

    @property
    def buffer_slot(self) -> int:
        return self.stream_index & 1

    @property
    def key(self) -> tuple[int, int]:
        return (self.cluster, self.stream_index)


@dataclass(frozen=True)
class LoadRecord:
    item: WorkItem
    start_cc: int
    end_cc: int
    lanes: tuple[int, ...]


@dataclass(frozen=True)
class ComputeRecord:
    item: WorkItem
    start_cc: int
    end_cc: int


@dataclass(frozen=True)
class CandidateResult:
    candidate: ScheduleCandidate
    makespan_cc: int
    lower_bound_cc: int
    compute_lower_bound_cc: int
    dma_lower_bound_cc: int
    initial_wait_cc: tuple[int, int]
    steady_stall_cc: tuple[int, int]
    compute_utilization: float
    dma_lane_utilization: float
    loads: tuple[LoadRecord, ...]
    computes: tuple[ComputeRecord, ...]
    history_validated: bool

    @property
    def overhead_cc(self) -> int:
        return self.makespan_cc - self.lower_bound_cc

    @property
    def over_lower_bound(self) -> float:
        if self.lower_bound_cc == 0:
            return 1.0
        return self.makespan_cc / self.lower_bound_cc

    @property
    def rank_key(self) -> tuple[int, int, int, str]:
        return (
            self.makespan_cc,
            sum(self.steady_stall_cc),
            self.overhead_cc,
            self.candidate.label,
        )


@dataclass
class _ClusterState:
    items: tuple[WorkItem, ...]
    next_load: int = 0
    next_compute: int = 0
    load_running: bool = False
    compute_running: bool = False
    load_busy_until: int = 0
    compute_busy_until: int = 0
    load_end: dict[int, int] = field(default_factory=dict)
    compute_end: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _PendingLoad:
    cluster: int
    item: WorkItem
    deadline_cc: int


@dataclass(frozen=True)
class _RunningLoad:
    record: LoadRecord


@dataclass(frozen=True)
class _RunningCompute:
    record: ComputeRecord


def _dma_duration(weight_bytes: int, lane_count: int) -> int:
    if lane_count not in (1, 2):
        raise ValueError(f"invalid lane count {lane_count}")
    return math.ceil(weight_bytes / (LANE_BW_BYTES_PER_CC * lane_count))


class NOuterSimulator:
    def __init__(self, config: NOuterConfig | None = None):
        self.config = config or default_config()

    def build_streams(self, group: GroupDescriptor) -> tuple[tuple[WorkItem, ...], ...]:
        streams: list[tuple[WorkItem, ...]] = []
        for cluster in (0, 1):
            experts = group.experts(cluster)
            items: list[WorkItem] = []
            for phase_index, phase in enumerate(self.config.phases):
                for block_id in range(phase.block_count):
                    for sequence_index, expert in enumerate(experts):
                        items.append(
                            WorkItem(
                                cluster=cluster,
                                stream_index=len(items),
                                phase_index=phase_index,
                                phase_name=phase.name,
                                block_id=block_id,
                                sequence_index=sequence_index,
                                expert=expert,
                                weight_bytes=phase.weight_block_bytes,
                                compute_cc=expert.compute_cc(phase),
                            )
                        )
            streams.append(tuple(items))
        return tuple(streams)

    def evaluate(self, candidate: ScheduleCandidate) -> CandidateResult:
        self._validate_group_capacity(candidate.group)
        streams = self.build_streams(candidate.group)
        states = [_ClusterState(streams[0]), _ClusterState(streams[1])]
        running_loads: list[_RunningLoad] = []
        running_computes: list[_RunningCompute] = []
        load_records: list[LoadRecord] = []
        compute_records: list[ComputeRecord] = []
        now = 0

        while not self._all_done(states):
            self._finish_events(now, states, running_loads, running_computes)
            self._start_ready_computes(now, states, running_computes, compute_records)
            self._start_ready_loads(now, states, running_loads, load_records)
            self._start_ready_computes(now, states, running_computes, compute_records)

            if self._all_done(states):
                break

            event_times = [run.record.end_cc for run in running_loads]
            event_times.extend(run.record.end_cc for run in running_computes)
            future_times = [end for end in event_times if end > now]
            if not future_times:
                raise RuntimeError("N-outer execution deadlocked")
            now = min(future_times)

        makespan = max(
            [record.end_cc for record in compute_records]
            + [record.end_cc for record in load_records]
            + [0]
        )
        return self.result_from_history(
            candidate,
            streams,
            load_records,
            compute_records,
            expected_makespan=makespan,
        )

    def result_from_history(
        self,
        candidate: ScheduleCandidate,
        streams: Sequence[Sequence[WorkItem]],
        load_records: Sequence[LoadRecord],
        compute_records: Sequence[ComputeRecord],
        *,
        expected_makespan: int | None = None,
    ) -> CandidateResult:
        """Build and validate common metrics for heuristic or exact histories."""

        makespan = max(
            [record.end_cc for record in compute_records]
            + [record.end_cc for record in load_records]
            + [0]
        )
        if expected_makespan is not None and makespan != expected_makespan:
            raise AssertionError("history makespan differs from the scheduler result")
        initial_wait, steady_stall = self._stall_metrics(streams, compute_records)
        total_compute = sum(item.compute_cc for stream in streams for item in stream)
        total_lane_cycles = sum(
            (record.end_cc - record.start_cc) * len(record.lanes)
            for record in load_records
        )
        compute_lb = max(sum(item.compute_cc for item in stream) for stream in streams)
        total_dma_bytes = sum(item.weight_bytes for stream in streams for item in stream)
        dma_lb = math.ceil(total_dma_bytes / (2 * LANE_BW_BYTES_PER_CC))
        lower_bound = max(compute_lb, dma_lb)
        result = CandidateResult(
            candidate=candidate,
            makespan_cc=makespan,
            lower_bound_cc=lower_bound,
            compute_lower_bound_cc=compute_lb,
            dma_lower_bound_cc=dma_lb,
            initial_wait_cc=initial_wait,
            steady_stall_cc=steady_stall,
            compute_utilization=(total_compute / (2 * makespan)) if makespan else 0.0,
            dma_lane_utilization=(total_lane_cycles / (2 * makespan)) if makespan else 0.0,
            loads=tuple(
                sorted(
                    load_records,
                    key=lambda record: (record.start_cc, record.item.cluster),
                )
            ),
            computes=tuple(
                sorted(compute_records, key=lambda record: (record.start_cc, record.item.cluster))
            ),
            history_validated=False,
        )
        self.validate_history(result, streams)
        return CandidateResult(**{**result.__dict__, "history_validated": True})

    def _validate_group_capacity(self, group: GroupDescriptor) -> None:
        limit = self.config.max_group_tokens_per_cluster
        if limit is None:
            return
        for cluster in (0, 1):
            total = sum(expert.ntokens for expert in group.experts(cluster))
            if total > limit:
                raise ValueError(
                    f"cluster{cluster} group has {total} tokens, exceeds limit {limit}"
                )

    @staticmethod
    def _all_done(states: Sequence[_ClusterState]) -> bool:
        return all(
            state.next_compute == len(state.items)
            and not state.compute_running
            and state.next_load == len(state.items)
            and not state.load_running
            for state in states
        )

    @staticmethod
    def _finish_events(
        now: int,
        states: Sequence[_ClusterState],
        running_loads: list[_RunningLoad],
        running_computes: list[_RunningCompute],
    ) -> None:
        completed_loads = [run for run in running_loads if run.record.end_cc == now]
        for run in completed_loads:
            item = run.record.item
            state = states[item.cluster]
            state.load_end[item.stream_index] = now
            state.load_running = False
            state.load_busy_until = now
            running_loads.remove(run)

        completed_computes = [run for run in running_computes if run.record.end_cc == now]
        for run in completed_computes:
            item = run.record.item
            state = states[item.cluster]
            state.compute_end[item.stream_index] = now
            state.compute_running = False
            state.compute_busy_until = now
            running_computes.remove(run)

    @staticmethod
    def _start_ready_computes(
        now: int,
        states: Sequence[_ClusterState],
        running: list[_RunningCompute],
        records: list[ComputeRecord],
    ) -> None:
        for cluster, state in enumerate(states):
            if state.compute_running or state.next_compute >= len(state.items):
                continue
            index = state.next_compute
            if state.load_end.get(index, math.inf) > now:
                continue
            if index > 0 and state.compute_end.get(index - 1, math.inf) > now:
                continue
            item = state.items[index]
            record = ComputeRecord(item=item, start_cc=now, end_cc=now + item.compute_cc)
            records.append(record)
            running.append(_RunningCompute(record))
            state.next_compute += 1
            state.compute_running = True
            state.compute_busy_until = record.end_cc

    def _start_ready_loads(
        self,
        now: int,
        states: Sequence[_ClusterState],
        running: list[_RunningLoad],
        records: list[LoadRecord],
    ) -> None:
        free_lanes = self._free_lanes(running)
        if not free_lanes:
            return
        pending = self._pending_loads(now, states)
        if not pending:
            return

        if (
            self.config.force_initial_split
            and now == 0
            and len(free_lanes) == 2
            and len(pending) == 2
            and all(request.item.stream_index == 0 for request in pending)
        ):
            for lane, request in zip(free_lanes, sorted(pending, key=lambda req: req.cluster)):
                self._launch_load(now, request, (lane,), states, running, records)
            return

        if len(free_lanes) == 2 and len(pending) >= 2:
            first, second = sorted(pending, key=self._pending_key)[:2]
            if self.config.dma_policy == DmaPolicy.SINGLE_ONLY:
                self._launch_load(now, first, (free_lanes[0],), states, running, records)
                self._launch_load(now, second, (free_lanes[1],), states, running, records)
                return
            action = self._choose_two_lane_action(now, first, second)
            if action == "split":
                self._launch_load(now, first, (free_lanes[0],), states, running, records)
                self._launch_load(now, second, (free_lanes[1],), states, running, records)
            elif action == "first_both":
                self._launch_load(now, first, tuple(free_lanes), states, running, records)
            else:
                self._launch_load(now, second, tuple(free_lanes), states, running, records)
            return

        request = min(pending, key=self._pending_key)
        if len(free_lanes) == 2 and self.config.dma_policy != DmaPolicy.SINGLE_ONLY:
            single_end = now + _dma_duration(request.item.weight_bytes, 1)
            lanes = tuple(free_lanes) if single_end > request.deadline_cc else (free_lanes[0],)
        else:
            lanes = (free_lanes[0],)
        self._launch_load(now, request, lanes, states, running, records)

    @staticmethod
    def _free_lanes(running: Iterable[_RunningLoad]) -> list[int]:
        occupied = {
            lane
            for transfer in running
            for lane in transfer.record.lanes
        }
        return [lane for lane in (0, 1) if lane not in occupied]

    @staticmethod
    def _pending_loads(now: int, states: Sequence[_ClusterState]) -> list[_PendingLoad]:
        pending: list[_PendingLoad] = []
        for cluster, state in enumerate(states):
            if state.load_running or state.next_load >= len(state.items):
                continue
            index = state.next_load
            if index >= 2 and state.compute_end.get(index - 2, math.inf) > now:
                continue
            if index == 0:
                deadline = 0
            elif state.compute_running:
                deadline = state.compute_busy_until
            else:
                deadline = state.compute_end.get(index - 1, now)
            pending.append(_PendingLoad(cluster, state.items[index], int(deadline)))
        return pending

    @staticmethod
    def _pending_key(request: _PendingLoad) -> tuple[int, int, int]:
        return (request.deadline_cc, request.item.compute_cc, request.cluster)

    @staticmethod
    def _choose_two_lane_action(
        now: int, first: _PendingLoad, second: _PendingLoad
    ) -> str:
        def lateness(end: int, deadline: int) -> int:
            return max(0, end - deadline)

        first_single = now + _dma_duration(first.item.weight_bytes, 1)
        second_single = now + _dma_duration(second.item.weight_bytes, 1)
        split_score = (
            max(
                lateness(first_single, first.deadline_cc),
                lateness(second_single, second.deadline_cc),
            ),
            lateness(first_single, first.deadline_cc)
            + lateness(second_single, second.deadline_cc),
            max(first_single, second_single),
            0,
        )

        first_both = now + _dma_duration(first.item.weight_bytes, 2)
        second_after = first_both + _dma_duration(second.item.weight_bytes, 2)
        first_score = (
            max(
                lateness(first_both, first.deadline_cc),
                lateness(second_after, second.deadline_cc),
            ),
            lateness(first_both, first.deadline_cc)
            + lateness(second_after, second.deadline_cc),
            second_after,
            1,
        )

        second_both = now + _dma_duration(second.item.weight_bytes, 2)
        first_after = second_both + _dma_duration(first.item.weight_bytes, 2)
        second_score = (
            max(
                lateness(second_both, second.deadline_cc),
                lateness(first_after, first.deadline_cc),
            ),
            lateness(second_both, second.deadline_cc)
            + lateness(first_after, first.deadline_cc),
            first_after,
            2,
        )
        return min(
            (split_score, "split"),
            (first_score, "first_both"),
            (second_score, "second_both"),
            key=lambda choice: choice[0],
        )[1]

    @staticmethod
    def _launch_load(
        now: int,
        request: _PendingLoad,
        lanes: tuple[int, ...],
        states: Sequence[_ClusterState],
        running: list[_RunningLoad],
        records: list[LoadRecord],
    ) -> None:
        state = states[request.cluster]
        item = request.item
        duration = _dma_duration(item.weight_bytes, len(lanes))
        record = LoadRecord(item=item, start_cc=now, end_cc=now + duration, lanes=lanes)
        records.append(record)
        running.append(_RunningLoad(record))
        state.next_load += 1
        state.load_running = True
        state.load_busy_until = record.end_cc

    @staticmethod
    def _stall_metrics(
        streams: Sequence[Sequence[WorkItem]], records: Sequence[ComputeRecord]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        by_key = {record.item.key: record for record in records}
        initial: list[int] = []
        steady: list[int] = []
        for cluster, stream in enumerate(streams):
            if not stream:
                initial.append(0)
                steady.append(0)
                continue
            initial.append(by_key[(cluster, 0)].start_cc)
            stall = 0
            for index in range(1, len(stream)):
                previous = by_key[(cluster, index - 1)]
                current = by_key[(cluster, index)]
                stall += current.start_cc - previous.end_cc
            steady.append(stall)
        return tuple(initial), tuple(steady)

    @staticmethod
    def validate_history(
        result: CandidateResult, streams: Sequence[Sequence[WorkItem]]
    ) -> None:
        loads = {record.item.key: record for record in result.loads}
        computes = {record.item.key: record for record in result.computes}
        expected = {item.key for stream in streams for item in stream}
        if set(loads) != expected or set(computes) != expected:
            raise AssertionError("history does not contain exactly one load/compute per item")

        for cluster, stream in enumerate(streams):
            for index, item in enumerate(stream):
                load = loads[item.key]
                compute = computes[item.key]
                if load.end_cc > compute.start_cc:
                    raise AssertionError("compute started before its weight load completed")
                if index > 0 and computes[(cluster, index - 1)].end_cc > compute.start_cc:
                    raise AssertionError("cluster compute stream overlapped itself")
                if index >= 2 and computes[(cluster, index - 2)].end_cc > load.start_cc:
                    raise AssertionError("DMA overwrote a live ping/pong weight slot")

        for lane in (0, 1):
            lane_records = sorted(
                (record for record in result.loads if lane in record.lanes),
                key=lambda record: record.start_cc,
            )
            for previous, current in zip(lane_records, lane_records[1:]):
                if previous.end_cc > current.start_cc:
                    raise AssertionError(f"DMA lane {lane} has overlapping transfers")
