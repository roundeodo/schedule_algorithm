#!/usr/bin/env python3
"""Canonical block-major N-outer execution model.

The dynamic scheduler supplies two complete ordered expert-slice lists.  The
static workers consume each list in true N-outer order::

    phase -> weight block -> expert slice -> token tile

Thus a cluster assigned ``[4, 2a, 2b]`` executes block 0 for all three slices
before returning to block 1.  A two-entry weight buffer permits the DMA worker
to load the next stream item while the VersaCore computes the current item.

Time is represented only in scheduler ticks.  One tick is the greatest common
divisor of every legal default load/compute duration (1408 accelerator cycles).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, IntFlag
from typing import Iterable, Sequence


TICK_CC = 1408
LANE_BYTES_PER_CC = 64


class Phase(str, Enum):
    GATE_UP = "gate_up"
    DOWN = "down"


class DmaMask(IntFlag):
    IDMA = 1
    XDMA = 2
    BOTH = IDMA | XDMA


class DmaPolicy(str, Enum):
    DEADLINE_AWARE = "deadline_aware"
    SINGLE_ONLY = "single_only"


@dataclass(frozen=True)
class PhaseSpec:
    phase: Phase
    block_count: int
    weight_block_bytes: int
    m4_compute_ticks: int
    m2_compute_ticks: int

    def __post_init__(self) -> None:
        if min(
            self.block_count,
            self.weight_block_bytes,
            self.m4_compute_ticks,
            self.m2_compute_ticks,
        ) <= 0:
            raise ValueError("phase constants must be positive")


def _default_phase(
    phase: Phase, block_count: int, weight_block_bytes: int
) -> PhaseSpec:
    m4_cc = math.ceil(weight_block_bytes / 64)
    m2_cc = math.ceil(weight_block_bytes / 128)
    if m4_cc % TICK_CC or m2_cc % TICK_CC:
        raise ValueError("default phase does not lie on the scheduler tick lattice")
    return PhaseSpec(
        phase,
        block_count,
        weight_block_bytes,
        m4_cc // TICK_CC,
        m2_cc // TICK_CC,
    )


@dataclass(frozen=True)
class ContractConfig:
    gate_up: PhaseSpec = _default_phase(
        Phase.GATE_UP, 8, 2 * 2048 * 176 // 2
    )
    down: PhaseSpec = _default_phase(Phase.DOWN, 8, 1408 * 256 // 2)
    dma_policy: DmaPolicy = DmaPolicy.DEADLINE_AWARE
    force_initial_split: bool = True
    weight_buffer_count: int = 2

    def __post_init__(self) -> None:
        if self.gate_up.phase != Phase.GATE_UP or self.down.phase != Phase.DOWN:
            raise ValueError("phase specifications are in the wrong positions")
        if self.weight_buffer_count != 2:
            raise ValueError("the frozen worker requires exactly two weight buffers")

    @property
    def phases(self) -> tuple[PhaseSpec, PhaseSpec]:
        return self.gate_up, self.down


@dataclass(frozen=True)
class ExpertSlice:
    """A real, non-padded token interval belonging to one expert."""

    eid: int
    token_start: int
    ntokens: int

    def __post_init__(self) -> None:
        if self.eid < 0 or self.token_start < 0 or self.ntokens <= 0:
            raise ValueError("invalid expert slice")

    @property
    def token_end(self) -> int:
        return self.token_start + self.ntokens

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

    def compute_ticks(self, phase: PhaseSpec) -> int:
        return (
            self.m4_iters * phase.m4_compute_ticks
            + self.m2_iters * phase.m2_compute_ticks
        )


@dataclass(frozen=True)
class GroupPlan:
    """Complete ordered expert-slice lists for one N-outer group."""

    cluster0: tuple[ExpertSlice, ...]
    cluster1: tuple[ExpertSlice, ...]
    group_id: int = 0

    def __post_init__(self) -> None:
        if self.group_id < 0 or (not self.cluster0 and not self.cluster1):
            raise ValueError("invalid/empty group")
        by_eid: dict[int, list[ExpertSlice]] = {}
        for item in (*self.cluster0, *self.cluster1):
            by_eid.setdefault(item.eid, []).append(item)
        for slices in by_eid.values():
            ordered = sorted(slices, key=lambda item: item.token_start)
            for left, right in zip(ordered, ordered[1:]):
                if left.token_end > right.token_start:
                    raise ValueError("expert slices overlap")

    def experts(self, cluster: int) -> tuple[ExpertSlice, ...]:
        if cluster == 0:
            return self.cluster0
        if cluster == 1:
            return self.cluster1
        raise ValueError("cluster must be zero or one")


@dataclass(frozen=True, order=True)
class ItemKey:
    cluster: int
    stream_index: int


@dataclass(frozen=True)
class WorkItem:
    key: ItemKey
    phase: Phase
    phase_index: int
    block_id: int
    sequence_index: int
    expert_slice: ExpertSlice
    weight_block_bytes: int
    compute_ticks: int

    @property
    def buffer_slot(self) -> int:
        return self.key.stream_index & 1


@dataclass(frozen=True)
class LoadEvent:
    item: WorkItem
    start_tick: int
    end_tick: int
    lanes: DmaMask


@dataclass(frozen=True)
class ComputeEvent:
    item: WorkItem
    start_tick: int
    end_tick: int


@dataclass(frozen=True)
class ScheduleResult:
    plan: GroupPlan
    streams: tuple[tuple[WorkItem, ...], tuple[WorkItem, ...]]
    loads: tuple[LoadEvent, ...]
    computes: tuple[ComputeEvent, ...]
    makespan_ticks: int
    lower_bound_ticks: int
    compute_lower_bound_ticks: int
    dma_lower_bound_ticks: int
    initial_wait_ticks: tuple[int, int]
    steady_stall_ticks: tuple[int, int]
    compute_utilization: float
    dma_lane_utilization: float
    validated: bool


@dataclass
class _ClusterState:
    items: tuple[WorkItem, ...]
    next_load: int = 0
    next_compute: int = 0
    running_load: LoadEvent | None = None
    running_compute: ComputeEvent | None = None
    load_end: dict[int, int] = field(default_factory=dict)
    compute_end: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _PendingLoad:
    cluster: int
    item: WorkItem
    deadline_tick: int


def dma_duration_ticks(weight_bytes: int, lanes: DmaMask) -> int:
    lane_count = int(bool(lanes & DmaMask.IDMA)) + int(bool(lanes & DmaMask.XDMA))
    if lane_count not in (1, 2):
        raise ValueError("a load must occupy one or two DMA lanes")
    return math.ceil(weight_bytes / (LANE_BYTES_PER_CC * lane_count * TICK_CC))


def build_streams(
    plan: GroupPlan, config: ContractConfig = ContractConfig()
) -> tuple[tuple[WorkItem, ...], tuple[WorkItem, ...]]:
    """Flatten one group in true block-major N-outer order."""

    streams: list[tuple[WorkItem, ...]] = []
    for cluster in (0, 1):
        items: list[WorkItem] = []
        experts = plan.experts(cluster)
        for phase_index, phase in enumerate(config.phases):
            for block_id in range(phase.block_count):
                for sequence_index, expert_slice in enumerate(experts):
                    items.append(
                        WorkItem(
                            key=ItemKey(cluster, len(items)),
                            phase=phase.phase,
                            phase_index=phase_index,
                            block_id=block_id,
                            sequence_index=sequence_index,
                            expert_slice=expert_slice,
                            weight_block_bytes=phase.weight_block_bytes,
                            compute_ticks=expert_slice.compute_ticks(phase),
                        )
                    )
        streams.append(tuple(items))
    return streams[0], streams[1]


class NOuterSimulator:
    """Ready-only producer/consumer simulation with a deadline-aware DMA arbiter."""

    def __init__(self, config: ContractConfig = ContractConfig()):
        self.config = config

    def schedule(self, plan: GroupPlan) -> ScheduleResult:
        streams = build_streams(plan, self.config)
        states = [_ClusterState(streams[0]), _ClusterState(streams[1])]
        loads: list[LoadEvent] = []
        computes: list[ComputeEvent] = []
        now = 0

        while not self._all_done(states):
            self._finish(now, states)
            self._start_computes(now, states, computes)
            self._start_loads(now, states, loads)
            self._start_computes(now, states, computes)
            if self._all_done(states):
                break
            future = [
                event.end_tick
                for state in states
                for event in (state.running_load, state.running_compute)
                if event is not None and event.end_tick > now
            ]
            if not future:
                raise RuntimeError("block-major N-outer execution deadlocked")
            now = min(future)

        makespan = max(
            (event.end_tick for event in (*loads, *computes)), default=0
        )
        both_clusters = bool(streams[0] and streams[1])
        compute_bounds: list[int] = []
        for stream in streams:
            if not stream:
                compute_bounds.append(0)
                continue
            initial_lanes = (
                DmaMask.IDMA
                if self.config.dma_policy == DmaPolicy.SINGLE_ONLY
                or (self.config.force_initial_split and both_clusters)
                else DmaMask.BOTH
            )
            compute_bounds.append(
                dma_duration_ticks(stream[0].weight_block_bytes, initial_lanes)
                + sum(item.compute_ticks for item in stream)
            )
        compute_lower_bound = max(compute_bounds)
        total_weight_bytes = sum(
            item.weight_block_bytes for stream in streams for item in stream
        )
        dma_lower_bound = math.ceil(
            total_weight_bytes / (2 * LANE_BYTES_PER_CC * TICK_CC)
        )
        lower_bound = max(compute_lower_bound, dma_lower_bound)
        initial_wait, steady_stall = self._stall_metrics(streams, computes)
        result = ScheduleResult(
            plan=plan,
            streams=streams,
            loads=tuple(sorted(loads, key=lambda event: (event.start_tick, event.item.key))),
            computes=tuple(
                sorted(computes, key=lambda event: (event.start_tick, event.item.key))
            ),
            makespan_ticks=makespan,
            lower_bound_ticks=lower_bound,
            compute_lower_bound_ticks=compute_lower_bound,
            dma_lower_bound_ticks=dma_lower_bound,
            initial_wait_ticks=initial_wait,
            steady_stall_ticks=steady_stall,
            compute_utilization=(
                sum(item.compute_ticks for stream in streams for item in stream)
                / (2 * makespan)
                if makespan
                else 0.0
            ),
            dma_lane_utilization=(
                sum(
                    (event.end_tick - event.start_tick)
                    * int(event.lanes).bit_count()
                    for event in loads
                )
                / (2 * makespan)
                if makespan
                else 0.0
            ),
            validated=False,
        )
        validate_schedule(result)
        return ScheduleResult(**{**result.__dict__, "validated": True})

    @staticmethod
    def _all_done(states: Sequence[_ClusterState]) -> bool:
        return all(
            state.next_compute == len(state.items)
            and state.running_compute is None
            and state.next_load == len(state.items)
            and state.running_load is None
            for state in states
        )

    @staticmethod
    def _finish(now: int, states: Sequence[_ClusterState]) -> None:
        for state in states:
            if state.running_load is not None and state.running_load.end_tick == now:
                index = state.running_load.item.key.stream_index
                state.load_end[index] = now
                state.running_load = None
            if state.running_compute is not None and state.running_compute.end_tick == now:
                index = state.running_compute.item.key.stream_index
                state.compute_end[index] = now
                state.running_compute = None

    @staticmethod
    def _start_computes(
        now: int, states: Sequence[_ClusterState], records: list[ComputeEvent]
    ) -> None:
        for state in states:
            index = state.next_compute
            if state.running_compute is not None or index >= len(state.items):
                continue
            if state.load_end.get(index, math.inf) > now:
                continue
            if index and state.compute_end.get(index - 1, math.inf) > now:
                continue
            item = state.items[index]
            event = ComputeEvent(item, now, now + item.compute_ticks)
            records.append(event)
            state.running_compute = event
            state.next_compute += 1

    def _start_loads(
        self, now: int, states: Sequence[_ClusterState], records: list[LoadEvent]
    ) -> None:
        free = self._free_lane_mask(states)
        pending = self._pending(now, states)
        if not free or not pending:
            return

        if (
            self.config.force_initial_split
            and now == 0
            and free == DmaMask.BOTH
            and len(pending) == 2
            and all(request.item.key.stream_index == 0 for request in pending)
        ):
            ordered = sorted(pending, key=lambda request: request.cluster)
            self._launch(now, ordered[0], DmaMask.IDMA, states, records)
            self._launch(now, ordered[1], DmaMask.XDMA, states, records)
            return

        if free == DmaMask.BOTH and len(pending) >= 2:
            first, second = sorted(pending, key=self._pending_key)[:2]
            if self.config.dma_policy == DmaPolicy.SINGLE_ONLY:
                action = "split"
            else:
                action = self._choose_two_lane_action(now, first, second)
            if action == "split":
                self._launch(now, first, DmaMask.IDMA, states, records)
                self._launch(now, second, DmaMask.XDMA, states, records)
            elif action == "first_both":
                self._launch(now, first, DmaMask.BOTH, states, records)
            else:
                self._launch(now, second, DmaMask.BOTH, states, records)
            return

        request = min(pending, key=self._pending_key)
        if free == DmaMask.BOTH and self.config.dma_policy != DmaPolicy.SINGLE_ONLY:
            single_end = now + dma_duration_ticks(
                request.item.weight_block_bytes, DmaMask.IDMA
            )
            lanes = DmaMask.BOTH if single_end > request.deadline_tick else DmaMask.IDMA
        else:
            lanes = DmaMask.IDMA if free & DmaMask.IDMA else DmaMask.XDMA
        self._launch(now, request, lanes, states, records)

    @staticmethod
    def _free_lane_mask(states: Sequence[_ClusterState]) -> DmaMask:
        occupied = DmaMask(0)
        for state in states:
            if state.running_load is not None:
                occupied |= state.running_load.lanes
        return DmaMask.BOTH & ~occupied

    @staticmethod
    def _pending(now: int, states: Sequence[_ClusterState]) -> list[_PendingLoad]:
        pending: list[_PendingLoad] = []
        for cluster, state in enumerate(states):
            index = state.next_load
            if state.running_load is not None or index >= len(state.items):
                continue
            # Loading item i reuses item i-2's ping/pong slot.
            if index >= 2 and state.compute_end.get(index - 2, math.inf) > now:
                continue
            if index == 0:
                deadline = 0
            elif state.running_compute is not None:
                deadline = state.running_compute.end_tick
            else:
                deadline = state.compute_end.get(index - 1, now)
            pending.append(_PendingLoad(cluster, state.items[index], int(deadline)))
        return pending

    @staticmethod
    def _pending_key(request: _PendingLoad) -> tuple[int, int, int]:
        return request.deadline_tick, request.item.compute_ticks, request.cluster

    @staticmethod
    def _choose_two_lane_action(
        now: int, first: _PendingLoad, second: _PendingLoad
    ) -> str:
        def late(end: int, deadline: int) -> int:
            return max(0, end - deadline)

        def score(first_end: int, second_end: int, tie: int) -> tuple[int, int, int, int]:
            l0 = late(first_end, first.deadline_tick)
            l1 = late(second_end, second.deadline_tick)
            return max(l0, l1), l0 + l1, max(first_end, second_end), tie

        first_single = now + dma_duration_ticks(
            first.item.weight_block_bytes, DmaMask.IDMA
        )
        second_single = now + dma_duration_ticks(
            second.item.weight_block_bytes, DmaMask.XDMA
        )
        choices = [(score(first_single, second_single, 0), "split")]

        first_both = now + dma_duration_ticks(first.item.weight_block_bytes, DmaMask.BOTH)
        second_after = first_both + dma_duration_ticks(
            second.item.weight_block_bytes, DmaMask.BOTH
        )
        choices.append((score(first_both, second_after, 1), "first_both"))

        second_both = now + dma_duration_ticks(
            second.item.weight_block_bytes, DmaMask.BOTH
        )
        first_after = second_both + dma_duration_ticks(
            first.item.weight_block_bytes, DmaMask.BOTH
        )
        # score() expects first then second completion.
        choices.append((score(first_after, second_both, 2), "second_both"))
        return min(choices, key=lambda choice: choice[0])[1]

    @staticmethod
    def _launch(
        now: int,
        request: _PendingLoad,
        lanes: DmaMask,
        states: Sequence[_ClusterState],
        records: list[LoadEvent],
    ) -> None:
        state = states[request.cluster]
        event = LoadEvent(
            request.item,
            now,
            now + dma_duration_ticks(request.item.weight_block_bytes, lanes),
            lanes,
        )
        records.append(event)
        state.running_load = event
        state.next_load += 1

    @staticmethod
    def _stall_metrics(
        streams: Sequence[Sequence[WorkItem]], records: Sequence[ComputeEvent]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        by_key = {event.item.key: event for event in records}
        initial: list[int] = []
        steady: list[int] = []
        for cluster, stream in enumerate(streams):
            if not stream:
                initial.append(0)
                steady.append(0)
                continue
            initial.append(by_key[ItemKey(cluster, 0)].start_tick)
            steady.append(
                sum(
                    by_key[ItemKey(cluster, index)].start_tick
                    - by_key[ItemKey(cluster, index - 1)].end_tick
                    for index in range(1, len(stream))
                )
            )
        return (initial[0], initial[1]), (steady[0], steady[1])


def validate_schedule(result: ScheduleResult) -> None:
    loads = {event.item.key: event for event in result.loads}
    computes = {event.item.key: event for event in result.computes}
    expected = {item.key for stream in result.streams for item in stream}
    if set(loads) != expected or set(computes) != expected:
        raise AssertionError("schedule must contain one load and compute per item")

    for cluster, stream in enumerate(result.streams):
        for index, item in enumerate(stream):
            load = loads[item.key]
            compute = computes[item.key]
            if load.end_tick > compute.start_tick:
                raise AssertionError("compute starts before its load completes")
            if index and computes[ItemKey(cluster, index - 1)].end_tick > compute.start_tick:
                raise AssertionError("one VersaCore executes overlapping work")
            if index >= 2 and computes[ItemKey(cluster, index - 2)].end_tick > load.start_tick:
                raise AssertionError("DMA overwrites a live ping/pong buffer")

    for lane in (DmaMask.IDMA, DmaMask.XDMA):
        intervals = sorted(
            (event for event in result.loads if event.lanes & lane),
            key=lambda event: (event.start_tick, event.end_tick, event.item.key),
        )
        for left, right in zip(intervals, intervals[1:]):
            if left.end_tick > right.start_tick:
                raise AssertionError("DMA lane is double-booked")

    if result.makespan_ticks != max(
        (event.end_tick for event in (*result.loads, *result.computes)), default=0
    ):
        raise AssertionError("makespan does not cover all events")


def schedule_group(
    plan: GroupPlan, *, config: ContractConfig = ContractConfig()
) -> ScheduleResult:
    return NOuterSimulator(config).schedule(plan)


def slices_from_counts(counts: Iterable[int], *, first_eid: int = 0) -> tuple[ExpertSlice, ...]:
    return tuple(
        ExpertSlice(first_eid + index, 0, int(count))
        for index, count in enumerate(counts)
        if int(count) > 0
    )
