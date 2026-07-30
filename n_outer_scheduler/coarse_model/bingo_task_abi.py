#!/usr/bin/env python3
"""Concrete, isolated Bingo task ABI for a selected N-outer macro history.

This module does not modify or reuse the current M-outer S1/S2/S3/S4 task
record.  That record has different execution semantics.  Instead it defines
the dynamic fields required by a future fixed N-outer Bingo DFG:

* one macro slot record per selected expert/token slice;
* one LOAD task per fixed weight block;
* one COMPUTE task per fixed weight block;
* dependency edges that encode ping/pong release and the selected global DMA
  service order.

The scheduler never chooses individual blocks.  Block tasks are a deterministic
lowering of the selected phase parameters, exactly as a fixed Bingo graph may
expand a coarse scheduler decision.  Absolute model timestamps are retained as
audit metadata only and are not used by :func:`replay_bingo_task_program`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .block_golden import (
    ArbitrationPolicy,
    BlockItem,
    GoldenResult,
    build_block_streams,
    replay_block_streams,
)
from .lowering import BingoMacroProgram, lower_history_to_bingo
from .search import SearchNode, validate_history
from .semantics import DmaBinding, ShapeName, default_phases


class BingoNOuterTaskKind(str, Enum):
    LOAD_WEIGHT = "load_weight"
    COMPUTE_RESIDENT_WEIGHT = "compute_resident_weight"


@dataclass(frozen=True)
class BingoNOuterStaticArgs:
    """Shape-independent parameters owned by the fixed N-outer DFG."""

    gate_up_blocks: int
    down_blocks: int
    gate_up_block_bytes: int
    down_block_bytes: int
    dma_lane_bytes_per_cc: int = 64
    weight_buffer_count: int = 2


@dataclass(frozen=True)
class BingoNOuterSlotArgs:
    """One scheduler-selected macro record copied to a cluster-local slot."""

    step_index: int
    action_kind: str
    cluster: int
    local_slot: int
    eid: int
    token_start: int
    ntokens: int
    gate_up_shape: ShapeName
    gate_up_dma_mask: int
    down_shape: ShapeName
    down_dma_mask: int
    gate_up_first_rank: int
    gate_up_stream_rank: int
    down_first_rank: int
    down_stream_rank: int

    @property
    def token_end(self) -> int:
        return self.token_start + self.ntokens


@dataclass(frozen=True)
class BingoNOuterTaskArgs:
    """Dynamic arguments for one fixed LOAD or COMPUTE DFG node."""

    task_id: int
    kind: BingoNOuterTaskKind
    cluster: int
    local_slot: int
    cluster_stream_index: int
    phase_index: int
    phase_name: str
    block_id: int
    block_count: int
    weight_buffer_slot: int
    eid: int
    token_start: int
    ntokens: int
    shape: ShapeName
    depends_on: tuple[int, ...]
    duration_cc: int
    dma_mask: int = 0
    weight_bytes: int = 0
    weight_block_offset: int = 0
    model_start_cc: int = 0
    model_end_cc: int = 0

    @property
    def token_end(self) -> int:
        return self.token_start + self.ntokens

    @property
    def dma_lanes(self) -> tuple[int, ...]:
        if self.kind != BingoNOuterTaskKind.LOAD_WEIGHT:
            return ()
        return tuple(lane for lane in (0, 1) if self.dma_mask & (1 << lane))


@dataclass(frozen=True)
class BingoNOuterTaskProgram:
    """Self-contained parameter image for a fixed N-outer Bingo graph."""

    static_args: BingoNOuterStaticArgs
    slots: tuple[BingoNOuterSlotArgs, ...]
    tasks: tuple[BingoNOuterTaskArgs, ...]
    issue_order: tuple[int, ...]
    dma_lane_task_order: tuple[tuple[int, ...], tuple[int, ...]]
    cluster_compute_task_order: tuple[tuple[int, ...], tuple[int, ...]]
    arbitration_policy: ArbitrationPolicy
    source_macro_makespan_cc: int
    source_block_makespan_cc: int
    history_validated: bool

    def task_by_id(self) -> dict[int, BingoNOuterTaskArgs]:
        return {task.task_id: task for task in self.tasks}


@dataclass(frozen=True)
class BingoNOuterTaskExecution:
    task_id: int
    start_cc: int
    end_cc: int


@dataclass(frozen=True)
class BingoNOuterReplayResult:
    executions: tuple[BingoNOuterTaskExecution, ...]
    makespan_cc: int
    dependencies_valid: bool
    resources_valid: bool
    ping_pong_valid: bool
    order_valid: bool
    token_ranges_valid: bool


def _slot_args(program: BingoMacroProgram) -> tuple[BingoNOuterSlotArgs, ...]:
    return tuple(
        BingoNOuterSlotArgs(
            step_index=record.step_index,
            action_kind=record.action_kind,
            cluster=record.cluster,
            local_slot=record.cluster_sequence_index,
            eid=record.eid,
            token_start=record.token_start,
            ntokens=record.ntokens,
            gate_up_shape=record.gate_up_shape,
            gate_up_dma_mask=record.gate_up_dma_mask,
            down_shape=record.down_shape,
            down_dma_mask=record.down_dma_mask,
            gate_up_first_rank=record.gate_up_first_rank,
            gate_up_stream_rank=record.gate_up_stream_rank,
            down_first_rank=record.down_first_rank,
            down_stream_rank=record.down_stream_rank,
        )
        for record in program.records
    )


def _golden_maps(golden: GoldenResult):
    loads = {
        (record.item.cluster, record.item.stream_index): record
        for record in golden.loads
    }
    computes = {
        (record.item.cluster, record.item.stream_index): record
        for record in golden.computes
    }
    return loads, computes


def lower_history_to_bingo_tasks(
    distribution: Sequence[int],
    node: SearchNode,
    *,
    policy: ArbitrationPolicy = ArbitrationPolicy.MACRO_ORDER,
) -> BingoNOuterTaskProgram:
    """Lower a validated macro history into a dependency-complete task image."""

    validate_history(distribution, node)
    macro_program = lower_history_to_bingo(distribution, node)
    slots = _slot_args(macro_program)
    slot_by_step_cluster = {
        (slot.step_index, slot.cluster): slot for slot in slots
    }

    streams = build_block_streams(node.history)
    golden = replay_block_streams(streams, policy=policy)
    golden_loads, golden_computes = _golden_maps(golden)

    load_id: dict[tuple[int, int], int] = {}
    compute_id: dict[tuple[int, int], int] = {}
    next_task_id = 0
    for cluster in (0, 1):
        for item in streams[cluster]:
            key = (cluster, item.stream_index)
            load_id[key] = next_task_id
            next_task_id += 1
            compute_id[key] = next_task_id
            next_task_id += 1

    lane_predecessors: dict[tuple[int, int], set[int]] = {
        key: set() for key in load_id
    }
    lane_order: list[list[int]] = [[], []]
    for lane in (0, 1):
        ordered = sorted(
            (
                record
                for record in golden.loads
                if record.item.binding & DmaBinding(1 << lane)
            ),
            key=lambda record: (
                record.start_cc,
                record.end_cc,
                record.item.service_rank,
                record.item.cluster,
                record.item.stream_index,
            ),
        )
        previous: int | None = None
        for record in ordered:
            key = (record.item.cluster, record.item.stream_index)
            current = load_id[key]
            lane_order[lane].append(current)
            if previous is not None:
                lane_predecessors[key].add(previous)
            previous = current

    tasks: list[BingoNOuterTaskArgs] = []
    cluster_compute_order: list[list[int]] = [[], []]
    for cluster in (0, 1):
        for item in streams[cluster]:
            key = (cluster, item.stream_index)
            slot = slot_by_step_cluster[(item.step_index, cluster)]
            load = golden_loads[key]
            compute = golden_computes[key]

            load_dependencies = set(lane_predecessors[key])
            # One fixed DMA worker advances each cluster stream. Even when two
            # consecutive blocks select different global lanes, the worker
            # cannot issue block i before it has retired block i-1.
            if item.stream_index:
                load_dependencies.add(
                    load_id[(cluster, item.stream_index - 1)]
                )
            if item.stream_index >= 2:
                load_dependencies.add(
                    compute_id[(cluster, item.stream_index - 2)]
                )
            tasks.append(
                BingoNOuterTaskArgs(
                    task_id=load_id[key],
                    kind=BingoNOuterTaskKind.LOAD_WEIGHT,
                    cluster=cluster,
                    local_slot=slot.local_slot,
                    cluster_stream_index=item.stream_index,
                    phase_index=item.phase_index,
                    phase_name=item.phase_name,
                    block_id=item.block_id,
                    block_count=item.block_count,
                    weight_buffer_slot=item.buffer_slot,
                    eid=item.eid,
                    token_start=item.token_start,
                    ntokens=item.ntokens,
                    shape=item.shape.name,
                    depends_on=tuple(sorted(load_dependencies)),
                    duration_cc=load.end_cc - load.start_cc,
                    dma_mask=int(item.binding),
                    weight_bytes=item.weight_bytes,
                    weight_block_offset=item.block_id * item.weight_bytes,
                    model_start_cc=load.start_cc,
                    model_end_cc=load.end_cc,
                )
            )

            compute_dependencies = {load_id[key]}
            if item.stream_index:
                compute_dependencies.add(
                    compute_id[(cluster, item.stream_index - 1)]
                )
            tasks.append(
                BingoNOuterTaskArgs(
                    task_id=compute_id[key],
                    kind=BingoNOuterTaskKind.COMPUTE_RESIDENT_WEIGHT,
                    cluster=cluster,
                    local_slot=slot.local_slot,
                    cluster_stream_index=item.stream_index,
                    phase_index=item.phase_index,
                    phase_name=item.phase_name,
                    block_id=item.block_id,
                    block_count=item.block_count,
                    weight_buffer_slot=item.buffer_slot,
                    eid=item.eid,
                    token_start=item.token_start,
                    ntokens=item.ntokens,
                    shape=item.shape.name,
                    depends_on=tuple(sorted(compute_dependencies)),
                    duration_cc=compute.end_cc - compute.start_cc,
                    model_start_cc=compute.start_cc,
                    model_end_cc=compute.end_cc,
                )
            )
            cluster_compute_order[cluster].append(compute_id[key])

    tasks.sort(key=lambda task: task.task_id)
    issue_order = tuple(
        task.task_id
        for task in sorted(
            tasks,
            key=lambda task: (
                task.model_start_cc,
                0 if task.kind == BingoNOuterTaskKind.LOAD_WEIGHT else 1,
                task.cluster,
                task.task_id,
            ),
        )
    )
    gate_up, down = default_phases()
    program = BingoNOuterTaskProgram(
        static_args=BingoNOuterStaticArgs(
            gate_up_blocks=gate_up.block_count,
            down_blocks=down.block_count,
            gate_up_block_bytes=gate_up.weight_block_bytes,
            down_block_bytes=down.weight_block_bytes,
        ),
        slots=slots,
        tasks=tuple(tasks),
        issue_order=issue_order,
        dma_lane_task_order=(tuple(lane_order[0]), tuple(lane_order[1])),
        cluster_compute_task_order=(
            tuple(cluster_compute_order[0]),
            tuple(cluster_compute_order[1]),
        ),
        arbitration_policy=policy,
        source_macro_makespan_cc=node.makespan_cc,
        source_block_makespan_cc=golden.makespan_cc,
        history_validated=False,
    )
    validate_bingo_task_program(distribution, program)
    return BingoNOuterTaskProgram(
        **{**program.__dict__, "history_validated": True}
    )


def replay_bingo_task_program(
    program: BingoNOuterTaskProgram,
) -> BingoNOuterReplayResult:
    """Earliest-time replay using dependencies, never model timestamps."""

    tasks = program.task_by_id()
    if len(tasks) != len(program.tasks):
        raise AssertionError("duplicate Bingo task ID")
    completed: dict[int, BingoNOuterTaskExecution] = {}
    remaining = set(tasks)
    issue_rank = {task_id: rank for rank, task_id in enumerate(program.issue_order)}
    while remaining:
        ready = [
            tasks[task_id]
            for task_id in remaining
            if all(dependency in completed for dependency in tasks[task_id].depends_on)
        ]
        if not ready:
            raise AssertionError("Bingo task graph is cyclic or has a missing dependency")
        for task in sorted(ready, key=lambda item: (issue_rank[item.task_id], item.task_id)):
            start = max(
                (completed[dependency].end_cc for dependency in task.depends_on),
                default=0,
            )
            completed[task.task_id] = BingoNOuterTaskExecution(
                task_id=task.task_id,
                start_cc=start,
                end_cc=start + task.duration_cc,
            )
            remaining.remove(task.task_id)

    executions = tuple(completed[index] for index in sorted(completed))
    result = BingoNOuterReplayResult(
        executions=executions,
        makespan_cc=max((item.end_cc for item in executions), default=0),
        dependencies_valid=True,
        resources_valid=_validate_replay_resources(program, executions),
        ping_pong_valid=_validate_ping_pong(program, executions),
        order_valid=_validate_compute_order(program, executions),
        token_ranges_valid=_validate_task_parameters(program),
    )
    if not all(
        (
            result.dependencies_valid,
            result.resources_valid,
            result.ping_pong_valid,
            result.order_valid,
            result.token_ranges_valid,
        )
    ):
        raise AssertionError("lowered Bingo task replay violates its contract")
    return result


def _execution_map(
    executions: Sequence[BingoNOuterTaskExecution],
) -> dict[int, BingoNOuterTaskExecution]:
    return {item.task_id: item for item in executions}


def _validate_replay_resources(
    program: BingoNOuterTaskProgram,
    executions: Sequence[BingoNOuterTaskExecution],
) -> bool:
    by_id = program.task_by_id()
    times = _execution_map(executions)
    for lane in (0, 1):
        intervals = sorted(
            (
                times[task_id].start_cc,
                times[task_id].end_cc,
                task_id,
            )
            for task_id in program.dma_lane_task_order[lane]
        )
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            return False
        if any(lane not in by_id[task_id].dma_lanes for _, _, task_id in intervals):
            return False
    for cluster in (0, 1):
        intervals = sorted(
            (
                times[task_id].start_cc,
                times[task_id].end_cc,
            )
            for task_id in program.cluster_compute_task_order[cluster]
        )
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            return False
    return True


def _validate_ping_pong(
    program: BingoNOuterTaskProgram,
    executions: Sequence[BingoNOuterTaskExecution],
) -> bool:
    times = _execution_map(executions)
    loads = {
        (task.cluster, task.cluster_stream_index): task
        for task in program.tasks
        if task.kind == BingoNOuterTaskKind.LOAD_WEIGHT
    }
    computes = {
        (task.cluster, task.cluster_stream_index): task
        for task in program.tasks
        if task.kind == BingoNOuterTaskKind.COMPUTE_RESIDENT_WEIGHT
    }
    for (cluster, index), load in loads.items():
        if load.weight_buffer_slot != (index & 1):
            return False
        if index >= 2:
            previous = computes[(cluster, index - 2)]
            if times[load.task_id].start_cc < times[previous.task_id].end_cc:
                return False
    return True


def _validate_compute_order(
    program: BingoNOuterTaskProgram,
    executions: Sequence[BingoNOuterTaskExecution],
) -> bool:
    times = _execution_map(executions)
    for cluster in (0, 1):
        ordered = program.cluster_compute_task_order[cluster]
        for left_id, right_id in zip(ordered, ordered[1:]):
            if times[right_id].start_cc < times[left_id].end_cc:
                return False
    return True


def _validate_task_parameters(program: BingoNOuterTaskProgram) -> bool:
    """Check that every lowered task is the deterministic slot expansion."""

    slots_by_cluster: list[list[BingoNOuterSlotArgs]] = [[], []]
    for slot in program.slots:
        if slot.cluster not in (0, 1):
            return False
        slots_by_cluster[slot.cluster].append(slot)
    for cluster in (0, 1):
        slots_by_cluster[cluster].sort(key=lambda item: item.local_slot)
        if [slot.local_slot for slot in slots_by_cluster[cluster]] != list(
            range(len(slots_by_cluster[cluster]))
        ):
            return False

    tasks_by_key: dict[
        tuple[int, int], dict[BingoNOuterTaskKind, BingoNOuterTaskArgs]
    ] = {}
    for task in program.tasks:
        bucket = tasks_by_key.setdefault(
            (task.cluster, task.cluster_stream_index), {}
        )
        if task.kind in bucket:
            return False
        bucket[task.kind] = task

    for cluster in (0, 1):
        expected: list[tuple[BingoNOuterSlotArgs, int, str, int, ShapeName, int, int]] = []
        for slot in slots_by_cluster[cluster]:
            for phase_index, (
                phase_name,
                block_count,
                block_bytes,
                shape,
                dma_mask,
            ) in enumerate(
                (
                    (
                        "gate_up",
                        program.static_args.gate_up_blocks,
                        program.static_args.gate_up_block_bytes,
                        slot.gate_up_shape,
                        slot.gate_up_dma_mask,
                    ),
                    (
                        "down",
                        program.static_args.down_blocks,
                        program.static_args.down_block_bytes,
                        slot.down_shape,
                        slot.down_dma_mask,
                    ),
                )
            ):
                expected.extend(
                    (
                        slot,
                        phase_index,
                        phase_name,
                        block_id,
                        shape,
                        dma_mask,
                        block_bytes,
                    )
                    for block_id in range(block_count)
                )
        keys = sorted(
            index for owner, index in tasks_by_key if owner == cluster
        )
        if keys != list(range(len(expected))):
            return False
        for stream_index, specification in enumerate(expected):
            slot, phase_index, phase_name, block_id, shape, dma_mask, block_bytes = specification
            bucket = tasks_by_key[(cluster, stream_index)]
            if set(bucket) != {
                BingoNOuterTaskKind.LOAD_WEIGHT,
                BingoNOuterTaskKind.COMPUTE_RESIDENT_WEIGHT,
            }:
                return False
            load = bucket[BingoNOuterTaskKind.LOAD_WEIGHT]
            compute = bucket[BingoNOuterTaskKind.COMPUTE_RESIDENT_WEIGHT]
            common_expected = (
                cluster,
                slot.local_slot,
                phase_index,
                phase_name,
                block_id,
                stream_index & 1,
                slot.eid,
                slot.token_start,
                slot.ntokens,
                shape,
            )
            for task in (load, compute):
                common_actual = (
                    task.cluster,
                    task.local_slot,
                    task.phase_index,
                    task.phase_name,
                    task.block_id,
                    task.weight_buffer_slot,
                    task.eid,
                    task.token_start,
                    task.ntokens,
                    task.shape,
                )
                if common_actual != common_expected:
                    return False
            if (
                load.block_count
                != (
                    program.static_args.gate_up_blocks
                    if phase_index == 0
                    else program.static_args.down_blocks
                )
                or load.dma_mask != dma_mask
                or load.weight_bytes != block_bytes
                or load.weight_block_offset != block_id * block_bytes
                or compute.block_count != load.block_count
            ):
                return False
    return True


def validate_bingo_task_program(
    distribution: Sequence[int], program: BingoNOuterTaskProgram
) -> None:
    """Structural validation independent of replay timestamps."""

    task_by_id = program.task_by_id()
    if set(task_by_id) != set(range(len(program.tasks))):
        raise AssertionError("Bingo task IDs must be contiguous")
    if set(program.issue_order) != set(task_by_id):
        raise AssertionError("issue order is not a task permutation")
    for task in program.tasks:
        if task.duration_cc <= 0:
            raise AssertionError("Bingo task duration must be positive")
        if any(dependency not in task_by_id for dependency in task.depends_on):
            raise AssertionError("Bingo task has an unknown dependency")
        if task.kind == BingoNOuterTaskKind.LOAD_WEIGHT:
            if task.dma_mask not in (1, 2, 3) or task.weight_bytes <= 0:
                raise AssertionError("invalid N-outer LOAD parameters")
        elif task.dma_mask != 0 or task.weight_bytes != 0:
            raise AssertionError("COMPUTE task unexpectedly carries DMA parameters")

    if not _validate_task_parameters(program):
        raise AssertionError("block tasks are not the deterministic macro-slot expansion")

    coverage: dict[int, list[tuple[int, int]]] = {}
    for slot in program.slots:
        coverage.setdefault(slot.eid, []).append((slot.token_start, slot.token_end))
    for eid, ntokens in enumerate(distribution):
        if ntokens <= 0:
            continue
        cursor = 0
        for start, end in sorted(coverage.get(eid, ())):
            if start != cursor:
                raise AssertionError("macro slot token ranges overlap or leave a gap")
            cursor = end
        if cursor != ntokens:
            raise AssertionError("macro slot token coverage is incomplete")

    replay = replay_bingo_task_program(program)
    if replay.makespan_cc != program.source_block_makespan_cc:
        raise AssertionError(
            "dependency-only Bingo replay does not reproduce the block golden"
        )
