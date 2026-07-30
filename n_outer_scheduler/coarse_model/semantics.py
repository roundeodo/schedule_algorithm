#!/usr/bin/env python3
"""Coarse-grained N-outer timing semantics for search and future RTL.

The scheduler operates on expert/token slices and whole Gate/Up or Down
phases.  Eight block-level double-buffer steps are represented by a closed
phase envelope; they are not candidate actions.  A phase reserves one explicit
DMA binding (iDMA, xDMA, or BOTH) for its aggregate block-prefetch demand.

This is intentionally an ideal model, like the resident four-stage reference:
DMA and compute overlap perfectly when their dependencies and resources allow
it.  Resource ownership, delayed service, pipeline fill/drain, and resulting
stall remain explicit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, IntFlag
from functools import lru_cache
from typing import Iterable, Sequence


LANE_BW_BYTES_PER_CC = 64


class DmaBinding(IntFlag):
    NONE = 0
    IDMA = 1
    XDMA = 2
    BOTH = IDMA | XDMA


SINGLE_BINDINGS = (DmaBinding.IDMA, DmaBinding.XDMA)


class ShapeName(str, Enum):
    M8 = "M8"
    M4 = "M4"
    M2 = "M2"


@dataclass(frozen=True)
class ShapeSpec:
    name: ShapeName
    m_dim: int
    compute_bw_bytes_per_cc: int


SHAPE_M8 = ShapeSpec(ShapeName.M8, 8, 32)
SHAPE_M4 = ShapeSpec(ShapeName.M4, 4, 64)
SHAPE_M2 = ShapeSpec(ShapeName.M2, 2, 128)
ALL_SHAPES = (SHAPE_M8, SHAPE_M4, SHAPE_M2)


@dataclass(frozen=True)
class MacroPhaseSpec:
    name: str
    block_count: int
    weight_block_bytes: int

    def __post_init__(self) -> None:
        if self.block_count <= 0 or self.weight_block_bytes <= 0:
            raise ValueError("phase block count and bytes must be positive")


def default_phases() -> tuple[MacroPhaseSpec, MacroPhaseSpec]:
    return (
        MacroPhaseSpec(
            name="gate_up",
            block_count=8,
            weight_block_bytes=2 * 2048 * 176 // 2,
        ),
        MacroPhaseSpec(
            name="down",
            block_count=8,
            weight_block_bytes=1408 * 256 // 2,
        ),
    )


@dataclass(frozen=True)
class ExpertSlice:
    eid: int
    token_start: int
    ntokens: int

    def __post_init__(self) -> None:
        if self.eid < 0 or self.token_start < 0 or self.ntokens <= 0:
            raise ValueError("invalid expert slice")

    @property
    def token_end(self) -> int:
        return self.token_start + self.ntokens


@dataclass(frozen=True)
class PhasePlan:
    shape: ShapeSpec
    dma: DmaBinding

    def __post_init__(self) -> None:
        if self.dma not in (*SINGLE_BINDINGS, DmaBinding.BOTH):
            raise ValueError("a phase must use IDMA, XDMA, or BOTH")


@dataclass(frozen=True)
class MacroTaskPlan:
    cluster: int
    expert_slice: ExpertSlice
    gate_up: PhasePlan
    down: PhasePlan

    def __post_init__(self) -> None:
        if self.cluster not in (0, 1):
            raise ValueError("cluster must be 0 or 1")


class ActionKind(str, Enum):
    SINGLE = "single"
    PAIR = "pair"
    SPLIT = "split"


class PrefetchTargetKind(str, Enum):
    SAME_EXPERT_NEXT_BLOCK = "same_expert_next_block"
    SAME_EXPERT_DOWN_FIRST = "same_expert_down_first"
    NEXT_EXPERT_GATE_UP_FIRST = "next_expert_gate_up_first"


@dataclass(frozen=True)
class PrefetchTarget:
    kind: PrefetchTargetKind
    eid: int
    phase_name: str
    block_id: int


def legal_prefetch_targets(
    *,
    current_slice: ExpertSlice,
    phase: MacroPhaseSpec,
    block_id: int,
    down_phase: MacroPhaseSpec | None = None,
    next_slice: ExpertSlice | None = None,
) -> tuple[PrefetchTarget, ...]:
    """Return lowering-realizable targets for the alternate weight buffer.

    While a phase has a successor block, two-buffer execution leaves no free
    policy choice: the alternate slot is required by that successor.  Only the
    final block opens a boundary-prefetch opportunity.  Gate/Up may prime the
    same expert's Down block 0; Down may prime the next scheduled slice's
    Gate/Up block 0.  Returning an empty tuple means the scheduler may leave
    the alternate slot unused.
    """

    if block_id < 0 or block_id >= phase.block_count:
        raise ValueError("block id is outside the phase")
    if block_id + 1 < phase.block_count:
        return (
            PrefetchTarget(
                kind=PrefetchTargetKind.SAME_EXPERT_NEXT_BLOCK,
                eid=current_slice.eid,
                phase_name=phase.name,
                block_id=block_id + 1,
            ),
        )
    if phase.name == "gate_up":
        if down_phase is None:
            raise ValueError("Gate/Up boundary requires a Down phase")
        return (
            PrefetchTarget(
                kind=PrefetchTargetKind.SAME_EXPERT_DOWN_FIRST,
                eid=current_slice.eid,
                phase_name=down_phase.name,
                block_id=0,
            ),
        )
    if phase.name == "down" and next_slice is not None:
        return (
            PrefetchTarget(
                kind=PrefetchTargetKind.NEXT_EXPERT_GATE_UP_FIRST,
                eid=next_slice.eid,
                phase_name="gate_up",
                block_id=0,
            ),
        )
    return ()


@dataclass(frozen=True)
class MacroActionPlan:
    kind: ActionKind
    tasks: tuple[MacroTaskPlan, ...]

    def __post_init__(self) -> None:
        clusters = [task.cluster for task in self.tasks]
        if not self.tasks or len(self.tasks) > 2 or len(clusters) != len(set(clusters)):
            raise ValueError("an action has one task per participating cluster")
        if self.kind == ActionKind.SINGLE and len(self.tasks) != 1:
            raise ValueError("SINGLE requires one task")
        if self.kind in (ActionKind.PAIR, ActionKind.SPLIT) and len(self.tasks) != 2:
            raise ValueError("PAIR/SPLIT require two tasks")
        left = self.tasks[0].expert_slice
        if self.kind == ActionKind.PAIR:
            if left.eid == self.tasks[1].expert_slice.eid:
                raise ValueError("PAIR requires different experts")
        if self.kind == ActionKind.SPLIT:
            right = self.tasks[1].expert_slice
            if left.eid != right.eid:
                raise ValueError("SPLIT requires the same expert")
            if max(left.token_start, right.token_start) < min(left.token_end, right.token_end):
                raise ValueError("SPLIT token slices overlap")


@dataclass(frozen=True)
class LaneState:
    idma_free_cc: int = 0
    xdma_free_cc: int = 0

    def free_cc(self, binding: DmaBinding) -> int:
        values = []
        if binding & DmaBinding.IDMA:
            values.append(self.idma_free_cc)
        if binding & DmaBinding.XDMA:
            values.append(self.xdma_free_cc)
        if not values:
            raise ValueError("NONE has no DMA resource")
        return max(values)

    def reserve(self, binding: DmaBinding, end_cc: int) -> "LaneState":
        return LaneState(
            idma_free_cc=end_cc if binding & DmaBinding.IDMA else self.idma_free_cc,
            xdma_free_cc=end_cc if binding & DmaBinding.XDMA else self.xdma_free_cc,
        )


@dataclass(frozen=True)
class MacroScheduleState:
    """State carried between selected macro actions.

    ``prefetch_release_cc[c]`` is the start of cluster ``c``'s final Down
    compute block.  The next scheduled expert may use that tail window to
    fetch its Gate/Up block 0.  It is a release time, not a promise that the
    block is ready; readiness is derived from the shared DMA-lane schedule.
    """

    cluster_free_cc: tuple[int, int] = (0, 0)
    prefetch_release_cc: tuple[int, int] = (0, 0)
    lane_state: LaneState = LaneState()

    def __post_init__(self) -> None:
        if any(value < 0 for value in (*self.cluster_free_cc, *self.prefetch_release_cc)):
            raise ValueError("schedule-state cycles must be non-negative")
        if any(
            self.prefetch_release_cc[c] > self.cluster_free_cc[c]
            for c in (0, 1)
        ):
            raise ValueError("a prefetch tail cannot start after cluster completion")


@dataclass(frozen=True)
class DmaInterval:
    cluster: int
    phase_name: str
    role: str
    start_cc: int
    end_cc: int
    binding: DmaBinding


@dataclass(frozen=True)
class PhaseTiming:
    cluster: int
    phase_name: str
    release_cc: int
    first_prefetch_release_cc: int
    first_dma_start_cc: int
    first_dma_end_cc: int
    dma_start_cc: int
    dma_end_cc: int
    compute_start_cc: int
    last_compute_start_cc: int
    end_cc: int
    binding: DmaBinding
    compute_block_cc: int
    load_block_cc: int
    pipeline_stall_cc: int
    resource_wait_cc: int
    first_resource_wait_cc: int
    fill_stall_cc: int


@dataclass(frozen=True)
class MacroTaskTiming:
    plan: MacroTaskPlan
    gate_up: PhaseTiming
    down: PhaseTiming

    @property
    def end_cc(self) -> int:
        return self.down.end_cc


@dataclass(frozen=True)
class MacroActionTiming:
    plan: MacroActionPlan
    task_timings: tuple[MacroTaskTiming, ...]
    service_order: tuple[tuple[int, str], ...]
    lane_state: LaneState
    next_state: MacroScheduleState
    dma_intervals: tuple[DmaInterval, ...]
    makespan_cc: int
    resource_stall_cc: int
    pipeline_stall_cc: int
    history_validated: bool


def dma_duration(weight_bytes: int, binding: DmaBinding) -> int:
    lanes = int(bool(binding & DmaBinding.IDMA)) + int(
        bool(binding & DmaBinding.XDMA)
    )
    if lanes == 0:
        raise ValueError("DMA binding cannot be NONE")
    return math.ceil(weight_bytes / (LANE_BW_BYTES_PER_CC * lanes))


def compute_block_cc(
    ntokens: int, shape: ShapeSpec, phase: MacroPhaseSpec
) -> int:
    """Compute all tokens against one N-outer weight block.

    The selected shape handles the first tile.  Remaining tokens use M2, which
    matches the resident four-stage tail rule while keeping the N-outer weight
    block resident across every token tile.
    """

    if ntokens <= 0:
        return 0
    first = math.ceil(phase.weight_block_bytes / shape.compute_bw_bytes_per_cc)
    remaining = max(0, ntokens - shape.m_dim)
    tail_iters = math.ceil(remaining / SHAPE_M2.m_dim)
    tail = tail_iters * math.ceil(
        phase.weight_block_bytes / SHAPE_M2.compute_bw_bytes_per_cc
    )
    return first + tail


def legal_bindings(
    ntokens: int, shape: ShapeSpec, phase: MacroPhaseSpec
) -> tuple[DmaBinding, ...]:
    """Return non-dominated normal and exposed-stall service modes.

    If one lane can hide the next block, BOTH is locally dominated and is not
    generated.  Otherwise BOTH is the preferred fully hidden mode, while the
    two single-lane modes remain legal fallbacks with explicit stall.
    """

    compute_cc = compute_block_cc(ntokens, shape, phase)
    single_cc = dma_duration(phase.weight_block_bytes, DmaBinding.IDMA)
    if single_cc <= compute_cc:
        return SINGLE_BINDINGS
    return (DmaBinding.BOTH, *SINGLE_BINDINGS)


def evaluate_phase(
    *,
    cluster: int,
    release_cc: int,
    ntokens: int,
    phase: MacroPhaseSpec,
    plan: PhasePlan,
    lanes: LaneState,
) -> tuple[PhaseTiming, LaneState]:
    """Evaluate one cold macro phase.

    This helper charges block 0 explicitly.  Cross-phase and cross-expert
    prefetch are evaluated by :func:`evaluate_action`, where the first-block
    release can precede the phase's compute release.
    """

    if plan.dma not in legal_bindings(ntokens, plan.shape, phase):
        raise ValueError(
            f"dominated/unsupported {plan.shape.name.value}+{plan.dma.name} mode"
        )
    load_cc = dma_duration(phase.weight_block_bytes, plan.dma)
    first, lanes = _reserve_dma(
        cluster=cluster,
        phase_name=phase.name,
        role="first_block",
        release_cc=release_cc,
        duration_cc=load_cc,
        binding=plan.dma,
        lanes=lanes,
    )
    timing, _, lanes = _finish_stream_phase(
        cluster=cluster,
        ntokens=ntokens,
        phase=phase,
        plan=plan,
        release_cc=release_cc,
        first_release_cc=release_cc,
        first=first,
        lanes=lanes,
    )
    return timing, lanes


@lru_cache(maxsize=4)
def _phase_orders(clusters: tuple[int, ...]) -> tuple[tuple[tuple[int, str], ...], ...]:
    """Finite RTL-oriented joint service-mode bank.

    For two clusters, 16 lockstep modes choose C0/C1 priority independently at
    each of the four phase operations.  Two whole-chain modes cover the case
    where one short/BOTH task should run through a phase boundary before the
    other stream receives service.  This replaces arbitrary enumeration of all
    70 topological merges with 18 explicit, auditable alternatives.
    """

    phase_ops = (
        "gate_up_first",
        "gate_up_stream",
        "down_first",
        "down_stream",
    )
    if len(clusters) == 1:
        cluster = clusters[0]
        return (tuple((cluster, operation) for operation in phase_ops),)
    if len(clusters) != 2:
        raise ValueError("macro action supports at most two clusters")
    left, right = clusters
    orders: list[tuple[tuple[int, str], ...]] = []
    for priority_bits in range(1 << len(phase_ops)):
        order: list[tuple[int, str]] = []
        for index, operation in enumerate(phase_ops):
            first, second = (
                (right, left)
                if priority_bits & (1 << index)
                else (left, right)
            )
            order.extend(((first, operation), (second, operation)))
        orders.append(tuple(order))
    orders.extend(
        (
            tuple((left, operation) for operation in phase_ops)
            + tuple((right, operation) for operation in phase_ops),
            tuple((right, operation) for operation in phase_ops)
            + tuple((left, operation) for operation in phase_ops),
        )
    )
    return tuple(dict.fromkeys(orders))


@lru_cache(maxsize=4)
def _exhaustive_phase_orders(
    clusters: tuple[int, ...],
) -> tuple[tuple[tuple[int, str], ...], ...]:
    """Calibration-only complete topological merge of phase operations."""

    phase_ops = (
        "gate_up_first",
        "gate_up_stream",
        "down_first",
        "down_stream",
    )
    orders: list[tuple[tuple[int, str], ...]] = []

    def visit(
        positions: tuple[int, ...], prefix: tuple[tuple[int, str], ...]
    ) -> None:
        if all(position == len(phase_ops) for position in positions):
            orders.append(prefix)
            return
        for index, cluster in enumerate(clusters):
            position = positions[index]
            if position == len(phase_ops):
                continue
            updated = list(positions)
            updated[index] += 1
            visit(
                tuple(updated),
                (*prefix, (cluster, phase_ops[position])),
            )

    visit(tuple(0 for _ in clusters), ())
    return tuple(orders)


def _binding_hot_phase_order(
    tasks: dict[int, MacroTaskPlan],
    gate_up_spec: MacroPhaseSpec,
    down_spec: MacroPhaseSpec,
) -> tuple[tuple[int, str], ...]:
    """Return one deterministic, non-searched RTL priority order.

    At each phase operation, BOTH precedes a single-lane transfer. Equal
    bindings prioritize the longer resident-block compute window, then the
    lower cluster ID. Each cluster's four operations remain topological.
    """

    result: list[tuple[int, str]] = []
    for operation, phase_spec, phase_name in (
        ("gate_up_first", gate_up_spec, "gate_up"),
        ("gate_up_stream", gate_up_spec, "gate_up"),
        ("down_first", down_spec, "down"),
        ("down_stream", down_spec, "down"),
    ):
        ranked = sorted(
            tasks,
            key=lambda cluster: (
                0
                if getattr(tasks[cluster], phase_name).dma == DmaBinding.BOTH
                else 1,
                -compute_block_cc(
                    tasks[cluster].expert_slice.ntokens,
                    getattr(tasks[cluster], phase_name).shape,
                    phase_spec,
                ),
                cluster,
            ),
        )
        result.extend((cluster, operation) for cluster in ranked)
    return tuple(result)


def _binding_chain_phase_order(
    tasks: dict[int, MacroTaskPlan],
    gate_up_spec: MacroPhaseSpec,
    down_spec: MacroPhaseSpec,
) -> tuple[tuple[int, str], ...]:
    """Run one cluster's four phase operations before the other cluster.

    A task using BOTH has the larger blocking footprint and receives priority;
    ties use the larger total resident-block compute window, then cluster ID.
    This is one deterministic order, not a service-order search.
    """

    operations = (
        "gate_up_first",
        "gate_up_stream",
        "down_first",
        "down_stream",
    )

    def rank(cluster: int) -> tuple[int, int, int]:
        task = tasks[cluster]
        both_count = sum(
            phase.dma == DmaBinding.BOTH for phase in (task.gate_up, task.down)
        )
        compute_total = sum(
            compute_block_cc(task.expert_slice.ntokens, phase.shape, spec)
            for phase, spec in (
                (task.gate_up, gate_up_spec),
                (task.down, down_spec),
            )
        )
        return (-both_count, -compute_total, cluster)

    cluster_order = sorted(tasks, key=rank)
    return tuple(
        (cluster, operation)
        for cluster in cluster_order
        for operation in operations
    )


def _reserve_dma(
    *,
    cluster: int,
    phase_name: str,
    role: str,
    release_cc: int,
    duration_cc: int,
    binding: DmaBinding,
    lanes: LaneState,
) -> tuple[DmaInterval, LaneState]:
    start = max(release_cc, lanes.free_cc(binding))
    end = start + duration_cc
    interval = DmaInterval(cluster, phase_name, role, start, end, binding)
    return interval, lanes.reserve(binding, end)


def _finish_stream_phase(
    *,
    cluster: int,
    ntokens: int,
    phase: MacroPhaseSpec,
    plan: PhasePlan,
    release_cc: int,
    first_release_cc: int,
    first: DmaInterval,
    lanes: LaneState,
) -> tuple[PhaseTiming, tuple[DmaInterval, ...], LaneState]:
    """Close one phase with a fixed ping/pong-safe max/add recurrence.

    Blocks are not exposed as scheduler candidates.  The recurrence is the
    implementation of one phase envelope and has a compile-time trip count
    (eight in the deployed configuration).  Each load waits for the previous
    use of its ping/pong slot, so a delayed stream can never catch up by
    illegally overwriting live weights.
    """

    compute_cc = compute_block_cc(ntokens, plan.shape, phase)
    load_cc = dma_duration(phase.weight_block_bytes, plan.dma)
    compute_start = max(release_cc, first.end_cc)
    compute_starts = [compute_start]
    compute_ends = [compute_start + compute_cc]
    internal: list[DmaInterval] = []
    resource_wait = 0
    pipeline_stall = 0
    for block_id in range(1, phase.block_count):
        load_release = (
            compute_start if block_id == 1 else compute_ends[block_id - 2]
        )
        interval, lanes = _reserve_dma(
            cluster=cluster,
            phase_name=phase.name,
            role=f"block_{block_id}",
            release_cc=load_release,
            duration_cc=load_cc,
            binding=plan.dma,
            lanes=lanes,
        )
        internal.append(interval)
        resource_wait += interval.start_cc - load_release
        block_compute_start = max(compute_ends[-1], interval.end_cc)
        pipeline_stall += block_compute_start - compute_ends[-1]
        compute_starts.append(block_compute_start)
        compute_ends.append(block_compute_start + compute_cc)
    end_cc = compute_ends[-1]
    if internal:
        dma_start_cc = internal[0].start_cc
        dma_end_cc = internal[-1].end_cc
    else:
        dma_start_cc = compute_start
        dma_end_cc = compute_start
    timing = PhaseTiming(
        cluster=cluster,
        phase_name=phase.name,
        release_cc=release_cc,
        first_prefetch_release_cc=first_release_cc,
        first_dma_start_cc=first.start_cc,
        first_dma_end_cc=first.end_cc,
        dma_start_cc=dma_start_cc,
        dma_end_cc=dma_end_cc,
        compute_start_cc=compute_start,
        last_compute_start_cc=compute_starts[-1],
        end_cc=end_cc,
        binding=plan.dma,
        compute_block_cc=compute_cc,
        load_block_cc=load_cc,
        pipeline_stall_cc=pipeline_stall,
        resource_wait_cc=resource_wait,
        first_resource_wait_cc=first.start_cc - first_release_cc,
        fill_stall_cc=compute_start - release_cc,
    )
    return timing, tuple(internal), lanes


def evaluate_action(
    plan: MacroActionPlan,
    *,
    cluster_free_cc: tuple[int, int] = (0, 0),
    lane_state: LaneState = LaneState(),
    state: MacroScheduleState | None = None,
    phases: tuple[MacroPhaseSpec, MacroPhaseSpec] | None = None,
    exhaustive_service_orders: bool = False,
    service_order_mode: str = "best18",
    forced_service_order: tuple[tuple[int, str], ...] | None = None,
) -> MacroActionTiming:
    """Choose the best finite joint service mode for one macro action.

    The public action remains phase-granular.  Internally, a fixed recurrence
    schedules each phase's compile-time block count so holes left by M8/single
    can serve the other cluster.  This is not a block candidate search: block
    order and ping/pong successors are fixed, and only the finite phase-level
    C0/C1 priority mode is compared.
    """

    gate_up_spec, down_spec = phases or default_phases()
    if state is not None and (
        cluster_free_cc != (0, 0) or lane_state != LaneState()
    ):
        raise ValueError("pass either state or legacy cluster/lane arguments")
    initial_state = state or MacroScheduleState(
        cluster_free_cc=cluster_free_cc,
        prefetch_release_cc=cluster_free_cc,
        lane_state=lane_state,
    )
    tasks = {task.cluster: task for task in plan.tasks}
    best: MacroActionTiming | None = None
    clusters = tuple(sorted(tasks))
    if service_order_mode not in (
        "best18",
        "fixed_c0",
        "binding_hot",
        "binding_chain",
    ):
        raise ValueError("unknown service-order mode")
    if forced_service_order is not None:
        expected = {
            (cluster, operation)
            for cluster in clusters
            for operation in (
                "gate_up_first",
                "gate_up_stream",
                "down_first",
                "down_stream",
            )
        }
        if (
            len(forced_service_order) != len(expected)
            or set(forced_service_order) != expected
        ):
            raise ValueError("forced service order is not a task-operation permutation")
        orders = (forced_service_order,)
    elif exhaustive_service_orders and service_order_mode != "best18":
        raise ValueError("exhaustive and fixed service-order modes conflict")
    elif exhaustive_service_orders:
        orders = _exhaustive_phase_orders(clusters)
    elif service_order_mode == "best18":
        orders = _phase_orders(clusters)
    elif service_order_mode == "fixed_c0":
        orders = (_phase_orders(clusters)[0],)
    elif service_order_mode == "binding_hot":
        orders = (_binding_hot_phase_order(tasks, gate_up_spec, down_spec),)
    else:
        orders = (_binding_chain_phase_order(tasks, gate_up_spec, down_spec),)
    for order in orders:
        candidate = _evaluate_joint_order(
            plan=plan,
            tasks=tasks,
            order=order,
            initial_state=initial_state,
            gate_up_spec=gate_up_spec,
            down_spec=down_spec,
        )
        validate_action_timing(candidate)
        candidate = MacroActionTiming(
            **{**candidate.__dict__, "history_validated": True}
        )
        rank = (
            candidate.makespan_cc,
            candidate.resource_stall_cc + candidate.pipeline_stall_cc,
            candidate.service_order,
        )
        if best is None or rank < (
            best.makespan_cc,
            best.resource_stall_cc + best.pipeline_stall_cc,
            best.service_order,
        ):
            best = candidate
    if best is None:
        raise ValueError("macro action has no legal phase-service order")
    return best


def _evaluate_joint_order(
    *,
    plan: MacroActionPlan,
    tasks: dict[int, MacroTaskPlan],
    order: tuple[tuple[int, str], ...],
    initial_state: MacroScheduleState,
    gate_up_spec: MacroPhaseSpec,
    down_spec: MacroPhaseSpec,
) -> MacroActionTiming:
    """Evaluate one joint priority mode with fixed successor recurrences."""

    order_rank = {operation: rank for rank, operation in enumerate(order)}
    lanes = initial_state.lane_state
    streams: dict[
        int, list[tuple[str, MacroPhaseSpec, PhasePlan, int]]
    ] = {}
    for cluster, task in tasks.items():
        stream = []
        for phase_name, phase_spec, phase_plan in (
            ("gate_up", gate_up_spec, task.gate_up),
            ("down", down_spec, task.down),
        ):
            if phase_plan.dma not in legal_bindings(
                task.expert_slice.ntokens, phase_plan.shape, phase_spec
            ):
                raise ValueError(
                    f"dominated/unsupported {phase_plan.shape.name.value}+"
                    f"{phase_plan.dma.name} mode"
                )
            stream.extend(
                (phase_name, phase_spec, phase_plan, block_id)
                for block_id in range(phase_spec.block_count)
            )
        streams[cluster] = stream

    next_load = {cluster: 0 for cluster in tasks}
    next_compute = {cluster: 0 for cluster in tasks}
    loads: dict[int, list[DmaInterval]] = {cluster: [] for cluster in tasks}
    load_releases: dict[int, list[int]] = {cluster: [] for cluster in tasks}
    compute_starts: dict[int, list[int]] = {cluster: [] for cluster in tasks}
    compute_ends: dict[int, list[int]] = {cluster: [] for cluster in tasks}
    gate_counts = {cluster: gate_up_spec.block_count for cluster in tasks}
    running_load_end: dict[int, int | None] = {cluster: None for cluster in tasks}
    running_compute_end: dict[int, int | None] = {
        cluster: None for cluster in tasks
    }
    lane_busy_until = {
        0: initial_state.lane_state.idma_free_cc,
        1: initial_state.lane_state.xdma_free_cc,
    }

    def load_release(cluster: int, index: int) -> int | None:
        phase_name, _, _, block_id = streams[cluster][index]
        gate_count = gate_counts[cluster]
        if index == 0:
            return initial_state.prefetch_release_cc[cluster]
        if index < gate_count:
            dependency = 0 if block_id == 1 else index - 2
            values = compute_starts if block_id == 1 else compute_ends
            return (
                values[cluster][dependency]
                if dependency < len(values[cluster])
                else None
            )
        if index == gate_count:
            dependency = gate_count - 1
            return (
                compute_starts[cluster][dependency]
                if dependency < len(compute_starts[cluster])
                else None
            )
        dependency = gate_count if block_id == 1 else index - 2
        values = compute_starts if block_id == 1 else compute_ends
        return (
            values[cluster][dependency]
            if dependency < len(values[cluster])
            else None
        )

    now = min(initial_state.prefetch_release_cc[cluster] for cluster in tasks)
    total_items = sum(len(stream) for stream in streams.values())
    while sum(next_compute.values()) < total_items:
        for cluster in tasks:
            if running_load_end[cluster] == now:
                running_load_end[cluster] = None
            if running_compute_end[cluster] == now:
                running_compute_end[cluster] = None

        for cluster in sorted(tasks):
            index = next_compute[cluster]
            if (
                running_compute_end[cluster] is not None
                or index >= len(streams[cluster])
                or index >= len(loads[cluster])
                or loads[cluster][index].end_cc > now
            ):
                continue
            dependency_end = (
                initial_state.cluster_free_cc[cluster]
                if index == 0
                else compute_ends[cluster][index - 1]
            )
            if dependency_end > now:
                continue
            _, phase_spec, phase_plan, _ = streams[cluster][index]
            compute_cc = compute_block_cc(
                tasks[cluster].expert_slice.ntokens,
                phase_plan.shape,
                phase_spec,
            )
            compute_starts[cluster].append(now)
            compute_ends[cluster].append(now + compute_cc)
            running_compute_end[cluster] = now + compute_cc
            next_compute[cluster] += 1

        pending = []
        for cluster in sorted(tasks):
            index = next_load[cluster]
            if (
                running_load_end[cluster] is not None
                or index >= len(streams[cluster])
            ):
                continue
            release = load_release(cluster, index)
            if release is None or release > now:
                continue
            phase_name, phase_spec, phase_plan, block_id = streams[cluster][index]
            required = []
            if phase_plan.dma & DmaBinding.IDMA:
                required.append(0)
            if phase_plan.dma & DmaBinding.XDMA:
                required.append(1)
            if any(lane_busy_until[lane] > now for lane in required):
                continue
            deadline = (
                initial_state.cluster_free_cc[cluster]
                if index == 0
                else compute_ends[cluster][index - 1]
            )
            operation = (
                cluster,
                f"{phase_name}_{'first' if block_id == 0 else 'stream'}",
            )
            pending.append(
                (deadline, order_rank[operation], cluster, release, required)
            )
        pending.sort()
        for _, _, cluster, release, required in pending:
            if any(lane_busy_until[lane] > now for lane in required):
                continue
            index = next_load[cluster]
            phase_name, phase_spec, phase_plan, block_id = streams[cluster][index]
            duration = dma_duration(phase_spec.weight_block_bytes, phase_plan.dma)
            interval = DmaInterval(
                cluster=cluster,
                phase_name=phase_name,
                role="first_block" if block_id == 0 else f"block_{block_id}",
                start_cc=now,
                end_cc=now + duration,
                binding=phase_plan.dma,
            )
            loads[cluster].append(interval)
            load_releases[cluster].append(release)
            running_load_end[cluster] = interval.end_cc
            next_load[cluster] += 1
            for lane in required:
                lane_busy_until[lane] = interval.end_cc

        events = []
        events.extend(
            value
            for value in (*running_load_end.values(), *running_compute_end.values())
            if value is not None and value > now
        )
        events.extend(value for value in lane_busy_until.values() if value > now)
        for cluster in tasks:
            if next_load[cluster] < len(streams[cluster]):
                release = load_release(cluster, next_load[cluster])
                if release is not None and release > now:
                    events.append(release)
            if next_compute[cluster] == 0:
                cluster_ready = initial_state.cluster_free_cc[cluster]
                if cluster_ready > now:
                    events.append(cluster_ready)
        if sum(next_compute.values()) == total_items:
            break
        if not events:
            raise RuntimeError("joint macro recurrence deadlocked")
        now = min(events)

    lanes = LaneState(
        idma_free_cc=lane_busy_until[0],
        xdma_free_cc=lane_busy_until[1],
    )

    phase_timings: dict[tuple[int, str], PhaseTiming] = {}
    all_intervals: list[DmaInterval] = []
    for cluster, task in tasks.items():
        gate_count = gate_counts[cluster]
        for phase_name, phase_spec, phase_plan, start_index in (
            ("gate_up", gate_up_spec, task.gate_up, 0),
            ("down", down_spec, task.down, gate_count),
        ):
            stop_index = start_index + phase_spec.block_count
            phase_loads = loads[cluster][start_index:stop_index]
            releases = load_releases[cluster][start_index:stop_index]
            starts = compute_starts[cluster][start_index:stop_index]
            ends = compute_ends[cluster][start_index:stop_index]
            release_cc = (
                initial_state.cluster_free_cc[cluster]
                if phase_name == "gate_up"
                else compute_ends[cluster][gate_count - 1]
            )
            pipeline_stall = sum(
                starts[index] - ends[index - 1]
                for index in range(1, len(starts))
            )
            internal = phase_loads[1:]
            timing = PhaseTiming(
                cluster=cluster,
                phase_name=phase_name,
                release_cc=release_cc,
                first_prefetch_release_cc=releases[0],
                first_dma_start_cc=phase_loads[0].start_cc,
                first_dma_end_cc=phase_loads[0].end_cc,
                dma_start_cc=(
                    internal[0].start_cc if internal else starts[0]
                ),
                dma_end_cc=(internal[-1].end_cc if internal else starts[0]),
                compute_start_cc=starts[0],
                last_compute_start_cc=starts[-1],
                end_cc=ends[-1],
                binding=phase_plan.dma,
                compute_block_cc=ends[0] - starts[0],
                load_block_cc=phase_loads[0].end_cc
                - phase_loads[0].start_cc,
                pipeline_stall_cc=pipeline_stall,
                resource_wait_cc=sum(
                    interval.start_cc - load_release
                    for interval, load_release in zip(internal, releases[1:])
                ),
                first_resource_wait_cc=phase_loads[0].start_cc - releases[0],
                fill_stall_cc=starts[0] - release_cc,
            )
            phase_timings[(cluster, phase_name)] = timing
            all_intervals.extend(phase_loads)

    task_timings = tuple(
        MacroTaskTiming(
            plan=tasks[cluster],
            gate_up=phase_timings[(cluster, "gate_up")],
            down=phase_timings[(cluster, "down")],
        )
        for cluster in sorted(tasks)
    )
    next_cluster_free = list(initial_state.cluster_free_cc)
    next_prefetch_release = list(initial_state.prefetch_release_cc)
    for timing in task_timings:
        cluster = timing.plan.cluster
        next_cluster_free[cluster] = timing.down.end_cc
        next_prefetch_release[cluster] = timing.down.last_compute_start_cc
    next_state = MacroScheduleState(
        cluster_free_cc=tuple(next_cluster_free),
        prefetch_release_cc=tuple(next_prefetch_release),
        lane_state=lanes,
    )
    return MacroActionTiming(
        plan=plan,
        task_timings=task_timings,
        service_order=order,
        lane_state=lanes,
        next_state=next_state,
        dma_intervals=tuple(all_intervals),
        makespan_cc=max(next_cluster_free),
        resource_stall_cc=sum(
            phase.resource_wait_cc + phase.first_resource_wait_cc
            for timing in task_timings
            for phase in (timing.gate_up, timing.down)
        ),
        pipeline_stall_cc=sum(
            phase.pipeline_stall_cc
            for timing in task_timings
            for phase in (timing.gate_up, timing.down)
        ),
        history_validated=False,
    )


def validate_action_timing(result: MacroActionTiming) -> None:
    intervals: dict[DmaBinding, list[tuple[int, int, str]]] = {
        DmaBinding.IDMA: [],
        DmaBinding.XDMA: [],
    }
    for task in result.task_timings:
        if task.down.release_cc < task.gate_up.end_cc:
            raise AssertionError("Down phase starts before Gate/Up completes")
        for phase in (task.gate_up, task.down):
            if phase.first_dma_start_cc < phase.first_prefetch_release_cc:
                raise AssertionError("first-block DMA starts before prefetch release")
            if phase.dma_start_cc < phase.release_cc:
                raise AssertionError("DMA starts before its phase is released")
            if phase.compute_start_cc < phase.release_cc:
                raise AssertionError("compute starts before its phase is released")
            if phase.compute_start_cc < phase.first_dma_end_cc:
                raise AssertionError("compute starts before block 0 is loaded")
            if phase.last_compute_start_cc + phase.compute_block_cc != phase.end_cc:
                raise AssertionError("invalid final compute window")
    for interval in result.dma_intervals:
        if interval.start_cc > interval.end_cc:
            raise AssertionError("negative DMA interval")
        for lane in SINGLE_BINDINGS:
            if interval.binding & lane:
                intervals[lane].append(
                    (
                        interval.start_cc,
                        interval.end_cc,
                        f"c{interval.cluster}:{interval.phase_name}:{interval.role}",
                    )
                )
    for lane, records in intervals.items():
        records.sort()
        for left, right in zip(records, records[1:]):
            if left[1] > right[0]:
                raise AssertionError(
                    f"{lane.name} overlap: {left[2]} and {right[2]}"
                )
