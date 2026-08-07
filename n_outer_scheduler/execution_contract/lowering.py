#!/usr/bin/env python3
"""Audit lowering for compact multi-slot N-outer schedules.

The compact scheduler words and runtime tables are the execution interface.
The expanded LOAD/COMPUTE commands remain verification-only dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import (
    DmaMask,
    GroupPlan,
    ItemKey,
    NOuterSimulator,
    Phase,
    SchedulePlan,
    ScheduleResult,
)
from .protocol import (
    SchedulerWordStream,
    decode_scheduler_words,
    emit_scheduler_words,
)
from .runtime_interface import (
    RuntimeLayout,
    RuntimeScheduleTables,
    decode_runtime_tables,
    lower_runtime_tables,
)


@dataclass(frozen=True)
class StaticRunnerTemplate:
    """History-independent Bingo topology.

    LOAD and COMPUTE are separate because they run on different cores and must
    overlap.  Weight blocks are loop iterations, not dynamic Bingo nodes.
    """

    nodes: tuple[str, ...] = (
        "host_prepare_schedule",
        "c0_dma_slot_worker",
        "c0_compute_slot_worker",
        "c1_dma_slot_worker",
        "c1_compute_slot_worker",
        "host_schedule_join",
    )

    @property
    def topology_signature(self) -> tuple[str, ...]:
        return self.nodes


@dataclass(frozen=True)
class RunnerLoadCommand:
    """Expanded verification event; not an RTL word or public SW call."""

    key: ItemKey
    slot_index: int
    phase: Phase
    block_id: int
    sequence_index: int
    eid: int
    token_start: int
    ntokens: int
    buffer_slot: int
    lanes: DmaMask
    duration_ticks: int
    wait_local_load: ItemKey | None
    wait_buffer_compute: ItemKey | None
    wait_lane_load: tuple[ItemKey | None, ItemKey | None]


@dataclass(frozen=True)
class RunnerComputeCommand:
    """Expanded verification event; not an RTL word or public SW call."""

    key: ItemKey
    slot_index: int
    phase: Phase
    block_id: int
    sequence_index: int
    eid: int
    token_start: int
    ntokens: int
    m4_iters: int
    m2_iters: int
    m4_tail_valid_tokens: int
    m2_valid_tokens: int
    buffer_slot: int
    duration_ticks: int
    wait_load: ItemKey
    wait_previous_compute: ItemKey | None


@dataclass(frozen=True)
class StaticRunnerProgram:
    template: StaticRunnerTemplate
    scheduler_stream: SchedulerWordStream
    runtime_tables: RuntimeScheduleTables
    loads: tuple[RunnerLoadCommand, ...]
    computes: tuple[RunnerComputeCommand, ...]
    lane_load_order: tuple[tuple[ItemKey, ...], tuple[ItemKey, ...]]


@dataclass(frozen=True)
class LoweredContract:
    plan: GroupPlan | SchedulePlan
    schedule: ScheduleResult
    runner_program: StaticRunnerProgram


def _lane_orders(schedule: ScheduleResult) -> tuple[tuple[ItemKey, ...], tuple[ItemKey, ...]]:
    orders: list[tuple[ItemKey, ...]] = []
    for lane in (DmaMask.IDMA, DmaMask.XDMA):
        orders.append(
            tuple(
                event.item.key
                for event in sorted(
                    (event for event in schedule.loads if event.lanes & lane),
                    key=lambda event: (
                        event.start_tick,
                        event.end_tick,
                        event.item.key,
                    ),
                )
            )
        )
    return orders[0], orders[1]


def lower_group(
    plan: GroupPlan,
    *,
    simulator: NOuterSimulator | None = None,
    runtime_layout: RuntimeLayout = RuntimeLayout(),
) -> LoweredContract:
    schedule_plan = SchedulePlan((plan,), schedule_id=plan.group_id)
    return _lower_schedule(
        plan,
        schedule_plan,
        simulator=simulator,
        runtime_layout=runtime_layout,
    )


def lower_schedule_plan(
    plan: SchedulePlan,
    *,
    simulator: NOuterSimulator | None = None,
    runtime_layout: RuntimeLayout = RuntimeLayout(),
) -> LoweredContract:
    return _lower_schedule(
        plan,
        plan,
        simulator=simulator,
        runtime_layout=runtime_layout,
    )


def _lower_schedule(
    source_plan: GroupPlan | SchedulePlan,
    schedule_plan: SchedulePlan,
    *,
    simulator: NOuterSimulator | None,
    runtime_layout: RuntimeLayout,
) -> LoweredContract:
    simulator = simulator or NOuterSimulator()
    scheduler_stream = emit_scheduler_words(schedule_plan)
    decoded = decode_scheduler_words(scheduler_stream)
    runtime_tables = lower_runtime_tables(decoded, layout=runtime_layout)
    runtime_decoded = decode_runtime_tables(runtime_tables)
    if decoded != schedule_plan or runtime_decoded != schedule_plan:
        raise AssertionError("compact/runtime lowering changed the schedule plan")
    schedule = simulator.schedule(decoded)
    lane_orders = _lane_orders(schedule)
    lane_predecessor: dict[tuple[ItemKey, int], ItemKey | None] = {}
    for lane, order in enumerate(lane_orders):
        previous: ItemKey | None = None
        for key in order:
            lane_predecessor[(key, lane)] = previous
            previous = key

    load_events = {event.item.key: event for event in schedule.loads}
    compute_events = {event.item.key: event for event in schedule.computes}
    loads: list[RunnerLoadCommand] = []
    computes: list[RunnerComputeCommand] = []
    for cluster, stream in enumerate(schedule.streams):
        for index, item in enumerate(stream):
            load = load_events[item.key]
            compute = compute_events[item.key]
            expert = item.expert_slice
            loads.append(
                RunnerLoadCommand(
                    key=item.key,
                    slot_index=item.slot_index,
                    phase=item.phase,
                    block_id=item.block_id,
                    sequence_index=item.sequence_index,
                    eid=expert.eid,
                    token_start=expert.token_start,
                    ntokens=expert.ntokens,
                    buffer_slot=item.buffer_slot,
                    lanes=load.lanes,
                    duration_ticks=load.end_tick - load.start_tick,
                    wait_local_load=ItemKey(cluster, index - 1) if index else None,
                    wait_buffer_compute=(
                        ItemKey(cluster, index - 2) if index >= 2 else None
                    ),
                    wait_lane_load=tuple(
                        lane_predecessor.get((item.key, lane))
                        if load.lanes & (DmaMask.IDMA if lane == 0 else DmaMask.XDMA)
                        else None
                        for lane in (0, 1)
                    ),
                )
            )
            computes.append(
                RunnerComputeCommand(
                    key=item.key,
                    slot_index=item.slot_index,
                    phase=item.phase,
                    block_id=item.block_id,
                    sequence_index=item.sequence_index,
                    eid=expert.eid,
                    token_start=expert.token_start,
                    ntokens=expert.ntokens,
                    m4_iters=expert.m4_iters,
                    m2_iters=expert.m2_iters,
                    m4_tail_valid_tokens=expert.m4_tail_valid_tokens,
                    m2_valid_tokens=expert.m2_valid_tokens,
                    buffer_slot=item.buffer_slot,
                    duration_ticks=compute.end_tick - compute.start_tick,
                    wait_load=item.key,
                    wait_previous_compute=(
                        ItemKey(cluster, index - 1) if index else None
                    ),
                )
            )

    program = StaticRunnerProgram(
        template=StaticRunnerTemplate(),
        scheduler_stream=scheduler_stream,
        runtime_tables=runtime_tables,
        loads=tuple(loads),
        computes=tuple(computes),
        lane_load_order=lane_orders,
    )
    validate_runner_program(program)
    return LoweredContract(source_plan, schedule, program)


def validate_runner_program(program: StaticRunnerProgram) -> None:
    loads = {command.key: command for command in program.loads}
    computes = {command.key: command for command in program.computes}
    if set(loads) != set(computes):
        raise AssertionError("runner LOAD/COMPUTE key sets differ")
    for command in program.loads:
        if command.duration_ticks <= 0 or command.buffer_slot != command.key.stream_index & 1:
            raise AssertionError("invalid LOAD command")
    for command in program.computes:
        if command.duration_ticks <= 0 or command.wait_load != command.key:
            raise AssertionError("invalid COMPUTE command")
        if command.buffer_slot != command.key.stream_index & 1:
            raise AssertionError("compute uses the wrong ping/pong buffer")
    for lane, order in enumerate(program.lane_load_order):
        previous: ItemKey | None = None
        mask = DmaMask.IDMA if lane == 0 else DmaMask.XDMA
        for key in order:
            if key not in loads or not loads[key].lanes & mask:
                raise AssertionError("lane order references the wrong LOAD")
            if loads[key].wait_lane_load[lane] != previous:
                raise AssertionError("lane predecessor chain is not contiguous")
            previous = key
