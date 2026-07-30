#!/usr/bin/env python3
"""Lower a complete block-major group into a fixed Bingo runner program."""

from __future__ import annotations

from dataclasses import dataclass

from .model import (
    DmaMask,
    ExpertSlice,
    GroupPlan,
    ItemKey,
    NOuterSimulator,
    Phase,
    ScheduleResult,
)


@dataclass(frozen=True)
class RtlGroupRecord:
    """One dynamic expert descriptor emitted in cluster-list order."""

    group_last: bool
    cluster_last: bool
    cluster: int
    eid: int
    token_start: int
    ntokens: int

    def __post_init__(self) -> None:
        if self.cluster not in (0, 1):
            raise ValueError("cluster must be zero or one")
        if not 0 <= self.eid < 64:
            raise ValueError("record supports at most 64 experts")
        if not 0 <= self.token_start < 256:
            raise ValueError("token_start does not fit eight bits")
        if not 1 <= self.ntokens <= 256:
            raise ValueError("ntokens does not fit the 1..256 encoding")


@dataclass(frozen=True)
class RtlGroupImage:
    group_id: int
    records: tuple[RtlGroupRecord, ...]


@dataclass(frozen=True)
class StaticRunnerTemplate:
    """History-independent Bingo topology.

    LOAD and COMPUTE are separate because they run on different cores and must
    overlap.  Weight blocks are loop iterations, not dynamic Bingo nodes.
    """

    nodes: tuple[str, ...] = (
        "group_start",
        "c0_load_worker",
        "c0_compute_worker",
        "c1_load_worker",
        "c1_compute_worker",
        "group_join",
    )

    @property
    def topology_signature(self) -> tuple[str, ...]:
        return self.nodes


@dataclass(frozen=True)
class RunnerLoadCommand:
    key: ItemKey
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
    key: ItemKey
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
    image: RtlGroupImage
    loads: tuple[RunnerLoadCommand, ...]
    computes: tuple[RunnerComputeCommand, ...]
    lane_load_order: tuple[tuple[ItemKey, ...], tuple[ItemKey, ...]]


@dataclass(frozen=True)
class LoweredContract:
    plan: GroupPlan
    schedule: ScheduleResult
    runner_program: StaticRunnerProgram


def emit_rtl_group(plan: GroupPlan) -> RtlGroupImage:
    records: list[RtlGroupRecord] = []
    flattened = [
        (cluster, index, expert, len(plan.experts(cluster)))
        for cluster in (0, 1)
        for index, expert in enumerate(plan.experts(cluster))
    ]
    for position, (cluster, index, expert, count) in enumerate(flattened):
        records.append(
            RtlGroupRecord(
                group_last=position + 1 == len(flattened),
                cluster_last=index + 1 == count,
                cluster=cluster,
                eid=expert.eid,
                token_start=expert.token_start,
                ntokens=expert.ntokens,
            )
        )
    return RtlGroupImage(plan.group_id, tuple(records))


def decode_rtl_group(image: RtlGroupImage) -> GroupPlan:
    clusters: list[list[ExpertSlice]] = [[], []]
    if not image.records or not image.records[-1].group_last:
        raise ValueError("RTL group image has no final record")
    if any(record.group_last for record in image.records[:-1]):
        raise ValueError("group_last appears before the final record")
    seen_cluster_last = [False, False]
    for record in image.records:
        if seen_cluster_last[record.cluster]:
            raise ValueError("record follows cluster_last")
        clusters[record.cluster].append(
            ExpertSlice(record.eid, record.token_start, record.ntokens)
        )
        if record.cluster_last:
            seen_cluster_last[record.cluster] = True
    for cluster in (0, 1):
        if clusters[cluster] and not seen_cluster_last[cluster]:
            raise ValueError("nonempty cluster is missing cluster_last")
    return GroupPlan(tuple(clusters[0]), tuple(clusters[1]), image.group_id)


def pack_rtl_record(record: RtlGroupRecord) -> int:
    """Pack the frozen 26-bit descriptor into the low bits of one word."""

    word = int(record.group_last)
    word |= int(record.cluster_last) << 1
    word |= record.cluster << 2
    word |= record.eid << 3
    word |= record.token_start << 9
    word |= (record.ntokens - 1) << 17
    return word


def unpack_rtl_record(word: int) -> RtlGroupRecord:
    if word < 0 or word >> 25:
        raise ValueError("reserved record bits must be zero")
    return RtlGroupRecord(
        group_last=bool(word & 1),
        cluster_last=bool((word >> 1) & 1),
        cluster=(word >> 2) & 1,
        eid=(word >> 3) & 0x3F,
        token_start=(word >> 9) & 0xFF,
        ntokens=((word >> 17) & 0xFF) + 1,
    )


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
    plan: GroupPlan, *, simulator: NOuterSimulator | None = None
) -> LoweredContract:
    simulator = simulator or NOuterSimulator()
    image = emit_rtl_group(plan)
    decoded = decode_rtl_group(image)
    if decoded != plan:
        raise AssertionError("RTL descriptor round trip changed the group plan")
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
        image=image,
        loads=tuple(loads),
        computes=tuple(computes),
        lane_load_order=lane_orders,
    )
    validate_runner_program(program)
    return LoweredContract(plan, schedule, program)


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
