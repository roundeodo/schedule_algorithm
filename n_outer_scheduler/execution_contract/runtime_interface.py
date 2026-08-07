#!/usr/bin/env python3
"""CVA6-style lowering from semantic slots to compact Bingo runtime tables."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .model import ExpertSlice, GroupPlan, SchedulePlan
from .protocol import PROTOCOL_VERSION, REQUIRED_FLAGS


class WorkerRole(IntEnum):
    DMA_SLOT_WORKER = 0
    COMPUTE_SLOT_WORKER = 1


@dataclass(frozen=True)
class RuntimeLayout:
    max_tokens_per_expert: int = 256
    slot_token_capacity: int = 256
    input_row_bytes: int = 2048 * 2
    intermediate_row_bytes: int = 1408 * 2
    output_row_bytes: int = 2048 * 2

    def __post_init__(self) -> None:
        if min(
            self.max_tokens_per_expert,
            self.slot_token_capacity,
            self.input_row_bytes,
            self.intermediate_row_bytes,
            self.output_row_bytes,
        ) <= 0:
            raise ValueError("runtime layout constants must be positive")


@dataclass(frozen=True)
class RuntimeScheduleHeader:
    version: int
    schedule_id: int
    slot_count: int
    total_slice_count: int
    flags: int


@dataclass(frozen=True)
class RuntimeSlotDesc:
    slot_ordinal: int
    group_id: int
    c0_first_slice: int
    c0_slice_count: int
    c1_first_slice: int
    c1_slice_count: int
    c0_token_count: int
    c1_token_count: int


@dataclass(frozen=True)
class RuntimeSliceDesc:
    slot_ordinal: int
    cluster: int
    local_index: int
    eid: int
    token_start: int
    ntokens: int
    token_ref_start: int
    input_l1_offset: int
    intermediate_l1_offset: int
    output_l1_offset: int
    m4_iters: int
    m2_iters: int
    m4_tail_valid_tokens: int
    m2_valid_tokens: int


@dataclass(frozen=True)
class RuntimeScheduleTables:
    header: RuntimeScheduleHeader
    slots: tuple[RuntimeSlotDesc, ...]
    slices: tuple[RuntimeSliceDesc, ...]


@dataclass(frozen=True)
class NOuterWorkerArgs:
    schedule_header_addr: int
    static_context_addr: int
    runtime_sync_addr: int
    cluster_id: int
    worker_role: WorkerRole

    def __post_init__(self) -> None:
        if self.cluster_id not in (0, 1):
            raise ValueError("worker cluster_id must be zero or one")
        if min(
            self.schedule_header_addr,
            self.static_context_addr,
            self.runtime_sync_addr,
        ) < 0:
            raise ValueError("worker addresses must be non-negative")


def _lower_cluster_slices(
    *,
    slot_ordinal: int,
    cluster: int,
    items: tuple[ExpertSlice, ...],
    layout: RuntimeLayout,
    destination: list[RuntimeSliceDesc],
) -> tuple[int, int]:
    first = len(destination)
    token_cursor = 0
    for local_index, item in enumerate(items):
        if item.token_end > layout.max_tokens_per_expert:
            raise ValueError("slice exceeds max_tokens_per_expert")
        destination.append(
            RuntimeSliceDesc(
                slot_ordinal=slot_ordinal,
                cluster=cluster,
                local_index=local_index,
                eid=item.eid,
                token_start=item.token_start,
                ntokens=item.ntokens,
                token_ref_start=(
                    item.eid * layout.max_tokens_per_expert + item.token_start
                ),
                input_l1_offset=token_cursor * layout.input_row_bytes,
                intermediate_l1_offset=(
                    token_cursor * layout.intermediate_row_bytes
                ),
                output_l1_offset=token_cursor * layout.output_row_bytes,
                m4_iters=item.m4_iters,
                m2_iters=item.m2_iters,
                m4_tail_valid_tokens=item.m4_tail_valid_tokens,
                m2_valid_tokens=item.m2_valid_tokens,
            )
        )
        token_cursor += item.ntokens
    if token_cursor > layout.slot_token_capacity:
        raise ValueError("slot side exceeds cluster workspace token capacity")
    return first, token_cursor


def lower_runtime_tables(
    plan: SchedulePlan, *, layout: RuntimeLayout = RuntimeLayout()
) -> RuntimeScheduleTables:
    slices: list[RuntimeSliceDesc] = []
    slots: list[RuntimeSlotDesc] = []
    for slot_ordinal, slot in enumerate(plan.slots):
        c0_first, c0_tokens = _lower_cluster_slices(
            slot_ordinal=slot_ordinal,
            cluster=0,
            items=slot.cluster0,
            layout=layout,
            destination=slices,
        )
        c1_first, c1_tokens = _lower_cluster_slices(
            slot_ordinal=slot_ordinal,
            cluster=1,
            items=slot.cluster1,
            layout=layout,
            destination=slices,
        )
        slots.append(
            RuntimeSlotDesc(
                slot_ordinal=slot_ordinal,
                group_id=slot.group_id,
                c0_first_slice=c0_first,
                c0_slice_count=len(slot.cluster0),
                c1_first_slice=c1_first,
                c1_slice_count=len(slot.cluster1),
                c0_token_count=c0_tokens,
                c1_token_count=c1_tokens,
            )
        )
    return RuntimeScheduleTables(
        header=RuntimeScheduleHeader(
            version=PROTOCOL_VERSION,
            schedule_id=plan.schedule_id,
            slot_count=len(plan.slots),
            total_slice_count=len(slices),
            flags=int(REQUIRED_FLAGS),
        ),
        slots=tuple(slots),
        slices=tuple(slices),
    )


def decode_runtime_tables(tables: RuntimeScheduleTables) -> SchedulePlan:
    header = tables.header
    if (
        header.version != PROTOCOL_VERSION
        or header.flags != int(REQUIRED_FLAGS)
        or header.slot_count != len(tables.slots)
        or header.total_slice_count != len(tables.slices)
    ):
        raise ValueError("runtime schedule header mismatch")
    slots: list[GroupPlan] = []
    for expected_ordinal, slot in enumerate(tables.slots):
        if slot.slot_ordinal != expected_ordinal:
            raise ValueError("runtime slot order mismatch")
        clusters: list[tuple[ExpertSlice, ...]] = []
        for cluster, first, count in (
            (0, slot.c0_first_slice, slot.c0_slice_count),
            (1, slot.c1_first_slice, slot.c1_slice_count),
        ):
            selected = tables.slices[first : first + count]
            if len(selected) != count:
                raise ValueError("runtime slice range is truncated")
            for local_index, item in enumerate(selected):
                if (
                    item.slot_ordinal != expected_ordinal
                    or item.cluster != cluster
                    or item.local_index != local_index
                ):
                    raise ValueError("runtime slice metadata mismatch")
            clusters.append(
                tuple(
                    ExpertSlice(item.eid, item.token_start, item.ntokens)
                    for item in selected
                )
            )
        slots.append(GroupPlan(clusters[0], clusters[1], slot.group_id))
    return SchedulePlan(tuple(slots), header.schedule_id)
