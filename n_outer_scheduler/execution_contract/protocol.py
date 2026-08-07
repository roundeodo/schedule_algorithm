#!/usr/bin/env python3
"""Compact 64-bit RTL output protocol for a multi-slot N-outer plan.

The stream contains one schedule header, one slot header per slot, and the
slot's C0 slices followed by its C1 slices.  It never contains phase/block
LOAD or COMPUTE events.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag

from .model import ExpertSlice, GroupPlan, SchedulePlan


PROTOCOL_VERSION = 1
DMA_POLICY_VERSION = 1


class RecordKind(IntEnum):
    SCHEDULE = 0
    SLOT = 1
    SLICE = 2


class ScheduleFlags(IntFlag):
    NO_GLOBAL_SLOT_BARRIER = 1 << 0
    FIXED_M4_M2_LOWERING = 1 << 1


REQUIRED_FLAGS = (
    ScheduleFlags.NO_GLOBAL_SLOT_BARRIER
    | ScheduleFlags.FIXED_M4_M2_LOWERING
)


@dataclass(frozen=True)
class SchedulerWordStream:
    words: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.words or any(word < 0 or word >= 1 << 64 for word in self.words):
            raise ValueError("scheduler stream requires nonempty 64-bit words")


def _kind(word: int) -> RecordKind:
    try:
        return RecordKind((word >> 62) & 0x3)
    except ValueError as error:
        raise ValueError("reserved scheduler record kind") from error


def _pack_schedule(plan: SchedulePlan, total_slices: int) -> int:
    if len(plan.slots) > 0xFF or total_slices > 0x3FF:
        raise ValueError("schedule exceeds compact protocol capacity")
    if plan.schedule_id > 0xFFFF:
        raise ValueError("schedule_id exceeds 16 bits")
    word = int(RecordKind.SCHEDULE) << 62
    word |= PROTOCOL_VERSION << 58
    word |= len(plan.slots) << 50
    word |= total_slices << 40
    word |= DMA_POLICY_VERSION << 36
    word |= int(REQUIRED_FLAGS) << 28
    word |= plan.schedule_id << 12
    return word


def _pack_slot(slot: GroupPlan, ordinal: int, *, last: bool) -> int:
    if ordinal > 0xFF or slot.group_id > 0xFF:
        raise ValueError("slot ordinal/group_id exceeds eight bits")
    if len(slot.cluster0) > 0xFF or len(slot.cluster1) > 0xFF:
        raise ValueError("slot slice count exceeds eight bits")
    word = int(RecordKind.SLOT) << 62
    word |= ordinal << 54
    word |= slot.group_id << 46
    word |= len(slot.cluster0) << 38
    word |= len(slot.cluster1) << 30
    word |= int(last) << 29
    return word


def _pack_slice(
    item: ExpertSlice, *, cluster: int, slot_ordinal: int, local_index: int
) -> int:
    if cluster not in (0, 1):
        raise ValueError("invalid cluster")
    if item.eid >= 64 or item.token_start >= 256 or item.ntokens > 256:
        raise ValueError("slice exceeds compact protocol fields")
    if slot_ordinal > 0xFF or local_index > 0xFF:
        raise ValueError("slice ordinal exceeds eight bits")
    word = int(RecordKind.SLICE) << 62
    word |= cluster << 61
    word |= item.eid << 55
    word |= item.token_start << 47
    word |= (item.ntokens - 1) << 39
    word |= slot_ordinal << 31
    word |= local_index << 23
    return word


def emit_scheduler_words(plan: SchedulePlan) -> SchedulerWordStream:
    total_slices = sum(
        len(slot.cluster0) + len(slot.cluster1) for slot in plan.slots
    )
    words = [_pack_schedule(plan, total_slices)]
    for ordinal, slot in enumerate(plan.slots):
        words.append(_pack_slot(slot, ordinal, last=ordinal + 1 == len(plan.slots)))
        words.extend(
            _pack_slice(item, cluster=0, slot_ordinal=ordinal, local_index=index)
            for index, item in enumerate(slot.cluster0)
        )
        words.extend(
            _pack_slice(item, cluster=1, slot_ordinal=ordinal, local_index=index)
            for index, item in enumerate(slot.cluster1)
        )
    return SchedulerWordStream(tuple(words))


def _decode_schedule(word: int) -> tuple[int, int, int]:
    if _kind(word) != RecordKind.SCHEDULE or word & 0xFFF:
        raise ValueError("invalid schedule header/reserved bits")
    version = (word >> 58) & 0xF
    slot_count = (word >> 50) & 0xFF
    total_slices = (word >> 40) & 0x3FF
    dma_policy = (word >> 36) & 0xF
    flags = ScheduleFlags((word >> 28) & 0xFF)
    schedule_id = (word >> 12) & 0xFFFF
    if version != PROTOCOL_VERSION or dma_policy != DMA_POLICY_VERSION:
        raise ValueError("unsupported N-outer protocol/policy version")
    if flags != REQUIRED_FLAGS:
        raise ValueError("unsupported N-outer schedule flags")
    if slot_count == 0 or total_slices == 0:
        raise ValueError("empty compact schedule")
    return schedule_id, slot_count, total_slices


def _decode_slot(word: int, expected_ordinal: int, expected_last: bool) -> tuple[int, int, int]:
    if _kind(word) != RecordKind.SLOT or word & ((1 << 29) - 1):
        raise ValueError("invalid slot record/reserved bits")
    ordinal = (word >> 54) & 0xFF
    group_id = (word >> 46) & 0xFF
    c0_count = (word >> 38) & 0xFF
    c1_count = (word >> 30) & 0xFF
    last = bool((word >> 29) & 1)
    if ordinal != expected_ordinal or last != expected_last:
        raise ValueError("slot order/last marker mismatch")
    if c0_count + c1_count == 0:
        raise ValueError("empty slot record")
    return group_id, c0_count, c1_count


def _decode_slice(
    word: int, *, expected_cluster: int, expected_slot: int, expected_index: int
) -> ExpertSlice:
    if _kind(word) != RecordKind.SLICE or word & ((1 << 23) - 1):
        raise ValueError("invalid slice record/reserved bits")
    cluster = (word >> 61) & 1
    eid = (word >> 55) & 0x3F
    token_start = (word >> 47) & 0xFF
    ntokens = ((word >> 39) & 0xFF) + 1
    slot_ordinal = (word >> 31) & 0xFF
    local_index = (word >> 23) & 0xFF
    if (
        cluster != expected_cluster
        or slot_ordinal != expected_slot
        or local_index != expected_index
    ):
        raise ValueError("slice order metadata mismatch")
    return ExpertSlice(eid, token_start, ntokens)


def decode_scheduler_words(stream: SchedulerWordStream) -> SchedulePlan:
    schedule_id, slot_count, total_slices = _decode_schedule(stream.words[0])
    cursor = 1
    slots: list[GroupPlan] = []
    decoded_slices = 0
    for ordinal in range(slot_count):
        if cursor >= len(stream.words):
            raise ValueError("truncated slot stream")
        group_id, c0_count, c1_count = _decode_slot(
            stream.words[cursor], ordinal, ordinal + 1 == slot_count
        )
        cursor += 1
        clusters: list[list[ExpertSlice]] = [[], []]
        for cluster, count in ((0, c0_count), (1, c1_count)):
            for local_index in range(count):
                if cursor >= len(stream.words):
                    raise ValueError("truncated slice stream")
                clusters[cluster].append(
                    _decode_slice(
                        stream.words[cursor],
                        expected_cluster=cluster,
                        expected_slot=ordinal,
                        expected_index=local_index,
                    )
                )
                cursor += 1
                decoded_slices += 1
        slots.append(GroupPlan(tuple(clusters[0]), tuple(clusters[1]), group_id))
    if cursor != len(stream.words) or decoded_slices != total_slices:
        raise ValueError("scheduler stream length/count mismatch")
    return SchedulePlan(tuple(slots), schedule_id)
