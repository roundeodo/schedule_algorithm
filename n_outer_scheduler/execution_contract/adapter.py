#!/usr/bin/env python3
"""Narrow one-slot candidate adapter for the block-major contract.

PAIR/SINGLE/SPLIT may construct the two lists inside one slot.  Search-only
action bookkeeping does not cross into execution, but selected slot boundaries
are preserved by :class:`SchedulePlan`.  This helper adapts exactly one slot.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from .model import ExpertSlice, GroupPlan


class SliceLike(Protocol):
    eid: int
    token_start: int
    ntokens: int


def _adapt_sequence(sequence: Iterable[SliceLike]) -> tuple[ExpertSlice, ...]:
    return tuple(
        ExpertSlice(int(item.eid), int(item.token_start), int(item.ntokens))
        for item in sequence
    )


def adapt_completed_candidate(
    cluster0: Iterable[SliceLike],
    cluster1: Iterable[SliceLike],
    *,
    group_id: int = 0,
) -> GroupPlan:
    """Adapt one completed slot; no search-internal action is preserved."""

    return GroupPlan(
        _adapt_sequence(cluster0),
        _adapt_sequence(cluster1),
        int(group_id),
    )
