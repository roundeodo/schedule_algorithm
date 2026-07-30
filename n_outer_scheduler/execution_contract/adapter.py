#!/usr/bin/env python3
"""Narrow candidate adapter for the block-major execution contract.

PAIR/SINGLE/SPLIT may be used by a search policy to construct the two lists,
but action boundaries never cross into execution.  Only the completed ordered
lists are adapted here.
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
    """Adapt only a completed candidate; no planning epoch is preserved."""

    return GroupPlan(
        _adapt_sequence(cluster0),
        _adapt_sequence(cluster1),
        int(group_id),
    )
