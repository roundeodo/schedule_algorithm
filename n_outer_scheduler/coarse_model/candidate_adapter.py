#!/usr/bin/env python3
"""Narrow adapter for injecting external SINGLE/PAIR/SPLIT semantics.

No four-stage timing, shape, cache, S1/S2/S3/S4, or result metadata crosses
this boundary.  Only expert/token slices and cluster assignments are accepted;
all N-outer modes and timing are generated locally.
"""

from __future__ import annotations

from typing import Iterable

from .candidates import CandidateSkeleton, SliceAssignment
from .semantics import ActionKind, ExpertSlice


def inject_candidate(
    kind: str,
    assignments: Iterable[tuple[int, int, int, int]],
) -> CandidateSkeleton:
    """Create a skeleton from ``(cluster, eid, token_start, ntokens)`` tuples."""

    try:
        action_kind = ActionKind(kind.lower())
    except ValueError as error:
        raise ValueError(f"unsupported external action kind {kind!r}") from error
    return CandidateSkeleton(
        action_kind,
        tuple(
            SliceAssignment(
                cluster,
                ExpertSlice(eid, token_start, ntokens),
            )
            for cluster, eid, token_start, ntokens in assignments
        ),
    )

