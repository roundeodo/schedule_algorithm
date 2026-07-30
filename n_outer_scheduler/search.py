#!/usr/bin/env python3
"""Candidate generation and ranking for the independent N-outer model."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from typing import Iterable

from .model import (
    CandidateResult,
    ExpertDescriptor,
    GroupDescriptor,
    NOuterSimulator,
    ScheduleCandidate,
)


@dataclass(frozen=True)
class SearchResult:
    best: CandidateResult
    ranked: tuple[CandidateResult, ...]
    generated_candidates: int
    rejected_candidates: int


def _sequence_variants(
    experts: tuple[ExpertDescriptor, ...],
) -> tuple[tuple[ExpertDescriptor, ...], ...]:
    if len(experts) <= 1:
        return (experts,)
    variants = (
        experts,
        tuple(sorted(experts, key=lambda expert: (-expert.ntokens, expert.eid))),
        tuple(sorted(experts, key=lambda expert: (expert.ntokens, expert.eid))),
    )
    return tuple(dict.fromkeys(variants))


def generate_partition_candidates(
    experts: Iterable[ExpertDescriptor],
    *,
    exhaustive_limit: int = 16,
    exhaustive_permutation_limit: int = 0,
) -> tuple[ScheduleCandidate, ...]:
    """Enumerate cluster partitions and three deterministic orderings per side.

    Cluster symmetry is removed by pinning the first (largest) expert to C0.
    This is an exhaustive partition search up to ``exhaustive_limit`` experts,
    but deliberately not an exhaustive permutation search.
    """

    ordered = tuple(sorted(experts, key=lambda expert: (-expert.ntokens, expert.eid)))
    if not ordered:
        raise ValueError("at least one expert is required")
    if exhaustive_permutation_limit < 0:
        raise ValueError("exhaustive_permutation_limit must be non-negative")
    if len(ordered) > exhaustive_limit:
        return _greedy_candidates(ordered)

    if len(ordered) <= exhaustive_permutation_limit:
        return _exhaustive_ordered_partition_candidates(ordered)

    candidates: list[ScheduleCandidate] = []
    tail = ordered[1:]
    for mask in range(1 << len(tail)):
        c0 = [ordered[0]]
        c1: list[ExpertDescriptor] = []
        for bit, expert in enumerate(tail):
            (c0 if mask & (1 << bit) else c1).append(expert)
        for seq0, seq1 in product(_sequence_variants(tuple(c0)), _sequence_variants(tuple(c1))):
            label = (
                f"c0={','.join(str(expert.eid) for expert in seq0)};"
                f"c1={','.join(str(expert.eid) for expert in seq1)}"
            )
            candidates.append(
                ScheduleCandidate(GroupDescriptor(seq0, seq1), label=label)
            )
    unique = {candidate.label: candidate for candidate in candidates}
    return tuple(unique.values())


def _exhaustive_ordered_partition_candidates(
    ordered: tuple[ExpertDescriptor, ...],
) -> tuple[ScheduleCandidate, ...]:
    """Enumerate every ordered two-cluster partition up to cluster symmetry.

    The largest expert is assigned to C0 to remove a global C0/C1 swap.  It is
    still permuted to every position inside C0, so within-cluster order is
    complete.  This mode is intended only for small diagnostic comparisons.
    """

    anchor, tail = ordered[0], ordered[1:]
    candidates: list[ScheduleCandidate] = []
    for mask in range(1 << len(tail)):
        c0_members = (anchor,) + tuple(
            expert for bit, expert in enumerate(tail) if mask & (1 << bit)
        )
        c1_members = tuple(
            expert for bit, expert in enumerate(tail) if not mask & (1 << bit)
        )
        c1_orders = permutations(c1_members) if c1_members else ((),)
        c1_orders = tuple(c1_orders)
        for seq0 in permutations(c0_members):
            for seq1 in c1_orders:
                label = (
                    f"c0={','.join(str(expert.eid) for expert in seq0)};"
                    f"c1={','.join(str(expert.eid) for expert in seq1)}"
                )
                candidates.append(
                    ScheduleCandidate(GroupDescriptor(tuple(seq0), tuple(seq1)), label=label)
                )
    return tuple(candidates)


def _greedy_candidates(
    experts: tuple[ExpertDescriptor, ...],
) -> tuple[ScheduleCandidate, ...]:
    candidates: list[ScheduleCandidate] = []
    for reverse_small in (False, True):
        c0: list[ExpertDescriptor] = []
        c1: list[ExpertDescriptor] = []
        loads = [0, 0]
        source = experts if not reverse_small else (experts[0], *reversed(experts[1:]))
        for expert in source:
            cluster = 0 if loads[0] <= loads[1] else 1
            (c0 if cluster == 0 else c1).append(expert)
            loads[cluster] += (expert.ntokens + 1) // 2
        label = (
            f"greedy-c0={','.join(str(expert.eid) for expert in c0)};"
            f"c1={','.join(str(expert.eid) for expert in c1)}"
        )
        candidates.append(
            ScheduleCandidate(GroupDescriptor(tuple(c0), tuple(c1)), label=label)
        )
    return tuple(candidates)


def search_candidates(
    simulator: NOuterSimulator,
    candidates: Iterable[ScheduleCandidate],
    *,
    top_k: int = 10,
) -> SearchResult:
    evaluated: list[CandidateResult] = []
    rejected = 0
    limit = simulator.config.max_group_tokens_per_cluster
    for candidate in candidates:
        if limit is not None and any(
            sum(expert.ntokens for expert in candidate.group.experts(cluster)) > limit
            for cluster in (0, 1)
        ):
            rejected += 1
            continue
        evaluated.append(simulator.evaluate(candidate))
    if not evaluated:
        raise ValueError("no candidate satisfies the group-capacity constraint")
    evaluated.sort(key=lambda result: result.rank_key)
    return SearchResult(
        best=evaluated[0],
        ranked=tuple(evaluated[:top_k]),
        generated_candidates=len(evaluated),
        rejected_candidates=rejected,
    )
