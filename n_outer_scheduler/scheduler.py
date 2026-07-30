#!/usr/bin/env python3
"""Scheduler policy for the independent N-outer static execution model.

The static workers fix phase/block traversal and two-buffer ownership.  The
scheduler selects the two ordered expert lists and, depending on mode, either
uses the fast deadline-aware DMA policy or searches exact per-event prefetch
grants.  The primary cost is the simulated completion time, never a weighted
proxy score.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .model import (
    LANE_BW_BYTES_PER_CC,
    CandidateResult,
    ExpertDescriptor,
    NOuterSimulator,
    ScheduleCandidate,
)
from .reference import ExactDmaPlanner, ExactDmaResult
from .search import generate_partition_candidates, search_candidates


class SchedulerMode(str, Enum):
    FAST = "fast"
    HYBRID = "hybrid"
    REFERENCE = "reference"


@dataclass(frozen=True)
class SchedulerOptions:
    mode: SchedulerMode = SchedulerMode.HYBRID
    exact_top_k: int = 10
    partition_exhaustive_limit: int = 16

    def __post_init__(self) -> None:
        if self.exact_top_k <= 0:
            raise ValueError("exact_top_k must be positive")
        if self.partition_exhaustive_limit <= 0:
            raise ValueError("partition_exhaustive_limit must be positive")


@dataclass(frozen=True)
class CostBreakdown:
    makespan_cc: int
    lower_bound_cc: int
    lower_bound_overhead_cc: int
    initial_prime_cc: int
    steady_stall_cc: int

    @property
    def objective(self) -> tuple[int, int, int]:
        """Lexicographic objective; there are no fitted coefficients."""

        return (
            self.makespan_cc,
            self.steady_stall_cc,
            self.lower_bound_overhead_cc,
        )


@dataclass(frozen=True)
class BandwidthAudit:
    valid: bool
    peak_bandwidth_bytes_per_cc: int
    lane_busy_cc: tuple[int, int]
    transferred_weight_bytes: int


@dataclass(frozen=True)
class PrefetchAudit:
    lookahead_loads: int
    overlapped_loads: int
    fully_hidden_loads: int
    phase_boundary_prefetches: int
    overlap_cc: int
    exposed_load_cc: int


@dataclass(frozen=True)
class SchedulerResult:
    mode: SchedulerMode
    candidate: ScheduleCandidate
    schedule: CandidateResult
    exact: ExactDmaResult | None
    cost: CostBreakdown
    bandwidth: BandwidthAudit
    prefetch: PrefetchAudit
    generated_candidates: int
    evaluated_candidates: int
    rejected_candidates: int
    exact_candidates: int
    dma_optimal_for_selected_candidate: bool
    optimal_within_generated_bank: bool
    partition_space_complete: bool
    permutation_space_complete: bool


class NOuterScheduler:
    """Select a legal N-outer mapping and its DMA-prefetch strategy."""

    def __init__(
        self,
        simulator: NOuterSimulator | None = None,
        options: SchedulerOptions | None = None,
    ):
        self.simulator = simulator or NOuterSimulator()
        self.options = options or SchedulerOptions()

    def schedule(self, experts: Iterable[ExpertDescriptor]) -> SchedulerResult:
        expert_tuple = tuple(experts)
        candidates = generate_partition_candidates(
            expert_tuple,
            exhaustive_limit=self.options.partition_exhaustive_limit,
        )
        exact_count = self._exact_candidate_count(len(candidates))
        ranking_depth = 1 if self.options.mode == SchedulerMode.FAST else exact_count
        fast_search = search_candidates(
            self.simulator,
            candidates,
            top_k=ranking_depth,
        )

        exact_results: list[ExactDmaResult] = []
        if self.options.mode != SchedulerMode.FAST:
            exact_results = [
                ExactDmaPlanner(self.simulator).evaluate(result.candidate)
                for result in fast_search.ranked[:exact_count]
            ]

        if exact_results:
            exact = min(exact_results, key=self._exact_rank_key)
            schedule = exact.schedule
            candidate = exact.candidate
        else:
            exact = None
            schedule = fast_search.best
            candidate = schedule.candidate

        bandwidth = audit_bandwidth(schedule)
        if not bandwidth.valid:
            raise AssertionError("selected N-outer schedule exceeds the DMA contract")
        prefetch = audit_prefetch(schedule)
        cost = CostBreakdown(
            makespan_cc=schedule.makespan_cc,
            lower_bound_cc=schedule.lower_bound_cc,
            lower_bound_overhead_cc=schedule.overhead_cc,
            initial_prime_cc=max(schedule.initial_wait_cc),
            steady_stall_cc=sum(schedule.steady_stall_cc),
        )
        return SchedulerResult(
            mode=self.options.mode,
            candidate=candidate,
            schedule=schedule,
            exact=exact,
            cost=cost,
            bandwidth=bandwidth,
            prefetch=prefetch,
            generated_candidates=len(candidates),
            evaluated_candidates=fast_search.generated_candidates,
            rejected_candidates=fast_search.rejected_candidates,
            exact_candidates=len(exact_results),
            dma_optimal_for_selected_candidate=exact is not None,
            optimal_within_generated_bank=(
                self.options.mode == SchedulerMode.REFERENCE
                and len(exact_results) == fast_search.generated_candidates
            ),
            partition_space_complete=(
                len(expert_tuple) <= self.options.partition_exhaustive_limit
            ),
            permutation_space_complete=False,
        )

    def _exact_candidate_count(self, generated: int) -> int:
        if self.options.mode == SchedulerMode.REFERENCE:
            return generated
        if self.options.mode == SchedulerMode.HYBRID:
            return min(self.options.exact_top_k, generated)
        return 0

    @staticmethod
    def _exact_rank_key(result: ExactDmaResult) -> tuple[int, int, int, str]:
        schedule = result.schedule
        return (
            schedule.makespan_cc,
            sum(schedule.steady_stall_cc),
            schedule.overhead_cc,
            result.candidate.label,
        )


def audit_bandwidth(schedule: CandidateResult) -> BandwidthAudit:
    events: list[tuple[int, int, int]] = []
    lane_busy = [0, 0]
    transferred = 0
    lane_valid = True
    for load in schedule.loads:
        if not load.lanes or any(lane not in (0, 1) for lane in load.lanes):
            lane_valid = False
            continue
        duration = load.end_cc - load.start_cc
        bandwidth = LANE_BW_BYTES_PER_CC * len(load.lanes)
        events.append((load.start_cc, 1, bandwidth))
        events.append((load.end_cc, 0, -bandwidth))
        transferred += load.item.weight_bytes
        for lane in load.lanes:
            lane_busy[lane] += duration

    current = 0
    peak = 0
    for _, _, delta in sorted(events):
        current += delta
        peak = max(peak, current)
        if current < 0:
            lane_valid = False
    valid = (
        lane_valid
        and current == 0
        and peak <= 2 * LANE_BW_BYTES_PER_CC
        and schedule.history_validated
    )
    return BandwidthAudit(
        valid=valid,
        peak_bandwidth_bytes_per_cc=peak,
        lane_busy_cc=tuple(lane_busy),
        transferred_weight_bytes=transferred,
    )


def audit_prefetch(schedule: CandidateResult) -> PrefetchAudit:
    compute_by_key = {record.item.key: record for record in schedule.computes}
    lookahead = 0
    overlapped = 0
    fully_hidden = 0
    phase_boundary = 0
    overlap_cc = 0
    exposed_cc = 0

    for load in schedule.loads:
        index = load.item.stream_index
        if index == 0:
            continue
        lookahead += 1
        previous = compute_by_key[(load.item.cluster, index - 1)]
        overlap = max(
            0,
            min(load.end_cc, previous.end_cc)
            - max(load.start_cc, previous.start_cc),
        )
        duration = load.end_cc - load.start_cc
        overlap_cc += overlap
        exposed_cc += duration - overlap
        if overlap > 0:
            overlapped += 1
            if load.end_cc <= previous.end_cc:
                fully_hidden += 1
            if load.item.phase_index != previous.item.phase_index:
                phase_boundary += 1

    return PrefetchAudit(
        lookahead_loads=lookahead,
        overlapped_loads=overlapped,
        fully_hidden_loads=fully_hidden,
        phase_boundary_prefetches=phase_boundary,
        overlap_cc=overlap_cc,
        exposed_load_cc=exposed_cc,
    )
