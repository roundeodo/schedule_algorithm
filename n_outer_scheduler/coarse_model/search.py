#!/usr/bin/env python3
"""History search and continuation scoring for the coarse N-outer model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .candidates import (
    CandidateSkeleton,
    RemainingExpert,
    WindowSpec,
    consume_candidate,
    generate_skeletons,
    materialize_modes,
)
from .semantics import (
    ALL_SHAPES,
    LANE_BW_BYTES_PER_CC,
    MacroActionPlan,
    MacroActionTiming,
    MacroPhaseSpec,
    MacroScheduleState,
    DmaBinding,
    compute_block_cc,
    default_phases,
    evaluate_action,
)


@dataclass(frozen=True)
class SearchConfig:
    window: WindowSpec = WindowSpec(8, 2)
    split_cuts: str = "balanced"
    beam_width: int = 32
    candidate_budget: int | None = None
    max_waves: int = 128

    def __post_init__(self) -> None:
        if self.beam_width <= 0 or self.max_waves <= 0:
            raise ValueError("beam width and max waves must be positive")
        if self.candidate_budget is not None and self.candidate_budget <= 0:
            raise ValueError("candidate budget must be positive")


@dataclass(frozen=True)
class SelectedStep:
    skeleton: CandidateSkeleton
    plan: MacroActionPlan
    timing: MacroActionTiming


@dataclass(frozen=True)
class SearchNode:
    remaining: tuple[RemainingExpert, ...]
    state: MacroScheduleState
    history: tuple[SelectedStep, ...] = ()
    resource_stall_cc: int = 0
    pipeline_stall_cc: int = 0
    continuation_lb_cc: int = 0
    continuation_estimate_cc: int = 0

    @property
    def complete(self) -> bool:
        return not self.remaining

    @property
    def makespan_cc(self) -> int:
        return max(self.state.cluster_free_cc)

    @property
    def rank_key(self) -> tuple[object, ...]:
        return (
            self.continuation_estimate_cc,
            self.continuation_lb_cc,
            self.makespan_cc,
            self.resource_stall_cc + self.pipeline_stall_cc,
            tuple(step.skeleton.label for step in self.history),
        )


@dataclass(frozen=True)
class SearchResult:
    node: SearchNode
    expanded_nodes: int
    evaluated_plans: int
    history_validated: bool


def remaining_from_distribution(distribution: Sequence[int]) -> tuple[RemainingExpert, ...]:
    return tuple(
        RemainingExpert(eid, 0, int(ntokens))
        for eid, ntokens in enumerate(distribution)
        if ntokens > 0
    )


def _two_resource_finish_lb(a: int, b: int, work: int) -> int:
    """Continuous-work lower bound for two resources with availability times."""

    early, late = sorted((a, b))
    if work <= late - early:
        return max(late, early + work)
    return max(late, math.ceil((early + late + work) / 2))


def continuation_lower_bound(
    state: MacroScheduleState,
    remaining: Sequence[RemainingExpert],
    *,
    phases: tuple[MacroPhaseSpec, MacroPhaseSpec] | None = None,
) -> int:
    """Admissible max(compute, DMA, committed-history) continuation bound.

    The bound assumes continuously divisible future work and no duplicated
    weights for SPLIT.  Those relaxations can only make the estimate earlier,
    so this is safe for reference search and simple enough for max/add/shift
    hardware scoring.
    """

    phase_specs = phases or default_phases()
    compute_work = 0
    for expert in remaining:
        compute_work += sum(
            phase.block_count
            * min(
                compute_block_cc(expert.ntokens, shape, phase)
                for shape in ALL_SHAPES
            )
            for phase in phase_specs
        )
    compute_finish = _two_resource_finish_lb(
        *state.cluster_free_cc, compute_work
    )

    lane_work = sum(
        math.ceil(phase.weight_block_bytes / LANE_BW_BYTES_PER_CC)
        * phase.block_count
        for phase in phase_specs
    ) * len(remaining)
    dma_finish = _two_resource_finish_lb(
        state.lane_state.idma_free_cc,
        state.lane_state.xdma_free_cc,
        lane_work,
    )
    return max(max(state.cluster_free_cc), compute_finish, dma_finish)


def continuation_lpt_estimate(
    state: MacroScheduleState,
    remaining: Sequence[RemainingExpert],
    *,
    phases: tuple[MacroPhaseSpec, MacroPhaseSpec] | None = None,
) -> int:
    """Hardware-friendly future estimate that preserves indivisible jobs.

    Unlike the lower bound, this list-schedules whole remaining experts in
    descending compute time onto the earlier cluster.  It is a ranking
    heuristic, not an optimality certificate.  The DMA component remains the
    relaxed resource lower bound so the score uses max/add/compare only.
    """

    phase_specs = phases or default_phases()
    jobs = sorted(
        (
            sum(
                phase.block_count
                * min(
                    compute_block_cc(expert.ntokens, shape, phase)
                    for shape in ALL_SHAPES
                )
                for phase in phase_specs
            )
            for expert in remaining
        ),
        reverse=True,
    )
    cluster_finish = list(state.cluster_free_cc)
    for job in jobs:
        cluster = min(range(2), key=lambda item: (cluster_finish[item], item))
        cluster_finish[cluster] += job
    lane_work = sum(
        math.ceil(phase.weight_block_bytes / LANE_BW_BYTES_PER_CC)
        * phase.block_count
        for phase in phase_specs
    ) * len(remaining)
    dma_finish = _two_resource_finish_lb(
        state.lane_state.idma_free_cc,
        state.lane_state.xdma_free_cc,
        lane_work,
    )
    return max(max(cluster_finish), dma_finish, max(state.cluster_free_cc))


def _expand_node(
    node: SearchNode,
    config: SearchConfig,
    phases: tuple[MacroPhaseSpec, MacroPhaseSpec],
) -> tuple[list[SearchNode], int]:
    children: list[SearchNode] = []
    evaluated = 0
    skeletons = generate_skeletons(
        node.remaining,
        window=config.window,
        split_cuts=config.split_cuts,
    )
    def static_bank_rank(
        entry: tuple[CandidateSkeleton, MacroActionPlan]
    ) -> tuple[object, ...]:
        skeleton, plan = entry
        kind_rank = {"pair": 0, "split": 1, "single": 2}[
            skeleton.kind.value
        ]
        counts = [task.expert_slice.ntokens for task in plan.tasks]
        balance = abs(counts[0] - counts[-1])
        fallback_penalty = 0
        lane_conflict = 0
        switch_penalty = 0
        for task in plan.tasks:
            for phase_plan, phase in zip(
                (task.gate_up, task.down), phases
            ):
                compute_cc = compute_block_cc(
                    task.expert_slice.ntokens, phase_plan.shape, phase
                )
                if phase_plan.dma != DmaBinding.BOTH:
                    single_load = math.ceil(
                        phase.weight_block_bytes / LANE_BW_BYTES_PER_CC
                    )
                    fallback_penalty += max(0, single_load - compute_cc)
            switch_penalty += int(task.gate_up.dma != task.down.dma)
        if len(plan.tasks) == 2:
            for phase_name in ("gate_up", "down"):
                left = getattr(plan.tasks[0], phase_name).dma
                right = getattr(plan.tasks[1], phase_name).dma
                lane_conflict += int(
                    left == right and left in (DmaBinding.IDMA, DmaBinding.XDMA)
                )
        return (
            kind_rank,
            balance,
            fallback_penalty,
            lane_conflict,
            switch_penalty,
            skeleton.label,
            tuple(
                (task.gate_up.dma.value, task.down.dma.value)
                for task in plan.tasks
            ),
        )

    if config.candidate_budget is None:
        bank = [
            (skeleton, plan)
            for skeleton in skeletons
            for plan in materialize_modes(skeleton, phases=phases)
        ]
        bank.sort(key=static_bank_rank)
    else:
        groups = {
            kind: [item for item in skeletons if item.kind.value == kind]
            for kind in ("pair", "split", "single")
        }

        def structure_rank(skeleton: CandidateSkeleton) -> tuple[object, ...]:
            counts = [
                assignment.expert_slice.ntokens
                for assignment in skeleton.assignments
            ]
            return (
                -sum(counts),
                abs(counts[0] - counts[-1]),
                skeleton.label,
            )

        for kind, items in groups.items():
            items.sort(key=structure_rank)
            if kind != "pair":
                continue
            ranked_experts = sorted(
                node.remaining, key=lambda item: (-item.ntokens, item.eid)
            )
            top_ids = {
                item.eid for item in ranked_experts[: config.window.top]
            }
            bottom_ids = {
                item.eid
                for item in (
                    ranked_experts[-config.window.bottom :]
                    if config.window.bottom
                    else ()
                )
            }
            expert_rank = {
                item.eid: rank for rank, item in enumerate(ranked_experts)
            }
            logical: dict[tuple[int, int], list[CandidateSkeleton]] = {}
            for skeleton in items:
                eids = tuple(
                    sorted(
                        assignment.expert_slice.eid
                        for assignment in skeleton.assignments
                    )
                )
                logical.setdefault(eids, []).append(skeleton)

            def logical_rank(
                key: tuple[int, int]
            ) -> tuple[int, int, int, int]:
                left, right = key
                is_hot_cold = (
                    (left in top_ids and right in bottom_ids)
                    or (right in top_ids and left in bottom_ids)
                )
                is_hot_hot = left in top_ids and right in top_ids
                family = 0 if is_hot_cold else 1 if is_hot_hot else 2
                return (
                    family,
                    min(expert_rank[left], expert_rank[right]),
                    max(expert_rank[left], expert_rank[right]),
                    left,
                )

            hot_cold = sorted(
                (key for key in logical if logical_rank(key)[0] == 0),
                key=logical_rank,
            )
            hot_hot = sorted(
                (key for key in logical if logical_rank(key)[0] == 1),
                key=logical_rank,
            )
            other = sorted(
                (key for key in logical if logical_rank(key)[0] == 2),
                key=logical_rank,
            )
            logical_order: list[tuple[int, int]] = []
            for index in range(max(len(hot_cold), len(hot_hot))):
                if index < len(hot_cold):
                    logical_order.append(hot_cold[index])
                if index < len(hot_hot):
                    logical_order.append(hot_hot[index])
            logical_order.extend(other)
            groups[kind] = [
                skeleton
                for key in logical_order
                for skeleton in sorted(logical[key], key=lambda item: item.label)
            ]
        budget = config.candidate_budget
        quotas = {
            "pair": budget // 2,
            "split": budget // 4,
            "single": budget - budget // 2 - budget // 4,
        }
        unused = 0
        for kind in quotas:
            if not groups[kind]:
                unused += quotas[kind]
                quotas[kind] = 0
        for kind in ("pair", "split", "single"):
            if groups[kind] and unused:
                quotas[kind] += unused
                unused = 0
        bank = []
        for kind in ("pair", "split", "single"):
            items = groups[kind]
            if not items or quotas[kind] == 0:
                continue
            # Keep both cluster orientations before spending the final quarter
            # of a family quota on alternate DMA service modes.  Treating a
            # C0/C1 assignment as a disposable mode is wrong once cluster
            # availability diverges across history steps.
            structure_slots = max(1, quotas[kind] - quotas[kind] // 4)
            items = items[: min(len(items), structure_slots)]

            def diverse_modes(
                skeleton: CandidateSkeleton,
            ) -> list[tuple[CandidateSkeleton, MacroActionPlan]]:
                ordered = sorted(
                    (
                        (skeleton, plan)
                        for plan in materialize_modes(skeleton, phases=phases)
                    ),
                    key=static_bank_rank,
                )

                def category(plan: MacroActionPlan) -> tuple[str, str]:
                    labels = []
                    for phase_name in ("gate_up", "down"):
                        bindings = [
                            getattr(task, phase_name).dma for task in plan.tasks
                        ]
                        if len(bindings) == 1:
                            labels.append(
                                "both" if bindings[0] == DmaBinding.BOTH else "single"
                            )
                        elif all(binding == DmaBinding.BOTH for binding in bindings):
                            labels.append("both_both")
                        elif all(
                            binding in (DmaBinding.IDMA, DmaBinding.XDMA)
                            for binding in bindings
                        ) and bindings[0] != bindings[1]:
                            labels.append("parallel_single")
                        elif DmaBinding.BOTH in bindings:
                            labels.append("mixed_both_single")
                        else:
                            labels.append("same_single")
                    return tuple(labels)

                preferred_categories = (
                    ("both_both", "both_both"),
                    ("parallel_single", "parallel_single"),
                    ("mixed_both_single", "mixed_both_single"),
                    ("both", "both"),
                    ("single", "single"),
                )
                result: list[tuple[CandidateSkeleton, MacroActionPlan]] = []
                used: set[int] = set()
                for wanted in preferred_categories:
                    for index, entry in enumerate(ordered):
                        if index not in used and category(entry[1]) == wanted:
                            result.append(entry)
                            used.add(index)
                            break
                result.extend(
                    entry for index, entry in enumerate(ordered) if index not in used
                )
                return result

            mode_lists = [diverse_modes(skeleton) for skeleton in items]
            depth = 0
            selected = 0
            while selected < quotas[kind]:
                progress = False
                for modes in mode_lists:
                    if depth < len(modes):
                        bank.append(modes[depth])
                        selected += 1
                        progress = True
                        if selected == quotas[kind]:
                            break
                if not progress:
                    break
                depth += 1

    for skeleton, plan in bank:
        next_remaining = consume_candidate(node.remaining, skeleton)
        evaluated += 1
        timing = evaluate_action(plan, state=node.state, phases=phases)
        next_state = timing.next_state
        child = SearchNode(
            remaining=next_remaining,
            state=next_state,
            history=(*node.history, SelectedStep(skeleton, plan, timing)),
            resource_stall_cc=node.resource_stall_cc
            + timing.resource_stall_cc,
            pipeline_stall_cc=node.pipeline_stall_cc
            + timing.pipeline_stall_cc,
            continuation_lb_cc=continuation_lower_bound(
                next_state, next_remaining, phases=phases
            ),
            continuation_estimate_cc=continuation_lpt_estimate(
                next_state, next_remaining, phases=phases
            ),
        )
        children.append(child)
    children.sort(key=lambda child: child.rank_key)
    return children, evaluated


def beam_search(
    distribution: Sequence[int],
    *,
    config: SearchConfig = SearchConfig(),
    phases: tuple[MacroPhaseSpec, MacroPhaseSpec] | None = None,
) -> SearchResult:
    """Run bounded best-first waves until at least one full history completes."""

    phase_specs = phases or default_phases()
    remaining = remaining_from_distribution(distribution)
    initial_state = MacroScheduleState()
    initial = SearchNode(
        remaining=remaining,
        state=initial_state,
        continuation_lb_cc=continuation_lower_bound(
            initial_state, remaining, phases=phase_specs
        ),
        continuation_estimate_cc=continuation_lpt_estimate(
            initial_state, remaining, phases=phase_specs
        ),
    )
    if not remaining:
        return SearchResult(initial, 0, 0, True)

    frontier = [initial]
    completed: list[SearchNode] = []
    expanded_nodes = 0
    evaluated_plans = 0
    for _ in range(config.max_waves):
        next_frontier: list[SearchNode] = []
        for node in frontier:
            if node.complete:
                completed.append(node)
                continue
            children, evaluated = _expand_node(node, config, phase_specs)
            expanded_nodes += 1
            evaluated_plans += evaluated
            for child in children:
                if child.complete:
                    completed.append(child)
                else:
                    next_frontier.append(child)
        if completed:
            best_complete = min(completed, key=lambda node: node.rank_key)
            if not next_frontier or all(
                node.continuation_lb_cc >= best_complete.makespan_cc
                for node in next_frontier
            ):
                validate_history(distribution, best_complete, phases=phase_specs)
                return SearchResult(
                    best_complete, expanded_nodes, evaluated_plans, True
                )
        if not next_frontier:
            break
        next_frontier.sort(key=lambda node: node.rank_key)
        frontier = next_frontier[: config.beam_width]
    if not completed:
        raise RuntimeError("search did not complete within max_waves")
    best = min(completed, key=lambda node: node.rank_key)
    validate_history(distribution, best, phases=phase_specs)
    return SearchResult(best, expanded_nodes, evaluated_plans, True)


def validate_history(
    distribution: Sequence[int],
    node: SearchNode,
    *,
    phases: tuple[MacroPhaseSpec, MacroPhaseSpec] | None = None,
) -> None:
    """Replay selected actions and verify resource state and token coverage."""

    phase_specs = phases or default_phases()
    remaining = remaining_from_distribution(distribution)
    state = MacroScheduleState()
    covered: dict[int, list[tuple[int, int]]] = {}
    for index, step in enumerate(node.history):
        expected_remaining = consume_candidate(remaining, step.skeleton)
        replay = evaluate_action(
            step.plan,
            state=state,
            phases=phase_specs,
            forced_service_order=step.timing.service_order,
        )
        if replay != step.timing:
            raise AssertionError(f"history step {index} is not deterministic")
        for assignment in step.skeleton.assignments:
            item = assignment.expert_slice
            covered.setdefault(item.eid, []).append((item.token_start, item.token_end))
        remaining = expected_remaining
        state = replay.next_state
    if remaining or node.remaining:
        raise AssertionError("history leaves unscheduled experts")
    if state != node.state:
        raise AssertionError("history terminal state mismatch")
    for eid, ntokens in enumerate(distribution):
        if ntokens <= 0:
            continue
        intervals = sorted(covered.get(eid, ()))
        cursor = 0
        for start, end in intervals:
            if start != cursor or end <= start:
                raise AssertionError(f"expert {eid} token coverage is not exact")
            cursor = end
        if cursor != ntokens:
            raise AssertionError(f"expert {eid} token coverage is incomplete")
