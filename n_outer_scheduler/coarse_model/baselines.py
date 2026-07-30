#!/usr/bin/env python3
"""Simple auditable N-outer baselines used before tuning a search scorer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .candidates import (
    CandidateSkeleton,
    RemainingExpert,
    SliceAssignment,
    canonical_shape,
    consume_candidate,
    bounded_joint_mode_bank,
    balanced_split_cuts,
    rtl_symmetric_mode_bank,
    plan_matches_pair_mode_policy,
)
from .search import (
    SearchNode,
    SelectedStep,
    continuation_lower_bound,
    continuation_lpt_estimate,
    remaining_from_distribution,
    validate_history,
)
from .semantics import (
    ActionKind,
    DmaBinding,
    MacroActionPlan,
    MacroScheduleState,
    MacroTaskPlan,
    PhasePlan,
    default_phases,
    evaluate_action,
)


@dataclass(frozen=True)
class BaselineResult:
    name: str
    node: SearchNode
    cluster_eids: tuple[tuple[int, ...], tuple[int, ...]]
    cluster_estimated_load_cc: tuple[int, int]
    history_validated: bool


def _lpt_partition(
    distribution: Sequence[int],
) -> tuple[list[list[RemainingExpert]], list[int]]:
    remaining = remaining_from_distribution(distribution)
    ranked = sorted(
        remaining,
        key=lambda expert: (
            -_standalone_fixed_lane_cost(expert),
            -expert.ntokens,
            expert.eid,
        ),
    )
    assignments: list[list[RemainingExpert]] = [[], []]
    estimated = [0, 0]
    for expert in ranked:
        cluster = min(range(2), key=lambda item: (estimated[item], item))
        assignments[cluster].append(expert)
        estimated[cluster] += _standalone_fixed_lane_cost(expert)
    return assignments, estimated


def _fixed_lane_task(expert: RemainingExpert, cluster: int) -> MacroTaskPlan:
    gate_up, down = default_phases()
    binding = DmaBinding.IDMA if cluster == 0 else DmaBinding.XDMA
    return MacroTaskPlan(
        cluster,
        expert.expert_slice,
        PhasePlan(canonical_shape(expert.ntokens, gate_up), binding),
        PhasePlan(canonical_shape(expert.ntokens, down), binding),
    )


def _standalone_fixed_lane_cost(expert: RemainingExpert) -> int:
    task = _fixed_lane_task(expert, 0)
    return evaluate_action(
        MacroActionPlan(ActionKind.SINGLE, (task,))
    ).makespan_cc


def fixed_lane_lpt(distribution: Sequence[int]) -> BaselineResult:
    """LPT partition with one permanent DMA lane per cluster.

    The baseline has no DMA arbitration and no split weights.  Its partition
    cost is the actual cold single-lane macro duration of each expert, so both
    compute-heavy hotspots and DMA-bound cold experts enter the same unit.
    """

    remaining = remaining_from_distribution(distribution)
    assignments, estimated = _lpt_partition(distribution)

    state = MacroScheduleState()
    active_remaining = remaining
    history: list[SelectedStep] = []
    resource_stall = 0
    pipeline_stall = 0
    # Planning one lane to completion before the other is safe: the unused
    # lane/cluster retain time zero and can still be scheduled independently.
    for cluster in (0, 1):
        for expert in assignments[cluster]:
            task = _fixed_lane_task(expert, cluster)
            plan = MacroActionPlan(ActionKind.SINGLE, (task,))
            skeleton = CandidateSkeleton(
                ActionKind.SINGLE,
                (SliceAssignment(cluster, expert.expert_slice),),
            )
            timing = evaluate_action(plan, state=state)
            history.append(SelectedStep(skeleton, plan, timing))
            active_remaining = consume_candidate(active_remaining, skeleton)
            state = timing.next_state
            resource_stall += timing.resource_stall_cc
            pipeline_stall += timing.pipeline_stall_cc
    node = SearchNode(
        remaining=active_remaining,
        state=state,
        history=tuple(history),
        resource_stall_cc=resource_stall,
        pipeline_stall_cc=pipeline_stall,
        continuation_lb_cc=continuation_lower_bound(state, active_remaining),
        continuation_estimate_cc=continuation_lpt_estimate(
            state, active_remaining
        ),
    )
    validate_history(distribution, node)
    return BaselineResult(
        name="fixed_lane_lpt",
        node=node,
        cluster_eids=tuple(
            tuple(expert.eid for expert in cluster) for cluster in assignments
        ),
        cluster_estimated_load_cc=tuple(estimated),
        history_validated=True,
    )


def paired_lpt_mode_search(
    distribution: Sequence[int],
    *,
    cluster1_order: str = "descending",
    beam_width: int = 8,
    mode_budget: int = 8,
    score_mode: str = "projected",
    forced_split: tuple[int, int] | None = None,
    service_order_mode: str = "best18",
    tie_break_mode: str = "stall",
    pair_mode_policy: str = "all",
    mode_bank_policy: str = "bounded_k4",
) -> BaselineResult:
    """Freeze an LPT partition/sequence and search only joint DMA modes."""

    if cluster1_order not in ("descending", "ascending"):
        raise ValueError("cluster1_order must be descending or ascending")
    if beam_width <= 0:
        raise ValueError("beam width must be positive")
    if mode_budget <= 0:
        raise ValueError("mode budget must be positive")
    if score_mode not in ("projected", "local"):
        raise ValueError("score_mode must be projected or local")
    if tie_break_mode not in ("stall", "bank_order", "both_on_tie"):
        raise ValueError(
            "tie_break_mode must be stall, bank_order, or both_on_tie"
        )
    if pair_mode_policy not in ("all", "no_mixed", "fixed_only"):
        raise ValueError("pair_mode_policy must be all, no_mixed, or fixed_only")
    if mode_bank_policy not in ("bounded_k4", "rtl_symmetric2"):
        raise ValueError(
            "mode_bank_policy must be bounded_k4 or rtl_symmetric2"
        )
    if forced_split is None:
        assignments, estimated = _lpt_partition(distribution)
    else:
        eid, cut = forced_split
        remaining = remaining_from_distribution(distribution)
        original = next((item for item in remaining if item.eid == eid), None)
        if original is None or not 0 < cut < original.ntokens:
            raise ValueError("forced SPLIT is outside the selected expert")
        left = RemainingExpert(eid, original.token_start, cut)
        right = RemainingExpert(
            eid,
            original.token_start + cut,
            original.ntokens - cut,
        )
        assignments = [[left], [right]]
        estimated = [
            _standalone_fixed_lane_cost(left),
            _standalone_fixed_lane_cost(right),
        ]
        ranked = sorted(
            (item for item in remaining if item.eid != eid),
            key=lambda expert: (
                -_standalone_fixed_lane_cost(expert),
                -expert.ntokens,
                expert.eid,
            ),
        )
        for expert in ranked:
            cluster = min(range(2), key=lambda item: (estimated[item], item))
            assignments[cluster].append(expert)
            estimated[cluster] += _standalone_fixed_lane_cost(expert)
    if cluster1_order == "ascending":
        if forced_split is not None:
            raise ValueError("forced SPLIT currently requires descending order")
        assignments[1] = list(reversed(assignments[1]))
    rounds: list[CandidateSkeleton] = []
    for index in range(max(len(assignments[0]), len(assignments[1]))):
        present = [
            (cluster, assignments[cluster][index])
            for cluster in (0, 1)
            if index < len(assignments[cluster])
        ]
        if len(present) == 2:
            kind = (
                ActionKind.SPLIT
                if present[0][1].eid == present[1][1].eid
                else ActionKind.PAIR
            )
            rounds.append(
                CandidateSkeleton(
                    kind,
                    tuple(
                        SliceAssignment(cluster, expert.expert_slice)
                        for cluster, expert in present
                    ),
                )
            )
        else:
            cluster, expert = present[0]
            rounds.append(
                CandidateSkeleton(
                    ActionKind.SINGLE,
                    (SliceAssignment(cluster, expert.expert_slice),),
                )
            )

    frontier: list[
        tuple[MacroScheduleState, tuple[SelectedStep, ...], int, int]
    ] = [(MacroScheduleState(), (), 0, 0)]
    active_remaining = remaining_from_distribution(distribution)
    for round_index, skeleton in enumerate(rounds):
        next_remaining = consume_candidate(active_remaining, skeleton)
        future_cost = [0, 0]
        for future in rounds[round_index + 1 :]:
            for assignment in future.assignments:
                expert = RemainingExpert(
                    assignment.expert_slice.eid,
                    assignment.expert_slice.token_start,
                    assignment.expert_slice.ntokens,
                )
                future_cost[assignment.cluster] += _standalone_fixed_lane_cost(
                    expert
                )
        children: list[
            tuple[
                tuple[int, int, int, tuple[tuple[int, int], ...]],
                MacroScheduleState,
                tuple[SelectedStep, ...],
                int,
                int,
            ]
        ] = []
        for state, history, resource_stall, pipeline_stall in frontier:
            plans = (
                bounded_joint_mode_bank(skeleton, budget=mode_budget)
                if mode_bank_policy == "bounded_k4"
                else rtl_symmetric_mode_bank(skeleton)
            )

            plans = tuple(
                plan
                for plan in plans
                if plan_matches_pair_mode_policy(
                    skeleton, plan, pair_mode_policy
                )
            )
            if not plans:
                raise AssertionError("pair-mode filter removed the fixed baseline")
            for mode_index, plan in enumerate(plans):
                timing = evaluate_action(
                    plan,
                    state=state,
                    service_order_mode=service_order_mode,
                )
                next_state = timing.next_state
                projected = (
                    max(
                        next_state.cluster_free_cc[0] + future_cost[0],
                        next_state.cluster_free_cc[1] + future_cost[1],
                        next_state.lane_state.idma_free_cc,
                        next_state.lane_state.xdma_free_cc,
                    )
                    if score_mode == "projected"
                    else max(
                        *next_state.cluster_free_cc,
                        next_state.lane_state.idma_free_cc,
                        next_state.lane_state.xdma_free_cc,
                    )
                )
                stall = (
                    resource_stall
                    + timing.resource_stall_cc
                    + pipeline_stall
                    + timing.pipeline_stall_cc
                )
                signature = tuple(
                    (task.gate_up.dma.value, task.down.dma.value)
                    for task in plan.tasks
                )
                if tie_break_mode == "bank_order":
                    rank = (
                        projected,
                        max(next_state.cluster_free_cc),
                        mode_index,
                        stall,
                        signature,
                    )
                elif tie_break_mode == "both_on_tie":
                    rank = (
                        projected,
                        max(next_state.cluster_free_cc),
                        -mode_index,
                        stall,
                        signature,
                    )
                else:
                    rank = (
                        projected,
                        max(next_state.cluster_free_cc),
                        stall,
                        mode_index,
                        signature,
                    )
                children.append(
                    (
                        rank,
                        next_state,
                        (*history, SelectedStep(skeleton, plan, timing)),
                        resource_stall + timing.resource_stall_cc,
                        pipeline_stall + timing.pipeline_stall_cc,
                    )
                )
        children.sort(key=lambda item: item[0])
        frontier = [
            (state, history, resource_stall, pipeline_stall)
            for _, state, history, resource_stall, pipeline_stall in children[
                :beam_width
            ]
        ]
        active_remaining = next_remaining

    state, history, resource_stall, pipeline_stall = min(
        frontier,
        key=lambda item: (
            max(item[0].cluster_free_cc),
            item[2] + item[3],
        ),
    )
    node = SearchNode(
        remaining=active_remaining,
        state=state,
        history=history,
        resource_stall_cc=resource_stall,
        pipeline_stall_cc=pipeline_stall,
        continuation_lb_cc=continuation_lower_bound(state, active_remaining),
        continuation_estimate_cc=continuation_lpt_estimate(
            state, active_remaining
        ),
    )
    validate_history(distribution, node)
    return BaselineResult(
        name=(
            f"paired_lpt_mode_search_{cluster1_order}_{score_mode}"
            + (
                ""
                if mode_bank_policy == "bounded_k4"
                else f"_{mode_bank_policy}"
            )
            if forced_split is None
            else f"split_e{forced_split[0]}_at{forced_split[1]}_"
            f"paired_lpt_{score_mode}"
            + (
                ""
                if mode_bank_policy == "bounded_k4"
                else f"_{mode_bank_policy}"
            )
        ),
        node=node,
        cluster_eids=tuple(
            tuple(expert.eid for expert in cluster) for cluster in assignments
        ),
        cluster_estimated_load_cc=tuple(estimated),
        history_validated=True,
    )


def split_hot_lpt_mode_search(
    distribution: Sequence[int],
    *,
    max_hot_experts: int = 4,
    beam_width: int = 1,
    mode_budget: int = 4,
    score_mode: str = "local",
    service_order_mode: str = "best18",
    tie_break_mode: str = "stall",
    pair_mode_policy: str = "all",
    mode_bank_policy: str = "bounded_k4",
) -> BaselineResult:
    """Try balanced legal SPLITs for the hottest experts and keep the best."""

    remaining = remaining_from_distribution(distribution)
    hot = sorted(remaining, key=lambda item: (-item.ntokens, item.eid))[
        :max_hot_experts
    ]
    candidates = []
    for expert in hot:
        for cut in balanced_split_cuts(expert.ntokens):
            candidates.append(
                paired_lpt_mode_search(
                    distribution,
                    beam_width=beam_width,
                    mode_budget=mode_budget,
                    score_mode=score_mode,
                    forced_split=(expert.eid, cut),
                    service_order_mode=service_order_mode,
                    tie_break_mode=tie_break_mode,
                    pair_mode_policy=pair_mode_policy,
                    mode_bank_policy=mode_bank_policy,
                )
            )
    if not candidates:
        raise ValueError("distribution contains no splittable expert")
    return min(
        candidates,
        key=lambda result: (
            result.node.makespan_cc,
            result.node.resource_stall_cc + result.node.pipeline_stall_cc,
            result.name,
        ),
    )
