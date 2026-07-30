#!/usr/bin/env python3
"""Macro-versus-block calibration and candidate-ranking regret metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .block_golden import (
    ArbitrationPolicy,
    replay_best_policy,
    replay_block_history,
)
from .candidates import (
    CandidateSkeleton,
    WindowSpec,
    bounded_joint_mode_bank,
    generate_skeletons,
    materialize_modes,
    plan_matches_pair_mode_policy,
    rtl_symmetric_mode_bank,
)
from .search import (
    SearchNode,
    SelectedStep,
    remaining_from_distribution,
    validate_history,
)
from .semantics import (
    MacroActionPlan,
    MacroScheduleState,
    default_phases,
    evaluate_action,
)


@dataclass(frozen=True)
class CandidateCalibration:
    skeleton: CandidateSkeleton
    plan: MacroActionPlan
    macro_makespan_cc: int
    golden_makespan_cc: int
    golden_policy: ArbitrationPolicy

    @property
    def timing_error_cc(self) -> int:
        return self.macro_makespan_cc - self.golden_makespan_cc


@dataclass(frozen=True)
class SkeletonRankingCalibration:
    skeleton_label: str
    entry_indices: tuple[int, ...]
    macro_selected_index: int
    golden_selected_index: int
    ranking_regret_cc: int


@dataclass(frozen=True)
class RankingCalibration:
    entries: tuple[CandidateCalibration, ...]
    skeleton_rankings: tuple[SkeletonRankingCalibration, ...]
    max_mode_ranking_regret_cc: int
    max_abs_timing_error_cc: int


@dataclass(frozen=True)
class HistoryModeRoundCalibration:
    step_index: int
    skeleton_label: str
    modes_evaluated: int
    selected_mode_index: int
    executable_oracle_mode_index: int
    selected_macro_prefix_cc: int
    selected_executable_prefix_cc: int
    executable_oracle_prefix_cc: int
    ranking_regret_cc: int
    macro_timing_error_cc: int


@dataclass(frozen=True)
class HistoryModeCalibration:
    rounds: tuple[HistoryModeRoundCalibration, ...]
    total_ranking_regret_cc: int
    max_ranking_regret_cc: int
    nonzero_regret_rounds: int
    max_abs_macro_timing_error_cc: int


def calibrate_root_candidates(
    distribution: Sequence[int],
    *,
    window: WindowSpec,
    max_plans: int | None = None,
) -> RankingCalibration:
    """Compare the same root candidates in macro and independent block models."""

    phases = default_phases()
    remaining = remaining_from_distribution(distribution)
    candidates = [
        (skeleton, plan)
        for skeleton in generate_skeletons(
            remaining, window=window, split_cuts="balanced"
        )
        for plan in materialize_modes(skeleton, phases=phases)
    ]
    if max_plans is not None:
        candidates = candidates[:max_plans]
    entries: list[CandidateCalibration] = []
    for skeleton, plan in candidates:
        timing = evaluate_action(plan, phases=phases)
        step = SelectedStep(skeleton, plan, timing)
        golden = replay_best_policy((step,), phases=phases)
        entries.append(
            CandidateCalibration(
                skeleton=skeleton,
                plan=plan,
                macro_makespan_cc=timing.makespan_cc,
                golden_makespan_cc=golden.makespan_cc,
                golden_policy=golden.policy,
            )
        )
    if not entries:
        raise ValueError("calibration requires at least one candidate")
    by_skeleton: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        by_skeleton.setdefault(entry.skeleton.label, []).append(index)
    rankings: list[SkeletonRankingCalibration] = []
    for label, indices in sorted(by_skeleton.items()):
        macro_index = min(
            indices,
            key=lambda index: (
                entries[index].macro_makespan_cc,
                entries[index].golden_makespan_cc,
            ),
        )
        golden_index = min(
            indices,
            key=lambda index: (
                entries[index].golden_makespan_cc,
                entries[index].macro_makespan_cc,
            ),
        )
        rankings.append(
            SkeletonRankingCalibration(
                skeleton_label=label,
                entry_indices=tuple(indices),
                macro_selected_index=macro_index,
                golden_selected_index=golden_index,
                ranking_regret_cc=(
                    entries[macro_index].golden_makespan_cc
                    - entries[golden_index].golden_makespan_cc
                ),
            )
        )
    return RankingCalibration(
        entries=tuple(entries),
        skeleton_rankings=tuple(rankings),
        max_mode_ranking_regret_cc=max(
            item.ranking_regret_cc for item in rankings
        ),
        max_abs_timing_error_cc=max(abs(item.timing_error_cc) for item in entries),
    )


def calibrate_history_mode_choices(
    distribution: Sequence[int],
    node: SearchNode,
    *,
    mode_budget: int = 4,
    service_order_mode: str = "best18",
    pair_mode_policy: str = "all",
    mode_bank_policy: str = "bounded_k4",
) -> HistoryModeCalibration:
    """Measure local mode-ranking regret along one committed macro history.

    At each round, the already committed prefix and the action skeleton remain
    fixed. Every mode in the same bounded bank is evaluated by the macro
    operator and independently lowered through the block replay using that
    mode's own service ranks. The metric therefore tests the scorer/timespan
    choice, not a different candidate generator or a different future.
    """

    if mode_budget <= 0:
        raise ValueError("mode budget must be positive")
    if mode_bank_policy not in ("bounded_k4", "rtl_symmetric2"):
        raise ValueError(
            "mode_bank_policy must be bounded_k4 or rtl_symmetric2"
        )
    validate_history(distribution, node)
    prefix: tuple[SelectedStep, ...] = ()
    state = MacroScheduleState()
    rounds: list[HistoryModeRoundCalibration] = []
    for step_index, selected_step in enumerate(node.history):
        plans = (
            bounded_joint_mode_bank(
                selected_step.skeleton, budget=mode_budget
            )
            if mode_bank_policy == "bounded_k4"
            else rtl_symmetric_mode_bank(selected_step.skeleton)
        )
        plans = tuple(
            plan
            for plan in plans
            if plan_matches_pair_mode_policy(
                selected_step.skeleton, plan, pair_mode_policy
            )
        )
        selected_mode_index = next(
            (
                index
                for index, plan in enumerate(plans)
                if plan == selected_step.plan
            ),
            None,
        )
        if selected_mode_index is None:
            raise AssertionError(
                "selected plan is absent from the calibrated bounded mode bank"
            )
        macro_cc: list[int] = []
        executable_cc: list[int] = []
        for plan in plans:
            timing = evaluate_action(
                plan,
                state=state,
                service_order_mode=service_order_mode,
            )
            candidate = SelectedStep(selected_step.skeleton, plan, timing)
            replay = replay_block_history(
                (*prefix, candidate), policy=ArbitrationPolicy.MACRO_ORDER
            )
            macro_cc.append(timing.makespan_cc)
            executable_cc.append(replay.makespan_cc)
        oracle_index = min(
            range(len(plans)),
            key=lambda index: (
                executable_cc[index],
                macro_cc[index],
                index,
            ),
        )
        selected_executable = executable_cc[selected_mode_index]
        oracle_executable = executable_cc[oracle_index]
        rounds.append(
            HistoryModeRoundCalibration(
                step_index=step_index,
                skeleton_label=selected_step.skeleton.label,
                modes_evaluated=len(plans),
                selected_mode_index=selected_mode_index,
                executable_oracle_mode_index=oracle_index,
                selected_macro_prefix_cc=macro_cc[selected_mode_index],
                selected_executable_prefix_cc=selected_executable,
                executable_oracle_prefix_cc=oracle_executable,
                ranking_regret_cc=selected_executable - oracle_executable,
                macro_timing_error_cc=(
                    macro_cc[selected_mode_index] - selected_executable
                ),
            )
        )
        prefix = (*prefix, selected_step)
        state = selected_step.timing.next_state
    return HistoryModeCalibration(
        rounds=tuple(rounds),
        total_ranking_regret_cc=sum(item.ranking_regret_cc for item in rounds),
        max_ranking_regret_cc=max(
            (item.ranking_regret_cc for item in rounds), default=0
        ),
        nonzero_regret_rounds=sum(item.ranking_regret_cc > 0 for item in rounds),
        max_abs_macro_timing_error_cc=max(
            (abs(item.macro_timing_error_cc) for item in rounds), default=0
        ),
    )
