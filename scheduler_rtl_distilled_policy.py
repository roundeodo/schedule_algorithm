#!/usr/bin/env python3
"""RTL-oriented top5+bottom1 bounded distilled scheduler.

Every round follows one fixed datapath:

1. materialize one statically compiled physical-profile bank;
2. locally reduce physical profiles that implement the same logical action;
3. evaluate every reduced action with one bounded continuation comparator;
4. commit the single global winner and any targeted S4PF attached to it.

The mirror has no base/recovery split, protected winner, recovery margin,
batch-level distribution classifier, beam expansion, SIM1, standalone S4PF
candidate or rollout.  S4PF is reduced locally with the concrete next task that
consumes it; the fixed trial order is C2 then C3 and local SINGLE then BOTH then
OFF.  The compiled profiles are hard-wired decode cases, not
runtime-programmable storage.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import statistics
from typing import Mapping

import four_stage_scheduler as reference
import scheduler_rtl_distilled_lowering as lowering
import scheduler_rtl_distilled_scoring as scoring
from scheduler_rtl_distilled_profiles import COMPILED_PROFILES
from scheduler_rtl_distilled_types import (
    LOGICAL_ACTION_PRIORITY,
    logical_action_order_key,
    MAX_PHYSICAL_CANDIDATES,
    POLICY_ID,
    TICK_CC,
    WINDOW,
)


CONTINUATION_SCORER = scoring.SCORER_ID
S4PF_MIN_CURRENT_GAIN_TICKS = 1


def assert_top5_bottom1_contract() -> None:
    """Keep observation, candidates and continuation scorer consistent."""
    if WINDOW != (5, 1):
        raise AssertionError(f"unexpected runtime window {WINDOW}")
    if scoring.WINDOW != WINDOW:
        raise AssertionError("candidate and continuation windows disagree")
    visible = {f"T{rank}" for rank in range(5)} | {"B0"}
    used = {
        selector
        for token in COMPILED_PROFILES
        for selector in token.logical.selectors
    }
    if used - visible:
        raise AssertionError(
            f"candidate bank exceeds top5+bottom1: {sorted(used - visible)}"
        )
    logical_actions = {
        (
            token.logical.mode,
            token.logical.family,
            token.logical.selectors,
            token.logical.split_rule,
        )
        for token in COMPILED_PROFILES
    }
    if logical_actions != set(LOGICAL_ACTION_PRIORITY):
        raise AssertionError("compiled profiles and fixed logical IDs disagree")


assert_top5_bottom1_contract()


@dataclass(frozen=True)
class ScheduleStep:
    mode: str
    candidate_slot: int
    action: reference.StageAction
    tag: str
    score: tuple[int, ...]
    selected_profile_slot: int
    local_profile_slots: tuple[int, ...]
    physical_candidate_count: int
    logical_candidate_count: int
    s4pf_actions: tuple[reference.StageAction, ...]


@dataclass(frozen=True)
class ScheduleResult:
    makespan_cc: int
    steps: tuple[ScheduleStep, ...]
    physical_candidate_count_max: int
    physical_candidate_count_mean: float
    logical_candidate_count_max: int
    logical_candidate_count_mean: float


@dataclass(frozen=True)
class CandidateSlot:
    """One locally reduced logical action presented to the global scorer."""

    slot: int
    action: reference.StageAction
    child: reference.BeamState
    physical_profile_slot: int
    s4pf_actions: tuple[reference.StageAction, ...]


@dataclass(frozen=True)
class CandidateSet:
    slots: tuple[CandidateSlot, ...]
    physical_count: int


def _mode(state: reference.BeamState) -> str:
    return lowering.mode(state)


def _logical_action_key(
    state: reference.BeamState,
    action: reference.StageAction,
) -> tuple[str, str, tuple[str, ...], str]:
    logical = lowering.logical_action_spec(state, action, WINDOW)
    return (
        logical.mode,
        logical.family,
        logical.selectors,
        logical.split_rule,
    )


def _physical_profile_key(
    action: reference.StageAction,
    child: reference.BeamState,
    s4pf_count: int,
    fixed_profile_slot: int,
) -> tuple[int, int, int, int, int, int]:
    """RTL-local reducer: finish first, then retain useful prefetches.

    The final field is the fixed combinational decode priority.  It is reached
    only after all timing and prefetch fields tie.
    """
    ends = (int(child.c2.task_end), int(child.c3.task_end))
    starts = [
        int(start)
        for eid, start in (
            (action.c2_eid, action.c2_start),
            (action.c3_eid, action.c3_start),
        )
        if eid >= 0
    ]
    _maximum, _minimum, _selected_sum, s2pf = (
        lowering.selected_action_features(action)
    )
    return (
        max(ends),
        sum(ends),
        max(starts, default=0),
        -int(s2pf),
        -int(s4pf_count),
        int(fixed_profile_slot),
    )


def _initial_state(
    token_distribution: Mapping[int, int],
    initial_cache_c2: int,
    initial_cache_c3: int,
) -> reference.BeamState:
    normalized = {
        int(eid): int(ntok)
        for eid, ntok in token_distribution.items()
        if int(ntok) > 0
    }
    state = reference.FourStageScheduler(
        normalized,
        initial_cache_c2=int(initial_cache_c2),
        initial_cache_c3=int(initial_cache_c3),
    )._initial_state()
    return scoring.normalize_state_bound(state)


def _materialize_candidate_set(
    state: reference.BeamState,
    *,
    enable_s4pf: bool,
) -> CandidateSet:
    runtime_profiles, fixed_priorities = lowering.runtime_profile_bank(
        state, COMPILED_PROFILES
    )
    physical_with_sources, _stats = lowering.materialize_candidates_with_sources(
        state,
        runtime_profiles,
    )
    physical = []
    physical_count = len(physical_with_sources)
    for action, runtime_sources in physical_with_sources:
        fixed_profile_slot = min(
            fixed_priorities[source] for source in runtime_sources
        )
        baseline_child = reference.apply_action(state, action)
        physical.append((action, baseline_child, (), fixed_profile_slot, False))
        if enable_s4pf:
            targeted = lowering.materialize_targeted_s4pf_variant(state, action)
            if targeted is not None:
                physical.append((*targeted, fixed_profile_slot, True))
    if physical_count > MAX_PHYSICAL_CANDIDATES:
        raise AssertionError(
            "physical candidate budget exceeded: "
            f"mode={_mode(state)} count={physical_count}"
        )

    grouped: dict[
        tuple[str, str, tuple[str, ...], str],
        list[
            tuple[
                reference.StageAction,
                reference.BeamState,
                tuple[reference.StageAction, ...],
                int,
                bool,
            ]
        ],
    ] = defaultdict(list)
    for action, child, s4pf_actions, fixed_profile_slot, targeted in physical:
        grouped[_logical_action_key(state, action)].append(
            (action, child, s4pf_actions, fixed_profile_slot, targeted)
        )

    reduced = []
    for logical in sorted(grouped, key=logical_action_order_key):
        baseline = min(
            (item for item in grouped[logical] if not item[4]),
            key=lambda item: _physical_profile_key(
                item[0],
                item[1],
                len(item[2]),
                item[3],
            ),
        )
        targeted_items = [item for item in grouped[logical] if item[4]]
        selected = baseline
        if targeted_items:
            targeted = min(
                targeted_items,
                key=lambda item: _physical_profile_key(
                    item[0],
                    item[1],
                    len(item[2]),
                    item[3],
                ),
            )
            baseline_max = max(
                int(baseline[1].c2.task_end),
                int(baseline[1].c3.task_end),
            )
            targeted_max = max(
                int(targeted[1].c2.task_end),
                int(targeted[1].c3.task_end),
            )
            if (
                baseline_max - targeted_max
                >= S4PF_MIN_CURRENT_GAIN_TICKS * TICK_CC
            ):
                selected = targeted
        reduced.append(selected)
    emitted: dict[
        tuple,
        tuple[
            reference.StageAction,
            reference.BeamState,
            tuple[reference.StageAction, ...],
            int,
        ],
    ] = {}
    for action, child, s4pf_actions, fixed_profile_slot, _targeted in reduced:
        emitted.setdefault(
            lowering.child_key(child),
            (action, child, s4pf_actions, fixed_profile_slot),
        )
    slots = tuple(
        CandidateSlot(
            slot=slot,
            action=action,
            child=child,
            physical_profile_slot=fixed_profile_slot,
            s4pf_actions=s4pf_actions,
        )
        for slot, (
            action,
            child,
            s4pf_actions,
            fixed_profile_slot,
        ) in enumerate(emitted.values())
    )
    if state.remaining and not slots:
        raise RuntimeError("compiled profile bank has no legal progress action")
    return CandidateSet(slots=slots, physical_count=physical_count)


def generate_candidate_slots(
    state: reference.BeamState,
    *,
    enable_s4pf: bool = True,
) -> tuple[CandidateSlot, ...]:
    return _materialize_candidate_set(state, enable_s4pf=enable_s4pf).slots


def _choose_one_round(
    state: reference.BeamState,
    *,
    enable_s4pf: bool,
) -> tuple[
    reference.StageAction,
    reference.BeamState,
    tuple[int, ...],
    CandidateSet,
    int,
]:
    candidate_set = _materialize_candidate_set(
        state,
        enable_s4pf=enable_s4pf,
    )
    score, selected_slot, action, child, _metadata = (
        scoring.select_continuation_transition_winner(
            state,
            [
                (slot.action, slot.child)
                for slot in candidate_set.slots
            ],
        )
    )
    return action, child, tuple(map(int, score)), candidate_set, selected_slot


def schedule(
    token_distribution: Mapping[int, int],
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
    *,
    enable_s4pf: bool = True,
) -> ScheduleResult:
    normalized = {
        int(eid): int(ntok)
        for eid, ntok in token_distribution.items()
        if int(ntok) > 0
    }
    state = _initial_state(normalized, initial_cache_c2, initial_cache_c3)
    steps: list[ScheduleStep] = []
    while state.remaining:
        round_state = state
        action, state, score, candidate_set, selected_slot = _choose_one_round(
            round_state,
            enable_s4pf=enable_s4pf,
        )
        s4pf_actions = candidate_set.slots[selected_slot].s4pf_actions

        replay = round_state
        for s4pf_action in s4pf_actions:
            replay = reference.apply_action(replay, s4pf_action)
        replay = reference.apply_action(replay, action)
        replay = scoring.normalize_state_bound(
            replay,
            parent_bound=int(round_state.f_score),
        )
        if replay != state:
            raise AssertionError("selected transition replay mismatch")
        steps.append(
            ScheduleStep(
                mode=_mode(round_state),
                candidate_slot=int(selected_slot),
                action=action,
                tag=action.tag,
                score=score,
                selected_profile_slot=candidate_set.slots[
                    selected_slot
                ].physical_profile_slot,
                local_profile_slots=tuple(
                    slot.physical_profile_slot for slot in candidate_set.slots
                ),
                physical_candidate_count=candidate_set.physical_count,
                logical_candidate_count=len(candidate_set.slots),
                s4pf_actions=s4pf_actions,
            )
        )

    replay_cc = reference.validate_schedule_history(
        state.history,
        normalized,
        initial_cache_c2=int(initial_cache_c2),
        initial_cache_c3=int(initial_cache_c3),
    )
    if replay_cc != state.g_score:
        raise AssertionError("schedule failed explicit-DMA replay")
    physical_counts = [step.physical_candidate_count for step in steps]
    logical_counts = [step.logical_candidate_count for step in steps]
    return ScheduleResult(
        makespan_cc=int(state.g_score),
        steps=tuple(steps),
        physical_candidate_count_max=max(physical_counts, default=0),
        physical_candidate_count_mean=(
            statistics.mean(physical_counts) if physical_counts else 0.0
        ),
        logical_candidate_count_max=max(logical_counts, default=0),
        logical_candidate_count_mean=(
            statistics.mean(logical_counts) if logical_counts else 0.0
        ),
    )
