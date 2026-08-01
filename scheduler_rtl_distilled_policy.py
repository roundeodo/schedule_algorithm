#!/usr/bin/env python3
"""RTL-oriented top5+bottom1 bounded distilled scheduler.

Every round follows one fixed datapath:

1. materialize one statically compiled physical-profile bank;
2. locally reduce physical profiles that implement the same logical action;
3. evaluate every reduced action with one bounded continuation comparator;
4. commit the single global winner.

The mirror has no base/recovery split, protected winner, recovery margin,
batch-level distribution classifier, beam expansion, SIM1, S4 prefetch or
rollout.  The compiled profiles are hard-wired decode cases, not
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
    physical_profile_slot: int


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
    state: reference.BeamState,
    action: reference.StageAction,
    fixed_profile_slot: int,
) -> tuple[int, int, int, int, int]:
    """RTL-local reducer: finish first, then retain useful S2 prefetch.

    The final field is the fixed combinational decode priority.  It is reached
    only after all timing and prefetch fields tie.
    """
    child = reference.apply_action(state, action)
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
) -> CandidateSet:
    runtime_profiles, fixed_priorities = lowering.runtime_profile_bank(
        state, COMPILED_PROFILES
    )
    physical_with_sources, _stats = lowering.materialize_candidates_with_sources(
        state,
        runtime_profiles,
    )
    physical = [
        (
            action,
            min(fixed_priorities[source] for source in runtime_sources),
        )
        for action, runtime_sources in physical_with_sources
    ]
    physical_count = len(physical)
    if physical_count > MAX_PHYSICAL_CANDIDATES:
        raise AssertionError(
            "physical candidate budget exceeded: "
            f"mode={_mode(state)} count={physical_count}"
        )

    grouped: dict[
        tuple[str, str, tuple[str, ...], str],
        list[tuple[reference.StageAction, int]],
    ] = defaultdict(list)
    for action, fixed_profile_slot in physical:
        grouped[_logical_action_key(state, action)].append(
            (action, fixed_profile_slot)
        )

    reduced = [
        min(
            grouped[logical],
            key=lambda item: _physical_profile_key(state, item[0], item[1]),
        )
        for logical in sorted(grouped, key=logical_action_order_key)
    ]
    emitted: dict[tuple, tuple[reference.StageAction, int]] = {}
    for action, fixed_profile_slot in reduced:
        child = reference.apply_action(state, action)
        emitted.setdefault(
            lowering.child_key(child), (action, fixed_profile_slot)
        )
    slots = tuple(
        CandidateSlot(
            slot=slot,
            action=action,
            physical_profile_slot=fixed_profile_slot,
        )
        for slot, (action, fixed_profile_slot) in enumerate(emitted.values())
    )
    if state.remaining and not slots:
        raise RuntimeError("compiled profile bank has no legal progress action")
    return CandidateSet(slots=slots, physical_count=physical_count)


def generate_candidate_slots(
    state: reference.BeamState,
) -> tuple[CandidateSlot, ...]:
    return _materialize_candidate_set(state).slots


def _choose_one_round(
    state: reference.BeamState,
) -> tuple[
    reference.StageAction,
    reference.BeamState,
    tuple[int, ...],
    CandidateSet,
    int,
]:
    candidate_set = _materialize_candidate_set(state)
    score, selected_slot, action, child, _metadata = (
        scoring.select_continuation_winner(
            state,
            [slot.action for slot in candidate_set.slots],
        )
    )
    return action, child, tuple(map(int, score)), candidate_set, selected_slot


def schedule(
    token_distribution: Mapping[int, int],
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
) -> ScheduleResult:
    normalized = {
        int(eid): int(ntok)
        for eid, ntok in token_distribution.items()
        if int(ntok) > 0
    }
    state = _initial_state(normalized, initial_cache_c2, initial_cache_c3)
    steps: list[ScheduleStep] = []
    while state.remaining:
        before = state
        action, state, score, candidate_set, selected_slot = _choose_one_round(
            state
        )
        replay = reference.apply_action(before, action)
        replay = scoring.normalize_state_bound(
            replay,
            parent_bound=int(before.f_score),
        )
        if replay != state:
            raise AssertionError("selected transition replay mismatch")
        steps.append(
            ScheduleStep(
                mode=_mode(before),
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
