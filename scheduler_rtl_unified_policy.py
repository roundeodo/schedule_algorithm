#!/usr/bin/env python3
"""Single-path RTL-oriented scheduler with a compiled T6+B2 candidate bank.

The runtime performs one bounded candidate pass, one fixed integer comparison
policy and one commit per round.  Candidate templates are Python constants
that mirror combinational decode cases; no JSON policy, ROM lookup,
distribution classifier, beam expansion, SIM1 or rollout is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import statistics
from typing import Iterable, Mapping

import evaluate_olmoe_fixed_token_banks as policy
import four_stage_scheduler as reference


POLICY_ID = "rtl-unified-t6b2-fixed13-v2"
WINDOW = (6, 2)
MAX_CONCRETE_CANDIDATES = 13
TICK_CC = policy.TICK_CC
SCORER = policy.HEAD5_HIST4_PAIRWISE_SCORER


@dataclass(frozen=True)
class ScheduleStep:
    mode: str
    candidate_slot: int
    action: reference.StageAction
    tag: str
    score: tuple[int, ...]
    candidate_count: int


@dataclass(frozen=True)
class ScheduleResult:
    makespan_cc: int
    steps: tuple[ScheduleStep, ...]
    candidate_count_max: int
    candidate_count_mean: float


@dataclass(frozen=True)
class CandidateSlot:
    """One deterministic concrete slot emitted for the current state."""

    slot: int
    action: reference.StageAction


def _logical(
    mode: str,
    family: str,
    *selectors: str,
    split_rule: str = "NONE",
) -> policy.ExplicitLogicalToken:
    return policy.ExplicitLogicalToken(
        mode=mode,
        family=family,
        selectors=tuple(selectors),
        split_rule=split_rule,
    )


def _profile(
    *,
    c2_s1: str = "NONE",
    c2_s3: str = "NONE",
    c3_s1: str = "NONE",
    c3_s3: str = "NONE",
    c2_dma_s1: str = "NONE",
    c2_dma_s3: str = "NONE",
    c2_s2pf: str = "NONE",
    c3_dma_s1: str = "NONE",
    c3_dma_s3: str = "NONE",
    c3_s2pf: str = "NONE",
    c2_s1_cached: bool = False,
    c2_s3_cached: bool = False,
    c3_s1_cached: bool = False,
    c3_s3_cached: bool = False,
) -> policy.ExplicitPhysicalProfile:
    return policy.ExplicitPhysicalProfile(
        c2_s1=c2_s1,
        c2_s3=c2_s3,
        c3_s1=c3_s1,
        c3_s3=c3_s3,
        c2_dma_s1=c2_dma_s1,
        c2_dma_s3=c2_dma_s3,
        c2_s2pf=c2_s2pf,
        c3_dma_s1=c3_dma_s1,
        c3_dma_s3=c3_dma_s3,
        c3_s2pf=c3_s2pf,
        s4pf_dma="NONE",
        c2_s1_cached=c2_s1_cached,
        c2_s3_cached=c2_s3_cached,
        c3_s1_cached=c3_s1_cached,
        c3_s3_cached=c3_s3_cached,
    )


def _token(
    logical: policy.ExplicitLogicalToken,
    physical: policy.ExplicitPhysicalProfile,
) -> policy.ExplicitCandidateToken:
    return policy.ExplicitCandidateToken(logical=logical, physical=physical)


def _compiled_candidate_tokens() -> tuple[policy.ExplicitCandidateToken, ...]:
    """Return fixed decode cases; mode/cluster valid bits select at runtime."""
    shape_b = reference.SHAPE_B.name
    shape_c = reference.SHAPE_C.name
    shape_a = reference.SHAPE_A.name
    return (
        # ONE_IDLE slots: B0 C/C, T0 B/B+S2PF and T3 B/B+S2PF.
        _token(
            _logical("ONE_IDLE", "SINGLE", "B0"),
            _profile(
                c3_s1=shape_c,
                c3_s3=shape_c,
                c3_dma_s1="BOTH",
                c3_dma_s3="BOTH",
            ),
        ),
        _token(
            _logical("ONE_IDLE", "SINGLE", "B0"),
            _profile(
                c2_s1=shape_c,
                c2_s3=shape_c,
                c2_dma_s1="BOTH",
                c2_dma_s3="BOTH",
            ),
        ),
        _token(
            _logical("ONE_IDLE", "SINGLE", "T0"),
            _profile(
                c2_s1=shape_b,
                c2_s3=shape_b,
                c2_dma_s1="BOTH",
                c2_s2pf="BOTH",
                c2_s3_cached=True,
            ),
        ),
        _token(
            _logical("ONE_IDLE", "SINGLE", "T0"),
            _profile(
                c3_s1=shape_b,
                c3_s3=shape_b,
                c3_dma_s1="BOTH",
                c3_s2pf="BOTH",
                c3_s3_cached=True,
            ),
        ),
        _token(
            _logical("ONE_IDLE", "SINGLE", "T3"),
            _profile(
                c2_s1=shape_b,
                c2_s3=shape_b,
                c2_dma_s1="BOTH",
                c2_s2pf="BOTH",
                c2_s3_cached=True,
            ),
        ),
        _token(
            _logical("ONE_IDLE", "SINGLE", "T3"),
            _profile(
                c3_s1=shape_b,
                c3_s3=shape_b,
                c3_dma_s1="BOTH",
                c3_s2pf="BOTH",
                c3_s3_cached=True,
            ),
        ),
        # SYNC slot 0: hot+cold asymmetric pair.
        _token(
            _logical("SYNC", "PAIR", "B0", "T0"),
            _profile(
                c2_s1=shape_a,
                c2_s3=shape_b,
                c3_s1=shape_b,
                c3_s3=shape_b,
                c2_dma_s1="IDMA",
                c2_s2pf="XDMA",
                c3_dma_s1="XDMA",
                c3_dma_s3="IDMA",
                c2_s3_cached=True,
            ),
        ),
        # SYNC slots 1/2: T0+T1 without and with C2 S2PF.
        _token(
            _logical("SYNC", "PAIR", "T0", "T1"),
            _profile(
                c2_s1=shape_b,
                c2_s3=shape_b,
                c3_s1=shape_b,
                c3_s3=shape_b,
                c2_dma_s1="IDMA",
                c2_dma_s3="IDMA",
                c3_dma_s1="XDMA",
                c3_dma_s3="XDMA",
            ),
        ),
        _token(
            _logical("SYNC", "PAIR", "T0", "T1"),
            _profile(
                c2_s1=shape_b,
                c2_s3=shape_b,
                c3_s1=shape_b,
                c3_s3=shape_b,
                c2_dma_s1="IDMA",
                c2_s2pf="IDMA",
                c3_dma_s1="XDMA",
                c3_dma_s3="IDMA",
                c2_s3_cached=True,
            ),
        ),
        # SYNC slot 3: hot+T4 asymmetric pair.
        _token(
            _logical("SYNC", "PAIR", "T0", "T4"),
            _profile(
                c2_s1=shape_a,
                c2_s3=shape_b,
                c3_s1=shape_b,
                c3_s3=shape_b,
                c2_dma_s1="IDMA",
                c2_s2pf="XDMA",
                c3_dma_s1="XDMA",
                c3_dma_s3="IDMA",
                c2_s3_cached=True,
            ),
        ),
        # SYNC slots 4/5: adjacent middle pairs with both S2PFs.
        *(
            _token(
                _logical("SYNC", "PAIR", left, right),
                _profile(
                    c2_s1=shape_b,
                    c2_s3=shape_b,
                    c3_s1=shape_b,
                    c3_s3=shape_b,
                    c2_dma_s1="IDMA",
                    c2_s2pf="IDMA",
                    c3_dma_s1="XDMA",
                    c3_s2pf="XDMA",
                    c2_s3_cached=True,
                    c3_s3_cached=True,
                ),
            )
            for left, right in (("T1", "T2"), ("T2", "T3"))
        ),
        # TERMINAL slot: C/C on the legal idle cluster.
        _token(
            _logical("TERMINAL", "SINGLE", "T0"),
            _profile(
                c2_s1=shape_c,
                c2_s3=shape_c,
                c2_dma_s1="BOTH",
                c2_dma_s3="BOTH",
            ),
        ),
        _token(
            _logical("TERMINAL", "SINGLE", "T0"),
            _profile(
                c3_s1=shape_c,
                c3_s3=shape_c,
                c3_dma_s1="BOTH",
                c3_dma_s3="BOTH",
            ),
        ),
        # TERMINAL split is an exact current-round candidate, not SIM1.
        _token(
            _logical("TERMINAL", "SPLIT", "T0", split_rule="BALANCED"),
            _profile(
                c2_s1=shape_b,
                c2_s3=shape_b,
                c3_s1=shape_b,
                c3_s3=shape_b,
                c2_dma_s1="IDMA",
                c2_dma_s3="IDMA",
                c3_dma_s1="XDMA",
                c3_dma_s3="XDMA",
            ),
        ),
    )


COMPILED_TOKENS = _compiled_candidate_tokens()


def _active_cluster(token: policy.ExplicitCandidateToken) -> int | None:
    active_c2 = token.physical.c2_s1 != "NONE"
    active_c3 = token.physical.c3_s1 != "NONE"
    if active_c2 == active_c3:
        return None
    return 2 if active_c2 else 3


def _profile_with_residency(
    profile: policy.ExplicitPhysicalProfile,
    *,
    c2_s1: bool,
    c2_s3: bool,
    c3_s1: bool,
    c3_s3: bool,
) -> policy.ExplicitPhysicalProfile:
    values = {
        field: getattr(profile, field)
        for field in profile.__dataclass_fields__
    }
    for prefix, s1_hit, s3_hit in (
        ("c2", c2_s1, c2_s3),
        ("c3", c3_s1, c3_s3),
    ):
        if values[f"{prefix}_s1"] == "NONE":
            continue
        if s1_hit:
            values[f"{prefix}_s1_cached"] = True
            values[f"{prefix}_dma_s1"] = "NONE"
        if s3_hit:
            values[f"{prefix}_s3_cached"] = True
            values[f"{prefix}_dma_s3"] = "NONE"
            values[f"{prefix}_s2pf"] = "NONE"
    return policy.ExplicitPhysicalProfile(**values)


def _hit_state(
    eid: int,
    snap: reference.FourStageSnap,
    start: int,
) -> tuple[bool, bool]:
    return (
        reference._swiglu_hit_for_candidate(eid, snap, start),
        reference._down_hit_for_candidate(eid, snap, start),
    )


def _runtime_profiles(
    state: reference.BeamState,
    token: policy.ExplicitCandidateToken,
) -> tuple[policy.ExplicitCandidateToken, ...]:
    logical = token.logical
    if logical.mode != _mode(state):
        return (token,)
    selected = tuple(
        policy._resolve_explicit_selector(state, selector, WINDOW)
        for selector in logical.selectors
    )
    if any(eid is None for eid in selected):
        return (token,)
    eids = tuple(int(eid) for eid in selected if eid is not None)
    variants: set[policy.ExplicitPhysicalProfile] = set()

    if logical.family == "PAIR" and len(eids) == 2:
        now = max(int(state.c2.task_end), int(state.c3.task_end))
        for eid2, eid3 in (eids, tuple(reversed(eids))):
            c2_s1, c2_s3 = _hit_state(eid2, state.c2, now)
            c3_s1, c3_s3 = _hit_state(eid3, state.c3, now)
            variants.add(
                _profile_with_residency(
                    token.physical,
                    c2_s1=c2_s1,
                    c2_s3=c2_s3,
                    c3_s1=c3_s1,
                    c3_s3=c3_s3,
                )
            )
    elif logical.family == "SPLIT" and len(eids) == 1:
        now = max(int(state.c2.task_end), int(state.c3.task_end))
        c2_s1, c2_s3 = _hit_state(eids[0], state.c2, now)
        c3_s1, c3_s3 = _hit_state(eids[0], state.c3, now)
        variants.add(
            _profile_with_residency(
                token.physical,
                c2_s1=c2_s1,
                c2_s3=c2_s3,
                c3_s1=c3_s1,
                c3_s3=c3_s3,
            )
        )
    elif logical.family == "SINGLE" and len(eids) == 1:
        cluster = _active_cluster(token)
        if cluster is not None:
            own = state.c2 if cluster == 2 else state.c3
            peer = state.c3 if cluster == 2 else state.c2
            starts = {int(own.task_end), int(peer.task_end)}
            if peer.s2pf_end >= 0:
                starts.add(int(peer.s2pf_end))
            for start in starts:
                s1_hit, s3_hit = _hit_state(eids[0], own, start)
                variants.add(
                    _profile_with_residency(
                        token.physical,
                        c2_s1=s1_hit if cluster == 2 else False,
                        c2_s3=s3_hit if cluster == 2 else False,
                        c3_s1=s1_hit if cluster == 3 else False,
                        c3_s3=s3_hit if cluster == 3 else False,
                    )
                )
    if not variants:
        variants.add(token.physical)
    return tuple(
        policy.ExplicitCandidateToken(logical=logical, physical=physical)
        for physical in sorted(variants)
    )


def _runtime_token_bank(
    state: reference.BeamState,
    tokens: Iterable[policy.ExplicitCandidateToken],
) -> tuple[policy.ExplicitCandidateToken, ...]:
    return tuple(
        sorted(
            {
                variant
                for token in tokens
                for variant in _runtime_profiles(state, token)
            }
        )
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
    return policy._bounded_policy_state(state, SCORER)


def _mode(state: reference.BeamState) -> str:
    return policy._explicit_mode(state)


def generate_candidate_slots(
    state: reference.BeamState,
) -> tuple[CandidateSlot, ...]:
    """Materialize and number the bounded candidate stream for RTL lockstep.

    Slot IDs are the deterministic emission indices for the current round,
    not persistent meanings shared by different modes.  The sequential RTL
    decoder can use the same counter while emitting legal concrete actions.
    """
    runtime_tokens = _runtime_token_bank(state, COMPILED_TOKENS)
    candidates, generation = policy.generate_direct_explicit_candidates(
        state,
        runtime_tokens,
        window=WINDOW,
        start_policy="bounded_release",
    )
    count = int(generation["concrete_candidates"])
    if count != len(candidates):
        raise AssertionError("candidate count does not match emitted stream")
    if count > MAX_CONCRETE_CANDIDATES:
        raise AssertionError(
            f"candidate budget exceeded: mode={_mode(state)} count={count}"
        )
    slots = []
    for slot, action in enumerate(candidates):
        slots.append(
            CandidateSlot(
                slot=slot,
                action=action,
            )
        )
    return tuple(slots)


def _choose_one_round(
    state: reference.BeamState,
) -> tuple[
    reference.StageAction,
    reference.BeamState,
    tuple[int, ...],
    int,
    int,
]:
    slots = generate_candidate_slots(state)
    candidates = [slot.action for slot in slots]
    if not candidates:
        raise RuntimeError("compiled candidate bank has no legal progress candidate")

    score, selected_index, action, child, _selector = (
        policy.select_practical_probe_candidate(
            state,
            candidates,
            scorer=SCORER,
            sync_tiebreak="hot_cold",
            window=WINDOW,
        )
    )
    selected_slot = int(slots[selected_index].slot)
    return action, child, score, len(slots), selected_slot


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
        action, state, score, count, selected_slot = (
            _choose_one_round(state)
        )
        replay = reference.apply_action(before, action)
        replay = policy._bounded_policy_state(
            replay,
            SCORER,
            before_f_score=int(before.f_score),
        )
        if replay != state:
            raise AssertionError("selected transition replay mismatch")
        steps.append(
            ScheduleStep(
                mode=_mode(before),
                candidate_slot=selected_slot,
                action=action,
                tag=action.tag,
                score=tuple(int(value) for value in score),
                candidate_count=count,
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
    counts = [step.candidate_count for step in steps]
    return ScheduleResult(
        makespan_cc=int(state.g_score),
        steps=tuple(steps),
        candidate_count_max=max(counts, default=0),
        candidate_count_mean=statistics.mean(counts) if counts else 0.0,
    )
