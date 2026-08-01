#!/usr/bin/env python3
"""T6+B2 union of protected adaptive and certified OLMoE policies.

Outside the certified OLMoE envelope, every round first reproduces the old
adaptive winner and admits only protected T0+B0 or B0@release0 replacements.
Inside the envelope, the independently replayed fixed14/head5-hist4 bank is
used.  Both modes use a fixed ROM, one sequential candidate pass, one best
reducer and one commit.  The supplied descriptor window is uniformly T0..T5
plus B0..B1; a mode may simply leave a visible selector unused.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import statistics
from typing import Iterable, Literal, Mapping

import evaluate_olmoe_fixed_token_banks as policy
import four_stage_scheduler as reference
import scheduler_hw_fixed_policy as fixed
import scheduler_rtl_adaptive_prefetch_policy as adaptive
from scheduler_olmoe_bounded_policy import (
    DEFAULT_TOKEN_BANK,
    POLICY_SCORER,
    POLICY_WINDOW as SOURCE_POLICY_WINDOW,
)
from scheduler_rtl_adaptive_prefetch_policy import (
    DEFAULT_S4_POLICY,
    S4Policy,
)


POLICY_ID = "rtl-adaptive-t6b2-protected-b0-certified-fixed14-union-v4"
POLICY_WINDOW = (6, 2)
CERTIFIED_TOTAL_EXPERTS = 64
CERTIFIED_ASSIGNMENTS = 140
FallbackReason = Literal[
    "none",
    "outside_certified_envelope",
    "candidate_bank_no_progress",
]
S4Gate = Literal["adaptive", "paired_single", "sync_paired_single"]


@dataclass(frozen=True)
class S4PrefetchEvent:
    cluster: int
    dma: reference.DmaBinding
    start_cc: int
    end_cc: int


@dataclass(frozen=True)
class HybridStep:
    action: reference.StageAction
    prefetch_events: tuple[S4PrefetchEvent, ...]
    candidate_count: int
    score: tuple


@dataclass(frozen=True)
class HybridScheduleResult:
    makespan_cc: int
    route: str
    fallback_reason: FallbackReason
    contract_eligible: bool
    steps: tuple[HybridStep, ...]
    candidate_count_max: int
    candidate_count_mean: float
    s4pf_single_count: int
    s4pf_both_count: int


def histogram_contract_eligible(
    token_distribution: Mapping[int, int],
) -> bool:
    """Return whether every descriptor below scorer head5 fits hist bins 1..4."""
    loads = sorted(
        (int(ntok) for ntok in token_distribution.values() if int(ntok) > 0),
        reverse=True,
    )
    return max(loads[SOURCE_POLICY_WINDOW[0] :], default=0) <= 4 * reference.FULL_M_DIM


def certified_olmoe_regime(
    token_distribution: Mapping[int, int],
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
    *,
    total_experts: int = CERTIFIED_TOTAL_EXPERTS,
) -> bool:
    """Frozen implementation envelope covered by all 65 LB=UB witnesses.

    Every field is available at initialization as a scalar counter or visible
    T6+B2 descriptor.  The bounds are the inclusive envelope of the frozen
    65-case proof set, not a learned per-case lookup table.  Initial cache hits
    remain on the old adaptive path because the proof set starts cache-empty.
    """
    if total_experts != CERTIFIED_TOTAL_EXPERTS:
        return False
    if initial_cache_c2 >= 0 or initial_cache_c3 >= 0:
        return False
    loads = sorted(
        (int(ntok) for ntok in token_distribution.values() if int(ntok) > 0),
        reverse=True,
    )
    if not loads or sum(loads) != CERTIFIED_ASSIGNMENTS:
        return False
    active = len(loads)
    cold_le2 = total_experts - active + sum(ntok <= 2 for ntok in loads)
    return (
        29 <= active <= 57
        and 40 <= cold_le2 <= 49
        and loads[0] <= 34
        and max(loads[5:], default=0) <= 7
        and histogram_contract_eligible(token_distribution)
    )


def _schedule_protected_t6b2(
    token_distribution: Mapping[int, int],
    initial_cache_c2: int,
    initial_cache_c3: int,
    *,
    s4_policy: S4Policy,
) -> HybridScheduleResult:
    """Run the old-winner-protected T6+B2 candidate pass every round.

    Candidate generation is repeated once here only to collect an audit count;
    the intended RTL emits each fixed candidate once into the same sequential
    score/reducer datapath.
    """
    cost_model = adaptive._COST_MODELS[s4_policy]
    with adaptive._use_cost_model(cost_model):
        state = fixed.initial_state(
            dict(token_distribution), initial_cache_c2, initial_cache_c3
        )
    candidate_counts: list[int] = []
    while state.remaining:
        with adaptive._use_cost_model(cost_model):
            transitions = fixed.generate_top6_bottom2_protected_successors(
                state,
                policy="balanced",
                top_policy="pruned",
                n1_policy="pruned",
            )
        candidate_counts.append(len(transitions))
        state = adaptive._choose_transition(
            state,
            cost_model,
            candidate_policy=fixed.TOP6_BOTTOM2_PROTECTED_CANDIDATE_POLICY,
            score_policy="protected_tail_b0_slack_1",
        ).state
    return HybridScheduleResult(
        makespan_cc=int(fixed.terminal_cost(state)),
        route="protected_t6b2",
        fallback_reason="none",
        contract_eligible=False,
        steps=(),
        candidate_count_max=max(candidate_counts, default=0),
        candidate_count_mean=(
            statistics.mean(candidate_counts) if candidate_counts else 0.0
        ),
        s4pf_single_count=0,
        s4pf_both_count=0,
    )


def _active_cluster(token: policy.ExplicitCandidateToken) -> int | None:
    physical = token.physical
    active_c2 = physical.c2_s1 != "NONE"
    active_c3 = physical.c3_s1 != "NONE"
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
    """Overlay runtime cache hits without adding a new candidate template.

    Cache bits mask the corresponding DMA fields in the same fixed physical
    template.  An S1-only S4PF hit preserves the template's down-weight path;
    a full initial-cache hit masks both S1 and S3 transfers.
    """
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


def _token_runtime_profiles(
    state: reference.BeamState,
    token: policy.ExplicitCandidateToken,
) -> tuple[policy.ExplicitCandidateToken, ...]:
    """Produce only cache overlays reachable by this token in this state."""
    logical = token.logical
    if logical.mode != policy._explicit_mode(state):
        return (token,)
    selected = tuple(
        policy._resolve_explicit_selector(state, selector, POLICY_WINDOW)
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
    expanded = {
        variant
        for token in tokens
        for variant in _token_runtime_profiles(state, token)
    }
    return tuple(sorted(expanded))


def _s4_mode_order(
    snap: reference.FourStageSnap,
    s4_policy: S4Policy,
) -> tuple[reference.DmaBinding, ...]:
    single = reference.DmaBinding.IDMA
    both = reference.DmaBinding.BOTH
    if s4_policy == "single_only":
        return (single,)
    if s4_policy == "both_only":
        return (both,)
    if s4_policy == "single_first":
        return (single, both)
    if s4_policy == "both_first":
        return (both, single)
    if s4_policy == "window_select":
        single_end = int(snap.dma3_end) + reference.dma_duration(
            reference.WEIGHT_BYTES_S1, single
        )
        return (single,) if single_end <= int(snap.task_end) else (both,)
    raise ValueError(f"unknown S4 policy {s4_policy!r}")


def _try_s4pf(
    snap: reference.FourStageSnap,
    peer: reference.FourStageSnap,
    *,
    cluster: int,
    s4_policy: S4Policy,
) -> tuple[reference.FourStageSnap, S4PrefetchEvent | None]:
    if snap.cur_eid < 0 or snap.pf_eid != -1:
        return snap, None
    for trial in _s4_mode_order(snap, s4_policy):
        binding = (
            reference.DmaBinding.IDMA
            if trial != reference.DmaBinding.BOTH and cluster == 2
            else reference.DmaBinding.XDMA
            if trial != reference.DmaBinding.BOTH
            else reference.DmaBinding.BOTH
        )
        start = int(snap.dma3_end)
        candidate = snap.with_prefetch(
            reference.PF_EID_GHOST,
            reference.SHAPE_A,
            start,
            binding,
        )
        if int(candidate.pf_end) > int(snap.task_end):
            continue
        feasible = (
            reference.bw_feasible(candidate, peer)
            if cluster == 2
            else reference.bw_feasible(peer, candidate)
        )
        if feasible:
            return candidate, S4PrefetchEvent(
                cluster=cluster,
                dma=binding,
                start_cc=start,
                end_cc=int(candidate.pf_end),
            )
    return snap, None


def _inject_s4_prefetches(
    state: reference.BeamState,
    *,
    s4_policy: S4Policy,
    s4_gate: S4Gate = "adaptive",
) -> tuple[reference.BeamState, tuple[S4PrefetchEvent, ...]]:
    """Preserve the adaptive baseline's sequential C2-then-C3 update order."""
    if not state.remaining:
        return state, ()
    if s4_gate in {"paired_single", "sync_paired_single"}:
        if (
            s4_gate == "sync_paired_single"
            and (
                state.c2.task_end != state.c3.task_end
                or len(state.remaining) < 2
                or int(state.remaining[0][1]) != int(state.remaining[1][1])
            )
        ):
            return state, ()
        c2, event2 = _try_s4pf(
            state.c2, state.c3, cluster=2, s4_policy="single_only"
        )
        c3, event3 = _try_s4pf(
            state.c3, c2, cluster=3, s4_policy="single_only"
        )
        # A lone cache hit changes SYNC/ONE_IDLE control asymmetrically.  The
        # paired gate commits neither transfer unless both owning lanes fit.
        # The stricter SYNC gate also requires an equal-load visible pair, so
        # the two S1 hits do not turn a balanced state into a hot/cold launch.
        if event2 is None or event3 is None:
            return state, ()
        return replace(state, c2=c2, c3=c3), (event2, event3)
    if s4_gate != "adaptive":
        raise ValueError(f"unknown S4 gate {s4_gate!r}")
    c2, event2 = _try_s4pf(
        state.c2, state.c3, cluster=2, s4_policy=s4_policy
    )
    c3, event3 = _try_s4pf(
        state.c3, c2, cluster=3, s4_policy=s4_policy
    )
    events = tuple(event for event in (event2, event3) if event is not None)
    return replace(state, c2=c2, c3=c3), events


class AdaptiveOlmoeScheduler:
    """T6+B2 regime union with a bit-exact adaptive fallback."""

    def __init__(self, token_bank: Path = DEFAULT_TOKEN_BANK):
        self.token_bank_path = token_bank.resolve()
        self.tokens = policy.load_explicit_token_bank(self.token_bank_path)

    @staticmethod
    def initial_state(
        token_distribution: Mapping[int, int],
        initial_cache_c2: int = -1,
        initial_cache_c3: int = -1,
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
        return policy._bounded_policy_state(state, POLICY_SCORER)

    def _choose_one_round(
        self,
        state: reference.BeamState,
    ) -> tuple[reference.StageAction, reference.BeamState, tuple, int]:
        runtime_tokens = _runtime_token_bank(state, self.tokens)
        candidates, generation = policy.generate_practical_probe_candidates(
            state,
            runtime_tokens,
            "bounded_release",
            "disabled",
            direct_generator=True,
            strict_token_bank=True,
            window=POLICY_WINDOW,
        )
        score, _tie, action, child, _selector = (
            policy.select_practical_probe_candidate(
                state,
                candidates,
                scorer=POLICY_SCORER,
                sync_tiebreak="hot_cold",
                window=POLICY_WINDOW,
            )
        )
        return action, child, score, int(generation["concrete_candidates"])

    def schedule(
        self,
        token_distribution: Mapping[int, int],
        initial_cache_c2: int = -1,
        initial_cache_c3: int = -1,
        *,
        enable_s4pf: bool = False,
        s4_policy: S4Policy = DEFAULT_S4_POLICY,
        s4_gate: S4Gate = "sync_paired_single",
        fallback: bool = True,
    ) -> HybridScheduleResult:
        normalized = {
            int(eid): int(ntok)
            for eid, ntok in token_distribution.items()
            if int(ntok) > 0
        }
        eligible = certified_olmoe_regime(
            normalized,
            initial_cache_c2,
            initial_cache_c3,
        )
        if not eligible:
            if not fallback:
                raise ValueError("distribution is outside the certified OLMoE envelope")
            return _schedule_protected_t6b2(
                normalized,
                initial_cache_c2,
                initial_cache_c3,
                s4_policy=s4_policy,
            )

        state = self.initial_state(normalized, initial_cache_c2, initial_cache_c3)
        steps: list[HybridStep] = []
        candidate_counts: list[int] = []
        try:
            while state.remaining:
                before = state
                action, child, score, candidate_count = self._choose_one_round(state)
                if enable_s4pf:
                    child, events = _inject_s4_prefetches(
                        child, s4_policy=s4_policy, s4_gate=s4_gate
                    )
                else:
                    events = ()
                # Reapply the selected transition from the exact current state.
                # This catches cache-flag, remaining-set and lowering drift.
                replay_child = reference.apply_action(before, action)
                replay_child = policy._bounded_policy_state(
                    replay_child,
                    POLICY_SCORER,
                    before_f_score=int(before.f_score),
                )
                if enable_s4pf:
                    replay_child, replay_events = _inject_s4_prefetches(
                        replay_child, s4_policy=s4_policy, s4_gate=s4_gate
                    )
                else:
                    replay_events = ()
                if replay_child != child or replay_events != events:
                    raise AssertionError("hybrid transition replay mismatch")
                candidate_counts.append(candidate_count)
                steps.append(
                    HybridStep(
                        action=action,
                        prefetch_events=events,
                        candidate_count=candidate_count,
                        score=score,
                    )
                )
                state = child
        except RuntimeError as exc:
            if "strict token bank has no progress candidate" not in str(exc) or not fallback:
                raise
            protected = _schedule_protected_t6b2(
                normalized,
                initial_cache_c2,
                initial_cache_c3,
                s4_policy=s4_policy,
            )
            return replace(
                protected,
                fallback_reason="candidate_bank_no_progress",
                contract_eligible=True,
            )

        events = [event for step in steps for event in step.prefetch_events]
        return HybridScheduleResult(
            makespan_cc=int(state.g_score),
            route="olmoe_upgrade",
            fallback_reason="none",
            contract_eligible=True,
            steps=tuple(steps),
            candidate_count_max=max(candidate_counts, default=0),
            candidate_count_mean=(
                statistics.mean(candidate_counts) if candidate_counts else 0.0
            ),
            s4pf_single_count=sum(
                event.dma != reference.DmaBinding.BOTH for event in events
            ),
            s4pf_both_count=sum(
                event.dma == reference.DmaBinding.BOTH for event in events
            ),
        )


def adaptive_olmoe_schedule(
    token_distribution: Mapping[int, int],
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
    *,
    enable_s4pf: bool = False,
    s4_policy: S4Policy = DEFAULT_S4_POLICY,
    s4_gate: S4Gate = "sync_paired_single",
    fallback: bool = True,
) -> int:
    return AdaptiveOlmoeScheduler().schedule(
        token_distribution,
        initial_cache_c2,
        initial_cache_c3,
        enable_s4pf=enable_s4pf,
        s4_policy=s4_policy,
        s4_gate=s4_gate,
        fallback=fallback,
    ).makespan_cc
