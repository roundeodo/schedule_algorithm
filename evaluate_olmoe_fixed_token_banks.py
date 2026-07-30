#!/usr/bin/env python3
"""Develop fixed-token scheduler banks against the frozen 65 OLMoE cases.

This is deliberately different from a ``K``-quota candidate generator.  Every
bank below is an explicit, state-relative token ROM: a token names one rank
selection, split rule, start-point index and physical shape rule.  Invalid
tokens are skipped, exactly as ``sched_candidate_generator.sv`` skips invalid
candidate IDs.  One evaluator lane can visit the resulting tokens
sequentially; there is no K-wide datapath and no target-dependent candidate
manufacturing.

The ``rtl_base`` bank must remain transition-equivalent to the current RTL
candidate generator.  Later banks are evidence-led additions and are kept
separate so that their candidate count and benefit can be audited.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import heapq
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Iterable, Optional


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import scheduler_hw_fixed_policy as fixed  # noqa: E402
import scheduler_rtl_adaptive_prefetch_policy as rtl  # noqa: E402
import four_stage_scheduler as reference  # noqa: E402
import analyze_window_witness_action_templates as witness_templates  # noqa: E402
from run_four_stage_reference import deserialize_action, serialize_action  # noqa: E402


DEFAULT_PROOF = (
    HERE / "results" / "policy_search" / "olmoe_top2_projection_65_optimal_v1.json"
)
DEFAULT_OUTPUT = (
    HERE / "results" / "policy_search" / "olmoe_65_fixed_token_banks_v1.json"
)
DEFAULT_WINDOW_AUDIT = (
    HERE
    / "results"
    / "policy_search"
    / "window_exact"
    / "olmoe_65_stagec_top8_coverage_audit_v1.json"
)
TICK_CC = rtl.TICK_CC
EXPLICIT_WINDOW = (8, 2)

if TICK_CC != reference.SCHEDULE_TIME_QUANTUM_CC:
    raise RuntimeError("scalar and explicit-DMA models use different tick scales")


@dataclass(frozen=True)
class BankSpec:
    name: str
    window: tuple[int, int]
    sync_pairs: tuple[tuple[str, str], ...]
    sync_splits: tuple[tuple[str, str], ...]
    one_idle_ranks: tuple[str, ...]
    sync_single_ranks: tuple[str, ...] = ()
    pair_profiles: tuple[str, ...] = ("RTL_PAIR",)


@dataclass(frozen=True)
class TargetSearchResult:
    feasible: bool
    exhaustive: bool
    termination: str
    trace: tuple[fixed.ScheduleStep, ...]
    expansions: int
    generated: int
    pruned_by_lower_bound: int
    memoized_dead_states: int
    runtime_s: float


@dataclass(frozen=True, order=True)
class ExplicitLogicalToken:
    """One state-relative rank/family token in the explicit-DMA model.

    Pair selectors are canonicalized as an unordered pair.  Cluster placement,
    shape, DMA lane and prefetch choices belong to ``ExplicitPhysicalProfile``
    and therefore remain real evaluated candidate axes.
    """

    mode: str
    family: str
    selectors: tuple[str, ...]
    split_rule: str = "NONE"


@dataclass(frozen=True, order=True)
class ExplicitPhysicalProfile:
    """Concrete physical fields of one fixed candidate token.

    Absolute starts are deliberately excluded.  Every legal event-aligned
    start matching this profile is emitted and counted as a separate concrete
    candidate; no hidden local selector chooses one on behalf of the scorer.
    """

    c2_s1: str
    c2_s3: str
    c3_s1: str
    c3_s3: str
    c2_dma_s1: str
    c2_dma_s3: str
    c2_s2pf: str
    c3_dma_s1: str
    c3_dma_s3: str
    c3_s2pf: str
    s4pf_dma: str
    c2_s1_cached: bool
    c2_s3_cached: bool
    c3_s1_cached: bool
    c3_s3_cached: bool


@dataclass(frozen=True, order=True)
class ExplicitCandidateToken:
    logical: ExplicitLogicalToken
    physical: ExplicitPhysicalProfile


# Current RTL token table:
#   SYNC     = pair 01/12/23 + split top0 half/front2
#   ONE_IDLE = top0 at up to three start points, fixed C/C + adaptive shape
#   TERMINAL = unchanged 10 solo + half split + up to two release tokens
RTL_BASE = BankSpec(
    name="rtl_base",
    window=(4, 0),
    sync_pairs=(("T0", "T1"), ("T1", "T2"), ("T2", "T3")),
    sync_splits=(("T0", "HALF"), ("T0", "FRONT2")),
    one_idle_ranks=("T0",),
)


# First fixed-token probe for the proven-sufficient top8+bottom2 window.  This
# is intentionally a named ROM, not a frequency-filled budget.  It extends the
# current adjacent-pair pattern through top8, adds two explicit cold pairings,
# permits the four hottest ranks to split, and exposes the coldest rank while a
# cluster is idle.  Results decide which entries survive; this is not yet the
# final bank.
TOP8_COLD1_PROBE = BankSpec(
    name="top8_bottom2_cold1_probe",
    window=(8, 2),
    sync_pairs=(
        ("T0", "T1"),
        ("T1", "T2"),
        ("T2", "T3"),
        ("T3", "T4"),
        ("T4", "T5"),
        ("T5", "T6"),
        ("T6", "T7"),
        ("T0", "B0"),
        ("B1", "B0"),
    ),
    sync_splits=(
        ("T0", "HALF"),
        ("T0", "FRONT2"),
        ("T1", "HALF"),
        ("T1", "FRONT2"),
        ("T2", "HALF"),
        ("T2", "FRONT2"),
        ("T3", "HALF"),
        ("T3", "FRONT2"),
    ),
    one_idle_ranks=("T0", "B0"),
    sync_single_ranks=("T0", "B0"),
    pair_profiles=("RTL_PAIR", "HOT_COLD_PAIR"),
)


TOP8_OBSERVED_WITNESS_PROBE = BankSpec(
    name="top8_bottom2_observed_witness_probe",
    window=TOP8_COLD1_PROBE.window,
    sync_pairs=TOP8_COLD1_PROBE.sync_pairs
    + (
        ("T0", "T7"),
        ("T1", "T7"),
        ("T2", "T7"),
        ("T3", "T7"),
        ("T4", "T7"),
        ("T1", "T5"),
        ("T0", "T3"),
    ),
    sync_splits=TOP8_COLD1_PROBE.sync_splits,
    one_idle_ranks=TOP8_COLD1_PROBE.one_idle_ranks,
    sync_single_ranks=TOP8_COLD1_PROBE.sync_single_ranks,
    pair_profiles=TOP8_COLD1_PROBE.pair_profiles,
)


BANKS = {
    bank.name: bank
    for bank in (RTL_BASE, TOP8_COLD1_PROBE, TOP8_OBSERVED_WITNESS_PROBE)
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ticks_text(cc: int) -> str:
    value = Fraction(int(cc), TICK_CC)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _target_cc(case: dict) -> int:
    value = Fraction(str(case["best_reference_ticks"])) * TICK_CC
    if value.denominator != 1:
        raise ValueError(f"{case['name']}: non-integral target {value}")
    return int(value)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _rank_index(label: str, count: int, window: tuple[int, int]) -> int | None:
    top, bottom = window
    if label.startswith("T"):
        rank = int(label[1:])
        return rank if rank < top and rank < count else None
    if label.startswith("B"):
        offset = int(label[1:])
        if offset >= bottom or offset >= count:
            return None
        index = count - 1 - offset  # B0 is the coldest visible expert.
        # A small tail may belong to both top and bottom.  It is still one legal
        # descriptor; state-key deduplication removes duplicate transitions.
        return index
    raise ValueError(f"unknown rank selector {label!r}")


def _selected(
    remaining: fixed.Remaining, label: str, window: tuple[int, int]
) -> tuple[int, int] | None:
    index = _rank_index(label, len(remaining), window)
    return remaining[index] if index is not None else None


def _append_pair(
    out: list[fixed.Transition],
    *,
    c2,
    c3,
    remaining: fixed.Remaining,
    left: tuple[int, int],
    right: tuple[int, int],
    tag: str,
    family: str = "pair_kj",
    profile: str = "RTL_PAIR",
) -> None:
    if left[0] == right[0]:
        return
    now = c2.task_end
    left_eid, left_ntok = left
    right_eid, right_ntok = right
    sw2 = fixed.cm._cc_swiglu_hit(left_eid, c2, now)
    dn2 = fixed.cm._cc_down_hit(left_eid, c2, now)
    sw3 = fixed.cm._cc_swiglu_hit(right_eid, c3, now)
    dn3 = fixed.cm._cc_down_hit(right_eid, c3, now)
    rem_after = fixed.cm._cc_remove_eids(remaining, left_eid, right_eid)
    if profile == "RTL_PAIR":
        s12, s32, s13, s33 = fixed.cm._cc_pick_shapes(
            left_ntok, right_ntok, sw2, dn2, sw3, dn3, now
        )
        fixed._append_pair(
            out,
            family=family,
            tag=tag,
            sa=fixed.cm._cc_mk_snap(
                now, s12, s32, left_ntok, left_eid, sw2, dn2
            ),
            s3a=s32,
            sb=fixed.cm._cc_mk_snap(
                now, s13, s33, right_ntok, right_eid, sw3, dn3
            ),
            s3b=s33,
            remaining=rem_after,
            policy="balanced",
        )
        return
    if profile != "HOT_COLD_PAIR":
        raise ValueError(profile)

    # One additional fixed physical token distilled from the dominant optimal
    # hot/cold profile.  Uncached ntok>=7 uses A/B; smaller work uses B/B.
    # Try S2PF only for the larger side, then the other side, then raw.  This is
    # a fixed three-entry legality microsequence, not a scored hidden oracle.
    s12 = fixed.cm.C_SHAPE_C if sw2 else (
        fixed.cm.C_SHAPE_A if left_ntok >= 7 else fixed.cm.C_SHAPE_B
    )
    s13 = fixed.cm.C_SHAPE_C if sw3 else (
        fixed.cm.C_SHAPE_A if right_ntok >= 7 else fixed.cm.C_SHAPE_B
    )
    s32 = fixed.cm.C_SHAPE_B
    s33 = fixed.cm.C_SHAPE_B
    sa = fixed.cm._cc_mk_snap(now, s12, s32, left_ntok, left_eid, sw2, dn2)
    sb = fixed.cm._cc_mk_snap(now, s13, s33, right_ntok, right_eid, sw3, dn3)
    order = (0, 1) if left_ntok >= right_ntok else (1, 0)
    selected_pair = None
    for side in order:
        if side == 0:
            trial_a = fixed.cm._cc_apply_s2pf(sa, s32, sa.dma1_end)
            trial = (trial_a, sb) if trial_a.s2pf_start >= 0 else None
        else:
            trial_b = fixed.cm._cc_apply_s2pf(sb, s33, sb.dma1_end)
            trial = (sa, trial_b) if trial_b.s2pf_start >= 0 else None
        if trial is not None and fixed.cm._cc_bw_ok(*trial):
            selected_pair = trial
            break
    if selected_pair is None and fixed.cm._cc_bw_ok(sa, sb):
        selected_pair = (sa, sb)
    if selected_pair is not None:
        out.append(
            fixed.Transition(
                fixed.PolicyState(selected_pair[0], selected_pair[1], rem_after),
                tag,
            )
        )


def _append_split(
    out: list[fixed.Transition],
    *,
    c2,
    c3,
    remaining: fixed.Remaining,
    selected: tuple[int, int],
    cut_rule: str,
    tag: str,
) -> None:
    eid, ntok = selected
    if cut_rule == "HALF":
        left = (ntok + 1) // 2
    elif cut_rule == "FRONT2":
        left = 2
    else:
        raise ValueError(cut_rule)
    if not 1 <= left < ntok:
        return
    right = ntok - left
    now = c2.task_end
    sw2 = fixed.cm._cc_swiglu_hit(eid, c2, now)
    dn2 = fixed.cm._cc_down_hit(eid, c2, now)
    sw3 = fixed.cm._cc_swiglu_hit(eid, c3, now)
    dn3 = fixed.cm._cc_down_hit(eid, c3, now)
    s12, s32, s13, s33 = fixed.cm._cc_pick_shapes(
        left, right, sw2, dn2, sw3, dn3, now
    )
    fixed._append_pair(
        out,
        family="split_top0",
        tag=tag,
        sa=fixed.cm._cc_mk_snap(now, s12, s32, left, eid, sw2, dn2),
        s3a=s32,
        sb=fixed.cm._cc_mk_snap(now, s13, s33, right, eid, sw3, dn3),
        s3b=s33,
        remaining=fixed.cm._cc_remove_eids(remaining, eid),
        policy="balanced",
    )


def _single_snap(
    eid: int,
    ntok: int,
    own,
    peer,
    start: int,
    cluster: int,
    profile: str,
):
    sw_hit = fixed.cm._cc_swiglu_hit(eid, own, start)
    down_hit = fixed.cm._cc_down_hit(eid, own, start)
    if profile == "FIXED_CC":
        s1 = s3 = fixed.cm.C_SHAPE_C
    elif profile == "ADAPTIVE":
        s1, s3 = fixed._adaptive_uncached_shapes(ntok)
    else:
        raise ValueError(profile)
    snap = fixed.cm._cc_mk_snap(start, s1, s3, ntok, eid, sw_hit, down_hit)
    if profile == "FIXED_CC" and snap.bw_s3 > 0:
        prefetched = fixed.cm._cc_apply_s2pf(snap, s3, snap.dma1_end)
        if prefetched.s2pf_start >= 0:
            prefetch_ok = (
                fixed.cm._cc_bw_ok(prefetched, peer)
                if cluster == 0
                else fixed.cm._cc_bw_ok(peer, prefetched)
            )
            if prefetch_ok:
                snap = prefetched
    feasible = (
        fixed.cm._cc_bw_ok(snap, peer)
        if cluster == 0
        else fixed.cm._cc_bw_ok(peer, snap)
    )
    return snap if feasible else None


def _append_single(
    out: list[fixed.Transition],
    *,
    c2,
    c3,
    remaining: fixed.Remaining,
    selected: tuple[int, int],
    cluster: int,
    start: int,
    profile: str,
    tag: str,
) -> None:
    eid, ntok = selected
    own, peer = (c2, c3) if cluster == 0 else (c3, c2)
    snap = _single_snap(eid, ntok, own, peer, start, cluster, profile)
    if snap is None:
        return
    next_c2, next_c3 = (snap, peer) if cluster == 0 else (peer, snap)
    out.append(
        fixed.Transition(
            fixed.PolicyState(
                next_c2,
                next_c3,
                fixed.cm._cc_remove_eids(remaining, eid),
            ),
            tag,
        )
    )


def generate_bank_successors(
    state: fixed.PolicyState, bank: BankSpec
) -> list[fixed.Transition]:
    """Materialize one explicit fixed token ROM for the current round."""
    if not state.remaining:
        return []
    c2, c3 = fixed._prepare(state.c2, state.c3)
    if len(state.remaining) == 1:
        return fixed._terminal_successors(
            c2,
            c3,
            state.remaining,
            policy="balanced",
            n1_policy="pruned",
        )

    out: list[fixed.Transition] = []
    if c2.task_end == c3.task_end:
        for token_id, (left_label, right_label) in enumerate(bank.sync_pairs):
            left = _selected(state.remaining, left_label, bank.window)
            right = _selected(state.remaining, right_label, bank.window)
            if left is None or right is None:
                continue
            for profile in bank.pair_profiles:
                _append_pair(
                    out,
                    c2=c2,
                    c3=c3,
                    remaining=state.remaining,
                    left=left,
                    right=right,
                    tag=(
                        f"bank_pair_{token_id}_{left_label}_{right_label}_"
                        f"{profile}"
                    ),
                    profile=profile,
                )
        for token_id, (rank_label, cut_rule) in enumerate(bank.sync_splits):
            selected = _selected(state.remaining, rank_label, bank.window)
            if selected is None:
                continue
            _append_split(
                out,
                c2=c2,
                c3=c3,
                remaining=state.remaining,
                selected=selected,
                cut_rule=cut_rule,
                tag=f"bank_split_{token_id}_{rank_label}_{cut_rule}",
            )
        for token_id, rank_label in enumerate(bank.sync_single_ranks):
            selected = _selected(state.remaining, rank_label, bank.window)
            if selected is None:
                continue
            for cluster in (0, 1):
                for profile in ("FIXED_CC", "ADAPTIVE"):
                    _append_single(
                        out,
                        c2=c2,
                        c3=c3,
                        remaining=state.remaining,
                        selected=selected,
                        cluster=cluster,
                        start=c2.task_end,
                        profile=profile,
                        tag=(
                            f"bank_sync_single_{token_id}_{rank_label}_"
                            f"c{cluster + 2}_{profile}"
                        ),
                    )
    else:
        idle_cluster = 0 if c2.task_end < c3.task_end else 1
        idle, busy = (c2, c3) if idle_cluster == 0 else (c3, c2)
        release_points = fixed.cm._cc_busy_time_points(busy, idle.task_end)
        for token_id, rank_label in enumerate(bank.one_idle_ranks):
            selected = _selected(state.remaining, rank_label, bank.window)
            if selected is None:
                continue
            for point_index, start in enumerate(release_points):
                for profile in ("FIXED_CC", "ADAPTIVE"):
                    _append_single(
                        out,
                        c2=c2,
                        c3=c3,
                        remaining=state.remaining,
                        selected=selected,
                        cluster=idle_cluster,
                        start=start,
                        profile=profile,
                        tag=(
                            f"bank_one_idle_{token_id}_{rank_label}_"
                            f"c{idle_cluster + 2}_p{point_index}_{profile}"
                        ),
                    )
    unique: dict[tuple, fixed.Transition] = {}
    for transition in out:
        unique.setdefault(fixed.state_key(transition.state), transition)
    if not unique:
        raise RuntimeError(
            f"{bank.name}: no legal token for state with {len(state.remaining)} remaining"
        )
    return list(unique.values())


def _candidate_key(transition: fixed.Transition, before: fixed.PolicyState) -> tuple:
    child = transition.state
    current = max(child.c2.task_end, child.c3.task_end)
    if len(before.remaining) == 1 or before.c2.task_end != before.c3.task_end:
        cost = current
    else:
        cost = fixed.hw_v2_continuation(
            child.c2, child.c3, child.remaining, policy="balanced"
        )
    return cost, len(child.remaining), current


def run_greedy_bank(
    token_dist: dict[int, int], bank: BankSpec
) -> tuple[int, tuple[fixed.ScheduleStep, ...], dict]:
    cost_model = rtl._COST_MODELS[rtl.DEFAULT_S4_POLICY]
    with rtl._use_cost_model(cost_model):
        state = fixed.initial_state(token_dist)
        trace: list[fixed.ScheduleStep] = []
        counts: list[int] = []
        mode_counts: Counter[str] = Counter()
        while state.remaining:
            transitions = generate_bank_successors(state, bank)
            counts.append(len(transitions))
            mode = (
                "TERMINAL"
                if len(state.remaining) == 1
                else "SYNC"
                if state.c2.task_end == state.c3.task_end
                else "ONE_IDLE"
            )
            mode_counts[mode] += 1
            chosen = min(transitions, key=lambda item: _candidate_key(item, state))
            trace.append(fixed.ScheduleStep(state, chosen.state, chosen.tag))
            state = chosen.state
        return (
            fixed.terminal_cost(state),
            tuple(trace),
            {
                "rounds": len(trace),
                "candidate_count_max": max(counts),
                "candidate_count_mean": statistics.mean(counts),
                "mode_rounds": dict(sorted(mode_counts.items())),
            },
        )


def _divisible_remaining_work_lb(state: fixed.PolicyState) -> int:
    """Safe two-processor LB using perfectly divisible, cache-free ideal work.

    ``_cc_best_task`` omits every DMA cost and gives each remaining expert its
    ideal serial compute duration.  Treating the sum as arbitrarily divisible
    between processors released at the two current task ends is a relaxation,
    so the resulting completion time cannot exceed a legal schedule optimum.
    """
    early, late = sorted((int(state.c2.task_end), int(state.c3.task_end)))
    work = sum(fixed.cm._cc_best_task(int(ntok)) for _eid, ntok in state.remaining)
    if early + work <= late:
        return late
    return max(late, (early + late + work + 1) // 2)


def target_search(
    token_dist: dict[int, int],
    bank: BankSpec,
    target_cc: int,
    *,
    time_limit_s: float = 300.0,
    expansion_limit: int = 2_000_000,
) -> TargetSearchResult:
    """Find a bank-only path at or below the certified target.

    Success is constructive.  Failure is a proof of candidate insufficiency
    only when ``exhaustive`` is true; time/expansion termination is unresolved.
    The target never affects candidate generation or ordering.
    """
    started = time.perf_counter()
    cost_model = rtl._COST_MODELS[rtl.DEFAULT_S4_POLICY]
    dead: set[tuple] = set()
    expansions = 0
    generated = 0
    pruned = 0
    termination = "exhausted"
    found_trace: Optional[tuple[fixed.ScheduleStep, ...]] = None

    class _StopSearch(Exception):
        pass

    with rtl._use_cost_model(cost_model):
        root = fixed.initial_state(token_dist)

        def visit(
            state: fixed.PolicyState,
            trace: tuple[fixed.ScheduleStep, ...],
        ) -> bool:
            nonlocal expansions, generated, pruned, termination, found_trace
            if time.perf_counter() - started >= time_limit_s:
                termination = "time_limit"
                raise _StopSearch
            if expansions >= expansion_limit:
                termination = "expansion_limit"
                raise _StopSearch
            lower_bound = _divisible_remaining_work_lb(state)
            if lower_bound > target_cc:
                pruned += 1
                return False
            if not state.remaining:
                if fixed.terminal_cost(state) <= target_cc:
                    found_trace = trace
                    termination = "target_reached"
                    return True
                return False
            key = fixed.state_key(state)
            if key in dead:
                return False
            expansions += 1
            transitions = generate_bank_successors(state, bank)
            generated += len(transitions)
            # Search ordering is not candidate generation and is not part of the
            # runtime policy.  Prefer the smallest safe child lower bound so a
            # constructive target path is found before exploring looser
            # branches; retain the RTL scorer only as the deterministic tie.
            transitions.sort(
                key=lambda item: (
                    _divisible_remaining_work_lb(item.state),
                    _candidate_key(item, state),
                )
            )
            for transition in transitions:
                step = fixed.ScheduleStep(state, transition.state, transition.tag)
                if visit(transition.state, trace + (step,)):
                    return True
            dead.add(key)
            return False

        try:
            feasible = visit(root, ())
        except _StopSearch:
            feasible = False

    exhaustive = termination in {"exhausted", "target_reached"}
    return TargetSearchResult(
        feasible=feasible,
        exhaustive=exhaustive,
        termination=termination,
        trace=found_trace or (),
        expansions=expansions,
        generated=generated,
        pruned_by_lower_bound=pruned,
        memoized_dead_states=len(dead),
        runtime_s=time.perf_counter() - started,
    )


def _explicit_family(action: reference.StageAction) -> str:
    if action.c2_eid >= 0 and action.c2_eid == action.c3_eid:
        return "SPLIT"
    if action.c2_eid >= 0 and action.c3_eid >= 0:
        return "PAIR"
    if action.c2_eid >= 0 or action.c3_eid >= 0:
        return "SINGLE"
    if action.pf_eid >= 0:
        return "PREFETCH"
    return "OTHER"


def _explicit_mode(state: reference.BeamState) -> str:
    if len(state.remaining) == 1:
        return "TERMINAL"
    return "SYNC" if state.c2.task_end == state.c3.task_end else "ONE_IDLE"


def _explicit_rank_label(
    state: reference.BeamState,
    eid: int,
    window: tuple[int, int] = EXPLICIT_WINDOW,
) -> str:
    if eid < 0:
        return "NONE"
    rank_by_eid = {
        int(candidate_eid): rank
        for rank, (candidate_eid, _ntok) in enumerate(state.remaining)
    }
    rank = rank_by_eid.get(int(eid))
    if rank is None:
        raise ValueError(f"E{eid} is not in remaining")
    top, bottom = window
    entries = len(state.remaining)
    if rank < min(top, entries):
        return f"T{rank}"
    if bottom and rank >= max(min(top, entries), entries - bottom):
        return f"B{entries - 1 - rank}"
    resident = "".join(
        str(cluster)
        for cluster, snap in ((2, state.c2), (3, state.c3))
        if snap.pf_eid == eid
    )
    if resident:
        return f"R{resident}"
    raise ValueError(f"E{eid} is outside top8+bottom2 and is not resident")


def _explicit_split_rule(action: reference.StageAction) -> str:
    if _explicit_family(action) != "SPLIT":
        return "NONE"
    low, high = sorted((int(action.c2_ntok), int(action.c3_ntok)))
    return "HALF" if low == high else f"CUT{low}"


def _explicit_physical_profile(
    action: reference.StageAction,
) -> ExplicitPhysicalProfile:
    shape_name = lambda shape: "NONE" if shape is None else str(shape.name)
    return ExplicitPhysicalProfile(
        c2_s1=shape_name(action.c2_shape_s1),
        c2_s3=shape_name(action.c2_shape_s3),
        c3_s1=shape_name(action.c3_shape_s1),
        c3_s3=shape_name(action.c3_shape_s3),
        c2_dma_s1=action.c2_dma_s1.name,
        c2_dma_s3=action.c2_dma_s3.name,
        c2_s2pf=action.c2_s2pf_dma.name,
        c3_dma_s1=action.c3_dma_s1.name,
        c3_dma_s3=action.c3_dma_s3.name,
        c3_s2pf=action.c3_s2pf_dma.name,
        s4pf_dma=action.pf_dma.name,
        c2_s1_cached=bool(action.c2_s1_cached),
        c2_s3_cached=bool(action.c2_s3_cached),
        c3_s1_cached=bool(action.c3_s1_cached),
        c3_s3_cached=bool(action.c3_s3_cached),
    )


def _explicit_logical_token(
    state: reference.BeamState,
    action: reference.StageAction,
    window: tuple[int, int] = EXPLICIT_WINDOW,
) -> ExplicitLogicalToken:
    family = _explicit_family(action)
    if family == "PAIR":
        selectors = tuple(
            sorted(
                (
                    _explicit_rank_label(state, action.c2_eid, window),
                    _explicit_rank_label(state, action.c3_eid, window),
                )
            )
        )
    elif family == "SPLIT":
        selectors = (_explicit_rank_label(state, action.c2_eid, window),)
    elif family == "SINGLE":
        eid = action.c2_eid if action.c2_eid >= 0 else action.c3_eid
        selectors = (_explicit_rank_label(state, eid, window),)
    elif family == "PREFETCH":
        selectors = (_explicit_rank_label(state, action.pf_eid, window),)
    else:
        raise ValueError(f"unsupported explicit action family {family}")
    return ExplicitLogicalToken(
        mode=_explicit_mode(state),
        family=family,
        selectors=selectors,
        split_rule=_explicit_split_rule(action),
    )


def _explicit_candidate_token(
    state: reference.BeamState,
    action: reference.StageAction,
    window: tuple[int, int] = EXPLICIT_WINDOW,
) -> ExplicitCandidateToken:
    return ExplicitCandidateToken(
        logical=_explicit_logical_token(state, action, window),
        physical=_explicit_physical_profile(action),
    )


def _resolve_explicit_selector(
    state: reference.BeamState,
    selector: str,
    window: tuple[int, int] = EXPLICIT_WINDOW,
) -> int | None:
    entries = len(state.remaining)
    top, bottom = window
    if selector.startswith("T"):
        rank = int(selector[1:])
        return int(state.remaining[rank][0]) if rank < min(top, entries) else None
    if selector.startswith("B"):
        offset = int(selector[1:])
        if offset >= bottom or offset >= entries:
            return None
        rank = entries - 1 - offset
        if rank < min(top, entries):
            # Overlapping top/bottom windows still name one descriptor.
            return int(state.remaining[rank][0])
        return int(state.remaining[rank][0])
    if selector.startswith("R"):
        resident_ids = []
        for cluster in selector[1:]:
            snap = state.c2 if cluster == "2" else state.c3 if cluster == "3" else None
            if snap is None or snap.pf_eid < 0:
                return None
            resident_ids.append(int(snap.pf_eid))
        if not resident_ids or len(set(resident_ids)) != 1:
            return None
        eid = resident_ids[0]
        return eid if any(item[0] == eid for item in state.remaining) else None
    raise ValueError(f"unknown explicit selector {selector!r}")


def _explicit_child_key(state: reference.BeamState) -> tuple:
    """Future-exact continuation class plus accumulated cluster work.

    ``four_stage_scheduler`` deliberately suppresses the opposite placement of
    equal-load, equally-resident expert IDs.  Its canonical fingerprint proves
    those states have identical legal continuations; retaining cluster work
    preserves the exact-search dominance contract.
    """
    return (state.fingerprint(), int(state.cluster_work_cc))


def _raw_actions_for_explicit_logical(
    state: reference.BeamState,
    logical: ExplicitLogicalToken,
    window: tuple[int, int] = EXPLICIT_WINDOW,
) -> list[reference.StageAction]:
    if logical.mode != _explicit_mode(state):
        return []
    selected = tuple(
        _resolve_explicit_selector(state, selector, window)
        for selector in logical.selectors
    )
    if any(eid is None for eid in selected):
        return []
    eids = tuple(int(eid) for eid in selected if eid is not None)
    if logical.family == "PAIR" and len(set(eids)) != 2:
        return []
    subset = tuple(item for item in state.remaining if item[0] in set(eids))
    if len(subset) != len(set(eids)):
        return []
    if logical.family == "PREFETCH":
        raw = reference.gen_prefetch_actions(
            state.c2,
            state.c3,
            subset,
            seed_mode=False,
            seed_all_visible=True,
        )
    else:
        raw = reference.gen_stage_actions(
            state.c2,
            state.c3,
            subset,
            seed_mode=False,
            seed_all_visible=True,
        )
    return [
        action
        for action in raw
        if _explicit_logical_token(state, action, window) == logical
    ]


_EXPLICIT_SHAPES = {
    "NONE": None,
    reference.SHAPE_A.name: reference.SHAPE_A,
    reference.SHAPE_B.name: reference.SHAPE_B,
    reference.SHAPE_C.name: reference.SHAPE_C,
}


def _explicit_shape(name: str) -> reference.Shape | None:
    try:
        return _EXPLICIT_SHAPES[name]
    except KeyError as exc:
        raise ValueError(f"unknown explicit shape {name!r}") from exc


def _explicit_dma(name: str) -> reference.DmaBinding:
    try:
        return reference.DmaBinding[name]
    except KeyError as exc:
        raise ValueError(f"unknown explicit DMA binding {name!r}") from exc


def _direct_pair_action(
    *,
    eid_a: int,
    ntok_a: int,
    shape_a_s1: reference.Shape,
    shape_a_s3: reference.Shape,
    start_a: int,
    s1_hit_a: bool,
    snap_a: reference.FourStageSnap,
    eid_b: int,
    ntok_b: int,
    shape_b_s1: reference.Shape,
    shape_b_s3: reference.Shape,
    start_b: int,
    s1_hit_b: bool,
    snap_b: reference.FourStageSnap,
    tag: str,
) -> reference.StageAction:
    return reference.StageAction(
        c2_eid=eid_a,
        c2_ntok=ntok_a,
        c2_shape_s1=shape_a_s1,
        c2_shape_s3=shape_a_s3,
        c2_start=start_a,
        c2_s1_cached=s1_hit_a,
        c2_s3_cached=snap_a.bw_s3 == 0,
        c3_eid=eid_b,
        c3_ntok=ntok_b,
        c3_shape_s1=shape_b_s1,
        c3_shape_s3=shape_b_s3,
        c3_start=start_b,
        c3_s1_cached=s1_hit_b,
        c3_s3_cached=snap_b.bw_s3 == 0,
        pf_cluster=-1,
        pf_eid=-1,
        pf_shape=None,
        pf_start=-1,
        tag=tag,
        c2_s2pf_start=snap_a.s2pf_start,
        c3_s2pf_start=snap_b.s2pf_start,
        c2_dma_s1=snap_a.dma_s1,
        c2_dma_s3=snap_a.dma_s3,
        c2_s2pf_dma=snap_a.s2pf_dma,
        c3_dma_s1=snap_b.dma_s1,
        c3_dma_s3=snap_b.dma_s3,
        c3_s2pf_dma=snap_b.s2pf_dma,
    )


def _direct_single_action(
    *,
    cluster: int,
    eid: int,
    ntok: int,
    shape_s1: reference.Shape,
    shape_s3: reference.Shape,
    start: int,
    s1_hit: bool,
    snap: reference.FourStageSnap,
) -> reference.StageAction:
    common = dict(
        pf_cluster=-1,
        pf_eid=-1,
        pf_shape=None,
        pf_start=-1,
        tag=f"DIRECT-SINGLE-C{cluster}(E{eid})",
    )
    if cluster == 2:
        return reference.StageAction(
            c2_eid=eid,
            c2_ntok=ntok,
            c2_shape_s1=shape_s1,
            c2_shape_s3=shape_s3,
            c2_start=start,
            c2_s1_cached=s1_hit,
            c2_s3_cached=snap.bw_s3 == 0,
            c3_eid=-1,
            c3_ntok=0,
            c3_shape_s1=None,
            c3_shape_s3=None,
            c3_start=-1,
            c3_s1_cached=False,
            c3_s3_cached=False,
            c2_s2pf_start=snap.s2pf_start,
            c2_dma_s1=snap.dma_s1,
            c2_dma_s3=snap.dma_s3,
            c2_s2pf_dma=snap.s2pf_dma,
            **common,
        )
    if cluster == 3:
        return reference.StageAction(
            c2_eid=-1,
            c2_ntok=0,
            c2_shape_s1=None,
            c2_shape_s3=None,
            c2_start=-1,
            c2_s1_cached=False,
            c2_s3_cached=False,
            c3_eid=eid,
            c3_ntok=ntok,
            c3_shape_s1=shape_s1,
            c3_shape_s3=shape_s3,
            c3_start=start,
            c3_s1_cached=s1_hit,
            c3_s3_cached=snap.bw_s3 == 0,
            c3_s2pf_start=snap.s2pf_start,
            c3_dma_s1=snap.dma_s1,
            c3_dma_s3=snap.dma_s3,
            c3_s2pf_dma=snap.s2pf_dma,
            **common,
        )
    raise ValueError(f"invalid cluster {cluster}")


def _source_s3_bindings(
    final_dma: reference.DmaBinding,
    s2pf_dma: reference.DmaBinding,
) -> tuple[reference.DmaBinding, ...]:
    # S2PF replaces the original S3 transfer before it can execute.  Its source
    # binding is therefore not future-observable.  Use one canonical non-zero
    # binding instead of enumerating three reference-only aliases.
    if s2pf_dma != reference.DmaBinding.NONE:
        return (reference.DmaBinding.IDMA,)
    return (final_dma,)


def _apply_direct_s2pf(
    snap: reference.FourStageSnap,
    shape_s3: reference.Shape,
    binding: reference.DmaBinding,
) -> reference.FourStageSnap:
    if binding == reference.DmaBinding.NONE:
        return snap
    # The lowered task contains has_s2pf but no arbitrary timestamp.  The only
    # realizable release is the final S1-load endpoint, exactly as in the RTL.
    return snap.with_s2_down_prefetch(shape_s3, snap.dma1_end, binding)


def _direct_single_cluster_is_legal(
    state: reference.BeamState, cluster: int
) -> bool:
    t2, t3 = int(state.c2.task_end), int(state.c3.task_end)
    if t2 < t3:
        return cluster == 2 or (
            cluster == 3 and reference._reserved_next_eid(state.c3) >= 0
        )
    if t3 < t2:
        return cluster == 3 or (
            cluster == 2 and reference._reserved_next_eid(state.c2) >= 0
        )
    return cluster == 2 or (cluster == 3 and state.c2 != state.c3)


def _direct_materialize_explicit_token(
    state: reference.BeamState,
    token: ExplicitCandidateToken,
    window: tuple[int, int] = EXPLICIT_WINDOW,
) -> tuple[list[reference.StageAction], dict]:
    """Directly lower one fixed token without calling a reference generator.

    This is the bounded runtime path.  It supports the PAIR/SINGLE families in
    the compressed bank and uses only token-selected shapes/bindings plus the
    finite event-aligned start and S2PF endpoint sets.  The full reference
    generator remains an audit oracle and is never called here.
    """
    logical, physical = token.logical, token.physical
    if logical.mode != _explicit_mode(state):
        return [], {"profile_attempts": 0, "low_level_variants": 0}
    if logical.family not in ("PAIR", "SPLIT", "SINGLE", "PREFETCH"):
        raise ValueError(f"direct bank does not support {logical}")
    if logical.family != "SPLIT" and logical.split_rule != "NONE":
        raise ValueError(f"non-SPLIT token has split rule {logical.split_rule}")
    if (
        logical.family != "PREFETCH"
        and _explicit_dma(physical.s4pf_dma) != reference.DmaBinding.NONE
    ):
        raise ValueError("consuming direct token cannot attach standalone S4PF")

    selected = tuple(
        _resolve_explicit_selector(state, selector, window)
        for selector in logical.selectors
    )
    if any(eid is None for eid in selected):
        return [], {"profile_attempts": 0, "low_level_variants": 0}
    selected_eids = tuple(int(eid) for eid in selected if eid is not None)
    ntok_by_eid = {int(eid): int(ntok) for eid, ntok in state.remaining}
    if any(eid not in ntok_by_eid for eid in selected_eids):
        return [], {"profile_attempts": 0, "low_level_variants": 0}

    actions: dict[tuple, reference.StageAction] = {}
    profile_attempts = 0
    low_level_variants = 0

    if logical.family in ("PAIR", "SPLIT"):
        if logical.family == "PAIR" and (
            len(selected_eids) != 2 or selected_eids[0] == selected_eids[1]
        ):
            return [], {"profile_attempts": 0, "low_level_variants": 0}
        if logical.family == "SPLIT" and len(selected_eids) != 1:
            return [], {"profile_attempts": 0, "low_level_variants": 0}
        shape_a_s1 = _explicit_shape(physical.c2_s1)
        shape_a_s3 = _explicit_shape(physical.c2_s3)
        shape_b_s1 = _explicit_shape(physical.c3_s1)
        shape_b_s3 = _explicit_shape(physical.c3_s3)
        if None in (shape_a_s1, shape_a_s3, shape_b_s1, shape_b_s3):
            raise ValueError("PAIR token has an inactive shape")
        dma_a_s1 = _explicit_dma(physical.c2_dma_s1)
        dma_b_s1 = _explicit_dma(physical.c3_dma_s1)
        dma_a_s2pf = _explicit_dma(physical.c2_s2pf)
        dma_b_s2pf = _explicit_dma(physical.c3_s2pf)
        source_a_s3 = _source_s3_bindings(
            _explicit_dma(physical.c2_dma_s3), dma_a_s2pf
        )
        source_b_s3 = _source_s3_bindings(
            _explicit_dma(physical.c3_dma_s3), dma_b_s2pf
        )
        now = max(int(state.c2.task_end), int(state.c3.task_end))
        if logical.family == "PAIR":
            work_assignments = (
                (
                    selected_eids[0],
                    ntok_by_eid[selected_eids[0]],
                    selected_eids[1],
                    ntok_by_eid[selected_eids[1]],
                ),
                (
                    selected_eids[1],
                    ntok_by_eid[selected_eids[1]],
                    selected_eids[0],
                    ntok_by_eid[selected_eids[0]],
                ),
            )
        else:
            split_eid = selected_eids[0]
            split_ntok = ntok_by_eid[split_eid]
            if logical.split_rule == "HALF":
                if split_ntok % 2:
                    return [], {"profile_attempts": 0, "low_level_variants": 0}
                cuts = (split_ntok // 2,)
            elif logical.split_rule == "BALANCED":
                if split_ntok < 2:
                    return [], {"profile_attempts": 0, "low_level_variants": 0}
                cuts = (split_ntok // 2,)
            elif logical.split_rule.startswith("CUT"):
                cut = int(logical.split_rule.removeprefix("CUT"))
                if cut <= 0 or cut >= split_ntok or cut != min(cut, split_ntok - cut):
                    return [], {"profile_attempts": 0, "low_level_variants": 0}
                cuts = tuple(dict.fromkeys((cut, split_ntok - cut)))
            else:
                raise ValueError(f"unknown split rule {logical.split_rule!r}")
            work_assignments = tuple(
                (split_eid, cut, split_eid, split_ntok - cut) for cut in cuts
            )
        for eid_a, ntok_a, eid_b, ntok_b in work_assignments:
            if (
                reference._reserved_next_eid(state.c2) >= 0
                and reference._reserved_next_eid(state.c2) != eid_a
            ) or (
                reference._reserved_next_eid(state.c3) >= 0
                and reference._reserved_next_eid(state.c3) != eid_b
            ):
                continue
            s1_hit_a = reference._swiglu_hit_for_candidate(eid_a, state.c2, now)
            s1_hit_b = reference._swiglu_hit_for_candidate(eid_b, state.c3, now)
            s3_hit_a = reference._down_hit_for_candidate(eid_a, state.c2, now)
            s3_hit_b = reference._down_hit_for_candidate(eid_b, state.c3, now)
            if (
                s1_hit_a != physical.c2_s1_cached
                or s1_hit_b != physical.c3_s1_cached
            ):
                continue
            for dma_a_s3 in source_a_s3:
                for dma_b_s3 in source_b_s3:
                    profile_attempts += 1
                    raw_a = reference.FourStageSnap.from_assign(
                        now,
                        shape_a_s1,
                        shape_a_s3,
                        ntok_a,
                        eid_a,
                        s1_hit_a,
                        s3_hit_a,
                        dma_s1=dma_a_s1,
                        dma_s3=dma_a_s3,
                    )
                    raw_b = reference.FourStageSnap.from_assign(
                        now,
                        shape_b_s1,
                        shape_b_s3,
                        ntok_b,
                        eid_b,
                        s1_hit_b,
                        s3_hit_b,
                        dma_s1=dma_b_s1,
                        dma_s3=dma_b_s3,
                    )
                    snap_a = _apply_direct_s2pf(
                        raw_a, shape_a_s3, dma_a_s2pf
                    )
                    snap_b = _apply_direct_s2pf(
                        raw_b, shape_b_s3, dma_b_s2pf
                    )
                    low_level_variants += 1
                    if reference.bw_feasible(snap_a, snap_b):
                        action = _direct_pair_action(
                            eid_a=eid_a,
                            ntok_a=ntok_a,
                            shape_a_s1=shape_a_s1,
                            shape_a_s3=shape_a_s3,
                            start_a=now,
                            s1_hit_a=s1_hit_a,
                            snap_a=snap_a,
                            eid_b=eid_b,
                            ntok_b=ntok_b,
                            shape_b_s1=shape_b_s1,
                            shape_b_s3=shape_b_s3,
                            start_b=now,
                            s1_hit_b=s1_hit_b,
                            snap_b=snap_b,
                            tag=(
                                f"DIRECT-SPLIT(E{eid_a}:{ntok_a},{ntok_b})"
                                if logical.family == "SPLIT"
                                else f"DIRECT-PAIR({eid_a}+{eid_b})"
                            ),
                        )
                        if _explicit_physical_profile(action) == physical:
                            child = reference.apply_action(state, action)
                            actions.setdefault(_explicit_child_key(child), action)
    elif logical.family == "SINGLE":
        if len(selected_eids) != 1:
            return [], {"profile_attempts": 0, "low_level_variants": 0}
        eid = selected_eids[0]
        active_c2 = _explicit_shape(physical.c2_s1) is not None
        active_c3 = _explicit_shape(physical.c3_s1) is not None
        if active_c2 == active_c3:
            raise ValueError("SINGLE token must activate exactly one cluster")
        cluster_id = 2 if active_c2 else 3
        if not _direct_single_cluster_is_legal(state, cluster_id):
            return [], {"profile_attempts": 0, "low_level_variants": 0}
        cluster = state.c2 if cluster_id == 2 else state.c3
        peer = state.c3 if cluster_id == 2 else state.c2
        cluster_reserved = reference._reserved_next_eid(cluster)
        peer_reserved = reference._reserved_next_eid(peer)
        if (cluster_reserved >= 0 and cluster_reserved != eid) or peer_reserved == eid:
            return [], {"profile_attempts": 0, "low_level_variants": 0}
        shape_s1 = _explicit_shape(physical.c2_s1 if active_c2 else physical.c3_s1)
        shape_s3 = _explicit_shape(physical.c2_s3 if active_c2 else physical.c3_s3)
        if shape_s1 is None or shape_s3 is None:
            raise ValueError("SINGLE token has an inactive active-side shape")
        dma_s1 = _explicit_dma(
            physical.c2_dma_s1 if active_c2 else physical.c3_dma_s1
        )
        final_dma_s3 = _explicit_dma(
            physical.c2_dma_s3 if active_c2 else physical.c3_dma_s3
        )
        s2pf_dma = _explicit_dma(
            physical.c2_s2pf if active_c2 else physical.c3_s2pf
        )
        expected_s1_hit = (
            physical.c2_s1_cached if active_c2 else physical.c3_s1_cached
        )
        ntok = ntok_by_eid[eid]
        cluster_end = int(cluster.task_end)
        s1_hit = reference._swiglu_hit_for_candidate(eid, cluster, cluster_end)
        s3_hit = reference._down_hit_for_candidate(eid, cluster, cluster_end)
        if s1_hit != expected_s1_hit:
            return [], {"profile_attempts": 0, "low_level_variants": 0}
        discovery_bindings = (
            reference.DMA_BINDINGS
            if s2pf_dma != reference.DmaBinding.NONE
            else (final_dma_s3,)
        )
        starts = set()
        for discovery_dma_s3 in discovery_bindings:
            starts.update(
                reference._start_candidates(
                    cluster_end,
                    cluster,
                    peer,
                    ntok,
                    shape_s1,
                    shape_s3,
                    dma_s1,
                    discovery_dma_s3,
                    s1_hit,
                    s3_hit,
                )
            )
        source_dma_s3 = _source_s3_bindings(final_dma_s3, s2pf_dma)[0]
        for start in sorted(starts):
            profile_attempts += 1
            raw = reference.FourStageSnap.from_assign(
                start,
                shape_s1,
                shape_s3,
                ntok,
                eid,
                s1_hit,
                s3_hit,
                dma_s1=dma_s1,
                dma_s3=source_dma_s3,
            )
            snap = _apply_direct_s2pf(raw, shape_s3, s2pf_dma)
            low_level_variants += 1
            action = _direct_single_action(
                cluster=cluster_id,
                eid=eid,
                ntok=ntok,
                shape_s1=shape_s1,
                shape_s3=shape_s3,
                start=start,
                s1_hit=s1_hit,
                snap=snap,
            )
            if _explicit_physical_profile(action) != physical:
                continue
            feasible = (
                reference.bw_feasible(snap, peer)
                if cluster_id == 2
                else reference.bw_feasible(peer, snap)
            )
            if not feasible:
                continue
            child = reference.apply_action(state, action)
            actions.setdefault(_explicit_child_key(child), action)
    else:
        if len(selected_eids) != 1:
            return [], {"profile_attempts": 0, "low_level_variants": 0}
        eid = selected_eids[0]
        active_c2 = _explicit_shape(physical.c2_s1) is not None
        active_c3 = _explicit_shape(physical.c3_s1) is not None
        if active_c2 == active_c3:
            raise ValueError("PREFETCH token must name exactly one cluster")
        cluster_id = 2 if active_c2 else 3
        cluster = state.c2 if cluster_id == 2 else state.c3
        peer = state.c3 if cluster_id == 2 else state.c2
        pf_shape = _explicit_shape(physical.c2_s1 if active_c2 else physical.c3_s1)
        pf_dma = _explicit_dma(physical.s4pf_dma)
        if pf_shape is None or pf_dma == reference.DmaBinding.NONE:
            raise ValueError("PREFETCH token has no physical transfer")
        if cluster.cur_eid < 0 or cluster.pf_eid != -1:
            return [], {"profile_attempts": 0, "low_level_variants": 0}
        for pf_start in reference._next_s1_prefetch_start_candidates(
            cluster, pf_dma, (peer,)
        ):
            profile_attempts += 1
            if peer.cur_eid >= 0 and pf_start < peer.task_start:
                continue
            snap = cluster.with_prefetch(eid, pf_shape, pf_start, pf_dma)
            feasible = (
                reference.bw_feasible(snap, peer)
                if cluster_id == 2
                else reference.bw_feasible(peer, snap)
            )
            low_level_variants += 1
            if not feasible:
                continue
            if cluster_id == 2:
                action = reference.StageAction(
                    c2_eid=-2,
                    c2_ntok=0,
                    c2_shape_s1=pf_shape,
                    c2_shape_s3=None,
                    c2_start=pf_start,
                    c2_s1_cached=False,
                    c2_s3_cached=False,
                    c3_eid=-1,
                    c3_ntok=0,
                    c3_shape_s1=None,
                    c3_shape_s3=None,
                    c3_start=-1,
                    c3_s1_cached=False,
                    c3_s3_cached=False,
                    pf_cluster=2,
                    pf_eid=eid,
                    pf_shape=pf_shape,
                    pf_start=pf_start,
                    tag=f"DIRECT-PF-C2(E{eid},{pf_dma.name})",
                    pf_dma=pf_dma,
                )
            else:
                action = reference.StageAction(
                    c2_eid=-1,
                    c2_ntok=0,
                    c2_shape_s1=None,
                    c2_shape_s3=None,
                    c2_start=-1,
                    c2_s1_cached=False,
                    c2_s3_cached=False,
                    c3_eid=-2,
                    c3_ntok=0,
                    c3_shape_s1=pf_shape,
                    c3_shape_s3=None,
                    c3_start=pf_start,
                    c3_s1_cached=False,
                    c3_s3_cached=False,
                    pf_cluster=3,
                    pf_eid=eid,
                    pf_shape=pf_shape,
                    pf_start=pf_start,
                    tag=f"DIRECT-PF-C3(E{eid},{pf_dma.name})",
                    pf_dma=pf_dma,
                )
            if _explicit_physical_profile(action) != physical:
                continue
            child = reference.apply_action(state, action)
            actions.setdefault(_explicit_child_key(child), action)
    return list(actions.values()), {
        "profile_attempts": profile_attempts,
        "low_level_variants": low_level_variants,
    }


def _bounded_release_action_allowed(
    state: reference.BeamState,
    action: reference.StageAction,
) -> bool:
    """Restrict local starts to directly registered release timestamps."""
    if action.c2_eid >= 0 and action.c3_eid >= 0:
        return True
    if action.c2_eid >= 0:
        start = int(action.c2_start)
        cluster, peer = state.c2, state.c3
    elif action.c3_eid >= 0:
        start = int(action.c3_start)
        cluster, peer = state.c3, state.c2
    else:
        return False
    allowed_starts = {int(cluster.task_end), int(peer.task_end)}
    if peer.s2pf_end >= 0:
        allowed_starts.add(int(peer.s2pf_end))
    return start in allowed_starts


def _direct_explicit_candidate_map(
    state: reference.BeamState,
    tokens: tuple[ExplicitCandidateToken, ...],
    window: tuple[int, int] = EXPLICIT_WINDOW,
    start_policy: str = "all",
) -> tuple[
    dict[tuple, reference.StageAction],
    dict[tuple, set[int]],
    dict,
]:
    """Lower a ROM and retain every token that realizes each exact child."""
    emitted: dict[tuple, reference.StageAction] = {}
    source_indices: dict[tuple, set[int]] = {}
    profile_attempts = 0
    low_level_variants = 0
    valid_tokens = 0

    def local_start_key(action: reference.StageAction) -> tuple:
        child = reference.apply_action(state, action)
        starts = [
            start
            for eid, start in (
                (action.c2_eid, action.c2_start),
                (action.c3_eid, action.c3_start),
            )
            if eid >= 0
        ]
        if start_policy == "earliest_start":
            return (max(starts, default=0), sum(starts), child.g_score)
        if start_policy in {"earliest_finish", "bounded_release"}:
            ends = (int(child.c2.task_end), int(child.c3.task_end))
            return (max(ends), sum(ends), max(starts, default=0))
        if start_policy == "latest_start":
            return (-min(starts, default=0), -sum(starts), child.g_score)
        raise ValueError(f"unknown explicit start policy {start_policy!r}")

    for token_index, token in enumerate(tokens):
        actions, stats = _direct_materialize_explicit_token(state, token, window)
        profile_attempts += stats["profile_attempts"]
        low_level_variants += stats["low_level_variants"]
        valid_tokens += bool(actions)
        if start_policy == "bounded_release" and actions:
            actions = [
                action
                for action in actions
                if _bounded_release_action_allowed(state, action)
            ]
        if start_policy != "all" and actions:
            actions = [min(actions, key=local_start_key)]
        for action in actions:
            child = reference.apply_action(state, action)
            key = _explicit_child_key(child)
            emitted.setdefault(key, action)
            source_indices.setdefault(key, set()).add(token_index)
    return emitted, source_indices, {
        "valid_composite_tokens": valid_tokens,
        "profile_attempts": profile_attempts,
        "low_level_variants": low_level_variants,
        "concrete_candidates": len(emitted),
        "start_policy": start_policy,
    }


def generate_direct_explicit_candidates(
    state: reference.BeamState,
    tokens: tuple[ExplicitCandidateToken, ...],
    window: tuple[int, int] = EXPLICIT_WINDOW,
    start_policy: str = "all",
) -> tuple[list[reference.StageAction], dict]:
    """Emit candidates from a fixed token ROM through the bounded lowering."""
    emitted, _source_indices, stats = _direct_explicit_candidate_map(
        state, tokens, window, start_policy
    )
    return list(emitted.values()), stats


def generate_explicit_union_candidates(
    state: reference.BeamState,
    tokens: tuple[ExplicitCandidateToken, ...],
    window: tuple[int, int] = EXPLICIT_WINDOW,
    start_policy: str = "all",
) -> tuple[list[reference.StageAction], dict]:
    """Materialize a fixed certificate-union ROM without hidden scoring.

    A logical token is generated once.  Each matching physical profile and
    every matching legal event-aligned start becomes a separately counted
    concrete candidate.  The certified target is not an input.
    """
    profiles_by_logical: dict[
        ExplicitLogicalToken, set[ExplicitPhysicalProfile]
    ] = {}
    for token in tokens:
        if token.logical.mode == _explicit_mode(state):
            profiles_by_logical.setdefault(token.logical, set()).add(token.physical)

    emitted: dict[tuple, reference.StageAction] = {}
    by_composite: dict[
        ExplicitCandidateToken, dict[tuple, reference.StageAction]
    ] = {}
    raw_actions = 0
    expanded_by_token = Counter()
    visible = reference.candidate_window_visible_eids(
        state.c2, state.c3, state.remaining, window
    )
    for logical, profiles in profiles_by_logical.items():
        raw = _raw_actions_for_explicit_logical(state, logical, window)
        raw_actions += len(raw)
        for action in raw:
            physical = _explicit_physical_profile(action)
            if physical not in profiles:
                continue
            if not reference.action_within_candidate_window(action, visible):
                raise AssertionError("fixed explicit token emitted a hidden expert")
            child = reference.apply_action(state, action)
            key = _explicit_child_key(child)
            composite = ExplicitCandidateToken(logical, physical)
            by_composite.setdefault(composite, {}).setdefault(key, action)

    def local_start_key(action: reference.StageAction) -> tuple:
        child = reference.apply_action(state, action)
        starts = [
            start
            for eid, start in (
                (action.c2_eid, action.c2_start),
                (action.c3_eid, action.c3_start),
            )
            if eid >= 0
        ]
        if start_policy == "earliest_start":
            return (max(starts, default=0), sum(starts), child.g_score, repr(action))
        if start_policy == "earliest_finish":
            ends = (int(child.c2.task_end), int(child.c3.task_end))
            return (max(ends), sum(ends), max(starts, default=0), repr(action))
        if start_policy == "latest_start":
            return (-min(starts, default=0), -sum(starts), child.g_score, repr(action))
        raise ValueError(f"unknown explicit start policy {start_policy!r}")

    for composite, actions in by_composite.items():
        expanded_by_token[repr(composite)] = len(actions)
        selected_actions = list(actions.values())
        if start_policy != "all" and selected_actions:
            selected_actions = [min(selected_actions, key=local_start_key)]
        for action in selected_actions:
            child = reference.apply_action(state, action)
            emitted.setdefault(_explicit_child_key(child), action)
    return list(emitted.values()), {
        "valid_logical_tokens": len(profiles_by_logical),
        "raw_reference_actions": raw_actions,
        "concrete_candidates": len(emitted),
        "start_policy": start_policy,
        "max_starts_per_composite_token": max(expanded_by_token.values(), default=0),
    }


def _extract_explicit_certificate_union(
    proof_cases: list[dict],
    window_audit_path: Path,
    window: tuple[int, int] = EXPLICIT_WINDOW,
) -> tuple[tuple[ExplicitCandidateToken, ...], list[dict], Counter]:
    audit = json.loads(window_audit_path.read_text(encoding="utf-8"))
    if not audit.get("complete"):
        raise RuntimeError(f"incomplete window audit: {window_audit_path}")
    window_name = f"top{window[0]}+bottom{window[1]}"
    audit_rows = [
        row for row in audit["results"] if row.get("window") == window_name
    ]
    proof_by_name = {str(case["name"]): case for case in proof_cases}
    if len(audit_rows) != len(proof_cases):
        raise RuntimeError(
            f"{window_name}: expected {len(proof_cases)} direct witnesses, got {len(audit_rows)}"
        )
    if any(
        row.get("window_status") != "proved_sufficient_direct"
        for row in audit_rows
    ):
        raise RuntimeError(f"{window_name}: not every case has a direct witness")

    source_cache: dict[Path, dict[str, dict]] = {}
    tokens: set[ExplicitCandidateToken] = set()
    histories = []
    token_uses: Counter = Counter()
    for audit_row in audit_rows:
        name = str(audit_row["name"])
        source_row, actions, source = witness_templates._materialize_history(
            audit_row, window, source_cache
        )
        proof_case = proof_by_name[name]
        if [int(value) for value in source_row["counts"]] != [
            int(value) for value in proof_case["counts"]
        ]:
            raise RuntimeError(f"{name}: proof/witness counts mismatch")
        token_dist = {
            eid: int(ntok)
            for eid, ntok in enumerate(proof_case["counts"])
            if int(ntok) > 0
        }
        target = _target_cc(proof_case)
        replay = reference.validate_schedule_history(actions, token_dist)
        if replay != target:
            raise RuntimeError(f"{name}: witness replay {replay} != target {target}")
        state = reference.FourStageScheduler(token_dist)._initial_state()
        case_tokens = []
        for action in actions:
            token = _explicit_candidate_token(state, action, window)
            tokens.add(token)
            token_uses[token] += 1
            case_tokens.append(token)
            state = reference.apply_action(state, action)
        if state.remaining:
            raise RuntimeError(f"{name}: direct witness did not terminate")
        histories.append(
            {
                "name": name,
                "counts": proof_case["counts"],
                "target_cc": target,
                "actions": actions,
                "tokens": tuple(case_tokens),
                "source": source,
            }
        )
    return tuple(sorted(tokens)), histories, token_uses


def _serialize_explicit_token(
    token: ExplicitCandidateToken, uses: int
) -> dict:
    logical = token.logical
    physical = token.physical
    return {
        "mode": logical.mode,
        "family": logical.family,
        "selectors": list(logical.selectors),
        "split_rule": logical.split_rule,
        "physical": {
            "c2_s1": physical.c2_s1,
            "c2_s3": physical.c2_s3,
            "c3_s1": physical.c3_s1,
            "c3_s3": physical.c3_s3,
            "c2_dma_s1": physical.c2_dma_s1,
            "c2_dma_s3": physical.c2_dma_s3,
            "c2_s2pf": physical.c2_s2pf,
            "c3_dma_s1": physical.c3_dma_s1,
            "c3_dma_s3": physical.c3_dma_s3,
            "c3_s2pf": physical.c3_s2pf,
            "s4pf_dma": physical.s4pf_dma,
            "c2_s1_cached": physical.c2_s1_cached,
            "c2_s3_cached": physical.c2_s3_cached,
            "c3_s1_cached": physical.c3_s1_cached,
            "c3_s3_cached": physical.c3_s3_cached,
        },
        "witness_uses": int(uses),
    }


def _deserialize_explicit_token(row: dict) -> ExplicitCandidateToken:
    return ExplicitCandidateToken(
        logical=ExplicitLogicalToken(
            mode=str(row["mode"]),
            family=str(row["family"]),
            selectors=tuple(str(value) for value in row["selectors"]),
            split_rule=str(row.get("split_rule", "NONE")),
        ),
        physical=ExplicitPhysicalProfile(**row["physical"]),
    )


def load_explicit_token_bank(path: Path) -> tuple[ExplicitCandidateToken, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["tokens"] if isinstance(payload, dict) else payload
    tokens = tuple(_deserialize_explicit_token(row) for row in rows)
    if not tokens:
        raise ValueError(f"empty explicit token bank: {path}")
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"duplicate explicit tokens in {path}")
    return tokens


def derive_budgeted_direct_token_bank(
    source_path: Path,
    output_path: Path,
    *,
    candidate_budget: int,
    selection_policy: str,
) -> dict:
    """Derive a fixed per-decision-mode ROM; no runtime hidden truncation."""
    if candidate_budget not in (16, 24, 32):
        raise ValueError("candidate budget must be 16, 24, or 32")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_rows = list(source["tokens"])
    if len({_deserialize_explicit_token(row) for row in source_rows}) != len(
        source_rows
    ):
        raise ValueError("source token bank contains duplicates")

    quotas = {
        "SYNC": {
            16: {"PAIR": 8, "SPLIT": 4, "SINGLE": 1, "PREFETCH": 3},
            24: {"PAIR": 12, "SPLIT": 6, "SINGLE": 2, "PREFETCH": 4},
            32: {"PAIR": 16, "SPLIT": 8, "SINGLE": 2, "PREFETCH": 6},
        },
        "ONE_IDLE": {
            16: {"SINGLE": 10, "PAIR": 2, "PREFETCH": 4},
            24: {"SINGLE": 15, "PAIR": 3, "PREFETCH": 6},
            32: {"SINGLE": 20, "PAIR": 5, "PREFETCH": 7},
        },
    }

    def priority(row: dict) -> tuple:
        return (
            -int(row.get("witness_uses", 0)),
            str(row["family"]),
            tuple(str(value) for value in row["selectors"]),
            str(row.get("split_rule", "NONE")),
            json.dumps(row["physical"], sort_keys=True),
        )

    selected_rows = []
    mode_summary = {}
    for mode in ("SYNC", "ONE_IDLE", "TERMINAL"):
        rows = sorted(
            (row for row in source_rows if row["mode"] == mode),
            key=priority,
        )
        if mode == "TERMINAL" or len(rows) <= candidate_budget:
            selected = rows
        elif selection_policy in ("frequency", "frequency_hot_cold"):
            selected = rows[:candidate_budget]
            if selection_policy == "frequency_hot_cold" and mode == "SYNC":
                anchors = [
                    row
                    for row in rows
                    if row["family"] == "PAIR"
                    and set(row["selectors"]) == {"T0", "B0"}
                ]
                if anchors:
                    anchor = anchors[0]
                    anchor_key = json.dumps(anchor, sort_keys=True)
                    if all(
                        json.dumps(row, sort_keys=True) != anchor_key
                        for row in selected
                    ):
                        selected[-1] = anchor
        elif selection_policy == "family_quota":
            selected = []
            selected_ids = set()
            for family, quota in quotas[mode][candidate_budget].items():
                for row in (item for item in rows if item["family"] == family):
                    if sum(item["family"] == family for item in selected) >= quota:
                        break
                    key = json.dumps(row, sort_keys=True)
                    selected_ids.add(key)
                    selected.append(row)
            for row in rows:
                if len(selected) >= candidate_budget:
                    break
                key = json.dumps(row, sort_keys=True)
                if key not in selected_ids:
                    selected_ids.add(key)
                    selected.append(row)
        else:
            raise ValueError(f"unknown token selection policy {selection_policy!r}")
        if len(selected) > candidate_budget and mode != "TERMINAL":
            raise AssertionError("budgeted mode exceeded candidate budget")
        selected_rows.extend(selected)
        mode_summary[mode] = {
            "source_tokens": len(rows),
            "selected_tokens": len(selected),
            "families": dict(
                sorted(Counter(row["family"] for row in selected).items())
            ),
            "witness_uses": sum(
                int(row.get("witness_uses", 0)) for row in selected
            ),
        }
    payload = {
        "schema": "olmoe-budgeted-direct-token-bank-v1",
        "final_bank": False,
        "target_used_by_generator": False,
        "manifest": {
            "source_bank": str(source_path.resolve()),
            "source_bank_sha256": _sha256(source_path),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "candidate_budget_per_decision_mode": candidate_budget,
            "selection_policy": selection_policy,
            "start_policy_required_by_audit": "earliest_finish",
        },
        "interpretation": {
            "budget": (
                "SYNC and ONE_IDLE are mutually exclusive; each contains at "
                "most K fixed ROM tokens before legality skipping"
            ),
            "frequency": (
                "witness frequency defines deterministic offline ordering only; "
                "it does not prove a removed token unnecessary"
            ),
            "next_gate": "explicit-DMA candidate certification at the frozen optimum",
        },
        "mode_summary": mode_summary,
        "tokens": selected_rows,
    }
    _atomic_json(output_path, payload)
    return payload


def distill_used_direct_token_bank(
    proof_path: Path,
    window_audit_path: Path,
    candidate_certificate_path: Path,
    source_bank_path: Path,
    output_path: Path,
) -> dict:
    """Union the fixed tokens actually used by a certified 65-case path set."""
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof_by_name = {str(row["name"]): row for row in proof["cases"]}
    certificate = json.loads(
        candidate_certificate_path.read_text(encoding="utf-8")
    )
    if not certificate.get("complete") or certificate.get("summary", {}).get(
        "candidate_sufficient"
    ) != len(proof_by_name):
        raise ValueError("candidate certificate is not complete for the proof set")
    source_payload = json.loads(source_bank_path.read_text(encoding="utf-8"))
    source_rows = list(source_payload["tokens"])
    source_tokens = tuple(_deserialize_explicit_token(row) for row in source_rows)
    source_uses = [int(row.get("witness_uses", 0)) for row in source_rows]
    start_policy = str(
        certificate.get("manifest", {}).get("start_policy", "earliest_finish")
    )
    window_audit = json.loads(window_audit_path.read_text(encoding="utf-8"))
    audit_by_name = {
        str(row["name"]): row
        for row in window_audit["results"]
        if row.get("window") == "top8+bottom2"
    }
    source_cache: dict[Path, dict[str, dict]] = {}
    token_uses = Counter()
    path_rows = []
    for row in certificate["cases"]:
        name = str(row["name"])
        case = proof_by_name[name]
        if row["status"] == "saved_optimal_path_covered":
            _source, actions, _meta = witness_templates._materialize_history(
                audit_by_name[name], EXPLICIT_WINDOW, source_cache
            )
        elif row["status"] == "alternative_optimal_path_found":
            actions = tuple(deserialize_action(raw) for raw in row["actions"])
        else:
            raise ValueError(f"{name}: uncertified status {row['status']}")
        token_dist = {
            eid: int(ntok)
            for eid, ntok in enumerate(case["counts"])
            if int(ntok) > 0
        }
        replay = reference.validate_schedule_history(actions, token_dist)
        if replay != _target_cc(case):
            raise ValueError(f"{name}: selected path misses certified target")
        state = reference.FourStageScheduler(token_dist)._initial_state()
        for action in actions:
            _emitted, sources, _stats = _direct_explicit_candidate_map(
                state, source_tokens, start_policy=start_policy
            )
            child = reference.apply_action(state, action)
            covering = sorted(sources.get(_explicit_child_key(child), ()))
            if not covering:
                raise ValueError(
                    f"{name}: certified transition is absent from source bank"
                )
            token_index = min(
                covering, key=lambda index: (-source_uses[index], index)
            )
            token = source_tokens[token_index]
            token_uses[token] += 1
            state = child
        if state.remaining:
            raise ValueError(f"{name}: selected path is non-terminal")
        path_rows.append(
            {
                "name": name,
                "status": row["status"],
                "actions": len(actions),
                "replay_ticks": _ticks_text(replay),
            }
        )
    tokens = tuple(sorted(token_uses))
    by_mode = Counter(token.logical.mode for token in tokens)
    by_family = Counter(token.logical.family for token in tokens)
    payload = {
        "schema": "olmoe-used-direct-token-bank-v1",
        "final_bank": False,
        "target_used_by_generator": False,
        "manifest": {
            "proof": str(proof_path.resolve()),
            "proof_sha256": _sha256(proof_path),
            "window_audit": str(window_audit_path.resolve()),
            "window_audit_sha256": _sha256(window_audit_path),
            "candidate_certificate": str(candidate_certificate_path.resolve()),
            "candidate_certificate_sha256": _sha256(candidate_certificate_path),
            "source_bank": str(source_bank_path.resolve()),
            "source_bank_sha256": _sha256(source_bank_path),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "start_policy": start_policy,
        },
        "summary": {
            "cases": len(path_rows),
            "tokens": len(tokens),
            "tokens_by_mode": dict(sorted(by_mode.items())),
            "tokens_by_family": dict(sorted(by_family.items())),
            "max_tokens_in_any_mode": max(by_mode.values(), default=0),
            "path_actions": sum(row["actions"] for row in path_rows),
        },
        "interpretation": {
            "scope": (
                "union of tokens used by the 65 replayed optimum paths in the "
                "input candidate certificate"
            ),
            "next_gate": (
                "the smaller fixed bank must independently pass candidate "
                "certification; use frequency only for ordering, never proof"
            ),
        },
        "tokens": [
            _serialize_explicit_token(token, token_uses[token]) for token in tokens
        ],
        "paths": path_rows,
    }
    _atomic_json(output_path, payload)
    return payload


def _swap_dma_name(name: str) -> str:
    return {"IDMA": "XDMA", "XDMA": "IDMA"}.get(name, name)


def _physical_profile_variant(
    profile: ExplicitPhysicalProfile,
    *,
    swap_clusters: bool,
    swap_lanes: bool,
) -> tuple:
    values = {
        field: getattr(profile, field)
        for field in ExplicitPhysicalProfile.__dataclass_fields__
    }
    if swap_clusters:
        for suffix in (
            "s1",
            "s3",
            "dma_s1",
            "dma_s3",
            "s2pf",
            "s1_cached",
            "s3_cached",
        ):
            left, right = f"c2_{suffix}", f"c3_{suffix}"
            values[left], values[right] = values[right], values[left]
    if swap_lanes:
        for field in (
            "c2_dma_s1",
            "c2_dma_s3",
            "c2_s2pf",
            "c3_dma_s1",
            "c3_dma_s3",
            "c3_s2pf",
            "s4pf_dma",
        ):
            values[field] = _swap_dma_name(values[field])
    return tuple(
        values[field] for field in ExplicitPhysicalProfile.__dataclass_fields__
    )


def _canonical_physical_profile(profile: ExplicitPhysicalProfile) -> tuple:
    """Collapse only exact cluster and DMA-lane symmetries."""
    return min(
        _physical_profile_variant(
            profile,
            swap_clusters=swap_clusters,
            swap_lanes=swap_lanes,
        )
        for swap_clusters in (False, True)
        for swap_lanes in (False, True)
    )


def _policy_observation(
    state: reference.BeamState,
    *,
    include_remaining_summary: bool,
) -> tuple:
    visible = reference.candidate_window_remaining(
        state.c2, state.c3, state.remaining, EXPLICIT_WINDOW
    )
    local_state = reference._canonical_state_future_key(
        state.c2, state.c3, visible
    )
    if not include_remaining_summary:
        return local_state
    return (
        local_state,
        len(state.remaining),
        sum(int(ntok) for _eid, ntok in state.remaining),
    )


def _canonical_policy_choice(
    state: reference.BeamState, action: reference.StageAction
) -> tuple:
    token = _explicit_candidate_token(state, action)
    return (
        token.logical.family,
        token.logical.selectors,
        token.logical.split_rule,
        _canonical_physical_profile(token.physical),
    )


def audit_policy_observation_aliases(histories: list[dict]) -> dict:
    """Check whether saved optima demand conflicting hardware-visible choices."""
    variants = {}
    for include_summary in (False, True):
        choices: dict[tuple, list[tuple[str, int, tuple]]] = {}
        for history in histories:
            token_dist = {
                eid: int(ntok)
                for eid, ntok in enumerate(history["counts"])
                if int(ntok) > 0
            }
            state = reference.FourStageScheduler(token_dist)._initial_state()
            for action_index, action in enumerate(history["actions"]):
                observation = _policy_observation(
                    state,
                    include_remaining_summary=include_summary,
                )
                choices.setdefault(observation, []).append(
                    (
                        history["name"],
                        action_index,
                        _canonical_policy_choice(state, action),
                    )
                )
                state = reference.apply_action(state, action)
        conflicts = [
            rows
            for rows in choices.values()
            if len({row[2] for row in rows}) > 1
        ]
        key = "with_remaining_count_and_token_sum" if include_summary else "window_state_only"
        variants[key] = {
            "witness_states": sum(len(rows) for rows in choices.values()),
            "unique_observations": len(choices),
            "repeated_observations": sum(len(rows) > 1 for rows in choices.values()),
            "conflicting_observations": len(conflicts),
            "max_choices_per_conflict": max(
                (len({row[2] for row in rows}) for rows in conflicts),
                default=0,
            ),
            "examples": [
                [
                    {"name": name, "action_index": index}
                    for name, index, _choice in rows[:6]
                ]
                for rows in conflicts[:5]
            ],
        }
    return {
        "observation": (
            "canonical cluster/DMA state, top8+bottom2 plus resident entries; "
            "the augmented form adds two scalar counters"
        ),
        "choice_normalization": (
            "family/rank/split/physical profile after exact cluster and DMA-lane symmetry"
        ),
        **variants,
        "interpretation": (
            "Zero augmented conflicts proves that the saved certificate paths "
            "admit a deterministic policy on these 65 traces. It does not yet "
            "prove that a compact arithmetic scorer can reproduce that policy."
        ),
    }


def audit_explicit_certificate_union(
    proof_cases: list[dict],
    window_audit_path: Path,
    *,
    case_limit: int | None = None,
    workers: int = 1,
) -> dict:
    """Prove a fixed, target-independent candidate upper bound constructively."""
    tokens, histories, token_uses = _extract_explicit_certificate_union(
        proof_cases, window_audit_path
    )
    observation_aliases = audit_policy_observation_aliases(histories)
    selected_histories = histories[:case_limit] if case_limit else histories
    if workers <= 0:
        raise ValueError("workers must be positive")
    jobs = [(history, tokens) for history in selected_histories]
    if workers == 1:
        audited = [_audit_one_explicit_history(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            audited = list(pool.map(_audit_one_explicit_history, jobs))
    rows = [item[0] for item in audited]
    candidate_counts = [value for item in audited for value in item[1]]
    raw_counts = [value for item in audited for value in item[2]]
    valid_logical_counts = [value for item in audited for value in item[3]]
    starts_per_token = [value for item in audited for value in item[4]]

    by_family = Counter(token.logical.family for token in tokens)
    by_mode = Counter(token.logical.mode for token in tokens)
    logical_tokens = {token.logical for token in tokens}
    full_audit = len(selected_histories) == len(histories)
    covered_cases = sum(row["direct_path_covered"] for row in rows)
    return {
        "role": (
            "constructive fixed-token upper bound distilled from all 65 direct "
            "top8+bottom2 optimal certificates"
        ),
        "final_bank": False,
        "target_used_by_generator": False,
        "window": list(EXPLICIT_WINDOW),
        "source_cases": len(histories),
        "audited_cases": len(rows),
        "audit_complete": full_audit,
        "direct_optimal_paths_covered": covered_cases,
        "all_audited_paths_covered": covered_cases == len(rows),
        "logical_tokens": len(logical_tokens),
        "composite_tokens": len(tokens),
        "composite_tokens_by_family": dict(sorted(by_family.items())),
        "composite_tokens_by_mode": dict(sorted(by_mode.items())),
        "concrete_candidate_count_on_witness_states": {
            "max": max(candidate_counts, default=0),
            "mean": statistics.mean(candidate_counts) if candidate_counts else 0.0,
        },
        "valid_logical_tokens_on_witness_states": {
            "max": max(valid_logical_counts, default=0),
            "mean": statistics.mean(valid_logical_counts) if valid_logical_counts else 0.0,
        },
        "raw_reference_actions_on_witness_states": {
            "max": max(raw_counts, default=0),
            "mean": statistics.mean(raw_counts) if raw_counts else 0.0,
        },
        "max_event_aligned_starts_per_composite_token": max(
            starts_per_token, default=0
        ),
        "required_next_gate": (
            "delete and parameterize tokens while retaining 65/65 constructive "
            "coverage; only then evaluate the closed-loop scorer"
        ),
        "policy_observation_aliases": observation_aliases,
        "tokens": [
            _serialize_explicit_token(token, token_uses[token]) for token in tokens
        ],
        "cases": rows,
    }


def _audit_one_explicit_history(job: tuple[dict, tuple[ExplicitCandidateToken, ...]]):
    """Pickle-safe worker for one constructive witness-path audit."""
    history, tokens = job
    token_dist = {
        eid: int(ntok)
        for eid, ntok in enumerate(history["counts"])
        if int(ntok) > 0
    }
    state = reference.FourStageScheduler(token_dist)._initial_state()
    first_miss = None
    candidate_counts = []
    raw_counts = []
    valid_logical_counts = []
    starts_per_token = []
    for index, witness_action in enumerate(history["actions"]):
        candidates, stats = generate_explicit_union_candidates(state, tokens)
        candidate_counts.append(stats["concrete_candidates"])
        raw_counts.append(stats["raw_reference_actions"])
        valid_logical_counts.append(stats["valid_logical_tokens"])
        starts_per_token.append(stats["max_starts_per_composite_token"])
        witness_child = reference.apply_action(state, witness_action)
        target_key = _explicit_child_key(witness_child)
        if not any(
            _explicit_child_key(reference.apply_action(state, action)) == target_key
            for action in candidates
        ):
            first_miss = {
                "action_index": index,
                "token": _serialize_explicit_token(
                    _explicit_candidate_token(state, witness_action), 0
                ),
                "stats": stats,
            }
            break
        state = witness_child
    covered = first_miss is None and not state.remaining
    row = {
        "name": history["name"],
        "target_ticks": _ticks_text(history["target_cc"]),
        "actions": len(history["actions"]),
        "direct_path_covered": covered,
        "first_miss": first_miss,
    }
    return (
        row,
        candidate_counts,
        raw_counts,
        valid_logical_counts,
        starts_per_token,
    )


def _audit_one_direct_explicit_history(
    job: tuple[dict, tuple[ExplicitCandidateToken, ...], str],
) -> dict:
    """Replay one certificate using only the bounded direct token lowering."""
    history, tokens, start_policy = job
    token_dist = {
        eid: int(ntok)
        for eid, ntok in enumerate(history["counts"])
        if int(ntok) > 0
    }
    state = reference.FourStageScheduler(token_dist)._initial_state()
    candidate_counts = []
    profile_attempts = []
    first_miss = None
    for index, witness_action in enumerate(history["actions"]):
        candidates, stats = generate_direct_explicit_candidates(
            state, tokens, start_policy=start_policy
        )
        candidate_counts.append(stats["concrete_candidates"])
        profile_attempts.append(stats["profile_attempts"])
        witness_child = reference.apply_action(state, witness_action)
        target_key = _explicit_child_key(witness_child)
        if not any(
            _explicit_child_key(reference.apply_action(state, action)) == target_key
            for action in candidates
        ):
            first_miss = {
                "action_index": index,
                "token": _serialize_explicit_token(
                    _explicit_candidate_token(state, witness_action), 0
                ),
                "candidate_stats": stats,
            }
            break
        state = witness_child
    return {
        "name": history["name"],
        "target_ticks": _ticks_text(history["target_cc"]),
        "actions": len(history["actions"]),
        "direct_path_covered": first_miss is None and not state.remaining,
        "first_miss": first_miss,
        "candidate_count_max": max(candidate_counts, default=0),
        "profile_attempts_max": max(profile_attempts, default=0),
        "start_policy": start_policy,
    }


def audit_direct_explicit_certificate_union(
    histories: list[dict],
    tokens: tuple[ExplicitCandidateToken, ...],
    *,
    workers: int = 1,
    start_policy: str = "all",
) -> dict:
    jobs = [(history, tokens, start_policy) for history in histories]
    if workers == 1:
        rows = [_audit_one_direct_explicit_history(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_audit_one_direct_explicit_history, jobs))
    covered = sum(row["direct_path_covered"] for row in rows)
    return {
        "cases": len(rows),
        "direct_optimal_paths_covered": covered,
        "all_paths_covered": covered == len(rows),
        "candidate_count_max": max(
            (row["candidate_count_max"] for row in rows), default=0
        ),
        "profile_attempts_max": max(
            (row["profile_attempts_max"] for row in rows), default=0
        ),
        "start_policy": start_policy,
        "rows": rows,
    }


def _direct_certificate_coverage_case(
    job: tuple[dict, tuple[ExplicitCandidateToken, ...], str],
) -> list[dict]:
    history, tokens, start_policy = job
    token_dist = {
        eid: int(ntok)
        for eid, ntok in enumerate(history["counts"])
        if int(ntok) > 0
    }
    state = reference.FourStageScheduler(token_dist)._initial_state()
    rows = []
    for action_index, witness_action in enumerate(history["actions"]):
        _emitted, sources, stats = _direct_explicit_candidate_map(
            state, tokens, start_policy=start_policy
        )
        witness_child = reference.apply_action(state, witness_action)
        target_key = _explicit_child_key(witness_child)
        covering = sorted(sources.get(target_key, ()))
        if not covering:
            raise AssertionError(
                f"{history['name']} action {action_index}: no direct covering token"
            )
        rows.append(
            {
                "name": history["name"],
                "action_index": action_index,
                "covering_token_indices": covering,
                "all_candidate_count": stats["concrete_candidates"],
            }
        )
        state = witness_child
    return rows


def build_direct_certificate_coverage_matrix(
    histories: list[dict],
    tokens: tuple[ExplicitCandidateToken, ...],
    *,
    workers: int = 1,
    start_policy: str = "all",
) -> list[dict]:
    jobs = [(history, tokens, start_policy) for history in histories]
    if workers == 1:
        nested = [_direct_certificate_coverage_case(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            nested = list(pool.map(_direct_certificate_coverage_case, jobs))
    return [row for case_rows in nested for row in case_rows]


def greedy_direct_token_cover(
    coverage_rows: list[dict],
    token_count: int,
) -> tuple[list[int], dict]:
    """Deterministic set cover followed by exact redundancy deletion."""
    cover_by_token = [set() for _ in range(token_count)]
    for row_index, row in enumerate(coverage_rows):
        for token_index in row["covering_token_indices"]:
            cover_by_token[int(token_index)].add(row_index)
    forced = {
        int(row["covering_token_indices"][0])
        for row in coverage_rows
        if len(row["covering_token_indices"]) == 1
    }
    selected = list(sorted(forced))
    covered = set().union(*(cover_by_token[index] for index in selected)) if selected else set()
    universe = set(range(len(coverage_rows)))
    while covered != universe:
        uncovered = universe - covered
        best = max(
            (index for index in range(token_count) if index not in selected),
            key=lambda index: (len(cover_by_token[index] & uncovered), -index),
        )
        gain = cover_by_token[best] & uncovered
        if not gain:
            raise AssertionError("coverage matrix contains an uncovered transition")
        selected.append(best)
        covered.update(gain)

    # Greedy order is not sacred.  Delete every non-forced entry whose removal
    # still leaves every certificate transition covered, iterating to a fixed
    # point because one deletion may make a later token indispensable.
    changed = True
    while changed:
        changed = False
        for token_index in list(reversed(selected)):
            if token_index in forced:
                continue
            trial = [index for index in selected if index != token_index]
            trial_cover = set().union(
                *(cover_by_token[index] for index in trial)
            ) if trial else set()
            if trial_cover == universe:
                selected = trial
                changed = True
    selected.sort()
    multiplicity = Counter(
        len(row["covering_token_indices"]) for row in coverage_rows
    )
    return selected, {
        "transitions": len(coverage_rows),
        "tokens_in": token_count,
        "forced_tokens": len(forced),
        "tokens_out": len(selected),
        "covering_token_multiplicity": {
            str(key): value for key, value in sorted(multiplicity.items())
        },
    }


def build_compressed_direct_token_bank(
    proof_path: Path,
    window_audit_path: Path,
    output_path: Path,
    *,
    workers: int,
    case_limit: int | None,
    start_policy: str,
) -> dict:
    """Construct and re-audit a target-independent direct-lowering token ROM.

    The cover is over saved optimal certificate transitions.  It is therefore
    a constructive upper bound for those histories, not proof that a removed
    token is unnecessary in every alternative optimal path or unseen state.
    """
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof_cases = list(proof["cases"])
    tokens, histories, token_uses = _extract_explicit_certificate_union(
        proof_cases, window_audit_path
    )
    selected_histories = histories[:case_limit] if case_limit else histories
    coverage = build_direct_certificate_coverage_matrix(
        selected_histories,
        tokens,
        workers=workers,
        start_policy=start_policy,
    )
    selected_indices, cover_stats = greedy_direct_token_cover(
        coverage, len(tokens)
    )
    selected_tokens = tuple(tokens[index] for index in selected_indices)
    audit = audit_direct_explicit_certificate_union(
        selected_histories,
        selected_tokens,
        workers=workers,
        start_policy=start_policy,
    )
    if not audit["all_paths_covered"]:
        raise AssertionError("compressed direct bank lost a covered certificate path")
    by_mode = Counter(token.logical.mode for token in selected_tokens)
    by_family = Counter(token.logical.family for token in selected_tokens)
    payload = {
        "schema": "olmoe-compressed-direct-token-bank-v1",
        "complete": len(selected_histories) == len(histories),
        "final_bank": False,
        "target_used_by_generator": False,
        "interpretation": {
            "covered": (
                "the fixed ROM and direct lowering contain every transition of "
                "the audited saved globally optimal histories"
            ),
            "not_proved": (
                "set-cover deletion does not prove removed tokens unnecessary "
                "for alternative optimal paths or unseen distributions"
            ),
            "candidate_budget": (
                "candidate_count_max is the actual per-state concrete count; "
                "the number of ROM tokens is not reported as K"
            ),
        },
        "manifest": {
            "proof": str(proof_path.resolve()),
            "proof_sha256": _sha256(proof_path),
            "window_audit": str(window_audit_path.resolve()),
            "window_audit_sha256": _sha256(window_audit_path),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "window": list(EXPLICIT_WINDOW),
            "start_policy": start_policy,
            "source_cases": len(histories),
            "audited_cases": len(selected_histories),
        },
        "compression": {
            **cover_stats,
            "selected_token_indices": selected_indices,
            "selected_tokens_by_mode": dict(sorted(by_mode.items())),
            "selected_tokens_by_family": dict(sorted(by_family.items())),
        },
        "audit": audit,
        "tokens": [
            _serialize_explicit_token(token, token_uses[token])
            for token in selected_tokens
        ],
    }
    _atomic_json(output_path, payload)
    return payload


def _direct_target_rank(state: reference.BeamState) -> tuple:
    return (
        len(state.remaining),
        reference.completion_estimate(state),
        abs(int(state.c2.task_end) - int(state.c3.task_end)),
        int(state.g_score),
    )


def run_direct_token_target_search(
    counts: list[int],
    target_cc: int,
    tokens: tuple[ExplicitCandidateToken, ...],
    *,
    start_policy: str,
    time_limit_s: float,
    max_expansions: int,
    initial_state: reference.BeamState | None = None,
) -> dict:
    """Search the exact graph induced by one fixed direct token ROM.

    The certified target appears only in admissible pruning.  It is never an
    input to token materialization, candidate ordering, or local start choice.
    """
    token_dist = {
        eid: int(ntok) for eid, ntok in enumerate(counts) if int(ntok) > 0
    }
    initial = (
        initial_state
        if initial_state is not None
        else reference.FourStageScheduler(token_dist)._initial_state()
    )
    target_capacity = 2 * int(target_cc)
    started = time.perf_counter()

    def within_target(state: reference.BeamState) -> bool:
        return (
            state.f_score <= target_cc
            and state.c2.task_end
            + state.c3.task_end
            + reference._minimum_cluster_work(state.remaining)
            <= target_capacity
        )

    if not within_target(initial):
        return {
            "feasible": False,
            "exhaustive": True,
            "termination": "root_bound",
            "expansions": 0,
            "generated": 0,
            "open_states": 0,
            "peak_open_states": 0,
            "candidate_count_max": 0,
            "runtime_s": time.perf_counter() - started,
            "actions": [],
        }

    rank_heap = []
    active_entries: set[int] = set()
    open_by_fingerprint: dict[tuple, tuple[int, int]] = {}
    closed_best_work: dict[tuple, int] = {}
    next_entry_id = 0
    peak_open = 0
    expansions = 0
    generated = 0
    candidate_count_max = 0

    def push(state: reference.BeamState) -> bool:
        nonlocal next_entry_id, peak_open
        if not within_target(state):
            return False
        fingerprint = state.fingerprint()
        closed_work = closed_best_work.get(fingerprint)
        if closed_work is not None and closed_work <= state.cluster_work_cc:
            return False
        previous = open_by_fingerprint.get(fingerprint)
        if previous is not None and previous[0] <= state.cluster_work_cc:
            return False
        if previous is not None:
            active_entries.discard(previous[1])
        entry_id = next_entry_id
        next_entry_id += 1
        open_by_fingerprint[fingerprint] = (state.cluster_work_cc, entry_id)
        active_entries.add(entry_id)
        heapq.heappush(
            rank_heap,
            (_direct_target_rank(state), entry_id, state),
        )
        peak_open = max(peak_open, len(active_entries))
        return True

    push(initial)
    termination = "open_exhausted"
    while rank_heap:
        while rank_heap and rank_heap[0][1] not in active_entries:
            heapq.heappop(rank_heap)
        if not rank_heap:
            break
        if time.perf_counter() - started >= time_limit_s:
            termination = "time_limit"
            break
        if expansions >= max_expansions:
            termination = "expansion_limit"
            break
        _rank, entry_id, state = heapq.heappop(rank_heap)
        active_entries.discard(entry_id)
        fingerprint = state.fingerprint()
        current = open_by_fingerprint.get(fingerprint)
        if current is not None and current[1] == entry_id:
            del open_by_fingerprint[fingerprint]
        closed_work = closed_best_work.get(fingerprint)
        if closed_work is not None and closed_work <= state.cluster_work_cc:
            continue
        closed_best_work[fingerprint] = state.cluster_work_cc
        expansions += 1
        actions, stats = generate_direct_explicit_candidates(
            state, tokens, start_policy=start_policy
        )
        generated += len(actions)
        candidate_count_max = max(candidate_count_max, len(actions))
        for action in actions:
            child = reference.apply_action(state, action)
            if not within_target(child):
                continue
            if not child.remaining:
                replay = reference.validate_schedule_history(
                    child.history, token_dist
                )
                if replay != child.g_score or replay > target_cc:
                    raise AssertionError("direct-token target witness failed replay")
                return {
                    "feasible": True,
                    "exhaustive": False,
                    "termination": "feasible",
                    "expansions": expansions,
                    "generated": generated,
                    "open_states": len(active_entries),
                    "peak_open_states": peak_open,
                    "candidate_count_max": candidate_count_max,
                    "runtime_s": time.perf_counter() - started,
                    "actions": [serialize_action(item) for item in child.history],
                }
            push(child)
    exhaustive = not active_entries
    return {
        "feasible": False,
        "exhaustive": exhaustive,
        "termination": termination,
        "expansions": expansions,
        "generated": generated,
        "open_states": len(active_entries),
        "peak_open_states": peak_open,
        "candidate_count_max": candidate_count_max,
        "runtime_s": time.perf_counter() - started,
        "actions": [],
    }


def _direct_target_search_job(job: tuple) -> dict:
    case, tokens, start_policy, time_limit_s, max_expansions = job
    result = run_direct_token_target_search(
        case["counts"],
        _target_cc(case),
        tokens,
        start_policy=start_policy,
        time_limit_s=time_limit_s,
        max_expansions=max_expansions,
    )
    return {
        "name": case["name"],
        "optimal_ticks": _ticks_text(_target_cc(case)),
        **result,
    }


def certify_direct_token_bank(
    proof_path: Path,
    window_audit_path: Path,
    bank_path: Path,
    output_path: Path,
    *,
    workers: int,
    start_policy: str,
    time_limit_s: float,
    max_expansions: int,
) -> dict:
    """Certify saved or alternative optimum paths in one fixed token graph."""
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof_cases = list(proof["cases"])
    tokens = load_explicit_token_bank(bank_path)
    _all_tokens, histories, _uses = _extract_explicit_certificate_union(
        proof_cases, window_audit_path
    )
    direct = audit_direct_explicit_certificate_union(
        histories,
        tokens,
        workers=workers,
        start_policy=start_policy,
    )
    direct_by_name = {row["name"]: row for row in direct["rows"]}
    missing_cases = [
        case
        for case in proof_cases
        if not direct_by_name[case["name"]]["direct_path_covered"]
    ]
    jobs = [
        (case, tokens, start_policy, time_limit_s, max_expansions)
        for case in missing_cases
    ]
    if workers == 1:
        alternatives = [_direct_target_search_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            alternatives = list(pool.map(_direct_target_search_job, jobs))
    alternative_by_name = {row["name"]: row for row in alternatives}
    rows = []
    for case in proof_cases:
        name = case["name"]
        direct_row = direct_by_name[name]
        if direct_row["direct_path_covered"]:
            rows.append(
                {
                    "name": name,
                    "status": "saved_optimal_path_covered",
                    "optimal_ticks": str(case["best_reference_ticks"]),
                    "candidate_count_max": direct_row["candidate_count_max"],
                    "saved_path_action_count": direct_row["actions"],
                }
            )
            continue
        alternative = alternative_by_name[name]
        status = (
            "alternative_optimal_path_found"
            if alternative["feasible"]
            else "candidate_insufficient"
            if alternative["exhaustive"]
            else "unresolved"
        )
        rows.append(
            {
                "name": name,
                "status": status,
                "saved_path_first_miss": direct_row["first_miss"],
                **alternative,
            }
        )
    status_counts = Counter(row["status"] for row in rows)
    sufficient = (
        status_counts["saved_optimal_path_covered"]
        + status_counts["alternative_optimal_path_found"]
    )
    payload = {
        "schema": "olmoe-direct-token-candidate-certification-v1",
        "complete": sufficient == len(rows),
        "interpretation": {
            "saved_optimal_path_covered": (
                "the fixed candidate graph directly contains the audited saved optimum"
            ),
            "alternative_optimal_path_found": (
                "target-feasibility search found and replayed another optimum in the same fixed graph"
            ),
            "candidate_insufficient": (
                "the fixed candidate graph was exhaustively searched at the certified target"
            ),
            "unresolved": "the graph search stopped with live OPEN states",
            "target_used_by_generator": False,
            "scorer_evaluated": False,
        },
        "manifest": {
            "proof": str(proof_path.resolve()),
            "proof_sha256": _sha256(proof_path),
            "window_audit": str(window_audit_path.resolve()),
            "window_audit_sha256": _sha256(window_audit_path),
            "token_bank": str(bank_path.resolve()),
            "token_bank_sha256": _sha256(bank_path),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "start_policy": start_policy,
            "time_limit_s_per_fallback_case": time_limit_s,
            "max_expansions_per_fallback_case": max_expansions,
        },
        "summary": {
            "cases": len(rows),
            "candidate_sufficient": sufficient,
            "saved_optimal_path_covered": status_counts[
                "saved_optimal_path_covered"
            ],
            "alternative_optimal_path_found": status_counts[
                "alternative_optimal_path_found"
            ],
            "candidate_insufficient": status_counts["candidate_insufficient"],
            "unresolved": status_counts["unresolved"],
            "token_rom_entries": len(tokens),
            "candidate_count_max": max(
                (row.get("candidate_count_max", 0) for row in rows), default=0
            ),
        },
        "cases": rows,
    }
    _atomic_json(output_path, payload)
    return payload


def generate_practical_probe_candidates(
    state: reference.BeamState,
    tokens: tuple[ExplicitCandidateToken, ...],
    start_policy: str,
    safety_policy: str,
    direct_generator: bool = False,
    strict_token_bank: bool = False,
    window: tuple[int, int] = EXPLICIT_WINDOW,
) -> tuple[list[reference.StageAction], dict]:
    """Closed-loop probe bank: consuming certificate tokens plus safe progress.

    Standalone prefetch is intentionally removed.  The fallback is a fixed
    seed-profile SINGLE menu for T0 and any concrete resident reservation; it
    guarantees progress after a wrong greedy decision reaches a state absent
    from the optimal certificates.  Every emitted fallback realization is a
    separately counted candidate.
    """
    union_actions, union_stats = (
        generate_direct_explicit_candidates(
            state, tokens, window=window, start_policy=start_policy
        )
        if direct_generator
        else generate_explicit_union_candidates(
            state, tokens, window=window, start_policy=start_policy
        )
    )
    if strict_token_bank:
        emitted = {
            _explicit_child_key(reference.apply_action(state, action)): action
            for action in union_actions
        }
        if not emitted:
            raise RuntimeError("strict token bank has no progress candidate")
        return list(emitted.values()), {
            "concrete_candidates": len(emitted),
            "certificate_union_candidates": len(emitted),
            "safety_single_candidates": 0,
            "union_raw_reference_actions": union_stats.get(
                "raw_reference_actions", 0
            ),
            "union_profile_attempts": union_stats.get("profile_attempts", 0),
            "union_low_level_variants": union_stats.get(
                "low_level_variants", 0
            ),
            "direct_generator": direct_generator,
            "strict_token_bank": True,
            "source_by_child": {key: "token_bank" for key in emitted},
            "safety_policy": "disabled",
        }
    union_actions = [action for action in union_actions if action.pf_eid < 0]
    selected_eids = {int(state.remaining[0][0])}
    selected_eids.update(
        int(snap.pf_eid)
        for snap in (state.c2, state.c3)
        if snap.pf_eid >= 0
        and any(eid == snap.pf_eid for eid, _ntok in state.remaining)
    )
    subset = tuple(
        item for item in state.remaining if int(item[0]) in selected_eids
    )
    fallback_raw = reference.gen_stage_actions(
        state.c2,
        state.c3,
        subset,
        seed_mode=True,
        seed_all_visible=True,
    )
    fallback = [
        action
        for action in fallback_raw
        if _explicit_family(action) == "SINGLE"
        and (
            action.c2_eid in selected_eids
            or action.c3_eid in selected_eids
        )
    ]
    if safety_policy != "all":
        by_eid: dict[int, list[reference.StageAction]] = {}
        for action in fallback:
            eid = action.c2_eid if action.c2_eid >= 0 else action.c3_eid
            by_eid.setdefault(int(eid), []).append(action)

        def safety_local_key(action: reference.StageAction) -> tuple:
            child = reference.apply_action(state, action)
            starts = [
                start
                for eid, start in (
                    (action.c2_eid, action.c2_start),
                    (action.c3_eid, action.c3_start),
                )
                if eid >= 0
            ]
            if safety_policy in ("earliest_finish_per_eid", "earliest_finish_global"):
                ends = (int(child.c2.task_end), int(child.c3.task_end))
                return (max(ends), sum(ends), max(starts, default=0), repr(action))
            if safety_policy == "earliest_start_per_eid":
                return (max(starts, default=0), sum(starts), child.g_score, repr(action))
            raise ValueError(f"unknown safety policy {safety_policy!r}")

        fallback = [min(actions, key=safety_local_key) for actions in by_eid.values()]
        if safety_policy == "earliest_finish_global" and fallback:
            fallback = [min(fallback, key=safety_local_key)]
    emitted: dict[tuple, tuple[reference.StageAction, str]] = {}
    for action in union_actions:
        emitted.setdefault(
            _explicit_child_key(reference.apply_action(state, action)),
            (action, "certificate_union"),
        )
    for action in fallback:
        emitted.setdefault(
            _explicit_child_key(reference.apply_action(state, action)),
            (action, "safety_single"),
        )
    if not emitted:
        raise RuntimeError("practical probe bank has no progress candidate")
    return [item[0] for item in emitted.values()], {
        "concrete_candidates": len(emitted),
        "certificate_union_candidates": sum(
            source == "certificate_union" for _action, source in emitted.values()
        ),
        "safety_single_candidates": sum(
            source == "safety_single" for _action, source in emitted.values()
        ),
        "union_raw_reference_actions": union_stats.get("raw_reference_actions", 0),
        "union_profile_attempts": union_stats.get("profile_attempts", 0),
        "union_low_level_variants": union_stats.get("low_level_variants", 0),
        "direct_generator": direct_generator,
        "source_by_child": {
            key: source for key, (_action, source) in emitted.items()
        },
        "safety_policy": safety_policy,
    }


def _selected_action_features(
    action: reference.StageAction,
) -> tuple[int, int, int, int]:
    selected_tokens = [
        int(ntok)
        for eid, ntok in (
            (action.c2_eid, action.c2_ntok),
            (action.c3_eid, action.c3_ntok),
        )
        if eid >= 0
    ]
    s2pf = sum(
        binding != reference.DmaBinding.NONE
        for binding in (action.c2_s2pf_dma, action.c3_s2pf_dma)
    )
    return (
        max(selected_tokens, default=0),
        min(selected_tokens, default=0),
        sum(selected_tokens),
        s2pf,
    )


@dataclass(frozen=True)
class BoundedRemainingCounters:
    """Scalar runtime state initialized once and decremented after selection."""

    count: int
    token_sum: int
    odd_count: int
    le2_count: int
    block_sum: int
    best_work_cc: int
    small_block_hist: tuple[int, int, int, int]


def _bounded_remaining_counters(
    remaining: tuple[tuple[int, int], ...],
) -> BoundedRemainingCounters:
    """Mirror maintained counters without exposing tail descriptors to scoring.

    The Python model recomputes these scalars as a consistency oracle.  RTL/SW
    initializes them with the distribution and then subtracts the selected
    expert contribution; no tail descriptor scan is required at runtime.
    """
    loads = [int(ntok) for _eid, ntok in remaining]
    block_counts = [
        (ntok + reference.FULL_M_DIM - 1) // reference.FULL_M_DIM
        for ntok in loads
    ]
    return BoundedRemainingCounters(
        count=len(loads),
        token_sum=sum(loads),
        odd_count=sum(ntok & 1 for ntok in loads),
        le2_count=sum(ntok <= 2 for ntok in loads),
        block_sum=sum(block_counts),
        best_work_cc=sum(reference._best_task_time(ntok) for ntok in loads),
        small_block_hist=tuple(
            sum(blocks == bucket for blocks in block_counts)
            for bucket in range(1, 5)
        ),
    )


def _bounded_compute_lb(
    c2_end: int,
    c3_end: int,
    total_blocks: int,
) -> int:
    if total_blocks <= 0:
        return max(c2_end, c3_end)
    phase_block = reference.SHAPE_C.T_s1 + reference.SHAPE_C.T_s3
    crossing_num = c3_end - c2_end + total_blocks * phase_block
    crossing_den = 2 * phase_block
    crossing_floor = crossing_num // crossing_den
    crossing_ceil = -(-crossing_num // crossing_den)
    candidates = {
        0,
        total_blocks,
        max(0, min(total_blocks, crossing_floor)),
        max(0, min(total_blocks, crossing_ceil)),
    }
    return min(
        max(
            c2_end + c2_blocks * phase_block,
            c3_end + (total_blocks - c2_blocks) * phase_block,
        )
        for c2_blocks in candidates
    )


def bounded_state_lower_bound_components(
    state: reference.BeamState,
) -> dict[str, int]:
    """Lower-bound fields from the hottest descriptor and scalar counters.

    The longest per-expert terms are monotone in load, so the first ranked
    descriptor is sufficient.  Remaining DMA bytes use the reference model's
    relaxed cache-slot accounting.  This is identical to the original
    zero-cache expression when no initial/S4-prefetch residency exists, while
    also making the bounded scorer well-defined for the adaptive-prefetch
    mirror.
    """
    counters = _bounded_remaining_counters(state.remaining)
    earliest = min(int(state.c2.task_end), int(state.c3.task_end))
    latest = max(int(state.c2.task_end), int(state.c3.task_end))
    compute_lb = _bounded_compute_lb(
        int(state.c2.task_end), int(state.c3.task_end), counters.block_sum
    )
    if state.remaining:
        top_only = state.remaining[:1]
        release_chain_lb = reference._release_aware_expert_chain_lb(
            int(state.c2.task_end), int(state.c3.task_end), top_only
        )
        critical_chain_lb = earliest + reference._critical_expert_chain_lb(
            state.c2, state.c3, top_only
        )
    else:
        release_chain_lb = latest
        critical_chain_lb = earliest
    mandatory_dma_bytes = reference._minimum_remaining_dma_bytes(
        state.c2,
        state.c3,
        state.remaining,
    )
    dma_release = reference._earliest_relaxed_dma_release(state.c2, state.c3)
    dma_capacity_lb = max(
        latest,
        reference._dma_capacity_finish_lb(
            state.c2, state.c3, dma_release, mandatory_dma_bytes
        ),
    )
    return {
        "committed_cc": latest,
        "compute_cc": compute_lb,
        "release_expert_chain_cc": release_chain_lb,
        "critical_chain_cc": critical_chain_lb,
        "mandatory_dma_bytes": mandatory_dma_bytes,
        "dma_release_cc": dma_release,
        "dma_capacity_cc": dma_capacity_lb,
        "combined_cc": max(
            latest,
            compute_lb,
            release_chain_lb,
            critical_chain_lb,
            dma_capacity_lb,
        ),
    }


def bounded_head_lpt_estimate(state: reference.BeamState, head: int) -> int:
    """RTL-oriented LPT: exact head jobs plus one aggregate tail-work scalar.

    The Python mirror derives ``tail_work`` from the full distribution only to
    emulate a counter.  Hardware needs no hidden descriptor list: software
    supplies the initial sum of ``best_task(ntok)`` once, and the scheduler
    subtracts the selected expert's contribution after each consuming action.
    """
    if head <= 0:
        raise ValueError("head must be positive")
    loads = [int(state.c2.task_end), int(state.c3.task_end)]
    for _eid, ntok in state.remaining[:head]:
        target = 0 if loads[0] <= loads[1] else 1
        loads[target] += reference._best_task_time(int(ntok))
    counters = _bounded_remaining_counters(state.remaining)
    head_work = sum(
        reference._best_task_time(int(ntok))
        for _eid, ntok in state.remaining[:head]
    )
    tail_work = counters.best_work_cc - head_work
    low = 0 if loads[0] <= loads[1] else 1
    high = 1 - low
    fill = min(loads[high] - loads[low], tail_work)
    loads[low] += fill
    tail_work -= fill
    if tail_work:
        loads[low] += tail_work // 2
        loads[high] += tail_work - tail_work // 2
    return max(loads)


def bounded_head_hist_lpt_estimate(
    state: reference.BeamState,
    *,
    head: int = 5,
    tail_max_blocks: int = 4,
) -> int:
    """Bounded LPT from ``head`` descriptors and four cold-tail counters.

    The histogram counters cover every remaining expert whose isolated task
    occupies one through four M blocks.  The visible head descriptors are
    subtracted before those bins are scheduled in descending order.  Any
    remaining work above four blocks is represented by one maintained
    aggregate-work scalar and balanced after the exact histogram.  Hardware
    therefore needs no hidden descriptor and the scorer remains defined for
    distributions outside the original OLMoE proof envelope.
    """
    if head <= 0:
        raise ValueError("head must be positive")
    if tail_max_blocks != 4:
        raise ValueError("the maintained runtime histogram has four bins")
    head_entries = state.remaining[:head]
    counters = _bounded_remaining_counters(state.remaining)
    tail_hist = list(counters.small_block_hist)
    for _eid, ntok in head_entries:
        blocks = (
            int(ntok) + reference.FULL_M_DIM - 1
        ) // reference.FULL_M_DIM
        if blocks <= tail_max_blocks:
            tail_hist[blocks - 1] -= 1
    if min(tail_hist, default=0) < 0:
        raise AssertionError("negative cold-tail histogram count")
    tail_best_work = counters.best_work_cc - sum(
        reference._best_task_time(int(ntok)) for _eid, ntok in head_entries
    )

    loads = [int(state.c2.task_end), int(state.c3.task_end)]
    for _eid, ntok in head_entries:
        target = 0 if loads[0] <= loads[1] else 1
        loads[target] += reference._best_task_time(int(ntok))
    phase_block_cc = reference._best_task_time(reference.FULL_M_DIM)
    histogram_work = sum(
        count * blocks * phase_block_cc
        for blocks, count in enumerate(tail_hist, start=1)
    )
    overflow_work = tail_best_work - histogram_work
    if overflow_work < 0:
        raise AssertionError("negative aggregate tail work")
    for blocks in range(tail_max_blocks, 0, -1):
        for _ in range(tail_hist[blocks - 1]):
            target = 0 if loads[0] <= loads[1] else 1
            loads[target] += blocks * phase_block_cc
    if overflow_work:
        low = 0 if loads[0] <= loads[1] else 1
        high = 1 - low
        fill = min(loads[high] - loads[low], overflow_work)
        loads[low] += fill
        overflow_work -= fill
        loads[low] += overflow_work // 2
        loads[high] += overflow_work - overflow_work // 2
    return max(int(state.f_score), max(loads))


def practical_probe_score(
    before: reference.BeamState,
    action: reference.StageAction,
    child: reference.BeamState,
    *,
    scorer: str,
    sync_tiebreak: str,
) -> tuple:
    """Evidence-led closed-loop scorer; full-tail LPT is a temporary probe.

    SYNC first minimizes the original full-distribution LPT estimate.  Equal
    scores prefer a consuming pair, preserve one hottest job in the decision,
    use the smaller partner, then prefer the smaller committed makespan.  This
    reproduces hot+cold launch followed by plateau-parity pairings.

    ONE_IDLE minimizes the later and then earlier cluster release.  The second
    term is essential: if several jobs fit under the busy cluster's slack, it
    chooses the short cold job so the idle cluster can issue again.

    The full-tail LPT input is not yet the final hardware scorer.  This probe
    isolates control/tie-break correctness before replacing the unseen tail by
    bounded-window plus aggregate counters.
    """
    max_ntok, min_ntok, sum_ntok, s2pf = _selected_action_features(action)
    mode = _explicit_mode(before)

    def sync_tail() -> tuple:
        if sync_tiebreak == "hot_cold":
            return (-max_ntok, sum_ntok, child.g_score, -s2pf)
        if sync_tiebreak == "small_pair":
            return (sum_ntok, child.g_score, -s2pf, -max_ntok)
        if sync_tiebreak == "earliest_commit":
            return (child.g_score, -s2pf, sum_ntok, -max_ntok)
        if sync_tiebreak == "equal_pair":
            return (
                max_ntok - min_ntok,
                sum_ntok,
                child.g_score,
                -s2pf,
                -max_ntok,
            )
        raise ValueError(f"unknown sync tiebreak {sync_tiebreak!r}")

    if scorer.startswith("lb_"):
        components = (
            bounded_state_lower_bound_components(child)
            if scorer
            in {
                "lb_f_head8_compute_dma",
                "lb_f_head5_hist4_compute_dma",
            }
            else reference.state_lower_bound_components(
                child.c2, child.c3, child.remaining
            )
        )
        early, late = sorted((int(child.c2.task_end), int(child.c3.task_end)))
        lpt = reference._lpt_completion_estimate(child)
        common_by_scorer = {
            "lb_f_lpt": (child.f_score, lpt),
            "lb_f_lpt_compute": (
                child.f_score,
                lpt,
                components["compute_cc"],
            ),
            "lb_f_lpt_compute_late": (
                child.f_score,
                lpt,
                components["compute_cc"],
                late,
            ),
            "lb_f_lpt_compute_early": (
                child.f_score,
                lpt,
                components["compute_cc"],
                early,
            ),
            "lb_f_lpt_compute_imbalance": (
                child.f_score,
                lpt,
                components["compute_cc"],
                late - early,
            ),
            "lb_f_lpt_compute_dma": (
                child.f_score,
                lpt,
                components["compute_cc"],
                components["dma_capacity_cc"],
            ),
            "lb_f_head8_compute_dma": (
                child.f_score,
                max(child.f_score, bounded_head_lpt_estimate(child, 8)),
                components["compute_cc"],
                components["dma_capacity_cc"],
            ),
            "lb_f_head5_hist4_compute_dma": (
                child.f_score,
                bounded_head_hist_lpt_estimate(child),
                components["compute_cc"],
                components["dma_capacity_cc"],
            ),
            "lb_f_lpt_compute_release": (
                child.f_score,
                lpt,
                components["compute_cc"],
                components["release_expert_chain_cc"],
            ),
        }
        if scorer == "lb_certified_lex":
            if mode == "SYNC":
                common = (
                    child.f_score,
                    lpt,
                    components["compute_cc"],
                    late - early,
                    child.cluster_work_cc,
                    -max_ntok,
                )
            else:
                common = (
                    child.f_score,
                    lpt,
                    components["compute_cc"],
                    components["dma_capacity_cc"],
                    early,
                    -components["critical_chain_cc"],
                    -components["release_expert_chain_cc"],
                )
        else:
            try:
                common = common_by_scorer[scorer]
            except KeyError as exc:
                raise ValueError(f"unknown practical scorer {scorer!r}") from exc
        if mode == "SYNC":
            return common + sync_tail()
        return common + (
            late,
            early,
            sum_ntok,
            child.g_score,
            -s2pf,
            len(child.remaining),
        )

    if mode == "SYNC":
        if scorer == "full_lpt":
            completion = reference._lpt_completion_estimate(child)
        elif scorer == "full_lpt_load":
            completion = reference._lpt_load_completion_estimate(child)
        elif scorer == "full_cache":
            completion = reference._cache_aware_completion_estimate(child)
        elif scorer == "full_dual":
            completion = reference.completion_estimate(child)
        elif scorer == "pathmax":
            completion = child.f_score
        elif scorer == "head4_8_min":
            completion = min(
                bounded_head_lpt_estimate(child, 4),
                bounded_head_lpt_estimate(child, 8),
            )
        elif scorer == "head4_8_max":
            completion = max(
                bounded_head_lpt_estimate(child, 4),
                bounded_head_lpt_estimate(child, 8),
            )
        elif scorer == "head4_8_sum":
            completion = (
                bounded_head_lpt_estimate(child, 4)
                + bounded_head_lpt_estimate(child, 8)
            )
        elif scorer.startswith("head4_8_at"):
            threshold = int(scorer.removeprefix("head4_8_at"))
            completion = bounded_head_lpt_estimate(
                child, 8 if len(child.remaining) <= threshold else 4
            )
        elif scorer.startswith("head") and scorer.endswith("_aggregate"):
            completion = bounded_head_lpt_estimate(
                child, int(scorer.removeprefix("head").removesuffix("_aggregate"))
            )
        else:
            raise ValueError(f"unknown practical scorer {scorer!r}")
        common = (completion, len(child.remaining))
        return common + sync_tail()
    early, late = sorted((int(child.c2.task_end), int(child.c3.task_end)))
    return (
        late,
        early,
        sum_ntok,
        child.g_score,
        -s2pf,
        len(child.remaining),
    )


ONE_PROGRESS_PAIRWISE_SCORER = "lb_f_lpt_compute_dma_one_progress_pairwise_v1"
SYNC_HOT_PAIRWISE_SCORER = "lb_f_lpt_compute_dma_regime_pairwise_v2"
MIN2_PLATEAU_PAIRWISE_SCORER = "lb_f_lpt_compute_dma_regime_pairwise_v3"
EXPANDED_PLATEAU_PAIRWISE_SCORER = "lb_f_lpt_compute_dma_regime_pairwise_v4"
BOUNDED_PAIRWISE_SCORER = "lb_f_head8_compute_dma_regime_pairwise_v1"
HEAD5_HIST4_PAIRWISE_SCORER = (
    "lb_f_head5_hist4_compute_dma_regime_pairwise_v1"
)
BOUNDED_PAIRWISE_SCORERS = {
    BOUNDED_PAIRWISE_SCORER,
    HEAD5_HIST4_PAIRWISE_SCORER,
}


def _pairwise_lpt(
    child: reference.BeamState,
    scorer: str,
) -> int:
    if scorer == HEAD5_HIST4_PAIRWISE_SCORER:
        return bounded_head_hist_lpt_estimate(child)
    if scorer == BOUNDED_PAIRWISE_SCORER:
        return max(child.f_score, bounded_head_lpt_estimate(child, 8))
    return reference._lpt_completion_estimate(child)


def _pairwise_components(
    child: reference.BeamState,
    scorer: str,
) -> dict[str, int]:
    if scorer in BOUNDED_PAIRWISE_SCORERS:
        return bounded_state_lower_bound_components(child)
    return reference.state_lower_bound_components(
        child.c2, child.c3, child.remaining
    )


def _bounded_policy_state(
    state: reference.BeamState,
    scorer: str,
    *,
    before_f_score: int | None = None,
) -> reference.BeamState:
    if scorer not in BOUNDED_PAIRWISE_SCORERS:
        return state
    components = bounded_state_lower_bound_components(state)
    bounded_f = components["combined_cc"]
    if before_f_score is not None:
        bounded_f = max(before_f_score, bounded_f)
    return replace(state, f_score=bounded_f)


def _practical_scalar_scorer(scorer: str) -> str:
    return (
        "lb_f_lpt_compute_dma"
        if scorer
        in {
            ONE_PROGRESS_PAIRWISE_SCORER,
            SYNC_HOT_PAIRWISE_SCORER,
            MIN2_PLATEAU_PAIRWISE_SCORER,
            EXPANDED_PLATEAU_PAIRWISE_SCORER,
        }
        else "lb_f_head5_hist4_compute_dma"
        if scorer == HEAD5_HIST4_PAIRWISE_SCORER
        else "lb_f_head8_compute_dma"
        if scorer == BOUNDED_PAIRWISE_SCORER
        else scorer
    )


def _one_progress_alt_score(
    before: reference.BeamState,
    action: reference.StageAction,
    child: reference.BeamState,
    scorer: str,
) -> tuple:
    components = _pairwise_components(child, scorer)
    max_ntok, _min_ntok, sum_ntok, s2pf = _selected_action_features(action)
    early, late = sorted((int(child.c2.task_end), int(child.c3.task_end)))
    return (
        child.f_score,
        _pairwise_lpt(child, scorer),
        -s2pf,
        -sum_ntok,
        components["compute_cc"],
        components["dma_capacity_cc"],
        late,
        early,
        child.g_score,
        -max_ntok,
    )


def _sync_hot_alt_score(
    before: reference.BeamState,
    action: reference.StageAction,
    child: reference.BeamState,
    scorer: str,
) -> tuple:
    components = _pairwise_components(child, scorer)
    max_ntok, _min_ntok, sum_ntok, s2pf = _selected_action_features(action)
    return (
        child.f_score,
        _pairwise_lpt(child, scorer),
        -max_ntok,
        sum_ntok,
        components["compute_cc"],
        components["dma_capacity_cc"],
        child.g_score,
        -s2pf,
    )


def select_practical_probe_candidate(
    state: reference.BeamState,
    candidates: list[reference.StageAction],
    *,
    scorer: str,
    sync_tiebreak: str,
    window: tuple[int, int] = EXPLICIT_WINDOW,
) -> tuple[tuple, int, reference.StageAction, reference.BeamState, dict]:
    """Select one candidate, including the bounded pairwise regime policy."""
    scalar_scorer = _practical_scalar_scorer(scorer)
    ranked = []
    for candidate_index, action in enumerate(candidates):
        child = reference.apply_action(state, action)
        child = _bounded_policy_state(
            child,
            scorer,
            before_f_score=int(state.f_score),
        )
        ranked.append(
            (
                practical_probe_score(
                    state,
                    action,
                    child,
                    scorer=scalar_scorer,
                    sync_tiebreak=sync_tiebreak,
                ),
                candidate_index,
                action,
                child,
            )
        )
    if scorer not in {
        ONE_PROGRESS_PAIRWISE_SCORER,
        SYNC_HOT_PAIRWISE_SCORER,
        MIN2_PLATEAU_PAIRWISE_SCORER,
        EXPANDED_PLATEAU_PAIRWISE_SCORER,
        BOUNDED_PAIRWISE_SCORER,
        HEAD5_HIST4_PAIRWISE_SCORER,
    }:
        score, candidate_id, action, child = min(
            ranked, key=lambda item: item[:2]
        )
        return score, candidate_id, action, child, {
            "selector": "scalar_min",
            "pairwise_overrides": 0,
        }

    remaining_counters = _bounded_remaining_counters(state.remaining)
    remaining_count = remaining_counters.count
    top_entries = state.remaining[: min(window[0], remaining_count)]
    bottom_entries = state.remaining[
        max(0, remaining_count - window[1]) :
    ]
    top_loads = [int(ntok) for _eid, ntok in top_entries]
    min_remaining_load = int(bottom_entries[-1][1]) if bottom_entries else 0
    le2_count = (
        remaining_counters.small_block_hist[0]
        if scorer == HEAD5_HIST4_PAIRWISE_SCORER
        else remaining_counters.le2_count
    )
    one_progress_gate_state = (
        _explicit_mode(state) == "ONE_IDLE"
        and remaining_counters.token_sum <= 84
        and remaining_counters.odd_count <= 9
        and (top_loads[4] if remaining_count > 4 else 0) <= 4
    )
    sync_hot_gate_state = (
        scorer
        in {
            SYNC_HOT_PAIRWISE_SCORER,
            MIN2_PLATEAU_PAIRWISE_SCORER,
            EXPANDED_PLATEAU_PAIRWISE_SCORER,
            BOUNDED_PAIRWISE_SCORER,
            HEAD5_HIST4_PAIRWISE_SCORER,
        }
        and _explicit_mode(state) == "SYNC"
        and remaining_count >= 2
        and top_loads[0] >= 2 * top_loads[1]
        and 32 * le2_count > 11 * remaining_counters.count
    )
    plateau_gate_state = (
        scorer
        in {
            MIN2_PLATEAU_PAIRWISE_SCORER,
            EXPANDED_PLATEAU_PAIRWISE_SCORER,
            BOUNDED_PAIRWISE_SCORER,
            HEAD5_HIST4_PAIRWISE_SCORER,
        }
        and _explicit_mode(state) == "ONE_IDLE"
        and remaining_count
        >= (
            8
            if scorer
            in {
                EXPANDED_PLATEAU_PAIRWISE_SCORER,
                BOUNDED_PAIRWISE_SCORER,
                HEAD5_HIST4_PAIRWISE_SCORER,
            }
            else 24
        )
        and top_loads[1] >= 5
        and top_loads[0] <= 6
        and abs(int(state.c2.task_end) - int(state.c3.task_end))
        == 3 * TICK_CC
    )
    tail_plateau_gate_state = (
        scorer
        in {
            EXPANDED_PLATEAU_PAIRWISE_SCORER,
            BOUNDED_PAIRWISE_SCORER,
            HEAD5_HIST4_PAIRWISE_SCORER,
        }
        and _explicit_mode(state) == "ONE_IDLE"
        and 2 <= remaining_count <= 7
        and top_loads[1] >= 5
        and top_loads[0] <= 6
        and abs(int(state.c2.task_end) - int(state.c3.task_end))
        == 6 * TICK_CC
    )
    slack_fill_gate_state = (
        scorer
        in {
            EXPANDED_PLATEAU_PAIRWISE_SCORER,
            BOUNDED_PAIRWISE_SCORER,
            HEAD5_HIST4_PAIRWISE_SCORER,
        }
        and _explicit_mode(state) == "ONE_IDLE"
        and 8 <= remaining_count <= 16
        and top_loads[1] >= 8
        and abs(int(state.c2.task_end) - int(state.c3.task_end))
        >= 9 * TICK_CC
    )
    rank_by_eid = {
        int(eid): rank for rank, (eid, _ntok) in enumerate(top_entries)
    }
    for offset, (eid, _ntok) in enumerate(reversed(bottom_entries)):
        rank_by_eid.setdefault(int(eid), remaining_count - 1 - offset)
    best = ranked[0]
    overrides = 0
    for candidate in ranked[1:]:
        base_winner = min((best, candidate), key=lambda item: item[:2])
        use_alt = False
        alt_winner = base_winner
        if one_progress_gate_state:
            alt_winner = min(
                (best, candidate),
                key=lambda item: (
                    _one_progress_alt_score(state, item[2], item[3], scorer),
                    item[1],
                ),
            )
        if one_progress_gate_state and _explicit_child_key(
            base_winner[3]
        ) != _explicit_child_key(alt_winner[3]):
            _base_max, _base_min, base_sum, base_s2pf = _selected_action_features(
                base_winner[2]
            )
            _alt_max, _alt_min, alt_sum, _alt_s2pf = _selected_action_features(
                alt_winner[2]
            )
            base_early = min(
                int(base_winner[3].c2.task_end),
                int(base_winner[3].c3.task_end),
            )
            alt_early = min(
                int(alt_winner[3].c2.task_end),
                int(alt_winner[3].c3.task_end),
            )
            use_alt = (
                base_s2pf == 0
                and alt_sum > base_sum
                and alt_early - base_early <= TICK_CC
            )
        if plateau_gate_state:
            progress_winner = min(
                (best, candidate),
                key=lambda item: (
                    _one_progress_alt_score(state, item[2], item[3], scorer),
                    item[1],
                ),
            )
            if _explicit_child_key(base_winner[3]) != _explicit_child_key(
                progress_winner[3]
            ):
                _base_max, _base_min, _base_sum, base_s2pf = (
                    _selected_action_features(base_winner[2])
                )
                _progress_max, _progress_min, _progress_sum, progress_s2pf = (
                    _selected_action_features(progress_winner[2])
                )
                base_ranks = {
                    rank_by_eid[eid]
                    for eid in (base_winner[2].c2_eid, base_winner[2].c3_eid)
                    if eid >= 0
                }
                progress_ranks = {
                    rank_by_eid[eid]
                    for eid in (
                        progress_winner[2].c2_eid,
                        progress_winner[2].c3_eid,
                    )
                    if eid >= 0
                }
                base_early = min(
                    int(base_winner[3].c2.task_end),
                    int(base_winner[3].c3.task_end),
                )
                base_late = max(
                    int(base_winner[3].c2.task_end),
                    int(base_winner[3].c3.task_end),
                )
                progress_early = min(
                    int(progress_winner[3].c2.task_end),
                    int(progress_winner[3].c3.task_end),
                )
                progress_late = max(
                    int(progress_winner[3].c2.task_end),
                    int(progress_winner[3].c3.task_end),
                )
                plateau_override = (
                    base_s2pf == 0
                    and progress_s2pf > 0
                    and (
                        0 in progress_ranks
                        or (
                            scorer
                            in {
                                EXPANDED_PLATEAU_PAIRWISE_SCORER,
                                BOUNDED_PAIRWISE_SCORER,
                                HEAD5_HIST4_PAIRWISE_SCORER,
                            }
                            and min(progress_ranks) <= 3
                        )
                    )
                    and max(base_ranks) >= remaining_count - 2
                    and progress_early == base_early
                    and progress_late - base_late <= 6 * TICK_CC
                )
                if plateau_override:
                    alt_winner = progress_winner
                    use_alt = True
        if tail_plateau_gate_state or slack_fill_gate_state:
            fill_winner = min(
                (best, candidate),
                key=lambda item: (
                    _one_progress_alt_score(state, item[2], item[3], scorer),
                    item[1],
                ),
            )
            if _explicit_child_key(base_winner[3]) != _explicit_child_key(
                fill_winner[3]
            ):
                _base_max, _base_min, base_sum, base_s2pf = (
                    _selected_action_features(base_winner[2])
                )
                _fill_max, _fill_min, fill_sum, fill_s2pf = (
                    _selected_action_features(fill_winner[2])
                )
                base_ranks = {
                    rank_by_eid[eid]
                    for eid in (base_winner[2].c2_eid, base_winner[2].c3_eid)
                    if eid >= 0
                }
                fill_ranks = {
                    rank_by_eid[eid]
                    for eid in (fill_winner[2].c2_eid, fill_winner[2].c3_eid)
                    if eid >= 0
                }
                base_early, base_late = sorted(
                    (
                        int(base_winner[3].c2.task_end),
                        int(base_winner[3].c3.task_end),
                    )
                )
                fill_early, fill_late = sorted(
                    (
                        int(fill_winner[3].c2.task_end),
                        int(fill_winner[3].c3.task_end),
                    )
                )
                base_components = _pairwise_components(base_winner[3], scorer)
                fill_components = _pairwise_components(fill_winner[3], scorer)
                common_fill_requirements = (
                    fill_s2pf > 0
                    and fill_sum > base_sum
                    and fill_early >= base_early
                    and fill_winner[3].f_score == base_winner[3].f_score
                    and _pairwise_lpt(fill_winner[3], scorer)
                    == _pairwise_lpt(base_winner[3], scorer)
                    and fill_components["dma_capacity_cc"]
                    == base_components["dma_capacity_cc"]
                )
                tail_plateau_override = (
                    tail_plateau_gate_state
                    and common_fill_requirements
                    and base_s2pf > 0
                    and 0 in fill_ranks
                    and 0 not in base_ranks
                    and fill_late - base_late <= 3 * TICK_CC
                )
                slack_fill_override = (
                    slack_fill_gate_state
                    and common_fill_requirements
                    and base_s2pf == 0
                    and min(fill_ranks) <= 1
                    and max(base_ranks) >= remaining_count - 2
                    and fill_late == base_late
                )
                if tail_plateau_override or slack_fill_override:
                    alt_winner = fill_winner
                    use_alt = True
        if sync_hot_gate_state:
            hot_winner = min(
                (best, candidate),
                key=lambda item: (
                    _sync_hot_alt_score(state, item[2], item[3], scorer),
                    item[1],
                ),
            )
            if _explicit_child_key(base_winner[3]) != _explicit_child_key(
                hot_winner[3]
            ):
                base_max, _base_min, _base_sum, _base_s2pf = (
                    _selected_action_features(base_winner[2])
                )
                hot_max, _hot_min, _hot_sum, _hot_s2pf = (
                    _selected_action_features(hot_winner[2])
                )
                base_ranks = {
                    rank_by_eid[eid]
                    for eid in (base_winner[2].c2_eid, base_winner[2].c3_eid)
                    if eid >= 0
                }
                hot_ranks = {
                    rank_by_eid[eid]
                    for eid in (hot_winner[2].c2_eid, hot_winner[2].c3_eid)
                    if eid >= 0
                }
                base_components = _pairwise_components(base_winner[3], scorer)
                hot_components = _pairwise_components(hot_winner[3], scorer)
                sync_override = (
                    0 in hot_ranks
                    and 0 not in base_ranks
                    and hot_max > base_max
                    and hot_winner[3].f_score == base_winner[3].f_score
                    and _pairwise_lpt(hot_winner[3], scorer)
                    == _pairwise_lpt(base_winner[3], scorer)
                    and hot_components["compute_cc"]
                    - base_components["compute_cc"]
                    <= 3 * TICK_CC
                    and hot_components["dma_capacity_cc"]
                    - base_components["dma_capacity_cc"]
                    <= 6 * TICK_CC
                    and (
                        base_components["dma_capacity_cc"] > 115 * TICK_CC
                        or (
                            scorer
                            in {
                                MIN2_PLATEAU_PAIRWISE_SCORER,
                                EXPANDED_PLATEAU_PAIRWISE_SCORER,
                                BOUNDED_PAIRWISE_SCORER,
                                HEAD5_HIST4_PAIRWISE_SCORER,
                            }
                            and min_remaining_load >= 2
                            and base_components["dma_capacity_cc"]
                            > 102 * TICK_CC
                        )
                    )
                )
                if sync_override:
                    alt_winner = hot_winner
                    use_alt = True
        if use_alt:
            best = alt_winner
            overrides += 1
        else:
            best = base_winner
    score, candidate_id, action, child = best
    return score, candidate_id, action, child, {
        "selector": (
            "regime_pairwise_v2"
            if scorer == SYNC_HOT_PAIRWISE_SCORER
            else "regime_pairwise_v3"
            if scorer == MIN2_PLATEAU_PAIRWISE_SCORER
            else "regime_pairwise_v4"
            if scorer == EXPANDED_PLATEAU_PAIRWISE_SCORER
            else "bounded_regime_pairwise_v1"
            if scorer == BOUNDED_PAIRWISE_SCORER
            else "head5_hist4_regime_pairwise_v1"
            if scorer == HEAD5_HIST4_PAIRWISE_SCORER
            else "one_progress_pairwise_v1"
        ),
        "pairwise_overrides": overrides,
        "one_progress_gate_state": one_progress_gate_state,
        "sync_hot_gate_state": sync_hot_gate_state,
        "plateau_gate_state": plateau_gate_state,
        "tail_plateau_gate_state": tail_plateau_gate_state,
        "slack_fill_gate_state": slack_fill_gate_state,
    }


def run_practical_probe_case(
    job: tuple[
        dict,
        tuple[ExplicitCandidateToken, ...],
        str,
        str,
        str,
        str,
        int,
        bool,
        bool,
        tuple[int, int],
    ]
):
    """Pickle-safe closed-loop run for one frozen case."""
    (
        case,
        tokens,
        scorer,
        sync_tiebreak,
        start_policy,
        safety_policy,
        min_token_uses,
        direct_generator,
        strict_token_bank,
        window,
    ) = job
    token_dist = {
        eid: int(ntok)
        for eid, ntok in enumerate(case["counts"])
        if int(ntok) > 0
    }
    target = _target_cc(case)
    state = _bounded_policy_state(
        reference.FourStageScheduler(token_dist)._initial_state(), scorer
    )
    selected = []
    candidate_counts = []
    safety_selected = 0
    while state.remaining:
        try:
            candidates, stats = generate_practical_probe_candidates(
                state,
                tokens,
                start_policy,
                safety_policy,
                direct_generator,
                strict_token_bank,
                window,
            )
        except RuntimeError as exc:
            if not strict_token_bank:
                raise
            return {
                "name": case["name"],
                "status": "candidate_dead_end",
                "error": str(exc),
                "scorer": scorer,
                "sync_tiebreak": sync_tiebreak,
                "start_policy": start_policy,
                "safety_policy": "disabled",
                "minimum_certificate_token_uses": min_token_uses,
                "direct_generator": direct_generator,
                "strict_token_bank": True,
                "optimal_ticks": _ticks_text(target),
                "makespan_ticks": None,
                "gap_ticks": None,
                "optimal": False,
                "rounds": len(selected),
                "candidate_count_max": max(candidate_counts, default=0),
                "candidate_count_mean": (
                    statistics.mean(candidate_counts) if candidate_counts else 0.0
                ),
                "safety_single_selected": safety_selected,
                "selected": selected,
            }
        candidate_counts.append(stats["concrete_candidates"])
        _score, _tie, action, child, selection_meta = (
            select_practical_probe_candidate(
                state,
                candidates,
                scorer=scorer,
                sync_tiebreak=sync_tiebreak,
                window=window,
            )
        )
        source = stats["source_by_child"][_explicit_child_key(child)]
        safety_selected += source == "safety_single"
        selected.append(
            {
                "family": _explicit_family(action),
                "source": source,
                "token": _serialize_explicit_token(
                    _explicit_candidate_token(state, action), 0
                ),
                "score": [int(value) for value in _score],
                "selection": selection_meta,
            }
        )
        state = child
    replay = reference.validate_schedule_history(state.history, token_dist)
    if replay != state.g_score:
        raise AssertionError("practical probe trace failed explicit-DMA replay")
    return {
        "name": case["name"],
        "status": "terminal",
        "scorer": scorer,
        "sync_tiebreak": sync_tiebreak,
        "start_policy": start_policy,
        "safety_policy": safety_policy,
        "minimum_certificate_token_uses": min_token_uses,
        "direct_generator": direct_generator,
        "strict_token_bank": strict_token_bank,
        "optimal_ticks": _ticks_text(target),
        "makespan_ticks": _ticks_text(state.g_score),
        "gap_ticks": _ticks_text(state.g_score - target),
        "optimal": state.g_score == target,
        "rounds": len(selected),
        "candidate_count_max": max(candidate_counts),
        "candidate_count_mean": statistics.mean(candidate_counts),
        "safety_single_selected": safety_selected,
        "selected": selected,
    }


def evaluate_practical_closed_loop(
    proof_cases: list[dict],
    tokens: tuple[ExplicitCandidateToken, ...],
    *,
    workers: int,
    case_limit: int | None,
    scorer: str,
    sync_tiebreak: str,
    start_policy: str,
    safety_policy: str,
    min_token_uses: int,
    direct_generator: bool = False,
    strict_token_bank: bool = False,
    window: tuple[int, int] = EXPLICIT_WINDOW,
) -> dict:
    cases = proof_cases[:case_limit] if case_limit else proof_cases
    jobs = [
        (
            case,
            tokens,
            scorer,
            sync_tiebreak,
            start_policy,
            safety_policy,
            min_token_uses,
            direct_generator,
            strict_token_bank,
            window,
        )
        for case in cases
    ]
    if workers == 1:
        rows = [run_practical_probe_case(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(run_practical_probe_case, jobs))
    terminal_rows = [row for row in rows if row["status"] == "terminal"]
    gaps = [Fraction(row["gap_ticks"]) for row in terminal_rows]
    return {
        "name": f"{scorer}_{sync_tiebreak}_progress_tiebreak_probe",
        "final_scorer": False,
        "standalone_prefetch_candidates": False,
        "minimum_certificate_token_uses": min_token_uses,
        "direct_generator": direct_generator,
        "strict_token_bank": strict_token_bank,
        "sync_tiebreak": sync_tiebreak,
        "start_policy": start_policy,
        "safety_policy": safety_policy,
        "certificate_composite_tokens": len(tokens),
        "target_used_by_policy": False,
        "scorer_input_contract": (
            (
                "top5 plus bottom1 descriptors, four cold-tail block-count "
                "bins, and maintained aggregate lower-bound counters"
            )
            if scorer == HEAD5_HIST4_PAIRWISE_SCORER
            else (
                "top8 remaining descriptors plus maintained tail-work, "
                "remaining-block, remaining-expert and mandatory-DMA counters"
            )
            if scorer == BOUNDED_PAIRWISE_SCORER
            else "full remaining list; offline performance upper bound"
            if scorer.startswith("full_")
            else (
                "full remaining LPT plus maintained lower-bound components; "
                "offline tie-break upper bound"
            )
            if scorer.startswith("lb_")
            else "admissible pathmax state bound and maintained aggregate work"
            if scorer == "pathmax"
            else f"first {scorer.removeprefix('head').removesuffix('_aggregate')} "
            "remaining descriptors plus one maintained aggregate tail-work scalar"
        ),
        "cases": len(rows),
        "terminal_cases": len(terminal_rows),
        "candidate_dead_end_cases": len(rows) - len(terminal_rows),
        "optimal_cases": sum(row["optimal"] for row in rows),
        "gap_ticks": {
            "sum_terminal_only": str(sum(gaps)),
            "mean_terminal_only": (
                float(sum(gaps) / len(gaps)) if gaps else None
            ),
            "max_terminal_only": str(max(gaps)) if gaps else None,
        },
        "candidate_count_max": max(
            row["candidate_count_max"] for row in rows
        ),
        "safety_single_selected": sum(row["safety_single_selected"] for row in rows),
        "rows": rows,
    }


def run_practical_probe_configuration(
    job: tuple[
        list[dict],
        tuple[ExplicitCandidateToken, ...],
        int | None,
        str,
        str,
        str,
        tuple[int, int],
    ]
) -> dict:
    """Pickle-safe matrix unit: one scorer configuration, all chosen cases."""
    (
        proof_cases,
        tokens,
        case_limit,
        scorer,
        sync_tiebreak,
        start_policy,
        window,
    ) = job
    return evaluate_practical_closed_loop(
        proof_cases,
        tokens,
        workers=1,
        case_limit=case_limit,
        scorer=scorer,
        sync_tiebreak=sync_tiebreak,
        start_policy=start_policy,
        safety_policy="disabled",
        min_token_uses=1,
        direct_generator=True,
        strict_token_bank=True,
        window=window,
    )


STRICT_MATRIX_SCORERS = (
    "full_lpt",
    "full_lpt_load",
    "full_cache",
    "full_dual",
    "pathmax",
    "head4_8_min",
    "head4_8_max",
    "head4_8_sum",
    "head4_8_at8",
    "head4_8_at16",
    "head4_8_at24",
    "head4_8_at32",
    "head4_aggregate",
    "head6_aggregate",
    "head8_aggregate",
    "lb_f_lpt",
    "lb_f_lpt_compute",
    "lb_f_lpt_compute_late",
    "lb_f_lpt_compute_early",
    "lb_f_lpt_compute_imbalance",
    "lb_f_lpt_compute_dma",
    "lb_f_lpt_compute_release",
    "lb_certified_lex",
    ONE_PROGRESS_PAIRWISE_SCORER,
    SYNC_HOT_PAIRWISE_SCORER,
    MIN2_PLATEAU_PAIRWISE_SCORER,
    EXPANDED_PLATEAU_PAIRWISE_SCORER,
    BOUNDED_PAIRWISE_SCORER,
    HEAD5_HIST4_PAIRWISE_SCORER,
)
STRICT_MATRIX_TIEBREAKS = (
    "hot_cold",
    "small_pair",
    "earliest_commit",
    "equal_pair",
)


def _compact_closed_loop_run(run: dict) -> dict:
    """Drop action histories while retaining every comparison outcome."""
    return {
        key: value for key, value in run.items() if key != "rows"
    } | {
        "rows": [
            {
                key: row.get(key)
                for key in (
                    "name",
                    "status",
                    "optimal_ticks",
                    "makespan_ticks",
                    "gap_ticks",
                    "optimal",
                    "rounds",
                    "candidate_count_max",
                    "candidate_count_mean",
                )
            }
            for row in run["rows"]
        ]
    }


def _closed_loop_rank_key(run: dict) -> tuple:
    """Rank only comparable completed runs ahead of dead-ended runs."""
    gap = Fraction(run["gap_ticks"]["sum_terminal_only"])
    max_gap = (
        Fraction(run["gap_ticks"]["max_terminal_only"])
        if run["gap_ticks"]["max_terminal_only"] is not None
        else Fraction(10**18)
    )
    return (
        int(run["candidate_dead_end_cases"]),
        -int(run["optimal_cases"]),
        gap,
        max_gap,
        str(run["name"]),
    )


def evaluate_strict_closed_loop_bank(
    proof_path: Path,
    token_bank_path: Path,
    output_path: Path,
    *,
    workers: int,
    case_limit: int | None,
    scorer: str,
    sync_tiebreak: str,
    start_policy: str,
    matrix: bool,
) -> dict:
    """Evaluate scoring only, with the supplied fixed ROM as a hard boundary.

    This deliberately bypasses the slower certificate-union and legacy-bank
    audits.  No safety candidate is manufactured after an off-path decision:
    such a state is reported as ``candidate_dead_end`` and remains visible in
    the scorer comparison.
    """
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof_cases = list(proof["cases"])
    tokens = load_explicit_token_bank(token_bank_path)
    configurations = (
        [
            (matrix_scorer, matrix_tiebreak)
            for matrix_scorer in STRICT_MATRIX_SCORERS
            for matrix_tiebreak in STRICT_MATRIX_TIEBREAKS
        ]
        if matrix
        else [(scorer, sync_tiebreak)]
    )
    started = time.perf_counter()
    if matrix:
        jobs = [
            (
                proof_cases,
                tokens,
                case_limit,
                run_scorer,
                run_tiebreak,
                start_policy,
                EXPLICIT_WINDOW,
            )
            for run_scorer, run_tiebreak in configurations
        ]
        if workers == 1:
            runs = [run_practical_probe_configuration(job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                runs = list(pool.map(run_practical_probe_configuration, jobs))
    else:
        runs = [
            evaluate_practical_closed_loop(
                proof_cases,
                tokens,
                workers=workers,
                case_limit=case_limit,
                scorer=scorer,
                sync_tiebreak=sync_tiebreak,
                start_policy=start_policy,
                safety_policy="disabled",
                min_token_uses=1,
                direct_generator=True,
                strict_token_bank=True,
            )
        ]
    best = min(runs, key=_closed_loop_rank_key)
    payload = {
        "schema": "olmoe-strict-fixed-token-scorer-v1",
        "complete": all(run["candidate_dead_end_cases"] == 0 for run in runs),
        "final_scorer": False,
        "interpretation": {
            "candidate_boundary": (
                "Only the supplied fixed state-relative token ROM is lowered; "
                "there is no safety fallback, dynamic candidate discovery, or "
                "target-dependent action generation."
            ),
            "complete_meaning": (
                "Every evaluated scorer reaches a terminal state on every case; "
                "optimality is reported separately and is not implied."
            ),
            "ranking": (
                "fewest dead ends, then most optimal cases, then minimum summed "
                "and maximum terminal-only gap"
            ),
        },
        "manifest": {
            "proof": str(proof_path.resolve()),
            "proof_sha256": _sha256(proof_path),
            "token_bank": str(token_bank_path.resolve()),
            "token_bank_sha256": _sha256(token_bank_path),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "window": list(EXPLICIT_WINDOW),
            "start_policy": start_policy,
            "case_limit": case_limit,
            "matrix": matrix,
        },
        "token_rom_entries": len(tokens),
        "configurations": len(runs),
        "best_configuration": {
            key: value for key, value in best.items() if key != "rows"
        },
        "runs": [_compact_closed_loop_run(run) for run in runs],
        "best_run_with_trace": best,
        "runtime_s": time.perf_counter() - started,
    }
    _atomic_json(output_path, payload)
    return payload


def _strict_policy_trajectory(
    case: dict,
    tokens: tuple[ExplicitCandidateToken, ...],
    *,
    scorer: str,
    sync_tiebreak: str,
    start_policy: str,
    window: tuple[int, int] = EXPLICIT_WINDOW,
) -> tuple[list[reference.BeamState], list[reference.StageAction]]:
    token_dist = {
        eid: int(ntok)
        for eid, ntok in enumerate(case["counts"])
        if int(ntok) > 0
    }
    state = _bounded_policy_state(
        reference.FourStageScheduler(token_dist)._initial_state(), scorer
    )
    states = [state]
    actions = []
    while state.remaining:
        candidates, _stats = generate_practical_probe_candidates(
            state,
            tokens,
            start_policy,
            "disabled",
            direct_generator=True,
            strict_token_bank=True,
            window=window,
        )
        _score, _repr, action, child, _selection_meta = (
            select_practical_probe_candidate(
                state,
                candidates,
                scorer=scorer,
                sync_tiebreak=sync_tiebreak,
                window=window,
            )
        )
        actions.append(action)
        states.append(child)
        state = child
    return states, actions


def audit_bounded_runtime_contract(
    proof_cases: list[dict],
    tokens: tuple[ExplicitCandidateToken, ...],
    *,
    scorer: str,
    sync_tiebreak: str,
    start_policy: str,
    window: tuple[int, int],
) -> dict:
    """Prove the bounded-state implementation against the offline scorer."""
    counter_mismatches = []
    component_mismatches = []
    pathmax_mismatches = []
    lpt_mismatches = []
    trajectory_mismatches = []
    transitions = 0
    states_checked = 0
    candidate_children_checked = 0
    component_names = (
        "committed_cc",
        "compute_cc",
        "release_expert_chain_cc",
        "critical_chain_cc",
        "mandatory_dma_bytes",
        "dma_release_cc",
        "dma_capacity_cc",
        "combined_cc",
    )
    for case in proof_cases:
        bounded_states, bounded_actions = _strict_policy_trajectory(
            case,
            tokens,
            scorer=scorer,
            sync_tiebreak=sync_tiebreak,
            start_policy=start_policy,
            window=window,
        )
        offline_states, offline_actions = _strict_policy_trajectory(
            case,
            tokens,
            scorer=EXPANDED_PLATEAU_PAIRWISE_SCORER,
            sync_tiebreak=sync_tiebreak,
            start_policy=start_policy,
            window=window,
        )
        bounded_trace = [serialize_action(action) for action in bounded_actions]
        offline_trace = [serialize_action(action) for action in offline_actions]
        if bounded_trace != offline_trace:
            first = next(
                (
                    index
                    for index, (bounded, offline) in enumerate(
                        zip(bounded_trace, offline_trace)
                    )
                    if bounded != offline
                ),
                min(len(bounded_trace), len(offline_trace)),
            )
            trajectory_mismatches.append(
                {
                    "name": case["name"],
                    "first_action_index": first,
                    "bounded_actions": len(bounded_trace),
                    "offline_actions": len(offline_trace),
                }
            )
        for state_index, state in enumerate(bounded_states):
            states_checked += 1
            candidates = []
            if state.remaining:
                candidates, _stats = generate_practical_probe_candidates(
                    state,
                    tokens,
                    start_policy,
                    "disabled",
                    direct_generator=True,
                    strict_token_bank=True,
                    window=window,
                )
            lpt_states = [("state", state)] + [
                (
                    f"candidate_{candidate_index}",
                    _bounded_policy_state(
                        reference.apply_action(state, action),
                        scorer,
                        before_f_score=int(state.f_score),
                    ),
                )
                for candidate_index, action in enumerate(candidates)
            ]
            candidate_children_checked += len(candidates)
            for locus, lpt_state in lpt_states:
                try:
                    bounded_lpt = (
                        bounded_head_hist_lpt_estimate(lpt_state)
                        if scorer == HEAD5_HIST4_PAIRWISE_SCORER
                        else max(
                            int(lpt_state.f_score),
                            bounded_head_lpt_estimate(lpt_state, 8),
                        )
                    )
                except AssertionError as exc:
                    lpt_mismatches.append(
                        {
                            "name": case["name"],
                            "state_index": state_index,
                            "locus": locus,
                            "contract_error": str(exc),
                        }
                    )
                    continue
                full_lpt = reference._lpt_completion_estimate(lpt_state)
                if bounded_lpt != full_lpt:
                    lpt_mismatches.append(
                        {
                            "name": case["name"],
                            "state_index": state_index,
                            "locus": locus,
                            "bounded_lpt": int(bounded_lpt),
                            "full_lpt": int(full_lpt),
                        }
                    )
            bounded_components = bounded_state_lower_bound_components(state)
            full_components = reference.state_lower_bound_components(
                state.c2, state.c3, state.remaining
            )
            different = {
                key: {
                    "bounded": int(bounded_components[key]),
                    "full": int(full_components[key]),
                }
                for key in component_names
                if bounded_components[key] != full_components[key]
            }
            if different:
                component_mismatches.append(
                    {
                        "name": case["name"],
                        "state_index": state_index,
                        "components": different,
                    }
                )
            if state_index < len(offline_states):
                offline_f = int(offline_states[state_index].f_score)
                if int(state.f_score) != offline_f:
                    pathmax_mismatches.append(
                        {
                            "name": case["name"],
                            "state_index": state_index,
                            "bounded_f_score": int(state.f_score),
                            "offline_f_score": offline_f,
                        }
                    )
        for action_index, (before, after) in enumerate(
            zip(bounded_states, bounded_states[1:])
        ):
            transitions += 1
            before_counters = _bounded_remaining_counters(before.remaining)
            after_counters = _bounded_remaining_counters(after.remaining)
            before_loads = dict(before.remaining)
            after_eids = {eid for eid, _ntok in after.remaining}
            consumed_loads = [
                int(ntok)
                for eid, ntok in before_loads.items()
                if eid not in after_eids
            ]
            expected = BoundedRemainingCounters(
                count=before_counters.count - len(consumed_loads),
                token_sum=before_counters.token_sum - sum(consumed_loads),
                odd_count=before_counters.odd_count
                - sum(ntok & 1 for ntok in consumed_loads),
                le2_count=before_counters.le2_count
                - sum(ntok <= 2 for ntok in consumed_loads),
                block_sum=before_counters.block_sum
                - sum(
                    (ntok + reference.FULL_M_DIM - 1) // reference.FULL_M_DIM
                    for ntok in consumed_loads
                ),
                best_work_cc=before_counters.best_work_cc
                - sum(reference._best_task_time(ntok) for ntok in consumed_loads),
                small_block_hist=tuple(
                    before_counters.small_block_hist[bucket - 1]
                    - sum(
                        (
                            ntok + reference.FULL_M_DIM - 1
                        )
                        // reference.FULL_M_DIM
                        == bucket
                        for ntok in consumed_loads
                    )
                    for bucket in range(1, 5)
                ),
            )
            if expected != after_counters:
                counter_mismatches.append(
                    {
                        "name": case["name"],
                        "action_index": action_index,
                        "expected": vars(expected),
                        "actual": vars(after_counters),
                    }
                )
    complete = not (
        counter_mismatches
        or component_mismatches
        or pathmax_mismatches
        or lpt_mismatches
        or trajectory_mismatches
    )
    return {
        "complete": complete,
        "cases": len(proof_cases),
        "transitions": transitions,
        "states": states_checked,
        "candidate_children": candidate_children_checked,
        "counter_update_mismatches": len(counter_mismatches),
        "full_lb_component_mismatch_states": len(component_mismatches),
        "pathmax_mismatch_states": len(pathmax_mismatches),
        "full_lpt_mismatch_states_or_candidates": len(lpt_mismatches),
        "offline_action_trace_mismatch_cases": len(trajectory_mismatches),
        "first_mismatches": {
            "counter": counter_mismatches[:3],
            "lower_bound": component_mismatches[:3],
            "pathmax": pathmax_mismatches[:3],
            "lpt": lpt_mismatches[:3],
            "trajectory": trajectory_mismatches[:3],
        },
        "runtime_contract": {
            "visible_descriptors": f"top{window[0]} plus bottom{window[1]}",
            "maintained_counters": (
                [
                    "remaining_count",
                    "remaining_token_sum",
                    "remaining_odd_count",
                    "remaining_shape_c_block_sum",
                    "remaining_block_hist_1_through_4",
                    "pathmax_f_score",
                ]
                if scorer == HEAD5_HIST4_PAIRWISE_SCORER
                else [
                    "remaining_count",
                    "remaining_token_sum",
                    "remaining_odd_count",
                    "remaining_le2_count",
                    "remaining_shape_c_block_sum",
                    "remaining_best_work_cc",
                    "pathmax_f_score",
                ]
            ),
            "derived_quantities": (
                [
                    "remaining_le2_count = remaining_block_hist_1",
                    "remaining_best_work_cc = block_sum * best_block_time_cc",
                ]
                if scorer == HEAD5_HIST4_PAIRWISE_SCORER
                else []
            ),
            "forbidden_hidden_state": [
                "full remaining descriptor scan",
                "target optimum",
                "child search",
                "S4 prefetch residency",
            ],
        },
    }


def audit_local_lowering_complexity(
    proof_cases: list[dict],
    tokens: tuple[ExplicitCandidateToken, ...],
    *,
    scorer: str,
    sync_tiebreak: str,
    start_policy: str,
    window: tuple[int, int],
) -> dict:
    """Count token-local physical variants before the global scorer."""
    decision_states = 0
    profile_attempts = []
    raw_actions = []
    bounded_actions = []
    scorer_candidates = []
    max_bounded_actions_per_token = 0
    selected_start_classes = Counter()
    for case in proof_cases:
        states, actions = _strict_policy_trajectory(
            case,
            tokens,
            scorer=scorer,
            sync_tiebreak=sync_tiebreak,
            start_policy=start_policy,
            window=window,
        )
        for state, selected_action in zip(states, actions):
            decision_states += 1
            state_attempts = 0
            state_raw = 0
            state_bounded = 0
            for token in tokens:
                materialized, stats = _direct_materialize_explicit_token(
                    state, token, window
                )
                eligible = [
                    action
                    for action in materialized
                    if _bounded_release_action_allowed(state, action)
                ]
                state_attempts += int(stats["profile_attempts"])
                state_raw += len(materialized)
                state_bounded += len(eligible)
                max_bounded_actions_per_token = max(
                    max_bounded_actions_per_token, len(eligible)
                )
            generated, _stats = generate_direct_explicit_candidates(
                state,
                tokens,
                window=window,
                start_policy=start_policy,
            )
            profile_attempts.append(state_attempts)
            raw_actions.append(state_raw)
            bounded_actions.append(state_bounded)
            scorer_candidates.append(len(generated))
            if selected_action.c2_eid >= 0 and selected_action.c3_eid >= 0:
                selected_start_classes["sync_pair"] += 1
            elif selected_action.c2_eid >= 0:
                start = int(selected_action.c2_start)
                cluster, peer = state.c2, state.c3
                if start == int(cluster.task_end):
                    selected_start_classes["cluster_release"] += 1
                elif peer.s2pf_end >= 0 and start == int(peer.s2pf_end):
                    selected_start_classes["peer_s2pf_end"] += 1
                elif start == int(peer.task_end):
                    selected_start_classes["peer_task_end"] += 1
                else:
                    selected_start_classes["outside_bounded_set"] += 1
            elif selected_action.c3_eid >= 0:
                start = int(selected_action.c3_start)
                cluster, peer = state.c3, state.c2
                if start == int(cluster.task_end):
                    selected_start_classes["cluster_release"] += 1
                elif peer.s2pf_end >= 0 and start == int(peer.s2pf_end):
                    selected_start_classes["peer_s2pf_end"] += 1
                elif start == int(peer.task_end):
                    selected_start_classes["peer_task_end"] += 1
                else:
                    selected_start_classes["outside_bounded_set"] += 1
    by_mode = Counter(token.logical.mode for token in tokens)
    static_local_variant_upper_bound = {
        "SYNC": 2 * by_mode.get("SYNC", 0),
        "ONE_IDLE": 3 * by_mode.get("ONE_IDLE", 0),
        "TERMINAL": 3 * by_mode.get("TERMINAL", 0),
    }
    return {
        "complete": selected_start_classes.get("outside_bounded_set", 0) == 0,
        "decision_states": decision_states,
        "rom_entries_by_mode": dict(sorted(by_mode.items())),
        "local_selector": {
            "PAIR": (
                "evaluate the two expert-to-cluster placements and keep the "
                "minimum (max_end, sum_end, latest_start)"
            ),
            "SINGLE": (
                "evaluate cluster_release, peer_s2pf_end and peer_task_end; "
                "discard illegal points and keep the same minimum"
            ),
        },
        "static_local_variant_upper_bound_by_mode": static_local_variant_upper_bound,
        "observed_on_65": {
            "profile_attempts_max": max(profile_attempts, default=0),
            "raw_materialized_actions_max": max(raw_actions, default=0),
            "bounded_release_actions_max": max(bounded_actions, default=0),
            "bounded_actions_per_token_max": max_bounded_actions_per_token,
            "post_local_selector_candidates_max": max(
                scorer_candidates, default=0
            ),
            "selected_start_classes": dict(sorted(selected_start_classes.items())),
        },
    }


def certify_pruned_closed_loop_bank(
    proof_path: Path,
    source_bank_path: Path,
    output_bank_path: Path,
    output_report_path: Path,
    *,
    keep_indices: tuple[int, ...],
    workers: int,
    scorer: str,
    sync_tiebreak: str,
    start_policy: str,
    window: tuple[int, int],
    selector_map: dict[str, str] | None = None,
    final_policy: bool = False,
) -> dict:
    """Materialize and causally certify one proposed implementation ROM."""
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof_cases = list(proof["cases"])
    source_payload = json.loads(source_bank_path.read_text(encoding="utf-8"))
    original_tokens = load_explicit_token_bank(source_bank_path)
    selector_map = dict(selector_map or {})
    projected: dict[ExplicitCandidateToken, dict] = {}
    projected_origins: dict[ExplicitCandidateToken, list[int]] = {}
    for original_index, (token, row) in enumerate(
        zip(original_tokens, source_payload["tokens"])
    ):
        projected_token = replace(
            token,
            logical=replace(
                token.logical,
                selectors=tuple(
                    selector_map.get(selector, selector)
                    for selector in token.logical.selectors
                ),
            ),
        )
        if projected_token not in projected:
            projected[projected_token] = _serialize_explicit_token(
                projected_token, int(row.get("witness_uses", 0))
            )
            projected_origins[projected_token] = [original_index]
        else:
            projected[projected_token]["witness_uses"] += int(
                row.get("witness_uses", 0)
            )
            projected_origins[projected_token].append(original_index)
    source_tokens = tuple(projected)
    source_rows = [projected[token] for token in source_tokens]
    source_origin_indices = [
        projected_origins[token] for token in source_tokens
    ]
    if not keep_indices or len(set(keep_indices)) != len(keep_indices):
        raise ValueError("keep_indices must be a non-empty unique sequence")
    if tuple(sorted(keep_indices)) != keep_indices:
        raise ValueError("keep_indices must be sorted")
    if keep_indices[0] < 0 or keep_indices[-1] >= len(source_tokens):
        raise ValueError("keep_indices outside source token ROM")
    tokens = tuple(source_tokens[index] for index in keep_indices)
    started = time.perf_counter()
    common = dict(
        workers=workers,
        case_limit=None,
        scorer=scorer,
        sync_tiebreak=sync_tiebreak,
        start_policy=start_policy,
        safety_policy="disabled",
        min_token_uses=1,
        direct_generator=True,
        strict_token_bank=True,
        window=window,
    )
    source_run = evaluate_practical_closed_loop(proof_cases, source_tokens, **common)
    pruned_run = evaluate_practical_closed_loop(proof_cases, tokens, **common)
    start_policy_runs = {start_policy: pruned_run}
    for ablation_policy in ("earliest_start", "earliest_finish", "all", "latest_start"):
        if ablation_policy == start_policy:
            continue
        start_policy_runs[ablation_policy] = evaluate_practical_closed_loop(
            proof_cases,
            tokens,
            **(common | {"start_policy": ablation_policy}),
        )
    leave_one_out_jobs = [
        (
            proof_cases,
            tokens[:index] + tokens[index + 1 :],
            None,
            scorer,
            sync_tiebreak,
            start_policy,
            window,
        )
        for index in range(len(tokens))
    ]
    if workers == 1:
        leave_one_out_runs = [
            run_practical_probe_configuration(job) for job in leave_one_out_jobs
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            leave_one_out_runs = list(
                pool.map(run_practical_probe_configuration, leave_one_out_jobs)
            )
    bank_payload = {
        "schema": "olmoe-bounded-fixed-token-rom-v1",
        "final_bank": False,
        "implementation_candidate": True,
        "target_used_by_generator": False,
        "manifest": {
            "source_bank": str(source_bank_path.resolve()),
            "source_bank_sha256": _sha256(source_bank_path),
            "source_original_token_rom_entries": len(original_tokens),
            "source_projected_origin_indices": source_origin_indices,
            "selector_map": selector_map,
            "source_indices": list(keep_indices),
            "window": list(window),
            "scorer": scorer,
            "start_policy": start_policy,
        },
        "summary": {
            "source_token_rom_entries": len(source_tokens),
            "token_rom_entries": len(tokens),
            "entries_by_mode": dict(
                sorted(Counter(token.logical.mode for token in tokens).items())
            ),
            "entries_by_family": dict(
                sorted(Counter(token.logical.family for token in tokens).items())
            ),
            "candidate_count_max_on_65": pruned_run["candidate_count_max"],
        },
        "interpretation": {
            "implementation_candidate": (
                "Closed-loop inclusion-minimal on the frozen 65 cases; this is "
                "not a universal proof that no smaller ROM can exist."
            ),
            "coverage_baseline": (
                "The immutable 29-entry source bank remains the independent "
                "65/65 candidate-sufficiency certificate."
            ),
        },
        "tokens": [source_rows[index] for index in keep_indices],
    }
    runtime_audit = audit_bounded_runtime_contract(
        proof_cases,
        tokens,
        scorer=scorer,
        sync_tiebreak=sync_tiebreak,
        start_policy=start_policy,
        window=window,
    )
    local_lowering_audit = audit_local_lowering_complexity(
        proof_cases,
        tokens,
        scorer=scorer,
        sync_tiebreak=sync_tiebreak,
        start_policy=start_policy,
        window=window,
    )
    leave_one_out = []
    for local_index, (source_index, run) in enumerate(
        zip(keep_indices, leave_one_out_runs)
    ):
        leave_one_out.append(
            {
                "local_index": local_index,
                "source_index": source_index,
                "mode": tokens[local_index].logical.mode,
                "family": tokens[local_index].logical.family,
                "selectors": list(tokens[local_index].logical.selectors),
                "terminal_cases": run["terminal_cases"],
                "candidate_dead_end_cases": run["candidate_dead_end_cases"],
                "optimal_cases": run["optimal_cases"],
                "gap_ticks": run["gap_ticks"],
                "removable": (
                    run["candidate_dead_end_cases"] == 0
                    and run["optimal_cases"] == len(proof_cases)
                ),
            }
        )
    expected_cases = len(proof_cases)
    complete = (
        source_run["candidate_dead_end_cases"] == 0
        and source_run["optimal_cases"] == expected_cases
        and pruned_run["candidate_dead_end_cases"] == 0
        and pruned_run["optimal_cases"] == expected_cases
        and not any(row["removable"] for row in leave_one_out)
        and runtime_audit["complete"]
        and local_lowering_audit["complete"]
    )
    if final_policy and not complete:
        raise RuntimeError("refusing to freeze an incomplete policy certificate")
    bank_payload["final_bank"] = bool(final_policy and complete)
    bank_payload["implementation_candidate"] = not bank_payload["final_bank"]
    _atomic_json(output_bank_path, bank_payload)
    report = {
        "schema": "olmoe-bounded-pruned-rom-certificate-v1",
        "complete": complete,
        "final_policy": bool(final_policy and complete),
        "manifest": {
            "proof": str(proof_path.resolve()),
            "proof_sha256": _sha256(proof_path),
            "source_bank": str(source_bank_path.resolve()),
            "source_bank_sha256": _sha256(source_bank_path),
            "pruned_bank": str(output_bank_path.resolve()),
            "pruned_bank_sha256": _sha256(output_bank_path),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "window": list(window),
            "scorer": scorer,
            "sync_tiebreak": sync_tiebreak,
            "start_policy": start_policy,
            "source_indices": list(keep_indices),
        },
        "summary": {
            "cases": expected_cases,
            "source_bank_optimal_cases": source_run["optimal_cases"],
            "source_bank_entries": len(source_tokens),
            "pruned_bank_optimal_cases": pruned_run["optimal_cases"],
            "pruned_bank_terminal_cases": pruned_run["terminal_cases"],
            "pruned_bank_entries": len(tokens),
            "pruned_candidate_count_max": pruned_run["candidate_count_max"],
            "pruned_gap_ticks": pruned_run["gap_ticks"],
            "leave_one_out_removable_entries": sum(
                row["removable"] for row in leave_one_out
            ),
        },
        "runtime_contract_audit": runtime_audit,
        "local_lowering_complexity": local_lowering_audit,
        "start_policy_ablation": {
            policy_name: {
                "terminal_cases": run["terminal_cases"],
                "candidate_dead_end_cases": run["candidate_dead_end_cases"],
                "optimal_cases": run["optimal_cases"],
                "gap_ticks": run["gap_ticks"],
                "candidate_count_max": run["candidate_count_max"],
            }
            for policy_name, run in sorted(start_policy_runs.items())
        },
        "leave_one_out": leave_one_out,
        "pruned_run": _compact_closed_loop_run(pruned_run),
        "runtime_s": time.perf_counter() - started,
    }
    _atomic_json(output_report_path, report)
    return report


def _strict_boundary_state_summary(state: reference.BeamState) -> dict:
    components = reference.state_lower_bound_components(
        state.c2, state.c3, state.remaining
    )
    return {
        "remaining": [[int(eid), int(ntok)] for eid, ntok in state.remaining],
        "remaining_count": len(state.remaining),
        "c2_task_end_ticks": _ticks_text(state.c2.task_end),
        "c3_task_end_ticks": _ticks_text(state.c3.task_end),
        "g_ticks": _ticks_text(state.g_score),
        "f_ticks": _ticks_text(state.f_score),
        "cluster_work_ticks": _ticks_text(state.cluster_work_cc),
        "lower_bound_components_ticks": {
            key: _ticks_text(value)
            for key, value in components.items()
            if key != "mandatory_dma_bytes"
        },
        "mandatory_dma_bytes": int(components["mandatory_dma_bytes"]),
    }


def diagnose_strict_scorer_boundaries(
    proof_path: Path,
    token_bank_path: Path,
    output_path: Path,
    *,
    scorer: str,
    sync_tiebreak: str,
    start_policy: str,
    time_limit_s: float,
    max_expansions: int,
) -> dict:
    """Locate the first policy transition that destroys target reachability.

    Reachability is monotone along a committed trajectory, so failed cases use
    binary search over their saved states.  The last reachable query also
    supplies one exact recovering action from the fixed candidate graph.  The
    target is used only by this offline diagnosis, never by the policy scorer.
    """
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    tokens = load_explicit_token_bank(token_bank_path)
    started = time.perf_counter()
    rows = []
    for case in proof["cases"]:
        target = _target_cc(case)
        states, actions = _strict_policy_trajectory(
            case,
            tokens,
            scorer=scorer,
            sync_tiebreak=sync_tiebreak,
            start_policy=start_policy,
        )
        final = states[-1]
        base = {
            "name": case["name"],
            "optimal_ticks": _ticks_text(target),
            "policy_ticks": _ticks_text(final.g_score),
            "gap_ticks": _ticks_text(final.g_score - target),
            "rounds": len(actions),
        }
        if final.g_score == target:
            rows.append({**base, "status": "policy_optimal"})
            continue

        query_cache: dict[int, dict] = {}

        def query(index: int) -> dict:
            if index not in query_cache:
                query_cache[index] = run_direct_token_target_search(
                    case["counts"],
                    target,
                    tokens,
                    start_policy=start_policy,
                    time_limit_s=time_limit_s,
                    max_expansions=max_expansions,
                    initial_state=states[index],
                )
            return query_cache[index]

        lo = 0
        hi = len(states) - 1
        unresolved = None
        while hi - lo > 1:
            mid = (lo + hi) // 2
            result = query(mid)
            if result["feasible"]:
                lo = mid
            elif result["exhaustive"]:
                hi = mid
            else:
                unresolved = {"state_index": mid, **result}
                break
        if unresolved is not None:
            rows.append(
                {
                    **base,
                    "status": "boundary_unresolved",
                    "unresolved_query": {
                        key: value
                        for key, value in unresolved.items()
                        if key != "actions"
                    },
                }
            )
            continue

        reachable = query(lo)
        unreachable = query(hi)
        if not reachable["feasible"] or not unreachable["exhaustive"]:
            raise AssertionError("binary boundary reachability invariant failed")
        prefix_actions = len(states[lo].history)
        if prefix_actions >= len(reachable["actions"]):
            raise AssertionError("reachable suffix did not contain a next action")
        recovery_action = deserialize_action(reachable["actions"][prefix_actions])
        selected_action = actions[lo]
        recovery_child = reference.apply_action(states[lo], recovery_action)
        selected_child = reference.apply_action(states[lo], selected_action)
        scalar_scorer = _practical_scalar_scorer(scorer)
        recovery_score = practical_probe_score(
            states[lo],
            recovery_action,
            recovery_child,
            scorer=scalar_scorer,
            sync_tiebreak=sync_tiebreak,
        )
        selected_score = practical_probe_score(
            states[lo],
            selected_action,
            selected_child,
            scorer=scalar_scorer,
            sync_tiebreak=sync_tiebreak,
        )
        rows.append(
            {
                **base,
                "status": "first_irrecoverable_transition_found",
                "last_reachable_state_index": lo,
                "first_unreachable_state_index": hi,
                "decision_mode": _explicit_mode(states[lo]),
                "state": _strict_boundary_state_summary(states[lo]),
                "selected_action": serialize_action(selected_action),
                "selected_score": [str(value) for value in selected_score],
                "recovering_action": serialize_action(recovery_action),
                "recovering_score": [str(value) for value in recovery_score],
                "selected_token": _serialize_explicit_token(
                    _explicit_candidate_token(states[lo], selected_action), 0
                ),
                "recovering_token": _serialize_explicit_token(
                    _explicit_candidate_token(states[lo], recovery_action), 0
                ),
                "binary_queries": [
                    {
                        "state_index": index,
                        **{
                            key: value
                            for key, value in result.items()
                            if key != "actions"
                        },
                    }
                    for index, result in sorted(query_cache.items())
                ],
            }
        )
    status_counts = Counter(row["status"] for row in rows)
    payload = {
        "schema": "olmoe-strict-scorer-boundary-v1",
        "complete": status_counts["boundary_unresolved"] == 0,
        "target_used_by_policy": False,
        "manifest": {
            "proof": str(proof_path.resolve()),
            "proof_sha256": _sha256(proof_path),
            "token_bank": str(token_bank_path.resolve()),
            "token_bank_sha256": _sha256(token_bank_path),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "scorer": scorer,
            "sync_tiebreak": sync_tiebreak,
            "start_policy": start_policy,
            "time_limit_s_per_query": time_limit_s,
            "max_expansions_per_query": max_expansions,
        },
        "summary": {
            "cases": len(rows),
            "policy_optimal": status_counts["policy_optimal"],
            "boundaries_found": status_counts[
                "first_irrecoverable_transition_found"
            ],
            "boundary_unresolved": status_counts["boundary_unresolved"],
            "boundary_modes": dict(
                sorted(
                    Counter(
                        row["decision_mode"]
                        for row in rows
                        if row["status"]
                        == "first_irrecoverable_transition_found"
                    ).items()
                )
            ),
        },
        "runtime_s": time.perf_counter() - started,
        "cases": rows,
    }
    _atomic_json(output_path, payload)
    return payload


def verify_rtl_base_equivalence(cases: Iterable[dict]) -> dict:
    """Compare the explicit ``rtl_base`` ROM against the RTL-validated mirror."""
    mismatches = []
    checked = 0
    for case in cases:
        token_dist = {
            eid: int(ntok)
            for eid, ntok in enumerate(case["counts"])
            if int(ntok) > 0
        }
        expected = rtl.adaptive_prefetch_schedule(token_dist)
        observed, _trace, _stats = run_greedy_bank(token_dist, RTL_BASE)
        checked += 1
        if expected != observed:
            mismatches.append(
                {
                    "name": case["name"],
                    "rtl_ticks": _ticks_text(expected),
                    "bank_ticks": _ticks_text(observed),
                }
            )
    return {
        "cases": checked,
        "equivalent_cases": checked - len(mismatches),
        "mismatches": mismatches,
    }


def _summarize(rows: list[dict], bank_name: str) -> dict:
    gaps = [Fraction(row[bank_name]["gap_ticks"]) for row in rows]
    counts = [row[bank_name]["candidate_count_max"] for row in rows]
    return {
        "cases": len(rows),
        "optimal_cases": sum(gap == 0 for gap in gaps),
        "gap_ticks": {
            "sum": str(sum(gaps)),
            "mean": float(sum(gaps) / len(gaps)),
            "p50": float(statistics.median(gaps)),
            "max": str(max(gaps)),
        },
        "candidate_count_max": max(counts),
        "candidate_count_mean_of_case_max": statistics.mean(counts),
    }


def evaluate(
    proof_path: Path,
    output_path: Path,
    window_audit_path: Path,
    *,
    explicit_case_limit: int | None = None,
    explicit_workers: int = 1,
    closed_loop_probe: bool = False,
    closed_loop_workers: int = 1,
    closed_loop_case_limit: int | None = None,
    closed_loop_scorer: str = "full_lpt",
    closed_loop_sync_tiebreak: str = "hot_cold",
    closed_loop_start_policy: str = "all",
    closed_loop_safety_policy: str = "all",
    closed_loop_min_token_uses: int = 1,
    closed_loop_token_bank: Path | None = None,
    closed_loop_direct_generator: bool = False,
    closed_loop_strict_token_bank: bool = False,
) -> dict:
    proof = json.loads(proof_path.read_text())
    cases = proof["cases"]
    equivalence = verify_rtl_base_equivalence(cases)
    if equivalence["mismatches"]:
        raise RuntimeError(
            f"rtl_base is not equivalent to current mirror: {equivalence['mismatches'][:3]}"
        )
    rows = []
    started = time.perf_counter()
    for case in cases:
        token_dist = {
            eid: int(ntok)
            for eid, ntok in enumerate(case["counts"])
            if int(ntok) > 0
        }
        target = _target_cc(case)
        row = {
            "name": case["name"],
            "counts": case["counts"],
            "optimal_ticks": _ticks_text(target),
        }
        for bank in BANKS.values():
            makespan, trace, stats = run_greedy_bank(token_dist, bank)
            row[bank.name] = {
                "makespan_ticks": _ticks_text(makespan),
                "gap_ticks": _ticks_text(makespan - target),
                **stats,
                "selected_tokens": [step.tag for step in trace],
            }
        rows.append(row)
    explicit_union = audit_explicit_certificate_union(
        cases,
        window_audit_path,
        case_limit=explicit_case_limit,
        workers=explicit_workers,
    )
    practical_closed_loop = None
    if closed_loop_probe:
        if closed_loop_token_bank is not None:
            explicit_tokens = load_explicit_token_bank(closed_loop_token_bank)
        else:
            explicit_tokens, _histories, token_uses = _extract_explicit_certificate_union(
                cases, window_audit_path
            )
            # Standalone PREFETCH has already been removed from the practical
            # closed-loop bank.  Filter it before materialization as well so its
            # raw reference variants do not inflate runtime.  Frequency pruning is
            # an explicit Pareto experiment, not evidence that a rare token is
            # semantically redundant.
            explicit_tokens = tuple(
                token
                for token in explicit_tokens
                if token.logical.family != "PREFETCH"
                and token_uses[token] >= closed_loop_min_token_uses
            )
        practical_closed_loop = evaluate_practical_closed_loop(
            cases,
            explicit_tokens,
            workers=closed_loop_workers,
            case_limit=closed_loop_case_limit,
            scorer=closed_loop_scorer,
            sync_tiebreak=closed_loop_sync_tiebreak,
            start_policy=closed_loop_start_policy,
            safety_policy=closed_loop_safety_policy,
            min_token_uses=closed_loop_min_token_uses,
            direct_generator=closed_loop_direct_generator,
            strict_token_bank=closed_loop_strict_token_bank,
        )
    report = {
        "schema": "olmoe-fixed-token-banks-v2",
        "complete": bool(
            explicit_union["audit_complete"]
            and explicit_union["all_audited_paths_covered"]
        ),
        "interpretation": {
            "candidate_contract": (
                "Each bank is an explicit state-relative token ROM evaluated "
                "sequentially by one lane; there is no K-wide candidate bus."
            ),
            "scalar_probe_scope": (
                "These results include the current RTL scorer. They do not yet "
                "model explicit DMA-lane identity and are retained only as the "
                "current-RTL equivalence baseline."
            ),
            "explicit_union_scope": (
                "The certificate-union bank is a constructive 65-case candidate "
                "upper bound, not the final RTL bank. Every legal start is a "
                "separately counted candidate and no score is used during generation."
            ),
        },
        "manifest": {
            "proof": str(proof_path.relative_to(HERE.parent)),
            "proof_sha256": _sha256(proof_path),
            "window_audit": str(window_audit_path.relative_to(HERE.parent)),
            "window_audit_sha256": _sha256(window_audit_path),
            "script": str(Path(__file__).resolve().relative_to(HERE.parent)),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "rtl_mirror": str((HERE / "scheduler_rtl_adaptive_prefetch_policy.py").relative_to(HERE.parent)),
            "rtl_mirror_sha256": _sha256(HERE / "scheduler_rtl_adaptive_prefetch_policy.py"),
            "closed_loop_token_bank": (
                str(closed_loop_token_bank)
                if closed_loop_token_bank is not None
                else None
            ),
            "closed_loop_token_bank_sha256": (
                _sha256(closed_loop_token_bank)
                if closed_loop_token_bank is not None
                else None
            ),
        },
        "banks": {
            bank.name: {
                "window": list(bank.window),
                "sync_pairs": [list(item) for item in bank.sync_pairs],
                "sync_splits": [list(item) for item in bank.sync_splits],
                "one_idle_ranks": list(bank.one_idle_ranks),
                "sync_single_ranks": list(bank.sync_single_ranks),
                "pair_profiles": list(bank.pair_profiles),
            }
            for bank in BANKS.values()
        },
        "rtl_base_equivalence": equivalence,
        "summary": {
            bank.name: _summarize(rows, bank.name) for bank in BANKS.values()
        },
        "explicit_certificate_union": explicit_union,
        "practical_closed_loop_probe": practical_closed_loop,
        "runtime_s": time.perf_counter() - started,
        "cases": rows,
    }
    _atomic_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", type=Path, default=DEFAULT_PROOF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window-audit", type=Path, default=DEFAULT_WINDOW_AUDIT)
    parser.add_argument(
        "--direct-compression-output",
        type=Path,
        default=None,
        help=(
            "build a direct-lowering certificate set-cover bank and exit; "
            "the formal run omits --direct-compression-case-limit"
        ),
    )
    parser.add_argument("--direct-compression-workers", type=int, default=1)
    parser.add_argument(
        "--direct-compression-case-limit",
        type=int,
        default=None,
        help="focused CLI regression only; omit for the authoritative 65-case bank",
    )
    parser.add_argument(
        "--direct-compression-start-policy",
        choices=(
            "all",
            "earliest_start",
            "earliest_finish",
            "bounded_release",
            "latest_start",
        ),
        default="all",
    )
    parser.add_argument(
        "--direct-candidate-certify-bank",
        type=Path,
        default=None,
        help="fixed token-bank JSON to certify by replay plus exact fallback",
    )
    parser.add_argument(
        "--direct-candidate-certify-output",
        type=Path,
        default=None,
    )
    parser.add_argument("--direct-candidate-workers", type=int, default=1)
    parser.add_argument(
        "--direct-candidate-start-policy",
        choices=(
            "all",
            "earliest_start",
            "earliest_finish",
            "bounded_release",
            "latest_start",
        ),
        default="earliest_finish",
    )
    parser.add_argument("--direct-candidate-time-limit-s", type=float, default=120.0)
    parser.add_argument("--direct-candidate-max-expansions", type=int, default=100000)
    parser.add_argument("--derive-budgeted-bank-source", type=Path, default=None)
    parser.add_argument("--derive-budgeted-bank-output", type=Path, default=None)
    parser.add_argument(
        "--derive-budgeted-bank-k", type=int, choices=(16, 24, 32), default=None
    )
    parser.add_argument(
        "--derive-budgeted-bank-policy",
        choices=("frequency", "frequency_hot_cold", "family_quota"),
        default="family_quota",
    )
    parser.add_argument("--distill-used-bank-certificate", type=Path, default=None)
    parser.add_argument("--distill-used-bank-source", type=Path, default=None)
    parser.add_argument("--distill-used-bank-output", type=Path, default=None)
    parser.add_argument("--certify-pruned-bank-source", type=Path, default=None)
    parser.add_argument("--certify-pruned-bank-output", type=Path, default=None)
    parser.add_argument("--certify-pruned-report-output", type=Path, default=None)
    parser.add_argument(
        "--certify-pruned-keep-indices",
        type=str,
        default=None,
        help="comma-separated, sorted source-ROM indices for causal certification",
    )
    parser.add_argument(
        "--certify-pruned-window",
        type=int,
        nargs=2,
        metavar=("TOP", "BOTTOM"),
        default=EXPLICIT_WINDOW,
        help="runtime descriptor window used to lower and score the ROM",
    )
    parser.add_argument(
        "--certify-pruned-scorer",
        choices=(BOUNDED_PAIRWISE_SCORER, HEAD5_HIST4_PAIRWISE_SCORER),
        default=BOUNDED_PAIRWISE_SCORER,
    )
    parser.add_argument(
        "--certify-pruned-selector-map",
        type=str,
        default="",
        help="comma-separated selector projection, for example B1:B0",
    )
    parser.add_argument(
        "--certify-pruned-final-policy",
        action="store_true",
        help="freeze final_bank/final_policy only if every certificate gate passes",
    )
    parser.add_argument(
        "--explicit-case-limit",
        type=int,
        default=None,
        help="focused development audit; omit for the authoritative 65-case run",
    )
    parser.add_argument(
        "--explicit-workers",
        type=int,
        default=1,
        help="parallel witness audits; each worker evaluates independent cases",
    )
    parser.add_argument(
        "--closed-loop-probe",
        action="store_true",
        help="run the consuming-candidate full-tail-LPT closed-loop probe",
    )
    parser.add_argument("--closed-loop-workers", type=int, default=1)
    parser.add_argument("--closed-loop-case-limit", type=int, default=None)
    parser.add_argument(
        "--closed-loop-scorer",
        choices=(
            "full_lpt",
            "full_lpt_load",
            "full_cache",
            "full_dual",
            "pathmax",
            "head4_8_min",
            "head4_8_max",
            "head4_8_sum",
            "head4_8_at8",
            "head4_8_at16",
            "head4_8_at24",
            "head4_8_at32",
            "head4_aggregate",
            "head6_aggregate",
            "head8_aggregate",
            "lb_f_lpt",
            "lb_f_lpt_compute",
            "lb_f_lpt_compute_late",
            "lb_f_lpt_compute_early",
            "lb_f_lpt_compute_imbalance",
            "lb_f_lpt_compute_dma",
            "lb_f_lpt_compute_release",
            "lb_certified_lex",
            ONE_PROGRESS_PAIRWISE_SCORER,
            SYNC_HOT_PAIRWISE_SCORER,
            MIN2_PLATEAU_PAIRWISE_SCORER,
            EXPANDED_PLATEAU_PAIRWISE_SCORER,
            BOUNDED_PAIRWISE_SCORER,
            HEAD5_HIST4_PAIRWISE_SCORER,
        ),
        default="full_lpt",
    )
    parser.add_argument(
        "--closed-loop-min-token-uses",
        type=int,
        default=1,
        help=(
            "Pareto ablation: retain certificate composite tokens observed at "
            "least this many times; safety SINGLE progress remains available"
        ),
    )
    parser.add_argument(
        "--closed-loop-sync-tiebreak",
        choices=("hot_cold", "small_pair", "earliest_commit", "equal_pair"),
        default="hot_cold",
        help="bounded SYNC comparator-field ablation after continuation ties",
    )
    parser.add_argument(
        "--closed-loop-start-policy",
        choices=(
            "all",
            "earliest_start",
            "earliest_finish",
            "bounded_release",
            "latest_start",
        ),
        default="all",
        help="local per-composite start selector; all counts every legal start",
    )
    parser.add_argument(
        "--closed-loop-safety-policy",
        choices=(
            "all",
            "earliest_start_per_eid",
            "earliest_finish_per_eid",
            "earliest_finish_global",
        ),
        default="all",
        help="bounded physical selector for the unconditional progress fallback",
    )
    parser.add_argument(
        "--closed-loop-token-bank",
        type=Path,
        default=None,
        help="versioned JSON token bank; bypass certificate-frequency extraction",
    )
    parser.add_argument(
        "--closed-loop-direct-generator",
        action="store_true",
        help="lower the fixed token bank directly without reference action generation",
    )
    parser.add_argument(
        "--closed-loop-strict-token-bank",
        action="store_true",
        help="disable all fallback candidates and score exactly the supplied fixed bank",
    )
    parser.add_argument(
        "--strict-closed-loop-output",
        type=Path,
        default=None,
        help=(
            "run only the strict fixed-bank scorer evaluation and exit; "
            "requires --closed-loop-token-bank"
        ),
    )
    parser.add_argument(
        "--strict-closed-loop-matrix",
        action="store_true",
        help=(
            "with --strict-closed-loop-output, compare the frozen scorer and "
            "SYNC tie-break matrix on the identical fixed candidate bank"
        ),
    )
    parser.add_argument(
        "--strict-boundary-output",
        type=Path,
        default=None,
        help=(
            "locate the first exact-target-irrecoverable decision of the "
            "selected strict fixed-bank scorer and exit"
        ),
    )
    args = parser.parse_args()
    if args.explicit_case_limit is not None and args.explicit_case_limit <= 0:
        raise SystemExit("--explicit-case-limit must be positive")
    if args.explicit_workers <= 0:
        raise SystemExit("--explicit-workers must be positive")
    if args.direct_compression_workers <= 0:
        raise SystemExit("--direct-compression-workers must be positive")
    if args.direct_candidate_workers <= 0:
        raise SystemExit("--direct-candidate-workers must be positive")
    if args.direct_candidate_time_limit_s <= 0:
        raise SystemExit("--direct-candidate-time-limit-s must be positive")
    if args.direct_candidate_max_expansions <= 0:
        raise SystemExit("--direct-candidate-max-expansions must be positive")
    if (
        args.direct_compression_case_limit is not None
        and args.direct_compression_case_limit <= 0
    ):
        raise SystemExit("--direct-compression-case-limit must be positive")
    if args.closed_loop_workers <= 0:
        raise SystemExit("--closed-loop-workers must be positive")
    if args.closed_loop_case_limit is not None and args.closed_loop_case_limit <= 0:
        raise SystemExit("--closed-loop-case-limit must be positive")
    if args.closed_loop_min_token_uses <= 0:
        raise SystemExit("--closed-loop-min-token-uses must be positive")
    if args.strict_closed_loop_matrix and args.strict_closed_loop_output is None:
        raise SystemExit(
            "--strict-closed-loop-matrix requires --strict-closed-loop-output"
        )
    if (
        args.strict_closed_loop_output is not None
        and args.closed_loop_token_bank is None
    ):
        raise SystemExit(
            "--strict-closed-loop-output requires --closed-loop-token-bank"
        )
    if args.strict_boundary_output is not None and args.closed_loop_token_bank is None:
        raise SystemExit(
            "--strict-boundary-output requires --closed-loop-token-bank"
        )
    if (
        args.strict_boundary_output is not None
        and args.strict_closed_loop_output is not None
    ):
        raise SystemExit(
            "--strict-boundary-output and --strict-closed-loop-output are exclusive"
        )
    if args.direct_compression_output is not None:
        payload = build_compressed_direct_token_bank(
            args.proof.resolve(),
            args.window_audit.resolve(),
            args.direct_compression_output.resolve(),
            workers=args.direct_compression_workers,
            case_limit=args.direct_compression_case_limit,
            start_policy=args.direct_compression_start_policy,
        )
        print(json.dumps(payload["compression"], indent=2))
        print(
            json.dumps(
                {
                    key: value
                    for key, value in payload["audit"].items()
                    if key != "rows"
                },
                indent=2,
            )
        )
        print(f"wrote {args.direct_compression_output.resolve()}")
        return 0
    if (
        args.direct_candidate_certify_bank is None
    ) != (
        args.direct_candidate_certify_output is None
    ):
        raise SystemExit(
            "--direct-candidate-certify-bank and "
            "--direct-candidate-certify-output must be provided together"
        )
    if args.direct_candidate_certify_bank is not None:
        payload = certify_direct_token_bank(
            args.proof.resolve(),
            args.window_audit.resolve(),
            args.direct_candidate_certify_bank.resolve(),
            args.direct_candidate_certify_output.resolve(),
            workers=args.direct_candidate_workers,
            start_policy=args.direct_candidate_start_policy,
            time_limit_s=args.direct_candidate_time_limit_s,
            max_expansions=args.direct_candidate_max_expansions,
        )
        print(json.dumps(payload["summary"], indent=2))
        print(f"wrote {args.direct_candidate_certify_output.resolve()}")
        return 0
    derive_values = (
        args.derive_budgeted_bank_source,
        args.derive_budgeted_bank_output,
        args.derive_budgeted_bank_k,
    )
    if any(value is not None for value in derive_values):
        if not all(value is not None for value in derive_values):
            raise SystemExit(
                "--derive-budgeted-bank-source/output/k must be provided together"
            )
        payload = derive_budgeted_direct_token_bank(
            args.derive_budgeted_bank_source.resolve(),
            args.derive_budgeted_bank_output.resolve(),
            candidate_budget=args.derive_budgeted_bank_k,
            selection_policy=args.derive_budgeted_bank_policy,
        )
        print(json.dumps(payload["mode_summary"], indent=2))
        print(f"wrote {args.derive_budgeted_bank_output.resolve()}")
        return 0
    distill_values = (
        args.distill_used_bank_certificate,
        args.distill_used_bank_source,
        args.distill_used_bank_output,
    )
    if any(value is not None for value in distill_values):
        if not all(value is not None for value in distill_values):
            raise SystemExit(
                "--distill-used-bank-certificate/source/output must be provided together"
            )
        payload = distill_used_direct_token_bank(
            args.proof.resolve(),
            args.window_audit.resolve(),
            args.distill_used_bank_certificate.resolve(),
            args.distill_used_bank_source.resolve(),
            args.distill_used_bank_output.resolve(),
        )
        print(json.dumps(payload["summary"], indent=2))
        print(f"wrote {args.distill_used_bank_output.resolve()}")
        return 0
    pruned_values = (
        args.certify_pruned_bank_source,
        args.certify_pruned_bank_output,
        args.certify_pruned_report_output,
        args.certify_pruned_keep_indices,
    )
    if any(value is not None for value in pruned_values):
        if not all(value is not None for value in pruned_values):
            raise SystemExit(
                "--certify-pruned-bank-source/output, "
                "--certify-pruned-report-output and "
                "--certify-pruned-keep-indices must be provided together"
            )
        try:
            keep_indices = tuple(
                int(value)
                for value in args.certify_pruned_keep_indices.split(",")
                if value.strip()
            )
        except ValueError as exc:
            raise SystemExit("invalid --certify-pruned-keep-indices") from exc
        try:
            selector_map = dict(
                item.split(":", 1)
                for item in args.certify_pruned_selector_map.split(",")
                if item
            )
        except ValueError as exc:
            raise SystemExit(
                "invalid --certify-pruned-selector-map; expected OLD:NEW"
            ) from exc
        certify_window = tuple(args.certify_pruned_window)
        if certify_window[0] <= 0 or certify_window[1] < 0:
            raise SystemExit("--certify-pruned-window requires TOP>0, BOTTOM>=0")
        payload = certify_pruned_closed_loop_bank(
            args.proof.resolve(),
            args.certify_pruned_bank_source.resolve(),
            args.certify_pruned_bank_output.resolve(),
            args.certify_pruned_report_output.resolve(),
            keep_indices=keep_indices,
            workers=args.closed_loop_workers,
            scorer=args.certify_pruned_scorer,
            sync_tiebreak=args.closed_loop_sync_tiebreak,
            start_policy=args.closed_loop_start_policy,
            window=certify_window,
            selector_map=selector_map,
            final_policy=args.certify_pruned_final_policy,
        )
        print(json.dumps(payload["summary"], indent=2))
        print(f"wrote {args.certify_pruned_bank_output.resolve()}")
        print(f"wrote {args.certify_pruned_report_output.resolve()}")
        return 0
    if args.strict_closed_loop_output is not None:
        payload = evaluate_strict_closed_loop_bank(
            args.proof.resolve(),
            args.closed_loop_token_bank.resolve(),
            args.strict_closed_loop_output.resolve(),
            workers=args.closed_loop_workers,
            case_limit=args.closed_loop_case_limit,
            scorer=args.closed_loop_scorer,
            sync_tiebreak=args.closed_loop_sync_tiebreak,
            start_policy=args.closed_loop_start_policy,
            matrix=args.strict_closed_loop_matrix,
        )
        print(json.dumps(payload["best_configuration"], indent=2))
        print(f"wrote {args.strict_closed_loop_output.resolve()}")
        return 0
    if args.strict_boundary_output is not None:
        payload = diagnose_strict_scorer_boundaries(
            args.proof.resolve(),
            args.closed_loop_token_bank.resolve(),
            args.strict_boundary_output.resolve(),
            scorer=args.closed_loop_scorer,
            sync_tiebreak=args.closed_loop_sync_tiebreak,
            start_policy=args.closed_loop_start_policy,
            time_limit_s=args.direct_candidate_time_limit_s,
            max_expansions=args.direct_candidate_max_expansions,
        )
        print(json.dumps(payload["summary"], indent=2))
        print(f"wrote {args.strict_boundary_output.resolve()}")
        return 0
    report = evaluate(
        args.proof.resolve(),
        args.output.resolve(),
        args.window_audit.resolve(),
        explicit_case_limit=args.explicit_case_limit,
        explicit_workers=args.explicit_workers,
        closed_loop_probe=args.closed_loop_probe,
        closed_loop_workers=args.closed_loop_workers,
        closed_loop_case_limit=args.closed_loop_case_limit,
        closed_loop_scorer=args.closed_loop_scorer,
        closed_loop_sync_tiebreak=args.closed_loop_sync_tiebreak,
        closed_loop_start_policy=args.closed_loop_start_policy,
        closed_loop_safety_policy=args.closed_loop_safety_policy,
        closed_loop_min_token_uses=args.closed_loop_min_token_uses,
        closed_loop_token_bank=(
            args.closed_loop_token_bank.resolve()
            if args.closed_loop_token_bank is not None
            else None
        ),
        closed_loop_direct_generator=args.closed_loop_direct_generator,
        closed_loop_strict_token_bank=args.closed_loop_strict_token_bank,
    )
    print(json.dumps(report["rtl_base_equivalence"], indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(
        json.dumps(
            {
                key: value
                for key, value in report["explicit_certificate_union"].items()
                if key not in {"tokens", "cases"}
            },
            indent=2,
        )
    )
    if report["practical_closed_loop_probe"] is not None:
        print(
            json.dumps(
                {
                    key: value
                    for key, value in report["practical_closed_loop_probe"].items()
                    if key != "rows"
                },
                indent=2,
            )
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
