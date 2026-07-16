#!/usr/bin/env python3
"""Derive and audit the bounded scheduler policy under the locked contract.

The current command audits the P2 bounded candidate architecture on a
deterministic, stratified sample of proven discovery states.  Future-value and
scorer commands will be added to this same file after the candidate structure
passes its audit; no parallel derivation pipeline is created.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import heapq
import itertools
import json
import math
import os
from pathlib import Path
import time

from analyze_scheduler_candidates import (
    action_family,
    canonical_template,
    decision_mode,
    quality_ok,
    rank_map,
    selected_eids,
    shape_name,
)
from four_stage_scheduler import (
    BeamState,
    DmaBinding,
    FourStageScheduler,
    FourStageSnap,
    SHAPE_A,
    SHAPE_B,
    SHAPE_C,
    StageAction,
    apply_action,
    bw_feasible,
    clear_scheduler_caches,
    completion_estimate,
    gen_prefetch_actions,
    gen_stage_actions,
    _isolated_task_time_lb,
    _down_hit_for_candidate,
    _reserved_next_eid,
    _start_candidates,
    _swiglu_hit_for_candidate,
    state_lower_bound_components,
    _cache_aware_completion_estimate,
    _lpt_completion_estimate,
)
from run_four_stage_reference import deserialize_action, serialize_action


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUTS = tuple(
    ROOT / "results" / "final_reference" / f"scheduler_reference_E{e}.json"
    for e in (8, 32, 64)
)
DEFAULT_OUT = Path("/tmp/scheduler_candidate_audit.json")
DEFAULT_FUTURE_DATASET = ROOT / "results" / "policy_search" / "future_value_dataset.jsonl"
TIME_QUANTUM_CC = 11_264
R8_RANKER_TIME_UNIT_CC = TIME_QUANTUM_CC // 2
TASK_PROFILES = {
    ("A", "B"), ("B", "B"), ("C", "C"),
    # Canonical conditional forms when exactly one stage is cache-ready.
    ("C", "B"), ("A", "C"), ("B", "C"),
}
FAMILIES = ("SINGLE", "PAIR", "SPLIT", "PREFETCH")
PROFILE_SHAPES = ((SHAPE_A, SHAPE_B), (SHAPE_B, SHAPE_B), (SHAPE_C, SHAPE_C))
# A/B has the widest constant-duration token plateau: tokens 1..4 share the
# same S1+S3 foreground completion.  Checking this fixed neighborhood around a
# monotone crossing preserves the scan's deterministic tie-break exactly.
SPLIT_CROSSING_RADIUS = 4


@dataclass(frozen=True)
class SampledState:
    case_key: str
    e_total: int
    step: int
    state: BeamState
    reference_action: StageAction
    reference_makespan: int


@dataclass(frozen=True)
class RankingState:
    """One state used to train the deployable R8 candidate ranker."""

    state_id: str
    case_key: str
    e_total: int
    step: int
    source: str
    state: BeamState
    reference_makespan: int
    trajectory_regret_cc: int


@dataclass(frozen=True)
class DirectMacro:
    """A bounded macro slot; no physical StageAction has been enumerated yet."""

    family: str
    key: tuple
    payload: tuple


def stable_u64(*parts: object) -> int:
    text = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def _cycles_to_quanta(value: int) -> int:
    return (max(0, int(value)) + TIME_QUANTUM_CC - 1) // TIME_QUANTUM_CC


def future_features(state: BeamState) -> tuple[int, dict[str, int], dict[str, int]]:
    """Return pathmax base, RTL-oriented integer features and LB components."""
    components = state_lower_bound_components(state.c2, state.c3, state.remaining)
    base = max(state.f_score, components["combined_cc"])
    tokens = sorted((ntok for _, ntok in state.remaining), reverse=True)
    earliest = min(state.c2.task_end, state.c3.task_end)
    latest = max(state.c2.task_end, state.c3.task_end)

    def ready_sets(snap: FourStageSnap) -> tuple[set[int], set[int]]:
        s1 = set()
        full = set()
        for eid, _ in state.remaining:
            if _swiglu_hit_for_candidate(eid, snap, snap.task_end):
                s1.add(eid)
            if _down_hit_for_candidate(eid, snap, snap.task_end):
                full.add(eid)
        return s1, full

    c2_s1_set, c2_full_set = ready_sets(state.c2)
    c3_s1_set, c3_full_set = ready_sets(state.c3)
    c2_s1, c2_full = len(c2_s1_set), len(c2_full_set)
    c3_s1, c3_full = len(c3_s1_set), len(c3_full_set)
    token_by_eid = dict(state.remaining)
    s1_union = c2_s1_set | c3_s1_set
    full_union = c2_full_set | c3_full_set
    dma_tail = max(
        state.c2.dma1_end,
        state.c2.dma3_end,
        state.c2.s2pf_end,
        state.c2.pf_end,
        state.c3.dma1_end,
        state.c3.dma3_end,
        state.c3.s2pf_end,
        state.c3.pf_end,
        earliest,
    )
    features = {
        "release_gap_q": _cycles_to_quanta(latest - earliest),
        "dma_busy_tail_q": _cycles_to_quanta(dma_tail - earliest),
        "compute_slack_q": _cycles_to_quanta(base - components["compute_cc"]),
        "release_chain_slack_q": _cycles_to_quanta(
            base - components["release_expert_chain_cc"]
        ),
        "critical_chain_slack_q": _cycles_to_quanta(
            base - components["critical_chain_cc"]
        ),
        "dma_slack_q": _cycles_to_quanta(base - components["dma_capacity_cc"]),
        "dma_work_q": _cycles_to_quanta(
            components["dma_capacity_cc"] - components["dma_release_cc"]
        ),
        "pathmax_gap_q": _cycles_to_quanta(base - components["combined_cc"]),
        "remaining_count": len(tokens),
        "remaining_tokens": sum(tokens),
        "largest_tokens": tokens[0] if tokens else 0,
        "second_tokens": tokens[1] if len(tokens) > 1 else 0,
        "odd_token_experts": sum(ntok & 1 for ntok in tokens),
        "small_le2_experts": sum(ntok <= 2 for ntok in tokens),
        "small_le4_experts": sum(ntok <= 4 for ntok in tokens),
        "c2_s1_ready": c2_s1,
        "c3_s1_ready": c3_s1,
        "c2_full_ready": c2_full,
        "c3_full_ready": c3_full,
        "unique_s1_ready": len(s1_union),
        "duplicate_s1_ready": len(c2_s1_set & c3_s1_set),
        "unique_full_ready": len(full_union),
        "duplicate_full_ready": len(c2_full_set & c3_full_set),
        "s1_ready_tokens": sum(token_by_eid[eid] for eid in s1_union),
        "full_ready_tokens": sum(token_by_eid[eid] for eid in full_union),
        "reserved_prefetches": int(_reserved_next_eid(state.c2) >= 0)
        + int(_reserved_next_eid(state.c3) >= 0),
        "c2_releases_first": int(state.c2.task_end < state.c3.task_end),
        "c3_releases_first": int(state.c3.task_end < state.c2.task_end),
    }
    return base, features, components


R8_RANKER_PROFILES = {
    "rtl-base": (),
    "rtl-timing": ("load_gap", "dma_tail", "pathmax_gap"),
    "rtl-timing-cache": (
        "load_gap",
        "dma_tail",
        "pathmax_gap",
        "ready_work",
        "duplicate_ready_work",
    ),
    "rtl-full": (
        "load_gap",
        "dma_tail",
        "pathmax_gap",
        "ready_work",
        "duplicate_ready_work",
        "odd_count",
        "small_count",
        "rank_sum",
        "family_pair",
        "family_split",
        "family_prefetch",
    ),
}
R8_RANKER_WEIGHT_DOMAIN = (-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16)
R8_RANKER_MODES = ("BOTH_IDLE", "ONE_IDLE", "LAST_EXPERT")


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _time_q(value: int) -> int:
    if value < 0:
        # Inactive DMA/prefetch endpoints use -1 and must lose the later max().
        return int(value)
    if value % R8_RANKER_TIME_UNIT_CC != 0:
        raise ValueError(f"time is not half-quantum aligned: {value}")
    return value // R8_RANKER_TIME_UNIT_CC


def _expert_work_q(ntok: int) -> int:
    """Isolated compute work in half-Tq units; shift/add only in RTL."""
    blocks = (int(ntok) + 1) >> 1
    return (blocks << 1) + (blocks << 2)


def r8_ranker_features(
    parent: BeamState, action: StageAction, child: BeamState
) -> tuple[int, dict[str, int]]:
    """Return an RTL-maintainable base and integer candidate features.

    The function deliberately avoids the full-list LPT scan.  Runtime state is
    limited to the ordered top8 window plus aggregates that can be initialized
    once and decremented when an action commits.
    """
    load2 = _time_q(child.c2.task_end)
    load3 = _time_q(child.c3.task_end)
    earliest = min(load2, load3)
    latest = max(load2, load3)
    f_score_q = _time_q(child.f_score)
    remaining = list(child.remaining)
    work_q = sum(_expert_work_q(ntok) for _, ntok in remaining)
    largest_work_q = _expert_work_q(remaining[0][1]) if remaining else 0

    # Lower-bound-like work base using only two release times, total remaining
    # work and top0 work.  All inputs are maintained incrementally by RTL.
    average_q = _ceil_div(load2 + load3 + work_q, 2)
    single_chain_q = earliest + largest_work_q
    work_base_q = max(latest, average_q, single_chain_q)
    base_q = max(f_score_q, work_base_q)

    dma_tail_q = max(
        _time_q(child.c2.dma1_end),
        _time_q(child.c2.dma3_end),
        _time_q(child.c2.s2pf_end),
        _time_q(child.c2.pf_end),
        _time_q(child.c3.dma1_end),
        _time_q(child.c3.dma3_end),
        _time_q(child.c3.s2pf_end),
        _time_q(child.c3.pf_end),
        earliest,
    )

    token_by_eid = dict(remaining)
    ready_by_cluster = []
    for snap in (child.c2, child.c3):
        ready = set()
        eid = snap.pf_eid
        if eid in token_by_eid and (
            _swiglu_hit_for_candidate(eid, snap, snap.task_end)
            or _down_hit_for_candidate(eid, snap, snap.task_end)
        ):
            ready.add(eid)
        ready_by_cluster.append(ready)
    ready_union = ready_by_cluster[0] | ready_by_cluster[1]
    ready_duplicate = ready_by_cluster[0] & ready_by_cluster[1]

    ranks = rank_map(parent)
    selected_ranks = [ranks[eid] for eid in selected_eids(action) if eid in ranks]
    family = action_family(action)
    terms = {
        "load_gap": latest - earliest,
        "dma_tail": max(0, dma_tail_q - earliest),
        "pathmax_gap": max(0, f_score_q - work_base_q),
        "ready_work": sum(
            _expert_work_q(token_by_eid[eid]) for eid in ready_union
        ),
        "duplicate_ready_work": sum(
            _expert_work_q(token_by_eid[eid]) for eid in ready_duplicate
        ),
        "odd_count": sum(ntok & 1 for _, ntok in remaining),
        "small_count": sum(ntok <= 4 for _, ntok in remaining),
        "rank_sum": sum(selected_ranks),
        "family_pair": int(family == "PAIR"),
        "family_split": int(family == "SPLIT"),
        "family_prefetch": int(family == "PREFETCH"),
    }
    return base_q, terms


def remaining_band(count: int) -> str:
    if count <= 1:
        return "1"
    if count <= 4:
        return "2_4"
    if count <= 8:
        return "5_8"
    if count <= 16:
        return "9_16"
    return "17_PLUS"


def collect_states(
    paths: list[Path],
    *,
    source_split: str,
    per_stratum: int,
    max_states: int,
    max_cases_per_file: int,
    min_remaining: int,
    max_remaining: int,
    require_r4_miss_r8_hit: bool,
    progress_every: int,
) -> list[SampledState]:
    """Keep deterministic lowest-hash states per structural stratum."""
    reservoirs: dict[tuple, list[tuple[int, str, int, SampledState]]] = defaultdict(
        list
    )
    cases = decisions = 0
    started = time.perf_counter()
    for path in paths:
        results = json.loads(path.read_text())["results"]
        eligible_in_file = 0
        for key, item in results.items():
            if item.get("dataset_split") != source_split or not quality_ok(
                item, "proven"
            ):
                continue
            if 0 <= max_cases_per_file <= eligible_in_file:
                break
            eligible_in_file += 1
            dist = {int(eid): int(ntok) for eid, ntok in item["dist"].items()}
            scheduler = FourStageScheduler(
                dist,
                initial_cache_c2=int(item.get("initial_cache_c2", -1)),
                initial_cache_c3=int(item.get("initial_cache_c3", -1)),
            )
            state = scheduler._initial_state()
            for step, raw in enumerate(item["actions"]):
                action = deserialize_action(raw)
                remaining_count = len(state.remaining)
                pool_relation_ok = True
                if require_r4_miss_r8_hit:
                    chosen = selected_eids(action)
                    pool4 = concrete_pool(state, 4, 2)
                    pool8 = concrete_pool(state, 8, 2)
                    pool_relation_ok = (
                        all(eid in pool8 for eid in chosen)
                        and any(eid not in pool4 for eid in chosen)
                    )
                if pool_relation_ok and remaining_count >= min_remaining and (
                    max_remaining < 0 or remaining_count <= max_remaining
                ):
                    stratum = (
                        int(item["e_total"]),
                        decision_mode(state),
                        remaining_band(remaining_count),
                        action_family(action),
                    )
                    sample = SampledState(
                        case_key=f"E{item['e_total']}:{key}",
                        e_total=int(item["e_total"]),
                        step=step,
                        state=state,
                        reference_action=action,
                        reference_makespan=int(item["makespan_cc"]),
                    )
                    score = stable_u64(sample.case_key, step)
                    heap = reservoirs[stratum]
                    # Python's min-heap stores negative hashes so the largest kept
                    # hash is replaced when a lower deterministic hash arrives.
                    entry = (-score, sample.case_key, sample.step, sample)
                    if len(heap) < per_stratum:
                        heapq.heappush(heap, entry)
                    elif score < -heap[0][0]:
                        heapq.heapreplace(heap, entry)
                decisions += 1
                state = apply_action(state, action)
            cases += 1
            clear_scheduler_caches()
            if progress_every > 0 and cases % progress_every == 0:
                kept = sum(len(values) for values in reservoirs.values())
                print(
                    f"sample cases={cases} decisions={decisions} kept={kept} "
                    f"elapsed_s={time.perf_counter()-started:.1f}",
                    flush=True,
                )

    samples = [entry[3] for heap in reservoirs.values() for entry in heap]
    groups: dict[tuple, list[SampledState]] = defaultdict(list)
    for sample in samples:
        groups[
            (
                decision_mode(sample.state),
                action_family(sample.reference_action),
                sample.e_total,
            )
        ].append(sample)
    for values in groups.values():
        values.sort(key=lambda sample: stable_u64(sample.case_key, sample.step))

    # Cover the common BOTH/ONE modes first, then LAST and the rare explicit
    # prefetch states.  Round-robin over (mode, family, E) prevents one
    # remaining-count band from consuming a small audit budget.
    group_order = []
    for mode, family in (
        ("BOTH_IDLE", "PAIR"),
        ("BOTH_IDLE", "SPLIT"),
        ("BOTH_IDLE", "SINGLE"),
        ("ONE_IDLE", "SINGLE"),
        ("LAST_EXPERT", "SINGLE"),
        ("BOTH_IDLE", "PREFETCH"),
        ("ONE_IDLE", "PREFETCH"),
        ("LAST_EXPERT", "PREFETCH"),
        ("ONE_IDLE", "PAIR"),
        ("ONE_IDLE", "SPLIT"),
        ("LAST_EXPERT", "SPLIT"),
    ):
        group_order.extend((mode, family, e_total) for e_total in (8, 32, 64))
    group_order.extend(sorted(key for key in groups if key not in group_order))

    chosen = []
    depth = 0
    while max_states < 0 or len(chosen) < max_states:
        added = False
        for key in group_order:
            values = groups.get(key, ())
            if depth >= len(values):
                continue
            chosen.append(values[depth])
            added = True
            if 0 <= max_states <= len(chosen):
                return chosen
        if not added:
            break
        depth += 1
    return chosen


def collect_manifest_states(
    paths: list[Path], manifest_path: Path, source_split: str
) -> list[SampledState]:
    """Replay only state IDs already fixed by a prior audit report."""
    manifest = json.loads(manifest_path.read_text())
    ordered_ids = [str(row["state_id"]) for row in manifest["rows"]]
    wanted: dict[str, set[int]] = defaultdict(set)
    for state_id in ordered_ids:
        case_key, step_text = state_id.rsplit(":", 1)
        wanted[case_key].add(int(step_text))

    found = {}
    for path in paths:
        results = json.loads(path.read_text())["results"]
        for key, item in results.items():
            case_key = f"E{item['e_total']}:{key}"
            if case_key not in wanted:
                continue
            if item.get("dataset_split") != source_split or not quality_ok(
                item, "proven"
            ):
                raise ValueError(
                    f"manifest state is not proven {source_split}: {case_key}"
                )
            dist = {int(eid): int(ntok) for eid, ntok in item["dist"].items()}
            scheduler = FourStageScheduler(
                dist,
                initial_cache_c2=int(item.get("initial_cache_c2", -1)),
                initial_cache_c3=int(item.get("initial_cache_c3", -1)),
            )
            state = scheduler._initial_state()
            for step, raw in enumerate(item["actions"]):
                action = deserialize_action(raw)
                if step in wanted[case_key]:
                    state_id = f"{case_key}:{step}"
                    found[state_id] = SampledState(
                        case_key=case_key,
                        e_total=int(item["e_total"]),
                        step=step,
                        state=state,
                        reference_action=action,
                        reference_makespan=int(item["makespan_cc"]),
                    )
                state = apply_action(state, action)
            clear_scheduler_caches()
    missing = [state_id for state_id in ordered_ids if state_id not in found]
    if missing:
        raise ValueError(f"manifest states not found: {missing}")
    return [found[state_id] for state_id in ordered_ids]


def select_samples(args: argparse.Namespace) -> list[SampledState]:
    if args.state_ids_from is not None:
        return collect_manifest_states(
            args.inputs, args.state_ids_from, args.source_split
        )
    return collect_states(
        args.inputs,
        source_split=args.source_split,
        per_stratum=args.states_per_stratum,
        max_states=args.max_states,
        max_cases_per_file=args.max_cases_per_file,
        min_remaining=args.sample_min_remaining,
        max_remaining=args.sample_max_remaining,
        require_r4_miss_r8_hit=args.sample_require_r4_miss_r8_hit,
        progress_every=args.progress_every,
    )


def _select_r8_lpt_child(
    state: BeamState,
) -> tuple[int, StageAction, BeamState, int]:
    actions = generate_direct_candidates(
        state, rank_limit=8, bottom_count=0, budget=32
    )
    if not actions:
        raise RuntimeError("R8/K32 generator produced no legal candidate")
    scored = []
    for candidate_index, action in enumerate(actions):
        child = apply_action(state, action)
        key = (
            _lpt_completion_estimate(child),
            len(child.remaining),
            max(child.c2.task_end, child.c3.task_end),
            candidate_index,
        )
        scored.append((key, action, child, candidate_index))
    _, action, child, candidate_index = min(scored, key=lambda row: row[0])
    return candidate_index, action, child, len(actions)


def collect_r8_policy_states(
    paths: list[Path],
    *,
    source_split: str,
    cases_per_e: int,
    per_stratum: int,
    max_states: int,
    max_cases_per_file: int,
    min_remaining: int,
    max_remaining: int,
    progress_every: int,
) -> list[RankingState]:
    """Collect deterministic states reached by the deployable R8/LPT policy."""
    selected_cases: dict[int, list[tuple[int, str, dict]]] = defaultdict(list)
    for path in paths:
        eligible_in_file = 0
        for key, item in json.loads(path.read_text())["results"].items():
            if item.get("dataset_split") != source_split or not quality_ok(
                item, "proven"
            ):
                continue
            if 0 <= max_cases_per_file <= eligible_in_file:
                break
            eligible_in_file += 1
            case_key = f"E{item['e_total']}:{key}"
            selected_cases[int(item["e_total"])].append(
                (stable_u64("r8-policy-case", case_key), case_key, item)
            )
    for e_total, values in selected_cases.items():
        values.sort(key=lambda row: (row[0], row[1]))
        selected_cases[e_total] = values[:cases_per_e]

    reservoirs: dict[tuple, list[RankingState]] = defaultdict(list)
    cases = 0
    started = time.perf_counter()
    for e_total in (8, 32, 64):
        for _, case_key, item in selected_cases.get(e_total, []):
            dist = {int(eid): int(ntok) for eid, ntok in item["dist"].items()}
            state = FourStageScheduler(
                dist,
                initial_cache_c2=int(item.get("initial_cache_c2", -1)),
                initial_cache_c3=int(item.get("initial_cache_c3", -1)),
            )._initial_state()
            trajectory = []
            max_decisions = 4 * len(state.remaining) + 8
            while state.remaining:
                step = len(trajectory)
                _, action, child, _ = _select_r8_lpt_child(state)
                trajectory.append((step, state, action_family(action)))
                state = child
                if len(trajectory) > max_decisions:
                    raise RuntimeError(f"R8 trajectory progress guard: {case_key}")
            reference = int(item["makespan_cc"])
            regret = int(state.g_score) - reference
            if regret < 0:
                raise RuntimeError(f"R8 policy undercut proven reference: {case_key}")
            regret_class = "positive" if regret > 0 else "exact"
            for step, parent, selected_family in trajectory:
                remaining_count = len(parent.remaining)
                if remaining_count < min_remaining or (
                    max_remaining >= 0 and remaining_count > max_remaining
                ):
                    continue
                sample = RankingState(
                    state_id=f"r8_lpt:{case_key}:{step}",
                    case_key=case_key,
                    e_total=e_total,
                    step=step,
                    source="r8_lpt",
                    state=parent,
                    reference_makespan=reference,
                    trajectory_regret_cc=regret,
                )
                stratum = (
                    regret_class,
                    e_total,
                    decision_mode(parent),
                    remaining_band(remaining_count),
                    selected_family,
                )
                reservoirs[stratum].append(sample)
            cases += 1
            clear_scheduler_caches()
            _equal_finish_left.cache_clear()
            _release_target_left.cache_clear()
            if progress_every > 0 and cases % progress_every == 0:
                print(
                    f"r8-state-sample cases={cases} raw_states="
                    f"{sum(len(v) for v in reservoirs.values())} "
                    f"elapsed_s={time.perf_counter()-started:.1f}",
                    flush=True,
                )

    groups = {}
    for stratum, values in reservoirs.items():
        values.sort(key=lambda sample: stable_u64(sample.state_id))
        groups[stratum] = values[:per_stratum]
    groups_by_e = {
        e_total: sorted(
            (key for key in groups if key[1] == e_total),
            key=lambda key: (
                0 if key[0] == "positive" else 1,
                key[2],
                key[3],
                key[4],
            ),
        )
        for e_total in (8, 32, 64)
    }
    group_order = []
    group_depth = 0
    while True:
        added_group = False
        for e_total in (8, 32, 64):
            values = groups_by_e[e_total]
            if group_depth < len(values):
                group_order.append(values[group_depth])
                added_group = True
        if not added_group:
            break
        group_depth += 1
    chosen = []
    depth = 0
    while max_states < 0 or len(chosen) < max_states:
        added = False
        for key in group_order:
            values = groups[key]
            if depth >= len(values):
                continue
            chosen.append(values[depth])
            added = True
            if 0 <= max_states <= len(chosen):
                return chosen
        if not added:
            break
        depth += 1
    return chosen


def select_ranking_states(args: argparse.Namespace) -> list[RankingState]:
    if args.ranking_state_source == "reference":
        reference_budget = args.max_states
        policy_budget = 0
    elif args.ranking_state_source == "r8_lpt":
        reference_budget = 0
        policy_budget = args.max_states
    else:
        reference_budget = args.max_states // 2 if args.max_states >= 0 else -1
        policy_budget = (
            args.max_states - reference_budget if args.max_states >= 0 else -1
        )

    selected = []
    if reference_budget != 0:
        reference_states = collect_states(
            args.inputs,
            source_split=args.source_split,
            per_stratum=args.states_per_stratum,
            max_states=reference_budget,
            max_cases_per_file=args.max_cases_per_file,
            min_remaining=args.sample_min_remaining,
            max_remaining=args.sample_max_remaining,
            require_r4_miss_r8_hit=False,
            progress_every=args.progress_every,
        )
        selected.extend(
            RankingState(
                state_id=f"reference:{sample.case_key}:{sample.step}",
                case_key=sample.case_key,
                e_total=sample.e_total,
                step=sample.step,
                source="reference",
                state=sample.state,
                reference_makespan=sample.reference_makespan,
                trajectory_regret_cc=0,
            )
            for sample in reference_states
        )
    if policy_budget != 0:
        selected.extend(
            collect_r8_policy_states(
                args.inputs,
                source_split=args.source_split,
                cases_per_e=args.ranking_cases_per_e,
                per_stratum=args.states_per_stratum,
                max_states=policy_budget,
                max_cases_per_file=args.max_cases_per_file,
                min_remaining=args.sample_min_remaining,
                max_remaining=args.sample_max_remaining,
                progress_every=args.progress_every,
            )
        )
    if len({sample.state_id for sample in selected}) != len(selected):
        raise RuntimeError("duplicate ranking state IDs")
    return selected


def concrete_pool(state: BeamState, rank_limit: int, bottom_count: int) -> set[int]:
    ranks = list(state.remaining)
    pool = {eid for eid, _ in ranks[:rank_limit]}
    if bottom_count > 0:
        pool.update(eid for eid, _ in ranks[-bottom_count:])
    pool.update(
        snap.pf_eid
        for snap in (state.c2, state.c3)
        if snap.pf_eid >= 0 and any(eid == snap.pf_eid for eid, _ in ranks)
    )
    return pool


def allowed_task_profile(action: StageAction, cluster: int) -> bool:
    if cluster == 2:
        eid = action.c2_eid
        s1, s3 = action.c2_shape_s1, action.c2_shape_s3
    else:
        eid = action.c3_eid
        s1, s3 = action.c3_shape_s1, action.c3_shape_s3
    if eid < 0:
        return True
    return (shape_name(s1), shape_name(s3)) in TASK_PROFILES


@lru_cache(maxsize=None)
def _equal_finish_left(
    total: int,
    eid: int,
    c2_start: int,
    c2_s1,
    c2_s3,
    c2_s1_cached: bool,
    c2_s3_cached: bool,
    c2_s2pf_start: int,
    c2_dma_s1,
    c2_dma_s3,
    c2_s2pf_dma,
    c3_start: int,
    c3_s1,
    c3_s3,
    c3_s1_cached: bool,
    c3_s3_cached: bool,
    c3_s2pf_start: int,
    c3_dma_s1,
    c3_dma_s3,
    c3_s2pf_dma,
) -> int:
    def ends(left: int) -> tuple[int, int]:
        right = total - left
        c2 = FourStageSnap.from_assign(
            c2_start,
            c2_s1,
            c2_s3,
            left,
            eid,
            c2_s1_cached,
            c2_s3_cached,
            c2_s2pf_start,
            c2_dma_s1,
            c2_dma_s3,
            c2_s2pf_dma,
        )
        c3 = FourStageSnap.from_assign(
            c3_start,
            c3_s1,
            c3_s3,
            right,
            eid,
            c3_s1_cached,
            c3_s3_cached,
            c3_s2pf_start,
            c3_dma_s1,
            c3_dma_s3,
            c3_s2pf_dma,
        )
        return c2.task_end, c3.task_end

    if total < 2:
        raise ValueError("split total must be at least two")
    # end2(left) is nondecreasing and end3(total-left) is nonincreasing.
    # Therefore the minimum makespan is adjacent to the zero crossing; no
    # token-by-token scan is needed.
    lo, hi = 1, total - 1
    while lo < hi:
        mid = (lo + hi) // 2
        end2, end3 = ends(mid)
        if end2 - end3 < 0:
            lo = mid + 1
        else:
            hi = mid
    candidates = set(
        range(
            max(1, lo - SPLIT_CROSSING_RADIUS),
            min(total - 1, lo + SPLIT_CROSSING_RADIUS) + 1,
        )
    )
    return min(
        candidates,
        key=lambda left: (
            max(ends(left)),
            abs(ends(left)[0] - ends(left)[1]),
            left,
        ),
    )


def equal_finish_left(action: StageAction) -> int:
    return _equal_finish_left(
        action.c2_ntok + action.c3_ntok,
        action.c2_eid,
        action.c2_start,
        action.c2_shape_s1,
        action.c2_shape_s3,
        action.c2_s1_cached,
        action.c2_s3_cached,
        action.c2_s2pf_start,
        action.c2_dma_s1,
        action.c2_dma_s3,
        action.c2_s2pf_dma,
        action.c3_start,
        action.c3_shape_s1,
        action.c3_shape_s3,
        action.c3_s1_cached,
        action.c3_s3_cached,
        action.c3_s2pf_start,
        action.c3_dma_s1,
        action.c3_dma_s3,
        action.c3_s2pf_dma,
    )


@lru_cache(maxsize=None)
def _release_target_left(
    action: StageAction,
    target_duration: int,
) -> int:
    total = action.c2_ntok + action.c3_ntok
    if total < 2:
        raise ValueError("split total must be at least two")

    def ends(left: int) -> tuple[int, int]:
        right = total - left
        c2 = FourStageSnap.from_assign(
            action.c2_start,
            action.c2_shape_s1,
            action.c2_shape_s3,
            left,
            action.c2_eid,
            action.c2_s1_cached,
            action.c2_s3_cached,
            action.c2_s2pf_start,
            action.c2_dma_s1,
            action.c2_dma_s3,
            action.c2_s2pf_dma,
        )
        c3 = FourStageSnap.from_assign(
            action.c3_start,
            action.c3_shape_s1,
            action.c3_shape_s3,
            right,
            action.c3_eid,
            action.c3_s1_cached,
            action.c3_s3_cached,
            action.c3_s2pf_start,
            action.c3_dma_s1,
            action.c3_dma_s3,
            action.c3_s2pf_dma,
        )
        return c2.task_end, c3.task_end

    # delta(left)=end2(left)-end3(total-left) is monotone.  |delta| is closest
    # to target_duration at one of the two crossings delta=-target/+target.
    candidates = set()
    for target in (-target_duration, target_duration):
        lo, hi = 1, total - 1
        while lo < hi:
            mid = (lo + hi) // 2
            end2, end3 = ends(mid)
            if end2 - end3 < target:
                lo = mid + 1
            else:
                hi = mid
        candidates.update(
            range(
                max(1, lo - SPLIT_CROSSING_RADIUS),
                min(total - 1, lo + SPLIT_CROSSING_RADIUS) + 1,
            )
        )
    return min(
        candidates,
        key=lambda left: (
            abs(abs(ends(left)[0] - ends(left)[1]) - target_duration),
            max(ends(left)),
            left,
        ),
    )


def split_rule(action: StageAction, state: BeamState | None = None) -> str | None:
    if action_family(action) != "SPLIT":
        return "-"
    left, right = action.c2_ntok, action.c3_ntok
    total = left + right
    half = total / 2
    if abs(left - half) <= 1.5:
        return f"HALF_OFFSET_{left - math.floor(half):+g}"
    for boundary in (1, 2, 4, 8):
        if left == boundary:
            return f"FRONT_{boundary}"
        if right == boundary:
            return f"TAIL_{boundary}"
    if state is not None:
        future = [
            (eid, ntok)
            for eid, ntok in state.remaining
            if eid != action.c2_eid
        ]
        for rank, (_, ntok) in enumerate(future[:4]):
            duration = _isolated_task_time_lb(ntok, False, False)
            template = replace(
                action,
                c2_ntok=0,
                c3_ntok=total,
                tag="",
            )
            target_left = _release_target_left(template, duration)
            if abs(left - target_left) <= 1:
                return f"RELEASE_R{rank}"
    if left == equal_finish_left(action):
        return "EQUAL_FINISH"
    return None


def action_key(action: StageAction) -> tuple:
    """Physical action identity excluding the descriptive tag."""
    return (
        action.c2_eid,
        action.c2_ntok,
        action.c2_shape_s1,
        action.c2_shape_s3,
        action.c2_start,
        action.c2_s1_cached,
        action.c2_s3_cached,
        action.c3_eid,
        action.c3_ntok,
        action.c3_shape_s1,
        action.c3_shape_s3,
        action.c3_start,
        action.c3_s1_cached,
        action.c3_s3_cached,
        action.pf_cluster,
        action.pf_eid,
        action.pf_shape,
        action.pf_start,
        action.c2_s2pf_start,
        action.c3_s2pf_start,
        action.c2_dma_s1,
        action.c2_dma_s3,
        action.c2_s2pf_dma,
        action.c3_dma_s1,
        action.c3_dma_s3,
        action.c3_s2pf_dma,
        action.pf_dma,
    )


def macro_key(action: StageAction, state: BeamState) -> tuple:
    family = action_family(action)
    if family == "PREFETCH":
        return (family, action.pf_cluster, action.pf_eid)
    if family == "SINGLE":
        cluster = 2 if action.c2_eid >= 0 else 3
        eid = action.c2_eid if cluster == 2 else action.c3_eid
        return (family, cluster, eid)
    if family == "PAIR":
        return (family, action.c2_eid, action.c3_eid)
    rule = split_rule(action, state)
    if rule == "EQUAL_FINISH" or (
        rule is not None and rule.startswith("RELEASE_R")
    ):
        return (family, action.c2_eid, rule)
    return (
        family,
        action.c2_eid,
        action.c2_ntok,
        action.c3_ntok,
        rule,
    )


def selector_priority(state: BeamState, eid: int) -> tuple:
    ranks = rank_map(state)
    if state.c2.pf_eid == eid or state.c3.pf_eid == eid:
        return (0, 0)
    rank = ranks[eid]
    tail = len(state.remaining) - 1 - rank
    if rank < 8:
        return (1, rank)
    if tail < 4:
        return (2, tail)
    return (3, rank)


def macro_priority(state: BeamState, key: tuple) -> tuple:
    family = key[0]
    if family == "PREFETCH":
        return (selector_priority(state, key[2]), key[1])
    if family == "SINGLE":
        return (selector_priority(state, key[2]), key[1])
    if family == "PAIR":
        a = selector_priority(state, key[1])
        b = selector_priority(state, key[2])
        ranks = rank_map(state)
        adjacent = int(abs(ranks[key[1]] - ranks[key[2]]) != 1)
        return (adjacent, max(a, b), min(a, b), key[1], key[2])
    rule = key[2] if len(key) == 3 else key[4]
    cut_tie = 0 if len(key) == 3 else key[2]
    if decision_mode(state) == "ONE_IDLE":
        # When only one cluster is idle, a useful split must justify waiting
        # for the peer.  Cover the balanced cut of several expert ranks before
        # spending slots on secondary cuts of only the hottest expert.
        if rule is not None and rule.startswith("HALF_OFFSET"):
            balance = abs(key[2] - key[3]) if len(key) != 3 else 0
            return (0, balance, selector_priority(state, key[1]), cut_tie)
        return (
            1,
            split_rule_priority(rule),
            selector_priority(state, key[1]),
            cut_tie,
        )
    return (selector_priority(state, key[1]), split_rule_priority(rule), cut_tie)


def split_rule_priority(rule: str | None) -> int:
    if rule is None:
        return 99
    if rule.startswith("HALF_OFFSET"):
        return 0
    order = {
        "RELEASE_R0": 1,
        "RELEASE_R1": 2,
        "RELEASE_R2": 3,
        "RELEASE_R3": 4,
        "EQUAL_FINISH": 5,
        "FRONT_1": 6,
        "TAIL_1": 7,
        "FRONT_2": 8,
        "TAIL_2": 9,
        "FRONT_4": 10,
        "TAIL_4": 11,
        "FRONT_8": 12,
        "TAIL_8": 13,
    }
    return order.get(rule, 99)


def profile_priority(action: StageAction) -> tuple:
    profile_orders = {
        "A": (("A", "B"), ("B", "B"), ("C", "C")),
        "B": (("B", "B"), ("A", "B"), ("C", "C")),
        "C": (("C", "C"), ("B", "B"), ("A", "B")),
    }

    def slot(cluster: int) -> int:
        if cluster == 2:
            eid, s1, s3 = action.c2_eid, action.c2_shape_s1, action.c2_shape_s3
            s1_cached, s3_cached = action.c2_s1_cached, action.c2_s3_cached
            s2pf = action.c2_s2pf_start >= 0
        else:
            eid, s1, s3 = action.c3_eid, action.c3_shape_s1, action.c3_shape_s3
            s1_cached, s3_cached = action.c3_s1_cached, action.c3_s3_cached
            s2pf = action.c3_s2pf_start >= 0
        if eid < 0:
            return -1
        ntok = action.c2_ntok if cluster == 2 else action.c3_ntok
        profile = (shape_name(s1), shape_name(s3))
        if s1_cached:
            order = (("C", "B"), ("C", "C"))
        elif s3_cached and not s2pf:
            order = (("A", "C"), ("B", "C"), ("C", "C"))
        else:
            target = "C" if ntok == 1 else ("B" if ntok <= 7 else "A")
            order = profile_orders[target] + (
                ("C", "B"), ("A", "C"), ("B", "C")
            )
        return order.index(profile) if profile in order else len(order)

    s2pf_pattern = (
        action.c2_s2pf_start >= 0,
        action.c3_s2pf_start >= 0,
    )
    family = action_family(action)
    if (
        family == "PAIR"
        and action.c2_ntok >= 8
        and action.c3_ntok >= 8
    ):
        s2pf_order = ((True, True), (False, False), (False, True), (True, False))
    else:
        s2pf_order = ((False, False), (True, True), (False, True), (True, False))
    s2pf_priority = s2pf_order.index(s2pf_pattern)
    start_sum = sum(
        start for start in (action.c2_start, action.c3_start) if start >= 0
    )
    # Shape and DMA objects inside action_key deliberately have no ordering.
    # repr() gives a deterministic final tie-break without comparing them.
    return (
        slot(2),
        slot(3),
        s2pf_priority,
        start_sum,
        repr(action_key(action)),
    )


def micro_class_key(action: StageAction) -> tuple:
    """Bounded micro choice before deterministic lane/start allocation."""

    def profile(cluster: int) -> tuple[str, str]:
        if cluster == 2:
            eid, s1, s3 = action.c2_eid, action.c2_shape_s1, action.c2_shape_s3
        else:
            eid, s1, s3 = action.c3_eid, action.c3_shape_s1, action.c3_shape_s3
        if eid < 0:
            return ("-", "-")
        return (shape_name(s1), shape_name(s3))

    return (
        profile(2),
        profile(3),
        action.c2_s2pf_start >= 0,
        action.c3_s2pf_start >= 0,
        shape_name(action.pf_shape),
    )


def fixed_lane_representative_priority(action: StageAction) -> tuple:
    """RTL-oriented earliest-start allocator with fixed lane preferences."""

    def preferred(cluster: int, shape, cached: bool) -> str:
        if shape is None or cached:
            return "NONE"
        if shape_name(shape) == "C":
            return "BOTH"
        return "IDMA" if cluster == 2 else "XDMA"

    def penalty(actual, expected: str) -> int:
        name = actual.name
        if name == expected:
            return 0
        if expected in ("IDMA", "XDMA") and name in ("IDMA", "XDMA"):
            return 1
        return 2

    binding_penalties = (
        penalty(
            action.c2_dma_s1,
            preferred(2, action.c2_shape_s1, action.c2_s1_cached),
        ),
        penalty(
            action.c2_dma_s3,
            preferred(2, action.c2_shape_s3, action.c2_s3_cached),
        ),
        penalty(
            action.c2_s2pf_dma,
            (
                preferred(2, action.c2_shape_s3, False)
                if action.c2_s2pf_start >= 0
                else "NONE"
            ),
        ),
        penalty(
            action.c3_dma_s1,
            preferred(3, action.c3_shape_s1, action.c3_s1_cached),
        ),
        penalty(
            action.c3_dma_s3,
            preferred(3, action.c3_shape_s3, action.c3_s3_cached),
        ),
        penalty(
            action.c3_s2pf_dma,
            (
                preferred(3, action.c3_shape_s3, False)
                if action.c3_s2pf_start >= 0
                else "NONE"
            ),
        ),
        penalty(
            action.pf_dma,
            (
                preferred(action.pf_cluster, action.pf_shape, False)
                if action.pf_cluster in (2, 3)
                else "NONE"
            ),
        ),
    )
    foreground_starts = tuple(
        start for start in (action.c2_start, action.c3_start) if start >= 0
    )
    background_starts = tuple(
        start
        for start in (
            action.c2_s2pf_start,
            action.c3_s2pf_start,
            action.pf_start,
        )
        if start >= 0
    )
    return (
        max(foreground_starts, default=-1),
        sum(foreground_starts),
        sum(binding_penalties),
        binding_penalties,
        max(background_starts, default=-1),
        sum(background_starts),
        repr(action_key(action)),
    )


def family_quotas(mode: str, budget: int) -> dict[str, int]:
    """Frequency-derived slots with one protected rare-family slot."""
    if budget not in (16, 24, 32):
        raise ValueError("candidate budget must be one of 16, 24, 32")
    if mode == "BOTH_IDLE":
        table = {
            16: (2, 10, 3, 1),
            24: (3, 15, 5, 1),
            32: (5, 19, 7, 1),
        }
    elif mode == "ONE_IDLE":
        table = {
            16: (11, 1, 3, 1),
            24: (18, 1, 4, 1),
            32: (26, 1, 4, 1),
        }
    else:
        table = {
            16: (14, 0, 1, 1),
            24: (22, 0, 1, 1),
            32: (30, 0, 1, 1),
        }
    values = table[budget]
    return dict(zip(FAMILIES, values))


def _pool_entries(
    state: BeamState, rank_limit: int, bottom_count: int
) -> list[tuple[int, int]]:
    pool = concrete_pool(state, rank_limit, bottom_count)
    return [(eid, ntok) for eid, ntok in state.remaining if eid in pool]


def _equivalent_entries(
    state: BeamState,
    entries: list[tuple[int, int]],
    *,
    now: int,
    multiplicity: int,
) -> list[tuple[int, int]]:
    """Retain only timing-distinct IDs; PAIR needs two IDs per class."""

    def cluster_key(eid: int, snap: FourStageSnap) -> tuple:
        named = snap.pf_eid == eid
        return (
            named,
            snap.pf_full if named else False,
            _swiglu_hit_for_candidate(eid, snap, now),
            _down_hit_for_candidate(eid, snap, now),
        )

    counts = Counter()
    kept = []
    for eid, ntok in entries:
        key = (ntok, cluster_key(eid, state.c2), cluster_key(eid, state.c3))
        if counts[key] >= multiplicity:
            continue
        counts[key] += 1
        kept.append((eid, ntok))
    return kept


def _profile_choices(s1_hit: bool, s3_hit: bool) -> tuple[tuple, ...]:
    # The retained A/B, B/B and C/C bank has only C/C when either stage is a
    # cache hit, because the reference generator canonicalizes a skipped stage
    # to Shape C.
    if s1_hit and s3_hit:
        return ((SHAPE_C, SHAPE_C),)
    if s1_hit:
        return ((SHAPE_C, SHAPE_B), (SHAPE_C, SHAPE_C))
    if s3_hit:
        return (
            (SHAPE_A, SHAPE_C),
            (SHAPE_B, SHAPE_C),
            (SHAPE_C, SHAPE_C),
        )
    return PROFILE_SHAPES


def _dedicated_binding(cluster: int, cached: bool) -> DmaBinding:
    if cached:
        return DmaBinding.NONE
    return DmaBinding.IDMA if cluster == 2 else DmaBinding.XDMA


def _single_lane_plans(
    cluster: int,
    s1,
    s3,
    hit1: bool,
    hit3: bool,
    use_s2pf: bool,
) -> tuple[tuple[DmaBinding, DmaBinding, DmaBinding], ...]:
    """Eight fixed lane modes replace a 3-by-3-by-3 binding product."""
    dedicated = DmaBinding.IDMA if cluster == 2 else DmaBinding.XDMA
    alternate = DmaBinding.XDMA if cluster == 2 else DmaBinding.IDMA
    preferred1 = DmaBinding.BOTH if s1 == SHAPE_C else dedicated
    preferred3 = DmaBinding.BOTH if s3 == SHAPE_C else dedicated
    raw = (
        (preferred1, preferred3, preferred3),
        (dedicated, dedicated, dedicated),
        (alternate, alternate, alternate),
        (DmaBinding.BOTH, DmaBinding.BOTH, DmaBinding.BOTH),
        (dedicated, alternate, dedicated),
        (dedicated, dedicated, alternate),
        (alternate, dedicated, alternate),
        (alternate, alternate, dedicated),
    )
    normalized = []
    for d1, d3, dpf in raw:
        normalized.append(
            (
                DmaBinding.NONE if hit1 else d1,
                DmaBinding.NONE if hit3 else d3,
                dpf if use_s2pf else DmaBinding.NONE,
            )
        )
    return tuple(dict.fromkeys(normalized))


def _pair_stage_action(
    state: BeamState,
    *,
    eid2: int,
    ntok2: int,
    profile2: tuple,
    s2pf2: bool,
    eid3: int,
    ntok3: int,
    profile3: tuple,
    s2pf3: bool,
    tag: str,
) -> StageAction | None:
    """Instantiate one pair/split slot with one fixed lane per cluster."""
    now = max(state.c2.task_end, state.c3.task_end)
    s12, s32 = profile2
    s13, s33 = profile3
    hit12 = _swiglu_hit_for_candidate(eid2, state.c2, now)
    hit32 = _down_hit_for_candidate(eid2, state.c2, now)
    hit13 = _swiglu_hit_for_candidate(eid3, state.c3, now)
    hit33 = _down_hit_for_candidate(eid3, state.c3, now)
    if (s2pf2 and hit32) or (s2pf3 and hit33):
        return None

    dedicated = (
        _dedicated_binding(2, hit12),
        _dedicated_binding(2, hit32),
        _dedicated_binding(3, hit13),
        _dedicated_binding(3, hit33),
    )
    opportunistic = list(dedicated)
    # If the peer skips a stage, Shape C may use both lanes without duplicating
    # a general binding search.  This is required for asymmetric cache states.
    if not hit12 and hit13 and s12 == SHAPE_C:
        opportunistic[0] = DmaBinding.BOTH
    if not hit13 and hit12 and s13 == SHAPE_C:
        opportunistic[2] = DmaBinding.BOTH
    if not hit32 and (hit33 or s2pf3) and s32 == SHAPE_C:
        opportunistic[1] = DmaBinding.BOTH
    if not hit33 and (hit32 or s2pf2) and s33 == SHAPE_C:
        opportunistic[3] = DmaBinding.BOTH

    candidates = []
    for d12, d32, d13, d33 in dict.fromkeys(
        (dedicated, tuple(opportunistic))
    ):
        base2 = FourStageSnap.from_assign(
            now, s12, s32, ntok2, eid2, hit12, hit32,
            dma_s1=d12, dma_s3=d32,
        )
        base3 = FourStageSnap.from_assign(
            now, s13, s33, ntok3, eid3, hit13, hit33,
            dma_s1=d13, dma_s3=d33,
        )
        snap2 = (
            FourStageSnap.from_assign(
                now, s12, s32, ntok2, eid2, hit12, hit32,
                base2.dma1_end, d12, d32, DmaBinding.IDMA,
            )
            if s2pf2 else base2
        )
        snap3 = (
            FourStageSnap.from_assign(
                now, s13, s33, ntok3, eid3, hit13, hit33,
                base3.dma1_end, d13, d33, DmaBinding.XDMA,
            )
            if s2pf3 else base3
        )
        if not bw_feasible(snap2, snap3):
            continue
        candidates.append(
            StageAction(
                c2_eid=eid2, c2_ntok=ntok2, c2_shape_s1=s12,
                c2_shape_s3=s32, c2_start=now, c2_s1_cached=hit12,
                c2_s3_cached=snap2.bw_s3 == 0,
                c3_eid=eid3, c3_ntok=ntok3, c3_shape_s1=s13,
                c3_shape_s3=s33, c3_start=now, c3_s1_cached=hit13,
                c3_s3_cached=snap3.bw_s3 == 0,
                pf_cluster=-1, pf_eid=-1, pf_shape=None, pf_start=-1,
                tag=tag,
                c2_s2pf_start=snap2.s2pf_start,
                c3_s2pf_start=snap3.s2pf_start,
                c2_dma_s1=snap2.dma_s1, c2_dma_s3=snap2.dma_s3,
                c2_s2pf_dma=snap2.s2pf_dma,
                c3_dma_s1=snap3.dma_s1, c3_dma_s3=snap3.dma_s3,
                c3_s2pf_dma=snap3.s2pf_dma,
            )
        )
    return min(candidates, key=fixed_lane_representative_priority) if candidates else None


def _single_stage_action(
    state: BeamState,
    *,
    cluster: int,
    eid: int,
    ntok: int,
    profile: tuple,
    use_s2pf: bool,
) -> StageAction | None:
    """Choose one earliest legal action from eight fixed lane modes."""
    current, peer = (state.c2, state.c3) if cluster == 2 else (state.c3, state.c2)
    start_floor = current.task_end
    s1, s3 = profile
    hit1 = _swiglu_hit_for_candidate(eid, current, start_floor)
    hit3 = _down_hit_for_candidate(eid, current, start_floor)
    if use_s2pf and hit3:
        return None
    candidates = []
    for d1, d3, dpf in _single_lane_plans(
        cluster, s1, s3, hit1, hit3, use_s2pf
    ):
        for start in _start_candidates(
            start_floor, current, peer, ntok, s1, s3,
            d1, d3, hit1, hit3,
        ):
            base = FourStageSnap.from_assign(
                start, s1, s3, ntok, eid, hit1, hit3,
                dma_s1=d1, dma_s3=d3,
            )
            snap = (
                FourStageSnap.from_assign(
                    start, s1, s3, ntok, eid, hit1, hit3,
                    base.dma1_end, d1, d3, dpf,
                )
                if use_s2pf else base
            )
            feasible = (
                bw_feasible(snap, peer)
                if cluster == 2 else bw_feasible(peer, snap)
            )
            if not feasible:
                continue
            idle = dict(
                c2_eid=-1, c2_ntok=0, c2_shape_s1=None, c2_shape_s3=None,
                c2_start=-1, c2_s1_cached=False, c2_s3_cached=False,
                c3_eid=-1, c3_ntok=0, c3_shape_s1=None, c3_shape_s3=None,
                c3_start=-1, c3_s1_cached=False, c3_s3_cached=False,
            )
            if cluster == 2:
                idle.update(
                    c2_eid=eid, c2_ntok=ntok, c2_shape_s1=s1,
                    c2_shape_s3=s3, c2_start=start, c2_s1_cached=hit1,
                    c2_s3_cached=snap.bw_s3 == 0,
                )
            else:
                idle.update(
                    c3_eid=eid, c3_ntok=ntok, c3_shape_s1=s1,
                    c3_shape_s3=s3, c3_start=start, c3_s1_cached=hit1,
                    c3_s3_cached=snap.bw_s3 == 0,
                )
            action = StageAction(
                **idle,
                pf_cluster=-1, pf_eid=-1, pf_shape=None, pf_start=-1,
                tag=f"SINGLE-C{cluster}(E{eid})",
                c2_s2pf_start=snap.s2pf_start if cluster == 2 else -1,
                c3_s2pf_start=snap.s2pf_start if cluster == 3 else -1,
                c2_dma_s1=snap.dma_s1 if cluster == 2 else DmaBinding.NONE,
                c2_dma_s3=snap.dma_s3 if cluster == 2 else DmaBinding.NONE,
                c2_s2pf_dma=snap.s2pf_dma if cluster == 2 else DmaBinding.NONE,
                c3_dma_s1=snap.dma_s1 if cluster == 3 else DmaBinding.NONE,
                c3_dma_s3=snap.dma_s3 if cluster == 3 else DmaBinding.NONE,
                c3_s2pf_dma=snap.s2pf_dma if cluster == 3 else DmaBinding.NONE,
            )
            candidates.append(action)
    return min(candidates, key=fixed_lane_representative_priority) if candidates else None


def _prefetch_variants(
    state: BeamState, cluster: int, eid: int
) -> list[StageAction]:
    current, peer = (state.c2, state.c3) if cluster == 2 else (state.c3, state.c2)
    if current.cur_eid < 0 or current.pf_eid != -1:
        return []
    # A target reserved by the peer is already owned.  Reserving the same
    # remaining expert on both clusters can consume both DMA lanes while the
    # next stage action is required to consume two different reservations,
    # producing a continuation dead end.
    if _reserved_next_eid(peer) == eid:
        return []
    start = current.dma3_end
    if peer.cur_eid >= 0 and start < peer.task_start:
        return []
    by_shape: dict[object, list[StageAction]] = defaultdict(list)
    for dma in (DmaBinding.IDMA, DmaBinding.XDMA, DmaBinding.BOTH):
        shape = SHAPE_C if dma == DmaBinding.BOTH else SHAPE_A
        snap = current.with_prefetch(eid, shape, start, dma)
        feasible = (
            bw_feasible(snap, peer)
            if cluster == 2 else bw_feasible(peer, snap)
        )
        if not feasible:
            continue
        action = StageAction(
            c2_eid=-2 if cluster == 2 else -1,
            c2_ntok=0,
            c2_shape_s1=shape if cluster == 2 else None,
            c2_shape_s3=None,
            c2_start=start if cluster == 2 else -1,
            c2_s1_cached=False,
            c2_s3_cached=False,
            c3_eid=-2 if cluster == 3 else -1,
            c3_ntok=0,
            c3_shape_s1=shape if cluster == 3 else None,
            c3_shape_s3=None,
            c3_start=start if cluster == 3 else -1,
            c3_s1_cached=False,
            c3_s3_cached=False,
            pf_cluster=cluster,
            pf_eid=eid,
            pf_shape=shape,
            pf_start=start,
            tag=f"PF-C{cluster}(E{eid},{dma.name})",
            pf_dma=dma,
        )
        by_shape[shape].append(action)
    result = [
        min(actions, key=fixed_lane_representative_priority)
        for actions in by_shape.values()
    ]
    peer_idle = peer.cur_eid < 0 or start >= peer.task_end
    result.sort(
        key=lambda action: (
            0 if (peer_idle and action.pf_shape == SHAPE_C) else 1,
            0 if action.pf_shape == SHAPE_A else 1,
            profile_priority(action),
        )
    )
    return result


def _static_split_rule(left: int, total: int) -> str | None:
    half = total / 2
    if abs(left - half) <= 1.5:
        return f"HALF_OFFSET_{left - math.floor(half):+g}"
    for boundary in (1, 2, 4, 8):
        if left == boundary:
            return f"FRONT_{boundary}"
        if total - left == boundary:
            return f"TAIL_{boundary}"
    return None


def _direct_macros(
    state: BeamState, rank_limit: int, bottom_count: int
) -> dict[str, list[DirectMacro]]:
    entries = _pool_entries(state, rank_limit, bottom_count)
    macros = {family: [] for family in FAMILIES}
    t2, t3 = state.c2.task_end, state.c3.task_end
    reserved2 = _reserved_next_eid(state.c2)
    reserved3 = _reserved_next_eid(state.c3)

    pair_now = max(t2, t3)
    pair_clusters_symmetric = reserved2 == reserved3 and all(
        _swiglu_hit_for_candidate(eid, state.c2, pair_now)
        == _swiglu_hit_for_candidate(eid, state.c3, pair_now)
        and _down_hit_for_candidate(eid, state.c2, pair_now)
        == _down_hit_for_candidate(eid, state.c3, pair_now)
        for eid, _ in entries
    )

    clusters = []
    if t2 < t3:
        clusters.append(2)
        if reserved3 >= 0:
            clusters.append(3)
    elif t3 < t2:
        clusters.append(3)
        if reserved2 >= 0:
            clusters.append(2)
    else:
        clusters.append(2)
        if state.c2 != state.c3:
            clusters.append(3)
    for cluster in clusters:
        single_entries = _equivalent_entries(
            state,
            entries,
            now=(state.c2 if cluster == 2 else state.c3).task_end,
            multiplicity=1,
        )
        own_reserved = reserved2 if cluster == 2 else reserved3
        peer_reserved = reserved3 if cluster == 2 else reserved2
        for eid, ntok in single_entries:
            if own_reserved >= 0 and eid != own_reserved:
                continue
            if peer_reserved >= 0 and eid == peer_reserved:
                continue
            key = ("SINGLE", cluster, eid)
            macros["SINGLE"].append(DirectMacro("SINGLE", key, (cluster, eid, ntok)))

    if len(state.remaining) >= 2:
        pair_entries = _equivalent_entries(
            state, entries, now=max(t2, t3), multiplicity=2
        )
        for i, (eid2, ntok2) in enumerate(pair_entries):
            for j, (eid3, ntok3) in enumerate(pair_entries):
                if i == j:
                    continue
                if pair_clusters_symmetric and j < i:
                    continue
                if reserved2 >= 0 and eid2 != reserved2:
                    continue
                if reserved3 >= 0 and eid3 != reserved3:
                    continue
                key = ("PAIR", eid2, eid3)
                macros["PAIR"].append(
                    DirectMacro("PAIR", key, (eid2, ntok2, eid3, ntok3))
                )

    split_entries = _equivalent_entries(
        state, entries, now=max(t2, t3), multiplicity=1
    )
    for eid, total in split_entries:
        if total < 2:
            continue
        if reserved2 >= 0 and eid != reserved2:
            continue
        if reserved3 >= 0 and eid != reserved3:
            continue
        explicit_cuts = set()
        lo = max(1, math.ceil(total / 2 - 1.5))
        hi = min(total - 1, math.floor(total / 2 + 1.5))
        explicit_cuts.update(range(lo, hi + 1))
        explicit_cuts.update(
            cut
            for cut in (1, 2, 4, 8, total - 1, total - 2, total - 4, total - 8)
            if 0 < cut < total
        )
        for left in sorted(explicit_cuts):
            rule = _static_split_rule(left, total)
            if rule is None:
                continue
            key = ("SPLIT", eid, left, total - left, rule)
            macros["SPLIT"].append(
                DirectMacro("SPLIT", key, (eid, total, rule, left))
            )
        future = [(future_eid, ntok) for future_eid, ntok in state.remaining if future_eid != eid]
        for rank in range(min(4, len(future))):
            rule = f"RELEASE_R{rank}"
            macros["SPLIT"].append(
                DirectMacro("SPLIT", ("SPLIT", eid, rule), (eid, total, rule, None))
            )
        macros["SPLIT"].append(
            DirectMacro(
                "SPLIT", ("SPLIT", eid, "EQUAL_FINISH"),
                (eid, total, "EQUAL_FINISH", None),
            )
        )

    for cluster, current in ((2, state.c2), (3, state.c3)):
        if current.cur_eid < 0 or current.pf_eid != -1:
            continue
        peer_reserved = reserved3 if cluster == 2 else reserved2
        prefetch_entries = _equivalent_entries(
            state, entries, now=min(t2, t3), multiplicity=2
        )
        for eid, _ in prefetch_entries:
            if peer_reserved == eid:
                continue
            key = ("PREFETCH", cluster, eid)
            macros["PREFETCH"].append(DirectMacro("PREFETCH", key, (cluster, eid)))

    for family in FAMILIES:
        unique = {macro.key: macro for macro in macros[family]}
        macros[family] = sorted(
            unique.values(), key=lambda macro: macro_priority(state, macro.key)
        )
    return macros


def _direct_micro_actions(state: BeamState, macro: DirectMacro) -> list[StageAction]:
    family = macro.family
    actions = []
    if family == "SINGLE":
        cluster, eid, ntok = macro.payload
        current = state.c2 if cluster == 2 else state.c3
        hit1 = _swiglu_hit_for_candidate(eid, current, current.task_end)
        hit3 = _down_hit_for_candidate(eid, current, current.task_end)
        for profile in _profile_choices(hit1, hit3):
            for use_s2pf in (False, True):
                action = _single_stage_action(
                    state, cluster=cluster, eid=eid, ntok=ntok,
                    profile=profile, use_s2pf=use_s2pf,
                )
                if action is not None:
                    actions.append(action)
    elif family == "PAIR":
        eid2, ntok2, eid3, ntok3 = macro.payload
        now = max(state.c2.task_end, state.c3.task_end)
        hit12 = _swiglu_hit_for_candidate(eid2, state.c2, now)
        hit32 = _down_hit_for_candidate(eid2, state.c2, now)
        hit13 = _swiglu_hit_for_candidate(eid3, state.c3, now)
        hit33 = _down_hit_for_candidate(eid3, state.c3, now)
        patterns = ((True, True), (False, False), (False, True), (True, False)) if ntok2 >= 8 and ntok3 >= 8 else ((False, False), (True, True), (False, True), (True, False))
        for profile2 in _profile_choices(hit12, hit32):
            for profile3 in _profile_choices(hit13, hit33):
                for s2pf2, s2pf3 in patterns:
                    action = _pair_stage_action(
                        state, eid2=eid2, ntok2=ntok2, profile2=profile2,
                        s2pf2=s2pf2, eid3=eid3, ntok3=ntok3,
                        profile3=profile3, s2pf3=s2pf3,
                        tag=f"PAIR({eid2}+{eid3})",
                    )
                    if action is not None:
                        actions.append(action)
    elif family == "SPLIT":
        eid, total, intended_rule, explicit_left = macro.payload
        now = max(state.c2.task_end, state.c3.task_end)
        hit12 = _swiglu_hit_for_candidate(eid, state.c2, now)
        hit32 = _down_hit_for_candidate(eid, state.c2, now)
        hit13 = _swiglu_hit_for_candidate(eid, state.c3, now)
        hit33 = _down_hit_for_candidate(eid, state.c3, now)
        for profile2 in _profile_choices(hit12, hit32):
            for profile3 in _profile_choices(hit13, hit33):
                for s2pf2, s2pf3 in ((False, False), (True, True), (False, True), (True, False)):
                    left = explicit_left if explicit_left is not None else max(1, total // 2)
                    action = _pair_stage_action(
                        state, eid2=eid, ntok2=left, profile2=profile2,
                        s2pf2=s2pf2, eid3=eid, ntok3=total-left,
                        profile3=profile3, s2pf3=s2pf3,
                        tag=f"SPLIT(E{eid}:{left},{total-left})",
                    )
                    if action is None:
                        continue
                    if explicit_left is None:
                        if intended_rule == "EQUAL_FINISH":
                            left = equal_finish_left(action)
                        else:
                            rank = int(intended_rule[-1])
                            future = [(future_eid, ntok) for future_eid, ntok in state.remaining if future_eid != eid]
                            if rank >= len(future):
                                continue
                            duration = _isolated_task_time_lb(future[rank][1], False, False)
                            left = _release_target_left(
                                replace(action, c2_ntok=0, c3_ntok=total, tag=""),
                                duration,
                            )
                        action = _pair_stage_action(
                            state, eid2=eid, ntok2=left, profile2=profile2,
                            s2pf2=s2pf2, eid3=eid, ntok3=total-left,
                            profile3=profile3, s2pf3=s2pf3,
                            tag=f"SPLIT(E{eid}:{left},{total-left})",
                        )
                    if action is None or split_rule(action, state) != intended_rule:
                        continue
                    if state.c2 == state.c3:
                        left_key = (
                            action.c2_ntok, shape_name(action.c2_shape_s1),
                            int(action.c2_dma_s1), shape_name(action.c2_shape_s3),
                            int(action.c2_dma_s3),
                        )
                        right_key = (
                            action.c3_ntok, shape_name(action.c3_shape_s1),
                            int(action.c3_dma_s1), shape_name(action.c3_shape_s3),
                            int(action.c3_dma_s3),
                        )
                        if left_key > right_key:
                            continue
                    actions.append(action)
    else:
        cluster, eid = macro.payload
        actions.extend(_prefetch_variants(state, cluster, eid))

    unique = {action_key(action): action for action in actions}
    if family == "PREFETCH":
        return list(unique.values())
    return sorted(unique.values(), key=profile_priority)


def _direct_family_stream(state: BeamState, macros: list[DirectMacro]):
    variants: dict[tuple, list[StageAction]] = {}
    depth = 0
    while True:
        added = False
        for macro in macros:
            values = variants.get(macro.key)
            if values is None:
                values = _direct_micro_actions(state, macro)
                variants[macro.key] = values
            if depth < len(values):
                added = True
                yield values[depth]
        if not added:
            return
        depth += 1


def validate_direct_action(state: BeamState, action: StageAction) -> None:
    """Reject a direct action that violates the reference state contract."""
    family = action_family(action)
    remaining = dict(state.remaining)
    if family == "PREFETCH":
        cluster = action.pf_cluster
        current, peer = (
            (state.c2, state.c3) if cluster == 2 else (state.c3, state.c2)
        )
        if action.pf_eid not in remaining:
            raise ValueError("prefetch target is not remaining")
        if current.cur_eid < 0 or current.pf_eid != -1:
            raise ValueError("prefetch requires an active cluster with an empty slot")
        if _reserved_next_eid(peer) == action.pf_eid:
            raise ValueError("prefetch target is already reserved by the peer")
        if action.pf_start != current.dma3_end:
            raise ValueError("prefetch must start at dma3_end")
        if peer.cur_eid >= 0 and action.pf_start < peer.task_start:
            raise ValueError("prefetch would modify sealed peer history")
        child = apply_action(state, action)
        if not bw_feasible(child.c2, child.c3):
            raise ValueError("prefetch violates DMA lane feasibility")
        return

    active = []
    for cluster, old, eid, ntok, start, cached1, cached3, s2pf_start in (
        (
            2, state.c2, action.c2_eid, action.c2_ntok, action.c2_start,
            action.c2_s1_cached, action.c2_s3_cached, action.c2_s2pf_start,
        ),
        (
            3, state.c3, action.c3_eid, action.c3_ntok, action.c3_start,
            action.c3_s1_cached, action.c3_s3_cached, action.c3_s2pf_start,
        ),
    ):
        if eid < 0:
            continue
        if eid not in remaining or start < old.task_end:
            raise ValueError("task consumes a missing expert or starts too early")
        expected1 = _swiglu_hit_for_candidate(eid, old, start)
        expected3 = _down_hit_for_candidate(eid, old, start)
        if cached1 != expected1:
            raise ValueError("S1 cache flag does not match residency")
        if s2pf_start < 0 and cached3 != expected3:
            raise ValueError("S3 cache flag does not match residency")
        if s2pf_start >= 0 and (expected3 or not cached3):
            raise ValueError("S2PF must replace an uncached S3 transfer")
        active.append((cluster, eid, ntok))
    if not active:
        raise ValueError("non-prefetch action has no task")
    if len(active) == 2 and active[0][1] == active[1][1]:
        eid = active[0][1]
        if active[0][2] + active[1][2] != remaining[eid]:
            raise ValueError("split token counts do not conserve the expert")
    else:
        for _, eid, ntok in active:
            if ntok != remaining[eid]:
                raise ValueError("non-split action changed an expert token count")

    reserved2 = _reserved_next_eid(state.c2)
    reserved3 = _reserved_next_eid(state.c3)
    selected2 = action.c2_eid
    selected3 = action.c3_eid
    if selected2 >= 0 and reserved2 >= 0 and selected2 != reserved2:
        raise ValueError("C2 did not consume its reserved prefetch target")
    if selected3 >= 0 and reserved3 >= 0 and selected3 != reserved3:
        raise ValueError("C3 did not consume its reserved prefetch target")
    if selected2 >= 0 and selected3 < 0 and reserved3 == selected2:
        raise ValueError("C2 stole C3's reserved prefetch target")
    if selected3 >= 0 and selected2 < 0 and reserved2 == selected3:
        raise ValueError("C3 stole C2's reserved prefetch target")
    child = apply_action(state, action)
    if not bw_feasible(child.c2, child.c3):
        raise ValueError("task action violates DMA lane feasibility")


def generate_direct_candidates(
    state: BeamState,
    *,
    rank_limit: int,
    bottom_count: int,
    budget: int,
) -> list[StageAction]:
    """Generate at most K actions without calling either full action generator."""
    macros = _direct_macros(state, rank_limit, bottom_count)
    streams = {
        family: _direct_family_stream(state, macros[family]) for family in FAMILIES
    }
    selected = []
    selected_keys = set()
    quotas = family_quotas(decision_mode(state), budget)

    def take(family: str, count: int) -> None:
        stream = streams[family]
        while count > 0:
            try:
                action = next(stream)
            except StopIteration:
                return
            key = action_key(action)
            if key in selected_keys:
                continue
            selected.append(action)
            selected_keys.add(key)
            count -= 1

    for family in FAMILIES:
        take(family, quotas[family])
    while len(selected) < budget:
        before = len(selected)
        for family in FAMILIES:
            take(family, 1)
            if len(selected) >= budget:
                break
        if len(selected) == before:
            break
    if len(selected) > budget:
        raise AssertionError("direct generator exceeded K")
    for action in selected:
        validate_direct_action(state, action)
    return selected


def filtered_actions(
    state: BeamState,
    actions: list[StageAction],
    *,
    rank_limit: int,
    bottom_count: int,
) -> list[StageAction]:
    pool = concrete_pool(state, rank_limit, bottom_count)
    kept = []
    for action in actions:
        if any(eid not in pool for eid in selected_eids(action)):
            continue
        if not allowed_task_profile(action, 2) or not allowed_task_profile(action, 3):
            continue
        if action_family(action) == "SPLIT" and split_rule(action, state) is None:
            continue
        kept.append(action)
    return kept


def select_bounded_candidates(
    state: BeamState,
    actions: list[StageAction],
    *,
    budget: int,
) -> list[StageAction]:
    groups: dict[str, dict[tuple, list[StageAction]]] = {
        family: defaultdict(list) for family in FAMILIES
    }
    for action in actions:
        family = action_family(action)
        groups[family][macro_key(action, state)].append(action)
    for family_groups in groups.values():
        for macro, physical_variants in tuple(family_groups.items()):
            micro_groups: dict[tuple, list[StageAction]] = defaultdict(list)
            for action in physical_variants:
                micro_groups[micro_class_key(action)].append(action)
            representatives = [
                min(
                    variants,
                    key=fixed_lane_representative_priority,
                )
                for variants in micro_groups.values()
            ]
            representatives.sort(key=profile_priority)
            family_groups[macro] = representatives

    selected: list[StageAction] = []
    selected_keys = set()
    leftovers = []
    quotas = family_quotas(decision_mode(state), budget)
    for family in FAMILIES:
        macro_items = sorted(
            groups[family].items(), key=lambda item: macro_priority(state, item[0])
        )
        used = 0
        depth = 0
        while used < quotas[family]:
            added = False
            for _, variants in macro_items:
                if depth >= len(variants):
                    continue
                action = variants[depth]
                key = action_key(action)
                if key not in selected_keys:
                    selected.append(action)
                    selected_keys.add(key)
                    used += 1
                    added = True
                    if used >= quotas[family]:
                        break
            if not added:
                break
            depth += 1
        for macro, variants in macro_items:
            for depth_index, action in enumerate(variants):
                if action_key(action) not in selected_keys:
                    leftovers.append(
                        (
                            FAMILIES.index(family),
                            repr(macro_priority(state, macro)),
                            depth_index,
                            profile_priority(action),
                            action,
                        )
                    )

    for _, _, _, _, action in sorted(leftovers, key=lambda item: item[:4]):
        if len(selected) >= budget:
            break
        key = action_key(action)
        if key not in selected_keys:
            selected.append(action)
            selected_keys.add(key)
    return selected


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def audit_candidates(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    samples = select_samples(args)
    rows = []
    exact = child_exact = template_hits = 0
    layer_hits = Counter()
    family_hits = Counter()
    family_total = Counter()
    for index, sample in enumerate(samples, 1):
        state = sample.state
        generated = gen_stage_actions(state.c2, state.c3, state.remaining)
        generated += gen_prefetch_actions(state.c2, state.c3, state.remaining)
        structural = filtered_actions(
            state,
            generated,
            rank_limit=args.rank_limit,
            bottom_count=args.bottom_count,
        )
        bounded = (
            generate_direct_candidates(
                state,
                rank_limit=args.rank_limit,
                bottom_count=args.bottom_count,
                budget=args.budget,
            )
            if args.generator == "direct"
            else select_bounded_candidates(state, structural, budget=args.budget)
        )
        ref_key = action_key(sample.reference_action)
        ref_macro = macro_key(sample.reference_action, state)
        generated_exact = any(action_key(action) == ref_key for action in generated)
        structural_exact = any(action_key(action) == ref_key for action in structural)
        exact_hit = any(action_key(action) == ref_key for action in bounded)
        reference_child = apply_action(state, sample.reference_action)
        bounded_children = [apply_action(state, action) for action in bounded]
        child_hit = any(
            child.fingerprint() == reference_child.fingerprint()
            for child in bounded_children
        )
        min_child_lower_bound = min(
            (child.f_score for child in bounded_children), default=None
        )
        min_child_estimate = min(
            (completion_estimate(child) for child in bounded_children), default=None
        )
        reference_template = canonical_template(state, sample.reference_action)
        structural_template_hit = any(
            canonical_template(state, action) == reference_template
            for action in structural
        )
        template_hit = any(
            canonical_template(state, action) == reference_template for action in bounded
        )
        structural_macro_hit = any(
            macro_key(action, state) == ref_macro for action in structural
        )
        bounded_macro_hit = any(
            macro_key(action, state) == ref_macro for action in bounded
        )
        ref_pool_ok = all(
            eid in concrete_pool(state, args.rank_limit, args.bottom_count)
            for eid in selected_eids(sample.reference_action)
        )
        ref_profile_ok = allowed_task_profile(
            sample.reference_action, 2
        ) and allowed_task_profile(sample.reference_action, 3)
        ref_split_ok = (
            action_family(sample.reference_action) != "SPLIT"
            or split_rule(sample.reference_action, state) is not None
        )
        for name, hit in (
            ("generated_exact", generated_exact),
            ("reference_pool", ref_pool_ok),
            ("reference_profile", ref_profile_ok),
            ("reference_split_rule", ref_split_ok),
            ("structural_exact", structural_exact),
            ("structural_macro", structural_macro_hit),
            ("structural_template", structural_template_hit),
            ("bounded_macro", bounded_macro_hit),
            ("bounded_exact", exact_hit),
            ("bounded_child", child_hit),
            ("bounded_template", template_hit),
        ):
            layer_hits[name] += int(hit)
        family = action_family(sample.reference_action)
        family_total[family] += 1
        family_hits[family] += int(child_hit)
        exact += int(exact_hit)
        child_exact += int(child_hit)
        template_hits += int(template_hit)
        rows.append(
            {
                "state_id": f"{sample.case_key}:{sample.step}",
                "case_key": sample.case_key,
                "step": sample.step,
                "e_total": sample.e_total,
                "mode": decision_mode(state),
                "remaining": len(state.remaining),
                "reference_family": family,
                "generated_actions": len(generated),
                "structural_actions": len(structural),
                "bounded_actions": len(bounded),
                "reference_makespan_cc": sample.reference_makespan,
                "minimum_child_lower_bound_cc": min_child_lower_bound,
                "minimum_child_completion_estimate_cc": min_child_estimate,
                "candidate_loss_lower_bound_cc": (
                    max(0, min_child_lower_bound - sample.reference_makespan)
                    if min_child_lower_bound is not None
                    else None
                ),
                "generated_exact_action_hit": generated_exact,
                "reference_pool_allowed": ref_pool_ok,
                "reference_profile_allowed": ref_profile_ok,
                "reference_split_rule_allowed": ref_split_ok,
                "structural_exact_action_hit": structural_exact,
                "structural_macro_hit": structural_macro_hit,
                "structural_template_hit": structural_template_hit,
                "bounded_macro_hit": bounded_macro_hit,
                "exact_action_hit": exact_hit,
                "exact_child_hit": child_hit,
                "canonical_template_hit": template_hit,
                "bounded_family_counts": dict(
                    Counter(action_family(action) for action in bounded)
                ),
            }
        )
        clear_scheduler_caches()
        _equal_finish_left.cache_clear()
        _release_target_left.cache_clear()
        print(
            f"audit {index}/{len(samples)} {sample.case_key}:{sample.step} "
            f"generated={len(generated)} structural={len(structural)} "
            f"bounded={len(bounded)} child_hit={int(child_hit)}",
            flush=True,
        )

    total = len(samples)
    report = {
        "schema": "scheduler_bounded_candidate_audit_v0",
        "provisional": True,
        "configuration": {
            "generator": args.generator,
            "rank_limit": args.rank_limit,
            "bottom_count": args.bottom_count,
            "budget": args.budget,
            "task_profiles": sorted("/".join(value) for value in TASK_PROFILES),
            "split_rules": (
                "half +/- 1, release targets R0..R3, local equal-finish, "
                "plus front/tail 1,2,4,8"
            ),
            "physical_variant_allocator": "fixed_lane_earliest_start",
        },
        "sampling": {
            "states_per_stratum": args.states_per_stratum,
            "max_states": args.max_states,
            "max_cases_per_file": args.max_cases_per_file,
            "min_remaining": args.sample_min_remaining,
            "max_remaining": args.sample_max_remaining,
            "state_ids_from": (
                str(args.state_ids_from) if args.state_ids_from is not None else None
            ),
            "require_r4_miss_r8_hit": args.sample_require_r4_miss_r8_hit,
            "states": total,
            "discovery_only": True,
            "proven_only": True,
        },
        "summary": {
            "exact_action_hits": exact,
            "exact_action_fraction": exact / total if total else None,
            "exact_child_hits": child_exact,
            "exact_child_fraction": child_exact / total if total else None,
            "canonical_template_hits": template_hits,
            "canonical_template_fraction": template_hits / total if total else None,
            "layer_hit_counts": dict(layer_hits),
            "layer_hit_fractions": {
                key: value / total if total else None
                for key, value in layer_hits.items()
            },
            "runtime_s": time.perf_counter() - started,
        },
        "child_hits_by_reference_family": {
            family: {
                "hits": family_hits[family],
                "states": family_total[family],
                "fraction": (
                    family_hits[family] / family_total[family]
                    if family_total[family]
                    else None
                ),
            }
            for family in FAMILIES
        },
        "rows": rows,
        "interpretation": [
            "This audit measures representation, not candidate-oracle regret.",
            "A missing reference child may still have an equally good alternative.",
            "The v0 quota and priority order remain provisional until forced-action continuation.",
        ],
    }
    atomic_write_json(args.out, report)
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.out}")
    return 0


def action_id(action: StageAction) -> str:
    encoded = json.dumps(
        serialize_action(action), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def continuation_result(
    child: BeamState,
    *,
    reference_child: BeamState,
    reference_makespan: int,
    time_limit_s: float | None,
    max_expansions: int | None,
    target_gap: float | None,
) -> dict:
    if not child.remaining:
        return {
            "upper_bound_cc": child.g_score,
            "lower_bound_cc": child.g_score,
            "proven_optimal": True,
            "optimality_gap": 0.0,
            "expansions": 0,
            "generated": 0,
            "runtime_s": 0.0,
            "termination": "terminal_action",
        }
    if child.fingerprint() == reference_child.fingerprint():
        return {
            "upper_bound_cc": reference_makespan,
            "lower_bound_cc": reference_makespan,
            "proven_optimal": True,
            "optimality_gap": 0.0,
            "expansions": 0,
            "generated": 0,
            "runtime_s": 0.0,
            "termination": "reference_child_equivalence",
        }

    scheduler = FourStageScheduler(dict(child.remaining), enable_prefetch=True)
    result = scheduler.run_anytime(
        time_limit_s=time_limit_s,
        max_expansions=max_expansions,
        target_gap=target_gap,
        initial_state=child,
    )
    if result.makespan < reference_makespan:
        raise RuntimeError(
            "forced continuation improved a proven reference prefix: "
            f"{result.makespan} < {reference_makespan}"
        )
    return {
        "upper_bound_cc": result.makespan,
        "lower_bound_cc": result.lower_bound,
        "proven_optimal": result.proven_optimal,
        "optimality_gap": result.optimality_gap,
        "expansions": result.expansions,
        "generated": result.generated,
        "runtime_s": result.runtime_s,
        "termination": result.termination,
    }


def update_oracle_summary(row: dict) -> None:
    evaluations = row["evaluations"]
    candidate_lbs = []
    candidate_ubs = []
    for candidate in row["candidates"]:
        result = evaluations.get(candidate["id"])
        candidate_lbs.append(
            int(result["lower_bound_cc"])
            if result is not None
            else int(candidate["child_lower_bound_cc"])
        )
        if result is not None:
            candidate_ubs.append(int(result["upper_bound_cc"]))
    lower = min(candidate_lbs) if candidate_lbs else None
    upper = min(candidate_ubs) if candidate_ubs else None
    reference = int(row["reference_makespan_cc"])
    row["candidate_oracle"] = {
        "lower_bound_cc": lower,
        "upper_bound_cc": upper,
        "loss_lower_bound_cc": max(0, lower - reference) if lower is not None else None,
        "loss_upper_bound_cc": max(0, upper - reference) if upper is not None else None,
        "evaluated_candidates": len(evaluations),
        "total_candidates": len(row["candidates"]),
    }


def audit_continuations(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    samples = select_samples(args)
    configuration = {
        "generator": args.generator,
        "source_split": args.source_split,
        "rank_limit": args.rank_limit,
        "bottom_count": args.bottom_count,
        "budget": args.budget,
        "states_per_stratum": args.states_per_stratum,
        "max_states": args.max_states,
        "max_cases_per_file": args.max_cases_per_file,
        "sample_min_remaining": args.sample_min_remaining,
        "sample_max_remaining": args.sample_max_remaining,
        "state_ids_from": (
            str(args.state_ids_from) if args.state_ids_from is not None else None
        ),
        "sample_require_r4_miss_r8_hit": args.sample_require_r4_miss_r8_hit,
        "time_limit_s": args.continuation_time_limit,
        "max_expansions": args.continuation_expansions,
        "target_gap": args.continuation_target_gap,
        "stop_on_zero": args.stop_on_zero,
        "candidate_revision": "direct-slot-conditional-cache-v8",
    }
    if args.resume and args.out.exists():
        report = json.loads(args.out.read_text())
        if report.get("configuration") != configuration:
            raise ValueError("resume configuration does not match existing output")
    else:
        report = {
            "schema": "scheduler_forced_continuation_audit_v0",
            "provisional": True,
            "configuration": configuration,
            "rows": [],
        }
    rows_by_id = {row["state_id"]: row for row in report["rows"]}

    time_limit = (
        None if args.continuation_time_limit < 0 else args.continuation_time_limit
    )
    expansions = (
        None if args.continuation_expansions < 0 else args.continuation_expansions
    )
    target_gap = (
        None if args.continuation_target_gap < 0 else args.continuation_target_gap
    )

    for index, sample in enumerate(samples, 1):
        state_id = f"{sample.case_key}:{sample.step}"
        state = sample.state
        if args.generator == "direct":
            generated = None
            structural = None
            bounded = generate_direct_candidates(
                state,
                rank_limit=args.rank_limit,
                bottom_count=args.bottom_count,
                budget=args.budget,
            )
        else:
            generated = gen_stage_actions(state.c2, state.c3, state.remaining)
            generated += gen_prefetch_actions(state.c2, state.c3, state.remaining)
            structural = filtered_actions(
                state,
                generated,
                rank_limit=args.rank_limit,
                bottom_count=args.bottom_count,
            )
            bounded = select_bounded_candidates(
                state,
                structural,
                budget=args.budget,
            )
        reference_child = apply_action(state, sample.reference_action)

        def reference_equivalent(child: BeamState) -> bool:
            if not child.remaining and not reference_child.remaining:
                return child.g_score == sample.reference_makespan
            return child.fingerprint() == reference_child.fingerprint()

        unique = {}
        for action in bounded:
            child = apply_action(state, action)
            key = (
                ("terminal", child.g_score)
                if not child.remaining
                else ("state", child.fingerprint())
            )
            unique.setdefault(key, (action, child))
        ordered = sorted(
            unique.values(),
            key=lambda pair: (
                int(not reference_equivalent(pair[1])),
                completion_estimate(pair[1]),
                pair[1].f_score,
                pair[1].g_score,
                action_id(pair[0]),
            ),
        )
        candidates = [
            {
                "id": action_id(action),
                "family": action_family(action),
                "child_lower_bound_cc": child.f_score,
                "action": serialize_action(action),
            }
            for action, child in ordered
        ]
        row = rows_by_id.get(state_id)
        if row is None:
            row = {
                "state_id": state_id,
                "case_key": sample.case_key,
                "step": sample.step,
                "e_total": sample.e_total,
                "mode": decision_mode(state),
                "remaining": len(state.remaining),
                "reference_family": action_family(sample.reference_action),
                "reference_makespan_cc": sample.reference_makespan,
                "generated_actions": len(generated) if generated is not None else None,
                "structural_actions": len(structural) if structural is not None else None,
                "bounded_actions": len(bounded),
                "unique_candidate_children": len(candidates),
                "candidates": candidates,
                "evaluations": {},
                "status": "in_progress",
            }
            report["rows"].append(row)
            rows_by_id[state_id] = row
        elif [candidate["id"] for candidate in row["candidates"]] != [
            candidate["id"] for candidate in candidates
        ]:
            raise ValueError(f"candidate set changed while resuming {state_id}")

        child_by_id = {action_id(action): child for action, child in ordered}
        update_oracle_summary(row)
        already_zero = row["candidate_oracle"]["loss_upper_bound_cc"] == 0
        for candidate in candidates:
            candidate_id = candidate["id"]
            if candidate_id in row["evaluations"]:
                continue
            if args.stop_on_zero and already_zero:
                break
            result = continuation_result(
                child_by_id[candidate_id],
                reference_child=reference_child,
                reference_makespan=sample.reference_makespan,
                time_limit_s=time_limit,
                max_expansions=expansions,
                target_gap=target_gap,
            )
            row["evaluations"][candidate_id] = result
            update_oracle_summary(row)
            already_zero = row["candidate_oracle"]["loss_upper_bound_cc"] == 0
            atomic_write_json(args.out, report)
            print(
                f"continuation {index}/{len(samples)} {state_id} "
                f"eval={len(row['evaluations'])}/{len(candidates)} "
                f"ub={result['upper_bound_cc']} lb={result['lower_bound_cc']} "
                f"oracle_loss_ub={row['candidate_oracle']['loss_upper_bound_cc']}",
                flush=True,
            )
        update_oracle_summary(row)
        if row["candidate_oracle"]["loss_upper_bound_cc"] == 0:
            row["status"] = "oracle_zero"
        elif len(row["evaluations"]) == len(candidates):
            row["status"] = "all_candidates_evaluated"
        else:
            row["status"] = "partial"
        clear_scheduler_caches()
        _equal_finish_left.cache_clear()
        _release_target_left.cache_clear()
        atomic_write_json(args.out, report)

    status_counts = Counter(row["status"] for row in report["rows"])
    report["summary"] = {
        "states": len(report["rows"]),
        "status_counts": dict(status_counts),
        "zero_oracle_loss_states": sum(
            row["candidate_oracle"]["loss_upper_bound_cc"] == 0
            for row in report["rows"]
        ),
        "runtime_s_this_invocation": time.perf_counter() - started,
    }
    atomic_write_json(args.out, report)
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.out}")
    return 0


def _load_jsonl_dataset(path: Path) -> tuple[dict | None, set[str], Counter]:
    meta = None
    entry_ids = set()
    kinds = Counter()
    if not path.exists():
        return meta, entry_ids, kinds
    with path.open() as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{lineno}") from exc
            kind = str(record.get("kind"))
            kinds[kind] += 1
            if kind == "meta":
                if meta is not None:
                    raise ValueError("future dataset contains multiple meta records")
                meta = record
            elif "entry_id" in record:
                entry_ids.add(str(record["entry_id"]))
    return meta, entry_ids, kinds


def _future_dataset_configuration(args: argparse.Namespace) -> dict:
    return {
        "schema": "scheduler_future_value_dataset_v0",
        "feature_revision": "integer-future-features-v1",
        "time_quantum_cc": TIME_QUANTUM_CC,
        "source_split": "discovery",
        "source_quality": "proven",
        "case_holdout_rule": "sha256(case_key) mod 5; zero is calibration",
        "candidate_policy": "R4+bottom2+residency,K32,direct-v8",
        "inputs": [str(path.resolve()) for path in args.inputs],
    }


def build_reference_future_dataset(args: argparse.Namespace) -> int:
    """Write exact optimal-path child values for every proven discovery case."""
    configuration = _future_dataset_configuration(args)
    final_path = args.dataset_out
    dataset_path = (
        final_path.with_suffix(final_path.suffix + ".rebuild.tmp")
        if args.rebuild_dataset else final_path
    )
    if args.rebuild_dataset and dataset_path.exists():
        dataset_path.unlink()
    meta, existing_ids, kinds = _load_jsonl_dataset(dataset_path)
    if meta is not None:
        if not args.resume:
            raise FileExistsError(
                f"{dataset_path} exists; pass --resume to continue it"
            )
        if meta.get("configuration") != configuration:
            raise ValueError("future dataset configuration does not match --resume")
    else:
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with dataset_path.open("w") as handle:
            handle.write(
                json.dumps(
                    {"kind": "meta", "configuration": configuration},
                    separators=(",", ":"),
                )
                + "\n"
            )

    started = time.perf_counter()
    cases = written = skipped = 0
    role_counts = Counter()
    residual_q = []
    with dataset_path.open("a") as handle:
        for path in args.inputs:
            results = json.loads(path.read_text())["results"]
            eligible_in_file = 0
            for key, item in results.items():
                if item.get("dataset_split") != "discovery" or not quality_ok(
                    item, "proven"
                ):
                    continue
                if 0 <= args.max_cases_per_file <= eligible_in_file:
                    break
                eligible_in_file += 1
                case_key = f"E{item['e_total']}:{key}"
                dist = {int(eid): int(ntok) for eid, ntok in item["dist"].items()}
                scheduler = FourStageScheduler(
                    dist,
                    initial_cache_c2=int(item.get("initial_cache_c2", -1)),
                    initial_cache_c3=int(item.get("initial_cache_c3", -1)),
                )
                state = scheduler._initial_state()
                target = int(item["makespan_cc"])
                role = "calibration" if stable_u64(case_key) % 5 == 0 else "fit"
                buffer = []
                for step, raw in enumerate(item["actions"]):
                    action = deserialize_action(raw)
                    child = apply_action(state, action)
                    entry_id = f"reference:{case_key}:{step}"
                    if entry_id in existing_ids:
                        skipped += 1
                    else:
                        base, features, _ = future_features(child)
                        if base > target:
                            raise RuntimeError(
                                f"admissible base exceeds proven target at {entry_id}: "
                                f"{base} > {target}"
                            )
                        residual = target - base
                        record = {
                            "kind": "reference",
                            "entry_id": entry_id,
                            "case_key": case_key,
                            "e_total": int(item["e_total"]),
                            "step": step,
                            "role": role,
                            "mode_before": decision_mode(state),
                            "remaining_before": len(state.remaining),
                            "remaining_after": len(child.remaining),
                            "action_family": action_family(action),
                            "target_final_cc": target,
                            "base_cc": base,
                            "target_residual_cc": residual,
                            "target_residual_q": _cycles_to_quanta(residual),
                            "features": features,
                        }
                        buffer.append(json.dumps(record, separators=(",", ":")))
                        existing_ids.add(entry_id)
                        written += 1
                        role_counts[role] += 1
                        residual_q.append(record["target_residual_q"])
                    state = child
                if state.remaining or state.g_score != target:
                    raise RuntimeError(f"reference replay mismatch at {case_key}")
                if buffer:
                    handle.write("\n".join(buffer) + "\n")
                    handle.flush()
                cases += 1
                clear_scheduler_caches()
                _equal_finish_left.cache_clear()
                _release_target_left.cache_clear()
                if args.progress_every > 0 and cases % args.progress_every == 0:
                    print(
                        f"future-reference cases={cases} written={written} "
                        f"skipped={skipped} elapsed_s={time.perf_counter()-started:.1f}",
                        flush=True,
                    )
    summary = {
        "cases": cases,
        "written_this_run": written,
        "skipped_existing": skipped,
        "existing_kind_counts_before": dict(kinds),
        "roles_written": dict(role_counts),
        "residual_q_min": min(residual_q) if residual_q else None,
        "residual_q_max": max(residual_q) if residual_q else None,
        "runtime_s": time.perf_counter() - started,
    }
    print(json.dumps(summary, indent=2))
    if args.rebuild_dataset:
        os.replace(dataset_path, final_path)
    print(f"wrote {final_path}")
    return 0


def _seed_continuation_results(paths: list[Path]) -> dict[tuple[str, str], tuple[dict, str]]:
    seeded = {}
    for path in paths:
        report = json.loads(path.read_text())
        for row in report.get("rows", []):
            state_id = str(row["state_id"])
            evaluations = row.get("evaluations", {})
            for candidate in row.get("candidates", []):
                candidate_id = str(candidate["id"])
                if candidate_id in evaluations:
                    seeded.setdefault(
                        (state_id, candidate_id),
                        (evaluations[candidate_id], str(path)),
                    )
            if "selected_action" in row and "selected_value" in row:
                selected = deserialize_action(row["selected_action"])
                seeded.setdefault(
                    (state_id, action_id(selected)),
                    (row["selected_value"], str(path)),
                )
    return seeded


def build_counterfactual_future_dataset(args: argparse.Namespace) -> int:
    """Append one forced-continuation label for every direct candidate."""
    configuration = _future_dataset_configuration(args)
    meta, existing_ids, kinds = _load_jsonl_dataset(args.dataset_out)
    if meta is None:
        raise FileNotFoundError(
            "reference future dataset is missing; run --dataset-phase reference first"
        )
    if meta.get("configuration") != configuration:
        raise ValueError("future dataset configuration does not match counterfactual run")
    samples = select_samples(args)
    seeded = _seed_continuation_results(args.seed_report)
    time_limit = None if args.continuation_time_limit < 0 else args.continuation_time_limit
    expansions = None if args.continuation_expansions < 0 else args.continuation_expansions
    target_gap = None if args.continuation_target_gap < 0 else args.continuation_target_gap
    started = time.perf_counter()
    counters = Counter()

    with args.dataset_out.open("a") as handle:
        for state_index, sample in enumerate(samples, 1):
            state_id = f"{sample.case_key}:{sample.step}"
            reference_child = apply_action(sample.state, sample.reference_action)
            actions = generate_direct_candidates(
                sample.state,
                rank_limit=args.rank_limit,
                bottom_count=args.bottom_count,
                budget=args.budget,
            )
            prepared = []
            result_by_child = {}
            for candidate_index, action in enumerate(actions):
                candidate_id = action_id(action)
                child = apply_action(sample.state, action)
                child_identity = (
                    ("terminal", child.g_score)
                    if not child.remaining else ("state", child.fingerprint())
                )
                child_key = hashlib.sha256(repr(child_identity).encode()).hexdigest()[:20]
                base, features, _ = future_features(child)
                seed = seeded.get((state_id, candidate_id))
                if seed is not None:
                    result_by_child.setdefault(child_identity, seed)
                prepared.append(
                    (
                        candidate_index,
                        candidate_id,
                        action,
                        child,
                        child_identity,
                        child_key,
                        base,
                        features,
                    )
                )

            buffer = []
            for (
                candidate_index,
                candidate_id,
                action,
                child,
                child_identity,
                child_key,
                base,
                features,
            ) in prepared:
                entry_id = f"counterfactual:{state_id}:{candidate_id}"
                if entry_id in existing_ids:
                    counters["existing"] += 1
                    continue
                cached = result_by_child.get(child_identity)
                if cached is not None:
                    result, source = cached
                    source_kind = "seed_report" if source != "same_child" else source
                    counters[source_kind] += 1
                else:
                    result = continuation_result(
                        child,
                        reference_child=reference_child,
                        reference_makespan=sample.reference_makespan,
                        time_limit_s=time_limit,
                        max_expansions=expansions,
                        target_gap=target_gap,
                    )
                    source = "search"
                    source_kind = "search"
                    counters[source_kind] += 1
                    result_by_child[child_identity] = (result, "same_child")
                record = {
                    "kind": "counterfactual",
                    "entry_id": entry_id,
                    "state_id": state_id,
                    "case_key": sample.case_key,
                    "e_total": sample.e_total,
                    "step": sample.step,
                    "role": (
                        "calibration"
                        if stable_u64(sample.case_key) % 5 == 0 else "fit"
                    ),
                    "mode_before": decision_mode(sample.state),
                    "remaining_before": len(sample.state.remaining),
                    "candidate_index": candidate_index,
                    "candidate_id": candidate_id,
                    "child_key": child_key,
                    "action_family": action_family(action),
                    "base_cc": base,
                    "features": features,
                    "reference_makespan_cc": sample.reference_makespan,
                    "value_lower_bound_cc": int(result["lower_bound_cc"]),
                    "value_upper_bound_cc": int(result["upper_bound_cc"]),
                    "value_proven_optimal": bool(result["proven_optimal"]),
                    "value_gap": float(result["optimality_gap"]),
                    "label_source": source_kind,
                    "label_source_detail": source,
                }
                buffer.append(json.dumps(record, separators=(",", ":")))
                existing_ids.add(entry_id)
                counters["written"] += 1
            if buffer:
                handle.write("\n".join(buffer) + "\n")
                handle.flush()
            clear_scheduler_caches()
            _equal_finish_left.cache_clear()
            _release_target_left.cache_clear()
            print(
                f"future-counterfactual {state_index}/{len(samples)} {state_id} "
                f"K={len(actions)} written={counters['written']} "
                f"searched={counters['search']} elapsed_s={time.perf_counter()-started:.1f}",
                flush=True,
            )
    summary = {
        "states": len(samples),
        "kind_counts_before": dict(kinds),
        **dict(counters),
        "seed_entries_loaded": len(seeded),
        "runtime_s": time.perf_counter() - started,
    }
    print(json.dumps(summary, indent=2))
    print(f"updated {args.dataset_out}")
    return 0


def _r8_ranking_dataset_configuration(args: argparse.Namespace) -> dict:
    return {
        "schema": "scheduler_r8_ranking_dataset_v1",
        "feature_revision": "rtl-aggregate-ranker-v1",
        "time_unit_cc": R8_RANKER_TIME_UNIT_CC,
        "source_split": args.source_split,
        "source_quality": "proven",
        "case_holdout_rule": "sha256(case_key) mod 5; zero is calibration",
        "candidate_policy": "R8+residency,K32,direct-v8",
        "ranking_state_source": args.ranking_state_source,
        "ranking_cases_per_e": args.ranking_cases_per_e,
        "states_per_stratum": args.states_per_stratum,
        "max_states": args.max_states,
        "max_cases_per_file": args.max_cases_per_file,
        "sample_min_remaining": args.sample_min_remaining,
        "sample_max_remaining": args.sample_max_remaining,
        "continuation_time_limit_s": args.continuation_time_limit,
        "continuation_expansions": args.continuation_expansions,
        "continuation_target_gap": args.continuation_target_gap,
        "inputs": [str(path.resolve()) for path in args.inputs],
    }


def _unrestricted_continuation_result(
    child: BeamState,
    *,
    time_limit_s: float | None,
    max_expansions: int | None,
    target_gap: float | None,
) -> dict:
    """Bound Q(child) without assuming that child lies on a reference path."""
    if not child.remaining:
        return {
            "upper_bound_cc": child.g_score,
            "lower_bound_cc": child.g_score,
            "proven_optimal": True,
            "optimality_gap": 0.0,
            "expansions": 0,
            "generated": 0,
            "runtime_s": 0.0,
            "termination": "terminal_action",
        }
    bounded_incumbents = []
    for rank_limit, bottom_count in ((8, 0), (4, 2)):
        terminal, _, _ = run_integer_policy(
            child,
            weights={},
            scorer_kind="lpt-estimate",
            tie_kind="rem-snap",
            rank_limit=rank_limit,
            bottom_count=bottom_count,
            budget=32,
        )
        bounded_incumbents.append(terminal)
    incumbent = min(bounded_incumbents, key=lambda state: state.g_score)
    scheduler = FourStageScheduler(dict(child.remaining), enable_prefetch=True)
    result = scheduler.run_anytime(
        time_limit_s=time_limit_s,
        max_expansions=max_expansions,
        target_gap=target_gap,
        initial_state=child,
        incumbent_state=incumbent,
    )
    if result.lower_bound > result.makespan:
        raise RuntimeError("continuation lower bound exceeds upper bound")
    return {
        "upper_bound_cc": result.makespan,
        "lower_bound_cc": result.lower_bound,
        "proven_optimal": result.proven_optimal,
        "optimality_gap": result.optimality_gap,
        "expansions": result.expansions,
        "generated": result.generated,
        "runtime_s": result.runtime_s,
        "termination": result.termination,
    }


def build_r8_ranking_dataset(args: argparse.Namespace) -> int:
    """Generate interval-valued Q labels for every candidate in each state."""
    configuration = _r8_ranking_dataset_configuration(args)
    meta, existing_ids, kinds = _load_jsonl_dataset(args.dataset_out)
    if meta is None:
        args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
        with args.dataset_out.open("w") as handle:
            handle.write(
                json.dumps(
                    {"kind": "meta", "configuration": configuration},
                    separators=(",", ":"),
                )
                + "\n"
            )
    else:
        if not args.resume:
            raise FileExistsError(
                f"{args.dataset_out} exists; pass --resume to continue it"
            )
        if meta.get("configuration") != configuration:
            raise ValueError("R8 ranking dataset configuration changed on resume")

    samples = select_ranking_states(args)
    time_limit = None if args.continuation_time_limit < 0 else args.continuation_time_limit
    expansions = None if args.continuation_expansions < 0 else args.continuation_expansions
    target_gap = None if args.continuation_target_gap < 0 else args.continuation_target_gap
    counters = Counter()
    started = time.perf_counter()
    with args.dataset_out.open("a") as handle:
        for state_index, sample in enumerate(samples, 1):
            actions = generate_direct_candidates(
                sample.state, rank_limit=8, bottom_count=0, budget=32
            )
            prepared = []
            result_by_child = {}
            for candidate_index, action in enumerate(actions):
                candidate_id = action_id(action)
                entry_id = f"r8_candidate:{sample.state_id}:{candidate_id}"
                child = apply_action(sample.state, action)
                identity = (
                    ("terminal", child.g_score)
                    if not child.remaining
                    else ("state", child.fingerprint())
                )
                base_q, terms = r8_ranker_features(sample.state, action, child)
                prepared.append(
                    (
                        candidate_index,
                        candidate_id,
                        entry_id,
                        action,
                        child,
                        identity,
                        base_q,
                        terms,
                    )
                )

            buffer = []
            for (
                candidate_index,
                candidate_id,
                entry_id,
                action,
                child,
                identity,
                base_q,
                terms,
            ) in prepared:
                if entry_id in existing_ids:
                    counters["existing"] += 1
                    continue
                if identity in result_by_child:
                    result = result_by_child[identity]
                    label_source = "same_child"
                    counters["same_child"] += 1
                else:
                    result = _unrestricted_continuation_result(
                        child,
                        time_limit_s=time_limit,
                        max_expansions=expansions,
                        target_gap=target_gap,
                    )
                    result_by_child[identity] = result
                    label_source = "search"
                    counters["search"] += 1
                if int(result["upper_bound_cc"]) < sample.reference_makespan:
                    raise RuntimeError(
                        f"candidate undercut proven root at {sample.state_id}"
                    )
                record = {
                    "kind": "r8_candidate",
                    "entry_id": entry_id,
                    "state_id": sample.state_id,
                    "state_source": sample.source,
                    "case_key": sample.case_key,
                    "e_total": sample.e_total,
                    "step": sample.step,
                    "role": (
                        "calibration"
                        if stable_u64(sample.case_key) % 5 == 0
                        else "fit"
                    ),
                    "mode_before": decision_mode(sample.state),
                    "remaining_before": len(sample.state.remaining),
                    "trajectory_regret_cc": sample.trajectory_regret_cc,
                    "candidate_index": candidate_index,
                    "candidate_id": candidate_id,
                    "action_family": action_family(action),
                    "base_q": base_q,
                    "terms": terms,
                    "value_lower_bound_cc": int(result["lower_bound_cc"]),
                    "value_upper_bound_cc": int(result["upper_bound_cc"]),
                    "value_proven_optimal": bool(result["proven_optimal"]),
                    "value_gap": float(result["optimality_gap"]),
                    "label_source": label_source,
                }
                buffer.append(json.dumps(record, separators=(",", ":")))
                existing_ids.add(entry_id)
                counters["written"] += 1
            if buffer:
                handle.write("\n".join(buffer) + "\n")
                handle.flush()
            clear_scheduler_caches()
            _equal_finish_left.cache_clear()
            _release_target_left.cache_clear()
            print(
                f"r8-ranking {state_index}/{len(samples)} {sample.state_id} "
                f"K={len(actions)} written={counters['written']} "
                f"searched={counters['search']} "
                f"elapsed_s={time.perf_counter()-started:.1f}",
                flush=True,
            )
    print(
        json.dumps(
            {
                "states": len(samples),
                "kind_counts_before": dict(kinds),
                **dict(counters),
                "runtime_s": time.perf_counter() - started,
            },
            indent=2,
        )
    )
    print(f"updated {args.dataset_out}")
    return 0


def _load_r8_ranking_groups(dataset_path: Path) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    meta = None
    with dataset_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("kind") == "meta":
                meta = row
            elif row.get("kind") == "r8_candidate":
                groups[str(row["state_id"])].append(row)
    if meta is None or meta.get("configuration", {}).get("schema") != (
        "scheduler_r8_ranking_dataset_v1"
    ):
        raise ValueError("not an R8 ranking dataset v1")
    for state_id, rows in groups.items():
        rows.sort(key=lambda row: int(row["candidate_index"]))
        indices = [int(row["candidate_index"]) for row in rows]
        if indices != list(range(len(rows))):
            raise ValueError(f"incomplete R8 candidate group {state_id}")
        for field in ("role", "mode_before", "case_key", "state_source"):
            if len({row[field] for row in rows}) != 1:
                raise ValueError(f"inconsistent {field} in {state_id}")
    return dict(groups)


def _r8_ranker_score(
    row: dict, weights_by_mode: dict[str, dict[str, int]]
) -> int:
    mode = str(row["mode_before"])
    weights = weights_by_mode.get(mode, {})
    correction = sum(
        int(weight) * int(row["terms"][term])
        for term, weight in weights.items()
    )
    return int(row["base_q"]) * 16 + correction


def _evaluate_r8_ranker(
    groups: dict[str, list[dict]],
    weights_by_mode: dict[str, dict[str, int]],
    *,
    role: str,
    mode_filter: str | None = None,
) -> dict:
    selections = []
    for state_id, rows in groups.items():
        if rows[0]["role"] != role:
            continue
        mode = str(rows[0]["mode_before"])
        if mode_filter is not None and mode != mode_filter:
            continue
        best_upper = min(int(row["value_upper_bound_cc"]) for row in rows)
        oracle_lower = min(int(row["value_lower_bound_cc"]) for row in rows)
        selected = min(
            rows,
            key=lambda row: (
                _r8_ranker_score(row, weights_by_mode),
                int(row["candidate_index"]),
            ),
        )
        selected_lower = int(selected["value_lower_bound_cc"])
        selected_upper = int(selected["value_upper_bound_cc"])
        certified_q = _cycles_to_quanta(max(0, selected_lower - best_upper))
        feasible_q = _cycles_to_quanta(max(0, selected_upper - best_upper))
        selections.append(
            {
                "state_id": state_id,
                "mode": mode,
                "candidate_index": int(selected["candidate_index"]),
                "candidate_id": selected["candidate_id"],
                "score": _r8_ranker_score(selected, weights_by_mode),
                "certified_regret_q": certified_q,
                "feasible_regret_q": feasible_q,
                "known_best_upper": selected_upper == best_upper,
                "oracle_interval_q": _cycles_to_quanta(best_upper - oracle_lower),
            }
        )
    certified = [row["certified_regret_q"] for row in selections]
    feasible = [row["feasible_regret_q"] for row in selections]
    summary = {
        "states": len(selections),
        "known_best_upper_states": sum(row["known_best_upper"] for row in selections),
        "certified_suboptimal_states": sum(value > 0 for value in certified),
        "certified_regret_sum_q": sum(certified),
        "feasible_regret_sum_q": sum(feasible),
        "feasible_regret_max_q": max(feasible, default=0),
        "oracle_interval_sum_q": sum(
            row["oracle_interval_q"] for row in selections
        ),
    }
    summary["objective"] = [
        summary["certified_regret_sum_q"],
        summary["certified_suboptimal_states"],
        summary["feasible_regret_sum_q"],
        summary["states"] - summary["known_best_upper_states"],
        summary["feasible_regret_max_q"],
    ]
    return {"summary": summary, "selections": selections}


def _ranker_complexity(weights_by_mode: dict[str, dict[str, int]]) -> list[int]:
    flat = [
        weight
        for weights in weights_by_mode.values()
        for weight in weights.values()
    ]
    return [sum(weight != 0 for weight in flat), sum(abs(weight) for weight in flat)]


def _fit_r8_mode_weights(
    groups: dict[str, list[dict]], profile_terms: tuple[str, ...], mode: str
) -> dict[str, int]:
    weights = {term: 0 for term in profile_terms}
    weights_by_mode = {mode: dict(weights)}
    current_eval = _evaluate_r8_ranker(
        groups, weights_by_mode, role="fit", mode_filter=mode
    )
    current_key = tuple(current_eval["summary"]["objective"] + _ranker_complexity(weights_by_mode))
    for _ in range(max(1, len(profile_terms) * 3)):
        best = None
        for term in profile_terms:
            for value in R8_RANKER_WEIGHT_DOMAIN:
                if value == weights[term]:
                    continue
                trial = dict(weights)
                trial[term] = value
                trial_by_mode = {mode: trial}
                evaluated = _evaluate_r8_ranker(
                    groups, trial_by_mode, role="fit", mode_filter=mode
                )
                key = tuple(
                    evaluated["summary"]["objective"]
                    + _ranker_complexity(trial_by_mode)
                )
                candidate = (key, term, value, trial, evaluated)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        if best is None or best[0] >= current_key:
            break
        current_key, _, _, weights, current_eval = best
    return weights


def fit_r8_ranking_scorer(args: argparse.Namespace) -> int:
    """Fit mode-selected shift/add weights against state-wise candidate regret."""
    groups = _load_r8_ranking_groups(args.dataset_out)
    if not groups:
        raise ValueError("R8 ranking dataset is empty")
    reports = []
    for profile, terms in R8_RANKER_PROFILES.items():
        weights_by_mode = {
            mode: _fit_r8_mode_weights(groups, terms, mode)
            for mode in R8_RANKER_MODES
        }
        fit = _evaluate_r8_ranker(groups, weights_by_mode, role="fit")
        calibration = _evaluate_r8_ranker(
            groups, weights_by_mode, role="calibration"
        )
        report = {
            "profile": profile,
            "terms": list(terms),
            "weights_by_mode": weights_by_mode,
            "complexity": _ranker_complexity(weights_by_mode),
            "fit": fit,
            "calibration": calibration,
        }
        reports.append(report)
        print(
            f"r8-ranker-fit {profile} complexity={report['complexity']} "
            f"fit={fit['summary']['objective']} "
            f"cal={calibration['summary']['objective']}",
            flush=True,
        )
    selected = min(
        reports,
        key=lambda report: tuple(
            report["calibration"]["summary"]["objective"]
            + report["complexity"]
        ),
    )
    output = {
        "schema": "scheduler_r8_integer_ranker_v1",
        "dataset": str(args.dataset_out),
        "state_count": len(groups),
        "candidate_policy": "R8+residency,K32,direct-v8",
        "base_scale": 16,
        "base_time_unit_cc": R8_RANKER_TIME_UNIT_CC,
        "coefficient_domain": list(R8_RANKER_WEIGHT_DOMAIN),
        "profiles": reports,
        "selected_profile": selected["profile"],
        "selected_weights_by_mode": selected["weights_by_mode"],
    }
    atomic_write_json(args.model_out, output)
    print(
        json.dumps(
            {
                "selected_profile": selected["profile"],
                "selected_weights_by_mode": selected["weights_by_mode"],
                "complexity": selected["complexity"],
                "fit": selected["fit"]["summary"],
                "calibration": selected["calibration"]["summary"],
            },
            indent=2,
        )
    )
    print(f"wrote {args.model_out}")
    return 0


def load_r8_ranker_model(model_path: Path) -> dict[str, dict[str, int]]:
    report = json.loads(model_path.read_text())
    if report.get("schema") != "scheduler_r8_integer_ranker_v1":
        raise ValueError(f"unsupported R8 ranker model: {model_path}")
    raw = report["selected_weights_by_mode"]
    weights = {
        str(mode): {str(term): int(value) for term, value in values.items()}
        for mode, values in raw.items()
    }
    allowed_terms = set(R8_RANKER_PROFILES[str(report["selected_profile"])])
    for mode in R8_RANKER_MODES:
        if set(weights.get(mode, {})) != allowed_terms:
            raise ValueError(f"R8 ranker terms changed for mode {mode}")
        if any(
            value not in R8_RANKER_WEIGHT_DOMAIN
            for value in weights[mode].values()
        ):
            raise ValueError(f"R8 ranker coefficient outside domain for {mode}")
    return weights


# Each correction coefficient is an integer in eighths of one time quantum.
# The sign domains encode physical monotonicity: longer release/DMA tails and
# duplicate cache copies cannot improve the score, while useful unique cache
# residency cannot make it worse.  This is deliberately a small RTL-oriented
# hypothesis family, not an unconstrained regression model.
SCORER_MODEL_TERMS = {
    "base": (),
    "timing": ("release", "dma"),
    "timing-cache-count": ("release", "dma", "duplicate", "ready_count"),
    "timing-cache-token": ("release", "dma", "duplicate", "ready_token8"),
    "timing-cache-hybrid": (
        "release",
        "dma",
        "duplicate",
        "ready_count",
        "ready_token8",
    ),
}
SCORER_WEIGHT_DOMAINS = {
    "release": (0, 1, 2, 4, 8),
    "dma": (0, 1, 2, 4, 8),
    "duplicate": (0, 1, 2, 4, 8),
    "ready_count": (0, -1, -2, -4, -8),
    "ready_token8": (0, -1, -2, -4, -8),
}


def _scorer_term_values(features: dict[str, int]) -> dict[str, int]:
    return {
        "release": int(features["release_gap_q"]),
        "dma": int(features["dma_busy_tail_q"]),
        "duplicate": int(features["duplicate_s1_ready"])
        + int(features["duplicate_full_ready"]),
        "ready_count": int(features["unique_s1_ready"])
        + int(features["unique_full_ready"]),
        # The only division in the model is a fixed right shift.  Round up so
        # a small but nonzero ready expert is not erased before calibration.
        "ready_token8": (
            int(features["s1_ready_tokens"])
            + int(features["full_ready_tokens"])
            + 7
        )
        >> 3,
    }


def _load_counterfactual_groups(dataset_path: Path) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    with dataset_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("kind") == "counterfactual":
                row["term_values"] = _scorer_term_values(row["features"])
                groups[str(row["state_id"])].append(row)
    for state_id, rows in groups.items():
        rows.sort(key=lambda row: int(row["candidate_index"]))
        roles = {row["role"] for row in rows}
        references = {int(row["reference_makespan_cc"]) for row in rows}
        if len(roles) != 1 or len(references) != 1:
            raise ValueError(f"inconsistent counterfactual group {state_id}")
        candidate_indices = [int(row["candidate_index"]) for row in rows]
        if candidate_indices != list(range(len(rows))):
            raise ValueError(f"incomplete counterfactual group {state_id}")
    return dict(groups)


def _integer_score(row: dict, weights: dict[str, int]) -> int:
    correction = sum(
        int(weight) * int(row["term_values"][term])
        for term, weight in weights.items()
    )
    return int(row["base_cc"]) * 8 + TIME_QUANTUM_CC * correction


def _evaluate_integer_scorer(
    groups: dict[str, list[dict]], weights: dict[str, int], role: str
) -> dict:
    selected_rows = []
    for state_id, rows in groups.items():
        if rows[0]["role"] != role:
            continue
        selected = min(
            rows,
            key=lambda row: (
                _integer_score(row, weights), int(row["candidate_index"])
            ),
        )
        reference = int(selected["reference_makespan_cc"])
        lower_loss = max(0, int(selected["value_lower_bound_cc"]) - reference)
        upper_loss = max(0, int(selected["value_upper_bound_cc"]) - reference)
        selected_rows.append(
            {
                "state_id": state_id,
                "candidate_index": int(selected["candidate_index"]),
                "candidate_id": selected["candidate_id"],
                "action_family": selected["action_family"],
                "score": _integer_score(selected, weights),
                "lower_loss_cc": lower_loss,
                "upper_loss_cc": upper_loss,
                "known_optimal": upper_loss == 0,
                "certified_suboptimal": lower_loss > 0,
            }
        )
    lower_q = [_cycles_to_quanta(row["lower_loss_cc"]) for row in selected_rows]
    upper_q = [_cycles_to_quanta(row["upper_loss_cc"]) for row in selected_rows]
    summary = {
        "states": len(selected_rows),
        "known_optimal_states": sum(row["known_optimal"] for row in selected_rows),
        "certified_suboptimal_states": sum(
            row["certified_suboptimal"] for row in selected_rows
        ),
        "lower_loss_sum_q": sum(lower_q),
        "upper_loss_sum_q": sum(upper_q),
        "upper_loss_max_q": max(upper_q, default=0),
    }
    # Training/calibration both use the same evidence ordering.  Proven lower
    # losses dominate, followed by candidates actually completed at the known
    # reference optimum; feasible but loose upper bounds are only tertiary.
    summary["objective"] = [
        summary["certified_suboptimal_states"],
        summary["lower_loss_sum_q"],
        summary["states"] - summary["known_optimal_states"],
        summary["upper_loss_sum_q"],
        summary["upper_loss_max_q"],
    ]
    return {"summary": summary, "selections": selected_rows}


def fit_integer_scorer(args: argparse.Namespace) -> int:
    """Fit a bounded shift/add scorer and select its fixed feature profile."""
    groups = _load_counterfactual_groups(args.dataset_out)
    if not groups:
        raise ValueError("counterfactual dataset is empty")
    expected_states = len(select_samples(args)) if args.state_ids_from else None
    if expected_states is not None and len(groups) != expected_states:
        raise ValueError(
            f"counterfactual dataset has {len(groups)} states, expected {expected_states}"
        )

    profile_reports = []
    for profile, terms in SCORER_MODEL_TERMS.items():
        domains = [SCORER_WEIGHT_DOMAINS[term] for term in terms]
        best = None
        configurations = itertools.product(*domains) if domains else [()]
        for values in configurations:
            weights = dict(zip(terms, values))
            fit = _evaluate_integer_scorer(groups, weights, "fit")
            complexity = [
                sum(weight != 0 for weight in weights.values()),
                sum(abs(weight) for weight in weights.values()),
            ]
            key = tuple(fit["summary"]["objective"] + complexity)
            if best is None or key < best[0]:
                best = (key, weights, fit)
        assert best is not None
        _, weights, fit = best
        calibration = _evaluate_integer_scorer(groups, weights, "calibration")
        profile_reports.append(
            {
                "profile": profile,
                "terms": list(terms),
                "weights_eighths": weights,
                "fit": fit,
                "calibration": calibration,
            }
        )
        print(
            f"scorer-fit {profile} weights={weights} "
            f"fit={fit['summary']['objective']} "
            f"cal={calibration['summary']['objective']}",
            flush=True,
        )

    # Calibration chooses only among the five profiles declared above; it
    # never changes a coefficient fitted on the fit partition.
    selected = min(
        profile_reports,
        key=lambda report: tuple(
            report["calibration"]["summary"]["objective"]
            + [
                sum(
                    weight != 0
                    for weight in report["weights_eighths"].values()
                ),
                sum(abs(weight) for weight in report["weights_eighths"].values()),
            ]
        ),
    )
    report = {
        "schema": "scheduler_integer_scorer_fit_v0",
        "dataset": str(args.dataset_out),
        "state_count": len(groups),
        "coefficient_unit": "one_eighth_of_11264_cycles",
        "selection_rule": (
            "fit chooses coefficients per fixed profile; calibration chooses profile"
        ),
        "profiles": profile_reports,
        "selected_profile": selected["profile"],
        "selected_weights_eighths": selected["weights_eighths"],
    }
    atomic_write_json(args.model_out, report)
    print(json.dumps({
        "selected_profile": report["selected_profile"],
        "selected_weights_eighths": report["selected_weights_eighths"],
        "fit": selected["fit"]["summary"],
        "calibration": selected["calibration"]["summary"],
    }, indent=2))
    print(f"wrote {args.model_out}")
    return 0


def load_selected_integer_model(
    model_path: Path, requested_profile: str | None = None
) -> tuple[str, dict[str, int]]:
    report = json.loads(model_path.read_text())
    if report.get("schema") != "scheduler_integer_scorer_fit_v0":
        raise ValueError(f"unsupported scorer model in {model_path}")
    profile = requested_profile or str(report["selected_profile"])
    if requested_profile is None:
        raw_weights = report["selected_weights_eighths"]
    else:
        matches = [
            item for item in report["profiles"] if item["profile"] == profile
        ]
        if len(matches) != 1:
            raise ValueError(f"model report does not contain profile {profile}")
        raw_weights = matches[0]["weights_eighths"]
    weights = {str(term): int(weight) for term, weight in raw_weights.items()}
    expected_terms = set(SCORER_MODEL_TERMS[profile])
    if set(weights) != expected_terms:
        raise ValueError(f"selected scorer terms do not match profile {profile}")
    for term, weight in weights.items():
        if weight not in SCORER_WEIGHT_DOMAINS[term]:
            raise ValueError(f"coefficient {term}={weight} is outside shift/add set")
    return profile, weights


def audit_base_scorer(args: argparse.Namespace) -> int:
    """Measure scorer loss after locking candidate-oracle loss separately."""
    started = time.perf_counter()
    samples = select_samples(args)
    model_profile = None
    model_weights = None
    if args.scorer == "integer-model-v0":
        model_profile, model_weights = load_selected_integer_model(
            args.model_out, args.model_profile
        )
    configuration = {
        "scorer_revision": args.scorer,
        "source_split": args.source_split,
        "candidate_revision": "direct-slot-conditional-cache-v8",
        "rank_limit": args.rank_limit,
        "bottom_count": args.bottom_count,
        "budget": args.budget,
        "state_ids_from": (
            str(args.state_ids_from) if args.state_ids_from is not None else None
        ),
        "time_limit_s": args.continuation_time_limit,
        "max_expansions": args.continuation_expansions,
        "target_gap": args.continuation_target_gap,
        "model_profile": model_profile,
        "model_weights_eighths": model_weights,
    }
    if args.resume and args.out.exists():
        report = json.loads(args.out.read_text())
        if report.get("configuration") != configuration:
            raise ValueError("scorer resume configuration does not match output")
    else:
        report = {
            "schema": "scheduler_base_scorer_audit_v0",
            "provisional": True,
            "configuration": configuration,
            "rows": [],
        }
    rows_by_id = {row["state_id"]: row for row in report["rows"]}
    time_limit = None if args.continuation_time_limit < 0 else args.continuation_time_limit
    expansions = None if args.continuation_expansions < 0 else args.continuation_expansions
    target_gap = None if args.continuation_target_gap < 0 else args.continuation_target_gap

    for index, sample in enumerate(samples, 1):
        state_id = f"{sample.case_key}:{sample.step}"
        if state_id in rows_by_id:
            continue
        actions = generate_direct_candidates(
            sample.state,
            rank_limit=args.rank_limit,
            bottom_count=args.bottom_count,
            budget=args.budget,
        )
        scored = []
        for candidate_index, action in enumerate(actions):
            child = apply_action(sample.state, action)
            base, features, _ = future_features(child)
            if args.scorer == "base-index-v0":
                scorer_key = (base, candidate_index)
            elif args.scorer == "base-release-tie-v0":
                scorer_key = (
                    base,
                    features["release_gap_q"],
                    features["dma_busy_tail_q"],
                    -features["unique_s1_ready"],
                    features["duplicate_s1_ready"],
                    candidate_index,
                )
            elif args.scorer == "base-plus-release-eighth-v0":
                scorer_key = (
                    base * 8
                    + features["release_gap_q"] * TIME_QUANTUM_CC,
                    features["dma_busy_tail_q"],
                    -features["unique_s1_ready"],
                    features["duplicate_s1_ready"],
                    candidate_index,
                )
            elif args.scorer == "integer-model-v0":
                assert model_weights is not None
                scorer_row = {
                    "base_cc": base,
                    "term_values": _scorer_term_values(features),
                }
                scorer_key = (
                    _integer_score(scorer_row, model_weights),
                    candidate_index,
                )
            else:
                raise AssertionError(args.scorer)
            scored.append(
                (scorer_key, base, candidate_index, action, child, features)
            )
        if not scored:
            raise RuntimeError(f"direct generator produced no candidate at {state_id}")
        scorer_key, base, candidate_index, action, child, features = min(
            scored, key=lambda item: item[0]
        )
        reference_child = apply_action(sample.state, sample.reference_action)
        result = continuation_result(
            child,
            reference_child=reference_child,
            reference_makespan=sample.reference_makespan,
            time_limit_s=time_limit,
            max_expansions=expansions,
            target_gap=target_gap,
        )
        row = {
            "state_id": state_id,
            "case_key": sample.case_key,
            "step": sample.step,
            "e_total": sample.e_total,
            "mode": decision_mode(sample.state),
            "remaining": len(sample.state.remaining),
            "candidate_count": len(actions),
            "selected_candidate_index": candidate_index,
            "selected_family": action_family(action),
            "selected_action": serialize_action(action),
            "score_key": list(scorer_key),
            "score_base_cc": base,
            "score_features": features,
            "reference_makespan_cc": sample.reference_makespan,
            "selected_value": result,
            "scorer_loss_lower_bound_cc": max(
                0, int(result["lower_bound_cc"]) - sample.reference_makespan
            ),
            "scorer_loss_upper_bound_cc": max(
                0, int(result["upper_bound_cc"]) - sample.reference_makespan
            ),
        }
        report["rows"].append(row)
        rows_by_id[state_id] = row
        atomic_write_json(args.out, report)
        clear_scheduler_caches()
        _equal_finish_left.cache_clear()
        _release_target_left.cache_clear()
        print(
            f"scorer {index}/{len(samples)} {state_id} K={len(actions)} "
            f"pick={candidate_index}:{action_family(action)} "
            f"loss_ub={row['scorer_loss_upper_bound_cc']}",
            flush=True,
        )

    losses = [int(row["scorer_loss_upper_bound_cc"]) for row in report["rows"]]
    ordered = sorted(losses)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)] if ordered else None
    report["summary"] = {
        "states": len(losses),
        "zero_scorer_loss_states": sum(loss == 0 for loss in losses),
        "mean_scorer_loss_cc": sum(losses) / len(losses) if losses else None,
        "p95_scorer_loss_cc": p95,
        "max_scorer_loss_cc": max(losses) if losses else None,
        "runtime_s_this_invocation": time.perf_counter() - started,
    }
    atomic_write_json(args.out, report)
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.out}")
    return 0


def run_integer_policy(
    initial: BeamState,
    *,
    weights: dict[str, int],
    scorer_kind: str,
    tie_kind: str,
    rank_limit: int,
    bottom_count: int,
    budget: int,
) -> tuple[BeamState, list[StageAction], int]:
    """Run the deployable round-by-round policy without future search."""
    if (
        scorer_kind == "lpt-estimate"
        and tie_kind == "rem-snap"
        and rank_limit == 4
        and bottom_count == 2
        and budget == 32
    ):
        from scheduler_policy_golden import run_policy as run_frozen_policy

        return run_frozen_policy(
            initial,
            candidate_generator=generate_direct_candidates,
        )
    state = initial
    history = []
    max_candidates = 0
    max_decisions = 4 * len(initial.remaining) + 8

    while state.remaining:
        actions = generate_direct_candidates(
            state,
            rank_limit=rank_limit,
            bottom_count=bottom_count,
            budget=budget,
        )
        if not actions:
            raise RuntimeError("integer policy has no legal candidate")
        max_candidates = max(max_candidates, len(actions))
        scored = []
        for candidate_index, action in enumerate(actions):
            child = apply_action(state, action)
            if scorer_kind == "lpt-estimate":
                score = _lpt_completion_estimate(child)
            elif scorer_kind == "cache-estimate":
                score = _cache_aware_completion_estimate(child)
            elif scorer_kind == "dual-estimate":
                score = completion_estimate(child)
            elif scorer_kind == "integer-ranker-v1":
                base_q, terms = r8_ranker_features(state, action, child)
                scorer_row = {
                    "mode_before": decision_mode(state),
                    "base_q": base_q,
                    "terms": terms,
                }
                score = _r8_ranker_score(scorer_row, weights)
            else:
                base, features, _ = future_features(child)
                if scorer_kind == "base":
                    score = base
                elif scorer_kind == "integer-model-v0":
                    scorer_row = {
                        "base_cc": base,
                        "term_values": _scorer_term_values(features),
                    }
                    score = _integer_score(scorer_row, weights)
                else:
                    raise AssertionError(scorer_kind)
            if tie_kind == "rem-snap":
                tie = (
                    len(child.remaining),
                    max(child.c2.task_end, child.c3.task_end),
                )
            elif tie_kind == "index":
                tie = ()
            else:
                raise AssertionError(tie_kind)
            scored.append(((score, *tie, candidate_index), action, child))
        _, action, state = min(scored, key=lambda row: row[0])
        history.append(action)
        if len(history) > max_decisions:
            raise RuntimeError("integer policy exceeded the progress guard")
    return state, history, max_candidates


def _rollout_summary(rows: list[dict]) -> dict:
    ratios = sorted(float(row["ratio"]) for row in rows)
    regrets = sorted(int(row["regret_cc"]) for row in rows)

    def percentile(values: list, fraction: float):
        if not values:
            return None
        return values[min(len(values) - 1, math.ceil(fraction * len(values)) - 1)]

    return {
        "cases": len(rows),
        "exact_cases": sum(row["regret_cc"] == 0 for row in rows),
        "positive_regret_cases": sum(row["regret_cc"] > 0 for row in rows),
        "beats_reference_cases": sum(row["regret_cc"] < 0 for row in rows),
        "ratio_mean": sum(ratios) / len(ratios) if ratios else None,
        "ratio_p95": percentile(ratios, 0.95),
        "ratio_max": max(ratios, default=None),
        "regret_mean_cc": sum(regrets) / len(regrets) if regrets else None,
        "regret_p95_cc": percentile(regrets, 0.95),
        "regret_max_cc": max(regrets, default=None),
        "max_candidates": max((row["max_candidates"] for row in rows), default=0),
        "max_decisions": max((row["decisions"] for row in rows), default=0),
    }


def audit_integer_policy_rollouts(args: argparse.Namespace) -> int:
    """Evaluate the fixed integer policy on a complete dataset partition."""
    if args.rollout_scorer == "integer-model-v0":
        profile, weights = load_selected_integer_model(
            args.model_out, args.model_profile
        )
        model = str(args.model_out)
    elif args.rollout_scorer == "integer-ranker-v1":
        if args.rank_limit != 8 or args.bottom_count != 0 or args.budget != 32:
            raise ValueError("integer-ranker-v1 requires R8, bottom0 and K32")
        profile = "mode-selected-r8-v1"
        weights = load_r8_ranker_model(args.model_out)
        model = str(args.model_out)
    else:
        profile, weights, model = "not-used", {}, None
    rows = []
    started = time.perf_counter()
    cases = 0
    for path in args.inputs:
        results = json.loads(path.read_text())["results"]
        eligible_in_file = 0
        for key, item in results.items():
            if item.get("dataset_split") != args.source_split or not quality_ok(
                item, args.quality
            ):
                continue
            if 0 <= args.max_cases_per_file <= eligible_in_file:
                break
            eligible_in_file += 1
            dist = {int(eid): int(ntok) for eid, ntok in item["dist"].items()}
            scheduler = FourStageScheduler(
                dist,
                initial_cache_c2=int(item.get("initial_cache_c2", -1)),
                initial_cache_c3=int(item.get("initial_cache_c3", -1)),
            )
            final, history, max_candidates = run_integer_policy(
                scheduler._initial_state(),
                weights=weights,
                scorer_kind=args.rollout_scorer,
                tie_kind=args.rollout_tie,
                rank_limit=args.rank_limit,
                bottom_count=args.bottom_count,
                budget=args.budget,
            )
            reference = int(item["makespan_cc"])
            regret = int(final.g_score) - reference
            if item.get("proven_optimal") and regret < 0:
                raise RuntimeError(
                    f"integer policy beat proven reference E{item['e_total']}:{key}"
                )
            history_blob = json.dumps(
                [serialize_action(action) for action in history],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            rows.append(
                {
                    "case_key": f"E{item['e_total']}:{key}",
                    "e_total": int(item["e_total"]),
                    "dataset_split": item.get("dataset_split"),
                    "quality_class": item.get("quality_class"),
                    "reference_proven": bool(item.get("proven_optimal")),
                    "reference_makespan_cc": reference,
                    "policy_makespan_cc": int(final.g_score),
                    "regret_cc": regret,
                    "ratio": int(final.g_score) / reference,
                    "decisions": len(history),
                    "max_candidates": max_candidates,
                    "history_sha256": hashlib.sha256(history_blob).hexdigest(),
                }
            )
            cases += 1
            clear_scheduler_caches()
            _equal_finish_left.cache_clear()
            _release_target_left.cache_clear()
            if args.progress_every > 0 and cases % args.progress_every == 0:
                print(
                    f"rollout cases={cases} "
                    f"elapsed_s={time.perf_counter()-started:.1f}",
                    flush=True,
                )
    by_e = {
        f"E{e_total}": _rollout_summary(
            [row for row in rows if row["e_total"] == e_total]
        )
        for e_total in (8, 32, 64)
    }
    report = {
        "schema": "scheduler_integer_policy_rollout_v0",
        "candidate_revision": "direct-slot-conditional-cache-v8",
        "model": model,
        "model_profile": profile,
        "model_weights_eighths": weights,
        "rollout_scorer": args.rollout_scorer,
        "rollout_tie": args.rollout_tie,
        "source_split": args.source_split,
        "quality": args.quality,
        "rank_limit": args.rank_limit,
        "bottom_count": args.bottom_count,
        "budget": args.budget,
        "max_cases_per_file": args.max_cases_per_file,
        "runtime_s": time.perf_counter() - started,
        "summary": {
            "overall": _rollout_summary(rows),
            **by_e,
        },
        "worst_rows": sorted(rows, key=lambda row: row["ratio"], reverse=True)[:100],
        "rows": rows,
    }
    atomic_write_json(args.out, report)
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.out}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "candidate-audit",
            "continuation-audit",
            "future-dataset",
            "scorer-fit",
            "scorer-audit",
            "rollout-audit",
            "r8-ranking-dataset",
            "r8-ranking-fit",
        ),
    )
    parser.add_argument("inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--dataset-out", type=Path, default=DEFAULT_FUTURE_DATASET
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=ROOT / "results" / "policy_search" / "integer_scorer_fit.json",
    )
    parser.add_argument(
        "--model-profile",
        choices=tuple(SCORER_MODEL_TERMS),
        help="evaluate one already fitted profile instead of the selected profile",
    )
    parser.add_argument(
        "--rollout-scorer",
        choices=(
            "integer-model-v0",
            "base",
            "lpt-estimate",
            "cache-estimate",
            "dual-estimate",
            "integer-ranker-v1",
        ),
        default="integer-model-v0",
    )
    parser.add_argument(
        "--rollout-tie",
        choices=("index", "rem-snap"),
        default="index",
    )
    parser.add_argument(
        "--dataset-phase",
        choices=("reference", "counterfactual"),
        default="reference",
    )
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument(
        "--ranking-state-source",
        choices=("reference", "r8_lpt", "mixed"),
        default="mixed",
        help="state distribution used for the R8 counterfactual rank dataset",
    )
    parser.add_argument(
        "--ranking-cases-per-e",
        type=int,
        default=256,
        help="deterministic proven cases per E used to collect R8 on-policy states",
    )
    parser.add_argument(
        "--source-split",
        choices=("discovery", "validation", "blind_test"),
        default="discovery",
    )
    parser.add_argument(
        "--quality",
        choices=("proven", "within3", "eligible"),
        default="proven",
    )
    parser.add_argument(
        "--seed-report",
        action="append",
        type=Path,
        default=[],
        help="reuse candidate/scorer continuation results when labeling candidates",
    )
    parser.add_argument("--rank-limit", type=int, choices=(4, 8), default=4)
    parser.add_argument(
        "--generator",
        choices=("direct", "enumerated"),
        default="direct",
        help="direct is the deployable bounded path; enumerated is offline audit only",
    )
    parser.add_argument(
        "--scorer",
        choices=(
            "base-index-v0",
            "base-release-tie-v0",
            "base-plus-release-eighth-v0",
            "integer-model-v0",
        ),
        default="base-index-v0",
    )
    parser.add_argument("--bottom-count", type=int, choices=(0, 2, 4), default=2)
    parser.add_argument("--budget", type=int, choices=(16, 24, 32), default=32)
    parser.add_argument("--states-per-stratum", type=int, default=1)
    parser.add_argument("--max-states", type=int, default=12)
    parser.add_argument(
        "--max-cases-per-file",
        type=int,
        default=-1,
        help="debug-only cap on eligible discovery cases read from each input",
    )
    parser.add_argument("--sample-min-remaining", type=int, default=1)
    parser.add_argument("--sample-max-remaining", type=int, default=-1)
    parser.add_argument(
        "--state-ids-from",
        type=Path,
        help="reuse the ordered state_id rows from an earlier audit report",
    )
    parser.add_argument(
        "--sample-require-r4-miss-r8-hit",
        action="store_true",
        help="sample only states whose reference expert pool needs R8 over R4",
    )
    parser.add_argument("--continuation-time-limit", type=float, default=2.0)
    parser.add_argument("--continuation-expansions", type=int, default=200)
    parser.add_argument("--continuation-target-gap", type=float, default=0.01)
    parser.add_argument("--stop-on-zero", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "candidate-audit":
        return audit_candidates(args)
    if args.command == "continuation-audit":
        return audit_continuations(args)
    if args.command == "future-dataset":
        if args.dataset_phase == "reference":
            return build_reference_future_dataset(args)
        return build_counterfactual_future_dataset(args)
    if args.command == "scorer-fit":
        return fit_integer_scorer(args)
    if args.command == "scorer-audit":
        return audit_base_scorer(args)
    if args.command == "rollout-audit":
        return audit_integer_policy_rollouts(args)
    if args.command == "r8-ranking-dataset":
        return build_r8_ranking_dataset(args)
    if args.command == "r8-ranking-fit":
        return fit_r8_ranking_scorer(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
