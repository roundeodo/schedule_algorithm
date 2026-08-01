#!/usr/bin/env python3
"""Evaluate a bounded, RTL-oriented candidate bank on the frozen OLMoE set.

This is the Stage-D *candidate oracle*, not the final scheduler scorer.  At
every reference state it exposes at most K concrete ``StageAction`` objects.
An exact target-feasibility search then asks whether that fixed candidate graph
contains any path to the already certified global optimum.  Candidate ordering
is local and target-independent; the optimum target is used only by the exact
search to prune states, never to manufacture or rank candidates.

The bank is intentionally factored into:

* bounded logical templates over TOP+BOTTOM+resident load classes; and
* at most three deterministic local physical profiles per template
  (fast-finish, S2-prefetch, and no-S2-prefetch/lane-light).  Standalone
  prefetch instead keeps one local realization for each legal DMA binding.

Reference action generation is invoked only for one named logical template at
a time and every returned concrete action counts against K.  The script never
enumerates the complete window action set and then hides an oracle-selected
winner behind one candidate ID.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import astuple, dataclass
from fractions import Fraction
import hashlib
import heapq
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Iterable, Optional


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import four_stage_scheduler as reference  # noqa: E402
from run_four_stage_reference import deserialize_action, serialize_action  # noqa: E402


BANK_VERSION = "load-class-local-physical-v1"
SUPPORTED_K = (16, 24, 32)
DEFAULT_PROOF = (
    HERE / "results" / "policy_search" / "olmoe_top2_projection_65_optimal_v1.json"
)
TICK_CC = reference.SCHEDULE_TIME_QUANTUM_CC


@dataclass(frozen=True)
class LogicalSpec:
    family: str
    eids: tuple[int, ...]
    cut: tuple[int, int] = ()
    label: str = ""


@dataclass(frozen=True)
class CandidateBatch:
    mode: str
    actions: tuple[reference.StageAction, ...]
    logical_specs: int
    raw_physical_actions: int
    family_specs: tuple[tuple[str, int], ...]
    family_candidates: tuple[tuple[str, int], ...]
    family_selected: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class CandidateSearchCheckpoint:
    target_makespan: int
    window: tuple[int, int]
    candidate_budget: int
    bank_version: str
    rank_heap: list[tuple]
    active_entries: set[int]
    open_by_fingerprint: dict[tuple, tuple[int, int]]
    closed_best_work: dict[tuple, int]
    next_entry_id: int
    expansions: int
    generated: int
    pruned_by_target: int
    peak_open_states: int
    runtime_s: float
    batch_totals: dict[str, int]
    batch_maxima: dict[str, int]
    mode_states: dict[str, int]
    selected_families: dict[str, int]


@dataclass(frozen=True)
class CandidateSearchResult:
    feasible: bool
    exhaustive: bool
    history: tuple[reference.StageAction, ...]
    termination: str
    checkpoint: Optional[CandidateSearchCheckpoint]
    expansions: int
    generated: int
    pruned_by_target: int
    open_states: int
    closed_states: int
    peak_open_states: int
    runtime_s: float
    batch_totals: dict[str, int]
    batch_maxima: dict[str, int]
    mode_states: dict[str, int]
    selected_families: dict[str, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ticks(cc: int) -> str:
    value = Fraction(int(cc), TICK_CC)
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _target_cc(value) -> int:
    ticks = Fraction(str(value))
    cc = ticks * TICK_CC
    if cc.denominator != 1:
        raise ValueError(f"target {value!r} is not an integer cycle count")
    return int(cc)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_pickle(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _family(action: reference.StageAction) -> str:
    if action.pf_eid >= 0:
        return "PREFETCH"
    if action.c2_eid >= 0 and action.c2_eid == action.c3_eid:
        return "SPLIT"
    if action.c2_eid >= 0 and action.c3_eid >= 0:
        return "PAIR"
    if action.c2_eid >= 0 or action.c3_eid >= 0:
        return "SINGLE"
    return "OTHER"


def _decision_mode(state: reference.BeamState) -> str:
    if len(state.remaining) == 1:
        return "TERMINAL"
    if state.c2.task_end == state.c3.task_end:
        return "SYNC"
    return "ONE_IDLE"


def _resident_eids(state: reference.BeamState) -> tuple[int, ...]:
    remaining = {eid for eid, _ntok in state.remaining}
    return tuple(
        dict.fromkeys(
            snap.pf_eid
            for snap in (state.c2, state.c3)
            if snap.pf_eid >= 0 and snap.pf_eid in remaining
        )
    )


def _load_groups(
    state: reference.BeamState,
    window: tuple[int, int],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    visible = reference.candidate_window_remaining(
        state.c2, state.c3, state.remaining, window
    )
    residents = set(_resident_eids(state))
    grouped: dict[int, list[tuple[int, int]]] = {}
    for eid, ntok in visible:
        if eid in residents:
            continue
        grouped.setdefault(int(ntok), []).append((int(eid), int(ntok)))
    return tuple(tuple(grouped[ntok]) for ntok in sorted(grouped, reverse=True))


def _dedupe_specs(specs: Iterable[LogicalSpec]) -> list[LogicalSpec]:
    selected = []
    seen = set()
    for spec in specs:
        eids = (
            tuple(sorted(spec.eids))
            if spec.family in {"PAIR", "WAIT_PAIR"}
            else spec.eids
        )
        key = (spec.family, eids, tuple(sorted(spec.cut)))
        if key in seen:
            continue
        seen.add(key)
        selected.append(spec)
    return selected


def _pair_specs(
    groups: tuple[tuple[tuple[int, int], ...], ...],
    residents: tuple[int, ...],
    family: str,
) -> list[LogicalSpec]:
    specs: list[LogicalSpec] = []

    # Concrete residents are named state and must be considered before generic
    # classes when an earlier S4PF reserved one cluster's next expert.
    if len(residents) >= 2:
        specs.append(LogicalSpec(family, residents[:2], label="R-R"))
    for resident in residents:
        for index in list(range(min(4, len(groups)))) + ([len(groups) - 1] if groups else []):
            if index < 0 or index >= len(groups):
                continue
            specs.append(
                LogicalSpec(
                    family,
                    (resident, groups[index][0][0]),
                    label=f"R-L{index}",
                )
            )

    # Same-class pairs are first-class templates.  The two concrete equal-load
    # IDs are timing-equivalent unless one is already resident, handled above.
    for index, group in enumerate(groups):
        if len(group) >= 2:
            specs.append(
                LogicalSpec(
                    family,
                    (group[0][0], group[1][0]),
                    label=f"L{index}-L{index}",
                )
            )
    for index in range(len(groups) - 1):
        specs.append(
            LogicalSpec(
                family,
                (groups[index][0][0], groups[index + 1][0][0]),
                label=f"L{index}-L{index + 1}",
            )
        )
    if groups:
        for index in (2, 3, 4, len(groups) - 1):
            if 0 < index < len(groups):
                specs.append(
                    LogicalSpec(
                        family,
                        (groups[0][0][0], groups[index][0][0]),
                        label=f"L0-L{index}",
                    )
                )
    return _dedupe_specs(specs)


def _split_specs(
    state: reference.BeamState,
    groups: tuple[tuple[tuple[int, int], ...], ...],
    residents: tuple[int, ...],
) -> list[LogicalSpec]:
    ntok_by_eid = dict(state.remaining)
    targets = [(eid, f"R{index}") for index, eid in enumerate(residents)]
    targets += [
        (group[0][0], f"L{index}")
        for index, group in enumerate(groups[:4])
    ]
    specs = []
    for eid, label in targets:
        ntok = int(ntok_by_eid[eid])
        if ntok < 2:
            continue
        half = (ntok // 2, ntok - ntok // 2)
        edge_low = max(1, ntok - 12) if ntok > 12 else min(4, ntok - 1)
        edge = (edge_low, ntok - edge_low)
        specs.append(LogicalSpec("SPLIT", (eid,), half, f"{label}:HALF"))
        specs.append(LogicalSpec("SPLIT", (eid,), edge, f"{label}:EDGE"))
    return _dedupe_specs(specs)


def _single_specs(
    groups: tuple[tuple[tuple[int, int], ...], ...],
    residents: tuple[int, ...],
) -> list[LogicalSpec]:
    specs = [
        LogicalSpec("SINGLE", (eid,), label=f"R{index}")
        for index, eid in enumerate(residents)
    ]
    specs += [
        LogicalSpec("SINGLE", (group[0][0],), label=f"L{index}")
        for index, group in enumerate(groups[:7])
    ]
    return _dedupe_specs(specs)


def _prefetch_specs(
    groups: tuple[tuple[tuple[int, int], ...], ...],
) -> list[LogicalSpec]:
    # All currently observable nonresident classes are represented.  K quotas,
    # not hidden frequency deletion, decide how many concrete profiles survive.
    return [
        LogicalSpec("PREFETCH", (group[0][0],), label=f"L{index}")
        for index, group in enumerate(groups[:7])
    ]


def _logical_specs(
    state: reference.BeamState,
    window: tuple[int, int],
) -> tuple[str, dict[str, list[LogicalSpec]]]:
    mode = _decision_mode(state)
    groups = _load_groups(state, window)
    residents = _resident_eids(state)
    families: dict[str, list[LogicalSpec]] = {
        "PAIR": [],
        "WAIT_PAIR": [],
        "SPLIT": [],
        "SINGLE": [],
        "PREFETCH": _prefetch_specs(groups),
    }
    if mode == "SYNC":
        families["PAIR"] = _pair_specs(groups, residents, "PAIR")
        families["SPLIT"] = _split_specs(state, groups, residents)
        families["SINGLE"] = _single_specs(groups, residents)
    elif mode == "ONE_IDLE":
        families["SINGLE"] = _single_specs(groups, residents)
        families["WAIT_PAIR"] = _pair_specs(groups, residents, "WAIT_PAIR")
    else:
        families["SINGLE"] = _single_specs(groups, residents)
    return mode, families


def _action_tie(action: reference.StageAction) -> str:
    return repr(astuple(action))


def _physical_metrics(
    action: reference.StageAction,
    child: reference.BeamState,
) -> dict[str, tuple]:
    bindings = (
        action.c2_dma_s1,
        action.c2_dma_s3,
        action.c2_s2pf_dma,
        action.c3_dma_s1,
        action.c3_dma_s3,
        action.c3_s2pf_dma,
        action.pf_dma,
    )
    both = sum(binding == reference.DmaBinding.BOTH for binding in bindings)
    lanes = sum(
        binding in (reference.DmaBinding.IDMA, reference.DmaBinding.XDMA)
        for binding in bindings
    )
    s2pf = sum(
        binding != reference.DmaBinding.NONE
        for binding in (action.c2_s2pf_dma, action.c3_s2pf_dma)
    )
    cache_gain = int(action.c2_s1_cached) + int(action.c2_s3_cached)
    cache_gain += int(action.c3_s1_cached) + int(action.c3_s3_cached)
    ends = (int(child.c2.task_end), int(child.c3.task_end))
    tie = _action_tie(action)
    return {
        "FAST": (max(ends), sum(ends), abs(ends[0] - ends[1]), -s2pf, tie),
        "S2PF": (-s2pf, -cache_gain, max(ends), sum(ends), both, tie),
        "NO_S2PF": (s2pf, both, lanes, max(ends), sum(ends), tie),
    }


def _matches_spec(action: reference.StageAction, spec: LogicalSpec) -> bool:
    family = _family(action)
    if spec.family == "PREFETCH":
        return family == "PREFETCH" and action.pf_eid == spec.eids[0]
    if spec.family in {"PAIR", "WAIT_PAIR"}:
        assigned = (action.c2_eid, action.c3_eid)
        return (
            family == "PAIR"
            and len(set(assigned)) == 2
            and set(assigned) == set(spec.eids)
        )
    if spec.family == "SPLIT":
        return (
            family == "SPLIT"
            and action.c2_eid == spec.eids[0]
            and tuple(sorted((action.c2_ntok, action.c3_ntok)))
            == tuple(sorted(spec.cut))
        )
    if spec.family == "SINGLE":
        assigned = [eid for eid in (action.c2_eid, action.c3_eid) if eid >= 0]
        return family == "SINGLE" and assigned == [spec.eids[0]]
    raise ValueError(spec.family)


def _eligible_physical_children(
    state: reference.BeamState,
    spec: LogicalSpec,
) -> tuple[list[tuple[reference.StageAction, reference.BeamState]], int]:
    selected = set(spec.eids)
    subset = tuple(item for item in state.remaining if item[0] in selected)
    if not subset:
        return [], 0
    if spec.family == "PREFETCH":
        raw = reference.gen_prefetch_actions(
            state.c2,
            state.c3,
            subset,
            seed_mode=True,
            seed_all_visible=True,
        )
    else:
        # The reference seed profile set is a fixed local physical menu:
        # A/B on one lane and C on BOTH for S1, B on one lane and C on BOTH
        # for S3, plus the deterministic optional S2PF realization.  Using it
        # avoids enumerating the complete physical Cartesian product before
        # selecting three profiles and makes the Python implementation match
        # the bounded hardware contract.
        raw = reference.gen_stage_actions(
            state.c2,
            state.c3,
            subset,
            seed_mode=True,
            seed_all_visible=True,
        )
    eligible = [action for action in raw if _matches_spec(action, spec)]
    if spec.family == "WAIT_PAIR":
        decision = min(state.c2.task_end, state.c3.task_end)
        eligible = [
            action
            for action in eligible
            if min(action.c2_start, action.c3_start) > decision
        ]
    elif spec.family == "PAIR" and state.c2.task_end != state.c3.task_end:
        return [], len(eligible)

    children = [(action, reference.apply_action(state, action)) for action in eligible]
    return children, len(eligible)


def _physical_profiles(
    state: reference.BeamState,
    spec: LogicalSpec,
) -> tuple[list[tuple[reference.StageAction, reference.BeamState]], int]:
    children, eligible_count = _eligible_physical_children(state, spec)
    profiles = []
    seen = set()
    if spec.family == "PREFETCH":
        # Binding is a real candidate axis.  Keep at most one earliest legal
        # start for BOTH, IDMA and XDMA; no binding is hidden in a local score.
        profile_actions = []
        for binding in (
            reference.DmaBinding.BOTH,
            reference.DmaBinding.IDMA,
            reference.DmaBinding.XDMA,
        ):
            options = [item for item in children if item[0].pf_dma == binding]
            if options:
                profile_actions.append(
                    min(options, key=lambda item: _physical_metrics(item[0], item[1])["FAST"])
                )
    else:
        profile_actions = [
            min(children, key=lambda item: _physical_metrics(item[0], item[1])[profile])
            for profile in ("FAST", "S2PF", "NO_S2PF")
            if children
        ]
    for action, child in profile_actions:
        key = (child.fingerprint(), int(child.cluster_work_cc))
        if key in seen:
            continue
        seen.add(key)
        profiles.append((action, child))
    return profiles, eligible_count


def _child_exact_key(state: reference.BeamState) -> tuple:
    """Concrete child identity used for sequential witness substitution."""
    return (
        state.c2,
        state.c3,
        state.remaining,
        int(state.cluster_work_cc),
    )


def _child_canonical_key(state: reference.BeamState) -> tuple:
    """The symmetry-aware identity used by the bounded-bank de-duplicator."""
    return (state.fingerprint(), int(state.cluster_work_cc))


def _build_candidate_pools(
    state: reference.BeamState,
    window: tuple[int, int],
) -> tuple[
    str,
    dict[str, list[LogicalSpec]],
    dict[str, list[tuple[reference.StageAction, reference.BeamState]]],
    int,
]:
    mode, specs_by_family = _logical_specs(state, window)
    pools: dict[str, list[tuple[reference.StageAction, reference.BeamState]]] = {}
    raw_physical = 0
    for family, specs in specs_by_family.items():
        per_spec = []
        for spec in specs:
            profiles, raw = _physical_profiles(state, spec)
            raw_physical += raw
            per_spec.append(profiles)
        # Profile-round ordering preserves logical diversity before adding a
        # second/third physical realization of the same template.
        pool = []
        for profile_index in range(3):
            for profiles in per_spec:
                if profile_index < len(profiles):
                    pool.append(profiles[profile_index])
        pools[family] = pool
    return mode, specs_by_family, pools, raw_physical


def _quotas(mode: str, k: int) -> dict[str, int]:
    if k not in SUPPORTED_K:
        raise ValueError(f"candidate budget must be one of {SUPPORTED_K}")
    if mode == "SYNC":
        return {
            16: {"PAIR": 8, "SPLIT": 4, "SINGLE": 1, "PREFETCH": 3},
            24: {"PAIR": 12, "SPLIT": 6, "SINGLE": 2, "PREFETCH": 4},
            32: {"PAIR": 16, "SPLIT": 8, "SINGLE": 2, "PREFETCH": 6},
        }[k]
    if mode == "ONE_IDLE":
        return {
            16: {"SINGLE": 10, "WAIT_PAIR": 2, "PREFETCH": 4},
            24: {"SINGLE": 15, "WAIT_PAIR": 3, "PREFETCH": 6},
            32: {"SINGLE": 20, "WAIT_PAIR": 4, "PREFETCH": 8},
        }[k]
    return {
        16: {"SINGLE": 12, "PREFETCH": 4},
        24: {"SINGLE": 18, "PREFETCH": 6},
        32: {"SINGLE": 24, "PREFETCH": 8},
    }[k]


def generate_bounded_candidates(
    state: reference.BeamState,
    window: tuple[int, int],
    k: int,
) -> CandidateBatch:
    """Return at most K target-independent concrete candidate actions."""
    mode, specs_by_family, pools, raw_physical = _build_candidate_pools(
        state, window
    )

    quotas = _quotas(mode, k)
    selected: list[tuple[reference.StageAction, reference.BeamState, str]] = []
    selected_keys = set()
    consumed_indices: dict[str, int] = {}

    def append(item, family: str) -> bool:
        action, child = item
        key = (child.fingerprint(), int(child.cluster_work_cc))
        if key in selected_keys:
            return False
        selected_keys.add(key)
        selected.append((action, child, family))
        return True

    for family, quota in quotas.items():
        index = 0
        accepted = 0
        pool = pools.get(family, [])
        while index < len(pool) and accepted < quota:
            accepted += int(append(pool[index], family))
            index += 1
        consumed_indices[family] = index

    # If a reserved family has fewer legal candidates, use the otherwise idle
    # slots.  This never increases the K-wide scorer input.
    fill_order = (
        ("PAIR", "SPLIT", "SINGLE", "PREFETCH")
        if mode == "SYNC"
        else ("SINGLE", "WAIT_PAIR", "PREFETCH")
    )
    while len(selected) < k:
        made_progress = False
        for family in fill_order:
            pool = pools.get(family, [])
            index = consumed_indices.get(family, 0)
            while index < len(pool):
                item = pool[index]
                index += 1
                consumed_indices[family] = index
                if append(item, family):
                    made_progress = True
                    break
            if len(selected) >= k:
                break
        if not made_progress:
            break

    visible = reference.candidate_window_visible_eids(
        state.c2, state.c3, state.remaining, window
    )
    actions = tuple(item[0] for item in selected)
    if len(actions) > k:
        raise AssertionError("candidate bank exceeded K")
    if any(
        not reference.action_within_candidate_window(action, visible)
        for action in actions
    ):
        raise AssertionError("candidate bank emitted a hidden expert")
    family_selected = Counter(item[2] for item in selected)
    return CandidateBatch(
        mode=mode,
        actions=actions,
        logical_specs=sum(len(specs) for specs in specs_by_family.values()),
        raw_physical_actions=raw_physical,
        family_specs=tuple(
            sorted((family, len(specs)) for family, specs in specs_by_family.items())
        ),
        family_candidates=tuple(
            sorted((family, len(pool)) for family, pool in pools.items())
        ),
        family_selected=tuple(sorted(family_selected.items())),
    )


def _spec_record(spec: LogicalSpec) -> dict:
    return {
        "family": spec.family,
        "eids": list(spec.eids),
        "cut": list(spec.cut),
        "label": spec.label,
    }


def _diagnose_witness_miss(
    state: reference.BeamState,
    witness_action: reference.StageAction,
    witness_child: reference.BeamState,
    window: tuple[int, int],
    k: int,
    batch: CandidateBatch,
) -> dict:
    """Locate the first bounded-generator stage that lost a witness child.

    This is deliberately stricter than a canonical fingerprint comparison:
    the direct-path audit substitutes actions sequentially and therefore needs
    the same concrete expert IDs and snapshots.  A canonical de-duplication
    loss is reported separately so it can later be repaired by an explicit
    symmetry/remapping rule rather than silently counted as coverage.
    """
    target_exact = _child_exact_key(witness_child)
    target_canonical = _child_canonical_key(witness_child)
    mode, specs_by_family, pools, _raw_physical = _build_candidate_pools(
        state, window
    )
    matching_specs = [
        spec
        for specs in specs_by_family.values()
        for spec in specs
        if _matches_spec(witness_action, spec)
    ]
    if not matching_specs:
        return {
            "classification": "logical_spec_missing",
            "mode": mode,
            "matching_specs": [],
            "target_pool_positions": {},
        }

    raw_exact_specs = []
    profile_exact_specs = []
    raw_candidate_counts = {}
    profile_candidate_counts = {}
    for spec in matching_specs:
        key = f"{spec.family}:{spec.label}:{','.join(map(str, spec.eids))}"
        eligible, _count = _eligible_physical_children(state, spec)
        profiles, _raw = _physical_profiles(state, spec)
        raw_candidate_counts[key] = len(eligible)
        profile_candidate_counts[key] = len(profiles)
        if any(_child_exact_key(child) == target_exact for _action, child in eligible):
            raw_exact_specs.append(spec)
        if any(_child_exact_key(child) == target_exact for _action, child in profiles):
            profile_exact_specs.append(spec)

    base = {
        "mode": mode,
        "matching_specs": [_spec_record(spec) for spec in matching_specs],
        "raw_exact_specs": [_spec_record(spec) for spec in raw_exact_specs],
        "profile_exact_specs": [
            _spec_record(spec) for spec in profile_exact_specs
        ],
        "raw_candidate_counts": raw_candidate_counts,
        "profile_candidate_counts": profile_candidate_counts,
        "target_pool_positions": {
            family: [
                index
                for index, (_action, child) in enumerate(pool)
                if _child_exact_key(child) == target_exact
            ]
            for family, pool in pools.items()
            if any(_child_exact_key(child) == target_exact for _action, child in pool)
        },
        "family_pool_sizes": {
            family: len(pool) for family, pool in sorted(pools.items())
        },
        "family_quotas": _quotas(mode, k),
        "family_selected": dict(batch.family_selected),
    }
    if not raw_exact_specs:
        base["classification"] = "seed_physical_menu_missing"
        return base
    if not profile_exact_specs:
        base["classification"] = "local_profile_pruned"
        return base

    selected_children = [
        reference.apply_action(state, action) for action in batch.actions
    ]
    if any(_child_exact_key(child) == target_exact for child in selected_children):
        # Defensive: the caller should only diagnose misses.
        base["classification"] = "selected_exact"
        return base
    canonical_indices = [
        index
        for index, child in enumerate(selected_children)
        if _child_canonical_key(child) == target_canonical
    ]
    if canonical_indices:
        base["classification"] = "canonical_dedupe_pruned"
        base["canonical_equivalent_selected_indices"] = canonical_indices
        return base
    base["classification"] = "quota_or_order_pruned"
    return base


def _target_rank(state: reference.BeamState) -> tuple:
    return (
        len(state.remaining),
        reference.completion_estimate(state),
        abs(state.c2.task_end - state.c3.task_end),
        state.g_score,
    )


def run_candidate_target_search(
    counts: list[int],
    target_makespan: int,
    window: tuple[int, int],
    k: int,
    *,
    time_limit_s: Optional[float] = None,
    max_expansions: Optional[int] = None,
    checkpoint: Optional[CandidateSearchCheckpoint] = None,
) -> CandidateSearchResult:
    """Exactly search the graph induced by ``generate_bounded_candidates``."""
    if time_limit_s is not None and time_limit_s <= 0:
        raise ValueError("time_limit_s must be positive")
    if max_expansions is not None and max_expansions <= 0:
        raise ValueError("max_expansions must be positive")
    window = reference.normalize_candidate_window(window)
    if window is None:
        raise ValueError("a finite window is required")
    token_dist = {
        eid: int(ntok) for eid, ntok in enumerate(counts) if int(ntok) > 0
    }
    scheduler = reference.FourStageScheduler(token_dist)
    target_capacity = 2 * int(target_makespan)
    started = time.perf_counter()

    def within_target(state: reference.BeamState) -> bool:
        return (
            state.f_score <= target_makespan
            and state.c2.task_end
            + state.c3.task_end
            + reference._minimum_cluster_work(state.remaining)
            <= target_capacity
        )

    if checkpoint is not None:
        if (
            checkpoint.target_makespan != target_makespan
            or checkpoint.window != window
            or checkpoint.candidate_budget != k
            or checkpoint.bank_version != BANK_VERSION
        ):
            raise ValueError("candidate checkpoint configuration mismatch")
        rank_heap = list(checkpoint.rank_heap)
        heapq.heapify(rank_heap)
        active_entries = set(checkpoint.active_entries)
        open_by_fingerprint = dict(checkpoint.open_by_fingerprint)
        closed_best_work = dict(checkpoint.closed_best_work)
        next_entry_id = int(checkpoint.next_entry_id)
        expansions = int(checkpoint.expansions)
        generated = int(checkpoint.generated)
        pruned_by_target = int(checkpoint.pruned_by_target)
        peak_open_states = int(checkpoint.peak_open_states)
        accumulated_runtime = float(checkpoint.runtime_s)
        batch_totals = Counter(checkpoint.batch_totals)
        batch_maxima = Counter(checkpoint.batch_maxima)
        mode_states = Counter(checkpoint.mode_states)
        selected_families = Counter(checkpoint.selected_families)
    else:
        initial = scheduler._initial_state()
        if not within_target(initial):
            return CandidateSearchResult(
                feasible=False,
                exhaustive=True,
                history=(),
                termination="root_bound",
                checkpoint=None,
                expansions=0,
                generated=0,
                pruned_by_target=1,
                open_states=0,
                closed_states=0,
                peak_open_states=0,
                runtime_s=time.perf_counter() - started,
                batch_totals={},
                batch_maxima={},
                mode_states={},
                selected_families={},
            )
        rank_heap = []
        active_entries: set[int] = set()
        open_by_fingerprint: dict[tuple, tuple[int, int]] = {}
        closed_best_work: dict[tuple, int] = {}
        next_entry_id = 0
        expansions = generated = pruned_by_target = 0
        peak_open_states = 0
        accumulated_runtime = 0.0
        batch_totals = Counter()
        batch_maxima = Counter()
        mode_states = Counter()
        selected_families = Counter()

    def push(state: reference.BeamState) -> bool:
        nonlocal next_entry_id, peak_open_states
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
            (_target_rank(state), state.f_score, state.g_score, entry_id, state),
        )
        peak_open_states = max(peak_open_states, len(active_entries))
        return True

    if checkpoint is None:
        push(initial)
    invocation_expansions = expansions
    termination = "open_exhausted"
    while rank_heap:
        while rank_heap and rank_heap[0][3] not in active_entries:
            heapq.heappop(rank_heap)
        if not rank_heap:
            break
        elapsed = time.perf_counter() - started
        if time_limit_s is not None and elapsed >= time_limit_s:
            termination = "time_limit"
            break
        if (
            max_expansions is not None
            and expansions - invocation_expansions >= max_expansions
        ):
            termination = "expansion_limit"
            break
        _rank, _f, _g, entry_id, state = heapq.heappop(rank_heap)
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

        batch = generate_bounded_candidates(state, window, k)
        generated += len(batch.actions)
        mode_states[batch.mode] += 1
        for field in ("logical_specs", "raw_physical_actions"):
            value = int(getattr(batch, field))
            batch_totals[field] += value
            batch_maxima[field] = max(batch_maxima[field], value)
        batch_totals["selected_candidates"] += len(batch.actions)
        batch_maxima["selected_candidates"] = max(
            batch_maxima["selected_candidates"], len(batch.actions)
        )
        for family, count in batch.family_selected:
            selected_families[family] += count

        for action in batch.actions:
            child = reference.apply_action(state, action)
            if not within_target(child):
                pruned_by_target += 1
                continue
            if not child.remaining:
                replay = reference.validate_schedule_history(child.history, token_dist)
                if replay != child.g_score or replay > target_makespan:
                    raise AssertionError("candidate witness failed physical replay")
                runtime = accumulated_runtime + time.perf_counter() - started
                return CandidateSearchResult(
                    feasible=True,
                    exhaustive=False,
                    history=child.history,
                    termination="feasible",
                    checkpoint=None,
                    expansions=expansions,
                    generated=generated,
                    pruned_by_target=pruned_by_target,
                    open_states=len(active_entries),
                    closed_states=len(closed_best_work),
                    peak_open_states=peak_open_states,
                    runtime_s=runtime,
                    batch_totals=dict(batch_totals),
                    batch_maxima=dict(batch_maxima),
                    mode_states=dict(mode_states),
                    selected_families=dict(selected_families),
                )
            if not push(child):
                pruned_by_target += 1

    exhaustive = not active_entries
    runtime = accumulated_runtime + time.perf_counter() - started
    continuation = None
    if not exhaustive:
        live_heap = [entry for entry in rank_heap if entry[3] in active_entries]
        heapq.heapify(live_heap)
        continuation = CandidateSearchCheckpoint(
            target_makespan=target_makespan,
            window=window,
            candidate_budget=k,
            bank_version=BANK_VERSION,
            rank_heap=live_heap,
            active_entries=set(active_entries),
            open_by_fingerprint=dict(open_by_fingerprint),
            closed_best_work=dict(closed_best_work),
            next_entry_id=next_entry_id,
            expansions=expansions,
            generated=generated,
            pruned_by_target=pruned_by_target,
            peak_open_states=peak_open_states,
            runtime_s=runtime,
            batch_totals=dict(batch_totals),
            batch_maxima=dict(batch_maxima),
            mode_states=dict(mode_states),
            selected_families=dict(selected_families),
        )
    return CandidateSearchResult(
        feasible=False,
        exhaustive=exhaustive,
        history=(),
        termination=termination,
        checkpoint=continuation,
        expansions=expansions,
        generated=generated,
        pruned_by_target=pruned_by_target,
        open_states=len(active_entries),
        closed_states=len(closed_best_work),
        peak_open_states=peak_open_states,
        runtime_s=runtime,
        batch_totals=dict(batch_totals),
        batch_maxima=dict(batch_maxima),
        mode_states=dict(mode_states),
        selected_families=dict(selected_families),
    )


def _manifest(
    proof: Path,
    window: tuple[int, int],
    k: int,
) -> dict:
    script = Path(__file__).resolve()
    reference_path = Path(reference.__file__).resolve()
    return {
        "schema": "olmoe_bounded_candidate_oracle_manifest_v1",
        "bank_version": BANK_VERSION,
        "window": list(window),
        "candidate_budget": k,
        "proof": str(proof.resolve()),
        "proof_sha256": _sha256(proof),
        "reference": str(reference_path),
        "reference_sha256": _sha256(reference_path),
        "script": str(script),
        "script_sha256": _sha256(script),
    }


def _ensure_manifest(work_dir: Path, expected: dict) -> None:
    path = work_dir / "manifest.json"
    work_dir.mkdir(parents=True, exist_ok=True)
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError(f"work-dir manifest mismatch: {path}")
    else:
        if any(work_dir.iterdir()):
            raise RuntimeError(f"non-empty work-dir has no manifest: {work_dir}")
        _atomic_json(path, expected)


def _result_payload(
    case: dict,
    target_cc: int,
    result: CandidateSearchResult,
) -> dict:
    if result.feasible:
        status = "candidate_sufficient"
    elif result.exhaustive:
        status = "candidate_insufficient"
    else:
        status = "unresolved"
    return {
        "name": case["name"],
        "counts": case["counts"],
        "optimal_ticks": str(case["best_reference_ticks"]),
        "target_cc": target_cc,
        "status": status,
        "feasible": result.feasible,
        "exhaustive": result.exhaustive,
        "termination": result.termination,
        "history_replay_valid": result.feasible,
        "actions": [serialize_action(action) for action in result.history],
        "expansions": result.expansions,
        "generated": result.generated,
        "pruned_by_target": result.pruned_by_target,
        "open_states": result.open_states,
        "closed_states": result.closed_states,
        "peak_open_states": result.peak_open_states,
        "runtime_s": result.runtime_s,
        "batch_totals": result.batch_totals,
        "batch_maxima": result.batch_maxima,
        "mode_states": result.mode_states,
        "selected_families": result.selected_families,
    }


def _write_aggregate(
    output: Path,
    manifest: dict,
    cases: list[dict],
    work_dir: Path,
) -> None:
    rows = []
    for case in cases:
        path = work_dir / f"{case['name']}.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            rows.append(
                {
                    "name": case["name"],
                    "counts": case["counts"],
                    "optimal_ticks": str(case["best_reference_ticks"]),
                    "status": "not_started",
                }
            )
    status = Counter(row["status"] for row in rows)
    payload = {
        "schema": "olmoe_bounded_candidate_oracle_v1",
        "interpretation": {
            "candidate_sufficient": "exact candidate graph contains a replay-valid globally optimal path",
            "candidate_insufficient": "exact candidate graph exhausted without reaching the certified optimum",
            "unresolved": "search stopped with live OPEN states",
            "not_started": "no search invocation has run",
            "scorer_not_evaluated": True,
        },
        "manifest": manifest,
        "summary": {
            "cases": len(rows),
            "candidate_sufficient": status["candidate_sufficient"],
            "candidate_insufficient": status["candidate_insufficient"],
            "unresolved": status["unresolved"],
            "not_started": status["not_started"],
            "complete": not (status["unresolved"] or status["not_started"]),
        },
        "cases": rows,
    }
    _atomic_json(output, payload)


def _action_family(action: reference.StageAction) -> str:
    if action.c2_eid >= 0 and action.c2_eid == action.c3_eid:
        return "SPLIT"
    if action.c2_eid >= 0 and action.c3_eid >= 0:
        return "PAIR"
    if action.c2_eid >= 0 or action.c3_eid >= 0:
        return "SINGLE"
    if action.pf_eid >= 0:
        return "PREFETCH"
    return "OTHER"


def _window_name(window: tuple[int, int]) -> str:
    top, bottom = window
    return f"top{top}" if bottom == 0 else f"top{top}+bottom{bottom}"


def _materialize_direct_witness(
    audit_row: dict,
    window: tuple[int, int],
    source_cache: dict[Path, dict[str, dict]],
) -> tuple[dict, tuple[reference.StageAction, ...], dict]:
    """Load the exact direct source selected by the completed window audit."""
    for source in audit_row.get("direct_witness_sources", []):
        path = Path(source["source"])
        resolved = path.resolve()
        if resolved not in source_cache:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            source_cache[resolved] = {
                str(row["name"]): row for row in payload["cases"]
            }
        rows = source_cache[resolved]
        name = str(audit_row["name"])
        if name not in rows:
            continue
        row = rows[name]
        if source.get("equal_load_id_relabel"):
            import analyze_directed_case_classification as relabel_audit

            found, actions = relabel_audit._symmetry_relabel_history(
                row, *window
            )
            if not found or actions is None:
                raise RuntimeError(
                    f"{name}: audited equal-load relabel no longer replays"
                )
        else:
            actions = tuple(
                deserialize_action(raw) for raw in row["actions"]
            )
        return row, actions, source
    raise RuntimeError(
        f"{audit_row['name']}: no materializable direct witness source"
    )


def _audit_direct_witness_coverage(
    *,
    proof_path: Path,
    window_audit_path: Path,
    output_path: Path,
    window: tuple[int, int],
    k: int,
) -> int:
    """Check whether the bounded bank reproduces each saved optimal path.

    Full coverage is a constructive candidate-sufficiency proof.  A miss only
    says that this particular optimal history is absent; another equal-optimal
    path may still exist and must be decided by the exact candidate oracle.
    """
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    audit = json.loads(window_audit_path.read_text(encoding="utf-8"))
    if not proof.get("complete") or proof.get("summary", {}).get(
        "proven_optimal"
    ) != 65:
        raise SystemExit("proof must contain all 65 certified optimal cases")
    if not audit.get("complete") or audit.get("schema") != "directed_window_grid_v1":
        raise SystemExit("direct witness audit must be a completed window grid")
    window_label = _window_name(window)
    audit_rows = [
        row
        for row in audit["results"]
        if row["window"] == window_label
    ]
    if len(audit_rows) != 65 or any(
        row.get("window_status") != "proved_sufficient_direct"
        for row in audit_rows
    ):
        raise SystemExit(
            f"{window_label}: window audit does not directly cover all 65 cases"
        )
    proof_by_name = {str(row["name"]): row for row in proof["cases"]}
    source_cache: dict[Path, dict[str, dict]] = {}
    rows = []
    total_actions = 0
    equivalent_actions = 0
    canonical_equivalent_actions = 0
    first_miss_families = Counter()
    miss_classifications = Counter()
    first_miss_classifications = Counter()
    selected_candidate_counts = []
    mode_counts = Counter()

    for index, audit_row in enumerate(audit_rows, 1):
        name = str(audit_row["name"])
        source_row, actions, source = _materialize_direct_witness(
            audit_row, window, source_cache
        )
        target = proof_by_name[name]
        if [int(value) for value in source_row["counts"]] != [
            int(value) for value in target["counts"]
        ]:
            raise RuntimeError(f"{name}: witness/proof distribution mismatch")
        token_dist = {
            eid: int(ntok)
            for eid, ntok in enumerate(target["counts"])
            if int(ntok) > 0
        }
        replay_cc = reference.validate_schedule_history(actions, token_dist)
        target_cc = _target_cc(target["best_reference_ticks"])
        if replay_cc != target_cc:
            raise RuntimeError(f"{name}: direct witness misses certified optimum")
        state = reference.FourStageScheduler(token_dist)._initial_state()
        misses = []
        action_records = []
        for action_index, witness_action in enumerate(actions):
            visible = reference.candidate_window_visible_eids(
                state.c2, state.c3, state.remaining, window
            )
            if not reference.action_within_candidate_window(
                witness_action, visible
            ):
                raise RuntimeError(
                    f"{name}: audited witness action {action_index} is hidden"
                )
            batch = generate_bounded_candidates(state, window, k)
            selected_candidate_counts.append(len(batch.actions))
            mode_counts[batch.mode] += 1
            witness_child = reference.apply_action(state, witness_action)
            witness_key = (
                witness_child.c2,
                witness_child.c3,
                witness_child.remaining,
                int(witness_child.cluster_work_cc),
            )
            equivalent = None
            canonical_equivalent = None
            for candidate_index, candidate in enumerate(batch.actions):
                child = reference.apply_action(state, candidate)
                key = _child_exact_key(child)
                if key == witness_key:
                    equivalent = (candidate_index, candidate, child)
                    break
                if (
                    canonical_equivalent is None
                    and _child_canonical_key(child)
                    == _child_canonical_key(witness_child)
                ):
                    canonical_equivalent = (candidate_index, candidate, child)
            covered = equivalent is not None
            canonical_covered = covered or canonical_equivalent is not None
            total_actions += 1
            equivalent_actions += int(covered)
            canonical_equivalent_actions += int(canonical_covered)
            record = {
                "index": action_index,
                "mode": batch.mode,
                "family": _action_family(witness_action),
                "tag": witness_action.tag,
                "candidate_count": len(batch.actions),
                "equivalent_candidate": covered,
                "canonical_equivalent_candidate": canonical_covered,
                "equivalent_candidate_index": (
                    equivalent[0] if equivalent is not None else None
                ),
                "canonical_equivalent_candidate_index": (
                    equivalent[0]
                    if equivalent is not None
                    else canonical_equivalent[0]
                    if canonical_equivalent is not None
                    else None
                ),
                "family_selected": dict(batch.family_selected),
            }
            action_records.append(record)
            if not canonical_covered:
                diagnosis = _diagnose_witness_miss(
                    state,
                    witness_action,
                    witness_child,
                    window,
                    k,
                    batch,
                )
                record["diagnosis"] = diagnosis
                miss_classifications[diagnosis["classification"]] += 1
                misses.append(record)
            # Always follow the concrete audited history.  Exact state equality
            # above (not merely a canonical equal-load fingerprint) guarantees
            # that an accepted candidate can be substituted without requiring
            # a new ID-remapping proof for later actions.
            state = witness_child
        if state.remaining:
            raise RuntimeError(f"{name}: direct candidate audit is non-terminal")
        if misses:
            first_miss_families[misses[0]["family"]] += 1
            first_miss_classifications[
                misses[0]["diagnosis"]["classification"]
            ] += 1
        rows.append(
            {
                "index": index,
                "name": name,
                "optimal_ticks": str(target["best_reference_ticks"]),
                "witness_source": str(Path(source["source"]).resolve()),
                "equal_load_id_relabel": bool(
                    source.get("equal_load_id_relabel")
                ),
                "actions": len(actions),
                "equivalent_actions": sum(
                    record["equivalent_candidate"] for record in action_records
                ),
                "canonical_equivalent_actions": sum(
                    record["canonical_equivalent_candidate"]
                    for record in action_records
                ),
                "full_known_optimal_witness_covered": all(
                    record["equivalent_candidate"] for record in action_records
                ),
                "all_source_states_have_canonical_equivalent": not misses,
                "first_miss": misses[0] if misses else None,
                "misses": misses,
                "action_records": action_records,
            }
        )

    covered_cases = sum(
        row["full_known_optimal_witness_covered"] for row in rows
    )
    canonical_step_covered_cases = sum(
        row["all_source_states_have_canonical_equivalent"] for row in rows
    )
    payload = {
        "schema": "olmoe_bounded_candidate_direct_witness_audit_v1",
        "complete": True,
        "interpretation": {
            "covered": (
                "exact concrete transition coverage proves that this saved globally optimal path can be replayed unchanged"
            ),
            "canonical_step_coverage": (
                "every saved source state has a symmetry-equivalent candidate; this is diagnostic evidence, not a constructive whole-path proof without propagated ID remapping"
            ),
            "missed": (
                "this saved path is absent; exact candidate search is still required before declaring insufficiency"
            ),
            "scorer_not_evaluated": True,
        },
        "manifest": {
            "bank_version": BANK_VERSION,
            "window": list(window),
            "candidate_budget": k,
            "proof": str(proof_path.resolve()),
            "proof_sha256": _sha256(proof_path),
            "window_audit": str(window_audit_path.resolve()),
            "window_audit_sha256": _sha256(window_audit_path),
            "reference": str(Path(reference.__file__).resolve()),
            "reference_sha256": _sha256(Path(reference.__file__).resolve()),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "summary": {
            "cases": len(rows),
            "full_known_optimal_witness_covered": covered_cases,
            "known_witness_missed": len(rows) - covered_cases,
            "all_source_states_have_canonical_equivalent": canonical_step_covered_cases,
            "actions": total_actions,
            "equivalent_actions": equivalent_actions,
            "equivalent_action_fraction": (
                equivalent_actions / total_actions if total_actions else 1.0
            ),
            "canonical_equivalent_actions": canonical_equivalent_actions,
            "canonical_equivalent_action_fraction": (
                canonical_equivalent_actions / total_actions
                if total_actions
                else 1.0
            ),
            "first_miss_families": dict(sorted(first_miss_families.items())),
            "miss_classifications": dict(sorted(miss_classifications.items())),
            "first_miss_classifications": dict(
                sorted(first_miss_classifications.items())
            ),
            "candidate_count": {
                "max": max(selected_candidate_counts, default=0),
                "mean": (
                    sum(selected_candidate_counts) / len(selected_candidate_counts)
                    if selected_candidate_counts
                    else 0.0
                ),
            },
            "decision_modes": dict(sorted(mode_counts.items())),
        },
        "cases": rows,
    }
    _atomic_json(output_path, payload)
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {output_path}")
    return 0


def _self_test() -> None:
    reference.clear_scheduler_caches()
    counts = [4, 2]
    token_dist = {0: 4, 1: 2}
    state = reference.FourStageScheduler(token_dist)._initial_state()
    first = generate_bounded_candidates(state, (1, 1), 32)
    second = generate_bounded_candidates(state, (1, 1), 32)
    if first.actions != second.actions:
        raise AssertionError("candidate generation is not deterministic")
    if not first.actions or len(first.actions) > 32:
        raise AssertionError("candidate budget invariant failed")
    result = run_candidate_target_search(
        counts,
        6 * TICK_CC,
        (1, 1),
        32,
        time_limit_s=30,
        max_expansions=10000,
    )
    if not result.feasible:
        raise AssertionError(f"small candidate oracle failed: {result.termination}")
    replay = reference.validate_schedule_history(result.history, token_dist)
    if replay != 6 * TICK_CC:
        raise AssertionError(f"small replay {replay} != {6 * TICK_CC}")
    print(
        json.dumps(
            {
                "self_test": "passed",
                "initial_candidates": len(first.actions),
                "search_expansions": result.expansions,
                "history_actions": len(result.history),
            },
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", type=Path, default=DEFAULT_PROOF)
    parser.add_argument("--window", nargs=2, type=int, metavar=("TOP", "BOTTOM"))
    parser.add_argument("--k", type=int, choices=SUPPORTED_K)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--time-slice-s", type=float, default=60.0)
    parser.add_argument("--max-expansions", type=int, default=100000)
    parser.add_argument("--max-invocations", type=int, default=0)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--direct-witness-audit",
        type=Path,
        help=(
            "completed 65-case direct window audit; check whether this K-wide "
            "bank reproduces each saved optimal path without exact graph search"
        ),
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    if args.direct_witness_audit is not None:
        if args.window is None or args.k is None or args.output is None:
            parser.error(
                "--direct-witness-audit requires --window, --k and --output"
            )
        return _audit_direct_witness_coverage(
            proof_path=args.proof,
            window_audit_path=args.direct_witness_audit,
            output_path=args.output,
            window=reference.normalize_candidate_window(tuple(args.window)),
            k=args.k,
        )
    if args.window is None or args.k is None or args.work_dir is None or args.output is None:
        parser.error("--window, --k, --work-dir and --output are required")

    proof = json.loads(args.proof.read_text(encoding="utf-8"))
    if not proof.get("complete") or proof.get("summary", {}).get("proven_optimal") != 65:
        raise SystemExit("proof must contain all 65 certified optimal cases")
    cases = list(proof["cases"])
    by_name = {case["name"]: case for case in cases}
    if args.case:
        missing = [name for name in args.case if name not in by_name]
        if missing:
            raise SystemExit(f"unknown case(s): {missing}")
        cases = [by_name[name] for name in args.case]
    window = reference.normalize_candidate_window(tuple(args.window))
    manifest = _manifest(args.proof, window, args.k)
    _ensure_manifest(args.work_dir, manifest)

    invocations = 0
    for case in cases:
        fragment = args.work_dir / f"{case['name']}.json"
        checkpoint_path = args.work_dir / f"{case['name']}.checkpoint.pkl"
        if fragment.exists():
            previous = json.loads(fragment.read_text(encoding="utf-8"))
            if previous.get("status") in {
                "candidate_sufficient",
                "candidate_insufficient",
            }:
                continue
        if args.max_invocations and invocations >= args.max_invocations:
            break
        checkpoint = None
        if checkpoint_path.exists():
            with checkpoint_path.open("rb") as stream:
                checkpoint = pickle.load(stream)
        target = _target_cc(case["best_reference_ticks"])
        print(
            f"[{invocations + 1}] {case['name']} target={_ticks(target)} "
            f"window={window} K={args.k}",
            flush=True,
        )
        reference.clear_scheduler_caches()
        result = run_candidate_target_search(
            case["counts"],
            target,
            window,
            args.k,
            time_limit_s=args.time_slice_s,
            max_expansions=args.max_expansions,
            checkpoint=checkpoint,
        )
        payload = _result_payload(case, target, result)
        _atomic_json(fragment, payload)
        if result.checkpoint is None:
            checkpoint_path.unlink(missing_ok=True)
        else:
            _atomic_pickle(checkpoint_path, result.checkpoint)
        print(
            f"  {payload['status']} termination={result.termination} "
            f"exp={result.expansions} open={result.open_states} "
            f"runtime={result.runtime_s:.3f}s",
            flush=True,
        )
        invocations += 1
        _write_aggregate(args.output, manifest, cases, args.work_dir)
    _write_aggregate(args.output, manifest, cases, args.work_dir)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
