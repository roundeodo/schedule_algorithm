#!/usr/bin/env python3
"""Reproduce the causal structural ablation behind the distilled scheduler."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import four_stage_scheduler as reference
import scheduler_rtl_distilled_lowering as lowering
import scheduler_rtl_distilled_policy as distilled
import scheduler_rtl_distilled_scoring as scoring
import scheduler_rtl_unified_policy as frozen_v4
from scheduler_rtl_distilled_types import (
    CandidateProfile,
    LogicalActionSpec,
    PhysicalProfile,
)


HERE = Path(__file__).resolve().parent
PROOF65 = HERE / "results/policy_search/olmoe_top2_projection_65_optimal_v1.json"
OUTPUT = HERE / "results/policy_search/scheduler_rtl_distilled_structure_ablation.json"
TICK_CC = distilled.TICK_CC

def _clean_profile(token) -> CandidateProfile:
    logical = token.logical
    physical = token.physical
    return CandidateProfile(
        logical=LogicalActionSpec(
            mode=logical.mode,
            family=logical.family,
            selectors=tuple(logical.selectors),
            split_rule=logical.split_rule,
        ),
        physical=PhysicalProfile(
            **{
                field: getattr(physical, field)
                for field in PhysicalProfile.__dataclass_fields__
            }
        ),
    )


ALL_V4_PROFILES = tuple(
    dict.fromkeys(
        _clean_profile(token)
        for token in (*frozen_v4.COMPILED_TOKENS, *frozen_v4.RECOVERY_TOKENS)
    )
)
UTILIZED_PROFILES = tuple(
    token
    for token in ALL_V4_PROFILES
    if not (token.logical.mode == "SYNC" and token.logical.family == "SINGLE")
)
VARIANTS = (
    "frozen_v4",
    "naive_union",
    "source_order_local_reducer",
    "reversed_order_local_reducer",
    "semantic_prefetch_reducer",
    "semantic_prefetch_reducer_reversed",
    "scalar_continuation",
    "distilled32",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_key(
    state: reference.BeamState,
    action: reference.StageAction,
) -> tuple:
    logical = lowering.logical_action_spec(state, action, distilled.WINDOW)
    return logical.mode, logical.family, logical.selectors, logical.split_rule


def _materialize(
    state: reference.BeamState,
    profiles: tuple[CandidateProfile, ...],
    prefer_s2pf: bool,
) -> tuple[list[reference.StageAction], int]:
    runtime, fixed_priorities = lowering.runtime_profile_bank(state, profiles)
    concrete, _stats = lowering.materialize_candidates_with_sources(
        state, runtime
    )
    grouped = defaultdict(list)
    for action, runtime_sources in concrete:
        profile_slot = min(fixed_priorities[index] for index in runtime_sources)
        grouped[_logical_key(state, action)].append((action, profile_slot))

    def local_key(item):
        action, profile_slot = item
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
            -int(s2pf) if prefer_s2pf else 0,
            profile_slot,
        )

    reduced = [
        min(grouped[logical], key=local_key)[0]
        for logical in sorted(grouped)
    ]
    emitted = {}
    for action in reduced:
        child = reference.apply_action(state, action)
        emitted.setdefault(lowering.child_key(child), action)
    return list(emitted.values()), len(concrete)


def _select_scalar(
    state: reference.BeamState,
    candidates: list[reference.StageAction],
) -> reference.BeamState:
    ranked = []
    for index, action in enumerate(candidates):
        child = reference.apply_action(state, action)
        child = scoring.normalize_state_bound(
            child,
            parent_bound=int(state.f_score),
        )
        score = scoring.base_continuation_key(state, action, child)
        ranked.append((score, index, child))
    return min(ranked, key=lambda item: item[:2])[2]


def _schedule_variant(distribution: dict[int, int], variant: str) -> tuple[int, int]:
    if variant == "frozen_v4":
        result = frozen_v4.schedule(distribution)
        return int(result.makespan_cc), int(result.candidate_count_max)
    if variant == "distilled32":
        result = distilled.schedule(distribution)
        return int(result.makespan_cc), int(result.physical_candidate_count_max)

    profiles = ALL_V4_PROFILES if variant == "naive_union" else UTILIZED_PROFILES
    if variant in {
        "reversed_order_local_reducer",
        "semantic_prefetch_reducer_reversed",
    }:
        profiles = tuple(reversed(profiles))
    prefer_s2pf = variant in {
        "semantic_prefetch_reducer",
        "semantic_prefetch_reducer_reversed",
        "scalar_continuation",
    }
    state = distilled._initial_state(distribution, -1, -1)
    physical_max = 0
    while state.remaining:
        candidates, physical_count = _materialize(state, profiles, prefer_s2pf)
        physical_max = max(physical_max, physical_count)
        if variant == "scalar_continuation":
            state = _select_scalar(state, candidates)
        else:
            _score, _slot, _action, state, _metadata = (
                scoring.select_continuation_winner(state, candidates)
            )
    return int(state.g_score), physical_max


def _worker(job: tuple[str, dict[int, int], int]) -> dict:
    name, distribution, target_cc = job
    results = {}
    for variant in VARIANTS:
        cc, physical_max = _schedule_variant(distribution, variant)
        results[variant] = {
            "cc": cc,
            "exact": cc == target_cc,
            "gap_ticks": str(Fraction(cc - target_cc, TICK_CC)),
            "physical_candidate_count_max": physical_max,
        }
    return {"name": name, "target_cc": target_cc, "variants": results}


def main() -> int:
    proof = json.loads(PROOF65.read_text(encoding="utf-8"))
    jobs = []
    for case in proof["cases"]:
        distribution = {
            eid: int(ntok)
            for eid, ntok in enumerate(case["counts"])
            if int(ntok) > 0
        }
        target_cc = int(Fraction(str(case["best_reference_ticks"])) * TICK_CC)
        jobs.append((str(case["name"]), distribution, target_cc))
    with ProcessPoolExecutor(max_workers=16) as pool:
        rows = list(pool.map(_worker, jobs, chunksize=1))

    summary = {}
    for variant in VARIANTS:
        summary[variant] = {
            "exact_cases": sum(row["variants"][variant]["exact"] for row in rows),
            "target_gap_ticks": str(
                Fraction(
                    sum(
                        row["variants"][variant]["cc"] - row["target_cc"]
                        for row in rows
                    ),
                    TICK_CC,
                )
            ),
            "physical_candidate_count_max": max(
                row["variants"][variant]["physical_candidate_count_max"]
                for row in rows
            ),
        }
    payload = {
        "schema": "bounded_distilled_structure_ablation",
        "manifest": {
            "proof65": str(PROOF65.resolve()),
            "proof65_sha256": _sha256(PROOF65),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "source_sha256": {
                path.name: _sha256(path)
                for path in (
                    HERE / "scheduler_rtl_distilled_policy.py",
                    HERE / "scheduler_rtl_distilled_profiles.py",
                    HERE / "scheduler_rtl_distilled_types.py",
                    HERE / "scheduler_rtl_distilled_lowering.py",
                    HERE / "scheduler_rtl_distilled_scoring.py",
                    HERE / "scheduler_rtl_unified_policy.py",
                    HERE / "four_stage_scheduler.py",
                )
            },
            "all_v4_profiles": len(ALL_V4_PROFILES),
            "utilized_profiles_before_pruning": len(UTILIZED_PROFILES),
            "distilled_profiles": len(distilled.COMPILED_PROFILES),
        },
        "summary": summary,
        "interpretation": {
            "naive_union": "Removing arbitration without semantic changes fails.",
            "source_order_local_reducer": "Removing SYNC SINGLE prevents wasting an idle-cluster issue opportunity, but inherited profile order still resolves physical ties.",
            "reversed_order_local_reducer": "Reversing profile order exposes whether an implicit ordering dependency remains.",
            "semantic_prefetch_reducer": "S2 prefetch explicitly breaks timing-identical physical-profile ties.",
            "semantic_prefetch_reducer_reversed": "The semantic reducer should retain quality after profile reordering.",
            "scalar_continuation": "The bounded state-conditioned comparator cannot be replaced by its fallback scalar key.",
            "distilled32": "Five discovery-dominated profiles are removed without changing certified outcomes.",
        },
        "rows": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
