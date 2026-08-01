#!/usr/bin/env python3
"""Freeze the 65-case proof and low-level coverage of the T6+B2 union."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import time

import evaluate_olmoe_fixed_token_banks as bounded
import four_stage_scheduler as reference
import scheduler_hw_fixed_policy as fixed
import scheduler_rtl_adaptive_olmoe_policy as union
import scheduler_rtl_adaptive_prefetch_policy as adaptive


HERE = Path(__file__).resolve().parent
DEFAULT_PROOF = (
    HERE
    / "results"
    / "policy_search"
    / "olmoe_top2_projection_65_optimal_v1.json"
)
DEFAULT_OUT = (
    HERE
    / "results"
    / "policy_search"
    / "scheduler_adaptive_t6b2_joint_union_65_v4.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _distribution(case: dict) -> dict[int, int]:
    return {
        eid: int(ntok)
        for eid, ntok in enumerate(case["counts"])
        if int(ntok) > 0
    }


def _target_cc(case: dict) -> int:
    target = Fraction(str(case["best_reference_ticks"])) * adaptive.TICK_CC
    if target.denominator != 1:
        raise ValueError(f"{case['name']}: non-integral target")
    return int(target)


def _target_states(
    scheduler: union.AdaptiveOlmoeScheduler,
    distribution: dict[int, int],
) -> list[reference.BeamState]:
    state = scheduler.initial_state(distribution)
    targets = []
    while state.remaining:
        _action, state, _score, _candidate_count = scheduler._choose_one_round(
            state
        )
        targets.append(state)
    return targets


def _cover_target_history(
    distribution: dict[int, int],
    targets: list[reference.BeamState],
) -> tuple[tuple[str, ...] | None, int, int]:
    """Find the saved target history in the current low-level generator.

    This search follows one already selected certified history.  It never
    searches alternative expert orders or scores candidates; branching exists
    only when eager-S4PF and explicit-S4PF-OFF children share task endpoints.
    """
    cost_model = adaptive._COST_MODELS[adaptive.DEFAULT_S4_POLICY]
    with adaptive._use_cost_model(cost_model):
        root = fixed.initial_state(distribution)
    memo: set[tuple] = set()
    candidate_count_max = 0
    visited = 0

    def visit(index: int, state: fixed.PolicyState) -> tuple[str, ...] | None:
        nonlocal candidate_count_max, visited
        if index == len(targets):
            return () if not state.remaining else None
        key = (index, fixed.state_key(state))
        if key in memo:
            return None
        memo.add(key)
        visited += 1
        target = targets[index]
        transitions = fixed.generate_top6_bottom2_fixed14_union_successors(
            state,
            policy="balanced",
            top_policy="pruned",
            n1_policy="pruned",
        )
        candidate_count_max = max(candidate_count_max, len(transitions))
        matches = [
            transition
            for transition in transitions
            if transition.state.remaining == tuple(target.remaining)
            and transition.state.c2.task_end == target.c2.task_end
            and transition.state.c3.task_end == target.c3.task_end
        ]
        matches.sort(
            key=lambda transition: (
                not transition.tag.startswith("fixed14"),
                transition.tag,
            )
        )
        for transition in matches:
            suffix = visit(index + 1, transition.state)
            if suffix is not None:
                return (transition.tag,) + suffix
        return None

    return visit(0, root), candidate_count_max, visited


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", type=Path, default=DEFAULT_PROOF)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    proof_path = args.proof.resolve()
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if not proof.get("complete") or len(proof.get("cases", ())) != 65:
        raise ValueError("proof must be the complete 65-case payload")

    scheduler = union.AdaptiveOlmoeScheduler()
    started = time.perf_counter()
    rows = []
    tag_counts: Counter[str] = Counter()
    for case in proof["cases"]:
        distribution = _distribution(case)
        target_cc = _target_cc(case)
        result = scheduler.schedule(distribution)
        old_cc = adaptive.adaptive_prefetch_schedule(distribution)
        targets = _target_states(scheduler, distribution)
        path, generator_candidate_max, coverage_states = _cover_target_history(
            distribution, targets
        )
        if path is not None:
            tag_counts.update(path)
        rows.append(
            {
                "name": case["name"],
                "target_ticks": bounded._ticks_text(target_cc),
                "old_adaptive_ticks": bounded._ticks_text(old_cc),
                "union_ticks": bounded._ticks_text(result.makespan_cc),
                "certified_regime": union.certified_olmoe_regime(distribution),
                "route": result.route,
                "optimal": result.makespan_cc == target_cc,
                "no_regression_vs_old": result.makespan_cc <= old_cc,
                "candidate_count_max": result.candidate_count_max,
                "low_level_history_covered": path is not None,
                "low_level_generator_candidate_count_max": generator_candidate_max,
                "low_level_coverage_states": coverage_states,
                "low_level_tags": list(path) if path is not None else None,
            }
        )

    source_names = (
        "scheduler_rtl_adaptive_prefetch_policy.py",
        "scheduler_hw_fixed_policy.py",
        "scheduler_rtl_adaptive_olmoe_policy.py",
        "scheduler_olmoe_bounded_policy.py",
        "evaluate_olmoe_fixed_token_banks.py",
        Path(__file__).name,
    )
    payload = {
        "schema": "scheduler_adaptive_t6b2_joint_union_65_v4",
        "complete": True,
        "runtime_s": time.perf_counter() - started,
        "configuration": {
            "policy_id": union.POLICY_ID,
            "window": {"head": 6, "bottom": 2},
            "normal_mode": (
                "old-winner-protected T6+B2 with B0@release0 and eager S4PF"
            ),
            "certified_mode": "fixed14 ROM with explicit S4PF=NONE profiles",
            "scorer": union.POLICY_SCORER,
            "candidate_reducer": "sequential best reducer",
            "proof": {
                "path": str(proof_path),
                "sha256": _sha256(proof_path),
            },
            "token_bank": {
                "path": str(scheduler.token_bank_path),
                "sha256": _sha256(scheduler.token_bank_path),
                "entries": len(scheduler.tokens),
            },
            "sources": {
                name: {
                    "path": str((HERE / name).resolve()),
                    "sha256": _sha256(HERE / name),
                }
                for name in source_names
            },
        },
        "summary": {
            "cases": len(rows),
            "certified_regime_cases": sum(row["certified_regime"] for row in rows),
            "optimal_cases": sum(row["optimal"] for row in rows),
            "no_regression_vs_old_cases": sum(
                row["no_regression_vs_old"] for row in rows
            ),
            "better_equal_worse_vs_old": {
                "better": sum(
                    Fraction(row["union_ticks"])
                    < Fraction(row["old_adaptive_ticks"])
                    for row in rows
                ),
                "equal": sum(
                    Fraction(row["union_ticks"])
                    == Fraction(row["old_adaptive_ticks"])
                    for row in rows
                ),
                "worse": sum(
                    Fraction(row["union_ticks"])
                    > Fraction(row["old_adaptive_ticks"])
                    for row in rows
                ),
            },
            "old_adaptive_optimal_cases": sum(
                Fraction(row["old_adaptive_ticks"])
                == Fraction(row["target_ticks"])
                for row in rows
            ),
            "old_adaptive_gap_ticks_sum": str(
                sum(
                    Fraction(row["old_adaptive_ticks"])
                    - Fraction(row["target_ticks"])
                    for row in rows
                )
            ),
            "union_gap_ticks_sum": str(
                sum(
                    Fraction(row["union_ticks"])
                    - Fraction(row["target_ticks"])
                    for row in rows
                )
            ),
            "certified_candidate_count_max": max(
                row["candidate_count_max"] for row in rows
            ),
            "low_level_history_covered_cases": sum(
                row["low_level_history_covered"] for row in rows
            ),
            "low_level_generator_candidate_count_max": max(
                row["low_level_generator_candidate_count_max"] for row in rows
            ),
            "selected_low_level_tags": dict(sorted(tag_counts.items())),
        },
        "rows": rows,
    }
    payload["complete"] = all(
        (
            payload["summary"]["cases"] == 65,
            payload["summary"]["certified_regime_cases"] == 65,
            payload["summary"]["optimal_cases"] == 65,
            payload["summary"]["no_regression_vs_old_cases"] == 65,
            payload["summary"]["low_level_history_covered_cases"] == 65,
        )
    )
    _atomic_write(args.out, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.out.resolve()}")
    return 0 if payload["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
