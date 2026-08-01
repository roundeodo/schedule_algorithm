#!/usr/bin/env python3
"""Evaluate simple legal largest-first policies on the frozen OLMoE corpus.

These baselines intentionally have no continuation scorer and no explicit
free-prefetch action.  At a synchronized boundary they either pair the two
largest remaining experts (``top_top``) or the largest and smallest
(``hot_cold``).  When only one cluster is free, they immediately issue the
largest remaining expert there.  For the fixed expert choice, all legal
four-stage shape, S2 down-prefetch and DMA-lane variants are generated and the
locally earliest completion is selected.  This gives the simple high-level
policy a favorable physical lowering without adding future search.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import astuple
from fractions import Fraction
import json
import os
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import four_stage_scheduler as reference  # noqa: E402
from run_four_stage_reference import serialize_action  # noqa: E402


DEFAULT_CERTIFICATE = (
    ROOT / "results" / "policy_search" / "olmoe_top2_projection_65_optimal_v1.json"
)
DEFAULT_OUT = (
    ROOT / "results" / "policy_search" / "olmoe_65_simple_balance_policy_v1.json"
)
TICK_CC = reference.SHAPE_C.T_s3
POLICIES = ("top_top", "hot_cold")


def _ticks(cc: int) -> str:
    value = Fraction(int(cc), TICK_CC)
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _action_eids(action: reference.StageAction) -> tuple[int, ...]:
    return tuple(
        eid for eid in (int(action.c2_eid), int(action.c3_eid)) if eid >= 0
    )


def _local_action_key(
    action: reference.StageAction, child: reference.BeamState
) -> tuple:
    ends = (int(child.c2.task_end), int(child.c3.task_end))
    # repr(astuple(...)) is used only as a stable deterministic final tie-break.
    return (
        max(ends),
        abs(ends[0] - ends[1]),
        sum(ends),
        repr(astuple(action)),
    )


def _selected_pair_counts(
    remaining: tuple[tuple[int, int], ...], policy: str
) -> tuple[int, int]:
    if len(remaining) < 2:
        raise ValueError("pair selection requires at least two experts")
    largest = int(remaining[0][1])
    partner = int(remaining[1][1] if policy == "top_top" else remaining[-1][1])
    return tuple(sorted((largest, partner), reverse=True))


def _action_counts(action: reference.StageAction) -> tuple[int, ...]:
    return tuple(
        sorted(
            (
                int(ntok)
                for eid, ntok in (
                    (action.c2_eid, action.c2_ntok),
                    (action.c3_eid, action.c3_ntok),
                )
                if int(eid) >= 0
            ),
            reverse=True,
        )
    )


def _run_policy(counts: list[int], policy: str) -> dict:
    if policy not in POLICIES:
        raise ValueError(policy)
    token_dist = {eid: int(ntok) for eid, ntok in enumerate(counts)}
    reference.clear_scheduler_caches()
    state = reference.FourStageScheduler(token_dist)._initial_state()
    decisions = []
    while state.remaining:
        t2 = int(state.c2.task_end)
        t3 = int(state.c3.task_end)
        if len(state.remaining) == 1 or t2 != t3:
            selected_remaining = (state.remaining[0],)
        elif policy == "top_top":
            selected_remaining = (state.remaining[0], state.remaining[1])
        else:
            selected_remaining = (state.remaining[0], state.remaining[-1])
        # Physical action generation depends on the selected experts and the
        # current snapshots, not on unselected tail identities when no capacity
        # pruning is requested.  Restricting this call avoids enumerating every
        # pair only to discard it below; apply_action still consumes the action
        # from the authoritative full remaining set in ``state``.
        actions = reference.gen_stage_actions(
            state.c2,
            state.c3,
            selected_remaining,
        )
        target = int(state.remaining[0][0])
        target_ntok = int(state.remaining[0][1])
        if len(state.remaining) == 1:
            if t2 < t3:
                eligible = [
                    action
                    for action in actions
                    if action.c2_eid == target and action.c3_eid < 0
                ]
            elif t3 < t2:
                eligible = [
                    action
                    for action in actions
                    if action.c3_eid == target and action.c2_eid < 0
                ]
            else:
                # A final split is retained: it is a bounded local tail choice,
                # not a continuation search over future experts.
                eligible = [
                    action for action in actions if set(_action_eids(action)) == {target}
                ]
        elif t2 == t3:
            pair_counts = _selected_pair_counts(state.remaining, policy)
            eligible = [
                action
                for action in actions
                if len(_action_eids(action)) == 2
                and len(set(_action_eids(action))) == 2
                and _action_counts(action) == pair_counts
            ]
        elif t2 < t3:
            eligible = [
                action
                for action in actions
                if action.c2_ntok == target_ntok and action.c3_eid < 0
            ]
        else:
            eligible = [
                action
                for action in actions
                if action.c3_ntok == target_ntok and action.c2_eid < 0
            ]
        if not eligible:
            raise RuntimeError(
                f"{policy}: no legal local action with {len(state.remaining)} "
                f"experts and ends {(t2, t3)}"
            )
        choices = [(action, reference.apply_action(state, action)) for action in eligible]
        action, child = min(
            choices,
            key=lambda item: _local_action_key(item[0], item[1]),
        )
        decisions.append(
            {
                "remaining_before": len(state.remaining),
                "candidate_variants": len(eligible),
                "selected": serialize_action(action),
                "ends_after_ticks": [
                    _ticks(child.c2.task_end),
                    _ticks(child.c3.task_end),
                ],
            }
        )
        state = child
    history = tuple(step["selected"] for step in decisions)
    validated = reference.validate_schedule_history(state.history, token_dist)
    if validated != state.g_score:
        raise RuntimeError(
            f"{policy}: validator {validated} != replay state {state.g_score}"
        )
    return {
        "makespan_cc": int(state.g_score),
        "makespan_ticks": _ticks(state.g_score),
        "terminal_ticks": [_ticks(state.c2.task_end), _ticks(state.c3.task_end)],
        "action_count": len(decisions),
        "decisions": decisions,
        "serialized_history": list(history),
        "history_replay_valid": True,
    }


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def _summary(rows: list[dict], policy: str) -> dict:
    gaps = [
        float(Fraction(row[policy]["makespan_ticks"]))
        - float(Fraction(row["optimal_ticks"]))
        for row in rows
    ]
    ratios = [
        float(Fraction(row[policy]["makespan_ticks"]))
        / float(Fraction(row["optimal_ticks"]))
        for row in rows
    ]
    return {
        "cases": len(rows),
        "optimal_cases": sum(abs(gap) < 1e-12 for gap in gaps),
        "gap_ticks": {
            "sum": sum(gaps),
            "mean": statistics.mean(gaps),
            "p50": statistics.median(gaps),
            "p95": _percentile(gaps, 0.95),
            "max": max(gaps),
        },
        "ratio": {
            "mean": statistics.mean(ratios),
            "p50": statistics.median(ratios),
            "p95": _percentile(ratios, 0.95),
            "max": max(ratios),
        },
    }


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _evaluate_case(payload: tuple[int, dict]) -> dict:
    index, case = payload
    row = {
        "index": index,
        "name": case["name"],
        "counts": case["counts"],
        "optimal_ticks": str(case["best_reference_ticks"]),
    }
    for policy in POLICIES:
        row[policy] = _run_policy(case["counts"], policy)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate", type=Path, default=DEFAULT_CERTIFICATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()
    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    cases = certificate["cases"]
    if args.limit >= 0:
        cases = cases[: args.limit]
    if args.workers <= 0:
        parser.error("--workers must be positive")
    rows = []
    started = time.perf_counter()
    indexed_cases = list(enumerate(cases, 1))
    if args.workers == 1:
        for completed, payload in enumerate(indexed_cases, 1):
            rows.append(_evaluate_case(payload))
            if args.progress_every > 0 and completed % args.progress_every == 0:
                print(
                    f"cases={completed}/{len(cases)} "
                    f"elapsed_s={time.perf_counter()-started:.3f}",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_evaluate_case, payload): payload[0]
                for payload in indexed_cases
            }
            for completed, future in enumerate(as_completed(futures), 1):
                rows.append(future.result())
                if args.progress_every > 0 and completed % args.progress_every == 0:
                    print(
                        f"cases={completed}/{len(cases)} "
                        f"elapsed_s={time.perf_counter()-started:.3f}",
                        flush=True,
                    )
    rows.sort(key=lambda row: row["index"])
    report = {
        "schema": "olmoe-simple-balance-policy-v1",
        "complete": len(cases) == 65,
        "configuration": {
            "policies": list(POLICIES),
            "continuation_scorer": "none",
            "free_prefetch_actions": False,
            "physical_variant_selection": "minimum immediate completion",
            "final_single_expert_split": True,
            "workers": args.workers,
        },
        "summary": {policy: _summary(rows, policy) for policy in POLICIES},
        "runtime_s": time.perf_counter() - started,
        "rows": rows,
    }
    _atomic_write(args.out, report)
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
