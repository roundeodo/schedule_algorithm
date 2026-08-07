#!/usr/bin/env python3
"""Evaluate multi-path search over the exact deployed distilled candidate graph.

The restricted search and the online policy share the observation window, the
28 compiled physical profiles, targeted S4 prefetch lowering, local physical
reduction, and continuation comparator.  The online policy commits the first
winner in every round.  This diagnostic retains alternative branches; the
comparator changes only their expansion order.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import heapq
import json
import os
from pathlib import Path
import time

import scheduler_rtl_distilled_policy as distilled
import scheduler_rtl_distilled_scoring as scoring
import scheduler_rtl_distilled_lowering as lowering
import verify_scheduler_rtl_unified_policy as datasets


HERE = Path(__file__).resolve().parent
SUITES = tuple(
    HERE / f"scheduler_strategy_coverage_E{experts}.json"
    for experts in (8, 32, 64)
)
REFERENCES = tuple(
    HERE
    / "results/legacy_scheduler_algorithm/final_reference"
    / f"scheduler_reference_E{experts}_compact.json"
    for experts in (8, 32, 64)
)
DEFAULT_OUT = (
    HERE
    / "results/policy_search"
    / "distilled_top5_bottom1_28profile_restricted_search.json"
)


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_references() -> dict[str, dict]:
    rows = {}
    for path in REFERENCES:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for case_id, row in payload["results"].items():
            key = f"E{int(row['e_total'])}:{int(case_id)}"
            if not row.get("history_validated", False):
                raise ValueError(f"{key}: offline history is not replay validated")
            rows[key] = {
                "reference_cc": int(row["makespan_cc"]),
                "reference_lower_bound_cc": int(row["lower_bound_cc"]),
                "reference_proven_optimal": bool(row["proven_optimal"]),
            }
    return rows


def _ordered_children(state):
    """Return all children, with the deployed comparator winner first."""
    slots = list(distilled.generate_candidate_slots(state, enable_s4pf=True))
    pending = [(slot.action, slot.child) for slot in slots]
    ordered = []
    while pending:
        _score, index, action, child, _metadata = (
            scoring.select_continuation_transition_winner(state, pending)
        )
        ordered.append((action, child))
        pending.pop(index)
    return ordered


def _frontier_key(state, sibling_rank: int) -> tuple[int, ...]:
    components = scoring.lower_bound_components(state)
    return (
        int(state.f_score),
        int(scoring.head5_hist4_lpt(state)),
        int(components["compute_cc"]),
        int(components["dma_capacity_cc"]),
        int(state.g_score),
        int(sibling_rank),
        len(state.remaining),
    )


def _restricted_search(job: dict, *, time_limit_s: float, max_expansions: int) -> dict:
    distribution = dict(job["distribution"])
    c2, c3 = int(job["c2"]), int(job["c3"])
    reference_cc = int(job["reference_cc"])
    global_lower_bound = int(job["reference_lower_bound_cc"])
    online = distilled.schedule(distribution, c2, c3, enable_s4pf=True)
    incumbent = int(online.makespan_cc)
    started = time.perf_counter()

    if incumbent == global_lower_bound:
        return {
            "key": job["key"],
            "e_total": int(job["e_total"]),
            "dataset_split": job["dataset_split"],
            "reference_cc": reference_cc,
            "reference_lower_bound_cc": global_lower_bound,
            "reference_proven_optimal": bool(job["reference_proven_optimal"]),
            "online_cc": incumbent,
            "restricted_cc": incumbent,
            "restricted_lower_bound_cc": incumbent,
            "restricted_optimal": True,
            "termination": "incumbent_equals_global_lower_bound",
            "expansions": 0,
            "generated": 0,
            "peak_frontier": 0,
            "runtime_s": time.perf_counter() - started,
        }

    root = distilled._initial_state(distribution, c2, c3)
    frontier = []
    next_id = 0
    heapq.heappush(frontier, (_frontier_key(root, 0), next_id, root))
    best_bound_by_state = {lowering.child_key(root): int(root.f_score)}
    peak_frontier = 1
    expansions = 0
    generated = 0
    termination = "frontier_exhausted"

    while frontier:
        if time.perf_counter() - started >= time_limit_s:
            termination = "time_limit"
            break
        if expansions >= max_expansions:
            termination = "expansion_limit"
            break
        priority, _entry_id, state = heapq.heappop(frontier)
        state_key = lowering.child_key(state)
        if best_bound_by_state.get(state_key) != int(state.f_score):
            continue
        state_bound = max(global_lower_bound, int(state.f_score))
        if state_bound >= incumbent:
            continue
        expansions += 1
        children = _ordered_children(state)
        generated += len(children)
        for sibling_rank, (_action, child) in enumerate(children):
            child_bound = max(global_lower_bound, int(child.f_score))
            if not child.remaining:
                incumbent = min(incumbent, int(child.g_score))
                if incumbent == global_lower_bound:
                    frontier.clear()
                    termination = "incumbent_equals_global_lower_bound"
                    break
                continue
            if child_bound >= incumbent:
                continue
            child_key = lowering.child_key(child)
            previous = best_bound_by_state.get(child_key)
            if previous is not None and previous <= int(child.f_score):
                continue
            best_bound_by_state[child_key] = int(child.f_score)
            next_id += 1
            heapq.heappush(
                frontier,
                (_frontier_key(child, sibling_rank), next_id, child),
            )
        peak_frontier = max(peak_frontier, len(frontier))

    live_lower_bound = min(
        (max(global_lower_bound, int(item[2].f_score)) for item in frontier),
        default=incumbent,
    )
    restricted_optimal = not frontier or live_lower_bound >= incumbent
    if restricted_optimal and termination not in {
        "incumbent_equals_global_lower_bound",
        "frontier_exhausted",
    }:
        termination = "frontier_bound_equals_incumbent"
    return {
        "key": job["key"],
        "e_total": int(job["e_total"]),
        "dataset_split": job["dataset_split"],
        "reference_cc": reference_cc,
        "reference_lower_bound_cc": global_lower_bound,
        "reference_proven_optimal": bool(job["reference_proven_optimal"]),
        "online_cc": int(online.makespan_cc),
        "restricted_cc": incumbent,
        "restricted_lower_bound_cc": live_lower_bound,
        "restricted_optimal": restricted_optimal,
        "termination": termination,
        "expansions": expansions,
        "generated": generated,
        "peak_frontier": peak_frontier,
        "runtime_s": time.perf_counter() - started,
    }


def _worker(args: tuple[dict, float, int]) -> dict:
    job, time_limit_s, max_expansions = args
    return _restricted_search(
        job,
        time_limit_s=time_limit_s,
        max_expansions=max_expansions,
    )


def _summary(rows: list[dict]) -> dict:
    if not rows:
        return {"cases": 0}
    exact = [row for row in rows if row["restricted_optimal"]]
    return {
        "cases": len(rows),
        "restricted_optimal_cases": len(exact),
        "restricted_improves_online_cases": sum(
            row["restricted_cc"] < row["online_cc"] for row in rows
        ),
        "restricted_equals_online_cases": sum(
            row["restricted_cc"] == row["online_cc"] for row in rows
        ),
        "restricted_matches_reference_cases": sum(
            row["restricted_cc"] == row["reference_cc"] for row in rows
        ),
        "online_total_cc": sum(row["online_cc"] for row in rows),
        "restricted_total_cc": sum(row["restricted_cc"] for row in rows),
        "reference_total_cc": sum(row["reference_cc"] for row in rows),
        "expansions": sum(row["expansions"] for row in rows),
        "runtime_s": sum(row["runtime_s"] for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--time-limit", type=float, default=1.0)
    parser.add_argument("--max-expansions", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    args = parser.parse_args()

    references = _load_references()
    jobs = datasets._dataset_jobs(tuple(path.resolve() for path in SUITES))
    for job in jobs:
        job.update(references[job["key"]])
    if args.limit is not None:
        jobs = jobs[: args.limit]
    worker_args = [
        (job, float(args.time_limit), int(args.max_expansions)) for job in jobs
    ]
    if args.workers == 1:
        rows = [_worker(item) for item in worker_args]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(_worker, worker_args))
    rows.sort(key=lambda row: (row["e_total"], row["key"]))
    payload = {
        "schema": "distilled_restricted_search_v1",
        "configuration": {
            "window": list(distilled.WINDOW),
            "compiled_profiles": len(distilled.COMPILED_PROFILES),
            "continuation_scorer": distilled.CONTINUATION_SCORER,
            "time_limit_s_per_case": float(args.time_limit),
            "max_expansions_per_case": int(args.max_expansions),
        },
        "complete": len(rows) == len(jobs),
        "summary": _summary(rows),
        "rows": {row["key"]: row for row in rows},
    }
    _atomic_write(args.out.resolve(), payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
