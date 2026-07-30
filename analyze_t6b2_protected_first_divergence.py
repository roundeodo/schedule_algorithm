#!/usr/bin/env python3
"""Audit the first closed-loop divergence caused by protected T6+B2 actions."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import compare_scheduler_adaptive_t6b2_30k as coverage
import scheduler_hw_fixed_policy as fixed
import scheduler_rtl_adaptive_prefetch_policy as adaptive


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = (
    HERE
    / "results"
    / "policy_search"
    / "scheduler_adaptive_t6b2_first_divergence_30k_v1.json"
)
EXPECTED_CASES = 29_928
HEAD_POLICY = "protected_headcritical_slack_1"
FINAL_POLICY = "protected_tail_b0_slack_1"


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


def _finish_old(state: fixed.PolicyState, cost_model) -> int:
    while state.remaining:
        state = adaptive._choose_transition(
            state,
            cost_model,
            candidate_policy="baseline",
            score_policy="legacy",
        ).state
    return int(fixed.terminal_cost(state))


def _first_divergence(
    distribution: dict[int, int],
    c2: int,
    c3: int,
    score_policy: str,
) -> dict | None:
    cost_model = adaptive._COST_MODELS[adaptive.DEFAULT_S4_POLICY]
    state = adaptive.initial_state(distribution, c2, c3)
    depth = 0
    while state.remaining:
        old = adaptive._choose_transition(
            state,
            cost_model,
            candidate_policy="baseline",
            score_policy="legacy",
        )
        selected = adaptive._choose_transition(
            state,
            cost_model,
            candidate_policy=fixed.TOP6_BOTTOM2_CANDIDATE_POLICY,
            score_policy=score_policy,
        )
        if fixed.state_key(old.state) != fixed.state_key(selected.state):
            break
        state = old.state
        depth += 1
    else:
        return None

    with adaptive._use_cost_model(cost_model):
        old_continuation = int(
            fixed.hw_v2_continuation(
                old.state.c2,
                old.state.c3,
                old.state.remaining,
                policy="balanced",
            )
        )
        selected_continuation = int(
            fixed.hw_v2_continuation(
                selected.state.c2,
                selected.state.c3,
                selected.state.remaining,
                policy="balanced",
            )
        )
        selected_head_finish = (
            min(selected.state.c2.task_end, selected.state.c3.task_end)
            + int(fixed.cm._cc_best_task(selected.state.remaining[0][1]))
            if selected.state.remaining
            else max(selected.state.c2.task_end, selected.state.c3.task_end)
        )

    old_rollout = _finish_old(old.state, cost_model)
    selected_rollout = _finish_old(selected.state, cost_model)
    return {
        "depth": depth,
        "mode": "SYNC" if state.c2.task_end == state.c3.task_end else "ONE_IDLE",
        "remaining_count": len(state.remaining),
        "remaining_loads": [int(ntok) for _eid, ntok in state.remaining],
        "input_task_ends_cc": [int(state.c2.task_end), int(state.c3.task_end)],
        "old_tag": old.tag,
        "selected_tag": selected.tag,
        "old_child_task_ends_cc": [
            int(old.state.c2.task_end),
            int(old.state.c3.task_end),
        ],
        "selected_child_task_ends_cc": [
            int(selected.state.c2.task_end),
            int(selected.state.c3.task_end),
        ],
        "old_continuation_cc": old_continuation,
        "selected_continuation_cc": selected_continuation,
        "selected_head_finish_cc": int(selected_head_finish),
        "old_rollout_cc": old_rollout,
        "selected_rollout_cc": selected_rollout,
        "rollout_delta_cc": selected_rollout - old_rollout,
    }


def _worker(job: dict) -> dict:
    distribution = {int(eid): int(ntok) for eid, ntok in job["dist"].items()}
    c2, c3 = int(job["c2"]), int(job["c3"])
    baseline = adaptive.adaptive_prefetch_schedule_result(distribution, c2, c3)
    head = adaptive.adaptive_prefetch_schedule_result(
        distribution,
        c2,
        c3,
        candidate_policy=fixed.TOP6_BOTTOM2_CANDIDATE_POLICY,
        score_policy=HEAD_POLICY,
    )
    final = adaptive.adaptive_prefetch_schedule_result(
        distribution,
        c2,
        c3,
        candidate_policy=fixed.TOP6_BOTTOM2_PROTECTED_CANDIDATE_POLICY,
        score_policy=FINAL_POLICY,
    )
    return {
        "key": job["key"],
        "e_total": int(job["e_total"]),
        "dataset_split": job["dataset_split"],
        "baseline_cc": int(baseline.makespan_cc),
        "headcritical_cc": int(head.makespan_cc),
        "final_cc": int(final.makespan_cc),
        "headcritical_first_divergence": (
            _first_divergence(distribution, c2, c3, HEAD_POLICY)
            if head.makespan_cc != baseline.makespan_cc
            else None
        ),
        "final_first_divergence": (
            _first_divergence(distribution, c2, c3, FINAL_POLICY)
            if final.makespan_cc != baseline.makespan_cc
            else None
        ),
    }


def _comparison(rows: list[dict], field: str) -> dict:
    return {
        "better": sum(row[field] < row["baseline_cc"] for row in rows),
        "equal": sum(row[field] == row["baseline_cc"] for row in rows),
        "worse": sum(row[field] > row["baseline_cc"] for row in rows),
        "aggregate_delta_cc": sum(
            row[field] - row["baseline_cc"] for row in rows
        ),
    }


def _divergence_summary(rows: list[dict], field: str) -> dict:
    affected = [row for row in rows if row[field] is not None]
    losses = [
        row for row in affected
        if row[field]["rollout_delta_cc"] > 0
    ]
    wins = [
        row for row in affected
        if row[field]["rollout_delta_cc"] < 0
    ]
    return {
        "affected_cases": len(affected),
        "beneficial_first_actions": len(wins),
        "harmful_first_actions": len(losses),
        "zero_delta_first_actions": len(affected) - len(wins) - len(losses),
        "mode_counts": dict(sorted(Counter(row[field]["mode"] for row in affected).items())),
        "remaining_count_counts": dict(
            sorted(Counter(str(row[field]["remaining_count"]) for row in affected).items())
        ),
        "selected_tag_counts": dict(
            sorted(Counter(row[field]["selected_tag"] for row in affected).items())
        ),
        "harmful_remaining_count_counts": dict(
            sorted(Counter(str(row[field]["remaining_count"]) for row in losses).items())
        ),
        "harmful_selected_tag_counts": dict(
            sorted(Counter(row[field]["selected_tag"] for row in losses).items())
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    args = parser.parse_args()
    jobs = coverage._load_jobs(coverage.DEFAULT_INPUTS)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_worker, jobs, chunksize=8))
    if len(rows) != EXPECTED_CASES:
        raise RuntimeError(f"expected {EXPECTED_CASES} rows, got {len(rows)}")

    payload = {
        "schema": "scheduler_adaptive_t6b2_first_divergence_30k_v1",
        "complete": True,
        "runtime_s": time.perf_counter() - started,
        "configuration": {
            "headcritical_policy": HEAD_POLICY,
            "final_policy": FINAL_POLICY,
            "inputs": [
                {"path": str(path.resolve()), "sha256": _sha256(path)}
                for path in coverage.DEFAULT_INPUTS
            ],
            "sources": {
                name: {
                    "path": str((HERE / name).resolve()),
                    "sha256": _sha256(HERE / name),
                }
                for name in (
                    "scheduler_hw_fixed_policy.py",
                    "scheduler_rtl_adaptive_prefetch_policy.py",
                    Path(__file__).name,
                )
            },
        },
        "summary": {
            "cases": len(rows),
            "headcritical_vs_baseline": _comparison(rows, "headcritical_cc"),
            "final_vs_baseline": _comparison(rows, "final_cc"),
            "headcritical_first_divergence": _divergence_summary(
                rows, "headcritical_first_divergence"
            ),
            "final_first_divergence": _divergence_summary(
                rows, "final_first_divergence"
            ),
        },
        "affected_rows": {
            row["key"]: row
            for row in rows
            if row["headcritical_first_divergence"] is not None
            or row["final_first_divergence"] is not None
        },
    }
    _atomic_write(args.out, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
