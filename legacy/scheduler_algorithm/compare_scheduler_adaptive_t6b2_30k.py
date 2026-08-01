#!/usr/bin/env python3
"""Paired sweep of old-winner-protected T6+B2 acceptance rules.

Every arm first reproduces the old adaptive winner.  Only added T0+B0/B0/B1
candidates may replace it, and only when a fixed continuation margin/slot
guard passes.  This isolates bottom-candidate value from global rescoring of
the old bank.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import scheduler_hw_fixed_policy as fixed
from scheduler_rtl_adaptive_prefetch_policy import (
    adaptive_prefetch_schedule_result,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUTS = tuple(
    ROOT / f"scheduler_strategy_coverage_E{experts}.json"
    for experts in (8, 32, 64)
)
DEFAULT_OUT = (
    ROOT
    / "results"
    / "policy_search"
    / "scheduler_adaptive_t6b2_tail_b0_sweep_30k_v10.json"
)
SCHEMA = "scheduler_adaptive_t6b2_tail_b0_sweep_30k_v10"
EXPECTED_CASES = 29_928
SCORE_ARMS = {
    "protected_sync_1_cc": "protected_sync_1",
    "protected_headcritical_slack_1_cc": "protected_headcritical_slack_1",
    "protected_tail_b0_slack_1_cc": "protected_tail_b0_slack_1",
}


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


def _comparison(rows: list[dict], lhs: str, rhs: str) -> dict:
    lhs_total = sum(int(row[lhs]) for row in rows)
    rhs_total = sum(int(row[rhs]) for row in rows)
    return {
        "lhs": lhs,
        "rhs": rhs,
        "cases": len(rows),
        "better": sum(int(row[lhs]) < int(row[rhs]) for row in rows),
        "equal": sum(int(row[lhs]) == int(row[rhs]) for row in rows),
        "worse": sum(int(row[lhs]) > int(row[rhs]) for row in rows),
        "lhs_total_cc": lhs_total,
        "rhs_total_cc": rhs_total,
        "aggregate_delta_cc": lhs_total - rhs_total,
        "aggregate_delta_pct": (
            (lhs_total / rhs_total - 1.0) * 100.0 if rhs_total else 0.0
        ),
    }


def _summary(rows: list[dict]) -> dict:
    summary = {
        "cases": len(rows),
        "added_candidate_selected_cases": sum(
            int(row["added_candidate_count"]) > 0 for row in rows
        ),
        "added_candidate_count": sum(
            int(row["added_candidate_count"]) for row in rows
        ),
        "added_tokens": dict(
            sorted(
                Counter(
                    token
                    for row in rows
                    for token in row["added_candidate_tokens"]
                ).items()
            )
        ),
    }
    for field, policy in SCORE_ARMS.items():
        summary[f"{policy}_vs_adaptive"] = _comparison(
            rows, field, "adaptive_baseline_cc"
        )
    return summary


def _configuration(inputs: tuple[Path, ...]) -> dict:
    return {
        "baseline": "rtl_adaptive_single_first_s4pf",
        "candidate_policy": fixed.TOP6_BOTTOM2_CANDIDATE_POLICY,
        "window": {"head": 6, "bottom": 2},
        "delta_only": [
            "SYNC T0+B0 pair",
            "ONE_IDLE B0/B1 C/C at existing release points",
        ],
        "unchanged": [
            "timing model",
            "shape and S2PF datapaths",
            "S4PF single-first policy",
            "best reducer and commit",
        ],
        "score_ablation": dict(SCORE_ARMS),
        "protected_contract": (
            "old legacy winner unless an added candidate passes the fixed "
            "continuation-margin and optional slack/mode guard"
        ),
        "inputs": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in inputs
        ],
        "sources": {
            name: {
                "path": str((ROOT / name).resolve()),
                "sha256": _sha256(ROOT / name),
            }
            for name in (
                "scheduler_hw_fixed_policy.py",
                "scheduler_rtl_adaptive_prefetch_policy.py",
                Path(__file__).name,
            )
        },
    }


def _payload(
    rows_by_key: dict[str, dict],
    *,
    configuration: dict,
    complete: bool,
    runtime_s: float,
) -> dict:
    rows = list(rows_by_key.values())
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets["overall"].append(row)
        buckets[f"E{row['e_total']}"] .append(row)
        buckets[f"split:{row['dataset_split']}"] .append(row)
        buckets[
            "cache:present" if row["initial_cache"] else "cache:none"
        ].append(row)
        buckets[
            "workload:strict_olmoe"
            if row["strict_olmoe_style"]
            else "workload:other"
        ].append(row)
    return {
        "schema": SCHEMA,
        "complete": bool(complete),
        "completed_cases": len(rows),
        "expected_cases": EXPECTED_CASES,
        "runtime_s": float(runtime_s),
        "configuration": configuration,
        "summary": {
            key: _summary(values) for key, values in sorted(buckets.items())
        },
        "rows": dict(sorted(rows_by_key.items())),
    }


def _worker(job: dict) -> dict:
    dist = {int(eid): int(ntok) for eid, ntok in job["dist"].items()}
    c2, c3 = int(job["c2"]), int(job["c3"])
    baseline = adaptive_prefetch_schedule_result(dist, c2, c3)
    results = {
        field: adaptive_prefetch_schedule_result(
            dist,
            c2,
            c3,
            candidate_policy=fixed.TOP6_BOTTOM2_CANDIDATE_POLICY,
            score_policy=policy,
        )
        for field, policy in SCORE_ARMS.items()
    }
    added = []
    for mode, candidate_id in results["protected_headcritical_slack_1_cc"].winner_trace:
        if mode == 1 and candidate_id == 5:
            added.append("SYNC_T0_B0")
        elif mode == 2 and candidate_id >= 6:
            bottom_rank = (candidate_id - 6) // 3
            release_index = (candidate_id - 6) % 3
            added.append(f"ONE_IDLE_B{bottom_rank}_R{release_index}")
    return {
        "key": job["key"],
        "e_total": int(job["e_total"]),
        "case_id": int(job["case_id"]),
        "dataset_split": job["dataset_split"],
        "construction": job["construction"],
        "initial_cache": c2 >= 0 or c3 >= 0,
        "strict_olmoe_style": bool(job["strict_olmoe_style"]),
        "adaptive_baseline_cc": int(baseline.makespan_cc),
        **{field: int(result.makespan_cc) for field, result in results.items()},
        "added_candidate_count": len(added),
        "added_candidate_tokens": added,
    }


def _load_jobs(inputs: tuple[Path, ...]) -> list[dict]:
    jobs = []
    for input_path in inputs:
        for case in json.loads(input_path.read_text(encoding="utf-8"))["cases"]:
            if not case.get("analysis_eligible", False):
                continue
            loads = [
                int(ntok) for ntok in case["dist"].values() if int(ntok) > 0
            ]
            e_total = int(case["e_total"])
            full_mean = sum(loads) / e_total
            hotness = max(loads) / full_mean
            cold_including_zero = (
                e_total - len(loads) + sum(ntok <= 2 for ntok in loads)
            )
            tail_ok = max(sorted(loads, reverse=True)[6:], default=0) <= 8
            strict_olmoe_style = (
                e_total == 64
                and len(loads) >= 29
                and 6.0 <= hotness <= 14.0
                and cold_including_zero / e_total >= 0.60
                and tail_ok
            )
            jobs.append(
                {
                    "key": f"E{e_total}:{int(case['case_id'])}",
                    "e_total": e_total,
                    "case_id": int(case["case_id"]),
                    "dataset_split": case.get("dataset_split"),
                    "construction": case.get("construction"),
                    "strict_olmoe_style": strict_olmoe_style,
                    "dist": case["dist"],
                    "c2": int(case.get("c2", -1)),
                    "c3": int(case.get("c3", -1)),
                }
            )
    if len(jobs) != EXPECTED_CASES:
        raise RuntimeError(f"expected {EXPECTED_CASES} cases, got {len(jobs)}")
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    inputs = tuple(args.input) if args.input else DEFAULT_INPUTS
    configuration = _configuration(inputs)
    jobs = _load_jobs(inputs)
    if args.limit is not None:
        jobs = jobs[: args.limit]

    rows_by_key: dict[str, dict] = {}
    prior_runtime = 0.0
    if args.out.exists():
        prior = json.loads(args.out.read_text(encoding="utf-8"))
        if prior.get("schema") != SCHEMA:
            raise ValueError("checkpoint schema mismatch")
        if prior.get("configuration") != configuration:
            raise ValueError("checkpoint configuration or source hash changed")
        rows_by_key.update(prior.get("rows", {}))
        prior_runtime = float(prior.get("runtime_s", 0.0))

    pending = [job for job in jobs if job["key"] not in rows_by_key]
    started = time.perf_counter()

    def checkpoint(complete: bool) -> None:
        _atomic_write(
            args.out,
            _payload(
                rows_by_key,
                configuration=configuration,
                complete=complete,
                runtime_s=prior_runtime + time.perf_counter() - started,
            ),
        )

    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for index, row in enumerate(pool.map(_worker, pending, chunksize=8), 1):
                rows_by_key[row["key"]] = row
                if args.checkpoint_every > 0 and index % args.checkpoint_every == 0:
                    checkpoint(False)
                    print(
                        f"t6b2-ablation completed={len(rows_by_key)}/{len(jobs)} "
                        f"elapsed_s={time.perf_counter() - started:.1f}",
                        flush=True,
                    )

    complete = args.limit is None and len(rows_by_key) == EXPECTED_CASES
    checkpoint(complete)
    overall = _payload(
        rows_by_key,
        configuration=configuration,
        complete=complete,
        runtime_s=prior_runtime + time.perf_counter() - started,
    )["summary"]["overall"]
    print(json.dumps(overall, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
