#!/usr/bin/env python3
"""Paired 29,928-case audit of adaptive and the certified T6+B2 union.

The driver is checkpointed and resumable.  Every row uses the same
distribution and initial cache state for:

* frozen adaptive baseline (single-first S4PF),
* protected additive T6+B2 bank or certified fixed14/head5-hist4 bank.

Cases outside the frozen 65-case implementation envelope reproduce the old
adaptive winner before admitting a protected bottom candidate.
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

from scheduler_rtl_adaptive_olmoe_policy import (
    POLICY_ID,
    AdaptiveOlmoeScheduler,
)
from scheduler_rtl_adaptive_prefetch_policy import adaptive_prefetch_schedule


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUTS = tuple(
    ROOT / f"scheduler_strategy_coverage_E{experts}.json"
    for experts in (8, 32, 64)
)
DEFAULT_OUT = (
    ROOT
    / "results"
    / "policy_search"
    / "scheduler_adaptive_t6b2_joint_union_30k_v4.json"
)
EXPECTED_CASES = 29_928
FINAL_RULE_EVIDENCE = (
    ROOT
    / "results"
    / "policy_search"
    / "scheduler_adaptive_t6b2_tail_b0_sweep_30k_v10.json"
)

_WORKER_SCHEDULER: AdaptiveOlmoeScheduler | None = None


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


def _summarize_rows(rows: list[dict]) -> dict:
    return {
        "cases": len(rows),
        "union_vs_adaptive": _comparison(
            rows, "union_cc", "adaptive_baseline_cc"
        ),
        "compute_only_ideal_gap": _comparison(
            rows, "union_cc", "compute_only_ideal_cc"
        ),
        "routes": dict(sorted(Counter(row["union_route"] for row in rows).items())),
        "fallback_reasons": dict(
            sorted(Counter(row["union_fallback_reason"] for row in rows).items())
        ),
        "candidate_count_max": max(
            (int(row["union_candidate_count_max"]) for row in rows),
            default=0,
        ),
    }


def _payload(
    rows_by_key: dict[str, dict],
    *,
    inputs: tuple[Path, ...],
    complete: bool,
    started: float,
    workers: int,
    expected_cases: int,
) -> dict:
    rows = list(rows_by_key.values())
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets["overall"].append(row)
        buckets[f"E{row['e_total']}"] .append(row)
        buckets[f"split:{row['dataset_split']}"] .append(row)
        buckets[f"route:{row['union_route']}"] .append(row)
        buckets[
            "workload:olmoe_style"
            if row["olmoe_style_target"]
            else "workload:other"
        ].append(row)
        buckets[
            "regime:certified"
            if row["certified_regime"]
            else "regime:protected"
        ].append(row)
    return {
        "schema": "scheduler_adaptive_t6b2_joint_union_30k_v4",
        "complete": bool(complete),
        "completed_cases": len(rows),
        "expected_cases": expected_cases,
        "runtime_s": time.perf_counter() - started,
        "configuration": {
            "policy_id": POLICY_ID,
            "baseline": "rtl_adaptive_single_first_s4pf",
            "upgrade_window": [6, 2],
            "generic_route": {
                "gate": "all_non_certified_inputs",
                "candidate_bank": (
                    "adaptive_plus_one_mode_specific_candidate:"
                    "SYNC_T0_B0_or_ONE_IDLE_B0_release0"
                ),
                "scorer": "protected_tail_b0_slack_1",
                "one_idle_valid": "B0_at_first_release_only",
                "acceptance": (
                    "fits_busy_slack_and_head_finish_ge_continuation_and_"
                    "continuation_advantage_ge_1_tick"
                ),
                "s4pf": "unchanged_adaptive_single_first",
                "selection_evidence": {
                    "path": str(FINAL_RULE_EVIDENCE.resolve()),
                    "sha256": _sha256(FINAL_RULE_EVIDENCE),
                },
            },
            "certified_route": {
                "candidate_rom_entries": 14,
                "scorer": "head5_hist4_regime_pairwise_v1",
                "s4pf": "disabled_by_token_profile",
            },
            "certified_envelope": {
                "e_total": 64,
                "assignment_sum": 140,
                "initial_cache": "none",
                "active_experts": [29, 57],
                "experts_le2_including_zero": [40, 49],
                "top1_ntok_max": 34,
                "tail_below_top5_ntok_max": 7
            },
            "olmoe_style_bucket": {
                "e_total": 64,
                "active_experts_min": 29,
                "top1_to_full_expert_mean_min": 6.0,
                "experts_le2_including_zero_fraction_min": 0.60,
                "tail_below_top5_ntok_max": 8,
                "purpose": "reporting_only_not_runtime_selection",
            },
            "workers": workers,
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
                    "scheduler_rtl_adaptive_olmoe_policy.py",
                    "scheduler_olmoe_bounded_policy.py",
                    "evaluate_olmoe_fixed_token_banks.py",
                    Path(__file__).name,
                )
            },
        },
        "summary": {
            key: _summarize_rows(values)
            for key, values in sorted(buckets.items())
        },
        "rows": dict(sorted(rows_by_key.items())),
    }


def _worker(job: dict) -> dict:
    global _WORKER_SCHEDULER
    if _WORKER_SCHEDULER is None:
        _WORKER_SCHEDULER = AdaptiveOlmoeScheduler()
    dist = {int(eid): int(ntok) for eid, ntok in job["dist"].items()}
    c2, c3 = int(job["c2"]), int(job["c3"])
    baseline = int(adaptive_prefetch_schedule(dist, c2, c3))
    union = _WORKER_SCHEDULER.schedule(
        dist,
        c2,
        c3,
        fallback=True,
    )
    return {
        "key": job["key"],
        "e_total": int(job["e_total"]),
        "case_id": int(job["case_id"]),
        "dataset_split": job["dataset_split"],
        "compute_only_ideal_cc": int(job["compute_only_ideal_cc"]),
        "olmoe_style_target": bool(job["olmoe_style_target"]),
        "adaptive_baseline_cc": baseline,
        "union_cc": int(union.makespan_cc),
        "certified_regime": bool(union.contract_eligible),
        "union_route": union.route,
        "union_fallback_reason": union.fallback_reason,
        "union_candidate_count_max": int(union.candidate_count_max),
    }


def _load_jobs(inputs: tuple[Path, ...], expected_cases: int) -> list[dict]:
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
            cold_including_zero = (
                e_total - len(loads) + sum(ntok <= 2 for ntok in loads)
            )
            tail_hist4_ok = max(sorted(loads, reverse=True)[5:], default=0) <= 8
            olmoe_style_target = (
                e_total == 64
                and tail_hist4_ok
                and len(loads) >= 29
                and max(loads) / full_mean >= 6.0
                and cold_including_zero / e_total >= 0.60
            )
            jobs.append(
                {
                    "key": f"E{e_total}:{int(case['case_id'])}",
                    "e_total": e_total,
                    "case_id": int(case["case_id"]),
                    "dataset_split": case.get("dataset_split"),
                    "compute_only_ideal_cc": int(case["compute_only_ideal_cc"]),
                    "olmoe_style_target": olmoe_style_target,
                    "dist": case["dist"],
                    "c2": int(case.get("c2", -1)),
                    "c3": int(case.get("c3", -1)),
                }
            )
    if len(jobs) != expected_cases:
        raise RuntimeError(f"expected {expected_cases} eligible jobs, got {len(jobs)}")
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-cases", type=int, default=EXPECTED_CASES)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or args.chunksize <= 0:
        raise SystemExit("--workers and --chunksize must be positive")
    inputs = tuple(path.resolve() for path in (args.input or DEFAULT_INPUTS))
    jobs = _load_jobs(inputs, args.expected_cases)
    if args.limit is not None:
        jobs = jobs[: args.limit]

    rows_by_key: dict[str, dict] = {}
    if args.resume and args.out.exists():
        saved = json.loads(args.out.read_text(encoding="utf-8"))
        rows_by_key.update(saved.get("rows", {}))
    pending = [job for job in jobs if job["key"] not in rows_by_key]
    started = time.perf_counter()
    print(
        f"adaptive-olmoe total={len(jobs)} resumed={len(rows_by_key)} "
        f"pending={len(pending)} workers={args.workers}",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for row in executor.map(_worker, pending, chunksize=args.chunksize):
            rows_by_key[row["key"]] = row
            completed = len(rows_by_key)
            if args.progress_every > 0 and completed % args.progress_every == 0:
                print(
                    f"adaptive-olmoe completed={completed}/{len(jobs)} "
                    f"elapsed_s={time.perf_counter() - started:.1f}",
                    flush=True,
                )
            if args.checkpoint_every > 0 and completed % args.checkpoint_every == 0:
                _atomic_write(
                    args.out,
                    _payload(
                        rows_by_key,
                        inputs=inputs,
                        complete=False,
                        started=started,
                        workers=args.workers,
                        expected_cases=args.expected_cases,
                    ),
                )

    complete = (
        args.limit is None
        and len(rows_by_key) == len(jobs)
        and len(rows_by_key) == args.expected_cases
    )
    payload = _payload(
        rows_by_key,
        inputs=inputs,
        complete=complete,
        started=started,
        workers=args.workers,
        expected_cases=args.expected_cases,
    )
    _atomic_write(args.out, payload)
    print(json.dumps(payload["summary"]["overall"], indent=2), flush=True)
    print(f"wrote {args.out.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
