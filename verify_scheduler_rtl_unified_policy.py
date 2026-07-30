#!/usr/bin/env python3
"""Checkpointed validation of the compiled unified scheduler policy."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import time

import scheduler_rtl_adaptive_prefetch_policy as adaptive
import scheduler_rtl_unified_policy as unified
from run_four_stage_reference import serialize_action


HERE = Path(__file__).resolve().parent
PROOF65 = HERE / "results/policy_search/olmoe_top2_projection_65_optimal_v1.json"
COVERAGE30K = tuple(
    HERE / f"scheduler_strategy_coverage_E{experts}.json"
    for experts in (8, 32, 64)
)
POSTFREEZE = tuple(
    Path("/tmp/scheduler_t6b2_postfreeze_v4")
    / f"scheduler_t6b2_postfreeze_E{experts}.json"
    for experts in (8, 32, 64)
)
DEFAULT_OUTPUTS = {
    "proof65": HERE / "results/policy_search/scheduler_rtl_unified_65_v4.json",
    "coverage30k": HERE / "results/policy_search/scheduler_rtl_unified_30k_v4.json",
    "postfreeze": HERE / "results/policy_search/scheduler_rtl_unified_postfreeze_v4.json",
}
EXPECTED_CASES = {"proof65": 65, "coverage30k": 29_928, "postfreeze": 11_928}
SCHEMA = "scheduler_rtl_unified_policy_validation_v3"


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


def _strict_olmoe_style(e_total: int, loads: list[int]) -> bool:
    if not loads or e_total != 64:
        return False
    full_mean = sum(loads) / e_total
    hotness = max(loads) / full_mean
    cold_including_zero = e_total - len(loads) + sum(load <= 2 for load in loads)
    return (
        len(loads) >= 29
        and 6.0 <= hotness <= 14.0
        and cold_including_zero / e_total >= 0.60
        and max(sorted(loads, reverse=True)[6:], default=0) <= 8
    )


def _proof_jobs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = []
    for case in payload["cases"]:
        distribution = {
            eid: int(ntok)
            for eid, ntok in enumerate(case["counts"])
            if int(ntok) > 0
        }
        target = Fraction(str(case["best_reference_ticks"])) * unified.TICK_CC
        if target.denominator != 1:
            raise ValueError(f"{case['name']}: non-integral target")
        jobs.append(
            {
                "key": str(case["name"]),
                "e_total": len(case["counts"]),
                "dataset_split": "proof65",
                "distribution": distribution,
                "c2": -1,
                "c3": -1,
                "compute_only_ideal_cc": None,
                "target_cc": int(target),
                "strict_olmoe_style": True,
            }
        )
    return jobs


def _dataset_jobs(paths: tuple[Path, ...]) -> list[dict]:
    jobs = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            if not case.get("analysis_eligible", False):
                continue
            distribution = {
                int(eid): int(ntok)
                for eid, ntok in case["dist"].items()
                if int(ntok) > 0
            }
            e_total = int(case["e_total"])
            jobs.append(
                {
                    "key": f"E{e_total}:{int(case['case_id'])}",
                    "e_total": e_total,
                    "dataset_split": case.get("dataset_split"),
                    "distribution": distribution,
                    "c2": int(case.get("c2", -1)),
                    "c3": int(case.get("c3", -1)),
                    "compute_only_ideal_cc": int(case["compute_only_ideal_cc"]),
                    "target_cc": None,
                    "strict_olmoe_style": _strict_olmoe_style(
                        e_total, list(distribution.values())
                    ),
                }
            )
    return jobs


def _worker(job: dict) -> dict:
    distribution = dict(job["distribution"])
    c2, c3 = int(job["c2"]), int(job["c3"])
    baseline = adaptive.adaptive_prefetch_schedule_result(distribution, c2, c3)
    result = unified.schedule(distribution, c2, c3)
    row = {
        "key": job["key"],
        "e_total": int(job["e_total"]),
        "dataset_split": job["dataset_split"],
        "initial_cache": c2 >= 0 or c3 >= 0,
        "strict_olmoe_style": bool(job["strict_olmoe_style"]),
        "compute_only_ideal_cc": job["compute_only_ideal_cc"],
        "target_cc": job["target_cc"],
        "adaptive_cc": int(baseline.makespan_cc),
        "unified_cc": int(result.makespan_cc),
        "candidate_count_max": int(result.candidate_count_max),
        "candidate_count_mean": float(result.candidate_count_mean),
        "candidate_count_max_by_mode": {
            mode: max(
                step.candidate_count
                for step in result.steps
                if step.mode == mode
            )
            for mode in sorted({step.mode for step in result.steps})
        },
        "selected_slot_max": max(
            (step.candidate_slot for step in result.steps), default=-1
        ),
    }
    if job["target_cc"] is not None:
        row["trace"] = [
            {
                "mode": step.mode,
                "candidate_slot": step.candidate_slot,
                "candidate_count": step.candidate_count,
                "score": list(step.score),
                "action": serialize_action(step.action),
            }
            for step in result.steps
        ]
    return row


def _comparison(rows: list[dict]) -> dict:
    unified_total = sum(row["unified_cc"] for row in rows)
    adaptive_total = sum(row["adaptive_cc"] for row in rows)
    summary = {
        "cases": len(rows),
        "better": sum(row["unified_cc"] < row["adaptive_cc"] for row in rows),
        "equal": sum(row["unified_cc"] == row["adaptive_cc"] for row in rows),
        "worse": sum(row["unified_cc"] > row["adaptive_cc"] for row in rows),
        "aggregate_delta_cc": unified_total - adaptive_total,
        "aggregate_delta_ticks": Fraction(
            unified_total - adaptive_total, unified.TICK_CC
        ).__str__(),
        "aggregate_delta_pct": (
            (unified_total / adaptive_total - 1.0) * 100.0
            if adaptive_total
            else 0.0
        ),
        "candidate_count_max": max(
            (row["candidate_count_max"] for row in rows), default=0
        ),
        "candidate_count_max_by_mode": {
            mode: max(
                row["candidate_count_max_by_mode"].get(mode, 0)
                for row in rows
            )
            for mode in ("SYNC", "ONE_IDLE", "TERMINAL")
        },
        "selected_slot_max": max(
            (row["selected_slot_max"] for row in rows), default=-1
        ),
    }
    targets = [row for row in rows if row["target_cc"] is not None]
    if targets:
        summary.update(
            optimal_cases=sum(
                row["unified_cc"] == row["target_cc"] for row in targets
            ),
            target_gap_ticks=Fraction(
                sum(row["unified_cc"] - row["target_cc"] for row in targets),
                unified.TICK_CC,
            ).__str__(),
        )
    return summary


def _summaries(rows: list[dict]) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets["overall"].append(row)
        buckets[f"E{row['e_total']}"].append(row)
        buckets[f"split:{row['dataset_split']}"].append(row)
        buckets[
            "cache:present" if row["initial_cache"] else "cache:none"
        ].append(row)
        buckets[
            "workload:strict_olmoe"
            if row["strict_olmoe_style"]
            else "workload:other"
        ].append(row)
    return {key: _comparison(value) for key, value in sorted(buckets.items())}


def _payload(
    suite: str,
    rows_by_key: dict[str, dict],
    configuration: dict,
    runtime_s: float,
    expected: int,
    complete: bool,
) -> dict:
    rows = list(rows_by_key.values())
    return {
        "schema": SCHEMA,
        "suite": suite,
        "complete": bool(complete),
        "completed_cases": len(rows),
        "expected_cases": expected,
        "runtime_s": float(runtime_s),
        "configuration": configuration,
        "summary": _summaries(rows),
        "rows": dict(sorted(rows_by_key.items())),
    }


def _audit_proof_traces(rows_by_key: dict[str, dict], jobs: list[dict]) -> None:
    """Regenerate every recorded proof slot and selected transition."""
    jobs_by_key = {job["key"]: job for job in jobs}
    for key, row in rows_by_key.items():
        job = jobs_by_key[key]
        state = unified._initial_state(
            dict(job["distribution"]), int(job["c2"]), int(job["c3"])
        )
        for round_index, recorded in enumerate(row["trace"]):
            slots = unified.generate_candidate_slots(state)
            action, child, score, count, selected_slot = unified._choose_one_round(
                state
            )
            if count < len(slots) or count != int(recorded["candidate_count"]):
                raise AssertionError(f"{key} round {round_index}: count mismatch")
            if selected_slot != int(recorded["candidate_slot"]):
                raise AssertionError(f"{key} round {round_index}: slot mismatch")
            if slots[selected_slot].action != action:
                raise AssertionError(
                    f"{key} round {round_index}: slot/action mismatch"
                )
            if serialize_action(action) != recorded["action"]:
                raise AssertionError(
                    f"{key} round {round_index}: serialized action mismatch"
                )
            if list(score) != recorded["score"]:
                raise AssertionError(f"{key} round {round_index}: score mismatch")
            state = child
        if state.remaining:
            raise AssertionError(f"{key}: trace is not terminal")
        if int(state.g_score) != int(row["unified_cc"]):
            raise AssertionError(f"{key}: terminal makespan mismatch")
        if int(row["unified_cc"]) != int(row["target_cc"]):
            raise AssertionError(f"{key}: certified target mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("proof65", "coverage30k", "postfreeze"),
        required=True,
    )
    parser.add_argument("--input", action="append", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-per-e", type=int)
    args = parser.parse_args()

    if args.input:
        inputs = tuple(path.resolve() for path in args.input)
    elif args.suite == "proof65":
        inputs = (PROOF65.resolve(),)
    elif args.suite == "coverage30k":
        inputs = tuple(path.resolve() for path in COVERAGE30K)
    else:
        inputs = tuple(path.resolve() for path in POSTFREEZE)
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing validation inputs: {missing}")

    jobs = (
        _proof_jobs(inputs[0])
        if args.suite == "proof65"
        else _dataset_jobs(inputs)
    )
    full_expected = EXPECTED_CASES[args.suite]
    if not args.input and len(jobs) != full_expected:
        raise RuntimeError(
            f"{args.suite}: expected {full_expected} jobs, got {len(jobs)}"
        )
    if args.sample_per_e is not None:
        counts: dict[int, int] = defaultdict(int)
        sampled = []
        for job in jobs:
            e_total = int(job["e_total"])
            if counts[e_total] < args.sample_per_e:
                sampled.append(job)
                counts[e_total] += 1
        jobs = sampled
    if args.limit is not None:
        jobs = jobs[: args.limit]
    expected = len(jobs)

    source_paths = (
        HERE / "scheduler_rtl_unified_policy.py",
        HERE / "evaluate_olmoe_fixed_token_banks.py",
        HERE / "four_stage_scheduler.py",
        HERE / "scheduler_rtl_adaptive_prefetch_policy.py",
        Path(__file__).resolve(),
    )
    configuration = {
        "policy_id": unified.POLICY_ID,
        "window": list(unified.WINDOW),
        "candidate_budget": unified.MAX_CONCRETE_CANDIDATES,
        "base_token_profiles": len(unified.COMPILED_TOKENS),
        "recovery_token_profiles": len(unified.RECOVERY_TOKENS),
        "recovery_margin_cc": unified.RECOVERY_MARGIN_CC,
        "scorer": unified.SCORER,
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in inputs
        ],
        "sources": {
            path.name: {"path": str(path), "sha256": _sha256(path)}
            for path in source_paths
        },
        "sample_per_e": args.sample_per_e,
        "limit": args.limit,
    }
    output = (args.out or DEFAULT_OUTPUTS[args.suite]).resolve()
    rows_by_key: dict[str, dict] = {}
    prior_runtime = 0.0
    if output.exists():
        prior = json.loads(output.read_text(encoding="utf-8"))
        if prior.get("schema") != SCHEMA or prior.get("suite") != args.suite:
            raise ValueError("checkpoint schema or suite mismatch")
        if prior.get("configuration") != configuration:
            raise ValueError("checkpoint inputs, source hashes or policy changed")
        rows_by_key.update(prior.get("rows", {}))
        prior_runtime = float(prior.get("runtime_s", 0.0))

    pending = [job for job in jobs if job["key"] not in rows_by_key]
    started = time.perf_counter()

    def checkpoint(complete: bool) -> None:
        _atomic_write(
            output,
            _payload(
                args.suite,
                rows_by_key,
                configuration,
                prior_runtime + time.perf_counter() - started,
                expected,
                complete,
            ),
        )

    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for index, row in enumerate(pool.map(_worker, pending, chunksize=4), 1):
                rows_by_key[row["key"]] = row
                if args.checkpoint_every > 0 and index % args.checkpoint_every == 0:
                    checkpoint(False)
                    print(
                        f"{args.suite} completed={len(rows_by_key)}/{expected} "
                        f"elapsed_s={time.perf_counter() - started:.1f}",
                        flush=True,
                    )

    complete = len(rows_by_key) == expected
    if complete and args.suite == "proof65":
        _audit_proof_traces(rows_by_key, jobs)
    checkpoint(complete)
    summary = _summaries(list(rows_by_key.values()))["overall"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
