#!/usr/bin/env python3
"""Checkpointed same-input validation of the bounded distilled scheduler."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import time

import scheduler_rtl_adaptive_prefetch_policy as adaptive
import scheduler_rtl_distilled_policy as distilled
import scheduler_rtl_unified_policy as frozen_v4
from run_four_stage_reference import serialize_action
import verify_scheduler_rtl_unified_policy as datasets


HERE = Path(__file__).resolve().parent
PROOF65 = HERE / "results/policy_search/olmoe_top2_projection_65_optimal_v1.json"
COVERAGE30K = tuple(
    HERE / f"scheduler_strategy_coverage_E{experts}.json"
    for experts in (8, 32, 64)
)
DEFAULT_OUTPUTS = {
    "proof65": (
        HERE
        / "results/policy_search/bounded_top5_bottom1_certificate_validation.json"
    ),
    "coverage30k": (
        HERE
        / "results/policy_search/bounded_top5_bottom1_random_validation.json"
    ),
}
EXPECTED_CASES = {"proof65": 65, "coverage30k": 29_928}
SCHEMA = "bounded_top5_bottom1_validation"


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


def _worker(job: dict) -> dict:
    distribution = dict(job["distribution"])
    c2, c3 = int(job["c2"]), int(job["c3"])
    adaptive_result = adaptive.adaptive_prefetch_schedule_result(
        distribution, c2, c3
    )
    v4_result = frozen_v4.schedule(distribution, c2, c3)
    result = distilled.schedule(distribution, c2, c3)
    selected_profile_uses = Counter(
        step.selected_profile_slot for step in result.steps
    )
    local_profile_winner_uses = Counter(
        profile_slot
        for step in result.steps
        for profile_slot in step.local_profile_slots
    )
    row = {
        "key": job["key"],
        "e_total": int(job["e_total"]),
        "dataset_split": job["dataset_split"],
        "initial_cache": c2 >= 0 or c3 >= 0,
        "strict_olmoe_style": bool(job["strict_olmoe_style"]),
        "compute_only_ideal_cc": job["compute_only_ideal_cc"],
        "target_cc": job["target_cc"],
        "adaptive_cc": int(adaptive_result.makespan_cc),
        "v4_cc": int(v4_result.makespan_cc),
        "distilled_cc": int(result.makespan_cc),
        "physical_candidate_count_max": int(
            result.physical_candidate_count_max
        ),
        "physical_candidate_count_mean": float(
            result.physical_candidate_count_mean
        ),
        "logical_candidate_count_max": int(result.logical_candidate_count_max),
        "logical_candidate_count_mean": float(
            result.logical_candidate_count_mean
        ),
        "physical_candidate_count_max_by_mode": {
            mode: max(
                step.physical_candidate_count
                for step in result.steps
                if step.mode == mode
            )
            for mode in sorted({step.mode for step in result.steps})
        },
        "logical_candidate_count_max_by_mode": {
            mode: max(
                step.logical_candidate_count
                for step in result.steps
                if step.mode == mode
            )
            for mode in sorted({step.mode for step in result.steps})
        },
        "selected_slot_max": max(
            (step.candidate_slot for step in result.steps), default=-1
        ),
        "selected_profile_uses": {
            str(slot): count
            for slot, count in sorted(selected_profile_uses.items())
        },
        "local_profile_winner_uses": {
            str(slot): count
            for slot, count in sorted(local_profile_winner_uses.items())
        },
    }
    if job["target_cc"] is not None:
        row["trace"] = [
            {
                "mode": step.mode,
                "candidate_slot": step.candidate_slot,
                "physical_candidate_count": step.physical_candidate_count,
                "logical_candidate_count": step.logical_candidate_count,
                "score": list(step.score),
                "selected_profile_slot": step.selected_profile_slot,
                "local_profile_slots": list(step.local_profile_slots),
                "action": serialize_action(step.action),
            }
            for step in result.steps
        ]
    return row


def _comparison(
    rows: list[dict],
    left: str,
    right: str,
) -> dict:
    left_total = sum(int(row[left]) for row in rows)
    right_total = sum(int(row[right]) for row in rows)
    return {
        "cases": len(rows),
        "better": sum(int(row[left]) < int(row[right]) for row in rows),
        "equal": sum(int(row[left]) == int(row[right]) for row in rows),
        "worse": sum(int(row[left]) > int(row[right]) for row in rows),
        "aggregate_delta_cc": left_total - right_total,
        "aggregate_delta_ticks": str(
            Fraction(left_total - right_total, distilled.TICK_CC)
        ),
        "aggregate_delta_pct": (
            (left_total / right_total - 1.0) * 100.0 if right_total else 0.0
        ),
    }


def _bucket_summary(rows: list[dict]) -> dict:
    summary = {
        "distilled_vs_v4": _comparison(rows, "distilled_cc", "v4_cc"),
        "distilled_vs_adaptive": _comparison(
            rows, "distilled_cc", "adaptive_cc"
        ),
        "v4_vs_adaptive": _comparison(rows, "v4_cc", "adaptive_cc"),
        "complexity": {
            "physical_candidate_count_max": max(
                (row["physical_candidate_count_max"] for row in rows),
                default=0,
            ),
            "logical_candidate_count_max": max(
                (row["logical_candidate_count_max"] for row in rows),
                default=0,
            ),
            "selected_slot_max": max(
                (row["selected_slot_max"] for row in rows), default=-1
            ),
            "selected_profile_slots": sorted(
                {
                    int(slot)
                    for row in rows
                    for slot in row["selected_profile_uses"]
                }
            ),
            "local_profile_winner_slots": sorted(
                {
                    int(slot)
                    for row in rows
                    for slot in row["local_profile_winner_uses"]
                }
            ),
            "physical_candidate_count_max_by_mode": {
                mode: max(
                    row["physical_candidate_count_max_by_mode"].get(mode, 0)
                    for row in rows
                )
                for mode in ("SYNC", "ONE_IDLE", "TERMINAL")
            },
            "logical_candidate_count_max_by_mode": {
                mode: max(
                    row["logical_candidate_count_max_by_mode"].get(mode, 0)
                    for row in rows
                )
                for mode in ("SYNC", "ONE_IDLE", "TERMINAL")
            },
        },
    }
    targets = [row for row in rows if row["target_cc"] is not None]
    if targets:
        summary["certificate"] = {
            "cases": len(targets),
            "optimal_cases": sum(
                row["distilled_cc"] == row["target_cc"] for row in targets
            ),
            "target_gap_ticks": str(
                Fraction(
                    sum(
                        row["distilled_cc"] - row["target_cc"]
                        for row in targets
                    ),
                    distilled.TICK_CC,
                )
            ),
        }
    return summary


def _summaries(rows: list[dict]) -> dict[str, dict]:
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
    return {key: _bucket_summary(value) for key, value in sorted(buckets.items())}


def _audit_proof_traces(rows_by_key: dict[str, dict], jobs: list[dict]) -> None:
    jobs_by_key = {job["key"]: job for job in jobs}
    for key, row in rows_by_key.items():
        job = jobs_by_key[key]
        state = distilled._initial_state(
            dict(job["distribution"]), int(job["c2"]), int(job["c3"])
        )
        for round_index, recorded in enumerate(row["trace"]):
            candidate_set = distilled._materialize_candidate_set(state)
            action, child, score, regenerated, selected_slot = (
                distilled._choose_one_round(state)
            )
            if candidate_set != regenerated:
                raise AssertionError(
                    f"{key} round {round_index}: materialization mismatch"
                )
            if selected_slot != int(recorded["candidate_slot"]):
                raise AssertionError(f"{key} round {round_index}: slot mismatch")
            if candidate_set.slots[selected_slot].action != action:
                raise AssertionError(
                    f"{key} round {round_index}: slot/action mismatch"
                )
            if candidate_set.physical_count != int(
                recorded["physical_candidate_count"]
            ) or len(candidate_set.slots) != int(
                recorded["logical_candidate_count"]
            ):
                raise AssertionError(
                    f"{key} round {round_index}: candidate count mismatch"
                )
            if serialize_action(action) != recorded["action"]:
                raise AssertionError(
                    f"{key} round {round_index}: serialized action mismatch"
                )
            if list(score) != recorded["score"]:
                raise AssertionError(f"{key} round {round_index}: score mismatch")
            selected = candidate_set.slots[selected_slot]
            if selected.physical_profile_slot != int(
                recorded["selected_profile_slot"]
            ):
                raise AssertionError(
                    f"{key} round {round_index}: selected profile mismatch"
                )
            if [slot.physical_profile_slot for slot in candidate_set.slots] != [
                int(value) for value in recorded["local_profile_slots"]
            ]:
                raise AssertionError(
                    f"{key} round {round_index}: local profile list mismatch"
                )
            state = child
        if state.remaining:
            raise AssertionError(f"{key}: trace is not terminal")
        if int(state.g_score) != int(row["distilled_cc"]):
            raise AssertionError(f"{key}: terminal makespan mismatch")
        if int(row["distilled_cc"]) != int(row["target_cc"]):
            raise AssertionError(f"{key}: certified target mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("proof65", "coverage30k"), required=True
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--sample-per-e", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    inputs = (
        (PROOF65.resolve(),)
        if args.suite == "proof65"
        else tuple(path.resolve() for path in COVERAGE30K)
    )
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing validation inputs: {missing}")
    jobs = (
        datasets._proof_jobs(inputs[0])
        if args.suite == "proof65"
        else datasets._dataset_jobs(inputs)
    )
    full_expected = EXPECTED_CASES[args.suite]
    if len(jobs) != full_expected:
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
        HERE / "scheduler_rtl_distilled_policy.py",
        HERE / "scheduler_rtl_distilled_profiles.py",
        HERE / "scheduler_rtl_unified_policy.py",
        HERE / "evaluate_olmoe_fixed_token_banks.py",
        HERE / "four_stage_scheduler.py",
        HERE / "scheduler_rtl_adaptive_prefetch_policy.py",
        Path(__file__).resolve(),
    )
    configuration = {
        "policy_id": distilled.POLICY_ID,
        "window": list(distilled.WINDOW),
        "physical_candidate_budget": distilled.MAX_PHYSICAL_CANDIDATES,
        "compiled_profiles": len(distilled.COMPILED_PROFILES),
        "continuation_scorer": distilled.CONTINUATION_SCORER,
        "start_policy": "earliest_finish",
        "local_reducer": [
            "latest_task_end",
            "sum_task_end",
            "latest_start",
            "prefer_s2_prefetch",
            "fixed_profile_priority",
        ],
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
        rows = list(rows_by_key.values())
        _atomic_write(
            output,
            {
                "schema": SCHEMA,
                "suite": args.suite,
                "complete": bool(complete),
                "completed_cases": len(rows),
                "expected_cases": expected,
                "runtime_s": prior_runtime + time.perf_counter() - started,
                "configuration": configuration,
                "summary": _summaries(rows),
                "rows": dict(sorted(rows_by_key.items())),
            },
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
    print(json.dumps(_summaries(list(rows_by_key.values()))["overall"], indent=2))
    print(f"wrote {output}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
