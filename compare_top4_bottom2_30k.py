#!/usr/bin/env python3
"""Compare top4+bottom2 policy against the frozen adaptive HW-v2 baseline.

The 29,928-case input is feature-stratified synthetic coverage data, not a
measured router distribution.  The script reuses the previously frozen
single-first baseline values and recomputes only the new policy.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import scheduler_top4_bottom2_policy as policy


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUTS = tuple(
    ROOT / f"scheduler_strategy_coverage_E{experts}.json"
    for experts in (8, 32, 64)
)
DEFAULT_BASELINE = (
    ROOT
    / "results"
    / "policy_search"
    / "scheduler_rtl_adaptive_prefetch_vs_hw_v2_30k.json"
)
DEFAULT_OUT = (
    ROOT / "results" / "policy_search" / "top4_bottom2_vs_adaptive_30k.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def comparison(rows: list[dict], rhs: str) -> dict:
    lhs_total = sum(row["top4_bottom2_cc"] for row in rows)
    rhs_total = sum(row[rhs] for row in rows)
    return {
        "lhs": "top4_bottom2_cc",
        "rhs": rhs,
        "cases": len(rows),
        "better": sum(row["top4_bottom2_cc"] < row[rhs] for row in rows),
        "equal": sum(row["top4_bottom2_cc"] == row[rhs] for row in rows),
        "worse": sum(row["top4_bottom2_cc"] > row[rhs] for row in rows),
        "lhs_total_cc": lhs_total,
        "rhs_total_cc": rhs_total,
        "aggregate_delta_cc": lhs_total - rhs_total,
        "aggregate_delta_pct": (lhs_total / rhs_total - 1.0) * 100.0,
        "max_improvement_ticks": max(
            (row[rhs] - row["top4_bottom2_cc"]) // policy.TICK_CC
            for row in rows
        ),
        "max_regression_ticks": max(
            (row["top4_bottom2_cc"] - row[rhs]) // policy.TICK_CC
            for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=2_000)
    args = parser.parse_args()
    inputs = tuple(args.input) if args.input else DEFAULT_INPUTS

    baseline_payload = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline_rows = baseline_payload["rows"]
    rows: list[dict] = []
    decisions: Counter[str] = Counter()
    started = time.perf_counter()
    stop = False

    for input_path in inputs:
        cases = json.loads(input_path.read_text(encoding="utf-8"))["cases"]
        for case in cases:
            if not case.get("analysis_eligible", False):
                continue
            key = f"E{int(case['e_total'])}:{int(case['case_id'])}"
            baseline = baseline_rows.get(key)
            if baseline is None:
                raise RuntimeError(f"missing frozen baseline row {key}")
            distribution = {
                int(eid): int(ntok) for eid, ntok in case["dist"].items()
            }
            result = policy.schedule_result(
                distribution,
                int(case.get("c2", -1)),
                int(case.get("c3", -1)),
            )
            decisions.update(step.decision for step in result.steps)
            rows.append(
                {
                    "key": key,
                    "e_total": int(case["e_total"]),
                    "case_id": int(case["case_id"]),
                    "dataset_split": case.get("dataset_split"),
                    "top4_bottom2_cc": int(result.makespan_cc),
                    "adaptive_single_first_cc": int(baseline["single_first_cc"]),
                    "algorithmic_hw_v2_cc": int(baseline["original_hw_v2_cc"]),
                }
            )
            if args.progress_every > 0 and len(rows) % args.progress_every == 0:
                print(
                    f"top4+bottom2 completed={len(rows)} "
                    f"elapsed_s={time.perf_counter() - started:.1f}",
                    flush=True,
                )
            if args.limit is not None and len(rows) >= args.limit:
                stop = True
                break
        if stop:
            break

    if args.limit is None and len(rows) != 29_928:
        raise RuntimeError(f"expected 29928 eligible cases, got {len(rows)}")

    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets["overall"].append(row)
        buckets[f"E{row['e_total']}"] .append(row)
        buckets[f"split:{row['dataset_split']}"] .append(row)
    summary = {
        key: {
            "vs_adaptive_single_first": comparison(
                values, "adaptive_single_first_cc"
            ),
            "vs_algorithmic_hw_v2": comparison(values, "algorithmic_hw_v2_cc"),
        }
        for key, values in sorted(buckets.items())
    }

    payload = {
        "schema": "top4_bottom2_vs_adaptive_30k_v1",
        "dataset_warning": (
            "Feature-stratified synthetic coverage; not a measured router "
            "probability distribution."
        ),
        "configuration": {
            "inputs": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in inputs
            ],
            "baseline": {
                "path": str(args.baseline.resolve()),
                "sha256": sha256(args.baseline),
            },
            "source_sha256": {
                "policy": sha256(ROOT / "scheduler_top4_bottom2_policy.py"),
                "driver": sha256(Path(__file__).resolve()),
            },
            "limit": args.limit,
        },
        "runtime_s": time.perf_counter() - started,
        "decision_counts": dict(sorted(decisions.items())),
        "summary": summary,
        "rows": {row["key"]: row for row in rows},
    }
    atomic_write(args.out, payload)
    print(json.dumps(summary["overall"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

