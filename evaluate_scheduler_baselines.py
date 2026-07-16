#!/usr/bin/env python3
"""Evaluate hardware-mirror baselines against the four-stage reference."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval_hw_mirror_s2pf_lite import hw_mirror_schedule  # noqa: E402


DEFAULT_INPUTS = (
    ROOT / "results" / "final_reference" / "scheduler_reference_E8_compact.json",
    ROOT / "results" / "final_reference" / "scheduler_reference_E32_compact.json",
    ROOT / "results" / "final_reference" / "scheduler_reference_E64_compact.json",
)
DEFAULT_OUT = ROOT / "results" / "final_reference" / "scheduler_baselines.json"


BASELINES = {
    "current_pruned": {
        "policy": "balanced",
        "top_policy": "pruned",
        "n1_policy": "pruned",
    },
    "expanded_outer": {
        "policy": "balanced",
        "top_policy": "full",
        "n1_policy": "full",
    },
    "top_full_n1_pruned": {
        "policy": "balanced",
        "top_policy": "full",
        "n1_policy": "pruned",
    },
}


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


def percentile(values: list[float], q: float):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def quality_ok(item: dict, quality: str) -> bool:
    if not item.get("analysis_eligible", False):
        return False
    if quality == "eligible":
        return True
    if quality == "within3":
        return float(item.get("optimality_gap", math.inf)) <= 0.03
    if quality == "proven":
        return bool(item.get("proven_optimal", False))
    raise ValueError(quality)


def stratified_keys(keys: list[str], count: int) -> list[str]:
    if count < 0 or count >= len(keys):
        return keys
    if count <= 0:
        return []
    if count == 1:
        return [keys[0]]
    indices = sorted(
        set(round(i * (len(keys) - 1) / (count - 1)) for i in range(count))
    )
    return [keys[index] for index in indices]


def summarize(rows: list[dict]) -> dict:
    buckets = defaultdict(list)
    for row in rows:
        for key in (
            "overall",
            f"baseline:{row['baseline']}",
            f"baseline:{row['baseline']}:split:{row['dataset_split']}",
            f"baseline:{row['baseline']}:E{row['e_total']}",
        ):
            buckets[key].append(row)
    result = {}
    for key, values in sorted(buckets.items()):
        ratios = [float(row["ratio"]) for row in values]
        result[key] = {
            "cases": len(values),
            "ratio_mean": statistics.mean(ratios),
            "ratio_p50": statistics.median(ratios),
            "ratio_p95": percentile(ratios, 0.95),
            "ratio_max": max(ratios),
            "exact": sum(abs(ratio - 1.0) <= 1e-12 for ratio in ratios),
            "le_1pct": sum(ratio <= 1.01 for ratio in ratios),
            "le_3pct": sum(ratio <= 1.03 for ratio in ratios),
            "beats_proven_reference": sum(
                row["reference_proven"] and row["ratio"] < 1.0 - 1e-12
                for row in values
            ),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--baselines", default=",".join(BASELINES))
    parser.add_argument("--quality", choices=("proven", "within3", "eligible"), default="within3")
    parser.add_argument("--dataset-split", action="append", choices=("discovery", "validation", "blind_test"))
    parser.add_argument("--sample-per-file", type=int, default=-1)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    names = [value for value in args.baselines.split(",") if value]
    if any(name not in BASELINES for name in names):
        parser.error(f"unknown baseline; choose {sorted(BASELINES)}")
    allowed_splits = set(args.dataset_split or [])
    rows = []
    started = time.perf_counter()
    cases = 0
    for path in args.inputs:
        payload = json.loads(path.read_text())
        results = payload["results"]
        keys = [
            key
            for key, item in results.items()
            if quality_ok(item, args.quality)
            and (not allowed_splits or item.get("dataset_split") in allowed_splits)
        ]
        for key in stratified_keys(keys, args.sample_per_file):
            item = results[key]
            dist = {int(eid): int(ntok) for eid, ntok in item["dist"].items()}
            reference = int(item["makespan_cc"])
            for name in names:
                config = BASELINES[name]
                makespan = hw_mirror_schedule(
                    dist,
                    int(item.get("initial_cache_c2", -1)),
                    int(item.get("initial_cache_c3", -1)),
                    **config,
                )
                rows.append(
                    {
                        "baseline": name,
                        "case_id": str(key),
                        "e_total": int(item["e_total"]),
                        "dataset_split": item.get("dataset_split"),
                        "quality_class": item.get("quality_class"),
                        "reference_proven": bool(item["proven_optimal"]),
                        "reference_makespan_cc": reference,
                        "baseline_makespan_cc": makespan,
                        "ratio": makespan / reference,
                    }
                )
            cases += 1
            if args.progress_every > 0 and cases % args.progress_every == 0:
                print(
                    f"cases={cases} rows={len(rows)} "
                    f"elapsed_s={time.perf_counter()-started:.1f}",
                    flush=True,
                )

    summary = summarize(rows)
    proven_mismatches = summary.get("overall", {}).get("beats_proven_reference", 0)
    report = {
        "baseline_configs": {name: BASELINES[name] for name in names},
        "quality": args.quality,
        "dataset_splits": sorted(allowed_splits),
        "sample_per_file": args.sample_per_file,
        "runtime_s": time.perf_counter() - started,
        "summary": summary,
        "model_consistency_warning": (
            f"{proven_mismatches} baseline rows beat a proven reference; inspect model parity"
            if proven_mismatches
            else None
        ),
        "worst_rows": sorted(rows, key=lambda row: row["ratio"], reverse=True)[:100],
        "rows": rows,
    }
    atomic_write(args.out, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary.get("overall", {}), indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
