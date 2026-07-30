#!/usr/bin/env python3
"""Compare S4PF single/both selection orders with fixed single-lane S2PF."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

from scheduler_hw_fixed_policy import hw_v2_schedule
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
    / "scheduler_rtl_adaptive_prefetch_vs_hw_v2_30k.json"
)
POLICIES = (
    "single_only",
    "both_only",
    "single_first",
    "both_first",
    "window_select",
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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def comparison(rows: list[dict], lhs: str, rhs: str) -> dict:
    lhs_total = sum(row[lhs] for row in rows)
    rhs_total = sum(row[rhs] for row in rows)
    return {
        "lhs": lhs,
        "rhs": rhs,
        "cases": len(rows),
        "better": sum(row[lhs] < row[rhs] for row in rows),
        "equal": sum(row[lhs] == row[rhs] for row in rows),
        "worse": sum(row[lhs] > row[rhs] for row in rows),
        "changed": sum(row[lhs] != row[rhs] for row in rows),
        "lhs_total_cc": lhs_total,
        "rhs_total_cc": rhs_total,
        "aggregate_delta_cc": lhs_total - rhs_total,
        "aggregate_delta_pct": (lhs_total / rhs_total - 1.0) * 100.0,
    }


def summarize(rows: list[dict], policies: tuple[str, ...]) -> dict:
    summary = {}
    for policy in policies:
        summary[f"{policy}_vs_hw_v2"] = comparison(
            rows, f"{policy}_cc", "original_hw_v2_cc"
        )
    if "single_first" in policies and "both_first" in policies:
        summary["single_first_vs_both_first"] = comparison(
            rows, "single_first_cc", "both_first_cc"
        )
    if "single_first" in policies and "single_only" in policies:
        summary["single_first_vs_single_only"] = comparison(
            rows, "single_first_cc", "single_only_cc"
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--policy", action="append", choices=POLICIES)
    parser.add_argument("--progress-every", type=int, default=2_000)
    args = parser.parse_args()
    inputs = tuple(args.input) if args.input else DEFAULT_INPUTS
    policies = tuple(args.policy) if args.policy else POLICIES

    rows = []
    started = time.perf_counter()
    stop = False
    for input_path in inputs:
        cases = json.loads(input_path.read_text())["cases"]
        for case in cases:
            if not case.get("analysis_eligible", False):
                continue
            dist = {int(eid): int(ntok) for eid, ntok in case["dist"].items()}
            c2 = int(case.get("c2", -1))
            c3 = int(case.get("c3", -1))
            row = {
                "key": f"E{int(case['e_total'])}:{int(case['case_id'])}",
                "e_total": int(case["e_total"]),
                "case_id": int(case["case_id"]),
                "dataset_split": case.get("dataset_split"),
                "original_hw_v2_cc": int(hw_v2_schedule(dist, c2, c3)),
            }
            for policy in policies:
                row[f"{policy}_cc"] = int(
                    adaptive_prefetch_schedule(dist, c2, c3, s4_policy=policy)
                )
            rows.append(row)
            if args.progress_every > 0 and len(rows) % args.progress_every == 0:
                print(
                    f"adaptive-prefetch completed={len(rows)} "
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

    payload = {
        "schema": "scheduler_rtl_adaptive_prefetch_vs_hw_v2_30k_v1",
        "configuration": {
            "s2pf_dma": "SINGLE",
            "s2pf_duration_ticks": 2,
            "s4pf_modes": {
                "SINGLE": {"duration_ticks": 4},
                "BOTH": {"duration_ticks": 2},
            },
            "policies": policies,
            "score_domain": "integer_tick_ceil",
            "inputs": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in inputs
            ],
            "source_sha256": {
                "original_hw_v2": sha256(ROOT / "scheduler_hw_fixed_policy.py"),
                "adaptive_rtl": sha256(
                    ROOT / "scheduler_rtl_adaptive_prefetch_policy.py"
                ),
                "driver": sha256(Path(__file__).resolve()),
            },
        },
        "runtime_s": time.perf_counter() - started,
        "summary": {
            key: summarize(values, policies) for key, values in sorted(buckets.items())
        },
        "rows": {row["key"]: row for row in rows},
    }
    atomic_write(args.out, payload)
    print(json.dumps(payload["summary"]["overall"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
