#!/usr/bin/env python3
"""Final paired comparison: frozen HW-v2, deployed HW, and four-stage reference."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

from eval_hw_mirror_s2pf_lite import hw_mirror_schedule
from scheduler_hw_fixed_policy import hw_v2_schedule


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUTS = tuple(ROOT / f"scheduler_strategy_coverage_E{e}.json" for e in (8, 32, 64))
DEFAULT_REFERENCES = tuple(
    ROOT / "results" / "final_reference" / f"scheduler_reference_E{e}_compact.json"
    for e in (8, 32, 64)
)
DEFAULT_ORACLE = ROOT / "results" / "policy_search" / "scheduler_hw_candidate_oracle_one_idle_shape_v2.json"
DEFAULT_OUT = ROOT / "results" / "policy_search" / "scheduler_hw_v2_30k_comparison.json"
HW_CONFIG = {"policy": "balanced", "top_policy": "pruned", "n1_policy": "pruned"}


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


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[max(0, min(len(values) - 1, math.ceil(len(values) * fraction) - 1))]


def summarize(rows: list[dict]) -> dict:
    old_total = sum(row["old_hw_cc"] for row in rows)
    new_total = sum(row["hw_v2_cc"] for row in rows)
    ratios = [row["hw_v2_cc"] / row["old_hw_cc"] for row in rows]
    proven = [row for row in rows if row["reference_proven_optimal"]]
    ref_total = sum(row["reference_cc"] for row in proven)
    old_proven = sum(row["old_hw_cc"] for row in proven)
    new_proven = sum(row["hw_v2_cc"] for row in proven)
    incumbent_total = sum(row["reference_cc"] for row in rows)
    return {
        "cases": len(rows),
        "hw_v2_vs_old": {
            "aggregate_ratio": new_total / old_total,
            "aggregate_delta_pct": (new_total / old_total - 1.0) * 100.0,
            "better": sum(row["hw_v2_cc"] < row["old_hw_cc"] for row in rows),
            "worse": sum(row["hw_v2_cc"] > row["old_hw_cc"] for row in rows),
            "equal": sum(row["hw_v2_cc"] == row["old_hw_cc"] for row in rows),
            "ratio_p95": percentile(ratios, 0.95),
            "ratio_max": max(ratios),
        },
        "vs_proven_four_stage": {
            "cases": len(proven),
            "old_hw_aggregate_gap_pct": (old_proven / ref_total - 1.0) * 100.0,
            "hw_v2_aggregate_gap_pct": (new_proven / ref_total - 1.0) * 100.0,
            "old_hw_exact": sum(row["old_hw_cc"] == row["reference_cc"] for row in proven),
            "hw_v2_exact": sum(row["hw_v2_cc"] == row["reference_cc"] for row in proven),
            "old_hw_beats_reference": sum(row["old_hw_cc"] < row["reference_cc"] for row in proven),
            "hw_v2_beats_reference": sum(row["hw_v2_cc"] < row["reference_cc"] for row in proven),
        },
        "vs_all_four_stage_incumbents": {
            "cases": len(rows),
            "old_hw_aggregate_delta_pct": (old_total / incumbent_total - 1.0) * 100.0,
            "hw_v2_aggregate_delta_pct": (new_total / incumbent_total - 1.0) * 100.0,
            "old_hw_beats_incumbent": sum(row["old_hw_cc"] < row["reference_cc"] for row in rows),
            "hw_v2_beats_incumbent": sum(row["hw_v2_cc"] < row["reference_cc"] for row in rows),
            "warning": "The 6339 unproven reference values are feasible incumbents, not certified optima.",
        },
    }


def oracle_attribution(rows_by_key: dict[str, dict], oracle_path: Path) -> dict:
    oracle = json.loads(oracle_path.read_text())
    oracle_name = f"oracle_w{max(oracle['configuration']['beam_widths'])}"
    sampled = []
    for item in oracle["rows"]:
        if item["key"] not in rows_by_key:
            raise ValueError(f"oracle case missing from 30K comparison: {item['key']}")
        row = rows_by_key[item["key"]]
        sampled.append(
            {
                **item,
                "hw_v2_cc": row["hw_v2_cc"],
                "old_hw_cc": row["old_hw_cc"],
            }
        )

    def split(values: list[dict]) -> dict:
        reference = sum(row["reference_cc"] for row in values)
        candidate = sum(row[oracle_name]["makespan_cc"] for row in values)
        selected = sum(row["hw_v2_cc"] for row in values)
        residual = selected - reference
        candidate_loss = candidate - reference
        selection_loss = selected - candidate
        return {
            "cases": len(values),
            "reference_cc": reference,
            "candidate_oracle_cc": candidate,
            "hw_v2_cc": selected,
            "candidate_loss_cc": candidate_loss,
            "selection_loss_cc": selection_loss,
            "candidate_loss_fraction_of_residual": candidate_loss / residual if residual else 0.0,
            "selection_loss_fraction_of_residual": selection_loss / residual if residual else 0.0,
            "candidate_matches_reference": sum(
                row[oracle_name]["makespan_cc"] == row["reference_cc"] for row in values
            ),
            "scorer_matches_candidate_oracle": sum(
                row["hw_v2_cc"] == row[oracle_name]["makespan_cc"] for row in values
            ),
        }

    exact = [row for row in sampled if row[oracle_name]["exact_candidate_optimum"]]
    return {
        "oracle_source": str(oracle_path.resolve()),
        "oracle_source_sha256": sha256(oracle_path),
        "beam_sample": split(sampled),
        "certified_no_pruning_subset": split(exact),
        "interpretation": (
            "On the no-pruning subset, candidate_loss is exact for the fixed candidate graph; "
            "selection_loss is the remaining HW-v2 scorer/control loss."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path)
    parser.add_argument("--reference", action="append", type=Path)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--progress-every", type=int, default=2000)
    args = parser.parse_args()
    inputs = tuple(args.input) if args.input else DEFAULT_INPUTS
    references = tuple(args.reference) if args.reference else DEFAULT_REFERENCES
    if len(inputs) != len(references):
        raise ValueError("input/reference count mismatch")

    rows = []
    started = time.perf_counter()
    for input_path, reference_path in zip(inputs, references):
        cases = json.loads(input_path.read_text())["cases"]
        reference = json.loads(reference_path.read_text())["results"]
        for case in cases:
            if not case.get("analysis_eligible", False):
                continue
            case_id = int(case["case_id"])
            truth = reference[str(case_id)]
            dist = {int(eid): int(ntok) for eid, ntok in case["dist"].items()}
            c2, c3 = int(case.get("c2", -1)), int(case.get("c3", -1))
            old_cc = int(hw_mirror_schedule(dist, c2, c3, **HW_CONFIG))
            new_cc = int(hw_v2_schedule(dist, c2, c3, **HW_CONFIG))
            rows.append(
                {
                    "key": f"E{int(case['e_total'])}:{case_id}",
                    "case_id": case_id,
                    "e_total": int(case["e_total"]),
                    "dataset_split": case.get("dataset_split"),
                    "active_n": int(case.get("active_n", len(dist))),
                    "m_total": int(case.get("m_total", 0)),
                    "construction": case.get("construction"),
                    "cache_regime": case.get("cache_regime"),
                    "old_hw_cc": old_cc,
                    "hw_v2_cc": new_cc,
                    "reference_cc": int(truth["makespan_cc"]),
                    "reference_lower_bound_cc": int(truth["lower_bound_cc"]),
                    "reference_proven_optimal": bool(truth.get("proven_optimal", False)),
                }
            )
            if args.progress_every > 0 and len(rows) % args.progress_every == 0:
                print(f"hw-v2-30k completed={len(rows)} elapsed_s={time.perf_counter()-started:.1f}", flush=True)

    if len(rows) != 29928:
        raise RuntimeError(f"expected 29928 eligible cases, got {len(rows)}")
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        for key in ("overall", f"E{row['e_total']}", f"split:{row['dataset_split']}"):
            buckets[key].append(row)
    rows_by_key = {row["key"]: row for row in rows}
    payload = {
        "schema": "scheduler_hw_v2_30k_comparison_v1",
        "configuration": {
            "old_hw": "eval_hw_mirror_s2pf_lite.hw_mirror_schedule",
            "hw_v2": "scheduler_hw_fixed_policy.hw_v2_schedule",
            "hw_config": HW_CONFIG,
            "inputs": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in inputs],
            "references": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in references],
            "source_sha256": {
                "old_hw": sha256(ROOT / "eval_hw_mirror_s2pf_lite.py"),
                "hw_v2": sha256(ROOT / "scheduler_hw_fixed_policy.py"),
                "driver": sha256(Path(__file__).resolve()),
            },
        },
        "runtime_s": time.perf_counter() - started,
        "summary": {key: summarize(values) for key, values in sorted(buckets.items())},
        "failure_attribution": oracle_attribution(rows_by_key, args.oracle),
        "rows": rows_by_key,
    }
    atomic_write(args.out, payload)
    print(json.dumps(payload["summary"]["overall"], indent=2))
    print(json.dumps(payload["failure_attribution"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
