#!/usr/bin/env python3
"""Closed-loop tail-policy ablation for the deployed hardware scheduler.

This tool changes only the continuation estimate used when scoring a child
with one remaining expert.  Candidate generation, timeline construction,
S2PF/S4PF policy, bandwidth checks, tie-breaking, and final n=1 execution all
remain those of ``eval_hw_mirror_s2pf_lite.py``.

The three compared variants are:

* ``original_sim1``: the deployed C/RTL mirror, including its restricted sim1;
* ``no_sim1``: use the ordinary greedy continuation for every non-empty tail;
* ``policy_consistent_tail``: predict n=1 with exactly the same candidate set
  and state updates used by the mirror's real final scheduling round.

The existing 30K comparison checkpoint is read-only.  Its hash is checked
again before writing the result so a concurrently changing checkpoint cannot
silently produce a mixed snapshot.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import eval_hw_mirror_s2pf_lite as hw


ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = ROOT / "results" / "policy_search" / "scheduler_strategies_30k.json"
DEFAULT_SUITES = (
    ROOT / "scheduler_strategy_coverage_E8.json",
    ROOT / "scheduler_strategy_coverage_E32.json",
    ROOT / "scheduler_strategy_coverage_E64.json",
)
DEFAULT_OUT = ROOT / "results" / "policy_search" / "scheduler_hw_tail_ablation.json"
HW_CONFIG = {
    "policy": "balanced",
    "top_policy": "pruned",
    "n1_policy": "pruned",
}
VARIANTS = ("original_sim1", "no_sim1", "policy_consistent_tail")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def no_sim1_continuation(c2, c3, remaining: tuple, *, policy: str) -> int:
    """Use the existing aggregate greedy score for every non-empty tail."""
    del policy
    if not remaining:
        return max(c2.task_end, c3.task_end)
    return hw.cm._cc_greedy_h(c2.task_end, c3.task_end, remaining)


def policy_consistent_n1_cost(c2, c3, eid: int, ntok: int, *, policy: str) -> int:
    """Return the makespan selected by the mirror's actual pruned n=1 round."""
    cm = hw.cm

    # The real main loop installs legal S4PF ghosts before constructing n=1
    # candidates.  Preserve the same sequential C2-then-C3 update order.
    if cm._cc_s4pf_ok_with_peer(c2, c3):
        c2 = cm._cc_apply_s4pf_ghost(c2)
    if cm._cc_s4pf_ok_with_peer(c3, c2):
        c3 = cm._cc_apply_s4pf_ghost(c3)

    t2 = c2.task_end
    t3 = c3.task_end
    t_now = max(t2, t3)
    best = cm.C_INF

    # Method A: the deployed pruned five-shape bank on either cluster.  Unlike
    # the old sim1, each cluster starts at its own task_end.
    for cluster in (0, 1):
        current, peer = (c2, c3) if cluster == 0 else (c3, c2)
        start = current.task_end
        sw_hit = cm._cc_swiglu_hit(eid, current, start)
        down_hit = cm._cc_down_hit(eid, current, start)
        for shape_s1, shape_s3 in hw.N1_PRUNED_SOLO_SHAPES:
            snap = cm._cc_mk_snap(
                start, shape_s1, shape_s3, ntok, eid, sw_hit, down_hit
            )
            if cm._cc_bw_ok(snap, peer):
                best = min(best, max(snap.task_end, peer.task_end))

    # The deployed n=1 split bank contains only ceil-half.
    if ntok >= 2:
        left = (ntok + 1) // 2
        right = ntok - left
        sw2 = cm._cc_swiglu_hit(eid, c2, t_now)
        dn2 = cm._cc_down_hit(eid, c2, t_now)
        sw3 = cm._cc_swiglu_hit(eid, c3, t_now)
        dn3 = cm._cc_down_hit(eid, c3, t_now)
        s12, s32, s13, s33 = cm._cc_pick_shapes(
            left, right, sw2, dn2, sw3, dn3, t_now
        )
        snap2 = cm._cc_mk_snap(t_now, s12, s32, left, eid, sw2, dn2)
        snap3 = cm._cc_mk_snap(t_now, s13, s33, right, eid, sw3, dn3)
        snap2, snap3 = hw._hw_try_s2pf_pair(
            "n1_split", snap2, s32, snap3, s33, policy=policy
        )
        if cm._cc_bw_ok(snap2, snap3):
            best = min(best, max(snap2.task_end, snap3.task_end))

    # Method B: the same two pruned busy-lane release endpoints used by the
    # real final round.  Index zero is the already-tested idle start.
    if t2 != t3:
        idle_cluster = 0 if t2 < t3 else 1
        idle, busy = (c2, c3) if idle_cluster == 0 else (c3, c2)
        for start in cm._cc_busy_time_points(busy, idle.task_end)[1:]:
            sw_hit = cm._cc_swiglu_hit(eid, idle, start)
            down_hit = cm._cc_down_hit(eid, idle, start)
            snap = cm._cc_mk_snap(
                start,
                cm.C_SHAPE_C,
                cm.C_SHAPE_C,
                ntok,
                eid,
                sw_hit,
                down_hit,
            )
            feasible = (
                cm._cc_bw_ok(snap, busy)
                if idle_cluster == 0
                else cm._cc_bw_ok(busy, snap)
            )
            if feasible:
                best = min(best, max(snap.task_end, busy.task_end))

    return t_now + cm._cc_best_task(ntok) if best == cm.C_INF else best


def policy_consistent_continuation(c2, c3, remaining: tuple, *, policy: str) -> int:
    if not remaining:
        return max(c2.task_end, c3.task_end)
    if len(remaining) == 1:
        eid, ntok = remaining[0]
        return policy_consistent_n1_cost(c2, c3, eid, ntok, policy=policy)
    return hw.cm._cc_greedy_h(c2.task_end, c3.task_end, remaining)


def load_cases(report: dict, suite_paths: tuple[Path, ...]) -> list[tuple[str, dict]]:
    wanted = set(report["results"])
    cases: dict[str, dict] = {}
    for path in suite_paths:
        payload = json.loads(path.read_text())
        for case in payload["cases"]:
            key = f"E{int(case['e_total'])}:{int(case['case_id'])}"
            if key in wanted:
                if key in cases:
                    raise ValueError(f"duplicate case {key}")
                cases[key] = case
    missing = wanted - set(cases)
    if missing:
        raise ValueError(f"missing {len(missing)} checkpoint cases from suites")
    return sorted(cases.items(), key=lambda item: (int(item[1]["e_total"]), int(item[1]["case_id"])))


def run_variant(case: dict, continuation) -> int:
    original = hw._hw_continuation_cost
    hw._hw_continuation_cost = continuation
    try:
        distribution = {int(eid): int(ntok) for eid, ntok in case["dist"].items()}
        return int(
            hw.hw_mirror_schedule(
                distribution,
                int(case.get("c2", -1)),
                int(case.get("c3", -1)),
                **HW_CONFIG,
            )
        )
    finally:
        hw._hw_continuation_cost = original


def comparison_summary(rows: list[dict], variant: str) -> dict:
    better = [row for row in rows if row[variant] < row["original_sim1"]]
    worse = [row for row in rows if row[variant] > row["original_sim1"]]
    original_total = sum(row["original_sim1"] for row in rows)
    variant_total = sum(row[variant] for row in rows)
    ratios = [row[variant] / row["original_sim1"] for row in rows]
    return {
        "cases": len(rows),
        "better": len(better),
        "worse": len(worse),
        "equal": len(rows) - len(better) - len(worse),
        "saved_cc": sum(row["original_sim1"] - row[variant] for row in better),
        "lost_cc": sum(row[variant] - row["original_sim1"] for row in worse),
        "variant_minus_original_cc": variant_total - original_total,
        "variant_over_original_aggregate": variant_total / original_total,
        "ratio_mean": sum(ratios) / len(ratios),
        "ratio_p95": percentile(ratios, 0.95),
        "ratio_max": max(ratios),
    }


def reference_summary(rows: list[dict], variant: str) -> dict:
    proven = [row for row in rows if row["reference_proven_optimal"]]
    reference_total = sum(row["reference_makespan_cc"] for row in proven)
    variant_total = sum(row[variant] for row in proven)
    return {
        "cases": len(proven),
        "exact": sum(row[variant] == row["reference_makespan_cc"] for row in proven),
        "variant_over_reference_aggregate": variant_total / reference_total,
        "ratio_max": max(row[variant] / row["reference_makespan_cc"] for row in proven),
    }


def summarize(rows: list[dict]) -> dict:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        for key in (
            "overall",
            f"E{row['e_total']}",
            f"split:{row['dataset_split']}",
            f"E{row['e_total']}:split:{row['dataset_split']}",
        ):
            buckets[key].append(row)
    result = {}
    for key, bucket in sorted(buckets.items()):
        result[key] = {
            "vs_original": {
                variant: comparison_summary(bucket, variant)
                for variant in VARIANTS[1:]
            },
            "vs_proven_reference": {
                variant: reference_summary(bucket, variant) for variant in VARIANTS
            },
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--suite", action="append", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    suite_paths = tuple(args.suite) if args.suite else DEFAULT_SUITES
    report_hash = file_sha256(args.report)
    report = json.loads(args.report.read_text())
    cases = load_cases(report, suite_paths)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        cases = cases[: args.limit]

    original_continuation = hw._hw_continuation_cost
    continuations = {
        "original_sim1": original_continuation,
        "no_sim1": no_sim1_continuation,
        "policy_consistent_tail": policy_consistent_continuation,
    }
    rows = []
    started = time.perf_counter()
    for index, (key, case) in enumerate(cases, 1):
        source = report["results"][key]
        makespans = {
            variant: run_variant(case, continuation)
            for variant, continuation in continuations.items()
        }
        recorded = int(source["makespan_cc"]["current_hardware_lite"])
        if makespans["original_sim1"] != recorded:
            raise RuntimeError(
                f"baseline mismatch for {key}: recomputed "
                f"{makespans['original_sim1']} != checkpoint {recorded}"
            )
        rows.append(
            {
                "key": key,
                "e_total": int(case["e_total"]),
                "case_id": int(case["case_id"]),
                "dataset_split": case.get("dataset_split"),
                "active_n": int(case.get("active_n", len(case["dist"]))),
                "m_total": int(case.get("m_total", 0)),
                "construction": case.get("construction"),
                "cache_regime": case.get("cache_regime"),
                "reference_makespan_cc": int(source["reference_makespan_cc"]),
                "reference_proven_optimal": bool(source["reference_proven_optimal"]),
                **makespans,
            }
        )
        if args.progress_every > 0 and index % args.progress_every == 0:
            print(
                f"tail-ablation completed={index}/{len(cases)} "
                f"elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )

    if file_sha256(args.report) != report_hash:
        raise RuntimeError("input 30K checkpoint changed during tail ablation")

    for variant, continuation in continuations.items():
        del continuation
        if variant not in VARIANTS:
            raise AssertionError(f"unexpected variant {variant}")
    hw._hw_continuation_cost = original_continuation

    def extremes(variant: str, reverse: bool) -> list[dict]:
        ordered = sorted(
            rows,
            key=lambda row: row[variant] / row["original_sim1"],
            reverse=reverse,
        )
        selected = []
        for row in ordered:
            if row[variant] == row["original_sim1"]:
                continue
            selected.append(
                {
                    **row,
                    "variant": variant,
                    "variant_over_original": row[variant] / row["original_sim1"],
                }
            )
            if len(selected) == 20:
                break
        return selected

    payload = {
        "schema": "scheduler_hw_tail_ablation_v1",
        "provisional": len(cases) != int(report.get("analysis_eligible_cases", len(cases))),
        "configuration": {
            "source_report": str(args.report.resolve()),
            "source_report_sha256": report_hash,
            "source_report_completed_cases": int(report.get("completed_cases", len(report["results"]))),
            "source_report_analysis_eligible_cases": int(report.get("analysis_eligible_cases", len(report["results"]))),
            "suites": [
                {"path": str(path.resolve()), "sha256": file_sha256(path)}
                for path in suite_paths
            ],
            "hw_mirror": {
                "path": str(Path(hw.__file__).resolve()),
                "sha256": file_sha256(Path(hw.__file__)),
                **HW_CONFIG,
            },
            "variants": list(VARIANTS),
            "changed_semantics": "continuation score only when one child expert remains",
        },
        "cases": len(rows),
        "runtime_s": time.perf_counter() - started,
        "summary": summarize(rows),
        "worst_regressions": {
            variant: extremes(variant, True) for variant in VARIANTS[1:]
        },
        "best_improvements": {
            variant: extremes(variant, False) for variant in VARIANTS[1:]
        },
        "results": {row["key"]: {k: v for k, v in row.items() if k != "key"} for row in rows},
    }
    atomic_write_json(args.out, payload)
    overall = payload["summary"]["overall"]
    print(f"wrote {args.out}")
    for variant in VARIANTS[1:]:
        stats = overall["vs_original"][variant]
        print(
            variant,
            f"better={stats['better']}",
            f"worse={stats['worse']}",
            f"equal={stats['equal']}",
            f"aggregate_delta_pct={(stats['variant_over_original_aggregate'] - 1.0) * 100.0:.6f}",
            f"max_regression_pct={(stats['ratio_max'] - 1.0) * 100.0:.6f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
