#!/usr/bin/env python3
"""Compare the four fixed S2PF/S4PF SINGLE/BOTH resource contracts.

The candidate bank and integer-tick scorer stay fixed.  Only prefetch DMA
bandwidth and duration change, which isolates the resource-contract effect.
The validated inherited-S2/S4-single and BOTH/BOTH columns are reused from the
existing paired 30K regression to avoid recomputing established baselines.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

import scheduler_rtl_prefetch_both_policy as rtl


ROOT = Path(__file__).resolve().parent
TICK_CC = rtl.TICK_CC
DEFAULT_INPUTS = tuple(
    ROOT / f"scheduler_strategy_coverage_E{e}.json" for e in (8, 32, 64)
)
DEFAULT_BASELINE = (
    ROOT
    / "results"
    / "policy_search"
    / "scheduler_rtl_prefetch_both_vs_hw_v2_30k.json"
)
DEFAULT_OUT = (
    ROOT / "results" / "policy_search" / "prefetch_fixed_modes_30k.json"
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


class FixedPrefetchCostModel(rtl._RtlPrefetchCostModel):
    def __init__(self, *, s2_both: bool, s4_both: bool) -> None:
        self.s2_bw = 128 if s2_both else 64
        self.s2_duration = (1 if s2_both else 2) * TICK_CC
        self.s4_bw = 128 if s4_both else 64
        self.s4_duration = (2 if s4_both else 4) * TICK_CC

    def _cc_snap_segs(self, snap):
        segments = []
        if snap.cur_eid >= 0 and snap.bw_s1 > 0:
            segments.append((snap.task_start, snap.dma1_end, snap.bw_s1))
        if snap.s2pf_start >= 0 and snap.s2pf_bw > 0:
            segments.append((snap.s2pf_start, snap.s2pf_end, self.s2_bw))
        if snap.cur_eid >= 0 and snap.bw_s3 > 0 and snap.dma3_end > snap.s2_end:
            segments.append((snap.s2_end, snap.dma3_end, snap.bw_s3))
        if snap.cur_eid >= 0 and snap.s4pf_valid:
            segments.append(
                (snap.dma3_end, snap.dma3_end + self.s4_duration, self.s4_bw)
            )
        return segments

    def _cc_apply_s2pf(self, snap, _shape_s3: int, start: int):
        if snap.bw_s3 == 0:
            return snap
        end = start + self.s2_duration
        if start < snap.dma1_end or end > snap.s2_end:
            return snap
        updated = replace(snap)
        updated.s2pf_start = start
        updated.s2pf_end = end
        updated.s2pf_bw = self.s2_bw
        updated.dma3_end = updated.s2_end
        updated.s3_end = updated.s2_end
        updated.s4_start = updated.s2_end
        updated.bw_s3 = 0
        updated.task_end = updated.s2_end + self._cc_best_s4(updated.ntok)
        return updated

    def _cc_s4pf_local_ok(self, snap) -> bool:
        return (
            snap.cur_eid >= 0
            and snap.pf_eid == -1
            and snap.dma3_end + self.s4_duration <= snap.task_end
        )

    def _cc_busy_time_points(self, busy, idle_time: int):
        points = [idle_time]
        s1_release = busy.dma1_end
        stage3_release = busy.s2pf_end if busy.s2pf_start >= 0 else busy.dma3_end
        s4pf_release = busy.dma3_end + self.s4_duration
        s1_valid = busy.cur_eid >= 0 and busy.bw_s1 > 0 and s1_release > idle_time
        stage3_valid = (
            busy.cur_eid >= 0
            and (busy.s2pf_start >= 0 or busy.bw_s3 > 0)
            and stage3_release > idle_time
        )
        s4pf_valid = (
            busy.cur_eid >= 0 and busy.s4pf_valid and s4pf_release > idle_time
        )
        if s1_valid:
            points.append(s1_release)
            if s4pf_valid and s4pf_release != s1_release:
                points.append(s4pf_release)
            elif stage3_valid and stage3_release != s1_release:
                points.append(stage3_release)
        elif stage3_valid:
            points.append(stage3_release)
            if s4pf_valid and s4pf_release != stage3_release:
                points.append(s4pf_release)
        elif s4pf_valid:
            points.append(s4pf_release)
        return points


MODELS = {
    "SS": FixedPrefetchCostModel(s2_both=False, s4_both=False),
    "SB": FixedPrefetchCostModel(s2_both=False, s4_both=True),
    "BS": FixedPrefetchCostModel(s2_both=True, s4_both=False),
    "BB": FixedPrefetchCostModel(s2_both=True, s4_both=True),
}


def schedule(dist: dict[int, int], c2: int, c3: int, mode: str) -> int:
    return rtl._schedule_result(
        dist, c2, c3, cost_model=MODELS[mode]
    ).makespan_cc


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
        "lhs_total_cc": lhs_total,
        "rhs_total_cc": rhs_total,
        "aggregate_delta_cc": lhs_total - rhs_total,
        "aggregate_delta_pct": (lhs_total / rhs_total - 1.0) * 100.0,
    }


def summarize(rows: list[dict]) -> dict:
    result = {}
    for mode in MODELS:
        result[f"{mode}_vs_inherited_single"] = comparison(
            rows, f"{mode}_cc", "inherited_s2_s4_single_cc"
        )
    result["SS_vs_BB"] = comparison(rows, "SS_cc", "BB_cc")
    result["SB_vs_SS_isolate_s4"] = comparison(rows, "SB_cc", "SS_cc")
    result["BS_vs_SS_isolate_s2"] = comparison(rows, "BS_cc", "SS_cc")
    result["BB_vs_SB_isolate_s2"] = comparison(rows, "BB_cc", "SB_cc")
    result["BB_vs_BS_isolate_s4"] = comparison(rows, "BB_cc", "BS_cc")
    totals = {mode: sum(row[f"{mode}_cc"] for row in rows) for mode in MODELS}
    result["fixed_mode_total_rank"] = sorted(totals.items(), key=lambda item: item[1])
    oracle_total = sum(min(row[f"{mode}_cc"] for mode in MODELS) for row in rows)
    best_fixed_total = min(totals.values())
    result["per_case_static_oracle"] = {
        "total_cc": oracle_total,
        "best_fixed_total_cc": best_fixed_total,
        "delta_vs_best_fixed_cc": oracle_total - best_fixed_total,
        "delta_vs_best_fixed_pct": (oracle_total / best_fixed_total - 1.0) * 100.0,
        "winner_counts_with_ties": {
            mode: sum(
                row[f"{mode}_cc"]
                == min(row[f"{candidate}_cc"] for candidate in MODELS)
                for row in rows
            )
            for mode in MODELS
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=2_000)
    args = parser.parse_args()
    inputs = tuple(args.input) if args.input else DEFAULT_INPUTS
    baseline = json.loads(args.baseline.read_text())["rows"]

    rows = []
    started = time.perf_counter()
    stop = False
    for input_path in inputs:
        for case in json.loads(input_path.read_text())["cases"]:
            if not case.get("analysis_eligible", False):
                continue
            if args.limit > 0 and len(rows) >= args.limit:
                stop = True
                break
            key = f"E{int(case['e_total'])}:{int(case['case_id'])}"
            prior = baseline[key]
            dist = {int(eid): int(ntok) for eid, ntok in case["dist"].items()}
            c2 = int(case.get("c2", -1))
            c3 = int(case.get("c3", -1))
            row = {
                "key": key,
                "e_total": int(case["e_total"]),
                "case_id": int(case["case_id"]),
                "dataset_split": case.get("dataset_split"),
                "inherited_s2_s4_single_cc": int(prior["old_prefetch_tick_cc"]),
                "SS_cc": schedule(dist, c2, c3, "SS"),
                "SB_cc": schedule(dist, c2, c3, "SB"),
                "BS_cc": schedule(dist, c2, c3, "BS"),
                "BB_cc": int(prior["current_rtl_cc"]),
            }
            rows.append(row)
            if args.progress_every > 0 and len(rows) % args.progress_every == 0:
                print(
                    f"prefetch-ablation completed={len(rows)} "
                    f"elapsed_s={time.perf_counter() - started:.1f}",
                    flush=True,
                )
        if stop:
            break

    expected = min(29_928, args.limit) if args.limit > 0 else 29_928
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} eligible cases, got {len(rows)}")

    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets["overall"].append(row)
        buckets[f"E{row['e_total']}"] .append(row)
        buckets[f"split:{row['dataset_split']}"] .append(row)
    payload = {
        "schema": "prefetch_fixed_modes_30k_v1",
        "configuration": {
            "mode_key": {
                "SS": "S2PF SINGLE, S4PF SINGLE",
                "SB": "S2PF SINGLE, S4PF BOTH",
                "BS": "S2PF BOTH, S4PF SINGLE",
                "BB": "S2PF BOTH, S4PF BOTH",
            },
            "candidate_bank": "frozen HW-v2",
            "score_domain": "integer_tick_ceil",
            "inputs": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in inputs
            ],
            "baseline": {
                "path": str(args.baseline.resolve()),
                "sha256": sha256(args.baseline),
            },
            "limit": args.limit,
        },
        "runtime_s": time.perf_counter() - started,
        "summary": {name: summarize(values) for name, values in sorted(buckets.items())},
        "rows": {row["key"]: row for row in rows},
    }
    atomic_write(args.out, payload)
    print(json.dumps(payload["summary"]["overall"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
