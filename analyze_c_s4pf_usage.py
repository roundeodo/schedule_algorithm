#!/usr/bin/env python3
"""Count committed S4 ghost-prefetch events in the C-accurate scheduler mirror."""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import eval_c_mirror_v2 as cm


DEFAULT_FILES = (
    ROOT / "scheduler_eval_inputs_E8_stratified_v6.json",
    ROOT / "scheduler_eval_inputs_E32_stratified_v6.json",
    ROOT / "scheduler_eval_inputs_E64_stratified_v6.json",
)


def stratified_indices(n_cases: int, n_pick: int | None) -> list[int]:
    if n_pick is None or n_pick >= n_cases:
        return list(range(n_cases))
    if n_pick <= 0:
        return []
    if n_pick == 1:
        return [0]
    return sorted(set(round(i * (n_cases - 1) / (n_pick - 1)) for i in range(n_pick)))


def inc(counter: dict[str, int], key: Any, amount: int = 1) -> None:
    counter[str(key)] += amount


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-per-file", type=int, default=-1)
    parser.add_argument("--out", type=Path, default=ROOT / "c_s4pf_usage_summary_full.json")
    args = parser.parse_args()

    original_ok = cm._cc_s4pf_ok_with_peer
    original_apply = cm._cc_apply_s4pf_ghost

    current: dict[str, Any] = {}
    current_events: list[dict[str, Any]] = []

    def ok_no_count(s: cm.CSnap, peer: cm.CSnap) -> bool:
        if not cm._cc_s4pf_local_ok(s):
            return False
        return cm._cc_bw_ok(original_apply(s), peer)

    def apply_counted(s: cm.CSnap) -> cm.CSnap:
        out = original_apply(s)
        if out.s4pf_valid:
            current_events.append(
                {
                    "file": current.get("file", ""),
                    "case_id": current.get("case_id", ""),
                    "profile": current.get("profile", ""),
                    "active_n": current.get("active_n", 0),
                    "m_total": current.get("m_total", 0),
                    "eid": s.cur_eid,
                    "task_end": s.task_end,
                    "s4_start": s.s4_start,
                    "s4pf_start": out.s4pf_start,
                }
            )
        return out

    cm._cc_s4pf_ok_with_peer = ok_no_count
    cm._cc_apply_s4pf_ghost = apply_counted

    t0 = time.perf_counter()
    rows = []
    try:
        for path in DEFAULT_FILES:
            payload = json.loads(path.read_text())
            cases = payload["cases"]
            sample = None if args.sample_per_file < 0 else args.sample_per_file
            idxs = stratified_indices(len(cases), sample)
            for idx in idxs:
                case = cases[idx]
                current.clear()
                current.update(
                    {
                        "file": path.name,
                        "case_id": case["case_id"],
                        "profile": case["profile"],
                        "active_n": case["active_n"],
                        "m_total": case["m_total"],
                    }
                )
                before = len(current_events)
                dist = {int(k): int(v) for k, v in case["dist"].items()}
                makespan = cm.c_mirror_v2_schedule(dist, int(case["c2"]), int(case["c3"]))
                n_events = len(current_events) - before
                rows.append(
                    {
                        "file": path.name,
                        "case_id": case["case_id"],
                        "profile": case["profile"],
                        "active_n": case["active_n"],
                        "m_total": case["m_total"],
                        "makespan": makespan,
                        "s4pf_count": n_events,
                    }
                )
    finally:
        cm._cc_s4pf_ok_with_peer = original_ok
        cm._cc_apply_s4pf_ghost = original_apply

    total_cases = len(rows)
    total_events = len(current_events)
    cases_with = sum(1 for r in rows if r["s4pf_count"] > 0)
    count_hist = collections.Counter(r["s4pf_count"] for r in rows)
    by_file = collections.defaultdict(int)
    cases_by_file = collections.defaultdict(int)
    cases_with_by_file = collections.defaultdict(int)
    by_profile = collections.defaultdict(int)
    cases_by_profile = collections.defaultdict(int)
    cases_with_by_profile = collections.defaultdict(int)
    by_active = collections.defaultdict(int)
    cases_by_active = collections.defaultdict(int)
    by_m_total = collections.defaultdict(int)
    by_eid = collections.Counter()

    for row in rows:
        inc(cases_by_file, row["file"])
        inc(cases_by_profile, row["profile"])
        inc(cases_by_active, row["active_n"])
        if row["s4pf_count"]:
            inc(cases_with_by_file, row["file"])
            inc(cases_with_by_profile, row["profile"])
        inc(by_file, row["file"], row["s4pf_count"])
        inc(by_profile, row["profile"], row["s4pf_count"])
        inc(by_active, row["active_n"], row["s4pf_count"])
        inc(by_m_total, row["m_total"], row["s4pf_count"])

    for ev in current_events:
        by_eid[str(ev["eid"])] += 1

    top_cases = sorted(rows, key=lambda r: r["s4pf_count"], reverse=True)[:30]
    report = {
        "sample_per_file": args.sample_per_file,
        "runtime_s": time.perf_counter() - t0,
        "summary": {
            "cases": total_cases,
            "s4pf_events": total_events,
            "cases_with_s4pf": cases_with,
            "case_fraction_with_s4pf": cases_with / total_cases if total_cases else 0.0,
            "events_per_case_mean": total_events / total_cases if total_cases else 0.0,
            "events_per_s4pf_case_mean": total_events / cases_with if cases_with else 0.0,
            "max_events_per_case": max((r["s4pf_count"] for r in rows), default=0),
        },
        "case_s4pf_count_hist": dict(sorted(count_hist.items())),
        "events_by_file": dict(sorted(by_file.items())),
        "cases_by_file": dict(sorted(cases_by_file.items())),
        "cases_with_s4pf_by_file": dict(sorted(cases_with_by_file.items())),
        "events_by_profile": dict(sorted(by_profile.items())),
        "cases_by_profile": dict(sorted(cases_by_profile.items())),
        "cases_with_s4pf_by_profile": dict(sorted(cases_with_by_profile.items())),
        "events_by_active_n": dict(sorted(by_active.items(), key=lambda kv: int(kv[0]))),
        "cases_by_active_n": dict(sorted(cases_by_active.items(), key=lambda kv: int(kv[0]))),
        "events_by_m_total": dict(sorted(by_m_total.items(), key=lambda kv: int(kv[0]))),
        "events_by_eid": dict(sorted(by_eid.items(), key=lambda kv: int(kv[0]))),
        "top_cases": top_cases,
    }
    args.out.write_text(json.dumps(report, indent=2))

    s = report["summary"]
    print(
        f"cases={s['cases']} s4pf_events={s['s4pf_events']} "
        f"cases_with_s4pf={s['cases_with_s4pf']} "
        f"case_fraction={s['case_fraction_with_s4pf']:.6f}"
    )
    print(
        f"events_per_case_mean={s['events_per_case_mean']:.6f} "
        f"events_per_s4pf_case_mean={s['events_per_s4pf_case_mean']:.6f} "
        f"max_events_per_case={s['max_events_per_case']}"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
