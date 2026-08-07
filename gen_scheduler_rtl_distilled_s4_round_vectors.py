#!/usr/bin/env python3
"""Emit selected random-corpus rounds that commit targeted S4PF."""

import json
from pathlib import Path

import scheduler_rtl_distilled_policy as policy
import verify_scheduler_rtl_unified_policy as datasets

from gen_scheduler_rtl_distilled_round_vectors import build_row, emit_rows


HERE = Path(__file__).resolve().parent
VALIDATION = HERE / "results/policy_search/bounded_top5_bottom1_fixed_lane_targeted_s4pf_random_validation.json"
COVERAGE = tuple(HERE / f"scheduler_strategy_coverage_E{experts}.json"
                 for experts in (8, 32, 64))


def main() -> int:
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    event_keys = {
        key for key, row in validation["rows"].items()
        if int(row["s4pf_events"]) > 0
    }
    jobs = {
        job["key"]: job
        for job in datasets._dataset_jobs(tuple(path.resolve() for path in COVERAGE))
        if job["key"] in event_keys
    }
    rows = []
    coverage = set()
    for key in sorted(jobs):
        job = jobs[key]
        state = policy._initial_state(job["distribution"], job["c2"], job["c3"])
        while state.remaining:
            before = state
            action, child, _score, candidate_set, selected_slot = policy._choose_one_round(
                before, enable_s4pf=True
            )
            selected = candidate_set.slots[selected_slot]
            if selected.s4pf_actions:
                rows.append(build_row(
                    before, action, child, candidate_set, selected_slot
                ))
                for prefetch in selected.s4pf_actions:
                    coverage.add((int(prefetch.pf_cluster), prefetch.pf_dma.name))
            state = child
        if len(rows) >= 48 and {
            (2, "IDMA"), (2, "BOTH"), (3, "XDMA"), (3, "BOTH")
        }.issubset(coverage):
            break
    if not rows:
        raise AssertionError("no targeted S4PF vectors collected")
    emit_rows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
