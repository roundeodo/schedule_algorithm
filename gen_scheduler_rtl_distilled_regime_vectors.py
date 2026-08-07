#!/usr/bin/env python3
"""Emit proof65 round-state regime classification vectors."""

from pathlib import Path

import scheduler_rtl_distilled_lowering as lowering
import scheduler_rtl_distilled_policy as policy
import scheduler_rtl_distilled_scoring as scoring
import verify_scheduler_rtl_unified_policy as datasets
from scheduler_rtl_distilled_types import TICK_CC


HERE = Path(__file__).resolve().parent
PROOF = HERE / "results/policy_search/olmoe_top2_projection_65_optimal_v1.json"
MODE = {"TERMINAL": 0, "SYNC": 1, "ONE_IDLE": 2}


def tick(value: int) -> int:
    if int(value) % TICK_CC:
        raise AssertionError(f"non-tick time {value}")
    return int(value) // TICK_CC


def main() -> int:
    rows = []
    for job in datasets._proof_jobs(PROOF.resolve()):
        state = policy._initial_state(job["distribution"], job["c2"], job["c3"])
        while state.remaining:
            counters = scoring.remaining_counters(state.remaining)
            top = list(state.remaining[:5])
            top += [(-1, 0)] * (5 - len(top))
            regime = scoring.classify_regime(state)
            rows.append((
                MODE[lowering.mode(state)],
                (
                    counters.count, counters.token_sum, counters.odd_count,
                    counters.block_sum, *counters.small_block_hist,
                ),
                top,
                tick(state.c2.task_end),
                tick(state.c3.task_end),
                (
                    int(regime.low_work_progress),
                    int(regime.sparse_hot_sync),
                    int(regime.mid_plateau),
                    int(regime.short_tail_plateau),
                    int(regime.large_slack_fill),
                ),
            ))
            _action, state, *_rest = policy._choose_one_round(
                state, enable_s4pf=True
            )
    print(len(rows))
    for mode, counters, top, c2_end, c3_end, regime in rows:
        print(mode, *counters, c2_end, c3_end, *regime)
        for eid, ntok in top:
            print(int(eid >= 0), max(0, int(eid)), int(ntok))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
