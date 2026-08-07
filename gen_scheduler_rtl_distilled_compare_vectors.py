#!/usr/bin/env python3
"""Emit every proof65 pairwise continuation fold for RTL lockstep."""

from pathlib import Path

import scheduler_rtl_distilled_lowering as lowering
import scheduler_rtl_distilled_policy as policy
import scheduler_rtl_distilled_scoring as scoring
import verify_scheduler_rtl_unified_policy as datasets
from scheduler_rtl_distilled_types import TICK_CC, WINDOW


HERE = Path(__file__).resolve().parent
PROOF = HERE / "results/policy_search/olmoe_top2_projection_65_optimal_v1.json"
MODE = {"TERMINAL": 0, "SYNC": 1, "ONE_IDLE": 2}


def tick(value: int) -> int:
    value = int(value)
    if value % TICK_CC:
        raise AssertionError(f"non-tick time {value}")
    return value // TICK_CC


def half_tick(value: int) -> int:
    value = int(value) * 2
    if value % TICK_CC:
        raise AssertionError(f"non-half-tick time {value}")
    return value // TICK_CC


def ranks(state) -> dict[int, int]:
    count = len(state.remaining)
    top = state.remaining[: min(WINDOW[0], count)]
    bottom = state.remaining[max(0, count - WINDOW[1]) :]
    result = {int(eid): rank for rank, (eid, _ntok) in enumerate(top)}
    for offset, (eid, _ntok) in enumerate(reversed(bottom)):
        result.setdefault(int(eid), count - 1 - offset)
    return result


def score_record(state, action, child) -> tuple[int, ...]:
    normalized = scoring.normalize_state_bound(
        child, parent_bound=int(state.f_score)
    )
    components = scoring.lower_bound_components(normalized)
    maximum, _minimum, selected_sum, s2pf = lowering.selected_action_features(
        action
    )
    ends = sorted((int(child.c2.task_end), int(child.c3.task_end)))
    rank_map = ranks(state)
    selected_ranks = {
        rank_map[int(eid)]
        for eid in (action.c2_eid, action.c3_eid)
        if int(eid) >= 0
    }
    if not selected_ranks:
        raise AssertionError("candidate selects no visible expert")
    return (
        half_tick(normalized.f_score),
        half_tick(scoring.head5_hist4_lpt(normalized)),
        half_tick(components["compute_cc"]),
        half_tick(components["dma_capacity_cc"]),
        tick(ends[0]),
        tick(ends[1]),
        tick(child.g_score),
        int(maximum),
        int(selected_sum),
        int(s2pf),
        len(child.remaining),
        min(selected_ranks),
        max(selected_ranks),
        int(0 in selected_ranks),
    )


def main() -> int:
    rows = []
    for job in datasets._proof_jobs(PROOF.resolve()):
        state = policy._initial_state(job["distribution"], job["c2"], job["c3"])
        while state.remaining:
            candidate_set = policy._materialize_candidate_set(state, enable_s4pf=True)
            candidates = [(slot.action, slot.child) for slot in candidate_set.slots]
            incumbent = candidates[0]
            regime = scoring.classify_regime(state)
            count = len(state.remaining)
            min_load = int(state.remaining[-1][1]) if state.remaining else 0
            for candidate in candidates[1:]:
                if lowering.child_key(incumbent[1]) == lowering.child_key(candidate[1]):
                    raise AssertionError("global candidate stream was not deduplicated")
                _score, winner, _action, _child, metadata = (
                    scoring.select_continuation_transition_winner(
                        state, [incumbent, candidate]
                    )
                )
                rows.append((
                    MODE[lowering.mode(state)],
                    (
                        int(regime.low_work_progress),
                        int(regime.sparse_hot_sync),
                        int(regime.mid_plateau),
                        int(regime.short_tail_plateau),
                        int(regime.large_slack_fill),
                    ),
                    count,
                    min_load,
                    score_record(state, *incumbent),
                    score_record(state, *candidate),
                    int(winner == 1),
                    int(metadata["pairwise_overrides"] != 0),
                ))
                if winner == 1:
                    incumbent = candidate
            _action, state, *_rest = policy._choose_one_round(
                state, enable_s4pf=True
            )

    print(len(rows))
    for mode, regime, count, min_load, lhs, rhs, winner, override in rows:
        print(mode, *regime, count, min_load, 1, winner, override)
        print(*lhs)
        print(*rhs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
