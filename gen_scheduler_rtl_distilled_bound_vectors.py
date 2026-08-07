#!/usr/bin/env python3
"""Emit F/H/C/D vectors for every proof65 logical candidate."""

from pathlib import Path

import four_stage_scheduler as reference
import scheduler_rtl_distilled_policy as policy
import scheduler_rtl_distilled_scoring as scoring
import verify_scheduler_rtl_unified_policy as datasets

from gen_scheduler_rtl_distilled_transition_vectors import state_fields
from scheduler_rtl_distilled_types import TICK_CC


HERE = Path(__file__).resolve().parent
PROOF = HERE / "results/policy_search/olmoe_top2_projection_65_optimal_v1.json"


def ht(value: int) -> int:
    scaled = int(value) * 2
    if scaled % TICK_CC:
        raise AssertionError(f"non-half-tick time {value}")
    return scaled // TICK_CC


def counter_fields(state: reference.BeamState, parent_bound: int) -> tuple[int, ...]:
    counters = scoring.remaining_counters(state.remaining)
    return (
        counters.count, counters.token_sum, counters.odd_count,
        counters.block_sum, *counters.small_block_hist, ht(parent_bound),
    )


def main() -> int:
    rows = []
    for job in datasets._proof_jobs(PROOF.resolve()):
        state = policy._initial_state(job["distribution"], job["c2"], job["c3"])
        while state.remaining:
            candidate_set = policy._materialize_candidate_set(state, enable_s4pf=True)
            for slot in candidate_set.slots:
                child = slot.child
                normalized = scoring.normalize_state_bound(
                    child, parent_bound=int(state.f_score)
                )
                components = scoring.lower_bound_components(normalized)
                head = list(child.remaining[:5])
                head += [(-1, 0)] * (5 - len(head))
                rows.append((
                    state_fields(child.c2), state_fields(child.c3),
                    counter_fields(child, int(state.f_score)), head,
                    (
                        ht(normalized.f_score), ht(scoring.head5_hist4_lpt(normalized)),
                        ht(components["compute_cc"]),
                        ht(components["dma_capacity_cc"]),
                    ),
                ))
            _action, state, *_rest = policy._choose_one_round(
                state, enable_s4pf=True
            )
    print(len(rows))
    for c2, c3, counters, head, expected in rows:
        print(*c2)
        print(*c3)
        print(*counters)
        for eid, ntok in head:
            print(int(eid >= 0), max(0, int(eid)), int(ntok))
        print(*expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
