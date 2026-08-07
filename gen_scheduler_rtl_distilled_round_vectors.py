#!/usr/bin/env python3
"""Emit proof65 full-round vectors for the distilled RTL engine."""

from pathlib import Path

import scheduler_rtl_distilled_lowering as lowering
import scheduler_rtl_distilled_policy as policy
import scheduler_rtl_distilled_scoring as scoring
import verify_scheduler_rtl_unified_policy as datasets
from scheduler_rtl_distilled_profiles import COMPILED_PROFILES

from gen_scheduler_rtl_distilled_bound_vectors import counter_fields
from gen_scheduler_rtl_distilled_compare_vectors import score_record
from gen_scheduler_rtl_distilled_transition_vectors import (
    MODE,
    SLOTS_BY_MODE,
    plan_fields,
    state_fields,
    t,
)


HERE = Path(__file__).resolve().parent
PROOF = HERE / "results/policy_search/olmoe_top2_projection_65_optimal_v1.json"


def build_row(before, action, child, candidate_set, selected_slot):
    selected = candidate_set.slots[selected_slot]
    profile_slot = int(selected.physical_profile_slot)
    token = COMPILED_PROFILES[profile_slot]
    mode = lowering.mode(before)
    mode_index = SLOTS_BY_MODE[mode].index(profile_slot)
    selected_eids = [
        lowering.resolve_selector(before, selector)
        for selector in token.logical.selectors
    ]
    swap = int(
        token.logical.family == "PAIR" and action.c2_eid == selected_eids[1]
    )
    start = action.c2_start if action.c2_eid >= 0 else action.c3_start
    remove = [int(selected_eids[0])]
    if token.logical.family == "PAIR":
        remove.append(int(selected_eids[1]))
    else:
        remove.append(remove[0])
    hot = list(before.remaining[:8])
    hot += [(-1, 0)] * (8 - len(hot))
    bottom = before.remaining[-1]
    s4pf = {2: 0, 3: 0}
    for prefetch in selected.s4pf_actions:
        s4pf[int(prefetch.pf_cluster)] = int(prefetch.pf_dma)
    return (
        state_fields(before.c2),
        state_fields(before.c3),
        counter_fields(before, int(before.f_score)),
        hot + [bottom],
        (
            profile_slot,
            mode_index,
            policy.LOGICAL_ACTION_PRIORITY[(
                token.logical.mode,
                token.logical.family,
                token.logical.selectors,
                token.logical.split_rule,
            )],
            swap,
            t(start),
            s4pf[2],
            s4pf[3],
        ),
        state_fields(child.c2),
        state_fields(child.c3),
        counter_fields(child, int(child.f_score)),
        plan_fields(action),
        (2 if token.logical.family == "PAIR" else 1,
         remove[0], remove[1]),
        score_record(before, action, child),
    )


def emit_rows(rows) -> None:
    print(len(rows))
    for row in rows:
        c2, c3, counters, descriptors, selected, cc2, cc3, child_counters, plan, remove, score = row
        print(*c2)
        print(*c3)
        print(*counters)
        for eid, ntok in descriptors:
            print(int(eid >= 0), max(0, int(eid)), int(ntok))
        print(*selected)
        print(*cc2)
        print(*cc3)
        print(*child_counters)
        print(*plan)
        print(*remove)
        print(*score)


def main() -> int:
    rows = []
    for job in datasets._proof_jobs(PROOF.resolve()):
        state = policy._initial_state(job["distribution"], job["c2"], job["c3"])
        while state.remaining:
            before = state
            action, child, _score, candidate_set, selected_slot = policy._choose_one_round(
                before, enable_s4pf=True
            )
            if candidate_set.slots[selected_slot].s4pf_actions:
                raise AssertionError("proof65 round vectors expect no S4PF")
            rows.append(build_row(before, action, child, candidate_set, selected_slot))
            state = child

    emit_rows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
