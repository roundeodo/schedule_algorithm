#!/usr/bin/env python3
"""Emit proof65 selected-transition lockstep vectors for the RTL evaluator."""

from pathlib import Path

import four_stage_scheduler as reference
import scheduler_rtl_distilled_policy as policy
from scheduler_rtl_distilled_profiles import COMPILED_PROFILES
from scheduler_rtl_distilled_types import LOGICAL_ACTION_PRIORITY
import scheduler_rtl_distilled_lowering as lowering
import verify_scheduler_rtl_unified_policy as datasets


HERE = Path(__file__).resolve().parent
PROOF = HERE / "results/policy_search/olmoe_top2_projection_65_optimal_v1.json"
TICK = reference.SCHEDULE_TIME_QUANTUM_CC
MODE = {"TERMINAL": 0, "SYNC": 1, "ONE_IDLE": 2}
SHAPE = {reference.SHAPE_A.name: 0, reference.SHAPE_B.name: 1, reference.SHAPE_C.name: 2}
def logical_id(token) -> int:
    logical = token.logical
    return LOGICAL_ACTION_PRIORITY[(
        logical.mode, logical.family, logical.selectors, logical.split_rule,
    )]


SLOTS_BY_MODE = {
    mode: sorted(
        (slot for slot, token in enumerate(COMPILED_PROFILES)
         if token.logical.mode == mode),
        key=lambda slot: (logical_id(COMPILED_PROFILES[slot]), slot),
    )
    for mode in MODE
}


def t(value: int) -> int:
    if value % TICK:
        raise AssertionError(f"non-tick time {value}")
    return value // TICK


def state_fields(snap: reference.FourStageSnap) -> tuple[int, ...]:
    cache_valid = int(snap.pf_eid >= 0)
    return (
        int(snap.cur_eid >= 0), max(0, int(snap.cur_eid)),
        t(snap.task_start), t(snap.task_end), t(snap.dma1_end),
        t(snap.s2_end), t(snap.dma3_end),
        t(snap.compute_end if snap.compute_end >= 0 else snap.task_end),
        int(snap.dma_s1), int(snap.dma_s3), int(snap.s2pf_dma),
        t(snap.s2pf_end) if snap.s2pf_end >= 0 else 0,
        cache_valid, max(0, int(snap.pf_eid)),
        t(snap.pf_end) if snap.pf_end >= 0 else 0, int(snap.pf_full),
    )


def plan_fields(action: reference.StageAction) -> tuple[int, ...]:
    tasks = []
    for cluster in (2, 3):
        eid = int(getattr(action, f"c{cluster}_eid"))
        if eid < 0:
            continue
        dma_s1 = getattr(action, f"c{cluster}_dma_s1")
        dma_s3 = getattr(action, f"c{cluster}_dma_s3")
        s2pf = getattr(action, f"c{cluster}_s2pf_dma")
        tasks.append((
            eid, int(getattr(action, f"c{cluster}_ntok")),
            int(getattr(action, f"c{cluster}_ntok")) * 0 +
            (0 if cluster == 2 else 0),
            cluster - 2,
            SHAPE[getattr(action, f"c{cluster}_shape_s1").name],
            SHAPE[getattr(action, f"c{cluster}_shape_s3").name],
            int(getattr(action, f"c{cluster}_s1_cached")),
            int(getattr(action, f"c{cluster}_s3_cached")),
            int(s2pf != reference.DmaBinding.NONE),
            int(dma_s1 == reference.DmaBinding.BOTH),
            int((s2pf if s2pf != reference.DmaBinding.NONE else dma_s3)
                == reference.DmaBinding.BOTH),
        ))
    # Split C3 starts at the C2 token count; all other token starts are zero.
    if len(tasks) == 2 and tasks[0][0] == tasks[1][0]:
        tasks[1] = tasks[1][:2] + (tasks[0][1],) + tasks[1][3:]
    while len(tasks) < 2:
        tasks.append((0,) * 11)
    valid_mask = 3 if len([x for x in (action.c2_eid, action.c3_eid) if x >= 0]) == 2 else 1
    return (valid_mask, *tasks[0], *tasks[1])


def main() -> int:
    jobs = datasets._proof_jobs(PROOF.resolve())
    print(len(jobs))
    for case_id, job in enumerate(jobs):
        state = policy._initial_state(job["distribution"], job["c2"], job["c3"])
        rows = []
        while state.remaining:
            action, child, _score, candidate_set, selected_slot = policy._choose_one_round(
                state, enable_s4pf=True
            )
            selected = candidate_set.slots[selected_slot]
            if selected.s4pf_actions:
                raise AssertionError("proof65 transition vectors expect no S4PF")
            profile_slot = int(selected.physical_profile_slot)
            token = COMPILED_PROFILES[profile_slot]
            logical = token.logical
            mode = lowering.mode(state)
            mode_index = SLOTS_BY_MODE[mode].index(profile_slot)
            visible_top = list(state.remaining[:5])
            visible_top += [(-1, 0)] * (5 - len(visible_top))
            bottom = state.remaining[-1]
            selected_eids = [
                lowering.resolve_selector(state, selector)
                for selector in logical.selectors
            ]
            swap = int(
                logical.family == "PAIR" and action.c2_eid == selected_eids[1]
            )
            start = action.c2_start if action.c2_eid >= 0 else action.c3_start
            remove_eids = [int(selected_eids[0])]
            if logical.family == "PAIR":
                remove_eids.append(int(selected_eids[1]))
            else:
                remove_eids.append(remove_eids[0])
            selected_tokens = [
                ntok for eid, ntok in (
                    (action.c2_eid, action.c2_ntok),
                    (action.c3_eid, action.c3_ntok),
                ) if eid >= 0
            ]
            s2pf_count = sum(binding != reference.DmaBinding.NONE for binding in (
                action.c2_s2pf_dma, action.c3_s2pf_dma
            ))
            rows.append((
                (MODE[mode], mode_index, swap, t(start), 0, 0,
                 2 if logical.family == "PAIR" else 1,
                 remove_eids[0], remove_eids[1],
                 max(selected_tokens), sum(selected_tokens), s2pf_count, t(start)),
                visible_top + [bottom],
                state_fields(child.c2), state_fields(child.c3), plan_fields(action),
            ))
            state = child
        print(case_id, len(rows), int(job["c2"]), int(job["c3"]))
        for header, descriptors, c2, c3, plan in rows:
            print(*header)
            for eid, ntok in descriptors:
                print(int(eid >= 0), max(0, int(eid)), int(ntok))
            print(*c2)
            print(*c3)
            print(*plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
