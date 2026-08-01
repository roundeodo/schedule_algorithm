#!/usr/bin/env python3
"""Directed OLMoE-style showcase with workload-encodable DMA bindings."""

from __future__ import annotations

from dataclasses import astuple
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path

import four_stage_scheduler as reference
from run_four_stage_reference import serialize_action
import scheduler_rtl_distilled_policy as distilled
import scheduler_rtl_unified_policy as frozen_v4


HERE = Path(__file__).resolve().parent
CASE_NAME = "olmoe_grid_hot8_12x_triple_a43_le2_42"
PROOF65 = HERE / "results/policy_search/olmoe_top2_projection_65_optimal_v1.json"
OUTPUT = HERE / "results/policy_search/scheduler_rtl_distilled_showcase.json"
TICK_CC = reference.SCHEDULE_TIME_QUANTUM_CC


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_binding(
    shape: reference.Shape,
    cached: bool,
    local: reference.DmaBinding,
) -> reference.DmaBinding:
    if cached:
        return reference.DmaBinding.NONE
    return reference.DmaBinding.BOTH if shape == reference.SHAPE_C else local


def _workload_encodable(action: reference.StageAction) -> bool:
    if action.pf_dma != reference.DmaBinding.NONE:
        return False
    for prefix, local in (
        ("c2", reference.DmaBinding.IDMA),
        ("c3", reference.DmaBinding.XDMA),
    ):
        if getattr(action, f"{prefix}_eid") < 0:
            continue
        if getattr(action, f"{prefix}_dma_s1") != _expected_binding(
            getattr(action, f"{prefix}_shape_s1"),
            getattr(action, f"{prefix}_s1_cached"),
            local,
        ):
            return False
        s2pf_dma = getattr(action, f"{prefix}_s2pf_dma")
        if s2pf_dma != reference.DmaBinding.NONE:
            if s2pf_dma != local:
                return False
            if getattr(action, f"{prefix}_dma_s3") != reference.DmaBinding.NONE:
                return False
        elif getattr(action, f"{prefix}_dma_s3") != _expected_binding(
            getattr(action, f"{prefix}_shape_s3"),
            getattr(action, f"{prefix}_s3_cached"),
            local,
        ):
            return False
    return True


def _local_physical_key(
    action: reference.StageAction,
    child: reference.BeamState,
) -> tuple:
    ends = (int(child.c2.task_end), int(child.c3.task_end))
    starts = tuple(
        int(start)
        for eid, start in (
            (action.c2_eid, action.c2_start),
            (action.c3_eid, action.c3_start),
        )
        if eid >= 0
    )
    s2pf = sum(
        binding != reference.DmaBinding.NONE
        for binding in (action.c2_s2pf_dma, action.c3_s2pf_dma)
    )
    return (
        max(ends),
        abs(ends[0] - ends[1]),
        sum(ends),
        max(starts, default=0),
        -s2pf,
        repr(astuple(action)),
    )


def _issue_queue(counts: list[int], mode: str) -> list[int]:
    descending = sorted(
        range(len(counts)), key=lambda eid: (-int(counts[eid]), eid)
    )
    if mode == "descending":
        return descending
    if mode == "ascending":
        return list(reversed(descending))
    if mode == "ends_inward":
        queue = []
        left, right = 0, len(descending) - 1
        while left <= right:
            queue.append(descending[left])
            left += 1
            if left <= right:
                queue.append(descending[right])
                right -= 1
        return queue
    raise ValueError(mode)


def _fixed_order(counts: list[int], mode: str) -> dict:
    distribution = {
        eid: int(ntok) for eid, ntok in enumerate(counts) if int(ntok) > 0
    }
    state = reference.FourStageScheduler(distribution)._initial_state()
    queue = _issue_queue(counts, mode)
    decisions = []
    while state.remaining:
        take = 2 if state.c2.task_end == state.c3.task_end and len(queue) >= 2 else 1
        selected = queue[:take]
        visible = tuple(
            sorted(
                ((eid, distribution[eid]) for eid in selected),
                key=lambda item: (-item[1], item[0]),
            )
        )
        eligible = []
        wanted = set(selected)
        for action in reference.gen_stage_actions(state.c2, state.c3, visible):
            assigned = tuple(
                eid for eid in (action.c2_eid, action.c3_eid) if eid >= 0
            )
            if take == 2 and (len(assigned) != 2 or set(assigned) != wanted):
                continue
            if take == 1 and (
                set(assigned) != wanted
                or (len(state.remaining) != 1 and len(assigned) != 1)
            ):
                continue
            if not _workload_encodable(action):
                continue
            eligible.append((action, reference.apply_action(state, action)))
        if not eligible:
            raise RuntimeError(f"{mode}: no workload-encodable action for {selected}")
        action, child = min(
            eligible, key=lambda pair: _local_physical_key(*pair)
        )
        decisions.append(
            {
                "selected_tokens": [distribution[eid] for eid in selected],
                "ends_ticks": [
                    str(Fraction(int(child.c2.task_end), TICK_CC)),
                    str(Fraction(int(child.c3.task_end), TICK_CC)),
                ],
                "action": serialize_action(action),
            }
        )
        queue = queue[take:]
        state = child
    replay = reference.validate_schedule_history(state.history, distribution)
    if replay != state.g_score:
        raise AssertionError("fixed-order replay mismatch")
    return {
        "makespan_cc": int(state.g_score),
        "makespan_ticks": str(Fraction(int(state.g_score), TICK_CC)),
        "decisions": decisions,
    }


def main() -> int:
    proof = json.loads(PROOF65.read_text(encoding="utf-8"))
    case = next(case for case in proof["cases"] if case["name"] == CASE_NAME)
    counts = [int(value) for value in case["counts"]]
    distribution = {eid: ntok for eid, ntok in enumerate(counts) if ntok > 0}
    target_ticks = Fraction(str(case["best_reference_ticks"]))
    target_cc = int(target_ticks * TICK_CC)
    fixed = {
        mode: _fixed_order(counts, mode)
        for mode in ("descending", "ascending", "ends_inward")
    }
    result = distilled.schedule(distribution)
    v4_result = frozen_v4.schedule(distribution)
    if result.makespan_cc != target_cc or v4_result.makespan_cc != target_cc:
        raise AssertionError("showcase scheduler misses certified optimum")
    payload = {
        "schema": "scheduler_rtl_distilled_showcase_v1",
        "manifest": {
            "proof65": str(PROOF65.resolve()),
            "proof65_sha256": _sha256(PROOF65),
            "sources": {
                path.name: _sha256(path)
                for path in (
                    Path(__file__).resolve(),
                    HERE / "scheduler_rtl_distilled_policy.py",
                    HERE / "scheduler_rtl_distilled_profiles.py",
                    HERE / "four_stage_scheduler.py",
                )
            },
        },
        "case": CASE_NAME,
        "distribution": counts,
        "assignment_total": sum(counts),
        "active_experts": len(counts),
        "conceptual_experts": 64,
        "certified_optimum_ticks": str(target_ticks),
        "distilled_ticks": str(Fraction(result.makespan_cc, TICK_CC)),
        "v4_ticks": str(Fraction(v4_result.makespan_cc, TICK_CC)),
        "fixed_orders": fixed,
        "time_reduction_vs_fixed_pct": {
            mode: (
                1.0 - result.makespan_cc / fixed_result["makespan_cc"]
            ) * 100.0
            for mode, fixed_result in fixed.items()
        },
        "speedup_vs_fixed": {
            mode: fixed_result["makespan_cc"] / result.makespan_cc
            for mode, fixed_result in fixed.items()
        },
        "complexity": {
            "physical_candidate_count_max": result.physical_candidate_count_max,
            "logical_candidate_count_max": result.logical_candidate_count_max,
        },
        "distilled_trace": [
            {
                "mode": step.mode,
                "physical_candidate_count": step.physical_candidate_count,
                "logical_candidate_count": step.logical_candidate_count,
                "selected_profile_slot": step.selected_profile_slot,
                "action": serialize_action(step.action),
            }
            for step in result.steps
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, OUTPUT)
    print(
        json.dumps(
            {
                "case": CASE_NAME,
                "optimum_ticks": str(target_ticks),
                "fixed_ticks": {
                    mode: row["makespan_ticks"] for mode, row in fixed.items()
                },
                "time_reduction_vs_fixed_pct": payload[
                    "time_reduction_vs_fixed_pct"
                ],
                "complexity": payload["complexity"],
            },
            indent=2,
        )
    )
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
