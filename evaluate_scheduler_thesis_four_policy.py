#!/usr/bin/env python3
"""Evaluate the four scheduler policies used by the thesis showcase.

The comparison deliberately separates expert-order decisions from physical
argument decisions:

``STATIC_DESC``
    One global descending expert queue.  Shape and DMA fields are fixed to
    S1=B, S3=B, C2=iDMA, C3=xDMA, with S2PF/S4PF disabled.

``DYNAMIC_DESC``
    The same global descending expert queue.  A local physical reducer chooses
    among every legal shape/DMA/S2PF realization, plus a targeted S4PF
    realization when it improves the concrete current transition.

``DYNAMIC_TWO_ENDED``
    C2 owns the hot end and C3 owns the cold end.  Each cluster immediately
    takes its next expert when it becomes free; there is no global slot
    barrier.  It uses exactly the same physical reducer as ``DYNAMIC_DESC``.

``FULL_SCHEDULER``
    The bounded distilled scheduler chooses the logical action, expert order,
    cluster mapping and physical realization with its continuation scorer.

The two fixed-order dynamic policies never use beam search, rollout or the
distilled continuation scorer.  They also never split an expert.  Therefore
the reported differences have a clean interpretation:

* DYNAMIC_DESC / STATIC_DESC measures dynamic physical-argument selection.
* FULL_SCHEDULER / DYNAMIC_DESC and / DYNAMIC_TWO_ENDED measure the additional
  benefit of dynamic logical scheduling and continuation-aware ordering.
"""

from __future__ import annotations

import argparse
from dataclasses import astuple
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import four_stage_scheduler as reference
from run_four_stage_reference import serialize_action
import scheduler_rtl_distilled_lowering as lowering
import scheduler_rtl_distilled_policy as distilled
from scheduler_rtl_distilled_types import (
    CandidateProfile,
    LogicalActionSpec,
    PhysicalProfile,
)


HERE = Path(__file__).resolve().parent
TICK_CC = reference.SCHEDULE_TIME_QUANTUM_CC
DEFAULT_OUTPUT = (
    HERE
    / "results/policy_search/scheduler_thesis_four_policy_showcases.json"
)
PROOF65 = HERE / "results/policy_search/olmoe_top2_projection_65_optimal_v1.json"

POLICY_STATIC_DESC = "STATIC_DESC"
POLICY_DYNAMIC_DESC = "DYNAMIC_DESC"
POLICY_DYNAMIC_TWO_ENDED = "DYNAMIC_TWO_ENDED"
POLICY_FULL = "FULL_SCHEDULER"

FIXED_SHAPE_S1 = reference.SHAPE_B
FIXED_SHAPE_S3 = reference.SHAPE_B
FIXED_DMA = {
    2: reference.DmaBinding.IDMA,
    3: reference.DmaBinding.XDMA,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ticks(cc: int) -> str:
    return str(Fraction(int(cc), TICK_CC))


def _normalize_counts(counts: Sequence[int]) -> list[int]:
    normalized = sorted(
        (int(value) for value in counts if int(value) > 0), reverse=True
    )
    if not normalized:
        raise ValueError("distribution must contain at least one active expert")
    if len(normalized) > 64:
        raise ValueError("showcase assumes at most 64 experts")
    return normalized


def _distribution(counts: Sequence[int]) -> dict[int, int]:
    return {eid: ntok for eid, ntok in enumerate(_normalize_counts(counts))}


def _remaining_queue(counts: Sequence[int]) -> list[int]:
    # Normalization assigns stable EIDs in descending load order.
    return list(range(len(_normalize_counts(counts))))


def _required_assignment(
    state: reference.BeamState,
    queue: Sequence[int],
    *,
    two_ended: bool,
) -> dict[int, int]:
    """Return the exact whole-expert assignment for the next issue event."""
    if not queue:
        return {}
    if len(queue) == 1:
        if state.c2.task_end <= state.c3.task_end:
            return {2: int(queue[0])}
        return {3: int(queue[0])}
    if state.c2.task_end == state.c3.task_end:
        return {2: int(queue[0]), 3: int(queue[-1] if two_ended else queue[1])}
    if state.c2.task_end < state.c3.task_end:
        return {2: int(queue[0])}
    return {3: int(queue[-1] if two_ended else queue[0])}


def _visible_selected(
    distribution: Mapping[int, int], required: Mapping[int, int]
) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            ((eid, int(distribution[eid])) for eid in required.values()),
            key=lambda item: (-item[1], item[0]),
        )
    )


def _matches_assignment(
    action: reference.StageAction, required: Mapping[int, int]
) -> bool:
    """Require exact cluster ownership and forbid SPLIT/extra assignments."""
    for cluster in (2, 3):
        expected = int(required.get(cluster, -1))
        if int(getattr(action, f"c{cluster}_eid")) != expected:
            return False
    return True


def _legal_consuming_actions(
    state: reference.BeamState,
    distribution: Mapping[int, int],
    required: Mapping[int, int],
) -> list[reference.StageAction]:
    visible = _visible_selected(distribution, required)
    return [
        action
        for action in reference.gen_stage_actions(state.c2, state.c3, visible)
        if _matches_assignment(action, required)
    ]


def _selector_for_eid(state: reference.BeamState, eid: int) -> str:
    rank = next(
        index
        for index, (candidate_eid, _ntok) in enumerate(state.remaining)
        if int(candidate_eid) == int(eid)
    )
    # STATIC_DESC only requests the next one or two descending entries.
    if rank > 1:
        raise AssertionError(f"static descending selected unexpected rank T{rank}")
    return f"T{rank}"


def _fixed_profile_token(
    state: reference.BeamState, required: Mapping[int, int]
) -> CandidateProfile:
    active = set(map(int, required))
    family = "PAIR" if len(required) == 2 else "SINGLE"
    selectors = tuple(
        sorted(_selector_for_eid(state, eid) for eid in required.values())
    )

    def shape(cluster: int, stage: int) -> str:
        if cluster not in active:
            return "NONE"
        return (FIXED_SHAPE_S1 if stage == 1 else FIXED_SHAPE_S3).name

    def dma(cluster: int) -> str:
        return FIXED_DMA[cluster].name if cluster in active else "NONE"

    return CandidateProfile(
        logical=LogicalActionSpec(
            mode=lowering.mode(state),
            family=family,
            selectors=selectors,
            split_rule="NONE",
        ),
        physical=PhysicalProfile(
            c2_s1=shape(2, 1),
            c2_s3=shape(2, 3),
            c3_s1=shape(3, 1),
            c3_s3=shape(3, 3),
            c2_dma_s1=dma(2),
            c2_dma_s3=dma(2),
            c2_s2pf="NONE",
            c3_dma_s1=dma(3),
            c3_dma_s3=dma(3),
            c3_s2pf="NONE",
            s4pf_dma="NONE",
            c2_s1_cached=False,
            c2_s3_cached=False,
            c3_s1_cached=False,
            c3_s3_cached=False,
        ),
    )


def _transition_key(
    action: reference.StageAction,
    child: reference.BeamState,
    s4pf_actions: tuple[reference.StageAction, ...],
) -> tuple:
    """One-step physical reducer shared by both dynamic fixed-order policies."""
    ends = (int(child.c2.task_end), int(child.c3.task_end))
    starts = tuple(
        int(start)
        for eid, start in (
            (action.c2_eid, action.c2_start),
            (action.c3_eid, action.c3_start),
        )
        if eid >= 0
    )
    s2pf_count = sum(
        binding != reference.DmaBinding.NONE
        for binding in (action.c2_s2pf_dma, action.c3_s2pf_dma)
    )
    return (
        max(ends),
        sum(ends),
        abs(ends[0] - ends[1]),
        max(starts, default=0),
        -len(s4pf_actions),
        -s2pf_count,
        repr(astuple(action)),
    )


def _choose_static_transition(
    state: reference.BeamState,
    required: Mapping[int, int],
) -> tuple[reference.StageAction, reference.BeamState, tuple[reference.StageAction, ...], int]:
    # Use direct profile lowering here instead of the reference action list.
    # That list intentionally deduplicates future-equivalent shapes (for
    # example A and B can have the same endpoint for a particular ntok), while
    # this baseline must retain the literal fixed argument words in its trace.
    candidates = []
    token = _fixed_profile_token(state, required)
    for action in lowering._materialize_one_profile(state, token):
        if not _matches_assignment(action, required):
            continue
        child = reference.apply_action(state, action)
        candidates.append((action, child, ()))
    if not candidates:
        raise RuntimeError("fixed B/B single-lane profile has no legal action")
    action, child, s4pf_actions = min(
        candidates, key=lambda item: _transition_key(*item)
    )
    return action, child, s4pf_actions, len(candidates)


def _choose_dynamic_transition(
    state: reference.BeamState,
    actions: Iterable[reference.StageAction],
) -> tuple[reference.StageAction, reference.BeamState, tuple[reference.StageAction, ...], int]:
    """Choose physical arguments only; do not inspect remaining future work."""
    candidates = []
    base_actions = list(actions)
    for action in base_actions:
        baseline_child = reference.apply_action(state, action)
        candidates.append((action, baseline_child, ()))
        targeted = lowering.materialize_targeted_s4pf_variant(state, action)
        if targeted is not None:
            candidates.append(targeted)
    if not candidates:
        raise RuntimeError("dynamic physical selector has no legal action")
    action, child, s4pf_actions = min(
        candidates, key=lambda item: _transition_key(*item)
    )
    return action, child, s4pf_actions, len(candidates)


def _update_queue(
    queue: Sequence[int], required: Mapping[int, int]
) -> list[int]:
    consumed = set(map(int, required.values()))
    return [eid for eid in queue if int(eid) not in consumed]


def _serialize_step(
    state: reference.BeamState,
    required: Mapping[int, int],
    action: reference.StageAction,
    child: reference.BeamState,
    s4pf_actions: tuple[reference.StageAction, ...],
    physical_candidate_count: int,
) -> dict:
    return {
        "start_state_ticks": {
            "C2": _ticks(state.c2.task_end),
            "C3": _ticks(state.c3.task_end),
        },
        "required_assignment": {
            f"C{cluster}": int(eid) for cluster, eid in required.items()
        },
        "physical_candidate_count": int(physical_candidate_count),
        "s4pf_actions": [serialize_action(item) for item in s4pf_actions],
        "action": serialize_action(action),
        "end_state_ticks": {
            "C2": _ticks(child.c2.task_end),
            "C3": _ticks(child.c3.task_end),
        },
    }


def _schedule_fixed_order(
    counts: Sequence[int], *, two_ended: bool, dynamic_physical: bool
) -> dict:
    normalized = _normalize_counts(counts)
    distribution = _distribution(normalized)
    state = reference.FourStageScheduler(distribution)._initial_state()
    queue = _remaining_queue(normalized)
    steps = []
    max_physical_candidates = 0
    while state.remaining:
        required = _required_assignment(state, queue, two_ended=two_ended)
        actions = _legal_consuming_actions(state, distribution, required)
        if not actions:
            raise RuntimeError(
                f"no legal whole-expert action for assignment {required}"
            )
        if dynamic_physical:
            action, child, s4pf_actions, candidate_count = (
                _choose_dynamic_transition(state, actions)
            )
        else:
            action, child, s4pf_actions, candidate_count = (
                _choose_static_transition(state, required)
            )
        max_physical_candidates = max(max_physical_candidates, candidate_count)
        steps.append(
            _serialize_step(
                state,
                required,
                action,
                child,
                s4pf_actions,
                candidate_count,
            )
        )
        queue = _update_queue(queue, required)
        state = child

    replay = reference.validate_schedule_history(state.history, distribution)
    if replay != state.g_score:
        raise AssertionError("fixed-order schedule failed explicit-DMA replay")
    if queue:
        raise AssertionError("fixed-order queue and reference state diverged")
    return {
        "makespan_cc": int(state.g_score),
        "makespan_ticks": _ticks(state.g_score),
        "physical_candidate_count_max": max_physical_candidates,
        "s2pf_event_count": sum(
            step["action"][f"c{cluster}_s2pf_dma"] != "NONE"
            for step in steps
            for cluster in (2, 3)
        ),
        "s4pf_event_count": sum(len(step["s4pf_actions"]) for step in steps),
        "steps": steps,
    }


def _serialize_full_result(result: distilled.ScheduleResult) -> dict:
    return {
        "makespan_cc": int(result.makespan_cc),
        "makespan_ticks": _ticks(result.makespan_cc),
        "physical_candidate_count_max": result.physical_candidate_count_max,
        "logical_candidate_count_max": result.logical_candidate_count_max,
        "s2pf_event_count": sum(
            binding != reference.DmaBinding.NONE
            for step in result.steps
            for binding in (
                step.action.c2_s2pf_dma,
                step.action.c3_s2pf_dma,
            )
        ),
        "s4pf_event_count": sum(len(step.s4pf_actions) for step in result.steps),
        "steps": [
            {
                "mode": step.mode,
                "candidate_slot": step.candidate_slot,
                "selected_profile_slot": step.selected_profile_slot,
                "physical_candidate_count": step.physical_candidate_count,
                "logical_candidate_count": step.logical_candidate_count,
                "score": list(step.score),
                "s4pf_actions": [
                    serialize_action(action) for action in step.s4pf_actions
                ],
                "action": serialize_action(step.action),
            }
            for step in result.steps
        ],
    }


def evaluate_distribution(counts: Sequence[int], *, name: str = "case") -> dict:
    normalized = _normalize_counts(counts)
    static_desc = _schedule_fixed_order(
        normalized, two_ended=False, dynamic_physical=False
    )
    dynamic_desc = _schedule_fixed_order(
        normalized, two_ended=False, dynamic_physical=True
    )
    dynamic_two_ended = _schedule_fixed_order(
        normalized, two_ended=True, dynamic_physical=True
    )
    full_result = distilled.schedule(_distribution(normalized))
    full = _serialize_full_result(full_result)
    policies = {
        POLICY_STATIC_DESC: static_desc,
        POLICY_DYNAMIC_DESC: dynamic_desc,
        POLICY_DYNAMIC_TWO_ENDED: dynamic_two_ended,
        POLICY_FULL: full,
    }
    full_cc = full["makespan_cc"]
    return {
        "name": name,
        "distribution": normalized,
        "assignment_total": sum(normalized),
        "input_tokens_top2": str(Fraction(sum(normalized), 2)),
        "active_experts": len(normalized),
        "conceptual_experts": 64,
        "policies": policies,
        "speedup": {
            "dynamic_params_vs_static_desc": (
                static_desc["makespan_cc"] / dynamic_desc["makespan_cc"]
            ),
            "full_vs_static_desc": static_desc["makespan_cc"] / full_cc,
            "full_vs_dynamic_desc": dynamic_desc["makespan_cc"] / full_cc,
            "full_vs_dynamic_two_ended": (
                dynamic_two_ended["makespan_cc"] / full_cc
            ),
        },
        "time_reduction_pct": {
            "dynamic_params_vs_static_desc": 100.0
            * (1.0 - dynamic_desc["makespan_cc"] / static_desc["makespan_cc"]),
            "full_vs_dynamic_desc": 100.0
            * (1.0 - full_cc / dynamic_desc["makespan_cc"]),
            "full_vs_dynamic_two_ended": 100.0
            * (1.0 - full_cc / dynamic_two_ended["makespan_cc"]),
        },
    }


def _load_default_specs() -> tuple[dict, ...]:
    proof_payload = json.loads(PROOF65.read_text(encoding="utf-8"))
    proof_by_name = {case["name"]: case for case in proof_payload["cases"]}
    proof_key = "olmoe_grid_hot8_12x_triple_a43_le2_42"
    proof_case = proof_by_name[proof_key]

    return (
        {
            "name": "certified_olmoe_triple_hot_long_cold_tail",
            "counts": [int(value) for value in proof_case["counts"]],
            "source": "65-case optimal certificate set",
            "source_key": proof_key,
            "characteristic": (
                "three local hotspots, medium experts, and a long <=2-token tail"
            ),
            "certified_optimum_ticks": str(proof_case["best_reference_ticks"]),
        },
        {
            "name": "synthetic_three_hot_medium_cold_m70",
            "counts": [28] * 3 + [6] * 4 + [2] * 16,
            "source": "structured synthetic family",
            "source_key": "M70-hot28x3-medium6x4-tail2x16",
            "characteristic": (
                "three 12.8x-average hotspots, four medium experts, and a cold tail"
            ),
        },
        {
            "name": "synthetic_parameter_and_order_stress_m92",
            "counts": [76, 40] + [2] * 32 + [1] * 4,
            "source": "structured synthetic family",
            "source_key": "M92-hot76-hot40-tail2x32-tail1x4",
            "characteristic": (
                "two skewed hotspots, a long one/two-assignment cold tail, "
                "38 active experts, and 26 inactive experts"
            ),
        },
        {
            "name": "synthetic_high_skew_olmoe_style_m60",
            "counts": [36, 22, 13, 6] + [2] * 17 + [1] * 9,
            "source": "structured synthetic family",
            "source_key": "M60-hot36-hot22-hot13-medium6-tail2x17-tail1x9",
            "characteristic": (
                "high-skew OLMoE-style profile with four loads above two, "
                "26 active cold experts, 30 active experts, and 60 "
                "conceptual experts at <=2"
            ),
        },
    )


def _audit_case(case: Mapping) -> None:
    policies = case["policies"]
    static = policies[POLICY_STATIC_DESC]
    for step in static["steps"]:
        if step["s4pf_actions"]:
            raise AssertionError("STATIC_DESC unexpectedly uses S4PF")
        action = step["action"]
        for cluster, lane in ((2, "IDMA"), (3, "XDMA")):
            if action[f"c{cluster}_eid"] < 0:
                continue
            if action[f"c{cluster}_shape_s1"] != FIXED_SHAPE_S1.name:
                raise AssertionError("STATIC_DESC changed S1 shape")
            if action[f"c{cluster}_shape_s3"] != FIXED_SHAPE_S3.name:
                raise AssertionError("STATIC_DESC changed S3 shape")
            if action[f"c{cluster}_dma_s1"] != lane:
                raise AssertionError("STATIC_DESC changed S1 lane")
            if action[f"c{cluster}_dma_s3"] != lane:
                raise AssertionError("STATIC_DESC changed S3 lane")
            if action[f"c{cluster}_s2pf_dma"] != "NONE":
                raise AssertionError("STATIC_DESC unexpectedly uses S2PF")

    for policy in (POLICY_DYNAMIC_DESC, POLICY_DYNAMIC_TWO_ENDED):
        consumed = []
        for step in policies[policy]["steps"]:
            action = step["action"]
            assigned = [
                action[f"c{cluster}_eid"]
                for cluster in (2, 3)
                if action[f"c{cluster}_eid"] >= 0
            ]
            if len(assigned) != len(set(assigned)):
                raise AssertionError(f"{policy} unexpectedly split an expert")
            consumed.extend(assigned)
        if sorted(consumed) != list(range(case["active_experts"])):
            raise AssertionError(f"{policy} did not consume every expert exactly once")

    speedup = case["speedup"]
    if case["name"] == "synthetic_parameter_and_order_stress_m92":
        if case["assignment_total"] != 184:
            raise AssertionError("M92 stress case changed assignment total")
        if case["active_experts"] != 38:
            raise AssertionError("M92 stress case changed active-expert count")
        if speedup["dynamic_params_vs_static_desc"] < 1.17:
            raise AssertionError("parameter-benefit showcase fell below 1.17x")
        if speedup["full_vs_dynamic_desc"] < 1.16:
            raise AssertionError("ordering-benefit showcase fell below 1.16x")
        if speedup["full_vs_dynamic_two_ended"] < 1.19:
            raise AssertionError("full versus two-ended fell below 1.19x")
    if case["name"] == "synthetic_three_hot_medium_cold_m70":
        if speedup["full_vs_dynamic_desc"] < 1.2:
            raise AssertionError("full versus descending fell below 1.2x")
        if speedup["full_vs_dynamic_two_ended"] < 1.2:
            raise AssertionError("full versus two-ended fell below 1.2x")
    if case["name"] == "synthetic_high_skew_olmoe_style_m60":
        if case["assignment_total"] != 120:
            raise AssertionError("M60 stress case changed assignment total")
        if case["active_experts"] != 30:
            raise AssertionError("M60 stress case changed active-expert count")
        conceptual_le2 = 64 - sum(
            int(value) > 2 for value in case["distribution"]
        )
        if conceptual_le2 != 60:
            raise AssertionError("M60 stress case changed <=2-expert count")
        if speedup["full_vs_dynamic_desc"] < 1.3:
            raise AssertionError("full versus descending fell below 1.3x")


def _parse_counts(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _write_payload(cases: Sequence[dict], output: Path) -> None:
    sources = (
        Path(__file__).resolve(),
        HERE / "four_stage_scheduler.py",
        HERE / "scheduler_rtl_distilled_policy.py",
        HERE / "scheduler_rtl_distilled_lowering.py",
        HERE / "scheduler_rtl_distilled_profiles.py",
        HERE / "scheduler_rtl_distilled_scoring.py",
        HERE / "scheduler_rtl_distilled_types.py",
        PROOF65,
    )
    payload = {
        "schema": "scheduler_thesis_four_policy_v1",
        "comparison_contract": {
            POLICY_STATIC_DESC: {
                "order": "global descending queue, refill on cluster availability",
                "shape_s1": FIXED_SHAPE_S1.name,
                "shape_s3": FIXED_SHAPE_S3.name,
                "C2_DMA": "IDMA",
                "C3_DMA": "XDMA",
                "S2PF": False,
                "S4PF": False,
                "split": False,
            },
            "shared_dynamic_physical_selector": {
                "used_by": [POLICY_DYNAMIC_DESC, POLICY_DYNAMIC_TWO_ENDED],
                "physical_space": "all legal shape/DMA/S2PF plus targeted S4PF",
                "objective": [
                    "current latest task end",
                    "current sum of task ends",
                    "current task-end imbalance",
                    "latest selected start",
                ],
                "future_lookahead": False,
                "beam_search": False,
                "split": False,
            },
            POLICY_DYNAMIC_DESC: {
                "order": "global descending queue, refill on cluster availability",
            },
            POLICY_DYNAMIC_TWO_ENDED: {
                "C2": "hottest remaining expert",
                "C3": "coldest remaining expert",
                "refill": "independent and immediate when owning cluster is free",
                "global_slot_barrier": False,
            },
            POLICY_FULL: {
                "implementation": "scheduler_rtl_distilled_policy.schedule",
                "logical_decisions": [
                    "PAIR/SINGLE/SPLIT",
                    "expert selection and order",
                    "cluster mapping",
                ],
                "physical_decisions": "compiled physical profiles with local reduction",
                "continuation_scorer": distilled.CONTINUATION_SCORER,
            },
        },
        "selection_contract": {
            "primary_requirements": [
                "show dynamic physical parameters matter",
                "show dynamic logical scheduling and order matter",
                "FULL_SCHEDULER is at least 1.2x faster than DYNAMIC_DESC or DYNAMIC_TWO_ENDED in each selected ordering showcase",
            ],
            "observed_max_speedup_vs_strong_dynamic_baseline": (
                "1.210x in the selected cases; no reproducible 1.3x-1.7x "
                "claim against dynamic two-ended for a distributed OLMoE-style case"
            ),
            "selection_policy": (
                "include certified, multi-hot medium/cold, parameter-stress, "
                "and 60-token OLMoE-style high-skew multi-hot/cold-tail evidence"
            ),
        },
        "sources": {path.name: _sha256(path) for path in sources},
        "cases": list(cases),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts",
        help="one comma-separated distribution; otherwise run built-in smoke cases",
    )
    parser.add_argument("--name", default="custom")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.counts:
        specs = ({"name": args.name, "counts": _parse_counts(args.counts)},)
    else:
        specs = _load_default_specs()
    cases = []
    for spec in specs:
        case = evaluate_distribution(spec["counts"], name=spec["name"])
        case.update({key: value for key, value in spec.items() if key != "counts"})
        if "certified_optimum_ticks" in spec:
            if (
                case["policies"][POLICY_FULL]["makespan_ticks"]
                != spec["certified_optimum_ticks"]
            ):
                raise AssertionError("full scheduler missed certified optimum")
        _audit_case(case)
        cases.append(case)
    _write_payload(cases, args.output)
    summary = {
        case["name"]: {
            "distribution": case["distribution"],
            "ticks": {
                policy: row["makespan_ticks"]
                for policy, row in case["policies"].items()
            },
            "speedup": case["speedup"],
        }
        for case in cases
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
