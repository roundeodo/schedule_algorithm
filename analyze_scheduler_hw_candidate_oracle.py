#!/usr/bin/env python3
"""Separate fixed-candidate selection loss from candidate-space loss.

The search never calls the four-stage action generator.  It follows only the
transitions exposed by ``scheduler_hw_fixed_policy.py``.  Therefore every
reported oracle makespan is achievable by the deployed fixed candidate bank.
If beam pruning occurs it is a feasible upper bound, not a proof of the true
candidate-space optimum; zero pruning certifies the candidate-space optimum.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import time

import analyze_scheduler_hw_scorers as scorers
import eval_hw_mirror_s2pf_lite as old_hw
import four_stage_scheduler as reference
import prove_top4_bottom2_directed as top4_proof
from run_four_stage_reference import deserialize_action, serialize_action
import scheduler_hw_fixed_policy as fixed
from scheduler_rtl_prefetch_both_policy import TICK_CC


ROOT = Path(__file__).resolve().parent
DEFAULT_SUITES = tuple(ROOT / f"scheduler_strategy_coverage_E{e}.json" for e in (8, 32, 64))
DEFAULT_REFERENCES = tuple(
    ROOT / "results" / "final_reference" / f"scheduler_reference_E{e}.json"
    for e in (8, 32, 64)
)
DEFAULT_OUT = ROOT / "results" / "policy_search" / "scheduler_hw_candidate_oracle.json"
HW_CONFIG = {"policy": "balanced", "top_policy": "pruned", "n1_policy": "pruned"}


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def active_bucket(active_n: int) -> str:
    if active_n <= 4:
        return "01_04"
    if active_n <= 8:
        return "05_08"
    if active_n <= 16:
        return "09_16"
    if active_n <= 32:
        return "17_32"
    return "33_64"


def stratified_pick(rows: list[dict], per_stratum: int) -> list[dict]:
    if per_stratum < 0:
        return rows
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["e_total"], row["dataset_split"], active_bucket(row["active_n"]))].append(row)
    selected = []
    for key in sorted(groups):
        values = sorted(groups[key], key=lambda row: row["case_id"])
        if len(values) <= per_stratum:
            selected.extend(values)
        elif per_stratum > 0:
            indices = sorted(
                set(round(i * (len(values) - 1) / (per_stratum - 1)) for i in range(per_stratum))
            ) if per_stratum > 1 else [len(values) // 2]
            selected.extend(values[index] for index in indices)
    return sorted(selected, key=lambda row: (row["e_total"], row["case_id"]))


def load_cases(
    suites: tuple[Path, ...], references: tuple[Path, ...], splits: set[str], proven_only: bool
) -> list[dict]:
    reference_by_key = {}
    for path in references:
        e_total = int(path.stem.rsplit("E", 1)[1])
        for raw_key, result in json.loads(path.read_text())["results"].items():
            reference_by_key[(e_total, int(raw_key))] = result
    rows = []
    for path in suites:
        for case in json.loads(path.read_text())["cases"]:
            key = (int(case["e_total"]), int(case["case_id"]))
            reference = reference_by_key[key]
            if not case.get("analysis_eligible", False):
                continue
            if case.get("dataset_split") not in splits:
                continue
            if proven_only and not reference.get("proven_optimal", False):
                continue
            rows.append(
                {
                    "key": f"E{key[0]}:{key[1]}",
                    "e_total": key[0],
                    "case_id": key[1],
                    "dataset_split": case.get("dataset_split"),
                    "active_n": int(case.get("active_n", len(case["dist"]))),
                    "construction": case.get("construction"),
                    "cache_regime": case.get("cache_regime"),
                    "dist": {int(eid): int(ntok) for eid, ntok in case["dist"].items()},
                    "c2": int(case.get("c2", -1)),
                    "c3": int(case.get("c3", -1)),
                    "reference_cc": int(reference["makespan_cc"]),
                    "reference_proven": bool(reference.get("proven_optimal", False)),
                }
            )
    return sorted(rows, key=lambda row: (row["e_total"], row["case_id"]))


def _parse_ticks(value: str | int) -> Fraction:
    return Fraction(str(value))


def _ticks_text(cc: int) -> str:
    value = Fraction(cc, TICK_CC)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def load_directed_cases(pairs: list[tuple[Path, Path]]) -> list[dict]:
    """Load characteristic-oriented suites and their four-stage pass results."""
    rows = []
    for pair_index, (suite_path, reference_path) in enumerate(pairs):
        suite = json.loads(suite_path.read_text())
        reference_payload = json.loads(reference_path.read_text())
        if not reference_payload.get("complete", False):
            raise ValueError(f"directed reference is incomplete: {reference_path}")
        reference_by_name = {row["name"]: row for row in reference_payload["cases"]}
        if len(reference_by_name) != len(reference_payload["cases"]):
            raise ValueError(f"directed reference has duplicate names: {reference_path}")
        suite_names = [case["name"] for case in suite["cases"]]
        if len(set(suite_names)) != len(suite_names):
            raise ValueError(f"directed suite has duplicate names: {suite_path}")
        if set(suite_names) != set(reference_by_name):
            missing = sorted(set(suite_names) - set(reference_by_name))
            extra = sorted(set(reference_by_name) - set(suite_names))
            raise ValueError(
                f"suite/reference case mismatch: missing={missing}, extra={extra}"
            )
        group = "min2" if "min2" in suite_path.stem else "base"
        for case_index, case in enumerate(suite["cases"]):
            name = case["name"]
            if name not in reference_by_name:
                raise KeyError(f"{name}: missing from {reference_path}")
            reference = reference_by_name[name]
            counts = [int(value) for value in case["active_counts"]]
            if counts != [int(value) for value in reference["counts"]]:
                raise ValueError(f"{name}: suite/reference distribution mismatch")
            if not reference.get("history_replay_valid", False):
                raise ValueError(f"{name}: reference history is not replay-validated")
            reference_cc = _parse_ticks(reference["best_reference_ticks"]) * TICK_CC
            lower_bound_cc = _parse_ticks(reference["certified_lower_bound_ticks"]) * TICK_CC
            top4_bottom2_cc = _parse_ticks(reference["mirror_policy_ticks"]) * TICK_CC
            for label, value in (
                ("reference", reference_cc),
                ("lower bound", lower_bound_cc),
                ("stored top4+bottom2 mirror", top4_bottom2_cc),
            ):
                if value.denominator != 1:
                    raise ValueError(f"{name}: non-integral {label} CC value {value}")
            rows.append(
                {
                    "key": name,
                    "name": name,
                    "group": group,
                    "pair_index": pair_index,
                    "case_id": case_index,
                    "e_total": int(case.get("metrics", {}).get("total_experts", 64)),
                    "dataset_split": "directed",
                    "active_n": len(counts),
                    "family": case.get("family"),
                    "profile": case.get("profile"),
                    "suite_role": case.get("suite_role"),
                    "metrics": case.get("metrics", {}),
                    "dist": dict(enumerate(counts)),
                    "c2": -1,
                    "c3": -1,
                    "reference_cc": int(reference_cc),
                    "reference_lower_bound_cc": int(lower_bound_cc),
                    "reference_proven": bool(reference.get("proven_optimal", False)),
                    "reference_termination": reference.get("termination"),
                    "reference_actions": reference.get("actions", []),
                    # This value belongs to the proof payload's historical
                    # policy snapshot.  ``run_directed`` recomputes both the
                    # current mirror and a replay-valid explicit-DMA lowering.
                    "stored_top4_bottom2_mirror_cc": int(top4_bottom2_cc),
                }
            )
    return rows


def score_state(state: fixed.PolicyState) -> int:
    return scorers.min_greedy_lpt4_task(
        state.c2, state.c3, state.remaining, policy=HW_CONFIG["policy"]
    )


def deduplicate(states: list[fixed.PolicyState]) -> list[fixed.PolicyState]:
    unique = {}
    for state in states:
        key = fixed.state_key(state)
        if key not in unique:
            unique[key] = state
    return list(unique.values())


def candidate_beam(
    root: fixed.PolicyState,
    *,
    width: int,
    branch_one_idle: bool,
    candidate_policy: str = "deployed",
    return_trace: bool = False,
) -> dict:
    if width <= 0:
        raise ValueError("beam width must be positive")
    Node = tuple[fixed.PolicyState, tuple[fixed.ScheduleStep, ...]]

    def deduplicate_nodes(nodes: list[Node]) -> list[Node]:
        unique = {}
        for node in nodes:
            unique.setdefault(fixed.state_key(node[0]), node)
        return list(unique.values())

    buckets: dict[int, list[Node]] = defaultdict(list)
    buckets[len(root.remaining)].append((root, ()))
    expanded = generated = deduplicated = pruned = peak = 0
    for remaining_count in range(len(root.remaining), 0, -1):
        nodes = deduplicate_nodes(buckets.pop(remaining_count, []))
        deduplicated += len(nodes)
        nodes.sort(
            key=lambda node: (
                score_state(node[0]),
                max(node[0].c2.task_end, node[0].c3.task_end),
            )
        )
        if len(nodes) > width:
            pruned += len(nodes) - width
            nodes = nodes[:width]
        peak = max(peak, len(nodes))
        for state, trace in nodes:
            expanded += 1
            if candidate_policy == "deployed":
                transitions = fixed.generate_successors(state, **HW_CONFIG)
            elif candidate_policy == "one_idle_shape_v2":
                transitions = fixed.generate_one_idle_shape_successors(state, **HW_CONFIG)
            elif candidate_policy == "resident_v2":
                transitions = fixed.generate_resident_successors(state, **HW_CONFIG)
            elif candidate_policy == "resident_shape_v2":
                transitions = fixed.generate_augmented_successors(state, **HW_CONFIG)
            else:
                raise ValueError(f"unknown candidate_policy {candidate_policy!r}")
            generated += len(transitions)
            both_idle = state.c2.task_end == state.c3.task_end
            if len(state.remaining) == 1 or (not both_idle and not branch_one_idle):
                transitions = [
                    min(
                        transitions,
                        key=lambda transition: max(
                            transition.state.c2.task_end,
                            transition.state.c3.task_end,
                        ),
                    )
                ]
            for transition in transitions:
                step = fixed.ScheduleStep(state, transition.state, transition.tag)
                buckets[len(transition.state.remaining)].append(
                    (transition.state, trace + (step,))
                )

    terminals = deduplicate_nodes(buckets.pop(0, []))
    if not terminals:
        raise RuntimeError("candidate beam produced no terminal state")
    best_state, best_trace = min(
        terminals,
        key=lambda node: fixed.terminal_cost(node[0]),
    )
    result = {
        "makespan_cc": fixed.terminal_cost(best_state),
        "exact_candidate_optimum": pruned == 0,
        "expanded_states": expanded,
        "generated_states": generated,
        "deduplicated_states_before_pruning": deduplicated,
        "pruned_states": pruned,
        "terminal_states": len(terminals),
        "peak_retained_states": peak,
    }
    if return_trace:
        result["_steps"] = best_trace
    return result


def _fixed_steps_trace(
    steps: tuple[fixed.ScheduleStep, ...],
) -> tuple[top4_proof.MirrorTransition, ...]:
    """Convert fixed-policy steps into the generic lowering format."""
    trace = []
    for step in steps:
        remaining_eids = {eid for eid, _ntok in step.after.remaining}
        selected = tuple(
            item for item in step.before.remaining if item[0] not in remaining_eids
        )
        if not selected:
            raise RuntimeError(f"HW-v2 transition {step.tag} consumed nothing")
        trace.append(
            top4_proof.MirrorTransition(
                step.before,
                step.after,
                step.tag,
                step.tag,
                selected,
            )
        )
    return tuple(trace)


def _fixed_hw_v2_trace(
    distribution: dict[int, int],
) -> tuple[top4_proof.MirrorTransition, ...]:
    """Expose frozen HW-v2 decisions in the generic lowering format."""
    steps = fixed.schedule_trace_with_scorer(
        distribution,
        continuation=fixed.hw_v2_continuation,
        candidate_policy=fixed.HW_V2_CANDIDATE_POLICY,
        **HW_CONFIG,
    )
    return _fixed_steps_trace(steps)


def _explicit_dma_lowering(
    distribution: dict[int, int],
    trace: tuple[top4_proof.MirrorTransition, ...],
    *,
    label: str,
) -> dict:
    """Choose the best replay-valid explicit-DMA realization of one trace.

    ``late`` and ``proactive`` attempt to materialize the mirror model's
    abstract prefetch decisions, and the better replay-valid result is used.
    Only if both fail, ``stage_only`` preserves the expert/cluster issue
    sequence while letting the reference model choose a conservative legal
    stage realization without claiming exact prefetch equivalence.
    """
    variants = []
    errors = []
    for lowering_mode, proactive in (("late", False), ("proactive", True)):
        try:
            state, _lowering = top4_proof.lower_policy_to_reference(
                distribution,
                proactive_prefetch=proactive,
                materialize_prefetch=True,
                mirror_trace=trace,
            )
            replay_cc = reference.validate_schedule_history(
                state.history, distribution
            )
            if replay_cc != state.g_score:
                raise RuntimeError(
                    f"{label}: {lowering_mode} replay {replay_cc} "
                    f"!= lowering score {state.g_score}"
                )
            variants.append((state.g_score, lowering_mode, state, _lowering))
        except (AssertionError, KeyError, RuntimeError, ValueError) as exc:
            errors.append(f"{lowering_mode}: {exc}")
    # Do not let the less faithful stage-only realization win merely because
    # it is faster.  It is a legality fallback only when neither attempt to
    # materialize the mirror's abstract prefetch requirements succeeds.
    if not variants:
        lowering_mode = "stage_only"
        try:
            state, _lowering = top4_proof.lower_policy_to_reference(
                distribution,
                materialize_prefetch=False,
                mirror_trace=trace,
            )
            replay_cc = reference.validate_schedule_history(
                state.history, distribution
            )
            if replay_cc != state.g_score:
                raise RuntimeError(
                    f"{label}: stage_only replay {replay_cc} "
                    f"!= lowering score {state.g_score}"
                )
            variants.append((state.g_score, lowering_mode, state, _lowering))
        except (AssertionError, KeyError, RuntimeError, ValueError) as exc:
            errors.append(f"stage_only: {exc}")
    if not variants:
        return {
            "cc": None,
            "mode": None,
            "action_count": None,
            "actions": [],
            "lowering": [],
            "errors": errors,
        }
    score, mode, state, lowering = min(
        variants, key=lambda item: (item[0], item[1])
    )
    return {
        "cc": score,
        "mode": mode,
        "action_count": len(state.history),
        "actions": [serialize_action(action) for action in state.history],
        "lowering": list(lowering),
        "errors": errors,
    }


def _consuming_action_signature(action: dict, *, full: bool) -> tuple | None:
    c2_active = int(action["c2_eid"]) >= 0
    c3_active = int(action["c3_eid"]) >= 0
    if not c2_active and not c3_active:
        return None
    if c2_active and c3_active and action["c2_eid"] == action["c3_eid"]:
        family = "SPLIT"
    elif c2_active and c3_active:
        family = "PAIR"
    else:
        family = "SINGLE"
    selection = (
        family,
        tuple(
            sorted(
                (
                    int(action["c2_ntok"]) if c2_active else 0,
                    int(action["c3_ntok"]) if c3_active else 0,
                ),
                reverse=True,
            )
        ),
    )
    if not full:
        return selection
    physical_fields = tuple(
        action[field]
        for field in (
            "c2_start_cc",
            "c2_shape_s1",
            "c2_shape_s3",
            "c2_s1_cached",
            "c2_s3_cached",
            "c2_s2pf_start_cc",
            "c2_dma_s1",
            "c2_dma_s3",
            "c2_s2pf_dma",
            "c3_start_cc",
            "c3_shape_s1",
            "c3_shape_s3",
            "c3_s1_cached",
            "c3_s3_cached",
            "c3_s2pf_start_cc",
            "c3_dma_s1",
            "c3_dma_s3",
            "c3_s2pf_dma",
        )
    )
    return selection + physical_fields


def _first_history_divergence(
    optimal_actions: list[dict], policy_actions: list[dict]
) -> dict:
    optimal_selection = [
        signature
        for action in optimal_actions
        if (signature := _consuming_action_signature(action, full=False))
        is not None
    ]
    policy_selection = [
        signature
        for action in policy_actions
        if (signature := _consuming_action_signature(action, full=False))
        is not None
    ]
    optimal_full = [
        signature
        for action in optimal_actions
        if (signature := _consuming_action_signature(action, full=True)) is not None
    ]
    policy_full = [
        signature
        for action in policy_actions
        if (signature := _consuming_action_signature(action, full=True)) is not None
    ]

    def first(left: list[tuple], right: list[tuple]) -> dict | None:
        for index in range(max(len(left), len(right))):
            lhs = left[index] if index < len(left) else None
            rhs = right[index] if index < len(right) else None
            if lhs != rhs:
                return {
                    "round": index,
                    "optimal": list(lhs) if lhs is not None else None,
                    "policy": list(rhs) if rhs is not None else None,
                }
        return None

    return {
        "first_expert_selection_or_family": first(
            optimal_selection, policy_selection
        ),
        "first_physical_action": first(optimal_full, policy_full),
        "interpretation": (
            "Comparison against one saved optimal history; divergence does not "
            "exclude another equal-optimal policy-compatible history."
        ),
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[max(0, min(len(values) - 1, math.ceil(len(values) * fraction) - 1))]


def summarize(rows: list[dict], oracle_name: str) -> dict:
    if not rows:
        return {"cases": 0}
    old_total = sum(row["old_hw_cc"] for row in rows)
    scorer_total = sum(row["new_scorer_cc"] for row in rows)
    oracle_total = sum(row[oracle_name]["makespan_cc"] for row in rows)
    reference_rows = [row for row in rows if row["reference_proven"]]
    reference_total = sum(row["reference_cc"] for row in reference_rows)
    candidate_ratios = [row[oracle_name]["makespan_cc"] / row["reference_cc"] for row in reference_rows]
    return {
        "cases": len(rows),
        "certified_candidate_optimum_cases": sum(row[oracle_name]["exact_candidate_optimum"] for row in rows),
        "candidate_matches_reference_cases": sum(
            row[oracle_name]["makespan_cc"] == row["reference_cc"] for row in reference_rows
        ),
        "old_hw_over_oracle_aggregate": old_total / oracle_total,
        "new_scorer_over_oracle_aggregate": scorer_total / oracle_total,
        "oracle_over_proven_reference_aggregate": oracle_total / reference_total,
        "candidate_ratio_p95": percentile(candidate_ratios, 0.95),
        "candidate_ratio_max": max(candidate_ratios),
        "old_selection_loss_cc": old_total - oracle_total,
        "new_scorer_selection_loss_cc": scorer_total - oracle_total,
        "candidate_loss_upper_bound_cc": oracle_total - reference_total,
    }


def summarize_directed(rows: list[dict], oracle_name: str) -> dict:
    if not rows:
        return {"cases": 0}
    proven_rows = [row for row in rows if row["reference_proven"]]
    hw_mirror_delta = [
        row["current_hw_v2_mirror_cc"] - row["reference_cc"] for row in rows
    ]
    hw_legal_rows = [
        row for row in rows if row["current_hw_v2_legal_cc"] is not None
    ]
    hw_legal_delta = [
        row["current_hw_v2_legal_cc"] - row["reference_cc"]
        for row in hw_legal_rows
    ]
    top4_mirror_delta = [
        row["current_top4_bottom2_mirror_cc"] - row["reference_cc"]
        for row in rows
    ]
    legal_rows = [
        row for row in rows if row["current_top4_bottom2_legal_cc"] is not None
    ]
    top4_legal_delta = [
        row["current_top4_bottom2_legal_cc"] - row["reference_cc"]
        for row in legal_rows
    ]
    selection_gain = [
        row["current_hw_v2_mirror_cc"] - row[oracle_name]["makespan_cc"]
        for row in rows
    ]
    oracle_delta = [row[oracle_name]["makespan_cc"] - row["reference_cc"] for row in rows]
    oracle_legal_rows = [
        row for row in rows if row[oracle_name]["legal_cc"] is not None
    ]
    oracle_legal_delta = [
        row[oracle_name]["legal_cc"] - row["reference_cc"]
        for row in oracle_legal_rows
    ]
    return {
        "cases": len(rows),
        "reference_proven_cases": len(proven_rows),
        "current_hw_v2_legal_lowering_cases": len(hw_legal_rows),
        "current_hw_v2_legal_lowering_failures": len(rows) - len(hw_legal_rows),
        "current_hw_v2_legal_above_reference_ub_cases": sum(
            value > 0 for value in hw_legal_delta
        ),
        "current_hw_v2_legal_equal_reference_ub_cases": sum(
            value == 0 for value in hw_legal_delta
        ),
        "current_hw_v2_legal_below_reference_ub_cases": sum(
            value < 0 for value in hw_legal_delta
        ),
        "current_hw_v2_legal_extra_over_reference_ub_ticks": _ticks_text(
            sum(hw_legal_delta)
        ),
        "current_hw_v2_mirror_above_reference_ub_cases_diagnostic": sum(
            value > 0 for value in hw_mirror_delta
        ),
        "current_hw_v2_mirror_equal_reference_ub_cases_diagnostic": sum(
            value == 0 for value in hw_mirror_delta
        ),
        "current_hw_v2_mirror_below_reference_ub_cases_ghost_warning": sum(
            value < 0 for value in hw_mirror_delta
        ),
        "top4_bottom2_legal_lowering_cases": len(legal_rows),
        "top4_bottom2_legal_lowering_failures": len(rows) - len(legal_rows),
        "top4_bottom2_legal_above_reference_ub_cases": sum(
            value > 0 for value in top4_legal_delta
        ),
        "top4_bottom2_legal_equal_reference_ub_cases": sum(
            value == 0 for value in top4_legal_delta
        ),
        "top4_bottom2_legal_below_reference_ub_cases": sum(
            value < 0 for value in top4_legal_delta
        ),
        "top4_bottom2_legal_extra_over_reference_ub_ticks": _ticks_text(
            sum(top4_legal_delta)
        ),
        "top4_bottom2_mirror_above_reference_ub_cases_diagnostic": sum(
            value > 0 for value in top4_mirror_delta
        ),
        "top4_bottom2_mirror_equal_reference_ub_cases_diagnostic": sum(
            value == 0 for value in top4_mirror_delta
        ),
        "top4_bottom2_mirror_below_reference_ub_cases_ghost_warning": sum(
            value < 0 for value in top4_mirror_delta
        ),
        "top4_bottom2_current_vs_stored_mirror_drift_cases": sum(
            row["current_top4_bottom2_mirror_cc"]
            != row["stored_top4_bottom2_mirror_cc"]
            for row in rows
        ),
        "oracle_improves_current_hw_cases": sum(value > 0 for value in selection_gain),
        "oracle_equal_current_hw_cases": sum(value == 0 for value in selection_gain),
        "oracle_worse_than_current_hw_cases": sum(value < 0 for value in selection_gain),
        "selection_gain_ticks": _ticks_text(sum(selection_gain)),
        "selection_gain_max_ticks": _ticks_text(max(selection_gain)),
        "oracle_legal_lowering_cases": len(oracle_legal_rows),
        "oracle_legal_lowering_failures": len(rows) - len(oracle_legal_rows),
        "oracle_legal_above_reference_ub_cases": sum(
            value > 0 for value in oracle_legal_delta
        ),
        "oracle_legal_equal_reference_ub_cases": sum(
            value == 0 for value in oracle_legal_delta
        ),
        "oracle_legal_below_reference_ub_cases": sum(
            value < 0 for value in oracle_legal_delta
        ),
        "oracle_legal_extra_over_reference_ub_ticks": _ticks_text(
            sum(oracle_legal_delta)
        ),
        "oracle_mirror_above_reference_ub_cases_diagnostic": sum(
            value > 0 for value in oracle_delta
        ),
        "oracle_mirror_equal_reference_ub_cases_diagnostic": sum(
            value == 0 for value in oracle_delta
        ),
        "oracle_mirror_below_reference_ub_cases_ghost_warning": sum(
            value < 0 for value in oracle_delta
        ),
        "oracle_mirror_extra_over_reference_ub_ticks_diagnostic": _ticks_text(
            sum(oracle_delta)
        ),
        "oracle_mirror_gap_max_ticks_diagnostic": _ticks_text(max(oracle_delta)),
        "certified_candidate_optimum_cases": sum(
            row[oracle_name]["exact_candidate_optimum"] for row in rows
        ),
        "oracle_matches_proven_optimum_cases": sum(
            row[oracle_name]["makespan_cc"] == row["reference_cc"] for row in proven_rows
        ),
        "oracle_legal_matches_proven_optimum_cases": sum(
            row[oracle_name]["legal_cc"] == row["reference_cc"]
            for row in proven_rows
            if row[oracle_name]["legal_cc"] is not None
        ),
    }


def run_directed(args, pairs: list[tuple[Path, Path]], widths: list[int]) -> int:
    if TICK_CC != reference.SCHEDULE_TIME_QUANTUM_CC:
        raise RuntimeError(
            f"tick mismatch: policy={TICK_CC}, reference="
            f"{reference.SCHEDULE_TIME_QUANTUM_CC}"
        )
    source_rows = load_directed_cases(pairs)
    if args.directed_case:
        requested = set(args.directed_case)
        available = {row["name"] for row in source_rows}
        missing = requested - available
        if missing:
            raise RuntimeError(f"unknown directed cases: {sorted(missing)}")
        source_rows = [row for row in source_rows if row["name"] in requested]
    if (
        args.expected_directed_cases > 0
        and len(source_rows) != args.expected_directed_cases
    ):
        raise RuntimeError(
            f"directed case count {len(source_rows)} != expected "
            f"{args.expected_directed_cases}"
        )
    if args.strict_directed:
        unproven = [row["name"] for row in source_rows if not row["reference_proven"]]
        if unproven:
            raise RuntimeError(
                f"strict directed analysis requires proved references: {unproven}"
            )
    rows = []
    started = time.perf_counter()
    for index, source in enumerate(source_rows, 1):
        reference_history = tuple(
            deserialize_action(action) for action in source["reference_actions"]
        )
        reference_replay_cc = reference.validate_schedule_history(
            reference_history, source["dist"]
        )
        if reference_replay_cc != source["reference_cc"]:
            raise RuntimeError(
                f"{source['key']}: reference replay {reference_replay_cc} "
                f"!= recorded {source['reference_cc']}"
            )
        current_cc = fixed.hw_v2_schedule(
            source["dist"], source["c2"], source["c3"], **HW_CONFIG
        )
        hw_v2_trace = _fixed_hw_v2_trace(source["dist"])
        traced_hw_cc = max(
            hw_v2_trace[-1].after.c2.task_end,
            hw_v2_trace[-1].after.c3.task_end,
        )
        if traced_hw_cc != current_cc:
            raise RuntimeError(
                f"{source['key']}: HW-v2 trace {traced_hw_cc} "
                f"!= schedule {current_cc}"
            )
        hw_v2_legal = _explicit_dma_lowering(
            source["dist"], hw_v2_trace, label=f"{source['key']} HW-v2"
        )
        mirror_trace = top4_proof._mirror_trace(source["dist"])
        current_top4_mirror_cc = max(
            mirror_trace[-1].after.c2.task_end,
            mirror_trace[-1].after.c3.task_end,
        )
        top4_legal = _explicit_dma_lowering(
            source["dist"], mirror_trace, label=f"{source['key']} top4+bottom2"
        )
        for label, lowering in (("HW-v2", hw_v2_legal), ("top4+bottom2", top4_legal)):
            if args.strict_directed and lowering["cc"] is None:
                raise RuntimeError(
                    f"{source['key']}: strict {label} lowering failed: "
                    f"{lowering['errors']}"
                )
            if (
                lowering["cc"] is not None
                and source["reference_proven"]
                and lowering["cc"] < source["reference_cc"]
            ):
                raise RuntimeError(
                    f"{source['key']}: legal {label} result {lowering['cc']} "
                    f"is below proved optimum {source['reference_cc']}"
                )
        hw_v2_divergence = (
            _first_history_divergence(
                source["reference_actions"], hw_v2_legal["actions"]
            )
            if hw_v2_legal["cc"] is not None
            else None
        )
        top4_divergence = (
            _first_history_divergence(
                source["reference_actions"], top4_legal["actions"]
            )
            if top4_legal["cc"] is not None
            else None
        )
        row = {
            key: value
            for key, value in source.items()
            if key not in {"dist", "reference_actions"}
        }
        row.update(
            {
                "current_hw_v2_cc": current_cc,
                "current_hw_v2_ticks": _ticks_text(current_cc),
                "current_hw_v2_mirror_cc": current_cc,
                "current_hw_v2_mirror_ticks": _ticks_text(current_cc),
                "current_hw_v2_legal_cc": hw_v2_legal["cc"],
                "current_hw_v2_legal_ticks": (
                    _ticks_text(hw_v2_legal["cc"])
                    if hw_v2_legal["cc"] is not None
                    else None
                ),
                "current_hw_v2_lowering_mode": hw_v2_legal["mode"],
                "current_hw_v2_legal_action_count": hw_v2_legal["action_count"],
                "current_hw_v2_legal_actions": hw_v2_legal["actions"],
                "current_hw_v2_lowering": hw_v2_legal["lowering"],
                "current_hw_v2_lowering_errors": hw_v2_legal["errors"],
                "current_hw_v2_first_divergence": hw_v2_divergence,
                "stored_top4_bottom2_mirror_ticks": _ticks_text(
                    source["stored_top4_bottom2_mirror_cc"]
                ),
                "current_top4_bottom2_mirror_cc": current_top4_mirror_cc,
                "current_top4_bottom2_mirror_ticks": _ticks_text(
                    current_top4_mirror_cc
                ),
                "current_top4_bottom2_legal_cc": top4_legal["cc"],
                "current_top4_bottom2_legal_ticks": (
                    _ticks_text(top4_legal["cc"])
                    if top4_legal["cc"] is not None
                    else None
                ),
                "current_top4_bottom2_lowering_mode": top4_legal["mode"],
                "current_top4_bottom2_legal_action_count": top4_legal[
                    "action_count"
                ],
                "current_top4_bottom2_legal_actions": top4_legal["actions"],
                "current_top4_bottom2_lowering": top4_legal["lowering"],
                "current_top4_bottom2_lowering_errors": top4_legal["errors"],
                "current_top4_bottom2_first_divergence": top4_divergence,
                "reference_best_ub_ticks": _ticks_text(source["reference_cc"]),
                "reference_lower_bound_ticks": _ticks_text(
                    source["reference_lower_bound_cc"]
                ),
            }
        )
        root = fixed.initial_state(source["dist"], source["c2"], source["c3"])
        for width in widths:
            oracle_name = f"oracle_w{width}"
            oracle = candidate_beam(
                root,
                width=width,
                branch_one_idle=True,
                candidate_policy=fixed.HW_V2_CANDIDATE_POLICY,
                return_trace=True,
            )
            oracle_steps = oracle.pop("_steps")
            oracle_trace = _fixed_steps_trace(oracle_steps)
            traced_oracle_cc = max(
                oracle_trace[-1].after.c2.task_end,
                oracle_trace[-1].after.c3.task_end,
            )
            if traced_oracle_cc != oracle["makespan_cc"]:
                raise RuntimeError(
                    f"{source['key']} {oracle_name}: trace {traced_oracle_cc} "
                    f"!= beam {oracle['makespan_cc']}"
                )
            oracle_legal = _explicit_dma_lowering(
                source["dist"],
                oracle_trace,
                label=f"{source['key']} {oracle_name}",
            )
            if args.strict_oracle_lowering and oracle_legal["cc"] is None:
                raise RuntimeError(
                    f"{source['key']} {oracle_name}: required oracle lowering "
                    f"failed: {oracle_legal['errors']}"
                )
            if (
                oracle_legal["cc"] is not None
                and source["reference_proven"]
                and oracle_legal["cc"] < source["reference_cc"]
            ):
                raise RuntimeError(
                    f"{source['key']} {oracle_name}: legal oracle result "
                    f"{oracle_legal['cc']} is below proved optimum "
                    f"{source['reference_cc']}"
                )
            oracle.update(
                {
                    "makespan_ticks": _ticks_text(oracle["makespan_cc"]),
                    "selection_gain_ticks": _ticks_text(
                        current_cc - oracle["makespan_cc"]
                    ),
                    "minus_reference_ub_ticks": _ticks_text(
                        oracle["makespan_cc"] - source["reference_cc"]
                    ),
                    "legal_cc": oracle_legal["cc"],
                    "legal_ticks": (
                        _ticks_text(oracle_legal["cc"])
                        if oracle_legal["cc"] is not None
                        else None
                    ),
                    "legal_minus_reference_ub_ticks": (
                        _ticks_text(oracle_legal["cc"] - source["reference_cc"])
                        if oracle_legal["cc"] is not None
                        else None
                    ),
                    "lowering_mode": oracle_legal["mode"],
                    "legal_action_count": oracle_legal["action_count"],
                    "legal_actions": oracle_legal["actions"],
                    "lowering": oracle_legal["lowering"],
                    "lowering_errors": oracle_legal["errors"],
                    "first_divergence": (
                        _first_history_divergence(
                            source["reference_actions"], oracle_legal["actions"]
                        )
                        if oracle_legal["cc"] is not None
                        else None
                    ),
                    "mirror_trace": [
                        {
                            "tag": transition.tag,
                            "selected_counts": [
                                ntok for _eid, ntok in transition.selected
                            ],
                        }
                        for transition in oracle_trace
                    ],
                }
            )
            row[oracle_name] = oracle
        rows.append(row)
        if args.progress_every > 0 and index % args.progress_every == 0:
            print(
                f"directed-candidate-oracle completed={index}/{len(source_rows)} "
                f"elapsed_s={time.perf_counter()-started:.1f}",
                flush=True,
            )

    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets["overall"].append(row)
        buckets[f"group:{row['group']}"] .append(row)
        buckets[f"hotness:{row['metrics'].get('top1_hotness_band', 'unknown')}"] .append(row)
        buckets[f"hotspots:{row['metrics'].get('local_hotspot_count', 'unknown')}"] .append(row)
    summary = {
        bucket: {
            f"oracle_w{width}": summarize_directed(values, f"oracle_w{width}")
            for width in widths
        }
        for bucket, values in sorted(buckets.items())
    }
    if len(widths) >= 2:
        narrow, wide = widths[-2], widths[-1]
        summary["width_convergence"] = {
            "narrow_width": narrow,
            "wide_width": wide,
            "wide_improves_cases": sum(
                row[f"oracle_w{wide}"]["makespan_cc"]
                < row[f"oracle_w{narrow}"]["makespan_cc"]
                for row in rows
            ),
            "equal_makespan_cases": sum(
                row[f"oracle_w{wide}"]["makespan_cc"]
                == row[f"oracle_w{narrow}"]["makespan_cc"]
                for row in rows
            ),
            "wide_total_gain_ticks": _ticks_text(
                sum(
                    row[f"oracle_w{narrow}"]["makespan_cc"]
                    - row[f"oracle_w{wide}"]["makespan_cc"]
                    for row in rows
                )
            ),
        }
    payload = {
        "schema": "scheduler_hw_directed_candidate_oracle_v2",
        "configuration": {
            "hw": HW_CONFIG,
            "candidate_policy": fixed.HW_V2_CANDIDATE_POLICY,
            "beam_widths": widths,
            "branch_one_idle": True,
            "strict_directed": args.strict_directed,
            "strict_oracle_lowering": args.strict_oracle_lowering,
            "expected_directed_cases": args.expected_directed_cases,
            "selected_directed_cases": list(args.directed_case),
            "tick_cc": TICK_CC,
            "source_sha256": {
                "analyze_scheduler_hw_candidate_oracle.py": file_sha256(
                    Path(__file__)
                ),
                "scheduler_hw_fixed_policy.py": file_sha256(
                    ROOT / "scheduler_hw_fixed_policy.py"
                ),
                "scheduler_top4_bottom2_policy.py": file_sha256(
                    ROOT / "scheduler_top4_bottom2_policy.py"
                ),
                "four_stage_scheduler.py": file_sha256(
                    ROOT / "four_stage_scheduler.py"
                ),
                "prove_top4_bottom2_directed.py": file_sha256(
                    ROOT / "prove_top4_bottom2_directed.py"
                ),
            },
            "pairs": [
                {
                    "suite": str(suite.resolve()),
                    "suite_sha256": file_sha256(suite),
                    "reference": str(reference.resolve()),
                    "reference_sha256": file_sha256(reference),
                }
                for suite, reference in pairs
            ],
        },
        "cases": len(rows),
        "runtime_s": time.perf_counter() - started,
        "summary": summary,
        "rows": rows,
        "interpretation": [
            "Current HW-v2 and every oracle path use the same timing model and frozen candidate generator.",
            "current_hw_v2_mirror_ticks is the frozen algorithmic mirror; only current_hw_v2_legal_ticks is its replay-valid explicit-DMA lowering.",
            "Stored top4+bottom2 mirror ticks are historical diagnostics from the reference payload; current mirror ticks are recomputed from the current source.",
            "Only current_top4_bottom2_legal_ticks is replay-valid explicit-DMA evidence; current_top4_bottom2_mirror_ticks may contain ghost-prefetch state.",
            "A stage_only lowering preserves the mirror expert-count/cluster issue sequence but not its abstract prefetch timing; it is a conservative replay-valid upper bound for that sequence, not an exact physical mirror replay.",
            "Current-HW minus oracle is selection/control loss within that candidate graph.",
            "When pruned_states is zero, only the mirror candidate-graph optimum is certified; formal physical candidate sufficiency requires the restricted explicit-DMA reference oracle.",
            "The saved oracle trace is lowered and replayed, but failure of that one mirror-optimal trace to reach the reference does not exclude another equal-mirror candidate trace with a better legal lowering.",
            "With beam pruning, oracle-to-reference gap is diagnostic rather than a formal candidate-space lower bound.",
            "A directed four-stage reference marked unproven is a legal best-known upper bound, not a certified optimum.",
            "An oracle result below the explicit-DMA reference requires legal lowering before it can be claimed as an improvement.",
        ],
    }
    atomic_write_json(args.out, payload)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", type=Path)
    parser.add_argument("--reference", action="append", type=Path)
    parser.add_argument(
        "--directed-pair",
        action="append",
        nargs=2,
        type=Path,
        metavar=("SUITE", "REFERENCE"),
        help="analyze a characteristic-oriented suite and completed pass result",
    )
    parser.add_argument(
        "--directed-case",
        action="append",
        default=[],
        help="optional directed case-name filter; may be repeated",
    )
    parser.add_argument(
        "--dataset-split",
        action="append",
        choices=("discovery", "validation", "blind_test"),
    )
    parser.add_argument("--include-unproven", action="store_true")
    parser.add_argument("--sample-per-stratum", type=int, default=4)
    parser.add_argument("--beam-width", action="append", type=int)
    parser.add_argument("--branch-one-idle", action="store_true")
    parser.add_argument(
        "--candidate-policy",
        choices=("deployed", "one_idle_shape_v2", "resident_v2", "resident_shape_v2"),
        default="deployed",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--strict-directed",
        action="store_true",
        help=(
            "require every directed reference to be proved and both deployed "
            "policy traces to have replay-valid explicit-DMA lowerings"
        ),
    )
    parser.add_argument(
        "--strict-oracle-lowering",
        action="store_true",
        help=(
            "additionally require post-hoc lowering of every mirror candidate-"
            "oracle trace; diagnostic only, not used for the formal baseline"
        ),
    )
    parser.add_argument(
        "--expected-directed-cases",
        type=int,
        default=0,
        help="reject directed analysis unless the loaded case count matches",
    )
    args = parser.parse_args()

    if args.expected_directed_cases < 0:
        raise SystemExit("--expected-directed-cases must be non-negative")

    suites = tuple(args.suite) if args.suite else DEFAULT_SUITES
    references = tuple(args.reference) if args.reference else DEFAULT_REFERENCES
    splits = set(args.dataset_split or ("discovery", "validation", "blind_test"))
    widths = sorted(set(args.beam_width or (8, 32, 128)))
    if args.directed_pair:
        return run_directed(args, [tuple(pair) for pair in args.directed_pair], widths)
    source_rows = load_cases(suites, references, splits, not args.include_unproven)
    source_rows = stratified_pick(source_rows, args.sample_per_stratum)
    rows = []
    started = time.perf_counter()
    for index, source in enumerate(source_rows, 1):
        old_cc = old_hw.hw_mirror_schedule(
            source["dist"], source["c2"], source["c3"], **HW_CONFIG
        )
        mirrored_cc = fixed.schedule_with_scorer(
            source["dist"], source["c2"], source["c3"], **HW_CONFIG
        )
        if old_cc != mirrored_cc:
            raise RuntimeError(f"fixed transition baseline mismatch at {source['key']}")
        new_cc = fixed.schedule_with_scorer(
            source["dist"],
            source["c2"],
            source["c3"],
            continuation=scorers.min_greedy_lpt4_task,
            **HW_CONFIG,
        )
        row = {key: value for key, value in source.items() if key != "dist"}
        row.update({"old_hw_cc": old_cc, "new_scorer_cc": new_cc})
        root = fixed.initial_state(source["dist"], source["c2"], source["c3"])
        for width in widths:
            row[f"oracle_w{width}"] = candidate_beam(
                root,
                width=width,
                branch_one_idle=args.branch_one_idle,
                candidate_policy=args.candidate_policy,
            )
        rows.append(row)
        if args.progress_every > 0 and index % args.progress_every == 0:
            print(
                f"candidate-oracle completed={index}/{len(source_rows)} "
                f"elapsed_s={time.perf_counter()-started:.1f}",
                flush=True,
            )

    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        for key in ("overall", f"E{row['e_total']}", f"split:{row['dataset_split']}"):
            buckets[key].append(row)
    summary = {
        bucket: {
            f"oracle_w{width}": summarize(values, f"oracle_w{width}")
            for width in widths
        }
        for bucket, values in sorted(buckets.items())
    }
    payload = {
        "schema": "scheduler_hw_candidate_oracle_v1",
        "configuration": {
            "hw": HW_CONFIG,
            "scorer": "min_greedy_lpt4_task",
            "beam_widths": widths,
            "branch_one_idle": args.branch_one_idle,
            "candidate_policy": args.candidate_policy,
            "sample_per_stratum": args.sample_per_stratum,
            "dataset_splits": sorted(splits),
            "proven_only": not args.include_unproven,
            "suites": [{"path": str(path.resolve()), "sha256": file_sha256(path)} for path in suites],
            "references": [{"path": str(path.resolve()), "sha256": file_sha256(path)} for path in references],
        },
        "cases": len(rows),
        "runtime_s": time.perf_counter() - started,
        "summary": summary,
        "rows": rows,
        "interpretation": [
            "Every oracle result is feasible within the deployed fixed candidate graph.",
            "When pruned_states is zero, only the mirror candidate-graph optimum is certified; formal physical candidate sufficiency requires the restricted explicit-DMA reference oracle.",
            "With beam pruning, oracle-reference gap is only an upper bound on candidate loss.",
            "Reference comparison is restricted to proven-optimal four-stage cases by default.",
        ],
    }
    atomic_write_json(args.out, payload)
    print(json.dumps(summary.get("overall", {}), indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
