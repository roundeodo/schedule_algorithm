#!/usr/bin/env python3
"""Classify directed cases by proof status and optimal-history visibility.

The report deliberately separates three questions:

* Was the reference makespan actually proved optimal?
* Can the saved legal history be expressed by a bounded expert window?
* If it can, does the policy-guided legal schedule still lose cycles?

Visibility of one optimal history proves that a window is sufficient for that
case.  Non-visibility of one history does *not* prove that the window is
insufficient because another equally optimal history may exist.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path

import four_stage_scheduler as reference
import prove_top4_bottom2_directed as proof
from run_four_stage_reference import deserialize_action
import scheduler_hw_fixed_policy as hw_base
import scheduler_rtl_adaptive_prefetch_policy as adaptive
import scheduler_top4_bottom2_policy as policy


WINDOWS = (
    (4, 0),
    (4, 2),
    (4, 4),
    (4, 8),
    (6, 0),
    (6, 2),
    (6, 4),
    (6, 8),
    (8, 0),
    (8, 2),
    (8, 4),
    (8, 8),
    (10, 0),
    (12, 0),
    (14, 0),
    (16, 0),
    (32, 0),
)
TICK_CC = policy.TICK_CC


def _window_name(top: int, bottom: int) -> str:
    return f"top{top}" if bottom == 0 else f"top{top}+bottom{bottom}"


def _visible(rank: int, entries: int, top: int, bottom: int) -> bool:
    return rank < min(top, entries) or (
        bottom > 0 and rank >= max(0, entries - bottom)
    )


def _action_family(tag: str) -> str:
    normalized = tag.upper()
    if normalized.startswith("PAIR"):
        return "PAIR"
    if normalized.startswith("SPLIT"):
        return "SPLIT"
    if normalized.startswith("SINGLE"):
        return "SINGLE"
    if normalized.startswith("PF-"):
        return "PREFETCH"
    return "OTHER"


def _symmetry_relabel_history(
    row: dict, top: int, bottom: int
) -> tuple[bool, tuple | None]:
    """Replay through a window after consistently permuting equal-load IDs.

    These directed cases start with empty caches, so experts with the same
    total token count are behaviorally interchangeable.  The DFS keeps an
    explicit prefetch-to-issue identity mapping and validates the relabelled
    terminal history in the independent four-stage model.
    """
    distribution = {eid: count for eid, count in enumerate(row["counts"])}
    actions = tuple(deserialize_action(action) for action in row["actions"])
    scheduler = reference.FourStageScheduler(distribution)
    initial = scheduler._initial_state()
    expected = Fraction(row["best_reference_ticks"]) * TICK_CC
    if expected.denominator != 1:
        raise RuntimeError(f"{row['name']}: non-integral expected score")

    def dfs(index, state, mapping: dict[int, int], assigned: frozenset[int], history):
        if index == len(actions):
            if state.remaining:
                return None
            relabelled = tuple(history)
            validated = reference.validate_schedule_history(relabelled, distribution)
            return relabelled if validated == int(expected) else None

        action = actions[index]
        remaining = state.remaining
        visible_indices = list(range(min(top, len(remaining))))
        if bottom:
            visible_indices.extend(
                range(
                    max(min(top, len(remaining)), len(remaining) - bottom),
                    len(remaining),
                )
            )
        visible_eids = {
            remaining[position][0] for position in dict.fromkeys(visible_indices)
        }
        referenced = []
        for eid in (action.pf_eid, action.c2_eid, action.c3_eid):
            if eid >= 0 and eid not in referenced:
                referenced.append(eid)
        for eid in referenced:
            if eid in mapping and mapping[eid] not in visible_eids:
                return None
        unmapped = [eid for eid in referenced if eid not in mapping]

        def assign(
            position: int,
            trial_mapping: dict[int, int],
            trial_assigned: frozenset[int],
        ):
            if position == len(unmapped):
                def mapped(eid: int) -> int:
                    return trial_mapping[eid] if eid >= 0 else eid

                relabelled = replace(
                    action,
                    c2_eid=mapped(action.c2_eid),
                    c3_eid=mapped(action.c3_eid),
                    pf_eid=mapped(action.pf_eid),
                )
                try:
                    child = reference.apply_action(state, relabelled)
                except (AssertionError, KeyError, RuntimeError, ValueError):
                    return None
                return dfs(
                    index + 1,
                    child,
                    trial_mapping,
                    trial_assigned,
                    history + (relabelled,),
                )

            original_eid = unmapped[position]
            required_count = distribution[original_eid]
            candidates = [
                eid
                for eid, count in remaining
                if eid in visible_eids
                and eid not in trial_assigned
                and count == required_count
            ]
            for physical_eid in candidates:
                next_mapping = dict(trial_mapping)
                next_mapping[original_eid] = physical_eid
                found = assign(
                    position + 1,
                    next_mapping,
                    trial_assigned | {physical_eid},
                )
                if found is not None:
                    return found
            return None

        return assign(0, mapping, assigned)

    relabelled = dfs(0, initial, {}, frozenset(), ())
    return relabelled is not None, relabelled


def _consuming_steps(row: dict) -> list[dict]:
    steps = []
    for raw in row["actions"]:
        action = deserialize_action(raw)
        counts = []
        if action.c2_eid >= 0 and action.c2_eid == action.c3_eid:
            counts.append(action.c2_ntok + action.c3_ntok)
        else:
            if action.c2_eid >= 0:
                counts.append(action.c2_ntok)
            if action.c3_eid >= 0:
                counts.append(action.c3_ntok)
        if not counts:
            continue
        steps.append(
            {
                "tag": action.tag,
                "family": _action_family(action.tag),
                "selected_counts": sorted(counts, reverse=True),
                "c2_count": action.c2_ntok if action.c2_eid >= 0 else None,
                "c3_count": action.c3_ntok if action.c3_eid >= 0 else None,
                "c2_shape_s1": (
                    action.c2_shape_s1.name if action.c2_shape_s1 is not None else None
                ),
                "c2_shape_s3": (
                    action.c2_shape_s3.name if action.c2_shape_s3 is not None else None
                ),
                "c3_shape_s1": (
                    action.c3_shape_s1.name if action.c3_shape_s1 is not None else None
                ),
                "c3_shape_s3": (
                    action.c3_shape_s3.name if action.c3_shape_s3 is not None else None
                ),
            }
        )
    return steps


def _first_policy_history_divergence(row: dict, policy_baseline: dict) -> dict:
    optimal_steps = _consuming_steps(row)
    policy_steps = [
        {
            "tag": step["reference_tag"],
            "family": _action_family(step["reference_tag"]),
            "selected_counts": sorted(step["selected_counts"], reverse=True),
            "decision": step["mirror_decision"],
        }
        for step in policy_baseline.get("lowering", ())
    ]
    if not policy_steps:
        return {
            "kind": "policy_lowering_unavailable",
            "index": None,
            "policy": None,
            "optimal": optimal_steps[0] if optimal_steps else None,
        }
    for index in range(max(len(policy_steps), len(optimal_steps))):
        policy_step = policy_steps[index] if index < len(policy_steps) else None
        optimal_step = optimal_steps[index] if index < len(optimal_steps) else None
        if policy_step is None or optimal_step is None:
            return {
                "kind": "different_consuming_step_count",
                "index": index,
                "policy": policy_step,
                "optimal": optimal_step,
            }
        if policy_step["selected_counts"] != optimal_step["selected_counts"]:
            return {
                "kind": "expert_count_selection_or_issue_order",
                "index": index,
                "policy": policy_step,
                "optimal": optimal_step,
            }
        if policy_step["family"] != optimal_step["family"]:
            return {
                "kind": "assignment_family",
                "index": index,
                "policy": policy_step,
                "optimal": optimal_step,
            }
    return {
        "kind": "same_expert_sequence_action_parameters_or_prefetch",
        "index": None,
        "policy": None,
        "optimal": None,
    }


def _selected_counts(before, after) -> list[int]:
    after_eids = {eid for eid, _count in after.remaining}
    return sorted(
        [count for eid, count in before.remaining if eid not in after_eids],
        reverse=True,
    )


def _candidate_key(state, transition) -> tuple[int, int, int, int]:
    child = transition.state
    current_makespan_cc = max(child.c2.task_end, child.c3.task_end)
    if len(state.remaining) == 1 or state.c2.task_end != state.c3.task_end:
        cost_ticks = adaptive._ceil_div(current_makespan_cc, policy.TICK_CC)
    else:
        continuation_cc = hw_base.hw_v2_continuation(
            child.c2, child.c3, child.remaining, policy="balanced"
        )
        cost_ticks = adaptive._ceil_div(continuation_cc, policy.TICK_CC)
    _token, candidate_id = adaptive.token_from_tag(state, transition.tag)
    return (
        cost_ticks,
        len(child.remaining),
        adaptive._ceil_div(current_makespan_cc, policy.TICK_CC),
        candidate_id,
    )


def _score_divergence_audit(row: dict, divergence: dict) -> dict | None:
    if divergence["kind"] != "expert_count_selection_or_issue_order":
        return None
    index = divergence["index"]
    distribution = {eid: count for eid, count in enumerate(row["counts"])}
    trace = proof._mirror_trace(distribution)
    if index is None or index >= len(trace):
        return {"status": "policy_trace_index_unavailable"}
    mirror = trace[index]
    if mirror.decision != "hw_v2_score":
        return {
            "status": "divergence_not_selected_by_hw_v2_score",
            "decision": mirror.decision,
        }
    state = mirror.before
    cost_model = adaptive._COST_MODELS[policy.DEFAULT_S4_POLICY]
    with adaptive._use_cost_model(cost_model):
        transitions = hw_base.generate_one_idle_shape_successors(
            state,
            policy="balanced",
            top_policy="pruned",
            n1_policy="pruned",
        )
        entries = []
        for transition in transitions:
            entries.append(
                {
                    "tag": transition.tag,
                    "family": _action_family(transition.tag),
                    "selected_counts": _selected_counts(state, transition.state),
                    "key": list(_candidate_key(state, transition)),
                }
            )
    chosen = next((entry for entry in entries if entry["tag"] == mirror.tag), None)
    optimal = divergence["optimal"]
    targets = [
        entry
        for entry in entries
        if entry["family"] == optimal["family"]
        and entry["selected_counts"] == optimal["selected_counts"]
    ]
    if not targets:
        return {
            "status": "optimal_high_level_action_missing_from_candidate_bank",
            "chosen": chosen,
            "target_family": optimal["family"],
            "target_selected_counts": optimal["selected_counts"],
        }
    target = min(targets, key=lambda entry: tuple(entry["key"]))
    if chosen is None:
        status = "chosen_transition_not_reconstructed"
    elif chosen["key"][0] == target["key"][0]:
        status = "continuation_tie_resolved_against_optimal_action"
    else:
        status = "continuation_score_prefers_nonoptimal_action"
    return {
        "status": status,
        "chosen": chosen,
        "best_matching_optimal_high_level_action": target,
        "matching_candidate_count": len(targets),
    }


def _audit_history(row: dict) -> dict:
    distribution = {eid: count for eid, count in enumerate(row["counts"])}
    scheduler = reference.FourStageScheduler(distribution)
    state = scheduler._initial_state()
    actions = tuple(deserialize_action(action) for action in row["actions"])
    compatible = {_window_name(*window): True for window in WINDOWS}
    issue_compatible = dict(compatible)
    prefetch_compatible = dict(compatible)
    first_violation: dict[str, dict] = {}
    issue_ranks: list[int] = []
    prefetch_ranks: list[int] = []
    family_counts: Counter[str] = Counter()
    dma_counts: Counter[str] = Counter()

    for index, action in enumerate(actions):
        remaining = state.remaining
        rank_by_eid = {eid: rank for rank, (eid, _count) in enumerate(remaining)}
        issue_eids = []
        for eid in (action.c2_eid, action.c3_eid):
            if eid >= 0 and eid not in issue_eids:
                issue_eids.append(eid)
        issue_at_step = [rank_by_eid[eid] for eid in issue_eids]
        issue_ranks.extend(issue_at_step)

        pf_at_step = []
        if action.pf_eid >= 0:
            pf_at_step = [rank_by_eid[action.pf_eid]]
            prefetch_ranks.extend(pf_at_step)

        family_counts[_action_family(action.tag)] += 1
        for binding in (
            action.c2_dma_s1,
            action.c2_dma_s3,
            action.c2_s2pf_dma,
            action.c3_dma_s1,
            action.c3_dma_s3,
            action.c3_s2pf_dma,
            action.pf_dma,
        ):
            if int(binding) != 0:
                dma_counts[reference.dma_name(binding)] += 1

        for top, bottom in WINDOWS:
            name = _window_name(top, bottom)
            issue_ok = all(
                _visible(rank, len(remaining), top, bottom)
                for rank in issue_at_step
            )
            prefetch_ok = all(
                _visible(rank, len(remaining), top, bottom)
                for rank in pf_at_step
            )
            issue_compatible[name] &= issue_ok
            prefetch_compatible[name] &= prefetch_ok
            compatible[name] &= issue_ok and prefetch_ok
            if not issue_ok or not prefetch_ok:
                first_violation.setdefault(
                    name,
                    {
                        "action_index": index,
                        "tag": action.tag,
                        "remaining_entries": len(remaining),
                        "remaining_counts": [count for _eid, count in remaining],
                        "issue_ranks_zero_based": issue_at_step,
                        "prefetch_ranks_zero_based": pf_at_step,
                        "issue_visible": issue_ok,
                        "prefetch_visible": prefetch_ok,
                    },
                )
        state = reference.apply_action(state, action)

    if state.remaining:
        raise RuntimeError(f"{row['name']}: history did not consume every expert")
    validated = reference.validate_schedule_history(actions, distribution)
    if validated != state.g_score:
        raise RuntimeError(
            f"{row['name']}: replay {validated} != state score {state.g_score}"
        )
    if Fraction(row["best_reference_ticks"]) * TICK_CC != validated:
        raise RuntimeError(
            f"{row['name']}: stored score does not match replayed history"
        )

    symmetry_compatible = {}
    relabelled_windows = []
    for window in WINDOWS:
        name = _window_name(*window)
        if compatible[name]:
            symmetry_compatible[name] = True
            continue
        found, _relabelled = _symmetry_relabel_history(row, *window)
        symmetry_compatible[name] = found
        if found:
            relabelled_windows.append(name)

    return {
        "history_replay_valid": True,
        "window_compatible": compatible,
        "symmetry_window_compatible": symmetry_compatible,
        "symmetry_relabelled_windows": relabelled_windows,
        "issue_window_compatible": issue_compatible,
        "prefetch_window_compatible": prefetch_compatible,
        "first_window_violation": first_violation,
        "max_issue_rank_zero_based": max(issue_ranks, default=-1),
        "max_prefetch_rank_zero_based": max(prefetch_ranks, default=-1),
        "action_family_counts": dict(sorted(family_counts.items())),
        "dma_binding_counts": dict(sorted(dma_counts.items())),
    }


def _classification(row: dict, audit: dict, policy_baseline: dict) -> str:
    if not row["proven_optimal"]:
        return "proof_pending"
    policy_available = policy_baseline["lowering_mode"] in {"late", "proactive"}
    policy_ticks = Fraction(policy_baseline["lowered_reference_ticks"])
    optimal_ticks = Fraction(row["best_reference_ticks"])
    # Equality here is stronger than inspecting the independently discovered
    # optimal history: the replay-validated policy-guided lowering itself is an
    # optimal witness and therefore proves the current semantic window suffices.
    if policy_available and policy_ticks == optimal_ticks:
        return "top4_bottom2_policy_guided_optimal"
    if audit["window_compatible"]["top4+bottom2"]:
        if not policy_available:
            return "top4_bottom2_visible_policy_lowering_failed"
        return "top4_bottom2_visible_selection_or_control_gap"
    return "top4_bottom2_window_sufficiency_unresolved_policy_gap"


def _summary(rows: list[dict]) -> dict:
    classifications = Counter(row["classification"] for row in rows)
    proven_rows = [row for row in rows if row["proven_optimal"]]
    best_known_rows = rows
    unproven_rows = [row for row in rows if not row["proven_optimal"]]

    def window_counts(source: list[dict], field: str) -> dict[str, int]:
        return {
            _window_name(*window): sum(
                row["history_audit"][field][_window_name(*window)]
                for row in source
            )
            for window in WINDOWS
        }

    return {
        "cases": len(rows),
        "proven_optimal": len(proven_rows),
        "unproven": len(rows) - len(proven_rows),
        "classification_counts": dict(sorted(classifications.items())),
        "proven_optimal_history_window_coverage": window_counts(
            proven_rows, "window_compatible"
        ),
        "proven_optimal_history_issue_window_coverage": window_counts(
            proven_rows, "issue_window_compatible"
        ),
        "proven_optimal_history_prefetch_window_coverage": window_counts(
            proven_rows, "prefetch_window_compatible"
        ),
        "best_known_history_window_coverage_diagnostic": window_counts(
            best_known_rows, "window_compatible"
        ),
        "proven_top6_sufficient": sum(
            row["top6_sufficiency"]["proven"] for row in proven_rows
        ),
        "proven_top6_unresolved": [
            row["name"]
            for row in proven_rows
            if not row["top6_sufficiency"]["proven"]
        ],
        "unproven_gap_buckets": dict(
            sorted(
                Counter(row["proof_pending_gap_bucket"] for row in unproven_rows).items()
            )
        ),
        "unproven_best_known_history_window_coverage_diagnostic": window_counts(
            unproven_rows, "window_compatible"
        ),
    }


def _top6_probe_witness(row: dict) -> bool:
    if not row.get("proven_optimal"):
        return False
    for trial in row.get("seed_beam_trials", ()):
        if trial.get("candidate_window") != {"top": 6, "bottom": 0}:
            continue
        if trial.get("window_history_found") is True:
            return True
    # Legacy window probes before ``window_history_found`` was recorded are
    # valid only when the bounded search strictly improved the outside-window
    # incumbent and reached the LB.  A non-improving feasible-history label is
    # deliberately not accepted.
    return row.get("termination") == "seed_beam_history_equals_root_lb"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--policy-baseline",
        type=Path,
        required=True,
        help="initial proof payload containing the original policy lowering",
    )
    parser.add_argument(
        "--top6-probe",
        type=Path,
        action="append",
        default=[],
        help="optional top6-restricted proof payload; may be repeated",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/policy_search/directed_case_classification.json"),
    )
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    baseline_payload = json.loads(args.policy_baseline.read_text(encoding="utf-8"))
    baseline_by_name = {row["name"]: row for row in baseline_payload["cases"]}
    top6_witness_sources: dict[str, list[str]] = {}
    for probe_path in args.top6_probe:
        probe_payload = json.loads(probe_path.read_text(encoding="utf-8"))
        for probe_row in probe_payload["cases"]:
            if _top6_probe_witness(probe_row):
                top6_witness_sources.setdefault(probe_row["name"], []).append(
                    str(probe_path)
                )
    rows = []
    for source in payload["cases"]:
        policy_baseline = baseline_by_name[source["name"]]
        audit = _audit_history(source)
        divergence = _first_policy_history_divergence(source, policy_baseline)
        row = {
            "name": source["name"],
            "origin": source["origin"],
            "tier": source["tier"],
            "family": source["family"],
            "profile": source["profile"],
            "batch_tokens": source["batch_tokens"],
            "active_experts": source["active_experts"],
            "counts": source["counts"],
            "root_lower_bound_ticks": source["root_lower_bound_ticks"],
            "policy_lowering_mode": policy_baseline["lowering_mode"],
            "policy_guided_legal_ticks": (
                policy_baseline["lowered_reference_ticks"]
                if policy_baseline["lowering_mode"] in {"late", "proactive"}
                else None
            ),
            "policy_lowering_fallback_ticks": (
                policy_baseline["lowered_reference_ticks"]
                if policy_baseline["lowering_mode"]
                == "reference_greedy_fallback"
                else None
            ),
            "mirror_policy_ticks_unverified": policy_baseline[
                "mirror_policy_ticks"
            ],
            "best_reference_ticks": source["best_reference_ticks"],
            "best_known_minus_root_lb_ticks": str(
                Fraction(source["best_reference_ticks"])
                - Fraction(source["root_lower_bound_ticks"])
            ),
            "proven_optimal": source["proven_optimal"],
            "proof_termination": source["termination"],
            "history_audit": audit,
            "first_policy_history_divergence": divergence,
            "divergence_candidate_score_audit": _score_divergence_audit(
                source, divergence
            ),
        }
        gap = Fraction(row["best_known_minus_root_lb_ticks"])
        row["proof_pending_gap_bucket"] = (
            None
            if source["proven_optimal"]
            else "gap_1_to_3"
            if gap <= 3
            else "gap_4_to_9"
            if gap <= 9
            else "gap_10_plus"
        )
        direct_top6 = (
            source["proven_optimal"]
            and audit["window_compatible"]["top6"]
        )
        probe_sources = top6_witness_sources.get(source["name"], [])
        row["top6_sufficiency"] = {
            "proven": bool(direct_top6 or probe_sources),
            "evidence": (
                "saved_optimal_history"
                if direct_top6
                else "restricted_probe"
                if probe_sources
                else "unresolved"
            ),
            "probe_sources": probe_sources,
        }
        row["classification"] = _classification(
            source, audit, policy_baseline
        )
        rows.append(row)

    result = {
        "schema": "directed_case_classification_v1",
        "source": str(args.input),
        "interpretation": {
            "compatible_optimal_history": (
                "proves the named window can express at least one saved optimal history"
            ),
            "incompatible_optimal_history": (
                "does not prove insufficiency; an alternative optimal history may exist"
            ),
            "best_known_unproven_history": (
                "diagnostic upper-bound evidence only, never an optimality claim"
            ),
        },
        "summary": _summary(rows),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
