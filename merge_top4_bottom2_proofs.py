#!/usr/bin/env python3
"""Merge directed proof passes, keeping the strongest verified row per case."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path

import four_stage_scheduler as reference
from run_four_stage_reference import deserialize_action


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ticks(cc: int, quantum: int) -> str:
    value = Fraction(cc, quantum)
    return str(value.numerator) if value.denominator == 1 else str(value)


def _campaign_row(
    path: Path, payload: dict, *, allow_restricted_witness: bool = False
) -> dict:
    """Convert one complete root-branch campaign into a mergeable proof row."""
    if payload.get("schema") != "target_root_branch_proof_v1":
        raise SystemExit(f"{path}: unsupported payload schema")
    if not payload.get("complete"):
        raise SystemExit(f"{path}: root campaign is incomplete")
    manifest = payload["manifest"]
    candidate_window = reference.normalize_candidate_window(
        tuple(manifest["candidate_window"])
        if manifest.get("candidate_window") is not None
        else None
    )
    if candidate_window is not None and not allow_restricted_witness:
        raise SystemExit(
            f"{path}: restricted-window campaign is a candidate-sufficiency "
            "result and cannot advance the global lower bound"
        )
    summary = payload["summary"]
    found = bool(summary.get("found_feasible_history"))
    exhausted = bool(summary.get("all_root_groups_exhaustive"))
    if candidate_window is not None and not found:
        raise SystemExit(
            f"{path}: a restricted campaign can be extracted only as a "
            "feasible witness; failure cannot advance the global lower bound"
        )
    if found == exhausted:
        raise SystemExit(
            f"{path}: campaign must be either feasible or fully exhaustive"
        )
    if int(summary["recorded_groups"]) != len(payload["branches"]):
        raise SystemExit(f"{path}: recorded-group count differs from branches")
    if exhausted and int(summary["recorded_groups"]) != int(
        manifest["semantic_root_groups"]
    ):
        raise SystemExit(f"{path}: exhaustive campaign omits root groups")

    prior_path = Path(manifest["prior_proof"])
    if not prior_path.is_file():
        raise SystemExit(f"{path}: missing prior proof {prior_path}")
    if _file_sha256(prior_path) != manifest["prior_proof_sha256"]:
        raise SystemExit(f"{path}: prior proof SHA-256 mismatch")
    reference_path = Path(__file__).resolve().parent / "four_stage_scheduler.py"
    if _file_sha256(reference_path) != manifest["reference_sha256"]:
        raise SystemExit(f"{path}: reference source SHA-256 mismatch")

    prior_payload = json.loads(prior_path.read_text(encoding="utf-8"))
    prior_by_name = {row["name"]: row for row in prior_payload["cases"]}
    case_name = manifest["case"]
    if case_name not in prior_by_name:
        raise SystemExit(f"{path}: prior proof is missing {case_name}")
    prior = prior_by_name[case_name]
    if [int(value) for value in prior["counts"]] != [
        int(value) for value in manifest["counts"]
    ]:
        raise SystemExit(f"{path}: distribution differs from prior proof")

    quantum = int(manifest["time_quantum_cc"])
    target = int(manifest["target_cc"])
    distribution = {
        eid: int(ntok)
        for eid, ntok in enumerate(manifest["counts"])
        if int(ntok) > 0
    }
    row = dict(prior)
    if candidate_window is not None:
        prior_lb = int(
            Fraction(prior["certified_lower_bound_ticks"]) * quantum
        )
        prior_ub = int(Fraction(prior["best_reference_ticks"]) * quantum)
        if (
            not prior.get("proven_optimal")
            or prior_lb != prior_ub
            or target != prior_ub
        ):
            raise SystemExit(
                f"{path}: restricted witness requires a globally proved "
                "LB=UB prior and exactly that optimum target"
            )
    evidence = {
        "campaign": str(path),
        "campaign_sha256": _file_sha256(path),
        "manifest_id": manifest["manifest_id"],
        "target_ticks": _ticks(target, quantum),
        "semantic_root_groups": int(manifest["semantic_root_groups"]),
        "root_children": int(manifest["root_children"]),
        "total_expansions": int(summary["total_expansions"]),
        "total_generated": int(summary["total_generated"]),
    }
    if found:
        feasible = [branch for branch in payload["branches"] if branch["feasible"]]
        if not feasible:
            raise SystemExit(f"{path}: feasible summary has no feasible branch")
        branch = min(feasible, key=lambda item: int(item["group_index"]))
        history = tuple(deserialize_action(action) for action in branch["actions"])
        replay_cc = reference.validate_schedule_history(history, distribution)
        if replay_cc != target:
            raise SystemExit(
                f"{path}: frontier history replay {replay_cc} != target {target}"
            )
        if candidate_window is not None:
            scheduler = reference.FourStageScheduler(distribution)
            state = scheduler._initial_state()
            for action_index, action in enumerate(history):
                visible = reference.candidate_window_visible_eids(
                    state.c2, state.c3, state.remaining, candidate_window
                )
                if not reference.action_within_candidate_window(action, visible):
                    raise SystemExit(
                        f"{path}: action {action_index} violates restricted "
                        f"window {candidate_window}"
                    )
                state = reference.apply_action(state, action)
            if state.remaining:
                raise SystemExit(f"{path}: restricted witness is not terminal")
        row.update(
            {
                "actions": branch["actions"],
                "best_reference_ticks": _ticks(replay_cc, quantum),
                "certified_lower_bound_ticks": _ticks(replay_cc, quantum),
                "proven_optimal": True,
                "optimality_gap": 0.0,
                "termination": (
                    "restricted_window_feasible_optimal_witness"
                    if candidate_window is not None
                    else "root_group_target_frontier_feasible_optimal"
                ),
                "history_replay_valid": True,
                "expansions": int(branch["expansions"]),
                "generated_states": int(branch["generated"]),
                "search_runtime_s": float(branch["runtime_s"]),
            }
        )
        evidence.update(
            {
                "result": "frontier_feasible",
                "feasible_group_index": int(branch["group_index"]),
                "semantic_group_key": branch["semantic_group_key"],
            }
        )
        if candidate_window is not None:
            evidence["result"] = "restricted_window_feasible_witness"
            evidence["candidate_window"] = list(candidate_window)
    else:
        if any(
            branch["feasible"] or not branch["exhaustive"]
            for branch in payload["branches"]
        ):
            raise SystemExit(f"{path}: exhaustive summary has unresolved branch")
        prior_history = tuple(
            deserialize_action(action) for action in prior["actions"]
        )
        replay_cc = reference.validate_schedule_history(prior_history, distribution)
        prior_ub = int(Fraction(prior["best_reference_ticks"]) * quantum)
        if replay_cc != prior_ub:
            raise SystemExit(
                f"{path}: prior history replay {replay_cc} != UB {prior_ub}"
            )
        advanced_lb = target + quantum
        if advanced_lb > prior_ub:
            raise SystemExit(f"{path}: advanced LB exceeds replay-valid UB")
        proven = advanced_lb == prior_ub
        row.update(
            {
                "certified_lower_bound_ticks": _ticks(advanced_lb, quantum),
                "proven_optimal": proven,
                "optimality_gap": (
                    0.0
                    if proven
                    else float(Fraction(prior_ub - advanced_lb, advanced_lb))
                ),
                "termination": (
                    "root_groups_frontier_infeasible_incumbent_optimal"
                    if proven
                    else "root_groups_frontier_infeasible_lb_advanced"
                ),
                "history_replay_valid": True,
                "expansions": int(summary["total_expansions"]),
                "generated_states": int(summary["total_generated"]),
            }
        )
        evidence["result"] = "frontier_exhaustive_infeasible"
        evidence["advanced_lower_bound_ticks"] = _ticks(advanced_lb, quantum)
    row["root_campaign_evidence"] = evidence
    return row


def _payload_rows(
    path: Path, payload: dict, *, allow_restricted_witness: bool = False
) -> list[dict]:
    if isinstance(payload.get("cases"), list):
        return payload["cases"]
    return [
        _campaign_row(
            path,
            payload,
            allow_restricted_witness=allow_restricted_witness,
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-restricted-witness",
        action="store_true",
        help=(
            "extract only feasible optimum histories from restricted-window "
            "campaigns; never advance a global lower bound from failure"
        ),
    )
    args = parser.parse_args()

    candidates_by_name: dict[str, list[tuple[dict, str]]] = {}
    order: list[str] = []
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in _payload_rows(
            path,
            payload,
            allow_restricted_witness=args.allow_restricted_witness,
        ):
            if not row.get("history_replay_valid"):
                raise SystemExit(f"{path}: unvalidated history for {row['name']}")
            name = row["name"]
            if name not in candidates_by_name:
                order.append(name)
                candidates_by_name[name] = []
            candidates_by_name[name].append((row, str(path)))

    rows = []
    for name in order:
        candidates = candidates_by_name[name]
        ub_row, ub_source = min(
            candidates,
            key=lambda item: Fraction(item[0]["best_reference_ticks"]),
        )
        lb_row, lb_source = max(
            candidates,
            key=lambda item: Fraction(item[0]["certified_lower_bound_ticks"]),
        )
        upper_bound = Fraction(ub_row["best_reference_ticks"])
        lower_bound = Fraction(lb_row["certified_lower_bound_ticks"])
        if lower_bound > upper_bound:
            raise SystemExit(
                f"{name}: merged lower bound {lower_bound} exceeds "
                f"validated upper bound {upper_bound}"
            )
        merged = dict(ub_row)
        merged["certified_lower_bound_ticks"] = str(lower_bound)
        merged["proven_optimal"] = lower_bound == upper_bound
        merged["optimality_gap"] = (
            0.0
            if lower_bound == upper_bound
            else float((upper_bound - lower_bound) / lower_bound)
        )
        if lower_bound == upper_bound and not ub_row.get("proven_optimal", False):
            merged["termination"] = "merged_independent_bounds_equal"
        merged["selected_proof_source"] = ub_source
        merged["selected_history_source"] = ub_source
        merged["selected_lower_bound_source"] = lb_source
        merged["merged_proof_sources"] = [source for _row, source in candidates]
        rows.append(merged)

    proven = [row for row in rows if row["proven_optimal"]]
    unproven = [row for row in rows if not row["proven_optimal"]]
    grid = [row for row in rows if row["origin"] == "systematic_grid"]

    def grouped(field: str) -> dict[str, dict[str, int]]:
        result = {}
        for value in sorted({str(row[field]) for row in grid}):
            members = [row for row in grid if str(row[field]) == value]
            result[value] = {
                "cases": len(members),
                "proven_optimal": sum(row["proven_optimal"] for row in members),
                "unproven": sum(not row["proven_optimal"] for row in members),
            }
        return result

    summary = {
        "cases": len(rows),
        "proven_optimal": len(proven),
        "unproven": len(unproven),
        "proof_sources": dict(
            sorted(Counter(row["selected_proof_source"] for row in rows).items())
        ),
        "grid_by_hot_experts": grouped("hot_experts"),
        "grid_by_batch_tokens": grouped("batch_tokens"),
        "grid_by_profile": grouped("profile"),
        "unproven_cases": [row["name"] for row in unproven],
    }
    output = {
        "schema": "top4_bottom2_directed_proof_merged_v1",
        "proof_model": "explicit_dma_lane_four_stage_anytime",
        "complete": True,
        "summary": summary,
        "cases": rows,
    }
    _atomic_write(args.output, output)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
