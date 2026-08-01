#!/usr/bin/env python3
"""Checkpoint exact target feasibility independently by canonical root child.

The union of the generated root children is the same complete future-distinct
action set used by ``FourStageScheduler.run_target_feasibility``.  A target is
declared infeasible only when every root-child search exhausts OPEN.  Any
feasible result is replayed in the full explicit-DMA model before it is saved.
Timeout fragments are diagnostics, never proofs, and are rerun by default.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import four_stage_scheduler as reference
from run_four_stage_reference import deserialize_action, serialize_action


HERE = Path(__file__).resolve().parent


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _ticks(cc: int) -> str:
    value = Fraction(cc, reference.SCHEDULE_TIME_QUANTUM_CC)
    return str(value.numerator) if value.denominator == 1 else str(value)


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _root_children(
    scheduler: reference.FourStageScheduler,
    target: int,
    candidate_window: tuple[int, int] | None = None,
) -> list[reference.BeamState]:
    initial = scheduler._initial_state()
    target_capacity = 2 * target
    work_budget = target_capacity - initial.cluster_work_cc
    generation_remaining = (
        initial.remaining
        if candidate_window is None
        else reference.candidate_window_remaining(
            initial.c2, initial.c3, initial.remaining, candidate_window
        )
    )
    hidden_work = (
        0
        if candidate_window is None
        else reference._minimum_cluster_work(initial.remaining)
        - reference._minimum_cluster_work(generation_remaining)
    )
    generator_bounds = {
        "work_budget_cc": work_budget - hidden_work,
        "capacity_limit_cc": target_capacity - hidden_work,
    }
    actions = reference.gen_stage_actions(
        initial.c2,
        initial.c3,
        generation_remaining,
        **generator_bounds,
    )
    if scheduler.enable_prefetch:
        actions += reference.gen_prefetch_actions(
            initial.c2,
            initial.c3,
            generation_remaining,
            **generator_bounds,
        )
    if candidate_window is not None:
        visible_eids = reference.candidate_window_visible_eids(
            initial.c2, initial.c3, initial.remaining, candidate_window
        )
        actions = [
            action
            for action in actions
            if reference.action_within_candidate_window(action, visible_eids)
        ]

    best_by_future: dict[tuple, reference.BeamState] = {}
    for action in actions:
        child = reference.apply_action(initial, action)
        if child.f_score > target:
            continue
        if (
            child.c2.task_end
            + child.c3.task_end
            + reference._minimum_cluster_work(child.remaining)
            > target_capacity
        ):
            continue
        key = child.fingerprint()
        previous = best_by_future.get(key)
        if (
            previous is None
            or child.cluster_work_cc < previous.cluster_work_cc
        ):
            best_by_future[key] = child

    def ordering(state: reference.BeamState) -> tuple:
        components = reference.state_lower_bound_components(
            state.c2, state.c3, state.remaining
        )
        action = state.history[-1]
        return (
            components["dma_capacity_cc"],
            len(state.remaining),
            state.f_score,
            state.g_score,
            action.tag,
            json.dumps(serialize_action(action), sort_keys=True),
        )

    return sorted(best_by_future.values(), key=ordering)


def _semantic_group_key(state: reference.BeamState) -> tuple:
    action = state.history[-1]
    if action.c2_eid >= 0 and action.c2_eid == action.c3_eid:
        return ("SPLIT", *sorted((action.c2_ntok, action.c3_ntok)))
    counts = tuple(
        sorted(ntok for ntok in (action.c2_ntok, action.c3_ntok) if ntok > 0)
    )
    if len(counts) == 2:
        return ("PAIR", *counts)
    if len(counts) == 1:
        return ("SINGLE", counts[0])
    return ("OTHER", action.tag)


def _summary(rows: list[dict], total_groups: int, total_children: int) -> dict:
    feasible = [row for row in rows if row["feasible"]]
    exhaustive = [row for row in rows if row["exhaustive"]]
    unresolved = [
        row for row in rows if not row["feasible"] and not row["exhaustive"]
    ]
    return {
        "root_children": total_children,
        "semantic_root_groups": total_groups,
        "recorded_groups": len(rows),
        "feasible_groups": len(feasible),
        "exhaustive_infeasible_groups": len(exhaustive),
        "unresolved_groups": len(unresolved),
        "all_root_groups_exhaustive": (
            len(rows) == total_groups and len(exhaustive) == total_groups
        ),
        "found_feasible_history": bool(feasible),
        "total_expansions": sum(row["expansions"] for row in rows),
        "total_generated": sum(row["generated"] for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-proof", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--time-limit-s", type=float, required=True)
    parser.add_argument("--rank-mode", default="dma")
    parser.add_argument(
        "--target-ticks",
        help=(
            "explicit target in scheduler ticks; required for a restricted "
            "candidate-window sufficiency campaign"
        ),
    )
    parser.add_argument(
        "--candidate-window",
        type=int,
        nargs=2,
        metavar=("TOP", "BOTTOM"),
        help="restrict every action to TOP+BOTTOM visible experts",
    )
    parser.add_argument("--max-expansions", type=int, default=200_000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--group-index",
        type=int,
        action="append",
        default=[],
        help="one-based semantic root-group index to run; may be repeated",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reuse-timeouts",
        action="store_true",
        help="reuse unresolved fragments instead of rerunning them",
    )
    parser.add_argument(
        "--repeat-until-complete",
        action="store_true",
        help="re-exec this campaign and resume checkpoints until exact completion",
    )
    args = parser.parse_args()
    if (
        args.time_limit_s <= 0
        or args.max_expansions <= 0
        or args.limit < 0
        or any(index <= 0 for index in args.group_index)
        or (
            args.repeat_until_complete
            and args.limit > 0
        )
    ):
        raise SystemExit("invalid non-positive search limit")

    prior = json.loads(args.prior_proof.read_text(encoding="utf-8"))
    prior_by_name = {row["name"]: row for row in prior["cases"]}
    if args.case not in prior_by_name:
        raise SystemExit(f"unknown case {args.case!r}")
    prior_row = prior_by_name[args.case]
    token_dist = {
        eid: int(ntok)
        for eid, ntok in enumerate(prior_row["counts"])
        if int(ntok) > 0
    }
    quantum = reference.SCHEDULE_TIME_QUANTUM_CC
    known_lb = int(Fraction(prior_row["certified_lower_bound_ticks"]) * quantum)
    incumbent = int(Fraction(prior_row["best_reference_ticks"]) * quantum)
    candidate_window = reference.normalize_candidate_window(
        tuple(args.candidate_window) if args.candidate_window is not None else None
    )
    if candidate_window is not None and args.target_ticks is None:
        raise SystemExit(
            "--target-ticks is required with --candidate-window; restricted "
            "failure is not a global lower-bound proof"
        )
    if args.target_ticks is not None:
        target_fraction = Fraction(args.target_ticks) * quantum
        if target_fraction.denominator != 1:
            raise SystemExit("--target-ticks is not an integral scheduler time")
        target = int(target_fraction)
        if target < 0 or target > incumbent:
            raise SystemExit("explicit target must be between zero and the incumbent")
        target_source = "explicit"
    else:
        target = ((known_lb + quantum - 1) // quantum) * quantum
        if target >= incumbent:
            raise SystemExit("frontier target is not below the incumbent")
        target_source = "next_global_frontier"
    if candidate_window is not None and (
        not prior_row.get("proven_optimal")
        or known_lb != incumbent
        or target != incumbent
    ):
        raise SystemExit(
            "restricted-window sufficiency requires an LB=UB prior and its "
            "proved-optimal makespan as the explicit target"
        )

    scheduler = reference.FourStageScheduler(token_dist)
    reference.clear_scheduler_caches()
    children = _root_children(scheduler, target, candidate_window)
    groups_by_key: dict[tuple, list[reference.BeamState]] = {}
    for child in children:
        groups_by_key.setdefault(_semantic_group_key(child), []).append(child)
    groups = sorted(groups_by_key.items(), key=lambda item: item[0])
    manifest = {
        "case": args.case,
        "counts": list(prior_row["counts"]),
        "target_cc": target,
        "target_ticks": _ticks(target),
        "target_source": target_source,
        "campaign_purpose": (
            "restricted_candidate_window_optimum_sufficiency"
            if candidate_window is not None
            else "global_target_frontier_proof"
        ),
        "candidate_window": (
            list(candidate_window) if candidate_window is not None else None
        ),
        "time_quantum_cc": quantum,
        "root_children": len(children),
        "semantic_root_groups": len(groups),
        "semantic_group_sizes": [
            {"key": list(key), "children": len(members)}
            for key, members in groups
        ],
        "rank_mode": args.rank_mode,
        "prior_proof": str(args.prior_proof.resolve()),
        "prior_proof_sha256": _source_hash(args.prior_proof),
        "reference_sha256": _source_hash(HERE / "four_stage_scheduler.py"),
    }
    manifest_id = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    manifest["manifest_id"] = manifest_id
    args.work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.work_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise SystemExit(f"manifest mismatch in {args.work_dir}")
    else:
        _atomic_write(manifest_path, manifest)

    indexed_groups = list(enumerate(groups, 1))
    if args.group_index:
        requested = set(args.group_index)
        missing = requested - {index for index, _group in indexed_groups}
        if missing:
            raise SystemExit(f"unknown group indices: {sorted(missing)}")
        indexed_groups = [
            item for item in indexed_groups if item[0] in requested
        ]
    selected = indexed_groups[: args.limit or None]
    rows_by_index: dict[int, dict] = {}
    found_feasible = False
    started = time.perf_counter()
    for index, (group_key, group_children) in selected:
        fragment = args.work_dir / f"{index:04d}.json"
        checkpoint_path = args.work_dir / f"{index:04d}.checkpoint.pkl"
        row = None
        resume_checkpoint = None
        if fragment.exists():
            candidate = json.loads(fragment.read_text(encoding="utf-8"))
            if candidate.get("manifest_id") != manifest_id:
                raise SystemExit(f"stale fragment {fragment}")
            if (
                candidate.get("feasible")
                or candidate.get("exhaustive")
                or args.reuse_timeouts
            ):
                row = candidate
            elif checkpoint_path.exists():
                expected_hash = candidate.get("checkpoint_sha256")
                actual_hash = _source_hash(checkpoint_path)
                if expected_hash != actual_hash:
                    raise SystemExit(f"checkpoint hash mismatch {checkpoint_path}")
                with checkpoint_path.open("rb") as stream:
                    resume_checkpoint = pickle.load(stream)
        if row is None:
            search_kwargs = (
                {"checkpoint": resume_checkpoint}
                if resume_checkpoint is not None
                else {"initial_states": tuple(group_children)}
            )
            result = scheduler.run_target_feasibility(
                target,
                time_limit_s=args.time_limit_s,
                max_expansions=args.max_expansions,
                rank_mode=args.rank_mode,
                candidate_window=candidate_window,
                **search_kwargs,
            )
            if result.feasible:
                replay = reference.validate_schedule_history(
                    result.history, token_dist
                )
                if replay > target:
                    raise AssertionError("feasible branch history misses target")
            row = {
                "manifest_id": manifest_id,
                "group_index": index,
                "semantic_group_key": list(group_key),
                "root_child_count": len(group_children),
                "root_actions": [
                    serialize_action(child.history[-1])
                    for child in group_children
                ],
                "feasible": result.feasible,
                "exhaustive": result.exhaustive,
                "termination": result.termination,
                "expansions": result.expansions,
                "generated": result.generated,
                "pruned_by_bound": result.pruned_by_bound,
                "open_states": result.open_states,
                "closed_states": result.closed_states,
                "peak_open_states": result.peak_open_states,
                "runtime_s": result.runtime_s,
                "actions": (
                    [serialize_action(action) for action in result.history]
                    if result.feasible
                    else []
                ),
            }
            if result.checkpoint is not None:
                _atomic_pickle(checkpoint_path, result.checkpoint)
                row["checkpoint_file"] = checkpoint_path.name
                row["checkpoint_sha256"] = _source_hash(checkpoint_path)
            _atomic_write(fragment, row)
        rows_by_index[index] = row
        print(
            f"[{index}/{len(groups)}] {group_key} {row['termination']} "
            f"feasible={row['feasible']} exhaustive={row['exhaustive']} "
            f"exp={row['expansions']} open={row['open_states']} "
            f"runtime={row['runtime_s']:.3f}s",
            flush=True,
        )
        if row["feasible"]:
            found_feasible = True
            break
        reference.clear_scheduler_caches()
        payload = {
            "schema": "target_root_branch_proof_v1",
            "complete": False,
            "manifest": manifest,
            "summary": _summary(
                list(rows_by_index.values()), len(groups), len(children)
            ),
            "branches": [rows_by_index[i] for i in sorted(rows_by_index)],
        }
        _atomic_write(args.output, payload)

    rows = [rows_by_index[i] for i in sorted(rows_by_index)]
    summary = _summary(rows, len(groups), len(children))
    selected_groups_exhaustive = bool(args.group_index) and all(
        rows_by_index.get(index, {}).get("exhaustive", False)
        for index in args.group_index
    )
    complete = (
        found_feasible
        or summary["all_root_groups_exhaustive"]
        or selected_groups_exhaustive
    )
    payload = {
        "schema": "target_root_branch_proof_v1",
        "complete": complete,
        "manifest": manifest,
        "summary": summary,
        "branches": rows,
    }
    _atomic_write(args.output, payload)
    print(json.dumps(summary, indent=2))
    print(f"elapsed_s={time.perf_counter() - started:.3f}")
    print(f"wrote {args.output}")
    if args.repeat_until_complete and not complete:
        print("campaign unresolved; re-execing from checkpoints", flush=True)
        os.execv(sys.executable, [sys.executable, *sys.argv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
