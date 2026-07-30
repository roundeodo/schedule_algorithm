#!/usr/bin/env python3
"""Exhaust one deterministic root branch below a validated global UB.

This is an exact branch-and-bound decision search.  It uses the complete
four-stage action generator and admissible ``f_score`` pruning only.  The
global incumbent history need not extend the selected root action: its
makespan is used purely as a strict threshold, while any newly found better
history necessarily extends the selected branch.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from fractions import Fraction
import heapq
from itertools import count
import json
import os
from pathlib import Path
import time

import four_stage_scheduler as reference
import prove_top4_bottom2_directed as proof
from run_four_stage_reference import deserialize_action, serialize_action
import scheduler_top4_bottom2_policy as policy


TICK_CC = policy.TICK_CC


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _ticks(cc: int) -> str:
    value = Fraction(cc, TICK_CC)
    return str(value.numerator) if value.denominator == 1 else str(value)


def _root_children(scheduler: reference.FourStageScheduler, upper_bound: int):
    root = scheduler._initial_state()
    actions = reference.gen_stage_actions(root.c2, root.c3, root.remaining)
    if scheduler.enable_prefetch:
        actions += reference.gen_prefetch_actions(root.c2, root.c3, root.remaining)
    unique = {}
    for action in actions:
        child = reference.apply_action(root, action)
        if child.f_score >= upper_bound:
            continue
        fingerprint = child.fingerprint()
        previous = unique.get(fingerprint)
        if previous is None or child.f_score < previous.f_score:
            unique[fingerprint] = child
    return sorted(
        unique.values(),
        key=lambda state: (
            reference.completion_estimate(state),
            state.f_score,
            state.g_score,
            state.history[-1].tag,
        ),
    )


@dataclass(frozen=True)
class BranchResult:
    input_upper_bound_cc: int
    best_makespan_cc: int
    best_history: tuple
    found_better: bool
    certified_lower_bound_cc: int
    proven_branch_optimum: bool
    proven_no_schedule_below_input_ub: bool
    expansions: int
    generated_states: int
    pruned_by_bound: int
    runtime_s: float
    termination: str


def search_branch(
    initial: reference.BeamState,
    *,
    upper_bound: int,
    time_limit_s: float | None,
    max_expansions: int | None,
) -> BranchResult:
    started = time.perf_counter()
    best_makespan = upper_bound
    best_history = ()
    rank_heap = []
    lb_heap = []
    active_entries = set()
    open_by_fp = {}
    closed_best = {}
    serial = count()

    def push(state: reference.BeamState) -> bool:
        if state.f_score >= best_makespan:
            return False
        fingerprint = state.fingerprint()
        closed_lb = closed_best.get(fingerprint)
        if closed_lb is not None and closed_lb <= state.f_score:
            return False
        previous = open_by_fp.get(fingerprint)
        if previous is not None and previous[0] <= state.f_score:
            return False
        if previous is not None:
            active_entries.discard(previous[1])
        entry_id = next(serial)
        open_by_fp[fingerprint] = (state.f_score, entry_id)
        active_entries.add(entry_id)
        estimate = reference.completion_estimate(state)
        heapq.heappush(
            rank_heap,
            (estimate, state.f_score, state.g_score, entry_id, state),
        )
        heapq.heappush(lb_heap, (state.f_score, entry_id))
        return True

    push(initial)
    expansions = 0
    generated = 0
    pruned = 0
    termination = "open_exhausted"

    while rank_heap:
        if time_limit_s is not None and time.perf_counter() - started >= time_limit_s:
            termination = "time_limit"
            break
        if max_expansions is not None and expansions >= max_expansions:
            termination = "expansion_limit"
            break
        while rank_heap and rank_heap[0][3] not in active_entries:
            heapq.heappop(rank_heap)
        if not rank_heap:
            break
        _estimate, _lb, _g, entry_id, state = heapq.heappop(rank_heap)
        active_entries.discard(entry_id)
        fingerprint = state.fingerprint()
        current_open = open_by_fp.get(fingerprint)
        if current_open is not None and current_open[1] == entry_id:
            del open_by_fp[fingerprint]
        if state.f_score >= best_makespan:
            pruned += 1
            continue
        closed_best[fingerprint] = state.f_score
        expansions += 1

        actions = reference.gen_stage_actions(state.c2, state.c3, state.remaining)
        actions += reference.gen_prefetch_actions(state.c2, state.c3, state.remaining)
        generated += len(actions)
        for action in actions:
            child = reference.apply_action(state, action)
            if not child.remaining:
                if child.g_score < best_makespan:
                    best_makespan = child.g_score
                    best_history = child.history
                continue
            if child.f_score >= best_makespan:
                pruned += 1
                continue
            push(child)

    while lb_heap and lb_heap[0][1] not in active_entries:
        heapq.heappop(lb_heap)
    open_lower_bound = lb_heap[0][0] if lb_heap else best_makespan
    exhaustive = not lb_heap or open_lower_bound >= best_makespan
    if exhaustive:
        termination = "branch_optimal"
        certified_lower_bound = best_makespan
    else:
        certified_lower_bound = min(best_makespan, open_lower_bound)
    found_better = bool(best_history)
    return BranchResult(
        input_upper_bound_cc=upper_bound,
        best_makespan_cc=best_makespan,
        best_history=best_history,
        found_better=found_better,
        certified_lower_bound_cc=certified_lower_bound,
        proven_branch_optimum=exhaustive,
        proven_no_schedule_below_input_ub=exhaustive and not found_better,
        expansions=expansions,
        generated_states=generated,
        pruned_by_bound=pruned,
        runtime_s=time.perf_counter() - started,
        termination=termination,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-input", type=Path, required=True)
    parser.add_argument("--prior-proof", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--branch-index", type=int, required=True)
    parser.add_argument("--time-limit-s", type=float, default=60.0)
    parser.add_argument("--max-expansions", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.branch_index < 0 or args.time_limit_s < 0 or args.max_expansions < 0:
        raise SystemExit("invalid branch index or limit")

    cases, _metadata = proof._load_external_cases(args.case_input)
    case_by_name = {case.name: case for case in cases}
    if args.case not in case_by_name:
        raise SystemExit(f"unknown case {args.case!r}")
    prior_payload = json.loads(args.prior_proof.read_text(encoding="utf-8"))
    prior_by_name = {row["name"]: row for row in prior_payload["cases"]}
    if args.case not in prior_by_name:
        raise SystemExit(f"prior proof is missing {args.case!r}")
    prior = prior_by_name[args.case]
    prior_history = tuple(deserialize_action(action) for action in prior["actions"])
    distribution = dict(enumerate(case_by_name[args.case].counts))
    validated_ub = reference.validate_schedule_history(prior_history, distribution)
    stored_ub = Fraction(prior["best_reference_ticks"]) * TICK_CC
    if stored_ub.denominator != 1 or validated_ub != int(stored_ub):
        raise SystemExit("prior history does not reproduce its stored UB")

    reference.clear_scheduler_caches()
    scheduler = reference.FourStageScheduler(distribution)
    children = _root_children(scheduler, validated_ub)
    if args.branch_index >= len(children):
        raise SystemExit(
            f"branch index {args.branch_index} outside 0..{len(children)-1}"
        )
    selected = children[args.branch_index]
    result = search_branch(
        selected,
        upper_bound=validated_ub,
        time_limit_s=args.time_limit_s or None,
        max_expansions=args.max_expansions or None,
    )
    payload = {
        "schema": "four_stage_root_branch_proof_v1",
        "case": args.case,
        "case_input": str(args.case_input.resolve()),
        "prior_proof": str(args.prior_proof.resolve()),
        "root_branches_below_ub": len(children),
        "branch_index": args.branch_index,
        "branch_first_action": serialize_action(selected.history[-1]),
        "branch_initial_lb_ticks": _ticks(selected.f_score),
        "input_upper_bound_ticks": _ticks(result.input_upper_bound_cc),
        "best_makespan_ticks": _ticks(result.best_makespan_cc),
        "certified_lower_bound_ticks": _ticks(result.certified_lower_bound_cc),
        "found_better": result.found_better,
        "proven_branch_optimum": result.proven_branch_optimum,
        "proven_no_schedule_below_input_ub": result.proven_no_schedule_below_input_ub,
        "expansions": result.expansions,
        "generated_states": result.generated_states,
        "pruned_by_bound": result.pruned_by_bound,
        "runtime_s": result.runtime_s,
        "termination": result.termination,
        "better_history": [serialize_action(action) for action in result.best_history],
    }
    _atomic_write(args.output, payload)
    print(json.dumps({key: payload[key] for key in (
        "case",
        "root_branches_below_ub",
        "branch_index",
        "branch_initial_lb_ticks",
        "best_makespan_ticks",
        "certified_lower_bound_ticks",
        "found_better",
        "proven_branch_optimum",
        "expansions",
        "generated_states",
        "runtime_s",
        "termination",
    )}, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
