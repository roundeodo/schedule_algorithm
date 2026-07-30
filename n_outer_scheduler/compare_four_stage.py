#!/usr/bin/env python3
"""Paired, auditable comparison of the N-outer and four-stage references.

This diagnostic intentionally uses small active-expert sets.  N-outer
enumerates every ordered no-SPLIT two-cluster partition and exactly solves the
DMA grant sequence for every mapping.  The four-stage side uses its certified
anytime search and validates the returned history.  The script first asserts
that both models account for identical compute work and weight bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from four_stage_scheduler import (
    WEIGHT_BYTES_TOTAL,
    FourStageScheduler,
    _best_task_time,
    validate_schedule_history,
)

from .model import ExpertDescriptor, NOuterSimulator, default_config
from .reference import ExactDmaPlanner
from .search import generate_partition_candidates


DEFAULT_CASES = (
    (2, 2),
    (4, 4),
    (8, 4),
    (16, 4, 2, 2),
    (16, 16, 4, 4),
)


def _parse_cases(value: str) -> tuple[tuple[int, ...], ...]:
    cases = tuple(
        tuple(int(token) for token in case.split(",") if token.strip())
        for case in value.split(";")
        if case.strip()
    )
    if not cases or any(not case or any(token <= 0 for token in case) for case in cases):
        raise argparse.ArgumentTypeError(
            "cases must be semicolon-separated positive token lists"
        )
    return cases


def _work_signature(tokens: tuple[int, ...]) -> dict[str, int]:
    config = default_config()
    experts = tuple(
        ExpertDescriptor(eid=eid, ntokens=ntokens)
        for eid, ntokens in enumerate(tokens)
    )
    nouter_compute = sum(
        phase.block_count * expert.compute_cc(phase)
        for expert in experts
        for phase in config.phases
    )
    four_stage_compute = sum(_best_task_time(ntokens) for ntokens in tokens)
    nouter_weight = len(experts) * sum(
        phase.block_count * phase.weight_block_bytes for phase in config.phases
    )
    four_stage_weight = len(experts) * WEIGHT_BYTES_TOTAL
    if nouter_compute != four_stage_compute:
        raise AssertionError("N-outer and four-stage compute work differ")
    if nouter_weight != four_stage_weight:
        raise AssertionError("N-outer and four-stage weight bytes differ")
    return {
        "compute_cc": nouter_compute,
        "weight_bytes": nouter_weight,
    }


def _nouter_exact(tokens: tuple[int, ...], exhaustive_limit: int) -> dict[str, object]:
    experts = tuple(
        ExpertDescriptor(eid=eid, ntokens=ntokens)
        for eid, ntokens in enumerate(tokens)
    )
    simulator = NOuterSimulator(default_config())
    candidates = generate_partition_candidates(
        experts,
        exhaustive_limit=exhaustive_limit,
        exhaustive_permutation_limit=exhaustive_limit,
    )
    started = time.perf_counter()
    exact = [ExactDmaPlanner(simulator).evaluate(candidate) for candidate in candidates]
    best = min(exact, key=lambda result: (result.makespan_cc, result.candidate.label))
    if not all(result.proven_optimal and result.plan_validated for result in exact):
        raise AssertionError("an N-outer fixed-candidate exact search was not validated")
    return {
        "makespan_cc": best.makespan_cc,
        "candidate_count": len(candidates),
        "cluster0": [expert.ntokens for expert in best.candidate.group.cluster0],
        "cluster1": [expert.ntokens for expert in best.candidate.group.cluster1],
        "candidate_label": best.candidate.label,
        "fixed_candidate_dma_optimal": True,
        "ordered_partition_space_complete": True,
        "split_supported": False,
        "history_validated": best.schedule.history_validated,
        "expanded_states": sum(result.expanded_states for result in exact),
        "runtime_s": time.perf_counter() - started,
    }


def _four_stage_exact(
    tokens: tuple[int, ...], time_limit_s: float, max_expansions: int
) -> dict[str, object]:
    distribution = {eid: ntokens for eid, ntokens in enumerate(tokens)}
    started = time.perf_counter()
    result = FourStageScheduler(distribution, beam_width=64).run_anytime(
        time_limit_s=time_limit_s,
        max_expansions=max_expansions,
        target_gap=0.0,
    )
    validated = validate_schedule_history(tuple(result.history), distribution)
    if validated != result.makespan:
        raise AssertionError("four-stage history does not reproduce its makespan")
    return {
        "makespan_cc": result.makespan,
        "lower_bound_cc": result.lower_bound,
        "proven_optimal": result.proven_optimal,
        "optimality_gap": result.optimality_gap,
        "history_validated": True,
        "uses_split": any("SPLIT" in action.tag for action in result.history),
        "actions": len(result.history),
        "expansions": result.expansions,
        "termination": result.termination,
        "runtime_s": time.perf_counter() - started,
    }


def _source_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    sources = {
        "n_outer/model.py": directory / "model.py",
        "n_outer/search.py": directory / "search.py",
        "n_outer/reference.py": directory / "reference.py",
        "n_outer/compare_four_stage.py": Path(__file__).resolve(),
        "four_stage_scheduler.py": directory.parent / "four_stage_scheduler.py",
    }
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sources.items()
    }


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=_parse_cases,
        default=DEFAULT_CASES,
        help='for example "2,2;4,4;16,4,2,2"',
    )
    parser.add_argument("--exhaustive-limit", type=int, default=6)
    parser.add_argument("--four-stage-time-limit", type=float, default=30.0)
    parser.add_argument("--four-stage-max-expansions", type=int, default=200_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(len(case) > args.exhaustive_limit for case in args.cases):
        parser.error("every case must fit --exhaustive-limit")

    cases: list[dict[str, object]] = []
    for tokens in args.cases:
        work = _work_signature(tokens)
        nouter = _nouter_exact(tokens, args.exhaustive_limit)
        four_stage = _four_stage_exact(
            tokens,
            args.four_stage_time_limit,
            args.four_stage_max_expansions,
        )
        delta = int(nouter["makespan_cc"]) - int(four_stage["makespan_cc"])
        item = {
            "tokens": list(tokens),
            "work_signature": work,
            "n_outer": nouter,
            "four_stage": four_stage,
            "delta_cc": delta,
            "n_outer_over_four_stage": (
                float(nouter["makespan_cc"]) / float(four_stage["makespan_cc"])
            ),
            "directly_comparable_search_space": not bool(four_stage["uses_split"]),
        }
        cases.append(item)
        print(
            f"tokens={tokens} n_outer={nouter['makespan_cc']} "
            f"four_stage={four_stage['makespan_cc']} delta={delta} "
            f"four_optimal={four_stage['proven_optimal']} "
            f"four_split={four_stage['uses_split']}"
        )

    payload = {
        "schema": "n_outer_vs_four_stage_paired_v1",
        "command": [sys.executable, "-m", "n_outer_scheduler.compare_four_stage", *sys.argv[1:]],
        "contracts": {
            "startup": "cold_first_weight_block",
            "n_outer_stream_order": "phase_block_expert",
            "n_outer_candidate_scope": "all ordered no-split partitions",
            "four_stage_cache": "empty",
        },
        "source_sha256": _source_hashes(),
        "cases": cases,
    }
    if args.output is not None:
        _write_atomic(args.output, payload)
        print(f"result_written={args.output}")


if __name__ == "__main__":
    main()
