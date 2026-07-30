#!/usr/bin/env python3
"""Run the independent N-outer candidate search on one token distribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .model import (
    CandidateResult,
    ExpertDescriptor,
    NOuterConfig,
    NOuterSimulator,
    default_config,
)
from .reference import ExactDmaPlanner, ExactDmaResult
from .search import SearchResult, generate_partition_candidates, search_candidates


def _parse_tokens(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("--tokens requires positive comma-separated integers")
    return values


def _candidate_payload(result: CandidateResult) -> dict[str, object]:
    group = result.candidate.group
    return {
        "label": result.candidate.label,
        "cluster0": [
            {"eid": expert.eid, "ntokens": expert.ntokens}
            for expert in group.cluster0
        ],
        "cluster1": [
            {"eid": expert.eid, "ntokens": expert.ntokens}
            for expert in group.cluster1
        ],
        "makespan_cc": result.makespan_cc,
        "lower_bound_cc": result.lower_bound_cc,
        "compute_lower_bound_cc": result.compute_lower_bound_cc,
        "dma_lower_bound_cc": result.dma_lower_bound_cc,
        "initial_wait_cc": list(result.initial_wait_cc),
        "steady_stall_cc": list(result.steady_stall_cc),
        "compute_utilization": result.compute_utilization,
        "dma_lane_utilization": result.dma_lane_utilization,
        "history_validated": result.history_validated,
    }


def _exact_payload(result: ExactDmaResult) -> dict[str, object]:
    group = result.candidate.group
    return {
        "label": result.candidate.label,
        "cluster0": [
            {"eid": expert.eid, "ntokens": expert.ntokens}
            for expert in group.cluster0
        ],
        "cluster1": [
            {"eid": expert.eid, "ntokens": expert.ntokens}
            for expert in group.cluster1
        ],
        "makespan_cc": result.makespan_cc,
        "relaxed_lower_bound_cc": result.relaxed_lower_bound_cc,
        "heuristic_makespan_cc": result.heuristic_makespan_cc,
        "heuristic_gap_cc": result.heuristic_gap_cc,
        "expanded_states": result.expanded_states,
        "memoized_states": result.memoized_states,
        "proven_optimal": result.proven_optimal,
        "plan_validated": result.plan_validated,
        "plan": [
            {
                "advance_cc": step.advance_cc,
                "grants": [
                    {"cluster": cluster, "lane_mask": lane_mask}
                    for cluster, lane_mask in step.grants
                ],
            }
            for step in result.decisions
        ],
    }


def _source_hashes() -> dict[str, str]:
    source_dir = Path(__file__).resolve().parent
    names = ("model.py", "search.py", "reference.py", "run_search.py")
    return {
        name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
        for name in names
    }


def _write_result(
    path: Path,
    *,
    args: argparse.Namespace,
    config: NOuterConfig,
    candidate_count: int,
    search_result: SearchResult,
    exact_scope: str | None,
    heuristic_results: tuple[CandidateResult, ...],
    exact_results: list[ExactDmaResult],
) -> None:
    payload = {
        "schema": "n_outer_candidate_reference",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, "-m", "n_outer_scheduler.run_search", *sys.argv[1:]],
        "source_sha256": _source_hashes(),
        "input": {"tokens": list(args.tokens)},
        "config": {
            "dma_policy": config.dma_policy.value,
            "force_initial_split": config.force_initial_split,
            "max_group_tokens_per_cluster": config.max_group_tokens_per_cluster,
            "phases": [
                {
                    "name": phase.name,
                    "block_count": phase.block_count,
                    "weight_block_bytes": phase.weight_block_bytes,
                    "m4_compute_cc": phase.m4_compute_cc,
                    "m2_compute_cc": phase.m2_compute_cc,
                }
                for phase in config.phases
            ],
        },
        "candidate_space": {
            "generated": candidate_count,
            "evaluated": search_result.generated_candidates,
            "rejected": search_result.rejected_candidates,
            "partition_space_complete": len(args.tokens) <= 16,
            "order_variants": ["input", "tokens_desc", "tokens_asc"],
            "permutation_space_complete": False,
        },
        "exact_scope": exact_scope,
        "heuristic_results": [
            _candidate_payload(result) for result in heuristic_results
        ],
        "exact_results": [_exact_payload(result) for result in exact_results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens",
        type=_parse_tokens,
        default=_parse_tokens("16,4,2,2,2,2,2,2"),
        help="active-expert token counts, default: 16,4,2,2,2,2,2,2",
    )
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--max-group-tokens", type=int)
    parser.add_argument(
        "--exact-top",
        type=int,
        default=0,
        help="prove DMA-grant optimality for top N heuristic candidates; -1 checks all",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write an auditable JSON result with source hashes and exact grant plans",
    )
    args = parser.parse_args()
    if args.top <= 0 or args.exact_top < -1:
        parser.error("--top must be positive and --exact-top must be -1 or non-negative")

    config = default_config()
    if args.max_group_tokens is not None:
        config = type(config)(
            phases=config.phases,
            dma_policy=config.dma_policy,
            force_initial_split=config.force_initial_split,
            max_group_tokens_per_cluster=args.max_group_tokens,
        )
    experts = tuple(
        ExpertDescriptor(eid=eid, ntokens=ntokens)
        for eid, ntokens in enumerate(args.tokens)
    )
    simulator = NOuterSimulator(config)
    candidates = generate_partition_candidates(experts)
    ranking_depth = (
        len(candidates)
        if args.exact_top == -1
        else max(args.top, args.exact_top)
    )
    result = search_candidates(simulator, candidates, top_k=ranking_depth)

    print(
        f"evaluated_candidates={result.generated_candidates} "
        f"rejected_candidates={result.rejected_candidates}"
    )
    for rank, candidate_result in enumerate(result.ranked[: args.top], start=1):
        group = candidate_result.candidate.group
        c0 = [expert.ntokens for expert in group.cluster0]
        c1 = [expert.ntokens for expert in group.cluster1]
        print(
            f"rank={rank} c0={c0} c1={c1} "
            f"makespan={candidate_result.makespan_cc} "
            f"lb={candidate_result.lower_bound_cc} "
            f"ratio={candidate_result.over_lower_bound:.6f} "
            f"initial_wait={candidate_result.initial_wait_cc} "
            f"steady_stall={candidate_result.steady_stall_cc} "
            f"compute_util={candidate_result.compute_utilization:.6f} "
            f"dma_util={candidate_result.dma_lane_utilization:.6f} "
            f"validated={candidate_result.history_validated}"
        )

    exact_results: list[ExactDmaResult] = []
    exact_scope: str | None = None
    if args.exact_top:
        targets = (
            result.ranked
            if args.exact_top == -1
            else result.ranked[: args.exact_top]
        )
        exact_results = [
            ExactDmaPlanner(simulator).evaluate(item.candidate) for item in targets
        ]
        exact_results.sort(key=lambda item: (item.makespan_cc, item.candidate.label))
        exact_scope = (
            "all_generated_candidates"
            if args.exact_top == -1
            else f"heuristic_top_{len(targets)}"
        )
        print(
            f"exact_scope={exact_scope} exact_outer_bank_complete={args.exact_top == -1} "
            "permutation_space_complete=False"
        )
        for rank, exact in enumerate(exact_results[: args.top], start=1):
            group = exact.candidate.group
            c0 = [expert.ntokens for expert in group.cluster0]
            c1 = [expert.ntokens for expert in group.cluster1]
            print(
                f"exact_rank={rank} c0={c0} c1={c1} "
                f"makespan={exact.makespan_cc} "
                f"heuristic={exact.heuristic_makespan_cc} "
                f"improvement={exact.heuristic_gap_cc} "
                f"lb={exact.relaxed_lower_bound_cc} "
                f"states={exact.expanded_states} "
                f"dma_proven={exact.proven_optimal} "
                f"plan_validated={exact.plan_validated}"
            )

    if args.output is not None:
        _write_result(
            args.output,
            args=args,
            config=config,
            candidate_count=len(candidates),
            search_result=result,
            exact_scope=exact_scope,
            heuristic_results=result.ranked,
            exact_results=exact_results,
        )
        print(f"result_written={args.output}")


if __name__ == "__main__":
    main()
