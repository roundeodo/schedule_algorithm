#!/usr/bin/env python3
"""Select an N-outer mapping and DMA-prefetch schedule."""

from __future__ import annotations

import argparse

from .model import ExpertDescriptor, NOuterConfig, NOuterSimulator, default_config
from .scheduler import NOuterScheduler, SchedulerMode, SchedulerOptions


def _parse_tokens(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("--tokens requires positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens",
        type=_parse_tokens,
        default=_parse_tokens("16,4,2,2,2,2,2,2"),
    )
    parser.add_argument(
        "--mode",
        type=SchedulerMode,
        choices=tuple(SchedulerMode),
        default=SchedulerMode.HYBRID,
    )
    parser.add_argument("--exact-top", type=int, default=10)
    parser.add_argument("--max-group-tokens", type=int)
    args = parser.parse_args()

    config = default_config()
    if args.max_group_tokens is not None:
        config = NOuterConfig(
            phases=config.phases,
            dma_policy=config.dma_policy,
            force_initial_split=config.force_initial_split,
            max_group_tokens_per_cluster=args.max_group_tokens,
        )
    experts = tuple(
        ExpertDescriptor(eid=eid, ntokens=ntokens)
        for eid, ntokens in enumerate(args.tokens)
    )
    scheduler = NOuterScheduler(
        NOuterSimulator(config),
        SchedulerOptions(mode=args.mode, exact_top_k=args.exact_top),
    )
    result = scheduler.schedule(experts)
    group = result.candidate.group

    print(f"mode={result.mode.value}")
    print(
        "cluster0="
        + ",".join(f"e{expert.eid}:{expert.ntokens}" for expert in group.cluster0)
    )
    print(
        "cluster1="
        + ",".join(f"e{expert.eid}:{expert.ntokens}" for expert in group.cluster1)
    )
    print(
        f"makespan_cc={result.cost.makespan_cc} "
        f"lower_bound_cc={result.cost.lower_bound_cc} "
        f"initial_prime_cc={result.cost.initial_prime_cc} "
        f"steady_stall_cc={result.cost.steady_stall_cc}"
    )
    print(
        f"bw_valid={result.bandwidth.valid} "
        f"peak_bw={result.bandwidth.peak_bandwidth_bytes_per_cc} "
        f"lane_busy_cc={result.bandwidth.lane_busy_cc}"
    )
    print(
        f"lookahead_loads={result.prefetch.lookahead_loads} "
        f"overlapped={result.prefetch.overlapped_loads} "
        f"fully_hidden={result.prefetch.fully_hidden_loads} "
        f"phase_boundary_pf={result.prefetch.phase_boundary_prefetches} "
        f"exposed_load_cc={result.prefetch.exposed_load_cc}"
    )
    print(
        f"generated={result.generated_candidates} "
        f"evaluated={result.evaluated_candidates} "
        f"exact={result.exact_candidates} "
        f"dma_optimal={result.dma_optimal_for_selected_candidate} "
        f"bank_optimal={result.optimal_within_generated_bank} "
        f"partition_complete={result.partition_space_complete} "
        f"permutation_complete={result.permutation_space_complete}"
    )


if __name__ == "__main__":
    main()
