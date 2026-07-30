#!/usr/bin/env python3
"""Compare simple N-outer baselines on frozen OLMoE distributions."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from four_stage_scheduler import SCHEDULE_TIME_QUANTUM_CC

from .baselines import fixed_lane_lpt, paired_lpt_mode_search
from .block_golden import replay_best_policy
from .run_evaluation import DEFAULT_CASE_FILE, _indices, _work_signature


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
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    parser.add_argument("--indices", type=_indices, default=(1, 3, 8, 10, 46))
    parser.add_argument("--mode-beam-width", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = json.loads(args.case_file.read_text(encoding="utf-8"))
    results = []
    for index in args.indices:
        case = source["cases"][index - 1]
        counts = tuple(int(value) for value in case["counts"])
        started = time.perf_counter()
        policies = [fixed_lane_lpt(counts)]
        policies.extend(
            paired_lpt_mode_search(
                counts,
                cluster1_order=order,
                beam_width=args.mode_beam_width,
            )
            for order in ("descending", "ascending")
        )
        policy_items = []
        for policy in policies:
            replay = replay_best_policy(policy.node.history)
            policy_items.append(
                {
                    "name": policy.name,
                    "macro_cc": policy.node.makespan_cc,
                    "block_cc": replay.makespan_cc,
                    "block_policy": replay.policy.value,
                    "macro_minus_block_cc": (
                        policy.node.makespan_cc - replay.makespan_cc
                    ),
                    "cluster_lengths": [
                        len(policy.cluster_eids[0]), len(policy.cluster_eids[1])
                    ],
                    "history_validated": policy.history_validated,
                    "block_validated": replay.history_validated,
                }
            )
        best = min(policy_items, key=lambda item: (item["block_cc"], item["name"]))
        four_stage_cc = int(case["best_reference_ticks"]) * SCHEDULE_TIME_QUANTUM_CC
        item = {
            "catalog_index": index,
            "name": case["name"],
            "counts": list(counts),
            "active_experts": len(counts),
            "work_signature": _work_signature(counts),
            "four_stage_certified_cc": four_stage_cc,
            "policies": policy_items,
            "best_nouter_policy": best["name"],
            "best_nouter_block_cc": best["block_cc"],
            "best_nouter_minus_four_stage_cc": best["block_cc"] - four_stage_cc,
            "runtime_s": time.perf_counter() - started,
        }
        results.append(item)
        print(
            f"[{index:02d}] active={len(counts)} best={best['name']} "
            f"nouter={best['block_cc']} four_stage={four_stage_cc} "
            f"runtime={item['runtime_s']:.2f}s",
            flush=True,
        )
    payload = {
        "schema": "coarse_nouter_baseline_matrix_v1",
        "contracts": {
            "same_atomic_work_checked": True,
            "fixed_lane_lpt": "no_lane_borrowing",
            "paired_mode_search": "fixed_lpt_partition_mode_only_search",
            "four_stage": "preexisting_certified_result",
            "nouter_global_optimality_claim": False,
        },
        "config": {
            "indices": list(args.indices),
            "mode_beam_width": args.mode_beam_width,
            "joint_mode_budget_per_skeleton": 8,
        },
        "cases": results,
    }
    if args.output:
        _write_atomic(args.output, payload)
        print(f"result_written={args.output}", flush=True)


if __name__ == "__main__":
    main()

