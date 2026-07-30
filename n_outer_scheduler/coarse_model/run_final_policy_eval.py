#!/usr/bin/env python3
"""Evaluate the frozen RTL-oriented N-outer policy family."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median

from four_stage_scheduler import SCHEDULE_TIME_QUANTUM_CC

from .baselines import (
    fixed_lane_lpt,
    paired_lpt_mode_search,
    split_hot_lpt_mode_search,
)
from .block_golden import replay_best_policy
from .bingo_task_abi import (
    lower_history_to_bingo_tasks,
    replay_bingo_task_program,
)
from .calibration import calibrate_history_mode_choices
from .lowering import lower_history_to_bingo, replay_bingo_program
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


def _source_hashes(case_file: Path) -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    paths = {
        "case_file": case_file,
        "semantics.py": directory / "semantics.py",
        "candidates.py": directory / "candidates.py",
        "baselines.py": directory / "baselines.py",
        "block_golden.py": directory / "block_golden.py",
        "lowering.py": directory / "lowering.py",
        "bingo_task_abi.py": directory / "bingo_task_abi.py",
        "calibration.py": directory / "calibration.py",
        "run_final_policy_eval.py": Path(__file__).resolve(),
    }
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }


def _policy_payload(distribution: tuple[int, ...], result) -> dict[str, object]:
    block = replay_best_policy(result.node.history)
    program = lower_history_to_bingo(distribution, result.node)
    lowered = replay_bingo_program(program)
    task_program = lower_history_to_bingo_tasks(distribution, result.node)
    task_replay = replay_bingo_task_program(task_program)
    if lowered.makespan_cc != task_replay.makespan_cc:
        raise AssertionError("the two independent Bingo lowering replays disagree")
    kinds = Counter(step.plan.kind.value for step in result.node.history)
    mode_profiles = Counter(
        str(
            tuple(
                (task.gate_up.dma.name, task.down.dma.name)
                for task in step.plan.tasks
            )
        )
        for step in result.node.history
    )
    return {
        "name": result.name,
        "macro_cc": result.node.makespan_cc,
        "block_best_cc": block.makespan_cc,
        "block_best_policy": block.policy.value,
        "lowered_macro_order_cc": lowered.makespan_cc,
        "task_abi_macro_order_cc": task_replay.makespan_cc,
        "macro_minus_block_best_cc": result.node.makespan_cc - block.makespan_cc,
        "macro_minus_task_abi_cc": result.node.makespan_cc
        - task_replay.makespan_cc,
        "task_abi_minus_block_best_cc": task_replay.makespan_cc
        - block.makespan_cc,
        "actions": dict(sorted(kinds.items())),
        "mode_profiles": dict(sorted(mode_profiles.items())),
        "macro_records": len(program.records),
        "lowered_block_tasks": len(task_program.tasks),
        "history_validated": result.history_validated,
        "block_validated": block.history_validated,
        "lowering_validated": program.history_validated,
        "task_abi_validated": (
            task_program.history_validated
            and task_replay.dependencies_valid
            and task_replay.resources_valid
            and task_replay.ping_pong_valid
            and task_replay.order_valid
            and task_replay.token_ranges_valid
            and task_replay.makespan_cc == task_program.source_block_makespan_cc
        ),
    }


def _calibration_payload(
    distribution: tuple[int, ...],
    result,
    *,
    pair_mode_policy: str,
    mode_bank_policy: str,
) -> dict[str, object]:
    calibration = calibrate_history_mode_choices(
        distribution,
        result.node,
        mode_budget=4,
        service_order_mode="binding_chain",
        pair_mode_policy=pair_mode_policy,
        mode_bank_policy=mode_bank_policy,
    )
    return {
        "policy": result.name,
        "round_count": len(calibration.rounds),
        "total_ranking_regret_cc": calibration.total_ranking_regret_cc,
        "max_ranking_regret_cc": calibration.max_ranking_regret_cc,
        "nonzero_regret_rounds": calibration.nonzero_regret_rounds,
        "max_abs_macro_timing_error_cc": (
            calibration.max_abs_macro_timing_error_cc
        ),
        "rounds": [asdict(item) for item in calibration.rounds],
    }


def _aggregate(cases: list[dict[str, object]]) -> dict[str, object]:
    main_ratios = [float(case["rtl_main_executable_over_four_stage"]) for case in cases]
    analysis_ratios = [
        float(case["analysis_best_executable_over_four_stage"]) for case in cases
    ]
    policy_counts = Counter(str(case["analysis_best_policy"]) for case in cases)
    result: dict[str, object] = {
        "case_count": len(cases),
        "rtl_main_policy": str(cases[0]["rtl_main_policy"]),
        "rtl_main_ratio_mean": mean(main_ratios),
        "rtl_main_ratio_median": median(main_ratios),
        "rtl_main_ratio_max": max(main_ratios),
        "analysis_best_ratio_mean": mean(analysis_ratios),
        "analysis_best_ratio_median": median(analysis_ratios),
        "analysis_best_ratio_max": max(analysis_ratios),
        "analysis_best_policy_counts": dict(sorted(policy_counts.items())),
        "rtl_main_better_than_fixed_cases": sum(
            int(case["rtl_main_minus_fixed_cc"]) < 0 for case in cases
        ),
        "rtl_main_equal_fixed_cases": sum(
            int(case["rtl_main_minus_fixed_cc"]) == 0 for case in cases
        ),
        "rtl_main_worse_than_fixed_cases": sum(
            int(case["rtl_main_minus_fixed_cc"]) > 0 for case in cases
        ),
        "offline_split_better_than_main_cases": sum(
            int(case["offline_split_minus_rtl_main_cc"]) < 0 for case in cases
        ),
        "rtl_main_macro_error_nonzero_cases": sum(
            int(case["rtl_main_macro_minus_executable_cc"]) != 0
            for case in cases
        ),
        "rtl_main_max_abs_macro_error_cc": max(
            abs(int(case["rtl_main_macro_minus_executable_cc"])) for case in cases
        ),
        "all_task_abi_validated": all(
            all(bool(policy["task_abi_validated"]) for policy in case["policies"])
            for case in cases
        ),
    }
    if all("mode_ranking_calibration" in case for case in cases):
        families = sorted(
            set.intersection(
                *(
                    set(case["mode_ranking_calibration"])
                    for case in cases
                )
            )
        )
        result["mode_ranking"] = {
            family: {
                "total_ranking_regret_cc": sum(
                    int(case["mode_ranking_calibration"][family]["total_ranking_regret_cc"])
                    for case in cases
                ),
                "max_ranking_regret_cc": max(
                    int(case["mode_ranking_calibration"][family]["max_ranking_regret_cc"])
                    for case in cases
                ),
                "nonzero_regret_rounds": sum(
                    int(case["mode_ranking_calibration"][family]["nonzero_regret_rounds"])
                    for case in cases
                ),
            }
            for family in families
        }
    if all("service_order_ablation" in case for case in cases):
        deltas = [
            int(case["service_order_ablation"]["binding_chain_minus_best18_cc"])
            for case in cases
        ]
        result["service_order_ablation"] = {
            "binding_chain_worse_cases": sum(delta > 0 for delta in deltas),
            "binding_chain_equal_cases": sum(delta == 0 for delta in deltas),
            "binding_chain_better_cases": sum(delta < 0 for delta in deltas),
            "max_binding_chain_loss_cc": max(deltas),
            "mean_binding_chain_delta_cc": mean(deltas),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    parser.add_argument(
        "--indices",
        type=_indices,
        help="one-based comma-separated case indices; omit for all cases",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--calibrate-mode-ranking",
        action="store_true",
        help="replay all K4 modes at every no-SPLIT/SPLIT history round",
    )
    parser.add_argument(
        "--calibrate-main-mode-ranking",
        action="store_true",
        help="replay K4 modes only for the deployable no-SPLIT main policy",
    )
    parser.add_argument(
        "--service-order-ablation",
        action="store_true",
        help="compare deterministic binding-chain priority with best-of-18",
    )
    parser.add_argument(
        "--main-pair-mode-policy",
        choices=("all", "no_mixed", "fixed_only"),
        default="no_mixed",
        help="static PAIR-mode filter for the deployable no-SPLIT policy",
    )
    parser.add_argument(
        "--main-mode-bank",
        choices=("bounded_k4", "rtl_symmetric2"),
        default="rtl_symmetric2",
        help="mode generator for the deployable no-SPLIT policy",
    )
    parser.add_argument(
        "--main-score-mode",
        choices=("local", "projected"),
        default="local",
        help="mode score used by the deployable no-SPLIT policy",
    )
    parser.add_argument(
        "--main-mode-tie-break",
        choices=("bank_order", "both_on_tie", "stall"),
        default="bank_order",
        help="tie break after equal macro completion scores",
    )
    args = parser.parse_args()
    source = json.loads(args.case_file.read_text(encoding="utf-8"))
    all_cases = source["cases"]
    indices = args.indices or tuple(range(1, len(all_cases) + 1))
    results = []
    for index in indices:
        case = all_cases[index - 1]
        distribution = tuple(int(value) for value in case["counts"])
        started = time.perf_counter()
        fixed = fixed_lane_lpt(distribution)
        no_split = paired_lpt_mode_search(
            distribution,
            beam_width=1,
            mode_budget=4,
            score_mode=args.main_score_mode,
            service_order_mode="binding_chain",
            tie_break_mode=args.main_mode_tie_break,
            pair_mode_policy=args.main_pair_mode_policy,
            mode_bank_policy=args.main_mode_bank,
        )
        split = split_hot_lpt_mode_search(
            distribution,
            max_hot_experts=1,
            beam_width=1,
            mode_budget=4,
            score_mode="local",
            service_order_mode="binding_chain",
            tie_break_mode="bank_order",
        )
        policy_results = (fixed, no_split, split)
        policies = [
            _policy_payload(distribution, result)
            for result in policy_results
        ]
        selected_index = min(
            range(len(policies)),
            key=lambda index: (
                policies[index]["task_abi_macro_order_cc"],
                policies[index]["macro_cc"],
                policies[index]["name"],
            ),
        )
        selected = policies[selected_index]
        fixed_payload, no_split_payload, split_payload = policies
        four_stage_cc = int(case["best_reference_ticks"]) * SCHEDULE_TIME_QUANTUM_CC
        item = {
            "catalog_index": index,
            "name": case["name"],
            "counts": list(distribution),
            "active_experts": len(distribution),
            "work_signature": _work_signature(distribution),
            "four_stage_certified_cc": four_stage_cc,
            "policies": policies,
            "rtl_main_policy": no_split_payload["name"],
            "rtl_main_macro_cc": no_split_payload["macro_cc"],
            "rtl_main_executable_cc": no_split_payload["task_abi_macro_order_cc"],
            "rtl_main_block_oracle_cc": no_split_payload["block_best_cc"],
            "rtl_main_macro_minus_executable_cc": no_split_payload[
                "macro_minus_task_abi_cc"
            ],
            "rtl_main_arbitration_regret_cc": no_split_payload[
                "task_abi_minus_block_best_cc"
            ],
            "rtl_main_executable_minus_four_stage_cc": no_split_payload[
                "task_abi_macro_order_cc"
            ] - four_stage_cc,
            "rtl_main_executable_over_four_stage": no_split_payload[
                "task_abi_macro_order_cc"
            ] / four_stage_cc,
            "rtl_main_minus_fixed_cc": no_split_payload["task_abi_macro_order_cc"]
            - fixed_payload["task_abi_macro_order_cc"],
            "offline_split_minus_rtl_main_cc": split_payload[
                "task_abi_macro_order_cc"
            ] - no_split_payload["task_abi_macro_order_cc"],
            "analysis_best_policy": selected["name"],
            "analysis_best_macro_cc": selected["macro_cc"],
            "analysis_best_executable_cc": selected["task_abi_macro_order_cc"],
            "analysis_best_block_oracle_cc": selected["block_best_cc"],
            "analysis_best_executable_over_four_stage": selected[
                "task_abi_macro_order_cc"
            ] / four_stage_cc,
            "runtime_s": time.perf_counter() - started,
        }
        if args.calibrate_mode_ranking or args.calibrate_main_mode_ranking:
            item["mode_ranking_calibration"] = {
                "no_split": _calibration_payload(
                    distribution,
                    no_split,
                    pair_mode_policy=args.main_pair_mode_policy,
                    mode_bank_policy=args.main_mode_bank,
                ),
            }
            if args.calibrate_mode_ranking:
                item["mode_ranking_calibration"]["split_top1"] = (
                    _calibration_payload(
                        distribution,
                        split,
                        pair_mode_policy="all",
                        mode_bank_policy="bounded_k4",
                    )
                )
            item["runtime_s"] = time.perf_counter() - started
        if args.service_order_ablation:
            best18 = paired_lpt_mode_search(
                distribution,
                beam_width=1,
                mode_budget=4,
                score_mode=args.main_score_mode,
                service_order_mode="best18",
                tie_break_mode=args.main_mode_tie_break,
                pair_mode_policy=args.main_pair_mode_policy,
                mode_bank_policy=args.main_mode_bank,
            )
            best18_payload = _policy_payload(distribution, best18)
            item["service_order_ablation"] = {
                "binding_chain_macro_cc": no_split_payload["macro_cc"],
                "binding_chain_executable_cc": no_split_payload[
                    "task_abi_macro_order_cc"
                ],
                "best18_macro_cc": best18_payload["macro_cc"],
                "best18_executable_cc": best18_payload["task_abi_macro_order_cc"],
                "binding_chain_minus_best18_cc": no_split_payload[
                    "task_abi_macro_order_cc"
                ] - best18_payload["task_abi_macro_order_cc"],
            }
            item["runtime_s"] = time.perf_counter() - started
        results.append(item)
        print(
            f"[{index:02d}/{len(all_cases)}] active={len(distribution)} "
            f"rtl_main={no_split_payload['task_abi_macro_order_cc']} "
            f"analysis_best={selected['name']}/"
            f"{selected['task_abi_macro_order_cc']} "
            f"four_stage={four_stage_cc} runtime={item['runtime_s']:.2f}s",
            flush=True,
        )
    payload = {
        "schema": "coarse_nouter_final_policy_eval_v3",
        "contracts": {
            "same_atomic_work_checked": True,
            "candidate_structure": "single_lane_lpt_partition_then_paired_heads",
            "joint_mode_budget": (
                2 if args.main_mode_bank == "rtl_symmetric2" else 4
            ),
            "main_mode_bank": args.main_mode_bank,
            "mode_search_beam": 1,
            "mode_score": args.main_score_mode,
            "mode_tie_break": args.main_mode_tie_break,
            "fixed_lane_baseline_is_bank_entry_zero": True,
            "main_pair_mode_policy": args.main_pair_mode_policy,
            "service_order": "one_deterministic_binding_chain_no_order_search",
            "service_order_search_count": 1,
            "rtl_main_policy": (
                "no_split_lpt_pair_heads_local_symmetric2"
                if args.main_mode_bank == "rtl_symmetric2"
                else "no_split_lpt_pair_heads_local_k4"
            ),
            "split_scope": "offline_top1_aligned_cut_full_history_ablation",
            "split_deployability_claim": False,
            "macro_model": "ready_only_fixed_block_recurrence",
            "block_model": "independent_calibration_and_lowering_replay",
            "bingo_task_abi": "macro_records_lowered_to_dependency_only_block_tasks",
            "reported_nouter_performance": "dependency_only_task_replay_macro_order",
            "block_best_role": "calibration_oracle_not_reported_execution",
            "mode_ranking_calibration": args.calibrate_mode_ranking,
            "main_mode_ranking_calibration": args.calibrate_main_mode_ranking,
            "service_order_ablation": args.service_order_ablation,
            "nouter_global_optimality_claim": False,
            "four_stage": "preexisting_65_case_certified_result",
        },
        "source_sha256": _source_hashes(args.case_file),
        "indices": list(indices),
        "cases": results,
        "aggregate": _aggregate(results),
    }
    _write_atomic(args.output, payload)
    print(f"result_written={args.output}", flush=True)


if __name__ == "__main__":
    main()
