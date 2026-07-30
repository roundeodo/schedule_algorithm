#!/usr/bin/env python3
"""Summarize the frozen 65-case N-outer policy evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from collections import Counter
from pathlib import Path


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires data")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return ordered[left]
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def summarize(payload: dict[str, object]) -> dict[str, object]:
    cases = payload.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("evaluation contains no cases")
    indices = [int(case["catalog_index"]) for case in cases]
    if len(indices) != len(set(indices)):
        raise ValueError("evaluation repeats a catalog index")

    main_counter: Counter[str] = Counter()
    analysis_best_counter: Counter[str] = Counter()
    executable_oracle_counter: Counter[str] = Counter()
    block_oracle_counter: Counter[str] = Counter()
    action_counter: Counter[str] = Counter()
    mode_counter: Counter[str] = Counter()
    main_ratios: list[float] = []
    macro_errors: list[float] = []
    arbitration_regrets: list[float] = []
    policy_selection_regrets: list[int] = []
    split_deltas: list[int] = []
    four_stage_relation = Counter()
    validation_failures: list[dict[str, object]] = []
    policy_regret_cases: list[dict[str, object]] = []

    for case in cases:
        policies = case["policies"]
        main_name = case.get("rtl_main_policy", case.get("selected_policy"))
        analysis_best_name = case.get(
            "analysis_best_policy", case.get("selected_policy")
        )
        main_policy = next(
            item for item in policies if item["name"] == main_name
        )
        analysis_best = next(
            item
            for item in policies
            if item["name"] == analysis_best_name
        )
        block_oracle = min(
            policies,
            key=lambda item: (
                int(item["block_best_cc"]),
                int(item["macro_cc"]),
                item["name"],
            ),
        )
        executable_oracle = min(
            policies,
            key=lambda item: (
                int(item["task_abi_macro_order_cc"]),
                int(item["macro_cc"]),
                item["name"],
            ),
        )
        main_counter[main_policy["name"]] += 1
        analysis_best_counter[analysis_best["name"]] += 1
        executable_oracle_counter[executable_oracle["name"]] += 1
        block_oracle_counter[block_oracle["name"]] += 1
        action_counter.update(main_policy["actions"])
        mode_counter.update(main_policy["mode_profiles"])

        four_stage = int(case["four_stage_certified_cc"])
        main_executable = int(main_policy["task_abi_macro_order_cc"])
        main_ratios.append(main_executable / four_stage)
        macro_errors.append(float(int(main_policy["macro_cc"]) - main_executable))
        arbitration_regrets.append(
            float(main_executable - int(main_policy["block_best_cc"]))
        )
        if main_executable < four_stage:
            four_stage_relation["win"] += 1
        elif main_executable == four_stage:
            four_stage_relation["tie"] += 1
        else:
            four_stage_relation["loss"] += 1

        analysis_executable = int(analysis_best["task_abi_macro_order_cc"])
        regret = analysis_executable - int(
            executable_oracle["task_abi_macro_order_cc"]
        )
        policy_selection_regrets.append(regret)
        if regret:
            policy_regret_cases.append(
                {
                    "catalog_index": case["catalog_index"],
                    "name": case["name"],
                    "macro_selected": analysis_best["name"],
                    "executable_oracle": executable_oracle["name"],
                    "regret_cc": regret,
                }
            )

        no_split = next(
            item
            for item in policies
            if item["name"].startswith("paired_lpt_mode_search_")
        )
        split = next(item for item in policies if item["name"].startswith("split_e"))
        split_deltas.append(
            int(split["task_abi_macro_order_cc"])
            - int(no_split["task_abi_macro_order_cc"])
        )

        for policy in policies:
            valid = all(
                bool(policy[field])
                for field in (
                    "history_validated",
                    "block_validated",
                    "lowering_validated",
                    "task_abi_validated",
                )
            )
            replay_equal = int(policy["task_abi_macro_order_cc"]) == int(
                policy["lowered_macro_order_cc"]
            )
            if not valid or not replay_equal:
                validation_failures.append(
                    {
                        "catalog_index": case["catalog_index"],
                        "policy": policy["name"],
                        "flags_valid": valid,
                        "macro_order_replay_equal": replay_equal,
                    }
                )

    policy_regret_cases.sort(
        key=lambda item: (-item["regret_cc"], item["catalog_index"])
    )
    split_improves = [-delta for delta in split_deltas if delta < 0]
    return {
        "schema": "coarse_nouter_final_policy_summary_v3",
        "source_schema": payload.get("schema"),
        "case_count": len(cases),
        "indices": indices,
        "complete_65": sorted(indices) == list(range(1, 66)),
        "all_histories_and_task_replays_valid": not validation_failures,
        "validation_failures": validation_failures,
        "rtl_main_policy_counts": dict(sorted(main_counter.items())),
        "analysis_best_macro_policy_counts": dict(
            sorted(analysis_best_counter.items())
        ),
        "executable_oracle_policy_counts": dict(
            sorted(executable_oracle_counter.items())
        ),
        "block_arbitration_oracle_policy_counts": dict(
            sorted(block_oracle_counter.items())
        ),
        "rtl_main_action_counts": dict(sorted(action_counter.items())),
        "rtl_main_mode_profile_counts": dict(sorted(mode_counter.items())),
        "rtl_main_executable_over_four_stage": _stats(main_ratios),
        "rtl_main_vs_four_stage_case_counts": dict(sorted(four_stage_relation.items())),
        "rtl_main_macro_minus_executable_cc": {
            **_stats(macro_errors),
            "optimistic_cases": sum(value < 0 for value in macro_errors),
            "exact_cases": sum(value == 0 for value in macro_errors),
            "conservative_cases": sum(value > 0 for value in macro_errors),
        },
        "rtl_main_executable_arbitration_regret_cc": {
            **_stats(arbitration_regrets),
            "nonzero_cases": sum(value > 0 for value in arbitration_regrets),
            "total_cc": sum(arbitration_regrets),
        },
        "analysis_best_macro_selection_regret_within_three_policy_bank_cc": {
            **_stats([float(value) for value in policy_selection_regrets]),
            "nonzero_cases": sum(value > 0 for value in policy_selection_regrets),
            "total_cc": sum(policy_selection_regrets),
            "cases": policy_regret_cases,
        },
        "top1_split_vs_no_split_executable_cc": {
            **_stats([float(value) for value in split_deltas]),
            "improves_cases": len(split_improves),
            "ties_cases": sum(value == 0 for value in split_deltas),
            "hurts_cases": sum(value > 0 for value in split_deltas),
            "mean_improvement_when_better_cc": (
                statistics.fmean(split_improves) if split_improves else 0.0
            ),
        },
        "embedded_aggregate": payload.get("aggregate"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    summary = summarize(payload)
    _write_atomic(args.output, summary)
    print(
        f"cases={summary['case_count']} complete_65={summary['complete_65']} "
        f"validated={summary['all_histories_and_task_replays_valid']}"
    )
    print(
        "rtl-main-executable/four-stage median="
        f"{summary['rtl_main_executable_over_four_stage']['median']:.6f} "
        "policy-selection-regret-cases="
        f"{summary['analysis_best_macro_selection_regret_within_three_policy_bank_cc']['nonzero_cases']}"
    )
    print(f"result_written={args.output}")


if __name__ == "__main__":
    main()
