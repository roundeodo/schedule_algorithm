#!/usr/bin/env python3
"""Evaluate the isolated coarse N-outer model on frozen OLMoE cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from four_stage_scheduler import (
    SCHEDULE_TIME_QUANTUM_CC,
    WEIGHT_BYTES_TOTAL,
    _best_task_time,
)

from .block_golden import replay_best_policy
from .candidates import WindowSpec
from .lowering import lower_history_to_bingo, replay_bingo_program
from .search import SearchConfig, beam_search
from .semantics import ALL_SHAPES, compute_block_cc, default_phases


DEFAULT_CASE_FILE = Path("results/policy_search/olmoe_top2_projection_65_optimal_v1.json")


def _indices(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("indices are one-based positive integers")
    return result


def _work_signature(counts: tuple[int, ...]) -> dict[str, int]:
    phases = default_phases()
    nouter_compute = sum(
        phase.block_count
        * min(compute_block_cc(ntokens, shape, phase) for shape in ALL_SHAPES)
        for ntokens in counts
        for phase in phases
    )
    four_stage_compute = sum(_best_task_time(ntokens) for ntokens in counts)
    nouter_weight = len(counts) * sum(
        phase.block_count * phase.weight_block_bytes for phase in phases
    )
    four_stage_weight = len(counts) * WEIGHT_BYTES_TOTAL
    if nouter_compute != four_stage_compute:
        raise AssertionError("N-outer and four-stage compute work differ")
    if nouter_weight != four_stage_weight:
        raise AssertionError("N-outer and four-stage weight bytes differ")
    return {"compute_cc": nouter_compute, "weight_bytes": nouter_weight}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--bottom", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--candidate-budget", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source = json.loads(args.case_file.read_text(encoding="utf-8"))
    source_cases = source["cases"]
    if any(index > len(source_cases) for index in args.indices):
        parser.error("an index exceeds the case catalog")
    config = SearchConfig(
        window=WindowSpec(args.top, args.bottom),
        beam_width=args.beam_width,
        candidate_budget=args.candidate_budget,
    )
    results: list[dict[str, object]] = []
    for index in args.indices:
        case = source_cases[index - 1]
        counts = tuple(int(value) for value in case["counts"])
        started = time.perf_counter()
        search = beam_search(counts, config=config)
        golden = replay_best_policy(search.node.history)
        program = lower_history_to_bingo(counts, search.node)
        lowered_replay = replay_bingo_program(program)
        certified_cc = int(case["best_reference_ticks"]) * SCHEDULE_TIME_QUANTUM_CC
        item = {
            "catalog_index": index,
            "name": case["name"],
            "counts": list(counts),
            "active_experts": len(counts),
            "work_signature": _work_signature(counts),
            "four_stage_certified_cc": certified_cc,
            "four_stage_certified_ticks": str(case["best_reference_ticks"]),
            "nouter_macro_cc": search.node.makespan_cc,
            "nouter_block_best_cc": golden.makespan_cc,
            "nouter_block_best_policy": golden.policy.value,
            "nouter_lowered_macro_order_cc": lowered_replay.makespan_cc,
            "macro_minus_block_best_cc": (
                search.node.makespan_cc - golden.makespan_cc
            ),
            "block_best_minus_four_stage_cc": (
                golden.makespan_cc - certified_cc
            ),
            "history_steps": len(search.node.history),
            "macro_records": len(program.records),
            "expanded_nodes": search.expanded_nodes,
            "evaluated_plans": search.evaluated_plans,
            "history_validated": search.history_validated,
            "lowering_validated": program.history_validated,
            "block_replay_validated": golden.history_validated,
            "runtime_s": time.perf_counter() - started,
        }
        results.append(item)
        print(
            f"[{index:02d}] {case['name']} active={len(counts)} "
            f"nouter_macro={search.node.makespan_cc} "
            f"nouter_block={golden.makespan_cc} "
            f"four_stage={certified_cc} "
            f"runtime={item['runtime_s']:.2f}s",
            flush=True,
        )

    directory = Path(__file__).resolve().parent
    payload = {
        "schema": "coarse_nouter_olmoe_evaluation_v1",
        "contracts": {
            "stream_order": "expert_phase_block",
            "startup": "cold_first_block_charged",
            "candidate_semantics": "single_pair_split_real_token_slices",
            "block_model_role": "calibration_and_lowering_replay_only",
            "comparison_scope": "same_work_different_loop_order",
            "nouter_optimality_claim": False,
            "four_stage_source": "preexisting_65_case_certified_result",
        },
        "config": {
            "top": args.top,
            "bottom": args.bottom,
            "beam_width": args.beam_width,
            "candidate_budget": args.candidate_budget,
            "indices": list(args.indices),
        },
        "source_sha256": {
            "case_file": _hash(args.case_file),
            "semantics.py": _hash(directory / "semantics.py"),
            "candidates.py": _hash(directory / "candidates.py"),
            "search.py": _hash(directory / "search.py"),
            "block_golden.py": _hash(directory / "block_golden.py"),
            "lowering.py": _hash(directory / "lowering.py"),
            "four_stage_scheduler.py": _hash(directory.parent.parent / "four_stage_scheduler.py"),
        },
        "cases": results,
    }
    if args.output is not None:
        _write_atomic(args.output, payload)
        print(f"result_written={args.output}", flush=True)


if __name__ == "__main__":
    main()

