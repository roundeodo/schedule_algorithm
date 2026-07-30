#!/usr/bin/env python3
"""Resumable paired 30K comparison of four scheduler strategies.

The four strategies are evaluated on identical distributions and initial
cache states:

* ``frozen_bounded``: the frozen R4+bottom2/K32 bounded-S2 policy;
* ``current_hardware_lite``: the Python mirror of the deployed C/RTL policy;
* ``historical_lite``: exact ``lite_scheduler.py`` from the last revision that
  contained it;
* ``historical_fast``: exact ``fast_scheduler.py`` from the same revision.

Historical modules are loaded read-only from Git objects, together with their
matching historical ``four_stage_scheduler.py`` and ``analytical_scheduler.py``.
They are deliberately not restored into the worktree.  This avoids mixing the
old aggregate-bandwidth helper API with the current explicit-DMA-lane helper
API while retaining the exact historical algorithms.

The report also joins the existing anytime reference result for every case.
Reference ratios are separated into all eligible and proven-optimal cases;
pairwise strategy comparisons never depend on reference completeness.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

from eval_hw_mirror_s2pf_lite import hw_mirror_schedule
from scheduler_policy_golden import BOUNDED_S2_CONFIG, run_distribution


ROOT = Path(__file__).resolve().parent
THESIS_ROOT = ROOT.parent
DEFAULT_INPUTS = (
    ROOT / "scheduler_strategy_coverage_E8.json",
    ROOT / "scheduler_strategy_coverage_E32.json",
    ROOT / "scheduler_strategy_coverage_E64.json",
)
DEFAULT_REFERENCES = (
    ROOT / "results" / "final_reference" / "scheduler_reference_E8_compact.json",
    ROOT / "results" / "final_reference" / "scheduler_reference_E32_compact.json",
    ROOT / "results" / "final_reference" / "scheduler_reference_E64_compact.json",
)
DEFAULT_OUT = (
    ROOT / "results" / "policy_search" / "scheduler_strategies_30k.json"
)
FREEZE_MANIFEST = ROOT / "results" / "policy_search" / "bounded_policy_freeze_v1.json"
HW_MIRROR_PATH = ROOT / "eval_hw_mirror_s2pf_lite.py"
HW_C_PATH = (
    THESIS_ROOT
    / "HeMAiA"
    / "target"
    / "sw"
    / "host"
    / "apps"
    / "offload_bingo_hw"
    / "single_chip"
    / "workloads"
    / "multi_cluster_MoE"
    / "moe_scheduler.c"
)
HW_RTL_PATHS = (
    THESIS_ROOT / "Scheduler_hw" / "sched_schedule_core.sv",
    THESIS_ROOT / "Scheduler_hw" / "sched_candidate_generator.sv",
    THESIS_ROOT / "Scheduler_hw" / "sched_score_unit.sv",
)
HW_CONFIG = {
    "policy": "balanced",
    "top_policy": "pruned",
    "n1_policy": "pruned",
}

STRATEGIES = (
    "frozen_bounded",
    "current_hardware_lite",
    "historical_lite",
    "historical_fast",
)
HISTORICAL_REVISION = "7a060e85be8fa3d8510cc76512cedc14339407d6"
HISTORICAL_SOURCE_SHA256 = {
    "four_stage_scheduler.py": "28e2e435381de8667138695c43ad54b05fbad1343a70dab4fdca1ca83c8bb672",
    "analytical_scheduler.py": "6791fbe422f0097dde8dc17f935585a9d42217a32c177b454558fc13dca48898",
    "lite_scheduler.py": "2a3d0fd104a8ac329bb4cfc8522f0a84bb6db7ab067551ae1e158bf743fb629b",
    "fast_scheduler.py": "21b9252a3a4c7fcb2a41fce14e0b95a87f6b6e297ed34f6f5b0e0b6a80c24afb",
}

_HISTORICAL_SCHEDULERS = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _git_blob(filename: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{HISTORICAL_REVISION}:{filename}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    source = result.stdout
    actual = hashlib.sha256(source).hexdigest()
    expected = HISTORICAL_SOURCE_SHA256[filename]
    if actual != expected:
        raise ValueError(
            f"historical source drift for {filename}: {actual} != {expected}"
        )
    return source


def _exec_historical_module(module_name: str, filename: str) -> types.ModuleType:
    module = types.ModuleType(module_name)
    module.__file__ = f"{ROOT}/.git-history/{HISTORICAL_REVISION}/{filename}"
    module.__package__ = ""
    sys.modules[module_name] = module
    source = _git_blob(filename)
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def historical_schedulers():
    """Load exact historical lite/fast functions in an isolated module stack."""
    global _HISTORICAL_SCHEDULERS
    if _HISTORICAL_SCHEDULERS is not None:
        return _HISTORICAL_SCHEDULERS

    saved_four = sys.modules.get("four_stage_scheduler")
    saved_analytical = sys.modules.get("analytical_scheduler")
    suffix = HISTORICAL_REVISION[:12]
    try:
        historical_four = _exec_historical_module(
            f"_historical_four_stage_scheduler_{suffix}",
            "four_stage_scheduler.py",
        )
        sys.modules["four_stage_scheduler"] = historical_four
        historical_analytical = _exec_historical_module(
            f"_historical_analytical_scheduler_{suffix}",
            "analytical_scheduler.py",
        )
        sys.modules["analytical_scheduler"] = historical_analytical
        historical_lite = _exec_historical_module(
            f"_historical_lite_scheduler_{suffix}",
            "lite_scheduler.py",
        )
        historical_fast = _exec_historical_module(
            f"_historical_fast_scheduler_{suffix}",
            "fast_scheduler.py",
        )
    finally:
        if saved_four is None:
            sys.modules.pop("four_stage_scheduler", None)
        else:
            sys.modules["four_stage_scheduler"] = saved_four
        if saved_analytical is None:
            sys.modules.pop("analytical_scheduler", None)
        else:
            sys.modules["analytical_scheduler"] = saved_analytical

    _HISTORICAL_SCHEDULERS = (
        historical_lite.lite_schedule,
        historical_fast.fast_schedule,
    )
    return _HISTORICAL_SCHEDULERS


def pair_summary(rows: list[dict], left: str, right: str) -> dict:
    deltas = [
        int(row["makespan_cc"][left]) - int(row["makespan_cc"][right])
        for row in rows
    ]
    ratios = [
        float(row["makespan_cc"][left]) / float(row["makespan_cc"][right])
        for row in rows
    ]
    return {
        "cases": len(rows),
        "left": left,
        "right": right,
        "left_better": sum(delta < 0 for delta in deltas),
        "right_better": sum(delta > 0 for delta in deltas),
        "equal": sum(delta == 0 for delta in deltas),
        "left_minus_right_cc_total": sum(deltas),
        "left_minus_right_cc_mean": statistics.mean(deltas) if deltas else None,
        "left_over_right_ratio_mean": statistics.mean(ratios) if ratios else None,
        "left_over_right_ratio_p05": percentile(ratios, 0.05),
        "left_over_right_ratio_p50": percentile(ratios, 0.50),
        "left_over_right_ratio_p95": percentile(ratios, 0.95),
        "left_over_right_ratio_min": min(ratios) if ratios else None,
        "left_over_right_ratio_max": max(ratios) if ratios else None,
    }


def reference_summary(rows: list[dict], strategy: str, *, proven_only: bool) -> dict:
    selected = [
        row for row in rows if not proven_only or row["reference_proven_optimal"]
    ]
    ratios = [
        float(row["makespan_cc"][strategy]) / float(row["reference_makespan_cc"])
        for row in selected
    ]
    return {
        "cases": len(selected),
        "strategy": strategy,
        "reference_scope": "proven_optimal" if proven_only else "all_best_ub",
        "exact": sum(abs(ratio - 1.0) <= 1e-12 for ratio in ratios),
        "within_1pct": sum(ratio <= 1.01 + 1e-12 for ratio in ratios),
        "within_3pct": sum(ratio <= 1.03 + 1e-12 for ratio in ratios),
        "beats_reference": sum(ratio < 1.0 - 1e-12 for ratio in ratios),
        "ratio_mean": statistics.mean(ratios) if ratios else None,
        "ratio_p50": percentile(ratios, 0.50),
        "ratio_p95": percentile(ratios, 0.95),
        "ratio_max": max(ratios) if ratios else None,
    }


def winner_summary(rows: list[dict]) -> dict:
    wins = {strategy: 0 for strategy in STRATEGIES}
    sole_wins = {strategy: 0 for strategy in STRATEGIES}
    for row in rows:
        best = min(row["makespan_cc"].values())
        leaders = [
            strategy
            for strategy in STRATEGIES
            if int(row["makespan_cc"][strategy]) == int(best)
        ]
        for strategy in leaders:
            wins[strategy] += 1
        if len(leaders) == 1:
            sole_wins[leaders[0]] += 1
    return {
        "cases": len(rows),
        "best_or_tied_best": wins,
        "sole_best": sole_wins,
    }


def bucket_summary(rows: list[dict]) -> dict:
    pairwise = {}
    for left_index, left in enumerate(STRATEGIES):
        for right in STRATEGIES[left_index + 1 :]:
            pairwise[f"{left}_vs_{right}"] = pair_summary(rows, left, right)
    reference_all = {
        strategy: reference_summary(rows, strategy, proven_only=False)
        for strategy in STRATEGIES
    }
    reference_proven = {
        strategy: reference_summary(rows, strategy, proven_only=True)
        for strategy in STRATEGIES
    }
    return {
        "winner_counts": winner_summary(rows),
        "pairwise": pairwise,
        "vs_reference_all_best_ub": reference_all,
        "vs_reference_proven_optimal": reference_proven,
    }


def summarize(results: dict[str, dict]) -> dict:
    buckets = defaultdict(list)
    for row in results.values():
        keys = (
            "overall",
            f"E{row['e_total']}",
            f"split:{row['dataset_split']}",
            f"E{row['e_total']}:split:{row['dataset_split']}",
        )
        for key in keys:
            buckets[key].append(row)
    return {key: bucket_summary(values) for key, values in sorted(buckets.items())}


def run_case(case: dict) -> tuple[str, dict]:
    lite_schedule, fast_schedule = historical_schedulers()
    e_total = int(case["e_total"])
    case_id = int(case["case_id"])
    key = f"E{e_total}:{case_id}"
    dist = {int(eid): int(ntok) for eid, ntok in case["dist"].items()}
    c2 = int(case.get("c2", -1))
    c3 = int(case.get("c3", -1))

    frozen = run_distribution(
        dist,
        initial_cache_c2=c2,
        initial_cache_c3=c3,
        config=BOUNDED_S2_CONFIG,
    )
    makespan_cc = {
        "frozen_bounded": int(frozen["makespan_cc"]),
        "current_hardware_lite": int(
            hw_mirror_schedule(dist, c2, c3, **HW_CONFIG)
        ),
        "historical_lite": int(lite_schedule(dist, c2, c3)),
        "historical_fast": int(fast_schedule(dist, c2, c3)),
    }
    reference = case["_reference"]
    return key, {
        "case_id": case_id,
        "e_total": e_total,
        "dataset_split": case.get("dataset_split"),
        "sample_class": case.get("sample_class"),
        "construction": case.get("construction"),
        "cache_regime": case.get("cache_regime"),
        "active_n": int(case.get("active_n", len(dist))),
        "m_total": int(case.get("m_total", 0)),
        "m_band": case.get("features", {}).get("m_band"),
        "makespan_cc": makespan_cc,
        "reference_makespan_cc": int(reference["makespan_cc"]),
        "reference_lower_bound_cc": int(reference["lower_bound_cc"]),
        "reference_optimality_gap": float(reference["optimality_gap"]),
        "reference_proven_optimal": bool(reference["proven_optimal"]),
        "reference_quality_class": reference.get("quality_class"),
        "frozen_decisions": int(frozen["decisions"]),
        "frozen_max_candidates": int(frozen["max_candidates"]),
        "frozen_history_sha256": frozen["history_sha256"],
    }


def configuration(input_paths: list[Path], reference_paths: list[Path]) -> dict:
    freeze = json.loads(FREEZE_MANIFEST.read_text())
    expected_policy_id = freeze["policy"]["policy_id"]
    if expected_policy_id != BOUNDED_S2_CONFIG.policy_id:
        raise ValueError("golden policy ID differs from the freeze manifest")
    frozen_sources = {}
    for name, expected in freeze["source_sha256"].items():
        path = ROOT / name
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"frozen source drift: {name}")
        frozen_sources[name] = actual
    hardware_sources = {
        str(HW_MIRROR_PATH.resolve()): file_sha256(HW_MIRROR_PATH),
        str(HW_C_PATH.resolve()): file_sha256(HW_C_PATH),
    }
    hardware_sources.update(
        {str(path.resolve()): file_sha256(path) for path in HW_RTL_PATHS}
    )
    return {
        "inputs": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in input_paths
        ],
        "references": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in reference_paths
        ],
        "strategies": {
            "frozen_bounded": {
                "policy_id": BOUNDED_S2_CONFIG.policy_id,
                "manifest": str(FREEZE_MANIFEST.resolve()),
                "manifest_sha256": file_sha256(FREEZE_MANIFEST),
                "source_sha256": frozen_sources,
            },
            "current_hardware_lite": {
                "config": HW_CONFIG,
                "source_sha256": hardware_sources,
            },
            "historical_lite": {
                "entrypoint": "lite_scheduler.py:lite_schedule",
                "git_revision": HISTORICAL_REVISION,
                "source_sha256": HISTORICAL_SOURCE_SHA256,
            },
            "historical_fast": {
                "entrypoint": "fast_scheduler.py:fast_schedule",
                "git_revision": HISTORICAL_REVISION,
                "source_sha256": HISTORICAL_SOURCE_SHA256,
            },
        },
        "historical_execution": (
            "exact Git blobs loaded with their matching historical four-stage "
            "and analytical modules; no historical files restored"
        ),
        "comparison_driver_sha256": file_sha256(Path(__file__).resolve()),
        "analysis_eligibility": "case.analysis_eligible=true",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument(
        "--references", nargs="+", type=Path, default=list(DEFAULT_REFERENCES)
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--checkpoint-every-s", type=float, default=300.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--limit-per-file", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")
    if args.checkpoint_every_s <= 0:
        raise ValueError("--checkpoint-every-s must be positive")
    if args.limit_per_file is not None and args.limit_per_file < 0:
        raise ValueError("--limit-per-file must be nonnegative")
    input_paths = [path.resolve() for path in args.inputs]
    reference_paths = [path.resolve() for path in args.references]
    if len(input_paths) != len(reference_paths):
        raise ValueError("--inputs and --references must contain the same count")
    config = configuration(input_paths, reference_paths)
    if args.out.exists():
        if not args.resume:
            raise FileExistsError(f"{args.out} exists; pass --resume")
        report = json.loads(args.out.read_text())
        if report.get("configuration") != config:
            raise ValueError("checkpoint configuration changed")
    else:
        report = {
            "schema": "scheduler_strategies_30k_v1",
            "provisional": True,
            "configuration": config,
            "source_cases": 0,
            "analysis_eligible_cases": 0,
            "excluded_analysis_ineligible_cases": 0,
            "results": {},
            "errors": {},
            "runtime_s_total": 0.0,
        }
    results = report["results"]
    errors = report["errors"]
    started = time.perf_counter()
    previous_runtime = float(report.get("runtime_s_total", 0.0))
    completed_this_run = 0
    last_checkpoint_at = time.perf_counter()

    payloads = []
    source_cases = 0
    eligible_cases = 0
    for input_path, reference_path in zip(input_paths, reference_paths):
        payload = json.loads(input_path.read_text())
        cases = payload.get("cases")
        reference_payload = json.loads(reference_path.read_text())
        reference_results = reference_payload.get("results")
        if not isinstance(cases, list):
            raise ValueError(f"input contains no cases list: {input_path}")
        if not isinstance(reference_results, dict):
            raise ValueError(f"reference contains no results map: {reference_path}")
        eligible = []
        for case in cases:
            source_cases += 1
            if not case.get("analysis_eligible", True):
                continue
            case_id = str(int(case["case_id"]))
            if case_id not in reference_results:
                raise ValueError(f"reference missing case {case_id}: {reference_path}")
            reference = reference_results[case_id]
            normalized_dist = {str(int(k)): int(v) for k, v in case["dist"].items()}
            reference_dist = {
                str(int(k)): int(v) for k, v in reference["dist"].items()
            }
            if normalized_dist != reference_dist:
                raise ValueError(f"distribution mismatch for case {case_id}")
            if int(case.get("c2", -1)) != int(reference.get("initial_cache_c2", -1)):
                raise ValueError(f"C2 cache mismatch for case {case_id}")
            if int(case.get("c3", -1)) != int(reference.get("initial_cache_c3", -1)):
                raise ValueError(f"C3 cache mismatch for case {case_id}")
            joined = dict(case)
            joined["_reference"] = reference
            eligible.append(joined)
            eligible_cases += 1
        payloads.append((input_path, eligible))
    report["source_cases"] = source_cases
    report["analysis_eligible_cases"] = eligible_cases
    report["excluded_analysis_ineligible_cases"] = source_cases - eligible_cases

    def checkpoint(provisional: bool) -> None:
        nonlocal last_checkpoint_at
        report["provisional"] = provisional
        report["completed_cases"] = len(results)
        report["runtime_s_total"] = previous_runtime + time.perf_counter() - started
        report["summary"] = summarize(results)
        if not provisional:
            report["results"] = dict(
                sorted(
                    results.items(),
                    key=lambda item: tuple(
                        int(value) for value in item[0][1:].split(":")
                    ),
                )
            )
            report["largest_frozen_improvements"] = {}
            report["largest_frozen_regressions"] = {}
            for baseline in STRATEGIES[1:]:
                report["largest_frozen_improvements"][baseline] = sorted(
                    report["results"].values(),
                    key=lambda row: (
                        row["makespan_cc"]["frozen_bounded"]
                        - row["makespan_cc"][baseline]
                    ),
                )[:100]
                report["largest_frozen_regressions"][baseline] = sorted(
                    report["results"].values(),
                    key=lambda row: (
                        row["makespan_cc"][baseline]
                        - row["makespan_cc"]["frozen_bounded"]
                    ),
                )[:100]
        atomic_write_json(args.out, report)
        last_checkpoint_at = time.perf_counter()

    # Validate and preload the Git-backed modules before starting workers.
    historical_schedulers()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for path, eligible in payloads:
            pending = [
                case
                for case in eligible
                if f"E{int(case['e_total'])}:{int(case['case_id'])}" not in results
            ]
            if args.limit_per_file is not None:
                pending = pending[: args.limit_per_file]
            futures = {executor.submit(run_case, case): case for case in pending}
            for future in as_completed(futures):
                case = futures[future]
                key = f"E{int(case['e_total'])}:{int(case['case_id'])}"
                try:
                    result_key, row = future.result()
                except Exception as exc:
                    errors[key] = repr(exc)
                    checkpoint(True)
                    raise RuntimeError(f"case failed: {key}") from exc
                if result_key != key:
                    raise RuntimeError(f"worker key mismatch: {result_key} != {key}")
                results[key] = row
                errors.pop(key, None)
                completed_this_run += 1
                if (
                    completed_this_run % args.save_every == 0
                    or time.perf_counter() - last_checkpoint_at
                    >= args.checkpoint_every_s
                ):
                    checkpoint(True)
                if args.progress_every > 0 and completed_this_run % args.progress_every == 0:
                    print(
                        f"strategy-30k completed={len(results)}/{eligible_cases} "
                        f"new={completed_this_run} "
                        f"elapsed_s={time.perf_counter()-started:.1f}",
                        flush=True,
                    )
    complete = len(results) == eligible_cases and not errors
    checkpoint(not complete)
    print(
        f"strategy-30k finished completed={len(results)}/{eligible_cases} "
        f"provisional={not complete} wrote {args.out}",
        flush=True,
    )
    if complete:
        print(json.dumps(report["summary"]["overall"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
