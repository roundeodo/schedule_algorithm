#!/usr/bin/env python3
"""Run the four-stage reference search on strategy-coverage inputs."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from four_stage_scheduler import (
    ALL_SHAPES,
    DmaBinding,
    FourStageScheduler,
    StageAction,
    clear_scheduler_caches,
    state_lower_bound_components,
    validate_schedule_history,
)


ROOT = Path(__file__).resolve().parent
MODEL_MODE = "explicit_dma_lane_four_stage_anytime"
LOWER_BOUND_MODE = "m2_block_release_chain_pathmax_v2"
DEFAULT_FILES = (
    ROOT / "scheduler_strategy_coverage_E8.json",
    ROOT / "scheduler_strategy_coverage_E32.json",
    ROOT / "scheduler_strategy_coverage_E64.json",
)


def serialize_action(action) -> dict:
    def shape_name(shape):
        return shape.name if shape is not None else None

    def dma_name(binding):
        return binding.name

    return {
        "tag": action.tag,
        "c2_eid": action.c2_eid,
        "c2_ntok": action.c2_ntok,
        "c2_start_cc": action.c2_start,
        "c2_shape_s1": shape_name(action.c2_shape_s1),
        "c2_shape_s3": shape_name(action.c2_shape_s3),
        "c2_s1_cached": action.c2_s1_cached,
        "c2_s3_cached": action.c2_s3_cached,
        "c2_s2pf_start_cc": action.c2_s2pf_start,
        "c2_dma_s1": dma_name(action.c2_dma_s1),
        "c2_dma_s3": dma_name(action.c2_dma_s3),
        "c2_s2pf_dma": dma_name(action.c2_s2pf_dma),
        "c3_eid": action.c3_eid,
        "c3_ntok": action.c3_ntok,
        "c3_start_cc": action.c3_start,
        "c3_shape_s1": shape_name(action.c3_shape_s1),
        "c3_shape_s3": shape_name(action.c3_shape_s3),
        "c3_s1_cached": action.c3_s1_cached,
        "c3_s3_cached": action.c3_s3_cached,
        "c3_s2pf_start_cc": action.c3_s2pf_start,
        "c3_dma_s1": dma_name(action.c3_dma_s1),
        "c3_dma_s3": dma_name(action.c3_dma_s3),
        "c3_s2pf_dma": dma_name(action.c3_s2pf_dma),
        "pf_cluster": action.pf_cluster,
        "pf_eid": action.pf_eid,
        "pf_shape": shape_name(action.pf_shape),
        "pf_start_cc": action.pf_start,
        "pf_dma": dma_name(action.pf_dma),
    }


SHAPE_BY_NAME = {shape.name: shape for shape in ALL_SHAPES}


def deserialize_action(item: dict) -> StageAction:
    """Rebuild a validated prior-pass action used to seed follow-up search."""

    def shape(field: str):
        name = item.get(field)
        if name is None:
            return None
        try:
            return SHAPE_BY_NAME[name]
        except KeyError as exc:
            raise ValueError(f"unknown serialized shape {name!r}") from exc

    def dma(field: str) -> DmaBinding:
        name = item.get(field, "NONE")
        try:
            return DmaBinding[name]
        except KeyError as exc:
            raise ValueError(f"unknown serialized DMA binding {name!r}") from exc

    return StageAction(
        c2_eid=int(item["c2_eid"]),
        c2_ntok=int(item["c2_ntok"]),
        c2_shape_s1=shape("c2_shape_s1"),
        c2_shape_s3=shape("c2_shape_s3"),
        c2_start=int(item["c2_start_cc"]),
        c2_s1_cached=bool(item["c2_s1_cached"]),
        c2_s3_cached=bool(item["c2_s3_cached"]),
        c3_eid=int(item["c3_eid"]),
        c3_ntok=int(item["c3_ntok"]),
        c3_shape_s1=shape("c3_shape_s1"),
        c3_shape_s3=shape("c3_shape_s3"),
        c3_start=int(item["c3_start_cc"]),
        c3_s1_cached=bool(item["c3_s1_cached"]),
        c3_s3_cached=bool(item["c3_s3_cached"]),
        pf_cluster=int(item["pf_cluster"]),
        pf_eid=int(item["pf_eid"]),
        pf_shape=shape("pf_shape"),
        pf_start=int(item["pf_start_cc"]),
        tag=str(item.get("tag", "")),
        c2_s2pf_start=int(item.get("c2_s2pf_start_cc", -1)),
        c3_s2pf_start=int(item.get("c3_s2pf_start_cc", -1)),
        c2_dma_s1=dma("c2_dma_s1"),
        c2_dma_s3=dma("c2_dma_s3"),
        c2_s2pf_dma=dma("c2_s2pf_dma"),
        c3_dma_s1=dma("c3_dma_s1"),
        c3_dma_s3=dma("c3_dma_s3"),
        c3_s2pf_dma=dma("c3_s2pf_dma"),
        pf_dma=dma("pf_dma"),
    )


def load_payload(path: Path):
    with path.open() as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "cases" not in payload:
        raise ValueError(f"{path} is not a scheduler strategy-coverage payload")
    return payload


def atomic_write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def run_case(
    case,
    beam_width: int,
    search_mode: str,
    time_limit_s: float,
    max_expansions: int,
    target_gap: float,
) -> dict:
    clear_scheduler_caches()
    dist = {int(k): int(v) for k, v in case["dist"].items()}
    c2 = int(case.get("c2", -1))
    c3 = int(case.get("c3", -1))
    t0 = time.time()
    scheduler = FourStageScheduler(
        dist,
        beam_width=beam_width,
        initial_cache_c2=c2,
        initial_cache_c3=c3,
    )
    initial_state = scheduler._initial_state()
    root_lb_components = state_lower_bound_components(
        initial_state.c2, initial_state.c3, initial_state.remaining
    )
    incumbent_history = None
    incumbent_seed = case.get("incumbent")
    incumbent_seed_cc = None
    if incumbent_seed is not None:
        incumbent_history = tuple(
            deserialize_action(action)
            for action in incumbent_seed.get("actions", [])
        )
        incumbent_seed_cc = validate_schedule_history(
            incumbent_history,
            dist,
            initial_cache_c2=c2,
            initial_cache_c3=c3,
        )
        declared_seed_cc = int(incumbent_seed["makespan_cc"])
        if incumbent_seed_cc != declared_seed_cc:
            raise ValueError(
                f"embedded incumbent makespan {declared_seed_cc} != "
                f"validated {incumbent_seed_cc}"
            )
    search_stats = {}
    if search_mode == "anytime":
        result = scheduler.run_anytime(
            time_limit_s=time_limit_s,
            max_expansions=max_expansions,
            target_gap=target_gap,
            incumbent_history=incumbent_history,
        )
        makespan = result.makespan
        history = result.history
        search_stats = {
            "lower_bound_cc": result.lower_bound,
            "optimality_gap": result.optimality_gap,
            "proven_optimal": result.proven_optimal,
            "expansions": result.expansions,
            "generated_states": result.generated,
            "pruned_by_bound": result.pruned_by_bound,
            "termination": result.termination,
        }
    else:
        makespan, history = scheduler.run()
    validated_makespan = validate_schedule_history(
        tuple(history), dist, initial_cache_c2=c2, initial_cache_c3=c3
    )
    if validated_makespan != makespan:
        raise RuntimeError(
            f"history makespan {validated_makespan} != search result {makespan}"
        )
    runtime = time.time() - t0
    ideal = case.get("compute_only_ideal_cc")
    ratio = (makespan / ideal) if ideal else None
    def action_family(action):
        if action.tag.startswith("PF-"):
            return "PREFETCH"
        if "SPLIT" in action.tag:
            return "SPLIT"
        if "PAIR" in action.tag:
            return "PAIR"
        if action.tag.startswith("SINGLE"):
            return "SINGLE"
        return "OTHER"

    family_counts = Counter(action_family(action) for action in history)
    output = {
        "case_id": case["case_id"],
        "e_total": case.get("e_total"),
        "m_total": case.get("m_total"),
        "assignment_total": case.get("assignment_total"),
        "active_n": case.get("active_n"),
        "dist": {str(eid): ntok for eid, ntok in sorted(dist.items())},
        "initial_cache_c2": c2,
        "initial_cache_c3": c3,
        "compute_only_ideal_cc": ideal,
        "analysis_eligible": case.get("analysis_eligible", True),
        "dataset_split": case.get("dataset_split"),
        "sample_class": case.get("sample_class"),
        "construction": case.get("construction"),
        "cache_regime": case.get("cache_regime"),
        "features": case.get("features", {}),
        "makespan_cc": makespan,
        "num_actions": len(history),
        "actions": [serialize_action(action) for action in history],
        "runtime_s": round(runtime, 6),
        "root_lb_components": root_lb_components,
        "lower_bound_mode": LOWER_BOUND_MODE,
        "incumbent_seed_cc": incumbent_seed_cc,
        "incumbent_seed_source": (
            incumbent_seed.get("source") if incumbent_seed is not None else None
        ),
        "over_compute_only_ideal": ratio,
        "action_family_counts": dict(sorted(family_counts.items())),
        "search_mode": search_mode,
        "history_validated": True,
        "status": "ok",
        **search_stats,
    }
    # Second-pass inputs embed the corresponding first-pass certificate so the
    # follow-up result stays self-contained and can be compared without loading
    # the full 30K first-pass sidecars again.
    if case.get("first_pass") is not None:
        output["first_pass"] = case["first_pass"]
    clear_scheduler_caches()
    return output


def annotate_file(
    path: Path,
    out_dir: Path,
    beam_width: int,
    limit: int,
    force: bool,
    save_every: int,
    max_active_n: int,
    search_mode: str,
    time_limit_s: float,
    max_expansions: int,
    target_gap: float,
    workers: int,
):
    payload = load_payload(path)
    cases = payload["cases"]
    if search_mode == "beam":
        search_tag = f"beam{beam_width}"
    else:
        gap_tag = "none" if target_gap is None else f"{target_gap:g}"
        time_tag = "none" if time_limit_s is None else f"{time_limit_s:g}"
        expansion_tag = "none" if max_expansions is None else str(max_expansions)
        search_tag = f"anytime_g{gap_tag}_t{time_tag}_x{expansion_tag}"
    out_path = out_dir / f"{path.stem}_{search_tag}_{MODEL_MODE}.json"
    if out_path.exists():
        result_payload = json.loads(out_path.read_text())
    else:
        result_payload = {
            "meta": {
                "input_file": path.name,
                "search_mode": search_mode,
                "beam_width": beam_width if search_mode == "beam" else None,
                "time_limit_s": time_limit_s if search_mode == "anytime" else None,
                "max_expansions": max_expansions if search_mode == "anytime" else None,
                "target_gap": target_gap if search_mode == "anytime" else None,
                "workers": workers,
                "model_mode": MODEL_MODE,
                "lower_bound_mode": LOWER_BOUND_MODE,
                "model": "gate_up_2048x1408_down_1408x2048_int4",
                "placement": (
                    "whole_block_ideal_overlap; partial_or_full_s2pf_at_dma1_end; "
                    "explicit IDMA/XDMA/BOTH lane binding; partial_or_full_s4pf; "
                    "chronological resource-event-aligned starts"
                ),
                "lowering_scope": (
                    "architectural reference with programmable DMA binding/start "
                    "relations; deployed compact ABI currently fixes 64-B lane roles"
                ),
            },
            "results": {},
        }
    results = result_payload["results"]
    done = 0
    failed = 0
    start_all = time.time()

    pending = []
    for idx, case in enumerate(cases):
        if max_active_n is not None and int(case.get("active_n", 0)) > max_active_n:
            continue
        case_id = str(case["case_id"])
        if not force and results.get(case_id, {}).get("status") == "ok":
            continue
        pending.append((idx, case))
        if limit is not None and len(pending) >= limit:
            break

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_case,
                case,
                beam_width,
                search_mode,
                time_limit_s,
                max_expansions,
                target_gap,
            ): (idx, case)
            for idx, case in pending
        }
        for future in as_completed(futures):
            idx, case = futures[future]
            case_id = str(case["case_id"])
            try:
                results[case_id] = future.result()
                done += 1
            except Exception as exc:
                results[case_id] = {
                    "case_id": case["case_id"],
                    "e_total": case.get("e_total"),
                    "analysis_eligible": case.get("analysis_eligible", True),
                    "dataset_split": case.get("dataset_split"),
                    "status": "error",
                    "error": repr(exc),
                }
                if case.get("first_pass") is not None:
                    results[case_id]["first_pass"] = case["first_pass"]
                failed += 1

            if save_every > 0 and (done + failed) % save_every == 0:
                atomic_write_json(out_path, result_payload)
                print(
                    f"{out_path.name}: saved done={done} failed={failed} idx={idx}",
                    flush=True,
                )

    result_payload["meta"]["completed_cases"] = sum(
        item.get("status") == "ok" for item in results.values()
    )
    result_payload["meta"]["last_run_s"] = round(time.time() - start_all, 6)
    eligible_ids = {
        str(case["case_id"])
        for case in cases
        if max_active_n is None or int(case.get("active_n", 0)) <= max_active_n
    }
    result_payload["meta"]["run_complete"] = limit is None and all(
        results.get(case_id, {}).get("status") == "ok" for case_id in eligible_ids
    )
    eligible_results = [
        results[case_id]
        for case_id in eligible_ids
        if results.get(case_id, {}).get("status") == "ok"
    ]
    result_payload["meta"]["proven_optimal_cases"] = sum(
        bool(item.get("proven_optimal")) for item in eligible_results
    )
    if search_mode == "anytime":
        if target_gap is None:
            quality_met = [
                bool(item.get("proven_optimal")) for item in eligible_results
            ]
        else:
            quality_met = [
                bool(item.get("proven_optimal"))
                or float(item.get("optimality_gap", float("inf"))) <= target_gap
                for item in eligible_results
            ]
        result_payload["meta"]["target_met_cases"] = sum(quality_met)
        result_payload["meta"]["limited_cases"] = sum(
            item.get("termination") in {"time_limit", "expansion_limit"}
            for item in eligible_results
        )
        result_payload["meta"]["quality_complete"] = (
            result_payload["meta"]["run_complete"]
            and len(quality_met) == len(eligible_ids)
            and all(quality_met)
        )
    atomic_write_json(out_path, result_payload)
    print(
        f"{out_path.name}: finished done={done} failed={failed} "
        f"total_ok={result_payload['meta']['completed_cases']} "
        f"quality_complete={result_payload['meta'].get('quality_complete')}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument(
        "--search-mode", choices=("beam", "anytime"), default="anytime"
    )
    parser.add_argument("--time-limit-s", type=float, default=60.0)
    parser.add_argument("--max-expansions", type=int, default=16)
    parser.add_argument(
        "--target-gap",
        type=float,
        default=0.03,
        help="stop a case only after the certified (UB-LB)/LB reaches this value",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--max-active-n", type=int, default=None)
    parser.add_argument("--files", nargs="*", type=Path, default=list(DEFAULT_FILES))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")

    for path in args.files:
        annotate_file(
            path,
            args.out_dir,
            args.beam_width,
            args.limit,
            args.force,
            args.save_every,
            args.max_active_n,
            args.search_mode,
            args.time_limit_s,
            args.max_expansions,
            args.target_gap,
            args.workers,
        )


if __name__ == "__main__":
    main()
