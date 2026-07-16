#!/usr/bin/env python3
"""Replay reference histories and census hardware-relevant action templates.

This is a deterministic, read-only analysis of existing reference results.  It
does not generate alternative actions, run continuation searches, or fit a
policy.  Its purpose is to establish the state/action population and measure
whether rank-bounded expert pools could represent reference-path actions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import time

from four_stage_scheduler import (
    BeamState,
    DmaBinding,
    FourStageScheduler,
    StageAction,
    apply_action,
    clear_scheduler_caches,
    validate_schedule_history,
)
from run_four_stage_reference import deserialize_action


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUTS = tuple(
    ROOT / "results" / "final_reference" / f"scheduler_reference_E{e}.json"
    for e in (8, 32, 64)
)
DEFAULT_OUT = ROOT / "results" / "policy_search" / "candidate_census.json"
POOL_CONFIGS = ((4, 0), (4, 2), (4, 4), (8, 0), (8, 2), (8, 4))


def quality_ok(item: dict, quality: str) -> bool:
    if not item.get("analysis_eligible", False):
        return False
    if quality == "proven":
        return bool(item.get("proven_optimal", False))
    if quality == "within3":
        return float(item.get("optimality_gap", math.inf)) <= 0.03
    if quality == "eligible":
        return True
    raise ValueError(quality)


def stratified_keys(keys: list[str], count: int) -> list[str]:
    if count < 0 or count >= len(keys):
        return keys
    if count == 0:
        return []
    if count == 1:
        return [keys[0]]
    indices = sorted(
        set(round(i * (len(keys) - 1) / (count - 1)) for i in range(count))
    )
    return [keys[index] for index in indices]


def action_family(action: StageAction) -> str:
    if (
        action.pf_cluster in (2, 3)
        or action.c2_eid == -2
        or action.c3_eid == -2
        or action.tag.startswith("PF-")
    ):
        return "PREFETCH"
    if action.c2_eid >= 0 and action.c3_eid >= 0:
        return "SPLIT" if action.c2_eid == action.c3_eid else "PAIR"
    return "SINGLE"


def decision_mode(state: BeamState) -> str:
    if len(state.remaining) == 1:
        return "LAST_EXPERT"
    if state.c2.task_end == state.c3.task_end:
        return "BOTH_IDLE"
    return "ONE_IDLE"


def selected_eids(action: StageAction) -> tuple[int, ...]:
    if action_family(action) == "PREFETCH":
        return (action.pf_eid,) if action.pf_eid >= 0 else ()
    return tuple(sorted({eid for eid in (action.c2_eid, action.c3_eid) if eid >= 0}))


def rank_map(state: BeamState) -> dict[int, int]:
    return {eid: rank for rank, (eid, _) in enumerate(state.remaining)}


def named_eids(state: BeamState) -> set[int]:
    return {
        snap.pf_eid
        for snap in (state.c2, state.c3)
        if snap.pf_eid >= 0 and any(eid == snap.pf_eid for eid, _ in state.remaining)
    }


def expert_equivalence_key(state: BeamState, eid: int) -> tuple:
    ntok = next(ntok for candidate, ntok in state.remaining if candidate == eid)

    def snap_role(snap) -> tuple:
        named = snap.pf_eid == eid
        return (named, bool(snap.pf_full) if named else False)

    return (ntok, snap_role(state.c2), snap_role(state.c3))


def pool_coverage(
    state: BeamState,
    action: StageAction,
    rank_limit: int,
    tail_count: int = 0,
) -> tuple[bool, bool]:
    """Return literal and timing-equivalence-aware expert-pool coverage."""
    chosen = selected_eids(action)
    if not chosen:
        return True, True
    ranks = rank_map(state)
    named = named_eids(state)
    pool = [
        eid
        for rank, (eid, _) in enumerate(state.remaining)
        if rank < rank_limit
        or (tail_count > 0 and rank >= len(state.remaining) - tail_count)
        or eid in named
    ]
    literal = all(eid in pool for eid in chosen)

    # Expert IDs have no timing meaning beyond token count and concrete
    # residency/prefetch naming.  Match chosen experts injectively against pool
    # members from the same future-observable equivalence class.
    available = Counter(expert_equivalence_key(state, eid) for eid in pool)
    needed = Counter(expert_equivalence_key(state, eid) for eid in chosen)
    equivalent = all(available[key] >= count for key, count in needed.items())
    return literal, equivalent


def shape_name(shape) -> str:
    if shape is None:
        return "-"
    return shape.name.split("(", 1)[0]


def rank_bucket(state: BeamState, eid: int, ranks: dict[int, int]) -> str:
    if eid < 0:
        return "NONE"
    roles = []
    if state.c2.pf_eid == eid:
        roles.append("C2_FULL" if state.c2.pf_full else "C2_S1")
    if state.c3.pf_eid == eid:
        roles.append("C3_FULL" if state.c3.pf_full else "C3_S1")
    if roles:
        return "+".join(roles)
    rank = ranks[eid]
    if rank < 4:
        return f"R{rank}"
    if rank < 8:
        return "R4_7"
    return "R8_PLUS"


def start_class(start: int, own, peer) -> str:
    if start < 0:
        return "NONE"
    if start == own.task_end:
        return "OWN_RELEASE"
    if start == max(own.task_end, peer.task_end):
        return "JOINT_RELEASE"
    if start == peer.task_end:
        return "PEER_RELEASE"
    own_events = {
        own.dma1_end: "OWN_DMA1_END",
        own.s2pf_end: "OWN_S2PF_END",
        own.dma3_end: "OWN_DMA3_END",
        own.pf_end: "OWN_PF_END",
    }
    peer_events = {
        peer.dma1_end: "PEER_DMA1_END",
        peer.s2pf_end: "PEER_S2PF_END",
        peer.dma3_end: "PEER_DMA3_END",
        peer.pf_end: "PEER_PF_END",
    }
    if start in own_events and start >= 0:
        return own_events[start]
    if start in peer_events and start >= 0:
        return peer_events[start]
    return "EVENT_OFFSET"


def split_cut_class(action: StageAction) -> str:
    if action_family(action) != "SPLIT":
        return "-"
    left, right = action.c2_ntok, action.c3_ntok
    total = left + right
    small = min(left, right)
    if abs(left - right) <= 1:
        balance = "BALANCED"
    elif 4 * small <= total:
        balance = "HEAVY_SKEW"
    elif 3 * small <= total:
        balance = "SKEW"
    else:
        balance = "NEAR_BALANCED"
    if left % 8 == 0 and right % 8 == 0:
        alignment = "M8"
    elif left % 4 == 0 and right % 4 == 0:
        alignment = "M4"
    elif left % 2 == 0 and right % 2 == 0:
        alignment = "M2"
    else:
        alignment = "ODD"
    return f"{balance}:{alignment}"


def swap_dma_name(name: str) -> str:
    if name == DmaBinding.IDMA.name:
        return DmaBinding.XDMA.name
    if name == DmaBinding.XDMA.name:
        return DmaBinding.IDMA.name
    return name


def slot_template(
    state: BeamState,
    action: StageAction,
    cluster: int,
    ranks: dict[int, int],
) -> dict:
    if cluster == 2:
        eid = action.c2_eid
        ntok = action.c2_ntok
        own, peer = state.c2, state.c3
        start = action.c2_start
        s1, s3 = action.c2_shape_s1, action.c2_shape_s3
        s1_cached, s3_cached = action.c2_s1_cached, action.c2_s3_cached
        dma_s1, dma_s3 = action.c2_dma_s1, action.c2_dma_s3
        s2pf_start, s2pf_dma = action.c2_s2pf_start, action.c2_s2pf_dma
    else:
        eid = action.c3_eid
        ntok = action.c3_ntok
        own, peer = state.c3, state.c2
        start = action.c3_start
        s1, s3 = action.c3_shape_s1, action.c3_shape_s3
        s1_cached, s3_cached = action.c3_s1_cached, action.c3_s3_cached
        dma_s1, dma_s3 = action.c3_dma_s1, action.c3_dma_s3
        s2pf_start, s2pf_dma = action.c3_s2pf_start, action.c3_s2pf_dma
    if eid == -2:
        kind = "PREFETCH"
        role = rank_bucket(state, action.pf_eid, ranks)
    elif eid >= 0:
        kind = "ASSIGN"
        role = rank_bucket(state, eid, ranks)
    else:
        kind = "NONE"
        role = "NONE"
    return {
        "kind": kind,
        "role": role,
        "ntok_class": (
            "0"
            if ntok <= 0
            else "1"
            if ntok == 1
            else "2_3"
            if ntok <= 3
            else "4_7"
            if ntok <= 7
            else "8_PLUS"
        ),
        "s1": shape_name(s1),
        "s3": shape_name(s3),
        "s1_cached": bool(s1_cached),
        "s3_cached": bool(s3_cached),
        "dma_s1": dma_s1.name,
        "dma_s3": dma_s3.name,
        "s2pf": s2pf_start >= 0,
        "s2pf_dma": s2pf_dma.name,
        "start": start_class(start, own, peer),
    }


def canonical_template(state: BeamState, action: StageAction) -> str:
    ranks = rank_map(state)
    payload = {
        "mode": decision_mode(state),
        "family": action_family(action),
        "wait": action.tag.startswith("WAIT-"),
        "c2": slot_template(state, action, 2, ranks),
        "c3": slot_template(state, action, 3, ranks),
        "split": split_cut_class(action),
        "pf_role": (
            rank_bucket(state, action.pf_eid, ranks)
            if action.pf_eid >= 0
            else "NONE"
        ),
        "pf_shape": shape_name(action.pf_shape),
        "pf_dma": action.pf_dma.name,
    }

    def encode(value: dict) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    # The physical IDMA/XDMA names are interchangeable.  Canonicalize the
    # complete action under a global lane swap while retaining cluster roles.
    swapped = json.loads(encode(payload))
    for key in ("dma_s1", "dma_s3", "s2pf_dma"):
        swapped["c2"][key] = swap_dma_name(swapped["c2"][key])
        swapped["c3"][key] = swap_dma_name(swapped["c3"][key])
    swapped["pf_dma"] = swap_dma_name(swapped["pf_dma"])
    return min(encode(payload), encode(swapped))


def counter_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common()}


def coverage_report(entry: dict) -> dict:
    total = entry["actions"]
    report = {
        "actions": total,
        "literal_covered": entry["literal"],
        "equivalence_covered": entry["equivalent"],
        "literal_fraction": entry["literal"] / total if total else None,
        "equivalence_fraction": entry["equivalent"] / total if total else None,
        "literal_misses_by_family": counter_dict(entry["literal_miss_family"]),
        "equivalence_misses_by_family": counter_dict(entry["equiv_miss_family"]),
        "top_equivalence_miss_signatures": counter_dict(
            Counter(dict(entry["equiv_miss_signature"].most_common(50)))
        ),
    }
    return report


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--dataset-split",
        action="append",
        choices=("discovery", "validation", "blind_test"),
        default=None,
        help="repeat to include multiple splits; default: discovery",
    )
    parser.add_argument(
        "--quality", choices=("proven", "within3", "eligible"), default="proven"
    )
    parser.add_argument(
        "--sample-per-file",
        type=int,
        default=-1,
        help="deterministic stratified sample after filters; -1 means all",
    )
    parser.add_argument("--top-templates", type=int, default=200)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--strict-validate",
        action="store_true",
        help="also run the independent full history validator for every case",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    splits = set(args.dataset_split or ("discovery",))
    started = time.perf_counter()

    cases = decisions = consuming_actions = prefetch_actions = 0
    by_e = Counter()
    by_split = Counter()
    by_quality = Counter()
    decision_modes = Counter()
    family_counts = Counter()
    mode_family_counts = Counter()
    remaining_counts = Counter()
    selected_rank_counts = Counter()
    selected_tail_rank_counts = Counter()
    max_rank_per_action = Counter()
    min_tail_rank_per_action = Counter()
    pair_rank_pairs = Counter()
    pair_tail_rank_pairs = Counter()
    split_rank_counts = Counter()
    prefetch_rank_counts = Counter()
    split_cut_counts = Counter()
    shape_profile_counts = Counter()
    dma_profile_counts = Counter()
    cache_pattern_counts = Counter()
    s2pf_pattern_counts = Counter()
    templates = Counter()
    coverage = {
        (rank_limit, tail_count): {
            "actions": 0,
            "literal": 0,
            "equivalent": 0,
            "literal_miss_family": Counter(),
            "equiv_miss_family": Counter(),
            "equiv_miss_signature": Counter(),
        }
        for rank_limit, tail_count in POOL_CONFIGS
    }
    input_summary = []

    for path in args.inputs:
        payload = json.loads(path.read_text())
        results = payload["results"]
        keys = [
            key
            for key, item in results.items()
            if item.get("dataset_split") in splits and quality_ok(item, args.quality)
        ]
        keys = stratified_keys(keys, args.sample_per_file)
        input_summary.append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "matching_cases": len(keys),
            }
        )

        for key in keys:
            item = results[key]
            dist = {int(eid): int(ntok) for eid, ntok in item["dist"].items()}
            c2_cache = int(item.get("initial_cache_c2", -1))
            c3_cache = int(item.get("initial_cache_c3", -1))
            actions = tuple(deserialize_action(raw) for raw in item["actions"])
            if args.strict_validate:
                validated = validate_schedule_history(
                    actions,
                    dist,
                    initial_cache_c2=c2_cache,
                    initial_cache_c3=c3_cache,
                )
                if validated != int(item["makespan_cc"]):
                    raise RuntimeError(
                        f"{path.name}:{key}: validator makespan {validated} "
                        f"!= reference {item['makespan_cc']}"
                    )

            state = FourStageScheduler(
                dist,
                initial_cache_c2=c2_cache,
                initial_cache_c3=c3_cache,
            )._initial_state()
            for action in actions:
                family = action_family(action)
                mode = decision_mode(state)
                ranks = rank_map(state)
                chosen = selected_eids(action)
                chosen_ranks = tuple(sorted(ranks[eid] for eid in chosen))
                chosen_tail_ranks = tuple(
                    sorted(len(state.remaining) - 1 - ranks[eid] for eid in chosen)
                )

                decisions += 1
                decision_modes[mode] += 1
                family_counts[family] += 1
                mode_family_counts[f"{mode}:{family}"] += 1
                remaining_counts[len(state.remaining)] += 1
                selected_rank_counts.update(chosen_ranks)
                selected_tail_rank_counts.update(chosen_tail_ranks)
                if chosen_ranks:
                    max_rank_per_action[max(chosen_ranks)] += 1
                    min_tail_rank_per_action[min(chosen_tail_ranks)] += 1
                if family == "PAIR":
                    pair_rank_pairs[str(chosen_ranks)] += 1
                    pair_tail_rank_pairs[str(chosen_tail_ranks)] += 1
                elif family == "SPLIT" and chosen_ranks:
                    split_rank_counts[chosen_ranks[0]] += 1
                elif family == "PREFETCH" and chosen_ranks:
                    prefetch_rank_counts[chosen_ranks[0]] += 1

                if family == "PREFETCH":
                    prefetch_actions += 1
                else:
                    consuming_actions += 1

                split_cut_counts[split_cut_class(action)] += 1
                shape_profile_counts[
                    f"{shape_name(action.c2_shape_s1)}/{shape_name(action.c2_shape_s3)}|"
                    f"{shape_name(action.c3_shape_s1)}/{shape_name(action.c3_shape_s3)}"
                ] += 1
                lane_pattern = (
                    action.c2_dma_s1.name,
                    action.c2_dma_s3.name,
                    action.c2_s2pf_dma.name,
                    action.c3_dma_s1.name,
                    action.c3_dma_s3.name,
                    action.c3_s2pf_dma.name,
                    action.pf_dma.name,
                )
                swapped_lane_pattern = tuple(swap_dma_name(name) for name in lane_pattern)
                dma_profile_counts[str(min(lane_pattern, swapped_lane_pattern))] += 1
                cache_pattern_counts[
                    str(
                        (
                            action.c2_s1_cached,
                            action.c2_s3_cached,
                            action.c3_s1_cached,
                            action.c3_s3_cached,
                        )
                    )
                ] += 1
                s2pf_pattern_counts[
                    str((action.c2_s2pf_start >= 0, action.c3_s2pf_start >= 0))
                ] += 1
                templates[canonical_template(state, action)] += 1

                for rank_limit, tail_count in POOL_CONFIGS:
                    literal, equivalent = pool_coverage(
                        state, action, rank_limit, tail_count
                    )
                    entry = coverage[(rank_limit, tail_count)]
                    entry["actions"] += 1
                    entry["literal"] += int(literal)
                    entry["equivalent"] += int(equivalent)
                    if not literal:
                        entry["literal_miss_family"][family] += 1
                    if not equivalent:
                        entry["equiv_miss_family"][family] += 1
                        remaining_bucket = (
                            "1"
                            if len(state.remaining) == 1
                            else "2_4"
                            if len(state.remaining) <= 4
                            else "5_8"
                            if len(state.remaining) <= 8
                            else "9_16"
                            if len(state.remaining) <= 16
                            else "17_PLUS"
                        )
                        signature = (
                            f"{mode}:{family}:remaining={remaining_bucket}:"
                            f"ranks={chosen_ranks}:tail={chosen_tail_ranks}"
                        )
                        entry["equiv_miss_signature"][signature] += 1

                state = apply_action(state, action)

            if state.remaining:
                raise RuntimeError(
                    f"{path.name}:{key}: replay left {len(state.remaining)} experts"
                )
            if state.g_score != int(item["makespan_cc"]):
                raise RuntimeError(
                    f"{path.name}:{key}: replay makespan {state.g_score} "
                    f"!= reference {item['makespan_cc']}"
                )
            if len(actions) != int(item.get("num_actions", len(actions))):
                raise RuntimeError(f"{path.name}:{key}: action count mismatch")

            cases += 1
            by_e[int(item["e_total"])] += 1
            by_split[str(item["dataset_split"])] += 1
            by_quality[str(item["quality_class"])] += 1
            clear_scheduler_caches()
            if args.progress_every > 0 and cases % args.progress_every == 0:
                print(
                    f"cases={cases} decisions={decisions} "
                    f"templates={len(templates)} elapsed_s={time.perf_counter()-started:.1f}",
                    flush=True,
                )

    top_templates = [
        {"count": count, "template": json.loads(signature)}
        for signature, count in templates.most_common(args.top_templates)
    ]
    report = {
        "schema": "scheduler_candidate_census_v1",
        "purpose": "reference-history replay and candidate-template census only",
        "filters": {
            "dataset_splits": sorted(splits),
            "quality": args.quality,
            "sample_per_file": args.sample_per_file,
            "strict_validate": args.strict_validate,
        },
        "inputs": input_summary,
        "summary": {
            "cases": cases,
            "decisions": decisions,
            "consuming_actions": consuming_actions,
            "prefetch_actions": prefetch_actions,
            "unique_canonical_templates": len(templates),
            "runtime_s": time.perf_counter() - started,
        },
        "cases_by_e": counter_dict(by_e),
        "cases_by_split": counter_dict(by_split),
        "cases_by_quality": counter_dict(by_quality),
        "decision_mode_counts": counter_dict(decision_modes),
        "action_family_counts": counter_dict(family_counts),
        "mode_family_counts": counter_dict(mode_family_counts),
        "remaining_expert_count": counter_dict(remaining_counts),
        "selected_expert_rank_occurrences": counter_dict(selected_rank_counts),
        "selected_expert_tail_rank_occurrences": counter_dict(
            selected_tail_rank_counts
        ),
        "max_selected_rank_per_action": counter_dict(max_rank_per_action),
        "min_selected_tail_rank_per_action": counter_dict(min_tail_rank_per_action),
        "pair_rank_pairs": counter_dict(pair_rank_pairs),
        "pair_tail_rank_pairs": counter_dict(pair_tail_rank_pairs),
        "split_rank_counts": counter_dict(split_rank_counts),
        "prefetch_rank_counts": counter_dict(prefetch_rank_counts),
        "split_cut_class_counts": counter_dict(split_cut_counts),
        "shape_profile_counts": counter_dict(shape_profile_counts),
        "dma_profile_counts": counter_dict(dma_profile_counts),
        "cache_pattern_counts": counter_dict(cache_pattern_counts),
        "s2pf_pattern_counts": counter_dict(s2pf_pattern_counts),
        "expert_pool_coverage": {
            (
                f"R{rank_limit}"
                f"{'_plus_bottom'+str(tail_count) if tail_count else ''}"
                "_plus_concrete_residency"
            ): coverage_report(entry)
            for (rank_limit, tail_count), entry in coverage.items()
        },
        "top_canonical_templates": top_templates,
        "interpretation_limits": [
            "Frequency and coverage do not prove that a rare template is removable.",
            "Alternative-action continuation values are intentionally absent.",
            "Candidate-oracle selection begins only after this census is locked.",
        ],
    }
    atomic_write_json(args.out, report)
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
