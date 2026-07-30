#!/usr/bin/env python3
"""Describe action templates used by replayed bounded-window optima.

This report is proposal data for the candidate generator, not a candidate
sufficiency proof.  Frequency can prioritize templates; only a closed-loop
candidate oracle may justify deleting one.  Every selected history is loaded
from the hash-audited direct sources recorded by completed window audits,
materialized under any required equal-load ID relabel, and replayed again.
"""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_directed_case_classification as history_audit  # noqa: E402
import four_stage_scheduler as reference  # noqa: E402
from run_four_stage_reference import deserialize_action  # noqa: E402


TICK_CC = reference.SCHEDULE_TIME_QUANTUM_CC


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _window_name(window: tuple[int, int]) -> str:
    top, bottom = window
    return f"top{top}" if bottom == 0 else f"top{top}+bottom{bottom}"


def _parse_window(name: str) -> tuple[int, int]:
    if "+bottom" in name:
        top, bottom = name.removeprefix("top").split("+bottom", 1)
        return int(top), int(bottom)
    return int(name.removeprefix("top")), 0


def _family(action: reference.StageAction) -> str:
    if action.c2_eid >= 0 and action.c2_eid == action.c3_eid:
        return "SPLIT"
    if action.c2_eid >= 0 and action.c3_eid >= 0:
        return "PAIR"
    if action.c2_eid >= 0 or action.c3_eid >= 0:
        return "SINGLE"
    if action.pf_eid >= 0:
        return "PREFETCH"
    return "OTHER"


def _shape_name(value) -> str:
    return "NONE" if value is None else str(value.name)


def _dma_name(value) -> str:
    return str(value.name)


def _adaptive_shapes(ntok: int, s1_cached: bool, s3_cached: bool) -> tuple[str, str]:
    if ntok <= 0:
        return "NONE", "NONE"
    if ntok >= 7:
        uncached = (reference.SHAPE_A.name, reference.SHAPE_B.name)
    elif ntok >= 3:
        uncached = (reference.SHAPE_B.name, reference.SHAPE_B.name)
    else:
        uncached = (reference.SHAPE_C.name, reference.SHAPE_C.name)
    return (
        reference.SHAPE_C.name if s1_cached else uncached[0],
        reference.SHAPE_C.name if s3_cached else uncached[1],
    )


def _rank_label(rank: int | None, entries: int, window: tuple[int, int]) -> str:
    if rank is None:
        return "NONE"
    top, bottom = window
    if rank < min(top, entries):
        return f"T{rank}"
    if bottom and rank >= max(min(top, entries), entries - bottom):
        return f"B{entries - 1 - rank}"
    return "RESIDENT"


def _class_label(
    eid: int,
    rank: int | None,
    state: reference.BeamState,
    visible: frozenset[int],
) -> str:
    """Normalize equal-load nonresident entries after window selection."""
    if eid < 0 or rank is None:
        return "NONE"
    resident = "".join(
        str(cluster)
        for cluster, snap in ((2, state.c2), (3, state.c3))
        if snap.pf_eid == eid
    )
    if resident:
        return f"R{resident}"
    loads = []
    for candidate_eid, ntok in state.remaining:
        if candidate_eid not in visible:
            continue
        if any(snap.pf_eid == candidate_eid for snap in (state.c2, state.c3)):
            continue
        if int(ntok) not in loads:
            loads.append(int(ntok))
    selected_load = int(state.remaining[rank][1])
    return f"L{loads.index(selected_load)}"


def _split_cut_label(action: reference.StageAction) -> str:
    if action.c2_eid < 0 or action.c2_eid != action.c3_eid:
        return "NONE"
    left, right = int(action.c2_ntok), int(action.c3_ntok)
    low, high = sorted((left, right))
    if low == high:
        return "HALF"
    return f"{low}+{high}"


def _hw_v2_high_level_verdict(action_row: dict) -> str:
    """Classify only expert/family expressibility, ignoring physical lowering."""
    if action_row["remaining_before"] == 1:
        return "covered_terminal_special"
    family = action_row["family"]
    mode = action_row["decision_mode"]
    ranks = action_row["rank_indices"]
    if family == "PREFETCH":
        return "missing_standalone_prefetch"
    if mode == "SYNC":
        if family == "PAIR":
            pair = tuple(sorted((ranks["c2"], ranks["c3"])))
            return (
                "covered_sync_pair"
                if pair in {(0, 1), (1, 2), (2, 3)}
                else "missing_sync_pair_rank"
            )
        if family == "SPLIT":
            return (
                "covered_sync_split_top0"
                if ranks["c2"] == ranks["c3"] == 0
                else "missing_sync_split_rank"
            )
        if family == "SINGLE":
            selected = ranks["c2"] if ranks["c2"] is not None else ranks["c3"]
            return (
                "covered_sync_fallback_top0"
                if selected == 0
                else "missing_sync_single_rank"
            )
        return "missing_other_family"
    if family != "SINGLE":
        return "missing_one_idle_non_single"
    selected = ranks["c2"] if ranks["c2"] is not None else ranks["c3"]
    return (
        "covered_one_idle_top0"
        if selected == 0
        else "missing_one_idle_rank"
    )


def _source_rows(path: Path, cache: dict[Path, dict[str, dict]]) -> dict[str, dict]:
    resolved = path.resolve()
    if resolved not in cache:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        cache[resolved] = {str(row["name"]): row for row in payload["cases"]}
    return cache[resolved]


def _materialize_history(
    audit_row: dict,
    window: tuple[int, int],
    source_cache: dict[Path, dict[str, dict]],
) -> tuple[dict, tuple[reference.StageAction, ...], dict]:
    name = str(audit_row["name"])
    for source in audit_row.get("direct_witness_sources", []):
        path = Path(source["source"])
        rows = _source_rows(path, source_cache)
        if name not in rows:
            continue
        row = rows[name]
        if source.get("equal_load_id_relabel"):
            found, actions = history_audit._symmetry_relabel_history(
                row, *window
            )
            if not found or actions is None:
                raise RuntimeError(f"{name}: audited equal-load relabel no longer replays")
        else:
            actions = tuple(deserialize_action(raw) for raw in row["actions"])
        return row, actions, source
    raise RuntimeError(f"{name}: no materializable direct witness source")


def _counter(counter: Counter) -> dict:
    return {str(key): int(counter[key]) for key in sorted(counter, key=str)}


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--window", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    requested_windows = set(args.window)
    source_cache: dict[Path, dict[str, dict]] = {}
    seen_case_windows: set[tuple[str, str]] = set()
    records = []
    audit_evidence = []
    for audit_path in args.audit:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit.get("complete"):
            raise SystemExit(f"window audit is incomplete: {audit_path}")
        audit_evidence.append(
            {"path": str(audit_path.resolve()), "sha256": _sha256(audit_path)}
        )
        for audit_row in audit["results"]:
            window_name = str(audit_row["window"])
            if requested_windows and window_name not in requested_windows:
                continue
            if audit_row.get("window_status") != "proved_sufficient_direct":
                continue
            key = (str(audit_row["name"]), window_name)
            if key in seen_case_windows:
                continue
            seen_case_windows.add(key)
            window = _parse_window(window_name)
            source_row, actions, source = _materialize_history(
                audit_row, window, source_cache
            )
            token_dist = {
                eid: int(ntok)
                for eid, ntok in enumerate(source_row["counts"])
                if int(ntok) > 0
            }
            state = reference.FourStageScheduler(token_dist)._initial_state()
            action_rows = []
            for index, action in enumerate(actions):
                rank_by_eid = {
                    eid: rank for rank, (eid, _ntok) in enumerate(state.remaining)
                }
                visible = reference.candidate_window_visible_eids(
                    state.c2, state.c3, state.remaining, window
                )
                if not reference.action_within_candidate_window(action, visible):
                    raise RuntimeError(
                        f"{key}: materialized action {index} violates its window"
                    )
                entries = len(state.remaining)
                ranks = {
                    "c2": rank_by_eid.get(int(action.c2_eid)),
                    "c3": rank_by_eid.get(int(action.c3_eid)),
                    "pf": rank_by_eid.get(int(action.pf_eid)),
                }
                labels = {
                    slot: _rank_label(rank, entries, window)
                    for slot, rank in ranks.items()
                }
                class_labels = {
                    "c2": _class_label(
                        int(action.c2_eid), ranks["c2"], state, visible
                    ),
                    "c3": _class_label(
                        int(action.c3_eid), ranks["c3"], state, visible
                    ),
                    "pf": _class_label(
                        int(action.pf_eid), ranks["pf"], state, visible
                    ),
                }
                visible_nonresident_loads = []
                resident_eids = {
                    snap.pf_eid
                    for snap in (state.c2, state.c3)
                    if snap.pf_eid in visible
                }
                for candidate_eid, ntok in state.remaining:
                    if candidate_eid not in visible or candidate_eid in resident_eids:
                        continue
                    if int(ntok) not in visible_nonresident_loads:
                        visible_nonresident_loads.append(int(ntok))
                family = _family(action)
                decision_mode = (
                    "SYNC"
                    if state.c2.task_end == state.c3.task_end
                    else "C2_EARLY"
                    if state.c2.task_end < state.c3.task_end
                    else "C3_EARLY"
                )
                physical = {
                    "family": family,
                    "decision_mode": decision_mode,
                    "c2_class": class_labels["c2"],
                    "c3_class": class_labels["c3"],
                    "pf_class": class_labels["pf"],
                    "split_cut": _split_cut_label(action),
                    "c2_s1": _shape_name(action.c2_shape_s1),
                    "c2_s3": _shape_name(action.c2_shape_s3),
                    "c3_s1": _shape_name(action.c3_shape_s1),
                    "c3_s3": _shape_name(action.c3_shape_s3),
                    "c2_dma_s1": _dma_name(action.c2_dma_s1),
                    "c2_dma_s3": _dma_name(action.c2_dma_s3),
                    "c2_s2pf": _dma_name(action.c2_s2pf_dma),
                    "c3_dma_s1": _dma_name(action.c3_dma_s1),
                    "c3_dma_s3": _dma_name(action.c3_dma_s3),
                    "c3_s2pf": _dma_name(action.c3_s2pf_dma),
                    "s4pf_dma": _dma_name(action.pf_dma),
                    "c2_s1_cached": bool(action.c2_s1_cached),
                    "c2_s3_cached": bool(action.c2_s3_cached),
                    "c3_s1_cached": bool(action.c3_s1_cached),
                    "c3_s3_cached": bool(action.c3_s3_cached),
                }
                selected_tokens = {
                    "c2": int(action.c2_ntok),
                    "c3": int(action.c3_ntok),
                }
                shape_matches = {}
                for slot in ("c2", "c3"):
                    ntok = selected_tokens[slot]
                    if ntok <= 0:
                        shape_matches[slot] = None
                        continue
                    expected_s1, expected_s3 = _adaptive_shapes(
                        ntok,
                        bool(getattr(action, f"{slot}_s1_cached")),
                        bool(getattr(action, f"{slot}_s3_cached")),
                    )
                    shape_matches[slot] = (
                        physical[f"{slot}_s1"] == expected_s1
                        and physical[f"{slot}_s3"] == expected_s3
                    )
                action_rows.append(
                    {
                        "index": index,
                        "remaining_before": entries,
                        "family": family,
                        "decision_mode": decision_mode,
                        "rank_indices": ranks,
                        "rank_labels": labels,
                        "class_labels": class_labels,
                        "visible_nonresident_load_classes": len(
                            visible_nonresident_loads
                        ),
                        "visible_resident_experts": len(resident_eids),
                        "selected_tokens": selected_tokens,
                        "adaptive_shape_matches": shape_matches,
                        "physical_template": physical,
                    }
                )
                action_rows[-1]["hw_v2_high_level_verdict"] = (
                    _hw_v2_high_level_verdict(action_rows[-1])
                )
                state = reference.apply_action(state, action)
            replay_cc = reference.validate_schedule_history(actions, token_dist)
            expected = Fraction(str(source_row["best_reference_ticks"])) * TICK_CC
            if expected.denominator != 1 or replay_cc != int(expected) or state.remaining:
                raise RuntimeError(f"{key}: terminal replay mismatch")
            records.append(
                {
                    "name": key[0],
                    "window": window_name,
                    "target_ticks": str(source_row["best_reference_ticks"]),
                    "source": source,
                    "action_count": len(actions),
                    "actions": action_rows,
                }
            )

    if not records:
        raise SystemExit("no direct bounded-window witnesses selected")
    by_window = {}
    for window_name in sorted({row["window"] for row in records}):
        selected = [row for row in records if row["window"] == window_name]
        actions = [action for row in selected for action in row["actions"]]
        families = Counter(action["family"] for action in actions)
        modes = Counter(action["decision_mode"] for action in actions)
        rank_labels = Counter(
            label
            for action in actions
            for label in action["rank_labels"].values()
            if label != "NONE"
        )
        class_labels = Counter(
            label
            for action in actions
            for label in action["class_labels"].values()
            if label != "NONE"
        )
        high_level = Counter(
            json.dumps(
                {
                    "family": action["family"],
                    "decision_mode": action["decision_mode"],
                    "classes": action["class_labels"],
                },
                sort_keys=True,
            )
            for action in actions
        )
        physical = Counter(
            json.dumps(action["physical_template"], sort_keys=True)
            for action in actions
        )
        hw_v2_verdicts = Counter(
            action["hw_v2_high_level_verdict"] for action in actions
        )
        shape_slots = [
            matched
            for action in actions
            for matched in action["adaptive_shape_matches"].values()
            if matched is not None
        ]
        shape_actions = [
            action
            for action in actions
            if action["family"] in {"PAIR", "SPLIT", "SINGLE"}
        ]
        visibility_by_mode = {}
        for mode in ("SYNC", "C2_EARLY", "C3_EARLY"):
            mode_actions = [
                action for action in actions if action["decision_mode"] == mode
            ]
            if not mode_actions:
                continue
            class_counts = sorted(
                action["visible_nonresident_load_classes"]
                for action in mode_actions
            )
            resident_counts = sorted(
                action["visible_resident_experts"] for action in mode_actions
            )
            visibility_by_mode[mode] = {
                "states": len(mode_actions),
                "nonresident_load_classes_max": max(class_counts),
                "nonresident_load_classes_p95": class_counts[
                    round(0.95 * (len(class_counts) - 1))
                ],
                "resident_experts_max": max(resident_counts),
            }
        by_window[window_name] = {
            "cases": len(selected),
            "actions": len(actions),
            "action_families": _counter(families),
            "decision_modes": _counter(modes),
            "rank_label_uses": _counter(rank_labels),
            "normalized_class_uses": _counter(class_labels),
            "unique_high_level_templates": len(high_level),
            "unique_physical_templates": len(physical),
            "hw_v2_high_level_verdicts": _counter(hw_v2_verdicts),
            "hw_v2_high_level_covered_actions": sum(
                count
                for verdict, count in hw_v2_verdicts.items()
                if verdict.startswith("covered_")
            ),
            "hw_v2_high_level_missing_actions": sum(
                count
                for verdict, count in hw_v2_verdicts.items()
                if verdict.startswith("missing_")
            ),
            "adaptive_shape_rule": {
                "matched_slots": sum(shape_slots),
                "total_slots": len(shape_slots),
                "all_slots_matched_actions": sum(
                    all(
                        value is None or value
                        for value in action["adaptive_shape_matches"].values()
                    )
                    for action in shape_actions
                ),
                "consuming_actions": len(shape_actions),
            },
            "visible_class_bounds_by_decision_mode": visibility_by_mode,
            "high_level_template_frequency": _counter(high_level),
            "physical_template_frequency": _counter(physical),
        }

    payload = {
        "schema": "window_witness_action_templates_v1",
        "interpretation": {
            "role": "descriptive proposal data for bounded candidate templates",
            "not_proven": "frequency does not prove candidate necessity or sufficiency",
            "required_next_step": "same-input K=16/24/32 explicit-DMA candidate oracle",
        },
        "audits": audit_evidence,
        "summary": {
            "case_window_histories": len(records),
            "unique_cases": len({row["name"] for row in records}),
            "by_window": by_window,
        },
        "records": records,
    }
    _atomic_write(args.output, payload)
    compact = {
        name: {
            key: value
            for key, value in row.items()
            if key not in {"high_level_template_frequency", "physical_template_frequency"}
        }
        for name, row in by_window.items()
    }
    print(json.dumps(compact, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
