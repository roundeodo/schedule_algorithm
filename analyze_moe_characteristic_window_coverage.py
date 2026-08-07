#!/usr/bin/env python3
"""Combine directed window witnesses for the 65 MoE-characteristic cases.

The directed cases were motivated by hot-anchor and long-cold-tail routing
properties reported for OLMoE, but they are synthetic Top-2 distributions used
to represent broader MoE routing characteristics.  Coverage requires a saved,
replay-valid optimal history whose every action is visible in the tested window.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import analyze_window_witness_action_templates as witness
import four_stage_scheduler as reference


HERE = Path(__file__).resolve().parent
LEGACY = HERE / "results" / "legacy_scheduler_algorithm" / "policy_search"
BASE_AUDIT = LEGACY / "olmoe_65_direct_window_witness_audit_v1.json"
TOP5_AUDIT = (
    LEGACY
    / "window_exact"
    / "olmoe_65_stagec_top5_coverage_audit_v1.json"
)
OUTPUT = (
    HERE
    / "results"
    / "policy_search"
    / "moe_characteristic_window_coverage.json"
)
WINDOWS = ((4, 0), (4, 2), (5, 1), (6, 2), (8, 8))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _window_name(window: tuple[int, int]) -> str:
    top, bottom = window
    return f"top{top}" if bottom == 0 else f"top{top}+bottom{bottom}"


def _resolve_archived_source(source: str) -> str:
    path = HERE / source
    if path.exists():
        return str(path)
    relative = Path(source)
    if relative.parts[0] != "results":
        raise FileNotFoundError(source)
    archived = HERE / "results" / "legacy_scheduler_algorithm" / Path(
        *relative.parts[1:]
    )
    if not archived.exists():
        raise FileNotFoundError(source)
    return str(archived)


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    base = json.loads(BASE_AUDIT.read_text(encoding="utf-8"))
    top5 = json.loads(TOP5_AUDIT.read_text(encoding="utf-8"))
    if not base.get("complete") or not top5.get("complete"):
        raise AssertionError("directed window evidence is incomplete")

    covered = {
        (str(row["name"]), str(row["window"])): bool(
            row["window_reaches_target"]
        )
        for row in base["results"]
    }
    source_cache: dict[Path, dict[str, dict]] = {}
    case_names = []
    for original in top5["results"]:
        audit_row = dict(original)
        case_names.append(str(audit_row["name"]))
        audit_row["direct_witness_sources"] = [
            {
                **source,
                "source": _resolve_archived_source(str(source["source"])),
            }
            for source in audit_row["direct_witness_sources"]
        ]
        source_row, actions, _source = witness._materialize_history(
            audit_row,
            (5, 1),
            source_cache,
        )
        distribution = {
            eid: int(ntok)
            for eid, ntok in enumerate(source_row["counts"])
            if int(ntok) > 0
        }
        state = reference.FourStageScheduler(distribution)._initial_state()
        visible_history = {window: True for window in WINDOWS}
        for action in actions:
            for window in WINDOWS:
                visible = reference.candidate_window_visible_eids(
                    state.c2,
                    state.c3,
                    state.remaining,
                    window,
                )
                visible_history[window] &= (
                    reference.action_within_candidate_window(action, visible)
                )
            state = reference.apply_action(state, action)
        if state.remaining:
            raise AssertionError(f"{audit_row['name']}: witness left expert work")
        for window, is_visible in visible_history.items():
            key = (str(audit_row["name"]), _window_name(window))
            covered[key] = covered.get(key, False) or bool(is_visible)

    if len(case_names) != 65 or len(set(case_names)) != 65:
        raise AssertionError("expected 65 unique directed cases")
    summary = {}
    for window in WINDOWS:
        name = _window_name(window)
        count = sum(covered.get((case, name), False) for case in case_names)
        summary[name] = {
            "window": list(window),
            "visible_descriptors": sum(window),
            "cases": len(case_names),
            "optimal_path_covered": count,
            "optimal_path_coverage_pct": 100.0 * count / len(case_names),
        }
    if summary["top5+bottom1"]["optimal_path_covered"] != 65:
        raise AssertionError("top5+bottom1 must retain all directed optimal paths")

    payload = {
        "schema": "moe_characteristic_window_coverage_v1",
        "dataset": {
            "name": "MoE-characteristic directed set",
            "cases": 65,
            "description": (
                "Synthetic Top-2 distributions representing a few hot experts, "
                "many active experts, and a long cold tail. OLMoE routing analysis "
                "motivated these properties; the cases are not measured OLMoE inputs."
            ),
        },
        "interpretation": (
            "Coverage requires at least one saved replay-valid optimal history "
            "that is visible under the stated window."
        ),
        "inputs": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in (BASE_AUDIT, TOP5_AUDIT)
        ],
        "source": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "summary": summary,
    }
    _atomic_write(OUTPUT, payload)
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
