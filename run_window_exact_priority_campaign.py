#!/usr/bin/env python3
"""Round-robin exact window-sufficiency campaigns over unresolved cases.

This is an offline proof orchestrator.  It never changes the reference action
graph: every time slice is delegated to ``run_target_root_branches.py`` and
therefore uses the same checkpointed exact target-feasibility search.  The
only heuristic choice is the order in which semantic root groups receive time.
A timeout remains unresolved; only a replay-valid feasible branch or complete
root-group exhaustion is evidence.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import four_stage_scheduler as reference
import run_target_root_branches as root_search


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_target_root_branches.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _window_name(window: tuple[int, int]) -> str:
    top, bottom = window
    return f"top{top}" if bottom == 0 else f"top{top}+bottom{bottom}"


def _case_rows(
    prior: dict, audit: dict, window: tuple[int, int], requested: set[str]
) -> list[dict]:
    prior_by_name = {row["name"]: row for row in prior["cases"]}
    window_name = _window_name(window)
    audit_rows = [
        row
        for row in audit["results"]
        if row["window"] == window_name
        and row["window_status"] == "unresolved"
        and (not requested or row["name"] in requested)
    ]
    missing = requested - {row["name"] for row in audit_rows}
    if missing:
        raise SystemExit(
            f"requested cases are absent or not unresolved for {window_name}: "
            f"{sorted(missing)}"
        )
    rows = []
    for audit_row in audit_rows:
        name = audit_row["name"]
        if name not in prior_by_name:
            raise SystemExit(f"audit case {name!r} is absent from prior proof")
        prior_row = prior_by_name[name]
        if not prior_row.get("proven_optimal"):
            raise SystemExit(f"case {name!r} lacks a proved-optimal target")
        if (
            Fraction(prior_row["certified_lower_bound_ticks"])
            != Fraction(prior_row["best_reference_ticks"])
        ):
            raise SystemExit(f"case {name!r} does not satisfy LB=UB")
        if Fraction(audit_row["target_ticks"]) != Fraction(
            prior_row["best_reference_ticks"]
        ):
            raise SystemExit(f"audit target mismatch for {name!r}")
        rows.append(prior_row)
    return rows


def _root_groups(
    row: dict, window: tuple[int, int]
) -> list[tuple[int, tuple, int]]:
    token_dist = {
        eid: int(ntok)
        for eid, ntok in enumerate(row["counts"])
        if int(ntok) > 0
    }
    target = int(
        Fraction(row["best_reference_ticks"])
        * reference.SCHEDULE_TIME_QUANTUM_CC
    )
    reference.clear_scheduler_caches()
    children = root_search._root_children(
        reference.FourStageScheduler(token_dist), target, window
    )
    sizes: dict[tuple, int] = {}
    for child in children:
        key = root_search._semantic_group_key(child)
        sizes[key] = sizes.get(key, 0) + 1
    return [
        (index, key, size)
        for index, (key, size) in enumerate(sorted(sizes.items()), 1)
    ]


def _group_priority(key: tuple) -> tuple:
    family = key[0]
    values = tuple(int(value) for value in key[1:])
    if family == "PAIR" and len(values) == 2 and values[0] == values[1]:
        # The min2 closure showed that equal medium pairs are frequently an
        # optimal way to preserve hot experts as later anchors.
        return (0, values[0])
    if family == "SINGLE" and len(values) == 1:
        # Prefer medium work over the smallest cold-tail singleton.
        return (1, -values[0])
    if family == "SPLIT" and len(values) == 2:
        return (2, abs(values[1] - values[0]), -sum(values), values)
    if family == "PAIR" and len(values) == 2:
        return (3, abs(values[1] - values[0]), values)
    return (4, key)


def _load_fragments(work_dir: Path) -> dict[int, dict]:
    rows = {}
    manifest_path = work_dir / "manifest.json"
    manifest_id = None
    if manifest_path.exists():
        manifest_id = json.loads(manifest_path.read_text(encoding="utf-8"))[
            "manifest_id"
        ]
    for path in sorted(work_dir.glob("[0-9][0-9][0-9][0-9].json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if manifest_id is not None and row.get("manifest_id") != manifest_id:
            raise SystemExit(f"stale root fragment {path}")
        rows[int(row["group_index"])] = row
    return rows


def _case_status(groups: list[tuple[int, tuple, int]], work_dir: Path) -> dict:
    fragments = _load_fragments(work_dir)
    expected_indices = {index for index, _key, _size in groups}
    unexpected_indices = set(fragments) - expected_indices
    if unexpected_indices:
        raise SystemExit(
            f"unexpected root-group fragments in {work_dir}: "
            f"{sorted(unexpected_indices)}"
        )
    feasible = [
        (index, row)
        for index, row in sorted(fragments.items())
        if row.get("feasible")
    ]
    exhaustive_indices = {
        index for index, row in fragments.items() if row.get("exhaustive")
    }
    feasible_fragment = None
    feasible_fragment_sha256 = None
    if feasible:
        feasible_fragment = f"{feasible[0][0]:04d}.json"
        feasible_fragment_sha256 = _sha256(work_dir / feasible_fragment)
    return {
        "semantic_root_groups": len(groups),
        "attempted_groups": len(fragments),
        "feasible": bool(feasible),
        "feasible_group": feasible[0][0] if feasible else None,
        "feasible_fragment": feasible_fragment,
        "feasible_fragment_sha256": feasible_fragment_sha256,
        "all_groups_exhaustive": (
            bool(expected_indices) and exhaustive_indices == expected_indices
        ),
        "exhaustive_groups": len(exhaustive_indices),
        "exhaustive_group_indices": sorted(exhaustive_indices),
        "unresolved_groups": sum(
            not row.get("feasible") and not row.get("exhaustive")
            for row in fragments.values()
        ),
        "total_expansions": sum(
            int(row.get("expansions", 0)) for row in fragments.values()
        ),
        "total_generated": sum(
            int(row.get("generated", 0)) for row in fragments.values()
        ),
    }


def _choose_group(
    groups: list[tuple[int, tuple, int]], work_dir: Path
) -> tuple[int, tuple, int] | None:
    fragments = _load_fragments(work_dir)
    ordered = sorted(groups, key=lambda item: (_group_priority(item[1]), item[0]))
    for group in ordered:
        if group[0] not in fragments:
            return group
    unresolved = [
        group
        for group in ordered
        if not fragments[group[0]].get("feasible")
        and not fragments[group[0]].get("exhaustive")
    ]
    if not unresolved:
        return None
    return min(
        unresolved,
        key=lambda group: (
            float(fragments[group[0]].get("runtime_s", 0.0)),
            _group_priority(group[1]),
            group[0],
        ),
    )


def _report(
    manifest: dict,
    rows: list[dict],
    groups_by_case: dict[str, list[tuple[int, tuple, int]]],
    work_root: Path,
) -> dict:
    cases = []
    for row in rows:
        name = row["name"]
        status = _case_status(groups_by_case[name], work_root / name)
        status.update(
            {
                "name": name,
                "target_ticks": str(row["best_reference_ticks"]),
                "work_dir": str((work_root / name).resolve()),
            }
        )
        cases.append(status)
    return {
        "schema": "window_exact_priority_campaign_v1",
        "complete": all(
            row["feasible"] or row["all_groups_exhaustive"] for row in cases
        ),
        "manifest": manifest,
        "summary": {
            "cases": len(cases),
            "feasible_cases": sum(row["feasible"] for row in cases),
            "proved_insufficient_cases": sum(
                row["all_groups_exhaustive"] for row in cases
            ),
            "unresolved_cases": sum(
                not row["feasible"] and not row["all_groups_exhaustive"]
                for row in cases
            ),
            "total_expansions": sum(row["total_expansions"] for row in cases),
            "total_generated": sum(row["total_generated"] for row in cases),
        },
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-proof", type=Path, required=True)
    parser.add_argument("--window-audit", type=Path, required=True)
    parser.add_argument("--candidate-window", type=int, nargs=2, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--time-slice-s", type=float, default=90.0)
    parser.add_argument("--max-expansions", type=int, default=100_000)
    parser.add_argument(
        "--max-invocations",
        type=int,
        default=0,
        help="stop cleanly after this many root time slices; zero runs to closure",
    )
    parser.add_argument("--rank-mode", default="hot_tail")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    window = reference.normalize_candidate_window(tuple(args.candidate_window))
    if (
        args.time_slice_s <= 0
        or args.max_expansions <= 0
        or args.max_invocations < 0
    ):
        raise SystemExit("search limits must be positive")

    prior = json.loads(args.prior_proof.read_text(encoding="utf-8"))
    audit = json.loads(args.window_audit.read_text(encoding="utf-8"))
    rows = _case_rows(prior, audit, window, set(args.case))
    if not rows:
        raise SystemExit("no unresolved cases selected")
    groups_by_case = {row["name"]: _root_groups(row, window) for row in rows}
    manifest = {
        "candidate_window": list(window),
        "window_name": _window_name(window),
        "cases": [row["name"] for row in rows],
        "targets": {
            row["name"]: str(row["best_reference_ticks"]) for row in rows
        },
        "group_priority": (
            "equal_pair_then_single_then_balanced_split_then_other_pair"
        ),
        "time_slice_s": args.time_slice_s,
        "max_expansions": args.max_expansions,
        "max_invocations": args.max_invocations,
        "rank_mode": args.rank_mode,
        "prior_proof": str(args.prior_proof.resolve()),
        "prior_proof_sha256": _sha256(args.prior_proof),
        "window_audit": str(args.window_audit.resolve()),
        "window_audit_sha256": _sha256(args.window_audit),
        "reference_sha256": _sha256(HERE / "four_stage_scheduler.py"),
        "runner_sha256": _sha256(RUNNER),
        "orchestrator_sha256": _sha256(Path(__file__)),
    }
    manifest_id = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    manifest["manifest_id"] = manifest_id
    args.work_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.work_root / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old != manifest:
            raise SystemExit(f"manifest mismatch in {args.work_root}")
    else:
        _atomic_write(manifest_path, manifest)

    if args.dry_run:
        payload = {
            "schema": "window_exact_priority_dry_run_v1",
            "manifest": manifest,
            "cases": [
                {
                    "name": row["name"],
                    "target_ticks": str(row["best_reference_ticks"]),
                    "priority": [
                        {"group_index": index, "key": list(key), "children": size}
                        for index, key, size in sorted(
                            groups_by_case[row["name"]],
                            key=lambda item: (_group_priority(item[1]), item[0]),
                        )
                    ],
                }
                for row in rows
            ],
        }
        _atomic_write(args.output, payload)
        print(json.dumps({"cases": len(rows), "dry_run": True}, indent=2))
        return 0

    started = time.perf_counter()
    invocations = 0
    while True:
        progressed = False
        for row in rows:
            name = row["name"]
            work_dir = args.work_root / name
            status = _case_status(groups_by_case[name], work_dir)
            if status["feasible"] or status["all_groups_exhaustive"]:
                continue
            selected = _choose_group(groups_by_case[name], work_dir)
            if selected is None:
                continue
            group_index, group_key, _group_size = selected
            command = [
                sys.executable,
                str(RUNNER),
                "--prior-proof",
                str(args.prior_proof.resolve()),
                "--case",
                name,
                "--target-ticks",
                str(row["best_reference_ticks"]),
                "--candidate-window",
                str(window[0]),
                str(window[1]),
                "--time-limit-s",
                str(args.time_slice_s),
                "--max-expansions",
                str(args.max_expansions),
                "--rank-mode",
                args.rank_mode,
                "--group-index",
                str(group_index),
                "--work-dir",
                str(work_dir.resolve()),
                "--output",
                str((work_dir / "last_result.json").resolve()),
            ]
            print(
                f"case={name} group={group_index} key={group_key}", flush=True
            )
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                raise SystemExit(
                    f"root runner failed for {name} group {group_index}: "
                    f"exit={result.returncode}"
                )
            progressed = True
            invocations += 1
            payload = _report(
                manifest, rows, groups_by_case, args.work_root
            )
            payload["elapsed_s"] = time.perf_counter() - started
            payload["invocations"] = invocations
            _atomic_write(args.output, payload)
            if args.max_invocations and invocations >= args.max_invocations:
                print(json.dumps(payload["summary"], indent=2), flush=True)
                return 0
        payload = _report(manifest, rows, groups_by_case, args.work_root)
        payload["elapsed_s"] = time.perf_counter() - started
        payload["invocations"] = invocations
        _atomic_write(args.output, payload)
        print(json.dumps(payload["summary"], indent=2), flush=True)
        if payload["complete"]:
            return 0
        if not progressed:
            raise SystemExit("no runnable unresolved root group remains")


if __name__ == "__main__":
    raise SystemExit(main())
