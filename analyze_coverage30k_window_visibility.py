#!/usr/bin/env python3
"""Audit which bounded Top+Bottom windows can replay saved 30K histories.

This is a visibility audit, not an online-policy comparison.  For every saved,
validated reference history, each action is replayed from the exact state that
precedes it.  The history is covered by a window only if every explicitly named
expert is either visible through that window or already resident in scheduler
state, matching ``four_stage_scheduler.candidate_window_visible_eids``.

Failure means only that the saved history is not representable.  It does not
prove that no alternative history with the same makespan exists in that window.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import four_stage_scheduler as reference
from run_four_stage_reference import deserialize_action


HERE = Path(__file__).resolve().parent
DEFAULT_INPUTS = tuple(
    HERE
    / "results"
    / "legacy_scheduler_algorithm"
    / "final_reference"
    / f"scheduler_reference_E{experts}.json"
    for experts in (8, 32, 64)
)
DEFAULT_OUTPUT = (
    HERE / "results" / "policy_search" / "coverage30k_window_visibility.json"
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


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _initial_state(row: dict) -> reference.BeamState:
    distribution = {
        int(eid): int(ntok)
        for eid, ntok in row["dist"].items()
        if int(ntok) > 0
    }
    return reference.FourStageScheduler(
        distribution,
        initial_cache_c2=int(row["initial_cache_c2"]),
        initial_cache_c3=int(row["initial_cache_c3"]),
    )._initial_state()


def _audit_row(row: dict) -> dict[str, bool]:
    state = _initial_state(row)
    covered = {_window_name(window): True for window in WINDOWS}
    history = []
    for serialized in row["actions"]:
        action = deserialize_action(serialized)
        history.append(action)
        for window in WINDOWS:
            name = _window_name(window)
            if not covered[name]:
                continue
            visible = reference.candidate_window_visible_eids(
                state.c2,
                state.c3,
                state.remaining,
                window,
            )
            if not reference.action_within_candidate_window(action, visible):
                covered[name] = False
        state = reference.apply_action(state, action)

    if state.remaining:
        raise AssertionError(f"case {row['case_id']}: history left expert work")
    if int(state.g_score) != int(row["makespan_cc"]):
        raise AssertionError(
            f"case {row['case_id']}: replay makespan {state.g_score} "
            f"!= stored {row['makespan_cc']}"
        )
    return covered


def _summarize(records: list[dict]) -> dict:
    result = {}
    for window in WINDOWS:
        name = _window_name(window)
        all_rows = records
        optimal_rows = [row for row in records if row["proven_optimal"]]
        covered_all = sum(row["covered"][name] for row in all_rows)
        covered_optimal = sum(row["covered"][name] for row in optimal_rows)
        result[name] = {
            "window": list(window),
            "visible_descriptors": sum(window),
            "validated_histories": len(all_rows),
            "validated_histories_covered": covered_all,
            "validated_history_coverage_pct": 100.0 * covered_all / len(all_rows),
            "proven_optimal_histories": len(optimal_rows),
            "proven_optimal_histories_covered": covered_optimal,
            "proven_optimal_history_coverage_pct": (
                100.0 * covered_optimal / len(optimal_rows)
            ),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=2000)
    args = parser.parse_args()

    inputs = tuple(args.input) if args.input else DEFAULT_INPUTS
    records = []
    started = time.perf_counter()
    stop = False
    for path in inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["results"].values():
            if not row.get("analysis_eligible", False):
                continue
            if row.get("status") != "ok" or not row.get("history_validated", False):
                raise AssertionError(f"{path.name}:{row['case_id']}: invalid source row")
            records.append(
                {
                    "key": f"E{int(row['e_total'])}:{int(row['case_id'])}",
                    "e_total": int(row["e_total"]),
                    "proven_optimal": bool(row["proven_optimal"]),
                    "covered": _audit_row(row),
                }
            )
            if args.progress_every and len(records) % args.progress_every == 0:
                print(
                    f"audited={len(records)} "
                    f"elapsed_s={time.perf_counter() - started:.1f}",
                    flush=True,
                )
            if args.limit is not None and len(records) >= args.limit:
                stop = True
                break
        if stop:
            break

    if args.limit is None and len(records) != 29_928:
        raise AssertionError(f"expected 29,928 eligible cases, got {len(records)}")

    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        buckets["overall"].append(record)
        buckets[f"E{record['e_total']}"] .append(record)
    payload = {
        "schema": "coverage30k_window_visibility_v1",
        "interpretation": {
            "success": (
                "Every action in the saved replay-valid history names only a "
                "window-visible or already-resident expert."
            ),
            "failure": (
                "The saved history is not representable; this does not prove "
                "that no equal-makespan history exists under the window."
            ),
            "scope": "Observation visibility only; no online scorer is evaluated.",
        },
        "configuration": {
            "windows": [list(window) for window in WINDOWS],
            "inputs": [
                {"path": str(path.resolve()), "sha256": _sha256(path)}
                for path in inputs
            ],
            "source": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "limit": args.limit,
        },
        "case_count": len(records),
        "proven_optimal_case_count": sum(
            record["proven_optimal"] for record in records
        ),
        "runtime_s": time.perf_counter() - started,
        "summary": {
            bucket: _summarize(bucket_records)
            for bucket, bucket_records in sorted(buckets.items())
        },
    }
    _atomic_write(args.output.resolve(), payload)
    print(json.dumps(payload["summary"]["overall"], indent=2))
    print(f"wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
