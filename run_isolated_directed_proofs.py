#!/usr/bin/env python3
"""Run one directed proof case per process and merge the replay-valid rows.

The explicit four-stage generator can retain large Python allocation arenas.
Process isolation keeps a multi-case pass bounded and leaves resumable case
fragments if the run is interrupted.  Fragments are removed after a complete
merge unless ``--keep-fragments`` is requested.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
PROVER = HERE / "prove_top4_bottom2_directed.py"


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _fraction(value: str) -> Fraction:
    return Fraction(value)


def _summary(rows: list[dict]) -> dict:
    return {
        "cases": len(rows),
        "proven_optimal": sum(bool(row["proven_optimal"]) for row in rows),
        "unproven": sum(not row["proven_optimal"] for row in rows),
        "termination_counts": dict(
            sorted(Counter(row["termination"] for row in rows).items())
        ),
        "best_known_gap_ticks": {
            row["name"]: str(
                _fraction(row["best_reference_ticks"])
                - _fraction(row["certified_lower_bound_ticks"])
            )
            for row in rows
            if not row["proven_optimal"]
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-input", type=Path, required=True)
    parser.add_argument("--prior-proof", type=Path)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="number of isolated proof subprocesses to run concurrently",
    )
    parser.add_argument(
        "--only-unproven",
        action="store_true",
        help="with --prior-proof, run only cases not yet proved optimal",
    )
    parser.add_argument("--time-limit-s", type=float, default=0.0)
    parser.add_argument("--max-expansions", type=int, default=200_000)
    parser.add_argument("--target-decision", action="store_true")
    parser.add_argument(
        "--target-rank-mode",
        choices=(
            "completion",
            "depth",
            "balance",
            "lpt",
            "cache",
            "hot_tail",
            "dma",
        ),
        default="depth",
    )
    parser.add_argument("--seed-beam-widths", default="8")
    parser.add_argument("--seed-beam-modes", default="completion,cache,lpt,f_g")
    parser.add_argument("--seed-window", default="")
    parser.add_argument(
        "--min-start-gap-ticks",
        type=int,
        default=-1,
        help="with --prior-proof, retain cases whose starting gap is at least this value",
    )
    parser.add_argument(
        "--max-start-gap-ticks",
        type=int,
        default=-1,
        help="with --prior-proof, retain cases whose starting gap is at most this value",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--keep-fragments", action="store_true")
    args = parser.parse_args()
    if (
        args.jobs <= 0
        or args.min_start_gap_ticks < -1
        or args.max_start_gap_ticks < -1
        or (
            args.min_start_gap_ticks >= 0
            and args.max_start_gap_ticks >= 0
            and args.min_start_gap_ticks > args.max_start_gap_ticks
        )
    ):
        raise SystemExit("invalid jobs or starting-gap range")
    if args.target_decision and args.time_limit_s <= 0:
        raise SystemExit("--target-decision requires --time-limit-s > 0")

    source = json.loads(args.case_input.read_text(encoding="utf-8"))
    names = [str(row["name"]) for row in source["cases"]]
    if args.case:
        requested = set(args.case)
        missing = requested - set(names)
        if missing:
            raise SystemExit(f"unknown cases: {sorted(missing)}")
        names = [name for name in names if name in requested]
    prior_by_name = {}
    if args.prior_proof is not None:
        prior_payload = json.loads(args.prior_proof.read_text(encoding="utf-8"))
        prior_by_name = {row["name"]: row for row in prior_payload["cases"]}
    if args.only_unproven:
        if args.prior_proof is None:
            raise SystemExit("--only-unproven requires --prior-proof")
        names = [
            name
            for name in names
            if name in prior_by_name and not prior_by_name[name]["proven_optimal"]
        ]
    if args.min_start_gap_ticks >= 0 or args.max_start_gap_ticks >= 0:
        if args.prior_proof is None:
            raise SystemExit("starting-gap selection requires --prior-proof")
        selected = []
        for name in names:
            if name not in prior_by_name:
                continue
            row = prior_by_name[name]
            gap = _fraction(row["best_reference_ticks"]) - _fraction(
                row["certified_lower_bound_ticks"]
            )
            if args.min_start_gap_ticks >= 0 and gap < args.min_start_gap_ticks:
                continue
            if args.max_start_gap_ticks >= 0 and gap > args.max_start_gap_ticks:
                continue
            selected.append(name)
        names = selected

    config = {
        "case_input": str(args.case_input.resolve()),
        "prior_proof": str(args.prior_proof.resolve()) if args.prior_proof else None,
        "time_limit_s": args.time_limit_s,
        "max_expansions": args.max_expansions,
        "target_decision": args.target_decision,
        "target_rank_mode": args.target_rank_mode,
        "seed_beam_widths": args.seed_beam_widths,
        "seed_beam_modes": args.seed_beam_modes,
        "seed_window": args.seed_window,
        "only_unproven": args.only_unproven,
        "jobs": args.jobs,
        "min_start_gap_ticks": args.min_start_gap_ticks,
        "max_start_gap_ticks": args.max_start_gap_ticks,
    }
    config_id = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    work_dir = args.work_dir or (
        args.output.parent / ".proof_fragments" / config_id
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    rows_by_index: dict[int, dict] = {}
    pending = []
    for index, name in enumerate(names, 1):
        fragment = work_dir / f"{index:03d}.json"
        row = None
        if fragment.exists():
            payload = json.loads(fragment.read_text(encoding="utf-8"))
            if payload.get("complete") and len(payload.get("cases", ())) == 1:
                candidate = payload["cases"][0]
                if candidate.get("name") == name and candidate.get(
                    "history_replay_valid"
                ):
                    row = candidate
                    print(f"[{index}/{len(names)}] resume {name}", flush=True)
        if row is not None:
            rows_by_index[index] = row
        else:
            pending.append((index, name, fragment))

    def run_one(index: int, name: str, fragment: Path) -> tuple[int, dict]:
        print(f"[{index}/{len(names)}] run {name}", flush=True)
        command = [
            sys.executable,
            str(PROVER),
            "--case-input",
            str(args.case_input.resolve()),
            "--case",
            name,
            "--time-limit-s",
            str(args.time_limit_s),
            "--max-expansions",
            str(args.max_expansions),
            "--seed-beam-widths",
            args.seed_beam_widths,
            "--seed-beam-modes",
            args.seed_beam_modes,
            "--output",
            str(fragment.resolve()),
        ]
        if args.prior_proof:
            command.extend(["--prior-proof", str(args.prior_proof.resolve())])
        if args.target_decision:
            command.append("--target-decision")
            command.extend(["--target-rank-mode", args.target_rank_mode])
        if args.seed_window:
            command.extend(["--seed-window", args.seed_window])
        completed = subprocess.run(command, cwd=HERE, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"case {name} failed with exit code {completed.returncode}; "
                f"resume with the same command"
            )
        payload = json.loads(fragment.read_text(encoding="utf-8"))
        row = payload["cases"][0]
        if not row.get("history_replay_valid"):
            raise RuntimeError(f"case {name}: missing replay validation")
        return index, row

    def write_partial() -> None:
        ordered = [rows_by_index[index] for index in sorted(rows_by_index)]
        partial = {
            "schema": "isolated_directed_proof_merge_v1",
            "complete": False,
            "run_config": config,
            "summary": _summary(ordered),
            "cases": ordered,
        }
        _atomic_write(args.output, partial)

    if rows_by_index:
        write_partial()
    if pending:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(run_one, index, name, fragment): (index, name)
                for index, name, fragment in pending
            }
            for future in as_completed(futures):
                index, name = futures[future]
                try:
                    completed_index, row = future.result()
                except RuntimeError as exc:
                    for other in futures:
                        other.cancel()
                    raise SystemExit(str(exc)) from exc
                rows_by_index[completed_index] = row
                print(
                    f"[{completed_index}/{len(names)}] complete {name} "
                    f"best={row['best_reference_ticks']} "
                    f"LB={row['certified_lower_bound_ticks']} "
                    f"proven={row['proven_optimal']}",
                    flush=True,
                )
                write_partial()

    rows = [rows_by_index[index] for index in range(1, len(names) + 1)]

    result = {
        "schema": "isolated_directed_proof_merge_v1",
        "complete": True,
        "run_config": config,
        "summary": _summary(rows),
        "cases": rows,
    }
    _atomic_write(args.output, result)
    if not args.keep_fragments:
        for fragment in work_dir.glob("*.json"):
            fragment.unlink()
        try:
            work_dir.rmdir()
            work_dir.parent.rmdir()
        except OSError:
            pass
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
