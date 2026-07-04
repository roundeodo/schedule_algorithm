#!/usr/bin/env python3
"""Annotate stratified-v6 scheduler inputs with beam64 results.

The script updates each case in-place with:
  - beam64_makespan_cc
  - beam64_num_actions
  - beam64_runtime_s
  - beam64_over_compute_only_ideal
  - beam64_status

It is resumable by default: cases with beam64_status == "ok" are skipped unless
--force is passed.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from four_stage_scheduler import FourStageScheduler


ROOT = Path(__file__).resolve().parent
BEAM_MODE = "semantic_pair_split_family_semantic_dedup"
DEFAULT_FILES = (
    ROOT / "scheduler_eval_inputs_E8_stratified_v6.json",
    ROOT / "scheduler_eval_inputs_E32_stratified_v6.json",
    ROOT / "scheduler_eval_inputs_E64_stratified_v6.json",
)


def load_payload(path: Path):
    with path.open() as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "cases" not in payload:
        raise ValueError(f"{path} is not a stratified-v6 payload")
    return payload


def atomic_write_json(path: Path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def run_case(case, beam_width: int):
    dist = {int(k): int(v) for k, v in case["dist"].items()}
    c2 = int(case.get("c2", -1))
    c3 = int(case.get("c3", -1))
    t0 = time.time()
    makespan, history = FourStageScheduler(
        dist,
        beam_width=beam_width,
        initial_cache_c2=c2,
        initial_cache_c3=c3,
    ).run()
    runtime = time.time() - t0
    ideal = case.get("compute_only_ideal_cc")
    ratio = (makespan / ideal) if ideal else None
    case[f"beam{beam_width}_makespan_cc"] = makespan
    case[f"beam{beam_width}_num_actions"] = len(history)
    case[f"beam{beam_width}_runtime_s"] = round(runtime, 6)
    case[f"beam{beam_width}_over_compute_only_ideal"] = ratio
    case[f"beam{beam_width}_mode"] = BEAM_MODE
    case[f"beam{beam_width}_status"] = "ok"


def annotate_file(
    path: Path,
    beam_width: int,
    limit: int,
    force: bool,
    save_every: int,
    max_active_n: int,
):
    payload = load_payload(path)
    cases = payload["cases"]
    done = 0
    failed = 0
    status_key = f"beam{beam_width}_status"
    start_all = time.time()

    for idx, case in enumerate(cases):
        if limit is not None and done >= limit:
            break
        if max_active_n is not None and int(case.get("active_n", 0)) > max_active_n:
            continue
        if not force and case.get(status_key) == "ok":
            continue
        try:
            run_case(case, beam_width)
            done += 1
        except Exception as exc:
            case[status_key] = "error"
            case[f"beam{beam_width}_error"] = repr(exc)
            failed += 1

        if save_every > 0 and (done + failed) % save_every == 0:
            atomic_write_json(path, payload)
            print(
                f"{path.name}: saved progress done={done} failed={failed} idx={idx}",
                flush=True,
            )

    payload.setdefault("meta", {})[f"beam{beam_width}_annotated_cases"] = sum(
        1 for c in cases if c.get(status_key) == "ok"
    )
    payload["meta"][f"beam{beam_width}_mode"] = BEAM_MODE
    payload["meta"][f"beam{beam_width}_last_update_s"] = round(time.time() - start_all, 6)
    atomic_write_json(path, payload)
    print(
        f"{path.name}: finished this run done={done} failed={failed} "
        f"total_ok={payload['meta'][f'beam{beam_width}_annotated_cases']}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--max-active-n", type=int, default=None)
    parser.add_argument("--files", nargs="*", type=Path, default=list(DEFAULT_FILES))
    args = parser.parse_args()

    for path in args.files:
        annotate_file(
            path,
            args.beam_width,
            args.limit,
            args.force,
            args.save_every,
            args.max_active_n,
        )


if __name__ == "__main__":
    main()
