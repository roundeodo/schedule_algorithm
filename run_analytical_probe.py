#!/usr/bin/env python3
"""Run a small analytical-scheduler timing probe on stratified-v6 inputs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from analytical_scheduler import analytical_schedule


DEFAULT_FILES = (
    ("E8", ROOT / "scheduler_eval_inputs_E8_stratified_v6.json"),
    ("E32", ROOT / "scheduler_eval_inputs_E32_stratified_v6.json"),
    ("E64", ROOT / "scheduler_eval_inputs_E64_stratified_v6.json"),
)


def split_counts(total: int, n_parts: int) -> list[int]:
    base = total // n_parts
    rem = total % n_parts
    return [base + (1 if i < rem else 0) for i in range(n_parts)]


def stratified_indices(n_cases: int, n_pick: int) -> list[int]:
    if n_pick <= 0:
        return []
    if n_pick == 1:
        return [0]
    return sorted(set(round(i * (n_cases - 1) / (n_pick - 1)) for i in range(n_pick)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=50)
    args = parser.parse_args()

    counts = split_counts(args.total, len(DEFAULT_FILES))
    all_times: list[float] = []
    wall0 = time.time()

    for (name, path), n_pick in zip(DEFAULT_FILES, counts):
        payload = json.loads(path.read_text())
        cases = payload["cases"]
        idxs = stratified_indices(len(cases), n_pick)
        print(f"\n{name}: running {len(idxs)} stratified cases", flush=True)

        times: list[float] = []
        for j, idx in enumerate(idxs, start=1):
            case = cases[idx]
            dist = {int(k): int(v) for k, v in case["dist"].items()}
            t0 = time.time()
            cc = analytical_schedule(dist, int(case["c2"]), int(case["c3"]))
            dt = time.time() - t0
            times.append(dt)
            all_times.append(dt)
            print(
                f"{name} {j:02d}/{len(idxs)} "
                f"case={idx} active={case['active_n']} m={case['m_total']} "
                f"profile={case['profile']} time_s={dt:.3f} cc={cc}",
                flush=True,
            )

        if times:
            print(
                f"{name} summary: mean={statistics.mean(times):.3f}s "
                f"median={statistics.median(times):.3f}s max={max(times):.3f}s",
                flush=True,
            )

    wall = time.time() - wall0
    if all_times:
        print("\nTOTAL")
        print(
            f"cases={len(all_times)} wall_s={wall:.3f} "
            f"mean_s={statistics.mean(all_times):.3f} "
            f"median_s={statistics.median(all_times):.3f} max_s={max(all_times):.3f}"
        )
        print(f"est_30000_h_by_mean={statistics.mean(all_times) * 30000 / 3600:.2f}")
        print(f"est_30000_h_by_wall_per_case={wall / len(all_times) * 30000 / 3600:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
