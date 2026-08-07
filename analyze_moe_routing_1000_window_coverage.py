#!/usr/bin/env python3
"""Generate and audit 1,000 MoE-routing-characteristic Top-2 cases.

The independent directed set varies the number and magnitude of hot experts,
the active-expert count, the one/two-token tail, expert IDs, and initial cache
residency.  Its ranges are motivated by published MoE routing observations; it
is not a trace set measured from one named model.

Each case is scheduled by the deterministic complete greedy rollout used to
seed the offline search.  Coverage means that every action in that legal
reference schedule is expressible through the tested observation window.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import four_stage_scheduler as reference


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "policy_search" / "moe_routing_1000_window_coverage.json"
BOUNDARY_AUDIT = (
    HERE / "results" / "policy_search" / "moe_characteristic_window_coverage.json"
)
SEED = 20260803
N_CASES = 1_000
E_TOTAL = 64
M_TOTAL = 70
ASSIGNMENT_TOTAL = 2 * M_TOTAL
ACTIVE_COUNTS = (29, 33, 38, 43)
HOT_COUNTS = (2, 3, 4)
COLD_FRACTIONS = (0.42, 0.44, 0.46, 0.49)
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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _add_middle_tokens(
    middle: list[int], target_sum: int, ceiling: int, rng: random.Random
) -> None:
    while sum(middle) < target_sum:
        choices = [index for index, value in enumerate(middle) if value < ceiling]
        if not choices:
            raise ValueError("middle experts cannot absorb the target assignment count")
        middle[rng.choice(choices)] += 1


def _make_case(case_id: int) -> dict:
    rng = random.Random(SEED + 104_729 * case_id)
    active_n = ACTIVE_COUNTS[case_id % len(ACTIVE_COUNTS)]
    hot_n = HOT_COUNTS[(case_id // len(ACTIVE_COUNTS)) % len(HOT_COUNTS)]
    cold_fraction = COLD_FRACTIONS[
        (case_id // (len(ACTIVE_COUNTS) * len(HOT_COUNTS)))
        % len(COLD_FRACTIONS)
    ]
    cold_n = round(active_n * cold_fraction)
    middle_n = active_n - hot_n - cold_n
    if middle_n <= 0:
        raise AssertionError("directed construction requires middle experts")

    hot_ranges = ((16, 30), (15, 18), (12, 15), (10, 14))
    hot = [rng.randint(*hot_ranges[index]) for index in range(hot_n)]
    hot.sort(reverse=True)

    # Alternate tails with both one- and two-token experts and tails whose
    # minimum is two.  This preserves the two cold-tail regimes in the earlier
    # directed campaign without retaining its small fixed grid.
    if case_id & 1:
        cold = [2] * cold_n
    else:
        cold = [1 + rng.randrange(2) for _ in range(cold_n)]

    middle_target = ASSIGNMENT_TOTAL - sum(hot) - sum(cold)
    minimum_middle = 3 * middle_n
    if middle_target < minimum_middle:
        deficit = minimum_middle - middle_target
        for index in range(len(hot)):
            reduction = min(deficit, hot[index] - 10)
            hot[index] -= reduction
            deficit -= reduction
            if deficit == 0:
                break
        middle_target = ASSIGNMENT_TOTAL - sum(hot) - sum(cold)
    middle = [3] * middle_n
    _add_middle_tokens(middle, middle_target, min(hot) - 1, rng)

    counts = sorted(hot + middle + cold, reverse=True)
    if len(counts) != active_n or sum(counts) != ASSIGNMENT_TOTAL:
        raise AssertionError("invalid directed Top-2 distribution")
    if max(counts) > M_TOTAL or min(counts) < 1:
        raise AssertionError("expert count violates Top-2 marginal bounds")

    eids = rng.sample(range(E_TOTAL), active_n)
    dist = {eid: ntok for eid, ntok in zip(eids, counts)}
    ranked_eids = [eid for eid, _ in sorted(dist.items(), key=lambda x: (-x[1], x[0]))]
    cold_eids = [eid for eid, ntok in dist.items() if ntok <= 2]
    cache_regime = case_id % 5
    if cache_regime == 0:
        c2, c3 = -1, -1
    elif cache_regime == 1:
        c2, c3 = ranked_eids[0], ranked_eids[1]
    elif cache_regime == 2:
        c2, c3 = ranked_eids[0], rng.choice(cold_eids)
    elif cache_regime == 3:
        c2, c3 = rng.sample(cold_eids, 2)
    else:
        c2, c3 = rng.sample(ranked_eids, 2)

    return {
        "case_id": case_id,
        "e_total": E_TOTAL,
        "m_total": M_TOTAL,
        "assignment_total": ASSIGNMENT_TOTAL,
        "active_n": active_n,
        "hot_n": hot_n,
        "cold_n": cold_n,
        "cold_fraction": cold_n / active_n,
        "dist": dist,
        "initial_cache_c2": c2,
        "initial_cache_c3": c3,
    }


def _audit_case(case: dict) -> dict:
    scheduler = reference.FourStageScheduler(
        case["dist"],
        enable_prefetch=False,
        initial_cache_c2=case["initial_cache_c2"],
        initial_cache_c3=case["initial_cache_c3"],
    )
    state = scheduler._initial_state()
    final = scheduler._greedy_incumbent(state)
    covered = {_window_name(window): True for window in WINDOWS}
    for action in final.history:
        for window in WINDOWS:
            visible = reference.candidate_window_visible_eids(
                state.c2, state.c3, state.remaining, window
            )
            covered[_window_name(window)] &= reference.action_within_candidate_window(
                action, visible
            )
        state = reference.apply_action(state, action)
    if state.remaining or state.g_score != final.g_score:
        raise AssertionError(f"case {case['case_id']}: invalid reference replay")
    return {
        **case,
        "reference_makespan_cc": final.g_score,
        "reference_actions": len(final.history),
        "covered": covered,
    }


def main() -> int:
    cases = [_make_case(case_id) for case_id in range(N_CASES)]
    if len({tuple(sorted(case["dist"].items())) for case in cases}) != N_CASES:
        raise AssertionError("directed distributions must be unique")
    workers = min(8, os.cpu_count() or 1)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(_audit_case, cases, chunksize=4))

    summary = {}
    for window in WINDOWS:
        name = _window_name(window)
        count = sum(record["covered"][name] for record in records)
        summary[name] = {
            "window": list(window),
            "visible_descriptors": sum(window),
            "cases": len(records),
            "reference_histories_covered": count,
            "reference_history_coverage_pct": 100.0 * count / len(records),
        }
    boundary = json.loads(BOUNDARY_AUDIT.read_text())
    combined = {}
    for window in WINDOWS:
        name = _window_name(window)
        boundary_row = boundary["summary"][name]
        covered = (
            summary[name]["reference_histories_covered"]
            + boundary_row["optimal_path_covered"]
        )
        cases_count = summary[name]["cases"] + boundary_row["cases"]
        combined[name] = {
            "window": list(window),
            "visible_descriptors": sum(window),
            "cases": cases_count,
            "reference_histories_covered": covered,
            "reference_history_coverage_pct": 100.0 * covered / cases_count,
        }

    payload = {
        "schema": "moe_routing_1000_window_coverage_v1",
        "dataset": {
            "name": "MoE-routing-characteristic directed set",
            "cases": N_CASES,
            "seed": SEED,
            "e_total": E_TOTAL,
            "topk": 2,
            "m_total": M_TOTAL,
            "active_expert_counts": list(ACTIVE_COUNTS),
            "hot_expert_counts": list(HOT_COUNTS),
            "cold_tail_fraction_targets": list(COLD_FRACTIONS),
            "initial_cache_regimes": 5,
        },
        "interpretation": (
            "Coverage requires every action in the complete legal offline-seed "
            "rollout to name only a window-visible or already-resident expert."
        ),
        "configuration": {
            "windows": [list(window) for window in WINDOWS],
            "workers": workers,
            "source": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "retained_boundary_audit": {
                "path": str(BOUNDARY_AUDIT.resolve()),
                "sha256": _sha256(BOUNDARY_AUDIT),
            },
        },
        "runtime_s": time.perf_counter() - started,
        "summary": {
            "generated_1000": summary,
            "combined_1065": combined,
        },
        "records": records,
    }
    _atomic_write(OUTPUT, payload)
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
