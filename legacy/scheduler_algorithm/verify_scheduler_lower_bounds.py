#!/usr/bin/env python3
"""Exhaustively verify scheduler lower bounds on tiny reachable state spaces.

This checker deliberately does not use LB pruning.  It enumerates every legal
stage/prefetch successor, computes the exact completion makespan of each
reachable state, and asserts ``state_lower_bound <= exact_completion``.  It
also checks that ``apply_action`` preserves the ancestor LB through pathmax.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from four_stage_scheduler import (
    BeamState,
    FourStageScheduler,
    apply_action,
    clear_scheduler_caches,
    gen_prefetch_actions,
    gen_stage_actions,
    state_lower_bound,
)


CASES = (
    ({0: 1}, -1, -1),
    ({0: 2}, -1, -1),
    ({0: 3}, -1, -1),
    ({0: 1, 1: 1}, -1, -1),
    ({0: 2, 1: 1}, -1, -1),
    ({0: 2, 1: 2}, -1, -1),
    ({0: 3, 1: 1}, -1, -1),
    ({0: 2, 1: 1}, 0, -1),
    ({0: 2, 1: 1}, 0, 1),
)


def verify_case(
    dist: dict[int, int],
    initial_cache_c2: int,
    initial_cache_c3: int,
    max_states: int,
) -> tuple[int, int]:
    clear_scheduler_caches()
    scheduler = FourStageScheduler(
        dist,
        initial_cache_c2=initial_cache_c2,
        initial_cache_c3=initial_cache_c3,
    )
    initial = scheduler._initial_state()
    memo = {}
    visiting = set()
    checked = 0

    def exact_completion(c2, c3, remaining) -> int | float:
        nonlocal checked
        key = (c2, c3, remaining)
        if key in memo:
            return memo[key]
        if key in visiting:
            raise AssertionError("reachable state graph contains a cycle")
        if len(memo) + len(visiting) >= max_states:
            raise RuntimeError(f"exact checker exceeded --max-states={max_states}")
        if not remaining:
            return max(c2.task_end, c3.task_end)

        visiting.add(key)
        raw_lb = state_lower_bound(c2, c3, remaining)
        state = BeamState(
            c2=c2,
            c3=c3,
            remaining=remaining,
            history=(),
            g_score=max(c2.task_end, c3.task_end),
            f_score=raw_lb,
        )
        actions = gen_stage_actions(c2, c3, remaining)
        actions += gen_prefetch_actions(c2, c3, remaining)
        best = math.inf
        for action in actions:
            child = apply_action(state, action)
            if child.f_score < state.f_score:
                raise AssertionError(
                    f"pathmax decreased LB {state.f_score} -> {child.f_score}"
                )
            best = min(
                best,
                exact_completion(child.c2, child.c3, child.remaining),
            )
        visiting.remove(key)
        memo[key] = best
        checked += 1
        if best < math.inf and raw_lb > best:
            raise AssertionError(
                f"inadmissible LB={raw_lb} > exact={best} for "
                f"remaining={remaining}"
            )
        return best

    exact = exact_completion(initial.c2, initial.c3, initial.remaining)
    if exact == math.inf:
        raise AssertionError(f"no complete schedule for test case {dist}")
    clear_scheduler_caches()
    return int(exact), checked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-states", type=int, default=200_000)
    args = parser.parse_args()
    if args.max_states <= 0:
        raise SystemExit("--max-states must be positive")

    total_states = 0
    for dist, c2, c3 in CASES:
        exact, checked = verify_case(dist, c2, c3, args.max_states)
        total_states += checked
        print(
            f"PASS dist={dist} cache=({c2},{c3}) "
            f"exact={exact} checked_states={checked}"
        )
    print(f"all lower-bound checks passed; checked_states={total_states}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
