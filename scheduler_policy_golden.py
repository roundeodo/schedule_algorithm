#!/usr/bin/env python3
"""Deterministic golden model for the frozen bounded RTL scheduler policy.

Candidate construction is shared with ``derive_scheduler_policy.py`` so that
the audited direct-v8 generator and the deployable policy cannot drift.  This
file owns the final runtime constants, future score, tie-break and commit loop.
It performs no beam search, continuation rollout or fitted-model inference.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable

from four_stage_scheduler import (
    BeamState,
    FourStageScheduler,
    StageAction,
    apply_action,
    clear_scheduler_caches,
    validate_schedule_history,
)
from run_four_stage_reference import serialize_action


POLICY_ID = "r4-b2-k32-direct-v8-lpt-rem-snap-v1"
CANDIDATE_REVISION = "direct-slot-conditional-cache-v8"
TIME_QUANTUM_CC = 11_264
RANK_LIMIT = 4
BOTTOM_COUNT = 2
CANDIDATE_BUDGET = 32


@dataclass(frozen=True)
class Decision:
    """One committed golden-policy decision and its auditable score key."""

    action: StageAction
    child: BeamState
    candidate_index: int
    candidate_count: int
    score_key: tuple[int, int, int, int]


CandidateGenerator = Callable[..., list[StageAction]]


def isolated_duration_cc(ntok: int) -> int:
    """Best isolated four-stage duration used by the integer LPT estimate.

    In the fixed timing model the best S1/S2 duration is
    ``ceil(ntok/2) * 2*Tq`` and the best S3/S4 duration is
    ``ceil(ntok/2) * Tq``.  The combined duration therefore needs only an
    increment, right shift, and shift-add multiplication by three.
    """
    if ntok <= 0:
        return 0
    half_token_blocks = (int(ntok) + 1) >> 1
    return 3 * TIME_QUANTUM_CC * half_token_blocks


def lpt_future_score_cc(state: BeamState) -> int:
    """Estimate final makespan by two-lane longest-processing-time placement."""
    end2 = int(state.c2.task_end)
    end3 = int(state.c3.task_end)
    # ``remaining`` is maintained in descending token order by the state
    # transition.  Sorting here makes that contract explicit and defensive;
    # equal-token experts have equal duration, so their ID order cannot change
    # the score.
    remaining = sorted(state.remaining, key=lambda item: (-item[1], item[0]))
    for _, ntok in remaining:
        duration = isolated_duration_cc(ntok)
        if end2 <= end3:
            end2 += duration
        else:
            end3 += duration
    return max(int(state.f_score), end2, end3)


def _default_candidate_generator() -> CandidateGenerator:
    # Lazy import avoids a module cycle: the derivation tool calls this golden
    # loop with its already-loaded generator, while standalone golden-model use
    # resolves the same function here.
    from derive_scheduler_policy import generate_direct_candidates

    return generate_direct_candidates


def select_action(
    state: BeamState,
    *,
    candidate_generator: CandidateGenerator | None = None,
) -> Decision:
    """Build, score and select one action under the frozen RTL policy."""
    generator = candidate_generator or _default_candidate_generator()
    actions = generator(
        state,
        rank_limit=RANK_LIMIT,
        bottom_count=BOTTOM_COUNT,
        budget=CANDIDATE_BUDGET,
    )
    if not actions:
        raise RuntimeError("golden policy has no legal candidate")
    if len(actions) > CANDIDATE_BUDGET:
        raise RuntimeError("golden policy exceeded the candidate budget")

    decisions = []
    for candidate_index, action in enumerate(actions):
        child = apply_action(state, action)
        score_key = (
            lpt_future_score_cc(child),
            len(child.remaining),
            max(child.c2.task_end, child.c3.task_end),
            candidate_index,
        )
        decisions.append(
            Decision(
                action=action,
                child=child,
                candidate_index=candidate_index,
                candidate_count=len(actions),
                score_key=score_key,
            )
        )
    return min(decisions, key=lambda decision: decision.score_key)


def run_policy(
    initial: BeamState,
    *,
    candidate_generator: CandidateGenerator | None = None,
) -> tuple[BeamState, list[StageAction], int]:
    """Run the frozen policy round by round until every expert is consumed."""
    state = initial
    history = []
    max_candidates = 0
    max_decisions = 4 * len(initial.remaining) + 8
    while state.remaining:
        decision = select_action(
            state,
            candidate_generator=candidate_generator,
        )
        history.append(decision.action)
        state = decision.child
        max_candidates = max(max_candidates, decision.candidate_count)
        if len(history) > max_decisions:
            raise RuntimeError("golden policy exceeded the progress guard")
    return state, history, max_candidates


def run_distribution(
    distribution: dict[int, int],
    *,
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
) -> dict:
    """Convenience API that returns a serialized, independently validated run."""
    scheduler = FourStageScheduler(
        distribution,
        initial_cache_c2=initial_cache_c2,
        initial_cache_c3=initial_cache_c3,
    )
    final, history, max_candidates = run_policy(scheduler._initial_state())
    validated = validate_schedule_history(
        tuple(history),
        scheduler.token_dist,
        initial_cache_c2=initial_cache_c2,
        initial_cache_c3=initial_cache_c3,
    )
    if validated != final.g_score:
        raise RuntimeError(
            f"golden replay makespan {validated} != state score {final.g_score}"
        )
    serialized = [serialize_action(action) for action in history]
    history_blob = json.dumps(
        serialized, sort_keys=True, separators=(",", ":")
    ).encode()
    result = {
        "schema": "scheduler_policy_golden_run_v1",
        "policy_id": POLICY_ID,
        "candidate_revision": CANDIDATE_REVISION,
        "rank_limit": RANK_LIMIT,
        "bottom_count": BOTTOM_COUNT,
        "candidate_budget": CANDIDATE_BUDGET,
        "makespan_cc": int(final.g_score),
        "decisions": len(history),
        "max_candidates": max_candidates,
        "history_sha256": hashlib.sha256(history_blob).hexdigest(),
        "actions": serialized,
    }
    # Candidate timing helpers are intentionally cached within one run. Clear
    # them at the public case boundary so long regressions do not retain state
    # from thousands of unrelated distributions.
    clear_scheduler_caches()
    from derive_scheduler_policy import _equal_finish_left, _release_target_left

    _equal_finish_left.cache_clear()
    _release_target_left.cache_clear()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-json",
        required=True,
        help='expert-token object, for example \'{"0": 16, "1": 8}\'',
    )
    parser.add_argument("--initial-cache-c2", type=int, default=-1)
    parser.add_argument("--initial-cache-c3", type=int, default=-1)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = json.loads(args.dist_json)
    if not isinstance(raw, dict):
        raise ValueError("--dist-json must decode to an object")
    result = run_distribution(
        {int(eid): int(ntok) for eid, ntok in raw.items()},
        initial_cache_c2=args.initial_cache_c2,
        initial_cache_c3=args.initial_cache_c3,
    )
    text = json.dumps(result, indent=2)
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(text + "\n")
        temporary.replace(args.out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
