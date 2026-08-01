#!/usr/bin/env python3
"""Bounded one-round-at-a-time mirror of the distilled OLMoE scheduler.

The policy uses a fixed state-relative token ROM, one current-round candidate
evaluation pass and maintained aggregate state.  It never consumes an optimum
target and never runs beam, exact or child-state search.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics
from typing import Mapping, Sequence

import four_stage_scheduler as reference
from run_four_stage_reference import serialize_action
import evaluate_olmoe_fixed_token_banks as policy


HERE = Path(__file__).resolve().parent
DEFAULT_TOKEN_BANK = (
    HERE
    / "results"
    / "policy_search"
    / "olmoe_t5b1_hist4_bounded14_token_bank_v1.json"
)
POLICY_ID = "olmoe-t5b1-hist4-fixed14-bounded-release-pairwise-v1"
POLICY_WINDOW = (5, 1)
POLICY_SCORER = policy.HEAD5_HIST4_PAIRWISE_SCORER


@dataclass(frozen=True)
class RoundDecision:
    action: reference.StageAction
    next_state: reference.BeamState
    score: tuple
    candidate_count: int
    selector_metadata: dict


@dataclass(frozen=True)
class ScheduleResult:
    makespan_cc: int
    history: tuple[reference.StageAction, ...]
    rounds: int
    candidate_count_max: int
    candidate_count_mean: float


class BoundedOlmoeScheduler:
    """Fixed-ROM policy with a single sequential candidate comparator."""

    def __init__(self, token_bank: Path = DEFAULT_TOKEN_BANK):
        self.token_bank_path = token_bank.resolve()
        self.tokens = policy.load_explicit_token_bank(self.token_bank_path)

    @staticmethod
    def initial_state(
        token_distribution: Mapping[int, int],
    ) -> reference.BeamState:
        normalized = {
            int(eid): int(ntok)
            for eid, ntok in token_distribution.items()
            if int(ntok) > 0
        }
        state = reference.FourStageScheduler(normalized)._initial_state()
        return policy._bounded_policy_state(state, POLICY_SCORER)

    def choose_one_round(self, state: reference.BeamState) -> RoundDecision:
        """Generate, score and select candidates for the current state only."""
        candidates, generation = policy.generate_practical_probe_candidates(
            state,
            self.tokens,
            "bounded_release",
            "disabled",
            direct_generator=True,
            strict_token_bank=True,
            window=POLICY_WINDOW,
        )
        score, _tie, action, child, selector = (
            policy.select_practical_probe_candidate(
                state,
                candidates,
                scorer=POLICY_SCORER,
                sync_tiebreak="hot_cold",
                window=POLICY_WINDOW,
            )
        )
        return RoundDecision(
            action=action,
            next_state=child,
            score=score,
            candidate_count=int(generation["concrete_candidates"]),
            selector_metadata=selector,
        )

    def schedule(
        self,
        token_distribution: Mapping[int, int],
    ) -> ScheduleResult:
        normalized = {
            int(eid): int(ntok)
            for eid, ntok in token_distribution.items()
            if int(ntok) > 0
        }
        state = self.initial_state(normalized)
        candidate_counts = []
        while state.remaining:
            decision = self.choose_one_round(state)
            candidate_counts.append(decision.candidate_count)
            state = decision.next_state
        replay_cc = reference.validate_schedule_history(state.history, normalized)
        if replay_cc != state.g_score:
            raise AssertionError("bounded policy history failed explicit-DMA replay")
        return ScheduleResult(
            makespan_cc=int(state.g_score),
            history=tuple(state.history),
            rounds=len(state.history),
            candidate_count_max=max(candidate_counts, default=0),
            candidate_count_mean=(
                statistics.mean(candidate_counts) if candidate_counts else 0.0
            ),
        )


def _distribution_from_counts(counts: Sequence[int]) -> dict[int, int]:
    return {
        eid: int(ntok)
        for eid, ntok in enumerate(counts)
        if int(ntok) > 0
    }


def verify_proof(
    scheduler: BoundedOlmoeScheduler,
    proof_path: Path,
) -> dict:
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    rows = []
    for case in proof["cases"]:
        result = scheduler.schedule(_distribution_from_counts(case["counts"]))
        target_cc = policy._target_cc(case)
        rows.append(
            {
                "name": case["name"],
                "target_ticks": policy._ticks_text(target_cc),
                "makespan_ticks": policy._ticks_text(result.makespan_cc),
                "optimal": result.makespan_cc == target_cc,
                "rounds": result.rounds,
                "candidate_count_max": result.candidate_count_max,
            }
        )
    return {
        "policy_id": POLICY_ID,
        "window": list(POLICY_WINDOW),
        "scorer": POLICY_SCORER,
        "cases": len(rows),
        "optimal_cases": sum(row["optimal"] for row in rows),
        "candidate_count_max": max(
            row["candidate_count_max"] for row in rows
        ),
        "failures": [row for row in rows if not row["optimal"]],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit_optimal_witness(
    scheduler: BoundedOlmoeScheduler,
    proof_path: Path,
    case_name: str,
    output_path: Path,
) -> dict:
    """Materialize one independently replayed optimal policy history.

    This function does not claim compatibility with a smaller candidate
    window.  ``evaluate_directed_window_grid.py`` remains the independent
    visibility and equal-load-ID-relabel gate.
    """
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    matches = [case for case in proof["cases"] if case["name"] == case_name]
    if len(matches) != 1:
        raise ValueError(
            f"proof must contain exactly one case named {case_name!r}"
        )
    case = matches[0]
    target_cc = policy._target_cc(case)
    lower_cc = Fraction(str(case["certified_lower_bound_ticks"])) * policy.TICK_CC
    if not case.get("proven_optimal") or lower_cc.denominator != 1:
        raise ValueError(f"{case_name}: source case is not a valid LB=UB proof")
    if int(lower_cc) != target_cc:
        raise ValueError(f"{case_name}: source case does not satisfy LB=UB")

    result = scheduler.schedule(_distribution_from_counts(case["counts"]))
    if result.makespan_cc != target_cc:
        raise ValueError(
            f"{case_name}: policy {policy._ticks_text(result.makespan_cc)} "
            f"does not reach target {policy._ticks_text(target_cc)}"
        )

    row = dict(case)
    row.update(
        actions=[serialize_action(action) for action in result.history],
        history_replay_valid=True,
        termination="bounded_policy_history_equals_certified_lb",
        selected_history_source=POLICY_ID,
        selected_proof_source=str(proof_path.resolve()),
        policy_id=POLICY_ID,
        policy_rounds=result.rounds,
        policy_candidate_count_max=result.candidate_count_max,
    )
    payload = {
        "schema": "olmoe_bounded_policy_optimal_witness_v1",
        "complete": True,
        "proof_model": "explicit_dma_lane_four_stage",
        "manifest": {
            "policy_id": POLICY_ID,
            "policy_source": str(Path(__file__).resolve()),
            "policy_source_sha256": _sha256(Path(__file__).resolve()),
            "token_bank": str(scheduler.token_bank_path),
            "token_bank_sha256": _sha256(scheduler.token_bank_path),
            "source_proof": str(proof_path.resolve()),
            "source_proof_sha256": _sha256(proof_path),
            "case": case_name,
            "window_claim": list(POLICY_WINDOW),
            "scorer": POLICY_SCORER,
        },
        "summary": {
            "cases": 1,
            "proven_optimal": 1,
            "history_replay_valid": 1,
            "makespan_ticks": policy._ticks_text(result.makespan_cc),
        },
        "cases": [row],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-bank", type=Path, default=DEFAULT_TOKEN_BANK)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--counts",
        type=str,
        help="comma-separated expert token counts",
    )
    group.add_argument(
        "--proof",
        type=Path,
        help="run the frozen proof-set regression",
    )
    parser.add_argument(
        "--witness-case",
        help="with --proof, emit this case's optimal bounded-policy history",
    )
    parser.add_argument(
        "--witness-output",
        type=Path,
        help="JSON path for --witness-case",
    )
    args = parser.parse_args()
    scheduler = BoundedOlmoeScheduler(args.token_bank)
    if args.proof is not None:
        if (args.witness_case is None) != (args.witness_output is None):
            raise SystemExit(
                "--witness-case and --witness-output must be provided together"
            )
        if args.witness_case is not None:
            payload = emit_optimal_witness(
                scheduler,
                args.proof.resolve(),
                args.witness_case,
                args.witness_output.resolve(),
            )
            print(json.dumps(payload["summary"], indent=2))
            print(f"wrote {args.witness_output.resolve()}")
            return 0
        print(json.dumps(verify_proof(scheduler, args.proof.resolve()), indent=2))
        return 0
    if args.witness_case is not None or args.witness_output is not None:
        raise SystemExit("--witness-case requires --proof")
    try:
        counts = [int(value) for value in args.counts.split(",")]
    except ValueError as exc:
        raise SystemExit("--counts must be comma-separated integers") from exc
    result = scheduler.schedule(_distribution_from_counts(counts))
    print(
        json.dumps(
            {
                "policy_id": POLICY_ID,
                "makespan_ticks": policy._ticks_text(result.makespan_cc),
                "rounds": result.rounds,
                "candidate_count_max": result.candidate_count_max,
                "candidate_count_mean": result.candidate_count_mean,
                "history": [serialize_action(action) for action in result.history],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
