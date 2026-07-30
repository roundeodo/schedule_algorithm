#!/usr/bin/env python3
"""Evaluate the top4+bottom2 policy on deterministic directed distributions.

The suite separates two claims:

* ``certificate`` cases are small structural tests.  Equality with the
  admissible four-stage lower bound is an exact optimality certificate.
* ``workload`` cases use E=64 and top-2 routed totals for batches of
  64/128/256 tokens.  They exercise local-hot, medium-hot, and flat-small-tail
  regimes.  A positive lower-bound gap is reported, never relabelled as a
  theoretical impossibility.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import four_stage_scheduler as reference
import scheduler_hw_fixed_policy as hw_v2
import scheduler_rtl_adaptive_prefetch_policy as adaptive
import scheduler_top4_bottom2_policy as top4_bottom2


TICK_CC = top4_bottom2.TICK_CC


@dataclass(frozen=True)
class DirectedCase:
    name: str
    tier: str
    family: str
    batch_tokens: int | None
    counts: tuple[int, ...]
    origin: str = "hand_directed"
    hot_experts: int | None = None
    medium_experts: int | None = None
    profile: str | None = None


def _case(
    name: str,
    tier: str,
    family: str,
    counts: list[int],
    *,
    batch_tokens: int | None = None,
    origin: str = "hand_directed",
    hot_experts: int | None = None,
    medium_experts: int | None = None,
    profile: str | None = None,
) -> DirectedCase:
    if not counts or any(value <= 0 for value in counts):
        raise ValueError(f"{name}: counts must be non-empty and positive")
    if len(counts) > 64:
        raise ValueError(f"{name}: active experts {len(counts)} exceeds E=64")
    if batch_tokens is not None and sum(counts) != 2 * batch_tokens:
        raise ValueError(
            f"{name}: routed assignments {sum(counts)} != 2*batch {2*batch_tokens}"
        )
    return DirectedCase(
        name=name,
        tier=tier,
        family=family,
        batch_tokens=batch_tokens,
        counts=tuple(sorted(counts, reverse=True)),
        origin=origin,
        hot_experts=hot_experts,
        medium_experts=medium_experts,
        profile=profile,
    )


def _weighted_group_counts(
    total: int,
    active: int,
    hot_experts: int,
    medium_experts: int,
    weights: tuple[int, int, int],
) -> tuple[list[int], list[int], list[int]]:
    """Allocate an exact routed-token total across three deterministic bands."""
    tail_experts = active - hot_experts - medium_experts
    if tail_experts <= 0 or total < active:
        raise ValueError("invalid active/hot/medium allocation")
    hot_weight, medium_weight, tail_weight = weights
    per_expert_weights = (
        [hot_weight] * hot_experts
        + [medium_weight] * medium_experts
        + [tail_weight] * tail_experts
    )
    remaining = total - active
    weight_sum = sum(per_expert_weights)
    quotient_remainder = [
        divmod(remaining * weight, weight_sum) for weight in per_expert_weights
    ]
    counts = [1 + quotient for quotient, _remainder in quotient_remainder]
    residue = total - sum(counts)
    bonus_order = sorted(
        range(active),
        key=lambda index: (-quotient_remainder[index][1], index),
    )
    for index in bonus_order[:residue]:
        counts[index] += 1
    return (
        counts[:hot_experts],
        counts[hot_experts : hot_experts + medium_experts],
        counts[hot_experts + medium_experts :],
    )


def generated_workload_cases() -> tuple[DirectedCase, ...]:
    """Systematic E64 top-2 grid with one through four local hot experts."""
    profiles = (
        ("sharp", (16, 5, 1), 6),
        ("local", (10, 4, 1), 4),
        ("broad", (7, 4, 2), 3),
    )
    cases: list[DirectedCase] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    for batch_tokens in (64, 128, 256):
        total = 2 * batch_tokens
        for active in (16, 24, 32, 48, 64):
            for hot_experts in (1, 2, 3, 4):
                for profile, weights, medium_divisor in profiles:
                    medium_experts = max(2, active // medium_divisor)
                    medium_experts = min(
                        medium_experts,
                        active - hot_experts - 1,
                    )
                    hot, medium, tail = _weighted_group_counts(
                        total,
                        active,
                        hot_experts,
                        medium_experts,
                        weights,
                    )
                    # Retain only cases with three observable bands.  This
                    # prevents a nominal hot-count label from describing a
                    # distribution whose integer allocation collapsed bands.
                    if min(hot) <= max(medium) or min(medium) < max(tail):
                        continue
                    counts = tuple(sorted(hot + medium + tail, reverse=True))
                    fingerprint = (batch_tokens, counts)
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    cases.append(
                        _case(
                            (
                                f"grid_m{batch_tokens}_a{active}_h{hot_experts}_"
                                f"med{medium_experts}_{profile}"
                            ),
                            "workload_grid",
                            f"{hot_experts}_hot_medium_flat_tail",
                            list(counts),
                            batch_tokens=batch_tokens,
                            origin="systematic_grid",
                            hot_experts=hot_experts,
                            medium_experts=medium_experts,
                            profile=profile,
                        )
                    )
    return tuple(cases)


def directed_cases() -> tuple[DirectedCase, ...]:
    hand_directed = (
        _case(
            "witness_16_16_4x4_2x5",
            "certificate",
            "two_hot_medium_cold",
            [16, 16] + [4] * 4 + [2] * 5,
        ),
        _case(
            "small_two_hot_cold",
            "certificate",
            "two_hot_flat_tail",
            [16, 16] + [2] * 8,
        ),
        _case(
            "small_two_hot_medium_cold",
            "certificate",
            "two_hot_medium_cold",
            [16, 16] + [4] * 4 + [2] * 6,
        ),
        _case(
            "small_one_hot_medium_cold",
            "certificate",
            "one_hot_medium_cold",
            [24] + [8] * 4 + [2] * 6,
        ),
        _case(
            "small_medium_band",
            "certificate",
            "medium_band",
            [8] * 4 + [4] * 4 + [2] * 4,
        ),
        _case(
            "m64_two_hot_4medium_26cold",
            "workload",
            "two_hot_medium_flat_tail",
            [24, 20] + [8] * 4 + [2] * 26,
            batch_tokens=64,
        ),
        _case(
            "m64_two_hot_8medium_24cold",
            "workload",
            "two_hot_medium_flat_tail",
            [16, 16] + [6] * 8 + [2] * 24,
            batch_tokens=64,
        ),
        _case(
            "m64_one_hot_6medium_24cold",
            "workload",
            "one_hot_medium_flat_tail",
            [20] + [10] * 6 + [2] * 24,
            batch_tokens=64,
        ),
        _case(
            "m64_medium_band_flat_tail",
            "workload",
            "medium_band_flat_tail",
            [8] * 8 + [4] * 8 + [2] * 16,
            batch_tokens=64,
        ),
        _case(
            "m64_two_hot_6medium_20cold",
            "workload",
            "two_hot_medium_flat_tail",
            [28, 24] + [6] * 6 + [2] * 20,
            batch_tokens=64,
        ),
        _case(
            "m128_two_hot_6medium_28small",
            "workload",
            "two_hot_medium_flat_tail",
            [40, 32] + [12] * 6 + [4] * 28,
            batch_tokens=128,
        ),
        _case(
            "m128_two_hot_12medium_50cold",
            "workload",
            "two_hot_medium_flat_tail",
            [32, 28] + [8] * 12 + [2] * 50,
            batch_tokens=128,
        ),
        _case(
            "m128_one_hot_4medium_36small",
            "workload",
            "one_hot_medium_flat_tail",
            [48] + [16] * 4 + [4] * 36,
            batch_tokens=128,
        ),
        _case(
            "m128_four_hot_8medium_32small",
            "workload",
            "local_hot_medium_flat_tail",
            [24] * 4 + [8] * 8 + [3] * 32,
            batch_tokens=128,
        ),
        _case(
            "m128_medium_hot_band",
            "workload",
            "medium_band_flat_tail",
            [20] * 6 + [4] * 30 + [2] * 8,
            batch_tokens=128,
        ),
        _case(
            "m256_two_hot_8medium_44small",
            "workload",
            "two_hot_medium_flat_tail",
            [64, 56] + [16] * 8 + [6] * 44,
            batch_tokens=256,
        ),
        _case(
            "m256_two_hot_10medium_50small",
            "workload",
            "two_hot_medium_flat_tail",
            [48, 44] + [12] * 10 + [6] * 50,
            batch_tokens=256,
        ),
        _case(
            "m256_eight_hot_32medium",
            "workload",
            "local_hot_medium_band",
            [32] * 8 + [8] * 32,
            batch_tokens=256,
        ),
        _case(
            "m256_six_hot_30medium_8small",
            "workload",
            "local_hot_medium_flat_tail",
            [40] * 6 + [8] * 30 + [4] * 8,
            batch_tokens=256,
        ),
        _case(
            "m256_twelve_medium_hot_32medium",
            "workload",
            "medium_hot_band",
            [24] * 12 + [7] * 32,
            batch_tokens=256,
        ),
        _case(
            "m256_near_balanced_two_bands",
            "workload",
            "near_balanced",
            [12] * 16 + [8] * 40,
            batch_tokens=256,
        ),
        _case(
            "m256_balanced_64active",
            "workload",
            "near_balanced",
            [10] * 32 + [6] * 32,
            batch_tokens=256,
        ),
    )
    return hand_directed + generated_workload_cases()


def _ticks_fraction(cc: int) -> str:
    value = Fraction(cc, TICK_CC)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _root_bounds(distribution: dict[int, int]) -> dict[str, int]:
    remaining = tuple(sorted(distribution.items(), key=lambda item: (-item[1], item[0])))
    initial = reference.make_initial_snap(-1)
    return reference.state_lower_bound_components(initial, initial, remaining)


def _fluid_compute_lb_cc(total_assignments: int) -> Fraction:
    # One Shape-C two-token block costs three ticks on one cluster.  Relax both
    # expert boundaries and indivisible blocks, then divide by two clusters.
    return Fraction(total_assignments * 3 * TICK_CC, 4)


def evaluate_case(case: DirectedCase) -> dict:
    distribution = {eid: count for eid, count in enumerate(case.counts)}
    bounds = _root_bounds(distribution)
    result = top4_bottom2.schedule_result(distribution)
    adaptive_cc = adaptive.adaptive_prefetch_schedule(distribution)
    hw_v2_cc = hw_v2.hw_v2_schedule(distribution)
    lower_bound_cc = bounds["combined_cc"]
    decision_counts = Counter(step.decision for step in result.steps)
    source_counts = Counter(step.source for step in result.steps)
    fluid_cc = _fluid_compute_lb_cc(sum(case.counts))

    return {
        "name": case.name,
        "tier": case.tier,
        "family": case.family,
        "origin": case.origin,
        "hot_experts": case.hot_experts,
        "medium_experts": case.medium_experts,
        "profile": case.profile,
        "batch_tokens": case.batch_tokens,
        "routed_assignments": sum(case.counts),
        "active_experts": len(case.counts),
        "counts": list(case.counts),
        "fluid_compute_lb_ticks": str(fluid_cc / TICK_CC),
        "four_stage_lb_ticks": _ticks_fraction(lower_bound_cc),
        "four_stage_lb_components_ticks": {
            key: _ticks_fraction(value)
            for key, value in bounds.items()
            if key.endswith("_cc")
        },
        "new_top4_bottom2_ticks": _ticks_fraction(result.makespan_cc),
        "adaptive_hw_v2_ticks": _ticks_fraction(adaptive_cc),
        "algorithmic_hw_v2_ticks": _ticks_fraction(hw_v2_cc),
        "new_minus_lb_ticks": _ticks_fraction(result.makespan_cc - lower_bound_cc),
        "new_minus_adaptive_ticks": _ticks_fraction(result.makespan_cc - adaptive_cc),
        "certificate": (
            "optimal_by_lower_bound"
            if result.makespan_cc == lower_bound_cc
            else "not_certified"
        ),
        "decision_counts": dict(sorted(decision_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "steps": [asdict(step) for step in result.steps],
    }


def _candidate_successors(state, cost_model):
    """All bounded candidates; policy branch guards are deliberately ignored."""
    if len(state.remaining) == 1:
        with adaptive._use_cost_model(cost_model):
            return hw_v2.generate_one_idle_shape_successors(
                state,
                policy="balanced",
                top_policy="pruned",
                n1_policy="pruned",
            )
    if state.c2.task_end != state.c3.task_end:
        return [
            transition
            for _priority, _source, transition in top4_bottom2._one_idle_candidates(
                state, cost_model
            )
        ]
    with adaptive._use_cost_model(cost_model):
        transitions = hw_v2.generate_one_idle_shape_successors(
            state,
            policy="balanced",
            top_policy="pruned",
            n1_policy="pruned",
        )
    hot_medium = top4_bottom2._hot_medium_transition(state, cost_model)
    if hot_medium is not None:
        transitions.append(hot_medium)
    return transitions


def _candidate_estimate(state, cost_model) -> int:
    if not state.remaining:
        return max(state.c2.task_end, state.c3.task_end)
    with adaptive._use_cost_model(cost_model):
        return hw_v2.hw_v2_continuation(
            state.c2, state.c3, state.remaining, policy="balanced"
        )


def bounded_candidate_oracle(
    counts: tuple[int, ...], beam_width: int
) -> tuple[int, tuple[str, ...]]:
    """Offline beam over the bounded candidate bank.

    The result is a feasible upper bound, not an optimality proof by itself.
    If it equals the independent four-stage lower bound, equality supplies the
    certificate.  This search is diagnostic only and is never called by the
    runtime policy.
    """
    distribution = {eid: count for eid, count in enumerate(counts)}
    cost_model = adaptive._COST_MODELS["single_first"]
    with adaptive._use_cost_model(cost_model):
        initial = hw_v2.initial_state(distribution)
    greedy_cc = top4_bottom2.schedule(distribution)
    best_cc = greedy_cc
    best_history: tuple[str, ...] = ("runtime_policy_seed",)
    beam = [(initial, tuple())]

    for _depth in range(2 * len(counts) + 5):
        next_by_state: dict[tuple, tuple[int, object, tuple[str, ...]]] = {}
        for state, history in beam:
            for transition in _candidate_successors(state, cost_model):
                child = transition.state
                child_history = history + (transition.tag,)
                if not child.remaining:
                    makespan = max(child.c2.task_end, child.c3.task_end)
                    if makespan < best_cc:
                        best_cc = makespan
                        best_history = child_history
                    continue
                estimate = _candidate_estimate(child, cost_model)
                if estimate >= best_cc:
                    continue
                fingerprint = hw_v2.state_key(child)
                previous = next_by_state.get(fingerprint)
                if previous is None or estimate < previous[0]:
                    next_by_state[fingerprint] = (
                        estimate,
                        child,
                        child_history,
                    )
        if not next_by_state:
            break
        ordered = sorted(
            next_by_state.values(),
            key=lambda item: (
                item[0],
                max(item[1].c2.task_end, item[1].c3.task_end),
                len(item[1].remaining),
            ),
        )[:beam_width]
        beam = [(item[1], item[2]) for item in ordered]
    return best_cc, best_history


def _summary(rows: list[dict]) -> dict:
    certified = [row for row in rows if row["certificate"] == "optimal_by_lower_bound"]
    better = [row for row in rows if Fraction(row["new_minus_adaptive_ticks"]) < 0]
    equal = [row for row in rows if Fraction(row["new_minus_adaptive_ticks"]) == 0]
    worse = [row for row in rows if Fraction(row["new_minus_adaptive_ticks"]) > 0]
    oracle_rows = [row for row in rows if "candidate_oracle_ticks" in row]
    hand_uncertified = [
        row
        for row in rows
        if row["origin"] == "hand_directed"
        and row["certificate"] != "optimal_by_lower_bound"
    ]

    def counts(field: str) -> dict[str, int]:
        return dict(
            sorted(
                Counter(str(row[field]) for row in rows).items(),
                key=lambda item: item[0],
            )
        )

    return {
        "cases": len(rows),
        "cases_by_origin": counts("origin"),
        "cases_by_tier": counts("tier"),
        "cases_by_hot_experts": counts("hot_experts"),
        "cases_by_batch_tokens": counts("batch_tokens"),
        "cases_by_profile": counts("profile"),
        "certified_optimal": len(certified),
        "better_than_adaptive_hw_v2": len(better),
        "equal_to_adaptive_hw_v2": len(equal),
        "worse_than_adaptive_hw_v2": len(worse),
        "uncertified_count": len(rows) - len(certified),
        "hand_directed_uncertified_cases": [row["name"] for row in hand_uncertified],
        "regressed_cases": [row["name"] for row in worse],
        "candidate_oracle_cases": len(oracle_rows),
        "candidate_oracle_hits_lower_bound": sum(
            row.get("candidate_oracle_certificate") == "optimal_by_lower_bound"
            for row in oracle_rows
        ),
        "runtime_selection_gap_cases": [
            row["name"]
            for row in oracle_rows
            if Fraction(row["candidate_oracle_ticks"])
            < Fraction(row["new_top4_bottom2_ticks"])
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/policy_search/top4_bottom2_directed.json"),
    )
    parser.add_argument(
        "--oracle-beam-width",
        type=int,
        default=512,
        help="offline bounded-candidate beam for certificate cases; 0 disables",
    )
    parser.add_argument(
        "--diagnostic-beam-widths",
        default="32,128",
        help=(
            "comma-separated offline beam widths for uncertified hand-directed "
            "workload cases and runtime regressions; empty string disables"
        ),
    )
    args = parser.parse_args()

    cases = directed_cases()
    rows = [evaluate_case(case) for case in cases]
    if args.oracle_beam_width < 0:
        raise SystemExit("--oracle-beam-width must be non-negative")
    try:
        diagnostic_widths = tuple(
            int(field.strip())
            for field in args.diagnostic_beam_widths.split(",")
            if field.strip()
        )
    except ValueError as exc:
        raise SystemExit("--diagnostic-beam-widths must contain integers") from exc
    if any(width <= 0 for width in diagnostic_widths):
        raise SystemExit("--diagnostic-beam-widths values must be positive")
    if args.oracle_beam_width:
        for case, row in zip(cases, rows):
            if case.tier != "certificate":
                continue
            oracle_cc, history = bounded_candidate_oracle(
                case.counts, args.oracle_beam_width
            )
            row["candidate_oracle_ticks"] = _ticks_fraction(oracle_cc)
            row["candidate_oracle_history"] = list(history)
            row["candidate_oracle_certificate"] = (
                "optimal_by_lower_bound"
                if oracle_cc == Fraction(row["four_stage_lb_ticks"]) * TICK_CC
                else "not_certified"
            )
    if diagnostic_widths:
        for case, row in zip(cases, rows):
            is_hand_uncertified = (
                case.origin == "hand_directed"
                and case.tier == "workload"
                and row["certificate"] != "optimal_by_lower_bound"
            )
            is_runtime_regression = Fraction(
                row["new_minus_adaptive_ticks"]
            ) > 0
            if not (is_hand_uncertified or is_runtime_regression):
                continue
            trials = []
            for width in diagnostic_widths:
                trial_cc, trial_history = bounded_candidate_oracle(
                    case.counts, width
                )
                trials.append((trial_cc, width, trial_history))
            oracle_cc, best_width, history = min(
                trials, key=lambda item: (item[0], item[1])
            )
            row["candidate_oracle_ticks"] = _ticks_fraction(oracle_cc)
            row["candidate_oracle_best_beam_width"] = best_width
            row["candidate_oracle_trials_ticks"] = {
                str(width): _ticks_fraction(trial_cc)
                for trial_cc, width, _trial_history in trials
            }
            row["candidate_oracle_history"] = list(history)
            row["candidate_oracle_certificate"] = (
                "optimal_by_lower_bound"
                if oracle_cc == Fraction(row["four_stage_lb_ticks"]) * TICK_CC
                else "not_certified"
            )
            row["runtime_minus_candidate_oracle_ticks"] = _ticks_fraction(
                Fraction(row["new_top4_bottom2_ticks"]) * TICK_CC - oracle_cc
            )
    summary = _summary(rows)
    payload = {
        "schema": "top4_bottom2_directed_v2",
        "model": {
            "base": "scheduler_rtl_adaptive_prefetch_policy single_first",
            "window": "semantic top4+bottom2",
            "runtime_search": "none",
        },
        "summary": summary,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(
        "name tier active LB new adaptive delta certificate"
    )
    for row in rows:
        print(
            row["name"],
            row["tier"],
            row["active_experts"],
            row["four_stage_lb_ticks"],
            row["new_top4_bottom2_ticks"],
            row["adaptive_hw_v2_ticks"],
            row["new_minus_adaptive_ticks"],
            row["certificate"],
        )
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output}")

    witness = next(row for row in rows if row["name"] == "witness_16_16_4x4_2x5")
    if witness["new_top4_bottom2_ticks"] != "45":
        raise SystemExit("witness regression: expected 45 ticks")
    if witness["certificate"] != "optimal_by_lower_bound":
        raise SystemExit("witness is no longer certified by the four-stage lower bound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
