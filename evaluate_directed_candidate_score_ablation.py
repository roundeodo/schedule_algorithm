#!/usr/bin/env python3
"""Screen bounded candidate/scorer changes on the directed suite.

Mirror results are diagnostic because the bounded model can contain ghost
prefetch state.  ``--legal-variant`` additionally lowers the exact variant
trace into the independent explicit-DMA reference and replay-validates it.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path

import four_stage_scheduler as reference
import prove_top4_bottom2_directed as proof
import scheduler_hw_fixed_policy as hw_base
import scheduler_rtl_adaptive_prefetch_policy as adaptive
import scheduler_top4_bottom2_policy as policy


@dataclass(frozen=True)
class Variant:
    name: str
    split_ranks: tuple[int, ...] = ()
    add_both_idle_single: bool = False
    continuation_head: int = 4
    tie_mode: str = "baseline"


VARIANTS = {
    variant.name: variant
    for variant in (
        Variant("baseline"),
        Variant("tie_split", tie_mode="split_first"),
        Variant("tie_current_first", tie_mode="current_first"),
        Variant("tie_greedy_current", tie_mode="greedy_current"),
        Variant("tie_guarded_split_half", tie_mode="guarded_split_half"),
        Variant("head6", continuation_head=6),
        Variant("split12", split_ranks=(1, 2)),
        Variant(
            "split12_tie_current_first",
            split_ranks=(1, 2),
            tie_mode="current_first",
        ),
        Variant(
            "split12_tie_greedy_current",
            split_ranks=(1, 2),
            tie_mode="greedy_current",
        ),
        Variant(
            "split12_tie_guarded_split_half",
            split_ranks=(1, 2),
            tie_mode="guarded_split_half",
        ),
        Variant(
            "split12_tie_guarded_non_top_split_half",
            split_ranks=(1, 2),
            tie_mode="guarded_non_top_split_half",
        ),
        Variant("split12_single", split_ranks=(1, 2), add_both_idle_single=True),
        Variant(
            "split12_single_tie_current_first",
            split_ranks=(1, 2),
            add_both_idle_single=True,
            tie_mode="current_first",
        ),
        Variant(
            "split12_single_tie_greedy_current",
            split_ranks=(1, 2),
            add_both_idle_single=True,
            tie_mode="greedy_current",
        ),
        Variant(
            "split12_single_tie_split",
            split_ranks=(1, 2),
            add_both_idle_single=True,
            tie_mode="split_first",
        ),
        Variant(
            "split12_single_head6",
            split_ranks=(1, 2),
            add_both_idle_single=True,
            continuation_head=6,
        ),
        Variant(
            "split12_single_head6_tie_split",
            split_ranks=(1, 2),
            add_both_idle_single=True,
            continuation_head=6,
            tie_mode="split_first",
        ),
    )
}


def _normalize_child(before, transition, tag):
    remaining_eids = {eid for eid, _count in transition.state.remaining}
    remaining = tuple(
        item for item in before.remaining if item[0] in remaining_eids
    )
    return hw_base.Transition(
        hw_base.PolicyState(
            transition.state.c2,
            transition.state.c3,
            remaining,
        ),
        tag,
    )


def _augmented_candidates(state, cost_model, variant: Variant):
    with adaptive._use_cost_model(cost_model):
        baseline = hw_base.generate_one_idle_shape_successors(
            state,
            policy="balanced",
            top_policy="pruned",
            n1_policy="pruned",
        )
        if state.c2.task_end != state.c3.task_end:
            return baseline

        extras = []
        for rank in variant.split_ranks:
            if rank >= len(state.remaining):
                continue
            reordered = (
                state.remaining[rank],
            ) + state.remaining[:rank] + state.remaining[rank + 1 :]
            trial_state = hw_base.PolicyState(state.c2, state.c3, reordered)
            for transition in hw_base.generate_one_idle_shape_successors(
                trial_state,
                policy="balanced",
                top_policy="pruned",
                n1_policy="pruned",
            ):
                if not transition.tag.startswith("split_0_"):
                    continue
                cut = transition.tag.rsplit("_", 1)[-1]
                extras.append(
                    _normalize_child(
                        state,
                        transition,
                        f"split_{rank}_{cut}",
                    )
                )

        if variant.add_both_idle_single and state.remaining:
            prepared_c2, prepared_c3 = hw_base._prepare(state.c2, state.c3)
            for transition in hw_base._one_idle_successors(
                prepared_c2, prepared_c3, state.remaining
            ):
                extras.append(
                    _normalize_child(
                        state,
                        transition,
                        f"both_idle_single_0__{transition.tag}",
                    )
                )

    unique = {}
    for transition in (*baseline, *extras):
        unique.setdefault(hw_base.state_key(transition.state), transition)
    return list(unique.values())


def _continuation(c2, c3, remaining, head: int) -> int:
    if head == 4:
        return hw_base.hw_v2_continuation(
            c2, c3, remaining, policy="balanced"
        )
    if not remaining:
        return max(c2.task_end, c3.task_end)
    greedy = hw_base.cm._cc_greedy_h(c2.task_end, c3.task_end, remaining)
    if len(remaining) <= 2:
        return greedy
    loads = [int(c2.task_end), int(c3.task_end)]
    for _eid, ntok in remaining[:head]:
        target = 0 if loads[0] <= loads[1] else 1
        loads[target] += int(hw_base.cm._cc_best_task(int(ntok)))
    tail_work = sum(
        hw_base.cm._cc_best_task(int(ntok))
        for _eid, ntok in remaining[head:]
    )
    hw_base._balance_divisible_work(loads, tail_work)
    return min(greedy, max(loads))


def _candidate_id(state, transition, ordinal: int) -> int:
    try:
        _token, candidate_id = adaptive.token_from_tag(state, transition.tag)
        return candidate_id
    except (AssertionError, KeyError, RuntimeError, ValueError):
        return 1000 + ordinal


def _choose_variant(state, cost_model, variant: Variant):
    transitions = _augmented_candidates(state, cost_model, variant)

    def metrics(item):
        ordinal, transition = item
        child = transition.state
        current = max(child.c2.task_end, child.c3.task_end)
        if len(state.remaining) == 1 or state.c2.task_end != state.c3.task_end:
            cost = adaptive._ceil_div(current, policy.TICK_CC)
        else:
            cost = adaptive._ceil_div(
                _continuation(
                    child.c2,
                    child.c3,
                    child.remaining,
                    variant.continuation_head,
                ),
                policy.TICK_CC,
            )
        candidate_id = _candidate_id(state, transition, ordinal)
        split_priority = 0 if transition.tag.startswith("split_") else 1
        return cost, current, len(child.remaining), candidate_id, split_priority

    def key(item):
        ordinal, transition = item
        cost, current, remaining_count, candidate_id, split_priority = metrics(item)
        if variant.tie_mode == "split_first":
            return (
                cost,
                split_priority,
                remaining_count,
                adaptive._ceil_div(current, policy.TICK_CC),
                candidate_id,
                transition.tag,
            )
        if variant.tie_mode == "current_first":
            return (
                cost,
                adaptive._ceil_div(current, policy.TICK_CC),
                remaining_count,
                candidate_id,
                transition.tag,
            )
        if variant.tie_mode == "greedy_current":
            child = transition.state
            greedy = hw_base.cm._cc_greedy_h(
                child.c2.task_end, child.c3.task_end, child.remaining
            )
            return (
                cost,
                adaptive._ceil_div(greedy, policy.TICK_CC),
                adaptive._ceil_div(current, policy.TICK_CC),
                remaining_count,
                candidate_id,
                transition.tag,
            )
        return (
            cost,
            remaining_count,
            adaptive._ceil_div(current, policy.TICK_CC),
            candidate_id,
            transition.tag,
        )

    indexed = list(enumerate(transitions))
    if variant.tie_mode in {
        "guarded_split_half",
        "guarded_non_top_split_half",
    }:
        baseline_chosen = min(indexed, key=key)
        baseline_cost, baseline_current, _remaining, _id, baseline_split = metrics(
            baseline_chosen
        )
        if baseline_split != 0:
            eligible = []
            now = max(state.c2.task_end, state.c3.task_end)
            for item in indexed:
                cost, current, remaining_count, candidate_id, split_priority = metrics(
                    item
                )
                if (
                    split_priority == 0
                    and (
                        variant.tie_mode != "guarded_non_top_split_half"
                        or not item[1].tag.startswith("split_0_")
                    )
                    and cost == baseline_cost
                    and 2 * (current - now) <= baseline_current - now
                ):
                    eligible.append(
                        (
                            current,
                            remaining_count,
                            candidate_id,
                            item[1].tag,
                            item,
                        )
                    )
            if eligible:
                return min(eligible)[-1][1]
        return baseline_chosen[1]
    return min(indexed, key=key)[1]


@contextmanager
def _variant_adaptive_chooser(variant: Variant):
    original = adaptive._choose_transition
    if variant.name == "baseline":
        yield
        return
    adaptive._choose_transition = (
        lambda state, cost_model: _choose_variant(state, cost_model, variant)
    )
    try:
        yield
    finally:
        adaptive._choose_transition = original


def _variant_trace(distribution: dict[int, int], variant: Variant):
    with _variant_adaptive_chooser(variant):
        return proof._mirror_trace(distribution)


def _legal_lowering(distribution, trace):
    variants = []
    errors = []
    for mode, proactive in (("late", False), ("proactive", True)):
        try:
            state, lowering = proof.lower_policy_to_reference(
                distribution,
                proactive_prefetch=proactive,
                mirror_trace=trace,
            )
            variants.append((state.g_score, mode, state, lowering))
        except RuntimeError as exc:
            errors.append(f"{mode}: {exc}")
    if not variants:
        return None, None, None, "; ".join(errors)
    _score, mode, state, lowering = min(variants, key=lambda item: item[:2])
    return state, mode, lowering, "; ".join(errors) or None


def _summary(rows, variant_name):
    proven = [row for row in rows if row["reference_proven_optimal"]]
    deltas = [Fraction(row["mirror_delta_vs_baseline_ticks"]) for row in rows]
    legal = [row for row in rows if row.get("legal_lowering_ticks") is not None]
    return {
        "variant": variant_name,
        "cases": len(rows),
        "mirror_better_equal_worse_vs_baseline": {
            "better": sum(delta < 0 for delta in deltas),
            "equal": sum(delta == 0 for delta in deltas),
            "worse": sum(delta > 0 for delta in deltas),
        },
        "mirror_exact_on_proven_diagnostic": sum(
            Fraction(row["mirror_ticks"]) == Fraction(row["reference_best_ticks"])
            for row in proven
        ),
        "mirror_regressions_on_proven": [
            row["name"]
            for row in proven
            if Fraction(row["mirror_delta_vs_baseline_ticks"]) > 0
        ],
        "legal_lowering_cases": len(legal),
        "legal_lowering_failures": len(rows) - len(legal),
        "legal_exact_on_proven": sum(
            row["reference_proven_optimal"]
            and Fraction(row["legal_lowering_ticks"])
            == Fraction(row["reference_best_ticks"])
            for row in legal
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy-baseline", type=Path, required=True)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="variant to evaluate; empty evaluates every variant",
    )
    parser.add_argument(
        "--legal-variant",
        action="append",
        default=[],
        help="variant for which explicit-DMA lowering is also run",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selected = args.variant or list(VARIANTS)
    unknown = (set(selected) | set(args.legal_variant)) - set(VARIANTS)
    if unknown:
        raise SystemExit(f"unknown variants: {sorted(unknown)}")
    reference_payload = json.loads(args.reference.read_text(encoding="utf-8"))
    baseline_payload = json.loads(args.policy_baseline.read_text(encoding="utf-8"))
    baseline_by_name = {row["name"]: row for row in baseline_payload["cases"]}
    source_rows = reference_payload["cases"][: args.limit or None]

    results = {}
    summaries = {}
    for variant_name in selected:
        variant = VARIANTS[variant_name]
        rows = []
        for source in source_rows:
            distribution = {
                eid: count for eid, count in enumerate(source["counts"])
            }
            trace = _variant_trace(distribution, variant)
            mirror_cc = max(trace[-1].after.c2.task_end, trace[-1].after.c3.task_end)
            baseline = baseline_by_name[source["name"]]
            row = {
                "name": source["name"],
                "reference_proven_optimal": source["proven_optimal"],
                "reference_best_ticks": source["best_reference_ticks"],
                "baseline_mirror_ticks": baseline["mirror_policy_ticks"],
                "mirror_ticks": proof._ticks(mirror_cc),
                "mirror_delta_vs_baseline_ticks": str(
                    Fraction(mirror_cc, policy.TICK_CC)
                    - Fraction(baseline["mirror_policy_ticks"])
                ),
                "legal_lowering_ticks": None,
                "legal_lowering_mode": None,
                "legal_lowering_error": None,
            }
            if variant_name in args.legal_variant:
                state, mode, _lowering, error = _legal_lowering(
                    distribution, trace
                )
                if state is not None:
                    row["legal_lowering_ticks"] = proof._ticks(state.g_score)
                    row["legal_lowering_mode"] = mode
                row["legal_lowering_error"] = error
            rows.append(row)
        results[variant_name] = rows
        summaries[variant_name] = _summary(rows, variant_name)
        print(json.dumps(summaries[variant_name], indent=2))

    payload = {
        "schema": "directed_candidate_score_ablation_v1",
        "reference": str(args.reference),
        "policy_baseline": str(args.policy_baseline),
        "interpretation": (
            "mirror is diagnostic; only replay-valid legal lowering is physical evidence"
        ),
        "summaries": summaries,
        "variants": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
