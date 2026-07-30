#!/usr/bin/env python3
"""Generate stratified deterministic OLMoE-like Top-2 scheduler cases.

The measured source statistics came from projecting the two highest-ranked
experts of a router trained for Top-8 routing.  Therefore these cases are not
presented as measurements from a model trained or evaluated with native Top-2
routing.  One case is an exact reconstructed observed histogram; the remaining
OLMoE-like cases are deterministic stress profiles constrained by reported
quantiles.  The systematic grid is stratified by hotspot load relative to the
mean over all 64 experts, hotspot multiplicity, active-expert population, and
cold-expert population.  A separate uniform-routing occupancy baseline is
included only as a control.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from dataclasses import dataclass
from pathlib import Path


TOTAL_EXPERTS = 64
TOKENS = 70
TOPK_PROJECTION = 2
TOTAL_ASSIGNMENTS = TOKENS * TOPK_PROJECTION
MEAN_ASSIGNMENTS_PER_EXPERT = TOTAL_ASSIGNMENTS / TOTAL_EXPERTS
# 12 tokens is 5.49x the exact 70-token Top-2 mean.  It separates the explicit
# local-hot leader group from the generated medium band in every grid profile.
LOCAL_HOT_LOAD_THRESHOLD = 12
DISCLAIMER = (
    "Top-2 projection of rankings from an OLMoE router trained for Top-8; "
    "not a native Top-2 model result."
)


@dataclass(frozen=True)
class Profile:
    name: str
    family: str
    evidence_kind: str
    counts: tuple[int, ...]
    description: str
    target_constraints: dict[str, object]


def _counts(*bands: tuple[int, int]) -> tuple[int, ...]:
    values: list[int] = []
    for value, repetitions in bands:
        values.extend([value] * repetitions)
    return tuple(values)


BASE_PROFILES = (
    Profile(
        name="olmoe_observed_ranked_window_001",
        family="observed_local_hot_many_cold",
        evidence_kind="observed_histogram_reconstructed_from_reported_ranking",
        counts=_counts(
            (16, 1), (15, 1), (6, 1), (5, 5), (4, 5), (3, 9),
            (2, 10), (1, 11), (0, 21),
        ),
        description=(
            "Exact 64-expert histogram reconstructed from the reported sorted "
            "window, total assignments, active-expert count, and <=2 count."
        ),
        target_constraints={
            "top1": 16,
            "top2": 15,
            "active_experts": 43,
            "experts_le_2": 42,
            "zero_experts": 21,
        },
    ),
    Profile(
        name="olmoe_iqr25_joint_constraint_profile",
        family="low_concentration_many_cold",
        evidence_kind="synthetic_joint_constraint_profile",
        counts=_counts(
            (15, 1), (10, 1), (5, 16), (4, 2), (2, 9), (1, 9), (0, 26)
        ),
        description=(
            "Deterministic profile matching the reported lower-quartile "
            "top1/top2, zero-expert, and <=2-expert constraints jointly."
        ),
        target_constraints={
            "top1": 15,
            "top2": 10,
            "experts_le_2": 44,
            "zero_experts": 26,
        },
    ),
    Profile(
        name="olmoe_median_joint_constraint_profile",
        family="median_local_hot_many_cold",
        evidence_kind="synthetic_joint_constraint_profile",
        counts=_counts(
            (22, 1), (12, 1), (6, 4), (5, 12), (2, 7), (1, 8), (0, 31)
        ),
        description=(
            "Deterministic profile matching the reported median top1/top2, "
            "zero-expert, and <=2-expert constraints jointly."
        ),
        target_constraints={
            "top1": 22,
            "top2": 12,
            "experts_le_2": 46,
            "zero_experts": 31,
        },
    ),
    Profile(
        name="olmoe_iqr75_joint_constraint_profile",
        family="high_concentration_many_zero",
        evidence_kind="synthetic_joint_constraint_profile",
        counts=_counts(
            (34, 1), (17, 1), (6, 3), (5, 10), (2, 7), (1, 7), (0, 35)
        ),
        description=(
            "Deterministic profile matching the reported upper-quartile "
            "top1/top2, zero-expert, and <=2-expert constraints jointly."
        ),
        target_constraints={
            "top1": 34,
            "top2": 17,
            "experts_le_2": 49,
            "zero_experts": 35,
        },
    ),
    Profile(
        name="olmoe_dual16_median_cold_profile",
        family="dual_hot_threshold_many_cold",
        evidence_kind="synthetic_threshold_profile",
        counts=_counts(
            (16, 2), (6, 6), (5, 10), (2, 7), (1, 8), (0, 31)
        ),
        description=(
            "Both leading experts are at the 16-token threshold while median "
            "zero-expert and <=2-expert counts are retained."
        ),
        target_constraints={
            "top1": 16,
            "top2": 16,
            "experts_le_2": 46,
            "zero_experts": 31,
        },
    ),
    Profile(
        name="olmoe_single_dominant_median_cold_profile",
        family="single_dominant_medium_band_many_cold",
        evidence_kind="synthetic_structure_profile",
        counts=_counts(
            (34, 1), (10, 1), (5, 10), (4, 6), (2, 7), (1, 8), (0, 31)
        ),
        description=(
            "One dominant expert, a separated second expert, a medium band, "
            "and the reported median cold/zero population."
        ),
        target_constraints={"experts_le_2": 46, "zero_experts": 31},
    ),
    Profile(
        name="olmoe_two_hot_plateau_median_cold_profile",
        family="two_hot_medium_plateau_many_cold",
        evidence_kind="synthetic_structure_profile",
        counts=_counts(
            (22, 1), (17, 1), (5, 15), (4, 1), (2, 7), (1, 8), (0, 31)
        ),
        description=(
            "Two separated hot experts followed by a broad equal medium band "
            "and the reported median cold/zero population."
        ),
        target_constraints={"experts_le_2": 46, "zero_experts": 31},
    ),
    Profile(
        name="olmoe_multi_hot_staircase_median_cold_profile",
        family="six_hot_staircase_many_cold",
        evidence_kind="synthetic_structure_profile",
        counts=_counts(
            (22, 1), (17, 1), (14, 1), (10, 1), (8, 1), (7, 1),
            (4, 4), (3, 8), (2, 7), (1, 8), (0, 31),
        ),
        description=(
            "Six nonuniform hot experts followed by medium and cold bands; "
            "tests more than the one-hot/two-hot special cases."
        ),
        target_constraints={"experts_le_2": 46, "zero_experts": 31},
    ),
    Profile(
        name="olmoe_four_hot_median_cold_profile",
        family="four_hot_medium_band_many_cold",
        evidence_kind="synthetic_structure_profile",
        counts=_counts(
            (20, 1), (18, 1), (16, 1), (14, 1), (4, 8), (3, 6),
            (2, 7), (1, 8), (0, 31),
        ),
        description=(
            "Four strong local hotspots, a medium band, and the reported "
            "median cold/zero population."
        ),
        target_constraints={"experts_le_2": 46, "zero_experts": 31},
    ),
    Profile(
        name="uniform_top2_occupancy_control",
        family="uniform_routing_occupancy_control",
        evidence_kind="deterministic_uniform_baseline",
        counts=_counts((4, 18), (3, 6), (2, 17), (1, 16), (0, 7)),
        description=(
            "Deterministic occupancy control matching the stated approximate "
            "uniform-routing counts for 0/1/2-token experts."
        ),
        target_constraints={"experts_le_2": 40, "zero_experts": 7},
    ),
)


def _constrained_profile(
    *,
    active_experts: int,
    experts_le_2: int,
    leaders: tuple[int, ...],
    archetype: str,
) -> Profile | None:
    """Construct one exact 140-assignment feature-grid profile.

    ``leaders`` fixes the local hotspot structure.  The positive cold band is
    divided between one- and two-token experts, while the remaining medium
    experts receive the most even integer allocation that stays below the last
    explicitly named leader.  Returning ``None`` means that combination of
    reported constraints is arithmetically infeasible.
    """
    zero_experts = TOTAL_EXPERTS - active_experts
    positive_cold = experts_le_2 - zero_experts
    medium_experts = active_experts - len(leaders) - positive_cold
    if positive_cold < 0 or medium_experts < 0:
        return None
    medium_cap = max(3, leaders[-1] - 1)
    preferred_twos = positive_cold // 2
    n2_order = sorted(
        range(positive_cold + 1),
        key=lambda n2: (abs(n2 - preferred_twos), n2),
    )
    for two_token_experts in n2_order:
        cold_sum = positive_cold + two_token_experts
        medium_sum = TOTAL_ASSIGNMENTS - sum(leaders) - cold_sum
        if medium_experts == 0:
            if medium_sum != 0:
                continue
            medium = []
        else:
            quotient, residue = divmod(medium_sum, medium_experts)
            if quotient < 3 or quotient + int(residue > 0) > medium_cap:
                continue
            medium = [quotient + 1] * residue + [quotient] * (
                medium_experts - residue
            )
        counts = tuple(
            list(leaders)
            + medium
            + [2] * two_token_experts
            + [1] * (positive_cold - two_token_experts)
            + [0] * zero_experts
        )
        return Profile(
            name=(
                f"olmoe_grid_{archetype}_a{active_experts}_"
                f"le2_{experts_le_2}"
            ),
            family=f"{archetype}_many_cold",
            evidence_kind="systematic_feature_grid_profile",
            counts=counts,
            description=(
                "Systematic exact-total profile spanning the reported active, "
                "cold, and local-hot regimes; it is a stress construction, "
                "not an observed router window."
            ),
            target_constraints={
                "active_experts": active_experts,
                "experts_le_2": experts_le_2,
                "zero_experts": zero_experts,
                "top1": leaders[0],
                "top2": leaders[1],
            },
        )
    return None


def _constrained_min2_profile(
    *,
    active_experts: int,
    experts_le_2: int,
    leaders: tuple[int, ...],
    archetype: str,
) -> Profile | None:
    """Construct an exact-total profile whose minimum positive load is two.

    Zero-load experts remain legal.  Every active cold expert is fixed to two
    tokens and every non-leader/non-cold expert is at least three tokens.  A
    ``None`` result is an arithmetic infeasibility under the requested active,
    cold, leader, and total-assignment constraints.
    """
    zero_experts = TOTAL_EXPERTS - active_experts
    two_token_experts = experts_le_2 - zero_experts
    medium_experts = active_experts - len(leaders) - two_token_experts
    if two_token_experts <= 0 or medium_experts < 0:
        return None

    medium_sum = (
        TOTAL_ASSIGNMENTS - sum(leaders) - 2 * two_token_experts
    )
    if medium_experts == 0:
        if medium_sum != 0:
            return None
        medium: list[int] = []
    else:
        quotient, residue = divmod(medium_sum, medium_experts)
        medium_cap = leaders[-1] - 1
        if quotient < 3 or quotient + int(residue > 0) > medium_cap:
            return None
        medium = [quotient + 1] * residue + [quotient] * (
            medium_experts - residue
        )

    counts = tuple(
        list(leaders)
        + medium
        + [2] * two_token_experts
        + [0] * zero_experts
    )
    return Profile(
        name=(
            f"olmoe_min2_grid_{archetype}_a{active_experts}_"
            f"le2_{experts_le_2}"
        ),
        family=f"min2_{archetype}_many_cold",
        evidence_kind="systematic_min2_feature_grid_profile",
        counts=counts,
        description=(
            "Systematic exact-total minimum-positive-two profile.  It is a "
            "feature-controlled stress construction, not an observed router "
            "window."
        ),
        target_constraints={
            "active_experts": active_experts,
            "experts_le_2": experts_le_2,
            "zero_experts": zero_experts,
            "top1": leaders[0],
            "top2": leaders[1],
            "minimum_positive_load": 2,
        },
    )


SYSTEMATIC_POPULATION_POINTS = (
    (29, 49),
    (33, 46),
    (38, 44),
    (43, 42),
)

SYSTEMATIC_ARCHETYPES = (
    # top1 / mean = 7.31x.  The observed 16/15 window is in this band.
    ("hot6_8x_dual", (16, 15)),
    ("hot6_8x_triple", (16, 15, 13)),
    ("hot6_8x_quad", (16, 15, 14, 12)),
    # top1 / mean = 10.06x, matching the reported median top1=22.
    ("hot8_12x_dual", (22, 17)),
    ("hot8_12x_triple", (22, 18, 14)),
    ("hot8_12x_quad", (22, 18, 15, 12)),
    # top1 / mean = 13.71x.
    ("hot12_14x_dual", (30, 18)),
    ("hot12_14x_triple", (30, 18, 14)),
    ("hot12_14x_quad", (30, 18, 14, 12)),
)


def _systematic_profiles() -> tuple[Profile, ...]:
    profiles = []
    seen = {profile.counts for profile in BASE_PROFILES}
    for active_experts, experts_le_2 in SYSTEMATIC_POPULATION_POINTS:
        for archetype, leaders in SYSTEMATIC_ARCHETYPES:
            profile = _constrained_profile(
                active_experts=active_experts,
                experts_le_2=experts_le_2,
                leaders=leaders,
                archetype=archetype,
            )
            if profile is None or profile.counts in seen:
                continue
            seen.add(profile.counts)
            profiles.append(profile)
    return tuple(profiles)


def _systematic_min2_profiles() -> tuple[Profile, ...]:
    profiles = []
    seen: set[tuple[int, ...]] = set()
    for active_experts, experts_le_2 in SYSTEMATIC_POPULATION_POINTS:
        for archetype, leaders in SYSTEMATIC_ARCHETYPES:
            profile = _constrained_min2_profile(
                active_experts=active_experts,
                experts_le_2=experts_le_2,
                leaders=leaders,
                archetype=archetype,
            )
            if profile is None or profile.counts in seen:
                continue
            seen.add(profile.counts)
            profiles.append(profile)
    return tuple(profiles)


PROFILES = BASE_PROFILES + _systematic_profiles()
MIN2_PROFILES = _systematic_min2_profiles()


def _hotness_band(top1_to_mean: float) -> str:
    if top1_to_mean < 6.0:
        return "below_6x_control"
    if top1_to_mean < 8.0:
        return "6_to_8x"
    if top1_to_mean < 12.0:
        return "8_to_12x"
    if top1_to_mean <= 14.0:
        return "12_to_14x"
    return "above_14x_tail"


def _metrics(counts: tuple[int, ...]) -> dict[str, int | float | str]:
    top1_to_mean = counts[0] / MEAN_ASSIGNMENTS_PER_EXPERT
    top2_to_mean = counts[1] / MEAN_ASSIGNMENTS_PER_EXPERT
    return {
        "total_experts": len(counts),
        "routed_assignments": sum(counts),
        "active_experts": sum(value > 0 for value in counts),
        "zero_experts": sum(value == 0 for value in counts),
        "experts_le_2": sum(value <= 2 for value in counts),
        "top1": counts[0],
        "top2": counts[1],
        "mean_assignments_per_expert": MEAN_ASSIGNMENTS_PER_EXPERT,
        "top1_to_mean": round(top1_to_mean, 6),
        "top2_to_mean": round(top2_to_mean, 6),
        "top1_hotness_band": _hotness_band(top1_to_mean),
        "local_hot_load_threshold": LOCAL_HOT_LOAD_THRESHOLD,
        "local_hotspot_count": sum(
            value >= LOCAL_HOT_LOAD_THRESHOLD for value in counts
        ),
        "minimum_positive_load": min(value for value in counts if value > 0),
    }


def _validate(profile: Profile) -> dict[str, int | float | str]:
    counts = profile.counts
    if len(counts) != TOTAL_EXPERTS:
        raise ValueError(f"{profile.name}: expected 64 experts, got {len(counts)}")
    if tuple(sorted(counts, reverse=True)) != counts:
        raise ValueError(f"{profile.name}: counts are not sorted")
    if any(value < 0 for value in counts):
        raise ValueError(f"{profile.name}: negative expert load")
    if sum(counts) != TOTAL_ASSIGNMENTS:
        raise ValueError(
            f"{profile.name}: expected {TOTAL_ASSIGNMENTS} assignments, "
            f"got {sum(counts)}"
        )
    metrics = _metrics(counts)
    for key, expected in profile.target_constraints.items():
        if isinstance(expected, int) and metrics[key] != expected:
            raise ValueError(
                f"{profile.name}: {key}={metrics[key]} does not match {expected}"
            )
    return metrics


def build_payload() -> dict[str, object]:
    rows = []
    for profile in PROFILES:
        metrics = _validate(profile)
        rows.append(
            {
                "name": profile.name,
                "tier": "olmoe_top2_projection_directed",
                "family": profile.family,
                "origin": "olmoe_top8_router_top2_projection",
                "profile": profile.evidence_kind,
                "batch_tokens": TOKENS,
                "counts_64": list(profile.counts),
                "active_counts": [value for value in profile.counts if value > 0],
                "metrics": metrics,
                "suite_role": (
                    "measured_anchor"
                    if profile.name == "olmoe_observed_ranked_window_001"
                    else "uniform_control"
                    if profile.name == "uniform_top2_occupancy_control"
                    else "stratified_core"
                    if profile.evidence_kind == "systematic_feature_grid_profile"
                    else "reported_statistic_anchor"
                    if profile.evidence_kind == "synthetic_joint_constraint_profile"
                    else "supplemental_stress"
                ),
                "description": profile.description,
                "target_constraints": profile.target_constraints,
            }
        )
    core_rows = [row for row in rows if row["suite_role"] == "stratified_core"]
    core_coverage = Counter(
        (
            row["metrics"]["top1_hotness_band"],
            row["metrics"]["local_hotspot_count"],
        )
        for row in core_rows
    )
    required_cells = {
        (band, multiplicity)
        for band in ("6_to_8x", "8_to_12x", "12_to_14x")
        for multiplicity in (2, 3, 4)
    }
    missing_cells = sorted(required_cells - set(core_coverage))
    if missing_cells:
        raise ValueError(f"stratified grid is missing cells: {missing_cells}")

    return {
        "schema": "olmoe_top8_router_top2_projection_cases_v3",
        "source_contract": {
            "router_training_topk": 8,
            "projection_topk": TOPK_PROJECTION,
            "tokens": TOKENS,
            "total_experts": TOTAL_EXPERTS,
            "total_assignments": TOTAL_ASSIGNMENTS,
            "mean_assignments_per_expert": MEAN_ASSIGNMENTS_PER_EXPERT,
            "local_hot_load_threshold": LOCAL_HOT_LOAD_THRESHOLD,
            "local_hot_threshold_to_mean": round(
                LOCAL_HOT_LOAD_THRESHOLD / MEAN_ASSIGNMENTS_PER_EXPERT, 6
            ),
            "disclaimer": DISCLAIMER,
        },
        "stratification_contract": {
            "primary_hotness_measure": "top1 / (total_assignments / 64)",
            "hotness_bands": ["6_to_8x", "8_to_12x", "12_to_14x"],
            "local_hotspot_definition": "expert load >= 12 tokens",
            "local_hotspot_multiplicities": [2, 3, 4],
            "active_expert_points": [29, 33, 38, 43],
            "experts_le_2_points": [49, 46, 44, 42],
            "coverage_by_hotness_and_multiplicity": {
                f"{band}|{multiplicity}": core_coverage[(band, multiplicity)]
                for band, multiplicity in sorted(required_cells)
            },
            "note": (
                "Grid profiles are deterministic feature-controlled stress cases, "
                "not additional measured router windows."
            ),
        },
        "reported_window_statistics": {
            "top1_median": 22,
            "top1_iqr": [15, 34],
            "top2_median": 12,
            "top2_iqr": [10, 17],
            "experts_le_2_median": 46,
            "experts_le_2_iqr": [44, 49],
            "zero_experts_median": 31,
            "zero_experts_iqr": [26, 35],
            "fraction_top1_ge_16_approx": 0.72,
            "fraction_top2_ge_16_approx": 0.30,
        },
        "cases": rows,
    }


def build_min2_payload() -> dict[str, object]:
    rows = []
    for profile in MIN2_PROFILES:
        metrics = _validate(profile)
        if metrics["minimum_positive_load"] != 2:
            raise ValueError(f"{profile.name}: minimum positive load is not two")
        rows.append(
            {
                "name": profile.name,
                "tier": "olmoe_top2_projection_directed",
                "family": profile.family,
                "origin": "olmoe_top8_router_top2_projection",
                "profile": profile.evidence_kind,
                "batch_tokens": TOKENS,
                "counts_64": list(profile.counts),
                "active_counts": [
                    value for value in profile.counts if value > 0
                ],
                "metrics": metrics,
                "suite_role": "stratified_min2_supplement",
                "description": profile.description,
                "target_constraints": profile.target_constraints,
            }
        )

    coverage = Counter(
        (
            row["metrics"]["top1_hotness_band"],
            row["metrics"]["local_hotspot_count"],
        )
        for row in rows
    )
    required_cells = {
        (band, multiplicity)
        for band in ("6_to_8x", "8_to_12x", "12_to_14x")
        for multiplicity in (2, 3, 4)
    }
    missing_cells = sorted(required_cells - set(coverage))
    if missing_cells:
        raise ValueError(f"minimum-positive-two grid is missing: {missing_cells}")

    return {
        "schema": "olmoe_top8_router_top2_projection_min2_supplement_v1",
        "source_contract": {
            "router_training_topk": 8,
            "projection_topk": TOPK_PROJECTION,
            "tokens": TOKENS,
            "total_experts": TOTAL_EXPERTS,
            "total_assignments": TOTAL_ASSIGNMENTS,
            "mean_assignments_per_expert": MEAN_ASSIGNMENTS_PER_EXPERT,
            "minimum_positive_load": 2,
            "zeros_allowed": True,
            "disclaimer": DISCLAIMER,
        },
        "stratification_contract": {
            "hotness_bands": ["6_to_8x", "8_to_12x", "12_to_14x"],
            "local_hotspot_definition": "expert load >= 12 tokens",
            "local_hotspot_multiplicities": [2, 3, 4],
            "requested_population_points": [
                list(point) for point in SYSTEMATIC_POPULATION_POINTS
            ],
            "coverage_by_hotness_and_multiplicity": {
                f"{band}|{multiplicity}": coverage[(band, multiplicity)]
                for band, multiplicity in sorted(required_cells)
            },
            "note": (
                "Only arithmetically feasible combinations are emitted; "
                "all active experts have at least two tokens."
            ),
        },
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=("base", "min2"),
        default="base",
    )
    parser.add_argument(
        "--output",
        type=Path,
    )
    args = parser.parse_args()
    if args.suite == "base":
        payload = build_payload()
        output = args.output or Path(
            "results/policy_search/olmoe_top2_projection_cases_v3.json"
        )
    else:
        payload = build_min2_payload()
        output = args.output or Path(
            "results/policy_search/olmoe_top2_projection_min2_supplement_v1.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(payload["cases"]), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
