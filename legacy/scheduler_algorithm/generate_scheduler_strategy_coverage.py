#!/usr/bin/env python3
"""Generate deterministic, feature-stratified MoE scheduler inputs.

The suite is intended for strategy discovery and validation, not as a claim
about the probability of distributions produced by a particular router.  It
combines directed timing/cache boundaries with constrained-random legal top-2
distributions, removes timing-equivalent duplicates, and records measured
features for later range-based policy analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
SEED = 20260711
TOPK = 2
TOTAL_EXPERTS = (8, 32, 64)
N_CASES_PER_E = 10_000
MAX_M_TOTAL = 256
DIRECTED_CASES_PER_E = 2_000
CORNER_CASES_PER_E = 24

SHAPE_C_M_DIM = 2
SHAPE_C_T_S1 = 22_528
SHAPE_C_T_S3 = 11_264
N_CLUSTERS = 2

# M_total boundaries exercise short workloads and changes around powers of two.
# Every integer M_total in 1..256 is additionally covered by the stratified set.
CRITICAL_M_TOTALS = (
    1, 2, 3, 4, 5, 7, 8, 9,
    15, 16, 17, 31, 32, 33,
    47, 48, 49, 63, 64, 65,
    95, 96, 97, 127, 128, 129,
    191, 192, 193, 255, 256,
)
CRITICAL_NTOK = (
    1, 2, 3, 4, 5, 7, 8, 9,
    15, 16, 17, 31, 32, 33,
    63, 64, 65, 127, 128, 129, 255, 256,
)

CONSTRUCTIONS = (
    "balanced",
    "balanced_jitter",
    "head_heavy",
    "two_hot",
    "zipf",
    "boundary_mix",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_only_ideal_cc(assignment_total: int) -> int:
    per_assignment_cc = (SHAPE_C_T_S1 + SHAPE_C_T_S3) / SHAPE_C_M_DIM
    return math.ceil(assignment_total * per_assignment_cc / N_CLUSTERS)


def m_band(m_total: int) -> str:
    if m_total <= 8:
        return "tiny_1_8"
    if m_total <= 32:
        return "small_9_32"
    if m_total <= 128:
        return "medium_33_128"
    return "large_129_256"


def density_bin(active_n: int, e_total: int, m_total: int) -> str:
    feasible = min(e_total, TOPK * m_total)
    density = active_n / feasible
    if density <= 0.25:
        return "sparse_le_25pct"
    if density <= 0.50:
        return "medium_25_50pct"
    if density <= 0.75:
        return "dense_50_75pct"
    return "full_gt_75pct"


def imbalance_bin(ratio: float) -> str:
    if ratio <= 1.25 + 1e-12:
        return "balanced_le_1p25"
    if ratio <= 2.0 + 1e-12:
        return "mild_1p25_2"
    if ratio <= 4.0 + 1e-12:
        return "medium_2_4"
    return "heavy_gt_4"


def tail_bin(tokens: list[int]) -> str:
    odd_fraction = sum(token & 1 for token in tokens) / len(tokens)
    if all(token % 8 == 0 for token in tokens):
        return "all_m8_aligned"
    if odd_fraction == 0:
        return "all_even"
    if odd_fraction <= 0.25:
        return "odd_light"
    if odd_fraction <= 0.75:
        return "odd_mixed"
    return "odd_heavy"


def gini(values: list[int]) -> float:
    ordered = sorted(values)
    total = sum(ordered)
    n = len(ordered)
    if total == 0:
        return 0.0
    weighted = sum((i + 1) * value for i, value in enumerate(ordered))
    return (2 * weighted) / (n * total) - (n + 1) / n


def distribution_features(
    tokens: list[int], e_total: int, m_total: int
) -> dict:
    total = sum(tokens)
    active_n = len(tokens)
    mean = total / active_n
    variance = sum((token - mean) ** 2 for token in tokens) / active_n
    probs = [token / total for token in tokens]
    entropy = -sum(prob * math.log(prob) for prob in probs)
    normalized_entropy = entropy / math.log(active_n) if active_n > 1 else 0.0
    top = sorted(tokens, reverse=True)
    ratio = top[0] / mean
    feasible_active = min(e_total, TOPK * m_total)
    return {
        "m_band": m_band(m_total),
        "active_density": active_n / feasible_active,
        "active_density_bin": density_bin(active_n, e_total, m_total),
        "max_ntok": top[0],
        "mean_ntok": mean,
        "max_to_mean": ratio,
        "imbalance_bin": imbalance_bin(ratio),
        "top1_share": top[0] / total,
        "top2_share": sum(top[:2]) / total,
        "top1_to_top2": top[0] / top[1] if active_n > 1 else 1.0,
        "coefficient_of_variation": math.sqrt(variance) / mean,
        "gini": gini(tokens),
        "normalized_entropy": normalized_entropy,
        "tail_bin": tail_bin(tokens),
        "odd_experts": sum(token & 1 for token in tokens),
        "mod2_hist": [sum(token % 2 == value for token in tokens) for value in range(2)],
        "mod4_hist": [sum(token % 4 == value for token in tokens) for value in range(4)],
        "mod8_hist": [sum(token % 8 == value for token in tokens) for value in range(8)],
    }


def balanced_tokens(active_n: int, target: int) -> list[int]:
    quotient, remainder = divmod(target, active_n)
    return [quotient + int(i < remainder) for i in range(active_n)]


def add_weighted(
    tokens: list[int], target: int, cap: int, weights: list[float], rng: random.Random
) -> list[int]:
    remaining = target - sum(tokens)
    while remaining > 0:
        available = [i for i, token in enumerate(tokens) if token < cap]
        if not available:
            raise ValueError("insufficient token capacity")
        total_weight = sum(weights[i] for i in available)
        pick = rng.choices(available, [weights[i] / total_weight for i in available])[0]
        tokens[pick] += 1
        remaining -= 1
    return tokens


def reduce_to_target(tokens: list[int], target: int) -> list[int]:
    while sum(tokens) > target:
        index = max(range(len(tokens)), key=lambda i: (tokens[i], -i))
        if tokens[index] <= 1:
            raise ValueError("cannot reduce a positive composition")
        tokens[index] -= 1
    return tokens


def allocate_tokens(
    active_n: int, m_total: int, construction: str, rng: random.Random
) -> list[int]:
    if active_n == 1:
        return [m_total]
    target = TOPK * m_total
    if not 2 <= active_n <= min(target, 64):
        raise ValueError("active_n violates top-2 routing capacity")

    if construction == "balanced":
        tokens = balanced_tokens(active_n, target)
    elif construction == "balanced_jitter":
        weights = [rng.uniform(0.75, 1.25) for _ in range(active_n)]
        tokens = add_weighted([1] * active_n, target, m_total, weights, rng)
    elif construction == "head_heavy":
        weights = [60.0] + [1.0] * (active_n - 1)
        tokens = add_weighted([1] * active_n, target, m_total, weights, rng)
    elif construction == "two_hot":
        weights = [40.0, 28.0] + [1.0] * (active_n - 2)
        tokens = add_weighted([1] * active_n, target, m_total, weights, rng)
    elif construction == "zipf":
        alpha = rng.uniform(0.7, 2.2)
        weights = [1.0 / ((i + 1) ** alpha) for i in range(active_n)]
        tokens = add_weighted([1] * active_n, target, m_total, weights, rng)
    elif construction == "boundary_mix":
        offset = rng.randrange(len(CRITICAL_NTOK))
        tokens = [
            min(m_total, CRITICAL_NTOK[(offset + i) % len(CRITICAL_NTOK)])
            for i in range(active_n)
        ]
        tokens = [max(1, token) for token in tokens]
        if sum(tokens) > target:
            reduce_to_target(tokens, target)
        elif sum(tokens) < target:
            weights = [1.0 / ((i + 1) ** 1.3) for i in range(active_n)]
            add_weighted(tokens, target, m_total, weights, rng)
    else:
        raise ValueError(construction)

    tokens.sort(reverse=True)
    if len(tokens) != active_n or min(tokens) < 1 or max(tokens) > m_total:
        raise AssertionError("invalid expert token allocation")
    if sum(tokens) != target:
        raise AssertionError("top-2 assignment total mismatch")
    return tokens


def assign_eids(e_total: int, tokens: list[int], rng: random.Random) -> list[tuple[int, int]]:
    eids = rng.sample(range(e_total), len(tokens))
    pairs = sorted(zip(eids, tokens), key=lambda item: (-item[1], item[0]))
    return pairs


def applicable_cache_regimes(active_n: int, e_total: int) -> tuple[str, ...]:
    regimes = ["none", "top1_one", "tail_one", "same_top1"]
    if active_n >= 2:
        regimes.extend(("top2_one", "top1_top2", "random_active_two"))
    if active_n < e_total:
        regimes.extend(("stale_one", "top1_stale"))
    return tuple(regimes)


def choose_cache(
    e_total: int,
    experts: list[tuple[int, int]],
    regime: str,
    rng: random.Random,
) -> tuple[int, int]:
    active = [eid for eid, _ in experts]
    inactive = [eid for eid in range(e_total) if eid not in set(active)]
    if regime == "none":
        return -1, -1
    if regime == "top1_one":
        return active[0], -1
    if regime == "top2_one":
        return active[1], -1
    if regime == "top1_top2":
        return active[0], active[1]
    if regime == "tail_one":
        return active[-1], -1
    if regime == "same_top1":
        return active[0], active[0]
    if regime == "random_active_two":
        c2, c3 = rng.sample(active, 2)
        return c2, c3
    if regime == "stale_one":
        return rng.choice(inactive), -1
    if regime == "top1_stale":
        return active[0], rng.choice(inactive)
    raise ValueError(f"inapplicable cache regime {regime!r}")


def normalized_signature(
    tokens: list[int], experts: list[tuple[int, int]], c2: int, c3: int
) -> tuple:
    rank = {eid: index for index, (eid, _) in enumerate(experts)}

    def cache_rank(eid: int) -> int:
        if eid < 0:
            return -1
        return rank.get(eid, -2)

    # Clusters are physically symmetric in the reference model.
    cache = tuple(sorted((cache_rank(c2), cache_rank(c3))))
    return tuple(tokens), cache


def active_anchors(e_total: int, m_total: int) -> list[int]:
    maximum = min(e_total, TOPK * m_total)
    if maximum < 2:
        return []
    values = {
        2,
        min(maximum, 3),
        min(maximum, 4),
        max(2, round(maximum * 0.25)),
        max(2, round(maximum * 0.50)),
        max(2, round(maximum * 0.75)),
        maximum,
    }
    return sorted(value for value in values if 2 <= value <= maximum)


def make_case(
    e_total: int,
    active_n: int,
    m_total: int,
    construction: str,
    cache_regime: str,
    sample_class: str,
    rng: random.Random,
) -> tuple[dict, tuple]:
    tokens = allocate_tokens(active_n, m_total, construction, rng)
    experts = assign_eids(e_total, tokens, rng)
    c2, c3 = choose_cache(e_total, experts, cache_regime, rng)
    assignment_total = sum(tokens)
    router_valid = (
        active_n >= TOPK
        and assignment_total == m_total * TOPK
        and max(tokens) <= m_total
    )
    features = distribution_features(tokens, e_total, m_total)
    case = {
        "e_total": e_total,
        "active_n": active_n,
        "router_valid": router_valid,
        "analysis_eligible": router_valid,
        "topk": TOPK,
        "m_total": m_total,
        "assignment_total": assignment_total,
        "compute_only_ideal_cc": compute_only_ideal_cc(assignment_total),
        "sample_class": sample_class,
        "construction": construction,
        "cache_regime": cache_regime,
        "features": features,
        "experts": [{"eid": eid, "ntokens": ntok} for eid, ntok in experts],
        "dist": {str(eid): ntok for eid, ntok in experts},
        "c2": c2,
        "c3": c3,
    }
    return case, normalized_signature(tokens, experts, c2, c3)


def assign_dataset_splits(cases: list[dict], rng: random.Random) -> None:
    """Assign exact 60/20/20 splits while covering every M and active_n."""
    labels = ("discovery", "validation", "blind_test")
    target = {
        "discovery": 3 * len(cases) // 5,
        "validation": len(cases) // 5,
        "blind_test": len(cases) // 5,
    }
    by_m = defaultdict(list)
    for case in cases:
        by_m[case["m_total"]].append(case)

    # Every M_total has at least seven cases, so each split receives one before
    # proportional assignment of the remainder.
    for group in by_m.values():
        rng.shuffle(group)
        n_validation = max(1, round(len(group) * 0.20))
        n_blind = max(1, round(len(group) * 0.20))
        if n_validation + n_blind >= len(group):
            n_validation = n_blind = 1
        for index, case in enumerate(group):
            if index < n_validation:
                case["dataset_split"] = "validation"
            elif index < n_validation + n_blind:
                case["dataset_split"] = "blind_test"
            else:
                case["dataset_split"] = "discovery"

    counts = Counter(case["dataset_split"] for case in cases)
    by_m_split = Counter((case["m_total"], case["dataset_split"]) for case in cases)
    by_active_split = Counter(
        (case["active_n"], case["dataset_split"])
        for case in cases
        if case["analysis_eligible"]
    )
    while counts != Counter(target):
        source = next(label for label in labels if counts[label] > target[label])
        destination = next(label for label in labels if counts[label] < target[label])
        movable = [
            case
            for case in cases
            if case["dataset_split"] == source
            and by_m_split[(case["m_total"], source)] > 1
            and (
                not case["analysis_eligible"]
                or by_active_split[(case["active_n"], source)] > 1
            )
        ]
        if not movable:
            raise RuntimeError("cannot rebalance dataset splits without losing coverage")
        case = rng.choice(movable)
        by_m_split[(case["m_total"], source)] -= 1
        by_m_split[(case["m_total"], destination)] += 1
        if case["analysis_eligible"]:
            by_active_split[(case["active_n"], source)] -= 1
            by_active_split[(case["active_n"], destination)] += 1
        case["dataset_split"] = destination
        counts[source] -= 1
        counts[destination] += 1


def generate_cases(
    e_total: int, rng: random.Random, split_label: str | None = None
) -> list[dict]:
    cases: list[dict] = []
    seen = set()
    construction_counts = Counter()
    cache_counts = Counter()
    active_counts = Counter()
    m_counts = Counter()

    def add(
        active_n: int,
        m_total: int,
        construction: str,
        cache_regime: str,
        sample_class: str,
    ) -> bool:
        case, signature = make_case(
            e_total,
            active_n,
            m_total,
            construction,
            cache_regime,
            sample_class,
            rng,
        )
        if signature in seen:
            return False
        seen.add(signature)
        cases.append(case)
        construction_counts[construction] += 1
        cache_counts[cache_regime] += 1
        active_counts[active_n] += 1
        m_counts[m_total] += 1
        return True

    # Keep the scheduler's n==1 branch as a small, explicitly ineligible unit set.
    corner_ms = list(CRITICAL_M_TOTALS)
    for i in range(CORNER_CASES_PER_E):
        m_total = corner_ms[i % len(corner_ms)]
        regime = "none" if i % 2 == 0 else "same_top1"
        add(1, m_total, "balanced", regime, "router_invalid_corner")

    # Directed crosses target stage/tail boundaries, density anchors, and cache states.
    directed_attempt = 0
    while len(cases) < DIRECTED_CASES_PER_E:
        m_total = CRITICAL_M_TOTALS[directed_attempt % len(CRITICAL_M_TOTALS)]
        anchors = active_anchors(e_total, m_total)
        active_n = anchors[(directed_attempt // len(CRITICAL_M_TOTALS)) % len(anchors)]
        construction = CONSTRUCTIONS[
            (directed_attempt * 5 + active_n) % len(CONSTRUCTIONS)
        ]
        regimes = applicable_cache_regimes(active_n, e_total)
        cache_regime = regimes[
            (directed_attempt * 7 + m_total) % len(regimes)
        ]
        add(active_n, m_total, construction, cache_regime, "directed_boundary")
        directed_attempt += 1
        if directed_attempt > 200_000:
            raise RuntimeError("could not construct unique directed cases")

    # Ensure every legal active_n is explicitly present in the policy-analysis set.
    for active_n in range(2, e_total + 1):
        if active_counts[active_n]:
            continue
        m_total = max(1, math.ceil(active_n / TOPK))
        add(active_n, m_total, "balanced", "none", "directed_boundary")

    # Explicitly hit every M_total before random fill.  Do not steer generation by
    # successful-case counters: deterministic constructions have fewer unique
    # outcomes and such feedback can repeatedly select an exhausted cell.
    for m_total in range(1, MAX_M_TOTAL + 1):
        if m_counts[m_total]:
            continue
        active_n = rng.choice(active_anchors(e_total, m_total))
        construction = rng.choice(CONSTRUCTIONS)
        regimes = applicable_cache_regimes(active_n, e_total)
        for _ in range(100):
            if add(
                active_n,
                m_total,
                construction,
                rng.choice(regimes),
                "directed_boundary",
            ):
                break
        else:
            raise RuntimeError(f"could not cover M_total={m_total}")

    # Fill with constrained-random cases.  Independent randomized choices avoid
    # coupling M_total, construction, and cache regime through one ordinal.
    # Actual measured bins are saved and audited after generation.
    attempts = 0
    while len(cases) < N_CASES_PER_E:
        attempts += 1
        if attempts > 1_000_000:
            raise RuntimeError("could not construct enough unique strategy cases")
        if rng.random() < 0.35:
            m_total = rng.choice(CRITICAL_M_TOTALS)
        else:
            m_total = rng.randint(1, MAX_M_TOTAL)
        anchors = active_anchors(e_total, m_total)
        if rng.random() < 0.70:
            active_n = rng.choice(anchors)
        else:
            active_n = rng.randint(2, min(e_total, TOPK * m_total))
        construction = rng.choice(CONSTRUCTIONS)
        regimes = applicable_cache_regimes(active_n, e_total)
        cache_regime = rng.choice(regimes)
        add(
            active_n,
            m_total,
            construction,
            cache_regime,
            "stratified_constrained_random",
        )

    if split_label is None:
        assign_dataset_splits(cases, rng)
    else:
        for case in cases:
            case["dataset_split"] = split_label
    rng.shuffle(cases)
    for case_id, case in enumerate(cases):
        case["case_id"] = case_id
    return cases


def nested_counts(cases: list[dict], *keys: str) -> dict:
    counts = Counter()
    for case in cases:
        values = []
        for key in keys:
            value = case
            for part in key.split("."):
                value = value[part]
            values.append(str(value))
        counts[tuple(values)] += 1
    return {"|".join(key): value for key, value in sorted(counts.items())}


def summarize(cases: list[dict]) -> dict:
    eligible = [case for case in cases if case["analysis_eligible"]]
    factor_keys = (
        "features.m_band",
        "features.active_density_bin",
        "features.imbalance_bin",
        "features.tail_bin",
        "cache_regime",
    )
    pairwise = {}
    for i, left in enumerate(factor_keys):
        for right in factor_keys[i + 1:]:
            pairwise[f"{left}__x__{right}"] = nested_counts(eligible, left, right)
    return {
        "n_cases": len(cases),
        "analysis_eligible_cases": len(eligible),
        "router_invalid_corner_cases": len(cases) - len(eligible),
        "dataset_split_counts": nested_counts(cases, "dataset_split"),
        "sample_class_counts": nested_counts(cases, "sample_class"),
        "active_n_counts": nested_counts(eligible, "active_n"),
        "m_total_counts": nested_counts(eligible, "m_total"),
        "construction_counts": nested_counts(eligible, "construction"),
        "cache_regime_counts": nested_counts(eligible, "cache_regime"),
        "m_band_counts": nested_counts(eligible, "features.m_band"),
        "active_density_bin_counts": nested_counts(
            eligible, "features.active_density_bin"
        ),
        "imbalance_bin_counts": nested_counts(eligible, "features.imbalance_bin"),
        "tail_bin_counts": nested_counts(eligible, "features.tail_bin"),
        "pairwise_factor_counts": pairwise,
    }


def validate_cases(cases: list[dict], e_total: int) -> None:
    if len(cases) != N_CASES_PER_E:
        raise AssertionError("wrong case count")
    if {case["active_n"] for case in cases if case["analysis_eligible"]} != set(
        range(2, e_total + 1)
    ):
        raise AssertionError("legal active_n coverage is incomplete")
    if {case["m_total"] for case in cases if case["analysis_eligible"]} != set(
        range(1, MAX_M_TOTAL + 1)
    ):
        raise AssertionError("M_total coverage is incomplete")
    signatures = set()
    for case in cases:
        tokens = [expert["ntokens"] for expert in case["experts"]]
        if case["analysis_eligible"]:
            if sum(tokens) != TOPK * case["m_total"]:
                raise AssertionError("assignment total mismatch")
            if max(tokens) > case["m_total"] or min(tokens) < 1:
                raise AssertionError("expert token cap violation")
        experts = [(expert["eid"], expert["ntokens"]) for expert in case["experts"]]
        signature = normalized_signature(tokens, experts, case["c2"], case["c3"])
        if signature in signatures:
            raise AssertionError("normalized duplicate input")
        signatures.add(signature)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cases-per-e", type=int, default=N_CASES_PER_E)
    parser.add_argument(
        "--directed-cases-per-e", type=int, default=DIRECTED_CASES_PER_E
    )
    parser.add_argument("--corner-cases-per-e", type=int, default=CORNER_CASES_PER_E)
    parser.add_argument("--e-total", nargs="+", type=int, default=list(TOTAL_EXPERTS))
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--prefix", default="scheduler_strategy_coverage")
    parser.add_argument(
        "--split-label",
        help="override generated discovery/validation/blind labels after balancing",
    )
    parser.add_argument(
        "--policy-freeze-manifest",
        type=Path,
        help="bind generated inputs to an already frozen policy manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global SEED, N_CASES_PER_E, DIRECTED_CASES_PER_E, CORNER_CASES_PER_E
    SEED = args.seed
    N_CASES_PER_E = args.cases_per_e
    DIRECTED_CASES_PER_E = args.directed_cases_per_e
    CORNER_CASES_PER_E = args.corner_cases_per_e
    if N_CASES_PER_E <= 0:
        raise ValueError("--cases-per-e must be positive")
    if not 0 <= CORNER_CASES_PER_E <= DIRECTED_CASES_PER_E <= N_CASES_PER_E:
        raise ValueError("require corner <= directed <= total cases per E")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    freeze_binding = None
    if args.policy_freeze_manifest is not None:
        if not args.policy_freeze_manifest.is_file():
            raise FileNotFoundError(args.policy_freeze_manifest)
        freeze_binding = {
            "path": str(args.policy_freeze_manifest.resolve()),
            "sha256": file_sha256(args.policy_freeze_manifest),
        }
    root_rng = random.Random(SEED)
    manifest = {
        "name": args.prefix,
        "seed": SEED,
        "topk": TOPK,
        "n_cases_per_e": N_CASES_PER_E,
        "directed_cases_per_e": DIRECTED_CASES_PER_E,
        "corner_cases_per_e": CORNER_CASES_PER_E,
        "e_total": list(args.e_total),
        "split_label": args.split_label,
        "policy_freeze": freeze_binding,
        "purpose": (
            "Feature-stratified strategy discovery and validation. This suite is "
            "coverage-balanced and must not be interpreted as a measured router "
            "probability distribution."
        ),
        "files": [],
        "file_sha256": {},
    }
    for e_total in args.e_total:
        rng = random.Random(root_rng.randrange(1 << 60))
        cases = generate_cases(e_total, rng, args.split_label)
        validate_cases(cases, e_total)
        payload = {
            "meta": {
                "name": f"{args.prefix}_E{e_total}",
                "seed": SEED,
                "e_total": e_total,
                "topk": TOPK,
                "max_m_total": MAX_M_TOTAL,
                "policy_freeze": freeze_binding,
                "description": (
                    "Directed timing/cache boundaries plus de-duplicated, "
                    "feature-stratified constrained-random top-2 distributions."
                ),
                "compute_only_ideal_formula": (
                    "ceil(assignment_total * ((22528 + 11264) / 2) / 2)"
                ),
                **summarize(cases),
            },
            "cases": cases,
        }
        out_path = out_dir / f"{args.prefix}_E{e_total}.json"
        with out_path.open("w") as handle:
            json.dump(payload, handle, indent=2)
        manifest["files"].append(out_path.name)
        manifest["file_sha256"][out_path.name] = file_sha256(out_path)
        print(f"wrote {out_path.name}")
        print(json.dumps({
            "analysis_eligible": payload["meta"]["analysis_eligible_cases"],
            "invalid_corners": payload["meta"]["router_invalid_corner_cases"],
            "imbalance_bins": payload["meta"]["imbalance_bin_counts"],
            "cache_regimes": payload["meta"]["cache_regime_counts"],
        }, indent=2))

    manifest_path = out_dir / f"{args.prefix}_manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"wrote {manifest_path.name}")


if __name__ == "__main__":
    main()
