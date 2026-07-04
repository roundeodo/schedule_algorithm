#!/usr/bin/env python3
"""
Controlled, representative scheduler input suite.

Design goals:
  1. Fix total expert pool E_total in {8, 32, 64}.
  2. Explicitly cover every active_n from 1..E_total.
  3. For router-valid cases (active_n >= topK), enforce:
       sum(ntokens) = M_total * topK
       1 <= ntokens_i <= M_total
     because one expert cannot receive more than all real input tokens.
  4. Cover token-count decision boundaries, not just random distributions.

active_n=1 is kept as an algorithm corner/unit case because the scheduler has
an n==1 branch, but it is marked router_valid=false for topK=2.
"""

import json
import math
import random
from collections import Counter
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
SEED = 20260623
TOPK = 2
TOTAL_EXPERTS = (8, 32, 64)
N_CASES_PER_E = 10_000

# Compute-only ideal lower bound in the scheduler abstraction.
# ShapeC processes 2 token-expert units in 22,528 cc for SwishGLU gate/up and
# 11,264 cc for down.  With two identical clusters kept fully busy, the ideal
# makespan is total token-expert work divided by two clusters, with no DMA wait,
# dependency bubble, or shape-tail waste.
SHAPE_C_M_DIM = 2
SHAPE_C_T_S1 = 22_528
SHAPE_C_T_S3 = 11_264
N_CLUSTERS = 2

# Values around shape and tiling boundaries.  23/24 are usually less
# interesting than 15/16/17 or 31/32/33 because ceil(/2), M_dim=2/4/8,
# split boundaries, and tail phases change around these values.
CRITICAL_M_TOTALS = (
    1, 2, 3, 4, 5, 7, 8, 9,
    15, 16, 17,
    31, 32, 33,
    47, 48, 49,
    63, 64, 65,
    95, 96, 97,
    127, 128, 129,
    191, 192, 193,
    255, 256,
)

CRITICAL_NTOK = (
    1, 2, 3, 4, 5, 7, 8, 9,
    15, 16, 17,
    31, 32, 33,
    63, 64, 65,
    127, 128, 129,
    255, 256,
)

PROFILES = (
    "all_tiny",
    "uniform",
    "balanced_noise",
    "zipf",
    "single_hot",
    "two_hot",
    "multi_hot",
    "bimodal",
    "shape_boundary",
    "pair_split_trap",
)

CACHE_MODES = (
    "none",
    "top0",
    "top1",
    "top0_top1",
    "top1_top0",
    "random_active",
    "tail_active",
    "inactive",
    "active_inactive",
)


def compute_only_ideal_cc(assignment_total):
    per_token_cc = (SHAPE_C_T_S1 + SHAPE_C_T_S3) / SHAPE_C_M_DIM
    return math.ceil(assignment_total * per_token_cc / N_CLUSTERS)


def choose_m_total(active_n, ordinal):
    if active_n <= 1:
        return CRITICAL_NTOK[ordinal % len(CRITICAL_NTOK)]
    min_m = (active_n + TOPK - 1) // TOPK
    candidates = [m for m in CRITICAL_M_TOTALS if m >= min_m]
    return candidates[ordinal % len(candidates)] if candidates else min_m


def add_weighted(tokens, target_sum, cap, weights, rng):
    n = len(tokens)
    while sum(tokens) < target_sum:
        avail = [i for i in range(n) if tokens[i] < cap]
        if not avail:
            raise ValueError("no capacity left while filling tokens")
        total_w = sum(max(0.0, weights[i]) for i in avail)
        if total_w <= 0:
            pick = rng.choice(avail)
        else:
            r = rng.random() * total_w
            acc = 0.0
            pick = avail[-1]
            for i in avail:
                acc += max(0.0, weights[i])
                if acc >= r:
                    pick = i
                    break
        tokens[pick] += 1
    return tokens


def reduce_to_sum(tokens, target_sum):
    while sum(tokens) > target_sum:
        idx = max(range(len(tokens)), key=lambda i: tokens[i])
        if tokens[idx] <= 1:
            raise ValueError("cannot reduce tokens below 1")
        tokens[idx] -= 1
    return tokens


def allocate_boundary_tokens(active_n, m_total, rng):
    target = m_total * TOPK
    vals = [min(m_total, CRITICAL_NTOK[i % len(CRITICAL_NTOK)]) for i in range(active_n)]
    vals = [max(1, v) for v in vals]
    if sum(vals) > target:
        reduce_to_sum(vals, target)
    elif sum(vals) < target:
        weights = [1.0 / ((i + 1) ** 1.3) for i in range(active_n)]
        add_weighted(vals, target, m_total, weights, rng)
    return vals


def allocate_tokens(active_n, m_total, profile, rng):
    if active_n == 1:
        return [m_total]
    target = m_total * TOPK
    if active_n > target:
        raise ValueError("active_n cannot exceed M_total*topK")
    tokens = [1] * active_n

    if profile == "all_tiny":
        weights = [12.0, 8.0] + [0.15] * max(0, active_n - 2)
    elif profile == "uniform":
        weights = [1.0] * active_n
    elif profile == "balanced_noise":
        weights = [rng.uniform(0.75, 1.25) for _ in range(active_n)]
    elif profile == "zipf":
        alpha = rng.uniform(0.75, 2.3)
        weights = [1.0 / ((i + 1) ** alpha) for i in range(active_n)]
    elif profile == "single_hot":
        weights = [50.0] + [1.0] * (active_n - 1)
    elif profile == "two_hot":
        weights = [35.0, 28.0] + [1.0] * max(0, active_n - 2)
    elif profile == "multi_hot":
        n_hot = min(active_n, rng.choice([3, 4, 6, 8]))
        weights = [rng.uniform(8.0, 20.0) if i < n_hot else 1.0 for i in range(active_n)]
    elif profile == "bimodal":
        head_n = max(1, active_n // rng.choice([4, 6, 8]))
        weights = [8.0 if i < head_n else 1.0 for i in range(active_n)]
    elif profile == "shape_boundary":
        return allocate_boundary_tokens(active_n, m_total, rng)
    elif profile == "pair_split_trap":
        # Forces one or two hot experts plus many small experts, exercising
        # SPLIT(top0), PAIR(topK,topJ), and WAIT/fill behaviour.
        weights = [45.0, 16.0] + [0.6] * max(0, active_n - 2)
    else:
        raise ValueError(profile)

    add_weighted(tokens, target, m_total, weights, rng)
    tokens.sort(reverse=True)
    assert len(tokens) == active_n
    assert min(tokens) >= 1
    assert max(tokens) <= m_total
    assert sum(tokens) == target
    return tokens


def assign_eids(e_total, tokens, rng):
    eids = rng.sample(range(e_total), len(tokens))
    pairs = list(zip(eids, tokens))
    pairs.sort(key=lambda x: (-x[1], x[0]))
    return pairs


def choose_cache(e_total, experts, mode, rng):
    active = [eid for eid, _ in experts]
    inactive = [eid for eid in range(e_total) if eid not in set(active)]
    if mode == "none":
        return -1, -1
    if mode == "top0":
        return active[0], -1
    if mode == "top1" and len(active) >= 2:
        return active[1], -1
    if mode == "top0_top1" and len(active) >= 2:
        return active[0], active[1]
    if mode == "top1_top0" and len(active) >= 2:
        return active[1], active[0]
    if mode == "random_active":
        c2 = rng.choice(active)
        rest = [eid for eid in active if eid != c2]
        c3 = rng.choice(rest) if rest and rng.random() < 0.5 else -1
        return c2, c3
    if mode == "tail_active":
        c2 = active[-1]
        c3 = active[-2] if len(active) >= 2 and rng.random() < 0.5 else -1
        return c2, c3
    if mode == "inactive" and inactive:
        return rng.choice(inactive), -1
    if mode == "active_inactive" and inactive:
        return active[0], rng.choice(inactive)
    return -1, -1


def targets_by_active_n(e_total, n_cases):
    base = n_cases // e_total
    rem = n_cases % e_total
    return {n: base + (1 if n <= rem else 0) for n in range(1, e_total + 1)}


def make_case(case_id, e_total, active_n, profile, cache_mode, ordinal, rng):
    m_total = choose_m_total(active_n, ordinal)
    tokens = allocate_tokens(active_n, m_total, profile, rng)
    experts = assign_eids(e_total, tokens, rng)
    c2, c3 = choose_cache(e_total, experts, cache_mode, rng)
    router_valid = active_n >= TOPK and sum(tokens) == m_total * TOPK and max(tokens) <= m_total
    assignment_total = sum(tokens)
    return {
        "case_id": case_id,
        "e_total": e_total,
        "active_n": active_n,
        "router_valid": router_valid,
        "topk": TOPK,
        "m_total": m_total,
        "assignment_total": assignment_total,
        "compute_only_ideal_cc": compute_only_ideal_cc(assignment_total),
        "profile": profile,
        "cache_mode": cache_mode,
        "experts": [{"eid": eid, "ntokens": ntok} for eid, ntok in experts],
        "dist": {str(eid): ntok for eid, ntok in experts},
        "c2": c2,
        "c3": c3,
    }


def summarize(cases):
    return {
        "active_n_counts": dict(sorted(Counter(c["active_n"] for c in cases).items())),
        "router_valid_counts": dict(sorted(Counter(str(c["router_valid"]) for c in cases).items())),
        "profile_counts": dict(sorted(Counter(c["profile"] for c in cases).items())),
        "cache_mode_counts": dict(sorted(Counter(c["cache_mode"] for c in cases).items())),
        "m_total_counts": dict(sorted(Counter(c["m_total"] for c in cases).items())),
        "max_ntok_min": min(max(e["ntokens"] for e in c["experts"]) for c in cases),
        "max_ntok_max": max(max(e["ntokens"] for e in c["experts"]) for c in cases),
    }


def main():
    root_rng = random.Random(SEED)
    manifest = {
        "name": "scheduler_eval_inputs_stratified_v6",
        "seed": SEED,
        "topk": TOPK,
        "n_cases_per_e": N_CASES_PER_E,
        "files": [],
    }
    for e_total in TOTAL_EXPERTS:
        rng = random.Random(root_rng.randrange(1 << 60))
        targets = targets_by_active_n(e_total, N_CASES_PER_E)
        cases = []
        for active_n, target in targets.items():
            for i in range(target):
                profile = PROFILES[i % len(PROFILES)]
                cache_mode = CACHE_MODES[(i // len(PROFILES)) % len(CACHE_MODES)]
                cases.append(make_case(len(cases), e_total, active_n, profile, cache_mode, i, rng))
        rng.shuffle(cases)
        for i, case in enumerate(cases):
            case["case_id"] = i
        payload = {
            "meta": {
                "name": f"scheduler_eval_inputs_E{e_total}_stratified_v6",
                "seed": SEED,
                "e_total": e_total,
                "topk": TOPK,
                "n_cases": len(cases),
                "description": (
                    "Controlled scheduler inputs. active_n is explicitly covered; "
                    "router-valid cases satisfy sum(ntokens)=M_total*topK and "
                    "ntokens_i<=M_total. active_n=1 is retained as an algorithm "
                    "corner for the scheduler's n==1 branch. compute_only_ideal_cc "
                    "is the two-cluster full-utilization lower bound in this "
                    "scheduler abstraction, excluding DMA wait, dependency bubbles, "
                    "and shape-tail waste."
                ),
                "compute_only_ideal_formula": (
                    "ceil(assignment_total * ((22528 + 11264) / 2) / 2)"
                ),
                **summarize(cases),
            },
            "cases": cases,
        }
        out_path = OUT_DIR / f"scheduler_eval_inputs_E{e_total}_stratified_v6.json"
        with out_path.open("w") as f:
            json.dump(payload, f, indent=2)
        manifest["files"].append(out_path.name)
        print(f"wrote {out_path}")
        print(json.dumps(payload["meta"], indent=2))

    manifest_path = OUT_DIR / "scheduler_eval_inputs_stratified_v6_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
