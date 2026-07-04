#!/usr/bin/env python3
"""
Evaluate the cache-first analytical scheduler against the beam reference.

This script intentionally samples initial cache residency across all active
experts, not only top0/top1.  Initial residency is full-expert residency and
therefore skips both S1 and S3 foreground DMA.  Stage-4 prefetch in the beam
model remains S1-only.
"""

import math
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from analytical_scheduler import analytical_schedule
from four_stage_scheduler import FourStageScheduler


def random_dist():
    m_total = random.choice([4, 8, 12, 16, 24, 32, 48, 64])
    n_experts = random.randint(1, min(8, m_total))
    dist_type = random.choices(
        ["uniform", "zipf", "hot", "bimodal", "single"], weights=[2, 4, 3, 2, 1]
    )[0]
    if dist_type == "single":
        return {0: m_total}
    if dist_type == "uniform":
        base = m_total // n_experts
        toks = [base] * n_experts
        toks[0] += m_total - sum(toks)
    elif dist_type == "zipf":
        alpha = random.uniform(0.6, 2.5)
        weights = [1.0 / (i + 1) ** alpha for i in range(n_experts)]
        total = sum(weights)
        toks = [max(1, round(weight / total * m_total)) for weight in weights]
        toks.sort(reverse=True)
        toks[0] += m_total - sum(toks)
    elif dist_type == "hot":
        hot_frac = random.uniform(0.35, 0.80)
        hot = max(1, round(m_total * hot_frac))
        rest = m_total - hot
        toks = [hot] + ([max(1, rest // max(1, n_experts - 1))] * (n_experts - 1) if n_experts > 1 else [])
        toks[0] += m_total - sum(toks)
    else:
        hot_group = max(1, n_experts // 2)
        cold_group = n_experts - hot_group
        hot_v = max(1, round(m_total * 0.7 / max(1, hot_group)))
        cold_v = max(1, round(m_total * 0.3 / max(1, cold_group)))
        toks = [hot_v] * hot_group + [cold_v] * cold_group
        toks.sort(reverse=True)
        toks[0] += m_total - sum(toks)
    toks = sorted([max(1, tok) for tok in toks], reverse=True)
    return {idx: tok for idx, tok in enumerate(toks) if tok > 0}


def sample_cache(dist):
    eids = list(dist.keys())
    if len(eids) == 1:
        mode = random.choices(["none", "c2", "c3"], weights=[2, 3, 3])[0]
    else:
        mode = random.choices(["none", "c2", "c3", "both"], weights=[2, 3, 3, 4])[0]

    if mode == "none":
        return -1, -1, mode
    if mode == "c2":
        return random.choice(eids), -1, mode
    if mode == "c3":
        return -1, random.choice(eids), mode

    c2_eid = random.choice(eids)
    c3_choices = [eid for eid in eids if eid != c2_eid]
    return c2_eid, random.choice(c3_choices), mode


def percentile(values, pct):
    if not values:
        return float("nan")
    idx = min(len(values) - 1, int(pct * len(values)))
    return values[idx]


def main():
    n_samples = int(os.environ.get("N_SAMPLES", "10000"))
    beam_width = int(os.environ.get("BEAM_WIDTH", "64"))
    print_every = int(os.environ.get("PRINT_EVERY", "500"))
    seed = int(os.environ.get("SEED", "42"))

    random.seed(seed)

    ratios = []
    worse = []
    better_count = 0
    skipped_beam = 0
    skipped_anal = 0
    by_mode = defaultdict(list)
    by_n = defaultdict(list)

    start_time = time.perf_counter()
    for i in range(n_samples):
        dist = random_dist()
        c2_eid, c3_eid, cache_mode = sample_cache(dist)

        try:
            ms_beam, _ = FourStageScheduler(
                dist,
                beam_width=beam_width,
                initial_cache_c2=c2_eid,
                initial_cache_c3=c3_eid,
            ).run()
        except Exception:
            skipped_beam += 1
            continue

        try:
            ms_anal = analytical_schedule(dist, c2_eid, c3_eid)
        except Exception:
            skipped_anal += 1
            ratios.append(float("nan"))
            continue

        ratio = ms_anal / ms_beam
        ratios.append(ratio)
        by_mode[cache_mode].append(ratio)
        by_n[len(dist)].append(ratio)
        if ms_anal < ms_beam:
            better_count += 1
        elif ms_anal > ms_beam:
            worse.append((ratio, i, sorted(dist.values(), reverse=True), c2_eid, c3_eid, cache_mode, ms_beam, ms_anal))

        if print_every > 0 and (i + 1) % print_every == 0:
            elapsed = time.perf_counter() - start_time
            valid = [value for value in ratios if not math.isnan(value)]
            mean_ratio = sum(valid) / len(valid) if valid else 0.0
            worse_so_far = sum(1 for value in valid if value > 1.0)
            eta = elapsed / (i + 1) * (n_samples - i - 1)
            print(
                f"  [{i+1}/{n_samples}] mean={mean_ratio:.4f} worse={worse_so_far} "
                f"elapsed={elapsed:.0f}s ETA={eta:.0f}s",
                flush=True,
            )

    elapsed = time.perf_counter() - start_time
    valid_ratios = sorted(value for value in ratios if not math.isnan(value))
    n_valid = len(valid_ratios)
    if n_valid == 0:
        raise RuntimeError("no valid samples")

    mean_ratio = sum(valid_ratios) / n_valid
    median_ratio = valid_ratios[n_valid // 2]
    pct_equal = sum(1 for value in valid_ratios if value == 1.0) / n_valid * 100
    within_5 = sum(1 for value in valid_ratios if value <= 1.05) / n_valid * 100
    within_10 = sum(1 for value in valid_ratios if value <= 1.10) / n_valid * 100

    print("\n============================================================")
    print(f"N_SAMPLES={n_samples} valid={n_valid} skipped_beam={skipped_beam} skipped_anal={skipped_anal}")
    print(f"BEAM_WIDTH={beam_width} seed={seed}")
    print(f"Total time: {elapsed:.1f}s ({elapsed / n_samples * 1000:.1f}ms/call)")
    print(f"mean_ratio = {mean_ratio:.4f}")
    print(f"median     = {median_ratio:.4f}")
    print(f"pct_equal  = {pct_equal:.1f}%")
    print(f"within  5% = {within_5:.1f}%")
    print(f"within 10% = {within_10:.1f}%")
    print(f"p95 = {percentile(valid_ratios, 0.95):.4f}  p99 = {percentile(valid_ratios, 0.99):.4f}  max = {valid_ratios[-1]:.4f}")
    print(f"worse_than_BW{beam_width} = {len(worse)}  better = {better_count}")

    print("\nBreakdown by cache mode:")
    for mode in ["none", "c2", "c3", "both"]:
        values = sorted(by_mode[mode])
        if not values:
            continue
        print(
            f"  {mode:4s}: count={len(values):4d} mean={sum(values)/len(values):.4f} "
            f"worse={sum(1 for value in values if value > 1.0)} max={values[-1]:.4f}"
        )

    print("\nBreakdown by n_experts:")
    for n_experts in sorted(by_n):
        values = sorted(by_n[n_experts])
        print(
            f"  n={n_experts}: count={len(values):4d} mean={sum(values)/len(values):.4f} "
            f"worse={sum(1 for value in values if value > 1.0)} max={values[-1]:.4f}"
        )

    if worse:
        print("\nTop-10 worst cases:")
        for ratio, idx, toks, c2_eid, c3_eid, mode, ms_beam, ms_anal in sorted(worse, reverse=True)[:10]:
            print(
                f"  i={idx:5d} ratio={ratio:.3f} mode={mode:4s} ntoks={toks[:8]} "
                f"c2={c2_eid} c3={c3_eid} beam={ms_beam} anal={ms_anal}"
            )


if __name__ == "__main__":
    main()