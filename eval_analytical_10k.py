#!/usr/bin/env python3
"""
eval_analytical_10k.py
用与 gen_hw_training.py 完全相同的 seed=42 生成 10000 个分布+缓存状态，
分别运行 FourStageScheduler(beam_width=16) 和 analytical_schedule，统计质量。
"""
import sys, os, random, time, math

sys.path.insert(0, os.path.dirname(__file__))
from four_stage_scheduler import FourStageScheduler
from analytical_scheduler import analytical_schedule


# ── 与 gen_hw_training.py 完全相同的分布生成器 ─────────────────────────────────
def random_dist():
    M = random.choice([4, 8, 12, 16, 24, 32, 48, 64])
    n = random.randint(1, min(8, M))
    dist_type = random.choices(
        ["uniform", "zipf", "hot", "bimodal", "single"], weights=[2, 4, 3, 2, 1]
    )[0]
    if dist_type == "single":
        return {0: M}
    if dist_type == "uniform":
        base = M // n
        toks = [base] * n
        toks[0] += M - sum(toks)
    elif dist_type == "zipf":
        alpha = random.uniform(0.6, 2.5)
        weights = [1.0 / (i + 1) ** alpha for i in range(n)]
        s = sum(weights)
        toks = [max(1, round(w / s * M)) for w in weights]
        toks.sort(reverse=True)
        toks[0] += M - sum(toks)
    elif dist_type == "hot":
        hot_frac = random.uniform(0.35, 0.80)
        hot = max(1, round(M * hot_frac))
        rest = M - hot
        toks = [hot] + ([max(1, rest // max(1, n - 1))] * (n - 1) if n > 1 else [])
        toks[0] += M - sum(toks)
    else:  # bimodal
        h = max(1, n // 2)
        c = n - h
        hot_v = max(1, round(M * 0.7 / max(1, h)))
        cold_v = max(1, round(M * 0.3 / max(1, c)))
        toks = [hot_v] * h + [cold_v] * c
        toks.sort(reverse=True)
        toks[0] += M - sum(toks)
    toks = sorted([max(1, t) for t in toks], reverse=True)
    return {i: t for i, t in enumerate(toks) if t > 0}


# ── 主循环 ─────────────────────────────────────────────────────────────────────
N_SAMPLES = int(os.environ.get("N_SAMPLES", "10000"))
BEAM_WIDTH = int(os.environ.get("BEAM_WIDTH", "16"))
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "50"))

random.seed(42)

ratios = []
worse = []  # (ratio, i, dist, cache_c2, cache_c3, ms_beam, ms_anal)
better_count = 0
skipped_beam = 0
skipped_anal = 0

t0 = time.perf_counter()

for i in range(N_SAMPLES):
    dist = random_dist()
    n = len(dist)
    # 与 gen_hw_training.py 完全相同的缓存采样逻辑
    cache_c2 = random.choice([-1, -1, -1, 0]) if n >= 1 else -1
    cache_c3_opts = [-1, -1, -1]
    if n >= 2 and cache_c2 >= 0:
        cache_c3_opts.append(1)
    cache_c3 = random.choice(cache_c3_opts)
    # cache_c2/c3 存的是排名 index，需要转为 expert_id
    items_sorted = sorted(dist.items(), key=lambda x: -x[1])
    c2_eid = items_sorted[cache_c2][0] if cache_c2 >= 0 else -1
    c3_eid = items_sorted[cache_c3][0] if cache_c3 >= 0 else -1

    # Beam reference
    try:
        ms_beam, _ = FourStageScheduler(
            dist,
            beam_width=BEAM_WIDTH,
            initial_cache_c2=c2_eid,
            initial_cache_c3=c3_eid,
        ).run()
    except Exception:
        skipped_beam += 1
        continue

    # Analytical
    try:
        ms_anal = analytical_schedule(dist, c2_eid, c3_eid)
    except Exception:
        skipped_anal += 1
        ratios.append(float("nan"))
        continue

    r = ms_anal / ms_beam
    ratios.append(r)
    if ms_anal < ms_beam:
        better_count += 1
    elif ms_anal > ms_beam:
        worse.append(
            (
                r,
                i,
                sorted(dist.values(), reverse=True)[:5],
                c2_eid,
                c3_eid,
                ms_beam,
                ms_anal,
            )
        )

    if (i + 1) % PRINT_EVERY == 0:
        elapsed = time.perf_counter() - t0
        valid = [x for x in ratios if not math.isnan(x)]
        mean_r = sum(valid) / len(valid) if valid else 0
        worse_so_far = sum(1 for x in valid if x > 1.0)
        print(
            f"  [{i+1}/{N_SAMPLES}]  mean={mean_r:.4f}  worse={worse_so_far}  "
            f"elapsed={elapsed:.0f}s  ETA={elapsed/(i+1)*(N_SAMPLES-i-1):.0f}s",
            flush=True,
        )

t1 = time.perf_counter()
elapsed = t1 - t0

# ── 统计 ───────────────────────────────────────────────────────────────────────
valid_ratios = sorted(r for r in ratios if not math.isnan(r))
N = len(valid_ratios)
mean_r = sum(valid_ratios) / N
median_r = valid_ratios[N // 2]
pct_optimal = sum(1 for r in valid_ratios if r == 1.0) / N * 100
within_5 = sum(1 for r in valid_ratios if r <= 1.05) / N * 100
within_10 = sum(1 for r in valid_ratios if r <= 1.10) / N * 100
p95 = valid_ratios[int(0.95 * N)]
p99 = valid_ratios[int(0.99 * N)]
max_r = valid_ratios[-1]

print(f"\n{'='*60}")
print(
    f"N_SAMPLES={N_SAMPLES}  valid={N}  skipped_beam={skipped_beam}  skipped_anal={skipped_anal}"
)
print(f"Total time: {elapsed:.1f}s  ({elapsed/N_SAMPLES*1000:.1f}ms/call)")
print(f"")
print(f"mean_ratio = {mean_r:.4f}")
print(f"median     = {median_r:.4f}")
print(f"pct_optimal (ratio=1.0) = {pct_optimal:.1f}%")
print(f"within  5% = {within_5:.1f}%")
print(f"within 10% = {within_10:.1f}%")
print(f"p95 = {p95:.4f}  p99 = {p99:.4f}  max = {max_r:.4f}")
print(f"worse_than_BW{BEAM_WIDTH} = {len(worse)}  better = {better_count}")
print(f"")

# Top 10 worst cases
if worse:
    print("Top-10 worst cases:")
    for r, i, toks, c2e, c3e, mb, ma in sorted(worse, reverse=True)[:10]:
        print(
            f"  i={i:5d} ratio={r:.3f} ntoks={toks} c2={c2e} c3={c3e} beam={mb} anal={ma}"
        )

# Distribution breakdown by n_experts
print("\nBreakdown by n_experts:")
from collections import defaultdict

by_n = defaultdict(list)
random.seed(42)
for i in range(N_SAMPLES):
    dist = random_dist()
    n = len(dist)
    cache_c2 = random.choice([-1, -1, -1, 0]) if n >= 1 else -1
    cache_c3_opts = [-1, -1, -1]
    if n >= 2 and cache_c2 >= 0:
        cache_c3_opts.append(1)
    cache_c3 = random.choice(cache_c3_opts)
    if i < len(ratios) and not math.isnan(ratios[i]):
        by_n[n].append(ratios[i])

for ne in sorted(by_n):
    rs = sorted(by_n[ne])
    m = sum(rs) / len(rs)
    w = sum(1 for r in rs if r > 1.0)
    print(f"  n={ne}: count={len(rs):4d}  mean={m:.4f}  worse={w}")
