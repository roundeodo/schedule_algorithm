#!/usr/bin/env python3
"""10000 case 对比 analytical_schedule vs lite_schedule 质量与速度。"""

import sys, random, time, os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analytical_scheduler import analytical_schedule
from lite_scheduler import lite_schedule


def random_dist(seed=None):
    rng = random.Random(seed)
    n_experts = rng.choices([1, 2, 3, 4, 5, 6, 7, 8], weights=[1, 3, 5, 6, 5, 4, 3, 2])[
        0
    ]
    expert_ids = rng.sample(range(32), n_experts)
    tokens = sorted(
        [max(1, int(rng.paretovariate(1.5) * 2)) for _ in range(n_experts)],
        reverse=True,
    )
    tokens = [min(t, 64) for t in tokens]
    return {eid: t for eid, t in zip(expert_ids, tokens)}


N = 10000
ratios = []
times_a, times_l = [], []
crashes_a = crashes_l = 0

print(f"运行 {N} 组随机测试（含缓存/无缓存），请稍候……", flush=True)
t_wall0 = time.perf_counter()

for i in range(N * 3):
    if len(ratios) >= N:
        break
    rng_seed = random.Random(i * 997 + 13)
    dist = random_dist(seed=i)
    keys = list(dist.keys())
    # 约 60% 概率有 c2，40% 概率有 c3（含无缓存情形）
    c2 = keys[0] if rng_seed.random() < 0.6 else -1
    c3 = keys[1] if len(keys) >= 2 and rng_seed.random() < 0.4 else -1

    try:
        t0 = time.perf_counter()
        a = analytical_schedule(dist, c2, c3)
        times_a.append(time.perf_counter() - t0)
    except Exception:
        crashes_a += 1
        continue

    try:
        t0 = time.perf_counter()
        l = lite_schedule(dist, c2, c3)
        times_l.append(time.perf_counter() - t0)
    except Exception:
        crashes_l += 1
        continue

    if a > 0:
        ratios.append((l / a, dist, c2, c3, a, l))

    if len(ratios) % 1000 == 0 and len(ratios) > 0:
        elapsed = time.perf_counter() - t_wall0
        print(f"  {len(ratios)}/{N} done ({elapsed:.0f}s elapsed)", flush=True)

n = len(ratios)
if n == 0:
    print("无有效数据")
    sys.exit(1)

rv = [r[0] for r in ratios]
elapsed_total = time.perf_counter() - t_wall0

print(f"\n{'='*55}")
print(f"有效对比:                {n}")
print(f"crashes (anal/lite):    {crashes_a}/{crashes_l}")
print(f"均值 ratio:              {sum(rv)/n:.4f}")
print(f"最大 ratio:              {max(rv):.4f}")
print(f"pct 完全相同 (≤1.001):   {sum(1 for r in rv if r<=1.001)/n*100:.1f}%")
print(f"pct <2%    (≤1.020):    {sum(1 for r in rv if r<=1.020)/n*100:.1f}%")
print(f"pct <5%    (≤1.050):    {sum(1 for r in rv if r<=1.050)/n*100:.1f}%")
print(f"pct <10%   (≤1.100):    {sum(1 for r in rv if r<=1.100)/n*100:.1f}%")
print(f"analytical 均时:        {sum(times_a)/len(times_a)*1000:.1f}ms")
print(f"lite       均时:        {sum(times_l)/len(times_l)*1000:.1f}ms")
print(f"speedup:                {sum(times_a)/sum(times_l):.2f}x")
print(f"总耗时:                  {elapsed_total:.0f}s")
print(f"{'='*55}")

worst = sorted(ratios, key=lambda x: x[0], reverse=True)[:10]
if worst[0][0] > 1.001:
    print("\n差距最大的前10个案例:")
    for r, dist, c2, c3, a, l in worst:
        toks = sorted(dist.values(), reverse=True)
        print(f"  ratio={r:.3f} toks={toks} c2={c2} c3={c3} anal={a} lite={l}")
