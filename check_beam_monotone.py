#!/usr/bin/env python3
"""
检查 beam search 是否单调：beam_width 越大结果越好？
以及确认使用 min(BW8,BW16,BW32,BW64) 作为 golden reference。
"""
import sys, random
sys.path.insert(0, '/esat/studscratch/r1015673/Thesis/Idea_Model')
sys.path.insert(0, '/esat/studscratch/r1015673/Thesis')

from four_stage_scheduler import FourStageScheduler
from Idea_Model.analytical_scheduler import random_dist, analytical_schedule

N = 300
non_monotone = 0
improved_by_min = 0
total_diffs = 0

ratios_vs_min = []

for i in range(N):
    dist = random_dist(seed=i)
    keys = list(dist.keys())
    c2 = keys[0] if random.Random(i*1000).random() < 0.25 else -1
    c3 = keys[1] if (c2 >= 0 and len(keys) >= 2 and random.Random(i*1000+1).random() < 0.25) else -1

    results = {}
    for bw in [4, 8, 16, 32, 64]:
        m, _ = FourStageScheduler(dist, beam_width=bw, initial_cache_c2=c2, initial_cache_c3=c3).run()
        results[bw] = m

    m_best = min(results.values())
    m_anal = analytical_schedule(dist, c2, c3)

    if not all(results[bw] >= results[min(results.keys())] for bw in sorted(results.keys())):
        non_monotone += 1

    if any(results[bw] < results[64] for bw in [4, 8, 16, 32]):
        improved_by_min += 1
        total_diffs += 1
        toks = sorted(dist.values(), reverse=True)
        print(f"i={i}: BW4={results[4]} BW8={results[8]} BW16={results[16]} BW32={results[32]} BW64={results[64]}  min={m_best}  anal={m_anal}  toks={toks[:5]}")

    ratios_vs_min.append(m_anal / m_best)

ratios_vs_min.sort()
n = len(ratios_vs_min)
print(f"\n== N={N} ==")
print(f"非单调（某BW<BW64）: {improved_by_min}")
print(f"\n分析法 vs min(BW4..64) golden:")
print(f"mean ratio = {sum(ratios_vs_min)/n:.4f}")
print(f"pct_optimal = {sum(1 for r in ratios_vs_min if r<=1.001)/n*100:.1f}%")
print(f"pct_5%      = {sum(1 for r in ratios_vs_min if r<=1.05)/n*100:.1f}%")
print(f"pct_10%     = {sum(1 for r in ratios_vs_min if r<=1.10)/n*100:.1f}%")
print(f"max ratio   = {max(ratios_vs_min):.4f}")
