#!/usr/bin/env python3
"""
mass_test.py — 大规模随机 + 穷举测试框架
==========================================

三阶段测试 + 三轮迭代优化分析:

  Phase 1: 穷举小 M (M=1..16) 所有整数分区，奇偶均含 (~916 个分区)
  Phase 2: 随机大 M (M=16,32,64,128)，每个 M 生成 50 例
            - Uniform 随机 (真正随机，含奇数)
            - Zipfian 热冷分布 (随机 s in [0.8,1.6])
            - 双峰 + 尾部分布
  Phase 3: Beam Width 灵敏度扫描 (beam=1..256，8 个代表性案例)

三轮迭代对比:
  v1: smart_split=False, beam_width=32  (基线)
  v2: smart_split=True,  beam_width=32  (候选集优化，等算力)
  v3: smart_split=True,  beam_width=128 (更大搜索宽度)

目标总运行时间: ~10-15 分钟
输出: mass_results.md
"""

import sys
import os
import math
import random
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# 加载调度器
sys.path.insert(0, os.path.dirname(__file__))
from beam_scheduler import (
    BeamScheduler,
    remaining_lb,
    SHAPE_A,
    SHAPE_B,
    SHAPE_C,
    ALL_SHAPES,
    _split_candidates,
)

# ============================================================
#  1. 测试数据生成器
# ============================================================


def integer_partitions(n: int, max_parts: int = 32, min_val: int = 1):
    """
    生成 n 的所有整数分区 (降序排列, 每部分 >= min_val).
    不限制具体 expert ID, 只关心 token 数量的多重集合.
    返回 tuple-of-ints, 如 (6,3,1) 表示 3 个 expert 各获 6,3,1 token.
    """

    def _helper(remaining, max_part, n_parts):
        if remaining == 0:
            yield ()
            return
        if n_parts == 0:
            return
        for part in range(min(remaining, max_part), min_val - 1, -1):
            for rest in _helper(remaining - part, part, n_parts - 1):
                yield (part,) + rest

    yield from _helper(n, n, max_parts)


def partition_to_dist(partition: tuple) -> Dict[int, int]:
    """将分区元组转为 {eid: ntok} 字典 (eid 从 0 开始)."""
    return {i: v for i, v in enumerate(partition)}


def random_dist_uniform(
    M: int, rng: random.Random, max_experts: int = 16
) -> Dict[int, int]:
    """
    Stars-and-bars 均匀随机分布.
    随机选 n 个 expert (1..min(M, max_experts))，再用隔板法分配 M 个 token.
    每个 expert 至少分得 1 个 token，token 数可以是奇数.
    """
    n = rng.randint(1, min(M, max_experts))
    if n >= M:
        return {i: 1 for i in range(min(n, M))}
    # 在 [1, M-1] 中随机放 n-1 个隔板
    separators = sorted(rng.sample(range(1, M), n - 1)) if n > 1 else []
    points = [0] + separators + [M]
    counts = [points[i + 1] - points[i] for i in range(len(points) - 1)]
    rng.shuffle(counts)  # 打乱，使大小与 eid 随机无关
    return {i: c for i, c in enumerate(counts)}


def random_dist_zipf(
    M: int, rng: random.Random, s: float = 1.2, max_experts: int = 16
) -> Dict[int, int]:
    """
    Zipfian 热冷分布: expert k 的权重 ∝ 1/k^s.
    真实 MoE 常见场景: 少数 expert 获大量 token, 多数 expert 获少量 token.
    """
    n = rng.randint(2, min(M, max_experts))
    weights = [1.0 / (k**s) for k in range(1, n + 1)]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]
    # 多项式采样
    counts = [0] * n
    for _ in range(M):
        r = rng.random()
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum or i == n - 1:
                counts[i] += 1
                break
    result = {i: c for i, c in enumerate(counts) if c > 0}
    return result if result else {0: M}


def random_dist_bimodal(
    M: int, rng: random.Random, max_experts: int = 16
) -> Dict[int, int]:
    """
    双峰分布: 1~2 个热门 expert + 若干冷门 expert (1~3 token 各).
    模拟部分 head 高度聚焦的情况.
    """
    n_hot = rng.randint(1, 2)
    n_cold = rng.randint(0, min(8, max_experts - n_hot))
    cold_total = n_cold  # 每个冷门 expert 固定 1 token
    hot_total = M - cold_total
    if hot_total < n_hot:
        # 退化为均匀
        return random_dist_uniform(M, rng, max_experts)
    # 分配 hot
    if n_hot == 1:
        hot_counts = [hot_total]
    else:
        split = rng.randint(n_hot - 1, hot_total - 1)
        hot_counts = [split, hot_total - split]
    counts = hot_counts + [1] * n_cold
    rng.shuffle(counts)
    return {i: c for i, c in enumerate(counts) if c > 0}


def generate_cases(
    M: int, n_random: int, rng: random.Random, max_experts: int = 16
) -> List[Tuple[str, Dict[int, int]]]:
    """
    为给定 M 生成多样化的随机 token 分布.
    n_random 个中:
      40% uniform random (真正随机, 含奇数)
      35% zipfian
      25% bimodal + tail
    """
    cases = []
    n_uniform = int(n_random * 0.40)
    n_zipf = int(n_random * 0.35)
    n_bi = n_random - n_uniform - n_zipf

    for i in range(n_uniform):
        dist = random_dist_uniform(M, rng, max_experts)
        cases.append((f"uniform_{i}", dist))
    for i in range(n_zipf):
        s = rng.uniform(0.8, 1.6)  # 随机化 Zipf 指数
        dist = random_dist_zipf(M, rng, s=s, max_experts=max_experts)
        cases.append((f"zipf_{i}", dist))
    for i in range(n_bi):
        dist = random_dist_bimodal(M, rng, max_experts)
        cases.append((f"bimodal_{i}", dist))

    rng.shuffle(cases)
    return cases


# ============================================================
#  2. 单次测试 & 指标计算
# ============================================================


def run_one(
    token_dist: Dict[int, int],
    beam_width: int = 64,
    smart_split: bool = True,
) -> dict:
    """
    运行一次 Beam Search 调度, 返回指标字典.
    指标:
      makespan  : 最优 makespan (cc)
      lb        : Johnson Rule 下界 (cc)
      ratio     : makespan / lb  (越接近 1.0 越好)
      dma_util  : DMA 利用率 (两通道总 fetch 时间 / 2*makespan)
      vc_util   : VersaCore 利用率 (两 cluster 总计算时间 / 2*makespan)
      elapsed   : 调度器用时 (s)
    """
    t0 = time.perf_counter()
    sched = BeamScheduler(
        token_dist,
        beam_width=beam_width,
        allow_split=True,
        smart_split=smart_split,
    )
    makespan, actions = sched.run()
    elapsed = time.perf_counter() - t0

    lb = remaining_lb(tuple(sorted(token_dist.items(), key=lambda x: -x[1])))
    ratio = makespan / lb if lb > 0 else 1.0

    total_fetch = total_compute = 0
    for action in actions:
        for eid, ntok, shape, cached, start in [
            (
                action.c2_eid,
                action.c2_ntok,
                action.c2_shape,
                action.c2_cached,
                action.c2_start,
            ),
            (
                action.c3_eid,
                action.c3_ntok,
                action.c3_shape if action.c3_shape else SHAPE_C,
                action.c3_cached,
                action.c3_start,
            ),
        ]:
            if eid < 0 or start < 0:
                continue
            total_compute += shape.T_task(ntok, cached)
            total_fetch += 0 if cached else shape.T_iter

    dma_util = total_fetch / (makespan * 2) if makespan > 0 else 0.0
    vc_util = total_compute / (makespan * 2) if makespan > 0 else 0.0

    return {
        "dist": token_dist,
        "M": sum(token_dist.values()),
        "n_exp": len(token_dist),
        "makespan": makespan,
        "lb": lb,
        "ratio": ratio,
        "dma_util": dma_util,
        "vc_util": vc_util,
        "elapsed": elapsed,
    }


# ============================================================
#  3. 批量测试
# ============================================================


def batch_test(
    cases: List[Tuple[str, Dict[int, int]]],
    beam_width: int,
    smart_split: bool,
    label: str = "",
    print_progress: bool = True,
) -> List[dict]:
    results = []
    t_batch = time.perf_counter()
    for idx, (name, dist) in enumerate(cases):
        r = run_one(dist, beam_width=beam_width, smart_split=smart_split)
        r["name"] = name
        results.append(r)
        if print_progress and (idx + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_batch
            mean_r = sum(x["ratio"] for x in results) / len(results)
            print(
                f"    [{label}] {idx+1}/{len(cases)}  "
                f"mean_ratio={mean_r:.4f}  ({elapsed:.1f}s)"
            )
    return results


# ============================================================
#  4. 统计分析
# ============================================================


def stats(values: List[float]) -> dict:
    if not values:
        return {}
    v = sorted(values)
    n = len(v)
    return {
        "mean": sum(v) / n,
        "min": v[0],
        "p25": v[n // 4],
        "median": v[n // 2],
        "p75": v[3 * n // 4],
        "p90": v[int(n * 0.9)],
        "p99": v[int(n * 0.99)],
        "max": v[-1],
        "pct_opt": sum(1 for x in v if x <= 1.001) / n,  # ratio ≤ 1.001 视为最优
    }


def analyze(results: List[dict], tag: str = "") -> dict:
    """
    汇总统计: ratio / dma_util / vc_util 的分布.
    返回统计摘要字典, 并打印到控制台.
    """
    if not results:
        return {}
    ratios = [r["ratio"] for r in results]
    dmas = [r["dma_util"] for r in results]
    vcs = [r["vc_util"] for r in results]
    r_stat = stats(ratios)
    d_stat = stats(dmas)
    v_stat = stats(vcs)

    print(f"\n  [{tag}] n={len(results)} cases")
    print(
        f"    ratio  : mean={r_stat['mean']:.4f}  "
        f"p25={r_stat['p25']:.4f}  median={r_stat['median']:.4f}  "
        f"p90={r_stat['p90']:.4f}  max={r_stat['max']:.4f}  "
        f"pct_optimal={r_stat['pct_opt']:.1%}"
    )
    print(
        f"    dma_util: mean={d_stat['mean']:.3f}  "
        f"median={d_stat['median']:.3f}  min={d_stat['min']:.3f}"
    )
    print(
        f"    vc_util : mean={v_stat['mean']:.3f}  "
        f"median={v_stat['median']:.3f}  min={v_stat['min']:.3f}"
    )

    return {"ratio": r_stat, "dma": d_stat, "vc": v_stat, "n": len(results)}


def worst_cases(results: List[dict], top_k: int = 15) -> List[dict]:
    """按 ratio 降序返回最差的 top_k 个案例."""
    return sorted(results, key=lambda r: -r["ratio"])[:top_k]


def ratio_histogram(results: List[dict], buckets=None) -> str:
    """生成 ratio 分布直方图 (文本形式)."""
    if buckets is None:
        buckets = [1.0, 1.01, 1.05, 1.10, 1.20, 1.40, 1.70, 2.00, float("inf")]
    labels = [
        "=1.00 (最优)",
        "1.00-1.01",
        "1.01-1.05",
        "1.05-1.10",
        "1.10-1.20",
        "1.20-1.40",
        "1.40-1.70",
        "1.70-2.00",
        ">2.00",
    ]
    counts = [0] * (len(buckets))
    for r in results:
        ratio = r["ratio"]
        for i in range(len(buckets)):
            if ratio <= buckets[i]:
                counts[i] += 1
                break
    n = len(results)
    lines = ["  Ratio 分布直方图:"]
    for label, cnt in zip(labels, counts):
        bar = "#" * int(cnt / n * 60)
        lines.append(f"    {label:16s}: {cnt:4d} ({cnt/n:5.1%}) {bar}")
    return "\n".join(lines)


def by_m_breakdown(results: List[dict]) -> str:
    """按 M 分组输出统计."""
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        groups[r["M"]].append(r["ratio"])
    lines = ["  按 M 分组统计 (ratio):"]
    lines.append(
        f"  {'M':>6}  {'n':>5}  {'mean':>7}  {'median':>7}  "
        f"{'p90':>7}  {'max':>7}  {'%optimal':>9}"
    )
    for M in sorted(groups.keys()):
        rs = sorted(groups[M])
        n = len(rs)
        mean_r = sum(rs) / n
        med = rs[n // 2]
        p90 = rs[int(n * 0.9)]
        mx = rs[-1]
        pct = sum(1 for x in rs if x <= 1.001) / n
        lines.append(
            f"  {M:>6}  {n:>5}  {mean_r:>7.4f}  {med:>7.4f}  "
            f"{p90:>7.4f}  {mx:>7.4f}  {pct:>8.1%}"
        )
    return "\n".join(lines)


# ============================================================
#  5. Beam Width 灵敏度扫描
# ============================================================


def beam_sweep(
    cases: List[Tuple[str, Dict[int, int]]],
    beam_widths: List[int],
    smart_split: bool = True,
) -> str:
    """
    对给定 cases 扫描不同 beam_width, 输出 makespan 收敛情况.
    """
    lines = []
    for name, dist in cases:
        M = sum(dist.values())
        lb = remaining_lb(tuple(sorted(dist.items(), key=lambda x: -x[1])))
        line_parts = [f"  {name} M={M} (lb={lb:,}):"]
        for bw in beam_widths:
            t0 = time.perf_counter()
            ms, _ = BeamScheduler(
                dist, beam_width=bw, allow_split=True, smart_split=smart_split
            ).run()
            elapsed = time.perf_counter() - t0
            line_parts.append(
                f"    beam={bw:>4}: makespan={ms:>10,}  ratio={ms/lb:.4f}  ({elapsed:.2f}s)"
            )
        lines.append("\n".join(line_parts))
    return "\n\n".join(lines)


# ============================================================
#  6. 主程序
# ============================================================


def main():
    import sys

    OUT_PATH = "/esat/studscratch/r1015673/Thesis/Idea_Model/mass_results.md"

    class _Tee:
        def __init__(self, *files):
            self._files = files

        def write(self, s):
            for f in self._files:
                try:
                    f.write(s)
                except (BrokenPipeError, OSError):
                    pass

        def flush(self):
            for f in self._files:
                f.flush()

    _orig = sys.stdout
    _fout = open(OUT_PATH, "w", encoding="utf-8")
    _fout.write("# Mass Test 调度器大规模验证报告\n\n")
    _fout.write(f"> **生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
    _fout.write("> **测试规模**: 穷举 M=1..16 所有整数分区 + 随机大 M  \n")
    _fout.write(
        "> **三轮对比**: v1(baseline beam=32) / v2(smart_split beam=32) / v3(smart_split beam=64)  \n\n"
    )
    _fout.write("```\n")
    sys.stdout = _Tee(_orig, _fout)

    rng = random.Random(2026)
    T_TOTAL = time.perf_counter()

    # ══════════════════════════════════════════════════════════
    # Phase 1: 穷举小 M (所有整数分区, 含奇数)
    # ══════════════════════════════════════════════════════════
    print("=" * 72)
    print("  Phase 1: 穷举小 M (M=1..16) 所有整数分区")
    print("  注: 分区仅关心 token 数量多重集, 奇偶均包含")
    print("=" * 72)

    small_cases: List[Tuple[str, Dict[int, int]]] = []
    for M in range(1, 17):
        parts = list(integer_partitions(M, max_parts=16))
        for p in parts:
            small_cases.append((f"M{M}", partition_to_dist(p)))
    print(f"  穷举总量: {len(small_cases)} 个分区\n")

    print("  ── v1: smart_split=False, beam=32 (基线) ──")
    r1_small = batch_test(
        small_cases,
        beam_width=32,
        smart_split=False,
        label="v1-small",
        print_progress=False,
    )
    analyze(r1_small, "v1 small")

    print("  ── v2: smart_split=True,  beam=32 (候选集优化，等算力) ──")
    r2_small = batch_test(
        small_cases,
        beam_width=32,
        smart_split=True,
        label="v2-small",
        print_progress=False,
    )
    analyze(r2_small, "v2 small")

    print("  ── v3: smart_split=True,  beam=64 (更大搜索宽度) ──")
    r3_small = batch_test(
        small_cases,
        beam_width=64,
        smart_split=True,
        label="v3-small",
        print_progress=False,
    )
    analyze(r3_small, "v3 small")

    print("\n" + by_m_breakdown(r3_small))
    print("\n" + ratio_histogram(r3_small))

    # 小 M 最差案例
    print("\n  [v3] 小 M 最差 10 例 (ratio 最高):")
    print(
        f"  {'M':>4} {'n_exp':>5} {'dist':>30} {'makespan':>10} {'lb':>10} {'ratio':>7}"
    )
    for r in worst_cases(r3_small, top_k=10):
        dist_str = str(sorted(r["dist"].values(), reverse=True))[:28]
        print(
            f"  {r['M']:>4} {r['n_exp']:>5} {dist_str:>30} "
            f"{r['makespan']:>10,} {r['lb']:>10,} {r['ratio']:>7.4f}"
        )

    # ══════════════════════════════════════════════════════════
    # Phase 2: 随机大 M
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  Phase 2: 随机大 M (M=16,32,64,128)  各 30 例")
    print("  分布类型: 40% uniform-random + 35% zipfian + 25% bimodal")
    print("  (含奇数 token 数，随机 expert 数量)")
    print("=" * 72)

    large_cases_all: List[Tuple[str, Dict[int, int]]] = []
    for M in [16, 32, 64, 128]:
        batch = generate_cases(M, n_random=30, rng=rng, max_experts=min(M, 16))
        for name, dist in batch:
            large_cases_all.append((f"M{M}_{name}", dist))
        print(f"  M={M}: 生成 {len(batch)} 例")

    print(f"\n  大 M 总量: {len(large_cases_all)} 例\n")

    print("  ── v1: smart_split=False, beam=32 ──")
    r1_large = batch_test(
        large_cases_all, beam_width=32, smart_split=False, label="v1-large"
    )
    analyze(r1_large, "v1 large")

    print("  ── v2: smart_split=True,  beam=32 ──")
    r2_large = batch_test(
        large_cases_all, beam_width=32, smart_split=True, label="v2-large"
    )
    analyze(r2_large, "v2 large")

    print("  ── v3: smart_split=True,  beam=64 ──")
    r3_large = batch_test(
        large_cases_all, beam_width=64, smart_split=True, label="v3-large"
    )
    analyze(r3_large, "v3 large")

    print("\n" + by_m_breakdown(r3_large))
    print("\n" + ratio_histogram(r3_large))

    print("\n  [v3] 大 M 最差 15 例:")
    print(
        f"  {'M':>4} {'n_exp':>5} {'dist(top5)':>35} {'makespan':>10} "
        f"{'lb':>10} {'ratio':>7}"
    )
    for r in worst_cases(r3_large, top_k=15):
        top5 = sorted(r["dist"].values(), reverse=True)[:5]
        dist_str = str(top5)[:33]
        print(
            f"  {r['M']:>4} {r['n_exp']:>5} {dist_str:>35} "
            f"{r['makespan']:>10,} {r['lb']:>10,} {r['ratio']:>7.4f}"
        )

    # ══════════════════════════════════════════════════════════
    # Phase 3: Beam Width 灵敏度扫描
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  Phase 3: Beam Width 灵敏度扫描")
    print("  使用 v3 最差案例 + 典型案例，测试 beam=1..128")
    print("=" * 72)

    worst3 = [(f"worst_{i}", r["dist"]) for i, r in enumerate(worst_cases(r3_large, 3))]
    canonical = [
        ("hot(6,1,1)", {0: 6, 1: 1, 2: 1}),
        ("hot(20,6,6)", {0: 20, 1: 6, 2: 6}),
        ("zipf(32,16,8,4,2,1)", {0: 32, 1: 16, 2: 8, 3: 4, 4: 2, 5: 1}),
        ("single(32)", {0: 32}),
        ("odd(7,5,3)", {0: 7, 1: 5, 2: 3}),
        ("odd(9,7,5,3,1)", {0: 9, 1: 7, 2: 5, 3: 3, 4: 1}),
        ("M128-zipf", {0: 64, 1: 32, 2: 16, 3: 8, 4: 4, 5: 2, 6: 2}),
    ]
    sweep_cases = worst3 + canonical
    beam_widths = [1, 4, 16, 32, 64, 128]

    print("\n" + beam_sweep(sweep_cases, beam_widths, smart_split=True))

    # ══════════════════════════════════════════════════════════
    # 综合对比: v1 vs v2 vs v3
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  综合对比: 三轮迭代效果")
    print("=" * 72)
    all_cases = small_cases + large_cases_all
    r1_all = r1_small + r1_large
    r2_all = r2_small + r2_large
    r3_all = r3_small + r3_large

    def cmp_row(label, r_list):
        ratios = [r["ratio"] for r in r_list]
        dmas = [r["dma_util"] for r in r_list]
        vcs = [r["vc_util"] for r in r_list]
        pct_opt = sum(1 for x in ratios if x <= 1.001) / len(ratios)
        t_total = sum(r["elapsed"] for r in r_list)
        print(
            f"  {label:40s}  mean_ratio={sum(ratios)/len(ratios):.4f}  "
            f"pct_opt={pct_opt:.1%}  "
            f"mean_dma={sum(dmas)/len(dmas):.3f}  "
            f"mean_vc={sum(vcs)/len(vcs):.3f}  "
            f"总耗时={t_total:.1f}s"
        )

    print(
        f"  {'配置':40s}  {'mean_ratio':10}  {'pct_opt':7}  "
        f"{'mean_dma':8}  {'mean_vc':7}  {'总耗时':6}"
    )
    print("  " + "-" * 100)
    cmp_row("v1: smart_split=False, beam=32  (基线)", r1_all)
    cmp_row("v2: smart_split=True,  beam=32  (候选集优化)", r2_all)
    cmp_row("v3: smart_split=True,  beam=64  (大搜索宽度)", r3_all)

    # 量化 smart_split 的加速比
    print("\n  [SPLIT 候选集优化加速比分析]")
    for ntok in [8, 16, 32, 64, 128]:
        from beam_scheduler import SHAPE_A, SHAPE_B, SHAPE_C

        for sA, sB in [(SHAPE_A, SHAPE_A), (SHAPE_A, SHAPE_B), (SHAPE_B, SHAPE_C)]:
            brute = ntok - 1
            smart = len(_split_candidates(ntok, sA, sB))
            speedup = brute / smart if smart > 0 else 1
            print(
                f"    ntok={ntok:>4}, ({sA.name},{sB.name}): "
                f"brute={brute:>4}, smart={smart:>3}, 加速={speedup:.1f}x"
            )

    # ══════════════════════════════════════════════════════════
    # 结论与调度规律总结
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  调度规律总结 (从测试结果中归纳)")
    print("=" * 72)

    # 按 n_experts 分析
    from collections import defaultdict

    by_nexp = defaultdict(list)
    for r in r3_all:
        by_nexp[r["n_exp"]].append(r["ratio"])
    print("\n  按 expert 数量分析 (v3, mean_ratio):")
    for ne in sorted(by_nexp.keys()):
        rs = by_nexp[ne]
        print(
            f"    n_experts={ne:>3}: n={len(rs):>5}  "
            f"mean={sum(rs)/len(rs):.4f}  "
            f"max={max(rs):.4f}"
        )

    # 奇偶分析
    odd_r = [r["ratio"] for r in r3_all if r["M"] % 2 == 1]
    even_r = [r["ratio"] for r in r3_all if r["M"] % 2 == 0]
    print(
        f"\n  奇数 M: n={len(odd_r)}  mean_ratio={sum(odd_r)/len(odd_r) if odd_r else 0:.4f}"
        f"  max={max(odd_r) if odd_r else 0:.4f}"
    )
    print(
        f"  偶数 M: n={len(even_r)}  mean_ratio={sum(even_r)/len(even_r) if even_r else 0:.4f}"
        f"  max={max(even_r) if even_r else 0:.4f}"
    )

    print(f"\n  总运行时间: {time.perf_counter()-T_TOTAL:.1f}s")
    print(f"  测试案例总数: {len(all_cases)}")

    sys.stdout = _orig
    _fout.write("```\n")
    _fout.close()
    print(f"\n[完整报告已保存 → {OUT_PATH}]")


if __name__ == "__main__":
    main()
