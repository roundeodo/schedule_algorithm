#!/usr/bin/env python3
"""
mass_test_4stage.py — 四阶段调度器大规模验证
================================================

测试策略:
  Phase 1: 穷举小 M (M=1..10) 所有整数分区          (~271 个案例)
  Phase 2: 随机中 M (M=16):  30 例 (uniform/zipf/bimodal 各 1/3)
  Phase 3: 随机大 M (M=32):  15 例
  Phase 4: 随机大 M (M=64):   5 例  (目标分布同款)
  Phase 5: beam_width 灵敏度  (3 个代表案例 × beam=8,16,32,64,128)

对比维度:
  A: FourStageScheduler beam=32  (主力)
  B: FourStageScheduler beam=64  (参考精度，仅 Phase 2 子集)
  另附: beam=1 (贪心) vs beam=32, 展示搜索收益

总预期运行时间: ~15-25 分钟
输出: mass_results_4stage.md
"""

import sys
import os
import math
import random
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from four_stage_scheduler import (
    FourStageScheduler,
    lb_remaining,
    compute_efficiency,
    SHAPE_A,
    SHAPE_B,
    SHAPE_C,
    ALL_SHAPES,
)

OUT_PATH = os.path.join(os.path.dirname(__file__), "mass_results_4stage.md")

# ============================================================
#  1. 分布生成器（与 mass_test.py 保持一致）
# ============================================================


def integer_partitions(n: int, max_parts: int = 32, min_val: int = 1):
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
    return {i: v for i, v in enumerate(partition)}


def random_dist_uniform(
    M: int, rng: random.Random, max_experts: int = 16
) -> Dict[int, int]:
    n = rng.randint(1, min(M, max_experts))
    if n >= M:
        return {i: 1 for i in range(min(n, M))}
    separators = sorted(rng.sample(range(1, M), n - 1)) if n > 1 else []
    points = [0] + separators + [M]
    counts = [points[i + 1] - points[i] for i in range(len(points) - 1)]
    rng.shuffle(counts)
    return {i: c for i, c in enumerate(counts)}


def random_dist_zipf(
    M: int, rng: random.Random, s: float = 1.2, max_experts: int = 16
) -> Dict[int, int]:
    n = rng.randint(2, min(M, max_experts))
    weights = [1.0 / (k**s) for k in range(1, n + 1)]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]
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
    n_hot = rng.randint(1, 2)
    n_cold = rng.randint(0, min(8, max_experts - n_hot))
    cold_total = n_cold
    hot_total = M - cold_total
    if hot_total < n_hot:
        return random_dist_uniform(M, rng, max_experts)
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
    cases = []
    n_uniform = int(n_random * 0.40)
    n_zipf = int(n_random * 0.35)
    n_bi = n_random - n_uniform - n_zipf
    for i in range(n_uniform):
        cases.append((f"uniform_{i}", random_dist_uniform(M, rng, max_experts)))
    for i in range(n_zipf):
        s = rng.uniform(0.8, 1.6)
        cases.append(
            (f"zipf_{i}", random_dist_zipf(M, rng, s=s, max_experts=max_experts))
        )
    for i in range(n_bi):
        cases.append((f"bimodal_{i}", random_dist_bimodal(M, rng, max_experts)))
    rng.shuffle(cases)
    return cases


# ============================================================
#  2. 单次测试
# ============================================================


def run_one(dist: Dict[int, int], beam_width: int = 32) -> dict:
    rem = tuple(sorted(dist.items(), key=lambda x: -x[1]))
    lb = lb_remaining(rem)
    t0 = time.perf_counter()
    ms, hist = FourStageScheduler(dist, beam_width=beam_width).run()
    elapsed = time.perf_counter() - t0
    ratio = ms / lb if lb > 0 else 1.0
    eff = compute_efficiency(hist, ms)

    # 统计 S1≠S3 次数（独立 shape 被利用）
    n_diff = sum(
        1 for a in hist if a.c2_eid >= 0 and a.c2_shape_s1 != a.c2_shape_s3
    ) + sum(1 for a in hist if a.c3_eid >= 0 and a.c3_shape_s1 != a.c3_shape_s3)
    n_tasks = sum(1 for a in hist if a.c2_eid >= 0) + sum(
        1 for a in hist if a.c3_eid >= 0
    )

    return {
        "dist": dist,
        "M": sum(dist.values()),
        "n_exp": len(dist),
        "makespan": ms,
        "lb": lb,
        "ratio": ratio,
        "c2_vc_util": eff["c2_vc_util"],
        "c3_vc_util": eff["c3_vc_util"],
        "n_diff": n_diff,
        "n_tasks": n_tasks,
        "elapsed": elapsed,
    }


# ============================================================
#  3. 批量测试
# ============================================================


def batch_test(
    cases: List[Tuple[str, Dict[int, int]]],
    beam_width: int,
    label: str = "",
    print_progress: bool = True,
    progress_every: int = 20,
) -> List[dict]:
    results = []
    t_start = time.perf_counter()
    for idx, (name, dist) in enumerate(cases):
        r = run_one(dist, beam_width=beam_width)
        r["name"] = name
        results.append(r)
        if print_progress and (idx + 1) % progress_every == 0:
            el = time.perf_counter() - t_start
            mean_r = sum(x["ratio"] for x in results) / len(results)
            print(
                f"  [{label}] {idx+1}/{len(cases)}  mean_ratio={mean_r:.4f}  ({el:.1f}s)"
            )
    return results


# ============================================================
#  4. 统计
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
        "p99": v[min(int(n * 0.99), n - 1)],
        "max": v[-1],
        "pct_opt": sum(1 for x in v if x <= 1.001) / n,
    }


def analyze(results: List[dict], tag: str = "") -> dict:
    if not results:
        return {}
    ratios = [r["ratio"] for r in results]
    vc2 = [r["c2_vc_util"] for r in results]
    vc3 = [r["c3_vc_util"] for r in results]
    n_diff_t = sum(r["n_diff"] for r in results)
    n_task_t = sum(r["n_tasks"] for r in results)
    t_total = sum(r["elapsed"] for r in results)

    r_stat = stats(ratios)
    print(f"\n  [{tag}]  n={len(results)}  total_time={t_total:.1f}s")
    print(
        f"    ratio : mean={r_stat['mean']:.4f}  p25={r_stat['p25']:.4f}"
        f"  median={r_stat['median']:.4f}  p90={r_stat['p90']:.4f}"
        f"  max={r_stat['max']:.4f}  pct_optimal={r_stat['pct_opt']:.1%}"
    )
    vc_mean = (sum(vc2) + sum(vc3)) / (2 * len(results))
    print(f"    vc_util (C2+C3 avg): {vc_mean*100:.1f}%")
    if n_task_t > 0:
        pct_diff = n_diff_t / n_task_t
        print(f"    S1≠S3 利用率: {n_diff_t}/{n_task_t} tasks ({pct_diff:.1%})")
    return {
        "tag": tag,
        "n": len(results),
        "r_stat": r_stat,
        "vc_mean": vc_mean,
        "t_total": t_total,
    }


def hist_ratio(results: List[dict]) -> str:
    buckets = [1.001, 1.01, 1.05, 1.10, 1.20, 1.40, 1.70, 2.00, 9999.0]
    labels = [
        "≤1.001",
        "1.001-1.01",
        "1.01-1.05",
        "1.05-1.10",
        "1.10-1.20",
        "1.20-1.40",
        "1.40-1.70",
        "1.70-2.00",
        ">2.00",
    ]
    counts = [0] * len(buckets)
    for r in results:
        for i, b in enumerate(buckets):
            if r["ratio"] <= b:
                counts[i] += 1
                break
    n = len(results)
    lines = ["  Ratio 直方图:"]
    for lbl, cnt in zip(labels, counts):
        bar = "#" * int(cnt / n * 50)
        lines.append(f"    {lbl:16s}: {cnt:4d} ({cnt/n:5.1%}) {bar}")
    return "\n".join(lines)


def by_m_breakdown(results: List[dict]) -> str:
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        groups[r["M"]].append(r["ratio"])
    lines = ["  按 M 分组 (ratio):"]
    lines.append(
        f"  {'M':>6}  {'n':>5}  {'mean':>7}  {'median':>7}  {'p90':>7}  {'max':>7}  {'%opt':>7}"
    )
    for M in sorted(groups.keys()):
        rs = sorted(groups[M])
        n = len(rs)
        mn = sum(rs) / n
        md = rs[n // 2]
        p9 = rs[int(n * 0.9)]
        mx = rs[-1]
        pc = sum(1 for x in rs if x <= 1.001) / n
        lines.append(
            f"  {M:>6}  {n:>5}  {mn:>7.4f}  {md:>7.4f}  {p9:>7.4f}  {mx:>7.4f}  {pc:>6.1%}"
        )
    return "\n".join(lines)


# ============================================================
#  5. Beam Width 灵敏度（小子集）
# ============================================================


def beam_sensitivity(
    cases: List[Tuple[str, Dict[int, int]]], beam_widths: List[int]
) -> str:
    lines = ["  Beam Width 灵敏度 (ratio vs LB):"]
    header = f"  {'Case':25s}" + "".join(f"  beam={b:>3d}" for b in beam_widths)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for name, dist in cases:
        lb = lb_remaining(tuple(sorted(dist.items(), key=lambda x: -x[1])))
        row = f"  {name:25s}"
        for bw in beam_widths:
            t0 = time.perf_counter()
            ms, _ = FourStageScheduler(dist, beam_width=bw).run()
            el = time.perf_counter() - t0
            row += f"  {ms/lb:.4f}({el:.0f}s)"
        lines.append(row)
    return "\n".join(lines)


# ============================================================
#  6. 主程序
# ============================================================


def main():
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
    _fout.write("# 四阶段调度器大规模验证报告\n\n")
    _fout.write(f"> **生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
    _fout.write(
        "> **调度器**: FourStageScheduler v2 (S1/S3 独立 Shape, 精确 BW 验证, Beam Search)  \n"
    )
    _fout.write("> **主力配置**: beam_width=32 | 精确验证: beam_width=64  \n\n")
    _fout.write("```\n")
    sys.stdout = _Tee(_orig, _fout)

    rng = random.Random(2026)
    T_TOTAL = time.perf_counter()

    all_results_b32: List[dict] = []
    all_summaries: List[dict] = []

    # ══════════════════════════════════════════════════════════
    #  Phase 1: 穷举小 M (M=1..10)
    # ══════════════════════════════════════════════════════════
    print("=" * 72)
    print("  Phase 1: 穷举小 M (M=1..10) 所有整数分区")
    print("=" * 72)

    small_cases: List[Tuple[str, Dict[int, int]]] = []
    for M in range(1, 11):
        for p in integer_partitions(M, max_parts=16):
            small_cases.append((f"M{M}_part", partition_to_dist(p)))
    print(f"  穷举案例数: {len(small_cases)}\n")

    r_small = batch_test(
        small_cases,
        beam_width=32,
        label="phase1-b32",
        print_progress=True,
        progress_every=50,
    )
    s = analyze(r_small, "Phase1 beam=32")
    all_results_b32 += r_small
    all_summaries.append(s)
    print(hist_ratio(r_small))
    print(by_m_breakdown(r_small))

    # ══════════════════════════════════════════════════════════
    #  Phase 2: 随机 M=16 (30 例)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  Phase 2: 随机 M=16 (30 例)")
    print("=" * 72)

    m16_cases = generate_cases(M=16, n_random=30, rng=rng)
    r_m16 = batch_test(
        m16_cases,
        beam_width=32,
        label="M16-b32",
        print_progress=True,
        progress_every=10,
    )
    s = analyze(r_m16, "M=16 beam=32")
    all_results_b32 += r_m16
    all_summaries.append(s)
    print(hist_ratio(r_m16))

    # beam=64 子集对比（仅 10 例）
    print("\n  -- 精度对比: beam=64 (10 例子集) --")
    r_m16_b64 = batch_test(
        m16_cases[:10], beam_width=64, label="M16-b64", print_progress=False
    )
    analyze(r_m16_b64, "M=16 beam=64 (10例)")

    # ══════════════════════════════════════════════════════════
    #  Phase 3: 随机 M=32 (15 例)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  Phase 3: 随机 M=32 (15 例)")
    print("=" * 72)

    m32_cases = generate_cases(M=32, n_random=15, rng=rng)
    r_m32 = batch_test(
        m32_cases, beam_width=32, label="M32-b32", print_progress=True, progress_every=5
    )
    s = analyze(r_m32, "M=32 beam=32")
    all_results_b32 += r_m32
    all_summaries.append(s)
    print(hist_ratio(r_m32))

    # ══════════════════════════════════════════════════════════
    #  Phase 4: 目标分布 M=64 (5 例 + 原始分布)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  Phase 4: M=64 (5 随机 + 1 原始目标分布)")
    print("=" * 72)

    TARGET_DIST = {
        0: 16,
        1: 11,
        2: 8,
        3: 7,
        4: 5,
        5: 4,
        6: 3,
        7: 3,
        8: 2,
        9: 2,
        10: 1,
        11: 1,
        12: 1,
    }
    m64_rnd = generate_cases(M=64, n_random=5, rng=rng)
    m64_cases = [("target_M64", TARGET_DIST)] + m64_rnd
    r_m64 = batch_test(
        m64_cases, beam_width=32, label="M64-b32", print_progress=True, progress_every=3
    )
    s = analyze(r_m64, "M=64 beam=32")
    all_results_b32 += r_m64
    all_summaries.append(s)

    # 目标分布详细结果
    tgt = r_m64[0]
    print(f"\n  ★ 目标分布 (M=64, 13 experts):")
    print(f"    makespan = {tgt['makespan']:,} cc")
    print(f"    LB       = {tgt['lb']:,} cc")
    print(f"    ratio    = {tgt['ratio']:.4f}")
    print(f"    C2 util  = {tgt['c2_vc_util']*100:.1f}%")
    print(f"    C3 util  = {tgt['c3_vc_util']*100:.1f}%")
    print(f"    S1≠S3    = {tgt['n_diff']}/{tgt['n_tasks']} tasks")

    # ══════════════════════════════════════════════════════════
    #  Phase 5: Beam Width 灵敏度
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  Phase 5: Beam Width 灵敏度分析")
    print("=" * 72)

    sensitivity_cases = [
        ("M8_hot3_3_2", {0: 3, 1: 3, 2: 2}),
        ("M16_zipf", {0: 8, 1: 4, 2: 2, 3: 1, 4: 1}),
        ("M64_target", TARGET_DIST),
    ]
    print(beam_sensitivity(sensitivity_cases, beam_widths=[1, 8, 16, 32, 64, 128]))

    # ══════════════════════════════════════════════════════════
    #  综合汇总
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  综合汇总 (全部 beam=32 结果)")
    print("=" * 72)

    analyze(all_results_b32, "ALL beam=32")
    print(hist_ratio(all_results_b32))
    print(by_m_breakdown(all_results_b32))

    # S1≠S3 总统计
    total_diff = sum(r["n_diff"] for r in all_results_b32)
    total_tasks = sum(r["n_tasks"] for r in all_results_b32)
    print(
        f"\n  S1/S3 独立 Shape 利用总计: {total_diff}/{total_tasks} ({total_diff/total_tasks*100:.1f}%)"
    )

    T_elapsed = time.perf_counter() - T_TOTAL
    print(f"\n  总运行时间: {T_elapsed:.1f}s ({T_elapsed/60:.1f} min)")
    print(f"  输出文件: {OUT_PATH}")

    _fout.write("```\n")
    _fout.close()
    sys.stdout = _orig
    print(f"\n✓ 报告已写入: {OUT_PATH}")


if __name__ == "__main__":
    main()
