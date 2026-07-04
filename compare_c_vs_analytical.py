#!/usr/bin/env python3
"""
compare_c_vs_analytical.py
==========================
比较 C 贪心 LPT 调度器（Python 重实现，含 S2 prefetch）和 Python 分析法调度器
在 10000 个随机分布上的 makespan 差距。

使用 ProcessPoolExecutor 并行化 analytical_schedule 调用（每 call ~1.8s）。
36 核约 8 分钟完成 10000 个测试。

运行：
  cd /esat/studscratch/r1015673/Thesis
  python3 Idea_Model/compare_c_vs_analytical.py [N_TESTS]

环境变量：
  N_WORKERS  (默认 = cpu_count，最多 36)
"""

import sys
import os
import random
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── C 贪心调度器（Python 重实现，镜像 moe_scheduler.c 含 Shape 感知 + S2 prefetch）─
_T_S1 = 90112  # ShapeA S1 (DMA+compute, 1-token window)
_T_S3 = 45056  # ShapeA S3 (DMA+compute, 1-token window)
_T_C_S1 = 22528  # ShapeC: per 2-token chunk S1/S2 compute; also ShapeA S1 DMA end
_T_C_S3 = 11264  # ShapeC: per 2-token chunk S3/S4 compute
_SHAPE_M = 8  # ShapeA M_dim
_T_DMA_S3 = 22528  # ShapeA alloc=64 down-weight DMA duration (S2 prefetch window)
_T_DMA_S1_A = 45056  # ShapeA alloc=64 gate+up DMA duration
# ShapeC DMA: gate+up = _T_C_S1 = 22528; down = _T_C_S3 = 11264


def _c_dur(ntok: int, hit: bool, avail_full_bw: bool):
    """单任务时长 + S1 DMA 结束时刻（相对于 t_start）。
    镜像 moe_scheduler.c 的 shape-aware 逻辑。
    返回 (dur, dma1_rel)
    """
    ntok = int(ntok)
    if hit:
        # cache hit: S1/S3 跳过，S2+S4 处理全部 ntok（ShapeC 公式）
        dur = ((ntok + 1) // 2) * (_T_C_S1 + _T_C_S3)
        return dur, 0  # 无 DMA
    if avail_full_bw:
        # ShapeC: ceil(ntok/2)*(T_C_S1+T_C_S3)，S1 DMA = T_C_S1
        dur = ((ntok + 1) // 2) * (_T_C_S1 + _T_C_S3)
        return dur, _T_C_S1
    # ShapeA with S2 down-weight prefetch check
    tail = max(0, ntok - _SHAPE_M)
    s2_win = ((tail + 1) // 2) * _T_C_S1
    s2_pf = s2_win >= _T_DMA_S3
    if s2_pf:
        t_s4 = ((ntok + 1) // 2) * _T_C_S3
        dur = _T_S1 + s2_win + t_s4
    else:
        t_s4 = ((tail + 1) // 2) * _T_C_S3
        dur = _T_S1 + s2_win + _T_S3 + t_s4
    return dur, _T_DMA_S1_A  # ShapeA S1 DMA = 45056


def c_greedy_schedule(token_dist: dict, cache_c2: int = -1, cache_c3: int = -1) -> int:
    """贪心 LPT 调度器（moe_scheduler.c 的 Python 重实现，含 shape 感知）。
    返回 makespan（cc）。
    """
    sorted_experts = sorted(token_dist.items(), key=lambda x: -x[1])
    free_at = [0, 0]  # [C2, C3]: task end time
    dma_free = [0, 0]  # [C2, C3]: S1 DMA end time
    cache = [cache_c2, cache_c3]
    for eid, ntok in sorted_experts:
        if ntok <= 0:
            continue
        ci = 0 if free_at[0] <= free_at[1] else 1
        other_ci = 1 - ci
        hit = cache[ci] == eid
        t_start = free_at[ci]
        # 对侧 S1 DMA 已结束 → 本侧可独占 128 B/cc → ShapeC
        avail_full = dma_free[other_ci] <= t_start
        dur, dma1_rel = _c_dur(ntok, hit, avail_full)
        free_at[ci] = t_start + dur
        dma_free[ci] = t_start + dma1_rel
    return max(free_at[0], free_at[1])


# ─── 分布生成器（与 eval_analytical_10k.py 相同逻辑，但用 seed 固定）──────────────
def _gen_test(seed: int):
    rng = random.Random(seed)
    M = rng.choice([4, 8, 12, 16, 24, 32])
    n = rng.randint(1, min(8, M))
    dist_type = rng.choices(
        ["uniform", "zipf", "hot", "bimodal", "single"], weights=[2, 4, 3, 2, 1]
    )[0]

    if dist_type == "single":
        toks = [M]
    elif dist_type == "uniform":
        base = M // n
        toks = [base] * n
        toks[0] += M - sum(toks)
    elif dist_type == "zipf":
        alpha = rng.uniform(0.6, 2.5)
        w = [1.0 / (i + 1) ** alpha for i in range(n)]
        s = sum(w)
        toks = [max(1, round(wi / s * M)) for wi in w]
        toks.sort(reverse=True)
        toks[0] += M - sum(toks)
    elif dist_type == "hot":
        hot = max(1, round(M * rng.uniform(0.35, 0.80)))
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
    dist = {i: t for i, t in enumerate(toks) if t > 0}

    # 缓存状态：前两个最重专家分别有 25% 概率命中
    items = sorted(dist.items(), key=lambda x: -x[1])
    c2 = items[0][0] if rng.random() < 0.25 else -1
    c3 = items[1][0] if len(items) >= 2 and rng.random() < 0.25 else -1
    return dist, c2, c3


# ─── Worker：在独立进程中运行单个测试─────────────────────────────────────────────
def _worker(seed: int):
    from analytical_scheduler import analytical_schedule

    dist, c2, c3 = _gen_test(seed)
    c_ms = c_greedy_schedule(dist, c2, c3)
    try:
        a_ms = analytical_schedule(dist, c2, c3)
    except Exception:
        a_ms = -1
    ntoks = sorted(dist.values(), reverse=True)[:5]
    return seed, c_ms, a_ms, ntoks, c2, c3


# ─── 主程序─────────────────────────────────────────────────────────────────────
def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    N_WORKERS = int(os.environ.get("N_WORKERS", str(min(mp.cpu_count(), 36))))
    OUT_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "compare_c_vs_analytical.txt"
    )

    print(f"C贪心 vs 分析法 对比测试  N={N}  workers={N_WORKERS}")
    print(f"结果将写入: {OUT_FILE}")
    print("=" * 70)

    results = []  # list of (c_ms, a_ms, ntoks, c2, c3)
    errors = 0
    t0 = time.time()
    done = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_worker, s): s for s in range(N)}
        for fut in as_completed(futures):
            try:
                seed, c_ms, a_ms, ntoks, c2, c3 = fut.result()
                if a_ms > 0:
                    results.append((c_ms, a_ms, ntoks, c2, c3))
                else:
                    errors += 1
            except Exception:
                errors += 1
            done += 1
            if done % 500 == 0:
                elapsed = time.time() - t0
                eta = elapsed / done * (N - done) if done > 0 else 0
                nvalid = len(results)
                mean_r = (
                    sum(c / a for c, a, *_ in results) / nvalid if nvalid > 0 else 0.0
                )
                print(
                    f"  [{done:5d}/{N}]  valid={nvalid}  mean_ratio={mean_r:.4f}"
                    f"  elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
                )

    elapsed = time.time() - t0
    N_valid = len(results)
    print(
        f"\n完成: {N_valid} 有效  {errors} 错误  总耗时 {elapsed:.1f}s  "
        f"({elapsed/max(N_valid, 1)*1000:.0f}ms/test)"
    )

    if N_valid == 0:
        print("无有效结果！")
        return

    # ── 统计─────────────────────────────────────────────────────────────────
    ratios = sorted(c / a for c, a, *_ in results)
    mean_r = sum(ratios) / N_valid

    def pct(p):
        return ratios[min(int(p / 100 * N_valid), N_valid - 1)]

    def cnt(lo, hi):
        return sum(1 for r in ratios if lo <= r < hi)

    lines = []
    lines.append(f"C贪心 / 分析法  对比报告  (N_valid={N_valid}  N_errors={errors})")
    lines.append("=" * 70)
    lines.append(f"均值比率:      {mean_r:.4f}  ({(mean_r-1)*100:+.2f}%)")
    lines.append(f"中位数比率:    {pct(50):.4f}  ({(pct(50)-1)*100:+.2f}%)")
    lines.append(f"P75:           {pct(75):.4f}  ({(pct(75)-1)*100:+.2f}%)")
    lines.append(f"P90:           {pct(90):.4f}  ({(pct(90)-1)*100:+.2f}%)")
    lines.append(f"P95:           {pct(95):.4f}  ({(pct(95)-1)*100:+.2f}%)")
    lines.append(f"P99:           {pct(99):.4f}  ({(pct(99)-1)*100:+.2f}%)")
    lines.append(f"最差比率:      {ratios[-1]:.4f}  ({(ratios[-1]-1)*100:+.2f}%)")
    lines.append("")
    lines.append("比率分布（C贪心 / 分析法，>1 表示 C 更差）:")
    lines.append(
        f"  C 更好或相同 (ratio ≤ 1.0):    {cnt(0, 1.000001):6d}  ({cnt(0,1.000001)/N_valid*100:5.1f}%)"
    )
    lines.append(
        f"  < 5%  更差   (1.00~1.05):      {cnt(1.000001, 1.05):6d}  ({cnt(1.000001,1.05)/N_valid*100:5.1f}%)"
    )
    lines.append(
        f"  5-10% 更差   (1.05~1.10):      {cnt(1.05, 1.10):6d}  ({cnt(1.05,1.10)/N_valid*100:5.1f}%)"
    )
    lines.append(
        f"  10-20% 更差  (1.10~1.20):      {cnt(1.10, 1.20):6d}  ({cnt(1.10,1.20)/N_valid*100:5.1f}%)"
    )
    lines.append(
        f"  20-50% 更差  (1.20~1.50):      {cnt(1.20, 1.50):6d}  ({cnt(1.20,1.50)/N_valid*100:5.1f}%)"
    )
    lines.append(
        f"  > 50%  更差  (ratio ≥ 1.50):   {cnt(1.50, 999):6d}  ({cnt(1.50,999)/N_valid*100:5.1f}%)"
    )
    lines.append("")
    bad_cases = sorted(
        [(c / a, c, a, ntoks, c2, c3) for c, a, ntoks, c2, c3 in results], reverse=True
    )[:10]
    lines.append("Top-10 最差案例:")
    for rank, (ratio, c_ms, a_ms, ntoks, c2, c3) in enumerate(bad_cases):
        lines.append(
            f"  [{rank+1:2d}] ratio={ratio:.3f}  C={c_ms:8d}  anal={a_ms:8d}"
            f"  ntoks={ntoks}  c2={c2}  c3={c3}"
        )

    output = "\n".join(lines)
    print("\n" + output)
    with open(OUT_FILE, "w") as f:
        f.write(output + "\n")
    print(f"\n结果已写入 {OUT_FILE}")


if __name__ == "__main__":
    main()
