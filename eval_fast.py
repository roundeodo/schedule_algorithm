#!/usr/bin/env python3
"""
eval_fast.py
============
加载预计算的 analytical 缓存，对任意调度器做秒级质量对比。
不需要重跑 analytical，10000 case 全部对比只需 ~30 秒（取决于 lite 速度）。

用法：
  # 先确保 analytical_cache.json 存在（precompute_analytical.py 生成）
  cd /esat/studscratch/r1015673/Thesis
  python3 Idea_Model/eval_fast.py              # 默认测 lite_scheduler
  python3 Idea_Model/eval_fast.py fast         # 测 fast_scheduler（解析版）

输出：ratio 分布统计 + 最差前10个 case
"""

import sys, time, os, json

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CACHE_PATH = os.path.join(os.path.dirname(__file__), "analytical_cache.json")


def load_cache():
    if not os.path.exists(CACHE_PATH):
        print(f"缓存文件不存在：{CACHE_PATH}")
        print("请先运行：python3 Idea_Model/precompute_analytical.py")
        sys.exit(1)
    with open(CACHE_PATH) as f:
        data = json.load(f)
    print(f"加载缓存：{len(data)} 条 analytical 结果")
    return data


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "lite"

    if mode == "fast":
        from fast_scheduler import fast_schedule as schedule

        scheduler_name = "fast_scheduler"
    else:
        from lite_scheduler import lite_schedule as schedule

        scheduler_name = "lite_scheduler"

    data = load_cache()
    n_total = len(data)

    ratios = []
    times_l = []
    crashes = 0

    print(f"测试 {scheduler_name} 对 {n_total} 个 case……", flush=True)
    t_wall0 = time.perf_counter()

    for rec in data:
        dist = {int(k): v for k, v in rec["dist"].items()}
        c2 = rec["c2"]
        c3 = rec["c3"]
        a = rec["analytical"]

        try:
            t0 = time.perf_counter()
            l = schedule(dict(dist), c2, c3)
            times_l.append(time.perf_counter() - t0)
        except Exception:
            crashes += 1
            continue

        if a > 0:
            ratios.append((l / a, dist, c2, c3, a, l))

    n = len(ratios)
    if n == 0:
        print("无有效数据")
        sys.exit(1)

    rv = [r[0] for r in ratios]
    elapsed = time.perf_counter() - t_wall0
    avg_l = sum(times_l) / len(times_l) * 1000 if times_l else 0

    print(f"\n{'='*55}")
    print(f"调度器:                  {scheduler_name}")
    print(f"有效对比:                {n}")
    print(f"crashes:                 {crashes}")
    print(f"均值 ratio:              {sum(rv)/n:.4f}")
    print(f"最大 ratio:              {max(rv):.4f}")
    print(f"中位数 ratio:            {sorted(rv)[n//2]:.4f}")
    print(f"pct 完全相同 (≤1.001):   {sum(1 for r in rv if r<=1.001)/n*100:.1f}%")
    print(f"pct <2%    (≤1.020):    {sum(1 for r in rv if r<=1.020)/n*100:.1f}%")
    print(f"pct <5%    (≤1.050):    {sum(1 for r in rv if r<=1.050)/n*100:.1f}%")
    print(f"pct <10%   (≤1.100):    {sum(1 for r in rv if r<=1.100)/n*100:.1f}%")
    print(f"lite 均时:               {avg_l:.1f}ms")
    print(f"eval 总耗时:             {elapsed:.1f}s")
    print(f"{'='*55}")

    worst = sorted(ratios, key=lambda x: x[0], reverse=True)[:10]
    if worst[0][0] > 1.001:
        print("\n差距最大的前10个案例:")
        for r, dist, c2, c3, a, l in worst:
            toks = sorted(dict(dist).values(), reverse=True)
            print(f"  ratio={r:.3f} toks={toks} c2={c2} c3={c3} anal={a} lite={l}")


if __name__ == "__main__":
    main()
