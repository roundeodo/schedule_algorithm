#!/usr/bin/env python3
"""
precompute_analytical.py
========================
一次性计算 10000 组随机输入的 analytical_schedule 结果，保存到磁盘。
之后用 eval_fast.py 对任意新调度器做对比，只需跑 lite 不需重跑 analytical。

用法：
  cd /esat/studscratch/r1015673/Thesis
  python3 Idea_Model/precompute_analytical.py
  （约 5-6 小时，只需跑一次）
"""

import sys, random, time, os, json

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analytical_scheduler import analytical_schedule

SAVE_PATH = os.path.join(os.path.dirname(__file__), "analytical_cache.json")
N = 10000


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


def main():
    # 检查是否已有部分结果
    existing = []
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            existing = json.load(f)
        print(f"已有 {len(existing)} 条缓存，继续从断点恢复……")

    existing_seeds = {r["seed"] for r in existing}
    results = list(existing)

    print(f"目标：{N} 条，当前 {len(results)} 条，还需 {N - len(results)} 条")
    t0 = time.perf_counter()
    last_save = time.perf_counter()

    for i in range(N * 3):
        if len(results) >= N:
            break
        if i in existing_seeds:
            continue

        rng_seed = random.Random(i * 997 + 13)
        dist_raw = random_dist(seed=i)
        keys = list(dist_raw.keys())
        c2 = keys[0] if rng_seed.random() < 0.6 else -1
        c3 = keys[1] if len(keys) >= 2 and rng_seed.random() < 0.4 else -1

        try:
            a = analytical_schedule(dict(dist_raw), c2, c3)
        except Exception as e:
            continue

        if a <= 0:
            continue

        results.append(
            {
                "seed": i,
                "dist": {str(k): v for k, v in dist_raw.items()},
                "c2": c2,
                "c3": c3,
                "analytical": a,
            }
        )

        # 每 100 条保存一次（断点续传）
        if len(results) % 100 == 0:
            with open(SAVE_PATH, "w") as f:
                json.dump(results, f)
            elapsed = time.perf_counter() - t0
            rate = len(results) / elapsed
            eta = (N - len(results)) / rate if rate > 0 else 0
            print(
                f"  {len(results)}/{N} ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)",
                flush=True,
            )
            last_save = time.perf_counter()

    # 最终保存
    with open(SAVE_PATH, "w") as f:
        json.dump(results[:N], f)
    elapsed = time.perf_counter() - t0
    print(f"\n完成！共 {len(results[:N])} 条，耗时 {elapsed:.0f}s，保存到 {SAVE_PATH}")


if __name__ == "__main__":
    main()
