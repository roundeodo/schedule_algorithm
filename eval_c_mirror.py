#!/usr/bin/env python3
"""
eval_c_mirror.py
================
精确镜像 moe_scheduler.c 简化后逻辑的 Python 评估脚本。

关键简化（vs lite_scheduler.py）：
  - pick_shapes(): 解析式形状选择，O(1)，与 C 版完全一致
  - 候选时间点：idle_t + bw_change_pts 中至多 2 个点（vs lite 的 7+ 个点）
    → 对应 C 代码 snap_segs 返回的 lo/hi 边界

用已有 analytical_cache.json（10000 条）直接对比，不重跑 analytical。
"""

import sys, os, json, time

sys.path.insert(0, os.path.dirname(__file__))

from four_stage_scheduler import (
    SHAPE_A,
    SHAPE_B,
    SHAPE_C,
    ALL_SHAPES,
    MAX_BW,
    PF_EID_GHOST,
    FourStageSnap,
    make_initial_snap,
    bw_feasible,
    with_optional_s2_down_prefetch,
    with_optional_s2_down_prefetch_pair,
    with_optional_next_s1_prefetch_pair,
    inject_ghost_prefetch_pair,
    _best_task_time,
    _best_concurrent_task_time,
)
from analytical_scheduler import _greedy_heuristic

CACHE_PATH = os.path.join(os.path.dirname(__file__), "analytical_cache.json")

# ─────────────────────────────────────────────────────────────────────────────
# 常量（与 C 代码完全一致）
# ─────────────────────────────────────────────────────────────────────────────
_T_DMA_S3_C = SHAPE_C.T_s3  # dma3 duration for ShapeC = 11264


def _best_s2_compute(ntok: int) -> int:
    """最优 S2 计算时间（ceil(ntok/2) * 22528）"""
    return ((ntok + 1) // 2) * 22528


# ─────────────────────────────────────────────────────────────────────────────
# pick_shapes: 镜像 C 版 pick_shapes()
# ─────────────────────────────────────────────────────────────────────────────
def pick_shapes(ntok_A, ntok_B, sw_A, dn_A, sw_B, dn_B, t_now):
    """返回 (s1_A, s3_A, s1_B, s3_B)，O(1)，与 moe_scheduler.c pick_shapes 完全一致。"""
    # S1 形状
    if sw_A or sw_B:
        s1_A = s1_B = SHAPE_C
    else:
        s1_A = s1_B = SHAPE_B

    # 解析 s2_end
    if sw_A:
        s2_A = t_now + _best_s2_compute(ntok_A)
    else:
        s2_A = t_now + s1_A.T_s1 + _best_s2_compute(max(0, ntok_A - s1_A.M_dim))

    if sw_B:
        s2_B = t_now + _best_s2_compute(ntok_B)
    else:
        s2_B = t_now + s1_B.T_s1 + _best_s2_compute(max(0, ntok_B - s1_B.M_dim))

    # S3 形状
    if dn_A and dn_B:
        s3_A, s3_B = SHAPE_B, SHAPE_B
    elif dn_A:
        s3_A, s3_B = SHAPE_B, SHAPE_C
    elif dn_B:
        s3_A, s3_B = SHAPE_C, SHAPE_B
    elif abs(s2_A - s2_B) >= _T_DMA_S3_C:
        s3_A, s3_B = SHAPE_C, SHAPE_C
    else:
        s3_A, s3_B = SHAPE_B, SHAPE_B

    return s1_A, s3_A, s1_B, s3_B


# ─────────────────────────────────────────────────────────────────────────────
# bw_seg_pts: 从 snap 提取 BW 边界时间点（镜像 C 版 snap_segs lo/hi）
# 最多返回 4 个点（2 段 × lo/hi）
# ─────────────────────────────────────────────────────────────────────────────
def bw_seg_pts(sn: FourStageSnap) -> list:
    """返回 snap 的 DMA 活跃区间边界点，对应 C snap_segs 的 lo/hi。"""
    pts = set()
    # S1 DMA: [task_start, dma1_end)
    if sn.bw_s1 > 0:
        pts.add(sn.task_start)
        pts.add(sn.dma1_end)
    # S2 down-prefetch: [s2pf_start, s2pf_end)
    if sn.s2pf_start >= 0 and sn.s2pf_bw > 0:
        pts.add(sn.s2pf_start)
        pts.add(sn.s2pf_end)
    # S3 DMA: [s2_end, dma3_end)
    if sn.bw_s3 > 0:
        pts.add(sn.s2_end)
        pts.add(sn.dma3_end)
    return sorted(pts)


# ─────────────────────────────────────────────────────────────────────────────
# 主调度函数（镜像 moe_scheduler.c 简化逻辑）
# ─────────────────────────────────────────────────────────────────────────────
def c_mirror_schedule(token_dist: dict, c2_eid: int = -1, c3_eid: int = -1) -> int:
    """
    镜像简化后 moe_scheduler.c 的调度逻辑：
    - pick_shapes: 解析式 O(1) 形状选择
    - 候选时间点：idle_t + bw_seg_pts 中 >= idle_t 的点（至多 ~4 个）
    """
    remaining = tuple(sorted(token_dist.items(), key=lambda x: x[1], reverse=True))
    c2 = make_initial_snap(c2_eid)
    c3 = make_initial_snap(c3_eid)

    def swiglu_hit(eid, snap, t):
        if snap.pf_end < 0 or snap.pf_end > t:
            return False
        return snap.pf_eid == PF_EID_GHOST or snap.pf_eid == eid

    def down_hit(eid, snap, t):
        return swiglu_hit(eid, snap, t) and snap.pf_full

    max_iters = len(remaining) * 4 + 10
    iters = 0

    while remaining and iters < max_iters:
        iters += 1
        c2, c3 = inject_ghost_prefetch_pair(c2, c3)
        t2, t3 = c2.task_end, c3.task_end
        both_idle = t2 == t3
        now = max(t2, t3)
        n = len(remaining)
        top0_eid, top0_ntok = remaining[0]

        # ── n == 1 ──────────────────────────────────────────────────────────
        if n == 1:
            c2_sw = swiglu_hit(top0_eid, c2, now)
            c2_dn = down_hit(top0_eid, c2, now)
            c3_sw = swiglu_hit(top0_eid, c3, now)
            c3_dn = down_hit(top0_eid, c3, now)
            best_cost = None
            best_result = None  # ('c2', sn) | ('c3', sn) | ('split', sna, snb)

            # Method A: Solo on each cluster, enumerate 9 (s1,s3) shape combos
            # (mirrors C code: for s1=0..2 for s3=0..2 + bw_ok against peer)
            for cc, cf, cl, t_start, other_sn in [
                (c2_sw, c2_dn, "c2", t2, c3),
                (c3_sw, c3_dn, "c3", t3, c2),
            ]:
                for s1 in ALL_SHAPES:
                    for s3 in ALL_SHAPES:
                        try:
                            sn = FourStageSnap.from_assign(
                                t_start, s1, s3, top0_ntok, top0_eid, cc, cf
                            )
                            sn = with_optional_s2_down_prefetch(sn, s3, other_sn)
                            ok = (
                                bw_feasible(sn, other_sn)
                                if cl == "c2"
                                else bw_feasible(other_sn, sn)
                            )
                            if not ok:
                                continue
                            ms = max(sn.task_end, other_sn.task_end)
                            if best_cost is None or ms < best_cost:
                                best_cost = ms
                                best_result = (cl, sn)
                        except Exception:
                            pass

            # SPLIT
            if top0_ntok >= 2:
                cut = (top0_ntok + 1) // 2
                rest = top0_ntok - cut
                s1a, s3a, s1b, s3b = pick_shapes(cut, rest, c2_sw, c2_dn, c3_sw, c3_dn, now)
                try:
                    sna = FourStageSnap.from_assign(
                        now, s1a, s3a, cut, top0_eid, c2_sw, c2_dn
                    )
                    snb = FourStageSnap.from_assign(
                        now, s1b, s3b, rest, top0_eid, c3_sw, c3_dn
                    )
                    sna, snb = with_optional_s2_down_prefetch_pair(sna, s3a, snb, s3b)
                    if bw_feasible(sna, snb):
                        cost = max(sna.task_end, snb.task_end)
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                            best_result = ("split", sna, snb)
                except Exception:
                    pass

            # Method B: early-start on idle cluster (not_both_idle)
            if not both_idle:
                idle_t = min(t2, t3)
                if t2 < t3:
                    idle_sn, busy_sn, idle_cl = c2, c3, "c2"
                    cc_idle, cf_idle = c2_sw, c2_dn
                else:
                    idle_sn, busy_sn, idle_cl = c3, c2, "c3"
                    cc_idle, cf_idle = c3_sw, c3_dn

                # 候选时间点：idle_t + BW 边界点（至多 ~4 个，对应 C snap_segs）
                cand_pts = sorted(t for t in bw_seg_pts(busy_sn) if t >= idle_t)
                try_starts = [idle_t] + cand_pts[:3]  # 最多 4 个点

                # Method B: use ShapeC (fastest), BW conflict → next time point
                # (mirrors C code: s1=2, s3=2 fixed)
                for t_st in try_starts:
                    cc = swiglu_hit(top0_eid, idle_sn, t_st)
                    cf = down_hit(top0_eid, idle_sn, t_st)
                    try:
                        new_sn = FourStageSnap.from_assign(
                            t_st, SHAPE_C, SHAPE_C, top0_ntok, top0_eid, cc, cf
                        )
                        ok = (
                            bw_feasible(new_sn, busy_sn)
                            if idle_cl == "c2"
                            else bw_feasible(busy_sn, new_sn)
                        )
                        if ok:
                            cost = max(new_sn.task_end, busy_sn.task_end)
                            if best_cost is None or cost < best_cost:
                                best_cost = cost
                                best_result = (idle_cl, new_sn)
                    except Exception:
                        pass

            if best_result is None:
                break
            if best_result[0] == "split":
                c2, c3 = best_result[1], best_result[2]
            elif best_result[0] == "c2":
                c2 = best_result[1]
            else:
                c3 = best_result[1]
            remaining = ()
            break

        # ── n >= 2, both_idle: PAIR(top0,top1) + SPLIT(top0) ───────────────
        elif both_idle:
            top1_eid, top1_ntok = remaining[1]
            new_rem_pair = tuple(
                r for r in remaining if r[0] != top0_eid and r[0] != top1_eid
            )
            new_rem_split = remaining[1:]

            c2c0 = swiglu_hit(top0_eid, c2, now)
            c2f0 = down_hit(top0_eid, c2, now)
            c3c0 = swiglu_hit(top0_eid, c3, now)
            c3f0 = down_hit(top0_eid, c3, now)
            c2c1 = swiglu_hit(top1_eid, c2, now)
            c2f1 = down_hit(top1_eid, c2, now)
            c3c1 = swiglu_hit(top1_eid, c3, now)
            c3f1 = down_hit(top1_eid, c3, now)

            best_cost = None
            best_which = None

            # PAIR 方向1: top0→C2, top1→C3
            s1a, s3a, s1b, s3b = pick_shapes(
                top0_ntok, top1_ntok, c2c0, c2f0, c3c1, c3f1, now
            )
            try:
                sna = FourStageSnap.from_assign(
                    now, s1a, s3a, top0_ntok, top0_eid, c2c0, c2f0
                )
                snb = FourStageSnap.from_assign(
                    now, s1b, s3b, top1_ntok, top1_eid, c3c1, c3f1
                )
                sna, snb = with_optional_s2_down_prefetch_pair(sna, s3a, snb, s3b)
                if bw_feasible(sna, snb):
                    cost = _greedy_heuristic(sna.task_end, snb.task_end, new_rem_pair)
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_which = ("pair", sna, snb, new_rem_pair)
            except Exception:
                pass

            # PAIR 方向2: top1→C2, top0→C3
            s1a, s3a, s1b, s3b = pick_shapes(
                top1_ntok, top0_ntok, c2c1, c2f1, c3c0, c3f0, now
            )
            try:
                sna = FourStageSnap.from_assign(
                    now, s1a, s3a, top1_ntok, top1_eid, c2c1, c2f1
                )
                snb = FourStageSnap.from_assign(
                    now, s1b, s3b, top0_ntok, top0_eid, c3c0, c3f0
                )
                sna, snb = with_optional_s2_down_prefetch_pair(sna, s3a, snb, s3b)
                if bw_feasible(sna, snb):
                    cost = _greedy_heuristic(sna.task_end, snb.task_end, new_rem_pair)
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_which = ("pair", sna, snb, new_rem_pair)
            except Exception:
                pass

            # SPLIT top0
            if top0_ntok >= 2:
                cut = (top0_ntok + 1) // 2
                rest = top0_ntok - cut
                s1a, s3a, s1b, s3b = pick_shapes(cut, rest, c2c0, c2f0, c3c0, c3f0, now)
                try:
                    sna = FourStageSnap.from_assign(
                        now, s1a, s3a, cut, top0_eid, c2c0, c2f0
                    )
                    snb = FourStageSnap.from_assign(
                        now, s1b, s3b, rest, top0_eid, c3c0, c3f0
                    )
                    sna, snb = with_optional_s2_down_prefetch_pair(sna, s3a, snb, s3b)
                    if bw_feasible(sna, snb):
                        cost = _greedy_heuristic(
                            sna.task_end, snb.task_end, new_rem_split
                        )
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                            best_which = ("pair", sna, snb, new_rem_split)
                except Exception:
                    pass

            if best_which is None:
                break
            _, sna, snb, new_rem = best_which
            c2, c3 = sna, snb
            remaining = new_rem

        # ── n >= 2, not_both_idle ────────────────────────────────────────────
        else:
            idle_t = min(t2, t3)
            if t2 < t3:
                idle_sn, busy_sn, idle_cl = c2, c3, "c2"
            else:
                idle_sn, busy_sn, idle_cl = c3, c2, "c3"

            sw_a = swiglu_hit(top0_eid, idle_sn, idle_t)
            dn_a = down_hit(top0_eid, idle_sn, idle_t)
            sw_b = swiglu_hit(top0_eid, busy_sn, now)
            dn_b = down_hit(top0_eid, busy_sn, now)

            new_rem = remaining[1:]
            best_cost = None
            best_which = None

            # Solo on idle cluster at BW boundary points
            cand_pts = sorted(t for t in bw_seg_pts(busy_sn) if t >= idle_t)
            try_starts = [idle_t] + cand_pts[:3]

            for t_st in try_starts:
                cc = swiglu_hit(top0_eid, idle_sn, t_st)
                cf = down_hit(top0_eid, idle_sn, t_st)
                try:
                    new_sn = FourStageSnap.from_assign(
                        t_st, SHAPE_C, SHAPE_C, top0_ntok, top0_eid, cc, cf
                    )
                    ok = (
                        bw_feasible(new_sn, busy_sn)
                        if idle_cl == "c2"
                        else bw_feasible(busy_sn, new_sn)
                    )
                    if ok:
                        new_c2 = new_sn if idle_cl == "c2" else busy_sn
                        new_c3 = busy_sn if idle_cl == "c2" else new_sn
                        cost = _greedy_heuristic(
                            new_c2.task_end, new_c3.task_end, new_rem
                        )
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                            best_which = (new_c2, new_c3, new_rem)
                except Exception:
                    pass

            # Solo on busy cluster (fallback: start at now, use ShapeC)
            try:
                cc = swiglu_hit(top0_eid, busy_sn, now)
                cf = down_hit(top0_eid, busy_sn, now)
                new_sn = FourStageSnap.from_assign(
                    now, SHAPE_C, SHAPE_C, top0_ntok, top0_eid, cc, cf
                )
                new_c2 = idle_sn if idle_cl == "c2" else new_sn
                new_c3 = new_sn if idle_cl == "c2" else idle_sn
                cost = _greedy_heuristic(new_c2.task_end, new_c3.task_end, new_rem)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_which = (new_c2, new_c3, new_rem)
            except Exception:
                pass

            if best_which is None:
                break
            c2, c3, remaining = best_which

    return max(c2.task_end, c3.task_end)


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────
def main():
    with open(CACHE_PATH) as f:
        data = json.load(f)
    print(f"加载缓存：{len(data)} 条 analytical 结果")
    print(
        f"测试 c_mirror_schedule（解析式形状 + 3个时间点）对 {len(data)} 个 case……",
        flush=True,
    )

    ratios = []
    times = []
    crashes = 0
    t_wall0 = time.perf_counter()

    for i, rec in enumerate(data):
        dist = {int(k): v for k, v in rec["dist"].items()}
        c2 = rec["c2"]
        c3 = rec["c3"]
        a = rec["analytical"]

        try:
            t0 = time.perf_counter()
            result = c_mirror_schedule(dict(dist), c2, c3)
            times.append(time.perf_counter() - t0)
        except Exception as e:
            crashes += 1
            continue

        if a > 0:
            ratios.append((result / a, dist, c2, c3, a, result))

        if (i + 1) % 2000 == 0:
            elapsed = time.perf_counter() - t_wall0
            print(f"  {i+1}/{len(data)} done ({elapsed:.1f}s)", flush=True)

    n = len(ratios)
    if n == 0:
        print("无有效数据")
        return

    rv = [r[0] for r in ratios]
    elapsed = time.perf_counter() - t_wall0
    avg_ms = sum(times) / len(times) * 1000 if times else 0

    print(f"\n{'='*57}")
    print(f"调度器:                    c_mirror (解析式 + 3pts)")
    print(f"有效对比:                  {n}")
    print(f"crashes:                   {crashes}")
    print(f"均值 ratio (vs analytical):{sum(rv)/n:>10.4f}")
    print(f"最大 ratio:                {max(rv):>10.4f}")
    print(f"中位数 ratio:              {sorted(rv)[n//2]:>10.4f}")
    print(f"pct 完全相同 (≤1.001):     {sum(1 for r in rv if r<=1.001)/n*100:>7.1f}%")
    print(f"pct <2%    (≤1.020):      {sum(1 for r in rv if r<=1.020)/n*100:>7.1f}%")
    print(f"pct <5%    (≤1.050):      {sum(1 for r in rv if r<=1.050)/n*100:>7.1f}%")
    print(f"pct <10%   (≤1.100):      {sum(1 for r in rv if r<=1.100)/n*100:>7.1f}%")
    print(f"均时:                      {avg_ms:>10.3f} ms/case")
    print(f"总耗时:                    {elapsed:>10.1f} s")
    print(f"{'='*57}")

    # 参考：lite_scheduler 的已知结果
    print(f"\n【参考】lite_scheduler (eval_lite_vs_anal_post_fix.txt):")
    print(f"  均值 ratio: 0.9863  最大 ratio: 1.2593  均时: 98.0ms  相同: 88.1%")

    worst = sorted(ratios, key=lambda x: x[0], reverse=True)[:10]
    if worst[0][0] > 1.001:
        print(f"\n差距最大的前10个案例:")
        for r, dist, c2i, c3i, a, result in worst:
            toks = sorted(dist.values(), reverse=True)
            print(
                f"  ratio={r:.3f}  toks={toks}  c2={c2i}  c3={c3i}  anal={a}  got={result}"
            )


if __name__ == "__main__":
    main()
