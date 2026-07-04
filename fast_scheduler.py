#!/usr/bin/env python3
"""
fast_scheduler.py
==================
极速贪心调度器：O(E) 复杂度，每步仅评估常数候选。

与 analytical_scheduler.py 的核心区别：
  ✗ 不做 PAIR(top0,topK) K=2,3 → 只评估 PAIR(top0,top1)（含两个方向）
  ✗ 不做 PAIR(topK,topJ) 变体（跳过 top0）
  ✗ 不做 sim1() lookahead → continuation cost 全用 _greedy_heuristic()
  ✓ PAIR/SPLIT 时枚举 3×3=9 种形状组合（vs 原版 N_SHAPES^4 × sim1，仍为常数）
  ✓ n==1 SPLIT 枚举 ceil/2, ceil/3, ceil/4 三个切分点
  ✓ bw_feasible() 精确检验，with_optional_s2_down_prefetch_pair 保留

复杂度分析（每步）：
  PAIR: 2方向 × 9形状 = 18 次 snap + bw_feasible
  SPLIT: 3切分 × 9形状 = 27 次 snap + bw_feasible
  !both_idle SOLO: 9形状 × start_pts(≤4) = 36 次
  总步数 O(E)，每步 O(1)，总计 O(E) ≈ 100 次操作（vs 原版 ~5M 次）

对应 C 实现：moe_scheduler.c 中 #ifdef MOE_SCHEDULE_FAST 快速路径
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from four_stage_scheduler import (
    ALL_SHAPES,
    MAX_BW,
    FourStageSnap,
    make_initial_snap,
    bw_feasible,
    with_optional_s2_down_prefetch,
    with_optional_s2_down_prefetch_pair,
    with_optional_next_s1_prefetch_pair,
    inject_ghost_prefetch_pair,
    PF_EID_GHOST,
    _best_task_time,
    _best_concurrent_task_time,
)
from analytical_scheduler import (
    best_solo_shape_s1,
    best_solo_shape_s3,
    best_conc_shape_s1,
    best_conc_shape_s3,
    _greedy_heuristic,
)

# ─────────────────────────────────────────────────────────────────────────────
# 辅助：全缓存命中时的最优初始 snap（S1/S3 均无需 DMA BW）
# ─────────────────────────────────────────────────────────────────────────────


def _best_cached_snap(eid: int, ntok: int, start: int = 0) -> FourStageSnap:
    """遍历 3×3=9 种形状组合，选 task_end 最小的全缓存命中 snap。"""
    best = None
    for s1 in ALL_SHAPES:
        for s3 in ALL_SHAPES:
            sn = FourStageSnap.from_assign(start, s1, s3, ntok, eid, True, True)
            if best is None or sn.task_end < best.task_end:
                best = sn
    return best


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：枚举 SPLIT 切分点列表
# ─────────────────────────────────────────────────────────────────────────────


def _split_cuts(ntok: int):
    """返回有代表性的切分点（去重），覆盖 half/third/quarter 及小切分点。

    策略：
    • ceil(ntok/k) for k=2,3,4：标准对称和非对称切分
    • cut=1（ntok≥3）、cut=2（ntok≥5）：极小切分，让一侧极快完成
    • ntok//2：向下取整的一半（奇数时 < ceil），如 ntok=9 → cut=4 (rest=5)
    • ntok//2-1：比等分小一的切分，如 ntok=14 → cut=6 (rest=8)，
      使 "短腿+长腿" 模式对齐 SPLIT(short)+serial(next_exp) ≈ SPLIT(long) 完成时刻
    """
    cuts = set()
    for denom in [2, 3, 4]:
        c = math.ceil(ntok / denom)
        if 1 <= c < ntok:
            cuts.add(c)
    # 极小切分点
    if ntok >= 3:
        cuts.add(1)
    if ntok >= 5:
        cuts.add(2)
    # 向下取整一半（奇数 ntok 有效）
    c_floor = ntok // 2
    if 1 <= c_floor < ntok:
        cuts.add(c_floor)
    # 比等分小一：产生 (short=ntok//2-1, long=ntok-ntok//2+1) 不对称分割
    c_floor_m1 = ntok // 2 - 1
    if c_floor_m1 >= 1:
        cuts.add(c_floor_m1)
    return sorted(cuts)


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：对一对 snap 枚举 9 种形状组合，返回最优 (ms, sna, snb)
# ─────────────────────────────────────────────────────────────────────────────


def _snap_bw_at(sn: FourStageSnap, t: int) -> int:
    """计算快照在时刻 t 的 DMA 带宽占用（不含 pf_bw，因 pf 是由 pf_pair 后添加的）。
    用于评估 busy 侧在 t_early 时刻的可用余量，作为 _best_pair_shapes 的次要排序键。
    """
    bw = 0
    # S1 DMA: active during [task_start, dma1_end)
    if sn.bw_s1 > 0 and sn.task_start <= t < sn.dma1_end:
        bw += sn.bw_s1
    # S2 down-prefetch: active during [s2pf_start, s2pf_end)
    if sn.s2pf_bw > 0 and sn.s2pf_start >= 0 and sn.s2pf_start <= t < sn.s2pf_end:
        bw += sn.s2pf_bw
    # S3 DMA: active during [s2_end, dma3_end)
    if sn.bw_s3 > 0 and sn.s2_end <= t < sn.dma3_end:
        bw += sn.bw_s3
    return bw


def _best_pair_shapes(
    start: int,
    eid_a: int,
    ntok_a: int,
    c_a: bool,
    f_a: bool,
    eid_b: int,
    ntok_b: int,
    c_b: bool,
    f_b: bool,
    rem_after: tuple,
):
    """
    在 t=start 同步发射 (eid_a→C2, eid_b→C3)，枚举 3×3 形状组合。
    返回 (greedy_cost, sna, snb) 或 (None, None, None) 若全不可行。

    次要排序：代价相同时，优先选 busy 侧在 t_early（= min(sna.end, snb.end)）时
    BW 占用更低的组合。这样能为 remaining 中的下一个 expert 在 t_early 预留最多 BW，
    最大化 early-start 的成功率。
    """
    best_cost = None
    best_busy_bw = None  # 次要键：busy 侧在 t_early 时的 BW（越小越好）
    best_sna = best_snb = None
    for s1a in ALL_SHAPES:
        for s3a in ALL_SHAPES:
            for s1b in ALL_SHAPES:
                for s3b in ALL_SHAPES:
                    try:
                        sna = FourStageSnap.from_assign(
                            start, s1a, s3a, ntok_a, eid_a, c_a, f_a
                        )
                        snb = FourStageSnap.from_assign(
                            start, s1b, s3b, ntok_b, eid_b, c_b, f_b
                        )
                        sna, snb = with_optional_s2_down_prefetch_pair(
                            sna, s3a, snb, s3b
                        )
                        if not bw_feasible(sna, snb):
                            continue
                        cost = _greedy_heuristic(sna.task_end, snb.task_end, rem_after)
                        # 次要键：busy 侧在 idle 侧结束时刻的 BW 占用
                        t_early = min(sna.task_end, snb.task_end)
                        busy_sn = snb if snb.task_end >= sna.task_end else sna
                        busy_bw = _snap_bw_at(busy_sn, t_early)
                        if (
                            best_cost is None
                            or cost < best_cost
                            or (cost == best_cost and busy_bw < best_busy_bw)
                        ):
                            best_cost = cost
                            best_busy_bw = busy_bw
                            best_sna, best_snb = sna, snb
                    except Exception:
                        pass
    return best_cost, best_sna, best_snb


# ─────────────────────────────────────────────────────────────────────────────
# 核心调度函数
# ─────────────────────────────────────────────────────────────────────────────


def _fast_schedule_core(
    remaining: tuple, init_c2: FourStageSnap, init_c3: FourStageSnap
) -> int:
    """主调度循环（O(E) 贪心），接受任意初始 cluster 快照。"""

    c2 = init_c2
    c3 = init_c3

    if not remaining:
        return max(c2.task_end, c3.task_end)

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
        c2_pre, c3_pre = c2, c3  # 保存 ghost 注入前的快照（用于 not_both_idle BW 评估）
        c2, c3 = inject_ghost_prefetch_pair(c2, c3)
        t2, t3 = c2.task_end, c3.task_end
        both_idle = t2 == t3
        now = max(t2, t3)
        n = len(remaining)
        top0_eid, top0_ntok = remaining[0]

        # ──────────────────────────────────────────────────
        # n == 1：最后一个 expert
        # ──────────────────────────────────────────────────
        if n == 1:
            best_cost = None
            best_which = None

            c2_sw = swiglu_hit(top0_eid, c2, now)
            c2_dn = down_hit(top0_eid, c2, now)
            c3_sw = swiglu_hit(top0_eid, c3, now)
            c3_dn = down_hit(top0_eid, c3, now)

            # 候选1/2: Solo C2/C3，枚举 3×3 形状
            for s1 in ALL_SHAPES:
                for s3 in ALL_SHAPES:
                    for cc, cf, which in [(c2_sw, c2_dn, "c2"), (c3_sw, c3_dn, "c3")]:
                        try:
                            sn = FourStageSnap.from_assign(
                                now, s1, s3, top0_ntok, top0_eid, cc, cf
                            )
                            sn = with_optional_s2_down_prefetch(sn, s3, None)
                            cost = sn.task_end
                            if best_cost is None or cost < best_cost:
                                best_cost = cost
                                best_which = (which, sn)
                        except Exception:
                            pass

            # 候选3: SPLIT——枚举 ceil/2, ceil/3, ceil/4 三个切分点 × 3×3 形状
            if top0_ntok >= 2:
                for cut in _split_cuts(top0_ntok):
                    rest = top0_ntok - cut
                    cost, sna, snb = _best_pair_shapes(
                        now, top0_eid, cut, c2_sw, c2_dn, top0_eid, rest, c3_sw, c3_dn, ()
                    )
                    if cost is not None and (best_cost is None or cost < best_cost):
                        best_cost = cost
                        best_which = ("split", sna, snb)

            # 候选4（仅 not-both-idle）：空闲侧提前启动 × 3×3 形状
            # 注意：使用 pf_pair 前的快照 c2_pre/c3_pre 做 BW 检查，
            # 避免 pf_bw 占满导致所有 early-start 候选不可行
            # 同时枚举 busy_sn 的 bw_change_pts，处理 s2pf 释放后可行的情况
            if not both_idle:
                idle_t = min(t2, t3)
                if t2 < t3:
                    # C2 空闲，C3 忙；使用 c3_pre（无 pf_bw）做 BW 检查
                    idle_sn, busy_sn_raw, idle_cl = c2, c3_pre, "c2"
                    cc_base, cf_base = swiglu_hit(top0_eid, c2, idle_t), down_hit(
                        top0_eid, c2, idle_t
                    )
                else:
                    # C3 空闲，C2 忙；使用 c2_pre（无 pf_bw）做 BW 检查
                    idle_sn, busy_sn_raw, idle_cl = c3, c2_pre, "c3"
                    cc_base, cf_base = swiglu_hit(top0_eid, c3, idle_t), down_hit(
                        top0_eid, c3, idle_t
                    )
                # 枚举 idle_t 及 busy_sn 的 BW 变化点作为候选启动时间
                try_starts = sorted(
                    t for t in ({idle_t} | busy_sn_raw.bw_change_pts()) if t >= idle_t
                )
                for t_st in try_starts:
                    cc = swiglu_hit(top0_eid, idle_sn, t_st)
                    cf = down_hit(top0_eid, idle_sn, t_st)
                    for s1 in ALL_SHAPES:
                        for s3 in ALL_SHAPES:
                            try:
                                new_sn = FourStageSnap.from_assign(
                                    t_st, s1, s3, top0_ntok, top0_eid, cc, cf
                                )
                                new_sn = with_optional_s2_down_prefetch(
                                    new_sn, s3, busy_sn_raw
                                )
                                ok = (
                                    bw_feasible(new_sn, busy_sn_raw)
                                    if idle_cl == "c2"
                                    else bw_feasible(busy_sn_raw, new_sn)
                                )
                                if ok:
                                    cost = max(new_sn.task_end, busy_sn_raw.task_end)
                                    if best_cost is None or cost < best_cost:
                                        best_cost = cost
                                        best_which = (idle_cl, new_sn)
                            except Exception:
                                pass

            if best_which is None:
                break
            if best_which[0] == "split":
                c2, c3 = best_which[1], best_which[2]
            elif best_which[0] == "c2":
                c2 = best_which[1]
            else:
                c3 = best_which[1]
            remaining = ()
            break

        # ──────────────────────────────────────────────────
        # n >= 2，both_idle：PAIR(top0,top1) 两方向 × 3×3 形状 + SPLIT(top0) × 切分点 × 3×3
        # ──────────────────────────────────────────────────
        elif both_idle:
            top1_eid, top1_ntok = remaining[1]
            new_rem_pair = tuple(
                r for r in remaining if r[0] != top0_eid and r[0] != top1_eid
            )
            new_rem_split = remaining[1:]

            c2_sw_0 = swiglu_hit(top0_eid, c2, now)
            c2_dn_0 = down_hit(top0_eid, c2, now)
            c3_sw_0 = swiglu_hit(top0_eid, c3, now)
            c3_dn_0 = down_hit(top0_eid, c3, now)
            c2c_1 = swiglu_hit(top1_eid, c2, now)
            c2f_1 = down_hit(top1_eid, c2, now)
            c3c_1 = swiglu_hit(top1_eid, c3, now)
            c3f_1 = down_hit(top1_eid, c3, now)

            best_cost = None
            best_which = None

            # PAIR 方向1: top0→C2, top1→C3
            cost, sna, snb = _best_pair_shapes(
                now,
                top0_eid,
                top0_ntok,
                c2_sw_0,
                c2_dn_0,
                top1_eid,
                top1_ntok,
                c3c_1,
                c3f_1,
                new_rem_pair,
            )
            if cost is not None and (best_cost is None or cost < best_cost):
                best_cost = cost
                best_which = ("pair", sna, snb, new_rem_pair)

            # PAIR 方向2: top1→C2, top0→C3
            cost, sna, snb = _best_pair_shapes(
                now,
                top1_eid,
                top1_ntok,
                c2c_1,
                c2f_1,
                top0_eid,
                top0_ntok,
                c3_sw_0,
                c3_dn_0,
                new_rem_pair,
            )
            if cost is not None and (best_cost is None or cost < best_cost):
                best_cost = cost
                best_which = ("pair", sna, snb, new_rem_pair)

            # SPLIT: top0 切分——枚举切分点 × 3×3 形状
            if top0_ntok >= 2:
                for cut in _split_cuts(top0_ntok):
                    rest = top0_ntok - cut
                    cost, sna, snb = _best_pair_shapes(
                        now,
                        top0_eid,
                        cut,
                        c2_sw_0,
                        c2_dn_0,
                        top0_eid,
                        rest,
                        c3_sw_0,
                        c3_dn_0,
                        new_rem_split,
                    )
                    if cost is not None and (best_cost is None or cost < best_cost):
                        best_cost = cost
                        best_which = ("split", sna, snb, new_rem_split)

            if best_which is None:
                # Fallback: solo top0 on C2
                s1_fb = best_solo_shape_s1(top0_ntok)
                s3_fb = best_solo_shape_s3(top0_ntok)
                try:
                    c2 = FourStageSnap.from_assign(
                        now, s1_fb, s3_fb, top0_ntok, top0_eid, c2_sw_0, c2_dn_0
                    )
                except Exception:
                    pass
                remaining = remaining[1:]
            else:
                _, sna, snb, new_rem = best_which
                c2, c3 = sna, snb
                remaining = new_rem

        # ──────────────────────────────────────────────────
        # n >= 2，not both_idle：空闲侧提前承接 top0 × 3×3 形状
        # 使用 pf_pair 前的快照做 BW 检查
        # ──────────────────────────────────────────────────
        else:
            idle_t = min(t2, t3)
            if t2 < t3:
                idle_sn, busy_sn_raw, idle_cl = c2, c3_pre, "c2"
            else:
                idle_sn, busy_sn_raw, idle_cl = c3, c2_pre, "c3"

            best_cost = None
            best_sn_end = None  # 次要键：idle 侧 task_end（越小越好）
            best_sn = None

            # 枚举 idle_t 及 busy_sn 的 BW 变化点作为候选启动时间
            try_starts = sorted(
                t for t in ({idle_t} | busy_sn_raw.bw_change_pts()) if t >= idle_t
            )
            for t_st in try_starts:
                cc = swiglu_hit(top0_eid, idle_sn, t_st)
                cf = down_hit(top0_eid, idle_sn, t_st)
                for s1 in ALL_SHAPES:
                    for s3 in ALL_SHAPES:
                        try:
                            new_sn = FourStageSnap.from_assign(
                                t_st, s1, s3, top0_ntok, top0_eid, cc, cf
                            )
                            new_sn = with_optional_s2_down_prefetch(
                                new_sn, s3, busy_sn_raw
                            )
                            ok = (
                                bw_feasible(new_sn, busy_sn_raw)
                                if idle_cl == "c2"
                                else bw_feasible(busy_sn_raw, new_sn)
                            )
                            if ok:
                                cost = max(new_sn.task_end, busy_sn_raw.task_end)
                                # 次要键：cost 相同时优先 idle 侧 task_end 更早，
                                # 让后续 expert 有更多时间提前启动
                                if (
                                    best_cost is None
                                    or cost < best_cost
                                    or (
                                        cost == best_cost
                                        and new_sn.task_end < best_sn_end
                                    )
                                ):
                                    best_cost = cost
                                    best_sn_end = new_sn.task_end
                                    best_sn = new_sn
                        except Exception:
                            pass

            if best_sn is not None:
                if idle_cl == "c2":
                    c2 = best_sn
                    # busy cluster（C3）的 PF_EID_GHOST 保持不变（对任意 eid 有效，无需回滚）
                else:
                    c3 = best_sn
                    # busy cluster（C2）的 PF_EID_GHOST 保持不变（对任意 eid 有效，无需回滚）

            remaining = remaining[1:]

    return max(c2.task_end, c3.task_end)


def fast_schedule(
    token_dist: dict,
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
) -> int:
    """O(E) 极速贪心调度器（含缓存预分配优化）。返回总 makespan（时钟周期数）。

    复杂度：O(E × 9形状 × 3候选) = O(E) 常数因子约 ~250 次操作（vs 原版 ~5M 次）。
    简化点：
      - 每步只评估 PAIR(top0,top1 两方向) + SPLIT(top0 三切分点)，不做 topK,topJ 变体
      - greedy_heuristic 估价代替 sim1() lookahead
      - 缓存预分配试 4 种方案取最优
    """
    remaining = tuple(sorted(token_dist.items(), key=lambda x: -x[1]))
    c2_idle = make_initial_snap(-1)
    c3_idle = make_initial_snap(-1)
    c2_base = make_initial_snap(initial_cache_c2)
    c3_base = make_initial_snap(initial_cache_c3)

    by_eid = {eid: ntok for eid, ntok in remaining}
    c2_ce = initial_cache_c2 if initial_cache_c2 in by_eid else -1
    c3_ce = initial_cache_c3 if initial_cache_c3 in by_eid else -1

    if c2_ce < 0 and c3_ce < 0:
        return _fast_schedule_core(remaining, c2_base, c3_base)

    # 有缓存：尝试 3~4 种预分配方案，取最优 makespan
    best_ms = _fast_schedule_core(remaining, c2_base, c3_base)

    if c2_ce >= 0:
        c2_snap = _best_cached_snap(c2_ce, by_eid[c2_ce])
        rem1 = tuple(r for r in remaining if r[0] != c2_ce)
        ms1 = _fast_schedule_core(rem1, c2_snap, c3_idle)
        best_ms = min(best_ms, ms1)

    if c3_ce >= 0:
        c3_snap = _best_cached_snap(c3_ce, by_eid[c3_ce])
        rem2 = tuple(r for r in remaining if r[0] != c3_ce)
        ms2 = _fast_schedule_core(rem2, c2_idle, c3_snap)
        best_ms = min(best_ms, ms2)

    if c2_ce >= 0 and c3_ce >= 0 and c2_ce != c3_ce:
        c2_snap = _best_cached_snap(c2_ce, by_eid[c2_ce])
        c3_snap = _best_cached_snap(c3_ce, by_eid[c3_ce])
        rem3 = tuple(r for r in remaining if r[0] not in {c2_ce, c3_ce})
        ms3 = _fast_schedule_core(rem3, c2_snap, c3_snap)
        best_ms = min(best_ms, ms3)

    return best_ms


# ─────────────────────────────────────────────────────────────────────────────
# 快速自测
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    from analytical_scheduler import analytical_schedule

    random.seed(42)
    tests = [
        ({0: 8}, -1, -1),
        ({0: 4, 1: 4}, -1, -1),
        ({0: 6, 1: 2}, -1, -1),
        ({0: 4, 1: 3, 2: 1}, -1, -1),
        ({0: 8, 1: 8}, -1, -1),
        ({0: 4, 1: 2, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1}, -1, -1),
        ({0: 8, 1: 6, 2: 4, 3: 2}, 0, -1),
        ({0: 5, 1: 3}, 0, 1),
        # 修复验证用例
        ({0: 2, 1: 2}, 0, -1),  # Bug1 case: c2e=0
        ({0: 12}, -1, -1),  # Bug2 case: single expert ntok=12
        ({0: 6, 1: 6, 2: 2, 3: 2}, -1, -1),  # Bug3 case
        ({0: 5, 1: 3}, -1, -1),  # Bug orig case
        # [9,1,1] ghost-pf + 次要键修复用例
        ({0: 9, 1: 1, 2: 1}, -1, -1),
        ({0: 9, 1: 1, 2: 1}, 0, -1),
        ({0: 9, 1: 1, 2: 1}, 1, -1),
        ({0: 9, 1: 1, 2: 1}, 2, -1),
    ]
    print(f"{'Test case':<35} {'fast':>8} {'anal':>8} {'ratio':>7}")
    print("-" * 65)
    all_pass = True
    for dist, c2e, c3e in tests:
        ms_fast = fast_schedule(dist, c2e, c3e)
        ms_anal = analytical_schedule(dist, c2e, c3e)
        ratio = ms_fast / ms_anal
        ok = ratio <= 1.05
        mark = "OK" if ok else "FAIL"
        if not ok:
            all_pass = False
        label = f"{dict(sorted(dist.items(),key=lambda x:-x[1]))} c2={c2e} c3={c3e}"
        print(f"{label:<35} {ms_fast:>8} {ms_anal:>8} {ratio:>7.4f}  {mark}")
    print("-" * 65)
    print("ALL PASS" if all_pass else "SOME FAILED")
