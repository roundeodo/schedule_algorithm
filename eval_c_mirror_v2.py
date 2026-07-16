#!/usr/bin/env python3
"""
eval_c_mirror_v2.py
===================
保留裁剪前较完整候选空间的 C-style Python baseline。

相比 eval_c_mirror.py（旧版 C 镜像），新增/修改：
  1. Ghost pf 注入 + 回滚（每次迭代开头，不只在 n=1）
  2. both_idle: PAIR K=1..min(3,n-1) + topK-topJ 交叉对 + M_dim 边界 SPLIT
  3. continuation_cost: n=1 → sim1_c (C 版简化 sim1)
                        n=2 小 token → 闭合式
                        else → greedy_h_c
  4. not_both_idle: 3 时间点 + S2 down-prefetch 尝试

对比对象：
  - analytical_schedule (10K 缓存，baseline)
  - fast_schedule       (fast_scheduler.py，上轮测试最好者)
  - c_mirror_v2         (本文件，full C-style baseline；不是当前 pruned C 的精确镜像)
"""

import sys, os, json, time, math
from dataclasses import dataclass, replace

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

CACHE_PATH = os.path.join(os.path.dirname(__file__), "analytical_cache.json")

EXACT_TAIL_MAX = 4  # 匹配 C: EXACT_TAIL_MAX = 4
_T_DMA_S3_C = SHAPE_C.t_dma_s3  # 11264 cc


# ─────────────────────────────────────────────────────────────────────────────
# 计时常量（完全匹配 C 代码中 best_task / best_conc）
# ─────────────────────────────────────────────────────────────────────────────
def _c_best_task(n: int) -> int:
    """单 cluster 处理 n token 最短时间（ShapeC，匹配 C best_task()）"""
    return ((n + 1) // 2) * 33792


def _c_best_conc(n: int) -> int:
    """双 cluster 并发处理 n token 最短时间（ShapeC，匹配 C best_conc()）"""
    return ((n + 3) // 4) * 67584


def _c_best_s2(n: int) -> int:
    return ((n + 1) // 2) * 22528


# ─────────────────────────────────────────────────────────────────────────────
# pick_shapes — 镜像 C pick_shapes()：O(1) 解析式形状选择
# ─────────────────────────────────────────────────────────────────────────────
def _pick_pair_shapes(ntok_A, ntok_B, sw_A, dn_A, sw_B, dn_B, t_now):
    """返回 [(s1A, s3A, s1B, s3B)]，与 moe_scheduler.c pick_shapes 完全一致。"""
    if sw_A or sw_B:
        s1_A = s1_B = SHAPE_C
    else:
        s1_A = s1_B = SHAPE_B

    if sw_A:
        s2_A = t_now + _c_best_s2(ntok_A)
    else:
        s2_A = t_now + s1_A.T_s1 + _c_best_s2(max(0, ntok_A - s1_A.M_dim))

    if sw_B:
        s2_B = t_now + _c_best_s2(ntok_B)
    else:
        s2_B = t_now + s1_B.T_s1 + _c_best_s2(max(0, ntok_B - s1_B.M_dim))

    if dn_A or dn_B:
        s3_A, s3_B = (
            SHAPE_C,
            SHAPE_C,
        )  # hit侧DMA=0，S4用ShapeC tile(2tok/11264cc)匹配best_s4
    elif abs(s2_A - s2_B) >= _T_DMA_S3_C:
        s3_A, s3_B = SHAPE_C, SHAPE_C
    else:
        s3_A, s3_B = SHAPE_B, SHAPE_B

    return [(s1_A, s3_A, s1_B, s3_B)]


# ─────────────────────────────────────────────────────────────────────────────
# DMA hi 端点提取（镜像 C snap_segs() 各段 .hi）
# ─────────────────────────────────────────────────────────────────────────────
def _dma_hi_pts(sn: FourStageSnap) -> set:
    """取 busy cluster 的 DMA 段结束时刻（BW 释放点）"""
    pts = {sn.dma1_end, sn.dma3_end}
    if hasattr(sn, "s2pf_end") and sn.s2pf_end >= 0:
        pts.add(sn.s2pf_end)
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# greedy_h_c — 镜像 C greedy_h()
# ─────────────────────────────────────────────────────────────────────────────
def _greedy_h_c(c2_end: int, c3_end: int, rem: tuple) -> int:
    """精确匹配 moe_scheduler.c greedy_h() 公式。"""
    max_e = max(c2_end, c3_end)
    nr = len(rem)
    if nr == 0:
        return max_e
    if nr == 1:
        nt = rem[0][1]
        te = min(c2_end, c3_end)
        tl = max_e
        sc = max(tl, te + _c_best_task(nt))
        sp = tl + _c_best_conc((nt + 1) // 2)
        return min(sc, sp)
    if nr == 2:
        te = min(c2_end, c3_end)
        tl = max_e
        bc0 = _c_best_conc(rem[0][1])
        bc1 = _c_best_conc(rem[1][1])
        pc = tl + max(bc0, bc1)
        ser = te + _c_best_task(rem[0][1]) + _c_best_task(rem[1][1])
        serc = max(ser, tl)
        return min(pc, serc)
    # general: Σ best_conc, max
    total = sum(_c_best_conc(nt) for _, nt in rem)
    mx = max(_c_best_conc(nt) for _, nt in rem)
    extra = max(mx, total // 2)
    return max_e + extra


# ─────────────────────────────────────────────────────────────────────────────
# sim1_c — 镜像 C sim1()（简化的 1-expert lookahead）
# ─────────────────────────────────────────────────────────────────────────────
def _sim1_c(
    c2_sn: FourStageSnap,
    c3_sn: FourStageSnap,
    eid: int,
    ntok: int,
    swiglu_hit_fn,
    down_hit_fn,
) -> int:
    """
    镜像 moe_scheduler.c sim1()：
      - 对每个 cluster 尝试 ShapeC solo（+ 若无 pf 也尝试 ShapeB）
      - 尝试 analytical SPLIT（pick_shapes + s2pf_pair）
    比 fast_lite _sim1 简单得多（无 9×9 枚举）。
    """
    t = max(c2_sn.task_end, c3_sn.task_end)
    best = None

    for sn_ci in [c2_sn, c3_sn]:
        cc = swiglu_hit_fn(eid, sn_ci, t)
        cf = down_hit_fn(eid, sn_ci, t)
        # ShapeC solo
        try:
            sn = FourStageSnap.from_assign(t, SHAPE_C, SHAPE_C, ntok, eid, cc, cf)
            if best is None or sn.task_end < best:
                best = sn.task_end
        except Exception:
            pass
        # ShapeB solo（无 pf 时额外尝试）
        if not cc:
            try:
                sn2 = FourStageSnap.from_assign(
                    t, SHAPE_B, SHAPE_B, ntok, eid, False, False
                )
                if best is None or sn2.task_end < best:
                    best = sn2.task_end
            except Exception:
                pass

    # Analytical SPLIT (pick_shapes, ceil/2)
    if ntok >= 2:
        ca = (ntok + 1) // 2
        cb = ntok - ca
        sw_a = swiglu_hit_fn(eid, c2_sn, t)
        dn_a = down_hit_fn(eid, c2_sn, t)
        sw_b = swiglu_hit_fn(eid, c3_sn, t)
        dn_b = down_hit_fn(eid, c3_sn, t)
        for s1a, s3a, s1b, s3b in _pick_pair_shapes(ca, cb, sw_a, dn_a, sw_b, dn_b, t):
            try:
                sna = FourStageSnap.from_assign(t, s1a, s3a, ca, eid, sw_a, dn_a)
                snb = FourStageSnap.from_assign(t, s1b, s3b, cb, eid, sw_b, dn_b)
                sna, snb = with_optional_s2_down_prefetch_pair(sna, s3a, snb, s3b)
                if bw_feasible(sna, snb):
                    e = max(sna.task_end, snb.task_end)
                    if best is None or e < best:
                        best = e
            except Exception:
                pass

    return best if best is not None else (t + _c_best_task(ntok))


# ─────────────────────────────────────────────────────────────────────────────
# continuation_cost_c — 镜像 C continuation_cost()
# ─────────────────────────────────────────────────────────────────────────────
def _continuation_cost_c(
    ta: FourStageSnap, tb: FourStageSnap, rem: tuple, swiglu_hit_fn, down_hit_fn
) -> int:
    """
    匹配 moe_scheduler.c continuation_cost()：
      nr=0: max(task_ends)
      nr=1: sim1_c (lookahead)
      nr=2 && total_tokens<=EXACT_TAIL_MAX: 闭合式
      else: greedy_h_c
    """
    nr = len(rem)
    if nr == 0:
        return max(ta.task_end, tb.task_end)
    if nr == 1:
        return _sim1_c(ta, tb, rem[0][0], rem[0][1], swiglu_hit_fn, down_hit_fn)
    if nr == 2 and (rem[0][1] + rem[1][1]) <= EXACT_TAIL_MAX:
        te = min(ta.task_end, tb.task_end)
        tl = max(ta.task_end, tb.task_end)
        ss = te + _c_best_task(rem[0][1]) + _c_best_task(rem[1][1])
        pa = tl + max(_c_best_conc(rem[0][1]), _c_best_conc(rem[1][1]))
        v1 = max(tl, ss)
        return min(v1, pa)
    return _greedy_h_c(ta.task_end, tb.task_end, rem)


# ─────────────────────────────────────────────────────────────────────────────
# 主调度函数
# ─────────────────────────────────────────────────────────────────────────────
def c_mirror_v2_schedule(
    token_dist: dict,
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
) -> int:
    """
    精确镜像当前 moe_scheduler.c 调度逻辑：
      - Ghost pf 注入 + 回滚（每次迭代）
      - both_idle: K=1..3 PAIR + topK-topJ + M_dim 边界 SPLIT
      - continuation_cost: sim1_c for n=1 remaining
      - not_both_idle: ShapeC × 3 DMA-hi 时间点 + S2pf 尝试
    """
    remaining = tuple(sorted(token_dist.items(), key=lambda x: -x[1]))
    c2 = make_initial_snap(initial_cache_c2)
    c3 = make_initial_snap(initial_cache_c3)

    def swiglu_hit(eid, snap, t):
        if snap.pf_end < 0 or snap.pf_end > t:
            return False
        return snap.pf_eid == PF_EID_GHOST or snap.pf_eid == eid

    def down_hit(eid, snap, t):
        return swiglu_hit(eid, snap, t) and snap.pf_full

    # ── 主循环 ────────────────────────────────────────────────────────────────
    while remaining:
        top0_eid, top0_ntok = remaining[0]
        t2, t3 = c2.task_end, c3.task_end
        now = max(t2, t3)
        both_idle = t2 == t3

        # ── Ghost pf 注入（每次迭代，匹配 C 代码）──────────────────────────────
        c2, c3 = inject_ghost_prefetch_pair(c2, c3)

        # 使用注入后的 snap 重新计算 swiglu_hit 标志
        c2c0 = swiglu_hit(top0_eid, c2, now)
        c2f0 = down_hit(top0_eid, c2, now)
        c3c0 = swiglu_hit(top0_eid, c3, now)
        c3f0 = down_hit(top0_eid, c3, now)

        # ── n=1 ───────────────────────────────────────────────────────────────
        if len(remaining) == 1:
            best_cost = None
            best_snap = None  # ("C2"/"C3"/split, snap or (sna, snb))

            # Method A: 9 (s1,s3) combo × 2 clusters（与 fast_lite 相同）
            for s1 in ALL_SHAPES:
                for s3 in ALL_SHAPES:
                    for ci, (sn_ci, cc, cf) in enumerate(
                        [
                            (c2, c2c0, c2f0),
                            (c3, c3c0, c3f0),
                        ]
                    ):
                        try:
                            sn = FourStageSnap.from_assign(
                                now, s1, s3, top0_ntok, top0_eid, cc, cf
                            )
                            cost = max(
                                sn.task_end, c3.task_end if ci == 0 else c2.task_end
                            )
                            if best_cost is None or cost < best_cost:
                                best_cost = cost
                                best_snap = ("C2" if ci == 0 else "C3", sn)
                        except Exception:
                            pass

            # SPLIT: ceil/2 和 floor/2（匹配 C n=1 的 2 个切点）
            if top0_ntok >= 2:
                h1 = (top0_ntok + 1) // 2
                h2 = top0_ntok // 2
                cuts = {h1}
                if h2 != h1 and 1 <= h2 <= top0_ntok - 1:
                    cuts.add(h2)
                for cut_A in cuts:
                    cut_B = top0_ntok - cut_A
                    if cut_A <= 0 or cut_B <= 0:
                        continue
                    for s1a, s3a, s1b, s3b in _pick_pair_shapes(
                        cut_A, cut_B, c2c0, c2f0, c3c0, c3f0, now
                    ):
                        try:
                            sna = FourStageSnap.from_assign(
                                now, s1a, s3a, cut_A, top0_eid, c2c0, c2f0
                            )
                            snb = FourStageSnap.from_assign(
                                now, s1b, s3b, cut_B, top0_eid, c3c0, c3f0
                            )
                            sna, snb = with_optional_s2_down_prefetch_pair(
                                sna, s3a, snb, s3b
                            )
                            if not bw_feasible(sna, snb):
                                continue
                            cost = max(sna.task_end, snb.task_end)
                            if best_cost is None or cost < best_cost:
                                best_cost = cost
                                best_snap = ("split", sna, snb)
                        except Exception:
                            pass

            # Method B: 3 时间点 × ShapeC（不 both_idle 时）
            if not both_idle:
                if t2 < t3:
                    idle_t, idle_cl = t2, "C2"
                    idle_sn, busy_sn = c2, c3
                else:
                    idle_t, idle_cl = t3, "C3"
                    idle_sn, busy_sn = c3, c2

                hi_pts = _dma_hi_pts(busy_sn)
                try_starts = sorted({idle_t} | {p for p in hi_pts if p > idle_t})[:3]

                for t_st in try_starts:
                    cc = swiglu_hit(top0_eid, idle_sn, t_st)
                    cf = down_hit(top0_eid, idle_sn, t_st)
                    try:
                        sn = FourStageSnap.from_assign(
                            t_st, SHAPE_C, SHAPE_C, top0_ntok, top0_eid, cc, cf
                        )
                        sn = with_optional_s2_down_prefetch(sn, SHAPE_C, busy_sn)
                        ok = (
                            bw_feasible(sn, busy_sn)
                            if idle_cl == "C2"
                            else bw_feasible(busy_sn, sn)
                        )
                        if not ok:
                            continue
                        cost = max(sn.task_end, busy_sn.task_end)
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                            best_snap = (idle_cl, sn)
                    except Exception:
                        pass

            # 提交 n=1
            if best_snap is None:
                c2 = FourStageSnap.from_assign(
                    now, SHAPE_C, SHAPE_C, top0_ntok, top0_eid, c2c0, c2f0
                )
            elif best_snap[0] == "split":
                _, sna, snb = best_snap
                c2, c3 = sna, snb
            else:
                which, sn = best_snap[0], best_snap[1]
                if which == "C2":
                    c2 = sn
                else:
                    c3 = sn
            remaining = ()
            break

        # ── both_idle ─────────────────────────────────────────────────────────
        if both_idle:
            best_pair_cost = None
            best_pair_snap = None
            best_pair_rem = None
            best_pair_ms = None
            best_pair_nrem = None

            def _try_pair(ea, na, sw_a, dn_a, eb, nb, sw_b, dn_b, rem_after):
                nonlocal best_pair_cost, best_pair_snap, best_pair_rem, best_pair_ms, best_pair_nrem
                for s1a, s3a, s1b, s3b in _pick_pair_shapes(
                    na, nb, sw_a, dn_a, sw_b, dn_b, now
                ):
                    try:
                        sna = FourStageSnap.from_assign(
                            now, s1a, s3a, na, ea, sw_a, dn_a
                        )
                        snb = FourStageSnap.from_assign(
                            now, s1b, s3b, nb, eb, sw_b, dn_b
                        )
                        ta, tb = with_optional_s2_down_prefetch_pair(sna, s3a, snb, s3b)
                        if not bw_feasible(ta, tb):
                            continue
                        cost = _continuation_cost_c(
                            ta, tb, rem_after, swiglu_hit, down_hit
                        )
                        ms = max(ta.task_end, tb.task_end)
                        nrem = len(rem_after)
                        # cand_better: 镜像 C 的 cand_better 优先级
                        better = False
                        if best_pair_cost is None:
                            better = True
                        elif cost < best_pair_cost:
                            better = True
                        elif cost == best_pair_cost:
                            if nrem < best_pair_nrem:
                                better = True
                            elif nrem == best_pair_nrem and ms < best_pair_ms:
                                better = True
                        if better:
                            best_pair_cost = cost
                            best_pair_snap = (ta, tb)
                            best_pair_rem = rem_after
                            best_pair_ms = ms
                            best_pair_nrem = nrem
                    except Exception:
                        pass

            # PAIR(top0, topK) K=1..min(3,n-1)
            maxK = min(3, len(remaining) - 1)
            for K in range(1, maxK + 1):
                K_eid, K_ntok = remaining[K]
                rem_after = tuple(
                    r for r in remaining if r[0] != top0_eid and r[0] != K_eid
                )
                # Dir1: top0→C2, topK→C3
                _try_pair(
                    top0_eid,
                    top0_ntok,
                    c2c0,
                    c2f0,
                    K_eid,
                    K_ntok,
                    swiglu_hit(K_eid, c3, now),
                    down_hit(K_eid, c3, now),
                    rem_after,
                )
                # Dir2: topK→C2, top0→C3
                _try_pair(
                    K_eid,
                    K_ntok,
                    swiglu_hit(K_eid, c2, now),
                    down_hit(K_eid, c2, now),
                    top0_eid,
                    top0_ntok,
                    c3c0,
                    c3f0,
                    rem_after,
                )

            # PAIR(topK, topJ) K>=1, J>K（要求 rem_after 非空）
            if len(remaining) >= 3:
                mKJ = min(3, len(remaining) - 1)
                for K in range(1, mKJ):
                    for J in range(K + 1, mKJ + 1):
                        if J >= len(remaining):
                            continue
                        K_eid, K_ntok = remaining[K]
                        J_eid, J_ntok = remaining[J]
                        rem_after = tuple(
                            r for r in remaining if r[0] != K_eid and r[0] != J_eid
                        )
                        if not rem_after:
                            continue  # C 代码: nra==0 则 continue
                        # Dir1: topK→C2, topJ→C3
                        _try_pair(
                            K_eid,
                            K_ntok,
                            swiglu_hit(K_eid, c2, now),
                            down_hit(K_eid, c2, now),
                            J_eid,
                            J_ntok,
                            swiglu_hit(J_eid, c3, now),
                            down_hit(J_eid, c3, now),
                            rem_after,
                        )
                        # Dir2: topJ→C2, topK→C3
                        _try_pair(
                            J_eid,
                            J_ntok,
                            swiglu_hit(J_eid, c2, now),
                            down_hit(J_eid, c2, now),
                            K_eid,
                            K_ntok,
                            swiglu_hit(K_eid, c3, now),
                            down_hit(K_eid, c3, now),
                            rem_after,
                        )

            # SPLIT(top0)：M_dim 边界切点（匹配 C 的 8 个切点）
            if top0_ntok >= 2:
                s_cuts = set()
                h1 = (top0_ntok + 1) // 2
                h2 = top0_ntok // 2
                s_cuts.add(h1)
                if h2 != h1:
                    s_cuts.add(h2)
                for mi in [8, 4, 2]:
                    if mi < top0_ntok:
                        s_cuts.add(mi)
                    if top0_ntok > mi:
                        s_cuts.add(top0_ntok - mi)
                rem_after = remaining[1:]
                for cut_A in s_cuts:
                    cut_B = top0_ntok - cut_A
                    if cut_A <= 0 or cut_B <= 0:
                        continue
                    _try_pair(
                        top0_eid,
                        cut_A,
                        c2c0,
                        c2f0,
                        top0_eid,
                        cut_B,
                        c3c0,
                        c3f0,
                        rem_after,
                    )

            # 提交 both_idle
            if best_pair_snap is not None:
                c2, c3 = best_pair_snap
                remaining = best_pair_rem
            else:
                # Fallback: solo top0 on C2
                sn = FourStageSnap.from_assign(
                    now, SHAPE_C, SHAPE_C, top0_ntok, top0_eid, c2c0, c2f0
                )
                c2 = sn
                remaining = remaining[1:]
                # C3 的 PF_EID_GHOST 保持不变（对任意 eid 有效，无需回滚）

            continue

        # ── not_both_idle ─────────────────────────────────────────────────────
        # not_both_idle 中不再需要 busy_pre 和 rollback
        if t2 < t3:
            idle_t, idle_cl = t2, "C2"
            idle_sn, busy_sn = c2, c3
        else:
            idle_t, idle_cl = t3, "C3"
            idle_sn, busy_sn = c3, c2

        hi_pts = _dma_hi_pts(busy_sn)
        try_starts = sorted({idle_t} | {p for p in hi_pts if p > idle_t})[:3]

        best_ms = None
        best_nb = None

        for t_st in try_starts:
            cc = swiglu_hit(top0_eid, idle_sn, t_st)
            cf = down_hit(top0_eid, idle_sn, t_st)
            try:
                sn = FourStageSnap.from_assign(
                    t_st, SHAPE_C, SHAPE_C, top0_ntok, top0_eid, cc, cf
                )
                # S2 down-prefetch 尝试（匹配 C not_both_idle 的 apply_s2pf 调用）
                sn = with_optional_s2_down_prefetch(sn, SHAPE_C, busy_sn)
                ok = (
                    bw_feasible(sn, busy_sn)
                    if idle_cl == "C2"
                    else bw_feasible(busy_sn, sn)
                )
                if not ok:
                    continue
                ms = max(sn.task_end, busy_sn.task_end)
                if best_ms is None or ms < best_ms:
                    best_ms = ms
                    best_nb = sn
            except Exception:
                pass

        # 提交 not_both_idle
        if best_nb is not None:
            if idle_cl == "C2":
                c2 = best_nb
            else:
                c3 = best_nb
        else:
            cc = swiglu_hit(top0_eid, idle_sn, idle_t)
            cf = down_hit(top0_eid, idle_sn, idle_t)
            fb = FourStageSnap.from_assign(
                idle_t, SHAPE_C, SHAPE_C, top0_ntok, top0_eid, cc, cf
            )
            if idle_cl == "C2":
                c2 = fb
            else:
                c3 = fb

        remaining = remaining[1:]
        # busy cluster 的 PF_EID_GHOST 保持不变（对任意 eid 有效，无需回滚）

    return max(c2.task_end, c3.task_end)


# ─────────────────────────────────────────────────────────────────────────────
# Full C-style baseline used to evaluate pruned mirrors.
#
# The legacy implementation above was built on FourStageSnap helpers.  The C
# scheduler now has BW-aware S4 prefetch state, so the executable mirror below
# keeps a local snap_t clone and shadows c_mirror_v2_schedule().
# ─────────────────────────────────────────────────────────────────────────────
C_TS1 = (90112, 45056, 22528)
C_TS3 = (45056, 22528, 11264)
C_TD1 = (45056, 45056, 22528)
C_TD3 = (22528, 22528, 11264)
C_MDIM = (8, 4, 2)
C_ALLOC = (64, 64, 128)
C_MAX_BW = 128
C_INF = 0xFFFFFFFF
C_EXACT_TAIL_MAX = 4
C_T_DMA3_C = 11264
C_SHAPE_A = 0
C_SHAPE_B = 1
C_SHAPE_C = 2
C_PF_EID_GHOST = -2


@dataclass
class CSnap:
    task_start: int
    task_end: int
    dma1_end: int
    s1_end: int
    s2_end: int
    dma3_end: int
    s3_end: int
    s4_start: int
    bw_s1: int
    bw_s3: int
    cur_eid: int
    pf_eid: int
    pf_start: int
    pf_end: int
    pf_bw: int
    pf_full: int
    s2pf_start: int
    s2pf_end: int
    s2pf_bw: int
    s4pf_valid: int
    s4pf_start: int
    ntok: int


def _cc_best_s2(r: int) -> int:
    return ((r + 1) // 2) * 22528


def _cc_best_s4(r: int) -> int:
    return ((r + 1) // 2) * 11264


def _cc_best_task(n: int) -> int:
    return ((n + 1) // 2) * 33792


def _cc_best_conc(n: int) -> int:
    return ((n + 3) // 4) * 67584


def _cc_idle_at(t: int) -> CSnap:
    return CSnap(
        task_start=t,
        task_end=t,
        dma1_end=t,
        s1_end=t,
        s2_end=t,
        dma3_end=t,
        s3_end=t,
        s4_start=t,
        bw_s1=0,
        bw_s3=0,
        cur_eid=-1,
        pf_eid=-1,
        pf_start=-1,
        pf_end=-1,
        pf_bw=0,
        pf_full=0,
        s2pf_start=-1,
        s2pf_end=-1,
        s2pf_bw=0,
        s4pf_valid=0,
        s4pf_start=0,
        ntok=0,
    )


def _cc_initial(cache_eid: int) -> CSnap:
    s = _cc_idle_at(0)
    if cache_eid >= 0:
        s.pf_eid = cache_eid
        s.pf_end = 0
        s.pf_full = 1
    return s


def _cc_mk_snap(start: int, s1: int, s3: int, ntok: int, eid: int, s1c: bool, s3c: bool) -> CSnap:
    r = _cc_idle_at(start)
    r.cur_eid = eid
    r.ntok = ntok
    if s1c:
        r.dma1_end = start
        r.s1_end = start
        r.bw_s1 = 0
        r.s2_end = start + _cc_best_s2(ntok)
    else:
        rm = ntok - C_MDIM[s1] if ntok > C_MDIM[s1] else 0
        r.dma1_end = start + C_TD1[s1]
        r.s1_end = start + C_TS1[s1]
        r.bw_s1 = C_ALLOC[s1]
        r.s2_end = r.s1_end + _cc_best_s2(rm)

    if s3c:
        r.dma3_end = r.s2_end
        r.s3_end = r.s2_end
        r.s4_start = r.s2_end
        r.bw_s3 = 0
        r.task_end = r.s2_end + _cc_best_s4(ntok)
    else:
        rm = ntok - C_MDIM[s3] if ntok > C_MDIM[s3] else 0
        r.dma3_end = r.s2_end + C_TD3[s3]
        r.s3_end = r.s2_end + C_TS3[s3]
        r.s4_start = r.s3_end
        r.bw_s3 = C_ALLOC[s3]
        r.task_end = r.s3_end + _cc_best_s4(rm)
    return r


def _cc_apply_s2pf(sn: CSnap, s3: int, ps: int) -> CSnap:
    pe = ps + C_TD3[s3]
    if sn.bw_s3 == 0:
        return sn
    if ps < sn.dma1_end or pe > sn.s2_end:
        return sn
    out = replace(sn)
    out.s2pf_start = ps
    out.s2pf_end = pe
    out.s2pf_bw = C_ALLOC[s3]
    out.dma3_end = out.s2_end
    out.s3_end = out.s2_end
    out.s4_start = out.s2_end
    out.bw_s3 = 0
    out.task_end = out.s2_end + _cc_best_s4(out.ntok)
    return out


def _cc_snap_segs(s: CSnap):
    out = []
    if s.cur_eid >= 0 and s.bw_s1 > 0:
        out.append((s.task_start, s.dma1_end, s.bw_s1))
    if s.s2pf_start >= 0 and s.s2pf_bw > 0:
        out.append((s.s2pf_start, s.s2pf_end, s.s2pf_bw))

    if s.cur_eid >= 0 and s.bw_s3 > 0 and s.dma3_end > s.s2_end:
        out.append((s.s2_end, s.dma3_end, s.bw_s3))
    if s.cur_eid >= 0 and s.s4pf_valid:
        out.append((s.s4pf_start, s.s4pf_start + C_TD1[C_SHAPE_A], C_ALLOC[C_SHAPE_A]))
    return out


def _cc_bw_ok(a: CSnap, b: CSnap) -> bool:
    sa = _cc_snap_segs(a)
    sb = _cc_snap_segs(b)
    for alo, ahi, abw in sa:
        for blo, bhi, bbw in sb:
            lo = max(alo, blo)
            hi = min(ahi, bhi)
            if lo < hi and abw + bbw > C_MAX_BW:
                return False
    return True


def _cc_s4pf_local_ok(s: CSnap) -> bool:
    return (
        s.cur_eid >= 0
        and s.pf_eid == -1
        and s.dma3_end + C_TD1[C_SHAPE_A] <= s.task_end
    )


def _cc_apply_s4pf_ghost(s: CSnap) -> CSnap:
    if not _cc_s4pf_local_ok(s):
        return s
    out = replace(s)
    out.pf_eid = C_PF_EID_GHOST
    out.pf_end = out.task_end
    out.pf_full = 0
    out.s4pf_valid = 1
    out.s4pf_start = out.dma3_end
    return out


def _cc_s4pf_ok_with_peer(s: CSnap, peer: CSnap) -> bool:
    if not _cc_s4pf_local_ok(s):
        return False
    return _cc_bw_ok(_cc_apply_s4pf_ghost(s), peer)


def _cc_swiglu_hit(eid: int, s: CSnap, t: int) -> bool:
    if s.pf_end < 0 or s.pf_end > t:
        return False
    return s.pf_eid == C_PF_EID_GHOST or s.pf_eid == eid


def _cc_down_hit(eid: int, s: CSnap, t: int) -> bool:
    return _cc_swiglu_hit(eid, s, t) and bool(s.pf_full)


def _cc_pick_shapes(na: int, nb: int, sw_a: bool, dn_a: bool, sw_b: bool, dn_b: bool, t0: int):
    if sw_a or sw_b:
        s1a = s1b = C_SHAPE_C
    else:
        s1a = s1b = C_SHAPE_B

    if sw_a:
        s2a = t0 + _cc_best_s2(na)
    else:
        r = na - C_MDIM[s1a] if na > C_MDIM[s1a] else 0
        s2a = t0 + C_TS1[s1a] + _cc_best_s2(r)
    if sw_b:
        s2b = t0 + _cc_best_s2(nb)
    else:
        r = nb - C_MDIM[s1b] if nb > C_MDIM[s1b] else 0
        s2b = t0 + C_TS1[s1b] + _cc_best_s2(r)

    if dn_a or dn_b:
        s3a = s3b = C_SHAPE_C
    elif abs(s2a - s2b) >= C_T_DMA3_C:
        s3a = s3b = C_SHAPE_C
    else:
        s3a = s3b = C_SHAPE_B
    return s1a, s3a, s1b, s3b


def _cc_s2pf_candidates(s: CSnap, s3: int):
    cand = []
    span = s.s2_end - s.dma1_end
    if s.bw_s3 > 0 and C_TD3[s3] <= span:
        lo = s.dma1_end
        hi = s.s2_end - C_TD3[s3]
        if hi >= lo:
            cand.append(lo)
            if lo <= s.s1_end <= hi and s.s1_end != s.dma1_end:
                cand.append(s.s1_end)
            if hi != lo:
                cand.append(hi)
    return cand


def _cc_try_s2pf_pair(sa: CSnap, s3a: int, sb: CSnap, s3b: int):
    ca = _cc_s2pf_candidates(sa, s3a)
    cb = _cc_s2pf_candidates(sb, s3b)
    best_sc = -1
    best_ss = (1 << 64) - 1
    best_a, best_b = sa, sb

    if _cc_bw_ok(sa, sb):
        best_sc = 0
        best_ss = 0

    for ps in ca:
        ta = _cc_apply_s2pf(sa, s3a, ps)
        if ta.s2pf_start < 0:
            continue
        if _cc_bw_ok(ta, sb):
            ss = ps
            if 1 > best_sc or (best_sc == 1 and ss < best_ss):
                best_sc, best_ss, best_a, best_b = 1, ss, ta, sb

    for ps in cb:
        tb = _cc_apply_s2pf(sb, s3b, ps)
        if tb.s2pf_start < 0:
            continue
        if _cc_bw_ok(sa, tb):
            ss = ps
            if 1 > best_sc or (best_sc == 1 and ss < best_ss):
                best_sc, best_ss, best_a, best_b = 1, ss, sa, tb

    for psa in ca:
        ta = _cc_apply_s2pf(sa, s3a, psa)
        if ta.s2pf_start < 0:
            continue
        for psb in cb:
            tb = _cc_apply_s2pf(sb, s3b, psb)
            if tb.s2pf_start < 0:
                continue
            if not _cc_bw_ok(ta, tb):
                continue
            ss = psa + psb
            if 2 > best_sc or (best_sc == 2 and ss < best_ss):
                best_sc, best_ss, best_a, best_b = 2, ss, ta, tb

    return best_a, best_b


def _cc_greedy_h(c2e: int, c3e: int, rem: tuple) -> int:
    max_e = max(c2e, c3e)
    nr = len(rem)
    if nr == 0:
        return max_e
    if nr == 1:
        nt = rem[0][1]
        te, tl = min(c2e, c3e), max_e
        sc = max(tl, te + _cc_best_task(nt))
        sp = tl + _cc_best_conc((nt + 1) // 2)
        return min(sc, sp)
    if nr == 2:
        te, tl = min(c2e, c3e), max_e
        bc0 = _cc_best_conc(rem[0][1])
        bc1 = _cc_best_conc(rem[1][1])
        pc = tl + max(bc0, bc1)
        ser = te + _cc_best_task(rem[0][1]) + _cc_best_task(rem[1][1])
        return min(pc, max(ser, tl))
    vals = [_cc_best_conc(nt) for _, nt in rem]
    return max_e + max(max(vals), sum(vals) // 2)


def _cc_sim1(c2: CSnap, c3: CSnap, eid: int, ntok: int) -> int:
    t = max(c2.task_end, c3.task_end)
    best = C_INF
    for sn_ci in (c2, c3):
        cc = _cc_swiglu_hit(eid, sn_ci, t)
        cf = _cc_down_hit(eid, sn_ci, t)
        sn = _cc_mk_snap(t, C_SHAPE_C, C_SHAPE_C, ntok, eid, cc, cf)
        best = min(best, sn.task_end)
        if not cc:
            sn2 = _cc_mk_snap(t, C_SHAPE_B, C_SHAPE_B, ntok, eid, False, False)
            best = min(best, sn2.task_end)

    if ntok >= 2:
        ca = (ntok + 1) // 2
        cb = ntok - ca
        sw_a = _cc_swiglu_hit(eid, c2, t)
        dn_a = _cc_down_hit(eid, c2, t)
        sw_b = _cc_swiglu_hit(eid, c3, t)
        dn_b = _cc_down_hit(eid, c3, t)
        s1a, s3a, s1b, s3b = _cc_pick_shapes(ca, cb, sw_a, dn_a, sw_b, dn_b, t)
        sna = _cc_mk_snap(t, s1a, s3a, ca, eid, sw_a, dn_a)
        snb = _cc_mk_snap(t, s1b, s3b, cb, eid, sw_b, dn_b)
        sna, snb = _cc_try_s2pf_pair(sna, s3a, snb, s3b)
        if _cc_bw_ok(sna, snb):
            best = min(best, max(sna.task_end, snb.task_end))

    return t + _cc_best_task(ntok) if best == C_INF else best


def _cc_continuation_cost(c2: CSnap, c3: CSnap, rem: tuple) -> int:
    nr = len(rem)
    if nr == 0:
        return max(c2.task_end, c3.task_end)
    if nr == 1:
        return _cc_sim1(c2, c3, rem[0][0], rem[0][1])
    if nr == 2 and rem[0][1] + rem[1][1] <= C_EXACT_TAIL_MAX:
        te = min(c2.task_end, c3.task_end)
        tl = max(c2.task_end, c3.task_end)
        ss = te + _cc_best_task(rem[0][1]) + _cc_best_task(rem[1][1])
        pa = tl + max(_cc_best_conc(rem[0][1]), _cc_best_conc(rem[1][1]))
        return min(max(tl, ss), pa)
    return _cc_greedy_h(c2.task_end, c3.task_end, rem)


def _cc_cand_better(best, cost: int, smx: int, smn: int, rem_len: int) -> bool:
    if best is None:
        return True
    bcost, bsmx, bsmn, blen = best
    if cost < bcost:
        return True
    if cost > bcost:
        return False
    if rem_len < blen:
        return True
    if rem_len > blen:
        return False
    if smx < bsmx:
        return True
    if smx > bsmx:
        return False
    return smn > bsmn


def _cc_busy_time_points(busy: CSnap, idle_t: int):
    pts = [idle_t]
    segs = _cc_snap_segs(busy) or []
    for _, hi, _ in segs:
        if len(pts) >= 3:
            break
        if hi > idle_t and hi not in pts:
            pts.append(hi)
    return pts


def _cc_remove_eids(rem: tuple, *eids: int) -> tuple:
    dead = set(eids)
    return tuple(r for r in rem if r[0] not in dead)


def c_mirror_v2_schedule(
    token_dist: dict,
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
) -> int:
    """Mirror the current workload moe_scheduler.c moe_plan() makespan."""
    remaining = tuple(sorted(((int(e), int(n)) for e, n in token_dist.items()), key=lambda x: -x[1]))
    c2 = _cc_initial(initial_cache_c2)
    c3 = _cc_initial(initial_cache_c3)

    while remaining:
        top0_eid, top0_ntok = remaining[0]
        t2, t3 = c2.task_end, c3.task_end
        tnow = max(t2, t3)
        both_idle = t2 == t3

        if _cc_s4pf_ok_with_peer(c2, c3):
            c2 = _cc_apply_s4pf_ghost(c2)
        if _cc_s4pf_ok_with_peer(c3, c2):
            c3 = _cc_apply_s4pf_ghost(c3)

        c2c0 = _cc_swiglu_hit(top0_eid, c2, tnow)
        c2f0 = _cc_down_hit(top0_eid, c2, tnow)
        c3c0 = _cc_swiglu_hit(top0_eid, c3, tnow)
        c3f0 = _cc_down_hit(top0_eid, c3, tnow)

        if len(remaining) == 1:
            best_cost = C_INF
            best_sn = None
            best_cl = 0
            is_split = False
            split_snb = None

            for ci in (0, 1):
                snap_ci = c2 if ci == 0 else c3
                peer = c3 if ci == 0 else c2
                tst = snap_ci.task_end
                cc = _cc_swiglu_hit(top0_eid, snap_ci, tst)
                cf = _cc_down_hit(top0_eid, snap_ci, tst)
                for s1 in (0, 1, 2):
                    for s3 in (0, 1, 2):
                        sn = _cc_mk_snap(tst, s1, s3, top0_ntok, top0_eid, cc, cf)
                        if not _cc_bw_ok(sn, peer):
                            continue
                        ms = max(sn.task_end, peer.task_end)
                        if ms < best_cost:
                            best_cost = ms
                            best_sn = sn
                            best_cl = ci
                            is_split = False

            if top0_ntok >= 2:
                cuts = []
                h1 = (top0_ntok + 1) // 2
                h2 = top0_ntok // 2
                cuts.append(h1)
                if h2 != h1 and 1 <= h2 <= top0_ntok - 1:
                    cuts.append(h2)
                for cut_a in cuts:
                    cut_b = top0_ntok - cut_a
                    s1a, s3a, s1b, s3b = _cc_pick_shapes(
                        cut_a, cut_b, c2c0, c2f0, c3c0, c3f0, tnow
                    )
                    sna = _cc_mk_snap(tnow, s1a, s3a, cut_a, top0_eid, c2c0, c2f0)
                    snb = _cc_mk_snap(tnow, s1b, s3b, cut_b, top0_eid, c3c0, c3f0)
                    sna, snb = _cc_try_s2pf_pair(sna, s3a, snb, s3b)
                    if not _cc_bw_ok(sna, snb):
                        continue
                    e = max(sna.task_end, snb.task_end)
                    if e < best_cost:
                        best_cost = e
                        best_sn = sna
                        split_snb = snb
                        is_split = True

            if not both_idle:
                idle_ci = 0 if t2 < t3 else 1
                idle_s = c2 if idle_ci == 0 else c3
                busy_s = c3 if idle_ci == 0 else c2
                idle_t = t2 if idle_ci == 0 else t3
                for tst in _cc_busy_time_points(busy_s, idle_t):
                    cc = _cc_swiglu_hit(top0_eid, idle_s, tst)
                    cf = _cc_down_hit(top0_eid, idle_s, tst)
                    sn = _cc_mk_snap(tst, C_SHAPE_C, C_SHAPE_C, top0_ntok, top0_eid, cc, cf)
                    ok = _cc_bw_ok(sn, busy_s) if idle_ci == 0 else _cc_bw_ok(busy_s, sn)
                    if not ok:
                        continue
                    ms = max(sn.task_end, busy_s.task_end)
                    if ms < best_cost:
                        best_cost = ms
                        best_sn = sn
                        best_cl = idle_ci
                        is_split = False

            remaining = ()
            if is_split:
                c2, c3 = best_sn, split_snb
            else:
                if best_cl == 0:
                    c2 = best_sn
                else:
                    c3 = best_sn
            break

        if both_idle:
            best_key = None
            best_snap = None
            best_rem = None

            def eval_pair(sa, s1a, s3a, sb, s1b, s3b, rem_after):
                nonlocal best_key, best_snap, best_rem
                ta, tb = _cc_try_s2pf_pair(sa, s3a, sb, s3b)
                if not _cc_bw_ok(ta, tb):
                    return
                cost = _cc_continuation_cost(ta, tb, rem_after)
                smx = max(ta.task_end, tb.task_end)
                smn = min(ta.task_end, tb.task_end)
                key = (cost, smx, smn, len(rem_after))
                if _cc_cand_better(best_key, cost, smx, smn, len(rem_after)):
                    best_key = key
                    best_snap = (ta, tb)
                    best_rem = rem_after

            max_k = min(3, len(remaining) - 1)
            for k in range(1, max_k + 1):
                keid, kntok = remaining[k]
                rem_after = _cc_remove_eids(remaining, top0_eid, keid)

                sw_a, dn_a = c2c0, c2f0
                sw_b = _cc_swiglu_hit(keid, c3, tnow)
                dn_b = _cc_down_hit(keid, c3, tnow)
                s1a, s3a, s1b, s3b = _cc_pick_shapes(
                    top0_ntok, kntok, sw_a, dn_a, sw_b, dn_b, tnow
                )
                sa = _cc_mk_snap(tnow, s1a, s3a, top0_ntok, top0_eid, sw_a, dn_a)
                sb = _cc_mk_snap(tnow, s1b, s3b, kntok, keid, sw_b, dn_b)
                eval_pair(sa, s1a, s3a, sb, s1b, s3b, rem_after)

                sw_a = _cc_swiglu_hit(keid, c2, tnow)
                dn_a = _cc_down_hit(keid, c2, tnow)
                sw_b, dn_b = c3c0, c3f0
                s1a, s3a, s1b, s3b = _cc_pick_shapes(
                    kntok, top0_ntok, sw_a, dn_a, sw_b, dn_b, tnow
                )
                sa = _cc_mk_snap(tnow, s1a, s3a, kntok, keid, sw_a, dn_a)
                sb = _cc_mk_snap(tnow, s1b, s3b, top0_ntok, top0_eid, sw_b, dn_b)
                eval_pair(sa, s1a, s3a, sb, s1b, s3b, rem_after)

            if len(remaining) >= 3:
                mkj = min(3, len(remaining) - 1)
                for k in range(1, mkj):
                    for j in range(k + 1, mkj + 1):
                        if j >= len(remaining):
                            continue
                        eid_k, nt_k = remaining[k]
                        eid_j, nt_j = remaining[j]
                        rem_after = _cc_remove_eids(remaining, eid_k, eid_j)
                        if not rem_after:
                            continue

                        sw_a = _cc_swiglu_hit(eid_k, c2, tnow)
                        dn_a = _cc_down_hit(eid_k, c2, tnow)
                        sw_b = _cc_swiglu_hit(eid_j, c3, tnow)
                        dn_b = _cc_down_hit(eid_j, c3, tnow)
                        s1a, s3a, s1b, s3b = _cc_pick_shapes(nt_k, nt_j, sw_a, dn_a, sw_b, dn_b, tnow)
                        sa = _cc_mk_snap(tnow, s1a, s3a, nt_k, eid_k, sw_a, dn_a)
                        sb = _cc_mk_snap(tnow, s1b, s3b, nt_j, eid_j, sw_b, dn_b)
                        eval_pair(sa, s1a, s3a, sb, s1b, s3b, rem_after)

                        sw_a = _cc_swiglu_hit(eid_j, c2, tnow)
                        dn_a = _cc_down_hit(eid_j, c2, tnow)
                        sw_b = _cc_swiglu_hit(eid_k, c3, tnow)
                        dn_b = _cc_down_hit(eid_k, c3, tnow)
                        s1a, s3a, s1b, s3b = _cc_pick_shapes(nt_j, nt_k, sw_a, dn_a, sw_b, dn_b, tnow)
                        sa = _cc_mk_snap(tnow, s1a, s3a, nt_j, eid_j, sw_a, dn_a)
                        sb = _cc_mk_snap(tnow, s1b, s3b, nt_k, eid_k, sw_b, dn_b)
                        eval_pair(sa, s1a, s3a, sb, s1b, s3b, rem_after)

            if top0_ntok >= 2:
                cuts = []
                h1 = (top0_ntok + 1) // 2
                h2 = top0_ntok // 2
                cuts.append(h1)
                if h2 != h1 and h2 >= 1:
                    cuts.append(h2)
                for md in (8, 4, 2):
                    if md < top0_ntok and md not in cuts:
                        cuts.append(md)
                    if top0_ntok > md:
                        k2 = top0_ntok - md
                        if k2 >= 1 and k2 not in cuts:
                            cuts.append(k2)

                rem_after = _cc_remove_eids(remaining, top0_eid)
                for cut_a in cuts:
                    cut_b = top0_ntok - cut_a
                    if cut_a == 0 or cut_b == 0:
                        continue
                    s1a, s3a, s1b, s3b = _cc_pick_shapes(
                        cut_a, cut_b, c2c0, c2f0, c3c0, c3f0, tnow
                    )
                    sa = _cc_mk_snap(tnow, s1a, s3a, cut_a, top0_eid, c2c0, c2f0)
                    sb = _cc_mk_snap(tnow, s1b, s3b, cut_b, top0_eid, c3c0, c3f0)
                    eval_pair(sa, s1a, s3a, sb, s1b, s3b, rem_after)

            if best_snap is not None:
                c2, c3 = best_snap
                remaining = best_rem
            else:
                c2 = _cc_mk_snap(tnow, C_SHAPE_C, C_SHAPE_C, top0_ntok, top0_eid, c2c0, c2f0)
                remaining = remaining[1:]
            continue

        idle_ci = 0 if t2 < t3 else 1
        idle_sn = c2 if idle_ci == 0 else c3
        busy_sn = c3 if idle_ci == 0 else c2
        idle_t = t2 if idle_ci == 0 else t3

        best_ms = C_INF
        best_nb = None
        for tst in _cc_busy_time_points(busy_sn, idle_t):
            cc = _cc_swiglu_hit(top0_eid, idle_sn, tst)
            cf = _cc_down_hit(top0_eid, idle_sn, tst)
            sn = _cc_mk_snap(tst, C_SHAPE_C, C_SHAPE_C, top0_ntok, top0_eid, cc, cf)
            if sn.bw_s3 > 0 and C_TD3[C_SHAPE_C] <= sn.s2_end - sn.dma1_end:
                hi = sn.s2_end - C_TD3[C_SHAPE_C]
                cand = _cc_apply_s2pf(sn, C_SHAPE_C, hi)
                if cand.s2pf_start >= 0:
                    ok2 = _cc_bw_ok(cand, busy_sn) if idle_ci == 0 else _cc_bw_ok(busy_sn, cand)
                    if ok2:
                        sn = cand
            ok = _cc_bw_ok(sn, busy_sn) if idle_ci == 0 else _cc_bw_ok(busy_sn, sn)
            if not ok:
                continue
            ms = max(sn.task_end, busy_sn.task_end)
            if ms < best_ms:
                best_ms = ms
                best_nb = sn

        remaining = remaining[1:]
        if best_nb is not None:
            if idle_ci == 0:
                c2 = best_nb
            else:
                c3 = best_nb
        else:
            cch = c2c0 if idle_ci == 0 else c3c0
            cfh = c2f0 if idle_ci == 0 else c3f0
            sf = _cc_mk_snap(idle_t, C_SHAPE_C, C_SHAPE_C, top0_ntok, top0_eid, cch, cfh)
            if idle_ci == 0:
                c2 = sf
            else:
                c3 = sf

    return max(c2.task_end, c3.task_end)


# ─────────────────────────────────────────────────────────────────────────────
# 三方对比（两阶段）
#   Phase 1: c_mirror_v2  vs analytical — 全量 10K（快，秒级）
#   Phase 2: fast_schedule vs analytical — 抽样 500 条（慢，fast 需 ~90s）
# ─────────────────────────────────────────────────────────────────────────────
def _load_cache():
    if not os.path.exists(CACHE_PATH):
        print(f"缓存文件不存在: {CACHE_PATH}")
        print("请先运行: python3 Idea_Model/precompute_analytical.py")
        sys.exit(1)
    with open(CACHE_PATH) as f:
        data = json.load(f)
    print(f"加载缓存: {len(data)} 条 analytical 结果")
    return data


def _pct(v, total):
    return f"{v/total*100:>6.2f}%"


def _print_stats(name, ratios, times, n_col=16):
    n = len(ratios)
    print(f"\n{'─'*55}")
    print(f"  {name}  (n={n})")
    print(f"{'─'*55}")
    print(f"  均值 ratio (vs analytical)  {sum(ratios)/n:>10.5f}")
    print(f"  最大 ratio                  {max(ratios):>10.5f}")
    print(
        f"  完全匹配  (≤1.001)          {sum(1 for r in ratios if r<=1.001)/n*100:>9.2f}%"
    )
    print(
        f"  <1%  退化 (≤1.010)          {sum(1 for r in ratios if r<=1.010)/n*100:>9.2f}%"
    )
    print(
        f"  <2%  退化 (≤1.020)          {sum(1 for r in ratios if r<=1.020)/n*100:>9.2f}%"
    )
    print(
        f"  <5%  退化 (≤1.050)          {sum(1 for r in ratios if r<=1.050)/n*100:>9.2f}%"
    )
    print(
        f"  <10% 退化 (≤1.100)          {sum(1 for r in ratios if r<=1.100)/n*100:>9.2f}%"
    )
    if times:
        print(
            f"  均时                        {sum(times)/len(times)*1e6:>9.1f} μs/case"
        )


if __name__ == "__main__":
    import random

    data = _load_cache()
    n_total = len(data)

    # ── Phase 1: c_mirror_v2 全量 10K ─────────────────────────────────────
    print(f"\n[Phase 1] c_mirror_v2 — 全量 {n_total} 个 case……", flush=True)
    t0_p1 = time.perf_counter()
    ratios_v2 = []
    times_v2 = []
    crashes_v2 = 0
    worst_v2_recs = []

    for idx, rec in enumerate(data):
        dist = {int(k): v for k, v in rec["dist"].items()}
        c2_i = rec["c2"]
        c3_i = rec["c3"]
        a = rec["analytical"]
        if a <= 0:
            continue
        try:
            t0 = time.perf_counter()
            v2 = c_mirror_v2_schedule(dict(dist), c2_i, c3_i)
            times_v2.append(time.perf_counter() - t0)
            r = v2 / a
            ratios_v2.append(r)
            worst_v2_recs.append(
                (r, sorted(dist.values(), reverse=True), c2_i, c3_i, a, v2)
            )
        except Exception:
            crashes_v2 += 1

        if (idx + 1) % 2000 == 0:
            print(
                f"  {idx+1}/{n_total} done ({time.perf_counter()-t0_p1:.0f}s)",
                flush=True,
            )

    print(
        f"  Phase 1 完成，耗时 {time.perf_counter()-t0_p1:.1f}s，crashes={crashes_v2}"
    )
    _print_stats("c_mirror_v2 (10K)", ratios_v2, times_v2)

    worst_v2_recs.sort(key=lambda x: x[0], reverse=True)
    if worst_v2_recs and worst_v2_recs[0][0] > 1.001:
        print("\n  c_mirror_v2 差距最大前 10 case:")
        for r, toks, c2i, c3i, a, v2 in worst_v2_recs[:10]:
            print(
                f"    ratio={r:.4f}  toks={toks}  c2={c2i}  c3={c3i}  anal={a}  v2={v2}"
            )

    # ── Phase 2: fast_schedule 抽样 500 条 ────────────────────────────────
    from fast_scheduler import fast_schedule

    N_FAST = 500
    sample_idx = list(range(0, n_total, n_total // N_FAST))[:N_FAST]
    sample = [data[i] for i in sample_idx]

    print(f"\n[Phase 2] fast_schedule — 抽样 {len(sample)} 个 case……", flush=True)
    t0_p2 = time.perf_counter()
    ratios_fast = []
    ratios_v2_s = []  # c_mirror_v2 在同一抽样上的结果（便于直接对比）
    times_fast = []
    crashes_fast = 0

    for idx, rec in enumerate(sample):
        dist = {int(k): v for k, v in rec["dist"].items()}
        c2_i = rec["c2"]
        c3_i = rec["c3"]
        a = rec["analytical"]
        if a <= 0:
            continue
        try:
            t0 = time.perf_counter()
            f = fast_schedule(dict(dist), c2_i, c3_i)
            times_fast.append(time.perf_counter() - t0)
            ratios_fast.append(f / a)
        except Exception:
            crashes_fast += 1

        try:
            v2 = c_mirror_v2_schedule(dict(dist), c2_i, c3_i)
            ratios_v2_s.append(v2 / a)
        except Exception:
            pass

        if (idx + 1) % 100 == 0:
            print(
                f"  {idx+1}/{len(sample)} done ({time.perf_counter()-t0_p2:.0f}s)",
                flush=True,
            )

    print(
        f"  Phase 2 完成，耗时 {time.perf_counter()-t0_p2:.1f}s，crashes={crashes_fast}"
    )

    # ── 汇总输出 ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  三方汇总对比（基准：analytical_schedule）")
    print(f"{'='*60}")
    print(f"  {'指标':<30} {'fast_sched':>10} {'c_mirr_v2':>10}")
    print(f"  {'(抽样 500)':<30} {'(500)':>10} {'(500)':>10}")
    print(f"  {'-'*52}")
    n_f = len(ratios_fast)
    n_v = len(ratios_v2_s)
    if n_f > 0 and n_v > 0:
        print(
            f"  {'均值 ratio':<30} {sum(ratios_fast)/n_f:>10.5f} {sum(ratios_v2_s)/n_v:>10.5f}"
        )
        print(
            f"  {'最大 ratio':<30} {max(ratios_fast):>10.5f} {max(ratios_v2_s):>10.5f}"
        )
        print(
            f"  {'完全匹配 (≤1.001)':<30} {sum(1 for r in ratios_fast if r<=1.001)/n_f*100:>9.2f}% {sum(1 for r in ratios_v2_s if r<=1.001)/n_v*100:>9.2f}%"
        )
        print(
            f"  {'<1% 退化 (≤1.010)':<30} {sum(1 for r in ratios_fast if r<=1.010)/n_f*100:>9.2f}% {sum(1 for r in ratios_v2_s if r<=1.010)/n_v*100:>9.2f}%"
        )
        print(
            f"  {'<5% 退化 (≤1.050)':<30} {sum(1 for r in ratios_fast if r<=1.050)/n_f*100:>9.2f}% {sum(1 for r in ratios_v2_s if r<=1.050)/n_v*100:>9.2f}%"
        )
        print(
            f"  {'<10% 退化 (≤1.100)':<30} {sum(1 for r in ratios_fast if r<=1.100)/n_f*100:>9.2f}% {sum(1 for r in ratios_v2_s if r<=1.100)/n_v*100:>9.2f}%"
        )
    if times_fast:
        print(
            f"\n  fast_schedule 均时    {sum(times_fast)/len(times_fast)*1e3:>8.2f} ms/case"
        )
    if times_v2:
        print(
            f"  c_mirror_v2  均时     {sum(times_v2)/len(times_v2)*1e6:>8.1f} μs/case"
        )
    print(f"{'='*60}")
