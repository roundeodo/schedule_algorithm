#!/usr/bin/env python3
"""
eval_fast_schedule.py
======================
对比三种调度器质量（10000 个随机分布）：
  1. analytical_schedule  — 参考基准
  2. lite_schedule        — 原始精简版
  3. fast_lite_schedule   — 本文件实现，镜像 C 简化版
                            (3 解析时间点 × ShapeC，无 9×9 形状搜索)

fast_lite_schedule 与 lite_schedule 的差异：
  - not_both_idle 主循环：
      原: {idle_t} | busy.bw_change_pts()  (最多 9 点) × 9 形状
      新: {idle_t, dma1_end, dma3_end[, s2pf_end]} (最多 3 点) × ShapeC
  - n=1 Method B（lite_schedule 已是 ShapeC，时间点未简化）：
      原: {idle_t} | busy.bw_change_pts() | busy_no_pf.bw_change_pts() × ShapeC
      新: {idle_t, dma1_end, dma3_end} (最多 3 点) × ShapeC
"""

import sys, os, random, time, math

sys.path.insert(0, os.path.dirname(__file__))

from four_stage_scheduler import (
    SHAPE_A,
    SHAPE_B,
    SHAPE_C,
    ALL_SHAPES,
    MAX_BW,
    FourStageSnap,
    make_initial_snap,
    bw_feasible,
    with_optional_s2_down_prefetch,
    with_optional_s2_down_prefetch_pair,
    with_optional_next_s1_prefetch_pair,
    _best_task_time,
    _best_s2_compute,
)
from analytical_scheduler import analytical_schedule
from lite_scheduler import (
    lite_schedule,
    _greedy_heuristic,
    best_solo_shape_s1,
    best_solo_shape_s3,
)

EXACT_TAIL_MAX = 4

_T_DMA_S3_C = SHAPE_C.t_dma_s3  # = 11264 cc


def _pick_pair_shapes(ntok_A, ntok_B, sw_A, dn_A, sw_B, dn_B, t_now):
    """解析 O(1) 形状选择，与 lite_scheduler._pick_pair_shapes 完全相同。"""
    if sw_A or sw_B:
        s1_A = SHAPE_C
        s1_B = SHAPE_C
    else:
        s1_A = SHAPE_B
        s1_B = SHAPE_B

    if sw_A:
        s2_A = t_now + _best_s2_compute(ntok_A)
    else:
        s2_A = t_now + s1_A.T_s1 + _best_s2_compute(max(0, ntok_A - s1_A.M_dim))

    if sw_B:
        s2_B = t_now + _best_s2_compute(ntok_B)
    else:
        s2_B = t_now + s1_B.T_s1 + _best_s2_compute(max(0, ntok_B - s1_B.M_dim))

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


def _dma_hi_pts(snap: FourStageSnap) -> set:
    """取 busy cluster 的 DMA 完成时刻（BW 释放点）作为候选起始时间。
    等价于 C 代码 snap_segs() 各段的 hi 端点。"""
    pts = {snap.dma1_end, snap.dma3_end}
    if snap.s2pf_end >= 0:
        pts.add(snap.s2pf_end)
    return pts


def fast_lite_schedule(
    token_dist: dict,
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
) -> int:
    """
    精简调度器：镜像 C moe_scheduler.c 的简化版。
    与 lite_schedule 相同，除了：
      - not_both_idle 时间点：最多 3 个（idle_t + DMA 结束时刻）
      - not_both_idle 形状：固定 ShapeC（最快；BW 冲突自动跳到下一时间点）
    """
    remaining = tuple(sorted(token_dist.items(), key=lambda x: -x[1]))

    c2 = make_initial_snap(initial_cache_c2)
    c3 = make_initial_snap(initial_cache_c3)

    def swiglu_hit(eid, snap, t):
        return snap.pf_eid == eid and snap.pf_end >= 0 and snap.pf_end <= t

    def down_hit(eid, snap, t):
        return swiglu_hit(eid, snap, t) and snap.pf_full

    def _sim1(c2_sn, c3_sn, e_eid, e_ntok):
        """lite_schedule 内 _sim1 的精简版（无修改）。"""
        deadline = max(c2_sn.task_end, c3_sn.task_end)

        def _eval_pf_pair(c2_s, c3_s):
            t2s, t3s = c2_s.task_end, c3_s.task_end
            now_s = max(t2s, t3s)
            c2_sw_s = swiglu_hit(e_eid, c2_s, now_s)
            c3_sw_s = swiglu_hit(e_eid, c3_s, now_s)
            c2_dn_s = down_hit(e_eid, c2_s, now_s)
            c3_dn_s = down_hit(e_eid, c3_s, now_s)
            best_s = None

            for s1 in ALL_SHAPES:
                for s3 in ALL_SHAPES:
                    for sw, dn in [(c2_sw_s, c2_dn_s), (c3_sw_s, c3_dn_s)]:
                        try:
                            sn = FourStageSnap.from_assign(
                                now_s, s1, s3, e_ntok, e_eid, sw, dn
                            )
                            cost = sn.task_end
                            if best_s is None or cost < best_s:
                                best_s = cost
                        except Exception:
                            pass

            if e_ntok >= 2:
                s_cuts = {math.ceil(e_ntok / 2), e_ntok // 2}
                for cut_A in s_cuts:
                    cut_B = e_ntok - cut_A
                    for s1_A in ALL_SHAPES:
                        for s1_B in ALL_SHAPES:
                            for s3_A in ALL_SHAPES:
                                for s3_B in ALL_SHAPES:
                                    try:
                                        sna = FourStageSnap.from_assign(
                                            now_s,
                                            s1_A,
                                            s3_A,
                                            cut_A,
                                            e_eid,
                                            c2_sw_s,
                                            c2_dn_s,
                                        )
                                        snb = FourStageSnap.from_assign(
                                            now_s,
                                            s1_B,
                                            s3_B,
                                            cut_B,
                                            e_eid,
                                            c3_sw_s,
                                            c3_dn_s,
                                        )
                                        sna, snb = with_optional_s2_down_prefetch_pair(
                                            sna, s3_A, snb, s3_B
                                        )
                                        if not bw_feasible(sna, snb):
                                            continue
                                        cost = max(sna.task_end, snb.task_end)
                                        if best_s is None or cost < best_s:
                                            best_s = cost
                                    except Exception:
                                        pass

            if t2s != t3s:
                if t2s < t3s:
                    idle_t_s, idle_sn_s, busy_sn_s = t2s, c2_s, c3_s
                    is_c2_idle = True
                else:
                    idle_t_s, idle_sn_s, busy_sn_s = t3s, c3_s, c2_s
                    is_c2_idle = False
                # ── 简化：3 时间点 × ShapeC ──────────────────────────────────
                hi_pts = _dma_hi_pts(busy_sn_s)
                try_starts_s = sorted({idle_t_s} | {t for t in hi_pts if t > idle_t_s})[
                    :3
                ]
                for t_st in try_starts_s:
                    sw = swiglu_hit(e_eid, idle_sn_s, t_st)
                    dn = down_hit(e_eid, idle_sn_s, t_st)
                    try:
                        sn = FourStageSnap.from_assign(
                            t_st, SHAPE_C, SHAPE_C, e_ntok, e_eid, sw, dn
                        )
                        sn = with_optional_s2_down_prefetch(sn, SHAPE_C, busy_sn_s)
                        if is_c2_idle:
                            if not bw_feasible(sn, busy_sn_s):
                                continue
                        else:
                            if not bw_feasible(busy_sn_s, sn):
                                continue
                        cost = max(sn.task_end, busy_sn_s.task_end)
                        if best_s is None or cost < best_s:
                            best_s = cost
                    except Exception:
                        pass

            return best_s

        pf_pairs = []
        c2_pf, c3_pf = with_optional_next_s1_prefetch_pair(c2_sn, c3_sn, e_eid)
        pf_pairs.append((c2_pf, c3_pf))
        for pf_shape in ALL_SHAPES:
            for sn_pf, sn_raw, is_c2 in [(c2_sn, c3_sn, True), (c3_sn, c2_sn, False)]:
                pf_start = max(sn_pf.s4_start, sn_pf.dma3_end)
                if pf_start < 0:
                    continue
                try:
                    cand = sn_pf.with_prefetch(e_eid, pf_shape, pf_start)
                    if cand.pf_end <= deadline and bw_feasible(
                        (cand if is_c2 else sn_raw),
                        (sn_raw if is_c2 else cand),
                    ):
                        pf_pairs.append((cand, sn_raw) if is_c2 else (sn_raw, cand))
                except Exception:
                    pass

        best_ms = None
        for c2_s, c3_s in pf_pairs:
            v = _eval_pf_pair(c2_s, c3_s)
            if v is not None and (best_ms is None or v < best_ms):
                best_ms = v
        return best_ms if best_ms is not None else (deadline + _best_task_time(e_ntok))

    # ── 主循环 ────────────────────────────────────────────────────────────────
    while remaining:
        top0_eid, top0_ntok = remaining[0]
        t2, t3 = c2.task_end, c3.task_end
        now = max(t2, t3)
        c2_sw_0 = swiglu_hit(top0_eid, c2, now)
        c3_sw_0 = swiglu_hit(top0_eid, c3, now)
        c2_dn_0 = down_hit(top0_eid, c2, now)
        c3_dn_0 = down_hit(top0_eid, c3, now)

        # ── n=1 ───────────────────────────────────────────────────────────────
        if len(remaining) == 1:
            best_cost_n1 = None
            best_snap_n1 = None

            c2_before_pf, c3_before_pf = c2, c3
            c2, c3 = with_optional_next_s1_prefetch_pair(c2, c3, top0_eid)

            # Method A
            for s1 in ALL_SHAPES:
                for s3 in ALL_SHAPES:
                    for ci, (sn_ci, sw, dn) in enumerate(
                        [
                            (c2, c2_sw_0, c2_dn_0),
                            (c3, c3_sw_0, c3_dn_0),
                        ]
                    ):
                        try:
                            sn = FourStageSnap.from_assign(
                                now, s1, s3, top0_ntok, top0_eid, sw, dn
                            )
                            cost = max(
                                sn.task_end, c3.task_end if ci == 0 else c2.task_end
                            )
                            if best_cost_n1 is None or cost < best_cost_n1:
                                best_cost_n1 = cost
                                best_snap_n1 = ("C2" if ci == 0 else "C3", sn)
                        except Exception:
                            pass

            # SPLIT
            if top0_ntok >= 2:
                s_cuts = {math.ceil(top0_ntok / 2), top0_ntok // 2}
                for mi in [8, 4, 2]:
                    if mi < top0_ntok:
                        s_cuts.add(mi)
                    if top0_ntok > mi:
                        s_cuts.add(top0_ntok - mi)
                for cut_A in s_cuts:
                    cut_B = top0_ntok - cut_A
                    if cut_A <= 0 or cut_B <= 0:
                        continue
                    for s1_A, s3_A, s1_B, s3_B in _pick_pair_shapes(
                        cut_A, cut_B, c2_sw_0, c2_dn_0, c3_sw_0, c3_dn_0, now
                    ):
                        try:
                            sna = FourStageSnap.from_assign(
                                now, s1_A, s3_A, cut_A, top0_eid, c2_sw_0, c2_dn_0
                            )
                            snb = FourStageSnap.from_assign(
                                now, s1_B, s3_B, cut_B, top0_eid, c3_sw_0, c3_dn_0
                            )
                            sna, snb = with_optional_s2_down_prefetch_pair(
                                sna, s3_A, snb, s3_B
                            )
                            if not bw_feasible(sna, snb):
                                continue
                            cost = max(sna.task_end, snb.task_end)
                            if best_cost_n1 is None or cost < best_cost_n1:
                                best_cost_n1 = cost
                                best_snap_n1 = ("split", sna, snb)
                        except Exception:
                            pass

            # Method B — 【简化】3 时间点 × ShapeC
            if t2 != t3:
                if t2 < t3:
                    idle_t_n1, idle_cl_n1 = t2, "C2"
                    busy_sn_n1, idle_sn_n1 = c3, c2
                    busy_no_pf_n1 = c3_before_pf
                else:
                    idle_t_n1, idle_cl_n1 = t3, "C3"
                    busy_sn_n1, idle_sn_n1 = c2, c3
                    busy_no_pf_n1 = c2_before_pf

                hi_pts = _dma_hi_pts(busy_sn_n1) | _dma_hi_pts(busy_no_pf_n1)
                try_starts_n1 = sorted(
                    {idle_t_n1} | {t for t in hi_pts if t > idle_t_n1}
                )[:3]

                for t_start in try_starts_n1:
                    hit = swiglu_hit(top0_eid, idle_sn_n1, t_start)
                    full = down_hit(top0_eid, idle_sn_n1, t_start)
                    for busy_alt in [busy_sn_n1, busy_no_pf_n1]:
                        try:
                            new_sn = FourStageSnap.from_assign(
                                t_start,
                                SHAPE_C,
                                SHAPE_C,
                                top0_ntok,
                                top0_eid,
                                hit,
                                full,
                            )
                            new_sn = with_optional_s2_down_prefetch(
                                new_sn, SHAPE_C, busy_alt
                            )
                            if idle_cl_n1 == "C2":
                                if not bw_feasible(new_sn, busy_alt):
                                    continue
                            else:
                                if not bw_feasible(busy_alt, new_sn):
                                    continue
                            cost = max(new_sn.task_end, busy_sn_n1.task_end)
                            if best_cost_n1 is None or cost < best_cost_n1:
                                best_cost_n1 = cost
                                best_snap_n1 = (idle_cl_n1, new_sn)
                        except Exception:
                            pass

            if best_snap_n1 is None:
                s1 = best_solo_shape_s1(top0_ntok)
                s3 = best_solo_shape_s3(top0_ntok)
                c2 = FourStageSnap.from_assign(
                    now, s1, s3, top0_ntok, top0_eid, c2_sw_0, c2_dn_0
                )
            elif best_snap_n1[0] == "split":
                _, sna, snb = best_snap_n1
                c2, c3 = sna, snb
            else:
                which, sn = best_snap_n1[0], best_snap_n1[1]
                if which == "C2":
                    c2 = sn
                else:
                    c3 = sn
            remaining = ()
            break

        # ── both_idle ─────────────────────────────────────────────────────────
        if t2 == t3:
            # [以下与 lite_schedule both_idle 完全相同]
            best_snap = None
            best_cost = None
            best_rem = None

            def _eval_pair(sna, s3a, snb, s3b, rem_after):
                nonlocal best_snap, best_cost, best_rem
                try:
                    ta, tb = with_optional_s2_down_prefetch_pair(sna, s3a, snb, s3b)
                    if not bw_feasible(ta, tb):
                        return
                    ms = max(ta.task_end, tb.task_end)
                    cost = _greedy_heuristic(ta.task_end, tb.task_end, rem_after)
                    if (
                        best_cost is None
                        or cost < best_cost
                        or (
                            cost == best_cost
                            and ms
                            < (
                                max(best_snap[0].task_end, best_snap[1].task_end)
                                if best_snap
                                else 0
                            )
                        )
                    ):
                        best_cost = cost
                        best_snap = (ta, tb)
                        best_rem = rem_after
                except Exception:
                    pass

            maxK = min(3, len(remaining) - 1)
            for K in range(1, maxK + 1):
                K_eid, K_ntok = remaining[K]
                rem_after = tuple(
                    r for r in remaining if r[0] != top0_eid and r[0] != K_eid
                )
                for ea, na, sw_a, dn_a, eb, nb, sw_b, dn_b in [
                    (
                        top0_eid,
                        top0_ntok,
                        c2_sw_0,
                        c2_dn_0,
                        K_eid,
                        K_ntok,
                        swiglu_hit(K_eid, c3, now),
                        down_hit(K_eid, c3, now),
                    ),
                    (
                        K_eid,
                        K_ntok,
                        swiglu_hit(K_eid, c2, now),
                        down_hit(K_eid, c2, now),
                        top0_eid,
                        top0_ntok,
                        c3_sw_0,
                        c3_dn_0,
                    ),
                ]:
                    for s1a, s3a, s1b, s3b in _pick_pair_shapes(
                        na, nb, sw_a, dn_a, sw_b, dn_b, now
                    ):
                        sna = FourStageSnap.from_assign(
                            now, s1a, s3a, na, ea, sw_a, dn_a
                        )
                        snb = FourStageSnap.from_assign(
                            now, s1b, s3b, nb, eb, sw_b, dn_b
                        )
                        _eval_pair(sna, s3a, snb, s3b, rem_after)

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
                            continue
                        for ea, na, sw_a, dn_a, eb, nb, sw_b, dn_b in [
                            (
                                K_eid,
                                K_ntok,
                                swiglu_hit(K_eid, c2, now),
                                down_hit(K_eid, c2, now),
                                J_eid,
                                J_ntok,
                                swiglu_hit(J_eid, c3, now),
                                down_hit(J_eid, c3, now),
                            ),
                            (
                                J_eid,
                                J_ntok,
                                swiglu_hit(J_eid, c2, now),
                                down_hit(J_eid, c2, now),
                                K_eid,
                                K_ntok,
                                swiglu_hit(K_eid, c3, now),
                                down_hit(K_eid, c3, now),
                            ),
                        ]:
                            for s1a, s3a, s1b, s3b in _pick_pair_shapes(
                                na, nb, sw_a, dn_a, sw_b, dn_b, now
                            ):
                                sna = FourStageSnap.from_assign(
                                    now, s1a, s3a, na, ea, sw_a, dn_a
                                )
                                snb = FourStageSnap.from_assign(
                                    now, s1b, s3b, nb, eb, sw_b, dn_b
                                )
                                _eval_pair(sna, s3a, snb, s3b, rem_after)

            # SPLIT(top0)
            if top0_ntok >= 2:
                s_cuts = {math.ceil(top0_ntok / 2), top0_ntok // 2}
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
                    for s1a, s3a, s1b, s3b in _pick_pair_shapes(
                        cut_A, cut_B, c2_sw_0, c2_dn_0, c3_sw_0, c3_dn_0, now
                    ):
                        sna = FourStageSnap.from_assign(
                            now, s1a, s3a, cut_A, top0_eid, c2_sw_0, c2_dn_0
                        )
                        snb = FourStageSnap.from_assign(
                            now, s1b, s3b, cut_B, top0_eid, c3_sw_0, c3_dn_0
                        )
                        _eval_pair(sna, s3a, snb, s3b, rem_after)

            if best_snap is not None:
                c2, c3 = best_snap
                remaining = best_rem
            else:
                s1 = best_solo_shape_s1(top0_ntok)
                s3 = best_solo_shape_s3(top0_ntok)
                c2 = FourStageSnap.from_assign(
                    now, s1, s3, top0_ntok, top0_eid, c2_sw_0, c2_dn_0
                )
                remaining = remaining[1:]

        else:
            # ── not_both_idle ── 【简化】3 时间点 × ShapeC ────────────────────
            if t2 < t3:
                idle_t, idle_cluster = t2, "C2"
                busy_snap = c3
                idle_snap = c2
            else:
                idle_t, idle_cluster = t3, "C3"
                busy_snap = c2
                idle_snap = c3

            hi_pts = _dma_hi_pts(busy_snap)
            try_starts = sorted({idle_t} | {t for t in hi_pts if t > idle_t})[:3]

            best_single_cost = None
            best_single_snap = None

            for t_start in try_starts:
                hit = swiglu_hit(top0_eid, idle_snap, t_start)
                full = down_hit(top0_eid, idle_snap, t_start)
                try:
                    new_sn = FourStageSnap.from_assign(
                        t_start, SHAPE_C, SHAPE_C, top0_ntok, top0_eid, hit, full
                    )
                    new_sn = with_optional_s2_down_prefetch(new_sn, SHAPE_C, busy_snap)
                    if idle_cluster == "C2":
                        if not bw_feasible(new_sn, busy_snap):
                            continue
                    else:
                        if not bw_feasible(busy_snap, new_sn):
                            continue
                    cost = max(new_sn.task_end, busy_snap.task_end)
                    if best_single_cost is None or cost < best_single_cost:
                        best_single_cost = cost
                        best_single_snap = new_sn
                except Exception:
                    pass

            if best_single_snap is not None:
                if idle_cluster == "C2":
                    c2 = best_single_snap
                else:
                    c3 = best_single_snap
            else:
                s1 = best_solo_shape_s1(top0_ntok)
                s3 = best_solo_shape_s3(top0_ntok)
                if idle_cluster == "C2":
                    c2 = FourStageSnap.from_assign(
                        idle_t,
                        s1,
                        s3,
                        top0_ntok,
                        top0_eid,
                        swiglu_hit(top0_eid, c2, idle_t),
                        down_hit(top0_eid, c2, idle_t),
                    )
                else:
                    c3 = FourStageSnap.from_assign(
                        idle_t,
                        s1,
                        s3,
                        top0_ntok,
                        top0_eid,
                        swiglu_hit(top0_eid, c3, idle_t),
                        down_hit(top0_eid, c3, idle_t),
                    )
            remaining = remaining[1:]

    return max(c2.task_end, c3.task_end)


# ─────────────────────────────────────────────────────────────────────────────
# 10K 对比测试
# ─────────────────────────────────────────────────────────────────────────────


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


if __name__ == "__main__":
    # analytical_schedule 很慢 (~300ms/case)，质量对比只用 500 案例
    N = 500
    ratios_lite, ratios_fast = [], []
    times_a, times_l, times_f = [], [], []
    crashes_a = crashes_l = crashes_f = 0
    t_wall0 = time.perf_counter()

    print(f"[质量对比] 运行 {N} 组，对比 analytical / lite / fast_lite……", flush=True)

    for i in range(N * 3):
        if len(ratios_fast) >= N:
            break
        rng_seed = random.Random(i * 997 + 13)
        dist = random_dist(seed=i)
        keys = list(dist.keys())
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

        try:
            t0 = time.perf_counter()
            f = fast_lite_schedule(dist, c2, c3)
            times_f.append(time.perf_counter() - t0)
        except Exception:
            crashes_f += 1
            continue

        if a > 0:
            ratios_lite.append((l / a, dist, c2, c3, a, l, f))
            ratios_fast.append((f / a, dist, c2, c3, a, l, f))

        if len(ratios_fast) % 1000 == 0 and len(ratios_fast) > 0:
            elapsed = time.perf_counter() - t_wall0
            print(f"  {len(ratios_fast)}/{N} done ({elapsed:.0f}s)", flush=True)

    n = len(ratios_fast)
    elapsed_total = time.perf_counter() - t_wall0

    rl = [r[0] for r in ratios_lite]
    rf = [r[0] for r in ratios_fast]

    print(f"\n{'='*62}")
    print(f"有效对比:                    {n}")
    print(f"crashes (anal/lite/fast):   {crashes_a}/{crashes_l}/{crashes_f}")
    print()
    print(f"{'指标':<28} {'lite':>10} {'fast_lite':>10}")
    print(f"{'-'*50}")
    print(f"{'均值 ratio (vs analytical)':<28} {sum(rl)/n:>10.4f} {sum(rf)/n:>10.4f}")
    print(f"{'最大 ratio':<28} {max(rl):>10.4f} {max(rf):>10.4f}")
    print(
        f"{'完全相同 (≤1.001)':<28} {sum(1 for r in rl if r<=1.001)/n*100:>9.1f}% {sum(1 for r in rf if r<=1.001)/n*100:>9.1f}%"
    )
    print(
        f"{'<1% 退化 (≤1.010)':<28} {sum(1 for r in rl if r<=1.010)/n*100:>9.1f}% {sum(1 for r in rf if r<=1.010)/n*100:>9.1f}%"
    )
    print(
        f"{'<2% 退化 (≤1.020)':<28} {sum(1 for r in rl if r<=1.020)/n*100:>9.1f}% {sum(1 for r in rf if r<=1.020)/n*100:>9.1f}%"
    )
    print(
        f"{'<5% 退化 (≤1.050)':<28} {sum(1 for r in rl if r<=1.050)/n*100:>9.1f}% {sum(1 for r in rf if r<=1.050)/n*100:>9.1f}%"
    )
    print(
        f"{'<10% 退化 (≤1.100)':<28} {sum(1 for r in rl if r<=1.100)/n*100:>9.1f}% {sum(1 for r in rf if r<=1.100)/n*100:>9.1f}%"
    )
    print()
    print(f"{'analytical 均时 (ms)':<28} {sum(times_a)/len(times_a)*1000:>10.2f}")
    print(f"{'lite 均时 (ms)':<28} {sum(times_l)/len(times_l)*1000:>10.2f}")
    print(f"{'fast_lite 均时 (ms)':<28} {sum(times_f)/len(times_f)*1000:>10.2f}")
    print(f"{'speedup lite→fast':<28} {sum(times_l)/sum(times_f):>10.2f}x")
    print(f"{'总耗时 (s)':<28} {elapsed_total:>10.0f}")
    print(f"{'='*62}")

    worst = sorted(ratios_fast, key=lambda x: x[0], reverse=True)[:10]
    if worst and worst[0][0] > 1.001:
        print("\nfast_lite 差距最大的前 10 个案例:")
        for r, dist, c2i, c3i, a, l, f in worst:
            toks = sorted(dist.values(), reverse=True)
            print(
                f"  ratio={r:.3f}  toks={toks}  c2={c2i}  c3={c3i}  "
                f"anal={a}  lite={l}  fast={f}"
            )

    # ── 纯速度对比：lite vs fast_lite（2000 案例，不跑 analytical）──────────
    print(f"\n[速度对比] 运行 2000 组，lite vs fast_lite……", flush=True)
    N2 = 2000
    tl2, tf2 = [], []
    for i in range(N2):
        dist2 = random_dist(seed=i + 50000)
        keys2 = list(dist2.keys())
        c2_ = keys2[0] if len(keys2) > 0 and (i % 3 != 0) else -1
        c3_ = keys2[1] if len(keys2) > 1 and (i % 2 == 0) else -1
        t0 = time.perf_counter()
        lite_schedule(dist2, c2_, c3_)
        tl2.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        fast_lite_schedule(dist2, c2_, c3_)
        tf2.append(time.perf_counter() - t0)

    print(f"  lite      均时: {sum(tl2)/N2*1000:.3f} ms")
    print(f"  fast_lite 均时: {sum(tf2)/N2*1000:.3f} ms")
    print(f"  speedup lite→fast: {sum(tl2)/sum(tf2):.2f}x")
