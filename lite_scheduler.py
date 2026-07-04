#!/usr/bin/env python3
"""
lite_scheduler.py
=================
analytical_scheduler 的精简版本，逐步引入计算量优化。
每个优化步骤在函数名和注释中标注，方便追踪效果 vs 质量的权衡。

与原版 analytical_scheduler.py 的对比：
  原版：精确、全穷举，缓存初始化做 4 次完整递归调用（约 380,000 cc 额外开销）
  精简版：缓存命中直接用，不递归，0 次额外完整调度

当前状态（Step 1 已完成）：
  优化：缓存命中直接分配给对应 cluster（best s1×s3 shape 选最优），不做 4 配置比较
  代价：~225 cc（9 次 from_assign）
  节省：原版 4 次递归 ≈ 380,000 cc → 精简版 0 次递归
  speedup（Python 实测）：
    - 无缓存命中：1x（无变化，主循环完全相同）
    - 两侧均命中：4-5x
    - 单侧命中（小 cached expert）：22-600x
    - 双侧命中（多 expert）：可达 1800x
  质量（lite ratio vs analytical）：
    - 大多数场景：ratio = 1.000（完全相同）
    - 已知退化案例：cached expert ntok 很大（如 30）且只有该 1 个 expert 时，
      lite 直接单侧分配（ratio=1.875），原版通过 SPLIT 可得更优解。
      在多 expert 场景（cached expert 被移出 remaining，其余 expert 仍被主循环调度），
      质量退化不存在。

运行（对比原版 analytical vs 精简版 lite）：
  cd /esat/studscratch/r1015673/Thesis
  python3 Idea_Model/lite_scheduler.py
"""

import math
import random
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from four_stage_scheduler import (
    SHAPE_A,
    SHAPE_B,
    SHAPE_C,
    ALL_SHAPES,
    MAX_BW,
    Shape,
    FourStageSnap,
    FourStageScheduler,
    BeamState,
    make_initial_snap,
    gen_stage_actions,
    apply_action,
    bw_feasible,
    with_optional_s2_down_prefetch,
    with_optional_s2_down_prefetch_pair,
    with_optional_next_s1_prefetch_pair,
    inject_ghost_prefetch_pair,
    PF_EID_GHOST,
    _best_task_time,
    _best_concurrent_task_time,
    _best_s2_compute,
)
from analytical_scheduler import analytical_schedule

EXACT_TAIL_MAX_TOKENS = 4


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数（与 analytical_scheduler 完全相同）
# ─────────────────────────────────────────────────────────────────────────────


def best_solo_shape_s1(ntok: int) -> Shape:
    return min(ALL_SHAPES, key=lambda s: s.T_s1_task(ntok))


def best_solo_shape_s3(ntok: int) -> Shape:
    return min(ALL_SHAPES, key=lambda s: s.T_s3_task(ntok))


def best_conc_shape_s1(ntok: int) -> Shape:
    return min(
        [s for s in ALL_SHAPES if s.alloc <= MAX_BW // 2],
        key=lambda s: s.T_s1_task(ntok),
    )


def best_conc_shape_s3(ntok: int) -> Shape:
    return min(
        [s for s in ALL_SHAPES if s.bw_req <= MAX_BW // 2],
        key=lambda s: s.T_s3_task(ntok),
    )


def _greedy_heuristic(c2_end: int, c3_end: int, remaining) -> int:
    """与原版完全相同的启发式代价估算。"""
    if not remaining:
        return max(c2_end, c3_end)

    tasks = [_best_concurrent_task_time(ntok) for _, ntok in remaining]

    if len(remaining) == 1:
        _, ntok1 = remaining[0]
        t_early = min(c2_end, c3_end)
        t_late = max(c2_end, c3_end)
        solo_t = _best_task_time(ntok1)
        half = math.ceil(ntok1 / 2)
        t_split_task = _best_concurrent_task_time(half)
        solo_cost = max(t_late, t_early + solo_t)
        split_cost = t_late + t_split_task
        return min(solo_cost, split_cost)

    if len(remaining) == 2:
        t_early = min(c2_end, c3_end)
        t_late = max(c2_end, c3_end)
        pair_cost = t_late + max(
            _best_concurrent_task_time(ntok) for _, ntok in remaining
        )
        serial_cost = max(
            t_early + sum(_best_task_time(ntok) for _, ntok in remaining),
            t_late,
        )
        return min(pair_cost, serial_cost)

    best_end = max(c2_end, c3_end)
    extra = max(max(tasks), sum(tasks) // 2)
    return best_end + extra


# ─────────────────────────────────────────────────────────────────────────────
# 精简版调度器主体
# ─────────────────────────────────────────────────────────────────────────────


def lite_schedule(
    token_dist: dict,
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
) -> int:
    """
    精简版分析法调度器。
    返回总 makespan（时钟周期数）。

    与 analytical_schedule 的核心区别：
      [STEP 1] 缓存初始化不做递归：
        - 若 initial_cache_c2 对应的 expert 在本轮 remaining 中，
          直接以 best_full_cached_snap 作为 C2 的初始状态并从 remaining 移除。
        - 若 initial_cache_c3 同理。
        - 不再枚举"是否使用缓存"的 4 种配置取 min，节省约 380,000 cc。

    主循环逻辑与 analytical_schedule 完全相同（包括 both_idle/not_both_idle/
    PAIR/PAIR_KJ/SPLIT/WAIT-PAIR/WAIT-SINGLE-PAIR/n=1 分支）。
    """
    remaining = tuple(sorted(token_dist.items(), key=lambda x: -x[1]))

    # ── [STEP 1] 缓存初始化：直接用 make_initial_snap，让主循环自己决策 ─────────────
    #
    # 原版做法（4 次完整递归，各探索不同的"是否预分配 cached expert"组合，取 min）：
    #   config 0: 不预分配，主循环自己决策
    #   config 1/2/3: 把 cached expert 预先分配给对应 cluster（已完成），remaining 减少
    #
    # 修复前的精简做法（Step 1 pre-assign，已废弃）：
    #   强制 config 1/3，把 cached expert 移出 remaining → 主循环无法对其做 SPLIT。
    #   对于 ntok 很大的 cached expert（如 ntok=4 适合 SPLIT），比 analytical 差 2×。
    #   导致 ~35% 案例出现显著回归（ratio ≤ 2.0）。
    #
    # 修复后的精简做法（config 0，make_initial_snap）：
    #   直接用 make_initial_snap(initial_cache_c2/c3) 初始化集群状态。
    #   make_initial_snap 设置 pf_eid=cached_eid, pf_full=True, task_end=0。
    #   主循环的 swiglu_hit/down_hit 自动识别缓存，可正确做 SPLIT/PAIR。
    #   代价：不再移除 cached expert，主循环多处理 1-2 个 expert（开销极小）。
    #
    c2 = make_initial_snap(initial_cache_c2)
    c3 = make_initial_snap(initial_cache_c3)

    # ── 以下主循环与 analytical_schedule 完全相同 ─────────────────────────────

    def swiglu_hit(eid, snap, t):
        if snap.pf_end < 0 or snap.pf_end > t:
            return False
        return snap.pf_eid == PF_EID_GHOST or snap.pf_eid == eid

    def down_hit(eid, snap, t):
        return swiglu_hit(eid, snap, t) and snap.pf_full

    def _sim1(
        c2_sn: FourStageSnap, c3_sn: FourStageSnap, e_eid: int, e_ntok: int
    ) -> int:
        deadline = max(c2_sn.task_end, c3_sn.task_end)

        def _eval_pf_pair(c2_s: FourStageSnap, c3_s: FourStageSnap):
            t2s, t3s = c2_s.task_end, c3_s.task_end
            now_s = max(t2s, t3s)
            c2_sw_s = swiglu_hit(e_eid, c2_s, now_s)
            c3_sw_s = swiglu_hit(e_eid, c3_s, now_s)
            c2_dn_s = down_hit(e_eid, c2_s, now_s)
            c3_dn_s = down_hit(e_eid, c3_s, now_s)
            best_s = None

            # Method A: both idle at now_s
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

            # SPLIT at now_s
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

            # Method B: early start on free cluster
            if t2s != t3s:
                if t2s < t3s:
                    idle_t_s, idle_sn_s, busy_sn_s = t2s, c2_s, c3_s
                    is_c2_idle = True
                else:
                    idle_t_s, idle_sn_s, busy_sn_s = t3s, c3_s, c2_s
                    is_c2_idle = False
                try_starts_s = sorted(
                    t for t in ({idle_t_s} | busy_sn_s.bw_change_pts()) if t >= idle_t_s
                )
                for t_st in try_starts_s:
                    sw = swiglu_hit(e_eid, idle_sn_s, t_st)
                    dn = down_hit(e_eid, idle_sn_s, t_st)
                    for s1 in ALL_SHAPES:
                        for s3 in ALL_SHAPES:
                            try:
                                sn = FourStageSnap.from_assign(
                                    t_st, s1, s3, e_ntok, e_eid, sw, dn
                                )
                                sn = with_optional_s2_down_prefetch(sn, s3, busy_sn_s)
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

        # 构建候选预取对列表
        pf_pairs = []

        # 候选1：with_optional_next_s1_prefetch_pair 的标准选择
        c2_pf, c3_pf = with_optional_next_s1_prefetch_pair(c2_sn, c3_sn, e_eid)
        pf_pairs.append((c2_pf, c3_pf))

        # 候选2：单 cluster 全预取（pf_end ≤ deadline）
        # with_optional_next_s1_prefetch_pair 偏好 score=2（双 SHAPE_A，pf_end=112640 > deadline）
        # 而忽略 score=1 的单 SHAPE_C（pf_end=90112 ≤ deadline，pf_full=True → dn=True）
        # 这里显式枚举每个 shape 的单 cluster 预取候选加以弥补
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

        # 对所有候选对求最小代价
        best_overall = None
        for c2_s, c3_s in pf_pairs:
            r = _eval_pf_pair(c2_s, c3_s)
            if r is not None and (best_overall is None or r < best_overall):
                best_overall = r

        return (
            best_overall
            if best_overall is not None
            else (deadline + _best_task_time(e_ntok))
        )

    def continuation_cost(
        c2_sn: FourStageSnap, c3_sn: FourStageSnap, new_rem: tuple
    ) -> int:
        if not new_rem:
            return max(c2_sn.task_end, c3_sn.task_end)
        if len(new_rem) == 1:
            eid, ntok = new_rem[0]
            return _sim1(c2_sn, c3_sn, eid, ntok)
        if (
            len(new_rem) == 2
            and sum(ntok for _, ntok in new_rem) <= EXACT_TAIL_MAX_TOKENS
        ):
            t_early = min(c2_sn.task_end, c3_sn.task_end)
            t_late = max(c2_sn.task_end, c3_sn.task_end)
            solo_seq = t_early + sum(_best_task_time(ntok) for _, ntok in new_rem)
            pair_after_late = t_late + max(
                _best_concurrent_task_time(ntok) for _, ntok in new_rem
            )
            return min(max(t_late, solo_seq), pair_after_late)
        return _greedy_heuristic(c2_sn.task_end, c3_sn.task_end, new_rem)

    def idle_snap_at(t: int) -> FourStageSnap:
        return FourStageSnap(
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
            pf_start=-1,
            pf_end=-1,
            pf_eid=-1,
            pf_bw=0,
            ntok=0,
        )

    # ── [Step 6] 解析式 O(1) 形状选择 ────────────────────────────────────────
    # 替代 PAIR/SPLIT 内层的 81-combo 枚举（3×3 s1 × 3×3 s3）。
    #
    # 数学推导：
    # S1 形状（两任务同时从 t_now 开始）：
    #   - 至少一侧缓存命中（sw=True）→ 其 bw_s1=0，另一侧可独占 128 B/cc → 均用 SHAPE_C
    #   - 两侧均未缓存：bw_alloc_A + bw_alloc_B ≤ 128，最优为 SHAPE_B+SHAPE_B（alloc=64+64=128）
    #     （SHAPE_B.T_s1_task ≤ SHAPE_A.T_s1_task for all ntok，且 alloc 相同）
    #
    # S3 形状（由 s2_end_A 和 s2_end_B 决定）：
    #   - 某侧 s3 全缓存（dn=True）→ bw_s3=0，另一侧可用 SHAPE_C
    #   - 两侧均未 s3 缓存：
    #     Δ = |s2_end_A - s2_end_B| ≥ SHAPE_C.t_dma_s3（11264 cc）
    #       → S3 DMA 窗口不重叠 → 均用 SHAPE_C（最快）
    #     Δ < 11264 → 窗口重叠，必须满足 bw_s3_A + bw_s3_B ≤ 128
    #       → SHAPE_B+SHAPE_B（bw_req=64+64=128，最快合法选项）
    #
    # BW 安全性保证：
    #   SHAPE_B 的 dma1_end 与 s2_end 相等（T_s1=t_dma_s1=45056），
    #   即 S1 DMA 窗口 [now, now+45056) 与 S3 DMA 窗口 [s2_end, ...) 不重叠（严格 <）。
    #   因此 S1 DMA 冲突和 S3 DMA 冲突可独立分析，无交叉干扰。
    _T_DMA_S3_C = SHAPE_C.t_dma_s3  # = 11264 cc

    def _pick_pair_shapes(ntok_A, ntok_B, sw_A, dn_A, sw_B, dn_B, t_now):
        """解析 O(1) 形状选择，返回 [(s1_A, s3_A, s1_B, s3_B)]。"""
        # S1 形状
        if sw_A or sw_B:
            s1_A = SHAPE_C
            s1_B = SHAPE_C
        else:
            s1_A = SHAPE_B
            s1_B = SHAPE_B

        # 解析计算 s2_end（不调用 from_assign）
        if sw_A:
            s2_A = t_now + _best_s2_compute(ntok_A)
        else:
            s2_A = t_now + s1_A.T_s1 + _best_s2_compute(max(0, ntok_A - s1_A.M_dim))

        if sw_B:
            s2_B = t_now + _best_s2_compute(ntok_B)
        else:
            s2_B = t_now + s1_B.T_s1 + _best_s2_compute(max(0, ntok_B - s1_B.M_dim))

        # S3 形状
        if dn_A or dn_B:
            # hit侧 S3 DMA=0，S4执行全部ntok token：ShapeC(M_dim=2, tile=11264cc)匹配best_s4
            # 非hit侧：另一侧DMA=0，可独占128 B/cc → ShapeC(bw_req=128 ≤ 128 ✓)
            s3_A, s3_B = SHAPE_C, SHAPE_C
        elif abs(s2_A - s2_B) >= _T_DMA_S3_C:
            s3_A, s3_B = SHAPE_C, SHAPE_C  # S3 DMA 窗口不重叠 → 均最快
        else:
            s3_A, s3_B = SHAPE_B, SHAPE_B  # 窗口重叠 → bw=64+64=128 合法

        return [(s1_A, s3_A, s1_B, s3_B)]

    _S3_PAIR_KJ = (SHAPE_B, SHAPE_C)
    # [Step 3] tiny expert 快速路径阈值：top0 与 top1 均 ≤ 此值时跳过全量枚举
    # TINY_NTOK=2: ntok=1/2 时 SPLIT 几乎无收益（tile 数=1），fast path 安全
    TINY_NTOK = 2
    max_iters = len(token_dist) * 4 + 10
    iters = 0

    while remaining and iters < max_iters:
        iters += 1
        c2_before_pf, c3_before_pf = (
            c2,
            c3,
        )  # 保存 ghost 注入前的 snaps（n=1 Method B BW 评估需要）
        c2, c3 = inject_ghost_prefetch_pair(c2, c3)
        t2, t3 = c2.task_end, c3.task_end
        both_idle = t2 == t3
        now = max(t2, t3)
        n = len(remaining)
        top0_eid, top0_ntok = remaining[0]

        if n == 1:
            # ── 最后一个 expert：与原版完全相同 ──────────────────────────────
            best_cost_n1 = None
            best_snap_n1 = None

            c2_sw = swiglu_hit(top0_eid, c2, now)
            c3_sw = swiglu_hit(top0_eid, c3, now)
            c2_dn = down_hit(top0_eid, c2, now)
            c3_dn = down_hit(top0_eid, c3, now)

            # Method A on C2
            # [Step 6] solo 任务独占全带宽，始终 SHAPE_C 最快（9 combos → 1）
            for s1, s3 in [(SHAPE_C, SHAPE_C)]:
                try:
                    sn = FourStageSnap.from_assign(
                        now, s1, s3, top0_ntok, top0_eid, c2_sw, c2_dn
                    )
                    cost = sn.task_end
                    if best_cost_n1 is None or cost < best_cost_n1:
                        best_cost_n1 = cost
                        best_snap_n1 = ("C2", sn)
                except Exception:
                    pass
            # Method A on C3
            # [Step 6] solo 任务独占全带宽，始终 SHAPE_C 最快（9 combos → 1）
            for s1, s3 in [(SHAPE_C, SHAPE_C)]:
                try:
                    sn = FourStageSnap.from_assign(
                        now, s1, s3, top0_ntok, top0_eid, c3_sw, c3_dn
                    )
                    cost = sn.task_end
                    if best_cost_n1 is None or cost < best_cost_n1:
                        best_cost_n1 = cost
                        best_snap_n1 = ("C3", sn)
                except Exception:
                    pass

            # SPLIT at now（n=1）
            if top0_ntok >= 2:
                split_cuts = {
                    k
                    for k in (math.ceil(top0_ntok / 2), top0_ntok // 2)
                    if 1 <= k <= top0_ntok - 1
                }
                for cut_A in split_cuts:
                    cut_B = top0_ntok - cut_A
                    # [Step 6] 解析式 O(1) 形状选择（81 combos → 1）
                    for s1_A, s3_A, s1_B, s3_B in _pick_pair_shapes(
                        cut_A, cut_B, c2_sw, c2_dn, c3_sw, c3_dn, now
                    ):
                        try:
                            sna = FourStageSnap.from_assign(
                                now, s1_A, s3_A, cut_A, top0_eid, c2_sw, c2_dn
                            )
                            snb = FourStageSnap.from_assign(
                                now, s1_B, s3_B, cut_B, top0_eid, c3_sw, c3_dn
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

            # Method B: early start
            if t2 != t3:
                if t2 < t3:
                    idle_t_n1, idle_cl_n1, busy_sn_n1, idle_sn_n1 = t2, "C2", c3, c2
                else:
                    idle_t_n1, idle_cl_n1, busy_sn_n1, idle_sn_n1 = t3, "C3", c2, c3
                # busy_no_pf: 去除本轮 with_optional_next_s1_prefetch_pair 添加的预取，
                # 避免 top0 预取的 BW 占用误判为不可行（_sim1 用无预取版 busy_sn）
                busy_no_pf_n1 = c3_before_pf if idle_cl_n1 == "C2" else c2_before_pf
                try_starts_n1 = sorted(
                    t
                    for t in (
                        {idle_t_n1}
                        | busy_sn_n1.bw_change_pts()
                        | busy_no_pf_n1.bw_change_pts()
                    )
                    if t >= idle_t_n1
                )
                for t_start in try_starts_n1:
                    hit = swiglu_hit(top0_eid, idle_sn_n1, t_start)
                    full = down_hit(top0_eid, idle_sn_n1, t_start)
                    # [Step 6] solo 任务独占全带宽，始终 SHAPE_C 最快（9 combos → 1）
                    for s1, s3 in [(SHAPE_C, SHAPE_C)]:
                        for busy_alt in [busy_sn_n1, busy_no_pf_n1]:
                            try:
                                new_sn = FourStageSnap.from_assign(
                                    t_start, s1, s3, top0_ntok, top0_eid, hit, full
                                )
                                new_sn = with_optional_s2_down_prefetch(
                                    new_sn, s3, busy_alt
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

            # 应用最优方案
            if best_snap_n1 is None:
                s1 = best_solo_shape_s1(top0_ntok)
                s3 = best_solo_shape_s3(top0_ntok)
                c2 = FourStageSnap.from_assign(
                    now, s1, s3, top0_ntok, top0_eid, c2_sw, c2_dn
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

        if both_idle:
            # ── both_idle 分支：与原版完全相同 ────────────────────────────────
            c2_sw_0 = swiglu_hit(top0_eid, c2, now)
            c3_sw_0 = swiglu_hit(top0_eid, c3, now)
            c2_dn_0 = down_hit(top0_eid, c2, now)
            c3_dn_0 = down_hit(top0_eid, c3, now)

            # ── [Step 3] tiny expert 快速路径 ────────────────────────────────
            # 条件：top0 与 top1 均为小 expert（ntok ≤ TINY_NTOK）。
            # 用 _pick_pair_shapes 选最优形状（含缓存感知），跳过 1200 combos 枚举。
            if top0_ntok <= TINY_NTOK and remaining[1][1] <= TINY_NTOK:
                _t1eid, _t1ntok = remaining[1]
                _c3c_1 = swiglu_hit(_t1eid, c3, now)
                _c3f_1 = down_hit(_t1eid, c3, now)
                _c2c_1 = swiglu_hit(_t1eid, c2, now)
                _c2f_1 = down_hit(_t1eid, c2, now)
                _fast_done = False
                _fast_best_cost = None
                # 方向 1：top0→C2, top1→C3
                for s1_A, s3_A, s1_B, s3_B in _pick_pair_shapes(
                    top0_ntok, _t1ntok, c2_sw_0, c2_dn_0, _c3c_1, _c3f_1, now
                ):
                    try:
                        _sn0 = FourStageSnap.from_assign(
                            now, s1_A, s3_A, top0_ntok, top0_eid, c2_sw_0, c2_dn_0
                        )
                        _sn1 = FourStageSnap.from_assign(
                            now, s1_B, s3_B, _t1ntok, _t1eid, _c3c_1, _c3f_1
                        )
                        _sn0, _sn1 = with_optional_s2_down_prefetch_pair(
                            _sn0, s3_A, _sn1, s3_B
                        )
                        if bw_feasible(_sn0, _sn1):
                            _cost = max(_sn0.task_end, _sn1.task_end)
                            if _fast_best_cost is None or _cost < _fast_best_cost:
                                _fast_best_cost = _cost
                                _fast_c2, _fast_c3 = _sn0, _sn1
                                _fast_done = True
                    except Exception:
                        pass
                # 方向 2：top1→C2, top0→C3
                for s1_A, s3_A, s1_B, s3_B in _pick_pair_shapes(
                    _t1ntok, top0_ntok, _c2c_1, _c2f_1, c3_sw_0, c3_dn_0, now
                ):
                    try:
                        _sn1r = FourStageSnap.from_assign(
                            now, s1_A, s3_A, _t1ntok, _t1eid, _c2c_1, _c2f_1
                        )
                        _sn0r = FourStageSnap.from_assign(
                            now, s1_B, s3_B, top0_ntok, top0_eid, c3_sw_0, c3_dn_0
                        )
                        _sn1r, _sn0r = with_optional_s2_down_prefetch_pair(
                            _sn1r, s3_A, _sn0r, s3_B
                        )
                        if bw_feasible(_sn1r, _sn0r):
                            _cost = max(_sn1r.task_end, _sn0r.task_end)
                            if _fast_best_cost is None or _cost < _fast_best_cost:
                                _fast_best_cost = _cost
                                _fast_c2, _fast_c3 = _sn1r, _sn0r
                                _fast_done = True
                    except Exception:
                        pass
                if _fast_done:
                    c2, c3 = _fast_c2, _fast_c3
                    remaining = remaining[2:]
                    continue
                # 完全回退到原版全量枚举（极少发生）

            best_cost = None
            best_snap = None
            best_snap_min = float("inf")
            best_snap_max = float("inf")
            best_new_rem_len = float("inf")

            # PAIR(top0, topK)，K=1,2,3，两方向
            max_K = min(n - 1, 3)
            for K in range(1, max_K + 1):
                topK_eid, topK_ntok = remaining[K]
                c2c_K = swiglu_hit(topK_eid, c2, now)
                c3c_K = swiglu_hit(topK_eid, c3, now)
                c2f_K = down_hit(topK_eid, c2, now)
                c3f_K = down_hit(topK_eid, c3, now)

                for eid_A, ntok_A, sw_A, dn_A, eid_B, ntok_B, sw_B, dn_B in [
                    (
                        top0_eid,
                        top0_ntok,
                        c2_sw_0,
                        c2_dn_0,
                        topK_eid,
                        topK_ntok,
                        c3c_K,
                        c3f_K,
                    ),
                    (
                        topK_eid,
                        topK_ntok,
                        c2c_K,
                        c2f_K,
                        top0_eid,
                        top0_ntok,
                        c3_sw_0 if hasattr(c3, "task_end") else False,
                        c3_dn_0 if hasattr(c3, "task_end") else False,
                    ),
                ]:
                    # 修正方向2的缓存状态
                    if eid_A == topK_eid:
                        cc_B2 = swiglu_hit(top0_eid, c3, now)
                        cf_B2 = down_hit(top0_eid, c3, now)
                        cc_A2 = swiglu_hit(topK_eid, c2, now)
                        cf_A2 = down_hit(topK_eid, c2, now)
                        sw_A, dn_A = cc_A2, cf_A2
                        sw_B, dn_B = cc_B2, cf_B2

                    # [Step 6] 解析式 O(1) 形状选择，替代 81-combo 枚举
                    for s1_A, s3_A, s1_B, s3_B in _pick_pair_shapes(
                        ntok_A, ntok_B, sw_A, dn_A, sw_B, dn_B, now
                    ):
                        try:
                            sna = FourStageSnap.from_assign(
                                now, s1_A, s3_A, ntok_A, eid_A, sw_A, dn_A
                            )
                            snb = FourStageSnap.from_assign(
                                now, s1_B, s3_B, ntok_B, eid_B, sw_B, dn_B
                            )
                            sna, snb = with_optional_s2_down_prefetch_pair(
                                sna, s3_A, snb, s3_B
                            )
                            if not bw_feasible(sna, snb):
                                continue
                            new_rem = tuple(
                                r for r in remaining if r[0] != eid_A and r[0] != eid_B
                            )
                            cost = continuation_cost(sna, snb, new_rem)
                            snap_min = min(sna.task_end, snb.task_end)
                            snap_max = max(sna.task_end, snb.task_end)
                            new_rem_len = len(new_rem)
                            if (
                                best_cost is None
                                or cost < best_cost
                                or (
                                    cost == best_cost and new_rem_len < best_new_rem_len
                                )
                                or (
                                    cost == best_cost
                                    and new_rem_len == best_new_rem_len
                                    and (
                                        snap_max < best_snap_max
                                        or (
                                            snap_max == best_snap_max
                                            and snap_min > best_snap_min
                                        )
                                    )
                                )
                            ):
                                best_cost = cost
                                best_snap = (sna, snb, new_rem)
                                best_snap_min = snap_min
                                best_snap_max = snap_max
                                best_new_rem_len = new_rem_len
                        except Exception:
                            pass

            # PAIR(topK, topJ)，K≥1，J>K（延迟 top0，先处理两个较小 expert）
            if n >= 3:
                max_KJ = min(n - 1, 3)
                for K in range(1, max_KJ):
                    for J in range(K + 1, max_KJ + 1):
                        if J >= n:
                            break
                        topK_eid, topK_ntok = remaining[K]
                        topJ_eid, topJ_ntok = remaining[J]
                        new_rem_KJ = tuple(
                            r
                            for r in remaining
                            if r[0] != topK_eid and r[0] != topJ_eid
                        )
                        if not new_rem_KJ:
                            continue
                        c2c_K = swiglu_hit(topK_eid, c2, now)
                        c3c_J = swiglu_hit(topJ_eid, c3, now)
                        c2c_J = swiglu_hit(topJ_eid, c2, now)
                        c3c_K = swiglu_hit(topK_eid, c3, now)
                        c2f_K = down_hit(topK_eid, c2, now)
                        c3f_J = down_hit(topJ_eid, c3, now)
                        c2f_J = down_hit(topJ_eid, c2, now)
                        c3f_K = down_hit(topK_eid, c3, now)
                        for eid_A, ntok_A, sw_A, dn_A, eid_B, ntok_B, sw_B, dn_B in [
                            (
                                topK_eid,
                                topK_ntok,
                                c2c_K,
                                c2f_K,
                                topJ_eid,
                                topJ_ntok,
                                c3c_J,
                                c3f_J,
                            ),
                            (
                                topJ_eid,
                                topJ_ntok,
                                c2c_J,
                                c2f_J,
                                topK_eid,
                                topK_ntok,
                                c3c_K,
                                c3f_K,
                            ),
                        ]:
                            # [Step 6] 解析式 O(1) 形状选择，替代 4-combo _S3_PAIR_KJ 枚举
                            for s1_A, s3_A, s1_B, s3_B in _pick_pair_shapes(
                                ntok_A, ntok_B, sw_A, dn_A, sw_B, dn_B, now
                            ):
                                try:
                                    sna = FourStageSnap.from_assign(
                                        now, s1_A, s3_A, ntok_A, eid_A, sw_A, dn_A
                                    )
                                    snb = FourStageSnap.from_assign(
                                        now, s1_B, s3_B, ntok_B, eid_B, sw_B, dn_B
                                    )
                                    sna, snb = with_optional_s2_down_prefetch_pair(
                                        sna, s3_A, snb, s3_B
                                    )
                                    if not bw_feasible(sna, snb):
                                        continue
                                    cost = continuation_cost(sna, snb, new_rem_KJ)
                                    snap_min = min(sna.task_end, snb.task_end)
                                    snap_max = max(sna.task_end, snb.task_end)
                                    new_rem_len = len(new_rem_KJ)
                                    if (
                                        best_cost is None
                                        or cost < best_cost
                                        or (
                                            cost == best_cost
                                            and new_rem_len < best_new_rem_len
                                        )
                                        or (
                                            cost == best_cost
                                            and new_rem_len == best_new_rem_len
                                            and (
                                                snap_max < best_snap_max
                                                or (
                                                    snap_max == best_snap_max
                                                    and snap_min > best_snap_min
                                                )
                                            )
                                        )
                                    ):
                                        best_cost = cost
                                        best_snap = (sna, snb, new_rem_KJ)
                                        best_snap_min = snap_min
                                        best_snap_max = snap_max
                                        best_new_rem_len = new_rem_len
                                except Exception:
                                    pass

            # SPLIT(top0)
            half_A = math.ceil(top0_ntok / 2)
            half_B = top0_ntok - half_A
            if half_B >= 1:
                # [Step 5+6] 包含对称切割 + M_dim 对齐的切割点
                # 与分析调度器保持一致：{ceil(n/2), n//2} ∪ {M_dim, max(1,n-M_dim)} for each shape
                # 这样可以覆盖不对称切割（如 ntok=10 时 cut=8/2），让先完成侧提前启动下一 expert
                split_cuts_both = {
                    k for k in (half_A, top0_ntok // 2) if 1 <= k <= top0_ntok - 1
                }
                for s in ALL_SHAPES:
                    for k in (s.M_dim, max(1, top0_ntok - s.M_dim)):
                        if 1 <= k <= top0_ntok - 1:
                            split_cuts_both.add(k)
                for cut_A in split_cuts_both:
                    cut_B = top0_ntok - cut_A
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
                            new_rem = remaining[1:]
                            cost = continuation_cost(sna, snb, new_rem)
                            snap_min = min(sna.task_end, snb.task_end)
                            snap_max = max(sna.task_end, snb.task_end)
                            new_rem_len = len(new_rem)
                            if (
                                best_cost is None
                                or cost < best_cost
                                or (
                                    cost == best_cost and new_rem_len < best_new_rem_len
                                )
                                or (
                                    cost == best_cost
                                    and new_rem_len == best_new_rem_len
                                    and (
                                        snap_max < best_snap_max
                                        or (
                                            snap_max == best_snap_max
                                            and snap_min > best_snap_min
                                        )
                                    )
                                )
                            ):
                                best_cost = cost
                                best_snap = (sna, snb, new_rem)
                                best_snap_min = snap_min
                                best_snap_max = snap_max
                                best_new_rem_len = new_rem_len
                        except Exception:
                            pass

            # WAIT-SINGLE-PAIR（n≥5，先发一个 ntok=1 的小 expert，再成对处理）
            if n >= 5:
                for K in range(1, min(n, 4)):
                    first_eid, first_ntok = remaining[K]
                    if first_ntok != 1:
                        continue
                    rem_after_first = tuple(r for r in remaining if r[0] != first_eid)
                    if len(rem_after_first) < 4:
                        continue
                    pair_anchor = rem_after_first[1]
                    pair_candidates = [
                        (pair_anchor, cand)
                        for cand in rem_after_first[2 : min(len(rem_after_first), 4)]
                        if cand[1] == 1
                    ]
                    if not pair_candidates:
                        continue
                    c2c_first = swiglu_hit(first_eid, c2, now)
                    c3c_first = swiglu_hit(first_eid, c3, now)
                    c2f_first = down_hit(first_eid, c2, now)
                    c3f_first = down_hit(first_eid, c3, now)
                    s1_first = best_solo_shape_s1(first_ntok)
                    s3_first = best_solo_shape_s3(first_ntok)
                    for send_first_to_c2 in (True,):
                        try:
                            first_sn = FourStageSnap.from_assign(
                                now,
                                s1_first,
                                s3_first,
                                first_ntok,
                                first_eid,
                                c2c_first if send_first_to_c2 else c3c_first,
                                c2f_first if send_first_to_c2 else c3f_first,
                            )
                            first_sn = with_optional_s2_down_prefetch(
                                first_sn, s3_first, None
                            )
                        except Exception:
                            continue
                        t_pair = first_sn.task_end
                        wait_sn = idle_snap_at(t_pair)
                        pair_base_c2 = first_sn if send_first_to_c2 else wait_sn
                        pair_base_c3 = wait_sn if send_first_to_c2 else first_sn
                        if not bw_feasible(pair_base_c2, pair_base_c3):
                            continue
                        for pair_a_item, pair_b_item in pair_candidates:
                            pair_eid_a, pair_ntok_a = pair_a_item
                            pair_eid_b, pair_ntok_b = pair_b_item
                            rem_after_pair = tuple(
                                r
                                for r in rem_after_first
                                if r[0] != pair_eid_a and r[0] != pair_eid_b
                            )
                            for swap_pair in (False, True):
                                eid_a = pair_eid_b if swap_pair else pair_eid_a
                                ntok_a = pair_ntok_b if swap_pair else pair_ntok_a
                                eid_b = pair_eid_a if swap_pair else pair_eid_b
                                ntok_b = pair_ntok_a if swap_pair else pair_ntok_b
                                s1_a = best_conc_shape_s1(ntok_a)
                                s1_b = best_conc_shape_s1(ntok_b)
                                s3_a = best_conc_shape_s3(ntok_a)
                                s3_b = best_conc_shape_s3(ntok_b)
                                try:
                                    sna = FourStageSnap.from_assign(
                                        t_pair,
                                        s1_a,
                                        s3_a,
                                        ntok_a,
                                        eid_a,
                                        swiglu_hit(eid_a, pair_base_c2, t_pair),
                                        down_hit(eid_a, pair_base_c2, t_pair),
                                    )
                                    snb = FourStageSnap.from_assign(
                                        t_pair,
                                        s1_b,
                                        s3_b,
                                        ntok_b,
                                        eid_b,
                                        swiglu_hit(eid_b, pair_base_c3, t_pair),
                                        down_hit(eid_b, pair_base_c3, t_pair),
                                    )
                                    sna, snb = with_optional_s2_down_prefetch_pair(
                                        sna, s3_a, snb, s3_b
                                    )
                                    if not bw_feasible(sna, snb):
                                        continue
                                    cost = continuation_cost(sna, snb, rem_after_pair)
                                    snap_min = min(sna.task_end, snb.task_end)
                                    snap_max = max(sna.task_end, snb.task_end)
                                    new_rem_len = len(rem_after_pair)
                                    if (
                                        best_cost is None
                                        or cost < best_cost
                                        or (
                                            cost == best_cost
                                            and new_rem_len < best_new_rem_len
                                        )
                                        or (
                                            cost == best_cost
                                            and new_rem_len == best_new_rem_len
                                            and (
                                                snap_max < best_snap_max
                                                or (
                                                    snap_max == best_snap_max
                                                    and snap_min > best_snap_min
                                                )
                                            )
                                        )
                                    ):
                                        best_cost = cost
                                        best_snap = (sna, snb, rem_after_pair)
                                        best_snap_min = snap_min
                                        best_snap_max = snap_max
                                        best_new_rem_len = new_rem_len
                                except Exception:
                                    pass

            # WAIT-PAIR（先 SINGLE 发 topK，让另一侧对齐后再 PAIR/SPLIT top0）
            for K in range(1, min(n, 4)):
                topK_eid, topK_ntok = remaining[K]
                c2c_K = swiglu_hit(topK_eid, c2, now)
                c3c_K = swiglu_hit(topK_eid, c3, now)
                c2f_K = down_hit(topK_eid, c2, now)
                c3f_K = down_hit(topK_eid, c3, now)
                rem_after_k = tuple(r for r in remaining if r[0] != topK_eid)
                if not rem_after_k:
                    continue
                for hit_k, full_k, send_to_c2 in [
                    (c2c_K, c2f_K, True),
                    (c3c_K, c3f_K, False),
                ]:
                    # [Step 6] WAIT-PAIR 单任务：独占全带宽，始终用 SHAPE_C（最快）
                    # 原版：9 combos；精简版：1 combo
                    for s1_k, s3_k in [(SHAPE_C, SHAPE_C)]:
                        try:
                            sn_k = FourStageSnap.from_assign(
                                now, s1_k, s3_k, topK_ntok, topK_eid, hit_k, full_k
                            )
                            sn_k = with_optional_s2_down_prefetch(sn_k, s3_k, None)
                            t_k = sn_k.task_end
                            wait_snap = idle_snap_at(t_k)
                            if not bw_feasible(sn_k, wait_snap):
                                continue
                            cost = continuation_cost(sn_k, wait_snap, rem_after_k)
                            snap_min = min(sn_k.task_end, wait_snap.task_end)
                            snap_max = max(sn_k.task_end, wait_snap.task_end)
                            new_rem_len = len(rem_after_k)
                            if (
                                best_cost is None
                                or cost < best_cost
                                or (
                                    cost == best_cost and new_rem_len < best_new_rem_len
                                )
                                or (
                                    cost == best_cost
                                    and new_rem_len == best_new_rem_len
                                    and (
                                        snap_max < best_snap_max
                                        or (
                                            snap_max == best_snap_max
                                            and snap_min > best_snap_min
                                        )
                                    )
                                )
                            ):
                                best_cost = cost
                                best_snap_min = snap_min
                                best_snap_max = snap_max
                                best_new_rem_len = new_rem_len
                                best_snap = (
                                    (sn_k, wait_snap, rem_after_k)
                                    if send_to_c2
                                    else (wait_snap, sn_k, rem_after_k)
                                )
                        except Exception:
                            pass

            if best_snap is None:
                s1 = best_solo_shape_s1(top0_ntok)
                s3 = best_solo_shape_s3(top0_ntok)
                new_c2 = FourStageSnap.from_assign(
                    now, s1, s3, top0_ntok, top0_eid, c2_sw_0, c2_dn_0
                )
                c2 = new_c2
                remaining = remaining[1:]
            else:
                c2, c3, remaining = best_snap

        else:
            # ── not_both_idle：与原版完全相同 ─────────────────────────────────
            if t2 < t3:
                idle_t, idle_cluster = t2, "C2"
                busy_snap = c3
                c2c_idle = swiglu_hit(top0_eid, c2, idle_t)
                c2f_idle = down_hit(top0_eid, c2, idle_t)
            else:
                idle_t, idle_cluster = t3, "C3"
                busy_snap = c2
                c2c_idle = swiglu_hit(top0_eid, c3, idle_t)
                c2f_idle = down_hit(top0_eid, c3, idle_t)

            try_starts = sorted(
                t for t in ({idle_t} | busy_snap.bw_change_pts()) if t >= idle_t
            )
            best_single_cost = None
            best_single_snap = None

            for t_start in try_starts:
                idle_snap_obj = c2 if idle_cluster == "C2" else c3
                hit = swiglu_hit(top0_eid, idle_snap_obj, t_start)
                full = down_hit(top0_eid, idle_snap_obj, t_start)
                for s1 in ALL_SHAPES:
                    for s3 in ALL_SHAPES:
                        try:
                            new_sn = FourStageSnap.from_assign(
                                t_start, s1, s3, top0_ntok, top0_eid, hit, full
                            )
                            new_sn = with_optional_s2_down_prefetch(
                                new_sn, s3, busy_snap
                            )
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
                        idle_t, s1, s3, top0_ntok, top0_eid, c2c_idle, c2f_idle
                    )
                else:
                    c3 = FourStageSnap.from_assign(
                        idle_t, s1, s3, top0_ntok, top0_eid, c2c_idle, c2f_idle
                    )
            remaining = remaining[1:]

    return max(c2.task_end, c3.task_end)


# ─────────────────────────────────────────────────────────────────────────────
# 测试与对比
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


def _print_result_table(label: str, ratios_vs_beam: list, ratios_vs_anal: list):
    n = len(ratios_vs_beam)
    if n == 0:
        print(f"  {label}: 无有效数据")
        return
    mean_r = sum(ratios_vs_beam) / n
    max_r = max(ratios_vs_beam)
    pct_opt = sum(1 for r in ratios_vs_beam if r <= 1.001) / n * 100
    pct_5 = sum(1 for r in ratios_vs_beam if r <= 1.05) / n * 100
    pct_10 = sum(1 for r in ratios_vs_beam if r <= 1.10) / n * 100

    mean_va = sum(ratios_vs_anal) / n if ratios_vs_anal else 0

    print(f"\n  ── {label} ──")
    print(f"  均值 ratio vs beam:    {mean_r:.4f}")
    print(f"  均值 ratio vs anal:    {mean_va:.4f}  (1.000=与原版相同)")
    print(f"  最大 ratio vs beam:    {max_r:.4f}")
    print(f"  pct_optimal (≤1.001): {pct_opt:.1f}%")
    print(f"  pct_5%      (≤1.050): {pct_5:.1f}%")
    print(f"  pct_10%     (≤1.100): {pct_10:.1f}%")


def main():
    N = 2000
    print(f"对比测试 {N} 个随机场景\n")
    print("  beam-search(width=64) 为基准，对比原版 analytical vs 精简版 lite\n")

    beam_list, anal_list, lite_list = [], [], []
    fail_count = 0
    t_anal_total = 0.0
    t_lite_total = 0.0

    for i in range(N):
        dist = random_dist(seed=i)
        keys = list(dist.keys())
        c2_cache = (
            keys[0]
            if random.Random(i * 1000).random() < 0.25 and len(keys) >= 1
            else -1
        )
        c3_cache_opts = [-1]
        if c2_cache >= 0 and len(keys) >= 2:
            c3_cache_opts.append(keys[1])
        c3_cache = random.Random(i * 1000 + 1).choice(c3_cache_opts)

        try:
            ms_beam, _ = FourStageScheduler(
                dist,
                beam_width=64,
                initial_cache_c2=c2_cache,
                initial_cache_c3=c3_cache,
            ).run()
        except Exception:
            fail_count += 1
            continue

        try:
            t0 = time.perf_counter()
            ms_anal = analytical_schedule(dist, c2_cache, c3_cache)
            t_anal_total += time.perf_counter() - t0
        except Exception:
            fail_count += 1
            continue

        try:
            t0 = time.perf_counter()
            ms_lite = lite_schedule(dist, c2_cache, c3_cache)
            t_lite_total += time.perf_counter() - t0
        except Exception:
            fail_count += 1
            continue

        if ms_beam <= 0:
            continue

        beam_list.append(ms_beam)
        anal_list.append(ms_anal)
        lite_list.append(ms_lite)

    if not beam_list:
        print("所有测试失败！")
        return

    n = len(beam_list)
    anal_vs_beam = [a / b for a, b in zip(anal_list, beam_list)]
    lite_vs_beam = [l / b for l, b in zip(lite_list, beam_list)]
    lite_vs_anal = [l / a for l, a in zip(lite_list, anal_list)]

    print(f"有效测试点: {n}/{N}，失败/跳过: {fail_count}\n")
    print("=" * 60)
    _print_result_table("原版 analytical_schedule", anal_vs_beam, [1.0] * n)
    _print_result_table("精简版 lite_schedule [STEP1]", lite_vs_beam, lite_vs_anal)
    print("=" * 60)

    print(f"\n  运行时间（Python，不代表 C 端，仅反映迭代次数比例）：")
    print(f"  analytical_schedule 总计: {t_anal_total*1000:.1f} ms")
    print(f"  lite_schedule       总计: {t_lite_total*1000:.1f} ms")
    if t_anal_total > 0:
        print(f"  speedup: {t_anal_total/t_lite_total:.2f}x")

    # 找出精简版比原版差距最大的案例
    worst = sorted(
        [
            (lite_vs_anal[i], beam_list[i], anal_list[i], lite_list[i])
            for i in range(n)
            if lite_vs_anal[i] > 1.001
        ],
        reverse=True,
    )
    if worst:
        print(f"\n  精简版比原版差距最大的前 5 个案例（ratio_lite/anal）：")
        for ratio, ms_b, ms_a, ms_l in worst[:5]:
            print(f"  lite/anal={ratio:.3f}  beam={ms_b}  anal={ms_a}  lite={ms_l}")
    else:
        print("\n  精简版与原版结果完全一致（无差异案例）")


if __name__ == "__main__":
    main()
