#!/usr/bin/env python3
"""
analytical_scheduler.py
=======================
分析法调度器：不用搜索，直接从物理方程推导每步决策。

核心思想：
  每步决策只需要评估 3 种候选：PAIR(top0,top1) / SPLIT(top0) / SINGLE(top0)
  对每种候选，用封闭形式计算代价，选最优。

与 beam-search 的区别：
  - beam-search: 每步展开所有合法 action，保留 64 个最优状态
  - 分析法: 每步只考虑 3 种结构性候选，贪心选择 1 步代价最小的

运行：
  cd /esat/studscratch/r1015673/Thesis
  python3 Idea_Model/analytical_scheduler.py
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from four_stage_scheduler import (
    SHAPE_A,
    SHAPE_B,
    SHAPE_C,
    ALL_SHAPES,
    WEIGHT_BYTES_S1,
    WEIGHT_BYTES_S3,
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
)

EXACT_TAIL_MAX_TOKENS = 4

# ─────────────────────────────────────────────────────────────────────────────
# 1.  分析法：形状选择
# ─────────────────────────────────────────────────────────────────────────────


def best_solo_shape_s1(ntok: int) -> "Shape":
    """单 expert 独占带宽（128 B/cc）时，S1 最优 shape（最小化 S1+S2 时长）。"""
    return min(ALL_SHAPES, key=lambda s: s.T_s1_task(ntok))


def best_solo_shape_s3(ntok: int) -> "Shape":
    """单 expert 独占带宽时，S3 最优 shape（最小化 S3+S4 时长）。"""
    return min(ALL_SHAPES, key=lambda s: s.T_s3_task(ntok))


def best_conc_shape_s1(ntok: int) -> "Shape":
    """并发时（每侧 BW ≤ 64 B/cc，alloc ≤ 64），S1 最优 shape。
    合法 shapes: ShapeA(alloc=64) 和 ShapeB(alloc=64)。ShapeC(alloc=128) 超限。
    """
    return min(
        [s for s in ALL_SHAPES if s.alloc <= MAX_BW // 2],
        key=lambda s: s.T_s1_task(ntok),
    )


def best_conc_shape_s3(ntok: int) -> "Shape":
    """并发时 S3 最优 shape。
    S3 阶段是否冲突取决于时序，保守估计仍用 bw≤64 约束（ShapeA 或 ShapeB）。
    实际运行时 bw_feasible 会精确检验，此处只给决策建议。
    """
    return min(
        [s for s in ALL_SHAPES if s.bw_req <= MAX_BW // 2],
        key=lambda s: s.T_s3_task(ntok),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  分析法：单步动作评估
# ─────────────────────────────────────────────────────────────────────────────


def _try_action(c2: FourStageSnap, c3: FourStageSnap, action, remaining):
    """尝试 apply 一个 action，返回 (new_c2, new_c3, new_remaining) 或 None（BW 违规）。"""
    try:
        new_state = apply_action(
            BeamState(
                c2=c2, c3=c3, remaining=remaining, history=(), g_score=0, f_score=0
            ),
            action,
        )
        return new_state.c2, new_state.c3, new_state.remaining
    except Exception:
        return None


def _greedy_heuristic(c2_end: int, c3_end: int, remaining) -> int:
    """分析法代价估算：当前决策之后的 makespan 下界估计。

    关键修正（remaining=1 时精确化）：
      free_cluster 在 t_early = min(c2,c3) 时刻就可以开始下一个任务；
      总 makespan = max(t_late, t_early + t_task)，而不是 t_late + t_task。
    multi-remaining 用并行任务时间下界估计。
    """
    if not remaining:
        return max(c2_end, c3_end)

    # 每个剩余任务在单个 cluster（BW ≤ 64）上的最短时间（用于 multi-remaining 估计）
    tasks = [_best_concurrent_task_time(ntok) for _, ntok in remaining]

    if len(remaining) == 1:
        # 精确下界：用 _best_task_time（无BW约束，允许ShapeC solo 128B/cc）
        # 理由：busy 侧 S3 DMA 可能在 t_early 之后很快结束，free 侧随即能用满 BW。
        # 这是可接受下界（actual ≥ t_early + _best_task_time，因为任务需要物理时间）。
        _, ntok1 = remaining[0]
        t_early = min(c2_end, c3_end)
        t_late = max(c2_end, c3_end)
        solo_t = _best_task_time(ntok1)  # 无 BW 约束（free 侧独占后的最短时长）
        half = math.ceil(ntok1 / 2)
        t_split_task = _best_concurrent_task_time(half)  # SPLIT：两侧各跑一半
        solo_cost = max(t_late, t_early + solo_t)
        # SPLIT 需要两侧都空闲才能同时运行，因此必须等到 t_late 之后才能开始
        # 若 t_early < t_late，空闲侧无法独立做 SPLIT（另一侧还在运行）
        split_cost = t_late + t_split_task
        return min(solo_cost, split_cost)

    # n==2：同时考虑"等 t_late 后 PAIR"和"空闲侧串行执行"两种方案
    if len(remaining) == 2:
        t_early = min(c2_end, c3_end)
        t_late = max(c2_end, c3_end)
        # 方案1：等 t_late 后 PAIR（两侧同时空闲后并发）
        pair_cost = t_late + max(
            _best_concurrent_task_time(ntok) for _, ntok in remaining
        )
        # 方案2：空闲侧从 t_early 开始串行执行两个 expert（不等 t_late）
        # 理由：t_early 时空闲侧可立即开始，无需等 busy 侧结束
        serial_cost = max(
            t_early + sum(_best_task_time(ntok) for _, ntok in remaining),
            t_late,
        )
        return min(pair_cost, serial_cost)

    # multi-remaining（≥3）：用 max(c2,c3) 作为基线（两侧均完成才能同步调度）
    best_end = max(c2_end, c3_end)
    extra = max(max(tasks), sum(tasks) // 2)
    return best_end + extra


# ─────────────────────────────────────────────────────────────────────────────
# 3.  分析法：贪心调度器主体
# ─────────────────────────────────────────────────────────────────────────────


def analytical_schedule(
    token_dist: dict,
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
    *,
    _initial_remaining=None,
    _initial_c2=None,
    _initial_c3=None,
    _allow_initial_cache_choices: bool = True,
) -> int:
    """
    贪心分析法调度器。
    返回总 makespan（时钟周期数）。

    决策逻辑（每步）：
    ┌─ n=1  → SINGLE(top0)，S1/S3 选最优 solo shape
    ├─ n≥2，both_idle
    │   ├─ 评估 PAIR(top0, top1)：ShapeB for S1（并发 BW 约束），best for S3
    │   ├─ 评估 SPLIT(top0)：ShapeB for S1，half tokens
    │   └─ 选代价（next_step_time + lb_remaining）最小的
    └─ n≥2，not both_idle（一个 cluster 空闲）
        └─ 分配给空闲的 cluster，使用最优 solo shape
    """

    remaining = (
        _initial_remaining
        if _initial_remaining is not None
        else tuple(sorted(token_dist.items(), key=lambda x: -x[1]))
    )
    c2 = _initial_c2 if _initial_c2 is not None else make_initial_snap(initial_cache_c2)
    c3 = _initial_c3 if _initial_c3 is not None else make_initial_snap(initial_cache_c3)

    def swiglu_hit(eid, snap, t):
        if snap.pf_end < 0 or snap.pf_end > t:
            return False
        return snap.pf_eid == PF_EID_GHOST or snap.pf_eid == eid

    def down_hit(eid, snap, t):
        return swiglu_hit(eid, snap, t) and snap.pf_full

    def rem_without(rem, *eids):
        drop = set(e for e in eids if e >= 0)
        return tuple(r for r in rem if r[0] not in drop)

    def best_full_cached_snap(eid, ntok, start=0):
        best = None
        for s1 in ALL_SHAPES:
            for s3 in ALL_SHAPES:
                sn = FourStageSnap.from_assign(start, s1, s3, ntok, eid, True, True)
                if best is None or sn.task_end < best.task_end:
                    best = sn
        return best

    def _sim1(
        c2_sn: FourStageSnap, c3_sn: FourStageSnap, e_eid: int, e_ntok: int
    ) -> int:
        """精确模拟：在 c2_sn/c3_sn 基础上调度最后一个 expert，返回最优 makespan。
        等价于运行 n==1 分支（包括 method A both-idle 和 method B early-start）。
        """
        c2_sn, c3_sn = with_optional_next_s1_prefetch_pair(c2_sn, c3_sn, e_eid)
        t2s, t3s = c2_sn.task_end, c3_sn.task_end
        now_s = max(t2s, t3s)
        c2_sw_s = swiglu_hit(e_eid, c2_sn, now_s)
        c3_sw_s = swiglu_hit(e_eid, c3_sn, now_s)
        c2_dn_s = down_hit(e_eid, c2_sn, now_s)
        c3_dn_s = down_hit(e_eid, c3_sn, now_s)
        best_s = None

        # Method A: both idle at now_s
        for s1 in ALL_SHAPES:
            for s3 in ALL_SHAPES:
                for cc, cf in [(c2_sw_s, c2_dn_s), (c3_sw_s, c3_dn_s)]:
                    try:
                        sn = FourStageSnap.from_assign(
                            now_s, s1, s3, e_ntok, e_eid, cc, cf
                        )
                        cost = sn.task_end
                        if best_s is None or cost < best_s:
                            best_s = cost
                    except Exception:
                        pass
        # SPLIT at now_s (both clusters free simultaneously)
        # 只取均分切割点（ceil/floor），避免大 ntok 时枚举过多
        # 关键：两侧 s1/s3 独立枚举（缓存命中侧不消耗 BW，可用 ShapeC）
        # 关键：两侧 cache 状态都用真实值（c2_sw_s, c3_sw_s）
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
                                        now_s, s1_A, s3_A, cut_A, e_eid, c2_sw_s, c2_dn_s
                                    )
                                    snb = FourStageSnap.from_assign(
                                        now_s, s1_B, s3_B, cut_B, e_eid, c3_sw_s, c3_dn_s
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

        # Method B: early start on free cluster (if not both_idle)
        if t2s != t3s:
            if t2s < t3s:
                idle_t_s, idle_sn_s, busy_sn_s = t2s, c2_sn, c3_sn
                is_c2_idle = True
            else:
                idle_t_s, idle_sn_s, busy_sn_s = t3s, c3_sn, c2_sn
                is_c2_idle = False
            try_starts_s = sorted(
                t for t in ({idle_t_s} | busy_sn_s.bw_change_pts()) if t >= idle_t_s
            )
            for t_st in try_starts_s:
                cc = swiglu_hit(e_eid, idle_sn_s, t_st)
                cf = down_hit(e_eid, idle_sn_s, t_st)
                for s1 in ALL_SHAPES:
                    for s3 in ALL_SHAPES:
                        try:
                            sn = FourStageSnap.from_assign(
                                t_st, s1, s3, e_ntok, e_eid, cc, cf
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
        return (
            best_s if best_s is not None else (max(t2s, t3s) + _best_task_time(e_ntok))
        )

    def continuation_cost(
        c2_sn: FourStageSnap, c3_sn: FourStageSnap, new_rem: tuple
    ) -> int:
        if not new_rem:
            return max(c2_sn.task_end, c3_sn.task_end)
        if len(new_rem) == 1:
            return _sim1(c2_sn, c3_sn, new_rem[0][0], new_rem[0][1])
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

    def split_hot_tail_cost(
        c2_sn: FourStageSnap, c3_sn: FourStageSnap, new_rem: tuple
    ) -> int:
        if not new_rem:
            return max(c2_sn.task_end, c3_sn.task_end)
        hot_eid, hot_ntok = new_rem[0]
        tail_rem = new_rem[1:]
        if hot_ntok < 2 or len(tail_rem) > 2:
            return continuation_cost(c2_sn, c3_sn, new_rem)

        c2_hot, c3_hot = with_optional_next_s1_prefetch_pair(c2_sn, c3_sn, hot_eid)
        if c2_hot.task_end != c3_hot.task_end:
            return continuation_cost(c2_hot, c3_hot, new_rem)

        start = c2_hot.task_end
        c2c_hot = swiglu_hit(hot_eid, c2_hot, start)
        c3c_hot = swiglu_hit(hot_eid, c3_hot, start)
        c2f_hot = down_hit(hot_eid, c2_hot, start)
        c3f_hot = down_hit(hot_eid, c3_hot, start)
        split_cuts = {math.ceil(hot_ntok / 2), hot_ntok // 2}
        for shape in ALL_SHAPES:
            split_cuts.add(shape.M_dim)
            split_cuts.add(max(1, hot_ntok - shape.M_dim))
        split_cuts = {k for k in split_cuts if 1 <= k <= hot_ntok - 1}

        best = None
        for cut_c2 in split_cuts:
            cut_c3 = hot_ntok - cut_c2
            for s1_c2 in ALL_SHAPES:
                for s1_c3 in ALL_SHAPES:
                    for s3_c2 in ALL_SHAPES:
                        for s3_c3 in ALL_SHAPES:
                            try:
                                split_c2 = FourStageSnap.from_assign(
                                    start,
                                    s1_c2,
                                    s3_c2,
                                    cut_c2,
                                    hot_eid,
                                    c2c_hot,
                                    c2f_hot,
                                )
                                split_c3 = FourStageSnap.from_assign(
                                    start,
                                    s1_c3,
                                    s3_c3,
                                    cut_c3,
                                    hot_eid,
                                    c3c_hot,
                                    c3f_hot,
                                )
                                split_c2, split_c3 = (
                                    with_optional_s2_down_prefetch_pair(
                                        split_c2, s3_c2, split_c3, s3_c3
                                    )
                                )
                                if not bw_feasible(split_c2, split_c3):
                                    continue
                                cost = continuation_cost(split_c2, split_c3, tail_rem)
                                if best is None or cost < best:
                                    best = cost
                            except Exception:
                                pass
        return best if best is not None else continuation_cost(c2_hot, c3_hot, new_rem)

    if _allow_initial_cache_choices and remaining:
        by_eid = {eid: ntok for eid, ntok in remaining}
        c2_eid = initial_cache_c2 if initial_cache_c2 in by_eid else -1
        c3_eid = initial_cache_c3 if initial_cache_c3 in by_eid else -1

        if c2_eid >= 0 or c3_eid >= 0:
            best_ms = analytical_schedule(
                token_dist,
                initial_cache_c2,
                initial_cache_c3,
                _initial_remaining=remaining,
                _initial_c2=c2,
                _initial_c3=c3,
                _allow_initial_cache_choices=False,
            )

            if c2_eid >= 0:
                c2_first = best_full_cached_snap(c2_eid, by_eid[c2_eid])
                best_ms = min(
                    best_ms,
                    analytical_schedule(
                        token_dist,
                        initial_cache_c2,
                        initial_cache_c3,
                        _initial_remaining=rem_without(remaining, c2_eid),
                        _initial_c2=c2_first,
                        _initial_c3=c3,
                        _allow_initial_cache_choices=False,
                    ),
                )

            if c3_eid >= 0:
                c3_first = best_full_cached_snap(c3_eid, by_eid[c3_eid])
                best_ms = min(
                    best_ms,
                    analytical_schedule(
                        token_dist,
                        initial_cache_c2,
                        initial_cache_c3,
                        _initial_remaining=rem_without(remaining, c3_eid),
                        _initial_c2=c2,
                        _initial_c3=c3_first,
                        _allow_initial_cache_choices=False,
                    ),
                )

            if c2_eid >= 0 and c3_eid >= 0:
                if c2_eid != c3_eid:
                    c2_first = best_full_cached_snap(c2_eid, by_eid[c2_eid])
                    c3_first = best_full_cached_snap(c3_eid, by_eid[c3_eid])
                    best_ms = min(
                        best_ms,
                        analytical_schedule(
                            token_dist,
                            initial_cache_c2,
                            initial_cache_c3,
                            _initial_remaining=rem_without(remaining, c2_eid, c3_eid),
                            _initial_c2=c2_first,
                            _initial_c3=c3_first,
                            _allow_initial_cache_choices=False,
                        ),
                    )
                else:
                    ntok = by_eid[c2_eid]
                    if ntok >= 2:
                        cuts = {math.ceil(ntok / 2), ntok // 2}
                        for cut_c2 in cuts:
                            cut_c3 = ntok - cut_c2
                            if cut_c2 <= 0 or cut_c3 <= 0:
                                continue
                            for s1_c2 in ALL_SHAPES:
                                for s3_c2 in ALL_SHAPES:
                                    for s1_c3 in ALL_SHAPES:
                                        for s3_c3 in ALL_SHAPES:
                                            c2_first = FourStageSnap.from_assign(
                                                0,
                                                s1_c2,
                                                s3_c2,
                                                cut_c2,
                                                c2_eid,
                                                True,
                                                True,
                                            )
                                            c3_first = FourStageSnap.from_assign(
                                                0,
                                                s1_c3,
                                                s3_c3,
                                                cut_c3,
                                                c3_eid,
                                                True,
                                                True,
                                            )
                                            if not bw_feasible(c2_first, c3_first):
                                                continue
                                            best_ms = min(
                                                best_ms,
                                                analytical_schedule(
                                                    token_dist,
                                                    initial_cache_c2,
                                                    initial_cache_c3,
                                                    _initial_remaining=rem_without(
                                                        remaining, c2_eid
                                                    ),
                                                    _initial_c2=c2_first,
                                                    _initial_c3=c3_first,
                                                    _allow_initial_cache_choices=False,
                                                ),
                                            )
            return best_ms

    max_iters = len(token_dist) * 4 + 10  # 安全上限，防死循环
    iters = 0

    while remaining and iters < max_iters:
        iters += 1
        c2, c3 = inject_ghost_prefetch_pair(c2, c3)
        t2, t3 = c2.task_end, c3.task_end
        both_idle = t2 == t3
        now = max(t2, t3)
        n = len(remaining)
        top0_eid, top0_ntok = remaining[0]

        if n == 1:
            # ── 最后一个 expert：穷举所有可能启动方案，选 makespan 最小的 ──
            # 方案来源：
            #   A. 等待两侧都空闲（now=max）后 SINGLE 或 SPLIT
            #   B. 在空闲侧提前启动（not-both-idle style）：
            #      遍历 busy 侧所有 BW 变化点，尝试各 shape，bw_feasible 检验

            best_cost_n1 = None
            best_snap_n1 = None  # ('C2', sn) / ('C3', sn) / ('split', sna, snb)

            # ── 方案 A: 两侧都空闲后再启动（now = max(t2,t3)）──────────────
            c2_sw = swiglu_hit(top0_eid, c2, now)
            c3_sw = swiglu_hit(top0_eid, c3, now)
            c2_dn = down_hit(top0_eid, c2, now)
            c3_dn = down_hit(top0_eid, c3, now)

            for s1 in ALL_SHAPES:
                for s3 in ALL_SHAPES:
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
            for s1 in ALL_SHAPES:
                for s3 in ALL_SHAPES:
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

            # SPLIT at now（两侧同时空闲后）
            # 关键：两侧 s1/s3 独立枚举，cache 用真实 c2_sw/c3_sw
            if top0_ntok >= 2:
                split_cuts = set()
                split_cuts.add(math.ceil(top0_ntok / 2))
                split_cuts.add(top0_ntok // 2)
                for s in ALL_SHAPES:
                    split_cuts.add(s.M_dim)
                    split_cuts.add(max(1, top0_ntok - s.M_dim))
                split_cuts = {k for k in split_cuts if 1 <= k <= top0_ntok - 1}
                for cut_A in split_cuts:
                    cut_B = top0_ntok - cut_A
                    for s1_A in ALL_SHAPES:
                        for s1_B in ALL_SHAPES:
                            for s3_A in ALL_SHAPES:
                                for s3_B in ALL_SHAPES:
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

            # ── 方案 B: 空闲侧提前启动（not-both-idle style）──────────────
            if t2 != t3:
                if t2 < t3:
                    idle_t_n1, idle_cl_n1, busy_sn_n1, idle_sn_n1 = t2, "C2", c3, c2
                else:
                    idle_t_n1, idle_cl_n1, busy_sn_n1, idle_sn_n1 = t3, "C3", c2, c3
                try_starts_n1 = sorted(
                    t
                    for t in ({idle_t_n1} | busy_sn_n1.bw_change_pts())
                    if t >= idle_t_n1
                )
                for t_start in try_starts_n1:
                    hit = swiglu_hit(top0_eid, idle_sn_n1, t_start)
                    full = down_hit(top0_eid, idle_sn_n1, t_start)
                    for s1 in ALL_SHAPES:
                        for s3 in ALL_SHAPES:
                            try:
                                new_sn = FourStageSnap.from_assign(
                                    t_start, s1, s3, top0_ntok, top0_eid, hit, full
                                )
                                new_sn = with_optional_s2_down_prefetch(
                                    new_sn, s3, busy_sn_n1
                                )
                                if idle_cl_n1 == "C2":
                                    if not bw_feasible(new_sn, busy_sn_n1):
                                        continue
                                else:
                                    if not bw_feasible(busy_sn_n1, new_sn):
                                        continue
                                cost = max(new_sn.task_end, busy_sn_n1.task_end)
                                if best_cost_n1 is None or cost < best_cost_n1:
                                    best_cost_n1 = cost
                                    best_snap_n1 = (idle_cl_n1, new_sn)
                            except Exception:
                                pass

            # 应用最优方案
            if best_snap_n1 is None:
                # 极端 fallback
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
            c2_sw_0 = swiglu_hit(top0_eid, c2, now)
            c3_sw_0 = swiglu_hit(top0_eid, c3, now)
            c2_dn_0 = down_hit(top0_eid, c2, now)
            c3_dn_0 = down_hit(top0_eid, c3, now)

            best_cost = None
            best_snap = None  # (new_c2, new_c3, new_remaining)
            best_snap_min = float("inf")  # tiebreaker for main PAIR loop
            best_snap_max = float("inf")

            # ── 候选族 PAIR(top0, topK)：K ∈ {1, 2, 3}（热热 / 热冷）────────
            # 同时考虑两个方向 (top0→C2, topK→C3) 和 (topK→C2, top0→C3)
            max_K = min(n - 1, 3)
            for K in range(1, max_K + 1):
                topK_eid, topK_ntok = remaining[K]
                c2c_K = swiglu_hit(topK_eid, c2, now)
                c3c_K = swiglu_hit(topK_eid, c3, now)
                c2f_K = down_hit(topK_eid, c2, now)
                c3f_K = down_hit(topK_eid, c3, now)

                # 方向 1: top0→C2, topK→C3
                for s1_A in ALL_SHAPES:
                    if (not c2_sw_0) and (not c3c_K) and s1_A.alloc > MAX_BW // 2:
                        continue
                    for s1_B in ALL_SHAPES:
                        if (not c3c_K) and (not c2_sw_0) and s1_B.alloc > MAX_BW // 2:
                            continue
                        for s3_A in ALL_SHAPES:
                            for s3_B in ALL_SHAPES:
                                try:
                                    sna = FourStageSnap.from_assign(
                                        now,
                                        s1_A,
                                        s3_A,
                                        top0_ntok,
                                        top0_eid,
                                        c2_sw_0,
                                        c2_dn_0,
                                    )
                                    snb = FourStageSnap.from_assign(
                                        now,
                                        s1_B,
                                        s3_B,
                                        topK_ntok,
                                        topK_eid,
                                        c3c_K,
                                        c3f_K,
                                    )
                                    sna, snb = with_optional_s2_down_prefetch_pair(
                                        sna, s3_A, snb, s3_B
                                    )
                                    if not bw_feasible(sna, snb):
                                        continue
                                    new_rem = tuple(
                                        r
                                        for r in remaining
                                        if r[0] != top0_eid and r[0] != topK_eid
                                    )
                                    cost = continuation_cost(sna, snb, new_rem)
                                    snap_min = min(sna.task_end, snb.task_end)
                                    snap_max = max(sna.task_end, snb.task_end)
                                    if (
                                        best_cost is None
                                        or cost < best_cost
                                        or (
                                            cost == best_cost
                                            and (
                                                snap_max < best_snap_max
                                                or (
                                                    snap_max == best_snap_max
                                                    and snap_min < best_snap_min
                                                )
                                            )
                                        )
                                    ):
                                        best_cost = cost
                                        best_snap = (sna, snb, new_rem)
                                        best_snap_min = snap_min
                                        best_snap_max = snap_max
                                except Exception:
                                    pass

                # 方向 2: topK→C2, top0→C3
                c2c_K2 = swiglu_hit(topK_eid, c2, now)
                c3c_02 = swiglu_hit(top0_eid, c3, now)
                c2f_K2 = down_hit(topK_eid, c2, now)
                c3f_02 = down_hit(top0_eid, c3, now)
                for s1_A in ALL_SHAPES:
                    if (not c2c_K2) and (not c3c_02) and s1_A.alloc > MAX_BW // 2:
                        continue
                    for s1_B in ALL_SHAPES:
                        if (not c3c_02) and (not c2c_K2) and s1_B.alloc > MAX_BW // 2:
                            continue
                        for s3_A in ALL_SHAPES:
                            for s3_B in ALL_SHAPES:
                                try:
                                    sna = FourStageSnap.from_assign(
                                        now,
                                        s1_A,
                                        s3_A,
                                        topK_ntok,
                                        topK_eid,
                                        c2c_K2,
                                        c2f_K2,
                                    )
                                    snb = FourStageSnap.from_assign(
                                        now,
                                        s1_B,
                                        s3_B,
                                        top0_ntok,
                                        top0_eid,
                                        c3c_02,
                                        c3f_02,
                                    )
                                    sna, snb = with_optional_s2_down_prefetch_pair(
                                        sna, s3_A, snb, s3_B
                                    )
                                    if not bw_feasible(sna, snb):
                                        continue
                                    new_rem = tuple(
                                        r
                                        for r in remaining
                                        if r[0] != top0_eid and r[0] != topK_eid
                                    )
                                    cost = continuation_cost(sna, snb, new_rem)
                                    snap_min = min(sna.task_end, snb.task_end)
                                    snap_max = max(sna.task_end, snb.task_end)
                                    if (
                                        best_cost is None
                                        or cost < best_cost
                                        or (
                                            cost == best_cost
                                            and (
                                                snap_max < best_snap_max
                                                or (
                                                    snap_max == best_snap_max
                                                    and snap_min < best_snap_min
                                                )
                                            )
                                        )
                                    ):
                                        best_cost = cost
                                        best_snap = (sna, snb, new_rem)
                                        best_snap_min = snap_min
                                        best_snap_max = snap_max
                                except Exception:
                                    pass

            # ── NEW: PAIR(topK, topJ) with K≥1, J>K（延迟 top0，先处理两个小 expert）──
            # 例如 [7,2,2]→先PAIR(top1,top2)，然后_sim1对top0做精确SPLIT，得到更优解
            # 注意：S3 只枚举 [SHAPE_B, SHAPE_C]（大多数情况最优），max_KJ<=3 控制开销
            _S3_PAIR_KJ = (SHAPE_B, SHAPE_C)
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
                            for s1_A in ALL_SHAPES:
                                if (
                                    (not sw_A)
                                    and (not sw_B)
                                    and s1_A.alloc > MAX_BW // 2
                                ):
                                    continue
                                for s1_B in ALL_SHAPES:
                                    if (
                                        (not sw_B)
                                        and (not sw_A)
                                        and s1_B.alloc > MAX_BW // 2
                                    ):
                                        continue
                                    for s3_A in _S3_PAIR_KJ:
                                        for s3_B in _S3_PAIR_KJ:
                                            try:
                                                sna = FourStageSnap.from_assign(
                                                    now,
                                                    s1_A,
                                                    s3_A,
                                                    ntok_A,
                                                    eid_A,
                                                    sw_A,
                                                    dn_A,
                                                )
                                                snb = FourStageSnap.from_assign(
                                                    now,
                                                    s1_B,
                                                    s3_B,
                                                    ntok_B,
                                                    eid_B,
                                                    sw_B,
                                                    dn_B,
                                                )
                                                sna, snb = (
                                                    with_optional_s2_down_prefetch_pair(
                                                        sna, s3_A, snb, s3_B
                                                    )
                                                )
                                                if not bw_feasible(sna, snb):
                                                    continue
                                                cost = continuation_cost(
                                                    sna, snb, new_rem_KJ
                                                )
                                                snap_min = min(
                                                    sna.task_end, snb.task_end
                                                )
                                                snap_max = max(
                                                    sna.task_end, snb.task_end
                                                )
                                                if (
                                                    best_cost is None
                                                    or cost < best_cost
                                                    or (
                                                        cost == best_cost
                                                        and (
                                                            snap_max < best_snap_max
                                                            or (
                                                                snap_max
                                                                == best_snap_max
                                                                and snap_min
                                                                < best_snap_min
                                                            )
                                                        )
                                                    )
                                                ):
                                                    best_cost = cost
                                                    best_snap = (sna, snb, new_rem_KJ)
                                                    best_snap_min = snap_min
                                                    best_snap_max = snap_max
                                            except Exception:
                                                pass

            # ── 候选 SPLIT(top0)：穷举切割点给 C2 / C3 ─────────────────────────
            # 关键：两侧 s1/s3 独立枚举，cache 用真实 c2_sw_0/c3_sw_0
            half_A = math.ceil(top0_ntok / 2)
            half_B = top0_ntok - half_A
            if half_B >= 1:
                # 切割点集合
                split_cuts_both = {half_A, top0_ntok // 2}
                for s in ALL_SHAPES:
                    split_cuts_both.add(s.M_dim)
                    split_cuts_both.add(max(1, top0_ntok - s.M_dim))
                split_cuts_both = {
                    k for k in split_cuts_both if 1 <= k <= top0_ntok - 1
                }

                for cut_A in split_cuts_both:
                    cut_B = top0_ntok - cut_A
                    for s1_A in ALL_SHAPES:
                        for s1_B in ALL_SHAPES:
                            for s3_A in ALL_SHAPES:
                                for s3_B in ALL_SHAPES:
                                    try:
                                        sna = FourStageSnap.from_assign(
                                            now,
                                            s1_A,
                                            s3_A,
                                            cut_A,
                                            top0_eid,
                                            c2_sw_0,
                                            c2_dn_0,
                                        )
                                        snb = FourStageSnap.from_assign(
                                            now,
                                            s1_B,
                                            s3_B,
                                            cut_B,
                                            top0_eid,
                                            c3_sw_0,
                                            c3_dn_0,
                                        )
                                        sna, snb = with_optional_s2_down_prefetch_pair(
                                            sna, s3_A, snb, s3_B
                                        )
                                        if not bw_feasible(sna, snb):
                                            continue
                                        new_rem = remaining[1:]  # top0 消耗完
                                        cost = continuation_cost(sna, snb, new_rem)
                                        snap_min = min(sna.task_end, snb.task_end)
                                        snap_max = max(sna.task_end, snb.task_end)
                                        if (
                                            best_cost is None
                                            or cost < best_cost
                                            or (
                                                cost == best_cost
                                                and (
                                                    snap_max < best_snap_max
                                                    or (
                                                        snap_max == best_snap_max
                                                        and snap_min < best_snap_min
                                                    )
                                                )
                                            )
                                        ):
                                            best_cost = cost
                                            best_snap = (sna, snb, new_rem)
                                            best_snap_min = snap_min
                                            best_snap_max = snap_max
                                    except Exception:
                                        pass

            # ── 候选 WAIT-SINGLE-PAIR：先发一个冷 expert，再成对处理两个冷 expert ──
            # BW16 的坏例显示，hot expert 很大且尾部冷 expert 多时，先用一个短
            # SINGLE 对齐时间，再 PAIR 两个冷 expert，可以给后续 hot SPLIT 留出更好窗口。
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
                        if send_first_to_c2:
                            pair_base_c2, pair_base_c3 = first_sn, wait_sn
                        else:
                            pair_base_c2, pair_base_c3 = wait_sn, first_sn
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
                                if swap_pair:
                                    eid_a, ntok_a = pair_eid_b, pair_ntok_b
                                    eid_b, ntok_b = pair_eid_a, pair_ntok_a
                                else:
                                    eid_a, ntok_a = pair_eid_a, pair_ntok_a
                                    eid_b, ntok_b = pair_eid_b, pair_ntok_b

                                s1_a = best_conc_shape_s1(ntok_a)
                                s1_b = best_conc_shape_s1(ntok_b)
                                s3_options_a = (best_conc_shape_s3(ntok_a),)
                                s3_options_b = (best_conc_shape_s3(ntok_b),)
                                for s3_a in s3_options_a:
                                    for s3_b in s3_options_b:
                                        try:
                                            sna = FourStageSnap.from_assign(
                                                t_pair,
                                                s1_a,
                                                s3_a,
                                                ntok_a,
                                                eid_a,
                                                swiglu_hit(
                                                    eid_a,
                                                    pair_base_c2,
                                                    t_pair,
                                                ),
                                                down_hit(
                                                    eid_a,
                                                    pair_base_c2,
                                                    t_pair,
                                                ),
                                            )
                                            snb = FourStageSnap.from_assign(
                                                t_pair,
                                                s1_b,
                                                s3_b,
                                                ntok_b,
                                                eid_b,
                                                swiglu_hit(
                                                    eid_b,
                                                    pair_base_c3,
                                                    t_pair,
                                                ),
                                                down_hit(
                                                    eid_b,
                                                    pair_base_c3,
                                                    t_pair,
                                                ),
                                            )
                                            sna, snb = (
                                                with_optional_s2_down_prefetch_pair(
                                                    sna, s3_a, snb, s3_b
                                                )
                                            )
                                            if not bw_feasible(sna, snb):
                                                continue
                                            if (
                                                len(rem_after_pair) <= 3
                                                and sum(
                                                    ntok
                                                    for _, ntok in rem_after_pair[1:]
                                                )
                                                <= EXACT_TAIL_MAX_TOKENS
                                            ):
                                                cost = split_hot_tail_cost(
                                                    sna, snb, rem_after_pair
                                                )
                                            else:
                                                cost = continuation_cost(
                                                    sna, snb, rem_after_pair
                                                )
                                            snap_min = min(sna.task_end, snb.task_end)
                                            snap_max = max(sna.task_end, snb.task_end)
                                            if (
                                                best_cost is None
                                                or cost < best_cost
                                                or (
                                                    cost == best_cost
                                                    and (
                                                        snap_max < best_snap_max
                                                        or (
                                                            snap_max == best_snap_max
                                                            and snap_min < best_snap_min
                                                        )
                                                    )
                                                )
                                            ):
                                                best_cost = cost
                                                best_snap = (
                                                    sna,
                                                    snb,
                                                    rem_after_pair,
                                                )
                                                best_snap_min = snap_min
                                                best_snap_max = snap_max
                                        except Exception:
                                            pass

            # ── 候选 WAIT-PAIR：先 SINGLE 发 topK（较小），让 top0 等到 topK 结束再 PAIR ──
            # 物理意义：toks=[9,2] → 先用 ShapeC 把 E2(2tok) 发到 C2（33792 cc），
            # 然后两侧同时开始 SPLIT(E1,4|5)——总 makespan 比贪心 SPLIT 低得多。
            # 实现：SINGLE(topK)→C2，构造 waiting_snap(t_k) 作为 C3（等待侧），
            # 下一步迭代会 both_idle at t_k，重新做 PAIR/SPLIT
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
                    for s1_k in ALL_SHAPES:
                        for s3_k in ALL_SHAPES:
                            try:
                                sn_k = FourStageSnap.from_assign(
                                    now, s1_k, s3_k, topK_ntok, topK_eid, hit_k, full_k
                                )
                                sn_k = with_optional_s2_down_prefetch(sn_k, s3_k, None)
                                t_k = sn_k.task_end
                                # 等待侧：构造 task_end=t_k 的 idle snap（BW=0）
                                wait_snap = idle_snap_at(t_k)
                                if not bw_feasible(sn_k, wait_snap):
                                    continue
                                cost = continuation_cost(sn_k, wait_snap, rem_after_k)
                                snap_min = min(sn_k.task_end, wait_snap.task_end)
                                snap_max = max(sn_k.task_end, wait_snap.task_end)
                                if (
                                    best_cost is None
                                    or cost < best_cost
                                    or (
                                        cost == best_cost
                                        and (
                                            snap_max < best_snap_max
                                            or (
                                                snap_max == best_snap_max
                                                and snap_min < best_snap_min
                                            )
                                        )
                                    )
                                ):
                                    best_cost = cost
                                    best_snap_min = snap_min
                                    best_snap_max = snap_max
                                    if send_to_c2:
                                        best_snap = (sn_k, wait_snap, rem_after_k)
                                    else:
                                        best_snap = (wait_snap, sn_k, rem_after_k)
                            except Exception:
                                pass

            if best_snap is None:
                # fallback: 只给 C2 分配 top0
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
            # ── 一个 cluster 空闲，另一个忙 ──────────────────────────────────
            # 关键修复：如果 busy 侧正在 S1/S3 DMA（占用 128 B/cc），空闲侧不能立刻启动，
            # 需推迟到 busy 侧 DMA 结束后。遍历 busy 侧所有 BW 变化点作为候选启动时刻。
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

            # 候选启动时刻：自身空闲时刻 + busy 侧所有 DMA 结束时刻
            try_starts = sorted(
                t for t in ({idle_t} | busy_snap.bw_change_pts()) if t >= idle_t
            )

            best_single_cost = None
            best_single_snap = None

            for t_start in try_starts:
                idle_snap = c2 if idle_cluster == "C2" else c3
                hit = swiglu_hit(top0_eid, idle_snap, t_start)
                full = down_hit(top0_eid, idle_snap, t_start)
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
                # 极端 fallback
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
# 4.  对比测试
# ─────────────────────────────────────────────────────────────────────────────


def random_dist(seed=None):
    rng = random.Random(seed)
    n_experts = rng.choices([1, 2, 3, 4, 5, 6, 7, 8], weights=[1, 3, 5, 6, 5, 4, 3, 2])[
        0
    ]
    expert_ids = rng.sample(range(32), n_experts)
    # 幂律分布（MoE 常见）
    tokens = sorted(
        [max(1, int(rng.paretovariate(1.5) * 2)) for _ in range(n_experts)],
        reverse=True,
    )
    tokens = [min(t, 64) for t in tokens]
    return {eid: t for eid, t in zip(expert_ids, tokens)}


def main():
    N = 2000
    print(f"测试 {N} 个随机场景：分析法 vs beam-search（beam_width=64）\n")

    ratios = []
    fail_count = 0
    hist_bins = {
        (1.0, 1.0): 0,
        (1.0, 1.05): 0,
        (1.05, 1.1): 0,
        (1.1, 1.2): 0,
        (1.2, 1.5): 0,
        (1.5, 99.0): 0,
    }
    worst_cases = []

    for i in range(N):
        dist = random_dist(seed=i)
        # 随机初始 cache（25% C2 命中，12% C3 命中）
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

        # beam-search 最优
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

        # 分析法
        try:
            ms_anal = analytical_schedule(dist, c2_cache, c3_cache)
        except Exception as e:
            fail_count += 1
            continue

        if ms_beam <= 0:
            continue

        ratio = ms_anal / ms_beam
        ratios.append(ratio)

        for lo, hi in hist_bins:
            if lo <= ratio < hi:
                hist_bins[(lo, hi)] += 1
                break

        if ratio > 1.15:
            worst_cases.append((ratio, dist, ms_beam, ms_anal))

    if not ratios:
        print("所有测试失败！")
        return

    worst_cases.sort(reverse=True)

    n = len(ratios)
    mean_r = sum(ratios) / n
    pct_optimal = sum(1 for r in ratios if r <= 1.001) / n * 100
    pct_5pct = sum(1 for r in ratios if r <= 1.05) / n * 100
    pct_10pct = sum(1 for r in ratios if r <= 1.10) / n * 100
    max_r = max(ratios)

    print("=" * 55)
    print(f"  有效测试点:    {n} / {N}")
    print(f"  失败 / 跳过:   {fail_count}")
    print(f"  均值 ratio:    {mean_r:.4f}")
    print(f"  最大 ratio:    {max_r:.4f}")
    print(f"  pct_optimal:  {pct_optimal:.1f}%  (ratio ≤ 1.001)")
    print(f"  pct_5%:       {pct_5pct:.1f}%   (ratio ≤ 1.05)")
    print(f"  pct_10%:      {pct_10pct:.1f}%  (ratio ≤ 1.10)")
    print("=" * 55)
    print("\n分布直方图:")
    labels = {
        (1.0, 1.0): " =1.000 (最优)",
        (1.0, 1.05): " (1.000, 1.05]",
        (1.05, 1.1): " (1.05,  1.10]",
        (1.1, 1.2): " (1.10,  1.20]",
        (1.2, 1.5): " (1.20,  1.50]",
        (1.5, 99.0): " >1.50",
    }
    for key, cnt in hist_bins.items():
        bar = "█" * (cnt * 40 // max(hist_bins.values(), default=1))
        print(f"  {labels[key]:20s}: {cnt:5d}  {bar}")

    if worst_cases:
        print(f"\n最差 {min(5, len(worst_cases))} 个案例（ratio 最高）:")
        for ratio, dist, ms_beam, ms_anal in worst_cases[:5]:
            toks = sorted(dist.values(), reverse=True)
            print(f"  ratio={ratio:.3f}  toks={toks}  beam={ms_beam}  anal={ms_anal}")


if __name__ == "__main__":
    main()
