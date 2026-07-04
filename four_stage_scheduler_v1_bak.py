#!/usr/bin/env python3
"""
四阶段专家调度器 (Four-Stage Expert Scheduler)
=============================================================================
将每个 expert 的计算分解为四个严格串行阶段:

  Stage 1 (SwishGLU Fetch+Compute):
      DMA 搬运 W_swish → Region_SWISH, VersaCore 同时计算首 M_dim 行
      DMA 带宽: bw_req  时长: T_half = ceil(W/2 / bw_req)
  Stage 2 (SwishGLU Compute-Only):
      复用 Region_SWISH, 计算剩余 token 行  带宽: 0
      时长: T_half * (n_iters - 1)   如 n_iters=1 则直接跳过
  Stage 3 (Down Proj Fetch+Compute):
      前置依赖: Stage 2 必须完成 (RAW hazard)
      DMA 搬运 W_down → Region_DOWN, VersaCore 计算首 M_dim 行
      带宽: bw_req  时长: T_half
  Stage 4 (Down Proj Compute-Only):
      复用 Region_DOWN, 计算剩余行  带宽: 0
      时长: T_half * (n_iters - 1)
      内存释放: Stage 4 开始时 Region_SWISH 可被 Prefetch 下一专家

  Prefetch 规则 (可选优化):
      当 Cluster 处于 Stage 4 时, Region_SWISH 已空, DMA 带宽为 0,
      允许 prefetch 下一专家的 SwishGLU 权重 (写入 Region_SWISH).
      若 prefetch 在 Stage 4 结束前完成 → 下一专家 Stage 1 变为 cache-hit.
      硬约束: prefetch 绝对不能占用对方 cluster 正在使用的带宽.

物理约束:
  - 两条 DMA 通道 (xDMA, iDMA), 各 64 B/cc
  - 任意时刻: xDMA_bw_used + iDMA_bw_used ≤ 128 B/cc
  - Shape_A/B alloc=64, Shape_C alloc=128 (占用两条通道)
  - VersaCore 计算和 DMA 可以完全流水线重叠 (只要在同一 cluster 内)

状态空间:
  - 每个 cluster 用 FourStageCluster 快照追踪:
      (stage, stage_end, dma_end, cached_eid, prefetch_end, prefetch_eid)
  - Beam Search 在所有合法 (action, cluster) 对上展开

动作类型:
  1. ASSIGN(eid, ntok, shape, cluster): 分配新 expert 给空闲 cluster
     → 进入 Stage 1 (若 eid 已 cache 则 Stage 1 退化为 0 延迟)
  2. PREFETCH(eid, shape, cluster): 在 Stage 4 期间预取下一 expert
  3. PAIR: 两个 cluster 同时 ASSIGN (对齐优化)
  4. SPLIT: 将大 expert token 拆给两个 cluster 分别 ASSIGN
"""

import math
import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ============================================================
#  物理常量 & Shape 定义
# ============================================================

WEIGHT_BYTES_TOTAL = 3 * 2048 * 1408 // 2   # 每个 expert 总权重: 4,325,376 B
WEIGHT_BYTES_HALF  = WEIGHT_BYTES_TOTAL // 2  # 每个阶段权重: 2,162,688 B
MAX_BW = 128   # B/cc


@dataclass(frozen=True)
class Shape:
    name: str
    M_dim: int
    bw_req: int   # 每条 fetch 所需带宽 B/cc

    @property
    def T_half(self) -> int:
        """单阶段 fetch 时长 (= Stage 1 或 Stage 3 的时长)."""
        return math.ceil(WEIGHT_BYTES_HALF / self.bw_req)

    @property
    def alloc(self) -> int:
        """DMA 通道占用量 (硬件最小分配粒度 64 B/cc)."""
        return 64 if self.bw_req <= 64 else 128

    def n_iters(self, ntok: int) -> int:
        """计算该 ntok 需要几次完整的 M_dim 迭代."""
        return math.ceil(ntok / self.M_dim)

    def T_stage12(self, ntok: int, s1_cached: bool = False) -> int:
        """Stage 1+2 总时长 (SwishGLU)."""
        if s1_cached:
            # cache hit: Stage 1 = 0, Stage 2 = T_half * (n_iters - 1) 但
            # 实际上 cache hit 说明权重已在 SRAM, 直接全算:
            # 所有迭代均为 compute-only, 时长 = T_half * n_iters
            return self.T_half * self.n_iters(ntok)
        # 正常: Stage 1 = T_half (fetch+compute 首轮), Stage 2 = T_half*(n_iters-1)
        return self.T_half * self.n_iters(ntok)  # 相同，但 Stage 1 占 DMA

    def T_stage34(self, ntok: int, s3_cached: bool = False) -> int:
        """Stage 3+4 总时长 (Down Proj)."""
        return self.T_half * self.n_iters(ntok)

    def T_total(self, ntok: int) -> int:
        """无 prefetch, 完整四阶段总时长."""
        return self.T_stage12(ntok) + self.T_stage34(ntok)

    def eta(self, ntok: int) -> float:
        return min(1.0, ntok / self.M_dim)


SHAPE_A = Shape("8x8x8",   M_dim=8,  bw_req=32)
SHAPE_B = Shape("4x8x16",  M_dim=4,  bw_req=64)
SHAPE_C = Shape("2x8x32",  M_dim=2,  bw_req=128)
ALL_SHAPES = [SHAPE_A, SHAPE_B, SHAPE_C]


# ============================================================
#  Cluster 四阶段状态快照
# ============================================================

# 阶段常量
ST_IDLE    = 0   # 空闲
ST_S1      = 1   # SwishGLU Fetch+Compute (有 DMA)
ST_S2      = 2   # SwishGLU Compute-Only  (无 DMA)
ST_S3      = 3   # Down Proj Fetch+Compute (有 DMA)
ST_S4      = 4   # Down Proj Compute-Only  (无 DMA, Region_SWISH 已释放)

STAGE_NAMES = {
    ST_IDLE: "idle",
    ST_S1:   "Swish-Fetch+Compute",
    ST_S2:   "Swish-Compute",
    ST_S3:   "Down-Fetch+Compute",
    ST_S4:   "Down-Compute",
}


@dataclass(frozen=True)
class FourStageSnap:
    """
    Cluster 的不可变状态快照 (用于 Beam Search 去重 & 哈希).

    字段语义:
      task_end     : 当前任务 (四阶段全部) 的结束时刻
      s1_end       : Stage 1 (SwishGLU fetch) 结束时刻  (DMA 释放点)
      s2_end       : Stage 2 结束时刻 = s3_start (= s1_end 若 n_iters=1)
      s3_end       : Stage 3 (Down fetch) 结束时刻      (DMA 释放点)
      s4_start     : Stage 4 开始时刻 = s3_end
      bw_in_use    : 当前阶段的 DMA 带宽占用 (0 / 64 / 128)
      cur_eid      : 当前处理的 expert id (-1 = idle)
      pf_end       : prefetch 完成时刻 (-1 = 无 prefetch)
      pf_eid       : prefetch 目标 expert id (-1 = 无)
      pf_bw        : prefetch 占用带宽 (0 = 无)
    """
    task_end:   int
    s1_end:     int
    s2_end:     int    # = s3_start
    s3_end:     int
    s4_start:   int    # = s3_end
    bw_in_use:  int    # DMA bw 当前阶段 (0/64/128)
    cur_eid:    int    # -1 = idle
    pf_end:     int    # prefetch 完成时刻 (-1 = 无)
    pf_eid:     int    # prefetch expert  (-1 = 无)
    pf_bw:      int    # prefetch 带宽占用

    def is_idle_at(self, t: int) -> bool:
        return t >= self.task_end

    def active_bw_at(self, t: int) -> int:
        """返回时刻 t 的 DMA 带宽占用 (包含 prefetch)."""
        base_bw = 0
        if self.cur_eid >= 0:
            # Stage 1: [task_start, s1_end)
            if t < self.s1_end:
                base_bw = self.bw_in_use
            # Stage 3: [s2_end, s3_end)
            elif self.s2_end <= t < self.s3_end:
                base_bw = self.bw_in_use
        pf_bw = self.pf_bw if (self.pf_end > 0 and t < self.pf_end) else 0
        return base_bw + pf_bw

    def stage_at(self, t: int) -> int:
        """返回时刻 t 所处阶段."""
        if self.cur_eid < 0 or t >= self.task_end:
            return ST_IDLE
        if t < self.s1_end:
            return ST_S1
        if t < self.s2_end:
            return ST_S2
        if t < self.s3_end:
            return ST_S3
        return ST_S4

    @classmethod
    def from_assign(
        cls,
        start: int,
        shape: Shape,
        ntok: int,
        eid: int,
        s1_cached: bool = False,
        s3_cached: bool = False,
    ) -> "FourStageSnap":
        """构建 ASSIGN 动作后的新快照."""
        ni = shape.n_iters(ntok)
        T_h = shape.T_half

        if s1_cached:
            # Stage 1 已预取: 无 DMA, Stage 1 退化为 0, Stage 2 = T_h * ni
            s1_end = start
        else:
            s1_end = start + T_h

        s2_end = start + T_h * ni    # Stage 1+2 总时长 = T_h * ni
        s3_end = s2_end + (0 if s3_cached else T_h)
        task_end = s2_end + T_h * ni  # Stage 3+4 总时长 = T_h * ni

        return cls(
            task_end  = task_end,
            s1_end    = s1_end,
            s2_end    = s2_end,
            s3_end    = s3_end,
            s4_start  = s3_end,
            bw_in_use = shape.alloc,
            cur_eid   = eid,
            pf_end    = -1,
            pf_eid    = -1,
            pf_bw     = 0,
        )

    def with_prefetch(self, pf_eid: int, pf_shape: Shape, pf_start: int) -> "FourStageSnap":
        """在 Stage 4 期间追加 prefetch 动作，返回更新后的快照."""
        pf_end = pf_start + pf_shape.T_half
        return FourStageSnap(
            task_end  = self.task_end,
            s1_end    = self.s1_end,
            s2_end    = self.s2_end,
            s3_end    = self.s3_end,
            s4_start  = self.s4_start,
            bw_in_use = self.bw_in_use,
            cur_eid   = self.cur_eid,
            pf_end    = pf_end,
            pf_eid    = pf_eid,
            pf_bw     = pf_shape.alloc,
        )


IDLE_SNAP = FourStageSnap(
    task_end=0, s1_end=0, s2_end=0, s3_end=0, s4_start=0,
    bw_in_use=0, cur_eid=-1, pf_end=-1, pf_eid=-1, pf_bw=0
)


# ============================================================
#  动作记录 (用于甘特图重建)
# ============================================================

@dataclass(frozen=True)
class StageAction:
    """一步调度动作，可同时涉及 C2 和 C3."""
    # C2 分配 (-1 = 本步 C2 不动)
    c2_eid:       int
    c2_ntok:      int
    c2_shape:     Optional[Shape]
    c2_start:     int
    c2_s1_cached: bool
    c2_s3_cached: bool
    # C3 分配 (-1 = 本步 C3 不动)
    c3_eid:       int
    c3_ntok:      int
    c3_shape:     Optional[Shape]
    c3_start:     int
    c3_s1_cached: bool
    c3_s3_cached: bool
    # Prefetch (在 C2 或 C3 的 Stage4 期间)
    pf_cluster:   int   # 2 or 3, -1=无
    pf_eid:       int
    pf_shape:     Optional[Shape]
    pf_start:     int
    tag:          str = ""


def _no_assign() -> dict:
    return dict(eid=-1, ntok=0, shape=None, start=-1,
                s1_cached=False, s3_cached=False)


# ============================================================
#  搜索状态
# ============================================================

@dataclass
class BeamState:
    c2: FourStageSnap
    c3: FourStageSnap
    remaining: Tuple[Tuple[int, int], ...]   # (eid, ntok) 降序
    history:   Tuple[StageAction, ...]
    g_score:   int
    f_score:   int

    def __lt__(self, other: "BeamState") -> bool:
        if self.f_score != other.f_score:
            return self.f_score < other.f_score
        return self.g_score > other.g_score

    def fingerprint(self) -> tuple:
        # 简化指纹：仅用关键时间点和 remaining
        return (
            self.c2.task_end, self.c2.s1_end, self.c2.s2_end, self.c2.s3_end,
            self.c2.pf_end, self.c2.pf_eid,
            self.c3.task_end, self.c3.s1_end, self.c3.s2_end, self.c3.s3_end,
            self.c3.pf_end, self.c3.pf_eid,
            self.remaining,
        )


# ============================================================
#  下界估算
# ============================================================

def lb_remaining(remaining: Tuple[Tuple[int, int], ...]) -> int:
    """
    两机并行下界 (Johnson's Rule).
    每个 expert 最快完成时间用 Shape_C 全四阶段:
      T_best = Shape_C.T_total(ntok)
    LB = max( sum / 2, max_single )
    """
    if not remaining:
        return 0
    tasks = [SHAPE_C.T_total(ntok) for _, ntok in remaining]
    return max(sum(tasks) // 2, max(tasks))


# ============================================================
#  候选集剪枝 (SPLIT 优化)
# ============================================================

def _split_candidates(hot_ntok: int, sA: Shape, sB: Shape) -> List[int]:
    """仅枚举 T_task 阶梯步进边界的 splitA 值."""
    cands: set = set()
    k = 1
    while k * sA.M_dim < hot_ntok:
        cands.add(k * sA.M_dim)
        k += 1
    k = 1
    while hot_ntok - k * sB.M_dim > 0:
        cands.add(hot_ntok - k * sB.M_dim)
        k += 1
    return sorted(cands) if cands else [max(1, hot_ntok // 2)]


# ============================================================
#  合法动作生成
# ============================================================

def _bw_free(c2: FourStageSnap, c3: FourStageSnap, t: int) -> int:
    return MAX_BW - c2.active_bw_at(t) - c3.active_bw_at(t)


def _earliest_assign_start(
    cluster: FourStageSnap, peer: FourStageSnap, alloc_need: int
) -> int:
    """cluster 任务结束后满足 BW 约束的最早可开始时刻."""
    base = cluster.task_end
    # 检查若干关键时间点
    candidates = sorted({
        base, peer.s1_end, peer.s2_end, peer.s3_end, peer.s4_start,
        peer.pf_end if peer.pf_end > 0 else base,
    })
    for t in candidates:
        if t >= base:
            if MAX_BW - peer.active_bw_at(t) >= alloc_need:
                return t
    return max(base, peer.s3_end)  # 保守回退


def gen_stage_actions(
    c2: FourStageSnap,
    c3: FourStageSnap,
    remaining: Tuple[Tuple[int, int], ...],
) -> List[StageAction]:
    """
    生成所有合法动作:
      1. ASSIGN-SINGLE: 较早空闲的 cluster 立即分配一个 expert
      2. PAIR:         两者同时空闲时同时分配两个 expert
      3. SPLIT:        同时空闲时将最热 expert 拆给两个 cluster
      4. WAIT-PAIR/SPLIT: 较早空闲的等待对齐后 PAIR/SPLIT
      5. PREFETCH:     在现有 Stage-4 阶段追加 prefetch (独立动作)
    """
    actions: List[StageAction] = []
    n = len(remaining)
    if n == 0:
        return actions

    t2, t3 = c2.task_end, c3.task_end
    both_idle = (t2 == t3)

    # ─── 辅助函数 ───────────────────────────────────────────

    def _make_c2_assign(eid, ntok, shape, start, s1c=False, s3c=False,
                        c3_eid=-1, c3_ntok=0, c3_shape=None, c3_start=-1,
                        c3_s1c=False, c3_s3c=False,
                        pf_cl=-1, pf_eid=-1, pf_shape=None, pf_start=-1,
                        tag=""):
        return StageAction(
            c2_eid=eid, c2_ntok=ntok, c2_shape=shape, c2_start=start,
            c2_s1_cached=s1c, c2_s3_cached=s3c,
            c3_eid=c3_eid, c3_ntok=c3_ntok, c3_shape=c3_shape, c3_start=c3_start,
            c3_s1_cached=c3_s1c, c3_s3_cached=c3_s3c,
            pf_cluster=pf_cl, pf_eid=pf_eid, pf_shape=pf_shape, pf_start=pf_start,
            tag=tag,
        )

    def _make_c3_assign(eid, ntok, shape, start, s1c=False, s3c=False, tag=""):
        return StageAction(
            c2_eid=-1, c2_ntok=0, c2_shape=None, c2_start=-1,
            c2_s1_cached=False, c2_s3_cached=False,
            c3_eid=eid, c3_ntok=ntok, c3_shape=shape, c3_start=start,
            c3_s1_cached=s1c, c3_s3_cached=s3c,
            pf_cluster=-1, pf_eid=-1, pf_shape=None, pf_start=-1,
            tag=tag,
        )

    # ─── PAIR & SPLIT (两者同时空闲) ────────────────────────

    if both_idle:
        now = t2

        # PAIR
        if n >= 2:
            for i in range(min(n, 6)):   # 限制枚举深度
                for j in range(min(n, 6)):
                    if i == j:
                        continue
                    eidA, ntokA = remaining[i]
                    eidB, ntokB = remaining[j]
                    s1cA = (eidA == c2.pf_eid and c2.pf_end > 0 and c2.pf_end <= now)
                    s1cB = (eidB == c3.pf_eid and c3.pf_end > 0 and c3.pf_end <= now)
                    for sA in ALL_SHAPES:
                        for sB in ALL_SHAPES:
                            bwA = 0 if s1cA else sA.alloc
                            bwB = 0 if s1cB else sB.alloc
                            if bwA + bwB > MAX_BW:
                                continue
                            actions.append(StageAction(
                                c2_eid=eidA, c2_ntok=ntokA, c2_shape=sA,
                                c2_start=now, c2_s1_cached=s1cA, c2_s3_cached=False,
                                c3_eid=eidB, c3_ntok=ntokB, c3_shape=sB,
                                c3_start=now, c3_s1_cached=s1cB, c3_s3_cached=False,
                                pf_cluster=-1, pf_eid=-1, pf_shape=None, pf_start=-1,
                                tag=f"PAIR({eidA}+{eidB})",
                            ))

        # SPLIT (最热 expert)
        hot_eid, hot_ntok = remaining[0]
        if hot_ntok >= 4:
            s1c2 = (hot_eid == c2.pf_eid and c2.pf_end > 0 and c2.pf_end <= now)
            s1c3 = (hot_eid == c3.pf_eid and c3.pf_end > 0 and c3.pf_end <= now)
            for sA in ALL_SHAPES:
                for sB in ALL_SHAPES:
                    bwA = 0 if s1c2 else sA.alloc
                    bwB = 0 if s1c3 else sB.alloc
                    if bwA + bwB > MAX_BW:
                        continue
                    for splitA in _split_candidates(hot_ntok, sA, sB):
                        splitB = hot_ntok - splitA
                        if splitB <= 0:
                            continue
                        actions.append(StageAction(
                            c2_eid=hot_eid, c2_ntok=splitA, c2_shape=sA,
                            c2_start=now, c2_s1_cached=s1c2, c2_s3_cached=False,
                            c3_eid=hot_eid, c3_ntok=splitB, c3_shape=sB,
                            c3_start=now, c3_s1_cached=s1c3, c3_s3_cached=False,
                            pf_cluster=-1, pf_eid=-1, pf_shape=None, pf_start=-1,
                            tag=f"SPLIT({splitA},{splitB})",
                        ))

    # ─── SINGLE (较早空闲的 cluster 独立分配) ───────────────

    if t2 <= t3:
        # C2 先空闲
        for eid, ntok in remaining:
            s1c = (eid == c2.pf_eid and c2.pf_end > 0 and c2.pf_end <= t2)
            for shape in ALL_SHAPES:
                alloc_need = 0 if s1c else shape.alloc
                start = _earliest_assign_start(c2, c3, alloc_need)
                if _bw_free(c2, c3, start) < alloc_need:
                    continue
                actions.append(_make_c2_assign(
                    eid, ntok, shape, start, s1c=s1c,
                    tag=f"SINGLE-C2(E{eid})",
                ))
    else:
        # C3 先空闲
        for eid, ntok in remaining:
            s1c = (eid == c3.pf_eid and c3.pf_end > 0 and c3.pf_end <= t3)
            for shape in ALL_SHAPES:
                alloc_need = 0 if s1c else shape.alloc
                start = _earliest_assign_start(c3, c2, alloc_need)
                if _bw_free(c2, c3, start) < alloc_need:
                    continue
                actions.append(_make_c3_assign(
                    eid, ntok, shape, start, s1c=s1c,
                    tag=f"SINGLE-C3(E{eid})",
                ))

    # ─── WAIT-TO-PAIR / WAIT-TO-SPLIT ───────────────────────

    if not both_idle:
        wait_t = max(t2, t3)
        if n >= 2:
            for i in range(min(n, 4)):
                for j in range(min(n, 4)):
                    if i == j:
                        continue
                    eidA, ntokA = remaining[i]
                    eidB, ntokB = remaining[j]
                    s1cA = (eidA == c2.pf_eid and c2.pf_end > 0 and c2.pf_end <= wait_t)
                    s1cB = (eidB == c3.pf_eid and c3.pf_end > 0 and c3.pf_end <= wait_t)
                    for sA in ALL_SHAPES:
                        for sB in ALL_SHAPES:
                            bwA = 0 if s1cA else sA.alloc
                            bwB = 0 if s1cB else sB.alloc
                            if bwA + bwB > MAX_BW:
                                continue
                            actions.append(StageAction(
                                c2_eid=eidA, c2_ntok=ntokA, c2_shape=sA,
                                c2_start=wait_t, c2_s1_cached=s1cA, c2_s3_cached=False,
                                c3_eid=eidB, c3_ntok=ntokB, c3_shape=sB,
                                c3_start=wait_t, c3_s1_cached=s1cB, c3_s3_cached=False,
                                pf_cluster=-1, pf_eid=-1, pf_shape=None, pf_start=-1,
                                tag=f"WAIT-PAIR({eidA}+{eidB})",
                            ))

        # WAIT-SPLIT
        hot_eid, hot_ntok = remaining[0]
        if hot_ntok >= 4:
            s1c2 = (hot_eid == c2.pf_eid and c2.pf_end > 0 and c2.pf_end <= wait_t)
            s1c3 = (hot_eid == c3.pf_eid and c3.pf_end > 0 and c3.pf_end <= wait_t)
            for sA in ALL_SHAPES:
                for sB in ALL_SHAPES:
                    bwA = 0 if s1c2 else sA.alloc
                    bwB = 0 if s1c3 else sB.alloc
                    if bwA + bwB > MAX_BW:
                        continue
                    for splitA in _split_candidates(hot_ntok, sA, sB):
                        splitB = hot_ntok - splitA
                        if splitB <= 0:
                            continue
                        actions.append(StageAction(
                            c2_eid=hot_eid, c2_ntok=splitA, c2_shape=sA,
                            c2_start=wait_t, c2_s1_cached=s1c2, c2_s3_cached=False,
                            c3_eid=hot_eid, c3_ntok=splitB, c3_shape=sB,
                            c3_start=wait_t, c3_s1_cached=s1c3, c3_s3_cached=False,
                            pf_cluster=-1, pf_eid=-1, pf_shape=None, pf_start=-1,
                            tag=f"WAIT-SPLIT({splitA},{splitB})",
                        ))

    return actions


def gen_prefetch_actions(
    c2: FourStageSnap,
    c3: FourStageSnap,
    remaining: Tuple[Tuple[int, int], ...],
    current_time: int,
) -> List[StageAction]:
    """
    生成 PREFETCH 动作 (只在 Stage-4 期间允许).
    返回在现有 action 基础上追加 prefetch 的扩展状态.
    这些动作不改变 remaining, 只修改 cluster 快照中的 pf_* 字段.
    """
    pf_actions = []
    if not remaining:
        return pf_actions

    next_eid, _ = remaining[0]  # 只 prefetch 最可能下一个的 expert

    # C2 处于 Stage 4?
    stage2_now = c2.stage_at(current_time)
    if stage2_now == ST_S4 and c2.pf_eid < 0:
        pf_start = max(current_time, c2.s4_start)
        for shape in ALL_SHAPES:
            pf_bw = shape.alloc
            # 硬约束: 不能占用 C3 正在使用的带宽
            c3_bw_now = c3.active_bw_at(pf_start)
            if c3_bw_now + pf_bw > MAX_BW:
                continue
            pf_actions.append(StageAction(
                c2_eid=-2, c2_ntok=0, c2_shape=shape, c2_start=pf_start,
                c2_s1_cached=False, c2_s3_cached=False,
                c3_eid=-1, c3_ntok=0, c3_shape=None, c3_start=-1,
                c3_s1_cached=False, c3_s3_cached=False,
                pf_cluster=2, pf_eid=next_eid, pf_shape=shape, pf_start=pf_start,
                tag=f"PREFETCH-C2(E{next_eid}@{pf_start})",
            ))

    # C3 处于 Stage 4?
    stage3_now = c3.stage_at(current_time)
    if stage3_now == ST_S4 and c3.pf_eid < 0:
        pf_start = max(current_time, c3.s4_start)
        for shape in ALL_SHAPES:
            pf_bw = shape.alloc
            c2_bw_now = c2.active_bw_at(pf_start)
            if c2_bw_now + pf_bw > MAX_BW:
                continue
            pf_actions.append(StageAction(
                c2_eid=-1, c2_ntok=0, c2_shape=None, c2_start=-1,
                c2_s1_cached=False, c2_s3_cached=False,
                c3_eid=-2, c3_ntok=0, c3_shape=shape, c3_start=pf_start,
                c3_s1_cached=False, c3_s3_cached=False,
                pf_cluster=3, pf_eid=next_eid, pf_shape=shape, pf_start=pf_start,
                tag=f"PREFETCH-C3(E{next_eid}@{pf_start})",
            ))

    return pf_actions


# ============================================================
#  动作应用 → 生成后继状态
# ============================================================

def apply_stage_action(state: BeamState, action: StageAction) -> BeamState:
    c2, c3 = state.c2, state.c3
    rem = list(state.remaining)
    consumed = set()

    new_c2, new_c3 = c2, c3

    # PREFETCH 动作 (c2_eid == -2 or c3_eid == -2)
    if action.c2_eid == -2:
        # C2 prefetch
        new_c2 = c2.with_prefetch(action.pf_eid, action.pf_shape, action.pf_start)
        new_rem = tuple(rem)
        g = max(new_c2.task_end, new_c3.task_end)
        f = g + lb_remaining(new_rem)
        return BeamState(
            c2=new_c2, c3=new_c3,
            remaining=new_rem,
            history=state.history + (action,),
            g_score=g, f_score=f,
        )
    if action.c3_eid == -2:
        # C3 prefetch
        new_c3 = c3.with_prefetch(action.pf_eid, action.pf_shape, action.pf_start)
        new_rem = tuple(rem)
        g = max(new_c2.task_end, new_c3.task_end)
        f = g + lb_remaining(new_rem)
        return BeamState(
            c2=new_c2, c3=new_c3,
            remaining=new_rem,
            history=state.history + (action,),
            g_score=g, f_score=f,
        )

    # C2 分配
    if action.c2_eid >= 0:
        new_c2 = FourStageSnap.from_assign(
            action.c2_start, action.c2_shape, action.c2_ntok, action.c2_eid,
            s1_cached=action.c2_s1_cached, s3_cached=action.c2_s3_cached,
        )
        consumed.add(action.c2_eid)

    # C3 分配
    if action.c3_eid >= 0:
        new_c3 = FourStageSnap.from_assign(
            action.c3_start, action.c3_shape, action.c3_ntok, action.c3_eid,
            s1_cached=action.c3_s1_cached, s3_cached=action.c3_s3_cached,
        )
        consumed.add(action.c3_eid)

    new_rem = tuple((e, n) for e, n in rem if e not in consumed)
    g = max(new_c2.task_end, new_c3.task_end)
    f = g + lb_remaining(new_rem)

    return BeamState(
        c2=new_c2, c3=new_c3,
        remaining=new_rem,
        history=state.history + (action,),
        g_score=g, f_score=f,
    )


# ============================================================
#  Beam Search 主体
# ============================================================

class FourStageScheduler:
    """
    四阶段 Beam Search 调度器.
    同时搜索 ASSIGN + PREFETCH 动作空间.
    """

    def __init__(
        self,
        token_dist: Dict[int, int],
        beam_width: int = 64,
        enable_prefetch: bool = True,
        max_steps: int = 1000,
    ):
        self.token_dist = token_dist
        self.beam_width = beam_width
        self.enable_prefetch = enable_prefetch
        self.max_steps = max_steps

        self.initial_remaining: Tuple[Tuple[int, int], ...] = tuple(
            sorted(token_dist.items(), key=lambda x: -x[1])
        )

    def run(self) -> Tuple[int, List[StageAction]]:
        init_state = BeamState(
            c2=IDLE_SNAP, c3=IDLE_SNAP,
            remaining=self.initial_remaining,
            history=(),
            g_score=0,
            f_score=lb_remaining(self.initial_remaining),
        )

        beam: List[BeamState] = [init_state]
        heapq.heapify(beam)

        best_makespan = float("inf")
        best_history: List[StageAction] = []
        seen: Dict[tuple, int] = {}

        for step in range(self.max_steps):
            if not beam:
                break

            next_candidates: List[BeamState] = []

            for state in beam:
                if not state.remaining:
                    ms = state.g_score
                    if ms < best_makespan:
                        best_makespan = ms
                        best_history = list(state.history)
                    continue

                # 生成 ASSIGN 动作
                assign_actions = gen_stage_actions(
                    state.c2, state.c3, state.remaining
                )

                for action in assign_actions:
                    child = apply_stage_action(state, action)
                    fp = child.fingerprint()
                    if fp in seen and seen[fp] <= child.f_score:
                        continue
                    seen[fp] = child.f_score
                    next_candidates.append(child)

                # 生成 PREFETCH 动作 (不消耗 remaining，作为状态扩展)
                if self.enable_prefetch:
                    t_now = max(state.c2.s4_start, state.c3.s4_start)
                    pf_actions = gen_prefetch_actions(
                        state.c2, state.c3, state.remaining, t_now
                    )
                    for action in pf_actions:
                        child = apply_stage_action(state, action)
                        fp = child.fingerprint()
                        if fp in seen and seen[fp] <= child.f_score:
                            continue
                        seen[fp] = child.f_score
                        next_candidates.append(child)

            if not next_candidates:
                break

            next_candidates.sort()
            beam = next_candidates[: self.beam_width]

        return best_makespan, best_history


# ============================================================
#  时间轴甘特图 (四阶段版)
# ============================================================

def format_four_stage_timeline(
    history: List[StageAction],
    token_dist: Dict[int, int],
    makespan: int,
) -> str:
    """
    输出四阶段调度甘特图.
    每行是一个时间段，列出 SRAM_xDMA / SRAM_iDMA / C2_VC / C3_VC.
    Stage 标注: S1=SwishGLU-Fetch+Compute, S2=SwishGLU-Compute,
                S3=Down-Fetch+Compute,    S4=Down-Compute
    """
    # 1. 重建各 cluster 的详细段列表
    segs_c2: List[dict] = []
    segs_c3: List[dict] = []
    pf_segs: List[dict] = []   # prefetch 段

    for action in history:
        if action.c2_eid >= 0:
            snap = FourStageSnap.from_assign(
                action.c2_start, action.c2_shape, action.c2_ntok, action.c2_eid,
                s1_cached=action.c2_s1_cached,
            )
            segs_c2.append(dict(eid=action.c2_eid, ntok=action.c2_ntok,
                                shape=action.c2_shape, snap=snap,
                                s1_cached=action.c2_s1_cached))
        if action.c3_eid >= 0:
            snap = FourStageSnap.from_assign(
                action.c3_start, action.c3_shape, action.c3_ntok, action.c3_eid,
                s1_cached=action.c3_s1_cached,
            )
            segs_c3.append(dict(eid=action.c3_eid, ntok=action.c3_ntok,
                                shape=action.c3_shape, snap=snap,
                                s1_cached=action.c3_s1_cached))
        if action.pf_cluster > 0:
            pf_segs.append(dict(cluster=action.pf_cluster,
                                eid=action.pf_eid, shape=action.pf_shape,
                                start=action.pf_start,
                                end=action.pf_start + action.pf_shape.T_half))

    # 2. 收集所有事件点
    events = {0, makespan}
    for seg in segs_c2 + segs_c3:
        sn = seg["snap"]
        events |= {sn.s1_end, sn.s2_end, sn.s3_end, sn.s4_start, sn.task_end}
        events.discard(0)
        events.add(0)
    for pf in pf_segs:
        events |= {pf["start"], pf["end"]}
    events = sorted(events)

    # 3. 逐段生成甘特行
    col_w = 36
    header = (f"{'Start':>10}  {'End':>10}  {'Dur':>8}  "
              f"{'SRAM_xDMA':<{col_w}}{'SRAM_iDMA':<{col_w}}"
              f"{'C2_VersaCore':<{col_w}}{'C3_VersaCore':<{col_w}}")
    sep = "-" * len(header)
    rows = []

    def _seg_at(segs, t):
        """找时刻 t 所在的 seg."""
        for sg in segs:
            sn = sg["snap"]
            if sn.s1_end == 0 and t >= sn.s2_end:
                # s1_cached case: 从 task_start 开始 (用 s2_end - T_half*ni 估算)
                pass
            s_start = sn.s2_end - sg["shape"].T_half * sg["shape"].n_iters(sg["ntok"])
            if s_start <= t < sn.task_end:
                return sg, sn
        return None, None

    def _vc_label(sg, sn, t, side):
        if sg is None:
            return "idle"
        stage = sn.stage_at(t)
        s_start = sn.s2_end - sg["shape"].T_half * sg["shape"].n_iters(sg["ntok"])
        ni = sg["shape"].n_iters(sg["ntok"])
        eta_v = sg["shape"].eta(sg["ntok"])
        label = f"E{sg['eid']}({sg['ntok']}tok,η={eta_v:.2f})"
        if stage == ST_S1:
            if sg["s1_cached"]:
                return f"{label} [S1-cache,SwishGLU-compute]"
            return f"{label} [S1:SwishGLU-fetch+compute]"
        elif stage == ST_S2:
            return f"{label} [S2:SwishGLU-compute]"
        elif stage == ST_S3:
            return f"{label} [S3:Down-fetch+compute]"
        elif stage == ST_S4:
            return f"{label} [S4:Down-compute]"
        return "idle"

    for i in range(len(events) - 1):
        t_s = events[i]
        t_e = events[i + 1]
        tmid = t_s

        sg2, sn2 = _seg_at(segs_c2, tmid)
        sg3, sn3 = _seg_at(segs_c3, tmid)

        # DMA 分配
        bw2 = sn2.active_bw_at(tmid) - (sn2.pf_bw if sn2 and sn2.pf_end > tmid else 0) if sn2 else 0
        bw3 = sn3.active_bw_at(tmid) - (sn3.pf_bw if sn3 and sn3.pf_end > tmid else 0) if sn3 else 0
        pf2_active = any(p["cluster"] == 2 and p["start"] <= tmid < p["end"] for p in pf_segs)
        pf3_active = any(p["cluster"] == 3 and p["start"] <= tmid < p["end"] for p in pf_segs)

        # 判断各 DMA 通道分配
        xdma = "idle"; idma = "idle"
        if bw2 == 128:
            xdma = f"→C2: E{sg2['eid']}({sg2['shape'].name})"
            idma = f"→C2: E{sg2['eid']}({sg2['shape'].name})"
        elif bw3 == 128:
            xdma = f"→C3: E{sg3['eid']}({sg3['shape'].name})"
            idma = f"→C3: E{sg3['eid']}({sg3['shape'].name})"
        elif bw2 == 64 and bw3 == 64:
            xdma = f"→C2: E{sg2['eid']}({sg2['shape'].name})"
            idma = f"→C3: E{sg3['eid']}({sg3['shape'].name})"
        elif bw2 == 64:
            xdma = f"→C2: E{sg2['eid']}({sg2['shape'].name})"
        elif bw3 == 64:
            xdma = f"→C3: E{sg3['eid']}({sg3['shape'].name})"

        # prefetch 覆盖
        if pf2_active:
            pf = next(p for p in pf_segs if p["cluster"] == 2 and p["start"] <= tmid < p["end"])
            xdma = f"→C2-PF: E{pf['eid']}({pf['shape'].name})"
        if pf3_active:
            pf = next(p for p in pf_segs if p["cluster"] == 3 and p["start"] <= tmid < p["end"])
            xdma = f"→C3-PF: E{pf['eid']}({pf['shape'].name})"

        vc2_label = _vc_label(sg2, sn2, tmid, 2) if sg2 else "idle"
        vc3_label = _vc_label(sg3, sn3, tmid, 3) if sg3 else "idle"

        dur = t_e - t_s
        row = (f"{t_s:>10,}  {t_e:>10,}  {dur:>8,}  "
               f"{xdma:<{col_w}}{idma:<{col_w}}"
               f"{vc2_label:<{col_w}}{vc3_label:<{col_w}}")
        rows.append(row)

    total_bw = "=" * len(header)
    return "\n".join([
        total_bw,
        f"  四阶段甘特图  [makespan={makespan:,} cc]",
        f"  S1=SwishGLU-Fetch+Compute  S2=SwishGLU-Compute  S3=Down-Fetch+Compute  S4=Down-Compute",
        total_bw,
        header,
        sep,
        *rows,
        sep,
    ])


# ============================================================
#  工具: 效率统计
# ============================================================

def compute_efficiency(
    history: List[StageAction],
    makespan: int,
) -> dict:
    """统计 VersaCore 和 DMA 的利用率."""
    c2_compute = 0; c3_compute = 0
    dma_c2_busy = 0; dma_c3_busy = 0

    for act in history:
        if act.c2_eid >= 0:
            sn = FourStageSnap.from_assign(
                act.c2_start, act.c2_shape, act.c2_ntok, act.c2_eid,
                s1_cached=act.c2_s1_cached,
            )
            c2_compute += sn.task_end - act.c2_start
            if not act.c2_s1_cached:
                dma_c2_busy += act.c2_shape.T_half  # S1
            dma_c2_busy += act.c2_shape.T_half      # S3

        if act.c3_eid >= 0:
            sn = FourStageSnap.from_assign(
                act.c3_start, act.c3_shape, act.c3_ntok, act.c3_eid,
                s1_cached=act.c3_s1_cached,
            )
            c3_compute += sn.task_end - act.c3_start
            if not act.c3_s1_cached:
                dma_c3_busy += act.c3_shape.T_half
            dma_c3_busy += act.c3_shape.T_half

        if act.pf_cluster > 0 and act.pf_shape is not None:
            if act.pf_cluster == 2:
                dma_c2_busy += act.pf_shape.T_half
            else:
                dma_c3_busy += act.pf_shape.T_half

    return {
        "makespan":       makespan,
        "c2_vc_util":     c2_compute / makespan if makespan > 0 else 0,
        "c3_vc_util":     c3_compute / makespan if makespan > 0 else 0,
        "dma_c2_util":    dma_c2_busy / makespan if makespan > 0 else 0,
        "dma_c3_util":    dma_c3_busy / makespan if makespan > 0 else 0,
        "c2_compute":     c2_compute,
        "c3_compute":     c3_compute,
        "c2_idle":        makespan - c2_compute,
        "c3_idle":        makespan - c3_compute,
    }
