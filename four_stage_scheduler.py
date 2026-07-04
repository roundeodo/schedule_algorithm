#!/usr/bin/env python3
"""
四阶段专家调度器 v2 (Four-Stage Expert Scheduler)
=============================================================================
架构说明
--------
每个 expert 的计算分解为四个严格串行阶段（同一 cluster 内不可重排）：

  Stage 1 (SwishGLU Fetch+Compute):
      DMA 搬运 W_swish → Region_SWISH，VersaCore 同时计算首批 M_dim_s1 个 token。
      DMA 带宽: bw_s1   时长: T_h1 = ceil(W_half / bw_s1)
  Stage 2 (SwishGLU Compute-Only):
      非 cache hit：复用 Region_SWISH 计算剩余 max(0, ntok-M_dim_s1) 个 token。
      Cache hit：  S1 完全跳过，S2 计算全部 ntok 个 token（整块矩阵，无 DMA）。
      S2 期间可 prefetch S3 down 权重（scheduler 决定是否启动）。
      S2 无前台 DMA 约束，自由选 ShapeC(M2) 最小化尾部浪费。
      时长（非 hit）: ceil(max(0, ntok-M_dim_s1)/2) × T_s1(ShapeC)
      时长（hit）：   ceil(ntok/2) × T_s1(ShapeC)
  Stage 3 (Down Proj Fetch+Compute):
      硬依赖: Stage 2 必须完成
      DMA 搬运 W_down → Region_DOWN，带宽: bw_s3，T_h3 = ceil(W_half / bw_s3)
      ⚠ S1 和 S3 独立选择 Shape（M_dim 和 bw 可以不同）
      若 down 权重已 ready（cache hit 或 S2 prefetch），则 S3 完全跳过（时长=0）。
  Stage 4 (Down Proj Compute-Only):
      非 cache hit：复用 Region_DOWN 计算剩余 max(0, ntok-M_dim_s3) 个 token。
      Cache hit：  S3 完全跳过，S4 计算全部 ntok 个 token（整块矩阵，无 DMA）。
      S4 期间可 prefetch 下一 expert 的 S1 gate/up 权重（scheduler 决定是否启动）。
      S4 无前台 DMA 约束，自由选 ShapeC(M2) 最小化尾部浪费。
      时长（非 hit）: ceil(max(0, ntok-M_dim_s3)/2) × T_s3(ShapeC)
      时长（hit）：   ceil(ntok/2) × T_s3(ShapeC)
      Stage 4 开始时 Region_SWISH 释放 → 可 Prefetch 下一 expert

关键时间公式:
    S1 时长    = shape_s1.T_s1                             [未 ready: 首批 M_dim_s1 tokens，DMA+计算等时]
                            或 0                                       [cache hit: S1 完全跳过]
    S2 时长    = ceil(max(0, ntok-M_dim_s1)/2)×T_s1(C)    [未 ready: 处理尾部 token]
                            或 ceil(ntok/2)×T_s1(C)                   [cache hit: 处理全部 token]
  S1+S2 共  = T_s1(shape_s1) + _best_s2_compute(ntok-M_dim_s1)   [未 ready]
             = _best_s2_compute(ntok)                                [cache hit]
    S3 时长    = shape_s3.T_s3                             [未 ready: 首批 M_dim_s3 tokens，DMA+计算等时]
                            或 0                                       [cache hit: S3 完全跳过]
    S4 时长    = ceil(max(0, ntok-M_dim_s3)/2)×T_s3(C)    [未 ready: 处理尾部 token]
                            或 ceil(ntok/2)×T_s3(C)                   [cache hit: 处理全部 token]
  S3+S4 共  = T_s3(shape_s3) + _best_s4_compute(ntok-M_dim_s3)   [未 ready]
             = _best_s4_compute(ntok)                                [cache hit]

BW 约束 (精确验证):
  任意时刻: C2.active_bw(t) + C3.active_bw(t) ≤ 128 B/cc
  验证方式: 枚举所有 BW 变化点（s1_end, s2_end, s3_end, pf_start/end）逐一检查

Beam Search:
  beam_width=64 → 每步保留 64 个最优状态展开（非贪心）
  f_score = g + lb_remaining 作为启发
"""

import math
import heapq
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

# ============================================================
#  物理常量
# ============================================================

WEIGHT_BYTES_TOTAL = 3 * 2048 * 1408 // 2  # 4,325,376 B  (gate+up+down)
WEIGHT_BYTES_S1 = 2 * 2048 * 1408 // 2  # 2,883,584 B  (SwishGLU: gate+up)
WEIGHT_BYTES_S3 = 1 * 2048 * 1408 // 2  # 1,441,792 B  (Down projection)
MAX_BW = 128  # B/cc

PF_EID_GHOST = -2  # snap.pf_eid 哨兵：「S4 窗口可预取某人，但具体是谁由分配后回填」
# swiglu_hit 遇到 PF_EID_GHOST 时对任意 eid 返回 True


# ============================================================
#  Shape 定义
# ============================================================


@dataclass(frozen=True)
class Shape:
    name: str
    M_dim: int
    bw_req: int  # B/cc

    @property
    def T_s1(self) -> int:
        """S1 (SwishGLU gate+up) 一次迭代计算时长。"""
        return math.ceil(WEIGHT_BYTES_S1 / self.bw_req)

    @property
    def T_s3(self) -> int:
        """S3 (Down projection) 一次迭代计算时长。"""
        return math.ceil(WEIGHT_BYTES_S3 / self.bw_req)

    @property
    def alloc(self) -> int:
        return 64 if self.bw_req <= 64 else 128

    @property
    def t_dma_s1(self) -> int:
        """S1 DMA 实际搬运时长（alloc-bound，gate+up）.

        ShapeA: alloc=64, t_dma_s1=45,056 cc，T_s1=90,112 cc
                → DMA 在 T_s1/2 时完成，后半段仅计算，BW 已释放。
        ShapeB/C: alloc=bw_req，t_dma_s1=T_s1（DMA 与计算同步结束）。
        """
        return math.ceil(WEIGHT_BYTES_S1 / self.alloc)

    @property
    def t_dma_s3(self) -> int:
        """S3 DMA 实际搬运时长（alloc-bound，down only）."""
        return math.ceil(WEIGHT_BYTES_S3 / self.alloc)

    @property
    def t_dma(self) -> int:
        """Prefetch 搬运时长（Prefetch 预取 S1 的 gate+up 权重）。"""
        return self.t_dma_s1

    def n_iters(self, ntok: int) -> int:
        return math.ceil(ntok / self.M_dim)

    def T_s1_task(self, ntok: int) -> int:
        """S1+S2 总时长（SwishGLU gate+up）。"""
        return self.T_s1 * self.n_iters(ntok)

    def T_s3_task(self, ntok: int) -> int:
        """S3+S4 总时长（Down projection）。"""
        return self.T_s3 * self.n_iters(ntok)

    def T_task(self, ntok: int) -> int:
        """单 expert 完整时长（S1+S2+S3+S4）。"""
        return self.T_s1_task(ntok) + self.T_s3_task(ntok)

    def eta(self, ntok: int) -> float:
        return min(1.0, ntok / self.M_dim)


SHAPE_A = Shape("A(M8,bw32)", M_dim=8, bw_req=32)
SHAPE_B = Shape("B(M4,bw64)", M_dim=4, bw_req=64)
SHAPE_C = Shape("C(M2,bw128)", M_dim=2, bw_req=128)
ALL_SHAPES = [SHAPE_A, SHAPE_B, SHAPE_C]

# Reference-beam knobs.  These keep the action space strictly richer than the
# analytical/fast schedulers while avoiding a combinatorial explosion.
PREFETCH_EID_DEPTH = 1
S2PF_VARIANT_LIMIT = 2
PAIR_HEAD_DEPTH = 5
WAIT_PAIR_HEAD_DEPTH = 4
PAIR_TAIL_DEPTH = 3
PAIR_MAX_CANDIDATES = 10
SPLIT_MAX_EXPERTS = 2
BOUNDARY_NTOKS = {
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    9,
    15,
    16,
    17,
    31,
    32,
    33,
    47,
    48,
    49,
    63,
    64,
    65,
    95,
    96,
    97,
    127,
    128,
    129,
    191,
    192,
    193,
    255,
    256,
}


def _best_s2_compute(remaining: int) -> int:
    """S2 最优 compute 时长：用 ShapeC(M2) 处理剩余 token，无 DMA 约束。
    所有 shape 的 per-token 吞吐相同，ShapeC M_dim=2 尾部浪费最少（至多 1 个槽位）。
    """
    if remaining <= 0:
        return 0
    return math.ceil(remaining / SHAPE_C.M_dim) * SHAPE_C.T_s1


def _best_s4_compute(remaining: int) -> int:
    """S4 最优 compute 时长：用 ShapeC(M2) 处理剩余 token，无 DMA 约束。"""
    if remaining <= 0:
        return 0
    return math.ceil(remaining / SHAPE_C.M_dim) * SHAPE_C.T_s3


# 当两 cluster 同时非缓存启动 S1 时，合法 (sA1,sB1) 对（alloc_A + alloc_B ≤ 128）
# ShapeA.alloc=64, ShapeB.alloc=64, ShapeC.alloc=128
# A+A=128, A+B=128, B+B=128 ✓；含 ShapeC 均超 128 ✗
CONCURRENT_S1_PAIRS = [
    (SHAPE_A, SHAPE_A),
    (SHAPE_A, SHAPE_B),
    (SHAPE_B, SHAPE_A),
    (SHAPE_B, SHAPE_B),
]

ST_IDLE = 0
ST_S1 = 1
ST_S2 = 2
ST_S3 = 3
ST_S4 = 4


# ============================================================
#  Cluster 状态快照
# ============================================================


@dataclass(frozen=True)
class FourStageSnap:
    """
    Cluster 完整状态快照。

    时间边界 (绝对时刻):
      task_start : 当前任务开始
      dma1_end   : Stage 1 DMA 搬运结束 (≤ s1_end; ShapeA 时 = task_start + 33,792)
    s1_end     : Stage 1 计算结束
      s2_end     : Stage 2 结束 = Stage 3 开始
      dma3_end   : Stage 3 DMA 搬运结束 (≤ s3_end; ShapeA 时 = s2_end + 33,792)
      s3_end     : Stage 3 计算结束 = Stage 4 开始
      task_end   : Stage 4 结束

    关键：ShapeA(bw_req=32, alloc=64) 在 S1 阶段 t_dma_s1=45,056 cc，
    计算则需 T_s1=90,112 cc。BW 约束仅对 [task_start, dma1_end) 窗口有效。
    ShapeB/C 的 t_dma_s1=T_s1（DMA 与计算同步结束）。

    DMA 带宽:
      bw_s1 : Stage 1 DMA 占用 (0 若 cached)
      bw_s3 : Stage 3 DMA 占用

        Prefetch:
            s2pf_start/s2pf_end/s2pf_bw : S2 期间预取当前 expert 的 down 权重
            pf_start/pf_end/pf_eid/pf_bw: S4 期间预取下一个 expert 的 S1 权重
    """

    task_start: int
    task_end: int
    dma1_end: int  # S1 DMA 实际结束（≤ s1_end；ShapeA 时 dma1_end=task_start+45,056）
    s1_end: int
    s2_end: int
    dma3_end: int  # S3 DMA 实际结束（≤ s3_end）
    s3_end: int
    s4_start: int  # = s3_end
    bw_s1: int
    bw_s3: int
    cur_eid: int  # -1 = idle
    pf_start: int  # -1 = 无
    pf_end: int  # -1 = 无
    pf_eid: int  # -1 = 无
    pf_bw: int
    pf_full: bool = False  # True only for full expert residency, not S1-only prefetch
    s2pf_start: int = -1  # -1 = 无；S2 down prefetch 起点
    s2pf_end: int = -1
    s2pf_bw: int = 0
    ntok: int = 0  # token count for this task (needed for cached S4 compute time)

    def is_idle_at(self, t: int) -> bool:
        return self.cur_eid < 0 or t >= self.task_end

    def active_bw_at(self, t: int) -> int:
        bw = 0
        if self.cur_eid >= 0:
            # BW 仅在 DMA 实际搬运期间（dma1_end/dma3_end）消耗，
            # 不延伸至计算结束（s1_end/s3_end）——ShapeA 修正关键处。
            if self.task_start <= t < self.dma1_end:
                bw = self.bw_s1
            elif self.s2_end <= t < self.dma3_end:
                bw = self.bw_s3
        if self.pf_start >= 0 and self.pf_start <= t < self.pf_end:
            bw += self.pf_bw
        if self.s2pf_start >= 0 and self.s2pf_start <= t < self.s2pf_end:
            bw += self.s2pf_bw
        return bw

    def stage_at(self, t: int) -> int:
        if self.cur_eid < 0 or t >= self.task_end:
            return ST_IDLE
        if self.task_start <= t < self.s1_end:
            return ST_S1
        if t < self.s2_end:
            return ST_S2
        if t < self.s3_end:
            return ST_S3
        return ST_S4

    def bw_change_pts(self) -> set:
        # 包含 dma1_end/dma3_end：ShapeA 在这两个时刻 BW 提前释放。
        pts = {
            self.task_start,
            self.dma1_end,
            self.s1_end,
            self.s2_end,
            self.dma3_end,
            self.s3_end,
            self.task_end,
        }
        if self.pf_start >= 0:
            pts |= {self.pf_start, self.pf_end}
        if self.s2pf_start >= 0:
            pts |= {self.s2pf_start, self.s2pf_end}
        return pts

    @classmethod
    def from_assign(
        cls,
        start: int,
        shape_s1: "Shape",
        shape_s3: "Shape",
        ntok: int,
        eid: int,
        s1_cached: bool = False,
        s3_cached: bool = False,
        s2pf_start: int = -1,
    ) -> "FourStageSnap":
        T_h1 = shape_s1.T_s1
        T_h3 = shape_s3.T_s3
        # ── S1/S2 phase ──────────────────────────────────────────────────────
        # Cache hit → S1 entirely skipped; S2 computes ALL ntok tokens with ShapeC.
        # Non-hit   → S1 overlap DMA+compute for first M_dim_s1 tokens; S2 tail.
        if s1_cached:
            s1_end = start
            s2_end = start + _best_s2_compute(ntok)
        else:
            remaining_s2 = max(0, ntok - shape_s1.M_dim)
            s1_end = start + T_h1
            s2_end = s1_end + _best_s2_compute(remaining_s2)
        # ── S2 down-weight prefetch ───────────────────────────────────────────
        use_s2pf = s2pf_start >= 0
        if use_s2pf:
            s2pf_end = s2pf_start + shape_s3.t_dma_s3
            if s2pf_start < start or s2pf_end > s2_end:
                raise ValueError("down prefetch must fit fully inside S1+S2")
        else:
            s2pf_end = -1
        # ── S3/S4 phase ──────────────────────────────────────────────────────
        # Cache hit (or S2 prefetch) → S3 entirely skipped; S4 computes ALL ntok.
        # Non-hit   → S3 overlap DMA+compute for first M_dim_s3 tokens; S4 tail.
        s3_ready = s3_cached or use_s2pf
        if s3_ready:
            s3_end = s2_end
            task_end = s2_end + _best_s4_compute(ntok)
        else:
            remaining_s4 = max(0, ntok - shape_s3.M_dim)
            s3_end = s2_end + T_h3
            task_end = s3_end + _best_s4_compute(remaining_s4)
        # ── DMA end points ────────────────────────────────────────────────────
        dma1_end = start if s1_cached else (start + shape_s1.t_dma_s1)
        dma3_end = s2_end if s3_ready else (s2_end + shape_s3.t_dma_s3)
        return cls(
            task_start=start,
            task_end=task_end,
            dma1_end=dma1_end,
            s1_end=s1_end,
            s2_end=s2_end,
            dma3_end=dma3_end,
            s3_end=s3_end,
            s4_start=s3_end,
            bw_s1=0 if s1_cached else shape_s1.alloc,
            bw_s3=0 if s3_ready else shape_s3.alloc,
            cur_eid=eid,
            pf_start=-1,
            pf_end=-1,
            pf_eid=-1,
            pf_bw=0,
            pf_full=False,
            s2pf_start=s2pf_start,
            s2pf_end=s2pf_end,
            s2pf_bw=shape_s3.alloc if use_s2pf else 0,
            ntok=ntok,
        )

    def with_s2_down_prefetch(
        self, shape_s3: "Shape", s2pf_start: int
    ) -> "FourStageSnap":
        s2pf_end = s2pf_start + shape_s3.t_dma_s3
        if self.bw_s3 == 0:
            return self
        if s2pf_start < self.task_start or s2pf_end > self.s2_end:
            raise ValueError("down prefetch must fit fully inside S1+S2")
        # S3 now skipped (weights loaded by S2 prefetch);
        # S4 computes ALL self.ntok tokens with ShapeC.
        new_task_end = self.s2_end + _best_s4_compute(self.ntok)
        return FourStageSnap(
            task_start=self.task_start,
            task_end=new_task_end,
            dma1_end=self.dma1_end,
            s1_end=self.s1_end,
            s2_end=self.s2_end,
            dma3_end=self.s2_end,
            s3_end=self.s2_end,  # S3 entirely skipped
            s4_start=self.s2_end,  # S4 starts right after S2
            bw_s1=self.bw_s1,
            bw_s3=0,
            cur_eid=self.cur_eid,
            pf_start=self.pf_start,
            pf_end=self.pf_end,
            pf_eid=self.pf_eid,
            pf_bw=self.pf_bw,
            pf_full=self.pf_full,
            s2pf_start=s2pf_start,
            s2pf_end=s2pf_end,
            s2pf_bw=shape_s3.alloc,
            ntok=self.ntok,
        )

    def with_prefetch(
        self, pf_eid: int, pf_shape: "Shape", pf_start: int
    ) -> "FourStageSnap":
        return FourStageSnap(
            task_start=self.task_start,
            task_end=self.task_end,
            dma1_end=self.dma1_end,
            s1_end=self.s1_end,
            s2_end=self.s2_end,
            dma3_end=self.dma3_end,
            s3_end=self.s3_end,
            s4_start=self.s4_start,
            bw_s1=self.bw_s1,
            bw_s3=self.bw_s3,
            cur_eid=self.cur_eid,
            pf_start=pf_start,
            pf_end=pf_start + pf_shape.t_dma_s1,  # Prefetch 预取 S1 权重（gate+up）
            pf_eid=pf_eid,
            pf_bw=pf_shape.alloc,
            pf_full=False,
            s2pf_start=self.s2pf_start,
            s2pf_end=self.s2pf_end,
            s2pf_bw=self.s2pf_bw,
            ntok=self.ntok,
        )


IDLE_SNAP = FourStageSnap(
    task_start=0,
    task_end=0,
    dma1_end=0,
    s1_end=0,
    s2_end=0,
    dma3_end=0,
    s3_end=0,
    s4_start=0,
    bw_s1=0,
    bw_s3=0,
    cur_eid=-1,
    pf_start=-1,
    pf_end=-1,
    pf_eid=-1,
    pf_bw=0,
    pf_full=False,
    ntok=0,
)


def make_initial_snap(cached_eid: int) -> FourStageSnap:
    """
    构造一个表示「调度开始前 SRAM 中已完整缓存某 expert 权重」的 cluster 快照。

    原理：将 cached_eid 设为 pf_eid，pf_end=0，pf_full=True。
    这样初始 resident hit 会同时跳过 Stage-1 和 Stage-3 前台 DMA。

    物理含义：前一轮推理结束时，cluster 最后处理的 expert（或最后 Prefetch 的 expert）
    的 W_swish 权重仍留在 Region_SWISH SRAM 中，新一轮调度可以直接复用。
    """
    if cached_eid < 0:
        return IDLE_SNAP
    return FourStageSnap(
        task_start=0,
        task_end=0,
        dma1_end=0,
        s1_end=0,
        s2_end=0,
        dma3_end=0,
        s3_end=0,
        s4_start=0,
        bw_s1=0,
        bw_s3=0,
        cur_eid=-1,
        pf_start=-1,
        pf_end=0,  # pf_end=0: 已在 t=0 完成，调度开始时即有效
        pf_eid=cached_eid,
        pf_bw=0,
        pf_full=True,
        ntok=0,
    )


# ============================================================
#  BW 约束精确验证
# ============================================================


@lru_cache(maxsize=1_000_000)
def bw_feasible(snap_a: FourStageSnap, snap_b: FourStageSnap) -> bool:
    """枚举所有 BW 变化点，检查总带宽约束。"""
    for t in snap_a.bw_change_pts() | snap_b.bw_change_pts():
        if snap_a.active_bw_at(t) + snap_b.active_bw_at(t) > MAX_BW:
            return False
    return True


@lru_cache(maxsize=1_000_000)
def _s2_down_prefetch_start_candidates(
    snap: FourStageSnap, shape_s3: Shape, peers: Tuple[FourStageSnap, ...] = ()
) -> List[int]:
    """枚举 S1+S2 内可让当前 expert 的 down DMA 完整结束的候选起点。"""
    if snap.cur_eid < 0 or snap.bw_s3 == 0:
        return []
    dma = shape_s3.t_dma_s3
    lo = snap.task_start
    hi = snap.s2_end - dma
    if hi < lo:
        return []
    cands = {lo, hi}
    for pt in (snap.dma1_end, snap.s1_end):
        if lo <= pt <= hi:
            cands.add(pt)
    for src in (snap,) + peers:
        for pt in src.bw_change_pts():
            if lo <= pt <= hi:
                cands.add(pt)
            aligned = pt - dma
            if lo <= aligned <= hi:
                cands.add(aligned)
    return sorted(cands)


def with_optional_s2_down_prefetch(
    snap: FourStageSnap, shape_s3: Shape, peer: Optional[FourStageSnap] = None
) -> FourStageSnap:
    """若 S1+S2 可完整隐藏当前 expert 的 down DMA，则返回带 down prefetch 的 snap。"""
    peers = (peer,) if peer is not None else ()
    for start in _s2_down_prefetch_start_candidates(snap, shape_s3, peers):
        cand = snap.with_s2_down_prefetch(shape_s3, start)
        if (peer is None and bw_feasible(cand, IDLE_SNAP)) or (
            peer is not None and bw_feasible(cand, peer)
        ):
            return cand
    return snap


def with_optional_s2_down_prefetch_pair(
    snap_a: FourStageSnap,
    shape_a_s3: Shape,
    snap_b: FourStageSnap,
    shape_b_s3: Shape,
) -> Tuple[FourStageSnap, FourStageSnap]:
    """为一对同时评估的 snap 选择可行的 S2 down-prefetch 组合。"""
    starts_a = [-1] + _s2_down_prefetch_start_candidates(snap_a, shape_a_s3, (snap_b,))
    starts_b = [-1] + _s2_down_prefetch_start_candidates(snap_b, shape_b_s3, (snap_a,))
    best_a, best_b = snap_a, snap_b
    best_score = -1
    best_start_sum = 10**18
    for start_a in starts_a:
        cand_a = (
            snap_a.with_s2_down_prefetch(shape_a_s3, start_a)
            if start_a >= 0
            else snap_a
        )
        for start_b in starts_b:
            cand_b = (
                snap_b.with_s2_down_prefetch(shape_b_s3, start_b)
                if start_b >= 0
                else snap_b
            )
            if not bw_feasible(cand_a, cand_b):
                continue
            score = int(cand_a.s2pf_start >= 0) + int(cand_b.s2pf_start >= 0)
            start_sum = (cand_a.s2pf_start if cand_a.s2pf_start >= 0 else 0) + (
                cand_b.s2pf_start if cand_b.s2pf_start >= 0 else 0
            )
            if score > best_score or (
                score == best_score and start_sum < best_start_sum
            ):
                best_a, best_b = cand_a, cand_b
                best_score = score
                best_start_sum = start_sum
    return best_a, best_b


@lru_cache(maxsize=1_000_000)
def _next_s1_prefetch_start_candidates(
    snap: FourStageSnap, pf_shape: Shape, peers: Tuple[FourStageSnap, ...] = ()
) -> List[int]:
    """枚举 S3+S4 内启动下一 expert gate/up 预取的候选起点。"""
    if snap.cur_eid < 0 or snap.pf_eid != -1:
        return []
    dma = pf_shape.t_dma_s1
    lo = snap.s2_end
    hi = snap.task_end
    if hi < lo:
        return []
    cands = {lo, snap.dma3_end, snap.s3_end, hi}
    for src in (snap,) + peers:
        for pt in src.bw_change_pts():
            if lo <= pt <= hi:
                cands.add(pt)
            aligned = pt - dma
            if lo <= aligned <= hi:
                cands.add(aligned)
    return sorted(cands)


def with_optional_next_s1_prefetch(
    snap: FourStageSnap,
    next_eid: int,
    peer: Optional[FourStageSnap] = None,
) -> FourStageSnap:
    """若 S3+S4 可启动下一 expert 的 gate/up 预取，则返回带 prefetch 的 snap。"""
    peers = (peer,) if peer is not None else ()
    best: Optional[FourStageSnap] = None
    for pf_shape in ALL_SHAPES:
        for start in _next_s1_prefetch_start_candidates(snap, pf_shape, peers):
            cand = snap.with_prefetch(next_eid, pf_shape, start)
            feasible = (
                bw_feasible(cand, peer)
                if peer is not None
                else bw_feasible(cand, IDLE_SNAP)
            )
            if not feasible:
                continue
            if best is None or cand.pf_end < best.pf_end:
                best = cand
            break
    return best if best is not None else snap


def with_optional_next_s1_prefetch_pair(
    snap_a: FourStageSnap,
    snap_b: FourStageSnap,
    next_eid: int,
) -> Tuple[FourStageSnap, FourStageSnap]:
    """为两个当前 task 选择可行的 S3+S4 next-S1 预取组合。"""
    cand_a = [snap_a]
    cand_b = [snap_b]
    for pf_shape in ALL_SHAPES:
        for start in _next_s1_prefetch_start_candidates(snap_a, pf_shape, (snap_b,)):
            cand = snap_a.with_prefetch(next_eid, pf_shape, start)
            if bw_feasible(cand, snap_b):
                cand_a.append(cand)
                break
    for pf_shape in ALL_SHAPES:
        for start in _next_s1_prefetch_start_candidates(snap_b, pf_shape, (snap_a,)):
            cand = snap_b.with_prefetch(next_eid, pf_shape, start)
            if bw_feasible(snap_a, cand):
                cand_b.append(cand)
                break

    best_a, best_b = snap_a, snap_b
    best_score = -1
    best_end_sum = 10**18
    for cand1 in cand_a:
        for cand2 in cand_b:
            if not bw_feasible(cand1, cand2):
                continue
            score = int(cand1.pf_eid == next_eid) + int(cand2.pf_eid == next_eid)
            end_sum = (cand1.pf_end if cand1.pf_eid == next_eid else 0) + (
                cand2.pf_end if cand2.pf_eid == next_eid else 0
            )
            if score > best_score or (score == best_score and end_sum < best_end_sum):
                best_a, best_b = cand1, cand2
                best_score = score
                best_end_sum = end_sum
    return best_a, best_b


def _apply_optional_s2pf_start(
    snap: FourStageSnap, shape_s3: Shape, start: int
) -> FourStageSnap:
    if start < 0:
        return snap
    try:
        return snap.with_s2_down_prefetch(shape_s3, start)
    except ValueError:
        return snap


def _snap_prefetch_key(snap: FourStageSnap) -> tuple:
    return (
        snap.task_end,
        snap.bw_s3,
        snap.s2pf_start,
        snap.s2pf_end,
        snap.s2pf_bw,
        snap.pf_start,
        snap.pf_end,
        snap.pf_eid,
        snap.pf_bw,
    )


def enumerate_s2_down_prefetch_variants(
    snap: FourStageSnap,
    shape_s3: Shape,
    peer: Optional[FourStageSnap] = None,
) -> List[FourStageSnap]:
    """Return a small reference set of S2 down-prefetch choices.

    The fast schedulers greedily select one S2PF placement.  The reference beam
    must include that placement, but also keep no-prefetch and a few boundary
    alternatives so that early BW occupation is not forced prematurely.
    """
    peer_snap = peer if peer is not None else IDLE_SNAP
    selected = {-1}
    greedy = with_optional_s2_down_prefetch(snap, shape_s3, peer)
    selected.add(greedy.s2pf_start)

    variants: List[FourStageSnap] = []
    seen = set()
    for start in sorted(selected):
        cand = _apply_optional_s2pf_start(snap, shape_s3, start)
        feasible = bw_feasible(cand, peer_snap)
        if not feasible:
            continue
        key = _snap_prefetch_key(cand)
        if key in seen:
            continue
        seen.add(key)
        variants.append(cand)

    variants.sort(key=lambda s: (s.task_end, s.s2pf_start < 0, s.s2pf_start))
    return variants[:S2PF_VARIANT_LIMIT]


def enumerate_s2_down_prefetch_pair_variants(
    snap_a: FourStageSnap,
    shape_a_s3: Shape,
    snap_b: FourStageSnap,
    shape_b_s3: Shape,
) -> List[Tuple[FourStageSnap, FourStageSnap]]:
    """Reference-beam S2PF variants for paired assignments.

    This deliberately contains the analytical/C greedy choice plus no-prefetch
    and one-sided/boundary choices, making the beam action space a superset of
    the downstream schedulers' S2PF behavior.
    """
    greedy_a, greedy_b = with_optional_s2_down_prefetch_pair(
        snap_a, shape_a_s3, snap_b, shape_b_s3
    )
    starts_a = {-1, greedy_a.s2pf_start}
    starts_b = {-1, greedy_b.s2pf_start}

    variants: List[Tuple[FourStageSnap, FourStageSnap]] = []
    seen = set()
    for start_a in sorted(starts_a):
        cand_a = _apply_optional_s2pf_start(snap_a, shape_a_s3, start_a)
        for start_b in sorted(starts_b):
            cand_b = _apply_optional_s2pf_start(snap_b, shape_b_s3, start_b)
            if not bw_feasible(cand_a, cand_b):
                continue
            key = (_snap_prefetch_key(cand_a), _snap_prefetch_key(cand_b))
            if key in seen:
                continue
            seen.add(key)
            variants.append((cand_a, cand_b))

    def rank(pair: Tuple[FourStageSnap, FourStageSnap]) -> tuple:
        a, b = pair
        is_greedy = a.s2pf_start == greedy_a.s2pf_start and b.s2pf_start == greedy_b.s2pf_start
        score = int(a.s2pf_start >= 0) + int(b.s2pf_start >= 0)
        return (not is_greedy, -score, max(a.task_end, b.task_end), a.s2pf_start, b.s2pf_start)

    variants.sort(key=rank)
    return variants[:S2PF_VARIANT_LIMIT]


def inject_ghost_prefetch_pair(
    snap_a: FourStageSnap,
    snap_b: FourStageSnap,
) -> Tuple[FourStageSnap, FourStageSnap]:
    """
    C 风格的 Ghost 预取注入（配对版本）。

    对每个 snap：若 cluster 正在运行（cur_eid >= 0），尚无任何预取（pf_eid == -1），
    且 S4 窗口长度 ≥ SHAPE_A.t_dma_s1，则注入 PF_EID_GHOST。

    PF_EID_GHOST 语义：「这个 cluster 的 S4 窗口可以预取下一个 expert 的 S1 权重，
    但具体预取谁由后续分配决策决定」。swiglu_hit 遇到 GHOST 时对任意 eid 返回 True。

    与 with_optional_next_s1_prefetch_pair 的区别：
      - 不绑定具体 eid（→ 任何分配到该 cluster 的 expert 都能享受 skip_s1）
      - 使用 SHAPE_A（bw=64 B/cc）保守估计，pf_end = s4_start + SHAPE_A.t_dma_s1
      - 只注入一次（pf_eid != -1 时跳过，包括 GHOST 已存在的情况）
    """

    def _try_ghost(snap: FourStageSnap, peer: FourStageSnap) -> FourStageSnap:
        if snap.cur_eid < 0 or snap.pf_eid != -1:
            return snap
        if snap.task_end - snap.s4_start < SHAPE_A.t_dma_s1:
            return snap
        candidate = snap.with_prefetch(PF_EID_GHOST, SHAPE_A, snap.s4_start)
        return candidate if bw_feasible(candidate, peer) else snap

    new_a = _try_ghost(snap_a, snap_b)
    new_b = _try_ghost(snap_b, new_a)
    return new_a, new_b


# ============================================================
#  动作记录
# ============================================================


@dataclass(frozen=True)
class StageAction:
    """
    一步调度动作。
    c2_eid=-1: C2 本步不动; c2_eid=-2: C2 做 prefetch（不消耗 remaining）
    S1 和 S3 独立记录 shape_s1 和 shape_s3。
    """

    c2_eid: int
    c2_ntok: int
    c2_shape_s1: Optional[Shape]
    c2_shape_s3: Optional[Shape]
    c2_start: int
    c2_s1_cached: bool
    c2_s3_cached: bool
    c3_eid: int
    c3_ntok: int
    c3_shape_s1: Optional[Shape]
    c3_shape_s3: Optional[Shape]
    c3_start: int
    c3_s1_cached: bool
    c3_s3_cached: bool
    pf_cluster: int
    pf_eid: int
    pf_shape: Optional[Shape]
    pf_start: int
    tag: str = ""
    c2_s2pf_start: int = -1
    c3_s2pf_start: int = -1


# ============================================================
#  Beam Search 状态
# ============================================================


@dataclass
class BeamState:
    c2: FourStageSnap
    c3: FourStageSnap
    remaining: Tuple[Tuple[int, int], ...]
    history: Tuple[StageAction, ...]
    g_score: int
    f_score: int

    def __lt__(self, other: "BeamState") -> bool:
        if self.f_score != other.f_score:
            return self.f_score < other.f_score
        return self.g_score > other.g_score

    def fingerprint(self) -> tuple:
        def snap_key(s: FourStageSnap) -> tuple:
            return (
                s.task_start,
                s.task_end,
                s.dma1_end,
                s.s1_end,
                s.s2_end,
                s.dma3_end,
                s.s3_end,
                s.s4_start,
                s.bw_s1,
                s.bw_s3,
                s.cur_eid,
                s.pf_start,
                s.pf_end,
                s.pf_eid,
                s.pf_bw,
                s.pf_full,
                s.s2pf_start,
                s.s2pf_end,
                s.s2pf_bw,
                s.ntok,
            )

        return (
            snap_key(self.c2),
            snap_key(self.c3),
            self.remaining,
        )


# ============================================================
#  下界估算
# ============================================================


def _best_task_time(ntok: int) -> int:
    """对单个 expert，在无 BW 约束下最优 (shape_s1, shape_s3) 的最短时长。"""
    return min(s.T_s1_task(ntok) for s in ALL_SHAPES) + min(
        s.T_s3_task(ntok) for s in ALL_SHAPES
    )


# 预计算: 两 cluster 并行时（各自 DMA 不超 MAX_BW/2=64 B/cc），ntok tokens 的最短时长
# 约束: s1.alloc * 2 ≤ MAX_BW  且  bw_s3 * 2 ≤ MAX_BW（ShapeC 的 128 B/cc 不能并发）
# 等价于: s1 ∈ {ShapeA,ShapeB}(alloc=64) 且 s3 ∈ {ShapeA,ShapeB}(bw_s3≤64)
_CONCURRENT_TASK_CACHE: Dict[int, int] = {}


def _best_concurrent_task_time(ntok: int) -> int:
    """两 cluster 同时运行时（BW 各限 64 B/cc）ntok tokens 的最短任务时长。

    两个 cluster 同时进行 S1/S3 DMA 时，每个的 bw_s1 ≤ 64 且 bw_s3 ≤ 64，
    即不能使用 ShapeC（alloc/bw_s3=128）作为并发 S1/S3。
    """
    if ntok in _CONCURRENT_TASK_CACHE:
        return _CONCURRENT_TASK_CACHE[ntok]
    best = 10**9  # 初始化为大值，仅考虑 BW 约束内的 shape 对
    for s1 in ALL_SHAPES:
        if s1.alloc * 2 > MAX_BW:
            continue  # ShapeC alloc=128，两个并发超限
        for s3 in ALL_SHAPES:
            bw_s3 = WEIGHT_BYTES_S3 / s3.T_s3
            if bw_s3 * 2 > MAX_BW:
                continue  # ShapeC bw_s3=128，两个并发超限
            sn = FourStageSnap.from_assign(0, s1, s3, ntok, eid=0)
            best = min(best, sn.task_end)
    _CONCURRENT_TASK_CACHE[ntok] = best
    return best


def lb_remaining(remaining: Tuple[Tuple[int, int], ...]) -> int:
    """两 cluster 调度下界：BW-aware 可纳下界。

    per-task 时长估算策略（取三者最小）：
      1) seq  = _best_task_time(ntok)
               独占所有 BW（仅当 n=1 时有效，n≥2 时存在伙伴竞争 BW）
      2) conc = _best_concurrent_task_time(ntok)
               与另一个 expert 并行，每侧 BW ≤ 64 B/cc（n≥2 时的实际约束）
      3) splt = _best_concurrent_task_time(ceil(ntok/2))
               SPLIT：两 cluster 各处理半边，BW ≤ 64 B/cc

    当 n=1 时：无伙伴竞争，per_task = min(seq, splt)（可用 ShapeC 独占 128 B/cc）
    当 n≥2 时：总有伙伴，per_task = min(conc, splt)
               conc 比 seq 更紧（ShapeC 独占 128 B/cc 不可行），修正了
               (1,1) 等小 token 场景下 lb 严重偏低的问题。

    聚合: lb = max(sum(per_tasks) // 2, max(per_tasks))
    可纳性已验证：lb ≤ 真实最优 makespan。
    """
    if not remaining:
        return 0
    n = len(remaining)
    tasks = []
    for _, ntok in remaining:
        half = math.ceil(ntok / 2)
        splt = _best_concurrent_task_time(half)
        if n == 1:
            # 单 expert：独占 BW（ShapeC 合法），但也可 SPLIT
            seq = _best_task_time(ntok)
            tasks.append(min(seq, splt))
        else:
            # 多 expert：存在伙伴竞争 BW，每侧 ≤ 64 B/cc
            conc = _best_concurrent_task_time(ntok)
            tasks.append(min(conc, splt))
    return max(sum(tasks) // 2, max(tasks))


def dma_lb_total(remaining: Tuple[Tuple[int, int], ...]) -> int:
    """纯 DMA 带宽下界（仅用于报告，不参与搜索）:
    每个 expert 需搬运 W_s1+W_s3=W_total 字节（gate+up+down），128 B/cc 是物理峰值。"""
    if not remaining:
        return 0
    return math.ceil(len(remaining) * WEIGHT_BYTES_TOTAL / MAX_BW)


# ============================================================
#  SPLIT 候选集
# ============================================================


def _split_candidates(hot_ntok: int, sA: Shape, sB: Shape) -> List[int]:
    """生成 SPLIT 分割点候选集（C2 分配的 token 数）。

    候选来源：
    1) ceil(ntok/2)：RTL 实现的标准均分，始终包含（最关键的候选）
    2) k*sA.M_dim：使 C2 的 S1 无尾部浪费的切割点
    3) ntok - k*sB.M_dim：使 C3 的 S1 无尾部浪费的切割点
    """
    cands: set = set()
    # 始终加入 ceil/floor 分割（与 RTL 的实现一致，且通常是最优切割点）
    cands.add(math.ceil(hot_ntok / 2))
    # M_dim 对齐的切割点（减少 S1 尾部浪费）
    k = 1
    while k * sA.M_dim < hot_ntok:
        cands.add(k * sA.M_dim)
        k += 1
    k = 1
    while k * sB.M_dim < hot_ntok:
        cands.add(hot_ntok - k * sB.M_dim)
        k += 1
    cands.discard(0)
    cands.discard(hot_ntok)
    return sorted(cands) if cands else [max(1, hot_ntok // 2)]


def _swiglu_hit_for_candidate(eid: int, snap: FourStageSnap, t: int) -> bool:
    if snap.pf_end < 0 or snap.pf_end > t:
        return False
    return snap.pf_eid == PF_EID_GHOST or snap.pf_eid == eid


def _down_hit_for_candidate(eid: int, snap: FourStageSnap, t: int) -> bool:
    return _swiglu_hit_for_candidate(eid, snap, t) and snap.pf_full


def _is_boundary_ntok(ntok: int) -> bool:
    return ntok in BOUNDARY_NTOKS


def _pair_candidate_indices(
    remaining: Tuple[Tuple[int, int], ...],
    c2: FourStageSnap,
    c3: FourStageSnap,
    now: int,
    is_wait: bool,
) -> List[int]:
    """PAIR candidate experts: semantic representatives under a small cap.

    Analytical/fast schedulers mostly reason about top experts.  The reference
    beam also keeps cold, cached/prefetched, and shape-boundary representatives,
    because those are the cases that can change the downstream decision.
    """
    n = len(remaining)
    head_depth = WAIT_PAIR_HEAD_DEPTH if is_wait else PAIR_HEAD_DEPTH
    idxs = set(range(min(n, head_depth)))

    # Cold experts can fill holes and change future alignment. Keep a few real
    # tail representatives rather than relying on rank quantiles.
    tail_start = max(0, n - PAIR_TAIL_DEPTH)
    idxs.update(range(tail_start, n))

    # Cache/prefetch-relevant experts are semantically special even if cold.
    for i, (eid, _) in enumerate(remaining):
        if (
            _swiglu_hit_for_candidate(eid, c2, now)
            or _swiglu_hit_for_candidate(eid, c3, now)
            or _down_hit_for_candidate(eid, c2, now)
            or _down_hit_for_candidate(eid, c3, now)
        ):
            idxs.add(i)

    # Boundary token counts are where shape/tail behaviour changes. Keep one
    # representative per boundary ntok.
    seen_boundary_ntok = set()
    for i, (_, ntok) in enumerate(remaining):
        if _is_boundary_ntok(ntok) and ntok not in seen_boundary_ntok:
            idxs.add(i)
            seen_boundary_ntok.add(ntok)

    # Keep representatives for distinct token-count classes from hot to cold.
    seen_ntok = set()
    for i, (_, ntok) in enumerate(remaining):
        if ntok in seen_ntok:
            continue
        idxs.add(i)
        seen_ntok.add(ntok)
        if len(idxs) >= PAIR_MAX_CANDIDATES:
            break

    # If still too many, keep semantic priority: hot/cache/boundary/cold.
    def priority(i: int) -> tuple:
        eid, ntok = remaining[i]
        cache_rel = (
            _swiglu_hit_for_candidate(eid, c2, now)
            or _swiglu_hit_for_candidate(eid, c3, now)
            or _down_hit_for_candidate(eid, c2, now)
            or _down_hit_for_candidate(eid, c3, now)
        )
        is_tail = i >= tail_start
        return (
            0 if i < head_depth else 1,
            0 if cache_rel else 1,
            0 if _is_boundary_ntok(ntok) else 1,
            0 if is_tail else 1,
            i,
        )

    return sorted(sorted((i for i in idxs if 0 <= i < n), key=priority)[:PAIR_MAX_CANDIDATES])


def _split_expert_indices(remaining: Tuple[Tuple[int, int], ...]) -> List[int]:
    """Experts worth trying SPLIT on.

    SPLIT is mainly useful for hot experts, but top1/top2 can matter when they
    are close to top0 or sit exactly on a shape boundary.
    """
    if not remaining:
        return []
    idxs = {0}
    top_ntok = remaining[0][1]
    for i, (_, ntok) in enumerate(remaining[1:], start=1):
        if len(idxs) >= SPLIT_MAX_EXPERTS:
            break
        close_to_top = ntok >= math.ceil(0.90 * top_ntok)
        substantial = ntok >= 16
        boundary_and_close = (
            _is_boundary_ntok(ntok)
            and substantial
            and ntok >= math.ceil(0.90 * top_ntok)
        )
        if (close_to_top and substantial) or boundary_and_close:
            idxs.add(i)
    return sorted(idxs)


# ============================================================
#  最早可开始时刻
# ============================================================


def _earliest_start(
    cluster_end: int,
    peer: FourStageSnap,
    ntok: int,
    shape_s1: Shape,
    shape_s3: Shape,
    s1_cached: bool = False,
    s3_cached: bool = False,
) -> int:
    """找最早的 start >= cluster_end，使得新任务与 peer 满足 BW 约束。"""
    candidates = sorted(
        {t for t in peer.bw_change_pts() if t >= cluster_end} | {cluster_end}
    )
    for t in candidates:
        snap_new = FourStageSnap.from_assign(
            t, shape_s1, shape_s3, ntok, 0, s1_cached, s3_cached
        )
        snap_new = with_optional_s2_down_prefetch(snap_new, shape_s3, peer)
        if bw_feasible(snap_new, peer):
            return t
    return max(cluster_end, peer.task_end)


def _start_candidates(cluster_end: int, peer: FourStageSnap) -> List[int]:
    """Reference-beam start candidates for assigning to an idle cluster."""
    cands = {cluster_end, peer.task_end}
    for pt in peer.bw_change_pts():
        if pt >= cluster_end:
            cands.add(pt)
    return sorted(cands)


# ============================================================
#  动作生成
# ============================================================


def gen_stage_actions(
    c2: FourStageSnap,
    c3: FourStageSnap,
    remaining: Tuple[Tuple[int, int], ...],
) -> List[StageAction]:
    """
    生成所有合法的 ASSIGN / WAIT 动作。
    对每种动作枚举所有 (shape_s1, shape_s3) 组合（9 种），
    用 bw_feasible 精确验证 BW 约束。
    """
    if not remaining:
        return []

    actions: List[StageAction] = []
    t2, t3 = c2.task_end, c3.task_end
    both_idle = t2 == t3
    n = len(remaining)

    def swiglu_hit(eid: int, snap: FourStageSnap, t: int) -> bool:
        if snap.pf_end < 0 or snap.pf_end > t:
            return False
        return snap.pf_eid == PF_EID_GHOST or snap.pf_eid == eid

    def down_hit(eid: int, snap: FourStageSnap, t: int) -> bool:
        return swiglu_hit(eid, snap, t) and snap.pf_full

    def make_pair(now, eidA, ntokA, sA1, sA3, c2_sw, eidB, ntokB, sB1, sB3, c3_sw, tag):
        c2_dn = down_hit(eidA, c2, now)
        c3_dn = down_hit(eidB, c3, now)
        sna = FourStageSnap.from_assign(now, sA1, sA3, ntokA, eidA, c2_sw, c2_dn)
        snb = FourStageSnap.from_assign(now, sB1, sB3, ntokB, eidB, c3_sw, c3_dn)
        out = []
        for va, vb in enumerate_s2_down_prefetch_pair_variants(sna, sA3, snb, sB3):
            if not bw_feasible(va, vb):
                continue
            out.append(
                StageAction(
                    c2_eid=eidA,
                    c2_ntok=ntokA,
                    c2_shape_s1=sA1,
                    c2_shape_s3=sA3,
                    c2_start=now,
                    c2_s1_cached=c2_sw,
                    c2_s3_cached=va.bw_s3 == 0,
                    c3_eid=eidB,
                    c3_ntok=ntokB,
                    c3_shape_s1=sB1,
                    c3_shape_s3=sB3,
                    c3_start=now,
                    c3_s1_cached=c3_sw,
                    c3_s3_cached=vb.bw_s3 == 0,
                    pf_cluster=-1,
                    pf_eid=-1,
                    pf_shape=None,
                    pf_start=-1,
                    tag=tag,
                    c2_s2pf_start=va.s2pf_start,
                    c3_s2pf_start=vb.s2pf_start,
                )
            )
        return out

    # ─── PAIR & SPLIT（包含 WAIT- 变体）────────────────────────────

    schedule_times = []
    if both_idle:
        schedule_times.append((t2, False))
    else:
        schedule_times.append((max(t2, t3), True))

    for now_time, is_wait in schedule_times:
        prefix = "WAIT-" if is_wait else ""
        pair_indices = _pair_candidate_indices(remaining, c2, c3, now_time, is_wait)
        split_indices = _split_expert_indices(remaining)

        # PAIR
        if n >= 2:
            for i in pair_indices:
                for j in pair_indices:
                    if i == j:
                        continue
                    eidA, ntokA = remaining[i]
                    eidB, ntokB = remaining[j]
                    c2_sw = swiglu_hit(eidA, c2, now_time)
                    c3_sw = swiglu_hit(eidB, c3, now_time)
                    # 当两 cluster 同时启动 S1 时，限制合法 S1 对（节省 ~55% 时间）
                    s1_pairs = (
                        [(s, t) for s in ALL_SHAPES for t in ALL_SHAPES]
                        if (c2_sw or c3_sw)
                        else CONCURRENT_S1_PAIRS
                    )
                    for sA1, sB1 in s1_pairs:
                        for sA3 in ALL_SHAPES:
                            for sB3 in ALL_SHAPES:
                                pair_actions = make_pair(
                                    now_time,
                                    eidA,
                                    ntokA,
                                    sA1,
                                    sA3,
                                    c2_sw,
                                    eidB,
                                    ntokB,
                                    sB1,
                                    sB3,
                                    c3_sw,
                                    f"{prefix}PAIR({eidA}+{eidB})",
                                )
                                actions.extend(pair_actions)

        # SPLIT hot/boundary representatives.  This keeps top0, plus at most a
        # couple of top1/topK experts when they are close to top0 or sit on a
        # shape boundary.
        for split_idx in split_indices:
            split_eid, split_ntok = remaining[split_idx]
            if split_ntok < 2:
                continue
            c2_sw = swiglu_hit(split_eid, c2, now_time)
            c3_sw = swiglu_hit(split_eid, c3, now_time)
            s1_pairs = (
                [(s, t) for s in ALL_SHAPES for t in ALL_SHAPES]
                if (c2_sw or c3_sw)
                else CONCURRENT_S1_PAIRS
            )
            for sA1, sB1 in s1_pairs:
                for sA3 in ALL_SHAPES:
                    for sB3 in ALL_SHAPES:
                        for spA in _split_candidates(split_ntok, sA1, sB1):
                            spB = split_ntok - spA
                            if spB <= 0:
                                continue
                            split_actions = make_pair(
                                now_time,
                                split_eid,
                                spA,
                                sA1,
                                sA3,
                                c2_sw,
                                split_eid,
                                spB,
                                sB1,
                                sB3,
                                c3_sw,
                                f"{prefix}SPLIT(E{split_eid}:{spA},{spB})",
                            )
                            actions.extend(split_actions)

    # ─── SINGLE（较早空闲的 cluster 立即分配）────────────────────────

    if t2 <= t3:
        for eid, ntok in remaining:
            c2_sw = swiglu_hit(eid, c2, t2)
            c2_dn = down_hit(eid, c2, t2)
            for s1 in ALL_SHAPES:
                for s3 in ALL_SHAPES:
                    for start in _start_candidates(t2, c3):
                        sn0 = FourStageSnap.from_assign(
                            start, s1, s3, ntok, eid, c2_sw, c2_dn
                        )
                        for sn in enumerate_s2_down_prefetch_variants(sn0, s3, c3):
                            if bw_feasible(sn, c3):
                                actions.append(
                                    StageAction(
                                        c2_eid=eid,
                                        c2_ntok=ntok,
                                        c2_shape_s1=s1,
                                        c2_shape_s3=s3,
                                        c2_start=start,
                                        c2_s1_cached=c2_sw,
                                        c2_s3_cached=sn.bw_s3 == 0,
                                        c3_eid=-1,
                                        c3_ntok=0,
                                        c3_shape_s1=None,
                                        c3_shape_s3=None,
                                        c3_start=-1,
                                        c3_s1_cached=False,
                                        c3_s3_cached=False,
                                        pf_cluster=-1,
                                        pf_eid=-1,
                                        pf_shape=None,
                                        pf_start=-1,
                                        tag=f"SINGLE-C2(E{eid})",
                                        c2_s2pf_start=sn.s2pf_start,
                                    )
                                )
    if t3 < t2:
        for eid, ntok in remaining:
            c3_sw = swiglu_hit(eid, c3, t3)
            c3_dn = down_hit(eid, c3, t3)
            for s1 in ALL_SHAPES:
                for s3 in ALL_SHAPES:
                    for start in _start_candidates(t3, c2):
                        sn0 = FourStageSnap.from_assign(
                            start, s1, s3, ntok, eid, c3_sw, c3_dn
                        )
                        for sn in enumerate_s2_down_prefetch_variants(sn0, s3, c2):
                            if bw_feasible(c2, sn):
                                actions.append(
                                    StageAction(
                                        c2_eid=-1,
                                        c2_ntok=0,
                                        c2_shape_s1=None,
                                        c2_shape_s3=None,
                                        c2_start=-1,
                                        c2_s1_cached=False,
                                        c2_s3_cached=False,
                                        c3_eid=eid,
                                        c3_ntok=ntok,
                                        c3_shape_s1=s1,
                                        c3_shape_s3=s3,
                                        c3_start=start,
                                        c3_s1_cached=c3_sw,
                                        c3_s3_cached=sn.bw_s3 == 0,
                                        pf_cluster=-1,
                                        pf_eid=-1,
                                        pf_shape=None,
                                        pf_start=-1,
                                        tag=f"SINGLE-C3(E{eid})",
                                        c3_s2pf_start=sn.s2pf_start,
                                    )
                                )

    return actions


def gen_prefetch_actions(
    c2: FourStageSnap,
    c3: FourStageSnap,
    remaining: Tuple[Tuple[int, int], ...],
    current_time: int,
) -> List[StageAction]:
    """在 S3+S4 期间生成下一 expert 的 S1 Prefetch 动作（不消耗 remaining）。"""
    if not remaining:
        return []
    pf_actions: List[StageAction] = []
    prefetch_targets = [eid for eid, _ in remaining[:PREFETCH_EID_DEPTH]]

    for cl, peer, cl_id in [(c2, c3, 2), (c3, c2, 3)]:
        if cl.cur_eid < 0:
            continue
        if cl.pf_eid != -1:
            continue
        for next_eid in prefetch_targets:
            for pf_shape in ALL_SHAPES:
                chosen_start = -1
                for pf_start in _next_s1_prefetch_start_candidates(cl, pf_shape, (peer,)):
                    if pf_start < current_time:
                        continue
                    cand = cl.with_prefetch(next_eid, pf_shape, pf_start)
                    if bw_feasible(cand, peer):
                        chosen_start = pf_start
                        break
                if chosen_start < 0:
                    continue
                if cl_id == 2:
                    pf_actions.append(
                        StageAction(
                            c2_eid=-2,
                            c2_ntok=0,
                            c2_shape_s1=pf_shape,
                            c2_shape_s3=None,
                            c2_start=chosen_start,
                            c2_s1_cached=False,
                            c2_s3_cached=False,
                            c3_eid=-1,
                            c3_ntok=0,
                            c3_shape_s1=None,
                            c3_shape_s3=None,
                            c3_start=-1,
                            c3_s1_cached=False,
                            c3_s3_cached=False,
                            pf_cluster=2,
                            pf_eid=next_eid,
                            pf_shape=pf_shape,
                            pf_start=chosen_start,
                            tag=f"PF-C2(E{next_eid},{pf_shape.name})",
                        )
                    )
                else:
                    pf_actions.append(
                        StageAction(
                            c2_eid=-1,
                            c2_ntok=0,
                            c2_shape_s1=None,
                            c2_shape_s3=None,
                            c2_start=-1,
                            c2_s1_cached=False,
                            c2_s3_cached=False,
                            c3_eid=-2,
                            c3_ntok=0,
                            c3_shape_s1=pf_shape,
                            c3_shape_s3=None,
                            c3_start=chosen_start,
                            c3_s1_cached=False,
                            c3_s3_cached=False,
                            pf_cluster=3,
                            pf_eid=next_eid,
                            pf_shape=pf_shape,
                            pf_start=chosen_start,
                            tag=f"PF-C3(E{next_eid},{pf_shape.name})",
                        )
                    )
    return pf_actions


# ============================================================
#  动作应用
# ============================================================


def apply_action(state: BeamState, action: StageAction) -> BeamState:
    c2, c3 = state.c2, state.c3
    rem = list(state.remaining)
    consumed: set = set()
    new_c2, new_c3 = c2, c3

    if action.c2_eid == -2:
        new_c2 = c2.with_prefetch(action.pf_eid, action.pf_shape, action.pf_start)
        new_rem = tuple(rem)
        g = max(new_c2.task_end, new_c3.task_end)
        return BeamState(
            c2=new_c2,
            c3=new_c3,
            remaining=new_rem,
            history=state.history + (action,),
            g_score=g,
            f_score=g + lb_remaining(new_rem),
        )
    if action.c3_eid == -2:
        new_c3 = c3.with_prefetch(action.pf_eid, action.pf_shape, action.pf_start)
        new_rem = tuple(rem)
        g = max(new_c2.task_end, new_c3.task_end)
        return BeamState(
            c2=new_c2,
            c3=new_c3,
            remaining=new_rem,
            history=state.history + (action,),
            g_score=g,
            f_score=g + lb_remaining(new_rem),
        )

    if action.c2_eid >= 0:
        new_c2 = FourStageSnap.from_assign(
            action.c2_start,
            action.c2_shape_s1,
            action.c2_shape_s3,
            action.c2_ntok,
            action.c2_eid,
            action.c2_s1_cached,
            action.c2_s3_cached,
            action.c2_s2pf_start,
        )
        consumed.add(action.c2_eid)
    if action.c3_eid >= 0:
        new_c3 = FourStageSnap.from_assign(
            action.c3_start,
            action.c3_shape_s1,
            action.c3_shape_s3,
            action.c3_ntok,
            action.c3_eid,
            action.c3_s1_cached,
            action.c3_s3_cached,
            action.c3_s2pf_start,
        )
        consumed.add(action.c3_eid)

    new_rem = tuple((e, n) for e, n in rem if e not in consumed)
    g = max(new_c2.task_end, new_c3.task_end)
    return BeamState(
        c2=new_c2,
        c3=new_c3,
        remaining=new_rem,
        history=state.history + (action,),
        g_score=g,
        f_score=g + lb_remaining(new_rem),
    )


def _state_family(state: BeamState) -> str:
    if not state.history:
        return "INIT"
    tag = state.history[-1].tag
    if tag.startswith("PF-"):
        return "PF"
    if "PAIR" in tag:
        return "PAIR"
    if "SPLIT" in tag:
        return "SPLIT"
    if tag.startswith("SINGLE"):
        return "SINGLE"
    return "OTHER"


def _last_action(state: BeamState) -> Optional[StageAction]:
    return state.history[-1] if state.history else None


def _action_work_score(action: Optional[StageAction]) -> int:
    if action is None:
        return 0
    return max(action.c2_ntok, 0) + max(action.c3_ntok, 0)


def _action_hit_score(action: Optional[StageAction]) -> int:
    if action is None:
        return 0
    return (
        int(action.c2_s1_cached)
        + int(action.c3_s1_cached)
        + int(action.c2_s3_cached)
        + int(action.c3_s3_cached)
    )


def _append_unique_state(
    out: List[BeamState], seen_ids: set, cand: BeamState, limit: int
) -> bool:
    if id(cand) in seen_ids:
        return False
    out.append(cand)
    seen_ids.add(id(cand))
    return len(out) >= limit


def _select_family_items(
    ordered: List[BeamState], family: str, quota: int
) -> List[BeamState]:
    family_items = [cand for cand in ordered if _state_family(cand) == family]
    if len(family_items) <= quota:
        return family_items

    selected: List[BeamState] = []
    seen_ids = set()

    # Quality lane inside the family.
    for cand in family_items:
        if _append_unique_state(selected, seen_ids, cand, quota):
            return selected
        if len(selected) >= max(1, quota // 2):
            break

    # Semantic lane: keep high-work and cache/prefetch-hit actions that may have
    # worse immediate f-score but better downstream alignment.
    semantic = sorted(
        family_items,
        key=lambda s: (
            -_action_hit_score(_last_action(s)),
            -_action_work_score(_last_action(s)),
            s.f_score,
            -s.g_score,
        ),
    )
    for cand in semantic:
        if _append_unique_state(selected, seen_ids, cand, quota):
            return selected

    for cand in family_items:
        if _append_unique_state(selected, seen_ids, cand, quota):
            return selected
    return selected


def _select_family_diverse_beam(
    ordered: List[BeamState], beam_width: int
) -> List[BeamState]:
    """Family-balanced lane used beside the quality lane."""
    if len(ordered) <= beam_width:
        return ordered

    quotas = {
        "PAIR": max(4, beam_width * 5 // 16),
        "SPLIT": max(4, beam_width * 5 // 16),
        "SINGLE": max(4, beam_width // 4),
        "PF": max(1, beam_width // 16),
    }
    selected: List[BeamState] = []
    selected_ids = set()

    for family, quota in quotas.items():
        for cand in _select_family_items(ordered, family, quota):
            if _append_unique_state(selected, selected_ids, cand, beam_width):
                break
        if len(selected) >= beam_width:
            break

    for cand in ordered:
        if len(selected) >= beam_width:
            break
        if id(cand) in selected_ids:
            continue
        selected.append(cand)
        selected_ids.add(id(cand))

    selected.sort()
    return selected


def _select_reference_beam(candidates: List[BeamState], beam_width: int) -> List[BeamState]:
    """Family-diverse reference beam.

    Plain f-score top-K can be crowded by many near-identical states from one
    action family, especially tiny-first SINGLE or same-cost SPLIT variants.
    The reference beam therefore reserves capacity for PAIR/SPLIT/SINGLE/PF
    families before filling remaining slots by f-score.
    """
    if len(candidates) <= beam_width:
        return candidates
    ordered = sorted(candidates)
    return _select_family_diverse_beam(ordered, beam_width)


# ============================================================
#  Beam Search 主体
# ============================================================


class FourStageScheduler:
    """
    四阶段 Beam Search 调度器（多步 look-ahead，非贪心）。

    beam_width=64 → 每步保留 64 个最优候选状态继续展开。
    全程枚举 (shape_s1, shape_s3) 9 种组合，BW 约束精确验证。
    """

    def __init__(
        self,
        token_dist: Dict[int, int],
        beam_width: int = 64,
        enable_prefetch: bool = True,
        max_steps: int = 2000,
        initial_cache_c2: int = -1,
        initial_cache_c3: int = -1,
    ):
        """
        Parameters
        ----------
        token_dist        : {expert_id: token_count} 当前轮的 top-K 路由结果
        beam_width        : beam search 宽度（推荐 32-64）
        enable_prefetch   : 是否允许 Stage-4 期间触发 prefetch
        initial_cache_c2  : C2 在调度开始前 SRAM 中已缓存的 expert ID（-1 表示空）
        initial_cache_c3  : C3 在调度开始前 SRAM 中已缓存的 expert ID（-1 表示空）

        注意: 两个 cluster 有独立 SRAM，因此允许缓存同一个 expert。被缓存的
        expert 不在 token_dist 中时，本轮不会产生 hit，但状态本身仍合法。
        """
        self.token_dist = token_dist
        self.beam_width = beam_width
        self.enable_prefetch = enable_prefetch
        self.max_steps = max_steps
        self.initial_cache_c2 = initial_cache_c2
        self.initial_cache_c3 = initial_cache_c3
        self.initial_remaining = tuple(sorted(token_dist.items(), key=lambda x: -x[1]))

    def run(self) -> Tuple[int, List[StageAction]]:
        init = BeamState(
            c2=make_initial_snap(self.initial_cache_c2),
            c3=make_initial_snap(self.initial_cache_c3),
            remaining=self.initial_remaining,
            history=(),
            g_score=0,
            f_score=lb_remaining(self.initial_remaining),
        )
        beam: List[BeamState] = [init]
        best_makespan = float("inf")
        best_history: List[StageAction] = []
        seen: Dict[tuple, int] = {}

        for _step in range(self.max_steps):
            if not beam:
                break
            next_candidates: List[BeamState] = []
            layer_best: Dict[tuple, BeamState] = {}

            def add_child(child: BeamState) -> None:
                fp = child.fingerprint()
                prev = layer_best.get(fp)
                if prev is not None and prev.f_score <= child.f_score:
                    return
                if fp in seen and seen[fp] <= child.f_score:
                    return
                layer_best[fp] = child

            for state in beam:
                if not state.remaining:
                    ms = max(state.c2.task_end, state.c3.task_end)
                    if ms < best_makespan:
                        best_makespan = ms
                        best_history = list(state.history)
                    continue

                active_state = state
                if self.enable_prefetch:
                    pc2, pc3 = inject_ghost_prefetch_pair(state.c2, state.c3)
                    if pc2 != state.c2 or pc3 != state.c3:
                        g = max(pc2.task_end, pc3.task_end)
                        active_state = BeamState(
                            c2=pc2,
                            c3=pc3,
                            remaining=state.remaining,
                            history=state.history,
                            g_score=g,
                            f_score=g + lb_remaining(state.remaining),
                        )

                for action in gen_stage_actions(
                    active_state.c2, active_state.c3, active_state.remaining
                ):
                    child = apply_action(active_state, action)
                    add_child(child)

                if self.enable_prefetch:
                    # 用 min 而非 max：让较早进入 S3 的 cluster 立即触发 prefetch，
                    # 不等待较晚的 cluster 也进入 S3。
                    t_now = min(active_state.c2.s2_end, active_state.c3.s2_end)
                    for action in gen_prefetch_actions(
                        active_state.c2,
                        active_state.c3,
                        active_state.remaining,
                        t_now,
                    ):
                        child = apply_action(active_state, action)
                        add_child(child)

            next_candidates = list(layer_best.values())

            if not next_candidates:
                break

            beam = _select_reference_beam(next_candidates, self.beam_width)
            for child in beam:
                fp = child.fingerprint()
                if fp not in seen or child.f_score < seen[fp]:
                    seen[fp] = child.f_score

        return best_makespan, best_history


# ============================================================
#  甘特图输出
# ============================================================


def format_timeline(
    history: List[StageAction],
    token_dist: Dict[int, int],
    makespan: int,
) -> str:
    segs_c2: List[dict] = []
    segs_c3: List[dict] = []
    pf_segs: List[dict] = []

    for act in history:
        if act.c2_eid >= 0:
            sn = FourStageSnap.from_assign(
                act.c2_start,
                act.c2_shape_s1,
                act.c2_shape_s3,
                act.c2_ntok,
                act.c2_eid,
                act.c2_s1_cached,
                act.c2_s3_cached,
                act.c2_s2pf_start,
            )
            segs_c2.append(
                dict(
                    eid=act.c2_eid,
                    ntok=act.c2_ntok,
                    ss1=act.c2_shape_s1,
                    ss3=act.c2_shape_s3,
                    snap=sn,
                    s1c=act.c2_s1_cached,
                    s3c=act.c2_s3_cached,
                )
            )
        if act.c3_eid >= 0:
            sn = FourStageSnap.from_assign(
                act.c3_start,
                act.c3_shape_s1,
                act.c3_shape_s3,
                act.c3_ntok,
                act.c3_eid,
                act.c3_s1_cached,
                act.c3_s3_cached,
                act.c3_s2pf_start,
            )
            segs_c3.append(
                dict(
                    eid=act.c3_eid,
                    ntok=act.c3_ntok,
                    ss1=act.c3_shape_s1,
                    ss3=act.c3_shape_s3,
                    snap=sn,
                    s1c=act.c3_s1_cached,
                    s3c=act.c3_s3_cached,
                )
            )
        if act.pf_cluster > 0:
            pf_segs.append(
                dict(
                    cluster=act.pf_cluster,
                    eid=act.pf_eid,
                    shape=act.pf_shape,
                    start=act.pf_start,
                    end=act.pf_start + act.pf_shape.t_dma_s1,
                )
            )

    events: set = {0, makespan}
    for seg in segs_c2 + segs_c3:
        sn = seg["snap"]
        events |= sn.bw_change_pts()
    for pf in pf_segs:
        events |= {pf["start"], pf["end"]}
    events = sorted(events)

    col_w = 42
    header = (
        f"{'Start':>10}  {'End':>10}  {'Dur':>8}  "
        f"{'xDMA':<{col_w}}{'iDMA':<{col_w}}"
        f"{'C2_VersaCore':<{col_w}}{'C3_VersaCore':<{col_w}}"
    )
    sep = "-" * len(header)
    rows = []

    def seg_at(segs, t):
        for sg in segs:
            sn = sg["snap"]
            if sn.task_start <= t < sn.task_end:
                return sg, sn
        return None, None

    def vc_label(sg, sn, t):
        if sg is None:
            return "idle"
        st = sn.stage_at(t)
        lbl = f"E{sg['eid']}({sg['ntok']}tok,S1={sg['ss1'].name},S3={sg['ss3'].name})"
        if st == ST_S1:
            sfx = "[S1-cache,SwishGLU-cmp]" if sg["s1c"] else "[S1:SwishGLU-fetch+cmp]"
        elif st == ST_S2:
            sfx = "[S2:SwishGLU-compute]"
        elif st == ST_S3:
            sfx = "[S3-cache,Down-cmp]" if sg["s3c"] else "[S3:Down-fetch+compute]"
        elif st == ST_S4:
            sfx = "[S4:Down-compute]"
        else:
            return "idle"
        return f"{lbl}{sfx}"

    for i in range(len(events) - 1):
        t_s = events[i]
        t_e = events[i + 1]
        sg2, sn2 = seg_at(segs_c2, t_s)
        sg3, sn3 = seg_at(segs_c3, t_s)

        # 使用 dma1_end/dma3_end 而非 s1_end/s3_end：
        # ShapeA 的 DMA 在 dma1_end 就完成（BW 提前释放），后半段 [dma1_end, s1_end) 无 DMA。
        bw2s1 = sn2.bw_s1 if (sn2 and sn2.task_start <= t_s < sn2.dma1_end) else 0
        bw2s3 = sn2.bw_s3 if (sn2 and sn2.s2_end <= t_s < sn2.dma3_end) else 0
        bw3s1 = sn3.bw_s1 if (sn3 and sn3.task_start <= t_s < sn3.dma1_end) else 0
        bw3s3 = sn3.bw_s3 if (sn3 and sn3.s2_end <= t_s < sn3.dma3_end) else 0
        pf2 = next(
            (p for p in pf_segs if p["cluster"] == 2 and p["start"] <= t_s < p["end"]),
            None,
        )
        pf3 = next(
            (p for p in pf_segs if p["cluster"] == 3 and p["start"] <= t_s < p["end"]),
            None,
        )

        xp = []
        ip = []
        if bw2s1 == 128:
            xp.append(f"C2-S1:{sg2['ss1'].name}")
            ip.append(f"C2-S1:{sg2['ss1'].name}")
        elif bw2s1:
            xp.append(f"C2-S1:{sg2['ss1'].name}")
        bw2s2pf = sn2.s2pf_bw if (sn2 and sn2.s2pf_start <= t_s < sn2.s2pf_end) else 0
        bw3s2pf = sn3.s2pf_bw if (sn3 and sn3.s2pf_start <= t_s < sn3.s2pf_end) else 0
        if bw2s2pf == 128:
            xp.append(f"C2-S2PF:{sg2['ss3'].name}")
            ip.append(f"C2-S2PF:{sg2['ss3'].name}")
        elif bw2s2pf:
            xp.append(f"C2-S2PF:{sg2['ss3'].name}")
        if bw2s3 == 128:
            xp.append(f"C2-S3:{sg2['ss3'].name}")
            ip.append(f"C2-S3:{sg2['ss3'].name}")
        elif bw2s3:
            xp.append(f"C2-S3:{sg2['ss3'].name}")
        if bw3s1 == 128:
            xp.append(f"C3-S1:{sg3['ss1'].name}")
            ip.append(f"C3-S1:{sg3['ss1'].name}")
        elif bw3s1:
            ip.append(f"C3-S1:{sg3['ss1'].name}")
        if bw3s2pf == 128:
            xp.append(f"C3-S2PF:{sg3['ss3'].name}")
            ip.append(f"C3-S2PF:{sg3['ss3'].name}")
        elif bw3s2pf:
            ip.append(f"C3-S2PF:{sg3['ss3'].name}")
        if bw3s3 == 128:
            xp.append(f"C3-S3:{sg3['ss3'].name}")
            ip.append(f"C3-S3:{sg3['ss3'].name}")
        elif bw3s3:
            ip.append(f"C3-S3:{sg3['ss3'].name}")
        if pf2:
            xp.append(f"C2-PF:E{pf2['eid']}({pf2['shape'].name})")
        if pf3:
            ip.append(f"C3-PF:E{pf3['eid']}({pf3['shape'].name})")

        xdma = ",".join(xp) if xp else "idle"
        idma = ",".join(ip) if ip else "idle"
        vc2 = vc_label(sg2, sn2, t_s)
        vc3 = vc_label(sg3, sn3, t_s)
        dur = t_e - t_s
        rows.append(
            f"{t_s:>10,}  {t_e:>10,}  {dur:>8,}  "
            f"{xdma:<{col_w}}{idma:<{col_w}}{vc2:<{col_w}}{vc3:<{col_w}}"
        )

    bar = "=" * len(header)
    return "\n".join(
        [
            bar,
            f"  四阶段甘特图  [makespan={makespan:,} cc]",
            f"  S1/S3 独立选择 Shape | S2/S4 在 ni>1 时自动存在",
            bar,
            header,
            sep,
            *rows,
            sep,
        ]
    )


# ============================================================
#  效率统计
# ============================================================


def compute_efficiency(history: List[StageAction], makespan: int) -> dict:
    c2_compute = c3_compute = 0
    dma_c2 = dma_c3 = 0
    for act in history:
        if act.c2_eid >= 0:
            sn = FourStageSnap.from_assign(
                act.c2_start,
                act.c2_shape_s1,
                act.c2_shape_s3,
                act.c2_ntok,
                act.c2_eid,
                act.c2_s1_cached,
                act.c2_s3_cached,
                act.c2_s2pf_start,
            )
            c2_compute += sn.task_end - act.c2_start
            if not act.c2_s1_cached:
                dma_c2 += act.c2_shape_s1.t_dma_s1
            if not act.c2_s3_cached:
                dma_c2 += act.c2_shape_s3.t_dma_s3
            if act.c2_s2pf_start >= 0:
                dma_c2 += act.c2_shape_s3.t_dma_s3
        if act.c3_eid >= 0:
            sn = FourStageSnap.from_assign(
                act.c3_start,
                act.c3_shape_s1,
                act.c3_shape_s3,
                act.c3_ntok,
                act.c3_eid,
                act.c3_s1_cached,
                act.c3_s3_cached,
                act.c3_s2pf_start,
            )
            c3_compute += sn.task_end - act.c3_start
            if not act.c3_s1_cached:
                dma_c3 += act.c3_shape_s1.t_dma_s1
            if not act.c3_s3_cached:
                dma_c3 += act.c3_shape_s3.t_dma_s3
            if act.c3_s2pf_start >= 0:
                dma_c3 += act.c3_shape_s3.t_dma_s3
        if act.pf_cluster > 0 and act.pf_shape:
            if act.pf_cluster == 2:
                dma_c2 += act.pf_shape.t_dma_s1
            else:
                dma_c3 += act.pf_shape.t_dma_s1
    ms = makespan or 1
    return dict(
        makespan=makespan,
        c2_vc_util=c2_compute / ms,
        c3_vc_util=c3_compute / ms,
        dma_c2_util=dma_c2 / ms,
        dma_c3_util=dma_c3 / ms,
        c2_compute=c2_compute,
        c3_compute=c3_compute,
        c2_idle=makespan - c2_compute,
        c3_idle=makespan - c3_compute,
    )
