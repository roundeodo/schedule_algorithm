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
      S1 weight DMA 结束后可 prefetch down 权重。若未在 S2 结束前完成，
      S3/S4 会等待 prefetch；这是合法的部分隐藏 action。
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
      down weight DMA 结束后可 prefetch 下一 expert 的 S1 gate/up 权重；
      可与剩余 S3 compute 和 S4 compute 重叠。若未在 compute 结束前完成，
      store/后续 slot 等待 prefetch；这同样是合法 action。
      S4 无前台 DMA 约束，自由选 ShapeC(M2) 最小化尾部浪费。
      时长（非 hit）: ceil(max(0, ntok-M_dim_s3)/2) × T_s3(ShapeC)
      时长（hit）：   ceil(ntok/2) × T_s3(ShapeC)
      down weight DMA 结束后可复用 DMA 通路 → 可 Prefetch 下一 expert

关键时间公式:
    S1 时长    = shape_s1.T_s1                             [未 ready: 整块 DMA+计算理想重叠]
                            或 0                                       [cache hit: S1 完全跳过]
    S2 时长    = ceil(max(0, ntok-M_dim_s1)/2)×T_s1(C)    [未 ready: 处理尾部 token]
                            或 ceil(ntok/2)×T_s1(C)                   [cache hit: 处理全部 token]
  S1+S2 共  = T_s1(shape_s1) + _best_s2_compute(ntok-M_dim_s1) [未 ready]
             = _best_s2_compute(ntok)                                [cache hit]
    S3 时长    = shape_s3.T_s3                             [未 ready: 整块 DMA+计算理想重叠]
                            或 0                                       [cache hit: S3 完全跳过]
    S4 时长    = ceil(max(0, ntok-M_dim_s3)/2)×T_s3(C)    [未 ready: 处理尾部 token]
                            或 ceil(ntok/2)×T_s3(C)                   [cache hit: 处理全部 token]
  S3+S4 共  = T_s3(shape_s3) + _best_s4_compute(ntok-M_dim_s3) [未 ready]
             = _best_s4_compute(ntok)                                [cache hit]

DMA 约束 (精确验证):
  iDMA/xDMA 是两条显式全局 lane，各 64 B/cc；BOTH 同时占两条。
  每个 S1/S3/S2PF/S4PF action 在搜索时确定 lane binding。
  验证方式: 枚举所有实际 DMA 区间的 start/end 变化点逐 lane 检查。

Beam Search:
  beam_width=K → 每步保留 K 个状态展开；reference 建议 K=128/256
  goal = remaining 为空；目标值 = max(C2.task_end, C3.task_end)
  f_score = release-aware compute / critical path / mandatory DMA 的可纳下界
  run_anytime 保留所有仍可能改进 incumbent 的唯一状态并报告最优性 gap

模型范围:
  使用完整 MoE 尺寸：Gate/Up=2048×1408，Down=1408×2048，INT4 weight。
  该 reference 是软件/RTL scheduler 四阶段抽象的上层理想策略集；不包含 DFG 的 token
  gather、output store、kernel launch/config 等实测开销，因此不是 cycle-accurate
  workload simulator。
"""

import heapq
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from itertools import count
from typing import Dict, List, Optional, Tuple

# ============================================================
#  物理常量
# ============================================================

WEIGHT_BYTES_TOTAL = 3 * 2048 * 1408 // 2  # 4,325,376 B  (gate+up+down)
WEIGHT_BYTES_S1 = 2 * 2048 * 1408 // 2  # 2,883,584 B  (SwishGLU: gate+up)
WEIGHT_BYTES_S3 = 1 * 2048 * 1408 // 2  # 1,441,792 B  (Down projection)
MAX_BW = 128  # B/cc
FULL_M_DIM = 2
FULL_BW = 128

PF_EID_GHOST = -2  # post-down-DMA 窗口可预取某人，具体 eid 由后续分配回填
# swiglu_hit 遇到 PF_EID_GHOST 时对任意 eid 返回 True


class DmaBinding(IntEnum):
    NONE = 0
    IDMA = 1
    XDMA = 2
    BOTH = 3


DMA_SINGLE_BINDINGS = (DmaBinding.IDMA, DmaBinding.XDMA)
DMA_BINDINGS = (*DMA_SINGLE_BINDINGS, DmaBinding.BOTH)


def dma_bw(binding: DmaBinding) -> int:
    return 64 * int(binding != DmaBinding.NONE) * (
        2 if binding == DmaBinding.BOTH else 1
    )


def dma_duration(weight_bytes: int, binding: DmaBinding) -> int:
    if binding == DmaBinding.NONE:
        return 0
    return math.ceil(weight_bytes / dma_bw(binding))


def default_dma_binding(shape: "Shape") -> DmaBinding:
    return DmaBinding.BOTH if shape.bw_req > 64 else DmaBinding.IDMA


def dma_name(binding: DmaBinding) -> str:
    return DmaBinding(binding).name


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
        """Legacy default-binding S1 DMA duration.

        ShapeA: alloc=64, t_dma_s1=45,056 cc，T_s1=90,112 cc
                → DMA 在 T_s1/2 时完成，后半段仅计算，BW 已释放。
        ShapeB/C: alloc=bw_req，t_dma_s1=T_s1（DMA 与计算同步结束）。
        Reference action timing uses ``dma_duration`` with its explicit binding;
        this property remains for analytical/lite compatibility only.
        """
        return dma_duration(WEIGHT_BYTES_S1, default_dma_binding(self))

    @property
    def t_dma_s3(self) -> int:
        """S3 DMA 实际搬运时长（alloc-bound，down only）."""
        return dma_duration(WEIGHT_BYTES_S3, default_dma_binding(self))

    @property
    def t_dma(self) -> int:
        """Prefetch 搬运时长（Prefetch 预取 S1 的 gate+up 权重）。"""
        return self.t_dma_s1

    def n_iters(self, ntok: int) -> int:
        return math.ceil(ntok / self.M_dim)

    def T_s1_task(self, ntok: int) -> int:
        """Non-resident S1+S2 latency in the ideal whole-block model."""
        if ntok <= 0:
            return 0
        tail = max(0, ntok - self.M_dim)
        full_compute = math.ceil(WEIGHT_BYTES_S1 / FULL_BW)
        return self.T_s1 + math.ceil(tail / FULL_M_DIM) * full_compute

    def T_s3_task(self, ntok: int) -> int:
        """Non-resident S3+S4 latency in the ideal whole-block model."""
        if ntok <= 0:
            return 0
        tail = max(0, ntok - self.M_dim)
        full_compute = math.ceil(WEIGHT_BYTES_S3 / FULL_BW)
        return self.T_s3 + math.ceil(tail / FULL_M_DIM) * full_compute

    def T_task(self, ntok: int) -> int:
        """单 expert 完整时长（S1+S2+S3+S4）。"""
        return self.T_s1_task(ntok) + self.T_s3_task(ntok)

    def eta(self, ntok: int) -> float:
        return min(1.0, ntok / self.M_dim)


SHAPE_A = Shape("A(M8,bw32)", M_dim=8, bw_req=32)
SHAPE_B = Shape("B(M4,bw64)", M_dim=4, bw_req=64)
SHAPE_C = Shape("C(M2,bw128)", M_dim=2, bw_req=128)
ALL_SHAPES = [SHAPE_A, SHAPE_B, SHAPE_C]
FOREGROUND_S1_SHAPES = (SHAPE_A, SHAPE_B, SHAPE_C)
FOREGROUND_S3_SHAPES = (SHAPE_B, SHAPE_C)
SHAPE_RANK = {shape: i for i, shape in enumerate(ALL_SHAPES)}

# Reference-beam action generation has no expert-rank cap.  Runtime is bounded
# by beam_width; generation removes hardware-illegal, exactly equivalent, or
# analytically dominated choices.
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
      dma1_end   : Stage 1 DMA 搬运结束 (≤ s1_end; ShapeA 时 = task_start + 45,056)
    s1_end     : Stage 1 计算结束
      s2_end     : Stage 2 结束 = Stage 3 开始
      dma3_end   : Stage 3 DMA 搬运结束 (≤ s3_end; ShapeA 时 = s2_end + 22,528)
      s3_end     : Stage 3 计算结束 = Stage 4 开始
      compute_end: Stage 4 计算结束
      task_end   : store 依赖可见的 slot 结束；可因部分隐藏 S4PF 晚于 compute_end

    Compute shape and DMA binding are independent.  Stage duration is
    max(shape compute time, binding transfer time), while lane occupancy ends
    at dma1_end/dma3_end.

    DMA 带宽:
      bw_s1 : Stage 1 DMA 占用 (0 若 cached)
      bw_s3 : Stage 3 DMA 占用

        Prefetch:
            s2pf_start/s2pf_end/s2pf_bw : dma1_end 后预取当前 expert 的 down 权重
            pf_start/pf_end/pf_eid/pf_bw: dma3_end 后预取下一个 expert 的 S1 权重
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
    compute_end: int = -1  # -1 only for legacy/manual idle snapshots
    dma_s1: DmaBinding = DmaBinding.NONE
    dma_s3: DmaBinding = DmaBinding.NONE
    s2pf_dma: DmaBinding = DmaBinding.NONE
    pf_dma: DmaBinding = DmaBinding.NONE

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

    def active_dma_mask_at(self, t: int) -> DmaBinding:
        mask = DmaBinding.NONE
        for binding in self.active_dma_bindings_at(t):
            mask = DmaBinding(mask | binding)
        return mask

    def active_dma_bindings_at(self, t: int) -> Tuple[DmaBinding, ...]:
        bindings: List[DmaBinding] = []
        if self.cur_eid >= 0:
            if self.task_start <= t < self.dma1_end:
                bindings.append(self.dma_s1)
            if self.s2_end <= t < self.dma3_end and self.bw_s3 > 0:
                bindings.append(self.dma_s3)
        if self.pf_start >= 0 and self.pf_start <= t < self.pf_end:
            bindings.append(self.pf_dma)
        if self.s2pf_start >= 0 and self.s2pf_start <= t < self.s2pf_end:
            bindings.append(self.s2pf_dma)
        return tuple(binding for binding in bindings if binding != DmaBinding.NONE)

    def stage_at(self, t: int) -> int:
        if self.cur_eid < 0 or t >= self.task_end:
            return ST_IDLE
        if self.task_start <= t < self.s1_end:
            return ST_S1
        if t < self.s2_end:
            return ST_S2
        if t < self.s3_end:
            return ST_S3
        compute_end = self.compute_end if self.compute_end >= 0 else self.task_end
        return ST_S4 if t < compute_end else ST_IDLE

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
    @lru_cache(maxsize=262_144)
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
        dma_s1: Optional[DmaBinding] = None,
        dma_s3: Optional[DmaBinding] = None,
        s2pf_dma: Optional[DmaBinding] = None,
    ) -> "FourStageSnap":
        resolved_s1_dma = (
            DmaBinding.NONE
            if s1_cached
            else DmaBinding(dma_s1 or default_dma_binding(shape_s1))
        )
        requested_s3_dma = DmaBinding(dma_s3 or default_dma_binding(shape_s3))
        resolved_s2pf_dma = DmaBinding(
            s2pf_dma or requested_s3_dma
        ) if s2pf_start >= 0 else DmaBinding.NONE
        T_h1 = max(
            shape_s1.T_s1,
            dma_duration(WEIGHT_BYTES_S1, resolved_s1_dma),
        )
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
        dma1_end = start + dma_duration(WEIGHT_BYTES_S1, resolved_s1_dma)
        # ── S2 down-weight prefetch ───────────────────────────────────────────
        # The DFG releases S2PF after the final S1 load.  S2PF may finish after
        # S2; in that case the no-op S3 chain waits for both dependencies.
        use_s2pf = s2pf_start >= 0
        if use_s2pf:
            s2pf_end = s2pf_start + dma_duration(
                WEIGHT_BYTES_S3, resolved_s2pf_dma
            )
            if s2pf_start < dma1_end:
                raise ValueError("down prefetch cannot start before dma1_end")
        else:
            s2pf_end = -1
        # ── S3/S4 phase ──────────────────────────────────────────────────────
        # Cache hit (or S2 prefetch) → S3 entirely skipped; S4 computes ALL ntok.
        # Non-hit   → S3 overlap DMA+compute for first M_dim_s3 tokens; S4 tail.
        s3_ready = s3_cached or use_s2pf
        resolved_s3_dma = (
            DmaBinding.NONE if s3_ready else requested_s3_dma
        )
        if s3_ready:
            s4_ready = max(s2_end, s2pf_end) if use_s2pf else s2_end
            s3_end = s4_ready
            compute_end = s4_ready + _best_s4_compute(ntok)
        else:
            T_h3 = max(
                shape_s3.T_s3,
                dma_duration(WEIGHT_BYTES_S3, resolved_s3_dma),
            )
            remaining_s4 = max(0, ntok - shape_s3.M_dim)
            s3_end = s2_end + T_h3
            compute_end = s3_end + _best_s4_compute(remaining_s4)
        # ── DMA end points ────────────────────────────────────────────────────
        dma3_end = (
            s3_end
            if s3_ready
            else s2_end + dma_duration(WEIGHT_BYTES_S3, resolved_s3_dma)
        )
        return cls(
            task_start=start,
            task_end=compute_end,
            dma1_end=dma1_end,
            s1_end=s1_end,
            s2_end=s2_end,
            dma3_end=dma3_end,
            s3_end=s3_end,
            s4_start=s3_end,
            bw_s1=dma_bw(resolved_s1_dma),
            bw_s3=dma_bw(resolved_s3_dma),
            cur_eid=eid,
            pf_start=-1,
            pf_end=-1,
            pf_eid=-1,
            pf_bw=0,
            pf_full=False,
            s2pf_start=s2pf_start,
            s2pf_end=s2pf_end,
            s2pf_bw=dma_bw(resolved_s2pf_dma),
            ntok=ntok,
            compute_end=compute_end,
            dma_s1=resolved_s1_dma,
            dma_s3=resolved_s3_dma,
            s2pf_dma=resolved_s2pf_dma,
        )

    def with_s2_down_prefetch(
        self,
        shape_s3: "Shape",
        s2pf_start: int,
        s2pf_dma: Optional[DmaBinding] = None,
    ) -> "FourStageSnap":
        resolved_s2pf_dma = DmaBinding(
            s2pf_dma or self.dma_s3 or default_dma_binding(shape_s3)
        )
        s2pf_end = s2pf_start + dma_duration(
            WEIGHT_BYTES_S3, resolved_s2pf_dma
        )
        if self.bw_s3 == 0:
            return self
        if s2pf_start < self.dma1_end:
            raise ValueError("down prefetch cannot start before dma1_end")
        # S3 now skipped (weights loaded by S2 prefetch);
        # S4 computes all tokens after both S2 compute and S2PF complete.
        s4_ready = max(self.s2_end, s2pf_end)
        new_compute_end = s4_ready + _best_s4_compute(self.ntok)
        return FourStageSnap(
            task_start=self.task_start,
            task_end=new_compute_end,
            dma1_end=self.dma1_end,
            s1_end=self.s1_end,
            s2_end=self.s2_end,
            dma3_end=s4_ready,
            s3_end=s4_ready,
            s4_start=s4_ready,
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
            s2pf_bw=dma_bw(resolved_s2pf_dma),
            ntok=self.ntok,
            compute_end=new_compute_end,
            dma_s1=self.dma_s1,
            dma_s3=DmaBinding.NONE,
            s2pf_dma=resolved_s2pf_dma,
            pf_dma=self.pf_dma,
        )

    def with_prefetch(
        self,
        pf_eid: int,
        pf_shape: "Shape",
        pf_start: int,
        pf_dma: Optional[DmaBinding] = None,
    ) -> "FourStageSnap":
        resolved_pf_dma = DmaBinding(
            pf_dma or default_dma_binding(pf_shape)
        )
        pf_end = pf_start + dma_duration(WEIGHT_BYTES_S1, resolved_pf_dma)
        compute_end = self.compute_end if self.compute_end >= 0 else self.task_end
        return FourStageSnap(
            task_start=self.task_start,
            task_end=max(self.task_end, pf_end),
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
            pf_end=pf_end,  # Prefetch 预取 S1 权重（gate+up）
            pf_eid=pf_eid,
            pf_bw=dma_bw(resolved_pf_dma),
            pf_full=False,
            s2pf_start=self.s2pf_start,
            s2pf_end=self.s2pf_end,
            s2pf_bw=self.s2pf_bw,
            ntok=self.ntok,
            compute_end=compute_end,
            dma_s1=self.dma_s1,
            dma_s3=self.dma_s3,
            s2pf_dma=self.s2pf_dma,
            pf_dma=resolved_pf_dma,
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
    compute_end=0,
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
        compute_end=0,
    )


# ============================================================
#  BW 约束精确验证
# ============================================================


@lru_cache(maxsize=1_000_000)
def bw_feasible(snap_a: FourStageSnap, snap_b: FourStageSnap) -> bool:
    """Check the two physical global DMA lanes at every interval boundary."""
    for t in _bw_event_points(snap_a) | _bw_event_points(snap_b):
        used = DmaBinding.NONE
        for binding in (
            *snap_a.active_dma_bindings_at(t),
            *snap_b.active_dma_bindings_at(t),
        ):
            if used & binding:
                return False
            used = DmaBinding(used | binding)
    return True


def _bw_event_points(snap: FourStageSnap) -> set:
    """Endpoints of actual DMA intervals, excluding compute-only boundaries."""
    points = set()
    if snap.cur_eid >= 0 and snap.bw_s1 > 0 and snap.task_start < snap.dma1_end:
        points.update((snap.task_start, snap.dma1_end))
    if snap.s2pf_start >= 0 and snap.s2pf_bw > 0:
        points.update((snap.s2pf_start, snap.s2pf_end))
    if snap.cur_eid >= 0 and snap.bw_s3 > 0 and snap.s2_end < snap.dma3_end:
        points.update((snap.s2_end, snap.dma3_end))
    if snap.pf_start >= 0 and snap.pf_bw > 0:
        points.update((snap.pf_start, snap.pf_end))
    return points


@lru_cache(maxsize=1_000_000)
def _s2_down_prefetch_start_candidates(
    snap: FourStageSnap, shape_s3: Shape, peers: Tuple[FourStageSnap, ...] = ()
) -> List[int]:
    """Return the only plan-realizable S2PF placement.

    Lowered tasks carry ``has_s2pf`` but no start timestamp.  The DFG releases
    the S2PF node after the final S1 load, so the ideal model starts it at
    ``dma1_end``.  Different synthetic placements would encode to the same task.
    """
    del peers
    if snap.cur_eid < 0 or snap.bw_s3 == 0:
        return []
    del shape_s3
    return [snap.dma1_end]


def with_optional_s2_down_prefetch(
    snap: FourStageSnap, shape_s3: Shape, peer: Optional[FourStageSnap] = None
) -> FourStageSnap:
    """贪心后级辅助：仅在 S2 内完全隐藏且带宽可行时开启。"""
    peers = (peer,) if peer is not None else ()
    for start in _s2_down_prefetch_start_candidates(snap, shape_s3, peers):
        for binding in DMA_BINDINGS:
            cand = snap.with_s2_down_prefetch(shape_s3, start, binding)
            # Greedy descendants enable only fully hidden S2PF.  The reference
            # search explicitly compares partially hidden variants against OFF.
            if cand.s2pf_end > snap.s2_end:
                continue
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
    starts_a = [-1] + _s2_down_prefetch_start_candidates(
        snap_a, shape_a_s3, (snap_b,)
    )
    starts_b = [-1] + _s2_down_prefetch_start_candidates(
        snap_b, shape_b_s3, (snap_a,)
    )
    best_a, best_b = snap_a, snap_b
    best_score = -1
    best_start_sum = 10**18
    variants_a = [snap_a]
    variants_b = [snap_b]
    for start_a in starts_a[1:]:
        for binding_a in DMA_BINDINGS:
            cand_a = snap_a.with_s2_down_prefetch(
                shape_a_s3, start_a, binding_a
            )
            if cand_a.s2pf_end <= snap_a.s2_end:
                variants_a.append(cand_a)
    for start_b in starts_b[1:]:
        for binding_b in DMA_BINDINGS:
            cand_b = snap_b.with_s2_down_prefetch(
                shape_b_s3, start_b, binding_b
            )
            if cand_b.s2pf_end <= snap_b.s2_end:
                variants_b.append(cand_b)
    for cand_a in variants_a:
        for cand_b in variants_b:
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
    snap: FourStageSnap,
    pf_dma: DmaBinding,
    peers: Tuple[FourStageSnap, ...] = (),
) -> List[int]:
    """Return the only lowering-realizable next-S1 prefetch placement.

    The workload DFG releases S4PF after the final down-weight load and carries
    no programmable start timestamp, so transfer begins at ``dma3_end``.  One
    lane (iDMA or xDMA) gives 64 B/cc; BOTH gives 128 B/cc.
    """
    del peers
    if snap.cur_eid < 0 or snap.pf_eid != -1:
        return []
    if pf_dma not in DMA_BINDINGS:
        return []
    lo = snap.dma3_end
    return [lo]


def with_optional_next_s1_prefetch(
    snap: FourStageSnap,
    next_eid: int,
    peer: Optional[FourStageSnap] = None,
) -> FourStageSnap:
    """贪心后级辅助：仅选择在当前 compute 内完全隐藏的 next-S1 prefetch。"""
    peers = (peer,) if peer is not None else ()
    best: Optional[FourStageSnap] = None
    for pf_dma in DMA_BINDINGS:
        pf_shape = SHAPE_C if pf_dma == DmaBinding.BOTH else SHAPE_A
        for start in _next_s1_prefetch_start_candidates(snap, pf_dma, peers):
            cand = snap.with_prefetch(next_eid, pf_shape, start, pf_dma)
            if cand.pf_end > snap.task_end:
                continue
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
    for pf_dma in DMA_BINDINGS:
        pf_shape = SHAPE_C if pf_dma == DmaBinding.BOTH else SHAPE_A
        for start in _next_s1_prefetch_start_candidates(snap_a, pf_dma, (snap_b,)):
            cand = snap_a.with_prefetch(next_eid, pf_shape, start, pf_dma)
            if cand.pf_end > snap_a.task_end:
                continue
            if bw_feasible(cand, snap_b):
                cand_a.append(cand)
                break
    for pf_dma in DMA_BINDINGS:
        pf_shape = SHAPE_C if pf_dma == DmaBinding.BOTH else SHAPE_A
        for start in _next_s1_prefetch_start_candidates(snap_b, pf_dma, (snap_a,)):
            cand = snap_b.with_prefetch(next_eid, pf_shape, start, pf_dma)
            if cand.pf_end > snap_b.task_end:
                continue
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
    snap: FourStageSnap,
    shape_s3: Shape,
    start: int,
    binding: Optional[DmaBinding] = None,
) -> FourStageSnap:
    if start < 0:
        return snap
    try:
        return snap.with_s2_down_prefetch(shape_s3, start, binding)
    except ValueError:
        return snap


@lru_cache(maxsize=262_144)
def enumerate_s2_down_prefetch_variants(
    snap: FourStageSnap,
    shape_s3: Shape,
    peer: Optional[FourStageSnap] = None,
) -> Tuple[FourStageSnap, ...]:
    """Enumerate the hardware-distinct S2PF off/on choices for one task."""
    peer_snap = peer if peer is not None else IDLE_SNAP
    starts = _s2_down_prefetch_start_candidates(
        snap, shape_s3, (peer_snap,) if peer is not None else ()
    )

    variants: List[FourStageSnap] = []
    seen = set()
    choices = [(-1, DmaBinding.NONE)] + [
        (start, binding) for start in starts for binding in DMA_BINDINGS
    ]
    for start, binding in choices:
        cand = _apply_optional_s2pf_start(snap, shape_s3, start, binding)
        feasible = bw_feasible(cand, peer_snap)
        if not feasible:
            continue
        key = _canonical_snap_pair_key(cand, peer_snap)
        if key in seen:
            continue
        seen.add(key)
        variants.append(cand)

    return tuple(variants)


@lru_cache(maxsize=262_144)
def enumerate_s2_down_prefetch_pair_variants(
    snap_a: FourStageSnap,
    shape_a_s3: Shape,
    snap_b: FourStageSnap,
    shape_b_s3: Shape,
) -> Tuple[Tuple[FourStageSnap, FourStageSnap], ...]:
    """Enumerate raw/A-only/B-only/both S2PF side masks when feasible."""
    starts_a = [-1, *_s2_down_prefetch_start_candidates(snap_a, shape_a_s3, (snap_b,))]
    starts_b = [-1, *_s2_down_prefetch_start_candidates(snap_b, shape_b_s3, (snap_a,))]

    variants: List[Tuple[FourStageSnap, FourStageSnap]] = []
    seen = set()
    choices_a = [(-1, DmaBinding.NONE)] + [
        (start, binding)
        for start in starts_a
        if start >= 0
        for binding in DMA_BINDINGS
    ]
    choices_b = [(-1, DmaBinding.NONE)] + [
        (start, binding)
        for start in starts_b
        if start >= 0
        for binding in DMA_BINDINGS
    ]
    for start_a, binding_a in choices_a:
        cand_a = _apply_optional_s2pf_start(
            snap_a, shape_a_s3, start_a, binding_a
        )
        for start_b, binding_b in choices_b:
            cand_b = _apply_optional_s2pf_start(
                snap_b, shape_b_s3, start_b, binding_b
            )
            if not bw_feasible(cand_a, cand_b):
                continue
            key = _canonical_snap_pair_key(cand_a, cand_b)
            if key in seen:
                continue
            seen.add(key)
            variants.append((cand_a, cand_b))

    return tuple(variants)


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
      - 使用 SHAPE_A（bw=64 B/cc）保守估计，从 dma3_end 开始
      - 只注入一次（pf_eid != -1 时跳过，包括 GHOST 已存在的情况）
    """

    def _try_ghost(snap: FourStageSnap, peer: FourStageSnap) -> FourStageSnap:
        if snap.cur_eid < 0 or snap.pf_eid != -1:
            return snap
        if snap.task_end - snap.dma3_end < SHAPE_A.t_dma_s1:
            return snap
        candidate = snap.with_prefetch(
            PF_EID_GHOST, SHAPE_A, snap.dma3_end, DmaBinding.IDMA
        )
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
    c2_dma_s1: DmaBinding = DmaBinding.NONE
    c2_dma_s3: DmaBinding = DmaBinding.NONE
    c2_s2pf_dma: DmaBinding = DmaBinding.NONE
    c3_dma_s1: DmaBinding = DmaBinding.NONE
    c3_dma_s3: DmaBinding = DmaBinding.NONE
    c3_s2pf_dma: DmaBinding = DmaBinding.NONE
    pf_dma: DmaBinding = DmaBinding.NONE


# ============================================================
#  Beam Search 状态
# ============================================================


def _swap_dma_lane(binding: DmaBinding) -> DmaBinding:
    if binding == DmaBinding.IDMA:
        return DmaBinding.XDMA
    if binding == DmaBinding.XDMA:
        return DmaBinding.IDMA
    return binding


def _snap_future_key(s: FourStageSnap, swap_lanes: bool = False) -> tuple:
    """Fields that can affect any later scheduling decision.

    s1_end/s3_end/s4_start/ntok are retained in the snapshot for reporting, but
    action generation observes DMA intervals, task availability, and residency
    only.  Omitting display-only fields removes behaviorally exact duplicates.
    """
    lane = _swap_dma_lane if swap_lanes else (lambda binding: binding)
    return (
        s.task_start,
        s.task_end,
        s.dma1_end,
        s.s2_end,
        s.dma3_end,
        s.bw_s1,
        s.bw_s3,
        int(lane(s.dma_s1)),
        int(lane(s.dma_s3)),
        int(s.cur_eid >= 0),
        s.pf_start,
        s.pf_end,
        s.pf_eid,
        s.pf_bw,
        int(lane(s.pf_dma)),
        s.pf_full,
        s.s2pf_start,
        s.s2pf_end,
        s.s2pf_bw,
        int(lane(s.s2pf_dma)),
    )


def _canonical_snap_pair_key(c2: FourStageSnap, c3: FourStageSnap) -> tuple:
    """Canonicalize identical clusters and the two identical 64-B/cc DMA lanes."""
    variants = []
    for swap_lanes in (False, True):
        a = _snap_future_key(c2, swap_lanes)
        b = _snap_future_key(c3, swap_lanes)
        variants.append((min(a, b), max(a, b)))
    return min(variants)


def _canonical_s2pf_base_pair_key(
    c2: FourStageSnap, c3: FourStageSnap
) -> tuple:
    """Canonical pair key before S2PF, where ntok still affects S4 recompute."""
    variants = []
    for swap_lanes in (False, True):
        a = (_snap_future_key(c2, swap_lanes), c2.ntok)
        b = (_snap_future_key(c3, swap_lanes), c3.ntok)
        variants.append((min(a, b), max(a, b)))
    return min(variants)


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
        return self.g_score < other.g_score

    def fingerprint(self) -> tuple:
        # C2/C3 have identical timing and BW semantics in this model.  Swapping
        # their complete snapshots cannot change any future action or makespan.
        # Canonicalizing the pair removes only this exact physical symmetry.
        a, b = _canonical_snap_pair_key(self.c2, self.c3)
        return (a, b, self.remaining)


# ============================================================
#  Cost / 可纳下界
# ============================================================


def _best_task_time(ntok: int) -> int:
    """对单个 expert，在无 BW 约束下最优 (shape_s1, shape_s3) 的最短时长。"""
    return min(s.T_s1_task(ntok) for s in ALL_SHAPES) + min(
        s.T_s3_task(ntok) for s in ALL_SHAPES
    )


@lru_cache(maxsize=None)
def _best_concurrent_task_time(ntok: int) -> int:
    """Best per-cluster time when each phase is limited to 64 B/cc.

    Analytical/lite schedulers use this as a continuation estimate.  It is not
    used by the reference beam's admissible lower bound because applying a
    fixed 64-B/cc restriction to all remaining work could overestimate states
    that later obtain 128 B/cc or cache-ready phases.
    """
    concurrent_shapes = tuple(shape for shape in ALL_SHAPES if shape.alloc <= 64)
    return min(shape.T_s1_task(ntok) for shape in concurrent_shapes) + min(
        shape.T_s3_task(ntok) for shape in concurrent_shapes
    )


def lb_remaining(remaining: Tuple[Tuple[int, int], ...]) -> int:
    """Remaining-work lower bound with all DMA/cache/tail costs relaxed.

    ``total_work`` assumes perfect Shape-C utilization across both clusters.
    ``critical_expert`` preserves the S1→S3 dependency of the largest single
    expert while allowing that expert to split perfectly across both clusters.
    Both relax the real problem, so their maximum is still admissible.
    """
    if not remaining:
        return 0
    phase_block = SHAPE_C.T_s1 + SHAPE_C.T_s3
    # Padding is paid independently by each expert.  Combining odd tails from
    # different experts would be physically impossible and made the old bound
    # unnecessarily loose.
    total_blocks = sum(
        math.ceil(ntok / FULL_M_DIM) for _, ntok in remaining
    )
    total_work = math.ceil(total_blocks / 2) * phase_block
    critical_expert = max(
        math.ceil(ntok / (2 * SHAPE_C.M_dim))
        * phase_block
        for _, ntok in remaining
    )
    return max(total_work, critical_expert)


def _release_aware_compute_lb(
    c2_end: int,
    c3_end: int,
    remaining: Tuple[Tuple[int, int], ...],
) -> int:
    """M=2-block relaxation with the two actual cluster release times.

    Every task portion occupies a cluster for an integer number of Shape-C
    token-pair blocks.  An expert with ``n`` tokens contributes
    ``ceil(n / 2)`` blocks.  For any integer ``a`` from zero through that
    count, assigning ``2*a`` tokens to C2 (with the endpoints represented by a
    non-split assignment) distributes exactly ``a`` blocks to C2 and the rest
    to C3.  Relaxing the real SPLIT synchronization therefore leaves only an
    integer two-machine load allocation, not continuously divisible cycles.

    The old cycles/2 bound could place half of a 33,792-cycle M=2 block on each
    cluster.  That is physically impossible and lost up to half a block of
    safe proof strength.  This minimization checks the two integer allocations
    adjacent to the continuous balance point and remains admissible because it
    still removes DMA, precedence, and split-start constraints.
    """
    if not remaining:
        return max(c2_end, c3_end)
    phase_block = SHAPE_C.T_s1 + SHAPE_C.T_s3
    total_blocks = sum(
        math.ceil(ntok / FULL_M_DIM) for _, ntok in remaining
    )

    # Continuous crossing of c2_end + k*P and c3_end + (K-k)*P.
    crossing_num = c3_end - c2_end + total_blocks * phase_block
    crossing_den = 2 * phase_block
    crossing_floor = crossing_num // crossing_den
    crossing_ceil = -(-crossing_num // crossing_den)
    candidates = {
        0,
        total_blocks,
        max(0, min(total_blocks, crossing_floor)),
        max(0, min(total_blocks, crossing_ceil)),
    }
    return min(
        max(
            c2_end + c2_blocks * phase_block,
            c3_end + (total_blocks - c2_blocks) * phase_block,
        )
        for c2_blocks in candidates
    )


def _release_aware_expert_chain_lb(
    c2_end: int,
    c3_end: int,
    remaining: Tuple[Tuple[int, int], ...],
) -> int:
    """Earliest compute-only completion of every individual expert.

    A whole expert can start on the earlier cluster.  A real SPLIT action must
    wait until both clusters are available, after which its M=2 blocks can be
    balanced between them.  Taking the better of those two relaxed choices for
    each expert, then the maximum over experts, is a safe precedence/distribution
    bound.  DMA and all interactions with the other experts are still removed.
    """
    if not remaining:
        return max(c2_end, c3_end)
    early, late = sorted((c2_end, c3_end))
    phase_block = SHAPE_C.T_s1 + SHAPE_C.T_s3
    finishes = []
    for _, ntok in remaining:
        blocks = math.ceil(ntok / FULL_M_DIM)
        whole_finish = early + blocks * phase_block
        split_finish = late + math.ceil(blocks / 2) * phase_block
        finishes.append(min(whole_finish, split_finish))
    return max(finishes)


@lru_cache(maxsize=4096)
def _isolated_task_time_lb(
    ntok: int, s1_cached: bool, s3_cached: bool
) -> int:
    """Best isolated four-stage chain with unconstrained access to BOTH lanes."""
    s1_choices = (SHAPE_C,) if s1_cached else FOREGROUND_S1_SHAPES
    s3_choices = (SHAPE_C,) if s3_cached else FOREGROUND_S3_SHAPES
    best = math.inf
    for shape_s1 in s1_choices:
        for shape_s3 in s3_choices:
            snap = FourStageSnap.from_assign(
                0,
                shape_s1,
                shape_s3,
                ntok,
                0,
                s1_cached,
                s3_cached,
                dma_s1=DmaBinding.NONE if s1_cached else DmaBinding.BOTH,
                dma_s3=DmaBinding.NONE if s3_cached else DmaBinding.BOTH,
            )
            best = min(best, snap.task_end)
            if not s3_cached:
                prefetched = snap.with_s2_down_prefetch(
                    shape_s3, snap.dma1_end, DmaBinding.BOTH
                )
                best = min(best, prefetched.task_end)
    return int(best)


def _critical_expert_chain_lb(
    c2: FourStageSnap,
    c3: FourStageSnap,
    remaining: Tuple[Tuple[int, int], ...],
) -> int:
    """Relaxed longest expert chain, allowing perfect two-cluster splitting.

    S1 is assumed prefetched for free because a preceding task may provide an
    S4PF window.  Down weights are treated as cached only when the state already
    contains a concrete full-residency hit.  Both split halves may use BOTH lanes
    simultaneously in this relaxation, so the bound cannot overestimate.
    """
    if not remaining:
        return 0

    def down_cached(eid: int, snap: FourStageSnap) -> bool:
        return snap.pf_eid == eid and snap.pf_full and snap.pf_end >= 0

    longest = 0
    for eid, ntok in remaining:
        c2_down = down_cached(eid, c2)
        c3_down = down_cached(eid, c3)
        best = min(
            _isolated_task_time_lb(ntok, True, c2_down),
            _isolated_task_time_lb(ntok, True, c3_down),
        )
        for left in range(1, ntok):
            right = ntok - left
            split_time = max(
                _isolated_task_time_lb(left, True, c2_down),
                _isolated_task_time_lb(right, True, c3_down),
            )
            best = min(best, split_time)
        longest = max(longest, best)
    return longest


def _minimum_remaining_dma_bytes(
    c2: FourStageSnap,
    c3: FourStageSnap,
    remaining: Tuple[Tuple[int, int], ...],
) -> int:
    """Relaxed mandatory bytes after assigning at most one expert per cache slot."""
    n_experts = len(remaining)
    if n_experts == 0:
        return 0
    snaps = (c2, c3)
    remaining_eids = {eid for eid, _ in remaining}
    concrete_s1 = {
        s.pf_eid
        for s in snaps
        if s.pf_eid in remaining_eids and s.pf_end >= 0
    }
    ghost_s1 = sum(s.pf_eid == PF_EID_GHOST and s.pf_end >= 0 for s in snaps)
    s1_slots = min(n_experts, len(concrete_s1) + ghost_s1)
    full_slots = len(
        {
            s.pf_eid
            for s in snaps
            if s.pf_eid in remaining_eids and s.pf_end >= 0 and s.pf_full
        }
    )
    return (
        max(0, n_experts - s1_slots) * WEIGHT_BYTES_S1
        + max(0, n_experts - full_slots) * WEIGHT_BYTES_S3
    )


def _earliest_relaxed_dma_release(c2: FourStageSnap, c3: FourStageSnap) -> int:
    """Earliest time at which any not-yet-chosen remaining transfer may start.

    A later search action may still attach next-S1 prefetch to an already
    committed task at dma3_end.  Starting the aggregate DMA relaxation only at
    task_end would therefore overestimate some states and make the LB unsafe.
    """
    releases = [c2.task_end, c3.task_end]
    for snap in (c2, c3):
        if snap.cur_eid >= 0 and snap.pf_eid == -1:
            releases.append(snap.dma3_end)
    return min(releases)


def _dma_capacity_finish_lb(
    c2: FourStageSnap,
    c3: FourStageSnap,
    start: int,
    required_bytes: int,
) -> int:
    """Earliest relaxed finish using free capacity of the two committed lanes.

    Remaining transfers may be divided arbitrarily across both lanes, which is
    more permissive than a real action.  Existing S1/S3/PF reservations are kept,
    so the result remains admissible while being tighter than bytes/128.
    """
    if required_bytes <= 0:
        return start

    points = {start}
    for snap in (c2, c3):
        points.update(t for t in _bw_event_points(snap) if t >= start)
    ordered = sorted(points)
    remaining_bytes = required_bytes

    for left, right in zip(ordered, ordered[1:]):
        if right <= left:
            continue
        used = c2.active_dma_mask_at(left) | c3.active_dma_mask_at(left)
        free_lanes = 2 - int(bool(used & DmaBinding.IDMA)) - int(
            bool(used & DmaBinding.XDMA)
        )
        if free_lanes == 0:
            continue
        rate = 64 * free_lanes
        capacity = (right - left) * rate
        if remaining_bytes <= capacity:
            return left + math.ceil(remaining_bytes / rate)
        remaining_bytes -= capacity

    tail_start = ordered[-1]
    return tail_start + math.ceil(remaining_bytes / MAX_BW)


def state_lower_bound_components(
    c2: FourStageSnap,
    c3: FourStageSnap,
    remaining: Tuple[Tuple[int, int], ...],
) -> Dict[str, int]:
    """Return each admissible final-makespan LB component."""
    earliest = min(c2.task_end, c3.task_end)
    latest = max(c2.task_end, c3.task_end)
    compute_lb = _release_aware_compute_lb(
        c2.task_end, c3.task_end, remaining
    )
    release_chain_lb = _release_aware_expert_chain_lb(
        c2.task_end, c3.task_end, remaining
    )
    chain_lb = earliest + _critical_expert_chain_lb(c2, c3, remaining)
    dma_bytes = _minimum_remaining_dma_bytes(c2, c3, remaining)
    dma_release = _earliest_relaxed_dma_release(c2, c3)
    dma_lb = max(latest, _dma_capacity_finish_lb(c2, c3, dma_release, dma_bytes))
    return {
        "committed_cc": latest,
        "compute_cc": compute_lb,
        "release_expert_chain_cc": release_chain_lb,
        "critical_chain_cc": chain_lb,
        "mandatory_dma_bytes": dma_bytes,
        "dma_release_cc": dma_release,
        "dma_capacity_cc": dma_lb,
        "combined_cc": max(
            latest, compute_lb, release_chain_lb, chain_lb, dma_lb
        ),
    }


def state_lower_bound(
    c2: FourStageSnap,
    c3: FourStageSnap,
    remaining: Tuple[Tuple[int, int], ...],
) -> int:
    """Admissible final-makespan LB: release-aware compute, chain, and DMA."""
    return state_lower_bound_components(c2, c3, remaining)["combined_cc"]


# ============================================================
#  SPLIT 候选集
# ============================================================


@lru_cache(maxsize=4096)
def _token_execution_signature(
    ntok: int, s1_cached: bool, s3_cached: bool
) -> tuple:
    """All timing observables by which one side of a SPLIT can affect the future."""
    if s1_cached:
        s1_profiles = ((0, _best_s2_compute(ntok)),)
    else:
        s1_profiles = tuple(
            sorted(
                {
                    (
                        dma_duration(WEIGHT_BYTES_S1, binding),
                        max(shape.T_s1, dma_duration(WEIGHT_BYTES_S1, binding))
                        + _best_s2_compute(max(0, ntok - shape.M_dim)),
                    )
                    for shape in FOREGROUND_S1_SHAPES
                    for binding in DMA_BINDINGS
                }
            )
        )
    if s3_cached:
        s3_profiles = ((0, _best_s4_compute(ntok)),)
    else:
        s3_profiles = tuple(
            sorted(
                {
                    (
                        dma_duration(WEIGHT_BYTES_S3, binding),
                        max(shape.T_s3, dma_duration(WEIGHT_BYTES_S3, binding))
                        + _best_s4_compute(max(0, ntok - shape.M_dim)),
                    )
                    for shape in FOREGROUND_S3_SHAPES
                    for binding in DMA_BINDINGS
                }
            )
        )
    # S2PF skips S3 and computes every token in S4.
    return (s1_profiles, s3_profiles, _best_s4_compute(ntok))


@lru_cache(maxsize=4096)
def _split_candidates(
    hot_ntok: int,
    c2_s1_cached: bool,
    c3_s1_cached: bool,
    c2_s3_cached: bool,
    c3_s3_cached: bool,
) -> Tuple[int, ...]:
    """One representative for each exactly future-distinct non-empty cut."""
    cuts = []
    seen = set()
    for left in range(1, hot_ntok):
        right = hot_ntok - left
        signature = (
            _token_execution_signature(left, c2_s1_cached, c2_s3_cached),
            _token_execution_signature(right, c3_s1_cached, c3_s3_cached),
        )
        if signature in seen:
            continue
        seen.add(signature)
        cuts.append(left)
    return tuple(cuts)


def _swiglu_hit_for_candidate(eid: int, snap: FourStageSnap, t: int) -> bool:
    if snap.pf_end < 0 or snap.pf_end > t:
        return False
    return snap.pf_eid == PF_EID_GHOST or snap.pf_eid == eid


def _down_hit_for_candidate(eid: int, snap: FourStageSnap, t: int) -> bool:
    return _swiglu_hit_for_candidate(eid, snap, t) and snap.pf_full


def _expert_equivalence_key(
    eid: int,
    ntok: int,
    c2: FourStageSnap,
    c3: FourStageSnap,
    now: int,
) -> tuple:
    """Future-observable properties of a remaining expert.

    Expert IDs are otherwise interchangeable in the timing model.  An ID is
    kept distinct when either cluster names it in a resident/prefetch slot,
    including a transfer that has not completed at ``now``.
    """

    def cluster_key(snap: FourStageSnap) -> tuple:
        named = snap.pf_eid == eid
        return (
            named,
            snap.pf_full if named else False,
            _swiglu_hit_for_candidate(eid, snap, now),
            _down_hit_for_candidate(eid, snap, now),
        )

    return (ntok, cluster_key(c2), cluster_key(c3))


def _reserved_next_eid(snap: FourStageSnap) -> int:
    """Concrete S4PF target that lowering requires as this cluster's next task."""
    if snap.cur_eid >= 0 and snap.pf_eid >= 0 and not snap.pf_full:
        return snap.pf_eid
    return -1


def _representative_expert_indices(
    remaining: Tuple[Tuple[int, int], ...],
    c2: FourStageSnap,
    c3: FourStageSnap,
    now: int,
    multiplicity: int = 1,
) -> List[int]:
    """Up to ``multiplicity`` IDs per exactly equivalent expert class."""
    seen: Dict[tuple, int] = {}
    indices = []
    for i, (eid, ntok) in enumerate(remaining):
        key = _expert_equivalence_key(eid, ntok, c2, c3, now)
        count = seen.get(key, 0)
        if count >= multiplicity:
            continue
        seen[key] = count + 1
        indices.append(i)
    return indices


def _pair_candidate_indices(
    remaining: Tuple[Tuple[int, int], ...],
    c2: FourStageSnap,
    c3: FourStageSnap,
    now: int,
    is_wait: bool,
) -> List[int]:
    """All distinct PAIR expert classes; no hot/rank candidate cap."""
    del is_wait
    return _representative_expert_indices(remaining, c2, c3, now, multiplicity=2)


def _split_expert_indices(
    remaining: Tuple[Tuple[int, int], ...],
    c2: FourStageSnap,
    c3: FourStageSnap,
    now: int,
) -> List[int]:
    """Every distinct expert class with at least two tokens can be split."""
    return [
        i
        for i in _representative_expert_indices(remaining, c2, c3, now)
        if remaining[i][1] >= 2
    ]


# ============================================================
#  合法启动边界
# ============================================================


@lru_cache(maxsize=65_536)
def _start_candidates(
    cluster_end: int,
    cluster: FourStageSnap,
    peer: FourStageSnap,
    ntok: int,
    shape_s1: Shape,
    shape_s3: Shape,
    dma_s1: DmaBinding,
    dma_s3: DmaBinding,
    s1_cached: bool,
    s3_cached: bool,
) -> Tuple[int, ...]:
    """Macro-stage starts for a task on an available cluster.

    A feasibility region changes whenever an endpoint of a new S1/S3/S2PF/S4PF
    interval crosses a peer DMA endpoint.  The earliest start in each region is
    therefore an event-aligned value ``peer_endpoint - local_offset``.
    """
    del cluster
    peer_history_floor = peer.task_start if peer.cur_eid >= 0 else cluster_end
    decision_floor = max(cluster_end, peer_history_floor)
    starts = {decision_floor, max(decision_floor, peer.task_end)}
    peer_releases = set()
    if peer.cur_eid >= 0 and peer.bw_s1 > 0:
        peer_releases.add(peer.dma1_end)
    if peer.s2pf_start >= 0 and peer.s2pf_bw > 0:
        peer_releases.add(peer.s2pf_end)
    if peer.cur_eid >= 0 and peer.bw_s3 > 0:
        peer_releases.add(peer.dma3_end)
    if peer.pf_start >= 0 and peer.pf_bw > 0:
        peer_releases.add(peer.pf_end)

    raw = FourStageSnap.from_assign(
        0,
        shape_s1,
        shape_s3,
        ntok,
        0,
        s1_cached,
        s3_cached,
        dma_s1=dma_s1,
        dma_s3=dma_s3,
    )
    local_variants = [raw]
    if not s3_cached:
        for binding in DMA_BINDINGS:
            local_variants.append(
                raw.with_s2_down_prefetch(shape_s3, raw.dma1_end, binding)
            )

    local_start_offsets = set()
    for snap in local_variants:
        if snap.bw_s1 > 0:
            local_start_offsets.add(snap.task_start)
        if snap.s2pf_start >= 0 and snap.s2pf_bw > 0:
            local_start_offsets.add(snap.s2pf_start)
        if snap.bw_s3 > 0:
            local_start_offsets.add(snap.s2_end)
        # A later action may attach next-S1 prefetch at dma3_end.  Include its
        # start offset so the current task can align that legal window too.
        local_start_offsets.add(snap.dma3_end)

    for peer_release in peer_releases:
        for local_offset in local_start_offsets:
            start = peer_release - local_offset
            if decision_floor <= start <= peer.task_end:
                starts.add(start)
    return tuple(sorted(starts))


# ============================================================
#  动作生成
# ============================================================


@lru_cache(maxsize=4096)
def _phase_profile_choices(
    ntok: int,
    stage: str,
    cached: bool,
) -> Tuple[Tuple[Shape, DmaBinding], ...]:
    """Collapse shape names that produce the same future-observable phase.

    The binding, DMA release, and total phase completion are retained exactly.
    Only profiles identical for this concrete token count are merged before the
    expensive PAIR/SPLIT Cartesian products are formed.
    """
    if cached:
        return ((SHAPE_C, DmaBinding.NONE),)
    if stage == "s1":
        shapes = FOREGROUND_S1_SHAPES
        weight_bytes = WEIGHT_BYTES_S1
        shape_compute = lambda shape: shape.T_s1
        tail_compute = lambda shape: _best_s2_compute(max(0, ntok - shape.M_dim))
    elif stage == "s3":
        shapes = FOREGROUND_S3_SHAPES
        weight_bytes = WEIGHT_BYTES_S3
        shape_compute = lambda shape: shape.T_s3
        tail_compute = lambda shape: _best_s4_compute(max(0, ntok - shape.M_dim))
    else:
        raise ValueError(f"unknown phase {stage!r}")

    choices: List[Tuple[Shape, DmaBinding]] = []
    seen = set()
    for shape in shapes:
        for binding in DMA_BINDINGS:
            transfer = dma_duration(weight_bytes, binding)
            phase_end = max(shape_compute(shape), transfer) + tail_compute(shape)
            key = (int(binding), transfer, phase_end)
            if key in seen:
                continue
            seen.add(key)
            choices.append((shape, binding))
    return tuple(choices)


@lru_cache(maxsize=4096)
def _gen_stage_actions_cached(
    c2: FourStageSnap,
    c3: FourStageSnap,
    remaining: Tuple[Tuple[int, int], ...],
    seed_mode: bool = False,
) -> Tuple[StageAction, ...]:
    """
    生成所有合法的 ASSIGN / WAIT 动作。
    对每种动作枚举 compute shape（S1=A/B/C，S3=B/C）和显式
    IDMA/XDMA/BOTH binding，用 bw_feasible 逐 lane 验证。
    """
    if not remaining:
        return ()

    actions: List[StageAction] = []
    action_state_seen = set()
    base_pair_seen = set()
    t2, t3 = c2.task_end, c3.task_end
    n = len(remaining)

    def phase_choices(
        ntok: int, stage: str, cached: bool
    ) -> Tuple[Tuple[Shape, DmaBinding], ...]:
        choices = _phase_profile_choices(ntok, stage, cached)
        if not seed_mode or cached:
            return choices
        # Seed rollout only constructs an incumbent.  Keep representative
        # single-lane and dual-lane modes here; full OPEN generation below uses
        # every future-distinct shape/binding profile.
        allowed = (
            {(SHAPE_A, DmaBinding.IDMA), (SHAPE_A, DmaBinding.XDMA),
             (SHAPE_B, DmaBinding.IDMA), (SHAPE_B, DmaBinding.XDMA),
             (SHAPE_C, DmaBinding.BOTH)}
            if stage == "s1"
            else {(SHAPE_B, DmaBinding.IDMA), (SHAPE_B, DmaBinding.XDMA),
                  (SHAPE_C, DmaBinding.BOTH)}
        )
        selected = tuple(choice for choice in choices if choice in allowed)
        return selected or choices[:1]

    def append_unique_action(
        out: List[StageAction],
        action: StageAction,
        next_c2: FourStageSnap,
        next_c3: FourStageSnap,
        consumed_eids: Tuple[int, ...],
    ) -> None:
        a, b = _canonical_snap_pair_key(next_c2, next_c3)
        key = (a, b, tuple(sorted(set(consumed_eids))))
        if key in action_state_seen:
            return
        action_state_seen.add(key)
        out.append(action)

    def swiglu_hit(eid: int, snap: FourStageSnap, t: int) -> bool:
        return _swiglu_hit_for_candidate(eid, snap, t)

    def down_hit(eid: int, snap: FourStageSnap, t: int) -> bool:
        return _down_hit_for_candidate(eid, snap, t)

    def concurrent_s1_pairs(
        ntok_a: int, c2_cached: bool, ntok_b: int, c3_cached: bool
    ) -> List[Tuple[Shape, DmaBinding, Shape, DmaBinding]]:
        pairs = []
        choices_a = phase_choices(ntok_a, "s1", c2_cached)
        choices_b = phase_choices(ntok_b, "s1", c3_cached)
        for shape_a, dma_a in choices_a:
            for shape_b, dma_b in choices_b:
                if dma_a & dma_b:
                    continue
                pairs.append((shape_a, dma_a, shape_b, dma_b))
        return pairs

    def seed_expert_indices(indices: List[int], limit: int) -> List[int]:
        if not seed_mode or len(indices) <= limit:
            return indices
        selected = list(indices[:limit])
        named = {c2.pf_eid, c3.pf_eid}
        selected.extend(i for i in indices if remaining[i][0] in named)
        return list(dict.fromkeys(selected))

    def seed_split_cuts(cuts: Tuple[int, ...], ntok: int) -> Tuple[int, ...]:
        if not seed_mode or len(cuts) <= 6:
            return cuts
        targets = {1, 2, ntok // 2, (ntok + 1) // 2, ntok - 2, ntok - 1}
        selected = {
            min(cuts, key=lambda cut: (abs(cut - target), cut))
            for target in targets
            if 0 < target < ntok
        }
        return tuple(sorted(selected))

    def make_pair(
        now,
        eidA,
        ntokA,
        sA1,
        dA1,
        sA3,
        dA3,
        c2_sw,
        eidB,
        ntokB,
        sB1,
        dB1,
        sB3,
        dB3,
        c3_sw,
        tag,
    ):
        c2_dn = down_hit(eidA, c2, now)
        c3_dn = down_hit(eidB, c3, now)
        sna = FourStageSnap.from_assign(
            now, sA1, sA3, ntokA, eidA, c2_sw, c2_dn,
            dma_s1=dA1, dma_s3=dA3,
        )
        snb = FourStageSnap.from_assign(
            now, sB1, sB3, ntokB, eidB, c3_sw, c3_dn,
            dma_s1=dB1, dma_s3=dB3,
        )
        base_a, base_b = _canonical_s2pf_base_pair_key(sna, snb)
        base_key = (base_a, base_b, tuple(sorted({eidA, eidB})))
        if base_key in base_pair_seen:
            return []
        base_pair_seen.add(base_key)
        out = []
        if seed_mode:
            greedy_pair = with_optional_s2_down_prefetch_pair(
                sna, sA3, snb, sB3
            )
            variants = [(sna, snb)]
            if greedy_pair != (sna, snb):
                variants.append(greedy_pair)
        else:
            variants = enumerate_s2_down_prefetch_pair_variants(
                sna, sA3, snb, sB3
            )
        for va, vb in variants:
            if not bw_feasible(va, vb):
                continue
            action = StageAction(
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
                c2_dma_s1=va.dma_s1,
                c2_dma_s3=va.dma_s3,
                c2_s2pf_dma=va.s2pf_dma,
                c3_dma_s1=vb.dma_s1,
                c3_dma_s3=vb.dma_s3,
                c3_s2pf_dma=vb.s2pf_dma,
            )
            append_unique_action(out, action, va, vb, (eidA, eidB))
        return out

    # ─── PAIR & SPLIT（包含 WAIT- 变体）────────────────────────────

    for now_time, is_wait in [(max(t2, t3), t2 != t3)]:
        prefix = "WAIT-" if is_wait else ""
        pair_indices = _pair_candidate_indices(remaining, c2, c3, now_time, is_wait)
        split_indices = _split_expert_indices(remaining, c2, c3, now_time)
        pair_indices = seed_expert_indices(pair_indices, 4)
        split_indices = seed_expert_indices(split_indices, 2)
        reserved_c2 = _reserved_next_eid(c2)
        reserved_c3 = _reserved_next_eid(c3)

        # PAIR
        if n >= 2:
            for i in pair_indices:
                for j in pair_indices:
                    if i == j:
                        continue
                    # At a fully symmetric state, (Ei->C2, Ej->C3) and its
                    # complete cluster swap have identical futures.
                    if c2 == c3 and j < i:
                        continue
                    eidA, ntokA = remaining[i]
                    eidB, ntokB = remaining[j]
                    if reserved_c2 >= 0 and eidA != reserved_c2:
                        continue
                    if reserved_c3 >= 0 and eidB != reserved_c3:
                        continue
                    c2_sw = swiglu_hit(eidA, c2, now_time)
                    c3_sw = swiglu_hit(eidB, c3, now_time)
                    c2_dn = down_hit(eidA, c2, now_time)
                    c3_dn = down_hit(eidB, c3, now_time)
                    s1_pairs = concurrent_s1_pairs(ntokA, c2_sw, ntokB, c3_sw)
                    for sA1, dA1, sB1, dB1 in s1_pairs:
                        for sA3, dA3 in phase_choices(
                            ntokA, "s3", c2_dn
                        ):
                            for sB3, dB3 in phase_choices(
                                ntokB, "s3", c3_dn
                            ):
                                pair_actions = make_pair(
                                    now_time,
                                    eidA,
                                    ntokA,
                                    sA1,
                                    dA1,
                                    sA3,
                                    dA3,
                                    c2_sw,
                                    eidB,
                                    ntokB,
                                    sB1,
                                    dB1,
                                    sB3,
                                    dB3,
                                    c3_sw,
                                    f"{prefix}PAIR({eidA}+{eidB})",
                                )
                                actions.extend(pair_actions)

        # SPLIT every future-distinct expert class at every non-empty cut.
        for split_idx in split_indices:
            split_eid, split_ntok = remaining[split_idx]
            if split_ntok < 2:
                continue
            if reserved_c2 >= 0 and split_eid != reserved_c2:
                continue
            if reserved_c3 >= 0 and split_eid != reserved_c3:
                continue
            c2_sw = swiglu_hit(split_eid, c2, now_time)
            c3_sw = swiglu_hit(split_eid, c3, now_time)
            c2_dn = down_hit(split_eid, c2, now_time)
            c3_dn = down_hit(split_eid, c3, now_time)
            # Token-dependent profile equivalence must be applied after choosing
            # a cut.  This avoids constructing shape products that are identical
            # for that concrete left/right token count.
            split_cuts = _split_candidates(
                split_ntok, c2_sw, c3_sw, c2_dn, c3_dn
            )
            for spA in seed_split_cuts(split_cuts, split_ntok):
                spB = split_ntok - spA
                if spB <= 0:
                    continue
                s1_pairs = concurrent_s1_pairs(spA, c2_sw, spB, c3_sw)
                for sA1, dA1, sB1, dB1 in s1_pairs:
                    for sA3, dA3 in phase_choices(spA, "s3", c2_dn):
                        for sB3, dB3 in phase_choices(spB, "s3", c3_dn):
                            if c2 == c3:
                                left = (
                                    spA, SHAPE_RANK[sA1], int(dA1),
                                    SHAPE_RANK[sA3], int(dA3),
                                )
                                right = (
                                    spB, SHAPE_RANK[sB1], int(dB1),
                                    SHAPE_RANK[sB3], int(dB3),
                                )
                                if left > right:
                                    continue
                            split_actions = make_pair(
                                now_time,
                                split_eid,
                                spA,
                                sA1,
                                dA1,
                                sA3,
                                dA3,
                                c2_sw,
                                split_eid,
                                spB,
                                sB1,
                                dB1,
                                sB3,
                                dB3,
                                c3_sw,
                                f"{prefix}SPLIT(E{split_eid}:{spA},{spB})",
                            )
                            actions.extend(split_actions)

    # ─── SINGLE（较早空闲的 cluster 立即分配）────────────────────────

    def add_single(cluster_id: int, cluster: FourStageSnap, peer: FourStageSnap) -> None:
        cluster_end = cluster.task_end
        cluster_reserved = _reserved_next_eid(cluster)
        peer_reserved = _reserved_next_eid(peer)
        single_indices = _representative_expert_indices(
            remaining, c2, c3, cluster_end
        )
        single_indices = seed_expert_indices(single_indices, 2)
        for idx in single_indices:
            eid, ntok = remaining[idx]
            if cluster_reserved >= 0 and eid != cluster_reserved:
                continue
            if peer_reserved >= 0 and eid == peer_reserved:
                continue
            s1_hit = swiglu_hit(eid, cluster, cluster_end)
            s3_hit = down_hit(eid, cluster, cluster_end)
            for s1, s1_dma in phase_choices(ntok, "s1", s1_hit):
                for s3, s3_dma in phase_choices(ntok, "s3", s3_hit):
                    for start in _start_candidates(
                        cluster_end,
                        cluster,
                        peer,
                        ntok,
                        s1,
                        s3,
                        s1_dma,
                        s3_dma,
                        s1_hit,
                        s3_hit,
                    ):
                        sn0 = FourStageSnap.from_assign(
                            start,
                            s1,
                            s3,
                            ntok,
                            eid,
                            s1_hit,
                            s3_hit,
                            dma_s1=s1_dma,
                            dma_s3=s3_dma,
                        )
                        if seed_mode:
                            greedy_sn = with_optional_s2_down_prefetch(sn0, s3, peer)
                            variants = (sn0,) if greedy_sn == sn0 else (sn0, greedy_sn)
                        else:
                            variants = enumerate_s2_down_prefetch_variants(
                                sn0, s3, peer
                            )
                        for sn in variants:
                            feasible = (
                                bw_feasible(sn, peer)
                                if cluster_id == 2
                                else bw_feasible(peer, sn)
                            )
                            if not feasible:
                                continue
                            if cluster_id == 2:
                                action = StageAction(
                                    c2_eid=eid,
                                    c2_ntok=ntok,
                                    c2_shape_s1=s1,
                                    c2_shape_s3=s3,
                                    c2_start=start,
                                    c2_s1_cached=s1_hit,
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
                                    c2_dma_s1=sn.dma_s1,
                                    c2_dma_s3=sn.dma_s3,
                                    c2_s2pf_dma=sn.s2pf_dma,
                                )
                            else:
                                action = StageAction(
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
                                    c3_s1_cached=s1_hit,
                                    c3_s3_cached=sn.bw_s3 == 0,
                                    pf_cluster=-1,
                                    pf_eid=-1,
                                    pf_shape=None,
                                    pf_start=-1,
                                    tag=f"SINGLE-C3(E{eid})",
                                    c3_s2pf_start=sn.s2pf_start,
                                    c3_dma_s1=sn.dma_s1,
                                    c3_dma_s3=sn.dma_s3,
                                    c3_s2pf_dma=sn.s2pf_dma,
                                )
                            next_c2, next_c3 = (
                                (sn, peer) if cluster_id == 2 else (peer, sn)
                            )
                            append_unique_action(
                                actions, action, next_c2, next_c3, (eid,)
                            )

    if t2 < t3:
        add_single(2, c2, c3)
        if _reserved_next_eid(c3) >= 0:
            add_single(3, c3, c2)
    elif t3 < t2:
        add_single(3, c3, c2)
        if _reserved_next_eid(c2) >= 0:
            add_single(2, c2, c3)
    else:
        add_single(2, c2, c3)
        if c2 != c3:
            add_single(3, c3, c2)

    return tuple(actions)


def gen_stage_actions(
    c2: FourStageSnap,
    c3: FourStageSnap,
    remaining: Tuple[Tuple[int, int], ...],
    *,
    seed_mode: bool = False,
) -> List[StageAction]:
    """Return cached, future-distinct legal stage actions as a mutable list."""
    return list(_gen_stage_actions_cached(c2, c3, remaining, seed_mode))


def gen_prefetch_actions(
    c2: FourStageSnap,
    c3: FourStageSnap,
    remaining: Tuple[Tuple[int, int], ...],
    *,
    seed_mode: bool = False,
) -> List[StageAction]:
    """在 S3+S4 期间生成下一 expert 的 S1 Prefetch 动作（不消耗 remaining）。"""
    if not remaining:
        return []
    pf_actions: List[StageAction] = []
    target_indices = _pair_candidate_indices(
        remaining, c2, c3, min(c2.task_end, c3.task_end), False
    )
    if seed_mode and len(target_indices) > 4:
        named = {c2.pf_eid, c3.pf_eid}
        target_indices = list(target_indices[:4]) + [
            index
            for index in target_indices[4:]
            if remaining[index][0] in named
        ]
        target_indices = list(dict.fromkeys(target_indices))
    prefetch_targets = [remaining[i][0] for i in target_indices]

    for cl, peer, cl_id in [(c2, c3, 2), (c3, c2, 3)]:
        if cl.cur_eid < 0:
            continue
        if cl.pf_eid != -1:
            continue
        for next_eid in prefetch_targets:
            for pf_dma in DMA_BINDINGS:
                pf_shape = SHAPE_C if pf_dma == DmaBinding.BOTH else SHAPE_A
                for pf_start in _next_s1_prefetch_start_candidates(
                    cl, pf_dma, (peer,)
                ):
                    # The peer snapshot no longer contains DMA intervals before
                    # its current task_start.  A retroactive PF in that sealed
                    # history would evade BW validation; the same valid schedule
                    # must add it earlier, before advancing the peer timeline.
                    if peer.cur_eid >= 0 and pf_start < peer.task_start:
                        continue
                    cand = cl.with_prefetch(
                        next_eid, pf_shape, pf_start, pf_dma
                    )
                    if not bw_feasible(cand, peer):
                        continue
                    if cl_id == 2:
                        pf_actions.append(
                            StageAction(
                                c2_eid=-2,
                                c2_ntok=0,
                                c2_shape_s1=pf_shape,
                                c2_shape_s3=None,
                                c2_start=pf_start,
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
                                pf_start=pf_start,
                                tag=f"PF-C2(E{next_eid},{dma_name(pf_dma)})",
                                pf_dma=pf_dma,
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
                                c3_start=pf_start,
                                c3_s1_cached=False,
                                c3_s3_cached=False,
                                pf_cluster=3,
                                pf_eid=next_eid,
                                pf_shape=pf_shape,
                                pf_start=pf_start,
                                tag=f"PF-C3(E{next_eid},{dma_name(pf_dma)})",
                                pf_dma=pf_dma,
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
        new_c2 = c2.with_prefetch(
            action.pf_eid, action.pf_shape, action.pf_start, action.pf_dma
        )
        new_rem = tuple(rem)
        g = max(new_c2.task_end, new_c3.task_end)
        return BeamState(
            c2=new_c2,
            c3=new_c3,
            remaining=new_rem,
            history=state.history + (action,),
            g_score=g,
            # Pathmax preserves every valid ancestor certificate.  A freshly
            # evaluated admissible LB need not be consistent and may decrease
            # after an expansion, but the parent bound remains valid for the
            # child's smaller completion set.
            f_score=max(
                state.f_score,
                state_lower_bound(new_c2, new_c3, new_rem),
            ),
        )
    if action.c3_eid == -2:
        new_c3 = c3.with_prefetch(
            action.pf_eid, action.pf_shape, action.pf_start, action.pf_dma
        )
        new_rem = tuple(rem)
        g = max(new_c2.task_end, new_c3.task_end)
        return BeamState(
            c2=new_c2,
            c3=new_c3,
            remaining=new_rem,
            history=state.history + (action,),
            g_score=g,
            f_score=max(
                state.f_score,
                state_lower_bound(new_c2, new_c3, new_rem),
            ),
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
            action.c2_dma_s1,
            action.c2_dma_s3,
            action.c2_s2pf_dma,
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
            action.c3_dma_s1,
            action.c3_dma_s3,
            action.c3_s2pf_dma,
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
        f_score=max(
            state.f_score,
            state_lower_bound(new_c2, new_c3, new_rem),
        ),
    )


def validate_schedule_history(
    history: Tuple[StageAction, ...],
    token_dist: Dict[int, int],
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
) -> int:
    """Rebuild a complete history and reject hidden temporal/DMA violations."""
    snaps = [make_initial_snap(initial_cache_c2), make_initial_snap(initial_cache_c3)]
    consumed: Dict[int, int] = {}
    lane_intervals: Dict[DmaBinding, List[Tuple[int, int, str]]] = {
        DmaBinding.IDMA: [],
        DmaBinding.XDMA: [],
    }

    def add_interval(start: int, end: int, binding: DmaBinding, label: str) -> None:
        if binding == DmaBinding.NONE or end <= start:
            return
        for lane in DMA_SINGLE_BINDINGS:
            if binding & lane:
                lane_intervals[lane].append((start, end, label))

    def assign(cluster_idx: int, action: StageAction) -> None:
        old = snaps[cluster_idx]
        if cluster_idx == 0:
            eid, ntok, start = action.c2_eid, action.c2_ntok, action.c2_start
            shape_s1, shape_s3 = action.c2_shape_s1, action.c2_shape_s3
            s1_cached, s3_cached = action.c2_s1_cached, action.c2_s3_cached
            s2pf_start = action.c2_s2pf_start
            dma_s1, dma_s3 = action.c2_dma_s1, action.c2_dma_s3
            s2pf_dma = action.c2_s2pf_dma
        else:
            eid, ntok, start = action.c3_eid, action.c3_ntok, action.c3_start
            shape_s1, shape_s3 = action.c3_shape_s1, action.c3_shape_s3
            s1_cached, s3_cached = action.c3_s1_cached, action.c3_s3_cached
            s2pf_start = action.c3_s2pf_start
            dma_s1, dma_s3 = action.c3_dma_s1, action.c3_dma_s3
            s2pf_dma = action.c3_s2pf_dma

        if start < old.task_end:
            raise ValueError(
                f"cluster {cluster_idx + 2} task E{eid} starts before prior task_end"
            )
        expected_s1_hit = _swiglu_hit_for_candidate(eid, old, start)
        if s1_cached != expected_s1_hit:
            raise ValueError(f"E{eid} S1 cache flag does not match resident state")
        if s2pf_start < 0:
            expected_down_hit = _down_hit_for_candidate(eid, old, start)
            if s3_cached != expected_down_hit:
                raise ValueError(f"E{eid} S3 cache flag does not match resident state")

        snap = FourStageSnap.from_assign(
            start,
            shape_s1,
            shape_s3,
            ntok,
            eid,
            s1_cached,
            s3_cached,
            s2pf_start,
            dma_s1,
            dma_s3,
            s2pf_dma,
        )
        add_interval(snap.task_start, snap.dma1_end, snap.dma_s1, f"E{eid}:S1")
        add_interval(snap.s2pf_start, snap.s2pf_end, snap.s2pf_dma, f"E{eid}:S2PF")
        add_interval(snap.s2_end, snap.dma3_end, snap.dma_s3, f"E{eid}:S3")
        snaps[cluster_idx] = snap
        consumed[eid] = consumed.get(eid, 0) + ntok

    for action in history:
        if action.c2_eid >= 0:
            assign(0, action)
        if action.c3_eid >= 0:
            assign(1, action)
        if action.pf_cluster in (2, 3):
            cluster_idx = action.pf_cluster - 2
            snap = snaps[cluster_idx]
            if snap.cur_eid < 0 or snap.pf_eid != -1:
                raise ValueError("S4PF does not attach to an unprefetched task")
            if action.pf_start != snap.dma3_end:
                raise ValueError("S4PF must start at dma3_end")
            add_interval(
                action.pf_start,
                action.pf_start + dma_duration(WEIGHT_BYTES_S1, action.pf_dma),
                action.pf_dma,
                f"E{action.pf_eid}:S4PF",
            )
            snaps[cluster_idx] = snap.with_prefetch(
                action.pf_eid, action.pf_shape, action.pf_start, action.pf_dma
            )

    expected = {eid: ntok for eid, ntok in token_dist.items() if ntok > 0}
    if consumed != expected:
        raise ValueError(f"history token coverage mismatch: {consumed} != {expected}")

    for lane, intervals in lane_intervals.items():
        intervals.sort()
        for previous, current in zip(intervals, intervals[1:]):
            if current[0] < previous[1]:
                raise ValueError(
                    f"{dma_name(lane)} overlap: {previous[2]} {previous[:2]} and "
                    f"{current[2]} {current[:2]}"
                )
    return max(snaps[0].task_end, snaps[1].task_end)


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


def _split_cut_class(action: StageAction) -> str:
    left, right = action.c2_ntok, action.c3_ntok
    small, total = min(left, right), left + right
    if small == 1:
        return "FRONT1"
    if small == 2:
        return "FRONT2"
    if abs(left - right) <= 2:
        return "BALANCED"
    if small * 4 <= total:
        return "EDGE"
    return "MID"


def _split_diversity_key(state: BeamState) -> tuple:
    action = _last_action(state)
    assert action is not None
    s1_pair = tuple(
        sorted(
            (
                SHAPE_RANK[action.c2_shape_s1],
                SHAPE_RANK[action.c3_shape_s1],
            )
        )
    )
    cache_pattern = tuple(sorted((action.c2_s1_cached, action.c3_s1_cached)))
    dma_pattern = tuple(
        sorted((int(action.c2_dma_s1), int(action.c3_dma_s1)))
    )
    return (
        action.c2_eid,
        _split_cut_class(action),
        s1_pair,
        dma_pattern,
        cache_pattern,
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
    # worse immediate f-score but better downstream alignment.  SPLIT receives
    # one best representative per cut/S1-shape class so balanced and front-M2
    # states cannot be crowded out by many equivalent offsets.
    if family == "SPLIT":
        grouped: Dict[tuple, BeamState] = {}
        for cand in family_items:
            key = _split_diversity_key(cand)
            prev = grouped.get(key)
            if prev is None or (cand.f_score, cand.g_score) < (
                prev.f_score,
                prev.g_score,
            ):
                grouped[key] = cand
        class_rank = {
            "BALANCED": 0,
            "FRONT2": 1,
            "FRONT1": 2,
            "MID": 3,
            "EDGE": 4,
        }
        semantic = sorted(
            grouped.values(),
            key=lambda s: (
                class_rank[_split_cut_class(_last_action(s))],
                s.f_score,
                s.g_score,
            ),
        )
    else:
        semantic = sorted(
            family_items,
            key=lambda s: (
                -_action_hit_score(_last_action(s)),
                -_action_work_score(_last_action(s)),
                s.f_score,
                s.g_score,
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


def _lpt_completion_estimate(state: BeamState) -> int:
    """Original non-cached LPT estimate, retained as a stable rollout lane."""
    ends = [state.c2.task_end, state.c3.task_end]
    durations = sorted(
        (_best_task_time(ntok) for _, ntok in state.remaining), reverse=True
    )
    for duration in durations:
        idx = 0 if ends[0] <= ends[1] else 1
        ends[idx] += duration
    return max(state.f_score, max(ends))


def _cache_aware_completion_estimate(state: BeamState) -> int:
    """Cache-aware list-scheduling estimate."""
    machines = [
        [state.c2.task_end, state.c2],
        [state.c3.task_end, state.c3],
    ]
    ordered = sorted(state.remaining, key=lambda item: (-item[1], item[0]))
    for eid, ntok in ordered:
        finishes = []
        for end, snap in machines:
            if snap is None:
                s1_cached = False
                s3_cached = False
            else:
                s1_cached = _swiglu_hit_for_candidate(eid, snap, end)
                s3_cached = _down_hit_for_candidate(eid, snap, end)
            finishes.append(
                end + _isolated_task_time_lb(ntok, s1_cached, s3_cached)
            )
        idx = 0 if finishes[0] <= finishes[1] else 1
        machines[idx][0] = finishes[idx]
        # The current model schedules each expert once.  After this estimated
        # assignment, the old concrete prefetch/residency on that cluster is gone.
        machines[idx][1] = None
    ends = [machines[0][0], machines[1][0]]
    return max(state.f_score, max(ends))


def completion_estimate(state: BeamState) -> int:
    """Dual-lane non-admissible estimate used only to order OPEN."""
    return min(
        _lpt_completion_estimate(state),
        _cache_aware_completion_estimate(state),
    )


@dataclass(frozen=True)
class AnytimeSearchResult:
    makespan: int
    history: Tuple[StageAction, ...]
    lower_bound: int
    optimality_gap: float
    proven_optimal: bool
    expansions: int
    generated: int
    pruned_by_bound: int
    runtime_s: float
    termination: str


# ============================================================
#  Beam Search 主体
# ============================================================


class FourStageScheduler:
    """
    四阶段 Beam Search 调度器（多步 look-ahead，非贪心）。

    beam_width controls retained states only; 128-256 is recommended for the
    reference run.  Every retained state expands the same complete action set.
    全程枚举 shape 与显式 DMA binding，逐 lane 约束精确验证。
    """

    def __init__(
        self,
        token_dist: Dict[int, int],
        beam_width: int = 64,
        enable_prefetch: bool = True,
        max_steps: Optional[int] = None,
        initial_cache_c2: int = -1,
        initial_cache_c3: int = -1,
    ):
        """
        Parameters
        ----------
        token_dist        : {expert_id: token_count} 当前轮的 top-K 路由结果
        beam_width        : beam search 宽度（reference 推荐 128-256）
        enable_prefetch   : 是否允许 Stage-4 期间触发 prefetch
        initial_cache_c2  : C2 在调度开始前 SRAM 中已缓存的 expert ID（-1 表示空）
        initial_cache_c3  : C3 在调度开始前 SRAM 中已缓存的 expert ID（-1 表示空）

        注意: 两个 cluster 有独立 SRAM，因此允许缓存同一个 expert。被缓存的
        expert 不在 token_dist 中时，本轮不会产生 hit，但状态本身仍合法。
        """
        if beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if any(ntok < 0 for ntok in token_dist.values()):
            raise ValueError("token counts must be non-negative")
        self.token_dist = {eid: ntok for eid, ntok in token_dist.items() if ntok > 0}
        self.beam_width = beam_width
        self.enable_prefetch = enable_prefetch
        self.initial_cache_c2 = initial_cache_c2
        self.initial_cache_c3 = initial_cache_c3
        self.initial_remaining = tuple(
            sorted(self.token_dist.items(), key=lambda x: (-x[1], x[0]))
        )
        # One expert assignment plus at most one prefetch action per resulting
        # cluster task is below 4*N even when every expert is split.
        self.max_steps = max_steps if max_steps is not None else 4 * len(self.initial_remaining) + 4

    def _initial_state(self) -> BeamState:
        c2 = make_initial_snap(self.initial_cache_c2)
        c3 = make_initial_snap(self.initial_cache_c3)
        return BeamState(
            c2=c2,
            c3=c3,
            remaining=self.initial_remaining,
            history=(),
            g_score=max(c2.task_end, c3.task_end),
            f_score=state_lower_bound(c2, c3, self.initial_remaining),
        )

    def _validated_incumbent_state(
        self, history: Tuple[StageAction, ...]
    ) -> BeamState:
        """Replay a prior complete schedule for use as an anytime incumbent."""
        validated_makespan = validate_schedule_history(
            history,
            self.token_dist,
            initial_cache_c2=self.initial_cache_c2,
            initial_cache_c3=self.initial_cache_c3,
        )
        state = self._initial_state()
        for action in history:
            state = apply_action(state, action)
        if state.remaining:
            raise ValueError("incumbent history does not consume every expert")
        if state.g_score != validated_makespan:
            raise ValueError(
                "incumbent replay makespan "
                f"{state.g_score} != validated {validated_makespan}"
            )
        return state

    def _greedy_incumbent(
        self,
        initial: BeamState,
        target_gap: Optional[float] = None,
    ) -> BeamState:
        """Build stage-only and prefetch-aware feasible rollouts; keep the best."""

        def rollout(
            allow_prefetch: bool,
            rank_fn,
            seed_mode: bool,
        ) -> Optional[BeamState]:
            """Return a complete greedy rollout, or ``None`` for a dead end.

            Prefetch reservations are legitimate search choices but a greedy
            rollout can reserve both DMA lanes in a way that leaves no legal
            next action.  Such a dead end only invalidates that optional
            rollout; it must not discard an already constructed stage-only
            incumbent.
            """
            state = initial
            while state.remaining:
                best_child: Optional[BeamState] = None
                actions = gen_stage_actions(
                    state.c2,
                    state.c3,
                    state.remaining,
                    seed_mode=seed_mode,
                )
                if allow_prefetch:
                    actions += gen_prefetch_actions(
                        state.c2, state.c3, state.remaining, seed_mode=seed_mode
                    )
                for action in actions:
                    child = apply_action(state, action)
                    if best_child is None or (
                        rank_fn(child), child.f_score, child.g_score
                    ) < (
                        rank_fn(best_child),
                        best_child.f_score,
                        best_child.g_score,
                    ):
                        best_child = child
                if best_child is None:
                    return None
                state = best_child
            return state

        stage_only = rollout(False, _lpt_completion_estimate, True)
        if stage_only is None:
            raise RuntimeError(
                "no legal stage-only action while constructing initial incumbent"
            )
        candidates = [stage_only]
        # Full rollout remains affordable for small active sets and preserves the
        # strong incumbent quality seen in the original reference experiments.
        if len(initial.remaining) <= 8:
            full_stage_only = rollout(False, _lpt_completion_estimate, False)
            if full_stage_only is not None:
                candidates.append(full_stage_only)
        best = min(candidates, key=lambda state: state.g_score)
        certified_gap = (
            0.0
            if initial.f_score >= best.g_score
            else (best.g_score - initial.f_score) / initial.f_score
        )
        if (
            not self.enable_prefetch
            or (target_gap is not None and certified_gap <= target_gap)
        ):
            return best
        prefetch_aware = rollout(True, _cache_aware_completion_estimate, True)
        if prefetch_aware is not None:
            candidates.append(prefetch_aware)
        return min(candidates, key=lambda state: state.g_score)

    def run(self) -> Tuple[int, List[StageAction]]:
        init = self._initial_state()
        beam: List[BeamState] = [init]
        best_makespan = float("inf")
        best_history: List[StageAction] = []
        seen: Dict[tuple, int] = {}

        for _step in range(self.max_steps):
            if not beam:
                break
            layer_best: Dict[tuple, BeamState] = {}

            def add_child(child: BeamState) -> None:
                nonlocal best_makespan, best_history
                if not child.remaining:
                    if child.g_score < best_makespan:
                        best_makespan = child.g_score
                        best_history = list(child.history)
                    return
                if child.f_score >= best_makespan:
                    return
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
                if state.f_score >= best_makespan:
                    continue

                for action in gen_stage_actions(
                    state.c2, state.c3, state.remaining
                ):
                    child = apply_action(state, action)
                    add_child(child)

                if self.enable_prefetch:
                    for action in gen_prefetch_actions(
                        state.c2,
                        state.c3,
                        state.remaining,
                    ):
                        child = apply_action(state, action)
                        add_child(child)

            next_candidates = list(layer_best.values())

            if not next_candidates:
                break

            beam = _select_reference_beam(next_candidates, self.beam_width)
            for child in beam:
                fp = child.fingerprint()
                if fp not in seen or child.f_score < seen[fp]:
                    seen[fp] = child.f_score

        if math.isinf(best_makespan):
            raise RuntimeError(
                f"beam search found no complete schedule within {self.max_steps} steps"
            )
        return int(best_makespan), best_history

    def run_anytime(
        self,
        *,
        time_limit_s: Optional[float] = None,
        max_expansions: Optional[int] = 10_000,
        target_gap: Optional[float] = None,
        incumbent_history: Optional[Tuple[StageAction, ...]] = None,
        initial_state: Optional[BeamState] = None,
        incumbent_state: Optional[BeamState] = None,
    ) -> AnytimeSearchResult:
        """Anytime best-first branch-and-bound with a reportable optimality gap.

        Every unique state with ``LB < incumbent`` remains in OPEN.  The
        non-admissible completion estimate changes expansion order only; safe
        pruning and the final certificate use ``f_score`` exclusively.

        ``initial_state`` permits certified continuation search from an
        already replayed intermediate state.  Its history, absolute times and
        pathmax bound are preserved.  A root ``incumbent_history`` and an
        intermediate state are mutually exclusive because they describe two
        different search roots.
        """
        if time_limit_s is not None and time_limit_s <= 0:
            raise ValueError("time_limit_s must be positive")
        if max_expansions is not None and max_expansions <= 0:
            raise ValueError("max_expansions must be positive")
        if target_gap is not None and target_gap < 0:
            raise ValueError("target_gap must be non-negative")

        if initial_state is not None and incumbent_history is not None:
            raise ValueError(
                "initial_state and incumbent_history are mutually exclusive"
            )
        if incumbent_history is not None and incumbent_state is not None:
            raise ValueError(
                "incumbent_history and incumbent_state are mutually exclusive"
            )

        started = time.perf_counter()
        initial = initial_state if initial_state is not None else self._initial_state()
        # A follow-up pass embeds its best validated schedule.  Reusing it
        # avoids rebuilding the same greedy rollout and immediately exposes any
        # certificate gain from a stronger LB.  First-pass callers have no
        # embedded history and retain the original greedy construction.
        if incumbent_state is not None:
            if incumbent_state.remaining:
                raise ValueError("incumbent_state must be terminal")
            if incumbent_state.g_score != max(
                incumbent_state.c2.task_end, incumbent_state.c3.task_end
            ):
                raise ValueError("incumbent_state makespan is inconsistent")
            prefix = tuple(initial.history)
            if tuple(incumbent_state.history[: len(prefix)]) != prefix:
                raise ValueError("incumbent_state does not extend initial_state")
            incumbent = incumbent_state
        elif incumbent_history is not None:
            incumbent = self._validated_incumbent_state(incumbent_history)
        else:
            incumbent = self._greedy_incumbent(initial, target_gap)
        best_makespan = incumbent.g_score
        best_history = incumbent.history

        rank_heap: List[tuple] = []
        lb_heap: List[tuple] = []
        active_entries: set = set()
        open_by_fp: Dict[tuple, Tuple[int, int]] = {}
        closed_best: Dict[tuple, int] = {}
        serial = count()

        def push(state: BeamState) -> bool:
            if state.f_score >= best_makespan:
                return False
            fp = state.fingerprint()
            closed_lb = closed_best.get(fp)
            if closed_lb is not None and closed_lb <= state.f_score:
                return False
            previous = open_by_fp.get(fp)
            if previous is not None and previous[0] <= state.f_score:
                return False
            if previous is not None:
                active_entries.discard(previous[1])
            entry_id = next(serial)
            open_by_fp[fp] = (state.f_score, entry_id)
            active_entries.add(entry_id)
            estimate = completion_estimate(state)
            heapq.heappush(
                rank_heap,
                (estimate, state.f_score, state.g_score, entry_id, state),
            )
            heapq.heappush(lb_heap, (state.f_score, entry_id))
            return True

        push(initial)
        expansions = 0
        generated = 0
        pruned_by_bound = 0
        termination = "open_exhausted"

        while rank_heap:
            while lb_heap and lb_heap[0][1] not in active_entries:
                heapq.heappop(lb_heap)
            certified_lb = lb_heap[0][0] if lb_heap else best_makespan
            certified_gap = (
                0.0
                if certified_lb >= best_makespan
                else (best_makespan - certified_lb) / certified_lb
            )
            if target_gap is not None and certified_gap <= target_gap:
                termination = "target_gap"
                break

            elapsed = time.perf_counter() - started
            if time_limit_s is not None and elapsed >= time_limit_s:
                termination = "time_limit"
                break
            if max_expansions is not None and expansions >= max_expansions:
                termination = "expansion_limit"
                break

            while rank_heap and rank_heap[0][3] not in active_entries:
                heapq.heappop(rank_heap)
            if not rank_heap:
                break
            _, _, _, entry_id, state = heapq.heappop(rank_heap)
            active_entries.discard(entry_id)
            fp = state.fingerprint()
            current_open = open_by_fp.get(fp)
            if current_open is not None and current_open[1] == entry_id:
                del open_by_fp[fp]

            if state.f_score >= best_makespan:
                pruned_by_bound += 1
                continue
            closed_best[fp] = state.f_score
            expansions += 1

            actions = gen_stage_actions(state.c2, state.c3, state.remaining)
            if self.enable_prefetch:
                actions += gen_prefetch_actions(
                    state.c2, state.c3, state.remaining
                )
            generated += len(actions)

            for action in actions:
                child = apply_action(state, action)
                if not child.remaining:
                    if child.g_score < best_makespan:
                        best_makespan = child.g_score
                        best_history = child.history
                    continue
                if child.f_score >= best_makespan:
                    pruned_by_bound += 1
                    continue
                push(child)

        while lb_heap and lb_heap[0][1] not in active_entries:
            heapq.heappop(lb_heap)
        open_lb = lb_heap[0][0] if lb_heap else best_makespan
        lower_bound = min(best_makespan, open_lb)
        proven_optimal = not lb_heap or open_lb >= best_makespan
        if proven_optimal:
            lower_bound = best_makespan
            termination = "optimal"
        gap = (
            0.0
            if proven_optimal or lower_bound == 0
            else (best_makespan - lower_bound) / lower_bound
        )
        return AnytimeSearchResult(
            makespan=int(best_makespan),
            history=best_history,
            lower_bound=int(lower_bound),
            optimality_gap=gap,
            proven_optimal=proven_optimal,
            expansions=expansions,
            generated=generated,
            pruned_by_bound=pruned_by_bound,
            runtime_s=time.perf_counter() - started,
            termination=termination,
        )


def clear_scheduler_caches() -> None:
    """Release per-case memoized states when worker processes are reused."""
    FourStageSnap.from_assign.cache_clear()
    bw_feasible.cache_clear()
    _s2_down_prefetch_start_candidates.cache_clear()
    _next_s1_prefetch_start_candidates.cache_clear()
    _start_candidates.cache_clear()
    enumerate_s2_down_prefetch_variants.cache_clear()
    enumerate_s2_down_prefetch_pair_variants.cache_clear()
    _best_concurrent_task_time.cache_clear()
    _isolated_task_time_lb.cache_clear()
    _token_execution_signature.cache_clear()
    _split_candidates.cache_clear()
    _phase_profile_choices.cache_clear()
    _gen_stage_actions_cached.cache_clear()


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
                act.c2_dma_s1,
                act.c2_dma_s3,
                act.c2_s2pf_dma,
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
                act.c3_dma_s1,
                act.c3_dma_s3,
                act.c3_s2pf_dma,
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
                    dma=act.pf_dma,
                    start=act.pf_start,
                    end=act.pf_start
                    + dma_duration(WEIGHT_BYTES_S1, act.pf_dma),
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
        d2s1 = (
            sn2.dma_s1
            if sn2 and sn2.task_start <= t_s < sn2.dma1_end
            else DmaBinding.NONE
        )
        d2s3 = (
            sn2.dma_s3
            if sn2 and sn2.s2_end <= t_s < sn2.dma3_end
            else DmaBinding.NONE
        )
        d3s1 = (
            sn3.dma_s1
            if sn3 and sn3.task_start <= t_s < sn3.dma1_end
            else DmaBinding.NONE
        )
        d3s3 = (
            sn3.dma_s3
            if sn3 and sn3.s2_end <= t_s < sn3.dma3_end
            else DmaBinding.NONE
        )
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
        d2s2pf = (
            sn2.s2pf_dma
            if sn2 and sn2.s2pf_start <= t_s < sn2.s2pf_end
            else DmaBinding.NONE
        )
        d3s2pf = (
            sn3.s2pf_dma
            if sn3 and sn3.s2pf_start <= t_s < sn3.s2pf_end
            else DmaBinding.NONE
        )

        def add_dma(label: str, binding: DmaBinding) -> None:
            if binding & DmaBinding.XDMA:
                xp.append(label)
            if binding & DmaBinding.IDMA:
                ip.append(label)

        if sg2:
            add_dma(f"C2-S1:{sg2['ss1'].name}", d2s1)
            add_dma(f"C2-S2PF:{sg2['ss3'].name}", d2s2pf)
            add_dma(f"C2-S3:{sg2['ss3'].name}", d2s3)
        if sg3:
            add_dma(f"C3-S1:{sg3['ss1'].name}", d3s1)
            add_dma(f"C3-S2PF:{sg3['ss3'].name}", d3s2pf)
            add_dma(f"C3-S3:{sg3['ss3'].name}", d3s3)
        if pf2:
            add_dma(f"C2-PF:E{pf2['eid']}", pf2["dma"])
        if pf3:
            add_dma(f"C3-PF:E{pf3['eid']}", pf3["dma"])

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
                act.c2_dma_s1,
                act.c2_dma_s3,
                act.c2_s2pf_dma,
            )
            c2_compute += sn.task_end - act.c2_start
            if not act.c2_s1_cached:
                dma_c2 += dma_duration(WEIGHT_BYTES_S1, act.c2_dma_s1)
            if not act.c2_s3_cached:
                dma_c2 += dma_duration(WEIGHT_BYTES_S3, act.c2_dma_s3)
            if act.c2_s2pf_start >= 0:
                dma_c2 += dma_duration(WEIGHT_BYTES_S3, act.c2_s2pf_dma)
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
                act.c3_dma_s1,
                act.c3_dma_s3,
                act.c3_s2pf_dma,
            )
            c3_compute += sn.task_end - act.c3_start
            if not act.c3_s1_cached:
                dma_c3 += dma_duration(WEIGHT_BYTES_S1, act.c3_dma_s1)
            if not act.c3_s3_cached:
                dma_c3 += dma_duration(WEIGHT_BYTES_S3, act.c3_dma_s3)
            if act.c3_s2pf_start >= 0:
                dma_c3 += dma_duration(WEIGHT_BYTES_S3, act.c3_s2pf_dma)
        if act.pf_cluster > 0 and act.pf_shape:
            if act.pf_cluster == 2:
                dma_c2 += dma_duration(WEIGHT_BYTES_S1, act.pf_dma)
            else:
                dma_c3 += dma_duration(WEIGHT_BYTES_S1, act.pf_dma)
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
