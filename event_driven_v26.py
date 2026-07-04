#!/usr/bin/env python3
"""
事件驱动双核调度器 v26
=======================================================
严格按用户规范实现，核心修正:

【修正1】η 语义 = min(1, M_i/M_dim)，但计算时间始终 = ceil(M_i/M_dim)×T_iter.
  1 tok + 8x8x8 (M_dim=8): T_iter=W/32, tiles=1, T_task=T_iter, η=1/8=0.125.
  -- 不是说"只算1行就快8倍"，而是说"搬了整轮权重，VC 7行空转，效率 12.5%"
  -- 强制规则: M=1 只许用 Shape_C (M_dim=2, BW=128), η=0.5, 比 Shape_A 的 0.125 高 4×.

【修正2】W4A8 (INT4 权重 INT8 激活): wpe=0.5, expert 总权重 = 3×2048×1408×0.5 = 4,325,376 B.
  T_iter_A = 4325376/32  = 135,168 cc  (Shape_A, BW=32)
  T_iter_B = 4325376/64  =  67,584 cc  (Shape_B, BW=64)
  T_iter_C = 4325376/128 =  33,792 cc  (Shape_C, BW=128)

【修正3】物理 DMA 通道: xDMA 64B/cc + iDMA 64B/cc, 两条完全独立通道.
  每条通道同一时刻只能被一个 cluster 占用; 两条可以同时工作.
  alloc_bw(shape) = 64 if bw_req≤64 else 128  (占 1 条 or 2 条通道)
  约束: alloc_bw(C2) + alloc_bw(C3) ≤ 128 (fetch 阶段才计入)

  关键推论:
    C2(Shape_B, xDMA=64) 运行时 iDMA 空闲 → C3 可立即用 iDMA 以 Shape_B 速度搬运.
    不需要 stagger 等待; M=1 不强制 Shape_C, BW 受限时降级到 Shape_B.
    Shape_C(alloc=128) 独占两条通道, 此时另一 cluster 无 DMA 可用.

形状空间:
  Shape_A: 8x8x8   M_dim=8  BW_req=32 B/cc
  Shape_B: 4x8x16  M_dim=4  BW_req=64 B/cc
  Shape_C: 2x8x32  M_dim=2  BW_req=128 B/cc
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from copy import deepcopy

# ============================================================
#  常量
# ============================================================

WEIGHT_BYTES = 3 * 2048 * 1408 // 2  # 4,325,376 B (W4A8, per expert)
MAX_BW = 128  # B/cc 全局 DMA 上限


@dataclass(frozen=True)
class Shape:
    name: str
    M_dim: int
    bw_req: int  # 该 shape 在 fetch 阶段的带宽需求 (B/cc)

    @property
    def T_iter(self) -> int:
        """搬运一次完整权重所需 cc."""
        return math.ceil(WEIGHT_BYTES / self.bw_req)

    def T_fetch(self, cached: bool) -> int:
        return 0 if cached else self.T_iter

    def T_task(self, ntok: int, cached: bool) -> int:
        tiles = math.ceil(ntok / self.M_dim)
        return tiles * self.T_iter

    @property
    def alloc(self) -> int:
        """实际占用的 DMA 通道带宽 (xDMA/iDMA 各 64 B/cc 为最小粒度)."""
        return 64 if self.bw_req <= 64 else 128

    def eta(self, ntok: int) -> float:
        """VC 算力有效率 = min(1, ntok/M_dim)."""
        return min(1.0, ntok / self.M_dim)


SHAPE_A = Shape("8x8x8", M_dim=8, bw_req=32)
SHAPE_B = Shape("4x8x16", M_dim=4, bw_req=64)
SHAPE_C = Shape("2x8x32", M_dim=2, bw_req=128)
ALL_SHAPES = [SHAPE_A, SHAPE_B, SHAPE_C]


# ============================================================
#  集群状态
# ============================================================


@dataclass
class ClusterState:
    """精确跟踪单个 Cluster 的时间轴状态."""

    cid: int
    # 任务结束时刻 (cluster 空闲时刻)
    task_end: int = 0
    # fetch 结束时刻 (cache miss 时 fetch 占用 DMA 直到此刻)
    fetch_end: int = 0
    # 当前任务 fetch 阶段的 BW 占用 (fetch 结束前有效)
    bw_in_use: int = 0
    # 缓存的 expert id (-1 = 无缓存)
    cached_eid: int = -1

    def active_bw_at(self, t: int) -> int:
        """t 时刻该 cluster 占用的 BW (仅 fetch 阶段)."""
        return self.bw_in_use if t < self.fetch_end else 0

    def is_idle_at(self, t: int) -> bool:
        return self.task_end <= t

    def is_in_compute_only_at(self, t: int) -> bool:
        """处于 compute-only 阶段: task 未完但 fetch 已结束."""
        return self.fetch_end <= t < self.task_end

    def apply_task(self, start: int, shape: Shape, ntok: int, cached: bool):
        """更新状态到执行完某任务后."""
        self.fetch_end = start + shape.T_fetch(cached)
        self.task_end = start + shape.T_task(ntok, cached)
        # 用 alloc (DMA 通道粒度 64) 而非 bw_req:
        # xDMA 和 iDMA 各 64 B/cc 独立, Shape_A/B 各占 1 条, Shape_C 占 2 条
        self.bw_in_use = 0 if cached else shape.alloc


# ============================================================
#  任务记录
# ============================================================


@dataclass
class TaskRecord:
    eid: int
    ntok: int
    cid: int
    shape: Shape
    cached: bool
    start: int
    fetch_end: int
    task_end: int
    eta: float
    rationale: str = ""


# ============================================================
#  候选方案 (支持双核同步 pair / 单核 single)
# ============================================================


@dataclass
class Assignment:
    """一次调度决策: 可能是单核或双核 pair."""

    tasks: List[Tuple[int, int, Shape, bool]]
    # (eid, ntok, shape, cached) per cluster, 索引 0=C2 1=C3
    score: float = 0.0
    rationale: str = ""


# ============================================================
#  Reward Function
# ============================================================


def compute_reward(
    # 两 cluster 当前状态 (决策前快照)
    c2_snap: ClusterState,
    c3_snap: ClusterState,
    # 本次决策: 为 cid_primary (+ 可选 cid_secondary) 分配
    primary_cid: int,
    primary_ntok: int,
    primary_shape: Shape,
    primary_cached: bool,
    primary_start: int,
    # 可选 secondary (pair 模式)
    secondary_ntok: Optional[int] = None,
    secondary_shape: Optional[Shape] = None,
    secondary_cached: bool = False,
    secondary_start: Optional[int] = None,
    queue_remaining: int = 0,
    remaining_ntoks: Optional[List[int]] = None,  # 实际剩余 expert 的 ntok 列表
) -> Tuple[float, str]:
    """
    综合评分 (越高越好).

    分项:
      1. η_primary    : 主 cluster VC 有效率 (weight 0.4)
      2. η_secondary  : 副 cluster VC 有效率 (weight 0.4, pair 时有效)
      3. bw_util      : 整体 DMA 带宽利用率 (weight 0.3)
      4. dma_idle_pen : DMA 闲置惩罚 (-1.0 if both fetch_end 相同且进入 compute_only)
      5. sync_end_pen : 同步完工软惩罚 (-0.3 if task_end 完全相同)
      6. vacuum_bonus : 带宽真空期 + 碎片专家 bonus (+0.2)
    """
    # --- 主 cluster 执行后状态 ---
    c2 = deepcopy(c2_snap)
    c3 = deepcopy(c3_snap)
    prim = c2 if primary_cid == 2 else c3
    prim.apply_task(primary_start, primary_shape, primary_ntok, primary_cached)

    sec = None
    if secondary_ntok is not None and secondary_shape is not None:
        sec = c3 if primary_cid == 2 else c2
        sec.apply_task(
            secondary_start or primary_start,
            secondary_shape,
            secondary_ntok,
            secondary_cached,
        )

    eta_p = primary_shape.eta(primary_ntok)
    eta_s = secondary_shape.eta(secondary_ntok) if sec else 0.0

    staggered = (
        secondary_start is not None
        and secondary_start > primary_start
        and sec is not None
    )

    # 估算 pair/single 完成时刻
    pair_end = max(prim.task_end, sec.task_end if sec else prim.task_end)

    # 剩余专家下界: 二机并行 makespan LB = max(sum/2, max_single)
    if remaining_ntoks:
        tasks_cc = [
            math.ceil(n / SHAPE_C.M_dim) * SHAPE_C.T_iter for n in remaining_ntoks
        ]
        remaining_lb = max(sum(tasks_cc) // 2, max(tasks_cc))
    else:
        remaining_lb = math.ceil(queue_remaining * SHAPE_C.T_iter / 2)

    # Compute-only Vacuum Bonus:
    # pair 内较大任务进入 compute_only 阶段时 DMA 全速可用 (128 B/cc),
    # 较小的 cluster (或空闲 cluster) 可以用 Shape_C 速度处理后续 expert.
    # 这部分"免费"容量从 remaining_lb 中扣除, 使评分函数偏爱 T_iter 短的 shape
    # (短 T_iter → 更早进入 compute_only → 更多 DMA 真空时间).
    if sec is not None:
        # 较大任务的 compute_only 窗口: [fetch_end, task_end]
        big = prim if prim.task_end >= sec.task_end else sec
        small = sec if prim.task_end >= sec.task_end else prim
        vacuum_start = max(
            big.fetch_end, small.task_end
        )  # C3 空闲 AND C2 compute_only 同时满足
        vacuum_end = big.task_end
        vacuum_time = max(0, vacuum_end - vacuum_start)
        free_capacity_cc = (vacuum_time // SHAPE_C.T_iter) * SHAPE_C.T_iter
    else:
        free_capacity_cc = 0

    adjusted_remaining_lb = max(0, remaining_lb - free_capacity_cc)
    est_makespan = pair_end + adjusted_remaining_lb

    # BW 利用率 (用于 tie-breaker)
    if staggered:
        bw_used = max(prim.bw_in_use, sec.bw_in_use if sec else 0)
    else:
        bw_used = prim.bw_in_use + (sec.bw_in_use if sec else 0)
    bw_util = min(bw_used / MAX_BW, 1.0)

    # --- 带宽真空期专属注入 bonus (单核模式) ---
    details = []
    if sec is None:
        peer = c3 if primary_cid == 2 else c2
        peer_in_compute = peer.is_in_compute_only_at(primary_start)
        if peer_in_compute and bw_used == 128:
            details.append("VACUUM_INJ")

    # 主评分 = -est_makespan (最小化 makespan); eta/bw 作为同 makespan 时 tie-breaker
    score = -est_makespan + 0.01 * (eta_p + eta_s + bw_util)
    tag = f"η_p={eta_p:.2f} η_s={eta_s:.2f} bw={bw_util:.2f}" + (
        f" [{' '.join(details)}]" if details else ""
    )
    return score, tag


# ============================================================
#  调度器主体
# ============================================================


class EventDrivenScheduler:
    """
    事件驱动双核调度器.

    state:
      c2, c3         : ClusterState
      remaining      : [(eid, ntok)] 待处理专家列表 (按 ntok 降序)
      clock          : 当前时钟 (最近的事件时刻)
      records        : 已派发任务记录
    """

    def __init__(
        self,
        token_dist: Dict[int, int],
        weight_bytes: int = WEIGHT_BYTES,
        cached_map: Optional[Dict[int, int]] = None,
    ):
        global WEIGHT_BYTES
        WEIGHT_BYTES = weight_bytes
        self.c2 = ClusterState(cid=2)
        self.c3 = ClusterState(cid=3)
        if cached_map:
            for eid, cid in cached_map.items():
                if cid == 2:
                    self.c2.cached_eid = eid
                else:
                    self.c3.cached_eid = eid

        self.remaining: List[Tuple[int, int]] = sorted(
            token_dist.items(), key=lambda x: -x[1]
        )
        self.clock = 0
        self.records: List[TaskRecord] = []

    # ----------------------------------------------------------
    #  约束检查
    # ----------------------------------------------------------

    def _bw_available_at(self, t: int) -> int:
        """t 时刻可用 DMA 带宽."""
        used = self.c2.active_bw_at(t) + self.c3.active_bw_at(t)
        return MAX_BW - used

    def _is_bw_legal(self, start: int, bw_req: int) -> bool:
        return self._bw_available_at(start) >= bw_req

    def _earliest_legal_start(self, cid: int, bw_req: int) -> int:
        """
        在 cluster cid 空闲后, 找最早满足 BW 约束的开始时刻.
        候选时刻集 = {cluster_free, c2.fetch_end, c3.fetch_end}.
        """
        cluster_free = self.c2.task_end if cid == 2 else self.c3.task_end
        candidates = sorted(
            {
                cluster_free,
                self.c2.fetch_end,
                self.c3.fetch_end,
            }
        )
        for t in candidates:
            if t >= cluster_free and self._bw_available_at(t) >= bw_req:
                return t
        # 极端情况: 等到所有 fetch 结束
        return max(cluster_free, self.c2.fetch_end, self.c3.fetch_end)

    # ----------------------------------------------------------
    #  强制规则
    # ----------------------------------------------------------

    def _allowed_shapes(self, ntok: int) -> List[Shape]:
        """所有 ntok 均开放全部 shape, 由 BW alloc 约束自动筛选.
        M=1 优先 Shape_C (更短 T_task), 但若 BW 不够可降级到 Shape_B.
        """
        return ALL_SHAPES

    # ----------------------------------------------------------
    #  候选生成 (前瞻搜索)
    # ----------------------------------------------------------

    def _gen_single_candidates(
        self, cid: int, remaining: List[Tuple[int, int]]
    ) -> List[Tuple[float, str, int, int, Shape, bool, int]]:
        """
        为 cid 生成单核候选方案.
        返回: (score, tag, eid, ntok, shape, cached, start)
        """
        candidates = []
        peer = self.c3 if cid == 2 else self.c2
        my = self.c2 if cid == 2 else self.c3

        for eid, ntok in remaining:
            cached = eid == my.cached_eid
            shapes = self._allowed_shapes(ntok)

            for shape in shapes:
                bw_need = 0 if cached else shape.alloc
                start = self._earliest_legal_start(cid, bw_need)

                # BW 约束检查 (用 alloc 而非 bw_req)
                if not cached and not self._is_bw_legal(start, shape.alloc):
                    continue

                score, tag = compute_reward(
                    self.c2,
                    self.c3,
                    primary_cid=cid,
                    primary_ntok=ntok,
                    primary_shape=shape,
                    primary_cached=cached,
                    primary_start=start,
                    queue_remaining=len(remaining) - 1,
                    remaining_ntoks=[n for e, n in remaining if e != eid],
                )
                candidates.append((score, tag, eid, ntok, shape, cached, start))

        # 真空期注入: 若 peer 在 compute_only, 扫描 ntok<=2
        peer_compute = peer.is_in_compute_only_at(my.task_end)
        if peer_compute:
            tiny = [(eid, n) for eid, n in remaining if n <= 2]
            for eid, ntok in tiny:
                cached = eid == my.cached_eid
                shape = SHAPE_C
                start = self._earliest_legal_start(cid, 0 if cached else shape.alloc)
                score, tag = compute_reward(
                    self.c2,
                    self.c3,
                    primary_cid=cid,
                    primary_ntok=ntok,
                    primary_shape=shape,
                    primary_cached=cached,
                    primary_start=start,
                    queue_remaining=len(remaining) - 1,
                    remaining_ntoks=[n for e, n in remaining if e != eid],
                )
                # 已有在 candidates 中的不重复加 (set 化)
                if not any(c[2] == eid for c in candidates if c[4] == shape):
                    candidates.append((score, tag, eid, ntok, shape, cached, start))

        return candidates

    def _gen_pair_candidates(
        self, remaining: List[Tuple[int, int]]
    ) -> List[Tuple[float, str, int, int, Shape, bool, int, int, Shape, bool, int]]:
        """
        两 cluster 同时空闲时枚举 pair (eA, eB, sA, sB).
        返回: (score, tag, eidA, ntokA, sA, cachedA, startA,
                             eidB, ntokB, sB, cachedB, startB)
        同时支持 Expert Splitting: eA==eB 时分配 (M_split_a, M_split_b).
        """
        cands = []
        now = max(self.c2.task_end, self.c3.task_end)
        n = len(remaining)

        # --- 普通 pair ---
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                eidA, ntokA = remaining[i]
                eidB, ntokB = remaining[j]
                cachedA = eidA == self.c2.cached_eid
                cachedB = eidB == self.c3.cached_eid
                for sA in self._allowed_shapes(ntokA):
                    for sB in self._allowed_shapes(ntokB):
                        bwA = 0 if cachedA else sA.alloc
                        bwB = 0 if cachedB else sB.alloc
                        if bwA + bwB > MAX_BW:
                            continue
                        startA = startB = now
                        rest_ntoks = [
                            rn for re, rn in remaining if re != eidA and re != eidB
                        ]
                        score, tag = compute_reward(
                            self.c2,
                            self.c3,
                            primary_cid=2,
                            primary_ntok=ntokA,
                            primary_shape=sA,
                            primary_cached=cachedA,
                            primary_start=startA,
                            secondary_ntok=ntokB,
                            secondary_shape=sB,
                            secondary_cached=cachedB,
                            secondary_start=startB,
                            queue_remaining=len(rest_ntoks),
                            remaining_ntoks=rest_ntoks,
                        )
                        cands.append(
                            (
                                score,
                                tag,
                                eidA,
                                ntokA,
                                sA,
                                cachedA,
                                startA,
                                eidB,
                                ntokB,
                                sB,
                                cachedB,
                                startB,
                            )
                        )

        # --- Expert Splitting ---
        # 对最大 ntok expert 尝试不对称拆分
        if n >= 1:
            hot_eid, hot_ntok = remaining[0]
            if hot_ntok >= 4:
                for splitA in range(1, hot_ntok):
                    splitB = hot_ntok - splitA
                    if splitA == splitB:
                        continue
                    for sA in self._allowed_shapes(splitA):
                        for sB in self._allowed_shapes(splitB):
                            bwA = sA.alloc
                            bwB = sB.alloc
                            if bwA + bwB > MAX_BW:
                                continue
                            rest_ntoks = [rn for _, rn in remaining[1:]]
                            score, tag = compute_reward(
                                self.c2,
                                self.c3,
                                primary_cid=2,
                                primary_ntok=splitA,
                                primary_shape=sA,
                                primary_cached=False,
                                primary_start=now,
                                secondary_ntok=splitB,
                                secondary_shape=sB,
                                secondary_cached=False,
                                secondary_start=now,
                                queue_remaining=len(rest_ntoks),
                                remaining_ntoks=rest_ntoks,
                            )
                            cands.append(
                                (
                                    score,
                                    f"SPLIT({splitA},{splitB}) {tag}",
                                    hot_eid,
                                    splitA,
                                    sA,
                                    False,
                                    now,
                                    hot_eid,
                                    splitB,
                                    sB,
                                    False,
                                    now,
                                )
                            )

        return cands

    # ----------------------------------------------------------
    #  执行一次调度事件
    # ----------------------------------------------------------

    def _dispatch(
        self, cid: int, eid: int, ntok: int, shape: Shape, cached: bool, start: int
    ) -> TaskRecord:
        cluster = self.c2 if cid == 2 else self.c3
        cluster.apply_task(start, shape, ntok, cached)
        cluster.cached_eid = eid  # 任务完成后 expert 驻留
        rec = TaskRecord(
            eid=eid,
            ntok=ntok,
            cid=cid,
            shape=shape,
            cached=cached,
            start=start,
            fetch_end=cluster.fetch_end,
            task_end=cluster.task_end,
            eta=shape.eta(ntok),
        )
        return rec

    # ----------------------------------------------------------
    #  主循环
    # ----------------------------------------------------------

    def run(self) -> List[TaskRecord]:
        while self.remaining:
            # 时钟推进到下一个事件 (最早空闲 cluster)
            t2, t3 = self.c2.task_end, self.c3.task_end
            both_idle = abs(t2 - t3) <= 1

            if both_idle and len(self.remaining) >= 2:
                # ---- PAIR / SPLIT 模式 ----
                self.clock = min(t2, t3)
                pair_cands = self._gen_pair_candidates(self.remaining)

                if pair_cands:
                    pair_cands.sort(key=lambda x: -x[0])
                    best = pair_cands[0]
                    (
                        score,
                        tag,
                        eidA,
                        ntokA,
                        sA,
                        cachedA,
                        startA,
                        eidB,
                        ntokB,
                        sB,
                        cachedB,
                        startB,
                    ) = best

                    recA = self._dispatch(2, eidA, ntokA, sA, cachedA, startA)
                    recA.rationale = f"PAIR-C2 {tag}"
                    recB = self._dispatch(3, eidB, ntokB, sB, cachedB, startB)
                    recB.rationale = f"PAIR-C3 {tag}"
                    self.records.extend([recA, recB])

                    # 若是 split, eid 相同 → 只移除一次
                    drop = set()
                    if eidA == eidB:
                        drop.add(eidA)
                    else:
                        drop.update([eidA, eidB])
                    self.remaining = [
                        (e, n) for e, n in self.remaining if e not in drop
                    ]
                    continue

            # ---- SINGLE 模式 ----
            if t2 <= t3:
                cid = 2
                self.clock = t2
            else:
                cid = 3
                self.clock = t3

            cands = self._gen_single_candidates(cid, self.remaining)

            # 若没有可用方案 (极端 BW 竞争), 等到 peer fetch_end
            if not cands:
                # 强制等待 peer fetch 释放 BW
                if cid == 2:
                    self.c2.task_end = max(self.c2.task_end, self.c3.fetch_end)
                else:
                    self.c3.task_end = max(self.c3.task_end, self.c2.fetch_end)
                continue

            cands.sort(key=lambda x: -x[0])
            score, tag, eid, ntok, shape, cached, start = cands[0]

            rec = self._dispatch(cid, eid, ntok, shape, cached, start)
            rec.rationale = tag
            self.records.append(rec)
            self.remaining = [(e, n) for e, n in self.remaining if e != eid]

        return self.records

    @property
    def makespan(self) -> int:
        return max(self.c2.task_end, self.c3.task_end)


# ============================================================
#  打印
# ============================================================


def print_schedule(records: List[TaskRecord], makespan: int):
    print(f"=== Makespan = {makespan:,} cc ===")
    print(
        f"{'Start':>8} {'C':>2} {'E':>3} {'tok':>3} {'shape':>8} "
        f"{'cache':>5} {'T_F':>8} {'T_task':>8} {'η':>5} | rationale"
    )
    print("-" * 90)
    for r in sorted(records, key=lambda x: (x.start, x.cid)):
        print(
            f"{r.start:>8,} C{r.cid} E{r.eid:>2d} {r.ntok:>3d} "
            f"{r.shape.name:>8} {'HIT' if r.cached else 'MISS':>5} "
            f"{r.fetch_end - r.start:>8,} {r.task_end:>8,} {r.eta:>5.2f} "
            f"| {r.rationale}"
        )


# ============================================================
#  Self-test
# ============================================================

if __name__ == "__main__":
    print("Shape 参数 (W4A8, expert=4,325,376 B):")
    for s in ALL_SHAPES:
        print(
            f"  {s.name}: M_dim={s.M_dim} BW={s.bw_req} "
            f"T_iter={s.T_iter:,}cc  η(M=1)={s.eta(1):.3f} "
            f"η(M=4)={s.eta(4):.3f} η(M=8)={s.eta(8):.3f}"
        )
    print()

    test_cases = [
        ("M=8 hot(6,1,1)", {0: 6, 1: 1, 2: 1}, {}),
        ("M=16 mixed(8,4,2,1,1)", {8: 8, 4: 4, 2: 2, 1: 1, 10: 1}, {}),
        ("M=16 uniform(4×4)", {0: 4, 1: 4, 2: 4, 3: 4}, {}),
        ("M=16 all cold (1×16)", {i: 1 for i in range(16)}, {}),
        ("M=4 uniform(1×4)", {0: 1, 1: 1, 2: 1, 3: 1}, {}),
        ("M=32 hot(20,6,6)", {0: 20, 1: 6, 2: 6}, {}),
        ("M=4 with cache E0@C2", {0: 4, 1: 4, 2: 4, 3: 4}, {0: 2}),
    ]

    for name, dist, cache in test_cases:
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")
        sched = EventDrivenScheduler(dist, cached_map=cache)
        recs = sched.run()
        print_schedule(recs, sched.makespan)
