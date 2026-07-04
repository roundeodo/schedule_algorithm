#!/usr/bin/env python3
"""
Beam Search 全局静态调度器 (v1)
=====================================================================
目标: 对给定 token 分布，找全局最优调度方案 (最小 makespan).

算法:
  - 状态 (State): (c2, c3, remaining, schedule)
  - 动作 (Action): 在当前时钟下，为一个或两个 cluster 分配 expert+shape
    包含: PAIR / EXPERT-SPLIT / SINGLE 三类
  - 展开: 每步从当前 beam 中每个状态生成所有合法后继，
    用 A*-like 下界 f(s) = actual_elapsed + remaining_lb 排序，
    保留 top-k (beam_width) 个状态继续搜索
  - 终止: remaining 为空，记录叶节点 makespan
  - 输出: makespan 最小的完整调度序列

物理约束 (继承自 v26):
  - 公共 SRAM 拥有两条独立 DMA 物理通道: SRAM_xDMA 和 SRAM_iDMA，各 64 B/cc
  - 这两条通道均可在任意时刻动态路由到 C0/C1/C2/C3 中的任意一个 cluster (1:1 约束)
  - 两条通道可以同时连接同一个 cluster (→ 128 B/cc 合力搬运), 也可以分别连接不同 cluster
  - DMA 通道粒度 = 64 B/cc，无法细分。故:
      alloc(Shape_A) = 64 B/cc  ← bw_req=32 但最小分配=1 条通道=64
      alloc(Shape_B) = 64 B/cc  ← bw_req=64 恰好占用 1 条通道
      alloc(Shape_C) = 128 B/cc ← bw_req=128 占用全部 2 条通道 (均指向同一 cluster)
  - BW 约束: alloc(C2_fetch) + alloc(C3_fetch) <= 128 B/cc (同一时刻总和)
  - T_iter = ceil(W / bw_req)
  - T_task(ntok, cached) = ceil(ntok/M_dim) * T_iter  (cached: T_fetch=0)
"""

import math
import heapq
import itertools
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, FrozenSet
from copy import deepcopy

# ============================================================
#  常量 & Shape
# ============================================================

WEIGHT_BYTES = 3 * 2048 * 1408 // 2  # 4,325,376 B
MAX_BW = 128  # B/cc


@dataclass(frozen=True)
class Shape:
    name: str
    M_dim: int
    bw_req: int

    @property
    def T_iter(self) -> int:
        return math.ceil(WEIGHT_BYTES / self.bw_req)

    @property
    def alloc(self) -> int:
        # 物理 DMA 通道粒度 = 64 B/cc (一条通道).
        # Shape_A (bw_req=32): 实际只需半条通道带宽，但硬件最小分配单元=1条通道 → alloc=64
        # Shape_B (bw_req=64): 精确用满 1 条通道 → alloc=64
        # Shape_C (bw_req=128): 占用两条通道 (xDMA+iDMA 全部) → alloc=128
        return 64 if self.bw_req <= 64 else 128

    def T_fetch(self, cached: bool) -> int:
        return 0 if cached else self.T_iter

    def T_task(self, ntok: int, cached: bool) -> int:
        return math.ceil(ntok / self.M_dim) * self.T_iter

    def eta(self, ntok: int) -> float:
        return min(1.0, ntok / self.M_dim)


SHAPE_A = Shape("8x8x8", M_dim=8, bw_req=32)
SHAPE_B = Shape("4x8x16", M_dim=4, bw_req=64)
SHAPE_C = Shape("2x8x32", M_dim=2, bw_req=128)
ALL_SHAPES = [SHAPE_A, SHAPE_B, SHAPE_C]


# ============================================================
#  Cluster 状态（可哈希快照）
# ============================================================


@dataclass(frozen=True)
class ClusterSnap:
    """不可变状态快照，用于 beam 状态哈希去重."""

    task_end: int
    fetch_end: int
    bw_in_use: int
    cached_eid: int

    def active_bw_at(self, t: int) -> int:
        return self.bw_in_use if t < self.fetch_end else 0

    def is_in_compute_only_at(self, t: int) -> bool:
        return self.fetch_end <= t < self.task_end

    def after_task(
        self, start: int, shape: Shape, ntok: int, cached: bool, new_cached_eid: int
    ) -> "ClusterSnap":
        return ClusterSnap(
            task_end=start + shape.T_task(ntok, cached),
            fetch_end=start + shape.T_fetch(cached),
            bw_in_use=0 if cached else shape.alloc,
            cached_eid=new_cached_eid,
        )


IDLE_CLUSTER = ClusterSnap(task_end=0, fetch_end=0, bw_in_use=0, cached_eid=-1)


# ============================================================
#  任务记录（调度序列中的一个动作）
# ============================================================


@dataclass(frozen=True)
class Action:
    """一次调度动作（可能同时分配两个 cluster）."""

    # C2 分配
    c2_eid: int
    c2_ntok: int
    c2_shape: Shape
    c2_cached: bool
    c2_start: int
    # C3 分配 (-1 表示本步 C3 不动)
    c3_eid: int = -1
    c3_ntok: int = 0
    c3_shape: Optional[Shape] = None
    c3_cached: bool = False
    c3_start: int = -1
    tag: str = ""

    @property
    def is_pair(self) -> bool:
        return self.c3_eid >= 0


# ============================================================
#  搜索状态（Beam 节点）
# ============================================================


@dataclass
class SearchState:
    """
    Beam Search 中的一个节点.

    f_score = makespan_lb (越小越优先)
    """

    c2: ClusterSnap
    c3: ClusterSnap
    # remaining: sorted tuple (eid, ntok) 降序 ntok, 保持 hashable
    remaining: Tuple[Tuple[int, int], ...]
    history: Tuple[Action, ...]  # 完整动作序列
    g_score: int  # 当前实际 elapsed (= max(c2.task_end, c3.task_end))
    f_score: int  # g + remaining_lb (A* 估价函数)

    # 优先队列比较: f_score 小优先, 相同则 g_score 大优先 (更完成的状态先)
    def __lt__(self, other: "SearchState") -> bool:
        if self.f_score != other.f_score:
            return self.f_score < other.f_score
        return self.g_score > other.g_score

    def fingerprint(self) -> tuple:
        """用于去重的状态指纹 (c2, c3, remaining)."""
        return (self.c2, self.c3, self.remaining)


# ============================================================
#  下界估算 (Johnson's Rule 两机 Flowshop LB)
# ============================================================


def remaining_lb(remaining: Tuple[Tuple[int, int], ...]) -> int:
    """
    剩余 expert 队列的 makespan 下界.
    每个 expert 用最快 shape (Shape_C) 单独跑的时间:
      t_i = ceil(ntok_i / M_dim_C) * T_iter_C
    两机并行:
      LB = max( sum(t_i)/2, max(t_i) )
    """
    if not remaining:
        return 0
    tasks = [math.ceil(n / SHAPE_C.M_dim) * SHAPE_C.T_iter for _, n in remaining]
    return max(sum(tasks) // 2, max(tasks))


# ============================================================
#  合法动作生成
# ============================================================


def _split_candidates(hot_ntok: int, sA: "Shape", sB: "Shape") -> List[int]:
    """
    计算 SPLIT 时 splitA 的最小有效候选集 (去除等价重复).

    T_task(splitA, sA) = ceil(splitA/M_dim_A)*T_iter_A 是关于 splitA 的阶梯函数:
      在 splitA = k*M_dim_A 处步进，区间内取值相同.
    T_task(splitB, sB) (splitB=hot_ntok-splitA) 在 splitA = hot_ntok-k*M_dim_B 处步进.

    只需在两者的步进边界取样，即可覆盖所有不等价的 (T_A, T_B) 配对.
    例: hot_ntok=20, M_dim_A=8, M_dim_B=4
      原始: 枚举 1..19 → 19 个值
      优化: {8,16} ∪ {4,8,12,16} = {4,8,12,16} → 4 个值  (4.75x 提速)
    证明完备性: 最优 splitA 一定在步进边界处 (否则可向边界移动同时保持或改善最大值).
    """
    cands: set = set()
    # T_A 的步进点 (splitA = k * M_dim_A)
    k = 1
    while k * sA.M_dim < hot_ntok:
        cands.add(k * sA.M_dim)
        k += 1
    # T_B 的步进点 (splitA = hot_ntok - k * M_dim_B)
    k = 1
    while hot_ntok - k * sB.M_dim > 0:
        cands.add(hot_ntok - k * sB.M_dim)
        k += 1
    # 若候选集为空 (极端情况: hot_ntok < min(M_dim_A, M_dim_B))，回退到中间切分
    return sorted(cands) if cands else [max(1, hot_ntok // 2)]


def _bw_free_at(c2: ClusterSnap, c3: ClusterSnap, t: int) -> int:
    return MAX_BW - c2.active_bw_at(t) - c3.active_bw_at(t)


def _earliest_start(cluster: ClusterSnap, peer: ClusterSnap, alloc_need: int) -> int:
    """cluster 空闲后满足 BW 约束的最早时刻."""
    base = cluster.task_end
    for t in sorted({base, peer.fetch_end, cluster.fetch_end}):
        if t >= base and (_bw_free_at_t := MAX_BW - peer.active_bw_at(t)) >= alloc_need:
            return t
    return max(base, peer.fetch_end)


def gen_actions(
    c2: ClusterSnap,
    c3: ClusterSnap,
    remaining: Tuple[Tuple[int, int], ...],
    allow_split: bool = True,
    smart_split: bool = True,
) -> List[Action]:
    """
    生成当前状态下所有合法动作，共四类:
      1. PAIR:       两 cluster 同时空闲，枚举 (eidA/sA, eidB/sB) 组合
      2. SPLIT:      同时空闲，将最热 expert 的 token 拆给两 cluster
      3. SINGLE:     一个 cluster 先空闲，立刻分配（不等待对方）
      4. WAIT-PAIR / WAIT-SPLIT:
                     两 cluster 不同时空闲时，让较早完成的 cluster 空转等待,
                     在 max(t2,t3) 时刻以 PAIR/SPLIT 方式同时启动.
                     代价=等待周期; 收益=双通道并行 + 负载对齐.
    """
    actions = []
    n = len(remaining)
    if n == 0:
        return actions

    t2, t3 = c2.task_end, c3.task_end
    both_idle = t2 == t3  # 严格同时空闲

    # ---- PAIR (需要 n >= 2, 两个不同 expert) ----
    if both_idle and n >= 2:
        now = t2
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                eidA, ntokA = remaining[i]
                eidB, ntokB = remaining[j]
                cachedA = eidA == c2.cached_eid
                cachedB = eidB == c3.cached_eid
                for sA in ALL_SHAPES:
                    for sB in ALL_SHAPES:
                        bwA = 0 if cachedA else sA.alloc
                        bwB = 0 if cachedB else sB.alloc
                        if bwA + bwB > MAX_BW:
                            continue
                        actions.append(
                            Action(
                                c2_eid=eidA,
                                c2_ntok=ntokA,
                                c2_shape=sA,
                                c2_cached=cachedA,
                                c2_start=now,
                                c3_eid=eidB,
                                c3_ntok=ntokB,
                                c3_shape=sB,
                                c3_cached=cachedB,
                                c3_start=now,
                                tag=f"PAIR({eidA}+{eidB})",
                            )
                        )

    # ---- SPLIT (n >= 1 即可; 将最热 expert 拆给两个 cluster 并行处理) ----
    # BUG FIX 1: 原代码把 SPLIT 放在 n>=2 块内，导致只剩 1 个 expert 时永远无法 SPLIT
    # BUG FIX 2: c2_cached/c3_cached 被写死为 False，但若拆分的 expert 正好是某 cluster
    #            上一轮处理过的 expert，应当利用 cache-hit (T_fetch=0, alloc=0)
    if both_idle and allow_split:
        now = t2
        hot_eid, hot_ntok = remaining[0]
        # 检查各 cluster 对热门 expert 的 cache 状态
        c2_hot_cached = hot_eid == c2.cached_eid
        c3_hot_cached = hot_eid == c3.cached_eid
        if hot_ntok >= 4:
            # smart_split=True: 只枚举 T_task 阶梯步进边界的 splitA 值 (4-8x 加速)
            # smart_split=False: 暴力枚举所有 splitA (用于对照实验)
            for sA in ALL_SHAPES:
                for sB in ALL_SHAPES:
                    bwA = 0 if c2_hot_cached else sA.alloc
                    bwB = 0 if c3_hot_cached else sB.alloc
                    if bwA + bwB > MAX_BW:
                        continue
                    splits = (
                        _split_candidates(hot_ntok, sA, sB)
                        if smart_split
                        else range(1, hot_ntok)
                    )
                    for splitA in splits:
                        splitB = hot_ntok - splitA
                        actions.append(
                            Action(
                                c2_eid=hot_eid,
                                c2_ntok=splitA,
                                c2_shape=sA,
                                c2_cached=c2_hot_cached,
                                c2_start=now,
                                c3_eid=hot_eid,
                                c3_ntok=splitB,
                                c3_shape=sB,
                                c3_cached=c3_hot_cached,
                                c3_start=now,
                                tag=f"SPLIT({splitA},{splitB})",
                            )
                        )

    # ---- SINGLE (先空闲的那个 cluster) ----
    if t2 <= t3:
        cid_free, c_free, c_peer = 2, c2, c3
    else:
        cid_free, c_free, c_peer = 3, c3, c2

    for eid, ntok in remaining:
        cached = eid == c_free.cached_eid
        for shape in ALL_SHAPES:
            alloc_need = 0 if cached else shape.alloc
            start = _earliest_start(c_free, c_peer, alloc_need)
            if not cached and (MAX_BW - c_peer.active_bw_at(start)) < alloc_need:
                continue
            if cid_free == 2:
                actions.append(
                    Action(
                        c2_eid=eid,
                        c2_ntok=ntok,
                        c2_shape=shape,
                        c2_cached=cached,
                        c2_start=start,
                        tag=f"SINGLE-C2({eid})",
                    )
                )
            else:
                actions.append(
                    Action(
                        c2_eid=-1,  # sentinel: 用 c3 字段
                        c2_ntok=0,
                        c2_shape=SHAPE_C,
                        c2_cached=False,
                        c2_start=-1,
                        c3_eid=eid,
                        c3_ntok=ntok,
                        c3_shape=shape,
                        c3_cached=cached,
                        c3_start=start,
                        tag=f"SINGLE-C3({eid})",
                    )
                )

    # ---- WAIT-TO-PAIR & WAIT-TO-SPLIT (较早空闲的 cluster 主动等待对齐) ----
    # 当 t2 ≠ t3 时，先完成的 cluster 空转等到 wait_t = max(t2,t3)，
    # 再以 PAIR/SPLIT 方式同时启动两个任务.
    # beam search 会通过 f_score 自动判断等待是否值得.
    if not both_idle:
        wait_t = max(t2, t3)
        # WAIT-TO-PAIR (需要 n >= 2)
        if n >= 2:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    eidA, ntokA = remaining[i]
                    eidB, ntokB = remaining[j]
                    cachedA = eidA == c2.cached_eid
                    cachedB = eidB == c3.cached_eid
                    for sA in ALL_SHAPES:
                        for sB in ALL_SHAPES:
                            bwA = 0 if cachedA else sA.alloc
                            bwB = 0 if cachedB else sB.alloc
                            if bwA + bwB > MAX_BW:
                                continue
                            actions.append(
                                Action(
                                    c2_eid=eidA,
                                    c2_ntok=ntokA,
                                    c2_shape=sA,
                                    c2_cached=cachedA,
                                    c2_start=wait_t,
                                    c3_eid=eidB,
                                    c3_ntok=ntokB,
                                    c3_shape=sB,
                                    c3_cached=cachedB,
                                    c3_start=wait_t,
                                    tag=f"WAIT-PAIR({eidA}+{eidB})",
                                )
                            )
        # WAIT-TO-SPLIT (n >= 1 即可；同样修复 cache-hit 被写死为 False 的问题)
        if allow_split:
            hot_eid, hot_ntok = remaining[0]
            c2_hot_cached = hot_eid == c2.cached_eid
            c3_hot_cached = hot_eid == c3.cached_eid
            if hot_ntok >= 4:
                for sA in ALL_SHAPES:
                    for sB in ALL_SHAPES:
                        bwA = 0 if c2_hot_cached else sA.alloc
                        bwB = 0 if c3_hot_cached else sB.alloc
                        if bwA + bwB > MAX_BW:
                            continue
                        splits = (
                            _split_candidates(hot_ntok, sA, sB)
                            if smart_split
                            else range(1, hot_ntok)
                        )
                        for splitA in splits:
                            splitB = hot_ntok - splitA
                            actions.append(
                                Action(
                                    c2_eid=hot_eid,
                                    c2_ntok=splitA,
                                    c2_shape=sA,
                                    c2_cached=c2_hot_cached,
                                    c2_start=wait_t,
                                    c3_eid=hot_eid,
                                    c3_ntok=splitB,
                                    c3_shape=sB,
                                    c3_cached=c3_hot_cached,
                                    c3_start=wait_t,
                                    tag=f"WAIT-SPLIT({splitA},{splitB})",
                                )
                            )

    return actions


# ============================================================
#  动作应用 → 生成后继状态
# ============================================================


def apply_action(
    state: SearchState,
    action: Action,
) -> SearchState:
    c2, c3 = state.c2, state.c3
    rem = list(state.remaining)

    new_c2, new_c3 = c2, c3
    consumed_eids = set()

    # 处理 C2 分配
    if action.c2_eid >= 0:
        new_c2 = c2.after_task(
            action.c2_start,
            action.c2_shape,
            action.c2_ntok,
            action.c2_cached,
            new_cached_eid=action.c2_eid,
        )
        consumed_eids.add(action.c2_eid)

    # 处理 C3 分配
    if action.is_pair and action.c3_eid >= 0:
        new_c3 = c3.after_task(
            action.c3_start,
            action.c3_shape,
            action.c3_ntok,
            action.c3_cached,
            new_cached_eid=action.c3_eid,
        )
        consumed_eids.add(action.c3_eid)
    elif not action.is_pair and action.c3_eid >= 0:
        # SINGLE-C3
        new_c3 = c3.after_task(
            action.c3_start,
            action.c3_shape,
            action.c3_ntok,
            action.c3_cached,
            new_cached_eid=action.c3_eid,
        )
        consumed_eids.add(action.c3_eid)

    # 移除已处理 expert (SPLIT 同一 eid 只移除一次)
    new_rem = tuple((e, n) for e, n in rem if e not in consumed_eids)

    g = max(new_c2.task_end, new_c3.task_end)
    f = g + remaining_lb(new_rem)

    return SearchState(
        c2=new_c2,
        c3=new_c3,
        remaining=new_rem,
        history=state.history + (action,),
        g_score=g,
        f_score=f,
    )


# ============================================================
#  Beam Search 主体
# ============================================================


class BeamScheduler:
    """
    Beam Search 全局静态调度器.

    参数:
      token_dist : {eid: ntok}
      beam_width : 每步保留的候选状态数 (越大越精确，越慢)
      allow_split: 是否允许 Expert Splitting
      max_steps  : 防止死循环的最大步数上限
    """

    def __init__(
        self,
        token_dist: Dict[int, int],
        beam_width: int = 64,
        allow_split: bool = True,
        max_steps: int = 500,
        cached_c2: int = -1,  # C2 初始缓存的 expert id (-1=冷启动)
        cached_c3: int = -1,  # C3 初始缓存的 expert id (-1=冷启动)
        smart_split: bool = True,  # True=候选集剪枝(推荐); False=暴力枚举(对照用)
    ):
        self.token_dist = token_dist
        self.beam_width = beam_width
        self.allow_split = allow_split
        self.max_steps = max_steps
        self.smart_split = smart_split
        self.init_c2 = ClusterSnap(
            task_end=0, fetch_end=0, bw_in_use=0, cached_eid=cached_c2
        )
        self.init_c3 = ClusterSnap(
            task_end=0, fetch_end=0, bw_in_use=0, cached_eid=cached_c3
        )

        # 初始剩余队列: 按 ntok 降序排列 (贪心优先处理大 expert)
        self.initial_remaining: Tuple[Tuple[int, int], ...] = tuple(
            sorted(token_dist.items(), key=lambda x: -x[1])
        )

    def run(self) -> Tuple[int, List[Action]]:
        """
        返回 (best_makespan, best_action_sequence).
        """
        init_state = SearchState(
            c2=self.init_c2,
            c3=self.init_c3,
            remaining=self.initial_remaining,
            history=(),
            g_score=0,
            f_score=remaining_lb(self.initial_remaining),
        )

        # beam: min-heap on f_score
        beam: List[SearchState] = [init_state]
        heapq.heapify(beam)

        best_makespan = float("inf")
        best_history: List[Action] = []

        seen: Dict[tuple, int] = {}  # fingerprint → best f_score seen

        for step in range(self.max_steps):
            if not beam:
                break

            # 展开当前 beam
            next_candidates: List[SearchState] = []

            for state in beam:
                if not state.remaining:
                    # 叶节点: 记录最优
                    ms = state.g_score
                    if ms < best_makespan:
                        best_makespan = ms
                        best_history = list(state.history)
                    continue

                # 生成所有合法动作
                actions = gen_actions(
                    state.c2,
                    state.c3,
                    state.remaining,
                    allow_split=self.allow_split,
                    smart_split=self.smart_split,
                )

                for action in actions:
                    child = apply_action(state, action)

                    # f_score 剪枝: 已知更优路径则跳过
                    fp = child.fingerprint()
                    if fp in seen and seen[fp] <= child.f_score:
                        continue
                    seen[fp] = child.f_score

                    next_candidates.append(child)

            if not next_candidates:
                break

            # 按 f_score 排序, 保留 top beam_width
            next_candidates.sort()
            beam = next_candidates[: self.beam_width]

        return best_makespan, best_history


# ============================================================
#  时间轴甘特图 (按事件点切分，逐段展示资源分配)
# ============================================================


def format_timeline(
    actions: List[Action], token_dist: Dict[int, int], makespan: int
) -> str:
    """
    输出逐段资源甘特表.
    时间段按所有事件点 (start/fetch_end/task_end) 切分，每段内资源状态恒定.

    列: Start | End | Dur | SRAM_xDMA | SRAM_iDMA | C2_VC | C3_VC

    DMA 分配规则 (两条通道均为 SRAM 侧，可动态路由到 C2 或 C3):
      - 若某 cluster 需 128 B/cc (Shape_C): xDMA 和 iDMA 同时连接该 cluster
      - 若两个 cluster 各需 64 B/cc: xDMA→C2, iDMA→C3
      - 若仅一个 cluster 需 64 B/cc:  xDMA→该 cluster, iDMA=idle
      - 均无 fetch:                    xDMA=idle, iDMA=idle
    """

    # 1. 重建每个 cluster 的任务段列表
    segs_c2: List[dict] = []
    segs_c3: List[dict] = []

    for action in actions:
        if action.c2_eid >= 0:
            s, ntok, cached, start = (
                action.c2_shape,
                action.c2_ntok,
                action.c2_cached,
                action.c2_start,
            )
            tf = 0 if cached else s.T_iter
            tt = s.T_task(ntok, cached)
            segs_c2.append(
                dict(
                    eid=action.c2_eid,
                    ntok=ntok,
                    shape=s,
                    cached=cached,
                    start=start,
                    fetch_end=start + tf,
                    task_end=start + tt,
                )
            )
        c3_eid = action.c3_eid
        if c3_eid >= 0 and action.c3_start >= 0:
            s, ntok, cached, start = (
                action.c3_shape,
                action.c3_ntok,
                action.c3_cached,
                action.c3_start,
            )
            tf = 0 if cached else s.T_iter
            tt = s.T_task(ntok, cached)
            segs_c3.append(
                dict(
                    eid=c3_eid,
                    ntok=ntok,
                    shape=s,
                    cached=cached,
                    start=start,
                    fetch_end=start + tf,
                    task_end=start + tt,
                )
            )

    # 2. 收集所有事件时间点，切分时间轴
    events = sorted(
        set(
            [0, makespan]
            + [seg["start"] for seg in segs_c2 + segs_c3]
            + [seg["fetch_end"] for seg in segs_c2 + segs_c3]
            + [seg["task_end"] for seg in segs_c2 + segs_c3]
        )
    )

    def active_segs(segs, t_s, t_e):
        return [sg for sg in segs if sg["start"] < t_e and sg["task_end"] > t_s]

    def fetching_segs(segs, t_s, t_e):
        """仅返回当前正处于 fetch 阶段 (t_s < fetch_end) 的任务段."""
        return [sg for sg in active_segs(segs, t_s, t_e) if t_s < sg["fetch_end"]]

    def dma_label(sg, cluster_name: str) -> str:
        return f"→{cluster_name}: E{sg['eid']}({sg['shape'].name})"

    def vc_cell(segs, t_s, t_e) -> str:
        parts = []
        for sg in active_segs(segs, t_s, t_e):
            phase = "fetch" if t_s < sg["fetch_end"] else "compute"
            eta = sg["shape"].eta(sg["ntok"])
            parts.append(f"E{sg['eid']}({sg['ntok']}tok,η={eta:.2f},{phase})")
        return " / ".join(parts) if parts else "idle"

    def compute_dma_assignment(t_s, t_e):
        """
        根据当前时间段 C2/C3 的 fetch 带宽需求，动态分配 SRAM_xDMA 和 SRAM_iDMA.
        返回: (xdma_cell_str, idma_cell_str)
        """
        fc2 = fetching_segs(segs_c2, t_s, t_e)
        fc3 = fetching_segs(segs_c3, t_s, t_e)
        bw2 = fc2[0]["shape"].alloc if fc2 else 0
        bw3 = fc3[0]["shape"].alloc if fc3 else 0

        if bw2 == 128:
            # Shape_C: 两条通道同时连接 C2
            lbl = dma_label(fc2[0], "C2")
            return (lbl, lbl)
        if bw3 == 128:
            # Shape_C: 两条通道同时连接 C3
            lbl = dma_label(fc3[0], "C3")
            return (lbl, lbl)
        if bw2 == 64 and bw3 == 64:
            # 两个 cluster 各占一条通道
            return (dma_label(fc2[0], "C2"), dma_label(fc3[0], "C3"))
        if bw2 == 64:
            return (dma_label(fc2[0], "C2"), "idle")
        if bw3 == 64:
            return (dma_label(fc3[0], "C3"), "idle")
        return ("idle", "idle")

    # 3. 输出表格
    W = 36
    HDR = (
        f"{'Start':>10} {'End':>10} {'Dur':>8}  "
        f"{'SRAM_xDMA':<{W}} {'SRAM_iDMA':<{W}} "
        f"{'C2_VC':<{W}} {'C3_VC':<{W}}"
    )
    SEP = "-" * len(HDR)

    lines = [
        "\n" + "=" * len(SEP),
        "  时间轴甘特图  (SRAM_xDMA/iDMA 均可动态路由至 C2 或 C3)",
        "=" * len(SEP),
        HDR,
        SEP,
    ]

    prev_row = None
    for i in range(len(events) - 1):
        t_s, t_e = events[i], events[i + 1]
        xdma, idma = compute_dma_assignment(t_s, t_e)
        row = (
            xdma,
            idma,
            vc_cell(segs_c2, t_s, t_e),
            vc_cell(segs_c3, t_s, t_e),
        )
        if row == prev_row:
            continue
        prev_row = row
        lines.append(
            f"{t_s:>10,} {t_e:>10,} {t_e-t_s:>8,}  "
            f"{row[0]:<{W}} {row[1]:<{W}} "
            f"{row[2]:<{W}} {row[3]:<{W}}"
        )
    lines.append(SEP)
    return "\n".join(lines)


# ============================================================
#  调度结果格式化输出
# ============================================================


def format_schedule(actions: List[Action], token_dist: Dict[int, int]) -> str:
    lines = []
    lines.append(f"   Start  C   E tok    shape cache      T_F   T_task     η | tag")
    lines.append("-" * 90)

    # 展开所有调度记录并按 start 排序
    records = []
    for action in actions:
        if action.c2_eid >= 0:
            s = action.c2_shape
            ntok = action.c2_ntok
            cached = action.c2_cached
            start = action.c2_start
            tf = 0 if cached else s.T_iter
            tt = s.T_task(ntok, cached)
            records.append(
                (
                    start,
                    2,
                    action.c2_eid,
                    ntok,
                    s,
                    cached,
                    tf,
                    tt,
                    s.eta(ntok),
                    action.tag,
                )
            )
        if action.is_pair and action.c3_eid >= 0:
            s = action.c3_shape
            ntok = action.c3_ntok
            cached = action.c3_cached
            start = action.c3_start
            tf = 0 if cached else s.T_iter
            tt = s.T_task(ntok, cached)
            records.append(
                (
                    start,
                    3,
                    action.c3_eid,
                    ntok,
                    s,
                    cached,
                    tf,
                    tt,
                    s.eta(ntok),
                    action.tag,
                )
            )
        elif not action.is_pair and action.c3_eid >= 0:
            s = action.c3_shape
            ntok = action.c3_ntok
            cached = action.c3_cached
            start = action.c3_start
            tf = 0 if cached else s.T_iter
            tt = s.T_task(ntok, cached)
            records.append(
                (
                    start,
                    3,
                    action.c3_eid,
                    ntok,
                    s,
                    cached,
                    tf,
                    tt,
                    s.eta(ntok),
                    action.tag,
                )
            )

    records.sort(key=lambda x: (x[0], x[1]))
    for start, cid, eid, ntok, shape, cached, tf, tt, eta, tag in records:
        cache_str = "HIT " if cached else "MISS"
        lines.append(
            f"  {start:6,d} C{cid} E{eid:2d} {ntok:3d}  {shape.name:>7s}  {cache_str}"
            f"  {tf:6,d}  {tt:7,d}  {eta:.2f} | {tag}"
        )
    return "\n".join(lines)


# ============================================================
#  测试套件
# ============================================================


def run_test(
    name: str,
    token_dist: Dict[int, int],
    beam_width: int = 128,
    show_timeline: bool = True,
    cached_c2: int = -1,
    cached_c3: int = -1,
):
    """单条测试，默认输出时间轴甘特图."""
    total_tok = sum(token_dist.values())
    print(f"\n{'='*70}")
    print(f"  {name}  (M={total_tok}, experts={len(token_dist)}, beam={beam_width})")
    if cached_c2 >= 0 or cached_c3 >= 0:
        print(f"  [init cache] C2←E{cached_c2}  C3←E{cached_c3}")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    sched = BeamScheduler(
        token_dist,
        beam_width=beam_width,
        allow_split=True,
        cached_c2=cached_c2,
        cached_c3=cached_c3,
    )
    makespan, actions = sched.run()
    elapsed = time.perf_counter() - t0

    print(f"=== Makespan = {makespan:,d} cc  (search: {elapsed:.2f}s) ===")
    print(format_schedule(actions, token_dist))

    if show_timeline:
        print(format_timeline(actions, token_dist, makespan))

    # 计算整体 η 和 DMA 利用率
    total_compute = 0
    total_fetch = 0
    for action in actions:
        for eid, ntok, shape, cached, start in [
            (
                action.c2_eid,
                action.c2_ntok,
                action.c2_shape,
                action.c2_cached,
                action.c2_start,
            ),
            (
                action.c3_eid,
                action.c3_ntok,
                action.c3_shape if action.c3_shape else SHAPE_C,
                action.c3_cached,
                action.c3_start,
            ),
        ]:
            if eid < 0 or start < 0:
                continue
            total_compute += shape.T_task(ntok, cached)
            total_fetch += 0 if cached else shape.T_iter

    dma_util = total_fetch / (makespan * 2) if makespan else 0
    vc_util = total_compute / (makespan * 2) if makespan else 0
    print(f"  DMA util={dma_util:.1%}  VC util={vc_util:.1%}")
    return makespan


def main():
    import sys, io

    OUT_PATH = "/esat/studscratch/r1015673/Thesis/Idea_Model/beam_results.md"

    # ── Tee: 同时写终端和文件 ──────────────────────────────────────
    class _Tee:
        def __init__(self, *files):
            self._files = files

        def write(self, s):
            for f in self._files:
                f.write(s)

        def flush(self):
            for f in self._files:
                f.flush()

    _orig = sys.stdout
    _fout = open(OUT_PATH, "w", encoding="utf-8")
    _fout.write("# Beam Scheduler 调度结果报告\n\n")
    _fout.write(f"> **生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
    _fout.write("> **模型类型**: 解析式解析预测模型（非 RTL 仿真）  \n")
    _fout.write("> **物理参数**: W4A8，expert_weight=4,325,376 B，")
    _fout.write("SRAM 侧两条独立 DMA: xDMA + iDMA，各 64 B/cc，合计 128 B/cc  \n")
    _fout.write("> **DMA 路由**: xDMA 和 iDMA 均可在任意时刻动态连接 C2 或 C3；\n")
    _fout.write(
        "> 甘特图中 SRAM_xDMA/SRAM_iDMA 列显示该通道实际连接的 cluster 和正在搬运的 expert  \n\n"
    )
    _fout.write("```\n")
    sys.stdout = _Tee(_orig, _fout)

    BW = 128  # beam width

    print("=" * 70)
    print("  Beam Scheduler — 解析式调度预测模型  (非 RTL 仿真)")
    print(
        "  SRAM_xDMA / SRAM_iDMA 各 64 B/cc，可动态路由至任意 cluster，总 BW 128 B/cc"
    )
    print("=" * 70)
    print("\nShape 参数 (W4A8, expert=4,325,376 B):")
    print("  [注] DMA 通道粒度=64 B/cc；Shape_A bw_req=32 仍占满 1 条通道 → alloc=64")
    for s in [SHAPE_A, SHAPE_B, SHAPE_C]:
        print(
            f"  {s.name}: M_dim={s.M_dim}  bw_req={s.bw_req:3d} B/cc"
            f"  alloc={s.alloc:3d} B/cc  T_iter={s.T_iter:,d} cc"
        )

    # ══════════════════════════════════════════════════════════
    # A. 基础对照集 (6 cases)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'#'*70}\n  A. 基础对照集\n{'#'*70}")
    run_test("[A1] M=8  hot(6,1,1)", {0: 6, 1: 1, 2: 1}, BW)
    run_test("[A2] M=16 mixed(8,4,2,1,1)", {8: 8, 4: 4, 2: 2, 1: 1, 10: 1}, BW)
    run_test("[A3] M=16 uniform(4×4)", {0: 4, 1: 4, 2: 4, 3: 4}, BW)
    run_test("[A4] M=16 all-cold(1×16)", {i: 1 for i in range(16)}, BW)
    run_test("[A5] M=4  uniform(1×4)", {0: 1, 1: 1, 2: 1, 3: 1}, BW)
    run_test("[A6] M=32 hot(20,6,6)", {0: 20, 1: 6, 2: 6}, BW)

    # ══════════════════════════════════════════════════════════
    # B. 分布形态多样性
    # ══════════════════════════════════════════════════════════
    print(f"\n{'#'*70}\n  B. 分布形态多样性\n{'#'*70}")
    run_test("[B1] M=32 zipf(16,8,4,2,1,1)", {0: 16, 1: 8, 2: 4, 3: 2, 4: 1, 5: 1}, BW)
    run_test("[B2] M=64 uniform(8×8)", {i: 8 for i in range(8)}, BW)
    run_test(
        "[B3] M=64 zipf(32,16,8,4,2,1,1)",
        {0: 32, 1: 16, 2: 8, 3: 4, 4: 2, 5: 1, 6: 1},
        BW,
    )
    run_test("[B4] M=16 dual-hot(8,8)", {0: 8, 1: 8}, BW)
    run_test("[B5] M=48 mixed(20,10,8,6,4)", {0: 20, 1: 10, 2: 8, 3: 6, 4: 4}, BW)
    run_test(
        "[B6] M=20 fragmented(5,4,3,3,2,1,1,1)",
        {0: 5, 1: 4, 2: 3, 3: 3, 4: 2, 5: 1, 6: 1, 7: 1},
        BW,
    )
    run_test("[B7] M=24 bimodal(12,12)", {0: 12, 1: 12}, BW)
    run_test(
        "[B8] M=32 bimodal+tail(16,12,2,1,1)", {0: 16, 1: 12, 2: 2, 3: 1, 4: 1}, BW
    )
    run_test("[B9] M=40 heavy-head(30,5,3,2)", {0: 30, 1: 5, 2: 3, 3: 2}, BW)
    run_test(
        "[B10] M=36 ladder(12,8,6,4,3,2,1)",
        {0: 12, 1: 8, 2: 6, 3: 4, 4: 3, 5: 2, 6: 1},
        BW,
    )

    # ══════════════════════════════════════════════════════════
    # C. 边界 / 极端场景
    # ══════════════════════════════════════════════════════════
    print(f"\n{'#'*70}\n  C. 边界 / 极端场景\n{'#'*70}")
    run_test("[C1] M=1  single token", {0: 1}, BW)
    run_test("[C2] M=2  two experts", {0: 1, 1: 1}, BW)
    run_test("[C3] M=32 single expert", {0: 32}, BW)
    run_test("[C4] M=64 single expert", {0: 64}, BW)
    run_test("[C5] M=8  micro(2,2,2,1,1)", {0: 2, 1: 2, 2: 2, 3: 1, 4: 1}, BW)
    run_test("[C6] M=3  asymm(2,1)", {0: 2, 1: 1}, BW)
    run_test("[C7] M=5  odd-sum(3,2)", {0: 3, 1: 2}, BW)
    run_test("[C8] M=16 all-one-expert", {0: 16}, BW)
    run_test("[C9] M=15 prime-frag(5,4,3,2,1)", {0: 5, 1: 4, 2: 3, 3: 2, 4: 1}, BW)
    run_test("[C10] M=7  triple(3,2,2)", {0: 3, 1: 2, 2: 2}, BW)

    # ══════════════════════════════════════════════════════════
    # D. 大 Batch 压力测试
    # ══════════════════════════════════════════════════════════
    print(f"\n{'#'*70}\n  D. 大 Batch 压力测试\n{'#'*70}")
    run_test("[D1] M=128 uniform(16×8)", {i: 16 for i in range(8)}, BW)
    run_test(
        "[D2] M=128 zipf(64,32,16,8,4,2,2)",
        {0: 64, 1: 32, 2: 16, 3: 8, 4: 4, 5: 2, 6: 2},
        BW,
    )
    run_test(
        "[D3] M=64  hot+tail(40,8,8,4,2,2)", {0: 40, 1: 8, 2: 8, 3: 4, 4: 2, 5: 2}, BW
    )
    run_test("[D4] M=256 uniform(32×8)", {i: 32 for i in range(8)}, BW)
    run_test("[D5] M=32  many-experts(2×16)", {i: 2 for i in range(16)}, BW)
    run_test("[D6] M=48  many-experts(3×16)", {i: 3 for i in range(16)}, BW)
    run_test(
        "[D7] M=96  zipf(48,24,12,6,3,2,1)",
        {0: 48, 1: 24, 2: 12, 3: 6, 4: 3, 5: 2, 6: 1},
        BW,
    )

    # ══════════════════════════════════════════════════════════
    # E. Cache-Hit 场景 (预热状态)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'#'*70}\n  E. Cache-Hit 场景\n{'#'*70}")
    # E1: C2 已缓存最热 expert，测试 cache-hit 是否被正确利用
    run_test("[E1] M=16 hot E0 cached@C2", {0: 8, 1: 4, 2: 4}, BW, cached_c2=0)
    # E2: 两个 cluster 各自缓存不同 expert → 同时命中
    run_test(
        "[E2] M=16 dual-hot both-cached", {0: 8, 1: 8}, BW, cached_c2=0, cached_c3=1
    )
    # E3: Cache 命中后还有大量冷 expert
    run_test("[E3] M=24 cached+cold", {0: 8, 1: 6, 2: 6, 3: 4}, BW, cached_c2=0)
    # E4: 错误缓存（cache expert 不在本次调度中） → 等价冷启动
    run_test("[E4] M=8  stale-cache (E99 not in batch)", {0: 4, 1: 4}, BW, cached_c2=99)
    # E5: 连续 batch 中热 expert 连续被相同 cluster 处理
    run_test(
        "[E5] M=32 hot-expert repeated, C2-cached", {0: 20, 1: 6, 2: 6}, BW, cached_c2=0
    )
    # E6: 两 cluster 都缓存同一 expert（不太可能，验证正确性）
    run_test(
        "[E6] M=16 both-cache same E0", {0: 8, 1: 4, 2: 4}, BW, cached_c2=0, cached_c3=0
    )

    # ══════════════════════════════════════════════════════════
    # F. Expert-Split 效果对比
    # ══════════════════════════════════════════════════════════
    print(f"\n{'#'*70}\n  F. Expert-Split 效果对比\n{'#'*70}")
    split_cases = [
        ("M=32 hot(20,6,6)", {0: 20, 1: 6, 2: 6}),
        ("M=64 hot(40,12,12)", {0: 40, 1: 12, 2: 12}),
        ("M=24 hot(20,2,2)", {0: 20, 1: 2, 2: 2}),
        ("M=32 single", {0: 32}),
        ("M=40 heavy-head(30,5,3,2)", {0: 30, 1: 5, 2: 3, 3: 2}),
    ]
    for name, dist in split_cases:
        ms_on, _ = BeamScheduler(dist, beam_width=BW, allow_split=True).run()
        ms_off, _ = BeamScheduler(dist, beam_width=BW, allow_split=False).run()
        gain = (ms_off - ms_on) / ms_off * 100 if ms_off else 0
        print(
            f"  {name:<30}: split={ms_on:>9,}cc  no-split={ms_off:>9,}cc  "
            f"gain={gain:+.1f}%"
        )

    # ══════════════════════════════════════════════════════════
    # G. WAIT-TO-PAIR 效果验证
    # ══════════════════════════════════════════════════════════
    print(f"\n{'#'*70}\n  G. WAIT-TO-PAIR 效果验证\n{'#'*70}")
    print("  (对比加入 WAIT-PAIR 前后的 makespan，需对比旧代码)")
    wait_cases = [
        ("[G1] M=16 mixed(8,4,2,1,1)", {8: 8, 4: 4, 2: 2, 1: 1, 10: 1}),
        ("[G2] M=48 mixed(20,10,8,6,4)", {0: 20, 1: 10, 2: 8, 3: 6, 4: 4}),
        ("[G3] M=32 hot(20,6,6)", {0: 20, 1: 6, 2: 6}),
    ]
    for name, dist in wait_cases:
        ms, acts = BeamScheduler(dist, beam_width=BW, allow_split=True).run()
        wait_used = any("WAIT" in a.tag for a in acts)
        print(f"  {name}: makespan={ms:,d}cc  WAIT策略被采用={wait_used}")

    # ══════════════════════════════════════════════════════════
    # H. Beam Width 灵敏度分析
    # ══════════════════════════════════════════════════════════
    print(f"\n{'#'*70}\n  H. Beam Width 灵敏度\n{'#'*70}")
    sensitivity_cases = [
        ("M=32 hot(20,6,6)", {0: 20, 1: 6, 2: 6}),
        ("M=64 zipf(32,16,8,4,2,1,1)", {0: 32, 1: 16, 2: 8, 3: 4, 4: 2, 5: 1, 6: 1}),
    ]
    for name, dist in sensitivity_cases:
        print(f"\n  {name}:")
        for bw in [1, 4, 16, 64, 256]:
            t0 = time.perf_counter()
            ms, _ = BeamScheduler(dist, beam_width=bw, allow_split=True).run()
            print(
                f"    beam={bw:4d}: makespan={ms:,d} cc  ({time.perf_counter()-t0:.3f}s)"
            )

    # ── 恢复 stdout，关闭文件 ─────────────────────────────────
    sys.stdout = _orig
    _fout.write("```\n")
    _fout.close()
    print(f"\n[完整报告（含所有甘特图）已保存 → {OUT_PATH}]")


if __name__ == "__main__":
    main()
