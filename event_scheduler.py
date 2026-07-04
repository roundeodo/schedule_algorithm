#!/usr/bin/env python3
"""
事件驱动调度器 (v22) — 单 cluster 决策 + BW 感知
============================================================

核心原则:
  A) 单 cluster 决策: 每次 cluster 空闲时调用一次调度器, 决定它下一步做什么.
     输入 = (剩余 experts, cache_map, peer 当前 phase, peer 预计结束时刻, DMA 通道状态).
     输出 = (eid, ntok, shape, dma_mode, load_bw, resident).

  B) BW 感知: 根据 peer 状态精确选择 load_bw
       peer busy + 用 DMA      → 我 load_bw = 64   (并行单通道)
       peer busy + 不用 DMA    → 我 resident
       peer 在 swiglu/driven   → 我 load_bw = 128  (DMA 此刻空闲)
       peer idle (算完)        → 我 load_bw = 128

  C) Cache 优先 + 配对优化:
     - 若我空闲且某未处理 expert 已在我的 TCDM → 直接 resident (0 DMA).
     - 若 peer 正在跑 resident expert, 我可独占 128 DMA_BOTH 跑冷门 ntok<4 expert.
     - 若 peer 也在流式 → 我们都限 64, 此时选择 ntok ≥ 4 的 expert 最划算
       (compute-bound, 两边同时收敛).

  D) 形状选择用 shape_lut.pick_shape(ntok, load_bw, resident).

物理硬件约束 (重要!):
  sram_xDMA 64B/cc, 同一时刻仅连 1 个 cluster.
  iDMA      64B/cc, 同一时刻仅连 1 个 cluster.
  合计      128B/cc, 但必须是两条独立通道.
  "DMA_BOTH (load_bw=128)" ≡ xDMA + iDMA 同时连到我一个 cluster.
  因此 DMA_BOTH 期间 peer 不能用 DMA — peer 必须 resident 或 idle.
"""

import math
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from config import (
    VersaCoreShape,
    SystemConfig,
    MoELayerConfig,
    ClusterConfig,
)
from shape_lut import pick_shape, estimate_cc, shape_bw_demand

# ============================================================
#  数据结构
# ============================================================


@dataclass
class Task:
    """一次派发到一个 cluster 的任务 (已决定 shape/DMA/BW)."""

    eid: int
    ntok: int
    cid: int
    shape: VersaCoreShape
    dma_mode: str  # 'xdma' | 'idma' | 'both' | 'none'
    load_bw: int  # 64 / 128 / 0
    resident: bool
    est_cc: int
    util: float
    rationale: str = ""

    # 事件时间戳 (执行期填充)
    start: int = 0
    gu_dma_end: int = 0  # gate+up DMA 释放时刻
    swig_end: int = 0  # swiglu 结束时刻 (down DMA 可申请)
    dn_dma_end: int = 0  # down DMA 释放时刻
    end: int = 0


@dataclass
class Plan:
    tasks: List[Task] = field(default_factory=list)
    strategy: str = "event_driven_v22"
    makespan: int = 0
    c2_end: int = 0
    c3_end: int = 0
    exit_eids: Dict[int, int] = field(default_factory=dict)


# ============================================================
#  硬件状态 (调度器运行时追踪)
# ============================================================


@dataclass
class HWState:
    """每次决策时刻的硬件快照."""

    now: int = 0
    c2_free_at: int = 0
    c3_free_at: int = 0
    c2_resident_eid: int = -1  # C2 当前驻留的 expert (≥0 表权重在)
    c3_resident_eid: int = -1
    xdma_free_at: int = 0
    idma_free_at: int = 0
    # peer phase 预测 (用于决定我是否能用 DMA_BOTH)
    c2_dma_end: int = 0  # C2 当前任务的 dn_dma_end (之后 DMA 才真空闲)
    c3_dma_end: int = 0
    c2_dma_mode: str = "none"  # C2 最近任务 dma 模式 ('xdma'/'idma'/'both'/'none')
    c3_dma_mode: str = "none"


# ============================================================
#  决策核心: 为单个 cluster 选择下一个任务
# ============================================================


def _dma_available_bw_for(
    cid: int,
    hw: HWState,
    remaining_count: int,
    K: int = 2048,
    N: int = 1408,
    wpe: float = 0.5,
) -> Tuple[int, str]:
    """
    计算 cid 开始新任务时可用 BW, 返回 (load_bw, dma_mode).

    决策策略 (关键修复: 避免 first-mover 独占 128 伤 peer):
      1. 若 peer 正在 DMA (now < peer_dma_end):
           peer 若占 xdma → 我拿 idma (64)
           peer 若占 idma → 我拿 xdma (64)
           peer 若占 both → 我必须等或走 resident  (此处返回 0 bw 触发等待)
      2. 若 peer DMA 空:
           a) remaining_count >= 2 且 peer 即将被调度 → 我只拿 xdma@64,
              留 idma 给 peer 形成并行 (全局最优于 DMA-bound 场景).
           b) remaining_count == 1 (最后一个 expert) 或 peer 已无任务
              → 我独占 both@128 加速.
    """
    peer = 3 if cid == 2 else 2
    peer_dma_end = hw.c2_dma_end if peer == 2 else hw.c3_dma_end
    peer_dma_mode = hw.c2_dma_mode if peer == 2 else hw.c3_dma_mode
    peer_free_at = hw.c2_free_at if peer == 2 else hw.c3_free_at
    peer_busy = peer_free_at > hw.now

    if hw.now < peer_dma_end:
        if peer_dma_mode == "xdma":
            return 64, "idma"
        if peer_dma_mode == "idma":
            return 64, "xdma"
        # peer 占 both → 我无 DMA
        return 0, "wait"

    # peer DMA 空.
    # 关键情形: peer 正在跑 resident 任务 (不占 DMA) → 我独占 128!
    if peer_busy and peer_dma_mode == "none":
        return 128, "both"

    # peer 即将也来决策? 看 remaining 是否有多余 expert 给 peer.
    # 若 remaining_count >= 2 且 peer 即将空闲 (peer_free_at 近),
    #    peer 会拿走另一个 → 我留 idma 给 peer, 自己拿 xdma@64.
    # 若 remaining_count == 1 → 独占 both, 没人跟我抢.
    if remaining_count <= 1:
        return 128, "both"
    return 64, "xdma"


def _choose_task_for_cluster(
    cid: int,
    remaining: List[Tuple[int, int]],
    hw: HWState,
    K: int = 2048,
    N: int = 1408,
    wpe: float = 0.5,
) -> Optional[Task]:
    """
    为 cid 从 remaining 列表中选一个 expert+shape.
    返回 Task 或 None (若 remaining 为空).

    决策优先级:
      1. Cache hit: 若 cid 有驻留 expert 且在 remaining 中 → resident
      2. BW 感知的 expert 选择:
         a) 若我独占 128: 优先 ntok<4 的冷门 (cc 减半), 否则最热门
         b) 若我只有 64 : 优先 ntok≥4 热门 (compute-bound, 不浪费 BW)
      3. 选最优 shape.
    """
    if not remaining:
        return None

    # --- 1) Cache hit ---
    my_resident = hw.c2_resident_eid if cid == 2 else hw.c3_resident_eid
    for eid, ntok in remaining:
        if eid == my_resident:
            shape, cc, util, _ = pick_shape(ntok, 0, resident=True, K=K, N=N, wpe=wpe)
            return Task(
                eid=eid,
                ntok=ntok,
                cid=cid,
                shape=shape,
                dma_mode="none",
                load_bw=0,
                resident=True,
                est_cc=cc,
                util=util,
                rationale=f"cache-hit on C{cid}",
            )

    # --- 2) BW 感知 ---
    load_bw, dma_mode = _dma_available_bw_for(
        cid, hw, remaining_count=len(remaining), K=K, N=N, wpe=wpe
    )

    # 若 peer 占 both, 我暂时无 DMA → 选最冷门 idle-wait
    # (由调度主循环让我 free_at 推迟到 peer_dma_end)
    if load_bw == 0:
        # 强制等 peer DMA 结束, 之后按 128 规划
        load_bw = 128
        dma_mode = "both"

    # 候选打分: 每个 expert 在当前 load_bw 下最优 shape 的 cc
    # 策略: 在 load_bw=128 时, 若存在 ntok<4 的 cold expert 就先吃它 (2× 加速)
    #       在 load_bw=64  时, 优先最热门 (让 peer 有 swiglu slack 可抢)
    if load_bw == 128:
        # 找 ntok<4 的冷门, 如果没有就选最热门
        cold = [(eid, ntok) for eid, ntok in remaining if ntok < 4]
        if cold:
            cold.sort(key=lambda x: x[1])  # 最冷先吃 (DMA-bound 最严重)
            eid, ntok = cold[0]
            rationale = f"hot-BW=128 吃冷门 ntok={ntok}"
        else:
            # 没冷门 → 选最大 (更好利用 compute)
            eid, ntok = max(remaining, key=lambda x: x[1])
            rationale = f"hot-BW=128 选最热 ntok={ntok}"
    else:
        # load_bw=64 → 并行流式, 选 ntok≥4 更划算
        big = [(eid, ntok) for eid, ntok in remaining if ntok >= 4]
        if big:
            eid, ntok = max(big, key=lambda x: x[1])
            rationale = f"parallel-BW=64 选 ntok={ntok}"
        else:
            # 只剩小 ntok, 选最大
            eid, ntok = max(remaining, key=lambda x: x[1])
            rationale = f"parallel-BW=64 仅剩小 ntok={ntok}"

    shape, cc, util, bw_demand = pick_shape(
        ntok, load_bw, resident=False, K=K, N=N, wpe=wpe
    )

    return Task(
        eid=eid,
        ntok=ntok,
        cid=cid,
        shape=shape,
        dma_mode=dma_mode,
        load_bw=load_bw,
        resident=False,
        est_cc=cc,
        util=util,
        rationale=rationale
        + f" shape={shape.meshRow}×{shape.tileSize}×{shape.meshCol} bw={load_bw}",
    )


# ============================================================
#  Pair 枚举 (v24: both-idle 全局配对)
# ============================================================


def _expert_mode_options(
    eid: int, ntok: int, cid: int, hw: HWState, K: int, N: int, wpe: float
):
    """返回 cid 执行 expert(eid,ntok) 的候选模式列表.
    每项: (mode_tag, load_bw, shape, cc, util, dma_mode).
    物理 BW 占用 = load_bw (0 / 64 / 128).
    """
    opts = []
    my_resident = hw.c2_resident_eid if cid == 2 else hw.c3_resident_eid
    if eid == my_resident:
        s, cc, u, _ = pick_shape(ntok, 0, True, K, N, wpe)
        opts.append(("resident", 0, s, cc, u, "none"))
    s, cc, u, _ = pick_shape(ntok, 64, False, K, N, wpe)
    dma_single = "xdma" if cid == 2 else "idma"
    opts.append(("stream64", 64, s, cc, u, dma_single))
    s, cc, u, _ = pick_shape(ntok, 128, False, K, N, wpe)
    opts.append(("stream128", 128, s, cc, u, "both"))
    return opts


def _pair_single_cc(ntok: int, K: int, N: int, wpe: float) -> int:
    """stream@64 下单 expert cc, 用于估算剩余池 lower-bound."""
    _, cc, _, _ = pick_shape(ntok, 64, False, K, N, wpe)
    return cc


def _enumerate_pair(
    remaining: List[Tuple[int, int]], hw: HWState, K: int, N: int, wpe: float
):
    """枚举 (i,j) 及各自 mode, 选配对后全局 makespan 最小.

    约束:
      - bw_A + bw_B ≤ 128 (物理 DMA 通道总和)
      - 若 A 选 stream128, B 必须 resident (独占 both)
      - 若 B 选 stream128, A 必须 resident (独占 both)

    评分:
      pair_cc = max(cc_A, cc_B)
      rest_lb = ceil(sum(stream64_cc(rest)) / 2)   # 剩余理想 2-bin 均分
      total   = pair_cc + rest_lb
      tie: 较小 imbalance = |cc_A - cc_B|

    返回 (i, j, optA, optB) 或 None.
    """
    if len(remaining) < 2:
        return None

    # 预算剩余池 cc (每个都用 stream64 估)
    single_cc = [_pair_single_cc(n, K, N, wpe) for _, n in remaining]

    # Cache-hit 约束 (v24): 若某 expert 在 C2/C3 的 resident 位, 强制它参与本 pair.
    # 理由: 下一步若该 cluster 跑其他 expert, 会覆盖 TCDM → cache 失效.
    cache_hit_idx = None
    cache_hit_cid = None
    for k, (eid, _) in enumerate(remaining):
        if eid == hw.c2_resident_eid and hw.c2_resident_eid >= 0:
            cache_hit_idx = k
            cache_hit_cid = 2
            break
        if eid == hw.c3_resident_eid and hw.c3_resident_eid >= 0:
            cache_hit_idx = k
            cache_hit_cid = 3
            break

    best = None
    n = len(remaining)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Cache-hit 约束: 命中 expert 必须在正确 cluster 上, 或作为 pair 的另一端
            if cache_hit_idx is not None:
                if cache_hit_cid == 2 and i != cache_hit_idx:
                    continue
                if cache_hit_cid == 3 and j != cache_hit_idx:
                    continue
            eidA, ntokA = remaining[i]
            eidB, ntokB = remaining[j]
            optsA = _expert_mode_options(eidA, ntokA, 2, hw, K, N, wpe)
            optsB = _expert_mode_options(eidB, ntokB, 3, hw, K, N, wpe)
            for mA in optsA:
                for mB in optsB:
                    bw_total = mA[1] + mB[1]
                    if bw_total > 128:
                        continue
                    # stream128 独占: peer 必须 resident (0 bw)
                    if mA[0] == "stream128" and mB[0] != "resident":
                        continue
                    if mB[0] == "stream128" and mA[0] != "resident":
                        continue
                    cc_a, cc_b = mA[3], mB[3]
                    pair_cc = max(cc_a, cc_b)
                    rest_sum = sum(single_cc[k] for k in range(n) if k != i and k != j)
                    rest_lb = math.ceil(rest_sum / 2)
                    imbalance = abs(cc_a - cc_b)
                    total = pair_cc + rest_lb
                    key = (total, imbalance, -(mA[4] + mB[4]))
                    if best is None or key < best[0]:
                        best = (key, i, j, mA, mB)
    return best


def _make_task_from_opt(eid: int, ntok: int, cid: int, opt, rationale: str) -> Task:
    mode_tag, load_bw, shape, cc, util, dma_mode = opt
    return Task(
        eid=eid,
        ntok=ntok,
        cid=cid,
        shape=shape,
        dma_mode=dma_mode,
        load_bw=load_bw,
        resident=(mode_tag == "resident"),
        est_cc=cc,
        util=util,
        rationale=rationale,
    )


# ============================================================
#  事件驱动主循环
# ============================================================


def _execute_task(task: Task, hw: HWState, moe_cfg, wpe: float, K: int):
    """填充 task 时间戳并更新 hw state."""
    # DMA start: 必须 >= hw.now, cluster 前一个任务的结束, 以及 DMA 通道空闲时刻
    cluster_free = hw.c2_free_at if task.cid == 2 else hw.c3_free_at
    dma_start = max(hw.now, cluster_free)
    if task.dma_mode in ("xdma", "both"):
        dma_start = max(dma_start, hw.xdma_free_at)
    if task.dma_mode in ("idma", "both"):
        dma_start = max(dma_start, hw.idma_free_at)
    task.start = dma_start

    if task.resident:
        task.gu_dma_end = dma_start
        task.swig_end = dma_start + task.est_cc
        task.dn_dma_end = dma_start
        task.end = dma_start + task.est_cc
    else:
        gu_bytes = moe_cfg.expert_gate_weight + moe_cfg.expert_up_weight
        dn_bytes = moe_cfg.expert_down_weight
        gu_dma = math.ceil(gu_bytes / task.load_bw) if task.load_bw > 0 else 0
        dn_dma = math.ceil(dn_bytes / task.load_bw) if task.load_bw > 0 else 0
        K_half = K // 2
        N = 1408
        gu_stage = math.ceil(task.est_cc * K / (K + K_half + 1))
        sw_cc = math.ceil(task.ntok * N / 128)
        task.gu_dma_end = dma_start + max(gu_dma, 10)
        task.swig_end = dma_start + gu_stage + sw_cc
        dn_dma_start = max(
            task.swig_end,
            hw.xdma_free_at if task.dma_mode in ("xdma", "both") else 0,
            hw.idma_free_at if task.dma_mode in ("idma", "both") else 0,
        )
        task.dn_dma_end = dn_dma_start + dn_dma
        task.end = max(dma_start + task.est_cc, task.dn_dma_end + 10)

    # 更新 hw
    cid = task.cid
    if cid == 2:
        hw.c2_free_at = task.end
        hw.c2_resident_eid = task.eid
        hw.c2_dma_end = task.dn_dma_end
        hw.c2_dma_mode = task.dma_mode
    else:
        hw.c3_free_at = task.end
        hw.c3_resident_eid = task.eid
        hw.c3_dma_end = task.dn_dma_end
        hw.c3_dma_mode = task.dma_mode
    if task.dma_mode in ("xdma", "both"):
        hw.xdma_free_at = task.dn_dma_end
    if task.dma_mode in ("idma", "both"):
        hw.idma_free_at = task.dn_dma_end


def schedule_event_driven(
    token_dist: Dict[int, int],
    sys_cfg: SystemConfig,
    moe_cfg: MoELayerConfig,
    cached_map: Optional[Dict[int, int]] = None,
    start_time: int = 0,
) -> Plan:
    """
    主入口: 用事件驱动单 cluster 决策模式生成调度计划.

    流程:
      1. remaining = list(token_dist.items()) (按 ntok 降序)
      2. HWState 初始化: c2/c3 的 resident_eid 来自 cached_map
      3. 循环: 挑 (c2_free_at, c3_free_at) 更小的那个 cluster 作为"当前空闲者"
               调 _choose_task_for_cluster 生成 task
               用 estimate_cc 细化时间 (含 phase 分离)
               更新 hw state
      4. 直到 remaining 为空
    """
    K = moe_cfg.hidden_size
    N = moe_cfg.moe_intermediate_size
    wpe = moe_cfg.wpe

    cached_map = cached_map or {}
    remaining = sorted(token_dist.items(), key=lambda x: -x[1])

    hw = HWState(
        now=start_time,
        c2_free_at=start_time,
        c3_free_at=start_time,
        c2_resident_eid=next((e for e, c in cached_map.items() if c == 2), -1),
        c3_resident_eid=next((e for e, c in cached_map.items() if c == 3), -1),
        xdma_free_at=start_time,
        idma_free_at=start_time,
        c2_dma_end=start_time,
        c3_dma_end=start_time,
    )

    plan = Plan()

    while remaining:
        # 选更早空闲的 cluster
        if hw.c2_free_at <= hw.c3_free_at:
            cid = 2
            my_free = hw.c2_free_at
        else:
            cid = 3
            my_free = hw.c3_free_at
        hw.now = my_free

        # [Expert-split / Clone opt] 两 cluster 同时空闲时尝试优化
        peer_cid = 3 if cid == 2 else 2
        peer_free = hw.c3_free_at if cid == 2 else hw.c2_free_at
        both_idle = (abs(peer_free - my_free) < 5) and len(remaining) >= 1
        if both_idle:
            hot_eid, hot_ntok = remaining[0]

            # --- [SPLIT opt] hot ntok ∈ {5,6,7}:
            # 同 cluster 内 "4-stream + (hot-4)-resident", peer 并行跑次热.
            # 原理: 第二 pass 权重已 DMA 进 TCDM, resident 无需 DMA, 速度快 4×.
            if (
                5 <= hot_ntok <= 7
                and hot_eid != hw.c2_resident_eid
                and hot_eid != hw.c3_resident_eid
            ):
                first_nt = 4
                second_nt = hot_ntok - first_nt
                s1, cc1, u1, _ = pick_shape(first_nt, 64, False, K, N, wpe)
                s2, cc2, u2, _ = pick_shape(second_nt, 0, True, K, N, wpe)
                split_cc_hot = cc1 + cc2  # C2 做 hot 的完整时间

                rest = remaining[1:]
                # stream@64 cc: 用于 NORMAL/CLONE 的并行 rest 估算
                rest_cc_64 = [pick_shape(n, 64, False, K, N, wpe)[1] for _, n in rest]
                # SPLIT 专用 rest_cc: 关键洞察 (v24)
                #   SPLIT 第二阶段 (resident) 不占 DMA, peer 此时可独占 128B/cc
                #   跑第一个 cold (cc 减半). 之后若还有 cold, 恢复 stream@64.
                rest_cc_split = []
                for idx, (_, n) in enumerate(rest):
                    bw = 128 if idx == 0 else 64
                    rest_cc_split.append(pick_shape(n, bw, False, K, N, wpe)[1])
                rest_cc = rest_cc_64  # NORMAL/CLONE 用基准

                # SPLIT: C2 跑 hot (split_cc_hot), C3 串行 rest (C3 自由分配)
                # 实际 C3 处理 rest 时 C2 在 split_cc_hot 后空闲, 会帮 C3 分担.
                # 简化: 两 cluster 可用总 slot = split_cc_hot (C2) + M_total (C3)
                #       但 C3 可能先完, 贡献剩余 slot.
                #       Greedy list-scheduling 估计:
                work_list = [split_cc_hot] + rest_cc_split
                # 将 work_list 按降序装 2 bin (LPT 算法)
                bins = [0, 0]
                for w in sorted(work_list, reverse=True):
                    idx = 0 if bins[0] <= bins[1] else 1
                    bins[idx] += w
                split_makespan = max(bins)

                # CLONE: 两 cluster 各 half_hot, 然后共同并行 rest
                half_a = hot_ntok // 2
                half_b = hot_ntok - half_a
                _, cca, _, _ = pick_shape(half_a, 64, False, K, N, wpe)
                _, ccb, _, _ = pick_shape(half_b, 64, False, K, N, wpe)
                clone_hot = max(cca, ccb)
                work_list_c = [clone_hot, clone_hot] + rest_cc
                bins_c = [0, 0]
                for w in sorted(work_list_c, reverse=True):
                    idx = 0 if bins_c[0] <= bins_c[1] else 1
                    bins_c[idx] += w
                clone_total = max(bins_c)

                # NORMAL: C2 hot || C3 next, 然后并行剩余
                _, cc_hot_full, _, _ = pick_shape(hot_ntok, 64, False, K, N, wpe)
                work_list_n = [cc_hot_full] + rest_cc
                bins_n = [0, 0]
                for w in sorted(work_list_n, reverse=True):
                    idx = 0 if bins_n[0] <= bins_n[1] else 1
                    bins_n[idx] += w
                normal_total = max(bins_n)

                # 选 split 的条件: 严格优于 clone 与 normal
                if split_makespan < clone_total and split_makespan < normal_total:
                    # 生成 split tasks
                    ta = Task(
                        eid=hot_eid,
                        ntok=first_nt,
                        cid=cid,
                        shape=s1,
                        dma_mode="xdma",
                        load_bw=64,
                        resident=False,
                        est_cc=cc1,
                        util=u1,
                        rationale=f"SPLIT-stream4 E{hot_eid}",
                    )
                    tb = Task(
                        eid=hot_eid,
                        ntok=second_nt,
                        cid=cid,
                        shape=s2,
                        dma_mode="none",
                        load_bw=0,
                        resident=True,
                        est_cc=cc2,
                        util=u2,
                        rationale=f"SPLIT-resident{second_nt} E{hot_eid}",
                    )
                    _execute_task(ta, hw, moe_cfg, wpe, K)
                    _execute_task(tb, hw, moe_cfg, wpe, K)
                    plan.tasks.extend([ta, tb])

                    # C3 并行跑次热 (若存在), 强制 idma@64 (xdma 已被 ta 占)
                    if len(remaining) >= 2:
                        next_eid, next_ntok = remaining[1]
                        s3, cc3, u3, _ = pick_shape(next_ntok, 64, False, K, N, wpe)
                        tc = Task(
                            eid=next_eid,
                            ntok=next_ntok,
                            cid=peer_cid,
                            shape=s3,
                            dma_mode="idma",
                            load_bw=64,
                            resident=False,
                            est_cc=cc3,
                            util=u3,
                            rationale=f"SPLIT-peer ntok={next_ntok}",
                        )
                        _execute_task(tc, hw, moe_cfg, wpe, K)
                        plan.tasks.append(tc)
                        remaining = [
                            (e, n) for e, n in remaining if e not in (hot_eid, next_eid)
                        ]
                    else:
                        remaining = [(e, n) for e, n in remaining if e != hot_eid]
                    continue

            # 仅当 hot ntok>=6 (拆分后每侧>=3, 可填满 4×8×8 M=4) 时才考虑 clone
            if (
                hot_ntok >= 6
                and hot_eid != hw.c2_resident_eid
                and hot_eid != hw.c3_resident_eid
            ):
                half_a = hot_ntok // 2
                half_b = hot_ntok - half_a
                # 估 clone: 两 cluster 各跑一半, 各用 xdma/idma@64
                sa, cca, ua, _ = pick_shape(half_a, 64, False, K, N, wpe)
                sb, ccb, ub, _ = pick_shape(half_b, 64, False, K, N, wpe)
                clone_cc = max(cca, ccb)
                # 估 normal: C2 拿 hot, C3 拿次热 (并行)
                normal_cc_hot, _ = estimate_cc(
                    hot_ntok,
                    K,
                    N,
                    pick_shape(hot_ntok, 64, False, K, N, wpe)[0],
                    64,
                    wpe,
                    False,
                )
                if len(remaining) >= 2:
                    _, nt2 = remaining[1]
                    normal_cc_cold, _ = estimate_cc(
                        nt2,
                        K,
                        N,
                        pick_shape(nt2, 64, False, K, N, wpe)[0],
                        64,
                        wpe,
                        False,
                    )
                    normal_cc = max(normal_cc_hot, normal_cc_cold)
                else:
                    normal_cc = normal_cc_hot
                # clone 胜出才用 (clone 消耗 1 个 expert, normal 消耗 2 个)
                # 考虑后续 expert 继续并行处理, clone 值 = clone_cc + remaining_cost
                # 简化: 只要 clone_cc < normal_cc * 0.85 就 clone
                if clone_cc < normal_cc * 0.85 or len(remaining) == 1:
                    # 生成两个 half-task
                    ta = Task(
                        eid=hot_eid,
                        ntok=half_a,
                        cid=2,
                        shape=sa,
                        dma_mode="xdma",
                        load_bw=64,
                        resident=False,
                        est_cc=cca,
                        util=ua,
                        rationale=f"CLONE E{hot_eid} half={half_a}",
                    )
                    tb = Task(
                        eid=hot_eid,
                        ntok=half_b,
                        cid=3,
                        shape=sb,
                        dma_mode="idma",
                        load_bw=64,
                        resident=False,
                        est_cc=ccb,
                        util=ub,
                        rationale=f"CLONE E{hot_eid} half={half_b}",
                    )
                    # 执行 ta / tb
                    _execute_task(ta, hw, moe_cfg, wpe, K)
                    _execute_task(tb, hw, moe_cfg, wpe, K)
                    plan.tasks.append(ta)
                    plan.tasks.append(tb)
                    remaining = [(e, n) for e, n in remaining if e != hot_eid]
                    continue

        # [PAIR opt v24] both_idle 且 remaining>=2 → 枚举 (eA,eB) 配对.
        # 准则: 总 bw ≤ 128, max(cc) 最小, 同时兼顾剩余池 2-bin 均分.
        if both_idle and len(remaining) >= 2:
            pair = _enumerate_pair(remaining, hw, K, N, wpe)
            if pair is not None:
                _, iA, iB, optA, optB = pair
                eidA, ntokA = remaining[iA]
                eidB, ntokB = remaining[iB]
                tA = _make_task_from_opt(
                    eidA, ntokA, 2, optA, f"PAIR C2 mode={optA[0]} ntok={ntokA}"
                )
                tB = _make_task_from_opt(
                    eidB, ntokB, 3, optB, f"PAIR C3 mode={optB[0]} ntok={ntokB}"
                )
                _execute_task(tA, hw, moe_cfg, wpe, K)
                _execute_task(tB, hw, moe_cfg, wpe, K)
                plan.tasks.extend([tA, tB])
                drop = {eidA, eidB}
                remaining = [(e, n) for e, n in remaining if e not in drop]
                continue

        task = _choose_task_for_cluster(cid, remaining, hw, K=K, N=N, wpe=wpe)
        if task is None:
            break

        _execute_task(task, hw, moe_cfg, wpe, K)

        # 从 remaining 移除
        remaining = [(e, n) for e, n in remaining if e != task.eid]
        plan.tasks.append(task)

    plan.c2_end = hw.c2_free_at
    plan.c3_end = hw.c3_free_at
    plan.makespan = max(plan.c2_end, plan.c3_end)
    # exit_eids: 每 cluster 最后 task 的 eid (可供下一层 cache)
    c2_last = [t for t in plan.tasks if t.cid == 2]
    c3_last = [t for t in plan.tasks if t.cid == 3]
    if c2_last:
        plan.exit_eids[2] = c2_last[-1].eid
    if c3_last:
        plan.exit_eids[3] = c3_last[-1].eid

    return plan


# ============================================================
#  调试 / 打印
# ============================================================


def print_plan(plan: Plan):
    print(
        f"=== Plan: {plan.strategy} makespan={plan.makespan:,}cc "
        f"C2={plan.c2_end:,} C3={plan.c3_end:,} ==="
    )
    for t in plan.tasks:
        shape_str = f"{t.shape.meshRow}×{t.shape.tileSize}×{t.shape.meshCol}"
        print(
            f"  C{t.cid} E{t.eid:2d} ntok={t.ntok:2d} "
            f"shape={shape_str:>8} dma={t.dma_mode:>4} bw={t.load_bw:>3} "
            f"cc={t.est_cc:>7,} util={t.util:5.1%} | {t.rationale}"
        )


if __name__ == "__main__":
    from config import SystemConfig, MoELayerConfig

    sys_cfg = SystemConfig.default_4cluster()
    moe_cfg = MoELayerConfig()

    print("\n--- Case 1: M=4 uniform (1,1,1,1) ---")
    plan = schedule_event_driven({0: 1, 1: 1, 2: 1, 3: 1}, sys_cfg, moe_cfg)
    print_plan(plan)

    print("\n--- Case 2: M=8 hot-dominated (6,1,1) ---")
    plan = schedule_event_driven({0: 6, 1: 1, 2: 1}, sys_cfg, moe_cfg)
    print_plan(plan)

    print("\n--- Case 3: M=16 uniform (4,4,4,4) ---")
    plan = schedule_event_driven({0: 4, 1: 4, 2: 4, 3: 4}, sys_cfg, moe_cfg)
    print_plan(plan)

    print("\n--- Case 4: M=16 mixed (8,4,2,1,1) with cache e8@C2 ---")
    plan = schedule_event_driven(
        {8: 8, 4: 4, 2: 2, 1: 1, 10: 1}, sys_cfg, moe_cfg, cached_map={8: 2}
    )
    print_plan(plan)

    print("\n--- Case 5: M=16 all cold (1×16) ---")
    plan = schedule_event_driven({i: 1 for i in range(16)}, sys_cfg, moe_cfg)
    print_plan(plan)
