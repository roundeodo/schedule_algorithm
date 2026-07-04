#!/usr/bin/env python3
"""
HeMAiA MoE 调度器
=================

职责: 生成调度方案 (SchedulePlan), 交给 model.py 执行.
不做cycle-accurate仿真, 只用轻量级cost估算搜索最优方案.

策略:
  1. 并行流式 (parallel_stream): C2用sram_xDMA(64), C3用iDMA(64)
  2. 全速流式 (solo_stream): 单cluster用sram_xDMA+iDMA(128B/cc)
  3. Token拆分 (token_split): 热门expert拆C2+C3, Phase1流式+Phase2驻留
  4. 驻留+流式 (resident_stream): 一个驻留计算, 另一个全速流式(128B/cc)
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

from config import (
    VersaCoreShape,
    generate_shapes,
    SystemConfig,
    MoELayerConfig,
)


# ============================================================
#  数据结构
# ============================================================


@dataclass
class ExpertTask:
    eid: int
    M: int  # token数


@dataclass
class ScheduleStep:
    """单条执行指令"""

    eid: int
    M: int
    cluster: int  # C2 or C3
    shape: VersaCoreShape
    dma_channels: str  # "xdma", "idma", "both", "none"
    load_bw: int  # 有效带宽 (bytes/cc)
    resident: bool  # 权重已驻留TCDM
    desc: str = ""


@dataclass
class Phase:
    """并行执行组 (C2和C3可同时执行)"""

    steps: List[ScheduleStep]
    desc: str = ""


@dataclass
class SchedulePlan:
    """完整调度方案"""

    phases: List[Phase]
    expert_tasks: List[ExpertTask]
    estimated_cc: int = 0  # 轻量估算
    strategy_name: str = ""


# ============================================================
#  Zipf 路由
# ============================================================


def zipf_route(
    M: int,
    n_experts: int = 64,
    topk: int = 2,
    zipf_s: float = 1.1,
    seed: int = 42,
    min_tokens: int = 2,
) -> List[ExpertTask]:
    """生成Zipf分布的routing结果"""
    rng = random.Random(seed)
    raw = [1.0 / (r**zipf_s) for r in range(1, n_experts + 1)]
    total = sum(raw)
    cdf, cum = [], 0.0
    for p in raw:
        cum += p / total
        cdf.append(cum)
    rank_to_eid = list(range(n_experts))
    rng.shuffle(rank_to_eid)
    counts = [0] * n_experts
    for _ in range(M):
        selected = set()
        while len(selected) < topk:
            u = rng.random()
            for rank, c in enumerate(cdf):
                if u <= c:
                    selected.add(rank)
                    break
        for rank in selected:
            counts[rank_to_eid[rank]] += 1
    result = [
        ExpertTask(eid=eid, M=cnt)
        for eid, cnt in enumerate(counts)
        if cnt >= min_tokens
    ]
    result.sort(key=lambda x: x.M, reverse=True)
    return result


# ============================================================
#  轻量Cost估算
# ============================================================


def gemm_cycles(M: int, K: int, N: int, shape: VersaCoreShape) -> int:
    R, T, C = shape.meshRow, shape.tileSize, shape.meshCol
    Mt = math.ceil(M / R)
    Kt = math.ceil(K / T)
    Nt = math.ceil(N / C)
    drain = math.ceil(R * C * 32 / 1024)
    return Mt * Nt * (Kt + drain) + 5


def b_demand(shape: VersaCoreShape, num_vc: int, wpe: float) -> float:
    """cluster总B矩阵带宽需求(bytes/cc)"""
    return num_vc * shape.tileSize * shape.meshCol * wpe


def vc_utilization(M: int, shape: VersaCoreShape) -> float:
    Mt = math.ceil(M / shape.meshRow)
    return M / (Mt * shape.meshRow) if Mt > 0 else 1.0


ELEM_MUL_RATE = 128  # SwiGLU elements/cycle


def expert_stream_cost(
    M: int, K: int, N: int, shape: VersaCoreShape, load_bw: int, num_vc: int, wpe: float
) -> int:
    """
    Streaming expert (gate+up → SwiGLU → down) 的cycle估算.
    dual-VC: 每个VC处理 ceil(M/num_vc) 行, 并行执行.
    """
    gate_up_w = 2 * K * N * wpe  # gate + up weights
    down_w = N * K * wpe
    bd = b_demand(shape, num_vc, wpe)
    M_per_vc = math.ceil(M / num_vc)

    def _stream(m, k, n, w_bytes):
        cc = gemm_cycles(m, k, n, shape)
        T = shape.tileSize
        first_tile = math.ceil(T * n * wpe / load_bw) if load_bw > 0 else 0
        if load_bw <= 0:
            return cc
        if bd <= load_bw:
            return first_tile + cc
        else:
            return first_tile + int(math.ceil(cc * bd / load_bw))

    gu_cc = _stream(M_per_vc, K, N, gate_up_w)
    sw_cc = math.ceil(M * N / ELEM_MUL_RATE)
    dn_cc = _stream(M_per_vc, N, K, down_w)
    return gu_cc + sw_cc + dn_cc


def expert_resident_cost(
    M: int, K: int, N: int, shape: VersaCoreShape, num_vc: int = 1
) -> int:
    """驻留权重时的expert cycle估算 (dual-VC M-split)"""
    M_per_vc = math.ceil(M / num_vc)
    gu_cc = gemm_cycles(M_per_vc, K, N, shape)
    sw_cc = math.ceil(M * N / ELEM_MUL_RATE)
    dn_cc = gemm_cycles(M_per_vc, N, K, shape)
    return gu_cc + sw_cc + dn_cc


# ============================================================
#  Shape选择
# ============================================================


def select_shape(
    M: int,
    K: int,
    N: int,
    load_bw: int,
    num_vc: int,
    wpe: float,
    mac_per_vc: int,
    streaming: bool = True,
) -> VersaCoreShape:
    """
    选择最优shape.

    streaming=True: 在load_bw约束下, 选择使 expert cost 最小的shape
    streaming=False: 纯compute, 选最小cycle的shape
    """
    shapes = generate_shapes(mac_per_vc, 8)
    best_shape = shapes[0]
    best_cost = float("inf")

    for shape in shapes:
        bd_val = b_demand(shape, num_vc, wpe)

        if streaming and load_bw > 0:
            cost = expert_stream_cost(M, K, N, shape, load_bw, num_vc, wpe)
        else:
            cost = expert_resident_cost(M, K, N, shape, num_vc)

        # gemm_cycles已通过ceil(M/meshRow)计入padding开销, 无需再乘util惩罚
        if cost < best_cost:
            best_cost = cost
            best_shape = shape

    return best_shape


# ============================================================
#  调度策略
# ============================================================


class Scheduler:
    """
    MoE调度器: 搜索最优调度方案.

    搜索空间:
      - 并行流式: C2用sram_xDMA(64), C3用iDMA(64)
      - 全速单发: 全部128B/cc给一个cluster
      - Token拆分: 热门expert拆C2+C3并行流式 → 驻留第二轮
      - 驻留+流式: 驻留cluster 0 DMA, 另一cluster 128B/cc

    取estimated_cc最小的方案输出.
    """

    def __init__(self, sys_cfg: SystemConfig, moe_cfg: MoELayerConfig):
        self.sys = sys_cfg
        self.moe = moe_cfg
        self.wpe = moe_cfg.weight_dtype_bytes
        self.K = moe_cfg.hidden_size
        self.N = moe_cfg.moe_intermediate_size
        # C2/C3的VC参数
        self.c2 = sys_cfg.clusters[2]
        self.c3 = sys_cfg.clusters[3]

    def _cost(
        self, M: int, bw: int, cid: int, resident: bool = False
    ) -> Tuple[int, VersaCoreShape]:
        """快速估算expert(M)在cluster cid上的cost"""
        c = self.sys.clusters[cid]
        shape = select_shape(
            M,
            self.K,
            self.N,
            bw,
            c.num_vc,
            self.wpe,
            c.vc_mac_count,
            streaming=not resident,
        )
        if resident:
            cc = expert_resident_cost(M, self.K, self.N, shape, c.num_vc)
        else:
            cc = expert_stream_cost(M, self.K, self.N, shape, bw, c.num_vc, self.wpe)
        return cc, shape

    def generate(self, tasks: List[ExpertTask]) -> SchedulePlan:
        """搜索最优调度方案"""
        candidates = []

        # 策略1: 并行流式 (sram_xDMA→C2 64, iDMA→C3 64)
        plan1 = self._strategy_parallel_stream(tasks)
        candidates.append(plan1)

        # 策略2: 全速顺序流式 (128B/cc 轮流)
        plan2 = self._strategy_sequential_full(tasks)
        candidates.append(plan2)

        # 策略3: Token拆分 + 并行 (如果最热expert够热)
        plan3 = self._strategy_token_split(tasks)
        if plan3:
            candidates.append(plan3)

        # 策略4: 混合 (尝试各种配对+驻留组合)
        plan4 = self._strategy_mixed(tasks)
        candidates.append(plan4)

        best = min(candidates, key=lambda p: p.estimated_cc)
        return best

    # ----------------------------------------------------------
    #  策略1: 并行流式
    # ----------------------------------------------------------

    def _strategy_parallel_stream(self, tasks: List[ExpertTask]) -> SchedulePlan:
        """
        配对: (E0,E1), (E2,E3), ...
        C2用sram_xDMA(64B/cc), C3用iDMA(64B/cc)
        """
        phases = []
        total_cc = 0
        pool = list(tasks)

        while len(pool) >= 2:
            t2, t3 = pool[0], pool[1]
            c2_cc, s2 = self._cost(t2.M, 64, 2)
            c3_cc, s3 = self._cost(t3.M, 64, 3)

            # 也试反向
            c2r_cc, s2r = self._cost(t3.M, 64, 2)
            c3r_cc, s3r = self._cost(t2.M, 64, 3)

            if max(c2_cc, c3_cc) <= max(c2r_cc, c3r_cc):
                phase = Phase(
                    steps=[
                        ScheduleStep(
                            t2.eid,
                            t2.M,
                            2,
                            s2,
                            "xdma",
                            64,
                            False,
                            f"E{t2.eid}(M={t2.M}) stream xDMA→C2",
                        ),
                        ScheduleStep(
                            t3.eid,
                            t3.M,
                            3,
                            s3,
                            "idma",
                            64,
                            False,
                            f"E{t3.eid}(M={t3.M}) stream iDMA→C3",
                        ),
                    ],
                    desc=f"parallel E{t2.eid}+E{t3.eid}",
                )
                total_cc += max(c2_cc, c3_cc)
            else:
                phase = Phase(
                    steps=[
                        ScheduleStep(
                            t3.eid,
                            t3.M,
                            2,
                            s2r,
                            "xdma",
                            64,
                            False,
                            f"E{t3.eid}(M={t3.M}) stream xDMA→C2",
                        ),
                        ScheduleStep(
                            t2.eid,
                            t2.M,
                            3,
                            s3r,
                            "idma",
                            64,
                            False,
                            f"E{t2.eid}(M={t2.M}) stream iDMA→C3",
                        ),
                    ],
                    desc=f"parallel E{t3.eid}+E{t2.eid}",
                )
                total_cc += max(c2r_cc, c3r_cc)

            phases.append(phase)
            pool = pool[2:]

        # 剩余1个
        if pool:
            t = pool[0]
            cc, s = self._cost(t.M, 128, 2)
            phases.append(
                Phase(
                    steps=[
                        ScheduleStep(
                            t.eid,
                            t.M,
                            2,
                            s,
                            "both",
                            128,
                            False,
                            f"E{t.eid}(M={t.M}) solo 128B/cc→C2",
                        )
                    ],
                    desc=f"solo E{t.eid}",
                )
            )
            total_cc += cc

        return SchedulePlan(
            phases=phases,
            expert_tasks=tasks,
            estimated_cc=total_cc,
            strategy_name="parallel_stream",
        )

    # ----------------------------------------------------------
    #  策略2: 全速顺序
    # ----------------------------------------------------------

    def _strategy_sequential_full(self, tasks: List[ExpertTask]) -> SchedulePlan:
        """每个expert独享128B/cc, C2和C3交替"""
        phases = []
        total_cc = 0
        for i, t in enumerate(tasks):
            cid = 2 if i % 2 == 0 else 3
            cc, s = self._cost(t.M, 128, cid)
            phases.append(
                Phase(
                    steps=[
                        ScheduleStep(
                            t.eid,
                            t.M,
                            cid,
                            s,
                            "both",
                            128,
                            False,
                            f"E{t.eid}(M={t.M}) solo 128B/cc→C{cid}",
                        )
                    ],
                    desc=f"solo E{t.eid} on C{cid}",
                )
            )
            total_cc += cc

        return SchedulePlan(
            phases=phases,
            expert_tasks=tasks,
            estimated_cc=total_cc,
            strategy_name="sequential_full",
        )

    # ----------------------------------------------------------
    #  策略3: Token拆分 + 并行
    # ----------------------------------------------------------

    def _strategy_token_split(self, tasks: List[ExpertTask]) -> Optional[SchedulePlan]:
        """
        最热expert拆分:
          Phase1: C2拿M/2 (sram_xDMA 64), C3拿M/2 (iDMA 64)
          Phase2: C2驻留计算剩余 (0 BW), C3流式cold expert (128B/cc)
          Phase3+: 继续配对剩余experts
        """
        if len(tasks) < 2:
            return None

        hot = tasks[0]
        if hot.M < 6:
            return None

        phases = []
        total_cc = 0

        # Phase1: hot expert split
        m_c2 = hot.M // 2
        m_c3 = hot.M - m_c2
        cc_c2_p1, s2_p1 = self._cost(m_c2, 64, 2)
        cc_c3_p1, s3_p1 = self._cost(m_c3, 64, 3)
        p1_cc = max(cc_c2_p1, cc_c3_p1)

        phases.append(
            Phase(
                steps=[
                    ScheduleStep(
                        hot.eid,
                        m_c2,
                        2,
                        s2_p1,
                        "xdma",
                        64,
                        False,
                        f"E{hot.eid}(M={m_c2}) split→C2 xDMA",
                    ),
                    ScheduleStep(
                        hot.eid,
                        m_c3,
                        3,
                        s3_p1,
                        "idma",
                        64,
                        False,
                        f"E{hot.eid}(M={m_c3}) split→C3 iDMA",
                    ),
                ],
                desc=f"Phase1: split E{hot.eid}(M={hot.M})",
            )
        )
        total_cc += p1_cc

        # Phase2+: 全部配对 C2(xDMA 64) + C3(iDMA 64)
        # 不做solo — 避免C2闲置, 充分利用双cluster并行度
        remaining = list(tasks[1:])

        while len(remaining) >= 2:
            t2, t3 = remaining[0], remaining[1]
            c2_cc, s2 = self._cost(t2.M, 64, 2)
            c3_cc, s3 = self._cost(t3.M, 64, 3)
            # 也试反向
            c2r, s2r = self._cost(t3.M, 64, 2)
            c3r, s3r = self._cost(t2.M, 64, 3)
            if max(c2_cc, c3_cc) <= max(c2r, c3r):
                phases.append(
                    Phase(
                        steps=[
                            ScheduleStep(
                                t2.eid,
                                t2.M,
                                2,
                                s2,
                                "xdma",
                                64,
                                False,
                                f"E{t2.eid}(M={t2.M}) xDMA→C2",
                            ),
                            ScheduleStep(
                                t3.eid,
                                t3.M,
                                3,
                                s3,
                                "idma",
                                64,
                                False,
                                f"E{t3.eid}(M={t3.M}) iDMA→C3",
                            ),
                        ],
                        desc=f"parallel E{t2.eid}+E{t3.eid}",
                    )
                )
                total_cc += max(c2_cc, c3_cc)
            else:
                phases.append(
                    Phase(
                        steps=[
                            ScheduleStep(
                                t3.eid,
                                t3.M,
                                2,
                                s2r,
                                "xdma",
                                64,
                                False,
                                f"E{t3.eid}(M={t3.M}) xDMA→C2",
                            ),
                            ScheduleStep(
                                t2.eid,
                                t2.M,
                                3,
                                s3r,
                                "idma",
                                64,
                                False,
                                f"E{t2.eid}(M={t2.M}) iDMA→C3",
                            ),
                        ],
                        desc=f"parallel E{t3.eid}+E{t2.eid}",
                    )
                )
                total_cc += max(c2r, c3r)
            remaining = remaining[2:]

        if remaining:
            t = remaining[0]
            cc, s = self._cost(t.M, 128, 2)
            phases.append(
                Phase(
                    steps=[ScheduleStep(t.eid, t.M, 2, s, "both", 128, False)],
                    desc=f"solo E{t.eid}",
                )
            )
            total_cc += cc

        return SchedulePlan(
            phases=phases,
            expert_tasks=tasks,
            estimated_cc=total_cc,
            strategy_name="token_split",
        )

    # ----------------------------------------------------------
    #  策略4: 混合搜索
    # ----------------------------------------------------------

    def _strategy_mixed(self, tasks: List[ExpertTask]) -> SchedulePlan:
        """
        贪心: 每轮选最优action (并行64+64, solo128, 驻留+流式)
        """
        phases = []
        total_cc = 0
        pool = list(tasks)
        c2_resident_eid = -1
        c3_resident_eid = -1

        while pool:
            best_action = None
            best_cc = float("inf")

            # Action 1: 并行流式 (前2个)
            if len(pool) >= 2:
                for i in range(min(3, len(pool))):
                    for j in range(i + 1, min(6, len(pool))):
                        ti, tj = pool[i], pool[j]
                        # C2=ti(xdma64), C3=tj(idma64)
                        ci, si = self._cost(ti.M, 64, 2)
                        cj, sj = self._cost(tj.M, 64, 3)
                        mk = max(ci, cj)
                        if mk < best_cc:
                            best_cc = mk
                            best_action = ("par", i, j, si, sj, False)
                        # 反向
                        ci2, si2 = self._cost(tj.M, 64, 2)
                        cj2, sj2 = self._cost(ti.M, 64, 3)
                        mk2 = max(ci2, cj2)
                        if mk2 < best_cc:
                            best_cc = mk2
                            best_action = ("par", j, i, si2, sj2, True)

            # Action 2: 单发128B/cc
            for k, t in enumerate(pool[:3]):
                for cid in [2, 3]:
                    cc, s = self._cost(t.M, 128, cid)
                    if cc < best_cc:
                        best_cc = cc
                        best_action = ("solo", k, cid, s)

            # Action 3: 驻留+并行
            for k, t in enumerate(pool[:3]):
                for cid in [2, 3]:
                    res_eid = c2_resident_eid if cid == 2 else c3_resident_eid
                    if res_eid == t.eid:
                        rcc, rs = self._cost(t.M, 0, cid, resident=True)
                        # 同时另一个cluster流式
                        o_cid = 3 if cid == 2 else 2
                        for k2, t2 in enumerate(pool[:3]):
                            if k2 == k:
                                continue
                            occ, os_ = self._cost(t2.M, 128, o_cid)
                            mk = max(rcc, occ)
                            if mk < best_cc:
                                best_cc = mk
                                best_action = ("res_par", k, cid, rs, k2, o_cid, os_)

            # 执行最优action
            if best_action is None:
                break

            if best_action[0] == "par":
                _, i, j, si, sj, flipped = best_action
                ti, tj = pool[i], pool[j]
                if flipped:
                    ti, tj = tj, ti
                phases.append(
                    Phase(
                        steps=[
                            ScheduleStep(
                                ti.eid,
                                ti.M,
                                2,
                                si,
                                "xdma",
                                64,
                                False,
                                f"E{ti.eid}(M={ti.M})",
                            ),
                            ScheduleStep(
                                tj.eid,
                                tj.M,
                                3,
                                sj,
                                "idma",
                                64,
                                False,
                                f"E{tj.eid}(M={tj.M})",
                            ),
                        ],
                        desc=f"par E{ti.eid}+E{tj.eid}",
                    )
                )
                total_cc += best_cc
                c2_resident_eid = ti.eid
                c3_resident_eid = tj.eid
                for idx in sorted([i, j], reverse=True):
                    pool.pop(idx)

            elif best_action[0] == "solo":
                _, k, cid, s = best_action
                t = pool[k]
                phases.append(
                    Phase(
                        steps=[
                            ScheduleStep(
                                t.eid,
                                t.M,
                                cid,
                                s,
                                "both",
                                128,
                                False,
                                f"E{t.eid}(M={t.M}) 128B→C{cid}",
                            )
                        ],
                        desc=f"solo E{t.eid} on C{cid}",
                    )
                )
                total_cc += best_cc
                if cid == 2:
                    c2_resident_eid = t.eid
                else:
                    c3_resident_eid = t.eid
                pool.pop(k)

            elif best_action[0] == "res_par":
                _, k, cid, rs, k2, o_cid, os_ = best_action
                t, t2 = pool[k], pool[k2]
                phases.append(
                    Phase(
                        steps=[
                            ScheduleStep(
                                t.eid,
                                t.M,
                                cid,
                                rs,
                                "none",
                                0,
                                True,
                                f"E{t.eid}(M={t.M}) resident C{cid}",
                            ),
                            ScheduleStep(
                                t2.eid,
                                t2.M,
                                o_cid,
                                os_,
                                "both",
                                128,
                                False,
                                f"E{t2.eid}(M={t2.M}) 128B→C{o_cid}",
                            ),
                        ],
                        desc=f"res E{t.eid} + stream E{t2.eid}",
                    )
                )
                total_cc += best_cc
                if o_cid == 2:
                    c2_resident_eid = t2.eid
                else:
                    c3_resident_eid = t2.eid
                for idx in sorted([k, k2], reverse=True):
                    pool.pop(idx)

        return SchedulePlan(
            phases=phases,
            expert_tasks=tasks,
            estimated_cc=total_cc,
            strategy_name="mixed",
        )


# ============================================================
#  Shared Expert Cost (used for comparison)
# ============================================================


def shared_expert_cost(M: int, sys_cfg: SystemConfig, moe_cfg: MoELayerConfig) -> int:
    """估算shared expert在C0+C1上的cycle数"""
    K = moe_cfg.hidden_size
    N = moe_cfg.shared_intermediate
    wpe = moe_cfg.weight_dtype_bytes
    tok = M * K  # token A bytes

    shapes = generate_shapes(512, 8)
    shape = min(shapes, key=lambda s: gemm_cycles(M, K, N, s))

    # token搬运
    tok_dma = math.ceil(tok / 64)
    p2p_c0_c1 = math.ceil(tok / 64)
    # gate+up (驻留, 并行)
    gate_cc = gemm_cycles(M, K, N, shape)
    up_cc = gemm_cycles(M, K, N, shape)
    # SwiGLU streaming
    sw_row = math.ceil(N / ELEM_MUL_RATE)
    p2p_row = math.ceil(N / 64)
    swiglu_cc = max(p2p_row, sw_row) * M + sw_row
    # half-down
    half_N = K // 2
    down_cc = gemm_cycles(M, N, half_N, shape)
    # merge
    merge_cc = math.ceil(M * half_N / 64)

    # C0 timeline: tok_dma + gate_cc + (wait swiglu) + half_down + merge
    # C1 timeline: tok_dma + p2p + up_cc + swiglu + half_down
    c0_total = tok_dma + gate_cc + down_cc + merge_cc
    c1_total = tok_dma + p2p_c0_c1 + up_cc + swiglu_cc + down_cc
    total = max(c0_total, c1_total) + merge_cc
    return total


# ============================================================
#  主入口
# ============================================================


def main():
    """运行调度器, 输出最优方案"""
    sys_cfg = SystemConfig.default_4cluster()
    moe_cfg = MoELayerConfig(weight_dtype_bits=4)  # INT4

    M_total = 64
    tasks = zipf_route(M_total, moe_cfg.n_routed_experts, moe_cfg.topk)
    n_active = len(tasks)
    total_tokens = sum(t.M for t in tasks)

    print(
        f"=== Routing: M={M_total}, active_experts={n_active}, "
        f"total_tokens={total_tokens} ==="
    )
    print(f"Top experts: {[(t.eid, t.M) for t in tasks[:10]]}")
    print()

    scheduler = Scheduler(sys_cfg, moe_cfg)
    plan = scheduler.generate(tasks)

    shared_cc = shared_expert_cost(M_total, sys_cfg, moe_cfg)

    print(f"=== Best Strategy: {plan.strategy_name} ===")
    print(f"Estimated routed cc: {plan.estimated_cc:,}")
    print(f"Shared expert cc:    {shared_cc:,}")
    print(f"Ratio: {plan.estimated_cc / shared_cc:.2%}")
    print()

    for i, phase in enumerate(plan.phases):
        print(f"Phase {i+1}: {phase.desc}")
        for step in phase.steps:
            print(
                f"  C{step.cluster}: E{step.eid}(M={step.M}) "
                f"shape={step.shape} "
                f"dma={step.dma_channels}({step.load_bw}B/cc) "
                f"{'resident' if step.resident else 'stream'}"
            )
    print()

    return plan, shared_cc


if __name__ == "__main__":
    main()
