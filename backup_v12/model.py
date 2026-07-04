#!/usr/bin/env python3
"""
HeMAiA MoE Cycle-Accurate System Model
=======================================

负责: 模拟硬件行为, 产生cycle-accurate timeline.
输入: 调度方案 (scheduler生成的SchedulePlan)
输出: Timeline (每个资源的事件序列) + TCDM快照 + makespan

互联拓扑:
  xDMA: 任意两个xDMA可P2P互联, 但一个xDMA同时只连一个(一对一)
    - sram_xDMA(64B/cc), C0_xDMA(64B/cc), ..., C3_xDMA(64B/cc)
  iDMA: 直连cluster TCDM(不占xDMA端口), 64B/cc, 同时只连一个cluster
  SRAM→cluster两条独立路径:
    (1) sram_xDMA ↔ cluster_xDMA (占双方xDMA端口)
    (2) iDMA → cluster_TCDM (只占iDMA, 不占cluster_xDMA)

计算模型:
  - GEMM tiling: 第一个tile暴露DMA延迟, 后续tile DMA与compute overlap
  - Dual-VC: C2/C3各2个256MAC VC, 使用相同shape
    - B矩阵带宽需求 = 2 × tileSize × meshCol × bytes_per_elem
    - 权重只搬一份(broadcast), 但消耗速率2x
  - VC利用率: meshRow > M时 util = M/(ceil(M/meshRow)*meshRow)
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from config import (
    VersaCoreShape,
    generate_shapes,
    ClusterConfig,
    SystemConfig,
    MoELayerConfig,
)

MB = 1024 * 1024


# ============================================================
#  基础计算函数
# ============================================================


def gemm_cycles(M: int, K: int, N: int, shape: VersaCoreShape) -> int:
    """GEMM(M,K,N)在给定shape下的计算周期数(单个VC)"""
    R, T, C = shape.meshRow, shape.tileSize, shape.meshCol
    Mt = math.ceil(M / R)
    Kt = math.ceil(K / T)
    Nt = math.ceil(N / C)
    drain = math.ceil(R * C * 32 / 1024)
    return Mt * Nt * (Kt + drain) + 5


def dma_cc(nbytes: int, bw: int) -> int:
    """搬运nbytes数据需要的周期数"""
    return math.ceil(nbytes / bw) if nbytes > 0 and bw > 0 else 0


def b_demand_per_vc(shape: VersaCoreShape, wpe: float) -> float:
    """单个VC的B矩阵每周期消耗(bytes/cc)"""
    return shape.tileSize * shape.meshCol * wpe


def b_demand_cluster(shape: VersaCoreShape, num_vc: int, wpe: float) -> float:
    """一个cluster所有VC的总B矩阵带宽需求(bytes/cc)"""
    return num_vc * shape.tileSize * shape.meshCol * wpe


def vc_utilization(M: int, shape: VersaCoreShape) -> float:
    """M维度利用率"""
    Mt = math.ceil(M / shape.meshRow)
    return M / (Mt * shape.meshRow) if Mt > 0 else 1.0


def bank_conflict_stretch(
    shape: VersaCoreShape, dma_active: bool, num_banks: int = 128, dma_ports: int = 16
) -> float:
    vc_demand = shape.a_banks + shape.b_banks
    if dma_active:
        total = vc_demand + dma_ports
        if total > num_banks:
            return total / num_banks
        p = min(1.0, dma_ports * vc_demand / (num_banks * num_banks))
        return 1.0 + p * 0.5
    return 1.0


ELEM_MUL_RATE = 128  # SwiGLU elements/cycle


# ============================================================
#  Timeline Events
# ============================================================


@dataclass
class Event:
    resource: str
    start: int
    end: int
    name: str
    data_bytes: int = 0
    macs: int = 0
    desc: str = ""

    @property
    def duration(self):
        return self.end - self.start


@dataclass
class TCDMSnapshot:
    time: int
    cluster: int
    contents: Dict[str, int]
    free_bytes: int
    event_desc: str


# ============================================================
#  资源追踪器
# ============================================================


class ResourceTracker:
    """追踪多个命名资源的空闲时刻"""

    def __init__(self):
        self.free_at: Dict[str, int] = {}

    def get(self, name: str) -> int:
        return self.free_at.get(name, 0)

    def set(self, name: str, t: int):
        self.free_at[name] = t

    def occupy(self, name: str, start: int, duration: int) -> Tuple[int, int]:
        """占用资源, 返回(actual_start, end)"""
        s = max(start, self.get(name))
        e = s + duration
        self.free_at[name] = e
        return s, e


class TCDMTracker:
    """追踪cluster TCDM内容"""

    def __init__(self, capacities: Dict[int, int]):
        self.capacity = dict(capacities)
        self.contents: Dict[int, Dict[str, int]] = {cid: {} for cid in capacities}

    def load(self, cid: int, label: str, nbytes: int):
        self.contents[cid][label] = nbytes

    def evict_all(self, cid: int):
        self.contents[cid].clear()

    def used(self, cid: int) -> int:
        return sum(self.contents[cid].values())

    def free(self, cid: int) -> int:
        return self.capacity[cid] - self.used(cid)

    def snapshot(self, cid: int, time: int, desc: str) -> TCDMSnapshot:
        return TCDMSnapshot(
            time=time,
            cluster=cid,
            contents=dict(self.contents[cid]),
            free_bytes=self.free(cid),
            event_desc=desc,
        )


# ============================================================
#  System Model (核心仿真器)
# ============================================================


class SystemModel:
    """
    Cycle-accurate 系统仿真模型.

    资源列表:
      DMA: sram_xDMA, iDMA, C0_xDMA, C1_xDMA, C2_xDMA, C3_xDMA
      VC:  C0_VC, C1_VC, C2_VC, C3_VC
      其他: C0_elemMul, C1_elemMul, C2_elemMul, C3_elemMul, Host
    """

    def __init__(self, sys: SystemConfig, moe: MoELayerConfig):
        self.sys = sys
        self.moe = moe
        self.res = ResourceTracker()
        self.tcdm = TCDMTracker({c.cluster_id: c.tcdm_size_bytes for c in sys.clusters})
        self.events: List[Event] = []
        self.tcdm_snapshots: List[TCDMSnapshot] = []

    def _ev(self, resource, start, duration, name, **kw):
        """记录一个事件"""
        ev = Event(
            resource=resource, start=start, end=start + duration, name=name, **kw
        )
        self.events.append(ev)
        return ev

    def _snap(self, cid, time, desc):
        s = self.tcdm.snapshot(cid, time, desc)
        self.tcdm_snapshots.append(s)

    # ----------------------------------------------------------
    #  DMA传输 (正确处理互联约束)
    # ----------------------------------------------------------

    def dma_sram_to_cluster(
        self,
        cid: int,
        nbytes: int,
        earliest: int,
        use_xdma: bool = True,
        use_idma: bool = True,
        label: str = "",
    ) -> Tuple[int, int]:
        """
        从SRAM搬数据到cluster.

        路径:
          use_xdma=True:  sram_xDMA ↔ C{cid}_xDMA (占用双方xDMA)
          use_idma=True:  iDMA → C{cid}_TCDM (只占iDMA)
          两者都True: 并行使用, 总BW=128B/cc

        返回 (start, end)
        """
        xdma_bw = self.sys.sram_xdma_bw_bytes_per_cc if use_xdma else 0
        idma_bw = self.sys.idma_bw_bytes_per_cc if use_idma else 0
        total_bw = xdma_bw + idma_bw
        if total_bw == 0:
            return earliest, earliest

        dur = dma_cc(nbytes, total_bw)

        # 确定最早可用时刻
        start = earliest
        if use_xdma:
            start = max(start, self.res.get("sram_xDMA"))
            start = max(start, self.res.get(f"C{cid}_xDMA"))
        if use_idma:
            start = max(start, self.res.get("iDMA"))

        end = start + dur

        # 占用资源
        if use_xdma:
            self.res.set("sram_xDMA", end)
            self.res.set(f"C{cid}_xDMA", end)
        if use_idma:
            self.res.set("iDMA", end)

        # 事件
        mode = (
            "xDMA+iDMA" if (use_xdma and use_idma) else ("xDMA" if use_xdma else "iDMA")
        )
        self._ev(
            f"DMA_{mode}→C{cid}",
            start,
            dur,
            label or f"SRAM→C{cid}",
            data_bytes=nbytes,
            desc=f"bw={total_bw}B/cc {mode}",
        )

        return start, end

    def dma_p2p(
        self, src_cid: int, dst_cid: int, nbytes: int, earliest: int, label: str = ""
    ) -> Tuple[int, int]:
        """
        cluster间P2P传输: C{src}_xDMA ↔ C{dst}_xDMA
        占用双方xDMA端口.
        """
        bw = self.sys.xdma_p2p_bw_bytes_per_cc
        dur = dma_cc(nbytes, bw)
        start = max(
            earliest, self.res.get(f"C{src_cid}_xDMA"), self.res.get(f"C{dst_cid}_xDMA")
        )
        end = start + dur
        self.res.set(f"C{src_cid}_xDMA", end)
        self.res.set(f"C{dst_cid}_xDMA", end)
        self._ev(
            f"C{src_cid}_xDMA",
            start,
            dur,
            label or f"P2P C{src_cid}→C{dst_cid}",
            data_bytes=nbytes,
        )
        return start, end

    def dma_sram_idma_only(
        self, cid: int, nbytes: int, earliest: int, label: str = ""
    ) -> Tuple[int, int]:
        """只用iDMA搬SRAM→cluster (不占cluster_xDMA)"""
        return self.dma_sram_to_cluster(
            cid, nbytes, earliest, use_xdma=False, use_idma=True, label=label
        )

    def dma_sram_xdma_only(
        self, cid: int, nbytes: int, earliest: int, label: str = ""
    ) -> Tuple[int, int]:
        """只用sram_xDMA↔cluster_xDMA搬数据 (占双方xDMA)"""
        return self.dma_sram_to_cluster(
            cid, nbytes, earliest, use_xdma=True, use_idma=False, label=label
        )

    # ----------------------------------------------------------
    #  Streaming GEMM (tiling边搬边算)
    # ----------------------------------------------------------

    def streaming_gemm(
        self,
        cid: int,
        M: int,
        K: int,
        N: int,
        shape: VersaCoreShape,
        weight_bytes: int,
        load_bw: int,
        earliest: int,
        label: str = "",
        dma_active: bool = True,
    ) -> Tuple[int, int, int]:
        """
        Streaming GEMM: 边搬边算.

        Tiling模型:
          - K维度分为Kt = ceil(K/tileSize)个tile
          - 每个K-tile权重 = tileSize × N × wpe bytes (一份, broadcast给所有VC)
          - dual-VC B需求 = num_vc × tileSize × meshCol × wpe (per cc)
          - 若 load_bw >= b_demand: compute-bound, 第一个tile等DMA, 后续overlap
          - 若 load_bw < b_demand: DMA-bound, VC减速 (slowdown = b_demand/load_bw)

        返回: (compute_start, compute_end, dma_end)
        """
        cluster = self.sys.clusters[cid]
        wpe = self.moe.weight_dtype_bytes
        num_vc = cluster.num_vc
        bd = b_demand_cluster(shape, num_vc, wpe)

        # dual-VC: 两个VC并行处理M的不同行, 每个VC处理 ceil(M/num_vc) 行
        M_per_vc = math.ceil(M / num_vc)
        compute_cc = gemm_cycles(M_per_vc, K, N, shape)
        dma_total = dma_cc(weight_bytes, load_bw) if load_bw > 0 else 0

        stretch = bank_conflict_stretch(shape, dma_active)
        compute_stretched = int(math.ceil(compute_cc * stretch))

        vc_free = self.res.get(f"C{cid}_VC")
        start = max(earliest, vc_free)

        if load_bw == 0 or dma_total == 0:
            # 驻留数据, 纯compute
            end = start + compute_stretched
            self._ev(
                f"C{cid}_VC",
                start,
                compute_stretched,
                label,
                macs=2 * num_vc * M * K * N,
                desc=f"shape{shape} resident M/vc={M_per_vc} util={vc_utilization(M_per_vc,shape):.0%}",
            )
            self.res.set(f"C{cid}_VC", end)
            return start, end, start

        # Tiling
        T = shape.tileSize
        Kt = math.ceil(K / T)
        tile_weight = int(T * N * wpe)
        first_tile_dma = dma_cc(tile_weight, load_bw)

        if bd <= load_bw:
            # Compute-bound: DMA能持续feed
            total = first_tile_dma + compute_stretched
            dma_end = start + dma_total
        else:
            # DMA-bound: VC减速
            slowdown = bd / load_bw
            total = (
                max(int(math.ceil(compute_stretched * slowdown)), dma_total)
                + first_tile_dma
            )
            dma_end = start + dma_total

        end = start + total
        bound = "DMA-bound" if bd > load_bw else "compute-bound"
        self._ev(
            f"C{cid}_VC",
            start,
            total,
            label,
            macs=2 * num_vc * M * K * N,
            desc=f"shape{shape} stream M/vc={M_per_vc} bw={load_bw} bd={bd:.0f} {bound}",
        )
        self.res.set(f"C{cid}_VC", end)
        return start, end, dma_end

    # ----------------------------------------------------------
    #  Resident GEMM (数据已在TCDM中)
    # ----------------------------------------------------------

    def resident_gemm(
        self,
        cid: int,
        M: int,
        K: int,
        N: int,
        shape: VersaCoreShape,
        earliest: int,
        label: str = "",
    ) -> Tuple[int, int]:
        """数据已驻留TCDM, 直接计算"""
        return self.streaming_gemm(
            cid, M, K, N, shape, 0, 0, earliest, label=label, dma_active=False
        )[:2]

    # ----------------------------------------------------------
    #  Expert完整执行 (gate+up → SwiGLU → down)
    # ----------------------------------------------------------

    def execute_expert(
        self,
        cid: int,
        eid: int,
        M: int,
        shape: VersaCoreShape,
        load_bw: int,
        dma_channels: str,
        earliest: int,
        resident: bool = False,
    ) -> Tuple[int, int]:
        """
        执行一个完整的routed expert.

        dma_channels: "xdma", "idma", "both", "none"
        resident: True = 权重已在TCDM

        流程: gate+up(streaming/resident) → SwiGLU → down(streaming/resident)
        返回: (expert_start, expert_end)
        """
        K = self.moe.hidden_size
        N = self.moe.moe_intermediate_size
        wpe = self.moe.weight_dtype_bytes
        gate_w = self.moe.expert_gate_weight_size
        up_w = self.moe.expert_up_weight_size
        down_w = self.moe.expert_down_weight_size

        if resident:
            # 全部驻留
            gu_s, gu_e = self.resident_gemm(
                cid, M, K, N, shape, earliest, f"E{eid} gate+up [{M},{K},{N}] resident"
            )
            sw_cc = math.ceil(M * N / ELEM_MUL_RATE)
            sw_s = gu_e
            sw_e = sw_s + sw_cc
            self._ev(f"C{cid}_elemMul", sw_s, sw_cc, f"E{eid} SwiGLU")
            dn_s, dn_e = self.resident_gemm(
                cid, M, N, K, shape, sw_e, f"E{eid} down [{M},{N},{K}] resident"
            )
            return gu_s, dn_e

        # Streaming: 需要搬运权重
        # gate+up权重交叉搬
        gu_bytes = gate_w + up_w
        # DMA资源占用
        gu_dma_start = self._occupy_dma_channels(cid, dma_channels, earliest)
        gu_dma_dur = dma_cc(gu_bytes, load_bw)
        gu_dma_end = gu_dma_start + gu_dma_dur

        gu_s, gu_e, _ = self.streaming_gemm(
            cid,
            M,
            K,
            N,
            shape,
            gu_bytes,
            load_bw,
            gu_dma_start,
            f"E{eid} gate+up [{M},{K},{N}] stream",
        )

        sw_cc = math.ceil(M * N / ELEM_MUL_RATE)
        sw_s = gu_e
        sw_e = sw_s + sw_cc
        self._ev(f"C{cid}_elemMul", sw_s, sw_cc, f"E{eid} SwiGLU")

        # down权重: gate+up DMA结束后开始搬
        dn_dma_start = max(gu_dma_end, sw_e)  # 也可以提前搬
        dn_dma_dur = dma_cc(down_w, load_bw)

        # 如果down DMA可以在swiglu期间开始
        dn_dma_actual_start = max(gu_dma_end, sw_s)  # swiglu开始时DMA可能已空闲
        dn_dma_actual_end = dn_dma_actual_start + dn_dma_dur

        dn_earliest = max(sw_e, dn_dma_actual_start)
        dn_s, dn_e, _ = self.streaming_gemm(
            cid,
            M,
            N,
            K,
            shape,
            down_w,
            load_bw,
            dn_earliest,
            f"E{eid} down [{M},{N},{K}] stream",
        )

        # 释放DMA资源
        total_dma_end = max(gu_dma_end, dn_dma_actual_end)
        self._release_dma_channels(cid, dma_channels, total_dma_end)

        # TCDM更新
        self.tcdm.evict_all(cid)
        self.tcdm.load(cid, f"E{eid}_weights", gate_w + up_w + down_w)
        self._snap(cid, dn_e, f"E{eid} done on C{cid}")

        return gu_s, dn_e

    def _occupy_dma_channels(self, cid, channels, earliest) -> int:
        """返回DMA通道最早可用时刻"""
        start = earliest
        if channels in ("xdma", "both"):
            start = max(start, self.res.get("sram_xDMA"), self.res.get(f"C{cid}_xDMA"))
        if channels in ("idma", "both"):
            start = max(start, self.res.get("iDMA"))
        return start

    def _release_dma_channels(self, cid, channels, end):
        if channels in ("xdma", "both"):
            self.res.set("sram_xDMA", end)
            self.res.set(f"C{cid}_xDMA", end)
        if channels in ("idma", "both"):
            self.res.set("iDMA", end)

    # ----------------------------------------------------------
    #  Shared Expert (C0+C1, 权重驻留, N方向half-down)
    # ----------------------------------------------------------

    def simulate_shared_expert(self, M: int) -> int:
        """
        模拟shared expert在C0+C1上的执行.
        返回: 完成时刻
        """
        K = self.moe.hidden_size  # 2048
        N = self.moe.shared_intermediate  # 2816
        half_down_N = K // 2  # 1024 (hidden/2)
        shapes_512 = generate_shapes(512, 8)
        shape = min(shapes_512, key=lambda s: gemm_cycles(M, K, N, s))
        tok_bytes = M * K

        # 1. Token A: SRAM → C0 (sram_xDMA ↔ C0_xDMA)
        tok_s, tok_e = self.dma_sram_xdma_only(0, tok_bytes, 0, "token_A SRAM→C0")

        # 2. Token A: C0 → C1 (P2P: C0_xDMA ↔ C1_xDMA)
        p2p_s, p2p_e = self.dma_p2p(0, 1, tok_bytes, tok_e, "token_A C0→C1")

        # 3. C0: gate GEMM + SiLU (驻留)
        gate_cc = gemm_cycles(M, K, N, shape)
        gate_s, gate_e = self.resident_gemm(
            0, M, K, N, shape, tok_e, f"shared gate+SiLU [{M},{K},{N}]"
        )

        # 4. C1: up GEMM (驻留)
        up_s, up_e = self.resident_gemm(
            1, M, K, N, shape, p2p_e, f"shared up [{M},{K},{N}]"
        )

        # 5. SwiGLU: gate_out P2P→C1, 逐行multiply
        p2p_row = dma_cc(N, 64)
        mul_row = math.ceil(N / ELEM_MUL_RATE)
        swiglu_done = max(gate_e + p2p_row, up_e) + mul_row
        self._ev(
            "C1_elemMul",
            gate_s,
            int(swiglu_done) - gate_s,
            "shared streaming SwiGLU",
            desc="row P2P + mul",
        )

        # 6. Half-down (N-split): C0做前半, C1做后半
        half_down_cc = gemm_cycles(M, N, half_down_N, shape)
        # C1有完整active_A, 直接算
        c1_dn_s, c1_dn_e = self.resident_gemm(
            1,
            M,
            N,
            half_down_N,
            shape,
            int(swiglu_done),
            f"shared C1 half-down [{M},{N},{half_down_N}]",
        )
        # C0需要streaming接收active_A from C1
        down_row_cc = gemm_cycles(1, N, half_down_N, shape)
        c0_pipe = p2p_row + down_row_cc + (M - 1) * max(p2p_row, down_row_cc)
        c0_dn_start = max(int(swiglu_done), self.res.get("C0_VC"))
        self._ev(
            "C0_VC",
            c0_dn_start,
            c0_pipe,
            f"shared C0 half-down [{M},{N},{half_down_N}]",
            macs=2 * M * N * half_down_N,
            desc=f"N-split: 前{half_down_N}维, streaming P2P from C1",
        )
        self.res.set("C0_VC", c0_dn_start + c0_pipe)

        # 7. Merge
        merge_cost = dma_cc(M * half_down_N, 64)
        merge_start = max(c0_dn_start + c0_pipe, c1_dn_e)
        merge_end = merge_start + merge_cost
        self._ev(
            "C0_xDMA",
            merge_start,
            merge_cost,
            "shared down merge",
            data_bytes=M * half_down_N,
        )
        self.res.set("C0_xDMA", merge_end)

        return merge_end

    # ----------------------------------------------------------
    #  Router
    # ----------------------------------------------------------

    def simulate_router(self, M: int) -> int:
        """模拟Router在C3上执行. 返回routing_ready时刻"""
        K = self.moe.hidden_size
        N_out = self.moe.n_routed_experts
        rtr_w = self.moe.router_weight_size
        shape = VersaCoreShape(2, 8, 16)

        # Router权重搬到C3 via iDMA
        _, rw_e = self.dma_sram_idma_only(3, rtr_w, 0, "router_w iDMA→C3")
        # Token A → C3 via sram_xDMA↔C3_xDMA
        _, tok_e = self.dma_sram_xdma_only(3, M * K, 0, "token_A SRAM→C3(router)")

        rtr_start = max(rw_e, tok_e, self.res.get("C3_VC"))
        rtr_cc = gemm_cycles(M, K, N_out, shape)
        self._ev(
            "C3_VC",
            rtr_start,
            rtr_cc,
            f"router [{M},{K},{N_out}]",
            macs=2 * M * K * N_out,
        )
        self.res.set("C3_VC", rtr_start + rtr_cc)

        topk_s, topk_e = self.res.occupy("Host", rtr_start + rtr_cc, 5000)
        scatter_s, scatter_e = self.res.occupy("Host", topk_e, 5000)
        softmax_s, softmax_e = self.res.occupy("Host", scatter_e, 15000)
        self._ev("Host", topk_s, 5000, "topK")
        self._ev("Host", scatter_s, 5000, "scatter_meta")
        self._ev("Host", softmax_s, 15000, "softmax")

        return scatter_e  # routing_ready

    # ----------------------------------------------------------
    #  Token A 分发到 C2/C3
    # ----------------------------------------------------------

    def distribute_tokens(self, M: int, routing_ready: int) -> Dict[int, int]:
        """
        分发token A到C2和C3 (并行).
        sram_xDMA↔C2_xDMA (64B/cc) + iDMA→C3 (64B/cc)
        返回: {cluster_id: token_ready_time}
        """
        tok = M * self.moe.hidden_size
        _, c2_e = self.dma_sram_xdma_only(2, tok, routing_ready, "token_A SRAM→C2")
        _, c3_e = self.dma_sram_idma_only(3, tok, routing_ready, "token_A iDMA→C3")
        return {2: c2_e, 3: c3_e}

    # ----------------------------------------------------------
    #  执行调度方案
    # ----------------------------------------------------------

    def execute_schedule(self, plan, routing_ready: int, tok_times: Dict[int, int]):
        """
        执行scheduler给出的调度方案.

        plan: List of ScheduleStep, 每个step定义一个expert在哪个cluster执行
        """
        for step in plan:
            earliest = max(
                tok_times.get(step.cluster, 0), self.res.get(f"C{step.cluster}_VC")
            )
            if step.resident:
                self.execute_expert(
                    step.cluster,
                    step.eid,
                    step.M,
                    step.shape,
                    0,
                    "none",
                    earliest,
                    resident=True,
                )
            else:
                self.execute_expert(
                    step.cluster,
                    step.eid,
                    step.M,
                    step.shape,
                    step.load_bw,
                    step.dma_channels,
                    earliest,
                )

    def get_makespan(self) -> int:
        if not self.events:
            return 0
        return max(ev.end for ev in self.events)
