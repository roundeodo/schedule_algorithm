#!/usr/bin/env python3
"""
HeMAiA MoE Cycle-Accurate System Model (v21)
=============================================

事件依赖模型: 每个操作记为Event, 通过资源占用时间戳追踪依赖.
bank追踪: 实时监控每个cluster的TCDM bank分配和冲突.
TCDM追踪: 实时监控每个cluster的TCDM内容和剩余空间.

量化方案: W4A8 (权重INT4 0.5B/elem, 激活INT8 1B/elem)

【v21核心变更】:
1) 所有cluster统一dual-VC (2×256MAC), 不再打包shared expert为单VC 512MAC
2) Shared expert在N方向对半切: C0负责左半[1408], C1负责右半[1408]
   每个cluster内部用与routed expert完全相同的dual-VC resident流程
3) Routed expert的gate+up phase 与 down phase 分离DMA资源占用:
   — gate+up DMA完成后DMA立即释放, swiglu期间其他cluster可抓这段slack
   — down DMA单独占用, 完成后释放

Shared Expert流水线 (v21, C0=左半1408, C1=右半1408, 各自dual-VC全驻留):
  1. token_A → C0(sram_xDMA) 和 C1(iDMA) 并行
  2. C0: dual-VC(VC0=gate_half, VC1=up_half) resident → swiglu_half (左半1408)
     C1: dual-VC(VC0=gate_half, VC1=up_half) resident → swiglu_half (右半1408) [并行]
  3. C0: dual-VC down N-split[M,1408,1024] resident
     C1: dual-VC down N-split[M,1408,1024] resident [并行]
  4. Merge: C0、C1的down输出需要对位相加 (因为并不是N-split而是原章shared expert的K方向分块归约)
     实现: C1 → C0 P2P 搬[M,2048] 部分和, C0做元素加

Routed Expert流式 (C0/C1/C2/C3 统一, dual-VC 2×256MAC):
  - gate+up权重在SRAM中交叉存储, 搬入即可用
  - tiling: per-tile级别追踪DMA与计算的overlap
  - dual-VC: VC0=gate, VC1=up 并行; down阶段N-split
  - 每个VC独立读A和B, streamer bank需求 = 2×A_banks + 2×B_banks
  - phase分离: gate+up DMA与down DMA对外为两次独立占用, swiglu期间DMA对其他cluster可用
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
    """
    单个VC的GEMM周期数.
    计算过程: Mt×Nt个输出tile, 每个tile做Kt次K累加 + drain次写回.
    drain = ceil(meshRow × meshCol × 32bit / 1024bit) 写回周期
    """
    R, T, C = shape.meshRow, shape.tileSize, shape.meshCol
    Mt = math.ceil(M / R)
    Kt = math.ceil(K / T)
    Nt = math.ceil(N / C)
    drain = math.ceil(R * C * 32 / 1024)
    return Mt * Nt * (Kt + drain) + 5


def dma_cc(nbytes: int, bw: int) -> int:
    """DMA搬运周期, 数据量必须是64B的整数倍"""
    if nbytes <= 0 or bw <= 0:
        return 0
    aligned = math.ceil(nbytes / 64) * 64  # 64B对齐
    return math.ceil(aligned / bw)


def vc_utilization(M: int, shape: VersaCoreShape) -> float:
    """VersaCore M维利用率: M不能整除meshRow时有padding浪费"""
    Mt = math.ceil(M / shape.meshRow)
    actual_rows = Mt * shape.meshRow
    return M / actual_rows if actual_rows > 0 else 1.0


def vc_real_utilization(
    M: int, shape: VersaCoreShape, compute_cc: int, total_cc: int
) -> float:
    """
    VersaCore真实利用率 = M维利用率 × 时间维利用率.

    时间维利用率 = compute周期 / 总周期 (DMA-bound时VersaCore空等).
    M维利用率 = 有效行数 / 实际行数 (padding浪费).
    """
    m_util = vc_utilization(M, shape)
    time_util = compute_cc / total_cc if total_cc > 0 else 1.0
    return m_util * time_util


# ============================================================
#  Event (任务记录)
# ============================================================


@dataclass
class Event:
    """一个硬件操作事件"""

    name: str  # 事件名
    resource: str  # 使用的硬件资源
    start: int  # 开始时刻
    end: int  # 结束时刻
    data_bytes: int = 0
    macs: int = 0
    desc: str = ""
    formula: str = ""  # 持续时间计算公式

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass
class TCDMSnapshot:
    """TCDM状态快照"""

    time: int
    cluster: int
    contents: Dict[str, int]
    used_bytes: int
    free_bytes: int
    desc: str


# ============================================================
#  资源追踪器
# ============================================================


class ResourceTracker:
    """追踪所有硬件资源的空闲时刻"""

    def __init__(self):
        self._free: Dict[str, int] = {}

    def get(self, name: str) -> int:
        return self._free.get(name, 0)

    def set(self, name: str, t: int):
        self._free[name] = t

    def earliest(self, *names: str) -> int:
        """多个资源中最早全部空闲的时刻"""
        return max(self.get(n) for n in names) if names else 0


class TCDMTracker:
    """每个cluster的TCDM内容追踪"""

    def __init__(self, capacities: Dict[int, int]):
        self.cap = dict(capacities)
        self.contents: Dict[int, Dict[str, int]] = {c: {} for c in capacities}

    def store(self, cid: int, label: str, size: int):
        self.contents[cid][label] = size

    def evict(self, cid: int, label: str):
        self.contents[cid].pop(label, None)

    def evict_all(self, cid: int):
        self.contents[cid].clear()

    def used(self, cid: int) -> int:
        return sum(self.contents[cid].values())

    def free(self, cid: int) -> int:
        return self.cap[cid] - self.used(cid)

    def snapshot(self, cid: int, time: int, desc: str) -> TCDMSnapshot:
        return TCDMSnapshot(
            time=time,
            cluster=cid,
            contents=dict(self.contents[cid]),
            used_bytes=self.used(cid),
            free_bytes=self.free(cid),
            desc=desc,
        )


class BankChecker:
    """计算bank冲突导致的性能拉伸

    C2/C3 dual-VC (2×256MAC, 各自独立streamer接口):
      每个VC独立读取A和B → streamer需求 = 2×A_banks + 2×B_banks
      与1个512MAC VC的总bank预算一致
      (512MAC [R×8×C]: A_banks(R*8/8) + B_banks(8*C*wpe/8))

    C0/C1 单VC:
      streamer需求 = A_banks + B_banks
    """

    @staticmethod
    def stretch_dual_vc(
        shape: VersaCoreShape,
        wpe: float,
        dma_ports: int = 0,
        total_banks: int = 64,
    ) -> float:
        """
        Dual-VC bank冲突系数.
        两个VC各自独立读A和B: 2×A_banks + 2×B_banks
        """
        streamer = 2 * shape.a_banks() + 2 * shape.b_banks(wpe)
        total = streamer + dma_ports
        if total <= total_banks:
            return 1.0
        return total / total_banks

    @staticmethod
    def stretch_single_vc(
        shape: VersaCoreShape,
        wpe: float,
        dma_ports: int = 0,
        total_banks: int = 64,
    ) -> float:
        """单VC bank冲突系数 (C0/C1)"""
        streamer = shape.a_banks() + shape.b_banks(wpe)
        total = streamer + dma_ports
        if total <= total_banks:
            return 1.0
        return total / total_banks


# ============================================================
#  系统仿真器
# ============================================================


class SystemModel:
    """
    Core simulation engine.

    硬件资源命名:
      DMA通道: sram_xDMA, iDMA, C0_xDMA, C1_xDMA, C2_xDMA, C3_xDMA
      计算:    C0_VC, C1_VC, C2_VC, C3_VC
      加速器:  C0_elem, C1_elem, C2_elem, C3_elem
      Host:    Host (topK/scatter/softmax等)
    """

    def __init__(self, sys: SystemConfig, moe: MoELayerConfig):
        self.sys = sys
        self.moe = moe
        self.res = ResourceTracker()
        self.tcdm = TCDMTracker({c.cluster_id: c.tcdm_size_bytes for c in sys.clusters})
        self.events: List[Event] = []
        self.snapshots: List[TCDMSnapshot] = []

        # 【v21】初始化TCDM驻留内容
        # C0: 左半shared expert 全驻留 (gate_half + up_half + down_half_block) = 4.125MB
        # C1: 右半shared expert 全驻留 = 4.125MB
        # 两者结构和大小与routed expert完全一致
        for cid in (0, 1):
            self.tcdm.store(cid, "shared_gate_half", moe.shared_half_gate_weight)
            self.tcdm.store(cid, "shared_up_half", moe.shared_half_up_weight)
            self.tcdm.store(cid, "shared_down_half", moe.shared_half_down_weight)

    def _ev(
        self,
        resource: str,
        start: int,
        duration: int,
        name: str,
        formula: str = "",
        **kw,
    ) -> Event:
        ev = Event(
            resource=resource,
            start=start,
            end=start + duration,
            name=name,
            formula=formula,
            **kw,
        )
        self.events.append(ev)
        return ev

    def _snap(self, cid: int, time: int, desc: str):
        self.snapshots.append(self.tcdm.snapshot(cid, time, desc))

    # ----------------------------------------------------------
    #  DMA操作 (严格追踪资源占用)
    # ----------------------------------------------------------

    def dma_xdma(
        self, src_xdma: str, dst_xdma: str, nbytes: int, earliest: int, label: str
    ) -> Event:
        """xDMA↔xDMA P2P传输 (占用双方xDMA端口)"""
        dur = dma_cc(nbytes, self.sys.p2p_bw)
        start = max(earliest, self.res.get(src_xdma), self.res.get(dst_xdma))
        self.res.set(src_xdma, start + dur)
        self.res.set(dst_xdma, start + dur)
        return self._ev(
            f"DMA_{src_xdma}↔{dst_xdma}",
            start,
            dur,
            label,
            data_bytes=nbytes,
            formula=f"ceil({nbytes}/{self.sys.p2p_bw})={dur}",
        )

    def dma_idma(self, cid: int, nbytes: int, earliest: int, label: str) -> Event:
        """iDMA→cluster TCDM (不占cluster xDMA端口)"""
        dur = dma_cc(nbytes, self.sys.idma_bw)
        start = max(earliest, self.res.get("iDMA"))
        self.res.set("iDMA", start + dur)
        return self._ev(
            f"iDMA→C{cid}",
            start,
            dur,
            label,
            data_bytes=nbytes,
            formula=f"ceil({nbytes}/{self.sys.idma_bw})={dur}",
        )

    def dma_sram_to_cluster(
        self,
        cid: int,
        nbytes: int,
        earliest: int,
        use_xdma: bool,
        use_idma: bool,
        label: str,
    ) -> Event:
        """SRAM→cluster, 可选路径组合"""
        xbw = self.sys.sram_xdma_bw if use_xdma else 0
        ibw = self.sys.idma_bw if use_idma else 0
        total_bw = xbw + ibw
        dur = dma_cc(nbytes, total_bw)
        start = earliest
        if use_xdma:
            start = max(start, self.res.get("sram_xDMA"), self.res.get(f"C{cid}_xDMA"))
        if use_idma:
            start = max(start, self.res.get("iDMA"))
        end = start + dur
        if use_xdma:
            self.res.set("sram_xDMA", end)
            self.res.set(f"C{cid}_xDMA", end)
        if use_idma:
            self.res.set("iDMA", end)
        mode = (
            "xDMA+iDMA" if (use_xdma and use_idma) else ("xDMA" if use_xdma else "iDMA")
        )
        return self._ev(
            f"SRAM({mode})→C{cid}",
            start,
            dur,
            label,
            data_bytes=nbytes,
            formula=f"ceil({nbytes}/{total_bw})={dur} [{mode}]",
        )

    # ----------------------------------------------------------
    #  GEMM计算
    # ----------------------------------------------------------

    def compute_gemm(
        self,
        cid: int,
        M: int,
        K: int,
        N: int,
        shape: VersaCoreShape,
        earliest: int,
        label: str,
        dma_active: bool = False,
    ) -> Event:
        """
        单VC resident GEMM (C0/C1, 权重已在TCDM).
        """
        cluster = self.sys.clusters[cid]
        cc = gemm_cycles(M, K, N, shape)

        # bank冲突检查 (单VC)
        dma_ports = cluster.xdma_tcdm_ports if dma_active else 0
        stretch = BankChecker.stretch_single_vc(
            shape, self.moe.wpe, dma_ports, cluster.tcdm_num_banks
        )
        actual_cc = math.ceil(cc * stretch)

        start = max(earliest, self.res.get(f"C{cid}_VC"))
        self.res.set(f"C{cid}_VC", start + actual_cc)

        util = vc_utilization(M, shape)
        return self._ev(
            f"C{cid}_VC",
            start,
            actual_cc,
            label,
            macs=2 * M * K * N,
            formula=f"gemm({M},{K},{N},{shape})={cc}"
            f"{'×%.2f(bank)' % stretch if stretch > 1 else ''}"
            f" util={util:.0%}",
            desc=f"single VC, shape={shape}",
        )

    def _streaming_gemm_dual_vc_pertile(
        self,
        cid: int,
        M: int,
        K: int,
        N: int,
        shape: VersaCoreShape,
        weight_bytes: int,
        load_bw: int,
        earliest: int,
        label: str,
    ) -> Event:
        """
        Per-tile细粒度streaming GEMM仿真 (双VC通用).

        将GEMM拆分为Mt×Nt×Kt个tile, 逐tile追踪:
          - DMA搬运B tile的时间 (每K-tile需要搬入两个VC各自的B: 2×T×C×wpe)
          - compute tile的时间 (Kt次K累加 + drain)
          - bank冲突: 当DMA和streamer同时访问TCDM时, 按bank_stretch拉伸
          - tile0延迟: 必须等第一个tile的B数据完全到达才能开始计算
          - 后续tile: DMA和计算overlap, 受限于 max(dma_per_tile, compute_per_tile)

        双VC: 两个VC并行, 每个VC独立读A和B
          streamer bank = 2×A_banks + 2×B_banks (保守, 无broadcast)

        返回: 包含duration的Event
        """
        cluster = self.sys.clusters[cid]
        wpe = self.moe.wpe
        R, T, C = shape.meshRow, shape.tileSize, shape.meshCol

        # tile维度
        Mt = math.ceil(M / R)
        Kt = math.ceil(K / T)
        Nt = math.ceil(N / C)
        drain = math.ceil(R * C * 32 / 1024)

        # 每个output tile的纯计算周期 (Kt次K累加 + drain)
        compute_per_out_tile = Kt + drain

        # 每K-tile DMA: 两个VC各自的B数据
        # 每K-tile每VC需要: T × C × wpe bytes (B矩阵)
        # 两个VC总需求: 2 × T × C × wpe bytes
        b_per_ktile = math.ceil(2 * T * C * wpe / 64) * 64  # 64B对齐
        dma_per_ktile = dma_cc(b_per_ktile, load_bw) if load_bw > 0 else 0

        # bank冲突系数 (DMA + streamer同时)
        dma_ports = cluster.xdma_tcdm_ports if load_bw > 0 else 0
        bank_s = BankChecker.stretch_dual_vc(
            shape, wpe, dma_ports, cluster.tcdm_num_banks
        )

        # 第一个Nt-tile(output column): 需要等第一个K-tile DMA完成
        # 后续K-tile: DMA和compute可以pipeline (如果DMA更快)
        # 有bank冲突时, compute被拉伸

        # 每output tile (Mt*Nt个):
        #   计算时间 = (Kt + drain) * bank_stretch (with DMA)
        #   DMA时间 = Kt * dma_per_ktile
        # 第一个tile: 先等第一个K-tile DMA, 然后compute和DMA pipeline

        # --- Per-tile simulation ---
        total_dma = dma_cc(weight_bytes, load_bw) if load_bw > 0 else 0

        if load_bw <= 0 or total_dma == 0:
            # Resident模式: 无DMA
            compute = Mt * Nt * compute_per_out_tile + 5
            bank_s_no_dma = BankChecker.stretch_dual_vc(
                shape, wpe, 0, cluster.tcdm_num_banks
            )
            actual_compute = math.ceil(compute * bank_s_no_dma)
            start = max(earliest, self.res.get(f"C{cid}_VC"))
            self.res.set(f"C{cid}_VC", start + actual_compute)
            return self._ev(
                f"C{cid}_VC",
                start,
                actual_compute,
                label,
                macs=2 * 2 * M * K * N,
                formula=f"pertile_resident: {Mt}×{Nt}×({Kt}+{drain})×{bank_s_no_dma:.2f}+5={actual_compute}",
                desc=f"resident dual-vc pertile",
            )

        # Streaming模式: 逐output-tile追踪
        # DMA连续搬运总weight, 计算逐tile进行
        # 关键: 第一个output tile的第一个K-tile必须等DMA完成
        # 之后DMA和compute pipeline

        # 简化per-tile model (避免过度复杂):
        # 把所有output tile展开为序列: tile[0]...tile[Mt*Nt-1]
        # 每个tile需要Kt个K-tile的数据
        # DMA端: 连续搬入(tile顺序), 每个K-tile: dma_per_ktile cc
        # Compute端: 每个tile: (Kt+drain)*bank_stretch cc (bank冲突仅在DMA活跃时)

        # 首先: 第一个tile
        # DMA搬第一个tile的Kt个K-tile: Kt * dma_per_ktile cc
        first_tile_dma = Kt * dma_per_ktile

        # 实际计算: pipeline中, 第一个K-tile DMA完成后compute开始
        # 后续K-tile: DMA和compute同时 → per K-tile时间 = max(dma_per_ktile, 1) (bank stretch)
        # tile的compute = 第一个K-tile等待 + (Kt-1)个K-tile pipeline + drain
        compute_with_bank = math.ceil(compute_per_out_tile * bank_s)

        # Per K-tile pipeline rate (DMA和compute哪个慢)
        pipeline_rate = max(
            dma_per_ktile, math.ceil(1 * bank_s)
        )  # 每K-tile被DMA或bank限制

        # 第一个tile: 等第一个K-tile DMA → pipeline剩余K-tile → drain
        tile0_time = dma_per_ktile + (Kt - 1) * pipeline_rate + drain

        # 后续tile: 不需要额外等DMA (DMA持续进行)
        # 但需要检查DMA是否跟得上
        # 每tile需要Kt个K-tile的B数据, compute需要compute_with_bank cc
        # DMA端: Kt*dma_per_ktile cc per tile
        # Compute端: compute_with_bank cc per tile (含bank stretch)
        # pipeline: 连续tile的瓶颈 = max(dma_per_tile, compute_with_bank)
        dma_per_outtile = Kt * dma_per_ktile
        tile_pipeline_rate = max(dma_per_outtile, compute_with_bank)

        # 总时间 = tile0 + (Mt*Nt - 1) * tile_pipeline_rate + 5(overhead)
        n_out_tiles = Mt * Nt
        if n_out_tiles <= 1:
            total = tile0_time + 5
        else:
            total = tile0_time + (n_out_tiles - 1) * tile_pipeline_rate + 5

        # 还需要与DMA总时间取max (DMA可能比计算慢)
        total = max(total, total_dma)

        # 确定bound类型
        pure_compute = gemm_cycles(M, K, N, shape)  # 无bank冲突的理想计算
        if total_dma > pure_compute:
            bound = "DMA-bound"
        elif bank_s > 1.0:
            bound = "bank-conflict"
        else:
            bound = "compute-bound"

        start = max(earliest, self.res.get(f"C{cid}_VC"))
        self.res.set(f"C{cid}_VC", start + total)

        return self._ev(
            f"C{cid}_VC",
            start,
            total,
            label,
            macs=2 * 2 * M * K * N,
            formula=f"pertile: {n_out_tiles}tiles "
            f"tile0={tile0_time} pipe={tile_pipeline_rate} "
            f"dma_total={total_dma} bank_s={bank_s:.2f} "
            f"→{total} [{bound}]",
            desc=f"{bound} dual-vc pertile bw={load_bw} "
            f"2A+2B banks={2*shape.a_banks()+2*shape.b_banks(wpe)}+{dma_ports}dma",
        )

    def streaming_gemm_dual_vc_gu(
        self,
        cid: int,
        M: int,
        K: int,
        N: int,
        shape: VersaCoreShape,
        weight_bytes: int,
        load_bw: int,
        earliest: int,
        label: str,
    ) -> Event:
        """
        Dual-VC Gate+Up流式: VC0做gate, VC1做up, 并行.

        gate和up权重在SRAM中交叉存储, DMA连续搬入.
        每个VC各自做完整GEMM(M,K,N), A和B各自独立读取.

        Per-tile细粒度追踪: DMA/compute/bank三方面的overlap和冲突.
        """
        return self._streaming_gemm_dual_vc_pertile(
            cid, M, K, N, shape, weight_bytes, load_bw, earliest, label
        )

    def streaming_gemm_dual_vc_down(
        self,
        cid: int,
        M: int,
        N_in: int,
        K_half: int,
        shape: VersaCoreShape,
        weight_bytes: int,
        load_bw: int,
        earliest: int,
        label: str,
    ) -> Event:
        """
        Dual-VC Down N-split流式: VC0做[M,N,K/2], VC1做[M,N,K/2].

        Per-tile细粒度追踪: 与gu相同的per-tile pipeline模型.
        """
        return self._streaming_gemm_dual_vc_pertile(
            cid, M, N_in, K_half, shape, weight_bytes, load_bw, earliest, label
        )

    def compute_gemm_dual_vc_gu(
        self,
        cid: int,
        M: int,
        K: int,
        N: int,
        shape: VersaCoreShape,
        earliest: int,
        label: str,
    ) -> Event:
        """
        Dual-VC gate+up resident GEMM (权重已在TCDM).
        VC0=gate, VC1=up, 并行, 每个VC做完整GEMM(M,K,N).
        Bank: 2×A_banks + 2×B_banks (每个VC独立读A和B, 无DMA)
        """
        cluster = self.sys.clusters[cid]
        cc = gemm_cycles(M, K, N, shape)

        stretch = BankChecker.stretch_dual_vc(
            shape, self.moe.wpe, 0, cluster.tcdm_num_banks
        )
        actual_cc = math.ceil(cc * stretch)

        start = max(earliest, self.res.get(f"C{cid}_VC"))
        self.res.set(f"C{cid}_VC", start + actual_cc)

        util = vc_utilization(M, shape)
        return self._ev(
            f"C{cid}_VC",
            start,
            actual_cc,
            label,
            macs=2 * 2 * M * K * N,
            formula=f"dual_vc_gu_resident: gemm({M},{K},{N},{shape})={cc}"
            f"{'×%.2f(bank)' % stretch if stretch > 1 else ''}"
            f" util={util:.0%}",
            desc=f"dual-vc gu resident, shape={shape}",
        )

    def compute_gemm_dual_vc_down(
        self,
        cid: int,
        M: int,
        N_in: int,
        K_half: int,
        shape: VersaCoreShape,
        earliest: int,
        label: str,
    ) -> Event:
        """
        Dual-VC down N-split resident GEMM (权重已在TCDM).
        VC0=[M,N,K/2], VC1=[M,N,K/2], 并行.
        """
        cluster = self.sys.clusters[cid]
        cc = gemm_cycles(M, N_in, K_half, shape)

        stretch = BankChecker.stretch_dual_vc(
            shape, self.moe.wpe, 0, cluster.tcdm_num_banks
        )
        actual_cc = math.ceil(cc * stretch)

        start = max(earliest, self.res.get(f"C{cid}_VC"))
        self.res.set(f"C{cid}_VC", start + actual_cc)

        util = vc_utilization(M, shape)
        return self._ev(
            f"C{cid}_VC",
            start,
            actual_cc,
            label,
            macs=2 * 2 * M * N_in * K_half,
            formula=f"dual_vc_dn_resident: gemm({M},{N_in},{K_half},{shape})={cc}"
            f"{'×%.2f(bank)' % stretch if stretch > 1 else ''}"
            f" util={util:.0%}",
            desc=f"dual-vc down_nsplit resident, shape={shape}",
        )

    def elemwise(self, cid: int, n_elements: int, earliest: int, label: str) -> Event:
        """SiLU / GLU等逐元素操作"""
        rate = self.sys.clusters[cid].elemwise_rate
        cc = math.ceil(n_elements / rate)
        start = max(earliest, self.res.get(f"C{cid}_elem"))
        self.res.set(f"C{cid}_elem", start + cc)
        return self._ev(
            f"C{cid}_elem", start, cc, label, formula=f"ceil({n_elements}/{rate})={cc}"
        )

    # ----------------------------------------------------------
    #  Shared Expert (v21: C0/C1 dual-VC, N方向切半各驻留半个shared)
    # ----------------------------------------------------------

    def simulate_shared_expert(self, M: int) -> int:
        """
        【v21】Shared expert在C0+C1上的执行 (dual-VC, N方向对半切).

        硬件: C0/C1各为 2×256MAC dual-VC, 权重布局与routed expert完全一致.
        Shared intermediate=2816 → 切为 C0左半[1408] + C1右半[1408].

        每个cluster内部 (与routed expert resident模式完全相同):
          - dual-VC gate+up: VC0=gate_half, VC1=up_half, resident, [M,K=2048,N=1408]
          - swiglu: elementwise, M×1408
          - dual-VC down N-split: 各VC做[M,N=1408,K_half=1024], 驻留
          - 输出: [M, K=2048] 部分和 (两cluster需相加)

        数据流:
          1. token_A(M×2048) → C0(sram_xDMA), C1(iDMA) 并行
          2. C0/C1 并行 gate+up (dual-VC resident) → swiglu
          3. C0/C1 并行 down (dual-VC N-split resident) → 部分和 [M,2048]
          4. C1 → C0 P2P 搬部分和, C0做elemwise加得最终输出

        返回: 完成时刻
        """
        K = self.moe.hidden_size  # 2048
        S_half = self.moe.shared_half_intermediate  # 1408
        K_half = K // 2  # 1024
        tok_bytes = M * K  # token_A (INT8)

        # 与routed expert一致的256MAC shape选择
        shapes = generate_shapes(256, 8)
        shape_gu = min(shapes, key=lambda s: gemm_cycles(M, K, S_half, s))
        shape_dn = min(shapes, key=lambda s: gemm_cycles(M, S_half, K_half, s))

        # --- Step 1: token_A → C0 和 C1 并行 ---
        ev_tok_c0 = self.dma_xdma(
            "sram_xDMA", "C0_xDMA", tok_bytes, 0, f"token_A→C0 ({tok_bytes}B)"
        )
        ev_tok_c1 = self.dma_idma(1, tok_bytes, 0, f"token_A→C1 ({tok_bytes}B)")
        self.tcdm.store(0, "token_A", tok_bytes)
        self.tcdm.store(1, "token_A", tok_bytes)

        # --- Step 2: C0/C1 并行 gate+up (dual-VC resident) ---
        ev_gu_c0 = self.compute_gemm_dual_vc_gu(
            0,
            M,
            K,
            S_half,
            shape_gu,
            ev_tok_c0.end,
            f"C0 shared gate+up [{M},{K},{S_half}] resident",
        )
        ev_gu_c1 = self.compute_gemm_dual_vc_gu(
            1,
            M,
            K,
            S_half,
            shape_gu,
            ev_tok_c1.end,
            f"C1 shared gate+up [{M},{K},{S_half}] resident",
        )

        # --- Step 3: C0/C1 并行 swiglu ---
        ev_sw_c0 = self.elemwise(
            0, M * S_half, ev_gu_c0.end, f"C0 shared SwiGLU ({M*S_half} elem)"
        )
        ev_sw_c1 = self.elemwise(
            1, M * S_half, ev_gu_c1.end, f"C1 shared SwiGLU ({M*S_half} elem)"
        )
        self.tcdm.store(0, "swiglu_half", M * S_half)
        self.tcdm.store(1, "swiglu_half", M * S_half)

        # --- Step 4: C0/C1 并行 down (dual-VC N-split resident) ---
        # 每cluster做 swiglu_half[M,1408] × W_down_half[1408,2048] → [M,2048] 部分和
        # dual-VC N-split: VC0做[M,1408,1024], VC1做[M,1408,1024]
        ev_dn_c0 = self.compute_gemm_dual_vc_down(
            0,
            M,
            S_half,
            K_half,
            shape_dn,
            ev_sw_c0.end,
            f"C0 shared down [{M},{S_half},{K_half}]×2 resident",
        )
        ev_dn_c1 = self.compute_gemm_dual_vc_down(
            1,
            M,
            S_half,
            K_half,
            shape_dn,
            ev_sw_c1.end,
            f"C1 shared down [{M},{S_half},{K_half}]×2 resident",
        )

        # --- Step 5: Merge — C1 → C0 P2P 部分和, C0做 elemwise add ---
        partial_bytes = M * K  # INT8存储的部分和
        merge_start = max(ev_dn_c0.end, ev_dn_c1.end)
        ev_p2p_merge = self.dma_xdma(
            "C1_xDMA",
            "C0_xDMA",
            partial_bytes,
            merge_start,
            f"shared merge C1→C0 ({partial_bytes}B)",
        )
        ev_add = self.elemwise(
            0, M * K, ev_p2p_merge.end, f"C0 shared merge-add ({M*K} elem)"
        )

        # TCDM清理中间结果 (保留驻留权重)
        for cid in (0, 1):
            self.tcdm.evict(cid, "token_A")
            self.tcdm.evict(cid, "swiglu_half")
        self._snap(0, ev_add.end, "shared expert done (dual-VC N-split)")
        self._snap(1, ev_add.end, "shared expert done (dual-VC N-split)")

        return ev_add.end

    # ----------------------------------------------------------
    #  Router (在C3上执行)
    # ----------------------------------------------------------

    def simulate_router(self, M: int) -> int:
        """模拟Router GEMM + topK + scatter + softmax"""
        K = self.moe.hidden_size
        N_out = self.moe.n_routed_experts
        rtr_w = self.moe.router_weight_size
        tok_bytes = M * K
        shape = VersaCoreShape(2, 8, 16)  # 小矩阵用小shape

        # router权重 + token_A → C3 (sram_xDMA和iDMA并行)
        ev_rw = self.dma_idma(3, rtr_w, 0, f"router_w iDMA→C3 ({rtr_w}B)")
        ev_tok = self.dma_xdma(
            "sram_xDMA", "C3_xDMA", tok_bytes, 0, f"token_A sram→C3 ({tok_bytes}B)"
        )
        self.tcdm.store(3, "router_w", rtr_w)
        self.tcdm.store(3, "token_A_router", tok_bytes)

        rtr_start = max(ev_rw.end, ev_tok.end)
        ev_rtr = self.compute_gemm(
            3, M, K, N_out, shape, rtr_start, f"router [{M},{K},{N_out}]"
        )
        # topK + scatter + softmax (Host端, 零DMA)
        ev_topk = self._ev("Host", ev_rtr.end, 5000, "topK", formula="~5000cc overhead")
        ev_scatter = self._ev("Host", ev_topk.end, 5000, "scatter", formula="~5000cc")
        ev_softmax = self._ev(
            "Host", ev_scatter.end, 15000, "softmax", formula="~15000cc"
        )
        self.res.set("Host", ev_softmax.end)

        # 清理router临时数据
        self.tcdm.evict(3, "router_w")
        self.tcdm.evict(3, "token_A_router")

        return ev_scatter.end  # routing_ready = 知道每个expert分配多少token

    # ----------------------------------------------------------
    #  Token分发到C2/C3 (从C0获取, 节省SRAM带宽)
    # ----------------------------------------------------------

    def distribute_tokens(self, M: int, routing_ready: int) -> Dict[int, int]:
        """
        分发token_A到C2和C3.
        可从C0获取(C0还有token_A的residue)或从SRAM获取.
        这里使用SRAM: sram_xDMA→C2, iDMA→C3 并行.
        """
        tok_bytes = M * self.moe.hidden_size
        ev_c2 = self.dma_sram_to_cluster(
            2,
            tok_bytes,
            routing_ready,
            use_xdma=True,
            use_idma=False,
            label=f"token_A→C2 ({tok_bytes}B)",
        )
        ev_c3 = self.dma_sram_to_cluster(
            3,
            tok_bytes,
            routing_ready,
            use_xdma=False,
            use_idma=True,
            label=f"token_A→C3 ({tok_bytes}B)",
        )
        self.tcdm.store(2, "token_A", tok_bytes)
        self.tcdm.store(3, "token_A", tok_bytes)
        return {2: ev_c2.end, 3: ev_c3.end}

    # ----------------------------------------------------------
    #  Routed Expert执行 (streaming或resident)
    # ----------------------------------------------------------

    def execute_routed_expert(
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
        执行一个routed expert (C2/C3, dual-VC):

        Gate+Up阶段: VC0=gate, VC1=up, 并行
          - 每个VC做完整GEMM(M, K=2048, N=1408)
          - gate+up权重在SRAM交叉存储, DMA连续搬入
          - DMA B需求: 2×T×C×wpe bytes/cycle (双VC独立B流)

        SwiGLU: gate_out经SiLU后与up_out逐元素相乘 → active_A

        Down阶段: N-split, VC0=[M,N,K/2], VC1=[M,N,K/2]
          - DMA搬入完整down权重, 两个VC各取自己的N切片
          - DMA B需求同上: 2×T×C×wpe bytes/cycle

        dma_channels: "xdma","idma","both","none"
        返回: (start, end)
        """
        K = self.moe.hidden_size  # 2048
        N = self.moe.moe_intermediate_size  # 1408
        K_half = K // 2  # 1024
        wpe = self.moe.wpe
        gate_w = self.moe.expert_gate_weight
        up_w = self.moe.expert_up_weight
        down_w = self.moe.expert_down_weight
        cluster = self.sys.clusters[cid]

        if resident:
            # 权重已在TCDM, 无DMA
            # Gate+Up: dual-VC并行 (resident)
            ev_gu = self.compute_gemm_dual_vc_gu(
                cid, M, K, N, shape, earliest, f"E{eid} gate+up [{M},{K},{N}] resident"
            )
            # SwiGLU
            ev_sw = self.elemwise(cid, M * N, ev_gu.end, f"E{eid} SwiGLU")
            # Down: dual-VC N-split (resident)
            ev_dn = self.compute_gemm_dual_vc_down(
                cid,
                M,
                N,
                K_half,
                shape,
                ev_sw.end,
                f"E{eid} down [{M},{N},{K_half}]×2 resident",
            )
            return ev_gu.start, ev_dn.end

        # --- Streaming mode (v21: gate+up和down的DMA占用分离) ---
        # Phase A: gate+up DMA 独立占用, 完成即释放 (swiglu期间DMA free)
        dma_start = earliest
        if dma_channels in ("xdma", "both"):
            dma_start = max(
                dma_start, self.res.get("sram_xDMA"), self.res.get(f"C{cid}_xDMA")
            )
        if dma_channels in ("idma", "both"):
            dma_start = max(dma_start, self.res.get("iDMA"))

        # Gate+Up: 双VC并行流式 (gate_w + up_w 交叉搬入)
        gu_bytes = gate_w + up_w
        gu_dma_dur = dma_cc(gu_bytes, load_bw)
        gu_dma_end = dma_start + gu_dma_dur

        # [v21] gate+up DMA结束后立即释放DMA资源, 允许其他cluster在本cluster做swiglu期间使用
        if dma_channels in ("xdma", "both"):
            self.res.set("sram_xDMA", gu_dma_end)
            self.res.set(f"C{cid}_xDMA", gu_dma_end)
        if dma_channels in ("idma", "both"):
            self.res.set("iDMA", gu_dma_end)

        ev_gu = self.streaming_gemm_dual_vc_gu(
            cid,
            M,
            K,
            N,
            shape,
            gu_bytes,
            load_bw,
            dma_start,
            f"E{eid} gate+up [{M},{K},{N}] stream (dual-VC)",
        )

        # SwiGLU — DMA此时空闲, 其他cluster的phase可插入
        ev_sw = self.elemwise(cid, M * N, ev_gu.end, f"E{eid} SwiGLU")

        # Phase B: down DMA 重新申请 — 可能被其他cluster占用, max()等
        dn_dma_start = ev_sw.end
        if dma_channels in ("xdma", "both"):
            dn_dma_start = max(
                dn_dma_start,
                self.res.get("sram_xDMA"),
                self.res.get(f"C{cid}_xDMA"),
            )
        if dma_channels in ("idma", "both"):
            dn_dma_start = max(dn_dma_start, self.res.get("iDMA"))

        dn_dma_dur = dma_cc(down_w, load_bw)
        dn_dma_end = dn_dma_start + dn_dma_dur

        ev_dn = self.streaming_gemm_dual_vc_down(
            cid,
            M,
            N,
            K_half,
            shape,
            down_w,
            load_bw,
            dn_dma_start,
            f"E{eid} down [{M},{N},{K_half}]×2 stream (dual-VC nsplit)",
        )

        # 释放down DMA资源 (第二段)
        if dma_channels in ("xdma", "both"):
            self.res.set("sram_xDMA", dn_dma_end)
            self.res.set(f"C{cid}_xDMA", dn_dma_end)
        if dma_channels in ("idma", "both"):
            self.res.set("iDMA", dn_dma_end)

        # TCDM更新
        self.tcdm.evict_all(cid)
        self.tcdm.store(cid, f"E{eid}_weights", gate_w + up_w + down_w)
        self._snap(cid, ev_dn.end, f"E{eid} done on C{cid}")

        return ev_gu.start, ev_dn.end

    # ----------------------------------------------------------
    #  工具方法
    # ----------------------------------------------------------

    def get_makespan(self) -> int:
        return max(ev.end for ev in self.events) if self.events else 0

    def get_resource_timeline(self) -> Dict[str, List[Event]]:
        """按资源分组的事件"""
        timeline = {}
        for ev in sorted(self.events, key=lambda e: e.start):
            timeline.setdefault(ev.resource, []).append(ev)
        return timeline
