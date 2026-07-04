#!/usr/bin/env python3
"""
形状选择 LUT (v23) — workload 全程资源饱和视角
============================================================

核心理念 (修正 v22 的 "BW 匹配" 误区):
  - 不再要求 shape_bw_demand ≤ load_bw.
  - 当 shape_bw_demand > load_bw (DMA-bound):
      total_cc = weight_bytes / load_bw  (与 shape 无关!)
      选 shape 唯一影响: M 维 padding 利用率 + bank 冲突.
      所以仍应选 **M util 最高** 的 shape, 避免浪费 padding.
  - 当 shape_bw_demand ≤ load_bw (compute-bound):
      total_cc = compute_cc (VC 喂满)
      DMA 早于 compute 完成 → xdma_free_at 提前释放,
      调度器立即把这段空闲 DMA 用于**预取下一个 expert 的权重**.
      所以 "shape bw_demand 小于 load_bw" 不是坏事, 只是 DMA 能兼职做别的.
  - 真正的浪费是: 选 M util 低的 shape, 让 VC 每 tile 有一半行空转.

形状选择原则 (workload 全程视角):
  a) 最大化 **VC 真实利用率** = m_util × (compute_cc / total_cc).
  b) M 方向利用率: Mt×R 要接近 ntok, 避免 padding.
  c) 不要避开 "bw_demand > load_bw" 的 shape — DMA-bound 时反而
     应选 m_util 最高的, 因为 cc 取决于 DMA, 与 shape 无关.

带宽需求表 (dual-VC, W4A8, 每 K-tile):
  Shape     meshRow×tileSize×meshCol   单VC B 需求   双VC B 需求
  1×8×32           1×8×32                128 B/cc       256 B/cc
  2×8×16           2×8×16                 64 B/cc       128 B/cc
  4×8×8            4×8×8                  32 B/cc        64 B/cc
  8×8×4            8×8×4                  16 B/cc        32 B/cc
  16×8×2          16×8×2                   8 B/cc        16 B/cc
  32×8×1          32×8×1                   4 B/cc         8 B/cc
"""

import math
from typing import Tuple, List, Dict
from config import VersaCoreShape, generate_shapes

# v24: 候选 shape 限定为三种 (VersaCore 架构决策).
# 理由: 其它 shape 要么 M 维粒度过粗 (1×8×32 R=1 永远 50%-100%),
# 要么 mesh_col 过窄导致 compute-per-tile < 1 cycle (bank 冲突).
# 这三种 shape 的 b_bw_demand 分别 = 256/128/64 B/cc (dual-VC),
# 正好覆盖 DMA_BOTH / 64×2 并行 / resident 三种物理模式.
SHAPES_256: List[VersaCoreShape] = [
    VersaCoreShape(2, 8, 32),  # bw=256B/cc, DMA-bound @ 128
    VersaCoreShape(4, 8, 16),  # bw=128B/cc, 匹配 DMA_BOTH
    VersaCoreShape(8, 8, 8),  # bw= 64B/cc, 匹配单通道并行
]
MB = 1024 * 1024


def shape_bw_demand(shape: VersaCoreShape, wpe: float) -> int:
    """双 VC 的 B 矩阵带宽需求 (bytes/cc)."""
    return int(2 * shape.tileSize * shape.meshCol * wpe)


def m_utilization(ntok: int, shape: VersaCoreShape) -> float:
    if ntok <= 0:
        return 0.0
    Mt = math.ceil(ntok / shape.meshRow)
    return ntok / (Mt * shape.meshRow)


def gemm_pure_cc(M: int, K: int, N: int, shape: VersaCoreShape) -> int:
    R, T, C = shape.meshRow, shape.tileSize, shape.meshCol
    Mt = math.ceil(M / R)
    Kt = math.ceil(K / T)
    Nt = math.ceil(N / C)
    drain = math.ceil(R * C * 32 / 1024)
    return Mt * Nt * (Kt + drain) + 5


def estimate_cc(
    ntok: int,
    K: int,
    N: int,
    shape: VersaCoreShape,
    load_bw: int,
    wpe: float,
    resident: bool,
) -> Tuple[int, float]:
    """
    估计 expert 总周期 (gate+up + swiglu + down) 以及真实 VC 利用率.

    物理约束:
      - 流式时 per-tile DMA 时间 = 2×T×C×wpe / load_bw
      - compute per-tile = 1 cycle (不含 drain)
      - 若 dma_per_tile > 1 → DMA-bound, versacore 空转
      - 若 compute_per_tile > dma_per_tile → compute-bound
    """
    if ntok <= 0:
        return 0, 0.0

    K_half = K // 2
    R, T, C = shape.meshRow, shape.tileSize, shape.meshCol

    # 纯计算 (两阶段 + swiglu)
    gu_compute = gemm_pure_cc(ntok, K, N, shape)
    dn_compute = gemm_pure_cc(ntok, N, K_half, shape)
    swiglu = math.ceil(ntok * N / 128)  # elemwise 128 elem/cc
    pure_compute = gu_compute + swiglu + dn_compute

    m_u = m_utilization(ntok, shape)

    if resident:
        # Bank 冲突: 2×A_banks + 2×B_banks (无 DMA 端口)
        a_banks = math.ceil(shape.meshRow * shape.tileSize / 8)
        b_banks = math.ceil(shape.tileSize * shape.meshCol * wpe / 8)
        streamer = 2 * a_banks + 2 * b_banks
        bank_s = streamer / 64 if streamer > 64 else 1.0
        total = math.ceil(pure_compute * bank_s)
        time_u = pure_compute / total if total > 0 else 1.0
        return total, m_u * time_u

    # --- 流式 ---
    bw_demand = shape_bw_demand(shape, wpe)
    # 每 K-tile 所需字节 = 2*T*C*wpe (对齐到 64B)
    b_per_ktile = math.ceil(bw_demand / 64) * 64
    dma_per_ktile = math.ceil(b_per_ktile / load_bw) if load_bw > 0 else 0

    # bank 冲突 (含 DMA 端口 16)
    a_banks = math.ceil(shape.meshRow * shape.tileSize / 8)
    b_banks = math.ceil(shape.tileSize * shape.meshCol * wpe / 8)
    streamer = 2 * a_banks + 2 * b_banks
    bank_total = streamer + 16
    bank_s = bank_total / 64 if bank_total > 64 else 1.0

    def stream_stage_cc(M: int, K_: int, N_: int) -> Tuple[int, int, int]:
        """返回 (total_cc, compute_cc, dma_cc). compute_cc = 纯 compute (无 DMA 等待)."""
        Mt = math.ceil(M / R)
        Kt = math.ceil(K_ / T)
        Nt = math.ceil(N_ / C)
        drain = math.ceil(R * C * 32 / 1024)
        compute_per_outtile = Kt + drain
        compute_with_bank = math.ceil(compute_per_outtile * bank_s)
        pipeline_rate = max(dma_per_ktile, math.ceil(bank_s))
        tile0 = dma_per_ktile + (Kt - 1) * pipeline_rate + drain
        dma_per_outtile = Kt * dma_per_ktile
        tile_pipeline = max(dma_per_outtile, compute_with_bank)
        n_out = Mt * Nt
        if n_out <= 1:
            total_cc = tile0 + 5
        else:
            total_cc = tile0 + (n_out - 1) * tile_pipeline + 5
        weight_bytes = int(K_ * N_ * wpe) * 2
        total_dma = math.ceil(weight_bytes / load_bw) if load_bw > 0 else 0
        total_cc = max(total_cc, total_dma)
        compute_only = Mt * Nt * compute_per_outtile
        return total_cc, compute_only, total_dma

    gu_total, gu_comp, gu_dma = stream_stage_cc(ntok, K, N)
    dn_total, dn_comp, dn_dma = stream_stage_cc(ntok, N, K_half)

    # v21 phase 分离: down DMA 可以紧跟 gate+up DMA (如果 compute 够久 swiglu 期间结束)
    # 简化: total ≈ gu_total + swiglu + dn_total
    total = gu_total + swiglu + dn_total

    compute_time = gu_comp + swiglu + dn_comp
    time_u = compute_time / total if total > 0 else 1.0
    return total, m_u * time_u


def pick_shape(
    ntok: int,
    load_bw: int,
    resident: bool = False,
    K: int = 2048,
    N: int = 1408,
    wpe: float = 0.5,
) -> Tuple[VersaCoreShape, int, float, int]:
    """
    为 (ntok, load_bw, resident) 选最优 shape (v23 workload 全程视角).
    返回: (shape, estimated_cc, vc_util, bw_demand).

    选择准则:
      1. estimate_cc 已正确建模 max(compute, dma) 与 pipeline.
      2. 直接按 (cc 升序, util 降序) 选 — 不再因 bw_demand > load_bw 预排除.
      3. DMA-bound 场景下所有 shape 的 cc 相近, tie-break 用 util, 自然选 M 维最优.
      4. Compute-bound 场景下大 shape 吃满 VC, 小 shape 被排除.
    """
    best = None
    for s in SHAPES_256:
        cc, util = estimate_cc(ntok, K, N, s, load_bw, wpe, resident)
        if cc <= 0:
            continue
        # ntok < meshRow 时 M 维利用率 < 1/R, padding 浪费 — 但若 DMA-bound 下
        # 仍等效, util 会自然反映, 不需特殊处理.
        key = (cc, -util)
        if best is None or key < best[0]:
            best = (key, s, util, shape_bw_demand(s, wpe), cc)

    if best is None:
        s = SHAPES_256[0]
        cc, util = estimate_cc(ntok, K, N, s, load_bw, wpe, resident)
        return s, cc, util, shape_bw_demand(s, wpe)

    _, s, util, bw, cc = best
    return s, cc, util, bw


def build_shape_lut(K: int = 2048, N: int = 1408, wpe: float = 0.5) -> Dict:
    """
    预生成完整 LUT: (ntok, bw_mode, resident) -> (shape, cc, util, bw_demand).

    bw_mode:
      0  : resident (load_bw=0)
      64 : 并行流式, 单通道
      128: 独占 DMA_BOTH
    """
    lut = {}
    for ntok in list(range(1, 17)) + [24, 32, 48, 64, 96, 128]:
        for bw, resident in [(0, True), (64, False), (128, False)]:
            shape, cc, util, bw_demand = pick_shape(ntok, bw, resident, K, N, wpe)
            lut[(ntok, bw, resident)] = {
                "shape": (shape.meshRow, shape.tileSize, shape.meshCol),
                "cc": cc,
                "util": util,
                "bw_demand": bw_demand,
            }
    return lut


if __name__ == "__main__":
    # 打印 LUT 供审阅
    print("=" * 80)
    print(
        f"{'ntok':>4} {'mode':>10} {'shape':>12} {'cc':>8} {'util':>6} {'bw_need':>8}"
    )
    print("-" * 80)
    for ntok in [1, 2, 4, 8, 12, 16, 24, 32, 64]:
        for bw, resident in [(0, True), (64, False), (128, False)]:
            shape, cc, util, bw_demand = pick_shape(ntok, bw, resident)
            mode = "resident" if resident else f"stream@{bw}"
            print(
                f"{ntok:>4} {mode:>10} "
                f"{shape.meshRow}×{shape.tileSize}×{shape.meshCol:<6} "
                f"{cc:>8} {util:>5.1%} {bw_demand:>6}B/cc"
            )
    print("=" * 80)
