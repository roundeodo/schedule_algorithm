"""
HeMAiA SoC 硬件配置 + DeepSeek V2 Lite MoE 层参数
===================================================

硬件配置:
  - 4个cluster (C0, C1, C2, C3), 每个有 xDMA(64B/cc), TCDM(5MB, 64 banks, 8B/bank/cc)
  - 每个cluster内有 SiLU加速器 和 乘法加速器
  - SRAM: sram_xDMA(64B/cc) + iDMA(64B/cc), 两者独立可并行
  - 互联: 任意xDMA间P2P一对一; iDMA直连cluster TCDM(不占xDMA端口)

量化方案: W4A8 (权重 INT4, 激活 INT8)
  - 权重(B矩阵): INT4 → 0.5 byte/elem
  - 激活(A矩阵): INT8 → 1 byte/elem
  - 中间结果(output): INT32 → 4 byte/elem (累加器)

【v21新硬件】所有cluster统一结构: 2×256MAC dual-VC, 5MB TCDM, 64 banks

驻留方案 (Shared Expert, W4A8) — 不再打包, N方向对半切:
  Shared intermediate=2816 → 切为 C0(左半1408) + C1(右半1408)
  C0(dual-VC): gate_half + up_half + down_half_block = 3×(2048×1408×0.5B) = 4.125MB
  C1(dual-VC): gate_half + up_half + down_half_block = 3×(2048×1408×0.5B) = 4.125MB
  Shared expert因此与routed expert权重布局完全一致, 复用同一dual-VC流程.
  C2/C3: 空(routed expert流式搬入, gate+up交叉存储)

【v21计算流程变更】单个expert按相位分段:
  Phase1: 对全部M做 gate+up(dual-VC并行) → swiglu
  Phase2: 对全部M做 down(dual-VC N-split)
  重点: Phase1和Phase2的DMA资源独立释放, 调度器可在phase间交错其他expert任务

Routed expert (INT4): gate+up+down = 3×(2048×1408×0.5B) = 4.125MB, 驻入5MB TCDM
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict


# ============================================================
#  VersaCore Shape 定义
# ============================================================


@dataclass
class VersaCoreShape:
    """
    VersaCore 阵列形状配置: meshRow × tileSize × meshCol
    每个周期完成 meshRow×tileSize×meshCol 次 MAC 操作

    streamer每周期从TCDM读取:
      A矩阵(激活, INT8): meshRow × tileSize bytes
      B矩阵(权重):        tileSize × meshCol × weight_bytes_per_elem
    """

    meshRow: int
    tileSize: int
    meshCol: int

    @property
    def macs(self) -> int:
        return self.meshRow * self.tileSize * self.meshCol

    def a_bytes(self) -> int:
        """A矩阵(激活)每周期读取量, 激活始终INT8"""
        return self.meshRow * self.tileSize

    def b_bytes(self, wpe: float) -> int:
        """B矩阵(权重)每周期读取量"""
        return int(self.tileSize * self.meshCol * wpe)

    def a_banks(self) -> int:
        """A矩阵占用bank数"""
        return math.ceil(self.a_bytes() / 8)

    def b_banks(self, wpe: float) -> int:
        """B矩阵占用bank数"""
        return math.ceil(self.b_bytes(wpe) / 8)

    def streamer_banks_gu(self, wpe: float, num_vc: int = 1) -> int:
        """
        Gate+Up阶段streamer bank需求 (dual-VC: VC0=gate, VC1=up)
        每个VC独立从TCDM读A和B, 无broadcast.
        总需求 = num_vc × (A_banks + B_banks)
        """
        return num_vc * (self.a_banks() + self.b_banks(wpe))

    def streamer_banks_down(self, wpe: float, num_vc: int = 1) -> int:
        """
        Down阶段streamer bank需求 (dual-VC: N-split, 各算[M,N,K/2])
        每个VC独立从TCDM读A和B, 无broadcast.
        总需求 = num_vc × (A_banks + B_banks)
        """
        return num_vc * (self.a_banks() + self.b_banks(wpe))

    def streamer_banks_single(self, wpe: float) -> int:
        """单VC streamer bank需求 (C0/C1, num_vc=1)"""
        return self.a_banks() + self.b_banks(wpe)

    def __repr__(self):
        return f"[{self.meshRow}x{self.tileSize}x{self.meshCol}]"


def generate_shapes(mac_count: int, tile_size: int = 8) -> List[VersaCoreShape]:
    """生成所有合法的 VersaCore shape (meshRow, meshCol 均为2的幂)"""
    spatial = mac_count // tile_size
    shapes = []
    r = 1
    while r <= spatial:
        c = spatial // r
        if r * c == spatial:
            shapes.append(VersaCoreShape(r, tile_size, c))
        r *= 2
    return shapes


SHAPES_512 = generate_shapes(512, 8)
SHAPES_256 = generate_shapes(256, 8)


# ============================================================
#  Cluster 硬件配置
# ============================================================


@dataclass
class ClusterConfig:
    """
    单个 Cluster 硬件参数
    - tcdm: 64 banks × 8B/bank = 512B/cc 总带宽
    - xDMA: 64B/cc, 通过16个TCDM端口写入 (占16 banks)
    - iDMA: 同上, 但直连TCDM不经xDMA
    """

    cluster_id: int
    tcdm_size_bytes: int  # TCDM容量 (bytes)
    tcdm_num_banks: int = 64  # bank数量
    tcdm_bank_width: int = 8  # 每bank每周期带宽 (bytes)
    xdma_bw: int = 64  # xDMA带宽 (bytes/cc)
    xdma_tcdm_ports: int = 16  # xDMA占用TCDM端口数 (=bank数)
    mac_count: int = 512  # 总MAC数
    num_vc: int = 1  # VersaCore数量
    vc_mac_count: int = 0  # 每个VC的MAC数 (0=自动)
    elemwise_rate: int = 128  # SiLU/乘法加速器吞吐 (elements/cc)

    def __post_init__(self):
        if self.vc_mac_count == 0:
            self.vc_mac_count = self.mac_count // self.num_vc


# ============================================================
#  系统级hardwire配置
# ============================================================


@dataclass
class SystemConfig:
    """
    HeMAiA SoC 系统配置
    互联: sram_xDMA(64B/cc) + iDMA(64B/cc) 独立并行
    """

    clusters: List[ClusterConfig]
    sram_xdma_bw: int = 64  # SRAM端xDMA带宽
    idma_bw: int = 64  # iDMA带宽
    p2p_bw: int = 64  # cluster间xDMA P2P带宽
    spm_size_bytes: int = (
        1024 * 1024 * 1024
    )  # 1GB SRAM (容纳64×4.125MB=264MB routed expert权重)

    @staticmethod
    def default_4cluster() -> "SystemConfig":
        """默认4-cluster配置 (v21: 所有cluster统一2×256MAC dual-VC, 5MB TCDM)"""
        MB = 1024 * 1024
        return SystemConfig(
            clusters=[
                ClusterConfig(
                    cluster_id=cid,
                    tcdm_size_bytes=5 * MB,
                    mac_count=512,
                    num_vc=2,
                    vc_mac_count=256,
                )
                for cid in range(4)
            ]
        )

    @property
    def num_clusters(self) -> int:
        return len(self.clusters)


# ============================================================
#  DeepSeek V2 Lite MoE 层参数
# ============================================================


@dataclass
class MoELayerConfig:
    """
    DeepSeek V2 Lite 单层 MoE 参数

    shared expert: 1个 (由原始2个shared expert合并为1个FFN, intermediate=2816)
    routed expert: 64个, topK=2, 每个 intermediate=1408
    """

    hidden_size: int = 2048
    moe_intermediate_size: int = 1408  # 每个routed expert的intermediate
    shared_intermediate_size: int = 2816  # shared expert的intermediate (= 2×1408)
    n_routed_experts: int = 64
    topk: int = 2
    weight_dtype_bits: int = 4  # 默认INT4

    @property
    def wpe(self) -> float:
        """每个权重元素的字节数"""
        return self.weight_dtype_bits / 8

    @property
    def dtype_label(self) -> str:
        return f"INT{self.weight_dtype_bits}"

    # ---- Shared expert 权重大小 (bytes) ----
    # v21: shared intermediate=2816 在 N 方向切为两半, 每半1408 (= moe_intermediate_size)
    # 每个cluster驻留一半shared: gate_half + up_half + down_half_block
    @property
    def shared_half_intermediate(self) -> int:
        """shared expert在N方向切半的维度 (1408)"""
        return self.shared_intermediate_size // 2

    @property
    def shared_half_gate_weight(self) -> int:
        """C0或C1驻留的gate权重: K × (S/2)"""
        return int(self.hidden_size * self.shared_half_intermediate * self.wpe)

    @property
    def shared_half_up_weight(self) -> int:
        return int(self.hidden_size * self.shared_half_intermediate * self.wpe)

    @property
    def shared_half_down_weight(self) -> int:
        """单cluster驻留的down块: (S/2) × K (输入S/2, 输出K, 合并时相加)"""
        return int(self.shared_half_intermediate * self.hidden_size * self.wpe)

    # ---- 兼容别名 (旧代码引用) ----
    @property
    def shared_gate_weight(self) -> int:
        return int(self.hidden_size * self.shared_intermediate_size * self.wpe)

    @property
    def shared_up_weight(self) -> int:
        return int(self.hidden_size * self.shared_intermediate_size * self.wpe)

    @property
    def shared_down_weight(self) -> int:
        return int(self.shared_intermediate_size * self.hidden_size * self.wpe)

    @property
    def c0_resident_size(self) -> int:
        """C0驻留: 半个shared (gate+up+down全套) = 4.125MB"""
        return (
            self.shared_half_gate_weight
            + self.shared_half_up_weight
            + self.shared_half_down_weight
        )

    @property
    def c1_resident_size(self) -> int:
        """C1驻留: 另外半个shared = 4.125MB"""
        return self.c0_resident_size

    # ---- Routed expert 权重大小 ----
    @property
    def expert_gate_weight(self) -> int:
        return int(self.hidden_size * self.moe_intermediate_size * self.wpe)

    @property
    def expert_up_weight(self) -> int:
        return int(self.hidden_size * self.moe_intermediate_size * self.wpe)

    @property
    def expert_down_weight(self) -> int:
        return int(self.moe_intermediate_size * self.hidden_size * self.wpe)

    @property
    def expert_total_weight(self) -> int:
        return self.expert_gate_weight + self.expert_up_weight + self.expert_down_weight

    # ---- Router ----
    @property
    def router_weight_size(self) -> int:
        return int(self.hidden_size * self.n_routed_experts * self.wpe)

    # ---- 计算量 (MAC ops) ----
    def shared_total_macs(self, M: int) -> int:
        """shared expert总MAC运算量: gate+up+down"""
        S = self.shared_intermediate_size
        K = self.hidden_size
        return M * K * S * 2 + M * K * S + M * S * K  # gate+up: 2×(M×K×S), down: M×S×K

    def shared_ideal_cc(self, M: int, total_mac: int = 1024) -> int:
        """shared expert理想下界 = 总MAC / 总MAC吞吐"""
        return math.ceil(
            3 * M * self.hidden_size * self.shared_intermediate_size / total_mac
        )

    def summary(self) -> str:
        MB = 1024 * 1024
        return (
            f"=== DeepSeek V2 Lite MoE 层参数 ({self.dtype_label}) ===\n"
            f"  hidden_size:          {self.hidden_size}\n"
            f"  shared_intermediate:  {self.shared_intermediate_size}\n"
            f"  moe_intermediate:     {self.moe_intermediate_size}\n"
            f"  n_routed_experts:     {self.n_routed_experts}, topK={self.topk}\n"
            f"  C0驻留(up+half_down): {self.c0_resident_size/MB:.3f} MB\n"
            f"  C1驻留(gate+half_down): {self.c1_resident_size/MB:.3f} MB\n"
            f"  单routed expert:      {self.expert_total_weight/MB:.3f} MB\n"
        )
