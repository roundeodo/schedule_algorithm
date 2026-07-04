"""
=== HeMAiA SoC 硬件配置 + DeepSeek V2 Lite MoE 层参数 (v3) ===

硬件互联拓扑 (修正版):
  xDMA系统:
    - 每个 cluster 有一个 cluster_xDMA, 带宽 64 bytes/cc
    - SPM SRAM 有一个 sram_xDMA, 带宽 64 bytes/cc
    - 任何 xDMA 可以和其他任何 xDMA 做 P2P 互联
    - 但一个 xDMA 同一时刻只能和一个 xDMA 连接 (一对一)
    - 例: sram_xDMA↔C2_xDMA, 此时 sram_xDMA 不能同时连 C3_xDMA
    - cluster 间 P2P: C0_xDMA↔C1_xDMA, 各 64B/cc

  iDMA系统:
    - 系统有一个 iDMA, 带宽 64 bytes/cc
    - iDMA 直连 cluster 的 TCDM (不占用 cluster 的 xDMA 端口!)
    - 同一时刻只能连一个 cluster
    - iDMA 和 sram_xDMA 是独立的, 可并行工作

  SRAM → cluster 搬运路径:
    路径1: sram_xDMA ↔ cluster_xDMA (P2P, 占用双方 xDMA 端口)
    路径2: iDMA → cluster_TCDM (直连, 不占 cluster 的 xDMA)
    两路径可并行: 总 BW = 128 B/cc (连同一或不同 cluster)
    连同一 cluster: 128B/cc, 但 cluster_xDMA 被 sram_xDMA 占用
    连不同 cluster: 各 64B/cc, 且 cluster_xDMA 可能仍可做其他 P2P

  C2/C3 VersaCore:
    - 各有 2 个 256MAC VersaCore, 使用相同 shape
    - dual-VC B矩阵带宽需求 = 2 × tileSize × meshCol × bytes_per_elem
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
    VersaCore 阵列形状配置
    三个维度: meshRow × tileSize × meshCol
    - meshRow: 输出矩阵 M 方向的空间展开
    - tileSize: K 方向的内积累加深度
    - meshCol: 输出矩阵 N 方向的空间展开
    每个周期完成 meshRow×tileSize×meshCol 次 MAC 操作
    """

    meshRow: int
    tileSize: int
    meshCol: int

    @property
    def macs(self) -> int:
        return self.meshRow * self.tileSize * self.meshCol

    @property
    def a_bytes_per_cycle(self) -> int:
        """每个计算周期, streamer 需要从 TCDM 读取的 A 数据量 (int8)"""
        return self.meshRow * self.tileSize

    @property
    def b_bytes_per_cycle(self) -> int:
        """每个计算周期, streamer 需要从 TCDM 读取的 B 数据量 (int8)"""
        return self.tileSize * self.meshCol

    @property
    def a_banks(self) -> int:
        return math.ceil(self.a_bytes_per_cycle / 8)

    @property
    def b_banks(self) -> int:
        return math.ceil(self.b_bytes_per_cycle / 8)

    @property
    def streamer_banks_total(self) -> int:
        return self.a_banks + self.b_banks

    def __repr__(self):
        return f"[{self.meshRow}x{self.tileSize}x{self.meshCol}]"


def generate_shapes(mac_count: int, tile_size: int = 8) -> List[VersaCoreShape]:
    """
    给定 MAC 数量和固定的 tileSize, 生成所有合法的 VersaCore shape
    meshRow × tileSize × meshCol = mac_count
    meshRow, meshCol 必须是 2 的幂且 >= 1
    """
    spatial = mac_count // tile_size
    shapes = []
    r = 1
    while r <= spatial:
        c = spatial // r
        if r * c == spatial:
            shapes.append(VersaCoreShape(r, tile_size, c))
        r *= 2
    return shapes


# 512-MAC 默认 shape 列表
SHAPES_512MAC: List[VersaCoreShape] = generate_shapes(512, 8)


# ============================================================
#  Cluster 硬件配置
# ============================================================


@dataclass
class ClusterConfig:
    """单个 Cluster 的硬件参数"""

    cluster_id: int
    tcdm_size_bytes: int
    tcdm_num_banks: int = 64
    tcdm_bank_width: int = 8
    xdma_bw_bytes_per_cc: int = 64
    xdma_tcdm_ports: int = 16
    mac_count: int = 512
    num_vc: int = 1  # VersaCore 数量 (C2/C3 = 2)
    vc_mac_count: int = 0  # 每个 VC 的 MAC 数 (0 = 自动 mac_count/num_vc)
    serial_cd_width_bits: int = 1024
    swishglu_elements_per_cc: int = 128

    def __post_init__(self):
        if self.vc_mac_count == 0:
            self.vc_mac_count = self.mac_count // self.num_vc

    @property
    def tcdm_total_bw(self) -> int:
        return self.tcdm_num_banks * self.tcdm_bank_width


# ============================================================
#  系统级硬件配置
# ============================================================


@dataclass
class SystemConfig:
    """
    整个 SoC 系统配置

    互联约束:
      - sram_xdma_bw: SRAM端的xDMA, 同时只能和一个cluster_xDMA P2P连接
      - idma_bw: iDMA, 直连cluster TCDM (不占xDMA端口), 同时只连一个cluster
      - 两者独立, 可并行; 总BW = sram_xdma_bw + idma_bw = 128 B/cc
    """

    clusters: List[ClusterConfig]
    sram_xdma_bw_bytes_per_cc: int = 64  # SRAM xDMA 带宽
    idma_bw_bytes_per_cc: int = 64  # iDMA 带宽
    xdma_p2p_bw_bytes_per_cc: int = 64  # cluster间 xDMA P2P 带宽
    spm_size_bytes: int = 32 * 1024 * 1024

    @staticmethod
    def default_4cluster() -> "SystemConfig":
        return SystemConfig(
            clusters=[
                ClusterConfig(
                    cluster_id=0,
                    tcdm_size_bytes=6 * 1024 * 1024,
                    mac_count=512,
                    num_vc=1,
                    vc_mac_count=512,
                ),
                ClusterConfig(
                    cluster_id=1,
                    tcdm_size_bytes=6 * 1024 * 1024,
                    mac_count=512,
                    num_vc=1,
                    vc_mac_count=512,
                ),
                ClusterConfig(
                    cluster_id=2,
                    tcdm_size_bytes=5 * 1024 * 1024,
                    mac_count=512,
                    num_vc=2,
                    vc_mac_count=256,
                ),
                ClusterConfig(
                    cluster_id=3,
                    tcdm_size_bytes=5 * 1024 * 1024,
                    mac_count=512,
                    num_vc=2,
                    vc_mac_count=256,
                ),
            ]
        )

    @staticmethod
    def parametric(
        num_clusters: int,
        tcdm_sizes: List[int],
        mac_count: int = 512,
        num_banks: int = 64,
        xdma_bw: int = 64,
        idma_bw: int = 64,
        spm_size: int = 32 * 1024 * 1024,
    ) -> "SystemConfig":
        """参数化硬件配置, 用于扫参"""
        assert len(tcdm_sizes) == num_clusters
        clusters = []
        for i in range(num_clusters):
            clusters.append(
                ClusterConfig(
                    cluster_id=i,
                    tcdm_size_bytes=tcdm_sizes[i],
                    tcdm_num_banks=num_banks,
                    mac_count=mac_count,
                )
            )
        return SystemConfig(
            clusters=clusters,
            xdma_p2p_bw_bytes_per_cc=xdma_bw,
            idma_bw_bytes_per_cc=idma_bw,
            spm_size_bytes=spm_size,
        )

    @property
    def num_clusters(self) -> int:
        return len(self.clusters)

    def combined_load_bw(self) -> int:
        return self.sram_xdma_bw_bytes_per_cc + self.idma_bw_bytes_per_cc


# ============================================================
#  DeepSeek V2 Lite MoE 层参数
# ============================================================


@dataclass
class MoELayerConfig:
    """
    DeepSeek V2 Lite 单层 MoE 参数

    支持 INT4 和 INT8:
      - INT8: weight_dtype_bits = 8
      - INT4: weight_dtype_bits = 4
    """

    hidden_size: int = 2048
    moe_intermediate_size: int = 1408
    n_shared_experts: int = 2
    n_routed_experts: int = 64
    topk: int = 2
    weight_dtype_bits: int = 8  # 8 = INT8, 4 = INT4

    @property
    def weight_dtype_bytes(self) -> float:
        return self.weight_dtype_bits / 8

    @property
    def dtype_label(self) -> str:
        return f"INT{self.weight_dtype_bits}"

    @property
    def shared_intermediate(self) -> int:
        return self.moe_intermediate_size * self.n_shared_experts

    # ---- 权重矩阵大小 (字节) ----

    @property
    def router_weight_size(self) -> int:
        return int(self.hidden_size * self.n_routed_experts * self.weight_dtype_bytes)

    @property
    def shared_gate_weight_size(self) -> int:
        return int(
            self.hidden_size * self.shared_intermediate * self.weight_dtype_bytes
        )

    @property
    def shared_up_weight_size(self) -> int:
        return int(
            self.hidden_size * self.shared_intermediate * self.weight_dtype_bytes
        )

    @property
    def shared_down_weight_size(self) -> int:
        return int(
            self.shared_intermediate * self.hidden_size * self.weight_dtype_bytes
        )

    @property
    def expert_gate_weight_size(self) -> int:
        return int(
            self.hidden_size * self.moe_intermediate_size * self.weight_dtype_bytes
        )

    @property
    def expert_up_weight_size(self) -> int:
        return int(
            self.hidden_size * self.moe_intermediate_size * self.weight_dtype_bytes
        )

    @property
    def expert_down_weight_size(self) -> int:
        return int(
            self.moe_intermediate_size * self.hidden_size * self.weight_dtype_bytes
        )

    @property
    def expert_total_weight_size(self) -> int:
        return (
            self.expert_gate_weight_size
            + self.expert_up_weight_size
            + self.expert_down_weight_size
        )

    # ---- GEMM 维度 [M, K, N] ----

    def shared_gate_dims(self, M: int) -> Tuple[int, int, int]:
        return (M, self.hidden_size, self.shared_intermediate)

    def shared_up_dims(self, M: int) -> Tuple[int, int, int]:
        return (M, self.hidden_size, self.shared_intermediate)

    def shared_down_dims(self, M: int) -> Tuple[int, int, int]:
        return (M, self.shared_intermediate, self.hidden_size)

    def expert_gate_dims(self, M: int) -> Tuple[int, int, int]:
        return (M, self.hidden_size, self.moe_intermediate_size)

    def expert_up_dims(self, M: int) -> Tuple[int, int, int]:
        return (M, self.hidden_size, self.moe_intermediate_size)

    def expert_down_dims(self, M: int) -> Tuple[int, int, int]:
        return (M, self.moe_intermediate_size, self.hidden_size)

    # ---- Activation 大小 (INT8 激活, 不受 weight dtype 影响) ----

    def input_activation_size(self, M: int) -> int:
        return M * self.hidden_size

    def shared_intermediate_act_size(self, M: int) -> int:
        return M * self.shared_intermediate

    def expert_intermediate_act_size(self, M: int) -> int:
        return M * self.moe_intermediate_size

    def output_activation_size(self, M: int) -> int:
        return M * self.hidden_size

    def summary(self) -> str:
        return (
            f"\n=== DeepSeek V2 Lite MoE 层参数 ({self.dtype_label}) ===\n"
            f"  hidden_size:          {self.hidden_size}\n"
            f"  moe_intermediate:     {self.moe_intermediate_size}\n"
            f"  shared_intermediate:  {self.shared_intermediate}\n"
            f"  n_routed_experts:     {self.n_routed_experts}\n"
            f"  topk:                 {self.topk}\n"
            f"  权重数据类型:         {self.dtype_label} ({self.weight_dtype_bytes} B/element)\n"
            f"  共享 gate/up 权重:    {self.shared_gate_weight_size / 1024 / 1024:.2f} MB each\n"
            f"  共享 down 权重:       {self.shared_down_weight_size / 1024 / 1024:.2f} MB\n"
            f"  单 expert gate/up:    {self.expert_gate_weight_size / 1024 / 1024:.2f} MB each\n"
            f"  单 expert down:       {self.expert_down_weight_size / 1024 / 1024:.2f} MB\n"
            f"  单 expert 总权重:     {self.expert_total_weight_size / 1024 / 1024:.2f} MB\n"
        )
