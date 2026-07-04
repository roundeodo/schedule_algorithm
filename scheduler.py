#!/usr/bin/env python3
"""
Scheduler v20: 全策略缓存感知调度 + 事件驱动DMA解耦 + 跨层expert缓存
=====================================================================

版本演进:
  v16: Per-Tile模型 + 细粒度调度策略池(7种) + 动态调度器
  v17: 新增 unified_dynamic_scheduler(DMA预取+专家克隆+shape切换)
  v18: 新增 event_driven_scheduler(DMA/compute解耦追踪, 全局makespan优化)
  v19: 跨层expert缓存探索(cached_eids → cached_map, TCDM容量分析)
  v20: 全部9种策略统一支持缓存感知调度 + exit_eids跨层传递

硬件模型 (v16修正, 延续至v20):
  - Bank: 2×A_banks + 2×B_banks (两个VC各自独立读A和B, 无broadcast)
  - Per-tile: 逐tile追踪DMA/compute pipeline和bank冲突
  - 量化: W4A8 (权重INT4, 激活INT8)

调度策略池 (9种, v20全部支持cached_map):
  1. phase_based: 热冷配对 + expert拆分 + 驻留phase
  2. greedy_balanced: 贪心负载均衡 @64B/cc
  3. sequential_full: 串行全带宽 @128B/cc
  4. bw_steal: 带宽窃取 — 先完成的cluster抢空闲DMA
  5. adaptive_split: 自适应拆分 — 穷举拆分点最优化
  6. online_greedy: 在线贪心 — 每步评估所有可选动作
  7. cold_batch: 冷门批量 — hot并行, cold用128B快速消化
  8. unified_dynamic: 统一动态 — DMA预取 + 专家克隆 + shape切换
  9. event_driven: 事件驱动 — DMA/计算解耦 + 全局makespan优化

v20核心特性:
  - 缓存预处理函数 _preprocess_cached(): 统一将缓存命中的expert分离为resident任务
  - cached_map: Dict[int, int]: 跨层缓存映射 (eid → cid), 表示eid的权重驻留在cluster cid
  - exit_eids: Dict[int, int]: 每个cluster最后执行的expert (cid → eid), 作为下一层的缓存映射
  - 缓存收益: ntok=1省74.9%, ntok=2省49.8%, ntok≥4无收益(compute-bound)
  - 缓存容量: C2/C3各可驻留1个完整expert (4.125MB / 5MB TCDM)

动态调度器 (schedule入口):
  - 遍历全部9种策略, 统一传递cached_map
  - cost函数选最优方案
  - 自动计算exit_eids供下一层使用

核心约束 (INT4, dual-VC):
  Shape [4×8×8]: 双VC demand = 2×(4+4) = 16 banks + DMA
  Shape [2×8×16]: 双VC demand = 2×(2+8) = 20 banks + DMA
  Shape [1×8×32]: 双VC demand = 2×(1+16) = 34 banks + DMA
"""

import math
import itertools
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from config import (
    VersaCoreShape,
    generate_shapes,
    ClusterConfig,
    SystemConfig,
    MoELayerConfig,
    SHAPES_256,
    SHAPES_512,
)
from model import gemm_cycles, dma_cc, vc_real_utilization


# ============================================================
#  数据结构
# ============================================================


@dataclass
class ExpertTask:
    """单个routed expert (或expert的一部分) 的执行任务"""

    eid: int  # expert编号
    ntok: int  # 分配的token数
    cid: int  # 目标cluster (2 or 3)
    shape: VersaCoreShape  # VersaCore shape
    dma_mode: str  # "xdma" | "idma" | "both" | "none"(驻留)
    load_bw: int  # DMA带宽 (64 or 128 or 0)
    phase: int = 0  # 执行阶段 (用于拆分expert)
    resident: bool = False  # 权重是否已驻留
    estimated_cc: int = 0
    vc_util: float = 0.0
    rationale: str = ""


@dataclass
class SchedulePlan:
    """完整调度方案"""

    tasks: List[ExpertTask]
    strategy: str
    estimated_cc: int = 0
    c2_cc: int = 0
    c3_cc: int = 0
    sram_xdma_util: float = 0.0
    sram_idma_util: float = 0.0
    avg_vc_util: float = 0.0
    # [v20] 跨层缓存传递: 记录每个cluster最后执行的expert (cid → eid)
    # 该expert的权重在执行完毕后仍驻留在TCDM中, 可供下一MoE层复用
    # 使用方法: 上一层的exit_eids → 下一层的cached_map (若eid匹配则缓存命中)
    exit_eids: Dict[int, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"策略={self.strategy} 估计={self.estimated_cc:,}cc "
            f"C2={self.c2_cc:,} C3={self.c3_cc:,} "
            f"VC_util={self.avg_vc_util:.1%} "
            f"xDMA={self.sram_xdma_util:.1%} iDMA={self.sram_idma_util:.1%}"
        )


# ============================================================
#  基础估计函数
# ============================================================


def _pertile_streaming_cc(
    ntok: int,
    K: int,
    N: int,
    shape: VersaCoreShape,
    load_bw: int,
    wpe: float,
    dma_ports: int = 16,
    total_banks: int = 64,
) -> int:
    """
    Per-tile细粒度计算单阶段streaming GEMM周期.
    与model.py中_streaming_gemm_dual_vc_pertile逻辑一致.
    """
    R, T, C = shape.meshRow, shape.tileSize, shape.meshCol
    Mt = math.ceil(ntok / R)
    Kt = math.ceil(K / T)
    Nt = math.ceil(N / C)
    drain = math.ceil(R * C * 32 / 1024)
    compute_per_out_tile = Kt + drain

    # 每K-tile DMA (双VC各需T×C×wpe, 总2×T×C×wpe, 64B对齐)
    b_per_ktile = math.ceil(2 * T * C * wpe / 64) * 64
    dma_per_ktile = dma_cc(b_per_ktile, load_bw) if load_bw > 0 else 0

    # bank冲突: 2×A_banks + 2×B_banks + DMA_ports
    a_banks = math.ceil(shape.a_bytes() / 8)
    b_banks = math.ceil(shape.b_bytes(wpe) / 8)
    streamer = 2 * a_banks + 2 * b_banks
    actual_dma_ports = dma_ports if load_bw > 0 else 0
    bank_total = streamer + actual_dma_ports
    bank_s = bank_total / total_banks if bank_total > total_banks else 1.0

    compute_with_bank = math.ceil(compute_per_out_tile * bank_s)
    pipeline_rate = max(dma_per_ktile, math.ceil(1 * bank_s))

    # tile0: 等第一K-tile DMA → pipeline剩余K-tile → drain
    tile0_time = dma_per_ktile + (Kt - 1) * pipeline_rate + drain

    # per output tile: max(dma, compute)
    dma_per_outtile = Kt * dma_per_ktile
    tile_pipeline_rate = max(dma_per_outtile, compute_with_bank)

    n_out_tiles = Mt * Nt
    if n_out_tiles <= 1:
        total = tile0_time + 5
    else:
        total = tile0_time + (n_out_tiles - 1) * tile_pipeline_rate + 5

    # 总DMA时间
    weight_bytes = int(K * N * wpe) * 2  # 双VC各自的B
    total_dma = dma_cc(weight_bytes, load_bw) if load_bw > 0 else 0
    total = max(total, total_dma)
    return total


def estimate_expert_cc(
    ntok: int,
    K: int,
    N: int,
    shape: VersaCoreShape,
    load_bw: int,
    wpe: float,
    num_vc: int,
    elemwise_rate: int,
    resident: bool = False,
) -> Tuple[int, float]:
    """
    估计一个routed expert的执行周期 (C2/C3 dual-VC, per-tile模型).
    返回: (total_cc, real_vc_utilization)

    Per-tile模型:
      - 将GEMM拆分为Mt×Nt×Kt个tile, 逐tile追踪DMA/compute pipeline
      - Bank冲突: 2×A_banks + 2×B_banks + DMA_ports(当DMA活跃时)
      - tile0: 必须等第一个K-tile DMA完成
      - 后续tile: DMA/compute pipeline, 受限于max(dma_per_tile, compute_per_tile)

    真实VC利用率 = M维利用率 × (compute_time / total_time)
    """
    if ntok <= 0:
        return 0, 0.0
    K_half = K // 2

    # 纯计算 (无bank冲突, 无DMA等待)
    single_gu = gemm_cycles(ntok, K, N, shape)
    single_dn = gemm_cycles(ntok, N, K_half, shape)
    swiglu = math.ceil(ntok * N / elemwise_rate)
    pure_compute = single_gu + swiglu + single_dn

    if resident:
        # 驻留: bank冲突(无DMA port)
        a_banks = math.ceil(shape.a_bytes() / 8)
        b_banks = math.ceil(shape.b_bytes(wpe) / 8)
        streamer = 2 * a_banks + 2 * b_banks
        bank_s = streamer / 64 if streamer > 64 else 1.0
        total = math.ceil(pure_compute * bank_s)
        real_util = vc_real_utilization(ntok, shape, pure_compute, total)
        return total, real_util

    # Streaming: per-tile gate+up + swiglu + per-tile down
    gu_total = _pertile_streaming_cc(ntok, K, N, shape, load_bw, wpe)

    # down: 权重=N×K×wpe, 但down阶段DMA需等gate+up DMA结束
    # down per-tile计算
    dn_total = _pertile_streaming_cc(ntok, N, K_half, shape, load_bw, wpe)

    # gate+up DMA结束时刻 (整体)
    gu_weight = int(2 * K * N * wpe)
    gu_dma = dma_cc(gu_weight, load_bw)

    # down DMA不能在gate+up DMA结束前开始
    dn_start = max(gu_total + swiglu, gu_dma)
    total = dn_start + dn_total

    real_util = vc_real_utilization(ntok, shape, pure_compute, total)
    return total, real_util


def best_shape_for(
    ntok: int,
    K: int,
    N: int,
    load_bw: int,
    wpe: float,
    num_vc: int,
    elemwise_rate: int,
    resident: bool = False,
) -> Tuple[VersaCoreShape, int, float]:
    """
    在所有256MAC shapes中选最优shape.
    Dual-VC: VC利用率按全M计算(每个VC各处理全部M行, gate/up分工)
    返回: (shape, estimated_cc, vc_utilization)

    [v23] 不再预排除 bw_demand>load_bw 的 shape —
    estimate_expert_cc 内部 pipeline 已正确建模 DMA-bound 的 max(compute, dma);
    DMA-bound 时所有 shape cc 几乎相等, 按 util tie-break 自然选 M 维最优.
    Compute-bound 时大 shape 自己就会因 pipeline_rate 被 bank 拖慢而落选.
    """
    shapes = generate_shapes(256, 8)
    best_s, best_cc, best_util = shapes[0], float("inf"), 0.0
    for s in shapes:
        cc, real_util = estimate_expert_cc(
            ntok, K, N, s, load_bw, wpe, num_vc, elemwise_rate, resident
        )
        if cc < best_cc or (cc == best_cc and real_util > best_util):
            best_s, best_cc, best_util = s, cc, real_util
    return best_s, best_cc, best_util


# ============================================================
#  Token分布生成器 (全面覆盖各种topK结果)
# ============================================================


def _distribute_tokens(total_slots: int, n_active: int) -> Dict[int, int]:
    """均匀分配total_slots到n_active个expert"""
    if n_active <= 0:
        return {}
    base = total_slots // n_active
    rem = total_slots % n_active
    return {i: base + (1 if i < rem else 0) for i in range(n_active)}


def generate_all_distributions(
    M: int, topK: int, n_experts: int
) -> List[Dict[int, int]]:
    """
    生成全面的token分布列表.

    对于小M (1,4,8,16): 生成所有可能的分布组合 (或代表性组合)
    对于大M (64,128): 采样100+种代表性分布

    分布模式:
      - uniform: 所有active expert均等
      - hot_heavy: 少量expert占多数token
      - cold_heavy: 大量expert各占1-2个token
      - mixed: 部分热门+部分冷门
      - extreme_hot: 1-2个expert占绝大多数
      - all_cold: 每个expert只有1个token
    """
    total = M * topK
    distributions = []
    seen = set()

    def _add(dist: Dict[int, int]):
        """去重后添加"""
        # 按token数降序排列作为签名
        sig = tuple(sorted(dist.values(), reverse=True))
        if sig not in seen:
            seen.add(sig)
            distributions.append(dict(dist))

    # === 1. Uniform: 所有active expert同等 ===
    for n_active in range(1, min(total, n_experts) + 1):
        if total >= n_active:
            _add(_distribute_tokens(total, n_active))

    # === 2. Hot-heavy: top-k个expert各占h个token, 剩余均分 ===
    for n_hot in range(1, min(8, total) + 1):
        for hot_frac in [0.5, 0.6, 0.7, 0.8, 0.9]:
            hot_total = max(n_hot, int(total * hot_frac))
            if hot_total > total:
                continue
            cold_total = total - hot_total
            per_hot = hot_total // n_hot
            if per_hot <= 0:
                continue
            n_cold = min(cold_total, n_experts - n_hot)
            if n_cold <= 0 and cold_total > 0:
                continue
            dist = {}
            for i in range(n_hot):
                dist[i] = per_hot + (1 if i < hot_total % n_hot else 0)
            if n_cold > 0 and cold_total > 0:
                per_cold = cold_total // n_cold
                for i in range(n_hot, n_hot + n_cold):
                    dist[i] = per_cold + (1 if (i - n_hot) < cold_total % n_cold else 0)
            if sum(dist.values()) == total and all(v > 0 for v in dist.values()):
                _add(dist)

    # === 3. Extreme: 1个expert占几乎所有token ===
    for n_big in [1, 2]:
        for rest_each in [1, 2]:
            big_tok = total - rest_each * min(
                total // rest_each - n_big, n_experts - n_big
            )
            if big_tok > 0:
                n_rest = min((total - big_tok * n_big) // rest_each, n_experts - n_big)
                if n_rest >= 0 and big_tok * n_big + n_rest * rest_each == total:
                    dist = {}
                    for i in range(n_big):
                        dist[i] = big_tok
                    for i in range(n_big, n_big + n_rest):
                        dist[i] = rest_each
                    if sum(dist.values()) == total:
                        _add(dist)

    # === 4. All-cold: 每个expert 1 token ===
    if total <= n_experts:
        _add({i: 1 for i in range(total)})

    # === 5. 特定模式: 指定token分配 ===
    patterns = [
        # (热门token数列表, 冷门expert数, 冷门每expert token数)
        ([total], 0, 0),  # 1个expert全部
        ([total // 2, total - total // 2], 0, 0),  # 2个expert平分
    ]
    if total >= 4:
        patterns.append(([total - 2, 1, 1], 0, 0))  # 1热+2冷(1tok)
    if total >= 6:
        patterns.append(([total - 4, 2, 1, 1], 0, 0))
    if total >= 8:
        hot = total // 2
        patterns.append(([hot, total - hot - 2, 1, 1], 0, 0))
    if total >= 10:
        patterns.append(([total - 6, 2, 2, 1, 1], 0, 0))

    for pat in patterns:
        hot_list = pat[0]
        if all(h > 0 for h in hot_list) and sum(hot_list) == total:
            dist = {i: h for i, h in enumerate(hot_list)}
            if len(dist) <= n_experts:
                _add(dist)

    # === 6. 对于大M, 补充随机采样 ===
    if total > 16:
        import random

        rng = random.Random(42)
        target_count = max(100, 200) - len(distributions)
        for _ in range(target_count * 3):
            if len(distributions) >= 200:
                break
            # 随机选active expert数
            n_active = rng.randint(2, min(total, n_experts))
            # 随机分配token (Dirichlet-like)
            weights = [rng.random() ** 0.5 for _ in range(n_active)]
            s = sum(weights)
            raw = [max(1, int(w / s * total)) for w in weights]
            # 修正总和
            diff = total - sum(raw)
            if diff > 0:
                for i in range(diff):
                    raw[i % n_active] += 1
            elif diff < 0:
                for i in range(abs(diff)):
                    idx = raw.index(max(raw))
                    if raw[idx] > 1:
                        raw[idx] -= 1
            if sum(raw) == total and all(r > 0 for r in raw):
                dist = {i: r for i, r in enumerate(raw)}
                _add(dist)

    return distributions


# ============================================================
#  [v20] 缓存预处理辅助函数
#  所有9种调度策略在入口处统一调用此函数,
#  将缓存命中的expert从待调度列表中分离, 生成resident任务
# ============================================================


def _preprocess_cached(
    experts: List[Tuple[int, int]],
    cached_map: Dict[int, int],
    K: int,
    N: int,
    wpe: float,
    num_vc: int,
    elem_rate: int,
) -> Tuple[List[ExpertTask], List[Tuple[int, int]], int, int]:
    """
    [v20] 缓存预处理: 分离缓存命中的expert, 生成resident任务.

    核心逻辑:
      1. 遍历所有expert, 检查其eid是否在cached_map中
      2. 命中: 以resident模式执行(dma_mode='none', load_bw=0), 权重已在TCDM中
         - 无需DMA传输, 释放DMA通道给其他expert使用
         - 选用resident最优shape (仅考虑bank冲突, 无DMA端口冲突)
         - 分配到缓存所在的cluster (cid = cached_map[eid])
      3. 未命中: 加入uncached列表, 由后续策略正常调度

    缓存收益分析:
      - ntok=1: DMA省67,584cc中的50,614cc = 节省74.9%
      - ntok=2: DMA省67,584cc中的33,642cc = 节省49.8%
      - ntok≥4: 计算时间≥DMA时间, 节省≈0% (compute-bound场景)

    返回: (cached_tasks, uncached_experts, c2_cached_cc, c3_cached_cc)
    """
    # 无缓存映射时, 所有expert均为未命中, 直接返回
    if not cached_map:
        return [], list(experts), 0, 0

    cached_tasks = []  # 缓存命中的expert任务列表
    uncached = []  # 缓存未命中的expert列表 (待后续策略调度)
    c2_cc, c3_cc = 0, 0  # C2/C3上缓存expert的计算时间累计

    for eid, ntok in experts:
        if eid in cached_map:
            # 缓存命中: 权重已驻留在目标cluster的TCDM中
            cid = cached_map[eid]  # 缓存所在的cluster编号

            # [v20优化] 自适应缓存旁路: 比较resident vs streaming成本
            # 当ntok≥4时, expert为compute-bound, resident几乎无DMA节省
            # 但强制绑定到固定cluster会降低调度灵活性, 反而可能伤害性能
            # 因此: 仅当resident时间 < streaming时间的95%时才使用缓存
            s_res, cc_res, u_res = best_shape_for(
                ntok, K, N, 0, wpe, num_vc, elem_rate, resident=True
            )
            s_str, cc_str, u_str = best_shape_for(
                ntok, K, N, 64, wpe, num_vc, elem_rate, resident=False
            )
            # 缓存收益比: resident时间 / streaming时间
            # 若 > 0.95 (节省不到5%), 则跳过缓存, 让策略自由分配cluster
            cache_benefit_ratio = cc_res / cc_str if cc_str > 0 else 1.0
            if cache_benefit_ratio > 0.95:
                # 缓存收益不足: 视为未命中, 交由策略自由调度
                uncached.append((eid, ntok))
                continue

            cached_tasks.append(
                ExpertTask(
                    eid=eid,
                    ntok=ntok,
                    cid=cid,
                    shape=s_res,
                    dma_mode="none",  # 无需DMA传输
                    load_bw=0,  # DMA带宽为0
                    resident=True,  # 标记为驻留模式
                    estimated_cc=cc_res,
                    vc_util=u_res,
                    rationale=f"缓存命中 resident {ntok}tok @C{cid} (省{(1-cache_benefit_ratio)*100:.0f}%)",
                )
            )
            # 累计各cluster的缓存计算时间, 用于初始化时间线
            if cid == 2:
                c2_cc += cc_res
            else:
                c3_cc += cc_res
        else:
            # 缓存未命中: 加入待调度列表
            uncached.append((eid, ntok))

    return cached_tasks, uncached, c2_cc, c3_cc


# ============================================================
#  核心调度逻辑: 细粒度phase-based调度
# ============================================================


def _expert_phases(
    experts_sorted: List[Tuple[int, int]],
    sys: SystemConfig,
    moe: MoELayerConfig,
    cached_map: Dict[int, int] = None,
) -> SchedulePlan:
    """
    Phase-based调度:

    Phase 1 (并行流式):
      - 选一对expert (热门+冷门), C2用xdma(64), C3用idma(64)并行
      - 热门expert用低B-demand shape (如4×8×8, B=32B/cc), 满足64B/cc
      - 冷门expert也用64B/cc匹配的shape

    Phase 2 (驻留+全BW):
      - 热门expert完成流式后权重驻留, 切换到计算最优shape继续算剩余token
      - 空闲cluster独享128B/cc, 用更高B-demand shape处理下一个expert

    Phase 3 (清理):
      - 处理剩余的冷门expert(1-2 token), 可能需要1×8×32 shape
    """
    K = moe.hidden_size
    N = moe.moe_intermediate_size
    wpe = moe.wpe
    shapes = generate_shapes(256, 8)
    num_vc = 2
    elem_rate = sys.clusters[2].elemwise_rate

    # 按token数降序排列
    experts = [(eid, ntok) for eid, ntok in experts_sorted if ntok > 0]
    if not experts:
        return SchedulePlan(tasks=[], strategy="phase_based")

    # [v20] 缓存预处理: 分离已缓存的expert, 缓存命中者直接resident执行
    cached_tasks, uncached, c2_cached, c3_cached = _preprocess_cached(
        experts, cached_map, K, N, wpe, num_vc, elem_rate
    )
    # [v20] 若所有expert均缓存命中, 无需DMA调度, 直接返回resident方案
    if not uncached and cached_tasks:
        total_c = max(c2_cached, c3_cached)
        avg_u = sum(t.vc_util * t.estimated_cc for t in cached_tasks) / max(
            1, sum(t.estimated_cc for t in cached_tasks)
        )
        return SchedulePlan(
            tasks=cached_tasks,
            strategy="phase_based",
            estimated_cc=total_c,
            c2_cc=c2_cached,
            c3_cc=c3_cached,
            avg_vc_util=avg_u,
        )

    # [v20] 初始化: 缓存expert的计算时间作为时间线起点, DMA从0开始(缓存expert不占DMA)
    tasks = list(cached_tasks)
    c2_total, c3_total = c2_cached, c3_cached
    xdma_busy, idma_busy = 0, 0  # DMA busy时间

    # 时间线追踪 (缓存expert不占用DMA, DMA通道从t=0可用)
    c2_free, c3_free = c2_cached, c3_cached
    sram_xdma_free, sram_idma_free = 0, 0

    remaining = list(uncached)  # [(eid, ntok)] 仅包含未缓存的expert

    while remaining:
        # 按token数降序排
        remaining.sort(key=lambda x: -x[1])

        # === 尝试并行: C2和C3同时处理两个expert ===
        if len(remaining) >= 2 and c2_free <= max(sram_xdma_free, sram_idma_free) + 100:
            e1_eid, e1_ntok = remaining[0]
            e2_eid, e2_ntok = remaining[1]

            # e1→C2(xDMA, 64B/cc), e2→C3(iDMA, 64B/cc)
            s1, cc1, u1 = best_shape_for(e1_ntok, K, N, 64, wpe, num_vc, elem_rate)
            s2, cc2, u2 = best_shape_for(e2_ntok, K, N, 64, wpe, num_vc, elem_rate)

            # 也试试交换
            s1b, cc1b, u1b = best_shape_for(e1_ntok, K, N, 64, wpe, num_vc, elem_rate)
            s2b, cc2b, u2b = best_shape_for(e2_ntok, K, N, 64, wpe, num_vc, elem_rate)

            # 并行makespan = max(cc1, cc2)
            parallel_cc = max(cc1, cc2)

            # 也尝试: 拆分热门expert
            # 如果e1很大, 可以拆成使得两个cluster在phase1结束时间相近
            best_split = None
            if e1_ntok > e2_ntok and e1_ntok >= 4:
                for split_tok in range(max(1, e2_ntok), e1_ntok):
                    # C2算split_tok个token, C3算e2_ntok个token
                    s_c2, cc_c2, u_c2 = best_shape_for(
                        split_tok, K, N, 64, wpe, num_vc, elem_rate
                    )
                    s_c3, cc_c3, u_c3 = best_shape_for(
                        e2_ntok, K, N, 64, wpe, num_vc, elem_rate
                    )
                    split_parallel = max(cc_c2, cc_c3)
                    # 剩余token在C2上驻留计算
                    remain_tok = e1_ntok - split_tok
                    s_res, cc_res, u_res = best_shape_for(
                        remain_tok, K, N, 0, wpe, num_vc, elem_rate, resident=True
                    )
                    total_with_split = split_parallel + cc_res

                    if best_split is None or split_parallel < best_split[0]:
                        best_split = (
                            split_parallel,
                            split_tok,
                            remain_tok,
                            s_c2,
                            s_c3,
                            s_res,
                            cc_c2,
                            cc_c3,
                            cc_res,
                            u_c2,
                            u_c3,
                            u_res,
                        )
                    # 尝试找到两个cluster时间最接近的split点
                    if abs(cc_c2 - cc_c3) < 1000:
                        break

            # 决定是否使用split
            use_split = False
            if best_split and e1_ntok > e2_ntok * 2:
                sp = best_split
                # split的total = max(phase1) + phase2_resident
                split_total_c2 = sp[6] + sp[8]  # C2: stream + resident
                # 对比不split的C2时间
                if split_total_c2 < cc1:
                    use_split = True

            if use_split and best_split:
                sp = best_split
                (
                    _,
                    split_tok,
                    remain_tok,
                    s_c2,
                    s_c3,
                    s_res,
                    cc_c2,
                    cc_c3,
                    cc_res,
                    u_c2,
                    u_c3,
                    u_res,
                ) = sp

                # Phase 1: C2 stream split_tok, C3 stream e2_ntok
                start1 = max(c2_free, sram_xdma_free)
                start2 = max(c3_free, sram_idma_free)

                tasks.append(
                    ExpertTask(
                        eid=e1_eid,
                        ntok=split_tok,
                        cid=2,
                        shape=s_c2,
                        dma_mode="xdma",
                        load_bw=64,
                        phase=1,
                        estimated_cc=cc_c2,
                        vc_util=u_c2,
                        rationale=f"热门拆分{e1_ntok}={split_tok}+{remain_tok}, phase1 stream@64B/cc",
                    )
                )
                tasks.append(
                    ExpertTask(
                        eid=e2_eid,
                        ntok=e2_ntok,
                        cid=3,
                        shape=s_c3,
                        dma_mode="idma",
                        load_bw=64,
                        phase=1,
                        estimated_cc=cc_c3,
                        vc_util=u_c3,
                        rationale=f"配对expert, stream@64B/cc",
                    )
                )

                phase1_end = max(start1 + cc_c2, start2 + cc_c3)
                xdma_busy += cc_c2
                idma_busy += cc_c3

                # Phase 2: C2 resident计算剩余token
                tasks.append(
                    ExpertTask(
                        eid=e1_eid,
                        ntok=remain_tok,
                        cid=2,
                        shape=s_res,
                        dma_mode="none",
                        load_bw=0,
                        phase=2,
                        resident=True,
                        estimated_cc=cc_res,
                        vc_util=u_res,
                        rationale=f"热门驻留计算剩余{remain_tok}tok",
                    )
                )

                c2_free = start1 + cc_c2 + cc_res
                c3_free = start2 + cc_c3
                sram_xdma_free = start1 + cc_c2
                sram_idma_free = start2 + cc_c3
                c2_total += cc_c2 + cc_res
                c3_total += cc_c3

                remaining.pop(0)  # remove e1
                remaining.pop(0)  # remove e2 (was at index 1, now 0)

            else:
                # 不拆分, 直接并行
                start1 = max(c2_free, sram_xdma_free)
                start2 = max(c3_free, sram_idma_free)

                tasks.append(
                    ExpertTask(
                        eid=e1_eid,
                        ntok=e1_ntok,
                        cid=2,
                        shape=s1,
                        dma_mode="xdma",
                        load_bw=64,
                        phase=1,
                        estimated_cc=cc1,
                        vc_util=u1,
                        rationale=f"并行流式@64B/cc, {e1_ntok}tok",
                    )
                )
                tasks.append(
                    ExpertTask(
                        eid=e2_eid,
                        ntok=e2_ntok,
                        cid=3,
                        shape=s2,
                        dma_mode="idma",
                        load_bw=64,
                        phase=1,
                        estimated_cc=cc2,
                        vc_util=u2,
                        rationale=f"并行流式@64B/cc, {e2_ntok}tok",
                    )
                )

                c2_free = start1 + cc1
                c3_free = start2 + cc2
                sram_xdma_free = start1 + cc1
                sram_idma_free = start2 + cc2
                xdma_busy += cc1
                idma_busy += cc2
                c2_total += cc1
                c3_total += cc2

                remaining.pop(0)
                remaining.pop(0)

        elif len(remaining) >= 1:
            # === 单expert: 独享128B/cc ===
            eid, ntok = remaining[0]

            # 选哪个cluster空闲
            if c2_free <= c3_free:
                cid = 2
                start = max(c2_free, sram_xdma_free, sram_idma_free)
            else:
                cid = 3
                start = max(c3_free, sram_xdma_free, sram_idma_free)

            s, cc, u = best_shape_for(ntok, K, N, 128, wpe, num_vc, elem_rate)

            tasks.append(
                ExpertTask(
                    eid=eid,
                    ntok=ntok,
                    cid=cid,
                    shape=s,
                    dma_mode="both",
                    load_bw=128,
                    phase=1,
                    estimated_cc=cc,
                    vc_util=u,
                    rationale=f"独享128B/cc, {ntok}tok",
                )
            )

            if cid == 2:
                c2_free = start + cc
                c2_total += cc
            else:
                c3_free = start + cc
                c3_total += cc
            sram_xdma_free = start + cc
            sram_idma_free = start + cc
            xdma_busy += cc
            idma_busy += cc

            remaining.pop(0)
        else:
            break

    makespan = max(c2_total, c3_total)
    total_time = max(c2_free, c3_free)

    # 计算利用率
    avg_util = sum(t.vc_util * t.estimated_cc for t in tasks) / max(
        1, sum(t.estimated_cc for t in tasks)
    )
    xdma_util = xdma_busy / total_time if total_time > 0 else 0
    idma_util = idma_busy / total_time if total_time > 0 else 0

    return SchedulePlan(
        tasks=tasks,
        strategy="phase_based",
        estimated_cc=total_time,
        c2_cc=c2_total,
        c3_cc=c3_total,
        sram_xdma_util=xdma_util,
        sram_idma_util=idma_util,
        avg_vc_util=avg_util,
    )


def _schedule_greedy_balanced(
    experts_sorted: List[Tuple[int, int]],
    sys: SystemConfig,
    moe: MoELayerConfig,
    cached_map: Dict[int, int] = None,
) -> SchedulePlan:
    """
    贪心均衡调度:
    - 按token数降序, 贪心分配到负载最小的cluster
    - 每个cluster用自己的DMA通道(C2=xdma, C3=idma)
    - 当只剩一个cluster有任务时, 切换到128B/cc
    """
    K = moe.hidden_size
    N = moe.moe_intermediate_size
    wpe = moe.wpe
    num_vc = 2
    elem_rate = sys.clusters[2].elemwise_rate

    experts = [(eid, ntok) for eid, ntok in experts_sorted if ntok > 0]

    # [v20] 缓存预处理: 分离缓存命中expert, 未命中者继续贪心分配
    cached_tasks, uncached, c2_cached, c3_cached = _preprocess_cached(
        experts, cached_map, K, N, wpe, num_vc, elem_rate
    )
    # [v20] 全部命中时直接返回
    if not uncached and cached_tasks:
        total_c = max(c2_cached, c3_cached)
        avg_u = sum(t.vc_util * t.estimated_cc for t in cached_tasks) / max(
            1, sum(t.estimated_cc for t in cached_tasks)
        )
        return SchedulePlan(
            tasks=cached_tasks,
            strategy="greedy_balanced",
            estimated_cc=total_c,
            c2_cc=c2_cached,
            c3_cc=c3_cached,
            avg_vc_util=avg_u,
        )

    # [v20] 缓存expert的计算时间作为cluster初始负载
    tasks = list(cached_tasks)
    c2_load, c3_load = c2_cached, c3_cached
    xdma_busy, idma_busy = 0, 0

    for eid, ntok in uncached:
        if c2_load <= c3_load:
            cid, dma, bw = 2, "xdma", 64
        else:
            cid, dma, bw = 3, "idma", 64

        s, cc, u = best_shape_for(ntok, K, N, bw, wpe, num_vc, elem_rate)
        tasks.append(
            ExpertTask(
                eid=eid,
                ntok=ntok,
                cid=cid,
                shape=s,
                dma_mode=dma,
                load_bw=bw,
                estimated_cc=cc,
                vc_util=u,
                rationale=f"greedy @{bw}B/cc",
            )
        )

        if cid == 2:
            c2_load += cc
            xdma_busy += cc
        else:
            c3_load += cc
            idma_busy += cc

    makespan = max(c2_load, c3_load)
    total_time = makespan
    avg_util = sum(t.vc_util * t.estimated_cc for t in tasks) / max(
        1, sum(t.estimated_cc for t in tasks)
    )

    return SchedulePlan(
        tasks=tasks,
        strategy="greedy_balanced",
        estimated_cc=makespan,
        c2_cc=c2_load,
        c3_cc=c3_load,
        sram_xdma_util=xdma_busy / total_time if total_time > 0 else 0,
        sram_idma_util=idma_busy / total_time if total_time > 0 else 0,
        avg_vc_util=avg_util,
    )


def _schedule_sequential_full(
    experts_sorted: List[Tuple[int, int]],
    sys: SystemConfig,
    moe: MoELayerConfig,
    cached_map: Dict[int, int] = None,
) -> SchedulePlan:
    """
    串行全带宽: 一次只有一个cluster活动, 独享128B/cc.
    适合expert数很少但每个token数很大的场景.
    """
    K = moe.hidden_size
    N = moe.moe_intermediate_size
    wpe = moe.wpe
    num_vc = 2
    elem_rate = sys.clusters[2].elemwise_rate

    experts = [(eid, ntok) for eid, ntok in experts_sorted if ntok > 0]

    # [v20] 缓存预处理: 缓存命中expert以resident模式执行, 不占用DMA链
    cached_tasks, uncached, c2_cached, c3_cached = _preprocess_cached(
        experts, cached_map, K, N, wpe, num_vc, elem_rate
    )
    # [v20] 全部命中时直接返回
    if not uncached and cached_tasks:
        total_c = max(c2_cached, c3_cached)
        avg_u = sum(t.vc_util * t.estimated_cc for t in cached_tasks) / max(
            1, sum(t.estimated_cc for t in cached_tasks)
        )
        return SchedulePlan(
            tasks=cached_tasks,
            strategy="sequential_full",
            estimated_cc=total_c,
            c2_cc=c2_cached,
            c3_cc=c3_cached,
            avg_vc_util=avg_u,
        )

    # [v20] 缓存expert可并行于串行DMA链, 因此c_free从cached_cc开始, DMA从0开始
    tasks = list(cached_tasks)
    c2_load, c3_load = c2_cached, c3_cached
    c2_free, c3_free = c2_cached, c3_cached
    serial_end = 0  # 追踪串行DMA链结束时间 (缓存expert不影响DMA链)
    toggle = True

    for eid, ntok in uncached:
        cid = 2 if toggle else 3
        toggle = not toggle
        start = max(c2_free if cid == 2 else c3_free, serial_end)
        s, cc, u = best_shape_for(ntok, K, N, 128, wpe, num_vc, elem_rate)
        tasks.append(
            ExpertTask(
                eid=eid,
                ntok=ntok,
                cid=cid,
                shape=s,
                dma_mode="both",
                load_bw=128,
                estimated_cc=cc,
                vc_util=u,
                rationale=f"sequential full @128B/cc",
            )
        )
        end = start + cc
        if cid == 2:
            c2_free = end
            c2_load += cc
        else:
            c3_free = end
            c3_load += cc
        serial_end = end

    # 总时间: 缓存expert并行 + 串行DMA链
    total = max(c2_free, c3_free)
    avg_util = sum(t.vc_util * t.estimated_cc for t in tasks) / max(
        1, sum(t.estimated_cc for t in tasks)
    )

    return SchedulePlan(
        tasks=tasks,
        strategy="sequential_full",
        estimated_cc=total,
        c2_cc=c2_load,
        c3_cc=c3_load,
        sram_xdma_util=1.0 if total > 0 else 0,
        sram_idma_util=1.0 if total > 0 else 0,
        avg_vc_util=avg_util,
    )


# ============================================================
#  策略4: 带宽窃取 (BW Steal)
# ============================================================


def _schedule_bw_steal(
    experts_sorted: List[Tuple[int, int]],
    sys: SystemConfig,
    moe: MoELayerConfig,
    cached_map: Dict[int, int] = None,
) -> SchedulePlan:
    """
    带宽窃取调度:
    - 两个cluster并行@64B/cc处理一对expert
    - 先完成的cluster释放其DMA通道
    - 后续expert可以窃取空闲DMA通道, 获得128B/cc
    - 特别适合: 一热一冷的组合 (冷门先完成, 热门接管全部带宽)
    """
    K = moe.hidden_size
    N = moe.moe_intermediate_size
    wpe = moe.wpe
    num_vc = 2
    elem_rate = sys.clusters[2].elemwise_rate

    experts = [(eid, ntok) for eid, ntok in experts_sorted if ntok > 0]
    if not experts:
        return SchedulePlan(tasks=[], strategy="bw_steal")

    # [v20] 缓存预处理: 缓存命中expert释放DMA通道, 有利于后续带宽窃取
    cached_tasks, uncached, c2_cached, c3_cached = _preprocess_cached(
        experts, cached_map, K, N, wpe, num_vc, elem_rate
    )
    # [v20] 全部命中时直接返回
    if not uncached and cached_tasks:
        total_c = max(c2_cached, c3_cached)
        avg_u = sum(t.vc_util * t.estimated_cc for t in cached_tasks) / max(
            1, sum(t.estimated_cc for t in cached_tasks)
        )
        return SchedulePlan(
            tasks=cached_tasks,
            strategy="bw_steal",
            estimated_cc=total_c,
            c2_cc=c2_cached,
            c3_cc=c3_cached,
            avg_vc_util=avg_u,
        )

    # [v20] 缓存expert计算时间作为时间线起点, DMA从0开始
    tasks = list(cached_tasks)
    c2_total, c3_total = c2_cached, c3_cached
    xdma_busy, idma_busy = 0, 0

    c2_free, c3_free = c2_cached, c3_cached
    sram_xdma_free, sram_idma_free = 0, 0

    remaining = list(uncached)  # [v20] 仅未缓存的expert参与调度

    while remaining:
        remaining.sort(key=lambda x: -x[1])

        if len(remaining) >= 2:
            e1_eid, e1_ntok = remaining[0]
            e2_eid, e2_ntok = remaining[1]

            # 并行: e1→C2@xdma64, e2→C3@idma64
            s1, cc1, u1 = best_shape_for(e1_ntok, K, N, 64, wpe, num_vc, elem_rate)
            s2, cc2, u2 = best_shape_for(e2_ntok, K, N, 64, wpe, num_vc, elem_rate)

            # 也尝试翻转
            if cc2 > cc1:
                s1, cc1, u1, s2, cc2, u2 = s2, cc2, u2, s1, cc1, u1
                e1_eid, e1_ntok, e2_eid, e2_ntok = e2_eid, e2_ntok, e1_eid, e1_ntok

            start1 = max(c2_free, sram_xdma_free)
            start2 = max(c3_free, sram_idma_free)

            tasks.append(
                ExpertTask(
                    eid=e1_eid,
                    ntok=e1_ntok,
                    cid=2,
                    shape=s1,
                    dma_mode="xdma",
                    load_bw=64,
                    phase=1,
                    estimated_cc=cc1,
                    vc_util=u1,
                    rationale=f"并行@64, {e1_ntok}tok 热",
                )
            )
            tasks.append(
                ExpertTask(
                    eid=e2_eid,
                    ntok=e2_ntok,
                    cid=3,
                    shape=s2,
                    dma_mode="idma",
                    load_bw=64,
                    phase=1,
                    estimated_cc=cc2,
                    vc_util=u2,
                    rationale=f"并行@64, {e2_ntok}tok 冷",
                )
            )

            c2_free = start1 + cc1
            c3_free = start2 + cc2
            sram_xdma_free = start1 + cc1
            sram_idma_free = start2 + cc2
            c2_total += cc1
            c3_total += cc2
            xdma_busy += cc1
            idma_busy += cc2

            remaining.pop(0)
            remaining.pop(0)

            # 带宽窃取: 如果C3先完成且有剩余expert
            if remaining and c3_free < c2_free:
                # C3完成后, iDMA空闲, 如果C2还在DMA中, iDMA可给C3用
                # C3独享128B/cc (如果xDMA也空了)
                steal_bw = 128 if sram_xdma_free <= c3_free else 64
                next_eid, next_ntok = remaining[0]
                s3, cc3, u3 = best_shape_for(
                    next_ntok, K, N, steal_bw, wpe, num_vc, elem_rate
                )
                start3 = max(c3_free, sram_idma_free)
                if steal_bw == 128:
                    start3 = max(start3, sram_xdma_free)

                tasks.append(
                    ExpertTask(
                        eid=next_eid,
                        ntok=next_ntok,
                        cid=3,
                        shape=s3,
                        dma_mode="both" if steal_bw == 128 else "idma",
                        load_bw=steal_bw,
                        phase=2,
                        estimated_cc=cc3,
                        vc_util=u3,
                        rationale=f"BW窃取@{steal_bw}, {next_ntok}tok",
                    )
                )
                c3_free = start3 + cc3
                c3_total += cc3
                if steal_bw == 128:
                    sram_xdma_free = start3 + cc3
                    xdma_busy += cc3
                sram_idma_free = start3 + cc3
                idma_busy += cc3
                remaining.pop(0)

            # 反之: C2先完成
            elif remaining and c2_free < c3_free:
                steal_bw = 128 if sram_idma_free <= c2_free else 64
                next_eid, next_ntok = remaining[0]
                s3, cc3, u3 = best_shape_for(
                    next_ntok, K, N, steal_bw, wpe, num_vc, elem_rate
                )
                start3 = max(c2_free, sram_xdma_free)
                if steal_bw == 128:
                    start3 = max(start3, sram_idma_free)

                tasks.append(
                    ExpertTask(
                        eid=next_eid,
                        ntok=next_ntok,
                        cid=2,
                        shape=s3,
                        dma_mode="both" if steal_bw == 128 else "xdma",
                        load_bw=steal_bw,
                        phase=2,
                        estimated_cc=cc3,
                        vc_util=u3,
                        rationale=f"BW窃取@{steal_bw}, {next_ntok}tok",
                    )
                )
                c2_free = start3 + cc3
                c2_total += cc3
                sram_xdma_free = start3 + cc3
                xdma_busy += cc3
                if steal_bw == 128:
                    sram_idma_free = start3 + cc3
                    idma_busy += cc3
                remaining.pop(0)

        elif len(remaining) == 1:
            eid, ntok = remaining[0]
            cid = 2 if c2_free <= c3_free else 3
            start = max(
                c2_free if cid == 2 else c3_free, sram_xdma_free, sram_idma_free
            )
            s, cc, u = best_shape_for(ntok, K, N, 128, wpe, num_vc, elem_rate)
            tasks.append(
                ExpertTask(
                    eid=eid,
                    ntok=ntok,
                    cid=cid,
                    shape=s,
                    dma_mode="both",
                    load_bw=128,
                    phase=1,
                    estimated_cc=cc,
                    vc_util=u,
                    rationale=f"独享128B/cc, {ntok}tok",
                )
            )
            if cid == 2:
                c2_free = start + cc
                c2_total += cc
            else:
                c3_free = start + cc
                c3_total += cc
            sram_xdma_free = start + cc
            sram_idma_free = start + cc
            xdma_busy += cc
            idma_busy += cc
            remaining.pop(0)
        else:
            break

    total_time = max(c2_free, c3_free)
    avg_util = sum(t.vc_util * t.estimated_cc for t in tasks) / max(
        1, sum(t.estimated_cc for t in tasks)
    )
    return SchedulePlan(
        tasks=tasks,
        strategy="bw_steal",
        estimated_cc=total_time,
        c2_cc=c2_total,
        c3_cc=c3_total,
        sram_xdma_util=xdma_busy / total_time if total_time > 0 else 0,
        sram_idma_util=idma_busy / total_time if total_time > 0 else 0,
        avg_vc_util=avg_util,
    )


# ============================================================
#  策略5: 自适应拆分 (Adaptive Split)
# ============================================================


def _schedule_adaptive_split(
    experts_sorted: List[Tuple[int, int]],
    sys: SystemConfig,
    moe: MoELayerConfig,
    cached_map: Dict[int, int] = None,
) -> SchedulePlan:
    """
    自适应拆分:
    - 对所有热门expert (ntok ≥ 4), 穷举所有可能的拆分点
    - 使用cost函数选最优拆分
    - Phase1: 拆分后的两部分分别在C2/C3并行@64B
    - Phase2: 驻留计算剩余token
    - 特别适合: 1热多冷场景 (拆分热门 → 两个cluster都满载)
    """
    K = moe.hidden_size
    N = moe.moe_intermediate_size
    wpe = moe.wpe
    num_vc = 2
    elem_rate = sys.clusters[2].elemwise_rate

    experts = [(eid, ntok) for eid, ntok in experts_sorted if ntok > 0]
    if not experts:
        return SchedulePlan(tasks=[], strategy="adaptive_split")

    # [v20] 缓存预处理: 缓存命中expert不参与拆分优化
    cached_tasks, uncached, c2_cached, c3_cached = _preprocess_cached(
        experts, cached_map, K, N, wpe, num_vc, elem_rate
    )
    # [v20] 全部命中时直接返回
    if not uncached and cached_tasks:
        total_c = max(c2_cached, c3_cached)
        avg_u = sum(t.vc_util * t.estimated_cc for t in cached_tasks) / max(
            1, sum(t.estimated_cc for t in cached_tasks)
        )
        return SchedulePlan(
            tasks=cached_tasks,
            strategy="adaptive_split",
            estimated_cc=total_c,
            c2_cc=c2_cached,
            c3_cc=c3_cached,
            avg_vc_util=avg_u,
        )

    # [v20] 缓存expert计算时间作为时间线起点
    tasks = list(cached_tasks)
    c2_total, c3_total = c2_cached, c3_cached
    xdma_busy, idma_busy = 0, 0
    c2_free, c3_free = c2_cached, c3_cached
    sram_xdma_free, sram_idma_free = 0, 0

    remaining = list(uncached)  # [v20] 仅未缓存expert参与拆分优化

    while remaining:
        remaining.sort(key=lambda x: -x[1])

        if len(remaining) >= 2:
            e1_eid, e1_ntok = remaining[0]
            e2_eid, e2_ntok = remaining[1]

            # 基准: 不拆分并行
            s1, cc1, u1 = best_shape_for(e1_ntok, K, N, 64, wpe, num_vc, elem_rate)
            s2, cc2, u2 = best_shape_for(e2_ntok, K, N, 64, wpe, num_vc, elem_rate)
            baseline_cc = max(cc1, cc2)

            # 穷举拆分e1 (步长=1, 上限e1_ntok-1)
            best_option = (
                "nosplit",
                baseline_cc,
                [
                    (e1_eid, e1_ntok, 2, s1, cc1, u1, "xdma", 64),
                    (e2_eid, e2_ntok, 3, s2, cc2, u2, "idma", 64),
                ],
            )

            if e1_ntok >= 4:
                for split_a in range(1, e1_ntok):
                    split_b = e1_ntok - split_a
                    # C2: e1拆分A部分, C3: e2
                    sa, cca, ua = best_shape_for(
                        split_a, K, N, 64, wpe, num_vc, elem_rate
                    )
                    # 剩余split_b在C2驻留计算
                    sr, ccr, ur = best_shape_for(
                        split_b, K, N, 0, wpe, num_vc, elem_rate, resident=True
                    )
                    # 总C2时间
                    total_c2 = cca + ccr
                    parallel_phase1 = max(cca, cc2)
                    total_option = parallel_phase1 + ccr  # C3空闲时C2还在驻留计算

                    if total_c2 < best_option[1] or (total_option < best_option[1]):
                        effective = max(total_c2, cc2)
                        if effective < best_option[1]:
                            best_option = (
                                "split",
                                effective,
                                [
                                    (e1_eid, split_a, 2, sa, cca, ua, "xdma", 64),
                                    (e2_eid, e2_ntok, 3, s2, cc2, u2, "idma", 64),
                                    (e1_eid, split_b, 2, sr, ccr, ur, "none", 0),
                                ],
                            )

            # 也尝试拆分e1并让C3同时算e1的一部分
            if e1_ntok >= 4 and len(remaining) >= 2:
                mid = e1_ntok // 2
                sa, cca, ua = best_shape_for(mid, K, N, 64, wpe, num_vc, elem_rate)
                sb, ccb, ub = best_shape_for(
                    e1_ntok - mid, K, N, 64, wpe, num_vc, elem_rate
                )
                parallel = max(cca, ccb)
                # 然后e2排队
                s2q, cc2q, u2q = best_shape_for(
                    e2_ntok, K, N, 128, wpe, num_vc, elem_rate
                )
                total_option2 = parallel + cc2q
                if total_option2 < best_option[1]:
                    best_option = (
                        "split_both",
                        total_option2,
                        [
                            (e1_eid, mid, 2, sa, cca, ua, "xdma", 64),
                            (e1_eid, e1_ntok - mid, 3, sb, ccb, ub, "idma", 64),
                            (e2_eid, e2_ntok, 2, s2q, cc2q, u2q, "both", 128),
                        ],
                    )

            # 执行最优方案
            for item in best_option[2]:
                eid, ntok, cid, shape, cc, util, dma, bw = item
                start = max(
                    c2_free if cid == 2 else c3_free,
                    sram_xdma_free if dma in ("xdma", "both") else 0,
                    sram_idma_free if dma in ("idma", "both") else 0,
                )
                tasks.append(
                    ExpertTask(
                        eid=eid,
                        ntok=ntok,
                        cid=cid,
                        shape=shape,
                        dma_mode=dma,
                        load_bw=bw,
                        phase=1 if dma != "none" else 2,
                        resident=(dma == "none"),
                        estimated_cc=cc,
                        vc_util=util,
                        rationale=f"adaptive_{best_option[0]} @{bw}B/cc",
                    )
                )
                if cid == 2:
                    c2_free = start + cc
                    c2_total += cc
                else:
                    c3_free = start + cc
                    c3_total += cc
                if dma in ("xdma", "both"):
                    sram_xdma_free = start + cc
                    xdma_busy += cc
                if dma in ("idma", "both"):
                    sram_idma_free = start + cc
                    idma_busy += cc

            remaining.pop(0)
            remaining.pop(0)

        elif len(remaining) == 1:
            eid, ntok = remaining[0]
            cid = 2 if c2_free <= c3_free else 3
            start = max(
                c2_free if cid == 2 else c3_free, sram_xdma_free, sram_idma_free
            )
            s, cc, u = best_shape_for(ntok, K, N, 128, wpe, num_vc, elem_rate)
            tasks.append(
                ExpertTask(
                    eid=eid,
                    ntok=ntok,
                    cid=cid,
                    shape=s,
                    dma_mode="both",
                    load_bw=128,
                    phase=1,
                    estimated_cc=cc,
                    vc_util=u,
                    rationale=f"最后expert@128B/cc",
                )
            )
            if cid == 2:
                c2_free = start + cc
                c2_total += cc
            else:
                c3_free = start + cc
                c3_total += cc
            sram_xdma_free = start + cc
            sram_idma_free = start + cc
            xdma_busy += cc
            idma_busy += cc
            remaining.pop(0)
        else:
            break

    total_time = max(c2_free, c3_free)
    avg_util = sum(t.vc_util * t.estimated_cc for t in tasks) / max(
        1, sum(t.estimated_cc for t in tasks)
    )
    return SchedulePlan(
        tasks=tasks,
        strategy="adaptive_split",
        estimated_cc=total_time,
        c2_cc=c2_total,
        c3_cc=c3_total,
        sram_xdma_util=xdma_busy / total_time if total_time > 0 else 0,
        sram_idma_util=idma_busy / total_time if total_time > 0 else 0,
        avg_vc_util=avg_util,
    )


# ============================================================
#  策略6: 在线动态调度器 (Online Greedy)
# ============================================================


def _schedule_online_greedy(
    experts_sorted: List[Tuple[int, int]],
    sys: SystemConfig,
    moe: MoELayerConfig,
    cached_map: Dict[int, int] = None,
) -> SchedulePlan:
    """
    在线动态调度器:
    - 维护精确的时间线(c2_free, c3_free, xdma_free, idma_free)
    - 每步: 找最早空闲的cluster, 从剩余expert中选最优分配
    - 选择标准: 最小化max(c2_free, c3_free)的增量
    - DMA分配: 根据当前哪些DMA通道空闲, 动态决定带宽
    - Shape选择: 对每个candidate评估所有shape, 选时间最短的

    这是真正的动态调度: 每一步的决策都依赖于前一步的结果
    """
    K = moe.hidden_size
    N = moe.moe_intermediate_size
    wpe = moe.wpe
    num_vc = 2
    elem_rate = sys.clusters[2].elemwise_rate

    experts = [(eid, ntok) for eid, ntok in experts_sorted if ntok > 0]
    if not experts:
        return SchedulePlan(tasks=[], strategy="online_greedy")

    # [v20] 缓存预处理: 缓存命中expert不占用DMA, 释放通道给在线决策
    cached_tasks, uncached, c2_cached, c3_cached = _preprocess_cached(
        experts, cached_map, K, N, wpe, num_vc, elem_rate
    )
    # [v20] 全部命中时直接返回
    if not uncached and cached_tasks:
        total_c = max(c2_cached, c3_cached)
        avg_u = sum(t.vc_util * t.estimated_cc for t in cached_tasks) / max(
            1, sum(t.estimated_cc for t in cached_tasks)
        )
        return SchedulePlan(
            tasks=cached_tasks,
            strategy="online_greedy",
            estimated_cc=total_c,
            c2_cc=c2_cached,
            c3_cc=c3_cached,
            avg_vc_util=avg_u,
        )

    # [v20] 缓存expert计算时间作为cluster起点, DMA从0开始(缓存不占DMA)
    tasks = list(cached_tasks)
    c2_total, c3_total = c2_cached, c3_cached
    xdma_busy, idma_busy = 0, 0

    # 精确时间线 (缓存expert不占用DMA, DMA通道从t=0可用)
    c2_free, c3_free = c2_cached, c3_cached
    xdma_free, idma_free = 0, 0

    remaining = list(uncached)  # [v20] 仅未缓存expert参与在线调度

    while remaining:
        # 找最早空闲的cluster
        now = min(c2_free, c3_free)
        cid_first = 2 if c2_free <= c3_free else 3
        cid_second = 3 if cid_first == 2 else 2

        c_first_free = c2_free if cid_first == 2 else c3_free
        c_second_free = c3_free if cid_first == 2 else c2_free

        # 判断当前DMA可用状态
        xdma_available = xdma_free <= c_first_free
        idma_available = idma_free <= c_first_free

        if xdma_available and idma_available:
            avail_bw = 128
            avail_dma = "both"
        elif xdma_available:
            avail_bw = 64
            avail_dma = "xdma"
        elif idma_available:
            avail_bw = 64
            avail_dma = "idma"
        else:
            # 两个DMA都忙, 等最早空闲的DMA
            wait_until = min(xdma_free, idma_free)
            avail_bw = 64
            avail_dma = "xdma" if xdma_free <= idma_free else "idma"

        # 是否可以两个cluster同时启动?
        can_parallel = (
            len(remaining) >= 2
            and abs(c2_free - c3_free) < 200  # 两个cluster几乎同时空闲
            and xdma_available
            and idma_available
        )

        if can_parallel:
            # 尝试所有(e_i, e_j)组合取最优
            best_combo = None
            best_makespan = float("inf")

            for i in range(min(len(remaining), 5)):
                for j in range(i + 1, min(len(remaining), 5)):
                    ei_eid, ei_ntok = remaining[i]
                    ej_eid, ej_ntok = remaining[j]

                    # e_i→C2@xdma, e_j→C3@idma
                    si, cci, ui = best_shape_for(
                        ei_ntok, K, N, 64, wpe, num_vc, elem_rate
                    )
                    sj, ccj, uj = best_shape_for(
                        ej_ntok, K, N, 64, wpe, num_vc, elem_rate
                    )
                    ms1 = max(c2_free + cci, c3_free + ccj)

                    # 反过来
                    si2, cci2, ui2 = best_shape_for(
                        ej_ntok, K, N, 64, wpe, num_vc, elem_rate
                    )
                    sj2, ccj2, uj2 = best_shape_for(
                        ei_ntok, K, N, 64, wpe, num_vc, elem_rate
                    )
                    ms2 = max(c2_free + cci2, c3_free + ccj2)

                    if ms1 <= ms2 and ms1 < best_makespan:
                        best_makespan = ms1
                        best_combo = (
                            i,
                            j,
                            ei_eid,
                            ei_ntok,
                            ej_eid,
                            ej_ntok,
                            si,
                            cci,
                            ui,
                            sj,
                            ccj,
                            uj,
                            False,
                        )
                    elif ms2 < best_makespan:
                        best_makespan = ms2
                        best_combo = (
                            i,
                            j,
                            ej_eid,
                            ej_ntok,
                            ei_eid,
                            ei_ntok,
                            si2,
                            cci2,
                            ui2,
                            sj2,
                            ccj2,
                            uj2,
                            True,
                        )

            if best_combo:
                _, _, e1_eid, e1_ntok, e2_eid, e2_ntok, s1, cc1, u1, s2, cc2, u2, _ = (
                    best_combo
                )
                start1 = max(c2_free, xdma_free)
                start2 = max(c3_free, idma_free)

                tasks.append(
                    ExpertTask(
                        eid=e1_eid,
                        ntok=e1_ntok,
                        cid=2,
                        shape=s1,
                        dma_mode="xdma",
                        load_bw=64,
                        phase=1,
                        estimated_cc=cc1,
                        vc_util=u1,
                        rationale=f"online并行@64, {e1_ntok}tok",
                    )
                )
                tasks.append(
                    ExpertTask(
                        eid=e2_eid,
                        ntok=e2_ntok,
                        cid=3,
                        shape=s2,
                        dma_mode="idma",
                        load_bw=64,
                        phase=1,
                        estimated_cc=cc2,
                        vc_util=u2,
                        rationale=f"online并行@64, {e2_ntok}tok",
                    )
                )

                c2_free = start1 + cc1
                c3_free = start2 + cc2
                xdma_free = start1 + cc1
                idma_free = start2 + cc2
                c2_total += cc1
                c3_total += cc2
                xdma_busy += cc1
                idma_busy += cc2

                # 移除已选expert (从后往前移除)
                idx_i, idx_j = best_combo[0], best_combo[1]
                if best_combo[12]:  # swapped
                    remaining.pop(idx_j)
                    remaining.pop(idx_i)
                else:
                    remaining.pop(idx_j)
                    remaining.pop(idx_i)
                continue

        # 单expert分配到最早空闲的cluster
        best_task = None
        best_end = float("inf")

        for idx in range(min(len(remaining), 5)):
            eid, ntok = remaining[idx]
            # 尝试各种DMA配置
            configs = []
            if xdma_available and idma_available:
                configs.append((cid_first, "both", 128))
            if xdma_available:
                configs.append((cid_first, "xdma", 64))
            if idma_available:
                configs.append((cid_first, "idma", 64))
            # 也可以等另一个cluster
            if c_second_free - c_first_free < 5000:
                if xdma_available:
                    configs.append((cid_second, "xdma", 64))
                if idma_available:
                    configs.append((cid_second, "idma", 64))

            for cid, dma, bw in configs:
                s, cc, u = best_shape_for(ntok, K, N, bw, wpe, num_vc, elem_rate)
                c_free = c2_free if cid == 2 else c3_free
                start = c_free
                if "xdma" in dma or dma == "both":
                    start = max(start, xdma_free)
                if "idma" in dma or dma == "both":
                    start = max(start, idma_free)
                end = start + cc

                if end < best_end:
                    best_end = end
                    best_task = (idx, eid, ntok, cid, s, cc, u, dma, bw, start)

        if best_task:
            idx, eid, ntok, cid, s, cc, u, dma, bw, start = best_task
            tasks.append(
                ExpertTask(
                    eid=eid,
                    ntok=ntok,
                    cid=cid,
                    shape=s,
                    dma_mode=dma,
                    load_bw=bw,
                    phase=1,
                    estimated_cc=cc,
                    vc_util=u,
                    rationale=f"online_greedy @{bw}B/cc",
                )
            )
            if cid == 2:
                c2_free = start + cc
                c2_total += cc
            else:
                c3_free = start + cc
                c3_total += cc
            if dma in ("xdma", "both"):
                xdma_free = start + cc
                xdma_busy += cc
            if dma in ("idma", "both"):
                idma_free = start + cc
                idma_busy += cc
            remaining.pop(idx)
        else:
            break

    total_time = max(c2_free, c3_free)
    avg_util = sum(t.vc_util * t.estimated_cc for t in tasks) / max(
        1, sum(t.estimated_cc for t in tasks)
    )
    return SchedulePlan(
        tasks=tasks,
        strategy="online_greedy",
        estimated_cc=total_time,
        c2_cc=c2_total,
        c3_cc=c3_total,
        sram_xdma_util=xdma_busy / total_time if total_time > 0 else 0,
        sram_idma_util=idma_busy / total_time if total_time > 0 else 0,
        avg_vc_util=avg_util,
    )


# ============================================================
#  策略7: 冷门批量处理 (Cold Batch)
# ============================================================


def _schedule_cold_batch(
    experts_sorted: List[Tuple[int, int]],
    sys: SystemConfig,
    moe: MoELayerConfig,
    cached_map: Dict[int, int] = None,
) -> SchedulePlan:
    """
    冷门批量处理:
    - 将expert分为hot(≥4tok)和cold(1-3tok)
    - Hot expert: 并行@64B/cc处理
    - Cold expert: 批量排队, 利用hot完成后释放的带宽
    - Cold expert在空闲cluster上独享128B/cc快速消化

    特别适合: 冷门专家多的分布 (大量1-2token的expert)
    """
    K = moe.hidden_size
    N = moe.moe_intermediate_size
    wpe = moe.wpe
    num_vc = 2
    elem_rate = sys.clusters[2].elemwise_rate

    experts = [(eid, ntok) for eid, ntok in experts_sorted if ntok > 0]
    if not experts:
        return SchedulePlan(tasks=[], strategy="cold_batch")

    # [v20] 缓存预处理: 缓存命中expert不参与hot/cold分类
    cached_tasks, uncached, c2_cached, c3_cached = _preprocess_cached(
        experts, cached_map, K, N, wpe, num_vc, elem_rate
    )
    # [v20] 全部命中时直接返回
    if not uncached and cached_tasks:
        total_c = max(c2_cached, c3_cached)
        avg_u = sum(t.vc_util * t.estimated_cc for t in cached_tasks) / max(
            1, sum(t.estimated_cc for t in cached_tasks)
        )
        return SchedulePlan(
            tasks=cached_tasks,
            strategy="cold_batch",
            estimated_cc=total_c,
            c2_cc=c2_cached,
            c3_cc=c3_cached,
            avg_vc_util=avg_u,
        )

    # [v20] 仅对未缓存expert进行hot/cold分类
    hot = [(eid, ntok) for eid, ntok in uncached if ntok >= 4]
    cold = [(eid, ntok) for eid, ntok in uncached if ntok < 4]

    # [v20] 缓存expert计算时间作为时间线起点
    tasks = list(cached_tasks)
    c2_total, c3_total = c2_cached, c3_cached
    xdma_busy, idma_busy = 0, 0
    c2_free, c3_free = c2_cached, c3_cached
    sram_xdma_free, sram_idma_free = 0, 0

    # Phase 1: 先处理hot expert (并行@64)
    remaining_hot = list(hot)
    while len(remaining_hot) >= 2:
        e1_eid, e1_ntok = remaining_hot.pop(0)
        e2_eid, e2_ntok = remaining_hot.pop(0)

        s1, cc1, u1 = best_shape_for(e1_ntok, K, N, 64, wpe, num_vc, elem_rate)
        s2, cc2, u2 = best_shape_for(e2_ntok, K, N, 64, wpe, num_vc, elem_rate)

        start1 = max(c2_free, sram_xdma_free)
        start2 = max(c3_free, sram_idma_free)

        tasks.append(
            ExpertTask(
                eid=e1_eid,
                ntok=e1_ntok,
                cid=2,
                shape=s1,
                dma_mode="xdma",
                load_bw=64,
                phase=1,
                estimated_cc=cc1,
                vc_util=u1,
                rationale=f"hot并行@64, {e1_ntok}tok",
            )
        )
        tasks.append(
            ExpertTask(
                eid=e2_eid,
                ntok=e2_ntok,
                cid=3,
                shape=s2,
                dma_mode="idma",
                load_bw=64,
                phase=1,
                estimated_cc=cc2,
                vc_util=u2,
                rationale=f"hot并行@64, {e2_ntok}tok",
            )
        )

        c2_free = start1 + cc1
        c3_free = start2 + cc2
        sram_xdma_free = start1 + cc1
        sram_idma_free = start2 + cc2
        c2_total += cc1
        c3_total += cc2
        xdma_busy += cc1
        idma_busy += cc2

    # 剩余hot + 所有cold合并处理
    remaining = remaining_hot + cold

    # Phase 2: 剩余expert轮流用128B/cc快速消化
    while remaining:
        eid, ntok = remaining.pop(0)
        cid = 2 if c2_free <= c3_free else 3
        start = max(c2_free if cid == 2 else c3_free, sram_xdma_free, sram_idma_free)
        s, cc, u = best_shape_for(ntok, K, N, 128, wpe, num_vc, elem_rate)
        tasks.append(
            ExpertTask(
                eid=eid,
                ntok=ntok,
                cid=cid,
                shape=s,
                dma_mode="both",
                load_bw=128,
                phase=2,
                estimated_cc=cc,
                vc_util=u,
                rationale=f"cold_batch@128, {ntok}tok",
            )
        )
        if cid == 2:
            c2_free = start + cc
            c2_total += cc
        else:
            c3_free = start + cc
            c3_total += cc
        sram_xdma_free = start + cc
        sram_idma_free = start + cc
        xdma_busy += cc
        idma_busy += cc

    total_time = max(c2_free, c3_free)
    avg_util = sum(t.vc_util * t.estimated_cc for t in tasks) / max(
        1, sum(t.estimated_cc for t in tasks)
    )
    return SchedulePlan(
        tasks=tasks,
        strategy="cold_batch",
        estimated_cc=total_time,
        c2_cc=c2_total,
        c3_cc=c3_total,
        sram_xdma_util=xdma_busy / total_time if total_time > 0 else 0,
        sram_idma_util=idma_busy / total_time if total_time > 0 else 0,
        avg_vc_util=avg_util,
    )


# ============================================================
#  策略8: 统一动态调度器 (Unified Dynamic Scheduler)
#  融合所有策略的核心思想 + DMA预取 + 专家克隆 + shape切换
# ============================================================


def _unified_dynamic_scheduler(
    experts_sorted: List[Tuple[int, int]],
    sys: SystemConfig,
    moe: MoELayerConfig,
    cached_map: Dict[int, int] = None,
) -> SchedulePlan:
    """
    统一动态调度器 - 融合所有策略精华 + 三大创新:

    创新1: DMA预取 (DMA Prefetch)
      当hot expert在cluster_A上compute-bound时, DMA通道空闲.
      利用这段DMA slack给cluster_B预取下一个cold expert的权重.
      M=8@[4×8×8]: DMA slack=68,202cc → 可预取4.163MB (恰好一整个expert!)
      这意味着cold expert到达cluster_B时, 权重已经部分或全部到位.

    创新2: 专家克隆 (Expert Clone)
      当只有1-2个active expert且tokens极多时, C2和C3都加载同一专家权重,
      各自处理一半的token, 实现2×加速.
      条件: compute_time >> 2 × dma_time (即加载两份权重仍比单cluster快)

    创新3: 动态shape切换
      同一expert的不同执行阶段使用不同shape:
      - 流式阶段: [4×8×8](B=64B/cc, 匹配DMA)
      - 驻留阶段: 选VC利用率最高的shape (e.g. M=1用[1×8×32])
      - 克隆阶段: 根据半M值选最优shape

    调度决策框架:
      每个调度点(某cluster空闲时), 评估所有可选动作:
      A) 从剩余expert中选一个, stream到当前cluster
      B) 从剩余expert中选一个, clone到两个cluster
      C) 利用DMA slack预取next expert到另一个cluster
      D) 当前cluster已有驻留权重, 切换shape继续算剩余token
      选择使max(c2_end, c3_end)最小化的动作.
    """
    K = moe.hidden_size
    N = moe.moe_intermediate_size
    wpe = moe.wpe
    num_vc = 2
    elem_rate = sys.clusters[2].elemwise_rate
    K_half = K // 2

    experts = [(eid, ntok) for eid, ntok in experts_sorted if ntok > 0]
    if not experts:
        return SchedulePlan(tasks=[], strategy="unified_dynamic")

    # --- DMA时间常量 ---
    expert_weight = int(3 * K * N * wpe)
    gu_weight = int(2 * K * N * wpe)
    dn_weight = int(K * N * wpe)

    # 缓存预处理
    cached_tasks, uncached, c2_cached, c3_cached = _preprocess_cached(
        experts, cached_map, K, N, wpe, num_vc, elem_rate
    )
    if not uncached and cached_tasks:
        total_c = max(c2_cached, c3_cached)
        avg_u = sum(t.vc_util * t.estimated_cc for t in cached_tasks) / max(
            1, sum(t.estimated_cc for t in cached_tasks)
        )
        return SchedulePlan(
            tasks=cached_tasks,
            strategy="unified_dynamic",
            estimated_cc=total_c,
            c2_cc=c2_cached,
            c3_cc=c3_cached,
            avg_vc_util=avg_u,
        )

    # [v20] 缓存预处理: DMA预取+克隆与缓存联合优化
    cached_tasks, uncached, c2_cached, c3_cached = _preprocess_cached(
        experts, cached_map, K, N, wpe, num_vc, elem_rate
    )
    # [v20] 全部命中时直接返回
    if not uncached and cached_tasks:
        total_c = max(c2_cached, c3_cached)
        avg_u = sum(t.vc_util * t.estimated_cc for t in cached_tasks) / max(
            1, sum(t.estimated_cc for t in cached_tasks)
        )
        return SchedulePlan(
            tasks=cached_tasks,
            strategy="unified_dynamic",
            estimated_cc=total_c,
            c2_cc=c2_cached,
            c3_cc=c3_cached,
            avg_vc_util=avg_u,
        )

    # [v20] 缓存expert计算时间初始化时间线, DMA从0开始
    tasks = list(cached_tasks)
    c2_total, c3_total = c2_cached, c3_cached
    xdma_busy, idma_busy = 0, 0

    # --- 精确时间线 (缓存expert不占用DMA, DMA通道从t=0可用) ---
    c2_free, c3_free = c2_cached, c3_cached
    xdma_free, idma_free = 0, 0  # DMA通道空闲时刻
    c2_dma_done, c3_dma_done = 0, 0  # 各cluster DMA结束时刻

    # --- 预取状态 ---
    # prefetched[cid] = (eid, bytes_loaded, total_bytes, prefetch_end_time)
    prefetched = {2: None, 3: None}

    remaining = list(uncached)

    def _estimate_stream(ntok, bw):
        """估计streaming expert的总时间和各阶段细节"""
        s, cc, u = best_shape_for(ntok, K, N, bw, wpe, num_vc, elem_rate)
        # 计算DMA slack (用于预取)
        gu_compute = gemm_cycles(ntok, K, N, s)
        dn_compute = gemm_cycles(ntok, N, K_half, s)
        gu_dma = dma_cc(gu_weight, bw)
        dn_dma = dma_cc(dn_weight, bw)
        gu_slack = max(0, gu_compute - gu_dma)
        dn_slack = max(0, dn_compute - dn_dma)
        return s, cc, u, gu_slack, dn_slack

    def _estimate_clone(ntok, bw):
        """估计克隆模式: 两个cluster各处理ntok//2个token"""
        half = max(1, ntok // 2)
        other = ntok - half
        s1, cc1, u1 = best_shape_for(half, K, N, bw, wpe, num_vc, elem_rate)
        s2, cc2, u2 = best_shape_for(other, K, N, bw, wpe, num_vc, elem_rate)
        # 克隆模式: 两份权重都要搬, DMA时间 = 2×dma@bw
        # 但如果bw=64, 可以C2=xdma, C3=idma并行搬 → 各只需1×dma
        return s1, cc1, u1, s2, cc2, u2, half, other

    def _prefetch_possible(slack_cc, bw):
        """在slack时间内, 以bw带宽, 能预取多少bytes"""
        return slack_cc * bw

    def _can_clone(ntok):
        """判断是否值得克隆: compute远超DMA时"""
        if ntok < 4:
            return False
        half = ntok // 2
        s, cc, _ = best_shape_for(half, K, N, 64, wpe, num_vc, elem_rate)
        # 克隆后每cluster的compute
        single_s, single_cc, _ = best_shape_for(ntok, K, N, 128, wpe, num_vc, elem_rate)
        # 克隆: 两cluster并行@64, 各算half → max(cc_c2, cc_c3)
        # 不克隆: 一个cluster@128 → single_cc
        # 克隆划算当 cc < single_cc
        return cc < single_cc * 0.75  # 至少快25%才值得

    # === 全局dominant expert检测: 三种方案比较 ===
    # 方案1(clone): C2+C3各@64处理一半dominant, 然后others串行@128
    # 方案2(greedy): 逐对配对@64
    # 方案3(split+resident): C2 stream(small)+resident(rest), C3独享DMA处理others
    remaining.sort(key=lambda x: -x[1])
    if len(remaining) >= 2 and remaining[0][1] >= 16:
        hot_eid, hot_ntok = remaining[0]
        others = remaining[1:]

        # --- 方案1: clone ---
        half1 = hot_ntok // 2
        half2 = hot_ntok - half1
        _, cc_clone1, _ = best_shape_for(half1, K, N, 64, wpe, num_vc, elem_rate)
        _, cc_clone2, _ = best_shape_for(half2, K, N, 64, wpe, num_vc, elem_rate)
        clone_phase = max(cc_clone1, cc_clone2)
        # clone后others用greedy配对估计
        others_greedy = sorted([ntok for _, ntok in others], reverse=True)
        others_greedy_cc = 0
        while len(others_greedy) >= 2:
            oa = others_greedy.pop(0)
            ob = others_greedy.pop(0)
            _, cc_oa, _ = best_shape_for(oa, K, N, 64, wpe, num_vc, elem_rate)
            _, cc_ob, _ = best_shape_for(ob, K, N, 64, wpe, num_vc, elem_rate)
            others_greedy_cc += max(cc_oa, cc_ob)
        if others_greedy:
            _, cc_ol, _ = best_shape_for(
                others_greedy[0], K, N, 128, wpe, num_vc, elem_rate
            )
            others_greedy_cc += cc_ol
        clone_total = clone_phase + others_greedy_cc

        # --- 方案2: greedy配对 ---
        greedy_toks = sorted([ntok for _, ntok in remaining], reverse=True)
        greedy_total = 0
        while len(greedy_toks) >= 2:
            ta = greedy_toks.pop(0)
            tb = greedy_toks.pop(0)
            _, cc_a, _ = best_shape_for(ta, K, N, 64, wpe, num_vc, elem_rate)
            _, cc_b, _ = best_shape_for(tb, K, N, 64, wpe, num_vc, elem_rate)
            greedy_total += max(cc_a, cc_b)
        if greedy_toks:
            _, cc_last, _ = best_shape_for(
                greedy_toks[0], K, N, 128, wpe, num_vc, elem_rate
            )
            greedy_total += cc_last

        # --- 方案3: split+resident (如果有others) ---
        split_total = float("inf")
        best_split_a = 2
        if others:
            first_other_ntok = others[0][1]
            _, cc_first_other, _ = best_shape_for(
                first_other_ntok, K, N, 64, wpe, num_vc, elem_rate
            )
            rest_others_cc = 0
            for _, ont in others[1:]:
                _, cc_ont, _ = best_shape_for(ont, K, N, 128, wpe, num_vc, elem_rate)
                rest_others_cc += cc_ont
            c3_time = cc_first_other + rest_others_cc

            for trial_a in range(1, min(hot_ntok, 33)):
                trial_b = hot_ntok - trial_a
                _, cc_sa, _ = best_shape_for(trial_a, K, N, 64, wpe, num_vc, elem_rate)
                _, cc_rb, _ = best_shape_for(
                    trial_b, K, N, 0, wpe, num_vc, elem_rate, resident=True
                )
                c2_time = cc_sa + cc_rb
                ms = max(c2_time, c3_time)
                if ms < split_total:
                    split_total = ms
                    best_split_a = trial_a

        # --- 选最优方案 ---
        best_alt = min(clone_total, split_total)
        if best_alt < greedy_total * 0.95:
            if clone_total <= split_total:
                # 执行clone方案: 只clone dominant, others留给greedy循环
                s1, cc1, u1 = best_shape_for(half1, K, N, 64, wpe, num_vc, elem_rate)
                s2, cc2, u2 = best_shape_for(half2, K, N, 64, wpe, num_vc, elem_rate)
                start_c2 = max(c2_free, xdma_free)
                start_c3 = max(c3_free, idma_free)
                tasks.append(
                    ExpertTask(
                        eid=hot_eid,
                        ntok=half1,
                        cid=2,
                        shape=s1,
                        dma_mode="xdma",
                        load_bw=64,
                        phase=1,
                        estimated_cc=cc1,
                        vc_util=u1,
                        rationale=f"dominant_clone {hot_ntok}→{half1}+{half2}, C2@xdma64",
                    )
                )
                tasks.append(
                    ExpertTask(
                        eid=hot_eid,
                        ntok=half2,
                        cid=3,
                        shape=s2,
                        dma_mode="idma",
                        load_bw=64,
                        phase=1,
                        estimated_cc=cc2,
                        vc_util=u2,
                        rationale=f"dominant_clone {hot_ntok}→{half1}+{half2}, C3@idma64",
                    )
                )
                c2_free = start_c2 + cc1
                c3_free = start_c3 + cc2
                xdma_free = start_c2 + cc1
                idma_free = start_c3 + cc2
                c2_total += cc1
                c3_total += cc2
                xdma_busy += cc1
                idma_busy += cc2

                # 只移除dominant expert, others留给greedy循环
                remaining = [e for e in remaining if e[0] != hot_eid]
            else:
                # 执行split+resident方案
                split_b = hot_ntok - best_split_a
                s_sa, cc_sa, u_sa, _, _ = _estimate_stream(best_split_a, 64)
                s_rb, cc_rb, u_rb = best_shape_for(
                    split_b, K, N, 0, wpe, num_vc, elem_rate, resident=True
                )

                start_c2 = max(c2_free, xdma_free)
                tasks.append(
                    ExpertTask(
                        eid=hot_eid,
                        ntok=best_split_a,
                        cid=2,
                        shape=s_sa,
                        dma_mode="xdma",
                        load_bw=64,
                        phase=1,
                        estimated_cc=cc_sa,
                        vc_util=u_sa,
                        rationale=f"dominant_split stream {best_split_a}tok",
                    )
                )
                # 第一个other配对@64 idma
                first_eid, first_ntok = others[0]
                s_f, cc_f, u_f, _, _ = _estimate_stream(first_ntok, 64)
                start_c3 = max(c3_free, idma_free)
                tasks.append(
                    ExpertTask(
                        eid=first_eid,
                        ntok=first_ntok,
                        cid=3,
                        shape=s_f,
                        dma_mode="idma",
                        load_bw=64,
                        phase=1,
                        estimated_cc=cc_f,
                        vc_util=u_f,
                        rationale=f"dominant_split pair {first_ntok}tok",
                    )
                )
                c2_free = start_c2 + cc_sa
                c3_free = start_c3 + cc_f
                xdma_free = start_c2 + cc_sa
                idma_free = start_c3 + cc_f
                c2_total += cc_sa
                c3_total += cc_f
                xdma_busy += cc_sa
                idma_busy += cc_f

                # C2: resident阶段 (0 DMA)
                tasks.append(
                    ExpertTask(
                        eid=hot_eid,
                        ntok=split_b,
                        cid=2,
                        shape=s_rb,
                        dma_mode="none",
                        load_bw=0,
                        phase=2,
                        resident=True,
                        estimated_cc=cc_rb,
                        vc_util=u_rb,
                        rationale=f"dominant_split resident {split_b}tok",
                    )
                )
                c2_free += cc_rb
                c2_total += cc_rb
                # C2做resident时DMA全空闲, C3独享128B/cc
                for other_eid, other_ntok in others[1:]:
                    s_o, cc_o, u_o = best_shape_for(
                        other_ntok, K, N, 128, wpe, num_vc, elem_rate
                    )
                    start_o = max(c3_free, xdma_free, idma_free)
                    tasks.append(
                        ExpertTask(
                            eid=other_eid,
                            ntok=other_ntok,
                            cid=3,
                            shape=s_o,
                            dma_mode="both",
                            load_bw=128,
                            phase=1,
                            estimated_cc=cc_o,
                            vc_util=u_o,
                            rationale=f"dominant_split rest@128 {other_ntok}tok",
                        )
                    )
                    c3_free = start_o + cc_o
                    xdma_free = start_o + cc_o
                    idma_free = start_o + cc_o
                    c3_total += cc_o
                    xdma_busy += cc_o
                    idma_busy += cc_o
                remaining.clear()

    # === 主调度循环 ===
    while remaining:
        remaining.sort(key=lambda x: -x[1])

        now = min(c2_free, c3_free)
        n_remaining = len(remaining)

        # --- 决策1: 只剩1个expert且token极多 → 尝试克隆 ---
        if n_remaining == 1:
            eid, ntok = remaining[0]

            if _can_clone(ntok) and ntok >= 8:
                # 克隆模式: C2和C3各装一半token
                s1, cc1, u1, s2, cc2, u2, half1, half2 = _estimate_clone(ntok, 64)
                start1 = max(c2_free, xdma_free)
                start2 = max(c3_free, idma_free)

                tasks.append(
                    ExpertTask(
                        eid=eid,
                        ntok=half1,
                        cid=2,
                        shape=s1,
                        dma_mode="xdma",
                        load_bw=64,
                        phase=1,
                        estimated_cc=cc1,
                        vc_util=u1,
                        rationale=f"克隆模式: {ntok}tok→{half1}+{half2}, C2@xdma64",
                    )
                )
                tasks.append(
                    ExpertTask(
                        eid=eid,
                        ntok=half2,
                        cid=3,
                        shape=s2,
                        dma_mode="idma",
                        load_bw=64,
                        phase=1,
                        estimated_cc=cc2,
                        vc_util=u2,
                        rationale=f"克隆模式: {ntok}tok→{half1}+{half2}, C3@idma64",
                    )
                )

                c2_free = start1 + cc1
                c3_free = start2 + cc2
                xdma_free = start1 + cc1
                idma_free = start2 + cc2
                c2_total += cc1
                c3_total += cc2
                xdma_busy += cc1
                idma_busy += cc2
                remaining.pop(0)
                continue
            else:
                # 不值得克隆, 独享128B/cc
                cid = 2 if c2_free <= c3_free else 3
                start = max(c2_free if cid == 2 else c3_free, xdma_free, idma_free)
                s, cc, u = best_shape_for(ntok, K, N, 128, wpe, num_vc, elem_rate)
                tasks.append(
                    ExpertTask(
                        eid=eid,
                        ntok=ntok,
                        cid=cid,
                        shape=s,
                        dma_mode="both",
                        load_bw=128,
                        phase=1,
                        estimated_cc=cc,
                        vc_util=u,
                        rationale=f"独享128B/cc, {ntok}tok",
                    )
                )
                if cid == 2:
                    c2_free = start + cc
                    c2_total += cc
                else:
                    c3_free = start + cc
                    c3_total += cc
                xdma_free = start + cc
                idma_free = start + cc
                xdma_busy += cc
                idma_busy += cc
                remaining.pop(0)
                continue

        # --- 决策2: ≥2个expert → 评估多种方案, 选最优 ---

        # 候选方案列表: (makespan_end, plan_items)
        candidates = []

        # === 方案A: 并行@64 (经典phase_based) ===
        for i in range(min(n_remaining, 6)):
            for j in range(i + 1, min(n_remaining, 6)):
                ei_eid, ei_ntok = remaining[i]
                ej_eid, ej_ntok = remaining[j]

                s_i, cc_i, u_i, gu_slack_i, dn_slack_i = _estimate_stream(ei_ntok, 64)
                s_j, cc_j, u_j, gu_slack_j, dn_slack_j = _estimate_stream(ej_ntok, 64)

                start_i = max(c2_free, xdma_free)
                start_j = max(c3_free, idma_free)
                end_i = start_i + cc_i
                end_j = start_j + cc_j
                makespan = max(end_i, end_j)

                # DMA预取分析: 两个expert的DMA slack总和
                total_slack_i = gu_slack_i + dn_slack_i
                total_slack_j = gu_slack_j + dn_slack_j

                # 如果有DMA slack, 可以预取下一个expert (第3个)
                prefetch_benefit = 0
                if n_remaining >= 3:
                    next_eid, next_ntok = remaining[
                        [k for k in range(n_remaining) if k != i and k != j][0]
                    ]
                    # 先完成的cluster释放DMA, 可以在另一个还在计算时开始预取
                    if end_i < end_j:
                        # C2先完成, xdma空闲, 在C3还在计算期间预取到C2
                        prefetch_avail = end_j - end_i  # 额外时间窗口
                        prefetch_bytes_avail = (total_slack_i + prefetch_avail) * 64
                    else:
                        prefetch_avail = end_i - end_j
                        prefetch_bytes_avail = (total_slack_j + prefetch_avail) * 64
                    # 如果能预取整个expert, 下一个expert就不需要等DMA
                    if prefetch_bytes_avail >= expert_weight:
                        # 理想: next expert变成resident, 计算时间大幅缩短
                        _, cc_next_res, u_next_res = best_shape_for(
                            next_ntok, K, N, 0, wpe, num_vc, elem_rate, resident=True
                        )
                        _, cc_next_stream, _ = best_shape_for(
                            next_ntok, K, N, 128, wpe, num_vc, elem_rate
                        )
                        prefetch_benefit = cc_next_stream - cc_next_res

                # 加权makespan (考虑预取收益)
                effective_makespan = (
                    makespan - prefetch_benefit * 0.5
                )  # 折扣: 预取不总是完美
                candidates.append(
                    (
                        effective_makespan,
                        makespan,
                        "parallel",
                        [
                            (
                                i,
                                ei_eid,
                                ei_ntok,
                                2,
                                s_i,
                                cc_i,
                                u_i,
                                "xdma",
                                64,
                                gu_slack_i + dn_slack_i,
                            ),
                            (
                                j,
                                ej_eid,
                                ej_ntok,
                                3,
                                s_j,
                                cc_j,
                                u_j,
                                "idma",
                                64,
                                gu_slack_j + dn_slack_j,
                            ),
                        ],
                    )
                )

                # 也尝试交换C2↔C3
                start_i2 = max(c2_free, idma_free)
                start_j2 = max(c3_free, xdma_free)
                end_i2 = start_i2 + cc_j  # j→C2
                end_j2 = start_j2 + cc_i  # i→C3
                makespan2 = max(end_i2, end_j2)
                if makespan2 < makespan:
                    candidates.append(
                        (
                            makespan2,
                            makespan2,
                            "parallel_swap",
                            [
                                (
                                    j,
                                    ej_eid,
                                    ej_ntok,
                                    2,
                                    s_j,
                                    cc_j,
                                    u_j,
                                    "idma",
                                    64,
                                    gu_slack_j + dn_slack_j,
                                ),
                                (
                                    i,
                                    ei_eid,
                                    ei_ntok,
                                    3,
                                    s_i,
                                    cc_i,
                                    u_i,
                                    "xdma",
                                    64,
                                    gu_slack_i + dn_slack_i,
                                ),
                            ],
                        )
                    )

        # === 方案B: 热门拆分 + 并行 (对1号expert拆分) ===
        # 关键: 拆分后C2做resident计算(无DMA), C3独享128B/cc处理剩余expert
        e1_eid, e1_ntok = remaining[0]
        if e1_ntok >= 4 and n_remaining >= 2:
            e2_eid, e2_ntok = remaining[1]

            # 预估C3处理剩余expert的总时间 (独享128B/cc, 串行)
            remaining_others = [remaining[k] for k in range(2, n_remaining)]
            others_cc_128 = 0
            for _, other_ntok in remaining_others:
                _, cc_other, _ = best_shape_for(
                    other_ntok, K, N, 128, wpe, num_vc, elem_rate
                )
                others_cc_128 += cc_other

            # 穷举拆分点(步进2以加速)
            step = max(1, e1_ntok // 20)
            for split_a in range(1, e1_ntok, step):
                split_b = e1_ntok - split_a
                s_a, cc_a, u_a, _, _ = _estimate_stream(split_a, 64)
                s_2, cc_2, u_2, _, _ = _estimate_stream(e2_ntok, 64)
                # 驻留阶段
                s_r, cc_r, u_r = best_shape_for(
                    split_b, K, N, 0, wpe, num_vc, elem_rate, resident=True
                )

                start_a = max(c2_free, xdma_free)
                start_2 = max(c3_free, idma_free)

                # C2: stream(split_a) + resident(split_b)
                c2_end = start_a + cc_a + cc_r
                # C3: stream(e2) + 独享128B/cc处理剩余expert
                c3_end = start_2 + cc_2 + others_cc_128
                makespan = max(c2_end, c3_end)

                # 构建完整task列表: e1 split + e2 + 剩余expert
                split_items = [
                    (0, e1_eid, split_a, 2, s_a, cc_a, u_a, "xdma", 64, 0),
                    (1, e2_eid, e2_ntok, 3, s_2, cc_2, u_2, "idma", 64, 0),
                    (-1, e1_eid, split_b, 2, s_r, cc_r, u_r, "none", 0, 0),  # resident
                ]
                # 剩余expert用C3@128B/cc
                for k_idx in range(2, n_remaining):
                    k_eid, k_ntok = remaining[k_idx]
                    sk, cck, uk = best_shape_for(
                        k_ntok, K, N, 128, wpe, num_vc, elem_rate
                    )
                    split_items.append(
                        (k_idx, k_eid, k_ntok, 3, sk, cck, uk, "both", 128, 0)
                    )

                candidates.append(
                    (
                        makespan,
                        makespan,
                        "hot_split",
                        split_items,
                    )
                )

        # === 方案C: 极端克隆 (两个expert都很热) ===
        if n_remaining >= 2:
            e1_eid, e1_ntok = remaining[0]
            e2_eid, e2_ntok = remaining[1]
            if _can_clone(e1_ntok) and e1_ntok >= 8 and e2_ntok >= 4:
                # 克隆e1到两个cluster, e2排队
                s1, cc1, u1, s2, cc2, u2, h1, h2 = _estimate_clone(e1_ntok, 64)
                clone_end = max(
                    max(c2_free, xdma_free) + cc1, max(c3_free, idma_free) + cc2
                )
                # 然后e2独享128B
                s_e2, cc_e2, u_e2 = best_shape_for(
                    e2_ntok, K, N, 128, wpe, num_vc, elem_rate
                )
                total_with_clone = clone_end + cc_e2

                # 对比: 不克隆并行
                s_1p, cc_1p, u_1p, _, _ = _estimate_stream(e1_ntok, 64)
                s_2p, cc_2p, u_2p, _, _ = _estimate_stream(e2_ntok, 64)
                parallel_makespan = max(
                    max(c2_free, xdma_free) + cc_1p, max(c3_free, idma_free) + cc_2p
                )

                if total_with_clone < parallel_makespan:
                    candidates.append(
                        (
                            total_with_clone,
                            total_with_clone,
                            "clone",
                            [
                                (0, e1_eid, h1, 2, s1, cc1, u1, "xdma", 64, 0),
                                (0, e1_eid, h2, 3, s2, cc2, u2, "idma", 64, 0),
                                (
                                    1,
                                    e2_eid,
                                    e2_ntok,
                                    -1,
                                    s_e2,
                                    cc_e2,
                                    u_e2,
                                    "both",
                                    128,
                                    0,
                                ),
                            ],  # cid=-1 → auto
                        )
                    )

        # === 选择最优方案 ===
        if not candidates:
            break

        candidates.sort(key=lambda x: x[0])
        _, real_makespan, strategy_name, items = candidates[0]

        # 执行选中方案
        indices_to_remove = set()
        for item in items:
            idx, eid, ntok, cid, shape, cc, util, dma, bw, slack = item

            # 自动选cluster
            if cid == -1:
                cid = 2 if c2_free <= c3_free else 3

            start = c2_free if cid == 2 else c3_free
            if dma in ("xdma", "both"):
                start = max(start, xdma_free)
            if dma in ("idma", "both"):
                start = max(start, idma_free)

            is_resident = dma == "none"

            tasks.append(
                ExpertTask(
                    eid=eid,
                    ntok=ntok,
                    cid=cid,
                    shape=shape,
                    dma_mode=dma,
                    load_bw=bw,
                    phase=1 if not is_resident else 2,
                    resident=is_resident,
                    estimated_cc=cc,
                    vc_util=util,
                    rationale=f"unified_{strategy_name} @{bw}B/cc "
                    f"{'slack=' + str(slack) if slack > 0 else ''}",
                )
            )

            if cid == 2:
                c2_free = start + cc
                c2_total += cc
            else:
                c3_free = start + cc
                c3_total += cc
            if dma in ("xdma", "both"):
                xdma_free = start + cc
                xdma_busy += cc
            if dma in ("idma", "both"):
                idma_free = start + cc
                idma_busy += cc

            if idx >= 0:
                indices_to_remove.add(idx)

        # 从remaining移除已处理的expert (从大到小移除避免索引偏移)
        for idx in sorted(indices_to_remove, reverse=True):
            if idx < len(remaining):
                remaining.pop(idx)

    total_time = max(c2_free, c3_free)
    avg_util = sum(t.vc_util * t.estimated_cc for t in tasks) / max(
        1, sum(t.estimated_cc for t in tasks)
    )
    return SchedulePlan(
        tasks=tasks,
        strategy="unified_dynamic",
        estimated_cc=total_time,
        c2_cc=c2_total,
        c3_cc=c3_total,
        sram_xdma_util=xdma_busy / total_time if total_time > 0 else 0,
        sram_idma_util=idma_busy / total_time if total_time > 0 else 0,
        avg_vc_util=avg_util,
    )


# ============================================================
#  策略8: 事件驱动调度器 (Event-Driven Scheduler)
#  核心创新: 将DMA通道空闲时刻与cluster计算完成时刻分开追踪
# ============================================================


def _event_driven_scheduler(
    experts_sorted: List[Tuple[int, int]],
    sys: SystemConfig,
    moe: MoELayerConfig,
    cached_map: Dict[int, int] = None,
) -> SchedulePlan:
    """
    事件驱动调度器 — 精确DMA时间线追踪 + 全局makespan最优化

    === 核心创新: DMA通道与cluster的解耦追踪 ===

    现有所有策略的共同缺陷:
      将DMA通道标记为"在整个expert计算期间占用".
      实际上, streaming模式下DMA在加载完所有weight tile后就空闲了,
      而cluster还需要继续计算最后若干tile. 对M>=8的expert:
      - DMA在total_cc的前50%时间就完成了
      - 后50%+时间, DMA通道完全空闲但未被利用

    本调度器的做法:
      - 分别追踪 cluster空闲时刻(c_free) 和 DMA通道空闲时刻(dma_free)
      - dma_free = start + expert_weight / bw  (DMA传输完成)
      - c_free   = start + total_streaming_cc   (计算完成)
      - 当 c_free > dma_free 时, DMA通道在[dma_free, c_free]内空闲
      - 该空闲DMA通道立即被重新分配, 为其他cluster加载expert

    === 调度流程 ===

    不使用固定的phase结构. 在每个"事件点"(任一cluster完成 或 DMA通道空闲时),
    评估所有可能的下一步动作, 选择使全局预估makespan最小的动作.

    动作类型:
      A) stream(E, cid, dma): 将expert E分配到cluster cid, 使用DMA通道dma @64B/cc
      B) stream_both(E, cid): 将expert E分配到cid, 使用两个DMA通道@128B/cc
      C) clone(E): C2和C3各加载一半token, 各用一个DMA @64B/cc

    全局makespan估计:
      对于每个候选动作, 计算:
      total_est = max(new_c2_free, new_c3_free) + remaining_lower_bound
      remaining_lower_bound = max(
          sum(expert_dma_bytes) / effective_bw,   # DMA下界
          sum(expert_compute_cc) / 2              # 双cluster计算下界
      )
    """
    K = moe.hidden_size
    N = moe.moe_intermediate_size
    wpe = moe.wpe
    num_vc = 2
    elem_rate = sys.clusters[2].elemwise_rate
    K_half = K // 2
    expert_weight = int(3 * K * N * wpe)  # gate+up+down total bytes

    experts = [(eid, ntok) for eid, ntok in experts_sorted if ntok > 0]
    if not experts:
        return SchedulePlan(tasks=[], strategy="event_driven")

    if cached_map is None:
        cached_map = {}

    tasks = []

    # === 精确时间线: 分别追踪cluster和DMA通道 ===
    c2_free = 0  # cluster2 计算完成时刻
    c3_free = 0  # cluster3 计算完成时刻
    xdma_free = 0  # SRAM xDMA 传输完成时刻
    idma_free = 0  # SRAM iDMA 传输完成时刻

    # 统计量
    c2_total = 0
    c3_total = 0
    xdma_busy = 0
    idma_busy = 0

    remaining = list(experts)

    # 缓存: (ntok, bw) → (shape, cc, vc_util)
    _cc_cache = {}

    def _expert_cc(ntok, bw, resident=False):
        """计算expert的streaming总时间 (含pipeline), 带缓存"""
        key = (ntok, bw, resident)
        if key not in _cc_cache:
            _cc_cache[key] = best_shape_for(
                ntok, K, N, bw, wpe, num_vc, elem_rate, resident=resident
            )
        return _cc_cache[key]

    # 预缓存DMA时间
    _dma_64 = dma_cc(expert_weight, 64)
    _dma_128 = dma_cc(expert_weight, 128)

    def _expert_dma_cc(bw):
        """expert权重的纯DMA传输时间"""
        if bw == 64:
            return _dma_64
        elif bw == 128:
            return _dma_128
        return dma_cc(expert_weight, bw) if bw > 0 else 0

    def _remaining_lower_bound(rem, c2f, c3f, xf, yf):
        """
        贪心前瞻模拟: 模拟剩余expert的快速调度, 返回估计的总makespan.

        与简单下界的关键区别:
        - 考虑cluster的实际空闲时刻 (一个可能远早于另一个)
        - 分别追踪DMA通道空闲 (DMA传输完 ≠ 计算完)
        - 先完成的cluster立刻获得下一个expert
        - 缓存命中的expert无需DMA, 立即开始计算

        这使得调度器能发现: "把hot expert放C2后, C3可以利用DMA slack连续@128处理cold experts"
        """
        if not rem:
            return max(c2f, c3f)
        # 贪心模拟: 每次把最大剩余expert分到最早空闲的cluster
        lc2, lc3, lxf, lyf = c2f, c3f, xf, yf
        sorted_rem = sorted(rem, key=lambda x: -x[1])
        idx = 0
        while idx < len(sorted_rem):
            eid_r, ntok = sorted_rem[idx]

            # 缓存命中: 无需DMA, 直接分配到最早空闲的cluster
            if eid_r in cached_map:
                cid_c = cached_map[eid_r]
                _, cc_r, _ = _expert_cc(ntok, 0, resident=True)
                if cid_c == 2:
                    lc2 = lc2 + cc_r
                else:
                    lc3 = lc3 + cc_r
                idx += 1
                continue

            # 选最早空闲的cluster
            if lc2 <= lc3:
                cid_l = 2
                cfree = lc2
            else:
                cid_l = 3
                cfree = lc3
            other_free = lc3 if cid_l == 2 else lc2

            # 尝试配对: 两个cluster都接近空闲且还有第二个expert
            if idx + 1 < len(sorted_rem) and abs(cfree - other_free) < _dma_64 // 2:
                _, ntok2 = sorted_rem[idx + 1]
                # 配对@64 vs 单个@128
                _, cc_a, _ = _expert_cc(ntok, 64)
                _, cc_b, _ = _expert_cc(ntok2, 64)
                start_a = max(lc2, lxf) if cid_l == 2 else max(lc2, lyf)
                start_b = max(lc3, lyf) if cid_l == 2 else max(lc3, lxf)
                pair_end = max(start_a + cc_a, start_b + cc_b)
                # 对比: 单@128后串行
                _, cc_128, _ = _expert_cc(ntok, 128)
                _, cc_b64, _ = _expert_cc(ntok2, 64)
                start_single = max(cfree, lxf, lyf)
                single_then_serial = max(
                    start_single + cc_128,
                    max(other_free, start_single + _expert_dma_cc(128)) + cc_b64,
                )
                if pair_end <= single_then_serial:
                    dma_t = _expert_dma_cc(64)
                    if cid_l == 2:
                        lc2 = start_a + cc_a
                        lc3 = start_b + cc_b
                        lxf = start_a + dma_t
                        lyf = start_b + dma_t
                    else:
                        lc3 = start_a + cc_a
                        lc2 = start_b + cc_b
                        lyf = start_a + dma_t
                        lxf = start_b + dma_t
                    idx += 2
                    continue

            # 尝试clone: 两cluster空闲时刻接近且expert够大
            if ntok >= 8 and abs(cfree - other_free) < _dma_64 // 2:
                h1, h2 = ntok // 2, ntok - ntok // 2
                _, cc1, _ = _expert_cc(h1, 64)
                _, cc2, _ = _expert_cc(h2, 64)
                start1 = max(lc2, lxf)
                start2 = max(lc3, lyf)
                clone_end = max(start1 + cc1, start2 + cc2)
                # 对比: 单cluster @128
                both_dma_free = max(lxf, lyf) <= cfree
                if both_dma_free:
                    _, cc_128, _ = _expert_cc(ntok, 128)
                    single_end = cfree + cc_128
                else:
                    _, cc_64, _ = _expert_cc(ntok, 64)
                    single_end = max(cfree, min(lxf, lyf)) + cc_64
                if clone_end < single_end:
                    lc2 = start1 + cc1
                    lc3 = start2 + cc2
                    dma_t = _expert_dma_cc(64)
                    lxf = start1 + dma_t
                    lyf = start2 + dma_t
                    idx += 1
                    continue

            # 确定DMA: 两个都空闲→@128, 否则用空闲的那个@64
            if max(lxf, lyf) <= cfree:
                bw_l = 128
                start_l = cfree
            elif min(lxf, lyf) <= cfree:
                bw_l = 64
                start_l = cfree
            else:
                # 两个DMA都忙, 等最早空闲的
                bw_l = 64
                start_l = max(cfree, min(lxf, lyf))
            _, cc_l, _ = _expert_cc(ntok, bw_l)
            dma_t_l = _expert_dma_cc(bw_l)
            compute_done_l = start_l + cc_l
            dma_done_l = start_l + dma_t_l
            if cid_l == 2:
                lc2 = compute_done_l
            else:
                lc3 = compute_done_l
            # DMA释放
            if bw_l == 128:
                lxf = dma_done_l
                lyf = dma_done_l
            elif lxf <= start_l:
                lxf = dma_done_l
            else:
                lyf = dma_done_l
            idx += 1
        return max(lc2, lc3)

    def _try_assign(eid, ntok, cid, dma_mode, c2f, c3f, xf, yf):
        """
        模拟将expert分配到指定cluster和DMA模式, 返回更新后的时间线.

        关键区别: DMA空闲时刻 ≠ cluster空闲时刻
        - dma_done = start + expert_weight / bw   (传输完成)
        - compute_done = start + streaming_cc       (计算完成, 含pipeline)
        - 当compute > dma时 (compute-bound), DMA在[dma_done, compute_done]内空闲
        """
        if dma_mode == "xdma":
            bw = 64
            start = max(c2f if cid == 2 else c3f, xf)
            s, cc, u = _expert_cc(ntok, bw)
            dma_t = _expert_dma_cc(bw)
            compute_done = start + cc
            dma_done = start + dma_t
            new_xf = dma_done  # DMA在传输完后立即释放!
            new_yf = yf
        elif dma_mode == "idma":
            bw = 64
            start = max(c2f if cid == 2 else c3f, yf)
            s, cc, u = _expert_cc(ntok, bw)
            dma_t = _expert_dma_cc(bw)
            compute_done = start + cc
            dma_done = start + dma_t
            new_xf = xf
            new_yf = dma_done  # DMA在传输完后立即释放!
        elif dma_mode == "both":
            bw = 128
            start = max(c2f if cid == 2 else c3f, xf, yf)
            s, cc, u = _expert_cc(ntok, bw)
            dma_t = _expert_dma_cc(bw)
            compute_done = start + cc
            dma_done = start + dma_t
            new_xf = dma_done
            new_yf = dma_done
        elif dma_mode == "none":
            # 缓存命中: 权重已驻留在TCDM, 无需DMA传输
            bw = 0
            start = c2f if cid == 2 else c3f
            s, cc, u = _expert_cc(ntok, 0, resident=True)
            compute_done = start + cc
            new_xf = xf  # DMA通道不受影响!
            new_yf = yf  # DMA通道不受影响!
        else:
            return None

        new_c2f = compute_done if cid == 2 else c2f
        new_c3f = compute_done if cid == 3 else c3f
        return (new_c2f, new_c3f, new_xf, new_yf, start, s, cc, u, bw, dma_mode, cid)

    # === 特殊情况: 只有1个expert ===
    if len(remaining) == 1:
        eid, ntok = remaining[0]

        # 如果缓存命中: 直接resident计算, 无需DMA (必须在缓存所在的cluster)
        if eid in cached_map:
            cid_cached = cached_map[eid]
            s_r, cc_r, u_r = _expert_cc(ntok, 0, resident=True)
            tasks.append(
                ExpertTask(
                    eid=eid,
                    ntok=ntok,
                    cid=cid_cached,
                    shape=s_r,
                    dma_mode="none",
                    load_bw=0,
                    phase=1,
                    resident=True,
                    estimated_cc=cc_r,
                    vc_util=u_r,
                    rationale=f"缓存命中 resident {ntok}tok @C{cid_cached}",
                )
            )
            if cid_cached == 2:
                c2_free = cc_r
                c2_total = cc_r
            else:
                c3_free = cc_r
                c3_total = cc_r
            remaining.clear()

        elif ntok >= 8:
            # 尝试clone: C2和C3各处理一半 @64
            half1, half2 = ntok // 2, ntok - ntok // 2
            s1, cc1, u1 = _expert_cc(half1, 64)
            s2, cc2, u2 = _expert_cc(half2, 64)
            clone_ms = max(cc1, cc2)
            # 对比: 单cluster @128
            _, cc_128, u_128 = _expert_cc(ntok, 128)
            if clone_ms < cc_128:
                tasks.append(
                    ExpertTask(
                        eid=eid,
                        ntok=half1,
                        cid=2,
                        shape=s1,
                        dma_mode="xdma",
                        load_bw=64,
                        phase=1,
                        estimated_cc=cc1,
                        vc_util=u1,
                        rationale=f"clone {ntok}→{half1}+{half2}",
                    )
                )
                tasks.append(
                    ExpertTask(
                        eid=eid,
                        ntok=half2,
                        cid=3,
                        shape=s2,
                        dma_mode="idma",
                        load_bw=64,
                        phase=1,
                        estimated_cc=cc2,
                        vc_util=u2,
                        rationale=f"clone {ntok}→{half1}+{half2}",
                    )
                )
                c2_free = cc1
                c3_free = cc2
                xdma_free = _expert_dma_cc(64)
                idma_free = _expert_dma_cc(64)
                c2_total = cc1
                c3_total = cc2
                xdma_busy = _expert_dma_cc(64)
                idma_busy = _expert_dma_cc(64)
                remaining.clear()

    # === 主调度循环: 事件驱动 ===
    while remaining:
        remaining.sort(key=lambda x: -x[1])
        n_rem = len(remaining)

        # 生成所有候选动作
        candidates = []  # (estimated_total, action_info)

        # 确定哪些cluster空闲, 哪些DMA通道空闲
        # 可同时分配两个expert (如果两个cluster都空闲)

        for i in range(min(n_rem, 4)):
            eid_i, ntok_i = remaining[i]

            # --- 缓存命中: 权重已驻留, 只能在缓存所在cluster执行 ---
            if eid_i in cached_map:
                cid_cached = cached_map[eid_i]
                res = _try_assign(
                    eid_i,
                    ntok_i,
                    cid_cached,
                    "none",
                    c2_free,
                    c3_free,
                    xdma_free,
                    idma_free,
                )
                if res:
                    nc2, nc3, nxf, nyf, st, s, cc, u, bw, dm, ci = res
                    rem_after = [(e, n) for e, n in remaining if e != eid_i]
                    total_est = _remaining_lower_bound(rem_after, nc2, nc3, nxf, nyf)
                    candidates.append(
                        (
                            total_est,
                            i,
                            eid_i,
                            ntok_i,
                            ci,
                            "none",
                            0,
                            s,
                            cc,
                            u,
                            st,
                            nc2,
                            nc3,
                            nxf,
                            nyf,
                        )
                    )
                # 缓存命中expert不走DMA (resident总是更优或同等)

            # --- 单expert分配到C2或C3 ---
            for cid in [2, 3]:
                # 用xdma @64
                res = _try_assign(
                    eid_i, ntok_i, cid, "xdma", c2_free, c3_free, xdma_free, idma_free
                )
                if res:
                    nc2, nc3, nxf, nyf, st, s, cc, u, bw, dm, ci = res
                    rem_after = [(e, n) for e, n in remaining if e != eid_i]
                    total_est = _remaining_lower_bound(rem_after, nc2, nc3, nxf, nyf)
                    candidates.append(
                        (
                            total_est,
                            i,
                            eid_i,
                            ntok_i,
                            ci,
                            dm,
                            bw,
                            s,
                            cc,
                            u,
                            st,
                            nc2,
                            nc3,
                            nxf,
                            nyf,
                        )
                    )

                # 用idma @64
                res = _try_assign(
                    eid_i, ntok_i, cid, "idma", c2_free, c3_free, xdma_free, idma_free
                )
                if res:
                    nc2, nc3, nxf, nyf, st, s, cc, u, bw, dm, ci = res
                    rem_after = [(e, n) for e, n in remaining if e != eid_i]
                    total_est = _remaining_lower_bound(rem_after, nc2, nc3, nxf, nyf)
                    candidates.append(
                        (
                            total_est,
                            i,
                            eid_i,
                            ntok_i,
                            ci,
                            dm,
                            bw,
                            s,
                            cc,
                            u,
                            st,
                            nc2,
                            nc3,
                            nxf,
                            nyf,
                        )
                    )

                # 用both @128
                res = _try_assign(
                    eid_i, ntok_i, cid, "both", c2_free, c3_free, xdma_free, idma_free
                )
                if res:
                    nc2, nc3, nxf, nyf, st, s, cc, u, bw, dm, ci = res
                    rem_after = [(e, n) for e, n in remaining if e != eid_i]
                    total_est = _remaining_lower_bound(rem_after, nc2, nc3, nxf, nyf)
                    candidates.append(
                        (
                            total_est,
                            i,
                            eid_i,
                            ntok_i,
                            ci,
                            dm,
                            bw,
                            s,
                            cc,
                            u,
                            st,
                            nc2,
                            nc3,
                            nxf,
                            nyf,
                        )
                    )

        # --- 配对分配: 两个expert同时上C2+C3 ---
        for i in range(min(n_rem, 3)):
            for j in range(i + 1, min(n_rem, 4)):
                eid_i, ntok_i = remaining[i]
                eid_j, ntok_j = remaining[j]
                # C2=xdma, C3=idma
                r_i = _try_assign(
                    eid_i, ntok_i, 2, "xdma", c2_free, c3_free, xdma_free, idma_free
                )
                r_j = _try_assign(
                    eid_j, ntok_j, 3, "idma", c2_free, c3_free, xdma_free, idma_free
                )
                if r_i and r_j:
                    nc2_i = r_i[0]
                    nc3_j = r_j[1]
                    nxf = r_i[2]
                    nyf = r_j[3]  # DMA各自释放时刻
                    rem_after = [
                        (e, n) for e, n in remaining if e != eid_i and e != eid_j
                    ]
                    total_est = _remaining_lower_bound(
                        rem_after, nc2_i, nc3_j, nxf, nyf
                    )
                    candidates.append(
                        (
                            total_est,
                            (i, j),
                            (eid_i, eid_j),
                            (ntok_i, ntok_j),
                            "pair",
                            "xdma+idma",
                            64,
                            (r_i[5], r_j[5]),
                            (r_i[6], r_j[6]),
                            (r_i[7], r_j[7]),
                            (r_i[4], r_j[4]),
                            nc2_i,
                            nc3_j,
                            nxf,
                            nyf,
                        )
                    )
                # C2=idma, C3=xdma (swap)
                r_i2 = _try_assign(
                    eid_i, ntok_i, 2, "idma", c2_free, c3_free, xdma_free, idma_free
                )
                r_j2 = _try_assign(
                    eid_j, ntok_j, 3, "xdma", c2_free, c3_free, xdma_free, idma_free
                )
                if r_i2 and r_j2:
                    nc2 = r_i2[0]
                    nc3 = r_j2[1]
                    nxf2 = r_j2[2]
                    nyf2 = r_i2[3]
                    rem_after = [
                        (e, n) for e, n in remaining if e != eid_i and e != eid_j
                    ]
                    total_est = _remaining_lower_bound(rem_after, nc2, nc3, nxf2, nyf2)
                    candidates.append(
                        (
                            total_est,
                            (i, j),
                            (eid_i, eid_j),
                            (ntok_i, ntok_j),
                            "pair_swap",
                            "idma+xdma",
                            64,
                            (r_i2[5], r_j2[5]),
                            (r_i2[6], r_j2[6]),
                            (r_i2[7], r_j2[7]),
                            (r_i2[4], r_j2[4]),
                            nc2,
                            nc3,
                            nxf2,
                            nyf2,
                        )
                    )

        # --- clone: 最大expert拆半, 两cluster各@64 ---
        if n_rem >= 1:
            eid_top, ntok_top = remaining[0]
            if ntok_top >= 8:
                h1, h2 = ntok_top // 2, ntok_top - ntok_top // 2
                s1, cc1, u1 = _expert_cc(h1, 64)
                s2, cc2, u2 = _expert_cc(h2, 64)
                start1 = max(c2_free, xdma_free)
                start2 = max(c3_free, idma_free)
                nc2 = start1 + cc1
                nc3 = start2 + cc2
                dma_t = _expert_dma_cc(64)
                nxf = start1 + dma_t
                nyf = start2 + dma_t
                rem_after = [(e, n) for e, n in remaining if e != eid_top]
                total_est = _remaining_lower_bound(rem_after, nc2, nc3, nxf, nyf)
                candidates.append(
                    (
                        total_est,
                        0,
                        eid_top,
                        ntok_top,
                        "clone",
                        "xdma+idma",
                        64,
                        (s1, s2),
                        (cc1, cc2),
                        (u1, u2),
                        (start1, start2),
                        nc2,
                        nc3,
                        nxf,
                        nyf,
                    )
                )

        # --- dominant split: 热门expert先stream少量tok, 再resident剩余tok ---
        if n_rem >= 1:
            eid_top, ntok_top = remaining[0]
            if ntok_top >= 6:  # 至少6tok才值得split
                for split_tok in [2, 4]:
                    if split_tok >= ntok_top:
                        continue
                    remain_tok = ntok_top - split_tok
                    for cid in [2, 3]:
                        for dm in ["xdma", "idma"]:
                            # phase1: stream split_tok @64
                            res_stream = _try_assign(
                                eid_top,
                                split_tok,
                                cid,
                                dm,
                                c2_free,
                                c3_free,
                                xdma_free,
                                idma_free,
                            )
                            if not res_stream:
                                continue
                            (
                                nc2_s,
                                nc3_s,
                                nxf_s,
                                nyf_s,
                                st_s,
                                s_s,
                                cc_s,
                                u_s,
                                bw_s,
                                dm_s,
                                ci_s,
                            ) = res_stream
                            # phase2: resident remain_tok (compute-only, weight already in TCDM)
                            s_r, cc_r, u_r = best_shape_for(
                                remain_tok,
                                K,
                                N,
                                0,
                                wpe,
                                num_vc,
                                elem_rate,
                                resident=True,
                            )
                            c_free_after_stream = nc2_s if cid == 2 else nc3_s
                            compute_done_r = c_free_after_stream + cc_r
                            nc2_r = compute_done_r if cid == 2 else nc2_s
                            nc3_r = compute_done_r if cid == 3 else nc3_s
                            rem_after = [(e, n) for e, n in remaining if e != eid_top]
                            total_est = _remaining_lower_bound(
                                rem_after, nc2_r, nc3_r, nxf_s, nyf_s
                            )
                            candidates.append(
                                (
                                    total_est,
                                    0,
                                    eid_top,
                                    ntok_top,
                                    "dom_split",
                                    dm,
                                    64,
                                    (s_s, s_r),
                                    (cc_s, cc_r),
                                    (u_s, u_r),
                                    (st_s, split_tok, remain_tok, cid),
                                    nc2_r,
                                    nc3_r,
                                    nxf_s,
                                    nyf_s,
                                )
                            )

        if not candidates:
            break

        # === 选择最优候选 ===
        candidates.sort(key=lambda x: x[0])
        best = candidates[0]

        # === 执行选中的动作 ===
        if best[4] == "pair" or best[4] == "pair_swap":
            # 配对分配
            (
                _,
                _,
                (eid_a, eid_b),
                (ntok_a, ntok_b),
                mode,
                dm_str,
                bw,
                (s_a, s_b),
                (cc_a, cc_b),
                (u_a, u_b),
                (st_a, st_b),
                new_c2,
                new_c3,
                new_xf,
                new_yf,
            ) = best
            if mode == "pair":
                dma_a, dma_b = "xdma", "idma"
                cid_a, cid_b = 2, 3
            else:
                dma_a, dma_b = "idma", "xdma"
                cid_a, cid_b = 2, 3
            tasks.append(
                ExpertTask(
                    eid=eid_a,
                    ntok=ntok_a,
                    cid=cid_a,
                    shape=s_a,
                    dma_mode=dma_a,
                    load_bw=64,
                    phase=1,
                    estimated_cc=cc_a,
                    vc_util=u_a,
                    rationale=f"ED并行@64 {ntok_a}tok",
                )
            )
            tasks.append(
                ExpertTask(
                    eid=eid_b,
                    ntok=ntok_b,
                    cid=cid_b,
                    shape=s_b,
                    dma_mode=dma_b,
                    load_bw=64,
                    phase=1,
                    estimated_cc=cc_b,
                    vc_util=u_b,
                    rationale=f"ED并行@64 {ntok_b}tok",
                )
            )
            c2_free = new_c2
            c3_free = new_c3
            xdma_free = new_xf
            idma_free = new_yf
            c2_total += cc_a
            c3_total += cc_b
            xdma_busy += _expert_dma_cc(64)
            idma_busy += _expert_dma_cc(64)
            remaining = [(e, n) for e, n in remaining if e != eid_a and e != eid_b]

        elif best[4] == "clone":
            (
                _,
                _,
                eid_top,
                ntok_top,
                _,
                _,
                _,
                (s1, s2),
                (cc1, cc2),
                (u1, u2),
                (st1, st2),
                new_c2,
                new_c3,
                new_xf,
                new_yf,
            ) = best
            h1, h2 = ntok_top // 2, ntok_top - ntok_top // 2
            tasks.append(
                ExpertTask(
                    eid=eid_top,
                    ntok=h1,
                    cid=2,
                    shape=s1,
                    dma_mode="xdma",
                    load_bw=64,
                    phase=1,
                    estimated_cc=cc1,
                    vc_util=u1,
                    rationale=f"ED clone {ntok_top}→{h1}+{h2}",
                )
            )
            tasks.append(
                ExpertTask(
                    eid=eid_top,
                    ntok=h2,
                    cid=3,
                    shape=s2,
                    dma_mode="idma",
                    load_bw=64,
                    phase=1,
                    estimated_cc=cc2,
                    vc_util=u2,
                    rationale=f"ED clone {ntok_top}→{h1}+{h2}",
                )
            )
            c2_free = new_c2
            c3_free = new_c3
            xdma_free = new_xf
            idma_free = new_yf
            c2_total += cc1
            c3_total += cc2
            dma_t = _expert_dma_cc(64)
            xdma_busy += dma_t
            idma_busy += dma_t
            remaining = [(e, n) for e, n in remaining if e != eid_top]

        elif best[4] == "dom_split":
            # dominant split: stream小部分 + resident大部分
            (
                _,
                _,
                eid_top,
                ntok_top,
                _,
                dm,
                bw,
                (s_s, s_r),
                (cc_s, cc_r),
                (u_s, u_r),
                (st_s, split_tok, remain_tok, cid),
                new_c2,
                new_c3,
                new_xf,
                new_yf,
            ) = best
            tasks.append(
                ExpertTask(
                    eid=eid_top,
                    ntok=split_tok,
                    cid=cid,
                    shape=s_s,
                    dma_mode=dm,
                    load_bw=64,
                    phase=1,
                    estimated_cc=cc_s,
                    vc_util=u_s,
                    rationale=f"ED split stream {split_tok}tok",
                )
            )
            tasks.append(
                ExpertTask(
                    eid=eid_top,
                    ntok=remain_tok,
                    cid=cid,
                    shape=s_r,
                    dma_mode="none",
                    load_bw=0,
                    phase=2,
                    estimated_cc=cc_r,
                    vc_util=u_r,
                    resident=True,
                    rationale=f"ED split resident {remain_tok}tok",
                )
            )
            c2_free = new_c2
            c3_free = new_c3
            xdma_free = new_xf
            idma_free = new_yf
            if cid == 2:
                c2_total += cc_s + cc_r
            else:
                c3_total += cc_s + cc_r
            dma_t = _expert_dma_cc(64)
            if dm == "xdma":
                xdma_busy += dma_t
            else:
                idma_busy += dma_t
            remaining = [(e, n) for e, n in remaining if e != eid_top]

        else:
            # 单expert分配
            (
                _,
                _,
                eid_i,
                ntok_i,
                cid,
                dm,
                bw,
                s,
                cc,
                u,
                st,
                new_c2,
                new_c3,
                new_xf,
                new_yf,
            ) = best
            tasks.append(
                ExpertTask(
                    eid=eid_i,
                    ntok=ntok_i,
                    cid=cid,
                    shape=s,
                    dma_mode=dm,
                    load_bw=bw,
                    phase=1,
                    estimated_cc=cc,
                    vc_util=u,
                    resident=(dm == "none"),
                    rationale=f"ED{'缓存' if dm == 'none' else '单发@' + str(bw)} {ntok_i}tok",
                )
            )
            c2_free = new_c2
            c3_free = new_c3
            xdma_free = new_xf
            idma_free = new_yf
            if cid == 2:
                c2_total += cc
            else:
                c3_total += cc
            if dm != "none":
                dma_t = _expert_dma_cc(bw)
                if dm == "xdma":
                    xdma_busy += dma_t
                elif dm == "idma":
                    idma_busy += dma_t
                else:
                    xdma_busy += dma_t // 2
                    idma_busy += dma_t // 2
            remaining = [(e, n) for e, n in remaining if e != eid_i]

    total_time = max(c2_free, c3_free)
    if total_time <= 0:
        total_time = 1
    avg_util = sum(t.vc_util * t.estimated_cc for t in tasks) / max(
        1, sum(t.estimated_cc for t in tasks)
    )

    # === 计算exit_eids: 每个cluster最后执行的expert (驻留在TCDM中) ===
    exit_eids = {}
    for t in tasks:
        if t.cid in (2, 3):
            exit_eids[t.cid] = t.eid  # 后面的覆盖前面的, 最终留下最后一个

    return SchedulePlan(
        tasks=tasks,
        strategy="event_driven",
        estimated_cc=total_time,
        c2_cc=c2_total,
        c3_cc=c3_total,
        sram_xdma_util=xdma_busy / total_time if total_time > 0 else 0,
        sram_idma_util=idma_busy / total_time if total_time > 0 else 0,
        avg_vc_util=avg_util,
        exit_eids=exit_eids,
    )


# ============================================================
#  Cost函数 + 调度入口
# ============================================================


def _event_driven_v22(
    experts: List[Tuple[int, int]],
    sys: SystemConfig,
    moe: MoELayerConfig,
    cached_map: Dict[int, int] = None,
) -> SchedulePlan:
    """
    v22 事件驱动调度器 wrapper.
    调用 event_scheduler.schedule_event_driven, 将其 Plan 转为 SchedulePlan.
    包含: 3-tier BW 决策 + cache hit + expert clone + SPLIT (stream4+resident).
    """
    from event_scheduler import schedule_event_driven

    token_dist = {eid: nt for eid, nt in experts if nt > 0}
    if not token_dist:
        return SchedulePlan(tasks=[], strategy="event_driven_v22")

    plan22 = schedule_event_driven(token_dist, sys, moe)

    tasks: List[ExpertTask] = []
    for t in plan22.tasks:
        tasks.append(
            ExpertTask(
                eid=t.eid,
                ntok=t.ntok,
                cid=t.cid,
                shape=t.shape,
                dma_mode=t.dma_mode,
                load_bw=t.load_bw,
                resident=t.resident,
                estimated_cc=t.est_cc,
                vc_util=t.util,
                rationale=t.rationale,
            )
        )

    if tasks:
        tot = sum(t.estimated_cc for t in tasks)
        avg_util = sum(t.vc_util * t.estimated_cc for t in tasks) / tot if tot else 0.0
    else:
        avg_util = 0.0

    return SchedulePlan(
        tasks=tasks,
        strategy="event_driven_v22",
        estimated_cc=plan22.makespan,
        c2_cc=plan22.c2_end,
        c3_cc=plan22.c3_end,
        avg_vc_util=avg_util,
    )


def cost_function(plan: SchedulePlan, shared_cc: int) -> float:
    """
    调度方案的cost值, 越小越好.

    cost = max(0, routed_cc/shared_cc - 1.0)
           + (1 - avg_vc_util) * 0.2
           + (1 - sram_util) * 0.1

    主目标: routed 不超过 shared (超出惩罚, 提前完成零惩罚 —
            makespan = max(routed, shared) 受 shared 兜底).
    次目标: VC利用率高, SRAM带宽利用率高.
    """
    if shared_cc <= 0:
        return float("inf")
    # v22: 直接按 routed/shared 排序 — routed 越短越好 (超出 shared 被线性惩罚;
    # 低于 shared 也线性奖励, 因为提前完成 → DMA/TCDM 可用于跨层缓存预取).
    time_penalty = plan.estimated_cc / shared_cc
    util_penalty = (1.0 - plan.avg_vc_util) * 0.002
    bw_penalty = (1.0 - (plan.sram_xdma_util + plan.sram_idma_util) / 2) * 0.001
    return time_penalty + util_penalty + bw_penalty


def schedule(
    M: int,
    token_dist: Dict[int, int],
    sys: SystemConfig,
    moe: MoELayerConfig,
    shared_cc: int = 0,
    cached_map: Dict[int, int] = None,
) -> SchedulePlan:
    """
    动态调度器: 尝试所有策略, 用cost函数选最优方案.

    策略池:
      1. phase_based: 热冷配对 + expert拆分 + shape切换
      2. greedy_balanced: 贪心负载均衡 @64B/cc
      3. sequential_full: 串行全带宽 @128B/cc
      4. bw_steal: 带宽窃取 — 先结束的cluster抢空闲DMA
      5. adaptive_split: 自适应拆分 — 穷举所有拆分点
      6. online_greedy: 在线贪心 — 每步评估所有可选动作
      7. cold_batch: 冷门批量 — hot并行, cold用128B快速消化
      8. unified_dynamic: 统一动态 — 融合所有策略 + DMA预取 + 专家克隆
      9. event_driven: 事件驱动 — DMA/计算解耦追踪 + 全局makespan优化
     10. event_driven_v22: v22 事件驱动 — 3-tier BW + cache + clone + SPLIT

    cached_map: 跨层缓存映射 {eid: cid} (eid的权重驻留在cluster cid的TCDM中)
    """
    experts = sorted(token_dist.items(), key=lambda x: -x[1])

    strategies = [
        _expert_phases,
        _schedule_greedy_balanced,
        _schedule_sequential_full,
        _schedule_bw_steal,
        _schedule_adaptive_split,
        _schedule_online_greedy,
        _schedule_cold_batch,
        _unified_dynamic_scheduler,
        _event_driven_scheduler,
        _event_driven_v22,
    ]

    best_plan = None
    best_cost = float("inf")

    # [v20] 统一传递cached_map: 每个策略内部调用_preprocess_cached分离缓存命中expert
    for fn in strategies:
        try:
            plan = fn(experts, sys, moe, cached_map=cached_map)
            c = cost_function(plan, shared_cc) if shared_cc > 0 else plan.estimated_cc
            if c < best_cost:
                best_cost = c
                best_plan = plan
        except Exception:
            continue

    # [v20] 计算exit_eids: 遍历tasks找每个cluster最后执行的expert
    # 该expert的权重在执行后仍驻留在TCDM中, 作为下一MoE层的缓存映射
    # exit_eids = {2: eid_last_on_C2, 3: eid_last_on_C3}
    if best_plan and not best_plan.exit_eids:
        for t in best_plan.tasks:
            if t.cid in (2, 3):
                best_plan.exit_eids[t.cid] = t.eid

    return best_plan
