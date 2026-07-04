#!/usr/bin/env python3
"""
HeMAiA MoE Performance Model v17 - 统一动态调度器 + DMA预取 + 专家克隆
========================================================================================

v17关键升级 (在v16基础上):
  - 新增策略8: unified_dynamic — 统一动态调度器
    融合所有7种策略精华 + DMA预取 + 专家克隆 + 动态shape切换
  - 新增策略9: prefetch_aware — DMA预取感知调度器
    利用compute-bound阶段的DMA slack为下一个expert预取权重
  - 三大创新:
    1) DMA预取: hot expert计算期间DMA空闲, 预取cold expert权重
    2) 专家克隆: C2+C3加载同一expert权重, 各算一半token, 2×加速
    3) 动态shape切换: 流式用[4×8×8], 驻留用计算最优shape
  - 底层不变: Bank模型2×(A+B), Per-Tile引擎, W4A8, SPM=1GB

功能:
1. 对每个M值, 遍历/采样大量topK分布
2. 用cost函数搜索最优调度策略 (9种)
3. 提取静态LUT规则
4. 测试动态调度器
5. 生成完整markdown报告 (合并时间行的任务流表, 公式表, 调度决策表)
"""

import math
import sys
import json
import os
from typing import Dict, List, Tuple
from config import (
    SystemConfig,
    MoELayerConfig,
    generate_shapes,
    SHAPES_256,
    VersaCoreShape,
)
from model import SystemModel, Event, gemm_cycles, dma_cc, vc_utilization
from scheduler import (
    schedule,
    generate_all_distributions,
    SchedulePlan,
    estimate_expert_cc,
    best_shape_for,
    cost_function,
    ExpertTask,
)


# ============================================================
#  完整MoE层仿真 (单次)
# ============================================================


def run_simulation(
    M: int,
    token_dist: Dict[int, int],
    sys_cfg: SystemConfig,
    moe_cfg: MoELayerConfig,
    plan: SchedulePlan = None,
) -> dict:
    """运行一次完整MoE层仿真, 返回详细结果"""
    model = SystemModel(sys_cfg, moe_cfg)

    # 1. Shared expert (C0+C1)
    shared_end = model.simulate_shared_expert(M)

    # 2. Router (C3)
    routing_ready = model.simulate_router(M)

    # 3. Token分发 (C2, C3)
    tok_ready = model.distribute_tokens(M, routing_ready)

    # 4. 调度
    if plan is None:
        plan = schedule(M, token_dist, sys_cfg, moe_cfg, shared_end)

    # 5. 执行routed expert
    c2_time, c3_time = 0, 0
    for task in plan.tasks:
        cid = task.cid
        earliest = tok_ready.get(cid, 0)
        start, end = model.execute_routed_expert(
            cid=cid,
            eid=task.eid,
            M=task.ntok,
            shape=task.shape,
            load_bw=task.load_bw,
            dma_channels=task.dma_mode,
            earliest=earliest,
            resident=task.resident,
        )
        if cid == 2:
            c2_time = max(c2_time, end)
        else:
            c3_time = max(c3_time, end)

    routed_end = max(c2_time, c3_time) if (c2_time + c3_time > 0) else 0
    total_mac = sum(c.mac_count for c in sys_cfg.clusters[:2])
    shared_ideal = moe_cfg.shared_ideal_cc(M, total_mac)

    return {
        "M": M,
        "shared_end": shared_end,
        "shared_ideal": shared_ideal,
        "routing_ready": routing_ready,
        "routed_end": routed_end,
        "routed_c2": c2_time,
        "routed_c3": c3_time,
        "makespan": max(shared_end, routed_end),
        "events": model.events,
        "plan": plan,
        "token_dist": token_dist,
        "n_active_experts": len(token_dist),
        "tcdm_snapshots": model.snapshots,
        "ratio": routed_end / shared_end if shared_end > 0 else 0,
    }


# ============================================================
#  批量训练: 对所有分布运行调度器
# ============================================================


def train_all(M: int, sys_cfg: SystemConfig, moe_cfg: MoELayerConfig) -> List[dict]:
    """对M值的所有代表性分布运行调度+仿真, 返回结果列表"""
    distributions = generate_all_distributions(
        M, moe_cfg.topk, moe_cfg.n_routed_experts
    )
    results = []

    for i, dist in enumerate(distributions):
        # 先计算shared_cc用于cost函数
        model_tmp = SystemModel(sys_cfg, moe_cfg)
        shared_cc = model_tmp.simulate_shared_expert(M)

        # 调度
        plan = schedule(M, dist, sys_cfg, moe_cfg, shared_cc)

        # 仿真
        result = run_simulation(M, dist, sys_cfg, moe_cfg, plan)
        result["dist_index"] = i
        results.append(result)

    return results


# ============================================================
#  静态LUT提取
# ============================================================


def classify_distribution(dist: Dict[int, int]) -> str:
    """
    将token分布分类为模式标签:
    - "all_uniform": 所有expert token数相同
    - "hot_dominated": 1-3个expert占>60%
    - "cold_dominated": >80%expert只有1-2 token
    - "mixed": 有热有冷
    - "single": 只有1个expert
    """
    if len(dist) <= 1:
        return "single"

    vals = sorted(dist.values(), reverse=True)
    total = sum(vals)
    n = len(vals)

    if max(vals) == min(vals):
        return "all_uniform"

    # 检查top3占比
    top3 = sum(vals[:3])
    if top3 / total > 0.6:
        return "hot_dominated"

    # 检查冷门比例
    cold_count = sum(1 for v in vals if v <= 2)
    if cold_count / n > 0.8:
        return "cold_dominated"

    return "mixed"


def extract_lut(all_results: Dict[int, List[dict]]) -> Dict:
    """
    从训练结果中提取静态LUT规则.

    LUT结构: {M: {pattern: best_strategy_params}}
    """
    lut = {}

    for M, results in all_results.items():
        lut[M] = {}

        # 按分布模式分组
        groups = {}
        for r in results:
            pattern = classify_distribution(r["token_dist"])
            groups.setdefault(pattern, []).append(r)

        for pattern, group in groups.items():
            # 找该模式下ratio最接近1.0的策略
            best = min(group, key=lambda r: abs(r["ratio"] - 1.0))
            plan = best["plan"]

            # 提取策略参数
            lut[M][pattern] = {
                "strategy": plan.strategy,
                "avg_ratio": sum(r["ratio"] for r in group) / len(group),
                "best_ratio": best["ratio"],
                "worst_ratio": max(r["ratio"] for r in group),
                "n_samples": len(group),
                "avg_vc_util": plan.avg_vc_util,
                "example_dist": best["token_dist"],
                "tasks_summary": [
                    (t.ntok, str(t.shape), t.dma_mode, t.load_bw, t.cid)
                    for t in plan.tasks[:8]
                ],
            }

    return lut


# ============================================================
#  Markdown报告生成
# ============================================================


def _fmt_dist(dist: Dict[int, int], max_show: int = 10) -> str:
    """格式化token分布"""
    vals = sorted(dist.values(), reverse=True)
    n = len(vals)
    if n <= max_show:
        return f"{n}experts: {vals}"
    return f"{n}experts: {vals[:max_show]}...({n-max_show} more)"


def generate_report(
    all_results: Dict[int, List[dict]],
    lut: Dict,
    sys_cfg: SystemConfig,
    moe_cfg: MoELayerConfig,
) -> str:
    """生成完整markdown报告"""
    MB = 1024 * 1024
    lines = []
    lines.append(
        "# HeMAiA MoE Performance Model v17 - 统一动态调度 + DMA预取 + 专家克隆 完整分析报告\n"
    )

    # === 系统配置 ===
    lines.append("## 1. 系统配置\n")
    lines.append("| 参数 | 值 |")
    lines.append("|------|---|")
    lines.append(f"| Hidden Size | {moe_cfg.hidden_size} |")
    lines.append(f"| Shared Intermediate | {moe_cfg.shared_intermediate_size} |")
    lines.append(f"| Routed Intermediate | {moe_cfg.moe_intermediate_size} |")
    lines.append(f"| Routed Experts | {moe_cfg.n_routed_experts} |")
    lines.append(f"| TopK | {moe_cfg.topk} |")
    lines.append(f"| Weight Type | INT{moe_cfg.weight_dtype_bits} |")
    lines.append(f"| C0驻留 | up+half_down = {moe_cfg.c0_resident_size/MB:.3f}MB |")
    lines.append(f"| C1驻留 | gate+half_down = {moe_cfg.c1_resident_size/MB:.3f}MB |")
    lines.append(f"| 单Routed Expert | {moe_cfg.expert_total_weight/MB:.3f}MB |")
    lines.append("")

    lines.append("| Cluster | MAC | VC | TCDM | 用途 |")
    lines.append("|---------|-----|----|----|------|")
    for c in sys_cfg.clusters:
        usage = {
            0: "shared up+half_down",
            1: "shared gate+half_down",
            2: "routed expert",
            3: "routed expert + router",
        }
        lines.append(
            f"| C{c.cluster_id} | {c.mac_count} | {c.num_vc}×{c.vc_mac_count} | "
            f"{c.tcdm_size_bytes/MB:.0f}MB | {usage[c.cluster_id]} |"
        )
    lines.append(
        f"\nSRAM xDMA: {sys_cfg.sram_xdma_bw}B/cc, iDMA: {sys_cfg.idma_bw}B/cc, "
        f"P2P: {sys_cfg.p2p_bw}B/cc\n"
    )

    # === 静态调度策略原理 ===
    lines.append("## 2. 静态调度策略原理\n")
    lines.append("### 2.1 调度目标\n")
    lines.append("- **主目标**: routed expert总执行时间 ≈ shared expert执行时间")
    lines.append(
        "- **次目标**: 所有cluster的VersaCore利用率高, SRAM xDMA+iDMA利用率高\n"
    )
    lines.append("### 2.2 Dual-VC硬件模型 (C2/C3) — v16修正\n")
    lines.append("C2和C3各有2个256MAC VersaCore, 工作模式如下:\n")
    lines.append("**Gate+Up阶段**: VC0=gate_proj, VC1=up_proj, **并行计算**")
    lines.append("  - 每个VC做完整GEMM(M, K=2048, N=1408)")
    lines.append(
        "  - **每个VC独立读A和B** (无broadcast), bank需求 = 2×A_banks + 2×B_banks"
    )
    lines.append("  - 双VC总DMA带宽需求 = 2 × T × C × wpe bytes/cycle")
    lines.append("  - 计算时间 = 单个GEMM时间 (因为并行)\n")
    lines.append("**Down阶段**: N-split, VC0=[M,N,K/2], VC1=[M,N,K/2]")
    lines.append("  - 每个VC独立读A和B切片, bank需求同gate+up")
    lines.append("  - 两个VC输出concat → [M, K=2048]\n")
    lines.append("**Bank模型 (v16)**: `2×A_banks + 2×B_banks`")
    lines.append("  - 每个VC独立占用A端口和B端口, 因此总bank需求是两个VC之和")
    lines.append("  - 示例 [4×8×8]: 2×(4+4) = 16 banks + DMA端口")
    lines.append("  - 示例 [2×8×16]: 2×(2+8) = 20 banks + DMA端口")
    lines.append("  - 示例 [1×8×32]: 2×(1+16) = 34 banks + DMA端口\n")
    lines.append("**Per-Tile流式模型 (v16)**:")
    lines.append("  - GEMM拆分为Mt×Nt output tiles × Kt K-tiles")
    lines.append("  - tile0有DMA延迟暴露, 后续tile以pipeline rate处理")
    lines.append("  - pipeline rate = max(dma_per_tile, compute×bank_stretch)")
    lines.append("  - bank_stretch = (2×A_banks + 2×B_banks + dma_ports) / 64\n")
    lines.append("### 2.3 带宽约束 (W4A8: 权重INT4, 激活INT8)\n")
    lines.append(
        "| Shape [R×T×C] | 单VC B需求 | 双VC B需求 | 2×A+2×B banks | @64B/cc | @128B/cc |"
    )
    lines.append("|------|------|------|------|------|------|")
    for s in generate_shapes(256, 8):
        bd_single = s.tileSize * s.meshCol * 0.5
        bd_dual = 2 * bd_single
        a_banks = s.meshRow
        b_banks = s.meshCol // 2  # INT4: wpe=0.5
        total_banks = 2 * (a_banks + b_banks)
        ok64 = "OK" if bd_dual <= 64 else "不足"
        ok128 = "OK" if bd_dual <= 128 else "不足"
        lines.append(
            f"| {s} | {bd_single:.0f}B/cc | {bd_dual:.0f}B/cc | {total_banks} | {ok64} | {ok128} |"
        )
    lines.append("")
    lines.append(
        "> **关键**: [4×8×8]是并行流式(64B/cc)时的最佳shape — 双VC B需求恰好=64B/cc\n"
    )
    lines.append("### 2.4 Phase-Based调度 (核心策略)\n")
    lines.append(
        "1. **Phase 1 (并行流式)**: 选一对expert, C2用sram_xDMA(64B/cc), C3用iDMA(64B/cc)并行"
    )
    lines.append("   - 双VC用[4×8×8], B需求=64B/cc, 恰好匹配单通道DMA带宽")
    lines.append("   - 可拆分热门expert: 前半段流式, 后半段驻留计算")
    lines.append("2. **Phase 2 (驻留+全BW)**: 热门expert权重已驻留, 无需DMA")
    lines.append("   - 空闲cluster独享128B/cc, 可用[2×8×16] (双VC B需求=128B/cc)")
    lines.append("3. **Phase 3 (清理)**: 处理剩余冷门expert (1-2 token)\n")

    # === 动态调度器原理 ===
    lines.append("## 3. 动态调度器原理\n")
    lines.append("动态调度器基于cost函数, 在runtime根据TopK结果选择最优调度方案.\n")
    lines.append("### Cost函数\n")
    lines.append("```")
    lines.append("cost = |routed_cc/shared_cc - 1.0|  // 主: 时间接近")
    lines.append("     + (1 - avg_vc_util) × 0.2     // 次: VC利用率")
    lines.append("     + (1 - avg_sram_util) × 0.1   // 次: SRAM带宽")
    lines.append("```\n")
    lines.append("策略搜索池 (9种):\n")
    lines.append("1. **phase_based**: 热冷配对 + expert拆分 + 驻留phase")
    lines.append("2. **greedy_balanced**: 贪心负载均衡 @64B/cc")
    lines.append("3. **sequential_full**: 串行全带宽 @128B/cc")
    lines.append("4. **bw_steal**: 带宽窃取 — 先结束的cluster抢空闲DMA")
    lines.append("5. **adaptive_split**: 自适应拆分 — 穷举所有拆分点")
    lines.append("6. **online_greedy**: 在线贪心 — 每步评估所有可选动作")
    lines.append("7. **cold_batch**: 冷门批量 — hot并行, cold用128B快速消化")
    lines.append(
        "8. **unified_dynamic**: 统一动态 — 融合所有策略 + DMA预取 + 专家克隆 + shape切换"
    )
    lines.append(
        "9. **prefetch_aware**: DMA预取感知 — 利用compute-bound的DMA slack预取下一个expert\n"
    )
    lines.append("")
    lines.append("### v17创新机制\n")
    lines.append("**DMA预取 (Prefetch)**:")
    lines.append("- 当hot expert在cluster_A计算时, DMA通道空闲(compute > DMA)")
    lines.append("- 利用空闲DMA为cluster_B预取下一个cold expert的权重")
    lines.append(
        "- M=8@[4×8×8]: DMA slack=68,202cc → 可预取4.163MB (恰好一整个expert!)"
    )
    lines.append("- M≥16: slack更大, 可以连续预取多个expert\n")
    lines.append("**专家克隆 (Expert Clone)**:")
    lines.append("- 当只有1-2个active expert且token极多时, C2+C3各加载同一份权重")
    lines.append("- 每个cluster处理一半的token, 实现2×计算加速")
    lines.append("- 条件: compute_time >> 2×dma_time (加载两份权重仍比单cluster快)\n")

    # === 训练结果汇总 ===
    lines.append("## 4. 训练结果汇总\n")
    for M in sorted(all_results.keys()):
        results = all_results[M]
        shared_cc = results[0]["shared_end"]
        ratios = [r["ratio"] for r in results]

        lines.append(f"### M = {M} ({len(results)}种分布)\n")
        lines.append(
            f"- Shared expert: {shared_cc:,} cc (ideal: {results[0]['shared_ideal']:,})"
        )
        lines.append(f"- Routed ratio 范围: [{min(ratios):.3f}, {max(ratios):.3f}]")
        lines.append(f"- 平均ratio: {sum(ratios)/len(ratios):.3f}")
        lines.append(
            f"- ratio ≤ 1.1的比例: {sum(1 for r in ratios if r <= 1.1)/len(ratios):.1%}"
        )
        lines.append(
            f"- ratio ≤ 1.2的比例: {sum(1 for r in ratios if r <= 1.2)/len(ratios):.1%}\n"
        )

        # 按模式分组统计
        lines.append(
            "| 分布模式 | 样本数 | 平均ratio | 最优ratio | 最差ratio | 平均VC利用率 |"
        )
        lines.append(
            "|---------|-------|----------|----------|----------|------------|"
        )
        groups = {}
        for r in results:
            pat = classify_distribution(r["token_dist"])
            groups.setdefault(pat, []).append(r)
        for pat, grp in sorted(groups.items()):
            rats = [r["ratio"] for r in grp]
            vcs = [r["plan"].avg_vc_util for r in grp]
            lines.append(
                f"| {pat} | {len(grp)} | {sum(rats)/len(rats):.3f} | "
                f"{min(rats):.3f} | {max(rats):.3f} | {sum(vcs)/len(vcs):.1%} |"
            )
        lines.append("")

        # 最优和最差案例
        best = min(results, key=lambda r: abs(r["ratio"] - 1.0))
        worst = max(results, key=lambda r: r["ratio"])
        lines.append(
            f"**最优案例**: ratio={best['ratio']:.3f}, "
            f"dist={_fmt_dist(best['token_dist'])}, "
            f"strategy={best['plan'].strategy}\n"
        )
        lines.append(
            f"**最差案例**: ratio={worst['ratio']:.3f}, "
            f"dist={_fmt_dist(worst['token_dist'])}, "
            f"strategy={worst['plan'].strategy}\n"
        )

    # === 静态LUT ===
    lines.append("## 5. 静态LUT (查找表)\n")
    lines.append(
        "| M | 分布模式 | 推荐策略 | 平均ratio | 最优ratio | 最差ratio | VC利用率 |"
    )
    lines.append("|---|---------|---------|----------|----------|----------|---------|")
    for M in sorted(lut.keys()):
        for pat, info in sorted(lut[M].items()):
            lines.append(
                f"| {M} | {pat} | {info['strategy']} | "
                f"{info['avg_ratio']:.3f} | {info['best_ratio']:.3f} | "
                f"{info['worst_ratio']:.3f} | {info['avg_vc_util']:.1%} |"
            )
    lines.append("")

    # === 详细任务流表 (对每个M选多个代表性分布) ===
    lines.append("## 6. 详细任务流表\n")
    for M in sorted(all_results.keys()):
        results = all_results[M]
        # 选ratio最接近1.0的 (最优)
        best = min(results, key=lambda r: abs(r["ratio"] - 1.0))
        # 选ratio最大的 (最差)
        worst = max(results, key=lambda r: r["ratio"])
        # 选一个中间案例 (hot_dominated或mixed且ratio在中位数附近)
        sorted_by_ratio = sorted(results, key=lambda r: r["ratio"])
        mid_idx = len(sorted_by_ratio) // 2
        mid = sorted_by_ratio[mid_idx]
        # 去重
        seen_sigs = set()
        for rep, label in [(best, "最优"), (mid, "中位"), (worst, "最差")]:
            sig = tuple(sorted(rep["token_dist"].values(), reverse=True))
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            lines.append(f"---\n")
            lines.append(f"**M={M} {label}案例** (ratio={rep['ratio']:.3f})\n")
            _write_task_flow_table(lines, rep, sys_cfg)
            _write_duration_formula_table(lines, rep)
            _write_scheduling_rationale_table(lines, rep, sys_cfg, moe_cfg)

    # === 分析总结 ===
    lines.append("## 7. 分析与结论\n")
    lines.append("### 7.1 DMA带宽瓶颈分析\n")
    K = moe_cfg.hidden_size
    N = moe_cfg.moe_intermediate_size
    wpe = moe_cfg.wpe
    expert_w = 3 * K * N * wpe
    dma_per_expert_64 = dma_cc(int(expert_w), 64)
    dma_per_expert_128 = dma_cc(int(expert_w), 128)
    lines.append(
        f"- 单个routed expert权重: gate+up+down = 3×{K}×{N}×{wpe} = {expert_w/(1024*1024):.3f}MB"
    )
    lines.append(f"- @64B/cc搬运时间: {dma_per_expert_64:,}cc (一对多并行搬运时)")
    lines.append(f"- @128B/cc搬运时间: {dma_per_expert_128:,}cc (独享xDMA+iDMA时)")
    lines.append("")

    lines.append("### 7.2 Per-Shape双VC带宽需求分析\n")
    lines.append(
        "| Shape [R×T×C] | 单VC B需求 | 双VC B需求 | K-tile(T cc) | DMA/tile@64B | DMA/tile@128B | @64B stall | @128B stall |"
    )
    lines.append("|------|------|------|------|------|------|------|------|")
    for s in generate_shapes(256, 8):
        bd_single = s.tileSize * s.meshCol * wpe
        bd_dual = 2 * bd_single
        T = s.tileSize
        dma64 = math.ceil(bd_dual / 64)
        dma128 = math.ceil(bd_dual / 128)
        stall64 = "YES" if dma64 > T else "NO"
        stall128 = "YES" if dma128 > T else "NO"
        lines.append(
            f"| {s} | {bd_single:.0f}B | {bd_dual:.0f}B | {T}cc | {dma64}cc | {dma128}cc | {stall64} | {stall128} |"
        )
    lines.append("")
    lines.append("> 注: 双VC B需求 = 2 × T × C × wpe (两个VC各自独立读A和B)")
    lines.append("> Bank需求 = 2×A_banks + 2×B_banks (无broadcast)")
    lines.append("> [4×8×8]双VC B=64B/cc, bank=16, 恰好匹配单通道DMA(64B/cc)")
    lines.append("> [2×8×16]双VC B=128B/cc, bank=20, 需要xDMA+iDMA同时工作(128B/cc)")
    lines.append(
        "> [1×8×32]双VC B=256B/cc, bank=34, 即使128B/cc也不够, DMA-bound不可避免\n"
    )

    lines.append("### 7.3 Routed vs Shared 时间比分析 (正确Dual-VC模型)\n")
    lines.append("并行调度下(C2=xDMA@64B/cc, C3=iDMA@64B/cc), 单expert时间:\n")
    lines.append(
        "| M | gu_compute | dn_compute | gu_dma | dn_dma | stream_total | shared_cc | Ratio理论 |"
    )
    lines.append(
        "|---|-----------|-----------|--------|--------|-------------|----------|----------|"
    )
    for M in [1, 4, 8, 16, 64, 128]:
        # 用[4x8x8]@64B/cc (并行64B/cc最佳shape)
        shape_48 = VersaCoreShape(4, 8, 8)
        shape_18x = VersaCoreShape(1, 8, 32)
        # dual-VC: 每个VC做完整GEMM(M,K,N)
        gu_48 = gemm_cycles(M, K, N, shape_48)
        dn_48 = gemm_cycles(M, N, K // 2, shape_48)
        sw = math.ceil(M * N / 128)
        gu_dma = dma_cc(int(2 * K * N * wpe), 64)
        dn_dma = dma_cc(int(N * K * wpe), 64)
        # streaming总时间
        gu_first = dma_cc(math.ceil(2 * 8 * N * wpe / 64) * 64, 64)
        gu_total = max(gu_dma, gu_first + gu_48)
        dn_first = dma_cc(math.ceil(2 * 8 * (K // 2) * wpe / 64) * 64, 64)
        dn_total = max(dn_dma, dn_first + dn_48)
        stream_total = gu_total + sw + dn_total
        model_tmp = SystemModel(sys_cfg, moe_cfg)
        shared_cc_m = model_tmp.simulate_shared_expert(M)
        ratio_theory = stream_total / shared_cc_m if shared_cc_m > 0 else 0
        lines.append(
            f"| {M} | {gu_48:,} | {dn_48:,} | {gu_dma:,} | {dn_dma:,} | "
            f"{stream_total:,} | {shared_cc_m:,} | {ratio_theory:.3f} |"
        )
    lines.append("")
    lines.append("> 使用[4×8×8]@64B/cc, 双VC B需求=64B/cc, 恰好匹配单通道DMA。")
    lines.append(
        "> M=1时 gu_compute=45,237cc vs gu_dma=45,056cc → compute-bound (刚好平衡)。"
    )
    lines.append("> M≥4时 compute远超DMA → compute-bound, DMA完全overlap。")
    lines.append("> 两个expert并行后, 理论ratio≈stream_total/shared_cc。\n")

    lines.append("### 7.4 关键发现\n")
    for M in sorted(all_results.keys()):
        results = all_results[M]
        shared = results[0]["shared_end"]
        avg_ratio = sum(r["ratio"] for r in results) / len(results)
        min_ratio = min(r["ratio"] for r in results)
        max_ratio = max(r["ratio"] for r in results)
        pct_le_11 = sum(1 for r in results if r["ratio"] <= 1.1) / len(results)
        pct_le_15 = sum(1 for r in results if r["ratio"] <= 1.5) / len(results)
        lines.append(
            f"- **M={M}**: shared={shared:,}cc, "
            f"ratio=[{min_ratio:.3f}, {max_ratio:.3f}], avg={avg_ratio:.3f}, "
            f"≤1.1: {pct_le_11:.1%}, ≤1.5: {pct_le_15:.1%}"
        )

    lines.append("")
    lines.append("### 7.5 调度策略效果分析\n")
    lines.append(
        "- **phase_based**: 对token集中的分布(2-4 active experts)最有效, "
        "可以充分利用xDMA+iDMA并行, ratio接近1.0"
    )
    lines.append(
        "- **greedy_balanced**: 对token分散的分布(多个cold expert)较好, "
        "负载均衡减少最长路径"
    )
    lines.append(
        "- **sequential_full**: 仅在极端情况(1个expert独占)有优势, " "独享128B/cc带宽"
    )
    lines.append(
        "- **bw_steal**: 利用先完成的cluster释放的DMA通道, " "特别适合一热一冷组合场景"
    )
    lines.append(
        "- **adaptive_split**: 穷举热门expert的拆分点, "
        "将一个热门拆分到两个cluster并行, 适合1热多冷"
    )
    lines.append(
        "- **online_greedy**: 真正的在线动态调度, 每步视野最优, "
        "对复杂分布(多种token数混合)效果最好"
    )
    lines.append(
        "- **cold_batch**: 先并行@64处理热门, 然后批量@128消化冷门, "
        "适合冷门expert特别多的分布"
    )
    lines.append(
        "- **核心限制**: 当active expert数 >> 2时, DMA带宽是不可避免的瓶颈。"
        "每对expert需要~67,584cc DMA时间, 而shared expert的计算也只有~M×17,000cc。"
        "n_pair = ceil(n_active/2), 串行DMA总时间 = n_pair × 67,584cc。"
    )

    return "\n".join(lines)


def _write_task_flow_table(lines: List[str], result: dict, sys_cfg: SystemConfig):
    """
    生成任务流表格 — v16: 同一时间段的事件合并到同一行.

    按时间段分组: 将start/end完全重叠或有部分重叠的事件归到同一行,
    每个资源列显示该时间段内该资源正在执行的任务.
    """
    M = result["M"]
    events = result["events"]
    if not events:
        return

    lines.append(f"### M={M} 任务流表 (dist: {_fmt_dist(result['token_dist'])})\n")

    # 收集所有资源
    resources = sorted(set(ev.resource for ev in events))

    # 收集所有时间边界点, 按时间段分组
    time_points = set()
    for ev in events:
        time_points.add(ev.start)
        time_points.add(ev.end)
    time_points = sorted(time_points)

    # 生成时间段
    intervals = []
    for i in range(len(time_points) - 1):
        t_start = time_points[i]
        t_end = time_points[i + 1]
        if t_start == t_end:
            continue
        # 找在这个区间内活跃的事件 (ev.start <= t_start and ev.end >= t_end)
        active = {}
        for ev in events:
            if ev.start <= t_start and ev.end >= t_end:
                active[ev.resource] = ev.name[:30]
        if active:
            intervals.append((t_start, t_end, active))

    # 合并相邻区间内活跃事件完全相同的行
    merged = []
    for t_start, t_end, active in intervals:
        if merged and merged[-1][2] == active:
            merged[-1] = (merged[-1][0], t_end, active)
        else:
            merged.append((t_start, t_end, active))

    # 表头
    hdr = "| Start | End | Dur |"
    for r in resources:
        hdr += f" {r} |"
    lines.append(hdr)
    lines.append("|" + "---|" * (3 + len(resources)))

    for t_start, t_end, active in merged:
        dur = t_end - t_start
        row = f"| {t_start:,} | {t_end:,} | {dur:,} |"
        for r in resources:
            row += f" {active.get(r, '')} |"
        lines.append(row)
    lines.append("")

    # TCDM快照
    if result["tcdm_snapshots"]:
        lines.append(f"#### TCDM状态 (M={M})\n")
        lines.append("| 时刻 | Cluster | 内容 | 已用 | 剩余 |")
        lines.append("|------|---------|------|------|------|")
        MB = 1024 * 1024
        for s in result["tcdm_snapshots"]:
            contents = ", ".join(f"{k}:{v/MB:.3f}MB" for k, v in s.contents.items())
            lines.append(
                f"| {s.time:,} | C{s.cluster} | {contents} | "
                f"{s.used_bytes/MB:.3f}MB | {s.free_bytes/MB:.3f}MB |"
            )
        lines.append("")


def _write_duration_formula_table(lines: List[str], result: dict):
    """持续时间公式表"""
    M = result["M"]
    events = result["events"]
    if not events:
        return

    lines.append(f"#### 持续时间公式表 (M={M})\n")
    lines.append("| # | Task | Resource | Start | End | Duration | Formula |")
    lines.append("|---|------|----------|-------|-----|----------|---------|")
    for i, ev in enumerate(sorted(events, key=lambda e: e.start)):
        lines.append(
            f"| {i} | {ev.name[:40]} | {ev.resource} | {ev.start:,} | "
            f"{ev.end:,} | {ev.duration:,} | {ev.formula if ev.formula else '-'} |"
        )
    lines.append("")


def _write_scheduling_rationale_table(
    lines: List[str], result: dict, sys_cfg: SystemConfig, moe_cfg: MoELayerConfig
):
    """调度决策表"""
    M = result["M"]
    plan = result["plan"]

    lines.append(f"#### 调度决策表 (M={M}, 策略={plan.strategy})\n")
    lines.append(f"- Token分布: {_fmt_dist(result['token_dist'])}")
    lines.append(
        f"- Routed CC: {result['routed_end']:,}, Shared CC: {result['shared_end']:,}, "
        f"Ratio: {result['ratio']:.3f}"
    )
    lines.append(
        f"- VC利用率: {plan.avg_vc_util:.1%}, "
        f"xDMA利用率: {plan.sram_xdma_util:.1%}, "
        f"iDMA利用率: {plan.sram_idma_util:.1%}\n"
    )

    lines.append(
        "| Expert | Tokens | Cluster | Shape | DMA | BW | Phase | VC利用率 | Est.CC | 决策理由 |"
    )
    lines.append(
        "|--------|--------|---------|-------|-----|-----|-------|---------|--------|---------|"
    )

    K = moe_cfg.hidden_size
    N = moe_cfg.moe_intermediate_size
    wpe = moe_cfg.wpe

    for t in plan.tasks:
        M_vpc = math.ceil(t.ntok / sys_cfg.clusters[t.cid].num_vc)
        lines.append(
            f"| E{t.eid} | {t.ntok} | C{t.cid} | {t.shape} | {t.dma_mode} | "
            f"{t.load_bw} | {t.phase} | {t.vc_util:.0%} | {t.estimated_cc:,} | "
            f"{t.rationale[:50]} |"
        )
    lines.append("")


# ============================================================
#  Main
# ============================================================


def main():
    sys_cfg = SystemConfig.default_4cluster()
    moe_cfg = MoELayerConfig()
    MB = 1024 * 1024

    print("=" * 80)
    print(
        "HeMAiA MoE Performance Model v17 (Unified Dynamic + DMA Prefetch + Expert Clone)"
    )
    print("=" * 80)
    print(moe_cfg.summary())

    # 驻留检查
    print("=== 驻留检查 ===")
    print(f"  C0 (up+half_down): {moe_cfg.c0_resident_size/MB:.3f}MB / 5MB")
    print(f"  C1 (gate+half_down): {moe_cfg.c1_resident_size/MB:.3f}MB / 5MB")
    print(f"  单routed expert: {moe_cfg.expert_total_weight/MB:.3f}MB / 5MB")
    print()

    M_values = [1, 4, 8, 16, 64, 128]
    all_results = {}

    for M in M_values:
        print(f"\n{'#'*60}")
        print(f"# Training M = {M}")
        print(f"{'#'*60}")

        dists = generate_all_distributions(M, moe_cfg.topk, moe_cfg.n_routed_experts)
        print(f"  生成了 {len(dists)} 种token分布")

        results = train_all(M, sys_cfg, moe_cfg)
        all_results[M] = results

        # 汇总
        ratios = [r["ratio"] for r in results]
        shared = results[0]["shared_end"]
        print(f"  Shared: {shared:,} cc")
        print(
            f"  Ratio: min={min(ratios):.3f} avg={sum(ratios)/len(ratios):.3f} "
            f"max={max(ratios):.3f}"
        )
        print(
            f"  ≤1.1: {sum(1 for r in ratios if r<=1.1)/len(ratios):.1%}  "
            f"≤1.5: {sum(1 for r in ratios if r<=1.5)/len(ratios):.1%}  "
            f"≤2.0: {sum(1 for r in ratios if r<=2.0)/len(ratios):.1%}"
        )

        # 按模式统计
        groups = {}
        for r in results:
            pat = classify_distribution(r["token_dist"])
            groups.setdefault(pat, []).append(r)
        for pat, grp in sorted(groups.items()):
            rats = [r["ratio"] for r in grp]
            print(
                f"    {pat:20s}: n={len(grp):3d} avg={sum(rats)/len(rats):.3f} "
                f"range=[{min(rats):.3f}, {max(rats):.3f}]"
            )

    # 提取LUT
    print("\n" + "=" * 60)
    print("提取静态LUT...")
    lut = extract_lut(all_results)
    for M in sorted(lut.keys()):
        print(f"\n  M={M}:")
        for pat, info in sorted(lut[M].items()):
            print(
                f"    {pat:20s}: strategy={info['strategy']:20s} "
                f"ratio=[{info['best_ratio']:.3f}, {info['worst_ratio']:.3f}] "
                f"n={info['n_samples']}"
            )

    # 保存LUT为JSON
    lut_json = {}
    for M in lut:
        lut_json[str(M)] = {}
        for pat, info in lut[M].items():
            lut_json[str(M)][pat] = {
                k: v for k, v in info.items() if k != "tasks_summary"
            }
            lut_json[str(M)][pat]["example_dist"] = {
                str(k): v for k, v in info["example_dist"].items()
            }

    with open("static_lut.json", "w") as f:
        json.dump(lut_json, f, indent=2, ensure_ascii=False)
    print("\n静态LUT已保存到 static_lut.json")

    # 生成完整报告
    print("\n生成markdown报告...")
    report = generate_report(all_results, lut, sys_cfg, moe_cfg)
    with open("v17_report.md", "w") as f:
        f.write(report)
    print(f"报告已保存到 v17_report.md ({len(report)} chars)")

    # === 汇总表 ===
    print("\n" + "=" * 120)
    print("全局汇总")
    print("=" * 120)
    print(
        f"{'M':>4} | {'Shared':>10} | {'#Dist':>5} | {'Avg Ratio':>9} | "
        f"{'Min Ratio':>9} | {'Max Ratio':>9} | {'≤1.1':>6} | {'≤1.5':>6} | {'≤2.0':>6}"
    )
    print("-" * 100)
    for M in M_values:
        results = all_results[M]
        ratios = [r["ratio"] for r in results]
        n = len(ratios)
        print(
            f"{M:>4} | {results[0]['shared_end']:>10,} | {n:>5} | "
            f"{sum(ratios)/n:>9.3f} | {min(ratios):>9.3f} | {max(ratios):>9.3f} | "
            f"{sum(1 for r in ratios if r<=1.1)/n:>6.1%} | "
            f"{sum(1 for r in ratios if r<=1.5)/n:>6.1%} | "
            f"{sum(1 for r in ratios if r<=2.0)/n:>6.1%}"
        )
    print("=" * 120)


if __name__ == "__main__":
    main()
