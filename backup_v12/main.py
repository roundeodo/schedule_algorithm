#!/usr/bin/env python3
"""
HeMAiA MoE Performance Model — 主入口
=====================================
config.py → scheduler.py → model.py → Markdown 报告
"""

import math
from config import SystemConfig, MoELayerConfig, generate_shapes
from scheduler import (
    Scheduler,
    zipf_route,
    shared_expert_cost,
    ExpertTask,
    SchedulePlan,
    Phase,
    ScheduleStep,
    gemm_cycles as sched_gemm_cycles,
)
from model import SystemModel, Event

MB = 1024 * 1024


def run_model(sys_cfg, moe_cfg, plan: SchedulePlan, M_total: int):
    """用model执行调度方案, 返回完整仿真结果"""
    model = SystemModel(sys_cfg, moe_cfg)

    # 1. Shared expert (C0+C1 并行, 权重驻留)
    shared_end = model.simulate_shared_expert(M_total)

    # 2. Router (C3)
    routing_ready = model.simulate_router(M_total)

    # 3. 分发token A到C2和C3
    tok_times = model.distribute_tokens(M_total, routing_ready)

    # 4. 执行routed expert调度方案
    for phase in plan.phases:
        for step in phase.steps:
            cid = step.cluster
            earliest = max(tok_times.get(cid, 0), model.res.get(f"C{cid}_VC"))

            if step.resident:
                model.execute_expert(
                    cid,
                    step.eid,
                    step.M,
                    step.shape,
                    0,
                    "none",
                    earliest,
                    resident=True,
                )
            else:
                model.execute_expert(
                    cid,
                    step.eid,
                    step.M,
                    step.shape,
                    step.load_bw,
                    step.dma_channels,
                    earliest,
                )

    return model, shared_end, routing_ready


def generate_markdown(
    model: SystemModel,
    plan: SchedulePlan,
    shared_end: int,
    routing_ready: int,
    M_total: int,
    moe_cfg: MoELayerConfig,
    tasks: list,
) -> str:
    """生成Markdown格式报告"""
    lines = []
    a = lines.append

    makespan = model.get_makespan()
    a(f"# HeMAiA MoE Performance Report (v12)")
    a(f"")
    a(f"## 配置")
    a(f"- **数据类型**: {moe_cfg.dtype_label}")
    a(
        f"- **Hidden**: {moe_cfg.hidden_size}, **Intermediate**: {moe_cfg.moe_intermediate_size}"
    )
    a(f"- **Shared intermediate**: {moe_cfg.shared_intermediate}")
    a(f"- **Routed experts**: {moe_cfg.n_routed_experts}, topK={moe_cfg.topk}")
    a(f"- **单expert权重**: {moe_cfg.expert_total_weight_size/MB:.2f} MB")
    a(f"- **总token数**: M={M_total}")
    a(f"")

    a(f"## 结果概览")
    a(f"| 指标 | 值 |")
    a(f"|---|---|")
    a(f"| **Makespan** | {makespan:,} cc |")
    a(f"| **Shared expert完成** | {shared_end:,} cc |")
    a(f"| **Routing ready** | {routing_ready:,} cc |")
    routed_events = [
        e for e in model.events if any(f"E{t.eid}" in e.name for t in plan.expert_tasks)
    ]
    if routed_events:
        routed_start = min(e.start for e in routed_events)
        routed_end = max(e.end for e in routed_events)
        routed_span = routed_end - routed_start
    else:
        routed_span = 0
    a(f"| **Routed expert span** | {routed_span:,} cc |")
    a(f"| **Routed/Shared** | {routed_span/shared_end:.2%} |" if shared_end > 0 else "")
    a(f"| **调度策略** | {plan.strategy_name} |")
    a(f"")

    a(f"## Expert路由分布")
    a(f"| Expert | Tokens | 占比 |")
    a(f"|---|---|---|")
    total_tok = sum(t.M for t in tasks)
    for t in tasks[:15]:
        a(f"| E{t.eid} | {t.M} | {t.M/total_tok:.1%} |")
    if len(tasks) > 15:
        a(f"| ... | ... | ... |")
    a(f"| **合计** | **{total_tok}** | **{len(tasks)}个expert** |")
    a(f"")

    a(f"## 调度方案详情")
    for i, phase in enumerate(plan.phases):
        a(f"### Phase {i+1}: {phase.desc}")
        a(f"| Cluster | Expert | M | Shape | DMA | BW | 模式 |")
        a(f"|---|---|---|---|---|---|---|")
        for step in phase.steps:
            mode = "resident" if step.resident else "stream"
            a(
                f"| C{step.cluster} | E{step.eid} | {step.M} "
                f"| {step.shape} | {step.dma_channels} | {step.load_bw}B/cc | {mode} |"
            )
        a(f"")

    a(f"## DMA互联拓扑")
    a(f"```")
    a(f"    sram_xDMA (64B/cc)  ←→  C0_xDMA / C1_xDMA / C2_xDMA / C3_xDMA")
    a(f"    iDMA (64B/cc)       →   cluster TCDM (不占xDMA端口)")
    a(f"    sram_xDMA + iDMA 可并行: 总 128B/cc")
    a(f"```")
    a(f"")

    a(f"## TCDM快照")
    a(f"| 时刻 | Cluster | 内容 | 空闲 | 事件 |")
    a(f"|---|---|---|---|---|")
    for snap in model.tcdm_snapshots[:20]:
        contents = ", ".join(f"{k}({v/1024:.0f}KB)" for k, v in snap.contents.items())
        a(
            f"| {snap.time:,} | C{snap.cluster} | {contents} "
            f"| {snap.free_bytes/1024:.0f}KB | {snap.event_desc} |"
        )
    a(f"")

    a(f"## 事件Timeline")
    a(f"| Resource | Start | End | Duration | Event | Detail |")
    a(f"|---|---|---|---|---|---|")
    sorted_events = sorted(model.events, key=lambda e: e.start)
    for ev in sorted_events:
        a(
            f"| {ev.resource} | {ev.start:,} | {ev.end:,} | {ev.duration:,} "
            f"| {ev.name} | {ev.desc} |"
        )
    a(f"")

    # 理想下界
    total_expert_w = sum(t.M for t in tasks) * 0  # not used
    total_w_bytes = len(tasks) * moe_cfg.expert_total_weight_size
    dma_lb = math.ceil(total_w_bytes / 128)  # 128B/cc total
    compute_cc_list = []
    shapes_256 = generate_shapes(256, 8)
    for t in tasks:
        # dual-VC: 每个VC处理 ceil(M/2) 行
        m_per_vc = math.ceil(t.M / 2)
        best_cc = min(
            sched_gemm_cycles(
                m_per_vc, moe_cfg.hidden_size, moe_cfg.moe_intermediate_size, s
            )
            for s in shapes_256
        )
        compute_cc_list.append(best_cc * 3)  # gate+up+down
    compute_lb = sum(compute_cc_list) // 2  # 两个cluster并行

    a(f"## 理想下界分析")
    a(f"| 维度 | 下界 | 实际 | 接近度 |")
    a(f"|---|---|---|---|")
    a(
        f"| DMA搬运 | {dma_lb:,} cc | {routed_span:,} cc | {dma_lb/routed_span:.1%} |"
        if routed_span > 0
        else ""
    )
    a(
        f"| 计算(双cluster) | {compute_lb:,} cc | {routed_span:,} cc | {compute_lb/routed_span:.1%} |"
        if routed_span > 0
        else ""
    )
    a(
        f"| 总下界 | {max(dma_lb,compute_lb):,} cc | {routed_span:,} cc | {max(dma_lb,compute_lb)/routed_span:.1%} |"
        if routed_span > 0
        else ""
    )
    a(f"")

    return "\n".join(lines)


def main():
    sys_cfg = SystemConfig.default_4cluster()
    moe_cfg = MoELayerConfig(weight_dtype_bits=4)  # INT4

    M_total = 64
    tasks = zipf_route(M_total, moe_cfg.n_routed_experts, moe_cfg.topk)
    n_active = len(tasks)
    total_tokens = sum(t.M for t in tasks)

    print(f"Routing: M={M_total}, active={n_active}, total_tokens={total_tokens}")
    print(f"Top: {[(t.eid, t.M) for t in tasks[:10]]}")

    scheduler = Scheduler(sys_cfg, moe_cfg)
    plan = scheduler.generate(tasks)
    print(f"\nBest strategy: {plan.strategy_name} (est={plan.estimated_cc:,} cc)")

    model, shared_end, routing_ready = run_model(sys_cfg, moe_cfg, plan, M_total)
    makespan = model.get_makespan()

    shared_est = shared_expert_cost(M_total, sys_cfg, moe_cfg)
    print(f"Shared:  {shared_end:,} cc (est={shared_est:,})")
    print(f"Routed:  via model simulation")
    print(f"Makespan: {makespan:,} cc")

    md = generate_markdown(
        model, plan, shared_end, routing_ready, M_total, moe_cfg, tasks
    )
    with open("output_v12.md", "w") as f:
        f.write(md)
    print(f"\n报告已写入 output_v12.md")


if __name__ == "__main__":
    main()
