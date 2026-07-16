#!/usr/bin/env python3
"""
Bingo Trace Timeline Analyzer
==============================
直接从 bingo_trace.json 提取真实 RTL 仿真时间线。
bingo_trace.json 中的时间戳来自 RTL $time，是绝对真实的仿真器时间 (ns)。

输出:
  1. 跨核统一时间线 (所有核心的任务按真实开始时间排序)
  2. 各核心独立时间线 (含任务间 gap)
  3. 核心利用率统计
  4. 调度开销分析 (GET_READY 等待时间)
"""

import argparse
import json
import sys
import os
import re
from collections import Counter, defaultdict

import verify_deps

# ─── 默认路径 ───
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRACE = os.path.join(_SCRIPT_DIR, "HeMAiA/target/sim/bin/logs/bingo_trace.json")
DEFAULT_UART_LOG = os.path.join(_SCRIPT_DIR, "HeMAiA/target/sim/bin/uart_chip_0_0.log")
DEFAULT_WORKLOAD_DIR = verify_deps.DEFAULT_WORKLOAD_DIR
_WORKLOADS_BASE = os.path.join(
    _SCRIPT_DIR,
    "HeMAiA/target/sw/host/apps/offload_bingo_hw/single_chip/workloads",
)


def detect_workload_from_uart(uart_log_path: str) -> str | None:
    """从 uart log 中提取 workload 名称，返回 workload 目录路径；找不到则返回 None。"""
    if not os.path.exists(uart_log_path):
        return None
    pattern = re.compile(r"\[Host\] Preparing ([\w]+) Workload")
    try:
        with open(uart_log_path) as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    name = m.group(1)
                    candidate = os.path.join(_WORKLOADS_BASE, name)
                    if os.path.isdir(candidate):
                        return candidate
    except OSError:
        pass
    return None


def parse_bingo_trace(trace_path):
    """解析 bingo_trace.json，返回所有 duration 事件 (ph='X')。"""
    with open(trace_path) as f:
        data = json.load(f)
    events = []
    for e in data["traceEvents"]:
        if e.get("ph") != "X":
            continue
        args = e.get("args", {})
        events.append(
            {
                "name": e["name"],
                "tid": e["tid"],
                "ts": e["ts"],  # start_ns (绝对仿真时间)
                "dur": e["dur"],  # duration_ns
                "end": e["ts"] + e["dur"],
                "kernel_type": args.get("kernel_type", ""),
                "dur_cc": args.get("dur_cc", 0),
                "freq_MHz": args.get("freq_MHz", 0),
            }
        )
    return events


def extract_phase_events(events):
    """从 duration 事件中提取 PHASE_* 级别事件，用于 phase 切换分析。"""
    phase_names = {
        "BINGO_TRACE_PHASE_DECISION": "PhaseDecision",
        "BINGO_TRACE_PHASE_SETUP": "PhaseSetup",
        "BINGO_TRACE_PHASE_SCATTER": "PhaseScatter",
    }
    phase_events = []
    for e in events:
        label = phase_names.get(e["name"])
        if label:
            phase_events.append(
                {
                    "tid": e["tid"],
                    "label": label,
                    "start_ns": e["ts"],
                    "end_ns": e["end"],
                    "dur_ns": e["dur"],
                    "dur_cc": e["dur_cc"],
                    "freq_MHz": e["freq_MHz"],
                }
            )
    return sorted(phase_events, key=lambda x: x["start_ns"])


def classify_kernel(sub_events):
    """根据 RUN_KERNEL 内的子事件判断 kernel 类型。"""
    names = {e["name"] for e in sub_events}
    # Host compute kernels (unique markers from 0x4xx)
    if "BINGO_TRACE_HOST_ENTRY" in names:
        return "Entry"
    if "BINGO_TRACE_HOST_EXIT" in names:
        return "Exit"
    if "BINGO_TRACE_HOST_ROUTER_SCHED" in names:
        return "RouterSched"
    if "BINGO_TRACE_HOST_EXPERT_DISPATCH_SW" in names:
        return "DispatchSW"
    if "BINGO_TRACE_HOST_EXPERT_DISPATCH_HW" in names:
        return "DispatchHW"
    if "BINGO_TRACE_HOST_EXPERT_DISPATCH_CERF" in names:
        return "DispatchCERF"
    if "BINGO_TRACE_HOST_MOE_PREPARE" in names:
        return "MoEPrepare"
    if "BINGO_TRACE_HOST_MOE_EXECUTE" in names:
        return "MoEExecute"
    if "BINGO_TRACE_HOST_SOFTMAX" in names:
        return "Softmax"
    if "BINGO_TRACE_HOST_SCATTER_META" in names:
        return "ScatterMeta"
    if "BINGO_TRACE_HOST_SWISH" in names:
        return "Swish"
    if "BINGO_TRACE_HOST_GLU" in names:
        return "GLU"
    if "BINGO_TRACE_HOST_ACCUMULATE" in names:
        return "Accumulate"
    if "BINGO_TRACE_HOST_SCATTER_PAD" in names:
        return "ScatterPad"
    if "BINGO_TRACE_HOST_CHECK_RESULT" in names:
        return "CheckResult"
    # Device MoE dynamic expert per-kernel markers (0x382-0x395)
    # Check _FULL variants before the plain _COMPUTE variants to avoid prefix collision
    if "BINGO_TRACE_DEV_MOE_GATHER_S1" in names:
        return "gather_s1"
    if "BINGO_TRACE_DEV_MOE_LOAD_GATE_UP" in names:
        return "load_gate_up_block"
    if "BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP_FULL" in names:
        return "compute_gate_up_full"
    if "BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP" in names:
        return "compute_gate_up_block"
    if "BINGO_TRACE_DEV_MOE_LOAD_DOWN" in names:
        return "load_down_block"
    if "BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_FULL" in names:
        return "compute_down_full"
    if "BINGO_TRACE_DEV_MOE_COMPUTE_DOWN" in names:
        return "compute_down_block"
    if "BINGO_TRACE_DEV_MOE_PREFETCH_S2" in names:
        return "prefetch_s2_down"
    if "BINGO_TRACE_DEV_MOE_PREFETCH_S4" in names:
        return "prefetch_s4_next_s1"
    if "BINGO_TRACE_DEV_MOE_STORE" in names:
        return "store"
    if "BINGO_TRACE_MOE_OUTPUT_PADDING_INIT" in names:
        return "PaddingInit"
    if "BINGO_TRACE_COMPLETION_RELAY" in names:
        return "CompletionRelay"
    # Device accelerator kernels — L15 MoE kernels (checked before generic GEMM_FULL)
    if "BINGO_TRACE_L15_FULL_CFG" in names:
        return "L15_Full"
    if "BINGO_TRACE_L15_SWIGLU_CFG" in names:
        return "L15_SwiGLU"
    if "BINGO_TRACE_L15_DOWN_CFG" in names:
        return "L15_Down"
    if "BINGO_TRACE_GEMM_FULL_CFG" in names:
        return "GEMM_Full"
    if "BINGO_TRACE_GEMM_MIN_CFG" in names:
        return "GEMM_Min"
    if "BINGO_TRACE_HOST_IDMA_CFG" in names:
        return "Host_DMA"
    # Dual DMA: dedicated outer marker takes priority over individual IDMA/XDMA markers
    if "BINGO_TRACE_DUAL_DMA_CFG" in names:
        return "Dual_DMA"
    if "BINGO_TRACE_IDMA_CFG" in names and "BINGO_TRACE_XDMA_CFG" in names:
        return "Dual_DMA"
    if "BINGO_TRACE_IDMA_CFG" in names:
        return "Dev_DMA"
    if "BINGO_TRACE_XDMA_CFG" in names:
        return "XDMA"
    if "BINGO_TRACE_DUMMY_KERNEL" in names:
        return "Dummy"
    return "Unknown"


def build_task_cycles(events):
    """
    把每个核心的事件序列组织成 task cycle:
      GET_READY → PREP → RUN_KERNEL (含子事件)

    返回 list of task dicts。
    """
    cores = {}
    for e in events:
        cores.setdefault(e["tid"], []).append(e)

    tasks = []
    for tid in sorted(cores.keys()):
        core_events = sorted(cores[tid], key=lambda x: x["ts"])

        # 提取 RUN_KERNEL 事件作为 task 的主体
        run_kernels = [
            e for e in core_events if e["name"] == "BINGO_TRACE_MGR_RUN_KERNEL"
        ]

        # 提取 GET_READY 和 PREP 事件
        get_readys = [
            e for e in core_events if e["name"] == "BINGO_TRACE_MGR_GET_READY"
        ]
        preps = [e for e in core_events if e["name"] == "BINGO_TRACE_MGR_PREP"]

        # 提取除 GET_READY / PREP / RUN_KERNEL 以外的所有事件作为子事件候选
        sub_candidates = [
            e
            for e in core_events
            if e["name"]
            not in (
                "BINGO_TRACE_MGR_GET_READY",
                "BINGO_TRACE_MGR_PREP",
                "BINGO_TRACE_MGR_RUN_KERNEL",
            )
        ]

        for seq_idx, rk in enumerate(run_kernels):
            # 找到该 RUN_KERNEL 时间范围内的所有子事件
            subs = [
                s
                for s in sub_candidates
                if s["ts"] >= rk["ts"] and s["end"] <= rk["end"]
            ]
            kernel_name = classify_kernel(subs)

            # 找最近的前一个 GET_READY 和 PREP
            prev_gr = None
            for gr in reversed(get_readys):
                if gr["end"] <= rk["ts"] + 100:  # 允许小误差
                    prev_gr = gr
                    break
            prev_prep = None
            for p in reversed(preps):
                if p["end"] <= rk["ts"] + 100:
                    prev_prep = p
                    break

            task = {
                "tid": tid,
                "seq": seq_idx + 1,
                "kernel": kernel_name,
                "marker_class": kernel_name,
                # RUN_KERNEL 时间 (任务真正执行的区间)
                "exec_start_ns": rk["ts"],
                "exec_end_ns": rk["end"],
                "exec_dur_ns": rk["dur"],
                "exec_dur_cc": rk["dur_cc"],
                "exec_freq_MHz": rk["freq_MHz"],
                # 调度开销
                "wait_ns": prev_gr["dur"] if prev_gr else 0,
                "prep_ns": prev_prep["dur"] if prev_prep else 0,
                # 整个 task cycle (从 GET_READY 开始到 RUN_KERNEL 结束)
                "cycle_start_ns": prev_gr["ts"] if prev_gr else rk["ts"],
                "cycle_end_ns": rk["end"],
                # 子事件列表
                "sub_events": subs,
            }
            tasks.append(task)

    return tasks


def parse_trace_tid_location(tid, base_hart_id=1, cores_per_cluster=2):
    """Map Perfetto tid string to static (cluster, core) when possible."""
    if tid == "Host Core":
        return (0, 2)
    # bingo_trace.py emits tid as "Cluster X Core Y" — parse directly
    match = re.match(r"Cluster\s+(\d+)\s+Core\s+(\d+)$", tid)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    # Fallback: legacy "Core N" hart-ID format
    match = re.match(r"Core\s+(\d+)$", tid)
    if not match:
        return None
    hart_id = int(match.group(1))
    local_id = hart_id - base_hart_id
    if local_id < 0:
        return None
    return (local_id // cores_per_cluster, local_id % cores_per_cluster)


def load_static_mapping(workload_dir):
    """Load generated DFG/header data used to infer node labels for trace tasks."""
    nodes, descriptors, dev_map = verify_deps.load_artifacts(workload_dir)
    errors, warnings = verify_deps.validate_artifacts(nodes, descriptors, dev_map)
    streams = verify_deps.build_expected_device_streams(nodes, descriptors, dev_map)
    return {
        "nodes": nodes,
        "descriptors": descriptors,
        "dev_map": dev_map,
        "streams": streams,
        "errors": errors,
        "warnings": warnings,
        "workload_dir": workload_dir,
    }


def annotate_tasks_with_static_nodes(
    tasks, static_map, base_hart_id=1, cores_per_cluster=2, strict=False
):
    """Attach inferred static node/kernel labels to MGR_RUN_KERNEL tasks.

    bingo_trace.json currently stores event type and hart only; it does not carry
    cur_global_task_id. Therefore this annotation is an ordered per-core
    inference from offload_bingo_hw.h descriptor order.
    """
    if static_map is None:
        for task in tasks:
            task["node_id"] = None
            task["static_kernel"] = None
            task["static_label"] = None
            task["static_note"] = "disabled"
        return []

    streams = static_map["streams"]
    per_key_index = defaultdict(int)
    warnings = []
    warning_counts = Counter()

    for task in sorted(tasks, key=lambda item: (item["tid"], item["seq"])):
        location = parse_trace_tid_location(
            task["tid"], base_hart_id, cores_per_cluster
        )
        task["static_location"] = location
        task["node_id"] = None
        task["static_kernel"] = None
        task["static_label"] = None
        task["static_note"] = "unmapped"
        if location is None:
            task["static_note"] = "no_tid_location"
            continue
        stream = streams.get(location)
        if not stream:
            task["static_note"] = f"no_static_stream_C{location[0]}_Core{location[1]}"
            continue
        stream_index = per_key_index[location]
        if stream_index >= len(stream):
            task["static_note"] = f"stream_exhausted_C{location[0]}_Core{location[1]}"
            warning_counts[task["static_note"]] += 1
            continue
        node = stream[stream_index]
        per_key_index[location] += 1
        task["node_id"] = node.node_id
        task["static_kernel"] = node.kernel
        task["static_label"] = verify_deps.short_kernel_name(node.kernel)
        task["static_note"] = "inferred_descriptor_order"
        if task.get("kernel") == "Unknown" and task["static_label"]:
            task["kernel"] = task["static_label"]

    if strict:
        for key, stream in streams.items():
            consumed = per_key_index[key]
            if consumed and consumed != len(stream):
                warnings.append(
                    f"C{key[0]}/Core{key[1]} trace consumed {consumed}/{len(stream)} static device tasks"
                )
    for reason, count in sorted(warning_counts.items()):
        warnings.append(f"{reason}: {count} tasks")
    return warnings


def task_node_label(task):
    return f"N{task['node_id']}" if task.get("node_id") is not None else "-"


def task_kernel_label(task):
    return task.get("static_label") or task["kernel"]


def print_unified_timeline(tasks):
    """打印按绝对时间排序的跨核统一时间线。"""
    sorted_tasks = sorted(tasks, key=lambda t: t["exec_start_ns"])

    print("=" * 130)
    print("UNIFIED CROSS-CORE TIMELINE (按绝对仿真时间排序)")
    print("  数据源: bingo_trace.json (RTL $time, 绝对真实时间)")
    print(f"  任务总数: {len(sorted_tasks)}")
    print("=" * 130)
    print()

    header = (
        f"  {'#':>3s}  {'Core':<12s}  {'Seq':>3s}  {'Node':>6s}  {'Kernel':<24s}"
        f"  {'start_ns':>14s}  {'end_ns':>14s}  {'dur_ns':>10s}"
        f"  {'dur_cc':>8s}  {'freq':>7s}  {'wait_ns':>10s}  {'prep_ns':>8s}"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    for i, t in enumerate(sorted_tasks, 1):
        print(
            f"  {i:3d}  {t['tid']:<12s}  {t['seq']:3d}  {task_node_label(t):>6s}  {task_kernel_label(t)[:24]:<24s}"
            f"  {t['exec_start_ns']:>14,}  {t['exec_end_ns']:>14,}  {t['exec_dur_ns']:>10,}"
            f"  {t['exec_dur_cc']:>8,}  {t['exec_freq_MHz']:>6.1f}M"
            f"  {t['wait_ns']:>10,}  {t['prep_ns']:>8,}"
        )

    # workload 跨度
    first = sorted_tasks[0]["exec_start_ns"]
    last = max(t["exec_end_ns"] for t in sorted_tasks)
    span = last - first
    print()
    print(
        f"  Workload 跨度: {first:,} ns → {last:,} ns  (span = {span:,} ns = {span/1000:.1f} us)"
    )


def print_per_core_timeline(tasks):
    """打印各核心独立时间线，含 gap 分析。"""
    cores = {}
    for t in tasks:
        cores.setdefault(t["tid"], []).append(t)

    print()
    print("=" * 130)
    print("PER-CORE TIMELINE (各核心独立时间线 + gap)")
    print("=" * 130)

    for tid in sorted(cores.keys()):
        core_tasks = sorted(cores[tid], key=lambda t: t["exec_start_ns"])
        print()
        print(f"  ─── {tid} ({len(core_tasks)} tasks) ───")
        print()

        header = (
            f"    {'Seq':>3s}  {'Node':>6s}  {'Kernel':<24s}"
            f"  {'start_ns':>14s}  {'end_ns':>14s}  {'dur_ns':>10s}"
            f"  {'dur_cc':>8s}  {'freq':>7s}  {'wait_ns':>10s}  {'gap_ns':>12s}"
        )
        print(header)
        print("    " + "─" * (len(header) - 4))

        prev_end = None
        for t in core_tasks:
            if prev_end is not None:
                gap = t["exec_start_ns"] - prev_end
                gap_str = f"+{gap:>10,}"
            else:
                gap_str = f"{'(first)':>12s}"

            print(
                f"    {t['seq']:3d}  {task_node_label(t):>6s}  {task_kernel_label(t)[:24]:<24s}"
                f"  {t['exec_start_ns']:>14,}  {t['exec_end_ns']:>14,}  {t['exec_dur_ns']:>10,}"
                f"  {t['exec_dur_cc']:>8,}  {t['exec_freq_MHz']:>6.1f}M"
                f"  {t['wait_ns']:>10,}  {gap_str}"
            )
            prev_end = t["exec_end_ns"]


def print_scheduling_overhead(tasks):
    """打印调度开销分析 (GET_READY 等待 + PREP 准备)。"""
    print()
    print("=" * 130)
    print("SCHEDULING OVERHEAD (调度开销分析)")
    print("  wait_ns = GET_READY 持续时间 (在 ready queue 等待新任务的时间)")
    print("  prep_ns = PREP 持续时间 (解析任务参数的准备时间)")
    print("=" * 130)
    print()

    cores = {}
    for t in tasks:
        cores.setdefault(t["tid"], []).append(t)

    for tid in sorted(cores.keys()):
        core_tasks = cores[tid]
        waits = [t["wait_ns"] for t in core_tasks]
        preps = [t["prep_ns"] for t in core_tasks]
        execs = [t["exec_dur_ns"] for t in core_tasks]

        total_wait = sum(waits)
        total_prep = sum(preps)
        total_exec = sum(execs)
        total = total_wait + total_prep + total_exec

        print(f"  {tid}:")
        print(f"    Tasks:      {len(core_tasks)}")
        print(
            f"    Wait total: {total_wait:>12,} ns  ({100*total_wait/total:.1f}% of core time)"
            if total > 0
            else ""
        )
        print(
            f"    Prep total: {total_prep:>12,} ns  ({100*total_prep/total:.1f}% of core time)"
            if total > 0
            else ""
        )
        print(
            f"    Exec total: {total_exec:>12,} ns  ({100*total_exec/total:.1f}% of core time)"
            if total > 0
            else ""
        )
        print(f"    Max wait:   {max(waits):>12,} ns" if waits else "")
        print(f"    Max exec:   {max(execs):>12,} ns" if execs else "")
        print()


def print_static_mapping_summary(tasks, static_map, mapping_warnings):
    print()
    print("=" * 130)
    print("STATIC NODE MAPPING (DFG/header -> bingo_trace tasks)")
    print("=" * 130)
    if static_map is None:
        print("  Static mapping disabled.")
        return

    print(f"  Workload dir: {static_map['workload_dir']}")
    print(
        "  Note: bingo_trace.json does not encode cur_global_task_id; Node labels below are inferred "
        "from per-core descriptor order."
    )
    print()

    mapped = [task for task in tasks if task.get("node_id") is not None]
    unmapped = [task for task in tasks if task.get("node_id") is None]
    dynamic_mapped = [
        task
        for task in mapped
        if task.get("static_kernel") and "moe_dynamic_expert" in task["static_kernel"]
    ]
    print(f"  Mapped tasks:          {len(mapped):5d} / {len(tasks)}")
    print(f"  Dynamic expert mapped: {len(dynamic_mapped):5d}")
    if unmapped:
        reasons = Counter(task.get("static_note", "unknown") for task in unmapped)
        print("  Unmapped reasons:")
        for reason, count in reasons.most_common():
            print(f"    {reason:<42s} {count:5d}")
    print()

    if static_map["errors"]:
        print("  Static artifact errors:")
        for item in static_map["errors"]:
            print(f"    - {item}")
        print()
    if static_map["warnings"]:
        print("  Static artifact warnings:")
        for item in static_map["warnings"]:
            print(f"    - {item}")
        print()
    if mapping_warnings:
        print("  Mapping warnings:")
        for item in mapping_warnings:
            print(f"    - {item}")
        print()


def print_utilization(tasks):
    """计算并打印各核心利用率。"""
    print()
    print("=" * 130)
    print("CORE UTILIZATION (核心利用率)")
    print("=" * 130)
    print()

    cores = {}
    for t in tasks:
        cores.setdefault(t["tid"], []).append(t)

    all_tasks = sorted(tasks, key=lambda t: t["exec_start_ns"])
    global_start = all_tasks[0]["cycle_start_ns"]
    global_end = max(t["exec_end_ns"] for t in all_tasks)
    global_span = global_end - global_start

    print(
        f"  全局时间跨度: {global_start:,} ns → {global_end:,} ns  (span = {global_span:,} ns = {global_span/1000:.1f} us)"
    )
    print()

    total_busy = 0
    for tid in sorted(cores.keys()):
        core_tasks = sorted(cores[tid], key=lambda t: t["exec_start_ns"])
        core_start = core_tasks[0]["cycle_start_ns"]
        core_end = core_tasks[-1]["exec_end_ns"]
        core_span = core_end - core_start
        busy = sum(t["exec_dur_ns"] for t in core_tasks)
        total_busy += busy
        idle = core_span - busy
        util = 100 * busy / core_span if core_span > 0 else 0

        print(f"  {tid}:")
        print(f"    活跃时间 (exec):  {busy:>12,} ns")
        print(f"    核心跨度:         {core_span:>12,} ns")
        print(f"    空闲时间:         {idle:>12,} ns")
        print(f"    利用率:           {util:>11.1f}%")

        # gap 统计
        gaps = []
        for i in range(1, len(core_tasks)):
            g = core_tasks[i]["exec_start_ns"] - core_tasks[i - 1]["exec_end_ns"]
            gaps.append(g)
        if gaps:
            print(f"    平均 gap:         {sum(gaps)/len(gaps):>12,.0f} ns")
            print(f"    最大 gap:         {max(gaps):>12,} ns")
            print(f"    最小 gap:         {min(gaps):>12,} ns")
        print()

    # 并行度分析 (简单重叠检测)
    print(f"  ─── 并行执行分析 ───")

    # 计算任意两个核心间执行时间的重叠
    core_list = sorted(cores.keys())
    for i in range(len(core_list)):
        for j in range(i + 1, len(core_list)):
            tid_a, tid_b = core_list[i], core_list[j]
            overlap = compute_overlap(cores[tid_a], cores[tid_b])
            print(f"  {tid_a} ∩ {tid_b}: {overlap:>12,} ns 重叠执行时间")

    print()


def compute_overlap(tasks_a, tasks_b):
    """计算两组任务执行时间的重叠。"""
    total = 0
    for a in tasks_a:
        for b in tasks_b:
            overlap_start = max(a["exec_start_ns"], b["exec_start_ns"])
            overlap_end = min(a["exec_end_ns"], b["exec_end_ns"])
            if overlap_end > overlap_start:
                total += overlap_end - overlap_start
    return total


def print_versacore_efficiency_analysis(tasks):
    """
    调度算法效果专项分析:

    C0/C1 (shared expert):
      VersaCore 有效占比 = 计算时间 / (计算 + 等待 C2/C3 完成的 tail 时间)
      反映 shared expert 结束后等待 individual expert 的尾部开销。

    C2/C3 (individual expert):
      在 [第一个 compute 开始, 最后 store 结束] 窗口内:
        VersaCore compute%  = 实际计算时间占比
        DMA-wait%           = VersaCore 空闲 且 DMA 正在运行 (流水线掩盖中, 正常)
        True-idle%          = 两者均空闲 (scheduler/barrier 纯开销, 越小越好)
        Final-store%        = 最后一个 compute 结束后的 DMA store 时间
    """
    COMPUTE_LABELS_SHARED = frozenset(
        {
            "dual_vc_l15_moe_swiglu",
            "dual_vc_l15_moe_down",
        }
    )
    COMPUTE_LABELS_INDIV = frozenset(
        {
            "compute_gate_up_block",
            "compute_gate_up_full",
            "compute_down_block",
            "compute_down_full",
        }
    )
    # 每 slot 固定 6 个 compute 任务: 2×gate_up_block + gate_up_full + 2×down_block + down_full
    COMPUTE_PER_SLOT = 6

    all_end = max(t["exec_end_ns"] for t in tasks)

    print()
    print("=" * 130)
    print("VERSACORE SCHEDULING EFFICIENCY  (VersaCore 调度效率专项分析)")
    print()
    print("  C0/C1 指标: VersaCore 有效占比 = compute / (compute + 等待 C2/C3 完成)")
    print("  C2/C3 指标: 在 pipeline 窗口 [first_compute -> last_store] 内:")
    print("    VersaCore compute%  = 实际计算时间 / 窗口")
    print(
        "    DMA-wait%           = VersaCore 空闲 且 DMA 在运行 (流水线掩盖 -- 正常但 DMA 是瓶颈)"
    )
    print("    True-idle%          = 两者均空闲 (scheduler/barrier 纯开销 -- 越小越好)")
    print("    Final-store%        = 最后 compute 后的 DMA store 时间")
    print("=" * 130)

    # ─── C0/C1 Shared Expert ───────────────────────────────────────────────────────────────
    print()
    print("  ══ Shared Expert (C0, C1): VersaCore 利用率分析 ══")
    print()

    for cl in [0, 1]:
        c0_compute = sorted(
            [
                t
                for t in tasks
                if t.get("static_location") == (cl, 0)
                and task_kernel_label(t) in COMPUTE_LABELS_SHARED
            ],
            key=lambda t: t["exec_start_ns"],
        )
        if not c0_compute:
            print(f"  C{cl}: 无 VersaCore compute 数据，跳过")
            continue

        compute_busy = sum(t["exec_dur_ns"] for t in c0_compute)
        last_compute_end = c0_compute[-1]["exec_end_ns"]

        # 找 compute 结束后的下一个任务 (通常是 exit kernel), 其 start 即为等待结束时刻
        c0_all_sorted = sorted(
            [t for t in tasks if t.get("static_location") == (cl, 0)],
            key=lambda t: t["exec_start_ns"],
        )
        next_task = next(
            (t for t in c0_all_sorted if t["exec_start_ns"] > last_compute_end), None
        )
        wait_ns = (
            next_task["exec_start_ns"] - last_compute_end
            if next_task
            else all_end - last_compute_end
        )
        total_span = compute_busy + wait_ns
        vc_useful = 100.0 * compute_busy / total_span if total_span > 0 else 0.0

        print(f"  C{cl} (Shared Expert {cl} -- core0 VersaCore):")
        for t in c0_compute:
            lbl = task_kernel_label(t)
            cc = t.get("exec_dur_cc", 0)
            print(f"    {lbl:<35s}  {t['exec_dur_ns']:>10,} ns  ({cc:,} cc)")
        print(f"    {'─' * 55}")
        print(
            f"    {'VersaCore 总计':<35s}  {compute_busy:>10,} ns  ({compute_busy / 1e3:.1f} us)"
        )
        print(
            f"    {'等待 C2/C3 完成 (tail wait)':<35s}  {wait_ns:>10,} ns  ({wait_ns / 1e3:.1f} us)"
        )
        print(
            f"    ★ VersaCore 有效占比:              {vc_useful:>8.1f}%  [= compute / (compute + tail)]"
        )
        print()

    # ─── C2/C3 Individual Expert ──────────────────────────────────────────────────────────
    print()
    print("  ══ Individual Expert (C2, C3): DMA-Compute 流水线效率 ══")
    print()

    for cl in [2, 3]:
        c0_compute = sorted(
            [
                t
                for t in tasks
                if t.get("static_location") == (cl, 0)
                and task_kernel_label(t) in COMPUTE_LABELS_INDIV
            ],
            key=lambda t: t["exec_start_ns"],
        )
        c1_all = sorted(
            [t for t in tasks if t.get("static_location") == (cl, 1)],
            key=lambda t: t["exec_start_ns"],
        )
        if not c0_compute:
            print(f"  C{cl}: 无 compute 数据，跳过")
            continue

        # Pipeline window = [first compute start, last store end]
        store_tasks = [t for t in c1_all if task_kernel_label(t) == "store"]
        pipeline_start = c0_compute[0]["exec_start_ns"]
        last_compute_end = c0_compute[-1]["exec_end_ns"]
        last_store_end = (
            max(t["exec_end_ns"] for t in store_tasks)
            if store_tasks
            else last_compute_end
        )
        pipeline_end = max(last_compute_end, last_store_end)
        pipeline_span = pipeline_end - pipeline_start

        compute_busy = sum(t["exec_dur_ns"] for t in c0_compute)

        # Gap analysis between consecutive compute tasks
        dma_wait_ns = 0
        true_idle_ns = 0
        gap_log = []  # (type, gap_ns) for debugging
        for i in range(1, len(c0_compute)):
            gap_s = c0_compute[i - 1]["exec_end_ns"]
            gap_e = c0_compute[i]["exec_start_ns"]
            gap = gap_e - gap_s
            if gap <= 0:
                continue
            dma_running = any(
                t["exec_start_ns"] < gap_e and t["exec_end_ns"] > gap_s for t in c1_all
            )
            if dma_running:
                dma_wait_ns += gap
                gap_log.append(("DMA-wait", gap))
            else:
                true_idle_ns += gap
                gap_log.append(("true-idle", gap))

        # Tail after last compute (store running)
        store_tail_ns = max(0, pipeline_end - last_compute_end)

        # DMA stats within pipeline window (tasks that overlap [pipeline_start, pipeline_end])
        window_c1 = [
            t
            for t in c1_all
            if t["exec_start_ns"] < pipeline_end and t["exec_end_ns"] > pipeline_start
        ]
        # Clip each DMA task to window and sum
        dma_busy_clipped = sum(
            min(t["exec_end_ns"], pipeline_end)
            - max(t["exec_start_ns"], pipeline_start)
            for t in window_c1
        )
        overlap_ns = compute_overlap(c0_compute, window_c1)
        dma_overlap_rate = (
            100.0 * overlap_ns / dma_busy_clipped if dma_busy_clipped > 0 else 0.0
        )

        vc_util = 100.0 * compute_busy / pipeline_span
        dma_wait_pct = 100.0 * dma_wait_ns / pipeline_span
        true_idle_pct = 100.0 * true_idle_ns / pipeline_span
        store_tail_pct = 100.0 * store_tail_ns / pipeline_span

        indiv_label = "INDIV_A" if cl == 2 else "INDIV_B"
        print(
            f"  C{cl} ({indiv_label}):  pipeline {pipeline_start:,} -> {pipeline_end:,} ns"
            f"  (span = {pipeline_span:,} ns = {pipeline_span / 1e3:.1f} us)"
        )
        print(
            f"  core0 compute 任务: {len(c0_compute)}  |  core1 DMA 任务 (窗口内): {len(window_c1)}"
        )
        print()

        # Compute breakdown
        by_type = defaultdict(list)
        for t in c0_compute:
            by_type[task_kernel_label(t)].append(t)

        print(f"    VersaCore (core0) 计算分解:")
        for lbl in (
            "compute_gate_up_block",
            "compute_gate_up_full",
            "compute_down_block",
            "compute_down_full",
        ):
            ts_list = by_type.get(lbl, [])
            if not ts_list:
                continue
            dur = sum(t["exec_dur_ns"] for t in ts_list)
            pct = 100.0 * dur / pipeline_span
            print(f"      {lbl:<35s}  x{len(ts_list):2d}  {dur:>10,} ns  ({pct:5.1f}%)")
        print(f"      {'─' * 65}")
        print(
            f"      {'VersaCore 合计':<40s}  {compute_busy:>10,} ns  ({vc_util:5.1f}%)"
        )
        print()

        print(f"    Gap / Stall 分解 (pipeline 窗口内 VersaCore 空闲时间):")
        print(
            f"      DMA-wait (DMA 运行中, VersaCore 等待):      {dma_wait_ns:>10,} ns  ({dma_wait_pct:5.1f}%)"
        )
        print(f"        -> DMA 是瓶颈, 流水线掩盖中但未完全掩盖")
        print(
            f"      True-idle (scheduler/barrier 纯开销):        {true_idle_ns:>10,} ns  ({true_idle_pct:5.1f}%)"
        )
        print(f"        -> 纯开销 (越小越好, < 5% 属正常)")
        print(
            f"      Final-store (compute 后 DMA store):          {store_tail_ns:>10,} ns  ({store_tail_pct:5.1f}%)"
        )
        print()

        print(f"    DMA (core1) 分析:")
        print(f"      DMA 总时间 (窗口内, clipped):   {dma_busy_clipped:>10,} ns")
        print(
            f"      DMA-compute 重叠时间:           {overlap_ns:>10,} ns  (掩盖率 {dma_overlap_rate:.1f}%)"
        )
        print()

        print(f"    ★ 调度效率摘要 [C{cl}]:")
        print(f"      VersaCore 利用率 (compute/window): {vc_util:>7.1f}%")
        print(
            f"      DMA 流水线等待 (DMA-wait):         {dma_wait_pct:>7.1f}%  <- DMA 带宽限制, 非调度问题"
        )
        print(
            f"      调度/barrier 纯开销 (True-idle):   {true_idle_pct:>7.1f}%  <- 调度优化目标"
        )
        print(f"      DMA 与 compute 重叠率:             {dma_overlap_rate:>7.1f}%")
        print()

        # Per-slot breakdown
        gather_tasks_sorted = sorted(
            [t for t in c1_all if task_kernel_label(t) == "gather_s1"],
            key=lambda t: t["exec_start_ns"],
        )
        stores_sorted = sorted(store_tasks, key=lambda t: t["exec_start_ns"])
        n_slots = len(gather_tasks_sorted)

        if n_slots > 0 and len(c0_compute) == n_slots * COMPUTE_PER_SLOT:
            print(
                f"    Per-slot 详细分解 ({n_slots} slots, {COMPUTE_PER_SLOT} compute/slot):"
            )
            print(
                f"    {'Slot':>4}  {'Gather-start':>13}  {'Store-end':>12}  "
                f"{'Span(us)':>8}  {'VCcomp(ns)':>10}  {'VCutil':>6}  "
                f"{'DMA-wait(ns)':>12}  {'True-idle(ns)':>13}"
            )
            print(f"    {'─' * 100}")

            for s in range(n_slots):
                sl_comp = c0_compute[s * COMPUTE_PER_SLOT : (s + 1) * COMPUTE_PER_SLOT]
                sl_last_end = sl_comp[-1]["exec_end_ns"]
                sl_gather_start = gather_tasks_sorted[s]["exec_start_ns"]

                # Find first store that starts after or at gather_start of this slot
                sl_store = next(
                    (t for t in stores_sorted if t["exec_start_ns"] >= sl_gather_start),
                    None,
                )
                sl_end = sl_store["exec_end_ns"] if sl_store else sl_last_end
                sl_span = sl_end - sl_gather_start

                # DMA tasks overlapping this slot's time range
                sl_c1 = [
                    t
                    for t in c1_all
                    if t["exec_start_ns"] < sl_end
                    and t["exec_end_ns"] > sl_gather_start
                ]

                sl_compute_busy = sum(t["exec_dur_ns"] for t in sl_comp)
                sl_vc_util = 100.0 * sl_compute_busy / sl_span if sl_span > 0 else 0.0

                sl_dma_wait = 0
                sl_true_idle = 0
                for i in range(1, len(sl_comp)):
                    gs = sl_comp[i - 1]["exec_end_ns"]
                    ge = sl_comp[i]["exec_start_ns"]
                    g = ge - gs
                    if g <= 0:
                        continue
                    dma_run = any(
                        t["exec_start_ns"] < ge and t["exec_end_ns"] > gs for t in sl_c1
                    )
                    if dma_run:
                        sl_dma_wait += g
                    else:
                        sl_true_idle += g

                print(
                    f"    {s:4d}  {sl_gather_start:>13,}  {sl_end:>12,}  "
                    f"{sl_span / 1e3:>8.1f}  {sl_compute_busy:>10,}  {sl_vc_util:>5.1f}%  "
                    f"{sl_dma_wait:>12,}  {sl_true_idle:>13,}"
                )
            print()
        else:
            print(
                f"    (per-slot 分解条件不满足: n_slots={n_slots}, compute={len(c0_compute)} != {n_slots * COMPUTE_PER_SLOT})"
            )
            print()

        print()


def print_gantt(tasks, cols=100):
    """打印简易 Gantt 图。"""
    sorted_tasks = sorted(tasks, key=lambda t: t["exec_start_ns"])
    global_start = sorted_tasks[0]["exec_start_ns"]
    global_end = max(t["exec_end_ns"] for t in sorted_tasks)
    span = global_end - global_start
    if span == 0:
        return

    print()
    print("=" * 130)
    scale = span / cols
    print(
        f"GANTT CHART ({cols} cols = {span:,} ns = {span/1000:.1f} us, 1 col ≈ {scale:,.0f} ns)"
    )
    print("=" * 130)
    print()

    # 核心标记: H=Host, G=GEMM Core, D=DMA Core
    core_char = {"Host Core": "H", "Core 1": "G", "Core 2": "D"}
    # Kernel类型首字母用于 Gantt 条内区分
    kernel_char = {
        "Entry": "N",
        "Exit": "X",
        "Dummy": "D",
        "RouterSched": "R",
        "DispatchSW": "s",
        "DispatchHW": "h",
        "DispatchCERF": "c",
        "Softmax": "S",
        "ScatterMeta": "M",
        "Swish": "W",
        "GLU": "L",
        "Accumulate": "A",
        "ScatterPad": "P",
        "CheckResult": "C",
        "MoEPrepare": "P",
        "MoEExecute": "E",
        "L15_Full": "F",
        "L15_SwiGLU": "f",
        "L15_Down": "j",
        "GEMM_Full": "G",
        "GEMM_Min": "g",
        "Host_DMA": "T",
        "Dev_DMA": "d",
        "XDMA": "x",
        "Dual_DMA": "B",
        "gather_s1": "a",
        "load_gate_up_block": "u",
        "compute_gate_up_block": "U",
        "prefetch_s2_down": "p",
        "load_down_block": "v",
        "compute_down_block": "V",
        "prefetch_s4_next_s1": "q",
        "store": "o",
        "Unknown": "?",
        "PaddingInit": "I",
        "CompletionRelay": "r",
    }

    header = f"  {'Core':<12s} {'Seq':>3s} {'Node':>6s} {'Kernel':<24s} |"
    print(header + "─" * cols + "|")

    for t in sorted_tasks:
        s = int((t["exec_start_ns"] - global_start) / scale)
        e = int((t["exec_end_ns"] - global_start) / scale)
        s = max(0, min(s, cols - 1))
        e = max(s + 1, min(e, cols))
        label = task_kernel_label(t)
        ch = kernel_char.get(
            label, kernel_char.get(t["kernel"], core_char.get(t["tid"], "?"))
        )
        bar = "." * s + ch * (e - s) + "." * (cols - e)
        print(
            f"  {t['tid']:<12s} {t['seq']:3d} {task_node_label(t):>6s} "
            f"{label[:24]:<24s} |{bar}|"
        )

    print()
    print("  Gantt 图例:")
    for kname, kch in sorted(kernel_char.items()):
        print(f"    {kch} = {kname}", end="  ")
    print()
    print()


def print_moe_scheduling_breakdown(tasks, events):
    """专门分析 Phase 3 (MoEPrepare) 和 Phase 4 (MoEExecute) 的内部时间分解。

    Phase 3 (node_prepare / __host_bingo_kernel_moe_prepare_request):
      BINGO_TRACE_HOST_MOE_PREPARE = init + token counting + request build
                                    + scheduler + optional schedule print
      └─ BINGO_TRACE_HOST_MOE_SCHED = 纯调度算法时间

    Phase 4 (node_execute / __host_bingo_kernel_moe_execute):
      BINGO_TRACE_HOST_MOE_EXECUTE = dynamic-arg init + programming + DMA slot fill
                                   + CAM update + inactive/active L3→L1 flush
      ├─ BINGO_TRACE_HOST_MOE_EXEC_INIT_* = Phase 4 init sub-stages
      └─ BINGO_TRACE_HOST_MOE_PRELOWER = scheduler result → node-level call args
    """
    print()
    print("=" * 130)
    print("MOE PHASE 3+4 SCHEDULING BREAKDOWN (调度+分发时间分解)")
    print(
        "  Phase 3 = MoEPrepare: schedule + lowering → writes per-cluster L3 stage args"
    )
    print(
        "  Phase 4 = MoEExecute: flushes runtime headers + active stage args to C2/C3 L1"
    )
    print("=" * 130)
    print()

    # 收集子事件持续时间
    def sub_dur(task, marker_name):
        """从该 task 的子事件中找对应 marker 的持续时间（B/E 配对 或 X 格式）。"""
        total = 0
        for s in task.get("sub_events", []):
            if s["name"] == marker_name:
                total += s["dur"]
        return total

    hw_sched_submarkers = [
        ("sort/rem init", "BINGO_TRACE_HOST_MOE_HW_SORT"),
        ("init/head writes", "BINGO_TRACE_HOST_MOE_HW_INIT_WRITE"),
        ("wait done", "BINGO_TRACE_HOST_MOE_HW_WAIT"),
        ("status decode", "BINGO_TRACE_HOST_MOE_HW_STATUS_DECODE"),
        ("refill/commit MMIO", "BINGO_TRACE_HOST_MOE_HW_CONTROL_WRITE"),
        ("restart write", "BINGO_TRACE_HOST_MOE_HW_RESTART"),
        ("task drain/pop", "BINGO_TRACE_HOST_MOE_HW_DRAIN_TASKS"),
    ]
    hw_lower_submarkers = [
        ("count slots", "BINGO_TRACE_HOST_MOE_HW_LOWER_COUNT"),
        ("task build", "BINGO_TRACE_HOST_MOE_HW_LOWER_TASK_BUILD"),
        ("pending patch", "BINGO_TRACE_HOST_MOE_HW_LOWER_PENDING_PATCH"),
        ("slot select", "BINGO_TRACE_HOST_MOE_HW_LOWER_SLOT_SELECT"),
        ("arg clear/program", "BINGO_TRACE_HOST_MOE_HW_LOWER_ARG_PROGRAM"),
        ("call-args prelower", "BINGO_TRACE_HOST_MOE_PRELOWER"),
        ("dma slot patch", "BINGO_TRACE_HOST_MOE_HW_LOWER_DMA_PATCH"),
        ("final state", "BINGO_TRACE_HOST_MOE_HW_LOWER_FINAL_STATE"),
    ]
    router_tasks = [t for t in tasks if task_kernel_label(t) == "RouterSched"]
    prepare_tasks = [t for t in tasks if task_kernel_label(t) == "MoEPrepare"]
    execute_tasks = [t for t in tasks if task_kernel_label(t) == "MoEExecute"]

    if not router_tasks and not prepare_tasks and not execute_tasks:
        print("  未检测到 MoEPrepare / MoEExecute 任务。")
        print("  可能原因:")
        print("    1. ENABLE_PHASE3_PHASE4 = False（DFG 中没有这两个节点）")
        print("    2. 仿真使用的是上次编译的二进制（marker 代码未编进去）")
        print(
            "    3. 这两个 kernel 没有内部子事件被 bingo_trace 捕获（需确认 bingo_trace.py 解析了 HOST 核心日志）"
        )
        print()
        return

    if router_tasks:
        print(f"  Phase 2 (RouterSched) tasks: {len(router_tasks)}")
        for t in router_tasks:
            exec_ns = t["exec_dur_ns"]
            sched_ns = sub_dur(t, "BINGO_TRACE_HOST_ROUTER_SCHED")
            print(
                f"    Node {task_node_label(t)} | Core {t['tid']} | Total = {exec_ns:,} ns"
            )
            if sched_ns > 0:
                print(
                    f"      router schedule      : {sched_ns:>12,} ns  ({100*sched_ns/exec_ns:.1f}%)"
                )
        print()

    print(f"  Phase 3 (MoEPrepare) tasks: {len(prepare_tasks)}")
    for t in prepare_tasks:
        exec_ns = t["exec_dur_ns"]
        init_ns = sub_dur(t, "BINGO_TRACE_HOST_MOE_PREPARE_INIT")
        sched_ns = sub_dur(t, "BINGO_TRACE_HOST_MOE_SCHED")
        hw_sched_ns = sub_dur(t, "BINGO_TRACE_HOST_MOE_HW_SCHED")
        hw_lower_ns = sub_dur(t, "BINGO_TRACE_HOST_MOE_HW_LOWER")
        token_count_ns = sub_dur(t, "BINGO_TRACE_HOST_MOE_TOKEN_COUNT")
        request_build_ns = sub_dur(t, "BINGO_TRACE_HOST_MOE_REQUEST_BUILD")
        sched_print_ns = sub_dur(t, "BINGO_TRACE_HOST_MOE_SCHED_PRINT")
        other_ns = exec_ns - init_ns - token_count_ns - request_build_ns - sched_ns - sched_print_ns

        print(
            f"    Node {task_node_label(t)} | Core {t['tid']} | Total = {exec_ns:,} ns"
        )
        if init_ns > 0 or token_count_ns > 0 or request_build_ns > 0 or sched_ns > 0:
            if init_ns > 0:
                print(
                    f"      init/check/memset    : {init_ns:>12,} ns  ({100*init_ns/exec_ns:.1f}%)"
                )
            print(
                f"      token count/scatter  : {token_count_ns:>12,} ns  ({100*token_count_ns/exec_ns:.1f}%)"
            )
            print(
                f"      CAM + request build  : {request_build_ns:>12,} ns  ({100*request_build_ns/exec_ns:.1f}%)"
            )
            print(
                f"      moe_schedule() 算法  : {sched_ns:>12,} ns  ({100*sched_ns/exec_ns:.1f}%)"
            )
            if hw_sched_ns > 0 or hw_lower_ns > 0:
                sched_other = sched_ns - hw_sched_ns - hw_lower_ns
                print(
                    f"        HW MMIO scheduler : {hw_sched_ns:>12,} ns  ({100*hw_sched_ns/exec_ns:.1f}% of node)"
                )
                hw_sub_sum = 0
                for _, marker in hw_sched_submarkers:
                    hw_sub_sum += sub_dur(t, marker)
                if hw_sub_sum > 0:
                    print("          HW scheduler sub-stages:")
                    for label, marker in hw_sched_submarkers:
                        ns = sub_dur(t, marker)
                        if ns > 0:
                            print(
                                f"            {label:<16}: {ns:>12,} ns  ({100*ns/exec_ns:.1f}% of node)"
                            )
                    hw_sched_other = hw_sched_ns - hw_sub_sum
                    if hw_sched_other > 0:
                        print(
                            f"            HW sched wrapper: {hw_sched_other:>12,} ns  ({100*hw_sched_other/exec_ns:.1f}% of node)"
                        )
                print(
                    f"        HW plan lowering  : {hw_lower_ns:>12,} ns  ({100*hw_lower_ns/exec_ns:.1f}% of node)"
                )
                lower_sub_sum = 0
                for _, marker in hw_lower_submarkers:
                    lower_sub_sum += sub_dur(t, marker)
                if lower_sub_sum > 0:
                    print("          HW lowering sub-stages:")
                    for label, marker in hw_lower_submarkers:
                        ns = sub_dur(t, marker)
                        if ns > 0:
                            print(
                                f"            {label:<18}: {ns:>12,} ns  ({100*ns/exec_ns:.1f}% of node)"
                            )
                    hw_lower_other = hw_lower_ns - lower_sub_sum
                    if hw_lower_other > 0:
                        print(
                            f"            HW lower wrapper: {hw_lower_other:>12,} ns  ({100*hw_lower_other/exec_ns:.1f}% of node)"
                        )
                if sched_other > 0:
                    print(
                        f"        scheduler wrapper : {sched_other:>12,} ns  ({100*sched_other/exec_ns:.1f}% of node)"
                    )
            if sched_print_ns > 0:
                print(
                    f"      schedule debug print : {sched_print_ns:>12,} ns  ({100*sched_print_ns/exec_ns:.1f}%)"
                )
            if other_ns > 0:
                print(
                    f"      其他未标记          : {other_ns:>12,} ns  ({100*other_ns/exec_ns:.1f}%)"
                )
        else:
            print(f"      (未检测到子事件，需重新编译后仿真)")
    print()

    print(f"  Phase 4 (MoEExecute) tasks: {len(execute_tasks)}")
    for t in execute_tasks:
        exec_ns = t["exec_dur_ns"]
        init_ns = sub_dur(t, "BINGO_TRACE_HOST_MOE_EXEC_INIT")
        init_setup_ns = sub_dur(t, "BINGO_TRACE_HOST_MOE_EXEC_INIT_SETUP")
        init_count_ns = sub_dur(t, "BINGO_TRACE_HOST_MOE_EXEC_INIT_COUNT")
        init_state_ns = sub_dur(t, "BINGO_TRACE_HOST_MOE_EXEC_INIT_HEADER")
        init_slot_c2_ns = sub_dur(t, "BINGO_TRACE_HOST_MOE_EXEC_INIT_SLOT_C2")
        init_slot_c3_ns = sub_dur(t, "BINGO_TRACE_HOST_MOE_EXEC_INIT_SLOT_C3")
        program_ns = sub_dur(t, "BINGO_TRACE_HOST_MOE_EXEC_PROGRAM")
        prelower_ns = sub_dur(t, "BINGO_TRACE_HOST_MOE_PRELOWER")
        dma_fill_ns = sub_dur(t, "BINGO_TRACE_HOST_MOE_EXEC_DMA_FILL")
        cam_ns = sub_dur(t, "BINGO_TRACE_HOST_MOE_EXEC_CAM")
        flush0_ns = sub_dur(t, "BINGO_TRACE_HOST_MOE_EXEC_FLUSH0")
        flush1_ns = sub_dur(t, "BINGO_TRACE_HOST_MOE_EXEC_FLUSH1")
        program_other_ns = program_ns - prelower_ns
        other_ns = exec_ns - init_ns - program_ns - dma_fill_ns - cam_ns - flush0_ns - flush1_ns
        print(
            f"    Node {task_node_label(t)} | Core {t['tid']} | Total = {exec_ns:,} ns"
        )
        if init_ns > 0 or program_ns > 0 or dma_fill_ns > 0 or flush0_ns > 0 or flush1_ns > 0:
            print(
                f"      init/clear templates : {init_ns:>12,} ns  ({100*init_ns/exec_ns:.1f}%)"
            )
            init_sub_sum = (
                init_setup_ns
                + init_count_ns
                + init_state_ns
                + init_slot_c2_ns
                + init_slot_c3_ns
            )
            if init_sub_sum > 0:
                if init_setup_ns > 0:
                    print(
                        f"        init setup        : {init_setup_ns:>12,} ns  ({100*init_setup_ns/exec_ns:.1f}% of node)"
                    )
                if init_count_ns > 0:
                    print(
                        f"        active count      : {init_count_ns:>12,} ns  ({100*init_count_ns/exec_ns:.1f}% of node)"
                    )
                if init_state_ns > 0:
                    print(
                        f"        runtime header build: {init_state_ns:>12,} ns  ({100*init_state_ns/exec_ns:.1f}% of node)"
                    )
                if init_slot_c2_ns > 0:
                    print(
                        f"        C2 dyn clear      : {init_slot_c2_ns:>12,} ns  ({100*init_slot_c2_ns/exec_ns:.1f}% of node)"
                    )
                if init_slot_c3_ns > 0:
                    print(
                        f"        C3 dyn clear      : {init_slot_c3_ns:>12,} ns  ({100*init_slot_c3_ns/exec_ns:.1f}% of node)"
                    )
                init_other_ns = init_ns - init_sub_sum
                if init_other_ns > 0:
                    print(
                        f"        init wrapper      : {init_other_ns:>12,} ns  ({100*init_other_ns/exec_ns:.1f}% of node)"
                    )
            print(
                f"      program task args    : {program_ns:>12,} ns  ({100*program_ns/exec_ns:.1f}%)"
            )
            print(
                f"        node-call prelower: {prelower_ns:>12,} ns  ({100*prelower_ns/exec_ns:.1f}% of node)"
            )
            if program_other_ns > 0:
                print(
                    f"        program wrapper   : {program_other_ns:>12,} ns  ({100*program_other_ns/exec_ns:.1f}% of node)"
                )
            print(
                f"      fill DMA slots       : {dma_fill_ns:>12,} ns  ({100*dma_fill_ns/exec_ns:.1f}%)"
            )
            print(
                f"      update CAM state     : {cam_ns:>12,} ns  ({100*cam_ns/exec_ns:.1f}%)"
            )
            print(
                f"      legacy inactive flush: {flush0_ns:>12,} ns  ({100*flush0_ns/exec_ns:.1f}%)"
            )
            print(
                f"      final active flush   : {flush1_ns:>12,} ns  ({100*flush1_ns/exec_ns:.1f}%)"
            )
            if other_ns > 0:
                print(
                    f"      其他未标记          : {other_ns:>12,} ns  ({100*other_ns/exec_ns:.1f}%)"
                )
        else:
            print(
                f"      L3→L1 args lowering  : {exec_ns:>12,} ns  (旧 trace 未检测到 PRELOWER 子事件)"
            )
    print()

    if prepare_tasks and execute_tasks:
        p_ns = sum(t["exec_dur_ns"] for t in prepare_tasks)
        e_ns = sum(t["exec_dur_ns"] for t in execute_tasks)
        total_ns = p_ns + e_ns
        prepare_init_total = sum(
            sub_dur(t, "BINGO_TRACE_HOST_MOE_PREPARE_INIT") for t in prepare_tasks
        )
        token_count_total = sum(
            sub_dur(t, "BINGO_TRACE_HOST_MOE_TOKEN_COUNT") for t in prepare_tasks
        )
        request_build_total = sum(
            sub_dur(t, "BINGO_TRACE_HOST_MOE_REQUEST_BUILD") for t in prepare_tasks
        )
        sched_ns = sum(sub_dur(t, "BINGO_TRACE_HOST_MOE_SCHED") for t in prepare_tasks)
        hw_sched_total = sum(
            sub_dur(t, "BINGO_TRACE_HOST_MOE_HW_SCHED") for t in prepare_tasks
        )
        hw_lower_total = sum(
            sub_dur(t, "BINGO_TRACE_HOST_MOE_HW_LOWER") for t in prepare_tasks
        )
        sched_print_total = sum(
            sub_dur(t, "BINGO_TRACE_HOST_MOE_SCHED_PRINT") for t in prepare_tasks
        )
        exec_init_total = sum(
            sub_dur(t, "BINGO_TRACE_HOST_MOE_EXEC_INIT") for t in execute_tasks
        )
        exec_init_setup_total = sum(
            sub_dur(t, "BINGO_TRACE_HOST_MOE_EXEC_INIT_SETUP") for t in execute_tasks
        )
        exec_init_count_total = sum(
            sub_dur(t, "BINGO_TRACE_HOST_MOE_EXEC_INIT_COUNT") for t in execute_tasks
        )
        exec_init_state_total = sum(
            sub_dur(t, "BINGO_TRACE_HOST_MOE_EXEC_INIT_HEADER") for t in execute_tasks
        )
        exec_init_slot_c2_total = sum(
            sub_dur(t, "BINGO_TRACE_HOST_MOE_EXEC_INIT_SLOT_C2") for t in execute_tasks
        )
        exec_init_slot_c3_total = sum(
            sub_dur(t, "BINGO_TRACE_HOST_MOE_EXEC_INIT_SLOT_C3") for t in execute_tasks
        )
        exec_program_total = sum(
            sub_dur(t, "BINGO_TRACE_HOST_MOE_EXEC_PROGRAM") for t in execute_tasks
        )
        prelower_ns = sum(
            sub_dur(t, "BINGO_TRACE_HOST_MOE_PRELOWER") for t in execute_tasks
        )
        exec_dma_fill_total = sum(
            sub_dur(t, "BINGO_TRACE_HOST_MOE_EXEC_DMA_FILL") for t in execute_tasks
        )
        exec_cam_total = sum(
            sub_dur(t, "BINGO_TRACE_HOST_MOE_EXEC_CAM") for t in execute_tasks
        )
        exec_flush0_total = sum(
            sub_dur(t, "BINGO_TRACE_HOST_MOE_EXEC_FLUSH0") for t in execute_tasks
        )
        exec_flush1_total = sum(
            sub_dur(t, "BINGO_TRACE_HOST_MOE_EXEC_FLUSH1") for t in execute_tasks
        )
        print(f"  ─── 汇总 ───")
        print(f"  Phase 3 总计  : {p_ns:>12,} ns  ({100*p_ns/total_ns:.1f}%)")
        if prepare_init_total > 0 or token_count_total > 0 or request_build_total > 0 or sched_ns > 0:
            print(
                f"    init/check/memset  : {prepare_init_total:>12,} ns  ({100*prepare_init_total/p_ns:.1f}% of Phase 3)"
            )
            print(
                f"    token count/scatter: {token_count_total:>12,} ns  ({100*token_count_total/p_ns:.1f}% of Phase 3)"
            )
            print(
                f"    CAM + request build: {request_build_total:>12,} ns  ({100*request_build_total/p_ns:.1f}% of Phase 3)"
            )
            print(
                f"    moe_schedule() 算法: {sched_ns:>12,} ns  ({100*sched_ns/p_ns:.1f}% of Phase 3)"
            )
            if hw_sched_total > 0 or hw_lower_total > 0:
                print(
                    f"      HW MMIO scheduler : {hw_sched_total:>12,} ns  ({100*hw_sched_total/p_ns:.1f}% of Phase 3)"
                )
                hw_sub_total = 0
                for _, marker in hw_sched_submarkers:
                    hw_sub_total += sum(sub_dur(t, marker) for t in prepare_tasks)
                if hw_sub_total > 0:
                    print("        HW scheduler sub-stages:")
                    for label, marker in hw_sched_submarkers:
                        ns = sum(sub_dur(t, marker) for t in prepare_tasks)
                        if ns > 0:
                            print(
                                f"          {label:<16}: {ns:>12,} ns  ({100*ns/p_ns:.1f}% of Phase 3)"
                            )
                    hw_sched_other = hw_sched_total - hw_sub_total
                    if hw_sched_other > 0:
                        print(
                            f"          HW sched wrapper: {hw_sched_other:>12,} ns  ({100*hw_sched_other/p_ns:.1f}% of Phase 3)"
                        )
                print(
                    f"      HW plan lowering  : {hw_lower_total:>12,} ns  ({100*hw_lower_total/p_ns:.1f}% of Phase 3)"
                )
                lower_sub_total = 0
                for _, marker in hw_lower_submarkers:
                    lower_sub_total += sum(sub_dur(t, marker) for t in prepare_tasks)
                if lower_sub_total > 0:
                    print("        HW lowering sub-stages:")
                    for label, marker in hw_lower_submarkers:
                        ns = sum(sub_dur(t, marker) for t in prepare_tasks)
                        if ns > 0:
                            print(
                                f"          {label:<18}: {ns:>12,} ns  ({100*ns/p_ns:.1f}% of Phase 3)"
                            )
                    hw_lower_other = hw_lower_total - lower_sub_total
                    if hw_lower_other > 0:
                        print(
                            f"          HW lower wrapper: {hw_lower_other:>12,} ns  ({100*hw_lower_other/p_ns:.1f}% of Phase 3)"
                        )
                sched_other_total = sched_ns - hw_sched_total - hw_lower_total
                if sched_other_total > 0:
                    print(
                        f"      scheduler wrapper : {sched_other_total:>12,} ns  ({100*sched_other_total/p_ns:.1f}% of Phase 3)"
                    )
            if sched_print_total > 0:
                print(
                    f"    schedule debug print: {sched_print_total:>12,} ns  ({100*sched_print_total/p_ns:.1f}% of Phase 3)"
                )
        print(f"  Phase 4 总计  : {e_ns:>12,} ns  ({100*e_ns/total_ns:.1f}%)")
        if exec_init_total > 0 or exec_program_total > 0 or exec_flush0_total > 0 or exec_flush1_total > 0:
            print(
                f"    init/clear templates: {exec_init_total:>12,} ns  ({100*exec_init_total/e_ns:.1f}% of Phase 4)"
            )
            exec_init_sub_sum = (
                exec_init_setup_total
                + exec_init_count_total
                + exec_init_state_total
                + exec_init_slot_c2_total
                + exec_init_slot_c3_total
            )
            if exec_init_sub_sum > 0:
                print(
                    f"      init setup        : {exec_init_setup_total:>12,} ns  ({100*exec_init_setup_total/e_ns:.1f}% of Phase 4)"
                )
                print(
                    f"      active count      : {exec_init_count_total:>12,} ns  ({100*exec_init_count_total/e_ns:.1f}% of Phase 4)"
                )
                print(
                    f"      runtime header build: {exec_init_state_total:>12,} ns  ({100*exec_init_state_total/e_ns:.1f}% of Phase 4)"
                )
                print(
                    f"      C2 dyn clear      : {exec_init_slot_c2_total:>12,} ns  ({100*exec_init_slot_c2_total/e_ns:.1f}% of Phase 4)"
                )
                print(
                    f"      C3 dyn clear      : {exec_init_slot_c3_total:>12,} ns  ({100*exec_init_slot_c3_total/e_ns:.1f}% of Phase 4)"
                )
                exec_init_other_total = exec_init_total - exec_init_sub_sum
                if exec_init_other_total > 0:
                    print(
                        f"      init wrapper      : {exec_init_other_total:>12,} ns  ({100*exec_init_other_total/e_ns:.1f}% of Phase 4)"
                    )
            print(
                f"    program task args   : {exec_program_total:>12,} ns  ({100*exec_program_total/e_ns:.1f}% of Phase 4)"
            )
            print(
                f"      node-call prelower: {prelower_ns:>12,} ns  ({100*prelower_ns/e_ns:.1f}% of Phase 4)"
            )
            print(
                f"    fill DMA slots      : {exec_dma_fill_total:>12,} ns  ({100*exec_dma_fill_total/e_ns:.1f}% of Phase 4)"
            )
            print(
                f"    update CAM state    : {exec_cam_total:>12,} ns  ({100*exec_cam_total/e_ns:.1f}% of Phase 4)"
            )
            print(
                f"    legacy inactive flush: {exec_flush0_total:>12,} ns  ({100*exec_flush0_total/e_ns:.1f}% of Phase 4)"
            )
            print(
                f"    final active flush   : {exec_flush1_total:>12,} ns  ({100*exec_flush1_total/e_ns:.1f}% of Phase 4)"
            )
        print(f"  Phase 3+4 总计: {total_ns:>12,} ns = {total_ns/1000:.2f} us")
        print()
        print(f"  结论:")
        if sched_ns > 0:
            if sched_ns > e_ns * 2:
                print(
                    f"    调度算法 ({sched_ns/1000:.1f} us) >> args lowering ({e_ns/1000:.1f} us): 瓶颈在调度，需优化 moe_schedule()"
                )
            elif e_ns > sched_ns * 2:
                print(
                    f"    args lowering ({e_ns/1000:.1f} us) >> 调度算法 ({sched_ns/1000:.1f} us): 瓶颈在 L3→L1 搬运"
                )
            else:
                print(
                    f"    调度 ({sched_ns/1000:.1f} us) ≈ args lowering ({e_ns/1000:.1f} us): 两者相近"
                )
        print()


def _build_sub_event_parent_tree(subs):
    """
    对一组子事件建立父子包含关系。
    规则：事件 B 的父节点 = 时间上严格包含 B 且 dur 最小的事件 A。
    严格包含：A.start <= B.start 且 A.end >= B.end 且 (A,B) 不完全相同。
    返回 parents 列表（与 subs 等长），parents[i] = 父节点下标或 None。
    """
    n = len(subs)
    ends = [s["ts"] + s["dur"] for s in subs]
    parents = [None] * n

    for i in range(n):
        best_j = None
        best_span = float("inf")
        for j in range(n):
            if i == j:
                continue
            # 检查 subs[j] 是否包含 subs[i]
            if subs[j]["ts"] <= subs[i]["ts"] and ends[j] >= ends[i]:
                # 排除完全相同（ts 和 dur 都一样）
                if subs[j]["ts"] == subs[i]["ts"] and ends[j] == ends[i]:
                    continue
                if subs[j]["dur"] < best_span:
                    best_span = subs[j]["dur"]
                    best_j = j
        parents[i] = best_j

    return parents


def print_sub_event_detail(tasks):
    """打印每个任务的子事件明细（层级树状，自动识别容器事件）。"""
    print()
    print("=" * 130)
    print("SUB-EVENT DETAIL (各任务子事件明细，缩进表示包含关系，[total] 为容器事件)")
    print("=" * 130)

    sorted_tasks = sorted(tasks, key=lambda t: t["exec_start_ns"])
    for i, t in enumerate(sorted_tasks, 1):
        if not t["sub_events"]:
            continue
        print(
            f"\n  Task #{i}: {t['tid']} seq={t['seq']} node={task_node_label(t)} "
            f"kernel={task_kernel_label(t)} marker_class={t.get('marker_class', t['kernel'])}"
            f"  exec=[{t['exec_start_ns']:,} - {t['exec_end_ns']:,}] ({t['exec_dur_ns']:,} ns)"
        )
        subs = sorted(t["sub_events"], key=lambda s: s["ts"])

        if not subs:
            continue

        # 建立父子关系树
        parents = _build_sub_event_parent_tree(subs)
        children = {idx: [] for idx in range(len(subs))}
        for idx, p in enumerate(parents):
            if p is not None:
                children[p].append(idx)

        roots = [idx for idx, p in enumerate(parents) if p is None]

        def _print_event(idx, depth):
            s = subs[idx]
            short_name = s["name"].replace("BINGO_TRACE_", "")
            is_container = bool(children[idx])
            indent = "    " + "  " * depth
            tag = "  [total]" if is_container else "         "
            # 容器事件加粗标注（ASCII 友好：用 >> 前缀区分）
            prefix = ">>" if is_container else "  "
            print(
                f"{indent}{prefix} {short_name:<25s}  ts={s['ts']:>12,}  dur={s['dur']:>10,} ns{tag}"
                f"  cc={s['dur_cc']:>6}  freq={s['freq_MHz']:.1f}M"
            )
            for child_idx in sorted(children[idx], key=lambda c: subs[c]["ts"]):
                _print_event(child_idx, depth + 1)

        for root_idx in sorted(roots, key=lambda r: subs[r]["ts"]):
            _print_event(root_idx, 0)


def print_moe_slot_skip_analysis(tasks, skip_threshold_cc=1000):
    """非侵入式分析每个 MoE dynamic slot 中哪些 DMA 节点实际执行了，哪些被 skip。

    判断依据：DMA 类节点（load_gate_up_block, load_down_block 等）
    如果 skip_s1=1 device kernel 只读一下 ctrl 标志就立刻返回，
    duration 极短（< skip_threshold_cc cycles）；
    如果实际运行 DMA，duration 在数万 cc 量级。

    完全基于 bingo_trace.json，不依赖任何 printf 插桩。
    """
    # 只考虑 C2/C3 indiv cluster 的任务
    INDIV_CLUSTERS = {2, 3}
    # DMA 节点的静态 label（来自 verify_deps.short_kernel_name）
    S1_LABELS = {"load_gate_up_block"}
    S3_LABELS = {"load_down_block"}
    S2PF_LABELS = {"prefetch_s2_down"}
    S4PF_LABELS = {"prefetch_s4_next_s1"}
    SLOT_START_LABEL = "gather_s1"  # 每个 slot 的第一个节点

    # per-slot 内节点顺序（用于节点明细表）
    CHAIN_ORDER = [
        "gather_s1",
        "load_gate_up_block",  # [0]
        "compute_gate_up_block",  # [0]
        "load_gate_up_block",  # [1]
        "compute_gate_up_block",  # [1]
        "prefetch_s2_down",
        "compute_gate_up_full",
        "load_down_block",  # [0]
        "compute_down_block",  # [0]
        "load_down_block",  # [1]
        "compute_down_block",  # [1]
        "prefetch_s4_next_s1",
        "compute_down_full",
        "store",
    ]

    # 按 cluster 分组
    cluster_tasks = defaultdict(list)
    for t in tasks:
        loc = t.get("static_location")
        if loc is None:
            continue
        cl = loc[0]
        if cl not in INDIV_CLUSTERS:
            continue
        cluster_tasks[cl].append(t)

    if not cluster_tasks:
        return

    print()
    print("=" * 130)
    print("MOE DYNAMIC SLOT SKIP ANALYSIS (非侵入式：基于节点 duration 推断 skip 行为)")
    print(
        f"  判断阈值: 节点 duration < {skip_threshold_cc} cc → 被 skip（仅检查 ctrl 标志后返回）"
    )
    print(
        f"           节点 duration >= {skip_threshold_cc} cc → 实际执行（DMA 真正传输数据）"
    )
    print("  适用范围: C2/C3 indiv cluster 的 dynamic slot 链")
    print("=" * 130)

    for cl in sorted(cluster_tasks.keys()):
        cl_label = {2: "C2 (indiv_A)", 3: "C3 (indiv_B)"}.get(cl, f"C{cl}")
        sorted_t = sorted(cluster_tasks[cl], key=lambda t: t["exec_start_ns"])

        # 按 cluster 的两个 core 分组（按绝对时间排序）
        core0_tasks = sorted(
            [t for t in sorted_t if t.get("static_location", (0, 99))[1] == 0],
            key=lambda t: t["exec_start_ns"],
        )
        core1_tasks = sorted(
            [t for t in sorted_t if t.get("static_location", (0, 99))[1] == 1],
            key=lambda t: t["exec_start_ns"],
        )

        # 按 static_label 分 slot：每遇到 gather_s1 (core1) 开一个新 slot
        slots = []
        cur_slot_core1 = []
        for t in core1_tasks:
            lbl = t.get("static_label") or t.get("kernel") or ""
            if lbl == SLOT_START_LABEL and cur_slot_core1:
                slots.append(cur_slot_core1)
                cur_slot_core1 = []
            cur_slot_core1.append(t)
        if cur_slot_core1:
            slots.append(cur_slot_core1)

        if not slots:
            print(f"\n  {cl_label}: 未找到以 gather_s1 开头的 slot 链，跳过。")
            continue

        print(f"\n  ─── {cl_label}: {len(slots)} slots ───")
        print()

        hdr = (
            f"    {'Slot':>4s}  {'Seq':>4s}  "
            f"{'skip_s1':>7s}  {'skip_s3':>7s}  {'s2_pf':>5s}  {'s4_pf':>5s}  "
            f"{'S1_dur_cc':>11s}  {'S3_dur_cc':>11s}  "
            f"{'slot_start_ns':>15s}  {'slot_dur_ns':>12s}  {'#nodes':>6s}"
        )
        print(hdr)
        print("    " + "─" * (len(hdr) - 4))

        total_skip_s1 = 0
        total_skip_s3 = 0
        total_s2pf = 0
        total_s4pf = 0

        for slot_idx, slot_nodes in enumerate(slots):
            # slot 时间范围：取本 slot core1 任务的时间 + 对应 core0 任务
            slot_start_ns = slot_nodes[0]["exec_start_ns"]
            slot_end_ns_c1 = max(t["exec_end_ns"] for t in slot_nodes)
            # 估计 core0 对应的任务时间范围（位于 slot 时间段内）
            # slot_idx 决定 core0 任务的 slice（每 slot 消耗 len(core0_per_slot) 个 core0 任务）
            # chain 中 core0 有: compute_gate_up_block×2, compute_gate_up_full,
            #                    compute_down_block×2, compute_down_full = 6 nodes
            CORE0_PER_SLOT = 6
            c0_start = slot_idx * CORE0_PER_SLOT
            c0_end = c0_start + CORE0_PER_SLOT
            slot_core0 = core0_tasks[c0_start:c0_end]
            all_slot_tasks = slot_nodes + slot_core0
            slot_end_ns = (
                max(t["exec_end_ns"] for t in all_slot_tasks)
                if all_slot_tasks
                else slot_end_ns_c1
            )
            slot_dur_ns = slot_end_ns - slot_start_ns

            # 收集各类节点（同时检查 static_label 和 kernel 字段）
            def node_label(t):
                return t.get("static_label") or t.get("kernel") or ""

            s1_nodes = [t for t in slot_nodes if node_label(t) in S1_LABELS]
            s3_nodes = [t for t in slot_nodes if node_label(t) in S3_LABELS]
            s2pf_nodes = [t for t in slot_nodes if node_label(t) in S2PF_LABELS]
            s4pf_nodes = [t for t in slot_nodes if node_label(t) in S4PF_LABELS]

            # skip 判断：该类所有节点 duration 都 < 阈值
            def all_skipped(nodes):
                if not nodes:
                    return None  # 不存在该节点
                return all(t["exec_dur_cc"] < skip_threshold_cc for t in nodes)

            def any_executed(nodes):
                if not nodes:
                    return False
                return any(t["exec_dur_cc"] >= skip_threshold_cc for t in nodes)

            s1_skip = all_skipped(s1_nodes)
            s3_skip = all_skipped(s3_nodes)
            s2pf_ran = any_executed(s2pf_nodes)
            s4pf_ran = any_executed(s4pf_nodes)

            s1_dur_str = (
                "+".join(str(t["exec_dur_cc"]) for t in s1_nodes) if s1_nodes else "—"
            )
            s3_dur_str = (
                "+".join(str(t["exec_dur_cc"]) for t in s3_nodes) if s3_nodes else "—"
            )

            def flag(v, true_str="YES", false_str=" no", none_str="  —"):
                if v is None:
                    return none_str
                return true_str if v else false_str

            skip_s1_str = flag(s1_skip, " SKIP", "  RUN", "   —")
            skip_s3_str = flag(s3_skip, " SKIP", "  RUN", "   —")
            s2pf_str = flag(s2pf_ran, "  yes", "   no", "   —")
            s4pf_str = flag(s4pf_ran, "  yes", "   no", "   —")

            # 验证标记：如果 skip_s1=SKIP 但没有 s4pf 且不是第一个 slot → 可疑
            warn = ""
            if s1_skip and not s4pf_ran and slot_idx > 0:
                prev_slot_nodes = slots[slot_idx - 1]
                prev_slot_core0 = core0_tasks[
                    (slot_idx - 1) * CORE0_PER_SLOT : slot_idx * CORE0_PER_SLOT
                ]
                prev_s4pf = [
                    t
                    for t in prev_slot_nodes + prev_slot_core0
                    if (t.get("static_label") or t.get("kernel") or "") in S4PF_LABELS
                    and t["exec_dur_cc"] >= skip_threshold_cc
                ]
                if not prev_s4pf:
                    warn = "  ⚠ skip_s1 but no prior S4_PF"

            print(
                f"    {slot_idx:4d}  {len(all_slot_tasks):4d}  "
                f"{skip_s1_str}  {skip_s3_str}  {s2pf_str}  {s4pf_str}  "
                f"{s1_dur_str:>11s}  {s3_dur_str:>11s}  "
                f"{slot_start_ns:>15,}  {slot_dur_ns:>12,}{warn}"
            )

            # ── per-node 明细表（slot 内各节点耗时）──
            print(
                f"\n      [Slot {slot_idx} 节点明细]  core1={len(slot_nodes)}nodes  core0={len(slot_core0)}nodes  span={slot_dur_ns:,} ns"
            )
            node_hdr = (
                f"      {'#':>3s}  {'Node':>6s}  {'Core':>4s}  {'Kernel':<26s}"
                f"  {'start_ns':>14s}  {'dur_ns':>10s}  {'dur_cc':>8s}  {'status':>6s}"
            )
            print(node_hdr)
            print("      " + "─" * (len(node_hdr) - 6))

            # 将 core0 和 core1 任务合并后按时间排序
            all_ordered = sorted(all_slot_tasks, key=lambda t: t["exec_start_ns"])
            for ni, nt in enumerate(all_ordered, 1):
                lbl = task_kernel_label(nt)
                core_num = nt.get("static_location", (0, "?"))[1]
                node_id_str = task_node_label(nt)
                status = "SKIP" if nt["exec_dur_cc"] < skip_threshold_cc else " RUN"
                print(
                    f"      {ni:3d}  {node_id_str:>6s}  C{core_num!s:<3s}  {lbl[:26]:<26s}"
                    f"  {nt['exec_start_ns']:>14,}  {nt['exec_dur_ns']:>10,}"
                    f"  {nt['exec_dur_cc']:>8,}  {status}"
                )
            print()

            if s1_skip:
                total_skip_s1 += 1
            if s3_skip:
                total_skip_s3 += 1
            if s2pf_ran:
                total_s2pf += 1
            if s4pf_ran:
                total_s4pf += 1

        n = len(slots)
        print(
            f"    汇总: skip_s1={total_skip_s1}/{n} ({100*total_skip_s1//n}%)"
            f"  skip_s3={total_skip_s3}/{n} ({100*total_skip_s3//n}%)"
            f"  s2_pf={total_s2pf}/{n}  s4_pf={total_s4pf}/{n}"
        )

    print()


def print_dma_cfg_run_breakdown(tasks):
    """分析每类 DMA 和 compute kernel 的 CFG 配置时间与 RUN 执行时间细分。

    DMA 指标（来自 moe_dynamic.h 插桩的子事件）：
      xDMA_CFG  = xdma_memcpy_1d_fast_full_addr() 的 30-CSR-write 阶段（配置开销）
      xDMA_WAIT = xdma_wait_task() 等待 xDMA 传输完成的阶段（实际传输时间）
      iDMA_WAIT = snrt_dma_wait_all() 等待 iDMA 传输完成的阶段（实际传输时间）
      Overhead  = 总时间 - 上述三项（fence / address calc / submit 等）

    Compute 指标（来自内层 dual_vc GEMM sub-events，若已插桩）：
      GEMM_CFG  = VersaCore streamer CSR 配置时间
      GEMM_RUN  = 实际矩阵计算时间（VersaCore busy）
    """
    DMA_KERNELS = frozenset(
        {
            "gather_s1",
            "load_gate_up_block",
            "load_down_block",
            "prefetch_s2_down",
            "prefetch_s4_next_s1",
            "store",
        }
    )
    COMPUTE_KERNELS = frozenset(
        {
            "compute_gate_up_block",
            "compute_gate_up_full",
            "compute_down_block",
            "compute_down_full",
        }
    )

    def sub_sum(subs, keyword):
        return sum(s["dur"] for s in subs if keyword in s["name"])

    def sub_sum_exact(subs, name):
        return sum(s["dur"] for s in subs if s["name"] == name)

    def sub_count(subs, keyword):
        return sum(1 for s in subs if keyword in s["name"])

    def pct(v, total):
        return f"{100.0 * v / total:5.1f}%" if total > 0 else "   — "

    print()
    print("=" * 130)
    print("DMA / COMPUTE  CFG vs RUN  BREAKDOWN  (配置时间 vs 执行时间细分)")
    print()
    print("  DMA kernel 子事件 (需重编 + 重跑仿真后才有数据):")
    print(
        "    xDMA_CFG  = xdma_memcpy_1d_fast_full_addr() 30-CSR-write 阶段  [配置开销]"
    )
    print(
        "    xDMA_WAIT = xdma_wait_task() 等待 xDMA 完成                     [实际传输]"
    )
    print(
        "    iDMA_WAIT = snrt_dma_wait_all() 等待 iDMA 完成                  [实际传输]"
    )
    print(
        "    Overhead  = 总时间 - xDMA_CFG - xDMA_WAIT - iDMA_WAIT           [fence/submit 等]"
    )
    print()
    print("  Compute kernel 子事件 (来自内层 dual_vc GEMM):")
    print("    GEMM_CFG = streamer CSR 配置   GEMM_RUN = 实际矩阵计算")
    print("=" * 130)

    CL_LABEL = {
        0: "C0 (shared_A)",
        1: "C1 (shared_B)",
        2: "C2 (indiv_A)",
        3: "C3 (indiv_B)",
    }
    DMA_ORDER = [
        "gather_s1",
        "load_gate_up_block",
        "load_down_block",
        "prefetch_s2_down",
        "prefetch_s4_next_s1",
        "store",
    ]
    CMP_ORDER = [
        "compute_gate_up_block",
        "compute_gate_up_full",
        "compute_down_block",
        "compute_down_full",
    ]
    CMP_DEVICE_MARKER = {
        "compute_gate_up_block": "BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP",
        "compute_gate_up_full": "BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP_FULL",
        "compute_down_block": "BINGO_TRACE_DEV_MOE_COMPUTE_DOWN",
        "compute_down_full": "BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_FULL",
    }

    for cl in [0, 1, 2, 3]:
        cl_tasks = [t for t in tasks if t.get("static_location", (None,))[0] == cl]
        if not cl_tasks:
            continue
        dma_cl = [t for t in cl_tasks if task_kernel_label(t) in DMA_KERNELS]
        cmp_cl = [t for t in cl_tasks if task_kernel_label(t) in COMPUTE_KERNELS]
        if not dma_cl and not cmp_cl:
            continue

        print(f"\n  ══ {CL_LABEL.get(cl, f'C{cl}')} ══")

        # ── DMA kernels ──────────────────────────────────────────────────────
        if dma_cl:
            print()
            print(f"  [DMA kernels]")
            hdr = (
                f"    {'Kernel':<26s}  {'N':>4s}  {'Total_ns':>10s}  "
                f"{'xCFG_ns':>10s}({'xCFG%':>5s})  "
                f"{'xWAIT_ns':>10s}({'xWT%':>4s})  "
                f"{'iWAIT_ns':>10s}({'iWT%':>4s})  "
                f"{'OH_ns':>10s}({'OH%':>4s})"
            )
            print(hdr)
            print(f"    {'─' * (len(hdr) - 4)}")

            for kname in DMA_ORDER:
                kt = [t for t in dma_cl if task_kernel_label(t) == kname]
                if not kt:
                    continue
                total_ns = sum(t["exec_dur_ns"] for t in kt)
                xcfg_ns = sum(sub_sum(t["sub_events"], "DMA_XDMA_CFG") for t in kt)
                xwait_ns = sum(sub_sum(t["sub_events"], "DMA_XDMA_WAIT") for t in kt)
                iwait_ns = sum(sub_sum(t["sub_events"], "DMA_IDMA_WAIT") for t in kt)
                oh_ns = total_ns - xcfg_ns - xwait_ns - iwait_ns

                no_data = xcfg_ns == 0 and xwait_ns == 0 and iwait_ns == 0
                note = "  ← 无子事件，需重编仿真" if no_data else ""
                print(
                    f"    {kname:<26s}  {len(kt):4d}  {total_ns:>10,}  "
                    f"{xcfg_ns:>10,}({pct(xcfg_ns,total_ns)})  "
                    f"{xwait_ns:>10,}({pct(xwait_ns,total_ns)})  "
                    f"{iwait_ns:>10,}({pct(iwait_ns,total_ns)})  "
                    f"{oh_ns:>10,}({pct(oh_ns,total_ns)}){note}"
                )

                if no_data:
                    continue

                # Per-instance 明细（仅当数量 <= 20 时）
                if len(kt) <= 20:
                    for t in sorted(kt, key=lambda x: x["exec_start_ns"]):
                        t_xcfg = sub_sum(t["sub_events"], "DMA_XDMA_CFG")
                        t_xwait = sub_sum(t["sub_events"], "DMA_XDMA_WAIT")
                        t_iwait = sub_sum(t["sub_events"], "DMA_IDMA_WAIT")
                        t_oh = t["exec_dur_ns"] - t_xcfg - t_xwait - t_iwait
                        n_xcfg = sub_count(t["sub_events"], "DMA_XDMA_CFG")
                        n_xwait = sub_count(t["sub_events"], "DMA_XDMA_WAIT")
                        n_iwait = sub_count(t["sub_events"], "DMA_IDMA_WAIT")
                        node_s = f"N{t.get('node_id', '?'):3}"
                        print(
                            f"      {node_s}  {t['exec_dur_cc']:>8} cc  "
                            f"xCFG×{n_xcfg}={t_xcfg:>8,}ns  "
                            f"xWAIT×{n_xwait}={t_xwait:>8,}ns  "
                            f"iWAIT×{n_iwait}={t_iwait:>8,}ns  "
                            f"OH={t_oh:>8,}ns"
                        )
            print()

            # 聚合统计：CFG 开销 vs 传输时间 vs 调度开销
            all_xcfg = sum(sub_sum(t["sub_events"], "DMA_XDMA_CFG") for t in dma_cl)
            all_xwait = sum(sub_sum(t["sub_events"], "DMA_XDMA_WAIT") for t in dma_cl)
            all_iwait = sum(sub_sum(t["sub_events"], "DMA_IDMA_WAIT") for t in dma_cl)
            all_total = sum(t["exec_dur_ns"] for t in dma_cl)
            all_oh = all_total - all_xcfg - all_xwait - all_iwait
            if all_total > 0 and (all_xcfg + all_xwait + all_iwait) > 0:
                xfer_ns = all_xwait + all_iwait
                print(
                    f"    ★ C{cl} DMA 汇总: 配置开销(xCFG)={pct(all_xcfg,all_total)}  "
                    f"传输等待(xWAIT+iWAIT)={pct(xfer_ns,all_total)}  "
                    f"其他开销={pct(all_oh,all_total)}"
                )
                if all_xcfg > 0:
                    per_call_cfg = (
                        all_xcfg
                        / sub_count(
                            [s for t in dma_cl for s in t["sub_events"]], "DMA_XDMA_CFG"
                        )
                        if sub_count(
                            [s for t in dma_cl for s in t["sub_events"]], "DMA_XDMA_CFG"
                        )
                        > 0
                        else 0
                    )
                    print(
                        f"      xDMA 单次 CFG 平均: {per_call_cfg:,.0f} ns  "
                        f"≈ {per_call_cfg/12:.0f} cc  (理论 30×CSR_write)"
                    )
            print()

        # ── Compute kernels ───────────────────────────────────────────────────
        if cmp_cl:
            print(f"  [Compute kernels]")
            has_gemm_subs = any(
                any(
                    "GEMM_FULL_CFG" in s["name"] or "GEMM_FULL_RUN" in s["name"]
                    for s in t["sub_events"]
                )
                for t in cmp_cl
            )
            if not has_gemm_subs:
                print(
                    f"    (未检测到 GEMM_FULL_CFG/RUN 子事件 — 内层 dual_vc kernel 未单独插桩)"
                )
                print(
                    f"    (仅展示总时间；outer START/END marker 已覆盖整个 GEMM 调用)"
                )
                for kname in CMP_ORDER:
                    kt = [t for t in cmp_cl if task_kernel_label(t) == kname]
                    if not kt:
                        continue
                    total_ns = sum(t["exec_dur_ns"] for t in kt)
                    total_cc = sum(t["exec_dur_cc"] for t in kt)
                    print(
                        f"    {kname:<26s}  ×{len(kt):2d}  "
                        f"total={total_ns:>10,}ns  ({total_cc:>8,}cc total, "
                        f"avg={total_cc//len(kt):>8,}cc)"
                    )
            else:
                hdr = (
                    f"    {'Kernel':<26s}  {'N':>4s}  {'Node_ns':>10s}  "
                    f"{'DEV_ns':>10s}  {'EntryGap':>10s}  "
                    f"{'Arg_ns':>10s}({'Arg/Gap':>7s})  "
                    f"{'CFG_ns':>10s}({'CFG/DEV':>7s})  "
                    f"{'RUN_ns':>10s}({'RUN/DEV':>7s})  "
                    f"{'DevOH_ns':>10s}({'OH/DEV':>6s})"
                )
                print(hdr)
                print(f"    {'─' * (len(hdr) - 4)}")
                for kname in CMP_ORDER:
                    kt = [t for t in cmp_cl if task_kernel_label(t) == kname]
                    if not kt:
                        continue
                    total_ns = sum(t["exec_dur_ns"] for t in kt)
                    dev_marker = CMP_DEVICE_MARKER.get(kname)
                    dev_ns = (
                        sum(sub_sum_exact(t["sub_events"], dev_marker) for t in kt)
                        if dev_marker
                        else 0
                    )
                    arg_ns = sum(sub_sum_exact(t["sub_events"], "BINGO_TRACE_KERNEL_ARG_PARSE") for t in kt)
                    cfg_ns = sum(sub_sum(t["sub_events"], "GEMM_FULL_CFG") for t in kt)
                    run_ns = sum(sub_sum(t["sub_events"], "GEMM_FULL_RUN") for t in kt)
                    entry_gap_ns = total_ns - dev_ns if dev_ns > 0 else 0
                    dev_oh_ns = dev_ns - cfg_ns - run_ns if dev_ns > 0 else total_ns - cfg_ns - run_ns
                    print(
                        f"    {kname:<26s}  {len(kt):4d}  {total_ns:>10,}  "
                        f"{dev_ns:>10,}  {entry_gap_ns:>10,}  "
                        f"{arg_ns:>10,}({pct(arg_ns,entry_gap_ns)})  "
                        f"{cfg_ns:>10,}({pct(cfg_ns,dev_ns)})  "
                        f"{run_ns:>10,}({pct(run_ns,dev_ns)})  "
                        f"{dev_oh_ns:>10,}({pct(dev_oh_ns,dev_ns)})"
                    )
                print(
                    "    NOTE: Node_ns is the real Bingo node execution time (RUN_KERNEL start→end). "
                    "DEV/Arg/CFG/RUN only break down where that node time is spent; they do not redefine node start/end."
                )
            print()


def print_phase_transition_analysis(tasks, phase_events):
    """分析 phase 切换开销：Exit→Decision→Setup→Scatter→Entry 完整 breakdown。"""
    print()
    print("=" * 130)
    print("PHASE TRANSITION ANALYSIS (阶段切换开销分析)")
    print("=" * 130)
    print()

    if not phase_events:
        print("  未检测到 PHASE_* trace 事件，跳过。")
        print()
        return

    # 打印所有 phase events
    print("  Phase-level events (CVA6 软件开销):")
    print(
        f"    {'#':>3s}  {'Label':<16s}  {'start_ns':>14s}  {'end_ns':>14s}  {'dur_ns':>10s}  {'dur_cc':>8s}"
    )
    print("    " + "─" * 75)
    for i, pe in enumerate(phase_events, 1):
        print(
            f"    {i:3d}  {pe['label']:<16s}  {pe['start_ns']:>14,}  {pe['end_ns']:>14,}"
            f"  {pe['dur_ns']:>10,}  {pe['dur_cc']:>8,}"
        )
    print()

    # 按 phase 过渡分组 (Decision + Setup + Scatter = 一次 phase switch)
    # 找所有 Exit tasks 和对应的下一个 Entry
    sorted_tasks = sorted(tasks, key=lambda t: t["exec_start_ns"])
    exits = [
        (i, t)
        for i, t in enumerate(sorted_tasks)
        if t["kernel"] == "Exit" or task_kernel_label(t) == "exit"
    ]
    entries = [
        (i, t)
        for i, t in enumerate(sorted_tasks)
        if t["kernel"] == "Entry" or task_kernel_label(t) == "entry"
    ]

    print("  Phase 切换开销 breakdown:")
    print(
        f"    {'过渡':>12s}  {'Exit_end':>14s}  {'Entry_start':>14s}  {'总GAP':>10s}"
        f"  {'Decision':>10s}  {'Setup':>10s}  {'Scatter':>10s}  {'其他':>10s}"
    )
    print("    " + "─" * 105)

    total_gap = 0
    total_decision = 0
    total_setup = 0
    total_scatter = 0
    transition_count = 0

    for exit_idx, (ei, exit_task) in enumerate(exits):
        exit_end = exit_task["exec_end_ns"]
        # 找紧随其后的 Entry
        next_entry = None
        for entry_idx, entry_task in entries:
            if entry_task["exec_start_ns"] > exit_end:
                next_entry = entry_task
                break
        if next_entry is None:
            continue

        gap = next_entry["exec_start_ns"] - exit_end
        if gap < 50000:  # gaps < 50us 不是 phase switch
            continue

        # 找这个 gap 窗口内的 phase events
        decision_dur = 0
        setup_dur = 0
        scatter_dur = 0
        for pe in phase_events:
            if (
                pe["start_ns"] >= exit_end
                and pe["end_ns"] <= next_entry["exec_start_ns"] + 200000
            ):
                if pe["label"] == "PhaseDecision":
                    decision_dur += pe["dur_ns"]
                elif pe["label"] == "PhaseSetup":
                    setup_dur += pe["dur_ns"]
                elif pe["label"] == "PhaseScatter":
                    scatter_dur += pe["dur_ns"]

        other = gap - decision_dur - setup_dur - scatter_dur
        total_gap += gap
        total_decision += decision_dur
        total_setup += setup_dur
        total_scatter += scatter_dur
        transition_count += 1

        label = f"Switch #{transition_count}"
        print(
            f"    {label:>12s}  {exit_end:>14,}  {next_entry['exec_start_ns']:>14,}  {gap:>10,}"
            f"  {decision_dur:>10,}  {setup_dur:>10,}  {scatter_dur:>10,}  {other:>10,}"
        )

    if transition_count > 0:
        print("    " + "─" * 105)
        other_total = total_gap - total_decision - total_setup - total_scatter
        print(
            f"    {'TOTAL':>12s}  {'':>14s}  {'':>14s}  {total_gap:>10,}"
            f"  {total_decision:>10,}  {total_setup:>10,}  {total_scatter:>10,}  {other_total:>10,}"
        )

        # 获取全局 span 做百分比
        all_start = sorted_tasks[0]["exec_start_ns"]
        all_end = max(t["exec_end_ns"] for t in sorted_tasks)
        global_span = all_end - all_start
        print()
        print(f"    Phase 切换总开销: {total_gap:,} ns = {total_gap/1000:.1f} us")
        print(
            f"    占全局时间跨度 ({global_span:,} ns) 的 {100*total_gap/global_span:.1f}%"
        )
        print(f"    其中:")
        print(
            f"      Decision (读token count):  {total_decision:>10,} ns ({100*total_decision/total_gap:.1f}%)"
        )
        print(
            f"      Setup (构建task graph):    {total_setup:>10,} ns ({100*total_setup/total_gap:.1f}%)"
        )
        print(
            f"      Scatter (DMA scatter pad): {total_scatter:>10,} ns ({100*total_scatter/total_gap:.1f}%)"
        )
        print(
            f"      其他 (fence/reinit/etc):   {other_total:>10,} ns ({100*other_total/total_gap:.1f}%)"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze Bingo Perfetto trace JSON.")
    parser.add_argument(
        "trace", nargs="?", default=DEFAULT_TRACE, help="Path to bingo_trace.json"
    )
    parser.add_argument(
        "--workload-dir",
        default=None,
        help="Workload directory with final_dfg.csv and offload_bingo_hw.h "
        "(默认: 自动从 uart log 检测，回退到 multi_cluster_MoE)",
    )
    parser.add_argument(
        "--uart-log",
        default=DEFAULT_UART_LOG,
        help="uart log 路径，用于自动检测 workload 名称",
    )
    parser.add_argument(
        "--no-static-map",
        action="store_true",
        help="Do not infer node/kernel labels from generated DFG/header artifacts",
    )
    parser.add_argument(
        "--base-hart-id",
        type=int,
        default=1,
        help="First SNAX hart id used to map trace Core N to cluster/core",
    )
    parser.add_argument(
        "--cores-per-cluster",
        type=int,
        default=2,
        help="SNAX harts per cluster for static trace mapping",
    )
    parser.add_argument(
        "--strict-static-map",
        action="store_true",
        help="Warn if a traced core consumes only part of its static stream",
    )
    args = parser.parse_args()

    # 自动检测 workload dir：用户未指定时从 uart log 推断
    if args.workload_dir is None:
        detected = detect_workload_from_uart(args.uart_log)
        if detected:
            print(
                f"[INFO] 自动检测到 workload: {os.path.basename(detected)}",
                file=sys.stderr,
            )
            args.workload_dir = detected
        else:
            print(
                f"[WARN] 无法从 uart log 检测 workload，回退到默认: {DEFAULT_WORKLOAD_DIR}",
                file=sys.stderr,
            )
            args.workload_dir = DEFAULT_WORKLOAD_DIR

    trace_path = args.trace
    if not os.path.exists(trace_path):
        print(f"ERROR: 找不到 {trace_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Trace file: {trace_path}")
    print()

    events = parse_bingo_trace(trace_path)
    print(f"解析到 {len(events)} 个 duration 事件")

    tasks = build_task_cycles(events)
    print(f"提取到 {len(tasks)} 个任务 (MGR_RUN_KERNEL)")
    if not tasks:
        print("没有检测到 MGR_RUN_KERNEL duration 事件，无法生成任务级 timeline。")
        return

    static_map = None
    mapping_warnings = []
    if not args.no_static_map:
        try:
            static_map = load_static_mapping(args.workload_dir)
            mapping_warnings = annotate_tasks_with_static_nodes(
                tasks,
                static_map,
                base_hart_id=args.base_hart_id,
                cores_per_cluster=args.cores_per_cluster,
                strict=args.strict_static_map,
            )
        except Exception as exc:
            print(f"WARNING: 静态 DFG/header 映射加载失败: {exc}", file=sys.stderr)
            annotate_tasks_with_static_nodes(tasks, None)
    else:
        annotate_tasks_with_static_nodes(tasks, None)

    phase_events = extract_phase_events(events)
    if phase_events:
        print(f"提取到 {len(phase_events)} 个 PHASE 级别事件")

    # 核心 → 事件数统计
    core_counts = Counter(t["tid"] for t in tasks)
    for tid, cnt in sorted(core_counts.items()):
        print(f"  {tid}: {cnt} tasks")
    print()

    # Kernel 类型统计
    kernel_counts = Counter(t["kernel"] for t in tasks)
    print("  Kernel 类型分布:")
    for kname, cnt in kernel_counts.most_common():
        total_dur = sum(t["exec_dur_ns"] for t in tasks if t["kernel"] == kname)
        print(f"    {kname:<14s}: {cnt:3d} 次, 总耗时 {total_dur:>12,} ns")
    print()

    static_kernel_counts = Counter(task_kernel_label(t) for t in tasks)
    if any(t.get("node_id") is not None for t in tasks):
        print("  静态 Kernel/Node 映射分布:")
        for kname, cnt in static_kernel_counts.most_common(20):
            total_dur = sum(
                t["exec_dur_ns"] for t in tasks if task_kernel_label(t) == kname
            )
            print(f"    {kname[:32]:<32s}: {cnt:3d} 次, 总耗时 {total_dur:>12,} ns")
        print()

    print_static_mapping_summary(tasks, static_map, mapping_warnings)

    # ─── 输出 ───
    print_unified_timeline(tasks)
    print_per_core_timeline(tasks)
    print_gantt(tasks)
    print_moe_scheduling_breakdown(tasks, events)
    print_moe_slot_skip_analysis(tasks)
    print_dma_cfg_run_breakdown(tasks)
    print_phase_transition_analysis(tasks, phase_events)
    print_utilization(tasks)
    print_versacore_efficiency_analysis(tasks)
    print_scheduling_overhead(tasks)
    print_sub_event_detail(tasks)


if __name__ == "__main__":
    main()
