#!/usr/bin/env python3
"""Verify generated Bingo DFG/header consistency for multi_cluster_MoE.

This checker intentionally reads generated artifacts instead of maintaining a
hand-written node table. The current MoE lowering creates many dynamic slot
nodes plus compiler-generated dummy nodes, so static validation must follow the
generated CSV/header exactly.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WORKLOAD_DIR = os.path.join(
    SCRIPT_DIR,
    "HeMAiA/target/sw/host/apps/offload_bingo_hw/single_chip/workloads/multi_cluster_MoE",
)


@dataclass(frozen=True)
class DfgNode:
    node_id: int
    chiplet: str
    cluster: int
    core: int
    node_type: str
    kernel: str


@dataclass(frozen=True)
class TaskDescriptor:
    chiplet_list: str
    index: int
    node_id: int
    value: int
    desc_type: int
    task_id: int
    assigned_chiplet: str
    assigned_cluster: int
    assigned_core: int
    dep_check_en: int
    dep_check_code: str
    dep_set_en: int
    dep_set_all: int
    dep_set_chiplet: str
    dep_set_cluster: int
    dep_set_code: str


@dataclass(frozen=True)
class DevTaskMapEntry:
    node_id: int
    dev_task_id: int
    comment: str


EXPECTED_DYNAMIC_CHAIN: Tuple[Tuple[str, int], ...] = (
    ("__snax_bingo_kernel_moe_dynamic_expert_gather_s1", 1),
    ("__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block", 1),
    ("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block", 0),
    ("__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block", 1),
    ("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block", 0),
    ("__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down", 1),
    ("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full", 0),
    ("__snax_bingo_kernel_moe_dynamic_expert_load_down_block", 1),
    ("__snax_bingo_kernel_moe_dynamic_expert_compute_down_block", 0),
    ("__snax_bingo_kernel_moe_dynamic_expert_load_down_block", 1),
    ("__snax_bingo_kernel_moe_dynamic_expert_compute_down_block", 0),
    ("__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1", 1),
    ("__snax_bingo_kernel_moe_dynamic_expert_compute_down_full", 0),
    ("__snax_bingo_kernel_moe_dynamic_expert_store", 1),
)

PHASE_NODE_KERNELS = {
    26: "__host_bingo_kernel_moe_router_schedule",
    27: "__host_bingo_kernel_moe_prepare_request",
    28: "__host_bingo_kernel_moe_execute",
}


def default_paths(workload_dir: str) -> Tuple[str, str]:
    return (
        os.path.join(workload_dir, "final_dfg.csv"),
        os.path.join(workload_dir, "offload_bingo_hw.h"),
    )


def parse_final_dfg(csv_path: str) -> Dict[int, DfgNode]:
    nodes: Dict[int, DfgNode] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"ID", "Chiplet", "Cluster", "Core", "Type", "Kernel"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"{csv_path} does not look like final_dfg.csv")
        for row in reader:
            node_id = int(row["ID"])
            nodes[node_id] = DfgNode(
                node_id=node_id,
                chiplet=row["Chiplet"],
                cluster=int(row["Cluster"]),
                core=int(row["Core"]),
                node_type=row["Type"],
                kernel=row["Kernel"],
            )
    return nodes


def _require_match(pattern: str, line: str, context: str) -> re.Match[str]:
    match = re.search(pattern, line)
    if not match:
        raise ValueError(f"Cannot parse {context}: {line.rstrip()}")
    return match


def parse_task_descriptors(header_path: str) -> List[TaskDescriptor]:
    with open(header_path) as f:
        lines = f.readlines()

    descriptors: List[TaskDescriptor] = []
    assign_re = re.compile(
        r"bingo_hw_scheduler_task_desc_list_chip_(\d+)\[(\d+)\]\s*=\s*"
        r"(0x[0-9A-Fa-f]+);\s*//\s*Node ID\s+(\d+)"
    )

    for i, line in enumerate(lines):
        match = assign_re.search(line)
        if not match:
            continue
        if i + 4 >= len(lines):
            raise ValueError(f"Descriptor at line {i + 1} is truncated")

        fields = _require_match(
            r"Fields:\s*Type=(\d+),\s*TaskID=(\d+)",
            lines[i + 1],
            f"descriptor fields after line {i + 1}",
        )
        assigned = _require_match(
            r"Assigned:\s*Chiplet=([0-9A-Fa-f]+),\s*Cluster=(\d+),\s*Core=(\d+)",
            lines[i + 2],
            f"descriptor assignment after line {i + 1}",
        )
        dep_check = _require_match(
            r"DepCheck:\s*En=(\d+),\s*Code=(0b[01]+)",
            lines[i + 3],
            f"descriptor dep_check after line {i + 1}",
        )
        dep_set = _require_match(
            r"DepSet:\s*En=(\d+),\s*All=(\d+),\s*Chiplet=([0-9A-Fa-f]+),"
            r"\s*Cluster=(\d+),\s*Code=(0b[01]+)",
            lines[i + 4],
            f"descriptor dep_set after line {i + 1}",
        )

        descriptors.append(
            TaskDescriptor(
                chiplet_list=match.group(1),
                index=int(match.group(2)),
                node_id=int(match.group(4)),
                value=int(match.group(3), 16),
                desc_type=int(fields.group(1)),
                task_id=int(fields.group(2)),
                assigned_chiplet=assigned.group(1),
                assigned_cluster=int(assigned.group(2)),
                assigned_core=int(assigned.group(3)),
                dep_check_en=int(dep_check.group(1)),
                dep_check_code=dep_check.group(2),
                dep_set_en=int(dep_set.group(1)),
                dep_set_all=int(dep_set.group(2)),
                dep_set_chiplet=dep_set.group(3),
                dep_set_cluster=int(dep_set.group(4)),
                dep_set_code=dep_set.group(5),
            )
        )
    return descriptors


def parse_dev_task_map(header_path: str) -> Dict[int, DevTaskMapEntry]:
    mapping: Dict[int, DevTaskMapEntry] = {}
    map_re = re.compile(
        r"global_task_id_to_dev_task_id_chip_\d+\[(\d+)\]\s*=\s*(-?\d+);"
        r"\s*//\s*Node ID\s+(\d+)(?:\s*->\s*Dev Task\s*(\d+))?\s*\(([^)]*)\)"
    )
    with open(header_path) as f:
        for line in f:
            match = map_re.search(line)
            if not match:
                continue
            array_idx = int(match.group(1))
            dev_task_id = int(match.group(2))
            node_id = int(match.group(3))
            if array_idx != node_id:
                raise ValueError(
                    f"global_task_id_to_dev_task_id index {array_idx} != Node ID {node_id}"
                )
            mapping[node_id] = DevTaskMapEntry(
                node_id=node_id,
                dev_task_id=dev_task_id,
                comment=match.group(5),
            )
    return mapping


def parse_nodes_from_header(
    header_path: str,
    descriptors: Optional[List[TaskDescriptor]] = None,
    dev_map: Optional[Dict[int, DevTaskMapEntry]] = None,
) -> Dict[int, DfgNode]:
    """Recover node placement and kernel names from a generated Bingo header.

    The generated dev-task map comments retain the final node location and
    kernel even when ``final_dfg.csv`` is no longer present.
    """
    if descriptors is None:
        descriptors = parse_task_descriptors(header_path)
    if dev_map is None:
        dev_map = parse_dev_task_map(header_path)

    descriptor_map = {desc.node_id: desc for desc in descriptors}
    comment_re = re.compile(
        r"^Node_ID\d+_Chiplet([0-9A-Fa-f]+)_Cluster(\d+)_Core(\d+)_Kernel(.*)$"
    )
    nodes: Dict[int, DfgNode] = {}
    for node_id, entry in dev_map.items():
        descriptor = descriptor_map.get(node_id)
        match = comment_re.match(entry.comment)
        if descriptor is None or match is None:
            continue
        nodes[node_id] = DfgNode(
            node_id=node_id,
            chiplet=match.group(1),
            cluster=int(match.group(2)),
            core=int(match.group(3)),
            node_type="normal" if descriptor.desc_type == 0 else "dummy",
            kernel=match.group(4),
        )
    return nodes


def node_is_device_kernel(node: DfgNode) -> bool:
    return node.kernel.startswith("__snax_")


def node_is_host_kernel(node: DfgNode) -> bool:
    return node.kernel.startswith("__host_")


def descriptor_by_node(
    descriptors: Iterable[TaskDescriptor],
) -> Dict[int, TaskDescriptor]:
    result: Dict[int, TaskDescriptor] = {}
    for desc in descriptors:
        if desc.node_id in result:
            raise ValueError(f"Duplicate descriptor for node {desc.node_id}")
        result[desc.node_id] = desc
    return result


def validate_artifacts(
    nodes: Dict[int, DfgNode],
    descriptors: List[TaskDescriptor],
    dev_map: Dict[int, DevTaskMapEntry],
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if not nodes:
        errors.append("final_dfg.csv contains no nodes")
        return errors, warnings

    max_node_id = max(nodes)
    missing_node_ids = [
        node_id for node_id in range(max_node_id + 1) if node_id not in nodes
    ]
    if missing_node_ids:
        errors.append(
            f"final_dfg.csv node IDs are not contiguous: missing {missing_node_ids[:10]}"
        )

    try:
        desc_by_node = descriptor_by_node(descriptors)
    except ValueError as exc:
        errors.append(str(exc))
        desc_by_node = {}

    missing_desc = sorted(set(nodes) - set(desc_by_node))
    extra_desc = sorted(set(desc_by_node) - set(nodes))
    if missing_desc:
        errors.append(f"Missing task descriptors for nodes: {missing_desc[:20]}")
    if extra_desc:
        errors.append(
            f"Descriptors reference nodes absent from final_dfg.csv: {extra_desc[:20]}"
        )

    missing_map = sorted(set(nodes) - set(dev_map))
    extra_map = sorted(set(dev_map) - set(nodes))
    if missing_map:
        errors.append(
            f"Missing global_task_id_to_dev_task_id entries: {missing_map[:20]}"
        )
    if extra_map:
        errors.append(f"Dev task map references unknown nodes: {extra_map[:20]}")

    for node_id, node in nodes.items():
        desc = desc_by_node.get(node_id)
        if desc:
            if desc.task_id != node_id:
                errors.append(
                    f"Node {node_id}: descriptor TaskID={desc.task_id}, expected {node_id}"
                )
            expected_desc_type = 0 if node.node_type == "normal" else 1
            if desc.desc_type != expected_desc_type:
                errors.append(
                    f"Node {node_id}: descriptor Type={desc.desc_type}, "
                    f"expected {expected_desc_type} for CSV Type={node.node_type}"
                )
            if (
                desc.assigned_chiplet != node.chiplet
                or desc.assigned_cluster != node.cluster
                or desc.assigned_core != node.core
            ):
                errors.append(
                    f"Node {node_id}: descriptor assignment "
                    f"C{desc.assigned_cluster}/Core{desc.assigned_core} "
                    f"does not match CSV C{node.cluster}/Core{node.core}"
                )

        map_entry = dev_map.get(node_id)
        if map_entry:
            if node.node_type != "normal":
                if map_entry.dev_task_id != -1:
                    errors.append(
                        f"Node {node_id}: dummy node maps to dev task {map_entry.dev_task_id}"
                    )
            elif node_is_device_kernel(node):
                if map_entry.dev_task_id < 0:
                    errors.append(
                        f"Node {node_id}: device kernel {node.kernel} maps to -1"
                    )
            elif node_is_host_kernel(node):
                if map_entry.dev_task_id != -1:
                    errors.append(
                        f"Node {node_id}: host kernel {node.kernel} maps to device task"
                    )

    for node_id, expected_kernel in PHASE_NODE_KERNELS.items():
        node = nodes.get(node_id)
        if not node:
            errors.append(f"Missing phase node {node_id} ({expected_kernel})")
        elif node.kernel != expected_kernel:
            errors.append(
                f"Node {node_id}: kernel={node.kernel}, expected {expected_kernel}"
            )

    phase_indices = {
        node_id: desc_by_node[node_id].index
        for node_id in PHASE_NODE_KERNELS
        if node_id in desc_by_node
    }
    if set(phase_indices) == set(PHASE_NODE_KERNELS):
        phase_ids = sorted(PHASE_NODE_KERNELS.keys())
        if not (
            phase_indices[phase_ids[0]]
            < phase_indices[phase_ids[1]]
            < phase_indices[phase_ids[2]]
        ):
            errors.append(
                f"Phase descriptor order is not router->prepare->execute: {phase_indices}"
            )

    dynamic_errors, dynamic_warnings = validate_dynamic_slot_chains(nodes, desc_by_node)
    errors.extend(dynamic_errors)
    warnings.extend(dynamic_warnings)

    return errors, warnings


def validate_dynamic_slot_chains(
    nodes: Dict[int, DfgNode], desc_by_node: Dict[int, TaskDescriptor]
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    dynamic_nodes = [
        node
        for node in sorted(nodes.values(), key=lambda item: item.node_id)
        if "__snax_bingo_kernel_moe_dynamic_expert_" in node.kernel
    ]

    if not dynamic_nodes:
        errors.append("No dynamic expert nodes found")
        return errors, warnings

    first_dynamic = dynamic_nodes[0].node_id
    last_dynamic = dynamic_nodes[-1].node_id
    expected_ids = list(range(first_dynamic, last_dynamic + 1))
    actual_ids = [node.node_id for node in dynamic_nodes]
    if actual_ids != expected_ids:
        errors.append(
            "Dynamic expert nodes are not a contiguous ID range: "
            f"first={first_dynamic}, last={last_dynamic}, count={len(actual_ids)}"
        )

    chain_len = len(EXPECTED_DYNAMIC_CHAIN)
    if len(dynamic_nodes) % chain_len != 0:
        errors.append(
            f"Dynamic node count {len(dynamic_nodes)} is not divisible by chain length {chain_len}"
        )
        return errors, warnings

    chain_count = len(dynamic_nodes) // chain_len
    cluster_counter = Counter()
    for chain_idx in range(chain_count):
        group = dynamic_nodes[chain_idx * chain_len : (chain_idx + 1) * chain_len]
        clusters = {node.cluster for node in group}
        if len(clusters) != 1:
            errors.append(
                f"Dynamic chain {chain_idx}: nodes span multiple clusters {sorted(clusters)}"
            )
            continue
        cluster = group[0].cluster
        cluster_counter[cluster] += 1
        if cluster not in (2, 3):
            warnings.append(
                f"Dynamic chain {chain_idx}: expected cluster 2/3, got {cluster}"
            )

        for offset, (expected_kernel, expected_core) in enumerate(
            EXPECTED_DYNAMIC_CHAIN
        ):
            node = group[offset]
            if node.kernel != expected_kernel:
                errors.append(
                    f"Dynamic chain {chain_idx} offset {offset}: kernel={node.kernel}, "
                    f"expected {expected_kernel}"
                )
            if node.core != expected_core:
                errors.append(
                    f"Dynamic chain {chain_idx} offset {offset} node {node.node_id}: "
                    f"Core{node.core}, expected Core{expected_core}"
                )
            desc = desc_by_node.get(node.node_id)
            if desc and desc.dep_set_en and not desc.dep_set_all:
                is_store_to_host_join = (
                    node.kernel.endswith("_store") and desc.dep_set_cluster == 0
                )
                if desc.dep_set_cluster != node.cluster and not is_store_to_host_join:
                    errors.append(
                        f"Dynamic node {node.node_id}: DepSet cluster {desc.dep_set_cluster} "
                        f"does not match assigned cluster {node.cluster}"
                    )

    if set(cluster_counter) != {2, 3}:
        warnings.append(
            f"Dynamic chains are not split across clusters 2 and 3: {dict(cluster_counter)}"
        )
    elif cluster_counter[2] != cluster_counter[3]:
        warnings.append(f"Dynamic chain count is imbalanced: {dict(cluster_counter)}")

    return errors, warnings


def build_expected_device_streams(
    nodes: Dict[int, DfgNode],
    descriptors: List[TaskDescriptor],
    dev_map: Dict[int, DevTaskMapEntry],
) -> Dict[Tuple[int, int], List[DfgNode]]:
    """Return expected per-(cluster, core) device run streams in descriptor order."""
    streams: Dict[Tuple[int, int], List[DfgNode]] = defaultdict(list)
    for desc in sorted(descriptors, key=lambda item: item.index):
        node = nodes.get(desc.node_id)
        map_entry = dev_map.get(desc.node_id)
        if not node or not map_entry:
            continue
        if node.node_type != "normal" or map_entry.dev_task_id < 0:
            continue
        if not node_is_device_kernel(node):
            continue
        streams[(node.cluster, node.core)].append(node)
    return dict(streams)


def short_kernel_name(kernel: str) -> str:
    for prefix in (
        "__snax_bingo_kernel_moe_dynamic_expert_",
        "__snax_bingo_kernel_",
        "__host_bingo_kernel_",
    ):
        if kernel.startswith(prefix):
            return kernel[len(prefix) :]
    return kernel


def print_summary(
    nodes: Dict[int, DfgNode],
    descriptors: List[TaskDescriptor],
    dev_map: Dict[int, DevTaskMapEntry],
    errors: List[str],
    warnings: List[str],
    verbose: bool = False,
) -> None:
    print("=" * 100)
    print("GENERATED DFG / HEADER CONSISTENCY CHECK")
    print("=" * 100)
    print(f"DFG nodes:              {len(nodes)}")
    print(f"Task descriptors:       {len(descriptors)}")
    print(f"Dev-task map entries:   {len(dev_map)}")
    print()

    node_type_counts = Counter(node.node_type for node in nodes.values())
    kernel_counts = Counter(
        short_kernel_name(node.kernel) for node in nodes.values() if node.kernel
    )
    print("Node type counts:")
    for name, count in sorted(node_type_counts.items()):
        print(f"  {name:<12s} {count:5d}")
    print()

    print("Most common kernels:")
    for kernel, count in kernel_counts.most_common(12):
        print(f"  {kernel:<42s} {count:5d}")
    print()

    streams = build_expected_device_streams(nodes, descriptors, dev_map)
    print("Expected device streams from descriptor order:")
    for (cluster, core), stream in sorted(streams.items()):
        dynamic_count = sum("moe_dynamic_expert" in node.kernel for node in stream)
        print(
            f"  C{cluster}/Core{core}: {len(stream):4d} device tasks "
            f"({dynamic_count:4d} dynamic expert tasks)"
        )
    print()

    dynamic_nodes = [
        node for node in nodes.values() if "moe_dynamic_expert" in node.kernel
    ]
    if dynamic_nodes:
        chain_len = len(EXPECTED_DYNAMIC_CHAIN)
        print(
            f"Dynamic expert nodes:   {len(dynamic_nodes)} "
            f"= {len(dynamic_nodes) // chain_len} chains x {chain_len} nodes"
        )
        by_cluster = Counter(node.cluster for node in dynamic_nodes)
        by_core = Counter((node.cluster, node.core) for node in dynamic_nodes)
        print(f"  By cluster: {dict(sorted(by_cluster.items()))}")
        print(
            "  By cluster/core: "
            + ", ".join(
                f"C{cluster}/Core{core}={count}"
                for (cluster, core), count in sorted(by_core.items())
            )
        )
        print()

    if verbose:
        print("First 8 expected dynamic chains:")
        sorted_dyn = sorted(dynamic_nodes, key=lambda item: item.node_id)
        chain_len = len(EXPECTED_DYNAMIC_CHAIN)
        for chain_idx in range(min(8, len(sorted_dyn) // chain_len)):
            group = sorted_dyn[chain_idx * chain_len : (chain_idx + 1) * chain_len]
            names = " -> ".join(short_kernel_name(node.kernel) for node in group)
            print(
                f"  Chain {chain_idx:02d} C{group[0].cluster}: N{group[0].node_id}-N{group[-1].node_id}: {names}"
            )
        print()

    if warnings:
        print("WARNINGS:")
        for item in warnings:
            print(f"  - {item}")
        print()

    if errors:
        print("ERRORS:")
        for item in errors:
            print(f"  - {item}")
        print()
        print("RESULT: FAIL")
    else:
        print("RESULT: PASS")


def load_artifacts(
    workload_dir: str,
) -> Tuple[Dict[int, DfgNode], List[TaskDescriptor], Dict[int, DevTaskMapEntry]]:
    csv_path, header_path = default_paths(workload_dir)
    nodes = parse_final_dfg(csv_path)
    descriptors = parse_task_descriptors(header_path)
    dev_map = parse_dev_task_map(header_path)
    return nodes, descriptors, dev_map


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify current multi_cluster_MoE generated dependency artifacts."
    )
    parser.add_argument(
        "--workload-dir",
        default=DEFAULT_WORKLOAD_DIR,
        help="Directory containing final_dfg.csv and offload_bingo_hw.h",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print extra dynamic-chain detail"
    )
    args = parser.parse_args(argv)

    nodes, descriptors, dev_map = load_artifacts(args.workload_dir)
    errors, warnings = validate_artifacts(nodes, descriptors, dev_map)
    print_summary(nodes, descriptors, dev_map, errors, warnings, verbose=args.verbose)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
