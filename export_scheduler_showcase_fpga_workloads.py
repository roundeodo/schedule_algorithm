#!/usr/bin/env python3
"""Lower the four thesis showcases into an FPGA-workload handoff format.

The exported JSON keeps scheduler rounds for audit, but execution slots are
cluster-local.  It also constructs a deterministic legal Top-2 token routing
whose expert marginals exactly match each showcased distribution.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import io
import json
import os
from pathlib import Path
from typing import Iterable

import four_stage_scheduler as reference
from run_four_stage_reference import deserialize_action


HERE = Path(__file__).resolve().parent
SHOWCASE_RESULT = (
    HERE / "results/policy_search/scheduler_thesis_four_policy_showcases.json"
)
OUTPUT_JSON = (
    HERE / "results/policy_search/scheduler_showcase_fpga_workloads.json"
)
OUTPUT_CSV = (
    HERE / "results/policy_search/scheduler_showcase_fpga_tasks.csv"
)
TICK_CC = reference.SCHEDULE_TIME_QUANTUM_CC

SHAPE_ID = {
    reference.SHAPE_A: 0,
    reference.SHAPE_B: 1,
    reference.SHAPE_C: 2,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tick(value_cc: int) -> int:
    value_cc = int(value_cc)
    if value_cc < 0:
        return -1
    quotient, remainder = divmod(value_cc, TICK_CC)
    if remainder:
        raise AssertionError(f"time {value_cc} cc is not on the tick lattice")
    return quotient


def _shape(shape: reference.Shape) -> dict:
    return {
        "id": SHAPE_ID[shape],
        "name": shape.name.split("(", 1)[0],
        "model_name": shape.name,
        "m_dim": int(shape.M_dim),
        "bw_req_bytes_per_cc": int(shape.bw_req),
        "alloc_bytes_per_cc": int(shape.alloc),
    }


def _action_family(action: reference.StageAction) -> str:
    e2, e3 = int(action.c2_eid), int(action.c3_eid)
    if e2 >= 0 and e3 >= 0:
        return "SPLIT" if e2 == e3 else "PAIR"
    if e2 >= 0 or e3 >= 0:
        return "SINGLE"
    raise AssertionError("showcase contains a non-consuming round action")


def _round_mode(state: reference.BeamState) -> str:
    if len(state.remaining) == 1:
        return "TERMINAL"
    return "SYNC" if state.c2.task_end == state.c3.task_end else "ONE_IDLE"


def _source_eid_map(case: dict) -> dict[int, int]:
    # The selected certificate is already rank ordered and every structured
    # synthetic case is defined directly in workload-EID order.
    return {eid: eid for eid in range(len(case["distribution"]))}


def _reconstruct_top2(counts: list[int]) -> tuple[list[dict], dict[int, list[int]]]:
    assignment_total = sum(counts)
    if assignment_total % 2:
        raise AssertionError("Top-2 assignment total must be even")
    n_tokens = assignment_total // 2
    if max(counts, default=0) > n_tokens:
        raise AssertionError("an expert cannot occur twice in one Top-2 token")
    heap = [(-int(count), int(eid)) for eid, count in enumerate(counts) if count]
    heapq.heapify(heap)
    tokens = []
    routed = {eid: [] for eid in range(len(counts))}
    for token_id in range(n_tokens):
        if len(heap) < 2:
            raise AssertionError("cannot reconstruct a loop-free Top-2 routing")
        neg_a, eid_a = heapq.heappop(heap)
        neg_b, eid_b = heapq.heappop(heap)
        if eid_a == eid_b:
            raise AssertionError("Top-2 reconstruction selected one expert twice")
        pair = sorted((eid_a, eid_b))
        tokens.append({"token_id": token_id, "workload_eids": pair})
        routed[eid_a].append(token_id)
        routed[eid_b].append(token_id)
        neg_a += 1
        neg_b += 1
        if neg_a:
            heapq.heappush(heap, (neg_a, eid_a))
        if neg_b:
            heapq.heappush(heap, (neg_b, eid_b))
    if heap:
        raise AssertionError("Top-2 reconstruction left unmatched assignments")
    for eid, count in enumerate(counts):
        if len(routed[eid]) != count:
            raise AssertionError(f"E{eid} reconstructed count mismatch")
    return tokens, routed


def _method_actions(case: dict, method: str) -> list[dict]:
    steps = case["policies"][method]["steps"]
    if method == "FULL_SCHEDULER":
        return [
            {
                "selected_profile_slot": int(step["selected_profile_slot"]),
                "s4pf_actions": list(step.get("s4pf_actions", [])),
                "action": step["action"],
            }
            for step in steps
        ]
    return [
        {
            "selected_profile_slot": None,
            "s4pf_actions": list(step.get("s4pf_actions", [])),
            "action": step["action"],
        }
        for step in steps
    ]


def _method_expected_ticks(case: dict, method: str) -> int:
    return int(case["policies"][method]["makespan_ticks"])


def _task_record(
    *,
    action: reference.StageAction,
    snap: reference.FourStageSnap,
    cluster: int,
    round_index: int,
    global_task_index: int,
    cluster_slot: int,
    family: str,
    counts: list[int],
    source_eids: dict[int, int],
    routed: dict[int, list[int]],
) -> dict:
    prefix = f"c{cluster}"
    eid = int(getattr(action, f"{prefix}_eid"))
    ntok = int(getattr(action, f"{prefix}_ntok"))
    token_start_rank = 0
    if family == "SPLIT" and cluster == 3:
        token_start_rank = int(action.c2_ntok)
    token_ids = routed[eid][token_start_rank : token_start_rank + ntok]
    if len(token_ids) != ntok:
        raise AssertionError("task token slice exceeds routed-token list")
    shape_s1 = getattr(action, f"{prefix}_shape_s1")
    shape_s3 = getattr(action, f"{prefix}_shape_s3")
    skip_s1 = bool(getattr(action, f"{prefix}_s1_cached"))
    skip_s3 = bool(getattr(action, f"{prefix}_s3_cached"))
    dma_s1 = getattr(action, f"{prefix}_dma_s1")
    dma_s3 = getattr(action, f"{prefix}_dma_s3")
    s2pf_dma = getattr(action, f"{prefix}_s2pf_dma")
    tail_s2 = ntok if skip_s1 else max(0, ntok - int(shape_s1.M_dim))
    tail_s4 = ntok if skip_s3 else max(0, ntok - int(shape_s3.M_dim))
    return {
        "global_task_index": global_task_index,
        "global_round": round_index,
        "cluster": f"C{cluster}",
        "cluster_model_id": cluster,
        "cluster_abi_id": cluster - 2,
        "cluster_slot": cluster_slot,
        "action_family": family,
        "action_tag": action.tag,
        "workload_eid": eid,
        "source_eid": source_eids[eid],
        "expert_total_ntokens": counts[eid],
        "token_start_rank": token_start_rank,
        "ntokens": ntok,
        "token_ids": token_ids,
        "shape_s1": _shape(shape_s1),
        "shape_s3": _shape(shape_s3),
        "dma_s1": {"id": int(dma_s1), "name": dma_s1.name},
        "dma_s3": {"id": int(dma_s3), "name": dma_s3.name},
        "skip_s1": skip_s1,
        "skip_s3": skip_s3,
        "has_s2pf": s2pf_dma != reference.DmaBinding.NONE,
        "s2pf_dma": {"id": int(s2pf_dma), "name": s2pf_dma.name},
        "m_s2_exec": (tail_s2 + 1) // 2,
        "m_s4_exec": (tail_s4 + 1) // 2,
        "skip_s2": tail_s2 == 0,
        "skip_s4": tail_s4 == 0,
        "timing_ticks": {
            "task_start": _tick(snap.task_start),
            "dma1_end": _tick(snap.dma1_end),
            "s1_end": _tick(snap.s1_end),
            "s2_end": _tick(snap.s2_end),
            "dma3_end": _tick(snap.dma3_end),
            "s3_end": _tick(snap.s3_end),
            "s4_start": _tick(snap.s4_start),
            "compute_end": _tick(snap.compute_end),
            "task_end": _tick(snap.task_end),
            "s2pf_start": _tick(snap.s2pf_start),
            "s2pf_end": _tick(snap.s2pf_end),
        },
        "s4pf": {
            "enabled": False,
            "target_workload_eid": -1,
            "target_source_eid": -1,
            "dma": {"id": 0, "name": "NONE"},
            "start_tick": -1,
            "end_tick": -1,
        },
    }


def _dma_ops(tasks: list[dict]) -> list[dict]:
    operations = []
    kind_id = {"S1": 1, "S3": 3, "S2_PREFETCH": 4, "S4_PREFETCH": 5}

    def add(task: dict, kind: str, dma: dict, eid: int, start: int, end: int):
        operations.append(
            {
                "dma_op_index": len(operations),
                "global_task_index": task["global_task_index"],
                "cluster": task["cluster"],
                "cluster_slot": task["cluster_slot"],
                "kind": kind,
                "kind_id": kind_id[kind],
                "dma": dma,
                "expert_id": eid,
                "start_tick": start,
                "end_tick": end,
            }
        )

    for task in tasks:
        timing = task["timing_ticks"]
        if not task["skip_s1"]:
            add(
                task,
                "S1",
                task["dma_s1"],
                task["workload_eid"],
                timing["task_start"],
                timing["dma1_end"],
            )
        if task["has_s2pf"]:
            add(
                task,
                "S2_PREFETCH",
                task["s2pf_dma"],
                task["workload_eid"],
                timing["s2pf_start"],
                timing["s2pf_end"],
            )
        elif not task["skip_s3"]:
            add(
                task,
                "S3",
                task["dma_s3"],
                task["workload_eid"],
                timing["s2_end"],
                timing["dma3_end"],
            )
        if task["s4pf"]["enabled"]:
            add(
                task,
                "S4_PREFETCH",
                task["s4pf"]["dma"],
                task["s4pf"]["target_workload_eid"],
                task["s4pf"]["start_tick"],
                task["s4pf"]["end_tick"],
            )
    return operations


def _validate_dma_lanes(method: dict) -> None:
    for lane_name, lane_mask in (("IDMA", 1), ("XDMA", 2)):
        intervals = sorted(
            (
                int(operation["start_tick"]),
                int(operation["end_tick"]),
                int(operation["dma_op_index"]),
            )
            for operation in method["dma_ops"]
            if int(operation["dma"]["id"]) & lane_mask
            and int(operation["end_tick"]) > int(operation["start_tick"])
        )
        for previous, current in zip(intervals, intervals[1:]):
            if current[0] < previous[1]:
                raise AssertionError(
                    f"{method['method']} {lane_name} overlap: "
                    f"op{previous[2]}={previous[:2]} op{current[2]}={current[:2]}"
                )


def _lower_method(
    case: dict,
    method: str,
    source_eids: dict[int, int],
    routed: dict[int, list[int]],
) -> dict:
    counts = [int(value) for value in case["distribution"]]
    distribution = {eid: ntok for eid, ntok in enumerate(counts)}
    state = reference.FourStageScheduler(distribution)._initial_state()
    tasks: list[dict] = []
    rounds = []
    cluster_slots = {2: 0, 3: 0}
    last_task = {2: -1, 3: -1}
    for round_index, encoded in enumerate(_method_actions(case, method)):
        mode = _round_mode(state)
        s4pf_items = []
        for item in encoded["s4pf_actions"]:
            action = deserialize_action(item)
            cluster = int(action.pf_cluster)
            previous = last_task[cluster]
            if previous < 0:
                raise AssertionError("S4PF has no previous same-cluster task")
            state = reference.apply_action(state, action)
            snap = state.c2 if cluster == 2 else state.c3
            target = int(action.pf_eid)
            tasks[previous]["s4pf"] = {
                "enabled": True,
                "target_workload_eid": target,
                "target_source_eid": source_eids[target],
                "dma": {"id": int(action.pf_dma), "name": action.pf_dma.name},
                "start_tick": _tick(action.pf_start),
                "end_tick": _tick(snap.pf_end),
            }
            s4pf_items.append(
                {
                    "cluster": f"C{cluster}",
                    "attached_to_global_task_index": previous,
                    "target_workload_eid": target,
                }
            )
        action = deserialize_action(encoded["action"])
        family = _action_family(action)
        child = reference.apply_action(state, action)
        round_task_indices = []
        for cluster in (2, 3):
            if int(getattr(action, f"c{cluster}_eid")) < 0:
                continue
            snap = child.c2 if cluster == 2 else child.c3
            task = _task_record(
                action=action,
                snap=snap,
                cluster=cluster,
                round_index=round_index,
                global_task_index=len(tasks),
                cluster_slot=cluster_slots[cluster],
                family=family,
                counts=counts,
                source_eids=source_eids,
                routed=routed,
            )
            tasks.append(task)
            round_task_indices.append(task["global_task_index"])
            last_task[cluster] = task["global_task_index"]
            cluster_slots[cluster] += 1
        rounds.append(
            {
                "global_round": round_index,
                "mode": mode,
                "action_family": family,
                "action_tag": action.tag,
                "selected_profile_slot": encoded["selected_profile_slot"],
                "task_indices": round_task_indices,
                "s4pf_actions": s4pf_items,
                "round_makespan_tick": _tick(child.g_score),
            }
        )
        state = child
    replay_cc = reference.validate_schedule_history(state.history, distribution)
    if replay_cc != state.g_score:
        raise AssertionError("explicit-DMA history replay mismatch")
    expected_ticks = _method_expected_ticks(case, method)
    if _tick(state.g_score) != expected_ticks:
        raise AssertionError(
            f"{case['name']} {method}: {_tick(state.g_score)} != {expected_ticks}"
        )
    covered = {eid: [] for eid in distribution}
    for task in tasks:
        eid = task["workload_eid"]
        start = task["token_start_rank"]
        covered[eid].extend(range(start, start + task["ntokens"]))
    for eid, ntok in distribution.items():
        if sorted(covered[eid]) != list(range(ntok)):
            raise AssertionError(f"{method}: E{eid} token slices are incomplete")
    streams = {
        f"C{cluster}": [
            task["global_task_index"]
            for task in tasks
            if task["cluster"] == f"C{cluster}"
        ]
        for cluster in (2, 3)
    }
    for cluster, indices in streams.items():
        slots = [tasks[index]["cluster_slot"] for index in indices]
        if slots != list(range(len(slots))):
            raise AssertionError(f"{method}: {cluster} slot sequence is not contiguous")
    return {
        "method": method,
        "makespan_ticks": expected_ticks,
        "n_rounds": len(rounds),
        "n_tasks": len(tasks),
        "n_dma_ops": 0,
        "rounds": rounds,
        "tasks": tasks,
        "cluster_streams": streams,
        "dma_ops": [],
    }


def _csv_text(cases: Iterable[dict]) -> str:
    columns = [
        "case",
        "method",
        "global_round",
        "global_task_index",
        "cluster",
        "cluster_abi_id",
        "cluster_slot",
        "action_family",
        "workload_eid",
        "source_eid",
        "expert_total_ntokens",
        "token_start_rank",
        "ntokens",
        "token_ids",
        "shape_s1_id",
        "shape_s1",
        "shape_s3_id",
        "shape_s3",
        "dma_s1",
        "dma_s3",
        "skip_s1",
        "skip_s3",
        "has_s2pf",
        "s2pf_dma",
        "m_s2_exec",
        "m_s4_exec",
        "skip_s2",
        "skip_s4",
        "task_start_tick",
        "dma1_end_tick",
        "s1_end_tick",
        "s2_end_tick",
        "dma3_end_tick",
        "s3_end_tick",
        "s4_start_tick",
        "compute_end_tick",
        "task_end_tick",
        "s2pf_start_tick",
        "s2pf_end_tick",
        "s4pf_target_eid",
        "s4pf_dma",
        "s4pf_start_tick",
        "s4pf_end_tick",
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for case in cases:
        for method in case["methods"]:
            for task in method["tasks"]:
                timing = task["timing_ticks"]
                writer.writerow(
                    {
                        "case": case["name"],
                        "method": method["method"],
                        "global_round": task["global_round"],
                        "global_task_index": task["global_task_index"],
                        "cluster": task["cluster"],
                        "cluster_abi_id": task["cluster_abi_id"],
                        "cluster_slot": task["cluster_slot"],
                        "action_family": task["action_family"],
                        "workload_eid": task["workload_eid"],
                        "source_eid": task["source_eid"],
                        "expert_total_ntokens": task["expert_total_ntokens"],
                        "token_start_rank": task["token_start_rank"],
                        "ntokens": task["ntokens"],
                        "token_ids": ";".join(map(str, task["token_ids"])),
                        "shape_s1_id": task["shape_s1"]["id"],
                        "shape_s1": task["shape_s1"]["name"],
                        "shape_s3_id": task["shape_s3"]["id"],
                        "shape_s3": task["shape_s3"]["name"],
                        "dma_s1": task["dma_s1"]["name"],
                        "dma_s3": task["dma_s3"]["name"],
                        "skip_s1": int(task["skip_s1"]),
                        "skip_s3": int(task["skip_s3"]),
                        "has_s2pf": int(task["has_s2pf"]),
                        "s2pf_dma": task["s2pf_dma"]["name"],
                        "m_s2_exec": task["m_s2_exec"],
                        "m_s4_exec": task["m_s4_exec"],
                        "skip_s2": int(task["skip_s2"]),
                        "skip_s4": int(task["skip_s4"]),
                        **{f"{key}_tick": value for key, value in timing.items()},
                        "s4pf_target_eid": task["s4pf"]["target_workload_eid"],
                        "s4pf_dma": task["s4pf"]["dma"]["name"],
                        "s4pf_start_tick": task["s4pf"]["start_tick"],
                        "s4pf_end_tick": task["s4pf"]["end_tick"],
                    }
                )
    return stream.getvalue()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    showcase = json.loads(SHOWCASE_RESULT.read_text(encoding="utf-8"))
    cases = []
    methods = (
        "STATIC_DESC",
        "DYNAMIC_DESC",
        "DYNAMIC_TWO_ENDED",
        "FULL_SCHEDULER",
    )
    for source_case in showcase["cases"]:
        counts = [int(value) for value in source_case["distribution"]]
        source_eids = _source_eid_map(source_case)
        tokens, routed = _reconstruct_top2(counts)
        expert_loads = [
            {
                "workload_eid": eid,
                "source_eid": source_eids[eid],
                "ntokens": ntok,
                "routed_token_ids": routed[eid],
            }
            for eid, ntok in enumerate(counts)
        ]
        lowered = [
            _lower_method(source_case, method, source_eids, routed)
            for method in methods
        ]
        for method in lowered:
            method["dma_ops"] = _dma_ops(method["tasks"])
            method["n_dma_ops"] = len(method["dma_ops"])
            _validate_dma_lanes(method)
        cases.append(
            {
                "name": source_case["name"],
                "characteristic": source_case["characteristic"],
                "source": source_case["source"],
                "source_key": source_case["source_key"],
                "conceptual_experts": 64,
                "active_experts": len(counts),
                "top_k": 2,
                "input_tokens": len(tokens),
                "assignment_total": sum(counts),
                "initial_cache": {"C2": -1, "C3": -1},
                "expert_id_contract": {
                    "schedule_field": "workload_eid",
                    "ordering": "descending ntokens, then ascending source_eid",
                    "source_eid_is_provenance_only": True,
                },
                "expert_loads": expert_loads,
                "token_routing": tokens,
                "methods": lowered,
            }
        )
    payload = {
        "schema": "scheduler_showcase_fpga_workloads_v1",
        "status": "model-derived workload specification; FPGA execution unvalidated",
        "time": {
            "unit": "tick",
            "cycles_per_tick": TICK_CC,
            "absolute_model_times_are_validation_targets": True,
        },
        "execution_contract": {
            "global_round_is_a_scheduler_group_not_a_runtime_barrier": True,
            "cluster_slot_is_the_normative_per_cluster_order": True,
            "global_task_index_order": "round order, C2 before C3 within a round",
            "both_dma_reserves_idma_and_xdma_simultaneously": True,
            "token_routing": (
                "deterministic legal Top-2 reconstruction from marginal expert "
                "loads; not the original router pairing"
            ),
        },
        "enums": {
            "shape": {"A": 0, "B": 1, "C": 2},
            "cluster_abi": {"C2": 0, "C3": 1},
            "dma": {"NONE": 0, "IDMA": 1, "XDMA": 2, "BOTH": 3},
            "dma_op_kind": {
                "S1": 1,
                "S3": 3,
                "S2_PREFETCH": 4,
                "S4_PREFETCH": 5,
            },
        },
        "manifest": {
            "showcase_result": str(SHOWCASE_RESULT.resolve()),
            "showcase_result_sha256": _sha256(SHOWCASE_RESULT),
            "generator": str(Path(__file__).resolve()),
            "generator_sha256": _sha256(Path(__file__).resolve()),
            "four_stage_scheduler_sha256": _sha256(
                HERE / "four_stage_scheduler.py"
            ),
        },
        "cases": cases,
    }
    csv_text = _csv_text(cases)
    _atomic_text(OUTPUT_CSV, csv_text)
    payload["manifest"]["flat_task_csv"] = str(OUTPUT_CSV.resolve())
    payload["manifest"]["flat_task_csv_sha256"] = _sha256(OUTPUT_CSV)
    _atomic_text(OUTPUT_JSON, json.dumps(payload, indent=2) + "\n")
    print(f"PASS cases={len(cases)} methods={len(cases) * len(methods)}")
    for case in cases:
        print(
            case["name"],
            {method["method"]: method["makespan_ticks"] for method in case["methods"]},
        )
    print(f"wrote {OUTPUT_JSON}")
    print(f"wrote {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
