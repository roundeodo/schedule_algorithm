#!/usr/bin/env python3
"""Lock the deployed C scheduler to the final Python distilled policy.

The Python policy is normative.  This test compiles the workload C source as a
native shared object, runs identical requests, and compares the complete
lowered task/DMA stream plus the tick-domain makespan.
"""

from __future__ import annotations

import argparse
import ctypes as C
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import random
import subprocess
import tempfile

import four_stage_scheduler as reference
import scheduler_rtl_distilled_policy as policy


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
C_DIR = ROOT / "HeMAiA/target/sw/host/apps/offload_bingo_hw/single_chip/workloads/multi_cluster_MoE"
C_SOURCE = C_DIR / "moe_scheduler.c"
PROOF65 = HERE / "results/policy_search/olmoe_top2_projection_65_optimal_v1.json"
S4_RESULT = HERE / "results/policy_search/bounded_top5_bottom1_fixed_lane_targeted_s4pf_random_validation.json"
MAX_EXPERTS = 64
MAX_TASKS = 128
MAX_DMA = 512


class Expert(C.Structure):
    _fields_ = [("expert_id", C.c_uint16), ("ntokens", C.c_uint16)]


class Request(C.Structure):
    _fields_ = [
        ("experts", Expert * MAX_EXPERTS),
        ("n_experts", C.c_uint16),
        ("cache_eid_c2", C.c_int16),
        ("cache_eid_c3", C.c_int16),
    ]


class Task(C.Structure):
    _fields_ = [
        ("cluster", C.c_int),
        ("expert_id", C.c_uint16),
        ("token_start_rank", C.c_uint16),
        ("ntokens", C.c_uint16),
        ("shape_s1", C.c_int),
        ("shape_s3", C.c_int),
        ("dma_s1", C.c_int),
        ("dma_s3", C.c_int),
        ("skip_s1", C.c_uint8),
        ("skip_s3", C.c_uint8),
        ("skip_s2", C.c_uint8),
        ("skip_s4", C.c_uint8),
        ("m_s2_exec", C.c_uint32),
        ("m_s4_exec", C.c_uint32),
    ]


class DmaOp(C.Structure):
    _fields_ = [
        ("task_idx", C.c_uint16),
        ("kind", C.c_int),
        ("dma", C.c_int),
        ("expert_id", C.c_int16),
    ]


class Schedule(C.Structure):
    _fields_ = [
        ("tasks", Task * MAX_TASKS),
        ("dma_ops", DmaOp * MAX_DMA),
        ("n_tasks", C.c_uint16),
        ("n_dma_ops", C.c_uint16),
    ]


class HwPlanDesc(C.Structure):
    _fields_ = [
        ("cluster", C.c_int),
        ("expert_id", C.c_uint16),
        ("token_start_rank", C.c_uint16),
        ("ntokens", C.c_uint16),
        ("shape_s1", C.c_int),
        ("shape_s3", C.c_int),
        ("skip_s1", C.c_uint8),
        ("skip_s3", C.c_uint8),
        ("has_s2pf", C.c_uint8),
        ("dma_s1", C.c_int),
        ("dma_s3", C.c_int),
        ("s2pf_dma", C.c_int),
    ]


class HwPlanEntry(C.Structure):
    _fields_ = [
        ("valid", C.c_uint8),
        ("desc", HwPlanDesc),
        ("allow_s4pf", C.c_uint8),
        ("s4pf_dma", C.c_int),
        ("s4pf_expert_id", C.c_int16),
    ]


@dataclass
class ExpectedTask:
    cluster: int
    eid: int
    start: int
    ntok: int
    s1: int
    s3: int
    d1: int
    d3: int
    s2pf: int
    skip1: int
    skip3: int
    m2: int
    m4: int
    s4_dma: int = 0
    s4_eid: int = -1


def compile_library(output: Path) -> None:
    subprocess.run(
        [
            "gcc", "-std=c99", "-O2", "-fPIC", "-shared",
            "-Wall", "-Wextra", "-Werror", "-DMOE_SCHEDULER_TEST_API",
            str(C_SOURCE), "-o", str(output),
        ],
        cwd=C_DIR,
        check=True,
    )


def expected_tasks(result: policy.ScheduleResult) -> list[ExpectedTask]:
    tasks: list[ExpectedTask] = []
    last = {2: -1, 3: -1}
    shape_id = {
        reference.SHAPE_A: 0,
        reference.SHAPE_B: 1,
        reference.SHAPE_C: 2,
    }
    for step in result.steps:
        for pf in step.s4pf_actions:
            index = last[int(pf.pf_cluster)]
            assert index >= 0
            tasks[index].s4_dma = int(pf.pf_dma)
            tasks[index].s4_eid = int(pf.pf_eid)
        action = step.action
        for cluster in (2, 3):
            eid = int(getattr(action, f"c{cluster}_eid"))
            if eid < 0:
                continue
            ntok = int(getattr(action, f"c{cluster}_ntok"))
            token_start = 0 if cluster == 2 else (
                int(action.c2_ntok) if action.c2_eid == eid else 0
            )
            sh1 = getattr(action, f"c{cluster}_shape_s1")
            sh3 = getattr(action, f"c{cluster}_shape_s3")
            skip1 = int(getattr(action, f"c{cluster}_s1_cached"))
            skip3 = int(getattr(action, f"c{cluster}_s3_cached"))
            d1 = int(getattr(action, f"c{cluster}_dma_s1"))
            d3 = int(getattr(action, f"c{cluster}_dma_s3"))
            s2pf = int(getattr(action, f"c{cluster}_s2pf_dma"))
            tail2 = ntok if skip1 else max(0, ntok - sh1.M_dim)
            tail4 = ntok if skip3 else max(0, ntok - sh3.M_dim)
            tasks.append(
                ExpectedTask(
                    cluster=cluster - 2,
                    eid=eid,
                    start=token_start,
                    ntok=ntok,
                    s1=shape_id[sh1],
                    s3=shape_id[sh3],
                    d1=d1,
                    d3=d3,
                    s2pf=s2pf,
                    skip1=skip1,
                    skip3=skip3,
                    m2=(tail2 + 1) // 2,
                    m4=(tail4 + 1) // 2,
                )
            )
            last[cluster] = len(tasks) - 1
    return tasks


def expected_dma(tasks: list[ExpectedTask], result: policy.ScheduleResult):
    # Recover whether the late transfer is S2PF from the normative action.
    ops = []
    for index, task in enumerate(tasks):
        if not task.skip1:
            ops.append((index, 1, task.d1, task.eid))
        if task.s2pf:
            ops.append((index, 4, task.s2pf, task.eid))
        elif not task.skip3:
            ops.append((index, 3, task.d3, task.eid))
        if task.s4_dma:
            ops.append((index, 5, task.s4_dma, task.s4_eid))
    return ops


def schedule_signature(schedule: Schedule):
    tasks = [
        (
            task.cluster, task.expert_id, task.token_start_rank, task.ntokens,
            task.shape_s1, task.shape_s3, task.dma_s1, task.dma_s3,
            task.skip_s1, task.skip_s3, task.skip_s2, task.skip_s4,
            task.m_s2_exec, task.m_s4_exec,
        )
        for task in schedule.tasks[: schedule.n_tasks]
    ]
    dma = [
        (op.task_idx, op.kind, op.dma, op.expert_id)
        for op in schedule.dma_ops[: schedule.n_dma_ops]
    ]
    return tasks, dma


def run_case(lib, distribution: dict[int, int], c2: int, c3: int, label: str) -> None:
    request = Request()
    entries = sorted(distribution.items())
    request.n_experts = len(entries)
    request.cache_eid_c2 = c2
    request.cache_eid_c3 = c3
    for i, (eid, ntok) in enumerate(entries):
        request.experts[i] = Expert(eid, ntok)
    actual = Schedule()
    ticks = C.c_uint32()
    status = lib.moe_schedule_debug(C.byref(request), C.byref(actual), C.byref(ticks))
    if status != 0:
        raise AssertionError(f"{label}: C status {status}")
    result = policy.schedule(distribution, c2, c3)
    expected = expected_tasks(result)
    if ticks.value * reference.SCHEDULE_TIME_QUANTUM_CC != result.makespan_cc:
        raise AssertionError(
            f"{label}: makespan C={ticks.value} ticks Python="
            f"{result.makespan_cc // reference.SCHEDULE_TIME_QUANTUM_CC} ticks"
        )
    if actual.n_tasks != len(expected):
        raise AssertionError(f"{label}: task count C={actual.n_tasks} Python={len(expected)}")
    for i, exp in enumerate(expected):
        got = actual.tasks[i]
        fields = (
            got.cluster, got.expert_id, got.token_start_rank, got.ntokens,
            got.shape_s1, got.shape_s3, got.dma_s1, got.dma_s3,
            got.skip_s1, got.skip_s3, got.m_s2_exec, got.m_s4_exec,
        )
        want = (
            exp.cluster, exp.eid, exp.start, exp.ntok, exp.s1, exp.s3,
            exp.d1, exp.d3, exp.skip1, exp.skip3, exp.m2, exp.m4,
        )
        if fields != want:
            raise AssertionError(f"{label}: task {i}\nC={fields}\nPython={want}")
    got_ops = [
        (actual.dma_ops[i].task_idx, actual.dma_ops[i].kind,
         actual.dma_ops[i].dma, actual.dma_ops[i].expert_id)
        for i in range(actual.n_dma_ops)
    ]
    want_ops = expected_dma(expected, result)
    if got_ops != want_ops:
        raise AssertionError(f"{label}: DMA\nC={got_ops}\nPython={want_ops}")

    # The production path lowers the internal Python-derived plan directly.
    # Also verify that its optional compact-plan export is lossless.
    plan = (HwPlanEntry * MAX_TASKS)()
    n_plan = C.c_uint16()
    status = lib.moe_make_hw_plan(C.byref(request), plan, C.byref(n_plan))
    if status != 0:
        raise AssertionError(f"{label}: compact-plan export status {status}")
    round_trip = Schedule()
    status = lib.moe_lower_hw_plan(
        C.byref(request), plan, n_plan.value, C.byref(round_trip)
    )
    if status != 0:
        raise AssertionError(f"{label}: compact-plan lowering status {status}")
    if schedule_signature(actual) != schedule_signature(round_trip):
        raise AssertionError(f"{label}: compact-plan round trip differs")
    reference.clear_scheduler_caches()


def proof_jobs():
    import verify_scheduler_rtl_unified_policy as datasets
    return datasets._proof_jobs(PROOF65.resolve())


def s4_jobs(limit: int):
    rows = json.loads(S4_RESULT.read_text())["rows"]
    sources = {}
    jobs = []
    for key, row in rows.items():
        if not row["s4pf_events"]:
            continue
        e_name, case_text = key.split(":")
        e_total = int(e_name[1:])
        if e_total not in sources:
            payload = json.loads(
                (HERE / f"scheduler_strategy_coverage_E{e_total}.json").read_text()
            )
            sources[e_total] = {int(case["case_id"]): case for case in payload["cases"]}
        case = sources[e_total][int(case_text)]
        jobs.append(
            (
                {int(eid): int(ntok) for eid, ntok in case["dist"].items()},
                int(case["c2"]), int(case["c3"]), f"s4-{key}",
            )
        )
        if len(jobs) >= limit:
            break
    return jobs


def coverage_jobs():
    jobs = []
    for e_total in (8, 32, 64):
        payload = json.loads(
            (HERE / f"scheduler_strategy_coverage_E{e_total}.json").read_text()
        )
        for case in payload["cases"]:
            jobs.append(
                (
                    {int(eid): int(ntok) for eid, ntok in case["dist"].items()},
                    int(case["c2"]),
                    int(case["c3"]),
                    f"coverage-E{e_total}:{int(case['case_id'])}",
                )
            )
    return jobs


def source_digest() -> str:
    digest = hashlib.sha256()
    paths = (
        Path(__file__).resolve(),
        C_SOURCE,
        C_DIR / "moe_scheduler.h",
        HERE / "four_stage_scheduler.py",
        HERE / "scheduler_rtl_distilled_policy.py",
        HERE / "scheduler_rtl_distilled_lowering.py",
        HERE / "scheduler_rtl_distilled_profiles.py",
        HERE / "scheduler_rtl_distilled_scoring.py",
        HERE / "scheduler_rtl_distilled_types.py",
    )
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_checkpoint(path: Path, digest: str) -> int:
    if not path.exists():
        return 0
    payload = json.loads(path.read_text())
    if payload.get("source_sha256") != digest:
        raise RuntimeError(
            f"checkpoint source hash differs: {path}; start a new validation"
        )
    return int(payload["next_coverage_index"])


def save_checkpoint(path: Path, digest: str, next_index: int, total: int) -> None:
    payload = {
        "schema": 1,
        "source_sha256": digest,
        "next_coverage_index": next_index,
        "coverage_total": total,
        "complete": next_index == total,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proof65", action="store_true")
    parser.add_argument("--random", type=int, default=0)
    parser.add_argument("--s4-samples", type=int, default=0)
    parser.add_argument("--coverage-all", action="store_true")
    parser.add_argument("--coverage-limit", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x5A17)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="moe_scheduler_c_") as tmp:
        library = Path(tmp) / "libmoe_scheduler.so"
        compile_library(library)
        lib = C.CDLL(str(library))
        lib.moe_schedule_debug.argtypes = [
            C.POINTER(Request), C.POINTER(Schedule), C.POINTER(C.c_uint32)
        ]
        lib.moe_schedule_debug.restype = C.c_int
        lib.moe_make_hw_plan.argtypes = [
            C.POINTER(Request), C.POINTER(HwPlanEntry), C.POINTER(C.c_uint16)
        ]
        lib.moe_make_hw_plan.restype = C.c_int
        lib.moe_lower_hw_plan.argtypes = [
            C.POINTER(Request), C.POINTER(HwPlanEntry), C.c_uint16,
            C.POINTER(Schedule),
        ]
        lib.moe_lower_hw_plan.restype = C.c_int
        cases = [
            ({0: 16}, -1, -1, "single16"),
            ({0: 16, 1: 16, 2: 4, 3: 4, 4: 4, 5: 4,
              6: 2, 7: 2, 8: 2, 9: 2, 10: 2}, -1, -1, "directed45"),
            ({0: 16, 1: 7, 2: 3}, 0, 2, "initial-cache-both"),
        ]
        if args.proof65:
            cases.extend(
                (job["distribution"], job["c2"], job["c3"], f"proof65-{i}")
                for i, job in enumerate(proof_jobs())
            )
        if args.s4_samples:
            cases.extend(s4_jobs(args.s4_samples))
        rng = random.Random(args.seed)
        for index in range(args.random):
            n = rng.randint(1, 64)
            dist = {eid: rng.randint(1, 96) for eid in range(n)}
            cases.append((dist, -1, -1, f"random-{index}"))
        for case_index, case in enumerate(cases):
            if args.verbose:
                print(f"RUN {case_index}/{len(cases)} {case[3]}", flush=True)
            run_case(lib, *case)
        validated = len(cases)
        if args.coverage_all:
            all_coverage = coverage_jobs()
            digest = source_digest()
            start = load_checkpoint(args.checkpoint, digest) if args.checkpoint else 0
            stop = len(all_coverage)
            if args.coverage_limit:
                stop = min(stop, start + args.coverage_limit)
            for index in range(start, stop):
                run_case(lib, *all_coverage[index])
                validated += 1
                if args.checkpoint:
                    save_checkpoint(args.checkpoint, digest, index + 1, len(all_coverage))
                if args.progress_every > 0 and (
                    (index + 1) % args.progress_every == 0 or index + 1 == stop
                ):
                    print(
                        f"PROGRESS coverage={index + 1}/{len(all_coverage)}",
                        flush=True,
                    )
                if (index + 1) % 32 == 0:
                    gc.collect()
        print(f"PASS Python/C lockstep cases={validated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
