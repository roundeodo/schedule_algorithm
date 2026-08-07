#!/usr/bin/env python3
"""Emit end-to-end MMIO/refill/task vectors for the distilled wrapper."""

import json
from pathlib import Path

import four_stage_scheduler as reference
import scheduler_rtl_distilled_policy as policy
import scheduler_rtl_distilled_scoring as scoring
import verify_scheduler_rtl_unified_policy as datasets
from verify_distilled_window_protocol import WindowState, _sorted_distribution


HERE = Path(__file__).resolve().parent
PROOF = HERE / "results/policy_search/olmoe_top2_projection_65_optimal_v1.json"
VALIDATION = HERE / "results/policy_search/bounded_top5_bottom1_fixed_lane_targeted_s4pf_random_validation.json"
COVERAGE = tuple(HERE / f"scheduler_strategy_coverage_E{experts}.json"
                 for experts in (8, 32, 64))
SHAPE = {
    reference.SHAPE_A.name: 0,
    reference.SHAPE_B.name: 1,
    reference.SHAPE_C.name: 2,
}


def descriptor(entry: tuple[int, int]) -> int:
    eid, ntok = entry
    return (int(ntok) & 0x1FF) | ((int(eid) & 0x3F) << 9) | (1 << 15)


def quad(entries: list[tuple[int, int]]) -> int:
    word = 0
    for slot, entry in enumerate(entries[:4]):
        word |= descriptor(entry) << (16 * slot)
    return word


def config(c2: int, c3: int, count: int) -> int:
    c2_field = 0x80 if int(c2) < 0 else int(c2) & 0x3F
    c3_field = 0x80 if int(c3) < 0 else int(c3) & 0x3F
    return c2_field | (c3_field << 8) | ((int(count) & 0x7F) << 16)


def aggregate(remaining: list[tuple[int, int]]) -> int:
    counters = scoring.remaining_counters(tuple(remaining))
    word = counters.token_sum | (counters.odd_count << 9) | (
        counters.block_sum << 16
    )
    for bucket, count in enumerate(counters.small_block_hist):
        word |= int(count) << (25 + 7 * bucket)
    return word


def stage_tail(ntok: int, shape: int, skip: bool) -> int:
    if skip:
        return ntok
    tile = (8, 4, 2)[shape]
    return max(0, ntok - tile)


def pack_task(task: dict, slot: int, s4_desc: int) -> int:
    ntok = int(task["ntok"])
    s1 = int(task["shape_s1"])
    s3 = int(task["shape_s3"])
    skip1 = bool(task["skip_s1"])
    skip3 = bool(task["skip_s3"])
    m2 = (stage_tail(ntok, s1, skip1) + 1) // 2
    m4 = (stage_tail(ntok, s3, skip3) + 1) // 2
    control = int(skip1) | (int(skip3) << 1) | (s1 << 2) | (s3 << 4)
    control |= int(task["cluster"]) << 6
    control |= int(slot) << 7
    word = int(task["eid"]) | (int(task["tok_start"]) << 6)
    word |= ntok << 15
    word |= int(task["has_s2pf"]) << 24
    word |= control << 25
    word |= (m2 & 0xFF) << 38
    word |= int(task["s1_both"]) << 46
    word |= (m4 & 0xFF) << 47
    word |= int(task["late_both"]) << 55
    word |= (int(s4_desc) & 0xFF) << 56
    return word


def action_tasks(action) -> list[dict]:
    result = []
    for cluster in (2, 3):
        eid = int(getattr(action, f"c{cluster}_eid"))
        if eid < 0:
            continue
        s2pf = getattr(action, f"c{cluster}_s2pf_dma")
        dma_s1 = getattr(action, f"c{cluster}_dma_s1")
        dma_s3 = getattr(action, f"c{cluster}_dma_s3")
        result.append({
            "cluster": cluster - 2,
            "eid": eid,
            "ntok": int(getattr(action, f"c{cluster}_ntok")),
            "tok_start": 0,
            "shape_s1": SHAPE[getattr(action, f"c{cluster}_shape_s1").name],
            "shape_s3": SHAPE[getattr(action, f"c{cluster}_shape_s3").name],
            "skip_s1": bool(getattr(action, f"c{cluster}_s1_cached")),
            "skip_s3": bool(getattr(action, f"c{cluster}_s3_cached")),
            "has_s2pf": s2pf != reference.DmaBinding.NONE,
            "s1_both": dma_s1 == reference.DmaBinding.BOTH,
            "late_both": (
                s2pf if s2pf != reference.DmaBinding.NONE else dma_s3
            ) == reference.DmaBinding.BOTH,
        })
    if len(result) == 2 and result[0]["eid"] == result[1]["eid"]:
        result[1]["tok_start"] = result[0]["ntok"]
    return result


def task_words(schedule) -> list[int]:
    pending = [None, None]
    slots = [0, 0]
    words = []
    for step in schedule.steps:
        targets = {2: reference.DmaBinding.NONE, 3: reference.DmaBinding.NONE}
        for prefetch in step.s4pf_actions:
            targets[int(prefetch.pf_cluster)] = prefetch.pf_dma
        for task in action_tasks(step.action):
            cluster = int(task["cluster"])
            if pending[cluster] is not None:
                previous, previous_slot = pending[cluster]
                binding = targets[cluster + 2]
                s4_desc = 0
                if binding != reference.DmaBinding.NONE:
                    operation = 2 if binding == reference.DmaBinding.BOTH else 1
                    s4_desc = operation | (int(task["eid"]) << 2)
                words.append(pack_task(previous, previous_slot, s4_desc))
            pending[cluster] = (task, slots[cluster])
            slots[cluster] += 1
    for cluster in (0, 1):
        if pending[cluster] is not None:
            task, slot = pending[cluster]
            words.append(pack_task(task, slot, 0))
    return words


def build_case(job: dict) -> tuple:
    remaining = _sorted_distribution(dict(job["distribution"]))
    window = WindowState.initialize(remaining)
    schedule = policy.schedule(
        dict(job["distribution"]), int(job["c2"]), int(job["c3"]),
        enable_s4pf=True,
    )
    refills = []
    for step in schedule.steps:
        consumed = {
            int(eid) for eid in (step.action.c2_eid, step.action.c3_eid)
            if int(eid) >= 0
        }
        window.consume(consumed)
        top_count, bottom_count = window.refill_request()
        if top_count or bottom_count:
            hidden = window.hidden()
            top_entries = hidden[:top_count]
            bottom_entries = list(reversed(hidden[len(hidden)-bottom_count:]))
            entries = top_entries + bottom_entries
            beats = [quad(entries[offset:offset + 4])
                     for offset in range(0, len(entries), 4)]
            refills.append((top_count, bottom_count, beats))
            window.apply_refill(top_count, bottom_count)
    initial = WindowState.initialize(remaining)
    initial_stream = initial.top + initial.bottom
    initial_words = [quad(initial_stream[offset:offset + 4])
                     for offset in range(0, 16, 4)]
    return (
        config(job["c2"], job["c3"], len(remaining)),
        aggregate(remaining),
        *initial_words,
        refills, task_words(schedule),
    )


def main() -> int:
    jobs = list(datasets._proof_jobs(PROOF.resolve()))
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    s4_key = next(
        key for key, row in validation["rows"].items()
        if int(row["s4pf_events"]) > 0
    )
    coverage_jobs = {
        job["key"]: job
        for job in datasets._dataset_jobs(tuple(path.resolve() for path in COVERAGE))
    }
    jobs.append(coverage_jobs[s4_key])
    cases = [build_case(job) for job in jobs]
    print(len(cases))
    for cfg, agg, w0, w1, w2, w3, refills, words in cases:
        print(f"{cfg:016x} {agg:016x} {w0:016x} {w1:016x} "
              f"{w2:016x} {w3:016x} "
              f"{len(refills)} {len(words)}")
        for top_count, bottom_count, beats in refills:
            padded = beats + [0] * (2 - len(beats))
            print(f"{top_count} {bottom_count} {len(beats)} "
                  f"{padded[0]:016x} {padded[1]:016x}")
        for word in words:
            print(f"{word:016x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
