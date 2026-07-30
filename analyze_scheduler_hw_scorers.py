#!/usr/bin/env python3
"""Closed-loop screening of fixed-cost scorers on the deployed HW candidates.

Only the continuation score is replaced.  The deployed candidate generator,
timeline model, prefetch policy, bandwidth checks, and tie-break remain fixed.
No scorer below generates a child candidate or performs a rollout.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import eval_hw_mirror_s2pf_lite as hw
import scheduler_hw_fixed_policy as fixed
from analyze_scheduler_hw_tail import policy_consistent_n1_cost


ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = ROOT / "results" / "policy_search" / "scheduler_strategies_30k.json"
DEFAULT_SUITES = tuple(ROOT / f"scheduler_strategy_coverage_E{e}.json" for e in (8, 32, 64))
DEFAULT_REFERENCES = tuple(
    ROOT / "results" / "final_reference" / f"scheduler_reference_E{e}.json"
    for e in (8, 32, 64)
)
DEFAULT_OUT = ROOT / "results" / "policy_search" / "scheduler_hw_scorer_screen.json"
HW_CONFIG = {"policy": "balanced", "top_policy": "pruned", "n1_policy": "pruned"}
CANDIDATE_POLICY = "deployed"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))]


def original_sim1(c2, c3, remaining: tuple, *, policy: str) -> int:
    return ORIGINAL_CONTINUATION(c2, c3, remaining, policy=policy)


def greedy_no_sim1(c2, c3, remaining: tuple, *, policy: str) -> int:
    del policy
    if not remaining:
        return max(c2.task_end, c3.task_end)
    return hw.cm._cc_greedy_h(c2.task_end, c3.task_end, remaining)


def _balance_divisible_work(loads: list[int], work: int) -> None:
    low = 0 if loads[0] <= loads[1] else 1
    high = 1 - low
    fill = min(loads[high] - loads[low], work)
    loads[low] += fill
    work -= fill
    if work:
        loads[low] += work // 2
        loads[high] += work - work // 2


def _lpt_prefix_score(c2, c3, remaining: tuple, *, prefix: int, task_time) -> int:
    loads = [int(c2.task_end), int(c3.task_end)]
    head = remaining[:prefix]
    for _, ntok in head:
        target = 0 if loads[0] <= loads[1] else 1
        loads[target] += int(task_time(int(ntok)))
    middle_work = sum(int(task_time(int(ntok))) for _, ntok in remaining[prefix:])
    _balance_divisible_work(loads, middle_work)
    return max(loads)


def lpt1_task(c2, c3, remaining: tuple, *, policy: str) -> int:
    del policy
    return _lpt_prefix_score(c2, c3, remaining, prefix=1, task_time=hw.cm._cc_best_task)


def lpt2_task(c2, c3, remaining: tuple, *, policy: str) -> int:
    del policy
    return _lpt_prefix_score(c2, c3, remaining, prefix=2, task_time=hw.cm._cc_best_task)


def lpt3_task(c2, c3, remaining: tuple, *, policy: str) -> int:
    del policy
    return _lpt_prefix_score(c2, c3, remaining, prefix=3, task_time=hw.cm._cc_best_task)


def balanced_concurrent(c2, c3, remaining: tuple, *, policy: str) -> int:
    del policy
    loads = [int(c2.task_end), int(c3.task_end)]
    _balance_divisible_work(
        loads, sum(hw.cm._cc_best_conc(int(ntok)) for _, ntok in remaining)
    )
    return max(loads)


def lpt4_concurrent(c2, c3, remaining: tuple, *, policy: str) -> int:
    del policy
    return _lpt_prefix_score(
        c2, c3, remaining, prefix=4, task_time=hw.cm._cc_best_conc
    )


def lpt4_task(c2, c3, remaining: tuple, *, policy: str) -> int:
    del policy
    return _lpt_prefix_score(
        c2, c3, remaining, prefix=4, task_time=hw.cm._cc_best_task
    )


def hybrid_lpt4_concurrent(c2, c3, remaining: tuple, *, policy: str) -> int:
    if len(remaining) <= 2:
        return greedy_no_sim1(c2, c3, remaining, policy=policy)
    return lpt4_concurrent(c2, c3, remaining, policy=policy)


def hybrid_lpt4_task(c2, c3, remaining: tuple, *, policy: str) -> int:
    if len(remaining) <= 2:
        return greedy_no_sim1(c2, c3, remaining, policy=policy)
    return lpt4_task(c2, c3, remaining, policy=policy)


def _blend_hybrid_task(c2, c3, remaining: tuple, *, policy: str, lpt_weight: int) -> int:
    greedy = greedy_no_sim1(c2, c3, remaining, policy=policy)
    if len(remaining) <= 2:
        return greedy
    lpt = lpt4_task(c2, c3, remaining, policy=policy)
    # Denominator four keeps the deployed arithmetic to shifts and adds.
    return ((4 - lpt_weight) * greedy + lpt_weight * lpt + 2) // 4


def blend_lpt4_task_25(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _blend_hybrid_task(c2, c3, remaining, policy=policy, lpt_weight=1)


def blend_lpt4_task_50(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _blend_hybrid_task(c2, c3, remaining, policy=policy, lpt_weight=2)


def blend_lpt4_task_75(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _blend_hybrid_task(c2, c3, remaining, policy=policy, lpt_weight=3)


def max_greedy_lpt4_task(c2, c3, remaining: tuple, *, policy: str) -> int:
    greedy = greedy_no_sim1(c2, c3, remaining, policy=policy)
    if len(remaining) <= 2:
        return greedy
    return max(greedy, lpt4_task(c2, c3, remaining, policy=policy))


def min_greedy_lpt4_task(c2, c3, remaining: tuple, *, policy: str) -> int:
    greedy = greedy_no_sim1(c2, c3, remaining, policy=policy)
    if len(remaining) <= 2:
        return greedy
    return min(greedy, lpt4_task(c2, c3, remaining, policy=policy))


def _min_greedy_lpt_prefix(c2, c3, remaining: tuple, *, policy: str, prefix: int) -> int:
    greedy = greedy_no_sim1(c2, c3, remaining, policy=policy)
    if len(remaining) <= 2:
        return greedy
    lpt = _lpt_prefix_score(
        c2, c3, remaining, prefix=prefix, task_time=hw.cm._cc_best_task
    )
    return min(greedy, lpt)


def min_greedy_lpt1_task(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _min_greedy_lpt_prefix(c2, c3, remaining, policy=policy, prefix=1)


def min_greedy_lpt2_task(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _min_greedy_lpt_prefix(c2, c3, remaining, policy=policy, prefix=2)


def min_greedy_lpt3_task(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _min_greedy_lpt_prefix(c2, c3, remaining, policy=policy, prefix=3)


def _future_s4pf_count(c2, c3) -> int:
    count = 0
    if hw.cm._cc_s4pf_ok_with_peer(c2, c3):
        c2 = hw.cm._cc_apply_s4pf_ghost(c2)
        count += 1
    if hw.cm._cc_s4pf_ok_with_peer(c3, c2):
        count += 1
    return count


def _adjusted_min_lpt4(
    c2,
    c3,
    remaining: tuple,
    *,
    policy: str,
    gap_weight_eighths: int = 0,
    s4pf_reward_eighths: int = 0,
) -> int:
    base = min_greedy_lpt4_task(c2, c3, remaining, policy=policy)
    release_gap = abs(int(c2.task_end) - int(c3.task_end))
    s4pf_value = hw.cm.C_TD1[hw.cm.C_SHAPE_A] * _future_s4pf_count(c2, c3)
    # Scale by eight so every tested coefficient is a shift/add fraction of
    # a cycle-domain feature.  Absolute scale is irrelevant to candidate rank.
    return 8 * base + gap_weight_eighths * release_gap - s4pf_reward_eighths * s4pf_value


def min_lpt4_gap_m2(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _adjusted_min_lpt4(c2, c3, remaining, policy=policy, gap_weight_eighths=-2)


def min_lpt4_gap_m1(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _adjusted_min_lpt4(c2, c3, remaining, policy=policy, gap_weight_eighths=-1)


def min_lpt4_gap_p1(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _adjusted_min_lpt4(c2, c3, remaining, policy=policy, gap_weight_eighths=1)


def min_lpt4_gap_p2(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _adjusted_min_lpt4(c2, c3, remaining, policy=policy, gap_weight_eighths=2)


def min_lpt4_s4pf_r1(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _adjusted_min_lpt4(c2, c3, remaining, policy=policy, s4pf_reward_eighths=1)


def min_lpt4_s4pf_r2(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _adjusted_min_lpt4(c2, c3, remaining, policy=policy, s4pf_reward_eighths=2)


def min_lpt4_s4pf_r4(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _adjusted_min_lpt4(c2, c3, remaining, policy=policy, s4pf_reward_eighths=4)


def min_lpt4_gap_m1_s4pf_r2(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _adjusted_min_lpt4(
        c2, c3, remaining, policy=policy, gap_weight_eighths=-1, s4pf_reward_eighths=2
    )


def min_lpt4_gap_p1_s4pf_r2(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _adjusted_min_lpt4(
        c2, c3, remaining, policy=policy, gap_weight_eighths=1, s4pf_reward_eighths=2
    )


def _guarded_min_lpt4(
    c2,
    c3,
    remaining: tuple,
    *,
    policy: str,
    max_greedy_tokens: int,
    sim1_tail: bool,
) -> int:
    if not remaining:
        return max(c2.task_end, c3.task_end)
    if len(remaining) == 1 and sim1_tail:
        return ORIGINAL_CONTINUATION(c2, c3, remaining, policy=policy)
    greedy = greedy_no_sim1(c2, c3, remaining, policy=policy)
    if len(remaining) <= 2 or sum(ntok for _, ntok in remaining) <= max_greedy_tokens:
        return greedy
    return min(greedy, lpt4_task(c2, c3, remaining, policy=policy))


def min_lpt4_sim1(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _guarded_min_lpt4(
        c2, c3, remaining, policy=policy, max_greedy_tokens=-1, sim1_tail=True
    )


def min_lpt4_guard_t4(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _guarded_min_lpt4(
        c2, c3, remaining, policy=policy, max_greedy_tokens=4, sim1_tail=False
    )


def min_lpt4_guard_t8(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _guarded_min_lpt4(
        c2, c3, remaining, policy=policy, max_greedy_tokens=8, sim1_tail=False
    )


def min_lpt4_guard_t12(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _guarded_min_lpt4(
        c2, c3, remaining, policy=policy, max_greedy_tokens=12, sim1_tail=False
    )


def min_lpt4_sim1_guard_t4(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _guarded_min_lpt4(
        c2, c3, remaining, policy=policy, max_greedy_tokens=4, sim1_tail=True
    )


def min_lpt4_sim1_guard_t8(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _guarded_min_lpt4(
        c2, c3, remaining, policy=policy, max_greedy_tokens=8, sim1_tail=True
    )


def min_lpt4_sim1_guard_t12(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _guarded_min_lpt4(
        c2, c3, remaining, policy=policy, max_greedy_tokens=12, sim1_tail=True
    )


def min_lpt4_exact_n1_diagnostic(c2, c3, remaining: tuple, *, policy: str) -> int:
    if len(remaining) == 1:
        eid, ntok = remaining[0]
        return policy_consistent_n1_cost(c2, c3, eid, ntok, policy=policy)
    return min_greedy_lpt4_task(c2, c3, remaining, policy=policy)


def min_lpt4_exact_n1_guard_t8_diagnostic(
    c2, c3, remaining: tuple, *, policy: str
) -> int:
    if len(remaining) == 1:
        eid, ntok = remaining[0]
        return policy_consistent_n1_cost(c2, c3, eid, ntok, policy=policy)
    return _guarded_min_lpt4(
        c2, c3, remaining, policy=policy, max_greedy_tokens=8, sim1_tail=False
    )


def _sim1_and_length_guard(
    c2, c3, remaining: tuple, *, policy: str, max_greedy_len: int
) -> int:
    if len(remaining) == 1:
        return ORIGINAL_CONTINUATION(c2, c3, remaining, policy=policy)
    greedy = greedy_no_sim1(c2, c3, remaining, policy=policy)
    if len(remaining) <= max_greedy_len:
        return greedy
    return min(greedy, lpt4_task(c2, c3, remaining, policy=policy))


def min_lpt4_sim1_guard_n3(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _sim1_and_length_guard(
        c2, c3, remaining, policy=policy, max_greedy_len=3
    )


def min_lpt4_sim1_guard_n4(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _sim1_and_length_guard(
        c2, c3, remaining, policy=policy, max_greedy_len=4
    )


def min_lpt4_sim1_guard_n5(c2, c3, remaining: tuple, *, policy: str) -> int:
    return _sim1_and_length_guard(
        c2, c3, remaining, policy=policy, max_greedy_len=5
    )


# ---------------------------------------------------------------------------
# Non-LPT scorer family 1: release/work/critical-chain/DMA lower bounds.
# ---------------------------------------------------------------------------


def _release_compute_components(c2, c3, remaining: tuple) -> tuple[int, int, int]:
    latest = max(int(c2.task_end), int(c3.task_end))
    if not remaining:
        return latest, latest, latest
    phase = hw.cm._cc_best_task(1)  # one M=2 compute block: 33,792 cycles
    total_blocks = sum((int(ntok) + 1) // 2 for _, ntok in remaining)
    crossing_num = int(c3.task_end) - int(c2.task_end) + total_blocks * phase
    crossing_den = 2 * phase
    floor = crossing_num // crossing_den
    ceil = -(-crossing_num // crossing_den)
    allocations = {
        0,
        total_blocks,
        max(0, min(total_blocks, floor)),
        max(0, min(total_blocks, ceil)),
    }
    work_lb = min(
        max(
            int(c2.task_end) + blocks_c2 * phase,
            int(c3.task_end) + (total_blocks - blocks_c2) * phase,
        )
        for blocks_c2 in allocations
    )
    early, late = sorted((int(c2.task_end), int(c3.task_end)))
    largest_blocks = (int(remaining[0][1]) + 1) // 2
    critical_lb = min(
        early + largest_blocks * phase,
        late + ((largest_blocks + 1) // 2) * phase,
    )
    return latest, work_lb, critical_lb


def _mandatory_dma_capacity_lb(c2, c3, remaining: tuple) -> int:
    if not remaining:
        return max(int(c2.task_end), int(c3.task_end))
    # Install the same deterministic S4PF ghosts that the next real round will
    # install.  Their bytes are already represented by committed segments and
    # must not be counted again as remaining DMA work.
    c2, c3 = fixed._prepare(c2, c3)
    count = len(remaining)
    valid_slots = sum(snap.pf_eid != -1 and snap.pf_end >= 0 for snap in (c2, c3))
    full_slots = sum(
        snap.pf_eid != -1 and snap.pf_end >= 0 and bool(snap.pf_full)
        for snap in (c2, c3)
    )
    s1_bytes = hw.cm.C_TD1[hw.cm.C_SHAPE_A] * hw.cm.C_ALLOC[hw.cm.C_SHAPE_A]
    s3_bytes = hw.cm.C_TD3[hw.cm.C_SHAPE_A] * hw.cm.C_ALLOC[hw.cm.C_SHAPE_A]
    required = (
        max(0, count - min(count, valid_slots)) * s1_bytes
        + max(0, count - min(count, full_slots)) * s3_bytes
    )
    start = min(int(c2.task_end), int(c3.task_end))
    latest = max(int(c2.task_end), int(c3.task_end))
    segments = [*hw.cm._cc_snap_segs(c2), *hw.cm._cc_snap_segs(c3)]
    points = {start}
    for lo, hi, _ in segments:
        if hi >= start:
            points.add(max(start, int(lo)))
            points.add(int(hi))
    ordered = sorted(points)
    for left, right in zip(ordered, ordered[1:]):
        if right <= left:
            continue
        used = sum(int(bw) for lo, hi, bw in segments if lo <= left < hi)
        free = max(0, hw.cm.C_MAX_BW - used)
        capacity = (right - left) * free
        if free and required <= capacity:
            return max(latest, left + (required + free - 1) // free)
        required -= min(required, capacity)
    tail = ordered[-1]
    return max(latest, tail + (required + hw.cm.C_MAX_BW - 1) // hw.cm.C_MAX_BW)


def multi_lower_bound_compute(c2, c3, remaining: tuple, *, policy: str) -> int:
    del policy
    return max(_release_compute_components(c2, c3, remaining))


def multi_lower_bound_dma(c2, c3, remaining: tuple, *, policy: str) -> int:
    del policy
    return max(
        *_release_compute_components(c2, c3, remaining),
        _mandatory_dma_capacity_lb(c2, c3, remaining),
    )


# ---------------------------------------------------------------------------
# Non-LPT scorer family 2: cache/DMA-aware bounded list scheduling.
# ---------------------------------------------------------------------------


def _earliest_bw_start(ready: int, duration: int, bw: int, peer) -> int:
    start = int(ready)
    segments = hw.cm._cc_snap_segs(peer)
    while True:
        conflicts = [
            int(hi)
            for lo, hi, peer_bw in segments
            if max(start, int(lo)) < min(start + duration, int(hi))
            and bw + int(peer_bw) > hw.cm.C_MAX_BW
        ]
        if not conflicts:
            return start
        start = max(conflicts)


def _isolated_finish(
    ready: int,
    eid: int,
    ntok: int,
    cache_snap,
    peer_snap,
    *,
    dma_aware: bool,
) -> int:
    sw_hit = cache_snap is not None and hw.cm._cc_swiglu_hit(eid, cache_snap, ready)
    down_hit = cache_snap is not None and hw.cm._cc_down_hit(eid, cache_snap, ready)
    if sw_hit:
        s1_options = [(int(ready), hw.cm._cc_best_s2(ntok))]
    else:
        s1_options = []
        for shape in (hw.cm.C_SHAPE_A, hw.cm.C_SHAPE_B, hw.cm.C_SHAPE_C):
            start = (
                _earliest_bw_start(
                    ready,
                    hw.cm.C_TD1[shape],
                    hw.cm.C_ALLOC[shape],
                    peer_snap,
                )
                if dma_aware
                else int(ready)
            )
            tail = max(0, ntok - hw.cm.C_MDIM[shape])
            s1_options.append(
                (start, hw.cm.C_TS1[shape] + hw.cm._cc_best_s2(tail))
            )

    best = hw.cm.C_INF
    for start, s1_phase in s1_options:
        s2_end = start + s1_phase
        if down_hit:
            best = min(best, s2_end + hw.cm._cc_best_s4(ntok))
            continue
        for shape in (hw.cm.C_SHAPE_A, hw.cm.C_SHAPE_B, hw.cm.C_SHAPE_C):
            s3_start = (
                _earliest_bw_start(
                    s2_end,
                    hw.cm.C_TD3[shape],
                    hw.cm.C_ALLOC[shape],
                    peer_snap,
                )
                if dma_aware
                else s2_end
            )
            tail = max(0, ntok - hw.cm.C_MDIM[shape])
            finish = s3_start + hw.cm.C_TS3[shape] + hw.cm._cc_best_s4(tail)
            best = min(best, finish)
    return int(best)


def _cache_list_score(c2, c3, remaining: tuple, *, dma_aware: bool) -> int:
    if not remaining:
        return max(int(c2.task_end), int(c3.task_end))
    c2, c3 = fixed._prepare(c2, c3)
    loads = [int(c2.task_end), int(c3.task_end)]
    committed = [c2, c3]
    cache_live = [True, True]
    for eid, ntok in remaining[:4]:
        finishes = []
        for lane in (0, 1):
            finishes.append(
                _isolated_finish(
                    loads[lane],
                    int(eid),
                    int(ntok),
                    committed[lane] if cache_live[lane] else None,
                    committed[1 - lane],
                    dma_aware=dma_aware,
                )
            )
        lane = 0 if finishes[0] <= finishes[1] else 1
        loads[lane] = finishes[lane]
        cache_live[lane] = False
    tail_work = sum(hw.cm._cc_best_task(int(ntok)) for _, ntok in remaining[4:])
    _balance_divisible_work(loads, tail_work)
    return max(loads)


def cache_list_top4(c2, c3, remaining: tuple, *, policy: str) -> int:
    del policy
    return _cache_list_score(c2, c3, remaining, dma_aware=False)


def cache_dma_list_top4(c2, c3, remaining: tuple, *, policy: str) -> int:
    del policy
    return _cache_list_score(c2, c3, remaining, dma_aware=True)


def min_greedy_cache_list(c2, c3, remaining: tuple, *, policy: str) -> int:
    return min(
        greedy_no_sim1(c2, c3, remaining, policy=policy),
        cache_list_top4(c2, c3, remaining, policy=policy),
    )


def min_greedy_cache_dma_list(c2, c3, remaining: tuple, *, policy: str) -> int:
    return min(
        greedy_no_sim1(c2, c3, remaining, policy=policy),
        cache_dma_list_top4(c2, c3, remaining, policy=policy),
    )


ORIGINAL_CONTINUATION = hw._hw_continuation_cost
SCORERS = {
    "original_sim1": original_sim1,
    "greedy_no_sim1": greedy_no_sim1,
    "balanced_concurrent": balanced_concurrent,
    "lpt4_concurrent": lpt4_concurrent,
    "lpt4_task": lpt4_task,
    "hybrid_lpt4_concurrent": hybrid_lpt4_concurrent,
    "hybrid_lpt4_task": hybrid_lpt4_task,
    "blend_lpt4_task_25": blend_lpt4_task_25,
    "blend_lpt4_task_50": blend_lpt4_task_50,
    "blend_lpt4_task_75": blend_lpt4_task_75,
    "max_greedy_lpt4_task": max_greedy_lpt4_task,
    "min_greedy_lpt4_task": min_greedy_lpt4_task,
    "lpt1_task": lpt1_task,
    "lpt2_task": lpt2_task,
    "lpt3_task": lpt3_task,
    "min_greedy_lpt1_task": min_greedy_lpt1_task,
    "min_greedy_lpt2_task": min_greedy_lpt2_task,
    "min_greedy_lpt3_task": min_greedy_lpt3_task,
    "min_lpt4_gap_m2": min_lpt4_gap_m2,
    "min_lpt4_gap_m1": min_lpt4_gap_m1,
    "min_lpt4_gap_p1": min_lpt4_gap_p1,
    "min_lpt4_gap_p2": min_lpt4_gap_p2,
    "min_lpt4_s4pf_r1": min_lpt4_s4pf_r1,
    "min_lpt4_s4pf_r2": min_lpt4_s4pf_r2,
    "min_lpt4_s4pf_r4": min_lpt4_s4pf_r4,
    "min_lpt4_gap_m1_s4pf_r2": min_lpt4_gap_m1_s4pf_r2,
    "min_lpt4_gap_p1_s4pf_r2": min_lpt4_gap_p1_s4pf_r2,
    "min_lpt4_sim1": min_lpt4_sim1,
    "min_lpt4_guard_t4": min_lpt4_guard_t4,
    "min_lpt4_guard_t8": min_lpt4_guard_t8,
    "min_lpt4_guard_t12": min_lpt4_guard_t12,
    "min_lpt4_sim1_guard_t4": min_lpt4_sim1_guard_t4,
    "min_lpt4_sim1_guard_t8": min_lpt4_sim1_guard_t8,
    "min_lpt4_sim1_guard_t12": min_lpt4_sim1_guard_t12,
    "min_lpt4_exact_n1_diagnostic": min_lpt4_exact_n1_diagnostic,
    "min_lpt4_exact_n1_guard_t8_diagnostic": min_lpt4_exact_n1_guard_t8_diagnostic,
    "min_lpt4_sim1_guard_n3": min_lpt4_sim1_guard_n3,
    "min_lpt4_sim1_guard_n4": min_lpt4_sim1_guard_n4,
    "min_lpt4_sim1_guard_n5": min_lpt4_sim1_guard_n5,
    "multi_lower_bound_compute": multi_lower_bound_compute,
    "multi_lower_bound_dma": multi_lower_bound_dma,
    "cache_list_top4": cache_list_top4,
    "cache_dma_list_top4": cache_dma_list_top4,
    "min_greedy_cache_list": min_greedy_cache_list,
    "min_greedy_cache_dma_list": min_greedy_cache_dma_list,
}


def load_cases(report: dict, suite_paths: tuple[Path, ...]) -> list[tuple[str, dict]]:
    wanted = set(report["results"])
    cases = {}
    for path in suite_paths:
        for case in json.loads(path.read_text())["cases"]:
            key = f"E{int(case['e_total'])}:{int(case['case_id'])}"
            if key in wanted:
                cases[key] = case
    missing = wanted - set(cases)
    if missing:
        raise ValueError(f"missing {len(missing)} checkpoint cases")
    return sorted(cases.items(), key=lambda item: (int(item[1]["e_total"]), int(item[1]["case_id"])))


def full_reference_report(reference_paths: tuple[Path, ...]) -> dict:
    results = {}
    for path in reference_paths:
        payload = json.loads(path.read_text())
        for result in payload["results"].values():
            if not result.get("analysis_eligible", False):
                continue
            key = f"E{int(result['e_total'])}:{int(result['case_id'])}"
            results[key] = {
                "reference_makespan_cc": int(result["makespan_cc"]),
                "reference_proven_optimal": bool(result.get("proven_optimal", False)),
            }
    return {"results": results, "analysis_eligible_cases": len(results)}


def run_scorer(case: dict, scorer) -> int:
    distribution = {int(eid): int(ntok) for eid, ntok in case["dist"].items()}
    return int(
        fixed.schedule_with_scorer(
            distribution,
            int(case.get("c2", -1)),
            int(case.get("c3", -1)),
            continuation=scorer,
            candidate_policy=CANDIDATE_POLICY,
            **HW_CONFIG,
        )
    )


def compare(rows: list[dict], scorer: str) -> dict:
    baseline = "original_sim1"
    better = [row for row in rows if row[scorer] < row[baseline]]
    worse = [row for row in rows if row[scorer] > row[baseline]]
    base_total = sum(row[baseline] for row in rows)
    score_total = sum(row[scorer] for row in rows)
    ratios = [row[scorer] / row[baseline] for row in rows]
    return {
        "cases": len(rows),
        "better": len(better),
        "worse": len(worse),
        "equal": len(rows) - len(better) - len(worse),
        "saved_cc": sum(row[baseline] - row[scorer] for row in better),
        "lost_cc": sum(row[scorer] - row[baseline] for row in worse),
        "scorer_over_original_aggregate": score_total / base_total,
        "ratio_mean": sum(ratios) / len(ratios),
        "ratio_p95": percentile(ratios, 0.95),
        "ratio_max": max(ratios),
    }


def versus_reference(rows: list[dict], scorer: str) -> dict:
    proven = [row for row in rows if row["reference_proven_optimal"]]
    reference_total = sum(row["reference_makespan_cc"] for row in proven)
    scorer_total = sum(row[scorer] for row in proven)
    ratios = [row[scorer] / row["reference_makespan_cc"] for row in proven]
    return {
        "cases": len(proven),
        "exact": sum(row[scorer] == row["reference_makespan_cc"] for row in proven),
        "scorer_over_reference_aggregate": scorer_total / reference_total,
        "ratio_mean": sum(ratios) / len(ratios),
        "ratio_p95": percentile(ratios, 0.95),
        "ratio_max": max(ratios),
    }


def summarize(rows: list[dict]) -> dict:
    buckets = defaultdict(list)
    for row in rows:
        for key in (
            "overall",
            f"E{row['e_total']}",
            f"split:{row['dataset_split']}",
            f"E{row['e_total']}:split:{row['dataset_split']}",
        ):
            buckets[key].append(row)
    return {
        key: {
            "vs_original": {name: compare(bucket, name) for name in SCORERS if name != "original_sim1"},
            "vs_proven_reference": {name: versus_reference(bucket, name) for name in SCORERS},
        }
        for key, bucket in sorted(buckets.items())
    }


def main() -> int:
    global CANDIDATE_POLICY, SCORERS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--suite", action="append", type=Path)
    parser.add_argument("--reference", action="append", type=Path)
    parser.add_argument(
        "--full-reference",
        action="store_true",
        help="evaluate every eligible suite case against final_reference instead of a checkpoint",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--scorer",
        action="append",
        choices=tuple(SCORERS),
        help="repeat to evaluate a subset; original_sim1 is always included",
    )
    parser.add_argument(
        "--candidate-policy",
        choices=("deployed", "one_idle_shape_v2", "resident_v2", "resident_shape_v2"),
        default="deployed",
    )
    args = parser.parse_args()

    CANDIDATE_POLICY = args.candidate_policy

    if args.scorer:
        requested = ["original_sim1", *args.scorer]
        SCORERS = {name: SCORERS[name] for name in dict.fromkeys(requested)}

    suite_paths = tuple(args.suite) if args.suite else DEFAULT_SUITES
    reference_paths = tuple(args.reference) if args.reference else DEFAULT_REFERENCES
    if args.full_reference:
        report_hash = None
        report = full_reference_report(reference_paths)
    else:
        report_hash = file_sha256(args.report)
        report = json.loads(args.report.read_text())
    cases = load_cases(report, suite_paths)
    if args.limit is not None:
        cases = cases[: args.limit]

    rows = []
    started = time.perf_counter()
    for index, (key, case) in enumerate(cases, 1):
        source = report["results"][key]
        scores = {name: run_scorer(case, scorer) for name, scorer in SCORERS.items()}
        if args.full_reference:
            distribution = {int(eid): int(ntok) for eid, ntok in case["dist"].items()}
            recorded = int(
                hw.hw_mirror_schedule(
                    distribution,
                    int(case.get("c2", -1)),
                    int(case.get("c3", -1)),
                    **HW_CONFIG,
                )
            )
        else:
            recorded = int(source["makespan_cc"]["current_hardware_lite"])
        if CANDIDATE_POLICY == "deployed" and scores["original_sim1"] != recorded:
            raise RuntimeError(
                f"baseline mismatch for {key}: transition={scores['original_sim1']} mirror={recorded}"
            )
        rows.append({
            "key": key,
            "e_total": int(case["e_total"]),
            "case_id": int(case["case_id"]),
            "dataset_split": case.get("dataset_split"),
            "active_n": int(case.get("active_n", len(case["dist"]))),
            "m_total": int(case.get("m_total", 0)),
            "construction": case.get("construction"),
            "cache_regime": case.get("cache_regime"),
            "reference_makespan_cc": int(source["reference_makespan_cc"]),
            "reference_proven_optimal": bool(source["reference_proven_optimal"]),
            "deployed_old_hw": recorded,
            **scores,
        })
        if args.progress_every > 0 and index % args.progress_every == 0:
            print(f"scorer-screen completed={index}/{len(cases)} elapsed_s={time.perf_counter()-started:.1f}", flush=True)

    if report_hash is not None and file_sha256(args.report) != report_hash:
        raise RuntimeError("input 30K checkpoint changed during scorer screen")

    summary = summarize(rows)
    payload = {
        "schema": "scheduler_hw_scorer_screen_v1",
        "provisional": len(cases) != int(report.get("analysis_eligible_cases", len(cases))),
        "configuration": {
            "source_report": str(args.report.resolve()),
            "source_report_sha256": report_hash,
            "source_report_cases": len(report["results"]),
            "suites": [{"path": str(path.resolve()), "sha256": file_sha256(path)} for path in suite_paths],
            "references": (
                [{"path": str(path.resolve()), "sha256": file_sha256(path)} for path in reference_paths]
                if args.full_reference
                else None
            ),
            "hw_mirror": {"path": str(Path(hw.__file__).resolve()), "sha256": file_sha256(Path(hw.__file__)), **HW_CONFIG},
            "scorers": list(SCORERS),
            "candidate_policy": (
                "deployed hardware candidates unchanged"
                if CANDIDATE_POLICY == "deployed"
                else "bounded fixed candidate augmentation under evaluation"
            ),
            "candidate_policy_revision": CANDIDATE_POLICY,
            "runtime_search": False,
        },
        "cases": len(rows),
        "runtime_s": time.perf_counter() - started,
        "summary": summary,
        "results": {row["key"]: {k: v for k, v in row.items() if k != "key"} for row in rows},
    }
    atomic_write_json(args.out, payload)
    print(f"wrote {args.out}")
    for name in SCORERS:
        ref = summary["overall"]["vs_proven_reference"][name]
        if name == "original_sim1":
            print(name, f"proven_gap_pct={(ref['scorer_over_reference_aggregate']-1)*100:.6f}", f"exact={ref['exact']}")
        else:
            comp = summary["overall"]["vs_original"][name]
            print(name, f"delta_pct={(comp['scorer_over_original_aggregate']-1)*100:.6f}", f"better={comp['better']}", f"worse={comp['worse']}", f"max_reg_pct={(comp['ratio_max']-1)*100:.6f}", f"proven_gap_pct={(ref['scorer_over_reference_aggregate']-1)*100:.6f}", f"exact={ref['exact']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
