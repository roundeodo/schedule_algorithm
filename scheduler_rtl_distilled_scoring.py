#!/usr/bin/env python3
"""Final bounded continuation arithmetic and regime-aware winner comparator."""

from __future__ import annotations

from dataclasses import dataclass, replace

import four_stage_scheduler as reference
import scheduler_rtl_distilled_lowering as lowering
from scheduler_rtl_distilled_types import TICK_CC, WINDOW


SCORER_ID = "bounded-regime-aware-continuation"

# Frozen integer thresholds.  All time thresholds are integer multiples of the
# scheduling quantum and require no programmable coefficients in RTL.
LOW_WORK_TOKEN_SUM_MAX = 84
LOW_WORK_ODD_COUNT_MAX = 9
LOW_WORK_FIFTH_LOAD_MAX = 4
HOTSPOT_RATIO = 2
COLD_COUNT_SCALE = 32
COLD_COUNT_THRESHOLD = 11
PLATEAU_MIN_REMAINING = 8
PLATEAU_SECOND_LOAD_MIN = 5
PLATEAU_FIRST_LOAD_MAX = 6
PLATEAU_IMBALANCE_TICKS = 3
TAIL_MAX_REMAINING = 7
TAIL_IMBALANCE_TICKS = 6
SLACK_MIN_REMAINING = 8
SLACK_MAX_REMAINING = 16
SLACK_SECOND_LOAD_MIN = 8
SLACK_IMBALANCE_TICKS = 9
HOT_DMA_HIGH_TICKS = 115
HOT_DMA_MIN2_TICKS = 102


@dataclass(frozen=True)
class RemainingCounters:
    count: int
    token_sum: int
    odd_count: int
    block_sum: int
    best_work_cc: int
    small_block_hist: tuple[int, int, int, int]


@dataclass(frozen=True)
class RegimeState:
    low_work_progress: bool
    sparse_hot_sync: bool
    mid_plateau: bool
    short_tail_plateau: bool
    large_slack_fill: bool


def remaining_counters(
    remaining: tuple[tuple[int, int], ...],
) -> RemainingCounters:
    """Counters initialized once by software and decremented after each commit."""
    loads = [int(ntok) for _eid, ntok in remaining]
    block_counts = [
        (ntok + reference.FULL_M_DIM - 1) // reference.FULL_M_DIM
        for ntok in loads
    ]
    return RemainingCounters(
        count=len(loads),
        token_sum=sum(loads),
        odd_count=sum(ntok & 1 for ntok in loads),
        block_sum=sum(block_counts),
        best_work_cc=sum(reference._best_task_time(ntok) for ntok in loads),
        small_block_hist=tuple(
            sum(blocks == bucket for blocks in block_counts)
            for bucket in range(1, 5)
        ),
    )


def _compute_capacity_bound(
    c2_end: int,
    c3_end: int,
    total_blocks: int,
) -> int:
    if total_blocks <= 0:
        return max(c2_end, c3_end)
    phase_block = reference.SHAPE_C.T_s1 + reference.SHAPE_C.T_s3
    crossing_num = c3_end - c2_end + total_blocks * phase_block
    crossing_den = 2 * phase_block
    crossing_floor = crossing_num // crossing_den
    crossing_ceil = -(-crossing_num // crossing_den)
    candidates = {
        0,
        total_blocks,
        max(0, min(total_blocks, crossing_floor)),
        max(0, min(total_blocks, crossing_ceil)),
    }
    return min(
        max(
            c2_end + c2_blocks * phase_block,
            c3_end + (total_blocks - c2_blocks) * phase_block,
        )
        for c2_blocks in candidates
    )


def lower_bound_components(state: reference.BeamState) -> dict[str, int]:
    """Bounded compute, critical-chain and DMA-capacity lower bounds."""
    counters = remaining_counters(state.remaining)
    c2_end = int(state.c2.task_end)
    c3_end = int(state.c3.task_end)
    earliest, latest = sorted((c2_end, c3_end))
    compute = _compute_capacity_bound(c2_end, c3_end, counters.block_sum)
    if state.remaining:
        hottest = state.remaining[:1]
        release_chain = reference._release_aware_expert_chain_lb(
            c2_end, c3_end, hottest
        )
        critical_chain = earliest + reference._critical_expert_chain_lb(
            state.c2, state.c3, hottest
        )
    else:
        release_chain = latest
        critical_chain = earliest
    mandatory_dma_bytes = reference._minimum_remaining_dma_bytes(
        state.c2, state.c3, state.remaining
    )
    dma_release = reference._earliest_relaxed_dma_release(state.c2, state.c3)
    dma_capacity = max(
        latest,
        reference._dma_capacity_finish_lb(
            state.c2,
            state.c3,
            dma_release,
            mandatory_dma_bytes,
        ),
    )
    return {
        "committed_cc": latest,
        "compute_cc": compute,
        "release_expert_chain_cc": release_chain,
        "critical_chain_cc": critical_chain,
        "mandatory_dma_bytes": mandatory_dma_bytes,
        "dma_release_cc": dma_release,
        "dma_capacity_cc": dma_capacity,
        "combined_cc": max(
            latest,
            compute,
            release_chain,
            critical_chain,
            dma_capacity,
        ),
    }


def normalize_state_bound(
    state: reference.BeamState,
    *,
    parent_bound: int | None = None,
) -> reference.BeamState:
    """Replace the reference bound with the final bounded monotone bound."""
    bound = lower_bound_components(state)["combined_cc"]
    if parent_bound is not None:
        bound = max(int(parent_bound), bound)
    return replace(state, f_score=bound)


def head5_hist4_lpt(state: reference.BeamState) -> int:
    """LPT estimate using five descriptors, four tail bins and aggregate work."""
    head_entries = state.remaining[:5]
    counters = remaining_counters(state.remaining)
    tail_hist = list(counters.small_block_hist)
    for _eid, ntok in head_entries:
        blocks = (
            int(ntok) + reference.FULL_M_DIM - 1
        ) // reference.FULL_M_DIM
        if blocks <= 4:
            tail_hist[blocks - 1] -= 1
    if min(tail_hist, default=0) < 0:
        raise AssertionError("negative cold-tail histogram count")

    tail_work = counters.best_work_cc - sum(
        reference._best_task_time(int(ntok)) for _eid, ntok in head_entries
    )
    loads = [int(state.c2.task_end), int(state.c3.task_end)]
    for _eid, ntok in head_entries:
        target = 0 if loads[0] <= loads[1] else 1
        loads[target] += reference._best_task_time(int(ntok))

    phase_block_cc = reference._best_task_time(reference.FULL_M_DIM)
    histogram_work = sum(
        count * blocks * phase_block_cc
        for blocks, count in enumerate(tail_hist, start=1)
    )
    overflow_work = tail_work - histogram_work
    if overflow_work < 0:
        raise AssertionError("negative aggregate tail work")
    for blocks in range(4, 0, -1):
        for _ in range(tail_hist[blocks - 1]):
            target = 0 if loads[0] <= loads[1] else 1
            loads[target] += blocks * phase_block_cc
    if overflow_work:
        low = 0 if loads[0] <= loads[1] else 1
        high = 1 - low
        fill = min(loads[high] - loads[low], overflow_work)
        loads[low] += fill
        overflow_work -= fill
        loads[low] += overflow_work // 2
        loads[high] += overflow_work - overflow_work // 2
    return max(int(state.f_score), max(loads))


def base_continuation_key(
    before: reference.BeamState,
    action: reference.StageAction,
    child: reference.BeamState,
) -> tuple[int, ...]:
    """Common F/H/C/D key and mode-specific secondary priorities."""
    maximum, _minimum, selected_sum, s2pf = lowering.selected_action_features(
        action
    )
    components = lower_bound_components(child)
    common = (
        int(child.f_score),
        int(head5_hist4_lpt(child)),
        int(components["compute_cc"]),
        int(components["dma_capacity_cc"]),
    )
    if lowering.mode(before) == "SYNC":
        return common + (
            -maximum,
            selected_sum,
            int(child.g_score),
            -s2pf,
        )
    early, late = sorted((int(child.c2.task_end), int(child.c3.task_end)))
    return common + (
        late,
        early,
        selected_sum,
        int(child.g_score),
        -s2pf,
        len(child.remaining),
    )


def _progress_key(
    action: reference.StageAction,
    child: reference.BeamState,
) -> tuple[int, ...]:
    components = lower_bound_components(child)
    maximum, _minimum, selected_sum, s2pf = lowering.selected_action_features(
        action
    )
    early, late = sorted((int(child.c2.task_end), int(child.c3.task_end)))
    return (
        int(child.f_score),
        int(head5_hist4_lpt(child)),
        -s2pf,
        -selected_sum,
        int(components["compute_cc"]),
        int(components["dma_capacity_cc"]),
        late,
        early,
        int(child.g_score),
        -maximum,
    )


def _hotspot_key(
    action: reference.StageAction,
    child: reference.BeamState,
) -> tuple[int, ...]:
    components = lower_bound_components(child)
    maximum, _minimum, selected_sum, s2pf = lowering.selected_action_features(
        action
    )
    return (
        int(child.f_score),
        int(head5_hist4_lpt(child)),
        -maximum,
        selected_sum,
        int(components["compute_cc"]),
        int(components["dma_capacity_cc"]),
        int(child.g_score),
        -s2pf,
    )


def classify_regime(state: reference.BeamState) -> RegimeState:
    counters = remaining_counters(state.remaining)
    count = counters.count
    top_entries = state.remaining[: min(WINDOW[0], count)]
    top_loads = [int(ntok) for _eid, ntok in top_entries]
    top_loads.extend([0] * (WINDOW[0] - len(top_loads)))
    imbalance = abs(int(state.c2.task_end) - int(state.c3.task_end))
    current_mode = lowering.mode(state)
    return RegimeState(
        low_work_progress=(
            current_mode == "ONE_IDLE"
            and counters.token_sum <= LOW_WORK_TOKEN_SUM_MAX
            and counters.odd_count <= LOW_WORK_ODD_COUNT_MAX
            and top_loads[4] <= LOW_WORK_FIFTH_LOAD_MAX
        ),
        sparse_hot_sync=(
            current_mode == "SYNC"
            and count >= 2
            and top_loads[0] >= HOTSPOT_RATIO * top_loads[1]
            and COLD_COUNT_SCALE * counters.small_block_hist[0]
            > COLD_COUNT_THRESHOLD * count
        ),
        mid_plateau=(
            current_mode == "ONE_IDLE"
            and count >= PLATEAU_MIN_REMAINING
            and top_loads[1] >= PLATEAU_SECOND_LOAD_MIN
            and top_loads[0] <= PLATEAU_FIRST_LOAD_MAX
            and imbalance == PLATEAU_IMBALANCE_TICKS * TICK_CC
        ),
        short_tail_plateau=(
            current_mode == "ONE_IDLE"
            and 2 <= count <= TAIL_MAX_REMAINING
            and top_loads[1] >= PLATEAU_SECOND_LOAD_MIN
            and top_loads[0] <= PLATEAU_FIRST_LOAD_MAX
            and imbalance == TAIL_IMBALANCE_TICKS * TICK_CC
        ),
        large_slack_fill=(
            current_mode == "ONE_IDLE"
            and SLACK_MIN_REMAINING <= count <= SLACK_MAX_REMAINING
            and top_loads[1] >= SLACK_SECOND_LOAD_MIN
            and imbalance >= SLACK_IMBALANCE_TICKS * TICK_CC
        ),
    )


def select_continuation_winner(
    state: reference.BeamState,
    candidates: list[reference.StageAction],
) -> tuple[tuple[int, ...], int, reference.StageAction, reference.BeamState, dict]:
    """Fold one fixed-order logical-candidate stream through one comparator."""
    if not candidates:
        raise ValueError("continuation comparator requires at least one candidate")
    ranked = []
    for candidate_index, action in enumerate(candidates):
        child = reference.apply_action(state, action)
        child = normalize_state_bound(child, parent_bound=int(state.f_score))
        ranked.append(
            (
                base_continuation_key(state, action, child),
                candidate_index,
                action,
                child,
            )
        )

    counters = remaining_counters(state.remaining)
    count = counters.count
    top_entries = state.remaining[: min(WINDOW[0], count)]
    bottom_entries = state.remaining[max(0, count - WINDOW[1]) :]
    min_remaining_load = int(bottom_entries[-1][1]) if bottom_entries else 0
    rank_by_eid = {
        int(eid): rank for rank, (eid, _ntok) in enumerate(top_entries)
    }
    for offset, (eid, _ntok) in enumerate(reversed(bottom_entries)):
        rank_by_eid.setdefault(int(eid), count - 1 - offset)

    regime = classify_regime(state)
    best = ranked[0]
    overrides = 0
    for candidate in ranked[1:]:
        base_winner = min((best, candidate), key=lambda item: item[:2])
        alternate = base_winner
        use_alternate = False

        if regime.low_work_progress:
            progress = min(
                (best, candidate),
                key=lambda item: (_progress_key(item[2], item[3]), item[1]),
            )
            if lowering.child_key(base_winner[3]) != lowering.child_key(progress[3]):
                _base_max, _base_min, base_sum, base_s2pf = (
                    lowering.selected_action_features(base_winner[2])
                )
                _alt_max, _alt_min, alt_sum, _alt_s2pf = (
                    lowering.selected_action_features(progress[2])
                )
                base_early = min(
                    int(base_winner[3].c2.task_end),
                    int(base_winner[3].c3.task_end),
                )
                alt_early = min(
                    int(progress[3].c2.task_end), int(progress[3].c3.task_end)
                )
                if (
                    base_s2pf == 0
                    and alt_sum > base_sum
                    and alt_early - base_early <= TICK_CC
                ):
                    alternate = progress
                    use_alternate = True

        if regime.mid_plateau:
            progress = min(
                (best, candidate),
                key=lambda item: (_progress_key(item[2], item[3]), item[1]),
            )
            if lowering.child_key(base_winner[3]) != lowering.child_key(progress[3]):
                _base_max, _base_min, _base_sum, base_s2pf = (
                    lowering.selected_action_features(base_winner[2])
                )
                _progress_max, _progress_min, _progress_sum, progress_s2pf = (
                    lowering.selected_action_features(progress[2])
                )
                base_ranks = {
                    rank_by_eid[eid]
                    for eid in (base_winner[2].c2_eid, base_winner[2].c3_eid)
                    if eid >= 0
                }
                progress_ranks = {
                    rank_by_eid[eid]
                    for eid in (progress[2].c2_eid, progress[2].c3_eid)
                    if eid >= 0
                }
                base_early = min(
                    int(base_winner[3].c2.task_end),
                    int(base_winner[3].c3.task_end),
                )
                base_late = max(
                    int(base_winner[3].c2.task_end),
                    int(base_winner[3].c3.task_end),
                )
                progress_early = min(
                    int(progress[3].c2.task_end), int(progress[3].c3.task_end)
                )
                progress_late = max(
                    int(progress[3].c2.task_end), int(progress[3].c3.task_end)
                )
                if (
                    base_s2pf == 0
                    and progress_s2pf > 0
                    and min(progress_ranks) <= 3
                    and max(base_ranks) >= count - 2
                    and progress_early == base_early
                    and progress_late - base_late <= 6 * TICK_CC
                ):
                    alternate = progress
                    use_alternate = True

        if regime.short_tail_plateau or regime.large_slack_fill:
            fill = min(
                (best, candidate),
                key=lambda item: (_progress_key(item[2], item[3]), item[1]),
            )
            if lowering.child_key(base_winner[3]) != lowering.child_key(fill[3]):
                _base_max, _base_min, base_sum, base_s2pf = (
                    lowering.selected_action_features(base_winner[2])
                )
                _fill_max, _fill_min, fill_sum, fill_s2pf = (
                    lowering.selected_action_features(fill[2])
                )
                base_ranks = {
                    rank_by_eid[eid]
                    for eid in (base_winner[2].c2_eid, base_winner[2].c3_eid)
                    if eid >= 0
                }
                fill_ranks = {
                    rank_by_eid[eid]
                    for eid in (fill[2].c2_eid, fill[2].c3_eid)
                    if eid >= 0
                }
                base_early, base_late = sorted(
                    (
                        int(base_winner[3].c2.task_end),
                        int(base_winner[3].c3.task_end),
                    )
                )
                fill_early, fill_late = sorted(
                    (int(fill[3].c2.task_end), int(fill[3].c3.task_end))
                )
                base_components = lower_bound_components(base_winner[3])
                fill_components = lower_bound_components(fill[3])
                common = (
                    fill_s2pf > 0
                    and fill_sum > base_sum
                    and fill_early >= base_early
                    and fill[3].f_score == base_winner[3].f_score
                    and head5_hist4_lpt(fill[3])
                    == head5_hist4_lpt(base_winner[3])
                    and fill_components["dma_capacity_cc"]
                    == base_components["dma_capacity_cc"]
                )
                tail_override = (
                    regime.short_tail_plateau
                    and common
                    and base_s2pf > 0
                    and 0 in fill_ranks
                    and 0 not in base_ranks
                    and fill_late - base_late <= 3 * TICK_CC
                )
                slack_override = (
                    regime.large_slack_fill
                    and common
                    and base_s2pf == 0
                    and min(fill_ranks) <= 1
                    and max(base_ranks) >= count - 2
                    and fill_late == base_late
                )
                if tail_override or slack_override:
                    alternate = fill
                    use_alternate = True

        if regime.sparse_hot_sync:
            hot = min(
                (best, candidate),
                key=lambda item: (_hotspot_key(item[2], item[3]), item[1]),
            )
            if lowering.child_key(base_winner[3]) != lowering.child_key(hot[3]):
                base_max, _base_min, _base_sum, _base_s2pf = (
                    lowering.selected_action_features(base_winner[2])
                )
                hot_max, _hot_min, _hot_sum, _hot_s2pf = (
                    lowering.selected_action_features(hot[2])
                )
                base_ranks = {
                    rank_by_eid[eid]
                    for eid in (base_winner[2].c2_eid, base_winner[2].c3_eid)
                    if eid >= 0
                }
                hot_ranks = {
                    rank_by_eid[eid]
                    for eid in (hot[2].c2_eid, hot[2].c3_eid)
                    if eid >= 0
                }
                base_components = lower_bound_components(base_winner[3])
                hot_components = lower_bound_components(hot[3])
                dma_large = (
                    base_components["dma_capacity_cc"] > HOT_DMA_HIGH_TICKS * TICK_CC
                    or (
                        min_remaining_load >= 2
                        and base_components["dma_capacity_cc"]
                        > HOT_DMA_MIN2_TICKS * TICK_CC
                    )
                )
                if (
                    0 in hot_ranks
                    and 0 not in base_ranks
                    and hot_max > base_max
                    and hot[3].f_score == base_winner[3].f_score
                    and head5_hist4_lpt(hot[3])
                    == head5_hist4_lpt(base_winner[3])
                    and hot_components["compute_cc"]
                    - base_components["compute_cc"]
                    <= 3 * TICK_CC
                    and hot_components["dma_capacity_cc"]
                    - base_components["dma_capacity_cc"]
                    <= 6 * TICK_CC
                    and dma_large
                ):
                    alternate = hot
                    use_alternate = True

        best = alternate if use_alternate else base_winner
        overrides += int(use_alternate)

    score, candidate_id, action, child = best
    return score, candidate_id, action, child, {
        "comparator": SCORER_ID,
        "pairwise_overrides": overrides,
        "regime": {
            "low_work_progress": regime.low_work_progress,
            "sparse_hot_sync": regime.sparse_hot_sync,
            "mid_plateau": regime.mid_plateau,
            "short_tail_plateau": regime.short_tail_plateau,
            "large_slack_fill": regime.large_slack_fill,
        },
    }
