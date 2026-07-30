#!/usr/bin/env python3
"""Frozen mirror for the previous fixed-BOTH integer-tick scheduler RTL.

This model keeps the frozen HW-v2 candidate bank and continuation formula, but
uses the physical prefetch contract implemented before the adaptive policy:

* S2PF always occupies both DMA lanes for one tick.
* S4PF always occupies both DMA lanes for two ticks.
* A busy-side S4PF release is retained as the final one-idle release point.
* Candidate scores are rounded to the integer tick domain before comparison.

The original :mod:`scheduler_hw_fixed_policy` remains unchanged and is the
algorithmic HW-v2 baseline used by the previous 30K result.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from threading import RLock
from typing import Iterator

import scheduler_hw_fixed_policy as base


TICK_CC = 11_264
S2PF_DMA_CC = 128
S2PF_DURATION_CC = 1 * TICK_CC
S4PF_DMA_CC = 128
S4PF_DURATION_CC = 2 * TICK_CC

PolicyState = base.PolicyState
Transition = base.Transition
initial_state = base.initial_state
terminal_cost = base.terminal_cost

_MODEL_LOCK = RLock()
_BASE_CM = base.cm
_SINGLE_SHAPES = tuple(base.hw.N1_PRUNED_SOLO_SHAPES)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


class _RtlPrefetchCostModel:
    """Delegate timing arithmetic to the old model and replace DMA semantics."""

    def __getattr__(self, name: str):
        return getattr(_BASE_CM, name)

    def _cc_snap_segs(self, snap):
        segments = []
        if snap.cur_eid >= 0 and snap.bw_s1 > 0:
            segments.append((snap.task_start, snap.dma1_end, snap.bw_s1))
        if snap.s2pf_start >= 0 and snap.s2pf_bw > 0:
            segments.append((snap.s2pf_start, snap.s2pf_end, S2PF_DMA_CC))
        if snap.cur_eid >= 0 and snap.bw_s3 > 0 and snap.dma3_end > snap.s2_end:
            segments.append((snap.s2_end, snap.dma3_end, snap.bw_s3))
        if snap.cur_eid >= 0 and snap.s4pf_valid:
            segments.append(
                (snap.dma3_end, snap.dma3_end + S4PF_DURATION_CC, S4PF_DMA_CC)
            )
        return segments

    def _cc_bw_ok(self, a, b) -> bool:
        # In the reachable two-cluster policy, scalar 64/128 occupancy is
        # equivalent to the RTL lane masks: C2 single=iDMA, C3 single=xDMA,
        # and every 128-B/cc transfer uses BOTH.
        for alo, ahi, abw in self._cc_snap_segs(a):
            for blo, bhi, bbw in self._cc_snap_segs(b):
                if max(alo, blo) < min(ahi, bhi) and abw + bbw > 128:
                    return False
        return True

    def _cc_apply_s2pf(self, snap, _shape_s3: int, start: int):
        if snap.bw_s3 == 0:
            return snap
        end = start + S2PF_DURATION_CC
        if start < snap.dma1_end or end > snap.s2_end:
            return snap
        updated = replace(snap)
        updated.s2pf_start = start
        updated.s2pf_end = end
        updated.s2pf_bw = S2PF_DMA_CC
        updated.dma3_end = updated.s2_end
        updated.s3_end = updated.s2_end
        updated.s4_start = updated.s2_end
        updated.bw_s3 = 0
        updated.task_end = updated.s2_end + self._cc_best_s4(updated.ntok)
        return updated

    def _cc_s4pf_local_ok(self, snap) -> bool:
        return (
            snap.cur_eid >= 0
            and snap.pf_eid == -1
            and snap.dma3_end + S4PF_DURATION_CC <= snap.task_end
        )

    def _cc_apply_s4pf_ghost(self, snap):
        if not self._cc_s4pf_local_ok(snap):
            return snap
        updated = replace(snap)
        updated.pf_eid = self.C_PF_EID_GHOST
        # RTL exposes the prefetched cache entry to the next round only after
        # the producer task completes, even though the DMA interval ends earlier.
        updated.pf_end = updated.task_end
        updated.pf_full = 0
        updated.s4pf_valid = 1
        updated.s4pf_start = updated.dma3_end
        return updated

    def _cc_s4pf_ok_with_peer(self, snap, peer) -> bool:
        if not self._cc_s4pf_local_ok(snap):
            return False
        return self._cc_bw_ok(self._cc_apply_s4pf_ghost(snap), peer)

    def _cc_busy_time_points(self, busy, idle_time: int):
        # Exact translation of moe_scheduler_core.make_early_start_ctx().
        points = [idle_time]
        s1_release = busy.dma1_end
        stage3_release = busy.s2pf_end if busy.s2pf_start >= 0 else busy.dma3_end
        s4pf_release = busy.dma3_end + S4PF_DURATION_CC

        s1_valid = (
            busy.cur_eid >= 0 and busy.bw_s1 > 0 and s1_release > idle_time
        )
        stage3_valid = (
            busy.cur_eid >= 0
            and (busy.s2pf_start >= 0 or busy.bw_s3 > 0)
            and stage3_release > idle_time
        )
        s4pf_valid = (
            busy.cur_eid >= 0 and busy.s4pf_valid and s4pf_release > idle_time
        )

        if s1_valid:
            points.append(s1_release)
            if s4pf_valid and s4pf_release != s1_release:
                points.append(s4pf_release)
            elif stage3_valid and stage3_release != s1_release:
                points.append(stage3_release)
        elif stage3_valid:
            points.append(stage3_release)
            if s4pf_valid and s4pf_release != stage3_release:
                points.append(s4pf_release)
        elif s4pf_valid:
            points.append(s4pf_release)
        return points


_RTL_CM = _RtlPrefetchCostModel()


@contextmanager
def _use_cost_model(cost_model) -> Iterator[None]:
    """Install a backend only for one atomic policy operation."""
    with _MODEL_LOCK:
        saved_base_cm = base.cm
        saved_hw_cm = base.hw.cm
        base.cm = cost_model
        base.hw.cm = cost_model
        try:
            yield
        finally:
            base.hw.cm = saved_hw_cm
            base.cm = saved_base_cm


def token_from_tag(state: PolicyState, tag: str) -> tuple[int, int]:
    """Map a Python transition tag to the RTL ``{mode, candidate_id}`` token."""
    if len(state.remaining) == 1:
        if tag.startswith("last_solo_c"):
            cluster = int(tag[len("last_solo_c")])
            shape_code = tag.rsplit("_", 1)[1]
            shape = (int(shape_code[0]), int(shape_code[1]))
            offset = 0 if cluster == 2 else len(_SINGLE_SHAPES)
            return 0, offset + _SINGLE_SHAPES.index(shape)
        if tag.startswith("last_split_"):
            return 0, 10
        if tag.startswith("last_release_c"):
            return 0, 11 + int(tag.rsplit("_", 1)[1])
        raise RuntimeError(f"unmapped LAST_EXPERT tag {tag}")

    if state.c2.task_end == state.c3.task_end:
        if tag == "pair_0_1":
            return 1, 0
        if tag == "pair_1_2":
            return 1, 1
        if tag == "pair_2_3":
            return 1, 2
        if tag.startswith("split_0_"):
            cut = int(tag.rsplit("_", 1)[1])
            half = (state.remaining[0][1] + 1) // 2
            return 1, 3 if cut == half else 4
        raise RuntimeError(f"unmapped BOTH_IDLE tag {tag}")

    if tag.startswith("one_idle_adaptive_c"):
        return 2, 3 + int(tag.rsplit("p", 1)[1])
    if tag.startswith("one_idle_c"):
        return 2, int(tag.rsplit("p", 1)[1])
    raise RuntimeError(f"unmapped ONE_IDLE tag {tag}")


def _choose_transition(state: PolicyState, cost_model) -> Transition:
    with _use_cost_model(cost_model):
        transitions = base.generate_one_idle_shape_successors(
            state, policy="balanced", top_policy="pruned", n1_policy="pruned"
        )

        def key(transition: Transition) -> tuple[int, int, int, int]:
            child = transition.state
            current_makespan_cc = max(child.c2.task_end, child.c3.task_end)
            if len(state.remaining) == 1 or state.c2.task_end != state.c3.task_end:
                cost_ticks = _ceil_div(current_makespan_cc, TICK_CC)
            else:
                continuation_cc = base.hw_v2_continuation(
                    child.c2, child.c3, child.remaining, policy="balanced"
                )
                cost_ticks = _ceil_div(continuation_cc, TICK_CC)
            _, candidate_id = token_from_tag(state, transition.tag)
            return (
                cost_ticks,
                len(child.remaining),
                _ceil_div(current_makespan_cc, TICK_CC),
                candidate_id,
            )

        return min(transitions, key=key)


def choose_transition(state: PolicyState) -> Transition:
    """Select one transition with the current RTL prefetch/tick contract."""
    return _choose_transition(state, _RTL_CM)


def choose_original_prefetch_tick_transition(state: PolicyState) -> Transition:
    """RTL-tick selector with the old inherited-prefetch resource contract."""
    return _choose_transition(state, _BASE_CM)


@dataclass(frozen=True)
class RtlScheduleResult:
    makespan_cc: int
    winner_trace: tuple[tuple[int, int], ...]

    @property
    def makespan_ticks(self) -> int:
        if self.makespan_cc % TICK_CC:
            raise ValueError("terminal makespan is not an integer tick")
        return self.makespan_cc // TICK_CC


def _schedule_result(
    token_dist: dict[int, int],
    initial_cache_c2: int,
    initial_cache_c3: int,
    *,
    cost_model,
) -> RtlScheduleResult:
    state = initial_state(token_dist, initial_cache_c2, initial_cache_c3)
    trace: list[tuple[int, int]] = []
    while state.remaining:
        chosen = _choose_transition(state, cost_model)
        trace.append(token_from_tag(state, chosen.tag))
        state = chosen.state
    return RtlScheduleResult(terminal_cost(state), tuple(trace))


def rtl_prefetch_both_schedule_result(
    token_dist: dict[int, int],
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
) -> RtlScheduleResult:
    return _schedule_result(
        token_dist,
        initial_cache_c2,
        initial_cache_c3,
        cost_model=_RTL_CM,
    )


def rtl_prefetch_both_schedule(
    token_dist: dict[int, int],
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
) -> int:
    return rtl_prefetch_both_schedule_result(
        token_dist, initial_cache_c2, initial_cache_c3
    ).makespan_cc


def original_prefetch_tick_schedule(
    token_dist: dict[int, int],
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
) -> int:
    return _schedule_result(
        token_dist,
        initial_cache_c2,
        initial_cache_c3,
        cost_model=_BASE_CM,
    ).makespan_cc
