#!/usr/bin/env python3
"""Integer-tick RTL mirror for single-S2PF and selectable S4PF DMA modes.

S2 prefetch is bound to the owning cluster's single DMA lane for two ticks.
S4 prefetch can use that single lane for four ticks or both lanes for two ticks.
The ``s4_policy`` argument controls the deterministic trial order.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from threading import RLock
from typing import Iterator, Literal

import scheduler_hw_fixed_policy as base
from scheduler_rtl_prefetch_both_policy import (
    RtlScheduleResult,
    TICK_CC,
    token_from_tag as _baseline_token_from_tag,
)


SINGLE_DMA_CC = 64
BOTH_DMA_CC = 128
S2PF_DURATION_CC = 2 * TICK_CC
S4PF_SINGLE_DURATION_CC = 4 * TICK_CC
S4PF_BOTH_DURATION_CC = 2 * TICK_CC

S4Policy = Literal[
    "single_only",
    "both_only",
    "single_first",
    "both_first",
    "window_select",
]
DEFAULT_S4_POLICY: S4Policy = "single_first"
ScorePolicy = Literal[
    "legacy",
    "continuation_all",
    "critical_1_2",
    "critical_5_8",
    "critical_3_4",
    "critical_7_8",
    "protected_1",
    "protected_2",
    "protected_4",
    "protected_8",
    "protected_slack_1",
    "protected_slack_2",
    "protected_slack_4",
    "protected_one_idle_slack_1",
    "protected_one_idle_slack_2",
    "protected_one_idle_slack_4",
    "protected_sync_1",
    "protected_sync_2",
    "protected_sync_4",
    "protected_state_t5_2",
    "protected_state_t5_4",
    "protected_state_t5_6",
    "protected_state_t5_8",
    "protected_top6_1",
    "protected_top6_2",
    "protected_top6_4",
    "protected_top6_slack_1",
    "protected_top6_slack_2",
    "protected_top6_slack_4",
    "protected_top6_one_idle_slack_1",
    "protected_top6_sync_1",
    "protected_consensus_1",
    "protected_consensus_slack_1",
    "protected_consensus_slack_2",
    "protected_consensus_slack_4",
    "protected_consensus_one_idle_slack_1",
    "protected_headcritical_slack_1",
    "protected_headcritical_slack_plus1_1",
    "protected_headcritical_slack_plus2_1",
    "protected_headcritical_one_idle_slack_1",
    "protected_tail_b0_slack_1",
    "protected_slack_once_1",
    "protected_consensus_slack_once_1",
    "protected_headcritical_slack_once_1",
]

_CRITICAL_RATIOS: dict[str, tuple[int, int]] = {
    "critical_1_2": (1, 2),
    "critical_5_8": (5, 8),
    "critical_3_4": (3, 4),
    "critical_7_8": (7, 8),
}

# (minimum continuation advantage in integer ticks, require ONE_IDLE task to
# fit entirely inside existing busy-side slack, allow SYNC, allow ONE_IDLE,
# optional maximum current T5 load for ONE_IDLE acceptance, use top6 instead
# of top4 in the continuation estimate).
# These are fixed comparator configurations for ablation; they do not unfold a
# child round or inspect an offline label.
_PROTECTED_POLICIES: dict[
    str, tuple[int, bool, bool, bool, int | None, bool]
] = {
    "protected_1": (1, False, True, True, None, False),
    "protected_2": (2, False, True, True, None, False),
    "protected_4": (4, False, True, True, None, False),
    "protected_8": (8, False, True, True, None, False),
    "protected_slack_1": (1, True, True, True, None, False),
    "protected_slack_2": (2, True, True, True, None, False),
    "protected_slack_4": (4, True, True, True, None, False),
    "protected_one_idle_slack_1": (1, True, False, True, None, False),
    "protected_one_idle_slack_2": (2, True, False, True, None, False),
    "protected_one_idle_slack_4": (4, True, False, True, None, False),
    "protected_sync_1": (1, False, True, False, None, False),
    "protected_sync_2": (2, False, True, False, None, False),
    "protected_sync_4": (4, False, True, False, None, False),
    "protected_state_t5_2": (1, True, True, True, 2, False),
    "protected_state_t5_4": (1, True, True, True, 4, False),
    "protected_state_t5_6": (1, True, True, True, 6, False),
    "protected_state_t5_8": (1, True, True, True, 8, False),
    "protected_top6_1": (1, False, True, True, None, True),
    "protected_top6_2": (2, False, True, True, None, True),
    "protected_top6_4": (4, False, True, True, None, True),
    "protected_top6_slack_1": (1, True, True, True, None, True),
    "protected_top6_slack_2": (2, True, True, True, None, True),
    "protected_top6_slack_4": (4, True, True, True, None, True),
    "protected_top6_one_idle_slack_1": (1, True, False, True, None, True),
    "protected_top6_sync_1": (1, False, True, False, None, True),
    "protected_consensus_1": (1, False, True, True, None, False),
    "protected_consensus_slack_1": (1, True, True, True, None, False),
    "protected_consensus_slack_2": (2, True, True, True, None, False),
    "protected_consensus_slack_4": (4, True, True, True, None, False),
    "protected_consensus_one_idle_slack_1": (1, True, False, True, None, False),
    "protected_headcritical_slack_1": (1, True, True, True, None, False),
    "protected_headcritical_slack_plus1_1": (1, True, True, True, None, False),
    "protected_headcritical_slack_plus2_1": (1, True, True, True, None, False),
    "protected_headcritical_one_idle_slack_1": (1, True, False, True, None, False),
    "protected_tail_b0_slack_1": (1, True, True, True, None, False),
}

_CONSENSUS_POLICIES = {
    "protected_consensus_1",
    "protected_consensus_slack_1",
    "protected_consensus_slack_2",
    "protected_consensus_slack_4",
    "protected_consensus_one_idle_slack_1",
}

_HEAD_CRITICAL_ALLOWANCE_TICKS = {
    "protected_headcritical_slack_1": 0,
    "protected_headcritical_slack_plus1_1": 1,
    "protected_headcritical_slack_plus2_1": 2,
    "protected_headcritical_one_idle_slack_1": 0,
    "protected_tail_b0_slack_1": 0,
}

_BOTTOM_B0_FIRST_RELEASE_ONLY_POLICIES = {
    "protected_tail_b0_slack_1",
}

_ONCE_POLICY_BASE: dict[str, ScorePolicy] = {
    "protected_slack_once_1": "protected_slack_1",
    "protected_consensus_slack_once_1": "protected_consensus_slack_1",
    "protected_headcritical_slack_once_1": "protected_headcritical_slack_1",
}

PolicyState = base.PolicyState
Transition = base.Transition
terminal_cost = base.terminal_cost

_MODEL_LOCK = RLock()
_BASE_CM = base.cm
_SINGLE_SHAPES = tuple(base.hw.N1_PRUNED_SOLO_SHAPES)


def token_from_tag(state: PolicyState, tag: str) -> tuple[int, int]:
    """Map baseline and additive T6+B2 tags to compact sequential IDs."""
    if tag == "t6b2_pair_t0_b0":
        return 1, 5
    if tag.startswith("t6b2_one_idle_b"):
        bottom_rank = int(tag[len("t6b2_one_idle_b")])
        point_index = int(tag.rsplit("p", 1)[1])
        return 2, 6 + 3 * bottom_rank + point_index
    fixed_pair_ids = {
        "fixed14_pair_t0_b0_ab_bb_c2hot": 6,
        "fixed14_pair_t0_b0_ab_bb_c3hot": 7,
        "fixed14_pair_t0_t4_ab_bb_c2hot": 8,
        "fixed14_pair_t0_t4_ab_bb_c3hot": 9,
        "fixed14_pair_t0_t1_none": 10,
        "fixed14_pair_t0_t1_c2pf": 11,
        "fixed14_pair_t0_t1_c3pf": 12,
        "fixed14_pair_t1_t2_bothpf": 13,
        "fixed14_pair_t2_t3_bothpf": 14,
    }
    if tag in fixed_pair_ids:
        return 1, fixed_pair_ids[tag]
    if tag.startswith("fixed14_single_t"):
        rank = int(tag[len("fixed14_single_t")])
        point_index = int(tag.rsplit("p", 1)[1])
        rank_slot = 0 if rank == 0 else 1
        return 2, 12 + 3 * rank_slot + point_index
    if tag.startswith("fixed14_single_b"):
        bottom_rank = int(tag[len("fixed14_single_b")])
        point_index = int(tag.rsplit("p", 1)[1])
        return 2, 18 + 3 * bottom_rank + point_index
    return _baseline_token_from_tag(state, tag)


@dataclass
class _AdaptiveSnap(_BASE_CM.CSnap):
    s4pf_dma_cc: int = 0


def _as_adaptive(snap) -> _AdaptiveSnap:
    if isinstance(snap, _AdaptiveSnap):
        return snap
    values = {field.name: getattr(snap, field.name) for field in fields(_BASE_CM.CSnap)}
    return _AdaptiveSnap(**values)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


class _AdaptivePrefetchCostModel:
    def __init__(self, s4_policy: S4Policy):
        self.s4_policy = s4_policy
        self._pending_s4pf_dma_cc = 0

    def __getattr__(self, name: str):
        return getattr(_BASE_CM, name)

    def _cc_initial(self, cache_eid: int):
        return _as_adaptive(_BASE_CM._cc_initial(cache_eid))

    def _cc_mk_snap(
        self,
        start: int,
        s1: int,
        s3: int,
        ntok: int,
        eid: int,
        s1c: bool,
        s3c: bool,
    ):
        return _as_adaptive(
            _BASE_CM._cc_mk_snap(start, s1, s3, ntok, eid, s1c, s3c)
        )

    @staticmethod
    def _s4pf_duration(dma_cc: int) -> int:
        if dma_cc == SINGLE_DMA_CC:
            return S4PF_SINGLE_DURATION_CC
        if dma_cc == BOTH_DMA_CC:
            return S4PF_BOTH_DURATION_CC
        raise ValueError(f"invalid S4PF DMA occupancy {dma_cc}")

    def _cc_snap_segs(self, snap):
        segments = []
        if snap.cur_eid >= 0 and snap.bw_s1 > 0:
            segments.append((snap.task_start, snap.dma1_end, snap.bw_s1))
        if snap.s2pf_start >= 0 and snap.s2pf_bw > 0:
            segments.append((snap.s2pf_start, snap.s2pf_end, SINGLE_DMA_CC))
        if snap.cur_eid >= 0 and snap.bw_s3 > 0 and snap.dma3_end > snap.s2_end:
            segments.append((snap.s2_end, snap.dma3_end, snap.bw_s3))
        if snap.cur_eid >= 0 and snap.s4pf_valid:
            duration = self._s4pf_duration(snap.s4pf_dma_cc)
            segments.append(
                (snap.dma3_end, snap.dma3_end + duration, snap.s4pf_dma_cc)
            )
        return segments

    def _cc_bw_ok(self, a, b) -> bool:
        # Reachable single-lane transfers are fixed to distinct lanes by cluster.
        for alo, ahi, abw in self._cc_snap_segs(a):
            for blo, bhi, bbw in self._cc_snap_segs(b):
                if max(alo, blo) < min(ahi, bhi) and abw + bbw > BOTH_DMA_CC:
                    return False
        return True

    def _cc_apply_s2pf(self, snap, _shape_s3: int, start: int):
        if snap.bw_s3 == 0:
            return snap
        end = start + S2PF_DURATION_CC
        if start < snap.dma1_end or end > snap.s2_end:
            return snap
        updated = replace(_as_adaptive(snap))
        updated.s2pf_start = start
        updated.s2pf_end = end
        updated.s2pf_bw = SINGLE_DMA_CC
        updated.dma3_end = updated.s2_end
        updated.s3_end = updated.s2_end
        updated.s4_start = updated.s2_end
        updated.bw_s3 = 0
        updated.task_end = updated.s2_end + self._cc_best_s4(updated.ntok)
        return updated

    def _s4pf_local_ok(self, snap, dma_cc: int) -> bool:
        return (
            snap.cur_eid >= 0
            and snap.pf_eid == -1
            and snap.dma3_end + self._s4pf_duration(dma_cc) <= snap.task_end
        )

    def _cc_s4pf_local_ok(self, snap) -> bool:
        return any(self._s4pf_local_ok(snap, mode) for mode in self._mode_order(snap))

    def _mode_order(self, snap=None) -> tuple[int, ...]:
        if self.s4_policy == "window_select":
            if snap is None:
                return (SINGLE_DMA_CC, BOTH_DMA_CC)
            return (
                (SINGLE_DMA_CC,)
                if self._s4pf_local_ok(snap, SINGLE_DMA_CC)
                else (BOTH_DMA_CC,)
            )
        return {
            "single_only": (SINGLE_DMA_CC,),
            "both_only": (BOTH_DMA_CC,),
            "single_first": (SINGLE_DMA_CC, BOTH_DMA_CC),
            "both_first": (BOTH_DMA_CC, SINGLE_DMA_CC),
        }[self.s4_policy]

    def _apply_s4pf_mode(self, snap, dma_cc: int):
        updated = replace(_as_adaptive(snap))
        updated.pf_eid = self.C_PF_EID_GHOST
        updated.pf_end = updated.task_end
        updated.pf_full = 0
        updated.s4pf_valid = 1
        updated.s4pf_start = updated.dma3_end
        updated.s4pf_dma_cc = dma_cc
        return updated

    def _cc_s4pf_ok_with_peer(self, snap, peer) -> bool:
        self._pending_s4pf_dma_cc = 0
        for dma_cc in self._mode_order(snap):
            if not self._s4pf_local_ok(snap, dma_cc):
                continue
            trial = self._apply_s4pf_mode(snap, dma_cc)
            if self._cc_bw_ok(trial, peer):
                self._pending_s4pf_dma_cc = dma_cc
                return True
        return False

    def _cc_apply_s4pf_ghost(self, snap):
        dma_cc = self._pending_s4pf_dma_cc
        self._pending_s4pf_dma_cc = 0
        if dma_cc == 0:
            return snap
        return self._apply_s4pf_mode(snap, dma_cc)

    def _cc_busy_time_points(self, busy, idle_time: int):
        points = [idle_time]
        s1_release = busy.dma1_end
        stage3_release = busy.s2pf_end if busy.s2pf_start >= 0 else busy.dma3_end
        s4pf_release = (
            busy.dma3_end + self._s4pf_duration(busy.s4pf_dma_cc)
            if busy.s4pf_valid
            else busy.dma3_end
        )

        s1_valid = busy.cur_eid >= 0 and busy.bw_s1 > 0 and s1_release > idle_time
        stage3_valid = (
            busy.cur_eid >= 0
            and (busy.s2pf_start >= 0 or busy.bw_s3 > 0)
            and stage3_release > idle_time
        )
        s4pf_valid = busy.cur_eid >= 0 and busy.s4pf_valid and s4pf_release > idle_time

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


_COST_MODELS = {
    policy: _AdaptivePrefetchCostModel(policy)
    for policy in (
        "single_only",
        "both_only",
        "single_first",
        "both_first",
        "window_select",
    )
}


@contextmanager
def _use_cost_model(cost_model) -> Iterator[None]:
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


def _choose_transition(
    state: PolicyState,
    cost_model,
    *,
    candidate_policy: str = "baseline",
    score_policy: ScorePolicy = "legacy",
) -> Transition:
    with _use_cost_model(cost_model):
        protected = _PROTECTED_POLICIES.get(score_policy)
        if protected is not None and candidate_policy not in {
            base.TOP6_BOTTOM2_CANDIDATE_POLICY,
            base.TOP6_BOTTOM2_PROTECTED_CANDIDATE_POLICY,
        }:
            raise ValueError("protected score requires the top6+bottom2 candidate bank")
        generator = (
            base.generate_one_idle_shape_successors
            if candidate_policy == "baseline"
            else base.generate_top6_bottom2_successors
            if candidate_policy == base.TOP6_BOTTOM2_CANDIDATE_POLICY
            else base.generate_top6_bottom2_protected_successors
            if candidate_policy == base.TOP6_BOTTOM2_PROTECTED_CANDIDATE_POLICY
            else base.generate_top6_bottom2_fixed14_union_successors
            if candidate_policy == base.TOP6_BOTTOM2_FIXED14_UNION_POLICY
            else None
        )
        if generator is None:
            raise ValueError(f"unknown candidate policy {candidate_policy!r}")
        transitions = generator(
            state, policy="balanced", top_policy="pruned", n1_policy="pruned"
        )

        def key(
            transition: Transition,
            active_score_policy: str = score_policy,
        ) -> tuple[int, int, int, int, int, int]:
            child = transition.state
            current_makespan_cc = max(child.c2.task_end, child.c3.task_end)
            critical_flag = 0
            critical_head_work_cc = 0
            if len(state.remaining) == 1:
                cost_ticks = _ceil_div(current_makespan_cc, TICK_CC)
            elif state.c2.task_end == state.c3.task_end or active_score_policy != "legacy":
                continuation = (
                    base.hw_v2_continuation_top6
                    if active_score_policy == "continuation_top6"
                    else base.hw_v2_continuation
                )
                continuation_cc = continuation(
                    child.c2, child.c3, child.remaining, policy="balanced"
                )
                cost_ticks = _ceil_div(continuation_cc, TICK_CC)
                # LPT can give the same lower bound to (a) dispatching the
                # largest remaining expert now and (b) repeatedly consuming
                # cold experts first.  On ONE_IDLE states, mark the latter as critical when the
                # earliest possible completion of the still-pending head
                # nearly consumes the full bound.  When two equal dominant
                # heads remain, their combined work is the secondary key, so
                # consuming one of them wins over consuming another cold
                # expert.  The
                # four ratios use shifts/adds and only break equal score ties;
                # they do not generate or simulate another candidate.
                ratio = _CRITICAL_RATIOS.get(active_score_policy)
                if (
                    ratio is not None
                    and state.c2.task_end != state.c3.task_end
                    and child.remaining
                ):
                    numerator, denominator = ratio
                    head_finish_cc = min(
                        child.c2.task_end, child.c3.task_end
                    ) + int(base.cm._cc_best_task(int(child.remaining[0][1])))
                    critical_flag = int(
                        denominator * head_finish_cc
                        >= numerator * continuation_cc
                    )
                    if critical_flag:
                        critical_head_work_cc = sum(
                            int(base.cm._cc_best_task(int(ntok)))
                            for _eid, ntok in child.remaining[:2]
                        )
            elif active_score_policy == "legacy":
                cost_ticks = _ceil_div(current_makespan_cc, TICK_CC)
            else:
                raise ValueError(f"unknown score policy {active_score_policy!r}")
            _, candidate_id = token_from_tag(state, transition.tag)
            return (
                cost_ticks,
                critical_flag,
                critical_head_work_cc,
                len(child.remaining),
                _ceil_div(current_makespan_cc, TICK_CC),
                candidate_id,
            )

        if protected is None:
            return min(transitions, key=key)

        # First reproduce the old adaptive winner exactly.  Added bottom
        # candidates may replace it only through the fixed acceptance guard;
        # otherwise the old trajectory is preserved rather than being
        # rescored by continuation_all.
        baseline = base.generate_one_idle_shape_successors(
            state, policy="balanced", top_policy="pruned", n1_policy="pruned"
        )
        baseline_winner = min(
            baseline, key=lambda transition: key(transition, "legacy")
        )
        baseline_states = {base.state_key(transition.state) for transition in baseline}
        added = [
            transition
            for transition in transitions
            if base.state_key(transition.state) not in baseline_states
        ]
        (
            min_advantage_ticks,
            require_slack,
            allow_sync,
            allow_one_idle,
            max_one_idle_t5,
            use_top6_continuation,
        ) = protected
        is_sync = state.c2.task_end == state.c3.task_end
        if (is_sync and not allow_sync) or (not is_sync and not allow_one_idle):
            return baseline_winner
        if not is_sync and max_one_idle_t5 is not None:
            # With five or fewer jobs B0/B1 alias the hot window and there is
            # no hidden middle to justify cold-fill reordering.  Treat that
            # tail as old-policy-only instead of making the T5 bound vacuous.
            if len(state.remaining) <= 5:
                return baseline_winner
            current_t5 = int(state.remaining[5][1])
            if current_t5 > max_one_idle_t5:
                return baseline_winner
        if require_slack and not is_sync:
            busy_end = max(int(state.c2.task_end), int(state.c3.task_end))
            added = [
                transition
                for transition in added
                if max(
                    int(transition.state.c2.task_end),
                    int(transition.state.c3.task_end),
                )
                <= busy_end
            ]
        if not added:
            return baseline_winner
        if (
            score_policy in _BOTTOM_B0_FIRST_RELEASE_ONLY_POLICIES
            and not is_sync
        ):
            # Fixed valid bits expose only the true cold end at the earliest
            # legal release point.  B1 and later release points can improve
            # the approximate continuation score while worsening the exact
            # terminal ordering once the visible tail aliases the hot view.
            added = [
                transition
                for transition in added
                if transition.tag.startswith("t6b2_one_idle_b0_")
                and transition.tag.endswith("_p0")
            ]
            if not added:
                return baseline_winner
        head_allowance_ticks = _HEAD_CRITICAL_ALLOWANCE_TICKS.get(score_policy)
        if head_allowance_ticks is not None and not is_sync:
            head_filtered = []
            for transition in added:
                child = transition.state
                if not child.remaining:
                    continue
                head_finish_cc = min(
                    int(child.c2.task_end), int(child.c3.task_end)
                ) + int(base.cm._cc_best_task(int(child.remaining[0][1])))
                continuation_cc = base.hw_v2_continuation(
                    child.c2, child.c3, child.remaining, policy="balanced"
                )
                if (
                    head_finish_cc + head_allowance_ticks * TICK_CC
                    >= continuation_cc
                ):
                    head_filtered.append(transition)
            added = head_filtered
            if not added:
                return baseline_winner
        if score_policy in _CONSENSUS_POLICIES and not is_sync:
            baseline_top4_ticks = key(baseline_winner, "continuation_all")[0]
            baseline_top6_ticks = key(baseline_winner, "continuation_top6")[0]
            acceptable = [
                transition
                for transition in added
                if (
                    baseline_top4_ticks
                    - key(transition, "continuation_all")[0]
                    >= min_advantage_ticks
                    and baseline_top6_ticks
                    - key(transition, "continuation_top6")[0]
                    >= min_advantage_ticks
                )
            ]
            if not acceptable:
                return baseline_winner
            return min(
                acceptable,
                key=lambda transition: (
                    key(transition, "continuation_all"),
                    key(transition, "continuation_top6"),
                ),
            )
        protected_continuation = (
            "continuation_top6" if use_top6_continuation else "continuation_all"
        )
        added_winner = min(
            added, key=lambda transition: key(transition, protected_continuation)
        )
        baseline_continuation_ticks = key(
            baseline_winner, protected_continuation
        )[0]
        added_continuation_ticks = key(
            added_winner, protected_continuation
        )[0]
        if (
            baseline_continuation_ticks - added_continuation_ticks
            >= min_advantage_ticks
        ):
            return added_winner
        return baseline_winner


def initial_state(
    token_dist: dict[int, int],
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
) -> PolicyState:
    cost_model = _COST_MODELS[DEFAULT_S4_POLICY]
    with _use_cost_model(cost_model):
        return base.initial_state(token_dist, initial_cache_c2, initial_cache_c3)


def choose_transition(state: PolicyState) -> Transition:
    return _choose_transition(state, _COST_MODELS[DEFAULT_S4_POLICY])


def adaptive_prefetch_schedule_result(
    token_dist: dict[int, int],
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
    *,
    s4_policy: S4Policy = DEFAULT_S4_POLICY,
    candidate_policy: str = "baseline",
    score_policy: ScorePolicy = "legacy",
) -> RtlScheduleResult:
    cost_model = _COST_MODELS[s4_policy]
    with _use_cost_model(cost_model):
        state = base.initial_state(token_dist, initial_cache_c2, initial_cache_c3)
    trace: list[tuple[int, int]] = []
    one_idle_bottom_used = False
    while state.remaining:
        active_score_policy = _ONCE_POLICY_BASE.get(score_policy, score_policy)
        if score_policy in _ONCE_POLICY_BASE and one_idle_bottom_used:
            # Preserve globally safe SYNC hot+cold use after the single
            # ONE_IDLE cold-fill opportunity has been consumed.
            active_score_policy = "protected_sync_1"
        chosen = _choose_transition(
            state,
            cost_model,
            candidate_policy=candidate_policy,
            score_policy=active_score_policy,
        )
        trace.append(token_from_tag(state, chosen.tag))
        if chosen.tag.startswith("t6b2_one_idle_b"):
            one_idle_bottom_used = True
        state = chosen.state
    return RtlScheduleResult(terminal_cost(state), tuple(trace))


def adaptive_prefetch_schedule(
    token_dist: dict[int, int],
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
    *,
    s4_policy: S4Policy = DEFAULT_S4_POLICY,
    candidate_policy: str = "baseline",
    score_policy: ScorePolicy = "legacy",
) -> int:
    return adaptive_prefetch_schedule_result(
        token_dist,
        initial_cache_c2,
        initial_cache_c3,
        s4_policy=s4_policy,
        candidate_policy=candidate_policy,
        score_policy=score_policy,
    ).makespan_cc
