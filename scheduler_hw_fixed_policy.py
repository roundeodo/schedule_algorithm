#!/usr/bin/env python3
"""Reusable transition model for the deployed fixed-candidate scheduler.

The original mirror in :mod:`eval_hw_mirror_s2pf_lite` intentionally exposes
only a complete greedy schedule.  This module exposes the *same* candidates as
state transitions so that selection/scoring loss can be separated from
candidate-space loss.  It does not add actions from the four-stage search.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass, replace
from typing import Callable, Iterable

import eval_hw_mirror_s2pf_lite as hw


cm = hw.cm
Remaining = tuple[tuple[int, int], ...]
Continuation = Callable[..., int]

HW_V2_CANDIDATE_POLICY = "one_idle_shape_v2"
TOP6_BOTTOM2_CANDIDATE_POLICY = "top6_bottom2_v1"
TOP6_BOTTOM2_PROTECTED_CANDIDATE_POLICY = "top6_bottom2_protected_v1"
TOP6_BOTTOM2_FIXED14_UNION_POLICY = "top6_bottom2_fixed14_union_v1"
TOP6_BOTTOM2_HEAD = 6
TOP6_BOTTOM2_TAIL = 2


@dataclass
class PolicyState:
    c2: cm.CSnap
    c3: cm.CSnap
    remaining: Remaining


@dataclass
class Transition:
    state: PolicyState
    tag: str


@dataclass(frozen=True)
class ScheduleStep:
    """One selected fixed-policy transition, including its input state."""

    before: PolicyState
    after: PolicyState
    tag: str


def initial_state(
    token_dist: dict[int, int], initial_cache_c2: int = -1, initial_cache_c3: int = -1
) -> PolicyState:
    remaining = tuple(
        sorted(
            ((int(eid), int(ntok)) for eid, ntok in token_dist.items()),
            key=lambda item: -item[1],
        )
    )
    return PolicyState(
        c2=cm._cc_initial(initial_cache_c2),
        c3=cm._cc_initial(initial_cache_c3),
        remaining=remaining,
    )


def state_key(state: PolicyState) -> tuple:
    return (astuple(state.c2), astuple(state.c3), state.remaining)


def top6_bottom2_window(remaining: Remaining) -> tuple[Remaining, Remaining]:
    """Return the fixed T0..T5 and B0..B1 descriptor views.

    ``remaining`` is sorted from hottest to coldest.  The bottom tuple is in
    selector order: B0 is the coldest remaining expert and B1 is the second
    coldest.  For short tails the two views may alias; generated child states
    are deduplicated before scoring.
    """
    head = tuple(remaining[:TOP6_BOTTOM2_HEAD])
    bottom = tuple(reversed(remaining[-TOP6_BOTTOM2_TAIL:]))
    return head, bottom


def terminal_cost(state: PolicyState) -> int:
    if state.remaining:
        raise ValueError("terminal_cost requires an empty remaining set")
    return max(state.c2.task_end, state.c3.task_end)


def _balance_divisible_work(loads: list[int], work: int) -> None:
    low = 0 if loads[0] <= loads[1] else 1
    high = 1 - low
    fill = min(loads[high] - loads[low], work)
    loads[low] += fill
    work -= fill
    if work:
        loads[low] += work // 2
        loads[high] += work - work // 2


def hw_v2_continuation(
    c2: cm.CSnap, c3: cm.CSnap, remaining: Remaining, *, policy: str
) -> int:
    """Fixed HW-v2 score: min(aggregate greedy, LPT top4 + tail balance).

    The function uses only the two candidate task ends, the first four experts
    after the candidate removal, and an aggregate sum for the remaining tail.
    It has no SIM1 expansion and does not generate a child candidate.
    """
    del policy
    if not remaining:
        return max(c2.task_end, c3.task_end)
    greedy = cm._cc_greedy_h(c2.task_end, c3.task_end, remaining)
    if len(remaining) <= 2:
        return greedy
    loads = [int(c2.task_end), int(c3.task_end)]
    for _, ntok in remaining[:4]:
        target = 0 if loads[0] <= loads[1] else 1
        loads[target] += int(cm._cc_best_task(int(ntok)))
    tail_work = sum(cm._cc_best_task(int(ntok)) for _, ntok in remaining[4:])
    _balance_divisible_work(loads, tail_work)
    return min(greedy, max(loads))


def hw_v2_continuation_top6(
    c2: cm.CSnap, c3: cm.CSnap, remaining: Remaining, *, policy: str
) -> int:
    """Use all T0..T5 descriptors before balancing an aggregate tail.

    This is the existing HW-v2 continuation datapath with two additional
    sequential compare/add placements.  It does not generate another action
    or inspect a child round.
    """
    del policy
    if not remaining:
        return max(c2.task_end, c3.task_end)
    greedy = cm._cc_greedy_h(c2.task_end, c3.task_end, remaining)
    if len(remaining) <= 2:
        return greedy
    loads = [int(c2.task_end), int(c3.task_end)]
    for _, ntok in remaining[:6]:
        target = 0 if loads[0] <= loads[1] else 1
        loads[target] += int(cm._cc_best_task(int(ntok)))
    tail_work = sum(cm._cc_best_task(int(ntok)) for _, ntok in remaining[6:])
    _balance_divisible_work(loads, tail_work)
    return min(greedy, max(loads))


def _prepare(c2: cm.CSnap, c3: cm.CSnap) -> tuple[cm.CSnap, cm.CSnap]:
    # Preserve the deployed sequential C2-then-C3 update order.
    if cm._cc_s4pf_ok_with_peer(c2, c3):
        c2 = cm._cc_apply_s4pf_ghost(c2)
    if cm._cc_s4pf_ok_with_peer(c3, c2):
        c3 = cm._cc_apply_s4pf_ghost(c3)
    return c2, c3


def _append_pair(
    out: list[Transition],
    *,
    family: str,
    tag: str,
    sa: cm.CSnap,
    s3a: int,
    sb: cm.CSnap,
    s3b: int,
    remaining: Remaining,
    policy: str,
) -> None:
    ta, tb = hw._hw_try_s2pf_pair(family, sa, s3a, sb, s3b, policy=policy)
    if cm._cc_bw_ok(ta, tb):
        out.append(Transition(PolicyState(ta, tb, remaining), tag))


def _mk_snap_with_s1_dma_bw(
    start: int,
    s1: int,
    s3: int,
    ntok: int,
    eid: int,
    s1_cached: bool,
    s3_cached: bool,
    *,
    s1_dma_bw: int | None = None,
) -> cm.CSnap:
    """Build one existing snap with an optional fixed DMA binding.

    Shape controls compute tiling.  The token ROM may independently bind the
    weight transfer to one lane (64) or both lanes (128).  Only the DMA end and
    occupancy change; the existing compute-stage endpoints stay untouched.
    """
    snap = cm._cc_mk_snap(
        start, s1, s3, ntok, eid, s1_cached, s3_cached
    )
    if s1_cached or s1_dma_bw is None or snap.bw_s1 == s1_dma_bw:
        return snap
    transfer_work = int(cm.C_TD1[s1]) * int(cm.C_ALLOC[s1])
    duration = (transfer_work + int(s1_dma_bw) - 1) // int(s1_dma_bw)
    updated = replace(snap)
    updated.dma1_end = int(start) + duration
    updated.bw_s1 = int(s1_dma_bw)
    return updated


def _apply_s2pf_with_dma_bw(
    snap: cm.CSnap,
    s3: int,
    start: int,
    dma_bw: int,
) -> cm.CSnap | None:
    """Apply the existing S2PF state update with a fixed DMA binding."""
    if snap.bw_s3 == 0:
        return snap
    transfer_work = int(cm.C_TD3[s3]) * int(cm.C_ALLOC[s3])
    duration = (transfer_work + int(dma_bw) - 1) // int(dma_bw)
    end = int(start) + duration
    if start < snap.dma1_end or end > snap.s2_end:
        return None
    updated = replace(snap)
    updated.s2pf_start = int(start)
    updated.s2pf_end = end
    updated.s2pf_bw = int(dma_bw)
    updated.dma3_end = updated.s2_end
    updated.s3_end = updated.s2_end
    updated.s4_start = updated.s2_end
    updated.bw_s3 = 0
    updated.task_end = updated.s2_end + cm._cc_best_s4(updated.ntok)
    return updated


def _append_fixed_pair(
    out: list[Transition],
    *,
    tag: str,
    c2: cm.CSnap,
    c3: cm.CSnap,
    remaining: Remaining,
    c2_s2pf_shape: int | None = None,
    c3_s2pf_shape: int | None = None,
    s2pf_dma_bw: int = 64,
) -> None:
    """Append one literal ROM profile through the existing bandwidth gate."""
    if c2_s2pf_shape is not None:
        c2 = _apply_s2pf_with_dma_bw(
            c2, c2_s2pf_shape, c2.dma1_end, s2pf_dma_bw
        )
        if c2 is None:
            return
    if c3_s2pf_shape is not None:
        c3 = _apply_s2pf_with_dma_bw(
            c3, c3_s2pf_shape, c3.dma1_end, s2pf_dma_bw
        )
        if c3 is None:
            return
    if cm._cc_bw_ok(c2, c3):
        out.append(Transition(PolicyState(c2, c3, remaining), tag))


def _terminal_successors(
    c2: cm.CSnap,
    c3: cm.CSnap,
    remaining: Remaining,
    *,
    policy: str,
    n1_policy: str,
) -> list[Transition]:
    eid, ntok = remaining[0]
    t2, t3 = c2.task_end, c3.task_end
    tnow = max(t2, t3)
    out: list[Transition] = []
    solo_shapes = (
        hw.N1_PRUNED_SOLO_SHAPES
        if n1_policy == "pruned"
        else tuple((s1, s3) for s1 in (0, 1, 2) for s3 in (0, 1, 2))
    )
    for cluster in (0, 1):
        own, peer = (c2, c3) if cluster == 0 else (c3, c2)
        start = own.task_end
        sw_hit = cm._cc_swiglu_hit(eid, own, start)
        down_hit = cm._cc_down_hit(eid, own, start)
        for s1, s3 in solo_shapes:
            snap = cm._cc_mk_snap(start, s1, s3, ntok, eid, sw_hit, down_hit)
            feasible = cm._cc_bw_ok(snap, peer) if cluster == 0 else cm._cc_bw_ok(peer, snap)
            if feasible:
                next_c2, next_c3 = (snap, peer) if cluster == 0 else (peer, snap)
                out.append(Transition(PolicyState(next_c2, next_c3, ()), f"last_solo_c{cluster+2}_{s1}{s3}"))

    if ntok >= 2:
        sw2 = cm._cc_swiglu_hit(eid, c2, tnow)
        dn2 = cm._cc_down_hit(eid, c2, tnow)
        sw3 = cm._cc_swiglu_hit(eid, c3, tnow)
        dn3 = cm._cc_down_hit(eid, c3, tnow)
        for left in hw._n1_split_cuts(ntok, n1_policy=n1_policy):
            right = ntok - left
            s12, s32, s13, s33 = cm._cc_pick_shapes(
                left, right, sw2, dn2, sw3, dn3, tnow
            )
            snap2 = cm._cc_mk_snap(tnow, s12, s32, left, eid, sw2, dn2)
            snap3 = cm._cc_mk_snap(tnow, s13, s33, right, eid, sw3, dn3)
            _append_pair(
                out,
                family="n1_split",
                tag=f"last_split_{left}",
                sa=snap2,
                s3a=s32,
                sb=snap3,
                s3b=s33,
                remaining=(),
                policy=policy,
            )

    if t2 != t3:
        idle_cluster = 0 if t2 < t3 else 1
        idle, busy = (c2, c3) if idle_cluster == 0 else (c3, c2)
        # Index zero is the ordinary idle-lane start already covered above.
        release_points = cm._cc_busy_time_points(busy, idle.task_end)
        if n1_policy == "pruned":
            release_points = release_points[1:]
        for release_index, start in enumerate(release_points):
            sw_hit = cm._cc_swiglu_hit(eid, idle, start)
            down_hit = cm._cc_down_hit(eid, idle, start)
            snap = cm._cc_mk_snap(
                start, cm.C_SHAPE_C, cm.C_SHAPE_C, ntok, eid, sw_hit, down_hit
            )
            feasible = (
                cm._cc_bw_ok(snap, busy)
                if idle_cluster == 0
                else cm._cc_bw_ok(busy, snap)
            )
            if feasible:
                next_c2, next_c3 = (
                    (snap, busy) if idle_cluster == 0 else (busy, snap)
                )
                out.append(
                    Transition(
                        PolicyState(next_c2, next_c3, ()),
                        f"last_release_c{idle_cluster+2}_{release_index}",
                    )
                )
    return out


def _both_idle_successors(
    c2: cm.CSnap,
    c3: cm.CSnap,
    remaining: Remaining,
    *,
    policy: str,
    top_policy: str,
) -> list[Transition]:
    top0_eid, top0_ntok = remaining[0]
    tnow = c2.task_end
    c2c0 = cm._cc_swiglu_hit(top0_eid, c2, tnow)
    c2f0 = cm._cc_down_hit(top0_eid, c2, tnow)
    c3c0 = cm._cc_swiglu_hit(top0_eid, c3, tnow)
    c3f0 = cm._cc_down_hit(top0_eid, c3, tnow)
    out: list[Transition] = []

    top0_ks = (
        [1]
        if top_policy == "pruned"
        else list(range(1, min(3, len(remaining) - 1) + 1))
    )
    for k in top0_ks:
        if k >= len(remaining):
            continue
        keid, kntok = remaining[k]
        rem_after = cm._cc_remove_eids(remaining, top0_eid, keid)
        sw2, dn2 = c2c0, c2f0
        sw3 = cm._cc_swiglu_hit(keid, c3, tnow)
        dn3 = cm._cc_down_hit(keid, c3, tnow)
        s12, s32, s13, s33 = cm._cc_pick_shapes(
            top0_ntok, kntok, sw2, dn2, sw3, dn3, tnow
        )
        _append_pair(
            out,
            family="pair_top0_topk",
            tag=f"pair_0_{k}",
            sa=cm._cc_mk_snap(tnow, s12, s32, top0_ntok, top0_eid, sw2, dn2),
            s3a=s32,
            sb=cm._cc_mk_snap(tnow, s13, s33, kntok, keid, sw3, dn3),
            s3b=s33,
            remaining=rem_after,
            policy=policy,
        )
        if top_policy == "full":
            sw2 = cm._cc_swiglu_hit(keid, c2, tnow)
            dn2 = cm._cc_down_hit(keid, c2, tnow)
            sw3, dn3 = c3c0, c3f0
            s12, s32, s13, s33 = cm._cc_pick_shapes(
                kntok, top0_ntok, sw2, dn2, sw3, dn3, tnow
            )
            _append_pair(
                out,
                family="pair_top0_topk",
                tag=f"pair_{k}_0",
                sa=cm._cc_mk_snap(tnow, s12, s32, kntok, keid, sw2, dn2),
                s3a=s32,
                sb=cm._cc_mk_snap(tnow, s13, s33, top0_ntok, top0_eid, sw3, dn3),
                s3b=s33,
                remaining=rem_after,
                policy=policy,
            )

    if len(remaining) >= 3:
        rank_pairs = (
            [(1, 2), (2, 3)]
            if top_policy == "pruned"
            else [
                (left, right)
                for left in range(1, min(3, len(remaining) - 1))
                for right in range(left + 1, min(3, len(remaining) - 1) + 1)
            ]
        )
        for left_rank, right_rank in rank_pairs:
            if right_rank >= len(remaining):
                continue
            left_eid, left_ntok = remaining[left_rank]
            right_eid, right_ntok = remaining[right_rank]
            rem_after = cm._cc_remove_eids(remaining, left_eid, right_eid)
            if not rem_after:
                continue
            sw2 = cm._cc_swiglu_hit(left_eid, c2, tnow)
            dn2 = cm._cc_down_hit(left_eid, c2, tnow)
            sw3 = cm._cc_swiglu_hit(right_eid, c3, tnow)
            dn3 = cm._cc_down_hit(right_eid, c3, tnow)
            s12, s32, s13, s33 = cm._cc_pick_shapes(
                left_ntok, right_ntok, sw2, dn2, sw3, dn3, tnow
            )
            _append_pair(
                out,
                family="pair_kj",
                tag=f"pair_{left_rank}_{right_rank}",
                sa=cm._cc_mk_snap(tnow, s12, s32, left_ntok, left_eid, sw2, dn2),
                s3a=s32,
                sb=cm._cc_mk_snap(tnow, s13, s33, right_ntok, right_eid, sw3, dn3),
                s3b=s33,
                remaining=rem_after,
                policy=policy,
            )

    if top0_ntok >= 2:
        rem_after = cm._cc_remove_eids(remaining, top0_eid)
        for cut in hw._split_cuts_for_policy(top0_ntok, top_policy=top_policy):
            other = top0_ntok - cut
            s12, s32, s13, s33 = cm._cc_pick_shapes(
                cut, other, c2c0, c2f0, c3c0, c3f0, tnow
            )
            _append_pair(
                out,
                family="split_top0",
                tag=f"split_0_{cut}",
                sa=cm._cc_mk_snap(tnow, s12, s32, cut, top0_eid, c2c0, c2f0),
                s3a=s32,
                sb=cm._cc_mk_snap(tnow, s13, s33, other, top0_eid, c3c0, c3f0),
                s3b=s33,
                remaining=rem_after,
                policy=policy,
            )

    if not out:
        fallback = cm._cc_mk_snap(
            tnow,
            cm.C_SHAPE_C,
            cm.C_SHAPE_C,
            top0_ntok,
            top0_eid,
            c2c0,
            c2f0,
        )
        out.append(
            Transition(
                PolicyState(fallback, c3, remaining[1:]), "both_idle_fallback"
            )
        )
    return out


def _one_idle_successors(
    c2: cm.CSnap, c3: cm.CSnap, remaining: Remaining
) -> list[Transition]:
    eid, ntok = remaining[0]
    idle_cluster = 0 if c2.task_end < c3.task_end else 1
    idle, busy = (c2, c3) if idle_cluster == 0 else (c3, c2)
    out: list[Transition] = []
    for point_index, start in enumerate(cm._cc_busy_time_points(busy, idle.task_end)):
        sw_hit = cm._cc_swiglu_hit(eid, idle, start)
        down_hit = cm._cc_down_hit(eid, idle, start)
        snap = cm._cc_mk_snap(
            start, cm.C_SHAPE_C, cm.C_SHAPE_C, ntok, eid, sw_hit, down_hit
        )
        if snap.bw_s3 > 0 and cm.C_TD3[cm.C_SHAPE_C] <= snap.s2_end - snap.dma1_end:
            prefetched = cm._cc_apply_s2pf(snap, cm.C_SHAPE_C, snap.dma1_end)
            if prefetched.s2pf_start >= 0:
                pf_ok = (
                    cm._cc_bw_ok(prefetched, busy)
                    if idle_cluster == 0
                    else cm._cc_bw_ok(busy, prefetched)
                )
                if pf_ok:
                    snap = prefetched
        feasible = (
            cm._cc_bw_ok(snap, busy)
            if idle_cluster == 0
            else cm._cc_bw_ok(busy, snap)
        )
        if feasible:
            next_c2, next_c3 = (
                (snap, busy) if idle_cluster == 0 else (busy, snap)
            )
            out.append(
                Transition(
                    PolicyState(next_c2, next_c3, remaining[1:]),
                    f"one_idle_c{idle_cluster+2}_p{point_index}",
                )
            )
    if not out:
        start = idle.task_end
        sw_hit = cm._cc_swiglu_hit(eid, idle, start)
        down_hit = cm._cc_down_hit(eid, idle, start)
        snap = cm._cc_mk_snap(
            start, cm.C_SHAPE_C, cm.C_SHAPE_C, ntok, eid, sw_hit, down_hit
        )
        next_c2, next_c3 = (snap, busy) if idle_cluster == 0 else (busy, snap)
        out.append(
            Transition(
                PolicyState(next_c2, next_c3, remaining[1:]), "one_idle_fallback"
            )
        )
    return out


def generate_successors(
    state: PolicyState,
    *,
    policy: str = "balanced",
    top_policy: str = "pruned",
    n1_policy: str = "pruned",
) -> list[Transition]:
    if not state.remaining:
        return []
    c2, c3 = _prepare(state.c2, state.c3)
    if len(state.remaining) == 1:
        return _terminal_successors(
            c2, c3, state.remaining, policy=policy, n1_policy=n1_policy
        )
    if c2.task_end == c3.task_end:
        return _both_idle_successors(
            c2, c3, state.remaining, policy=policy, top_policy=top_policy
        )
    return _one_idle_successors(c2, c3, state.remaining)


def _adaptive_uncached_shapes(ntok: int) -> tuple[int, int]:
    """One fixed token-threshold profile distilled from proven references."""
    if ntok >= 7:
        return cm.C_SHAPE_A, cm.C_SHAPE_B
    if ntok >= 3:
        return cm.C_SHAPE_B, cm.C_SHAPE_B
    return cm.C_SHAPE_C, cm.C_SHAPE_C


def _concrete_resident(snap: cm.CSnap, remaining: Remaining, now: int) -> int | None:
    if snap.pf_eid < 0 or not snap.pf_full or snap.pf_end > now:
        return None
    return snap.pf_eid if any(eid == snap.pf_eid for eid, _ in remaining) else None


def generate_augmented_successors(
    state: PolicyState,
    *,
    policy: str = "balanced",
    top_policy: str = "pruned",
    n1_policy: str = "pruned",
    add_resident: bool = True,
    add_one_idle_shape: bool = True,
) -> list[Transition]:
    """Bounded v2 candidate bank: deployed bank plus two evidence-led families.

    Added candidates are limited to concrete resident experts when both lanes
    are idle, and one token-threshold shape alternative when one lane is idle.
    There is no rank expansion, standalone prefetch, child generation, or
    variable-size candidate loop.
    """
    base = generate_successors(
        state, policy=policy, top_policy=top_policy, n1_policy=n1_policy
    )
    if len(state.remaining) <= 1:
        return base
    c2, c3 = _prepare(state.c2, state.c3)
    extras: list[Transition] = []
    if c2.task_end != c3.task_end and add_one_idle_shape:
        eid, ntok = state.remaining[0]
        idle_cluster = 0 if c2.task_end < c3.task_end else 1
        idle, busy = (c2, c3) if idle_cluster == 0 else (c3, c2)
        s1, s3 = _adaptive_uncached_shapes(ntok)
        for point_index, start in enumerate(cm._cc_busy_time_points(busy, idle.task_end)):
            sw_hit = cm._cc_swiglu_hit(eid, idle, start)
            down_hit = cm._cc_down_hit(eid, idle, start)
            snap = cm._cc_mk_snap(start, s1, s3, ntok, eid, sw_hit, down_hit)
            feasible = (
                cm._cc_bw_ok(snap, busy)
                if idle_cluster == 0
                else cm._cc_bw_ok(busy, snap)
            )
            if feasible:
                next_c2, next_c3 = (
                    (snap, busy) if idle_cluster == 0 else (busy, snap)
                )
                extras.append(
                    Transition(
                        PolicyState(next_c2, next_c3, state.remaining[1:]),
                        f"one_idle_adaptive_c{idle_cluster+2}_p{point_index}",
                    )
                )
    elif c2.task_end == c3.task_end and add_resident:
        now = c2.task_end
        residents = (
            _concrete_resident(c2, state.remaining, now),
            _concrete_resident(c3, state.remaining, now),
        )
        token_by_eid = dict(state.remaining)
        for cluster, eid in enumerate(residents):
            if eid is None:
                continue
            ntok = token_by_eid[eid]
            own, peer = (c2, c3) if cluster == 0 else (c3, c2)
            snap = cm._cc_mk_snap(
                now, cm.C_SHAPE_C, cm.C_SHAPE_C, ntok, eid, True, True
            )
            feasible = cm._cc_bw_ok(snap, peer) if cluster == 0 else cm._cc_bw_ok(peer, snap)
            if feasible:
                next_c2, next_c3 = (snap, peer) if cluster == 0 else (peer, snap)
                extras.append(
                    Transition(
                        PolicyState(
                            next_c2,
                            next_c3,
                            cm._cc_remove_eids(state.remaining, eid),
                        ),
                        f"resident_single_c{cluster+2}",
                    )
                )
            if ntok >= 2:
                left = ntok - 1 if cluster == 0 else 1
                right = ntok - left
                sw2 = cm._cc_swiglu_hit(eid, c2, now)
                dn2 = cm._cc_down_hit(eid, c2, now)
                sw3 = cm._cc_swiglu_hit(eid, c3, now)
                dn3 = cm._cc_down_hit(eid, c3, now)
                s12, s32, s13, s33 = cm._cc_pick_shapes(
                    left, right, sw2, dn2, sw3, dn3, now
                )
                _append_pair(
                    extras,
                    family="split_resident",
                    tag=f"resident_split_c{cluster+2}",
                    sa=cm._cc_mk_snap(now, s12, s32, left, eid, sw2, dn2),
                    s3a=s32,
                    sb=cm._cc_mk_snap(now, s13, s33, right, eid, sw3, dn3),
                    s3b=s33,
                    remaining=cm._cc_remove_eids(state.remaining, eid),
                    policy=policy,
                )

        if residents[0] is not None and residents[1] is not None and residents[0] != residents[1]:
            eid2, eid3 = residents
            ntok2, ntok3 = token_by_eid[eid2], token_by_eid[eid3]
            sw2 = cm._cc_swiglu_hit(eid2, c2, now)
            dn2 = cm._cc_down_hit(eid2, c2, now)
            sw3 = cm._cc_swiglu_hit(eid3, c3, now)
            dn3 = cm._cc_down_hit(eid3, c3, now)
            s12, s32, s13, s33 = cm._cc_pick_shapes(
                ntok2, ntok3, sw2, dn2, sw3, dn3, now
            )
            _append_pair(
                extras,
                family="pair_resident",
                tag="resident_pair",
                sa=cm._cc_mk_snap(now, s12, s32, ntok2, eid2, sw2, dn2),
                s3a=s32,
                sb=cm._cc_mk_snap(now, s13, s33, ntok3, eid3, sw3, dn3),
                s3b=s33,
                remaining=cm._cc_remove_eids(state.remaining, eid2, eid3),
                policy=policy,
            )

    unique = {}
    for transition in [*base, *extras]:
        unique.setdefault(state_key(transition.state), transition)
    return list(unique.values())


def generate_one_idle_shape_successors(
    state: PolicyState,
    *,
    policy: str = "balanced",
    top_policy: str = "pruned",
    n1_policy: str = "pruned",
) -> list[Transition]:
    return generate_augmented_successors(
        state,
        policy=policy,
        top_policy=top_policy,
        n1_policy=n1_policy,
        add_resident=False,
        add_one_idle_shape=True,
    )


def _top6_bottom2_pair_successors(
    c2: cm.CSnap,
    c3: cm.CSnap,
    remaining: Remaining,
    *,
    policy: str,
) -> list[Transition]:
    """Add the fixed hot+cold pair through the existing pair datapath."""
    if len(remaining) < 2 or c2.task_end != c3.task_end:
        return []
    head, bottom = top6_bottom2_window(remaining)
    if not head or not bottom:
        return []
    hot_eid, hot_ntok = head[0]
    cold_eid, cold_ntok = bottom[0]
    if hot_eid == cold_eid:
        return []

    now = c2.task_end
    # Preserve the distilled physical orientation: cold on C2, hot on C3.
    sw2 = cm._cc_swiglu_hit(cold_eid, c2, now)
    dn2 = cm._cc_down_hit(cold_eid, c2, now)
    sw3 = cm._cc_swiglu_hit(hot_eid, c3, now)
    dn3 = cm._cc_down_hit(hot_eid, c3, now)
    s12, s32, s13, s33 = cm._cc_pick_shapes(
        cold_ntok, hot_ntok, sw2, dn2, sw3, dn3, now
    )
    out: list[Transition] = []
    _append_pair(
        out,
        family="pair_top_bottom",
        tag="t6b2_pair_t0_b0",
        sa=cm._cc_mk_snap(now, s12, s32, cold_ntok, cold_eid, sw2, dn2),
        s3a=s32,
        sb=cm._cc_mk_snap(now, s13, s33, hot_ntok, hot_eid, sw3, dn3),
        s3b=s33,
        remaining=cm._cc_remove_eids(remaining, hot_eid, cold_eid),
        policy=policy,
    )
    return out


def _top6_bottom2_single_successors(
    c2: cm.CSnap,
    c3: cm.CSnap,
    remaining: Remaining,
) -> list[Transition]:
    """Add B0/B1 ONE_IDLE singles through the existing C/C datapath."""
    if not remaining or c2.task_end == c3.task_end:
        return []
    _head, bottom = top6_bottom2_window(remaining)
    idle_cluster = 0 if c2.task_end < c3.task_end else 1
    idle, busy = (c2, c3) if idle_cluster == 0 else (c3, c2)
    out: list[Transition] = []
    for bottom_rank, (eid, ntok) in enumerate(bottom):
        for point_index, start in enumerate(
            cm._cc_busy_time_points(busy, idle.task_end)
        ):
            sw_hit = cm._cc_swiglu_hit(eid, idle, start)
            down_hit = cm._cc_down_hit(eid, idle, start)
            snap = cm._cc_mk_snap(
                start,
                cm.C_SHAPE_C,
                cm.C_SHAPE_C,
                ntok,
                eid,
                sw_hit,
                down_hit,
            )
            if (
                snap.bw_s3 > 0
                and cm.C_TD3[cm.C_SHAPE_C] <= snap.s2_end - snap.dma1_end
            ):
                prefetched = cm._cc_apply_s2pf(
                    snap, cm.C_SHAPE_C, snap.dma1_end
                )
                if prefetched.s2pf_start >= 0:
                    pf_ok = (
                        cm._cc_bw_ok(prefetched, busy)
                        if idle_cluster == 0
                        else cm._cc_bw_ok(busy, prefetched)
                    )
                    if pf_ok:
                        snap = prefetched
            feasible = (
                cm._cc_bw_ok(snap, busy)
                if idle_cluster == 0
                else cm._cc_bw_ok(busy, snap)
            )
            if not feasible:
                continue
            next_c2, next_c3 = (
                (snap, busy) if idle_cluster == 0 else (busy, snap)
            )
            out.append(
                Transition(
                    PolicyState(
                        next_c2,
                        next_c3,
                        cm._cc_remove_eids(remaining, eid),
                    ),
                    f"t6b2_one_idle_b{bottom_rank}_c{idle_cluster+2}_p{point_index}",
                )
            )
    return out


def generate_top6_bottom2_successors(
    state: PolicyState,
    *,
    policy: str = "balanced",
    top_policy: str = "pruned",
    n1_policy: str = "pruned",
) -> list[Transition]:
    """Baseline candidates plus fixed T0+B0 and B0/B1 additions.

    This is an additive candidate-space ablation.  It reuses the current
    shape, S2PF, bandwidth and score datapaths; no rollout, dynamic search or
    alternative timing model is introduced.
    """
    baseline = generate_one_idle_shape_successors(
        state,
        policy=policy,
        top_policy=top_policy,
        n1_policy=n1_policy,
    )
    if len(state.remaining) <= 1:
        return baseline
    c2, c3 = _prepare(state.c2, state.c3)
    if c2.task_end == c3.task_end:
        extras = _top6_bottom2_pair_successors(
            c2, c3, state.remaining, policy=policy
        )
    else:
        extras = _top6_bottom2_single_successors(c2, c3, state.remaining)
    unique = {}
    for transition in [*baseline, *extras]:
        unique.setdefault(state_key(transition.state), transition)
    return list(unique.values())


def generate_top6_bottom2_protected_successors(
    state: PolicyState,
    *,
    policy: str = "balanced",
    top_policy: str = "pruned",
    n1_policy: str = "pruned",
) -> list[Transition]:
    """Old bank plus the final single protected T6+B2 candidate per mode.

    SYNC exposes T0+B0.  ONE_IDLE exposes only B0 at the earliest legal
    release point.  B1 and later release points were ablation candidates; the
    final score valid gate rejects them unconditionally, so the RTL-oriented
    bank does not generate or score them.
    """
    baseline = generate_one_idle_shape_successors(
        state,
        policy=policy,
        top_policy=top_policy,
        n1_policy=n1_policy,
    )
    if len(state.remaining) <= 1:
        return baseline
    c2, c3 = _prepare(state.c2, state.c3)
    if c2.task_end == c3.task_end:
        extras = _top6_bottom2_pair_successors(
            c2, c3, state.remaining, policy=policy
        )
    else:
        extras = [
            transition
            for transition in _top6_bottom2_single_successors(
                c2, c3, state.remaining
            )
            if transition.tag.startswith("t6b2_one_idle_b0_")
            and transition.tag.endswith("_p0")
        ]
    unique = {}
    for transition in [*baseline, *extras]:
        unique.setdefault(state_key(transition.state), transition)
    return list(unique.values())


def _fixed14_sync_successors(
    c2: cm.CSnap,
    c3: cm.CSnap,
    remaining: Remaining,
) -> list[Transition]:
    """Literal SYNC profiles distilled from the replayed optimal bank.

    Selectors are state-relative T0..T5/B0..B1.  Every profile is a fixed ROM
    entry; there is no rank loop in the intended implementation.  The Python
    helper uses a compact table to avoid duplicating the same snap plumbing.
    """
    if len(remaining) < 2 or c2.task_end != c3.task_end:
        return []
    head, bottom = top6_bottom2_window(remaining)
    now = c2.task_end
    out: list[Transition] = []

    def pair(
        left: tuple[int, int],
        right: tuple[int, int],
        *,
        left_shapes: tuple[int, int],
        right_shapes: tuple[int, int],
        left_pf: bool,
        right_pf: bool,
        tag: str,
    ) -> None:
        left_eid, left_ntok = left
        right_eid, right_ntok = right
        if left_eid == right_eid:
            return
        left_s1, left_s3 = left_shapes
        right_s1, right_s3 = right_shapes
        left_snap = _mk_snap_with_s1_dma_bw(
            now,
            left_s1,
            left_s3,
            left_ntok,
            left_eid,
            cm._cc_swiglu_hit(left_eid, c2, now),
            cm._cc_down_hit(left_eid, c2, now),
        )
        right_snap = _mk_snap_with_s1_dma_bw(
            now,
            right_s1,
            right_s3,
            right_ntok,
            right_eid,
            cm._cc_swiglu_hit(right_eid, c3, now),
            cm._cc_down_hit(right_eid, c3, now),
        )
        _append_fixed_pair(
            out,
            tag=tag,
            c2=left_snap,
            c3=right_snap,
            remaining=cm._cc_remove_eids(remaining, left_eid, right_eid),
            c2_s2pf_shape=left_s3 if left_pf else None,
            c3_s2pf_shape=right_s3 if right_pf else None,
            s2pf_dma_bw=64,
        )

    # The two asymmetric A/B--B/B profiles are emitted in both physical
    # orientations.  This is a fixed factor of two and preserves cache/DMA
    # symmetry without a runtime search loop.
    for selector_name, other in (
        ("b0", bottom[0] if bottom else None),
        ("t4", head[4] if len(head) > 4 else None),
    ):
        if other is None:
            continue
        hot = head[0]
        pair(
            hot,
            other,
            left_shapes=(cm.C_SHAPE_A, cm.C_SHAPE_B),
            right_shapes=(cm.C_SHAPE_B, cm.C_SHAPE_B),
            left_pf=True,
            right_pf=False,
            tag=f"fixed14_pair_t0_{selector_name}_ab_bb_c2hot",
        )
        pair(
            other,
            hot,
            left_shapes=(cm.C_SHAPE_B, cm.C_SHAPE_B),
            right_shapes=(cm.C_SHAPE_A, cm.C_SHAPE_B),
            left_pf=False,
            right_pf=True,
            tag=f"fixed14_pair_t0_{selector_name}_ab_bb_c3hot",
        )

    # Symmetric B/B pair profiles from the 14-entry bank.  These are separate
    # candidates because no-S2PF, one-side S2PF and both-side S2PF have
    # different DMA occupancy even when their task ends coincide.
    pair_specs = (
        (0, 1, False, False, "t0_t1_none"),
        (0, 1, True, False, "t0_t1_c2pf"),
        (0, 1, False, True, "t0_t1_c3pf"),
        (1, 2, True, True, "t1_t2_bothpf"),
        (2, 3, True, True, "t2_t3_bothpf"),
    )
    for left_rank, right_rank, left_pf, right_pf, name in pair_specs:
        if right_rank >= len(head):
            continue
        pair(
            head[left_rank],
            head[right_rank],
            left_shapes=(cm.C_SHAPE_B, cm.C_SHAPE_B),
            right_shapes=(cm.C_SHAPE_B, cm.C_SHAPE_B),
            left_pf=left_pf,
            right_pf=right_pf,
            tag=f"fixed14_pair_{name}",
        )
    return out


def _fixed14_one_idle_successors(
    c2: cm.CSnap,
    c3: cm.CSnap,
    remaining: Remaining,
) -> list[Transition]:
    """Fixed cold C/C and T0/T3 B/B singles with S4PF disabled."""
    if not remaining or c2.task_end == c3.task_end:
        return []
    head, bottom = top6_bottom2_window(remaining)
    idle_cluster = 0 if c2.task_end < c3.task_end else 1
    idle, busy = (c2, c3) if idle_cluster == 0 else (c3, c2)
    out: list[Transition] = []
    release_points = cm._cc_busy_time_points(busy, idle.task_end)

    # The distilled cold-fill profiles explicitly carry S4PF=NONE.  Keeping
    # them separate from the deployed eager-S4PF bottom candidates is what
    # lets the fixed ROM represent the certified hot+cold overlap path.
    for bottom_rank, (eid, ntok) in enumerate(bottom):
        for point_index, start in enumerate(release_points):
            snap = _mk_snap_with_s1_dma_bw(
                start,
                cm.C_SHAPE_C,
                cm.C_SHAPE_C,
                ntok,
                eid,
                cm._cc_swiglu_hit(eid, idle, start),
                cm._cc_down_hit(eid, idle, start),
                s1_dma_bw=128,
            )
            feasible = (
                cm._cc_bw_ok(snap, busy)
                if idle_cluster == 0
                else cm._cc_bw_ok(busy, snap)
            )
            if not feasible:
                continue
            next_c2, next_c3 = (
                (snap, busy) if idle_cluster == 0 else (busy, snap)
            )
            out.append(
                Transition(
                    PolicyState(
                        next_c2,
                        next_c3,
                        cm._cc_remove_eids(remaining, eid),
                    ),
                    f"fixed14_single_b{bottom_rank}_c{idle_cluster+2}_p{point_index}",
                )
            )

    for rank in (0, 3):
        if rank >= len(head):
            continue
        eid, ntok = head[rank]
        for point_index, start in enumerate(release_points):
            snap = _mk_snap_with_s1_dma_bw(
                start,
                cm.C_SHAPE_B,
                cm.C_SHAPE_B,
                ntok,
                eid,
                cm._cc_swiglu_hit(eid, idle, start),
                cm._cc_down_hit(eid, idle, start),
                s1_dma_bw=128,
            )
            snap = _apply_s2pf_with_dma_bw(
                snap,
                cm.C_SHAPE_B,
                snap.dma1_end,
                128,
            )
            if snap is None:
                continue
            feasible = (
                cm._cc_bw_ok(snap, busy)
                if idle_cluster == 0
                else cm._cc_bw_ok(busy, snap)
            )
            if not feasible:
                continue
            next_c2, next_c3 = (
                (snap, busy) if idle_cluster == 0 else (busy, snap)
            )
            out.append(
                Transition(
                    PolicyState(
                        next_c2,
                        next_c3,
                        cm._cc_remove_eids(remaining, eid),
                    ),
                    f"fixed14_single_t{rank}_c{idle_cluster+2}_p{point_index}",
                )
            )
    return out


def generate_top6_bottom2_fixed14_union_successors(
    state: PolicyState,
    *,
    policy: str = "balanced",
    top_policy: str = "pruned",
    n1_policy: str = "pruned",
) -> list[Transition]:
    """Old adaptive bank plus T6+B2 and the fixed14 physical templates."""
    baseline = generate_top6_bottom2_successors(
        state,
        policy=policy,
        top_policy=top_policy,
        n1_policy=n1_policy,
    )
    if len(state.remaining) <= 1:
        return baseline
    # Baseline candidates above retain the deployed eager S4PF preparation.
    # The distilled ROM profiles explicitly encode S4PF=NONE, so evaluate
    # those profiles from the unmodified input snapshots.  This is a fixed
    # per-template enable bit, not a second policy rollout.
    c2, c3 = state.c2, state.c3
    extras = (
        _fixed14_sync_successors(c2, c3, state.remaining)
        if c2.task_end == c3.task_end
        else _fixed14_one_idle_successors(c2, c3, state.remaining)
    )
    unique = {}
    for transition in [*baseline, *extras]:
        unique.setdefault(state_key(transition.state), transition)
    return list(unique.values())


def generate_resident_successors(
    state: PolicyState,
    *,
    policy: str = "balanced",
    top_policy: str = "pruned",
    n1_policy: str = "pruned",
) -> list[Transition]:
    return generate_augmented_successors(
        state,
        policy=policy,
        top_policy=top_policy,
        n1_policy=n1_policy,
        add_resident=True,
        add_one_idle_shape=False,
    )


def _best_immediate(transitions: Iterable[Transition]) -> Transition:
    return min(
        transitions,
        key=lambda transition: max(
            transition.state.c2.task_end, transition.state.c3.task_end
        ),
    )


def _run_with_scorer(
    token_dist: dict[int, int],
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
    *,
    continuation: Continuation = hw._hw_continuation_cost,
    policy: str = "balanced",
    top_policy: str = "pruned",
    n1_policy: str = "pruned",
    candidate_policy: str = "deployed",
) -> tuple[PolicyState, tuple[ScheduleStep, ...]]:
    """Run deployed greedy control and retain every selected transition."""
    state = initial_state(token_dist, initial_cache_c2, initial_cache_c3)
    trace: list[ScheduleStep] = []
    while state.remaining:
        if candidate_policy == "deployed":
            generator = generate_successors
        elif candidate_policy == "one_idle_shape_v2":
            generator = generate_one_idle_shape_successors
        elif candidate_policy == "resident_v2":
            generator = generate_resident_successors
        elif candidate_policy == "resident_shape_v2":
            generator = generate_augmented_successors
        elif candidate_policy == TOP6_BOTTOM2_CANDIDATE_POLICY:
            generator = generate_top6_bottom2_successors
        elif candidate_policy == TOP6_BOTTOM2_PROTECTED_CANDIDATE_POLICY:
            generator = generate_top6_bottom2_protected_successors
        else:
            raise ValueError(f"unknown candidate_policy {candidate_policy!r}")
        transitions = generator(
            state, policy=policy, top_policy=top_policy, n1_policy=n1_policy
        )
        if len(state.remaining) == 1 or state.c2.task_end != state.c3.task_end:
            chosen = _best_immediate(transitions)
        else:
            def key(transition: Transition) -> tuple[int, int, int]:
                child = transition.state
                return (
                    continuation(
                        child.c2, child.c3, child.remaining, policy=policy
                    ),
                    len(child.remaining),
                    max(child.c2.task_end, child.c3.task_end),
                )

            chosen = min(transitions, key=key)
        trace.append(ScheduleStep(state, chosen.state, chosen.tag))
        state = chosen.state
    return state, tuple(trace)


def schedule_trace_with_scorer(
    token_dist: dict[int, int],
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
    *,
    continuation: Continuation = hw._hw_continuation_cost,
    policy: str = "balanced",
    top_policy: str = "pruned",
    n1_policy: str = "pruned",
    candidate_policy: str = "deployed",
) -> tuple[ScheduleStep, ...]:
    """Return the exact transition sequence selected by the fixed policy."""
    _terminal, trace = _run_with_scorer(
        token_dist,
        initial_cache_c2,
        initial_cache_c3,
        continuation=continuation,
        policy=policy,
        top_policy=top_policy,
        n1_policy=n1_policy,
        candidate_policy=candidate_policy,
    )
    return trace


def schedule_with_scorer(
    token_dist: dict[int, int],
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
    *,
    continuation: Continuation = hw._hw_continuation_cost,
    policy: str = "balanced",
    top_policy: str = "pruned",
    n1_policy: str = "pruned",
    candidate_policy: str = "deployed",
) -> int:
    """Reproduce deployed greedy control while allowing a scorer callback."""
    terminal, _trace = _run_with_scorer(
        token_dist,
        initial_cache_c2,
        initial_cache_c3,
        continuation=continuation,
        policy=policy,
        top_policy=top_policy,
        n1_policy=n1_policy,
        candidate_policy=candidate_policy,
    )
    return terminal_cost(terminal)


def hw_v2_schedule(
    token_dist: dict[int, int],
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
    *,
    policy: str = "balanced",
    top_policy: str = "pruned",
    n1_policy: str = "pruned",
) -> int:
    """Run the frozen algorithmic HW-v2 policy."""
    return schedule_with_scorer(
        token_dist,
        initial_cache_c2,
        initial_cache_c3,
        continuation=hw_v2_continuation,
        policy=policy,
        top_policy=top_policy,
        n1_policy=n1_policy,
        candidate_policy=HW_V2_CANDIDATE_POLICY,
    )
