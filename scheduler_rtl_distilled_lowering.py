#!/usr/bin/env python3
"""Bounded candidate-profile lowering for the final RTL scheduler mirror.

The module contains no search, learned parameters, legacy candidate banks or
top-level prefetch candidates.  It resolves the fixed Top5+Bottom1 profiles
into legal four-stage actions for one runtime state.  S4PF is lowered together
with a concrete next same-cluster consumer, matching the fixed C2-then-C3 and
local-SINGLE-then-BOTH trial order.  The no-S4PF realization is always kept as
the local baseline.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import four_stage_scheduler as reference
from scheduler_rtl_distilled_types import (
    CandidateProfile,
    LogicalActionSpec,
    PhysicalProfile,
    WINDOW,
)


_SHAPES = {
    "NONE": None,
    reference.SHAPE_A.name: reference.SHAPE_A,
    reference.SHAPE_B.name: reference.SHAPE_B,
    reference.SHAPE_C.name: reference.SHAPE_C,
}

# Targeted S4PF is disabled in the short tail.  With eight or fewer remaining
# descriptors, the continuation comparator is already close to terminal and a
# small local timing change can reorder the exact tail without enough future
# reuse to recover it.  The threshold is validated as part of the OFF/ON
# closed-loop comparison rather than being a workload classifier.
S4PF_MIN_REMAINING = 9


def mode(state: reference.BeamState) -> str:
    if len(state.remaining) == 1:
        return "TERMINAL"
    return "SYNC" if state.c2.task_end == state.c3.task_end else "ONE_IDLE"


def resolve_selector(
    state: reference.BeamState,
    selector: str,
    window: tuple[int, int] = WINDOW,
) -> int | None:
    """Resolve a Top/Bottom rank selector without exposing hidden descriptors."""
    entries = len(state.remaining)
    top, bottom = window
    if selector.startswith("T"):
        rank = int(selector[1:])
        return int(state.remaining[rank][0]) if rank < min(top, entries) else None
    if selector.startswith("B"):
        offset = int(selector[1:])
        if offset >= bottom or offset >= entries:
            return None
        return int(state.remaining[entries - 1 - offset][0])
    raise ValueError(f"unsupported bounded selector {selector!r}")


def _rank_label(
    state: reference.BeamState,
    eid: int,
    window: tuple[int, int] = WINDOW,
) -> str:
    if eid < 0:
        return "NONE"
    rank_by_eid = {
        int(candidate_eid): rank
        for rank, (candidate_eid, _ntok) in enumerate(state.remaining)
    }
    rank = rank_by_eid.get(int(eid))
    if rank is None:
        raise ValueError(f"E{eid} is not in remaining")
    top, bottom = window
    entries = len(state.remaining)
    if rank < min(top, entries):
        return f"T{rank}"
    if bottom and rank >= max(min(top, entries), entries - bottom):
        return f"B{entries - 1 - rank}"
    raise ValueError(
        f"E{eid} is outside top{top}+bottom{bottom} observation window"
    )


def _family(action: reference.StageAction) -> str:
    if action.c2_eid >= 0 and action.c2_eid == action.c3_eid:
        return "SPLIT"
    if action.c2_eid >= 0 and action.c3_eid >= 0:
        return "PAIR"
    if action.c2_eid >= 0 or action.c3_eid >= 0:
        return "SINGLE"
    raise ValueError("the final scheduler supports consuming actions only")


def _split_rule(action: reference.StageAction) -> str:
    if _family(action) != "SPLIT":
        return "NONE"
    low, high = sorted((int(action.c2_ntok), int(action.c3_ntok)))
    return "HALF" if low == high else "BALANCED"


def logical_action_spec(
    state: reference.BeamState,
    action: reference.StageAction,
    window: tuple[int, int] = WINDOW,
) -> LogicalActionSpec:
    family = _family(action)
    if family == "PAIR":
        selectors = tuple(
            sorted(
                (
                    _rank_label(state, action.c2_eid, window),
                    _rank_label(state, action.c3_eid, window),
                )
            )
        )
    elif family == "SPLIT":
        selectors = (_rank_label(state, action.c2_eid, window),)
    else:
        eid = action.c2_eid if action.c2_eid >= 0 else action.c3_eid
        selectors = (_rank_label(state, eid, window),)
    return LogicalActionSpec(
        mode=mode(state),
        family=family,
        selectors=selectors,
        split_rule=_split_rule(action),
    )


def physical_profile(action: reference.StageAction) -> PhysicalProfile:
    shape_name = lambda shape: "NONE" if shape is None else str(shape.name)
    return PhysicalProfile(
        c2_s1=shape_name(action.c2_shape_s1),
        c2_s3=shape_name(action.c2_shape_s3),
        c3_s1=shape_name(action.c3_shape_s1),
        c3_s3=shape_name(action.c3_shape_s3),
        c2_dma_s1=action.c2_dma_s1.name,
        c2_dma_s3=action.c2_dma_s3.name,
        c2_s2pf=action.c2_s2pf_dma.name,
        c3_dma_s1=action.c3_dma_s1.name,
        c3_dma_s3=action.c3_dma_s3.name,
        c3_s2pf=action.c3_s2pf_dma.name,
        s4pf_dma=action.pf_dma.name,
        c2_s1_cached=bool(action.c2_s1_cached),
        c2_s3_cached=bool(action.c2_s3_cached),
        c3_s1_cached=bool(action.c3_s1_cached),
        c3_s3_cached=bool(action.c3_s3_cached),
    )


def child_key(state: reference.BeamState) -> tuple:
    """Exact future-equivalence key used to deduplicate concrete actions."""
    return state.fingerprint(), int(state.cluster_work_cc)


def selected_action_features(
    action: reference.StageAction,
) -> tuple[int, int, int, int]:
    selected_tokens = [
        int(ntok)
        for eid, ntok in (
            (action.c2_eid, action.c2_ntok),
            (action.c3_eid, action.c3_ntok),
        )
        if eid >= 0
    ]
    s2pf_count = sum(
        binding != reference.DmaBinding.NONE
        for binding in (action.c2_s2pf_dma, action.c3_s2pf_dma)
    )
    return (
        max(selected_tokens, default=0),
        min(selected_tokens, default=0),
        sum(selected_tokens),
        int(s2pf_count),
    )


def _shape(name: str) -> reference.Shape | None:
    try:
        return _SHAPES[name]
    except KeyError as exc:
        raise ValueError(f"unknown fixed shape {name!r}") from exc


def _dma(name: str) -> reference.DmaBinding:
    try:
        return reference.DmaBinding[name]
    except KeyError as exc:
        raise ValueError(f"unknown fixed DMA binding {name!r}") from exc


def _single_dma_for_cluster(cluster: int) -> reference.DmaBinding:
    """Return the fixed owning lane used by all single-lane prefetches."""
    if cluster == 2:
        return reference.DmaBinding.IDMA
    if cluster == 3:
        return reference.DmaBinding.XDMA
    raise ValueError(f"invalid cluster {cluster}")


def _active_cluster(profile: PhysicalProfile) -> int | None:
    active_c2 = profile.c2_s1 != "NONE"
    active_c3 = profile.c3_s1 != "NONE"
    if active_c2 == active_c3:
        return None
    return 2 if active_c2 else 3


def _profile_with_residency(
    profile: PhysicalProfile,
    *,
    c2_s1: bool,
    c2_s3: bool,
    c3_s1: bool,
    c3_s3: bool,
) -> PhysicalProfile:
    values = {
        field: getattr(profile, field)
        for field in profile.__dataclass_fields__
    }
    for prefix, s1_hit, s3_hit in (
        ("c2", c2_s1, c2_s3),
        ("c3", c3_s1, c3_s3),
    ):
        if values[f"{prefix}_s1"] == "NONE":
            continue
        if s1_hit:
            values[f"{prefix}_s1_cached"] = True
            values[f"{prefix}_dma_s1"] = "NONE"
        if s3_hit:
            values[f"{prefix}_s3_cached"] = True
            values[f"{prefix}_dma_s3"] = "NONE"
            values[f"{prefix}_s2pf"] = "NONE"
    return PhysicalProfile(**values)


def _hit_state(
    eid: int,
    snap: reference.FourStageSnap,
    start: int,
) -> tuple[bool, bool]:
    return (
        reference._swiglu_hit_for_candidate(eid, snap, start),
        reference._down_hit_for_candidate(eid, snap, start),
    )


def _runtime_variants(
    state: reference.BeamState,
    token: CandidateProfile,
) -> tuple[CandidateProfile, ...]:
    logical = token.logical
    if logical.mode != mode(state):
        return (token,)
    selected = tuple(resolve_selector(state, selector) for selector in logical.selectors)
    if any(eid is None for eid in selected):
        return (token,)
    eids = tuple(int(eid) for eid in selected if eid is not None)
    variants: set[PhysicalProfile] = set()

    if logical.family == "PAIR" and len(eids) == 2:
        now = max(int(state.c2.task_end), int(state.c3.task_end))
        for eid2, eid3 in (eids, tuple(reversed(eids))):
            c2_s1, c2_s3 = _hit_state(eid2, state.c2, now)
            c3_s1, c3_s3 = _hit_state(eid3, state.c3, now)
            variants.add(
                _profile_with_residency(
                    token.physical,
                    c2_s1=c2_s1,
                    c2_s3=c2_s3,
                    c3_s1=c3_s1,
                    c3_s3=c3_s3,
                )
            )
    elif logical.family == "SPLIT" and len(eids) == 1:
        now = max(int(state.c2.task_end), int(state.c3.task_end))
        c2_s1, c2_s3 = _hit_state(eids[0], state.c2, now)
        c3_s1, c3_s3 = _hit_state(eids[0], state.c3, now)
        variants.add(
            _profile_with_residency(
                token.physical,
                c2_s1=c2_s1,
                c2_s3=c2_s3,
                c3_s1=c3_s1,
                c3_s3=c3_s3,
            )
        )
    elif logical.family == "SINGLE" and len(eids) == 1:
        cluster = _active_cluster(token.physical)
        if cluster is not None:
            own = state.c2 if cluster == 2 else state.c3
            peer = state.c3 if cluster == 2 else state.c2
            starts = {int(own.task_end), int(peer.task_end)}
            if peer.s2pf_end >= 0:
                starts.add(int(peer.s2pf_end))
            for start in starts:
                s1_hit, s3_hit = _hit_state(eids[0], own, start)
                variants.add(
                    _profile_with_residency(
                        token.physical,
                        c2_s1=s1_hit if cluster == 2 else False,
                        c2_s3=s3_hit if cluster == 2 else False,
                        c3_s1=s1_hit if cluster == 3 else False,
                        c3_s3=s3_hit if cluster == 3 else False,
                    )
                )
    if not variants:
        variants.add(token.physical)
    return tuple(
        CandidateProfile(logical=logical, physical=profile)
        for profile in sorted(variants)
    )


def runtime_profile_bank(
    state: reference.BeamState,
    profiles: Iterable[CandidateProfile],
) -> tuple[tuple[CandidateProfile, ...], tuple[int, ...]]:
    source_slots: dict[CandidateProfile, set[int]] = defaultdict(set)
    for profile_slot, token in enumerate(profiles):
        for variant in _runtime_variants(state, token):
            source_slots[variant].add(profile_slot)
    runtime_profiles = tuple(sorted(source_slots))
    fixed_priorities = tuple(
        min(source_slots[profile]) for profile in runtime_profiles
    )
    return runtime_profiles, fixed_priorities


def _source_s3_binding(
    final_dma: reference.DmaBinding,
    s2pf_dma: reference.DmaBinding,
) -> reference.DmaBinding:
    # A valid S2PF replaces the original S3 transfer.  The old source binding
    # is not future-observable, so use one canonical non-zero binding.
    return (
        reference.DmaBinding.IDMA
        if s2pf_dma != reference.DmaBinding.NONE
        else final_dma
    )


def _apply_s2pf(
    snap: reference.FourStageSnap,
    shape_s3: reference.Shape,
    binding: reference.DmaBinding,
) -> reference.FourStageSnap:
    if binding == reference.DmaBinding.NONE:
        return snap
    return snap.with_s2_down_prefetch(shape_s3, snap.dma1_end, binding)


def _pair_action(
    *,
    eid_a: int,
    ntok_a: int,
    shape_a_s1: reference.Shape,
    shape_a_s3: reference.Shape,
    start_a: int,
    s1_hit_a: bool,
    snap_a: reference.FourStageSnap,
    eid_b: int,
    ntok_b: int,
    shape_b_s1: reference.Shape,
    shape_b_s3: reference.Shape,
    start_b: int,
    s1_hit_b: bool,
    snap_b: reference.FourStageSnap,
    tag: str,
) -> reference.StageAction:
    return reference.StageAction(
        c2_eid=eid_a,
        c2_ntok=ntok_a,
        c2_shape_s1=shape_a_s1,
        c2_shape_s3=shape_a_s3,
        c2_start=start_a,
        c2_s1_cached=s1_hit_a,
        c2_s3_cached=snap_a.bw_s3 == 0,
        c3_eid=eid_b,
        c3_ntok=ntok_b,
        c3_shape_s1=shape_b_s1,
        c3_shape_s3=shape_b_s3,
        c3_start=start_b,
        c3_s1_cached=s1_hit_b,
        c3_s3_cached=snap_b.bw_s3 == 0,
        pf_cluster=-1,
        pf_eid=-1,
        pf_shape=None,
        pf_start=-1,
        tag=tag,
        c2_s2pf_start=snap_a.s2pf_start,
        c3_s2pf_start=snap_b.s2pf_start,
        c2_dma_s1=snap_a.dma_s1,
        c2_dma_s3=snap_a.dma_s3,
        c2_s2pf_dma=snap_a.s2pf_dma,
        c3_dma_s1=snap_b.dma_s1,
        c3_dma_s3=snap_b.dma_s3,
        c3_s2pf_dma=snap_b.s2pf_dma,
    )


def _single_action(
    *,
    cluster: int,
    eid: int,
    ntok: int,
    shape_s1: reference.Shape,
    shape_s3: reference.Shape,
    start: int,
    s1_hit: bool,
    snap: reference.FourStageSnap,
) -> reference.StageAction:
    common = dict(
        pf_cluster=-1,
        pf_eid=-1,
        pf_shape=None,
        pf_start=-1,
        tag=f"DIRECT-SINGLE-C{cluster}(E{eid})",
    )
    inactive = dict(
        c2_eid=-1,
        c2_ntok=0,
        c2_shape_s1=None,
        c2_shape_s3=None,
        c2_start=-1,
        c2_s1_cached=False,
        c2_s3_cached=False,
    )
    if cluster == 2:
        return reference.StageAction(
            c2_eid=eid,
            c2_ntok=ntok,
            c2_shape_s1=shape_s1,
            c2_shape_s3=shape_s3,
            c2_start=start,
            c2_s1_cached=s1_hit,
            c2_s3_cached=snap.bw_s3 == 0,
            c3_eid=-1,
            c3_ntok=0,
            c3_shape_s1=None,
            c3_shape_s3=None,
            c3_start=-1,
            c3_s1_cached=False,
            c3_s3_cached=False,
            c2_s2pf_start=snap.s2pf_start,
            c2_dma_s1=snap.dma_s1,
            c2_dma_s3=snap.dma_s3,
            c2_s2pf_dma=snap.s2pf_dma,
            **common,
        )
    if cluster == 3:
        return reference.StageAction(
            **inactive,
            c3_eid=eid,
            c3_ntok=ntok,
            c3_shape_s1=shape_s1,
            c3_shape_s3=shape_s3,
            c3_start=start,
            c3_s1_cached=s1_hit,
            c3_s3_cached=snap.bw_s3 == 0,
            c3_s2pf_start=snap.s2pf_start,
            c3_dma_s1=snap.dma_s1,
            c3_dma_s3=snap.dma_s3,
            c3_s2pf_dma=snap.s2pf_dma,
            **common,
        )
    raise ValueError(f"invalid cluster {cluster}")


def _single_cluster_is_legal(state: reference.BeamState, cluster: int) -> bool:
    t2, t3 = int(state.c2.task_end), int(state.c3.task_end)
    if t2 < t3:
        return cluster == 2 or (
            cluster == 3 and reference._reserved_next_eid(state.c3) >= 0
        )
    if t3 < t2:
        return cluster == 3 or (
            cluster == 2 and reference._reserved_next_eid(state.c2) >= 0
        )
    return cluster == 2 or (cluster == 3 and state.c2 != state.c3)


def _s4pf_action(
    *,
    cluster: int,
    target_eid: int,
    start: int,
    binding: reference.DmaBinding,
) -> reference.StageAction:
    """Build one hidden action targeting the next same-cluster expert.

    The action is inserted immediately before its consumer in the Python
    history.  An RTL pending record can hold the preceding task until this
    target EID is known, so no wildcard cache state is needed in the selected
    round-to-round policy state.
    """
    shape = (
        reference.SHAPE_C
        if binding == reference.DmaBinding.BOTH
        else reference.SHAPE_A
    )
    inactive = dict(
        c2_eid=-1,
        c2_ntok=0,
        c2_shape_s1=None,
        c2_shape_s3=None,
        c2_start=-1,
        c2_s1_cached=False,
        c2_s3_cached=False,
        c3_eid=-1,
        c3_ntok=0,
        c3_shape_s1=None,
        c3_shape_s3=None,
        c3_start=-1,
        c3_s1_cached=False,
        c3_s3_cached=False,
    )
    inactive[f"c{cluster}_eid"] = -2
    inactive[f"c{cluster}_shape_s1"] = shape
    inactive[f"c{cluster}_start"] = int(start)
    return reference.StageAction(
        **inactive,
        pf_cluster=int(cluster),
        pf_eid=int(target_eid),
        pf_shape=shape,
        pf_start=int(start),
        pf_dma=binding,
        tag=f"AUTO-S4PF-C{cluster}({binding.name})",
    )


def _materialize_one_profile(
    state: reference.BeamState,
    token: CandidateProfile,
) -> list[reference.StageAction]:
    logical, profile = token.logical, token.physical
    if logical.mode != mode(state):
        return []
    if logical.family not in {"PAIR", "SPLIT", "SINGLE"}:
        raise ValueError(f"unsupported final action family {logical.family!r}")
    if profile.s4pf_dma != "NONE":
        raise ValueError("the final scheduler has no standalone S4 prefetch")

    selected = tuple(resolve_selector(state, selector) for selector in logical.selectors)
    if any(eid is None for eid in selected):
        return []
    eids = tuple(int(eid) for eid in selected if eid is not None)
    ntok_by_eid = {int(eid): int(ntok) for eid, ntok in state.remaining}
    actions: dict[tuple, reference.StageAction] = {}

    if logical.family in {"PAIR", "SPLIT"}:
        if logical.family == "PAIR" and (len(eids) != 2 or eids[0] == eids[1]):
            return []
        if logical.family == "SPLIT" and len(eids) != 1:
            return []
        shape_a_s1 = _shape(profile.c2_s1)
        shape_a_s3 = _shape(profile.c2_s3)
        shape_b_s1 = _shape(profile.c3_s1)
        shape_b_s3 = _shape(profile.c3_s3)
        if None in (shape_a_s1, shape_a_s3, shape_b_s1, shape_b_s3):
            raise ValueError("PAIR/SPLIT profile must activate both clusters")
        dma_a_s1 = _dma(profile.c2_dma_s1)
        dma_b_s1 = _dma(profile.c3_dma_s1)
        dma_a_s2pf = _dma(profile.c2_s2pf)
        dma_b_s2pf = _dma(profile.c3_s2pf)
        dma_a_s3 = _source_s3_binding(_dma(profile.c2_dma_s3), dma_a_s2pf)
        dma_b_s3 = _source_s3_binding(_dma(profile.c3_dma_s3), dma_b_s2pf)
        now = max(int(state.c2.task_end), int(state.c3.task_end))

        if logical.family == "PAIR":
            assignments = (
                (eids[0], ntok_by_eid[eids[0]], eids[1], ntok_by_eid[eids[1]]),
                (eids[1], ntok_by_eid[eids[1]], eids[0], ntok_by_eid[eids[0]]),
            )
        else:
            eid = eids[0]
            ntok = ntok_by_eid[eid]
            if logical.split_rule == "HALF":
                if ntok % 2:
                    return []
                cut = ntok // 2
            elif logical.split_rule == "BALANCED":
                if ntok < 2:
                    return []
                cut = ntok // 2
            else:
                raise ValueError(f"unsupported final split rule {logical.split_rule!r}")
            assignments = ((eid, cut, eid, ntok - cut),)

        for eid_a, ntok_a, eid_b, ntok_b in assignments:
            if (
                reference._reserved_next_eid(state.c2) >= 0
                and reference._reserved_next_eid(state.c2) != eid_a
            ) or (
                reference._reserved_next_eid(state.c3) >= 0
                and reference._reserved_next_eid(state.c3) != eid_b
            ):
                continue
            s1_hit_a = reference._swiglu_hit_for_candidate(eid_a, state.c2, now)
            s1_hit_b = reference._swiglu_hit_for_candidate(eid_b, state.c3, now)
            s3_hit_a = reference._down_hit_for_candidate(eid_a, state.c2, now)
            s3_hit_b = reference._down_hit_for_candidate(eid_b, state.c3, now)
            if s1_hit_a != profile.c2_s1_cached or s1_hit_b != profile.c3_s1_cached:
                continue
            raw_a = reference.FourStageSnap.from_assign(
                now,
                shape_a_s1,
                shape_a_s3,
                ntok_a,
                eid_a,
                s1_hit_a,
                s3_hit_a,
                dma_s1=dma_a_s1,
                dma_s3=dma_a_s3,
            )
            raw_b = reference.FourStageSnap.from_assign(
                now,
                shape_b_s1,
                shape_b_s3,
                ntok_b,
                eid_b,
                s1_hit_b,
                s3_hit_b,
                dma_s1=dma_b_s1,
                dma_s3=dma_b_s3,
            )
            snap_a = _apply_s2pf(raw_a, shape_a_s3, dma_a_s2pf)
            snap_b = _apply_s2pf(raw_b, shape_b_s3, dma_b_s2pf)
            if not reference.bw_feasible(snap_a, snap_b):
                continue
            action = _pair_action(
                eid_a=eid_a,
                ntok_a=ntok_a,
                shape_a_s1=shape_a_s1,
                shape_a_s3=shape_a_s3,
                start_a=now,
                s1_hit_a=s1_hit_a,
                snap_a=snap_a,
                eid_b=eid_b,
                ntok_b=ntok_b,
                shape_b_s1=shape_b_s1,
                shape_b_s3=shape_b_s3,
                start_b=now,
                s1_hit_b=s1_hit_b,
                snap_b=snap_b,
                tag=(
                    f"DIRECT-SPLIT(E{eid_a}:{ntok_a},{ntok_b})"
                    if logical.family == "SPLIT"
                    else f"DIRECT-PAIR({eid_a}+{eid_b})"
                ),
            )
            if physical_profile(action) == profile:
                child = reference.apply_action(state, action)
                actions.setdefault(child_key(child), action)
        return list(actions.values())

    if len(eids) != 1:
        return []
    eid = eids[0]
    cluster_id = _active_cluster(profile)
    if cluster_id is None:
        raise ValueError("SINGLE profile must activate exactly one cluster")
    if not _single_cluster_is_legal(state, cluster_id):
        return []
    cluster = state.c2 if cluster_id == 2 else state.c3
    peer = state.c3 if cluster_id == 2 else state.c2
    cluster_reserved = reference._reserved_next_eid(cluster)
    peer_reserved = reference._reserved_next_eid(peer)
    if (cluster_reserved >= 0 and cluster_reserved != eid) or peer_reserved == eid:
        return []
    shape_s1 = _shape(profile.c2_s1 if cluster_id == 2 else profile.c3_s1)
    shape_s3 = _shape(profile.c2_s3 if cluster_id == 2 else profile.c3_s3)
    if shape_s1 is None or shape_s3 is None:
        raise ValueError("SINGLE profile has an inactive active-side shape")
    dma_s1 = _dma(profile.c2_dma_s1 if cluster_id == 2 else profile.c3_dma_s1)
    final_dma_s3 = _dma(
        profile.c2_dma_s3 if cluster_id == 2 else profile.c3_dma_s3
    )
    s2pf_dma = _dma(profile.c2_s2pf if cluster_id == 2 else profile.c3_s2pf)
    expected_s1_hit = (
        profile.c2_s1_cached if cluster_id == 2 else profile.c3_s1_cached
    )
    ntok = ntok_by_eid[eid]
    cluster_end = int(cluster.task_end)
    s1_hit = reference._swiglu_hit_for_candidate(eid, cluster, cluster_end)
    s3_hit = reference._down_hit_for_candidate(eid, cluster, cluster_end)
    if s1_hit != expected_s1_hit:
        return []

    discovery_bindings = (
        reference.DMA_BINDINGS
        if s2pf_dma != reference.DmaBinding.NONE
        else (final_dma_s3,)
    )
    starts = set()
    for discovery_dma_s3 in discovery_bindings:
        starts.update(
            reference._start_candidates(
                cluster_end,
                cluster,
                peer,
                ntok,
                shape_s1,
                shape_s3,
                dma_s1,
                discovery_dma_s3,
                s1_hit,
                s3_hit,
            )
        )
    source_dma_s3 = _source_s3_binding(final_dma_s3, s2pf_dma)
    for start in sorted(starts):
        raw = reference.FourStageSnap.from_assign(
            start,
            shape_s1,
            shape_s3,
            ntok,
            eid,
            s1_hit,
            s3_hit,
            dma_s1=dma_s1,
            dma_s3=source_dma_s3,
        )
        snap = _apply_s2pf(raw, shape_s3, s2pf_dma)
        action = _single_action(
            cluster=cluster_id,
            eid=eid,
            ntok=ntok,
            shape_s1=shape_s1,
            shape_s3=shape_s3,
            start=start,
            s1_hit=s1_hit,
            snap=snap,
        )
        if physical_profile(action) != profile:
            continue
        feasible = (
            reference.bw_feasible(snap, peer)
            if cluster_id == 2
            else reference.bw_feasible(peer, snap)
        )
        if feasible:
            child = reference.apply_action(state, action)
            actions.setdefault(child_key(child), action)
    return list(actions.values())


def materialize_targeted_s4pf_variant(
    state: reference.BeamState,
    action: reference.StageAction,
) -> tuple[
    reference.StageAction,
    reference.BeamState,
    tuple[reference.StageAction, ...],
] | None:
    """Lower S4PF jointly with the concrete action that consumes it.

    The no-S4PF action remains available to the caller.  This helper tries the
    fixed local SINGLE lane first and BOTH second for each active cluster, in
    C2-then-C3 order.  A targeted prefetch is retained only after the resulting
    cached physical profile can be materialized and its complete child state is
    known.  Consequently no prefetch DMA interval survives into an unknown
    future round.
    """
    if len(state.remaining) < S4PF_MIN_REMAINING:
        return None

    augmented = state
    prefetch_actions: list[reference.StageAction] = []
    prefetched_clusters: set[int] = set()

    for cluster in (2, 3):
        eid = int(getattr(action, f"c{cluster}_eid"))
        if eid < 0 or bool(getattr(action, f"c{cluster}_s1_cached")):
            continue
        own = augmented.c2 if cluster == 2 else augmented.c3
        peer = augmented.c3 if cluster == 2 else augmented.c2
        if own.cur_eid < 0 or own.pf_eid != -1:
            continue
        start = int(own.dma3_end)
        # If the peer has already advanced past this point, its replaced
        # snapshot no longer carries every DMA interval that could overlap a
        # retroactive prefetch.  Such a trial cannot be certified from the
        # bounded state and must remain OFF.
        if peer.cur_eid >= 0 and start < int(peer.task_start):
            continue
        compute_end = int(
            own.compute_end if own.compute_end >= 0 else own.task_end
        )
        for binding in (
            _single_dma_for_cluster(cluster),
            reference.DmaBinding.BOTH,
        ):
            shape = (
                reference.SHAPE_C
                if binding == reference.DmaBinding.BOTH
                else reference.SHAPE_A
            )
            trial = own.with_prefetch(eid, shape, start, binding)
            if int(trial.pf_end) > compute_end:
                continue
            feasible = (
                reference.bw_feasible(trial, peer)
                if cluster == 2
                else reference.bw_feasible(peer, trial)
            )
            if not feasible:
                continue
            prefetch = _s4pf_action(
                cluster=cluster,
                target_eid=eid,
                start=start,
                binding=binding,
            )
            augmented = reference.apply_action(augmented, prefetch)
            prefetch_actions.append(prefetch)
            prefetched_clusters.add(cluster)
            break

    if not prefetch_actions:
        return None

    base_profile = physical_profile(action)
    cached_profile = _profile_with_residency(
        base_profile,
        c2_s1=2 in prefetched_clusters,
        c2_s3=bool(action.c2_s3_cached),
        c3_s1=3 in prefetched_clusters,
        c3_s3=bool(action.c3_s3_cached),
    )
    token = CandidateProfile(
        logical=logical_action_spec(state, action, WINDOW),
        physical=cached_profile,
    )
    candidates = _materialize_one_profile(augmented, token)
    if not candidates:
        return None

    def transition_key(candidate: reference.StageAction) -> tuple[int, int, int]:
        child = reference.apply_action(augmented, candidate)
        starts = [
            int(start)
            for eid, start in (
                (candidate.c2_eid, candidate.c2_start),
                (candidate.c3_eid, candidate.c3_start),
            )
            if eid >= 0
        ]
        ends = (int(child.c2.task_end), int(child.c3.task_end))
        return max(ends), sum(ends), max(starts, default=0)

    selected = min(candidates, key=transition_key)
    child = reference.apply_action(augmented, selected)
    return selected, child, tuple(prefetch_actions)


def materialize_candidates_with_sources(
    state: reference.BeamState,
    profiles: tuple[CandidateProfile, ...],
) -> tuple[list[tuple[reference.StageAction, tuple[int, ...]]], dict[str, int]]:
    """Lower profiles, retain their earliest-finish action and deduplicate children."""
    emitted: dict[tuple, reference.StageAction] = {}
    source_indices: dict[tuple, set[int]] = {}
    valid_profiles = 0
    concrete_before_dedup = 0

    def earliest_finish(action: reference.StageAction) -> tuple[int, int, int]:
        child = reference.apply_action(state, action)
        starts = [
            int(start)
            for eid, start in (
                (action.c2_eid, action.c2_start),
                (action.c3_eid, action.c3_start),
            )
            if eid >= 0
        ]
        ends = (int(child.c2.task_end), int(child.c3.task_end))
        return max(ends), sum(ends), max(starts, default=0)

    for profile_index, profile in enumerate(profiles):
        actions = _materialize_one_profile(state, profile)
        valid_profiles += bool(actions)
        concrete_before_dedup += len(actions)
        if actions:
            actions = [min(actions, key=earliest_finish)]
        for action in actions:
            key = child_key(reference.apply_action(state, action))
            emitted.setdefault(key, action)
            source_indices.setdefault(key, set()).add(profile_index)

    return [
        (action, tuple(sorted(source_indices[key])))
        for key, action in emitted.items()
    ], {
        "valid_profiles": valid_profiles,
        "concrete_before_dedup": concrete_before_dedup,
        "physical_candidates": len(emitted),
    }
