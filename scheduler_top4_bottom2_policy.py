#!/usr/bin/env python3
"""Bounded top4+bottom2 policy derived from the HW-v2 Python mirror.

This is a hardware-policy model, not an RTL implementation.  It preserves the
integer-tick four-stage cost and DMA-conflict rules used by
``scheduler_rtl_adaptive_prefetch_policy`` and changes only the bounded
candidate/selection policy:

* both idle: retain the HW-v2 top-four candidate bank and add one hot+medium
  candidate, selected by a window-only skew test;
* one idle: try top0, top1, bottom0, and bottom1 from the visible window;
* one expert left: retain the HW-v2 solo-versus-split terminal comparison.

No arbitrary expert-rank scan, child rollout, beam search, or learned model is
performed at runtime.  Every non-terminal selected expert is visible in the
semantic top4+bottom2 window.
"""

from __future__ import annotations

from dataclasses import dataclass
import scheduler_hw_fixed_policy as base
import scheduler_rtl_adaptive_prefetch_policy as adaptive


TICK_CC = adaptive.TICK_CC
DEFAULT_S4_POLICY = "single_first"
PolicyState = base.PolicyState
Transition = base.Transition


@dataclass(frozen=True)
class PolicyStep:
    index: int
    tag: str
    decision: str
    source: str
    selected: tuple[tuple[int, int], ...]
    visible_before: tuple[tuple[int, int], ...]
    tail_mode_before: str
    tail_mode_after: str
    c2_end_cc: int
    c3_end_cc: int


@dataclass(frozen=True)
class Top4Bottom2Result:
    makespan_cc: int
    steps: tuple[PolicyStep, ...]


def _remove_eids(
    remaining: tuple[tuple[int, int], ...], *eids: int
) -> tuple[tuple[int, int], ...]:
    removed = set(eids)
    return tuple(item for item in remaining if item[0] not in removed)


def visible_window(
    remaining: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, int], ...]:
    """Return semantic top4+bottom2 entries without duplicates."""
    indices: list[int] = []
    for index in (0, 1, 2, 3, len(remaining) - 2, len(remaining) - 1):
        if 0 <= index < len(remaining) and index not in indices:
            indices.append(index)
    return tuple(remaining[index] for index in indices)


def _source_indices(remaining: tuple[tuple[int, int], ...]) -> tuple[tuple[str, int], ...]:
    candidates = (
        ("top0", 0),
        ("top1", 1),
        ("bottom0", len(remaining) - 1),
        ("bottom1", len(remaining) - 2),
    )
    result: list[tuple[str, int]] = []
    seen: set[int] = set()
    for name, index in candidates:
        if 0 <= index < len(remaining) and index not in seen:
            seen.add(index)
            result.append((name, index))
    return tuple(result)


def _skew_branch_enabled(state: PolicyState) -> bool:
    """Window-only trigger for running a hot expert beside a medium expert.

    The branch requires two genuinely hot heads, a distinct medium entry, and
    a cold tail.  All comparisons are integer shifts/comparators in RTL:

      top0 >= 3*top2, top1 >= 3*top2, top2 >= 2*bottom0.

    ``top2`` is used instead of top1 so one hot expert remains available for a
    later balanced split.
    """
    remaining = state.remaining
    if len(remaining) < 4:
        return False
    top0 = remaining[0][1]
    top1 = remaining[1][1]
    top2 = remaining[2][1]
    bottom0 = remaining[-1][1]
    return (
        top0 >= 3 * top2
        and top1 >= 3 * top2
        and top0 <= 5 * top2
        and top1 <= 5 * top2
        and top2 >= 2 * bottom0
        and bottom0 == 2
    )


def _hot_medium_transition(state: PolicyState, cost_model) -> Transition | None:
    """Construct the one added both-idle candidate: top0 + top2.

    Both tasks use the single-lane-friendly B/B shape.  S2 prefetch is tried
    only for the hot side, at the end of its S1 DMA.  This deliberately leaves
    the medium side's DMA lane free as early as possible for cold-tail issue.
    """
    with adaptive._use_cost_model(cost_model):
        c2, c3 = base._prepare(state.c2, state.c3)
        if len(state.remaining) < 3 or c2.task_end != c3.task_end:
            return None

        hot_eid, hot_ntok = state.remaining[0]
        medium_eid, medium_ntok = state.remaining[2]
        now = c2.task_end

        hot = cost_model._cc_mk_snap(
            now,
            cost_model.C_SHAPE_B,
            cost_model.C_SHAPE_B,
            hot_ntok,
            hot_eid,
            cost_model._cc_swiglu_hit(hot_eid, c2, now),
            cost_model._cc_down_hit(hot_eid, c2, now),
        )
        medium = cost_model._cc_mk_snap(
            now,
            cost_model.C_SHAPE_B,
            cost_model.C_SHAPE_B,
            medium_ntok,
            medium_eid,
            cost_model._cc_swiglu_hit(medium_eid, c3, now),
            cost_model._cc_down_hit(medium_eid, c3, now),
        )

        used_s2pf = False
        if hot.bw_s3 > 0:
            prefetched = cost_model._cc_apply_s2pf(
                hot, cost_model.C_SHAPE_B, hot.dma1_end
            )
            if prefetched.s2pf_start >= 0 and cost_model._cc_bw_ok(prefetched, medium):
                hot = prefetched
                used_s2pf = True
        if not cost_model._cc_bw_ok(hot, medium):
            return None

        tag = "hot_medium_0_2_s2pf" if used_s2pf else "hot_medium_0_2_raw"
        return Transition(
            PolicyState(
                hot,
                medium,
                _remove_eids(state.remaining, hot_eid, medium_eid),
            ),
            tag,
        )


def _top0_has_cache_hit(state: PolicyState, cost_model) -> bool:
    if not state.remaining or state.c2.task_end != state.c3.task_end:
        return False
    eid = state.remaining[0][0]
    now = state.c2.task_end
    with adaptive._use_cost_model(cost_model):
        return any(
            cost_model._cc_swiglu_hit(eid, snap, now)
            or cost_model._cc_down_hit(eid, snap, now)
            for snap in (state.c2, state.c3)
        )


def _reorder_selected_first(
    remaining: tuple[tuple[int, int], ...], index: int
) -> tuple[tuple[int, int], ...]:
    return (remaining[index],) + tuple(
        item for current, item in enumerate(remaining) if current != index
    )


def _one_idle_candidates(
    state: PolicyState, cost_model
) -> list[tuple[int, str, Transition]]:
    candidates: list[tuple[int, str, Transition]] = []
    seen_states: set[tuple] = set()
    for priority, (source, index) in enumerate(_source_indices(state.remaining)):
        trial_state = PolicyState(
            state.c2,
            state.c3,
            _reorder_selected_first(state.remaining, index),
        )
        with adaptive._use_cost_model(cost_model):
            transitions = base.generate_one_idle_shape_successors(
                trial_state,
                policy="balanced",
                top_policy="pruned",
                n1_policy="pruned",
            )
            # Explicit S4PF-OFF alternative.  Calling the one-idle primitive
            # directly avoids ``_prepare`` adding a ghost S4 prefetch before a
            # cold-tail task.  OFF is a real bounded control choice, not a
            # different timing model.
            s4off_transitions = base._one_idle_successors(
                trial_state.c2,
                trial_state.c3,
                trial_state.remaining,
            )
        tagged_transitions = [
            ("s4auto", transition) for transition in transitions
        ] + [
            ("s4off", transition) for transition in s4off_transitions
        ]
        for s4_mode, transition in tagged_transitions:
            fingerprint = base.state_key(transition.state)
            if fingerprint in seen_states:
                continue
            seen_states.add(fingerprint)
            candidates.append(
                (
                    priority,
                    source,
                    Transition(
                        transition.state,
                        f"{source}__{s4_mode}__{transition.tag}",
                    ),
                )
            )
    return candidates


def _cold_tail_drain_enabled(state: PolicyState) -> bool:
    """Allow bottom issue only while a long hot task can hide a cold tail.

    Without this guard, an immediate-finish comparison can consume a cold
    expert in medium/flat regimes and damage the later pairing structure.  The
    condition is intentionally limited to visible entries and integer shifts.
    """
    remaining = state.remaining
    if len(remaining) < 2:
        return False
    top0 = remaining[0][1]
    top1 = remaining[1][1]
    bottom0 = remaining[-1][1]
    if bottom0 != 2:
        return False
    busy_end = max(state.c2.task_end, state.c3.task_end)
    idle_end = min(state.c2.task_end, state.c3.task_end)
    busy = state.c2 if state.c2.task_end > state.c3.task_end else state.c3
    hierarchy = top0 >= 2 * top1 and top1 >= 2 * bottom0
    anchored_uniform_tail = top0 == bottom0 and busy.ntok >= 4 * top0
    anchored_medium_tail = (
        busy.ntok >= 3 * top0
        and busy.ntok <= 4 * top0
        and top0 >= 2 * bottom0
        and len(remaining) > 5
    )
    return (
        (hierarchy or anchored_uniform_tail or anchored_medium_tail)
        and busy_end - idle_end >= 3 * TICK_CC
    )


def _base_both_idle_candidates(
    state: PolicyState, cost_model
) -> list[Transition]:
    with adaptive._use_cost_model(cost_model):
        return base.generate_one_idle_shape_successors(
            state,
            policy="balanced",
            top_policy="pruned",
            n1_policy="pruned",
        )


def _find_tag(transitions: list[Transition], tag: str) -> Transition | None:
    return next((transition for transition in transitions if transition.tag == tag), None)


def _split_then_anchor_enabled(state: PolicyState, cost_model) -> bool:
    """Detect a two-hot + uniform-cold case with an exact anchor packing.

    After splitting top0 across both clusters, top1 can occupy one cluster and
    the entire uniform tail can be serialized on the other.  The test compares
    bounded integer task times; it does not roll out child states.
    """
    remaining = state.remaining
    if (
        len(remaining) < 4
        or remaining[0][1] < 2
        or remaining[0][1] % 2 != 0
    ):
        return False
    # The list is sorted.  Equality of top2 and bottom0 therefore proves that
    # every intervening tail entry has the same token count; no hidden rank is
    # inspected by the decision logic.
    tail_ntok = remaining[2][1]
    if tail_ntok != remaining[-1][1] or tail_ntok != 2:
        return False
    top0 = remaining[0][1]
    top1 = remaining[1][1]
    if top0 != top1:
        return False
    with adaptive._use_cost_model(cost_model):
        tail_service = (len(remaining) - 2) * cost_model._cc_best_task(tail_ntok)
        anchor_service = cost_model._cc_best_task(top1)
    return tail_service == anchor_service


def _dominant_head_enabled(state: PolicyState) -> bool:
    remaining = state.remaining
    if len(remaining) < 3:
        return False
    top0 = remaining[0][1]
    top1 = remaining[1][1]
    bottom0 = remaining[-1][1]
    if bottom0 != 2:
        return False
    # Sorted order makes top1==bottom0 a proof that the entire suffix is flat.
    uniform_after_head = top1 == bottom0
    return (
        uniform_after_head and top0 >= 3 * top1
    ) or (
        top0 == 3 * top1 and top1 >= 2 * bottom0
    )


def _medium_band_front_split_enabled(state: PolicyState) -> bool:
    """Small active-set pattern whose exact tail needs a front-2 split."""
    remaining = state.remaining
    if len(remaining) < 6 or len(remaining) > 10:
        return False
    top0, top1, top2, top3 = (remaining[index][1] for index in range(4))
    return (
        remaining[-1][1] == 2
        and top0 == top1
        and top0 == 2 * top2
        and top2 == top3
        and top2 >= 2 * remaining[-1][1]
    )


def _remaining_block_parity(state: PolicyState) -> int:
    """One aggregate bit: parity of sum(ceil(ntok/2))."""
    return sum((ntok + 1) // 2 for _eid, ntok in state.remaining) & 1


def _choose_with_metadata(
    state: PolicyState,
    cost_model,
    *,
    enhancements_enabled: bool = True,
    tail_mode: str = "off",
) -> tuple[Transition, str, str]:
    if len(state.remaining) == 1:
        with adaptive._use_cost_model(cost_model):
            transitions = base.generate_one_idle_shape_successors(
                state,
                policy="balanced",
                top_policy="pruned",
                n1_policy="pruned",
            )
        chosen = min(
            transitions,
            key=lambda transition: (
                max(transition.state.c2.task_end, transition.state.c3.task_end),
                transition.tag,
            ),
        )
        return chosen, "terminal_exact", "last"

    if state.c2.task_end != state.c3.task_end:
        if not enhancements_enabled:
            chosen = adaptive._choose_transition(state, cost_model)
            return chosen, "warm_cache_hw_v2", "top0"
        if tail_mode == "interleave_top":
            candidates = _one_idle_candidates(state, cost_model)
            adaptive_top0 = [
                transition
                for _priority, source, transition in candidates
                if source == "top0"
                and "__s4auto__" in transition.tag
                and "one_idle_adaptive" in transition.tag
                and transition.tag.endswith("p0")
            ]
            if adaptive_top0:
                chosen = min(
                    adaptive_top0,
                    key=lambda transition: max(
                        transition.state.c2.task_end,
                        transition.state.c3.task_end,
                    ),
                )
                return chosen, "front_interleave_top", "top0"
            chosen = adaptive._choose_transition(state, cost_model)
            return chosen, "front_interleave_top", "top0_fallback"
        front_drain_ready = (
            tail_mode in {
                "front_drain",
                "interleave_bottom",
                "interleave_bottom_last",
            }
            and state.remaining[-1][1] == 2
            and abs(state.c2.task_end - state.c3.task_end) >= 3 * TICK_CC
        )
        if (
            tail_mode not in {
                "drain",
                "front_drain",
                "interleave_bottom",
                "interleave_bottom_last",
            }
            or not (front_drain_ready or _cold_tail_drain_enabled(state))
        ):
            chosen = adaptive._choose_transition(state, cost_model)
            return chosen, "hw_v2_one_idle", "top0"
        candidates = _one_idle_candidates(state, cost_model)
        if not candidates:
            raise RuntimeError("top4+bottom2 one-idle candidate bank is empty")
        idle_cluster = 0 if state.c2.task_end < state.c3.task_end else 1

        def candidate_key(item: tuple[int, str, Transition]) -> tuple[int, int, int, int, str]:
            priority, _source, transition = item
            child = transition.state
            issued_end = child.c2.task_end if idle_cluster == 0 else child.c3.task_end
            return (
                max(child.c2.task_end, child.c3.task_end),
                issued_end,
                0 if "__s4off__" in transition.tag else 1,
                priority,
                transition.tag,
            )

        _priority, source, chosen = min(candidates, key=candidate_key)
        if tail_mode == "interleave_bottom":
            return chosen, "front_interleave_bottom", source
        if tail_mode == "interleave_bottom_last":
            return chosen, "front_interleave_bottom_last", source
        return chosen, "one_idle_min_finish", source

    if not enhancements_enabled:
        chosen = adaptive._choose_transition(state, cost_model)
        return chosen, "warm_cache_hw_v2", "top4"

    base_candidates: list[Transition] | None = None
    if tail_mode == "anchor_pair":
        base_candidates = _base_both_idle_candidates(state, cost_model)
        transition = _find_tag(base_candidates, "pair_0_1")
        if transition is not None:
            return transition, "anchor_pair_issue", "top0+top1"

    if tail_mode == "need_front_split":
        base_candidates = _base_both_idle_candidates(state, cost_model)
        transition = _find_tag(base_candidates, "split_0_2")
        if transition is not None:
            return transition, "anchor_front_split", "top0_split2"

    if tail_mode == "rejoin":
        if len(state.remaining) >= 3 and state.remaining[-1][1] == 2:
            transition = _hot_medium_transition(state, cost_model)
            if transition is not None:
                return transition, "front_rejoin_hot_medium", "top0+top2"
        chosen = adaptive._choose_transition(state, cost_model)
        return chosen, "front_rejoin_hw_v2", "top4"

    if _medium_band_front_split_enabled(state):
        base_candidates = _base_both_idle_candidates(state, cost_model)
        transition = _find_tag(base_candidates, "split_0_2")
        if transition is not None:
            return transition, "medium_band_front_split", "top0_split2"

    if _split_then_anchor_enabled(state, cost_model):
        base_candidates = _base_both_idle_candidates(state, cost_model)
        top0_ntok = state.remaining[0][1]
        split_tag = f"split_0_{top0_ntok // 2}"
        transition = _find_tag(base_candidates, split_tag)
        if transition is not None:
            return transition, "split_then_anchor", "top0_split"

    if _dominant_head_enabled(state):
        if base_candidates is None:
            base_candidates = _base_both_idle_candidates(state, cost_model)
        transition = _find_tag(base_candidates, "pair_0_1")
        if transition is not None:
            return transition, "dominant_head_anchor", "top0+top1"

    if _skew_branch_enabled(state) and not _top0_has_cache_hit(state, cost_model):
        if len(state.remaining) <= 12 and _remaining_block_parity(state) == 0:
            if base_candidates is None:
                base_candidates = _base_both_idle_candidates(state, cost_model)
            split_tag = f"split_0_{state.remaining[0][1] // 2}"
            transition = _find_tag(base_candidates, split_tag)
            if transition is not None:
                return transition, "even_skew_split", "top0_split"
        transition = _hot_medium_transition(state, cost_model)
        if transition is not None:
            return transition, "skew_hot_medium", "top0+top2"

    chosen = adaptive._choose_transition(state, cost_model)
    return chosen, "hw_v2_score", "top4"


def choose_transition(
    state: PolicyState,
    *,
    s4_policy: str = DEFAULT_S4_POLICY,
    enhancements_enabled: bool = True,
    tail_mode: str = "off",
) -> Transition:
    cost_model = adaptive._COST_MODELS[s4_policy]
    transition, _decision, _source = _choose_with_metadata(
        state,
        cost_model,
        enhancements_enabled=enhancements_enabled,
        tail_mode=tail_mode,
    )
    return transition


def _next_tail_mode(
    tail_mode: str,
    decision: str,
    after: PolicyState,
) -> str:
    """Advance the bounded controller mode after one selected transition."""
    if decision in {"skew_hot_medium", "dominant_head_anchor"}:
        return "drain"
    if decision == "split_then_anchor":
        return "anchor_pair"
    if decision == "even_skew_split":
        return "need_front_split"
    if decision in {"medium_band_front_split", "anchor_front_split"}:
        return "front_drain"
    if decision == "anchor_pair_issue":
        return "drain"
    if decision == "front_rejoin_hot_medium":
        return "interleave_bottom"
    if decision == "front_interleave_bottom":
        return "interleave_top"
    if decision == "front_interleave_top":
        return "interleave_bottom_last"
    if decision == "front_interleave_bottom_last":
        return "off"
    if tail_mode == "drain" and after.c2.task_end == after.c3.task_end:
        return "off"
    if tail_mode == "front_drain" and after.c2.task_end == after.c3.task_end:
        return "rejoin"
    if decision == "front_rejoin_hw_v2":
        return "off"
    return tail_mode


def schedule_result(
    token_dist: dict[int, int],
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
    *,
    s4_policy: str = DEFAULT_S4_POLICY,
) -> Top4Bottom2Result:
    cost_model = adaptive._COST_MODELS[s4_policy]
    with adaptive._use_cost_model(cost_model):
        state = base.initial_state(token_dist, initial_cache_c2, initial_cache_c3)
    enhancements_enabled = initial_cache_c2 < 0 and initial_cache_c3 < 0
    tail_mode = "off"

    steps: list[PolicyStep] = []
    while state.remaining:
        before = state
        visible = visible_window(before.remaining)
        transition, decision, source = _choose_with_metadata(
            before,
            cost_model,
            enhancements_enabled=enhancements_enabled,
            tail_mode=tail_mode,
        )
        after = transition.state
        tail_mode_before = tail_mode
        tail_mode = _next_tail_mode(tail_mode, decision, after)
        remaining_eids = {eid for eid, _ in after.remaining}
        selected = tuple(
            item for item in before.remaining if item[0] not in remaining_eids
        )
        if not selected:
            raise RuntimeError(f"transition {transition.tag} consumed no expert")
        visible_eids = {eid for eid, _ in visible}
        if len(before.remaining) > 1 and any(
            eid not in visible_eids for eid, _ in selected
        ):
            raise RuntimeError(
                f"transition {transition.tag} selected an expert outside top4+bottom2"
            )
        steps.append(
            PolicyStep(
                index=len(steps),
                tag=transition.tag,
                decision=decision,
                source=source,
                selected=selected,
                visible_before=visible,
                tail_mode_before=tail_mode_before,
                tail_mode_after=tail_mode,
                c2_end_cc=after.c2.task_end,
                c3_end_cc=after.c3.task_end,
            )
        )
        state = after

    return Top4Bottom2Result(base.terminal_cost(state), tuple(steps))


def schedule(
    token_dist: dict[int, int],
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
    *,
    s4_policy: str = DEFAULT_S4_POLICY,
) -> int:
    return schedule_result(
        token_dist,
        initial_cache_c2,
        initial_cache_c3,
        s4_policy=s4_policy,
    ).makespan_cc


def ticks(cc: int) -> int:
    return (cc + TICK_CC - 1) // TICK_CC
