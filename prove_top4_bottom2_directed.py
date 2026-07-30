#!/usr/bin/env python3
"""Build and verify exact four-stage certificates for directed distributions.

The bounded hardware mirror is not itself an optimality proof.  This tool uses
its expert/cluster decisions only as a lowering hint, reconstructs a concrete
history in the independent explicit-DMA-lane four-stage model, replays that
history with the reference validator, and then runs admissible branch-and-bound
when equality with the root lower bound does not already close the proof.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import json
import os
from pathlib import Path
import time

import evaluate_top4_bottom2_directed as directed
import four_stage_scheduler as reference
from run_four_stage_reference import deserialize_action, serialize_action
import scheduler_hw_fixed_policy as hw_base
import scheduler_rtl_adaptive_prefetch_policy as adaptive
import scheduler_top4_bottom2_policy as policy
import construct_olmoe_block_schedules as block_constructor


TICK_CC = policy.TICK_CC


@dataclass(frozen=True)
class MirrorTransition:
    before: hw_base.PolicyState
    after: hw_base.PolicyState
    tag: str
    decision: str
    selected: tuple[tuple[int, int], ...]


def _ticks(cc: int) -> str:
    value = Fraction(cc, TICK_CC)
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _mirror_trace(token_dist: dict[int, int]) -> tuple[MirrorTransition, ...]:
    cost_model = adaptive._COST_MODELS[policy.DEFAULT_S4_POLICY]
    with adaptive._use_cost_model(cost_model):
        state = hw_base.initial_state(token_dist)
    tail_mode = "off"
    trace: list[MirrorTransition] = []
    while state.remaining:
        before = state
        transition, decision, _source = policy._choose_with_metadata(
            before,
            cost_model,
            enhancements_enabled=True,
            tail_mode=tail_mode,
        )
        after = transition.state
        remaining_eids = {eid for eid, _ in after.remaining}
        selected = tuple(
            item for item in before.remaining if item[0] not in remaining_eids
        )
        if not selected:
            raise RuntimeError(f"mirror transition {transition.tag} consumed nothing")
        trace.append(
            MirrorTransition(before, after, transition.tag, decision, selected)
        )
        tail_mode = policy._next_tail_mode(tail_mode, decision, after)
        state = after
    return tuple(trace)


# Internal compute boundaries can differ between exactly future-equivalent
# shape representatives.  These are the observables that affect later issue,
# DMA conflicts, S2PF, and final completion.
_LOWERING_OBSERVABLES = (
    "task_start",
    "task_end",
    "dma1_end",
    "s2_end",
    "dma3_end",
    "bw_s1",
    "bw_s3",
    "s2pf_start",
    "s2pf_end",
    "s2pf_bw",
    "ntok",
)


def _selected_on_cluster(
    mirror: MirrorTransition, cluster: int
) -> bool:
    selected_eids = {eid for eid, _ in mirror.selected}
    return getattr(mirror.after, f"c{cluster}").cur_eid in selected_eids


def _matching_action_candidates(
    state: reference.BeamState,
    mirror: MirrorTransition,
    *,
    seed_mode: bool,
) -> list[tuple[int, int, int, reference.StageAction, reference.BeamState]]:
    remaining_by_eid = dict(state.remaining)
    selected_counts = Counter(ntok for _eid, ntok in mirror.selected)
    actions = reference.gen_stage_actions(
        state.c2,
        state.c3,
        state.remaining,
        seed_mode=seed_mode,
    )
    candidates = []
    for action in actions:
        consumed_eids = {
            eid for eid in (action.c2_eid, action.c3_eid) if eid >= 0
        }
        if Counter(remaining_by_eid[eid] for eid in consumed_eids) != selected_counts:
            continue

        matches_clusters = True
        for cluster in (2, 3):
            action_eid = getattr(action, f"c{cluster}_eid")
            desired = _selected_on_cluster(mirror, cluster)
            if desired != (action_eid >= 0):
                matches_clusters = False
                break
            if desired:
                desired_ntok = getattr(mirror.after, f"c{cluster}").ntok
                if getattr(action, f"c{cluster}_ntok") != desired_ntok:
                    matches_clusters = False
                    break
        if not matches_clusters:
            continue

        child = reference.apply_action(state, action)
        mismatch = 0
        for cluster in (2, 3):
            if not _selected_on_cluster(mirror, cluster):
                continue
            target = getattr(mirror.after, f"c{cluster}")
            actual = getattr(child, f"c{cluster}")
            mismatch += sum(
                abs(getattr(actual, field) - getattr(target, field))
                for field in _LOWERING_OBSERVABLES
            )
        candidates.append(
            (mismatch, child.g_score, child.f_score, action, child)
        )
    return candidates


def _required_prefetch_target_count(
    mirror: MirrorTransition, cluster: int
) -> int | None:
    """Return the full expert count when the mirror requires an S1 hit."""
    target = getattr(mirror.after, f"c{cluster}")
    if not _selected_on_cluster(mirror, cluster) or target.bw_s1 != 0:
        return None
    return dict(mirror.before.remaining)[target.cur_eid]


def _insert_required_prefetches(
    state: reference.BeamState,
    mirror: MirrorTransition,
    *,
    clusters: tuple[int, ...] = (2, 3),
) -> tuple[reference.BeamState, list[dict]]:
    """Materialize only cache hits that have a legal explicit PF history.

    The mirror's ghost S4PF may be attached after its physical start time.  The
    reference generator rejects such retroactive actions.  When no explicit
    action exists, lowering leaves the task non-cached and remains feasible;
    it never treats a ghost hit as evidence.
    """
    log = []
    for cluster in clusters:
        target_count = _required_prefetch_target_count(mirror, cluster)
        if target_count is None:
            continue
        target_start = getattr(mirror.after, f"c{cluster}").task_start
        snap = getattr(state, f"c{cluster}")
        remaining_by_eid = dict(state.remaining)
        if (
            snap.pf_eid in remaining_by_eid
            and remaining_by_eid[snap.pf_eid] == target_count
            and snap.pf_end <= target_start
        ):
            continue

        candidates = []
        peer_cluster = 3 if cluster == 2 else 2
        peer_snap = getattr(state, f"c{peer_cluster}")
        mirror_is_split = (
            len(mirror.selected) == 1
            and mirror.after.c2.cur_eid == mirror.after.c3.cur_eid
            and _selected_on_cluster(mirror, 2)
            and _selected_on_cluster(mirror, 3)
        )
        for action in reference.gen_prefetch_actions(
            state.c2, state.c3, state.remaining
        ):
            if action.pf_cluster != cluster:
                continue
            if remaining_by_eid[action.pf_eid] != target_count:
                continue
            if (
                not mirror_is_split
                and peer_snap.pf_eid >= 0
                and action.pf_eid == peer_snap.pf_eid
            ):
                continue
            pf_end = action.pf_start + reference.dma_duration(
                reference.WEIGHT_BYTES_S1, action.pf_dma
            )
            if pf_end > target_start:
                continue
            # Prefer a single physical lane and the latest legal placement.
            candidates.append(
                (
                    int(action.pf_dma == reference.DmaBinding.BOTH),
                    -action.pf_start,
                    int(action.pf_dma),
                    action,
                )
            )
        if not candidates:
            log.append(
                {
                    "cluster": cluster,
                    "target_count": target_count,
                    "status": "ghost_not_materializable",
                }
            )
            continue
        candidates.sort(key=lambda item: item[:3])
        action = candidates[0][3]
        state = reference.apply_action(state, action)
        log.append(
            {
                "cluster": cluster,
                "target_count": target_count,
                "status": "explicit_prefetch_inserted",
                "reference_tag": action.tag,
                "pf_start_ticks": _ticks(action.pf_start),
            }
        )
    return state, log


def lower_policy_to_reference(
    token_dist: dict[int, int],
    *,
    proactive_prefetch: bool = False,
    materialize_prefetch: bool = True,
    mirror_trace: tuple[MirrorTransition, ...] | None = None,
) -> tuple[reference.BeamState, tuple[dict, ...]]:
    """Return a replay-validated explicit-lane history guided by the policy."""
    trace = _mirror_trace(token_dist) if mirror_trace is None else mirror_trace
    scheduler = reference.FourStageScheduler(token_dist)
    state = scheduler._initial_state()
    lowering_log = []

    if not materialize_prefetch:
        # A single greedy physical-shape choice can make a perfectly feasible
        # expert/cluster issue sequence dead-end several rounds later.  Use
        # depth-first backtracking along the fixed sequence, ordered by mirror
        # mismatch, and memoize failed (depth, physical state) pairs.  This
        # normally follows the old greedy path once and only explores nearby
        # alternatives at the first dead end, rather than materializing a
        # large cross-product at every round.
        failed: set[tuple[int, tuple]] = set()
        expanded = 0
        expansion_limit = 100_000

        def visit(
            index: int, parent: reference.BeamState
        ) -> tuple[reference.BeamState, tuple[dict, ...]] | None:
            nonlocal expanded
            if index == len(trace):
                return (parent, ()) if not parent.remaining else None
            key = (index, parent.fingerprint())
            if key in failed:
                return None
            if expanded >= expansion_limit:
                return None
            expanded += 1
            mirror = trace[index]
            candidates = _matching_action_candidates(
                parent, mirror, seed_mode=True
            )
            used_full_generation = False
            if not candidates:
                candidates = _matching_action_candidates(
                    parent, mirror, seed_mode=False
                )
                used_full_generation = True
            candidates.sort(
                key=lambda item: (
                    item[0],
                    item[2],
                    item[1],
                    abs(item[4].c2.task_end - item[4].c3.task_end),
                )
            )
            for mismatch, _g_score, _f_score, action, child in candidates:
                suffix = visit(index + 1, child)
                if suffix is None:
                    continue
                terminal, suffix_log = suffix
                entry = {
                    "index": index,
                    "mirror_tag": mirror.tag,
                    "mirror_decision": mirror.decision,
                    "selected_counts": [ntok for _eid, ntok in mirror.selected],
                    "reference_tag": action.tag,
                    "observable_mismatch_cc": mismatch,
                    "full_generation_fallback": used_full_generation,
                    "prefetch_materialization": [
                        {"status": "stage_only_no_prefetch_materialization"}
                    ],
                    "c2_end_ticks": _ticks(child.c2.task_end),
                    "c3_end_ticks": _ticks(child.c3.task_end),
                }
                return terminal, (entry,) + suffix_log
            failed.add(key)
            return None

        result = visit(0, state)
        if result is None:
            qualifier = (
                f" after reaching expansion limit {expansion_limit}"
                if expanded >= expansion_limit
                else ""
            )
            raise RuntimeError(
                f"stage-only lowering search found no terminal{qualifier}"
            )
        state, lowering_log = result
        validated = reference.validate_schedule_history(
            state.history, token_dist
        )
        if validated != state.g_score:
            raise RuntimeError(
                f"lowered history replay {validated} != state score "
                f"{state.g_score}"
            )
        return state, lowering_log

    for index, mirror in enumerate(trace):
        prefetch_log = []
        if not materialize_prefetch:
            prefetch_log = [
                {
                    "status": "stage_only_no_prefetch_materialization",
                }
            ]
        elif proactive_prefetch:
            # A concrete S4PF must be committed while the producing task is
            # still represented in the state.  For each cluster, look only at
            # its next assigned task; a reservation cannot skip an intervening
            # task on that cluster.
            for cluster in (2, 3):
                target = next(
                    (
                        future
                        for future in trace[index:]
                        if _selected_on_cluster(future, cluster)
                    ),
                    None,
                )
                if target is None:
                    continue
                state, inserted = _insert_required_prefetches(
                    state, target, clusters=(cluster,)
                )
                prefetch_log.extend(inserted)
        else:
            state, prefetch_log = _insert_required_prefetches(state, mirror)
        candidates = _matching_action_candidates(state, mirror, seed_mode=True)
        used_full_generation = False
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        if not candidates or candidates[0][0] != 0:
            full_candidates = _matching_action_candidates(
                state, mirror, seed_mode=False
            )
            used_full_generation = True
            if full_candidates:
                candidates.extend(full_candidates)
        if not candidates:
            raise RuntimeError(
                f"cannot lower mirror step {index} {mirror.tag}: "
                f"selected={mirror.selected}"
            )
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        mismatch, _g_score, _f_score, action, state = candidates[0]
        lowering_log.append(
            {
                "index": index,
                "mirror_tag": mirror.tag,
                "mirror_decision": mirror.decision,
                "selected_counts": [ntok for _eid, ntok in mirror.selected],
                "reference_tag": action.tag,
                "observable_mismatch_cc": mismatch,
                "full_generation_fallback": used_full_generation,
                "prefetch_materialization": prefetch_log,
                "c2_end_ticks": _ticks(state.c2.task_end),
                "c3_end_ticks": _ticks(state.c3.task_end),
            }
        )

    if state.remaining:
        raise RuntimeError("lowered reference history is not terminal")
    validated = reference.validate_schedule_history(state.history, token_dist)
    if validated != state.g_score:
        raise RuntimeError(
            f"lowered history replay {validated} != state score {state.g_score}"
        )
    return state, tuple(lowering_log)


def seed_beam_incumbent(
    scheduler: reference.FourStageScheduler,
    incumbent: reference.BeamState,
    beam_width: int,
    rank_mode: str,
    candidate_window: tuple[int, int] | None = None,
    incumbent_is_window_valid: bool | None = None,
) -> tuple[reference.BeamState, dict, bool]:
    """Find a better feasible history with the reference model's seed actions.

    This is an upper-bound search only.  Missing an action cannot prove
    infeasibility; every returned history is nevertheless a legal full-model
    history and equality with the independent lower bound is a valid proof.
    """
    started = time.perf_counter()
    initial = scheduler._initial_state()
    best = incumbent
    best_is_window_valid = (
        candidate_window is None
        if incumbent_is_window_valid is None
        else incumbent_is_window_valid
    )
    beam = [initial]
    seen: dict[tuple, int] = {}
    expanded = 0
    generated = 0

    def block_rank(state: reference.BeamState, cache_bonus: bool) -> tuple:
        if state.remaining:
            maximum = max(
                (ntok + reference.FULL_M_DIM - 1) // reference.FULL_M_DIM
                for _eid, ntok in state.remaining
            )
            histogram = [0] * (maximum + 1)
            for _eid, ntok in state.remaining:
                histogram[
                    (ntok + reference.FULL_M_DIM - 1)
                    // reference.FULL_M_DIM
                ] += 1
            remaining_blocks = block_constructor._minimum_block_class_plan(
                tuple(histogram)
            )[0]
        else:
            remaining_blocks = 0
        projected = (
            max(state.c2.task_end, state.c3.task_end)
            + remaining_blocks * block_constructor.BLOCK_CC
        )
        ready_cold_hits = 0
        if cache_bonus:
            remaining_by_eid = dict(state.remaining)
            for snap in (state.c2, state.c3):
                if (
                    snap.pf_eid in remaining_by_eid
                    and remaining_by_eid[snap.pf_eid] <= 2
                    and snap.pf_end >= 0
                    and snap.pf_end <= snap.task_end
                ):
                    ready_cold_hits += 1
            projected -= ready_cold_hits * block_constructor.BLOCK_CC
        return (
            projected,
            -ready_cold_hits,
            abs(state.c2.task_end - state.c3.task_end),
            state.f_score,
            state.g_score,
        )

    def visible_remaining(state: reference.BeamState):
        if candidate_window is None:
            return state.remaining
        top, bottom = candidate_window
        entries = len(state.remaining)
        indices = list(range(min(top, entries)))
        if bottom:
            indices.extend(range(max(top, entries - bottom), entries))
        # A previously issued concrete prefetch is already resident scheduler
        # state, even if intervening issues move its target outside the newly
        # ranked window.  Hiding it here would make the probe stricter than the
        # bounded hardware and incorrectly discard legal cache-hit continuations.
        rank_by_eid = {
            eid: index for index, (eid, _count) in enumerate(state.remaining)
        }
        for snap in (state.c2, state.c3):
            if snap.pf_eid in rank_by_eid:
                indices.append(rank_by_eid[snap.pf_eid])
        return tuple(state.remaining[index] for index in dict.fromkeys(indices))

    def generated_actions(state: reference.BeamState):
        action_remaining = visible_remaining(state)
        if candidate_window is None:
            return (
                reference.gen_stage_actions(
                    state.c2, state.c3, action_remaining, seed_mode=True
                )
                + reference.gen_prefetch_actions(
                    state.c2, state.c3, action_remaining, seed_mode=True
                )
            )

        # The window itself is the expert-rank bound.  Generate every expert
        # choice inside it once, while retaining seed_mode's compact shape/DMA
        # profiles.  The previous rotation workaround repeatedly regenerated
        # the same physical actions and still covered only cyclicly adjacent
        # rank subsets rather than every pair in the visible window.
        return (
            reference.gen_stage_actions(
                state.c2,
                state.c3,
                action_remaining,
                seed_mode=True,
                seed_all_visible=True,
            )
            + reference.gen_prefetch_actions(
                state.c2,
                state.c3,
                action_remaining,
                seed_mode=True,
                seed_all_visible=True,
            )
        )

    for _depth in range(scheduler.max_steps):
        if not beam or (best_is_window_valid and best.g_score == initial.f_score):
            break
        layer: dict[tuple, reference.BeamState] = {}
        for state in beam:
            if (
                state.f_score >= best.g_score
                if best_is_window_valid
                else state.f_score > best.g_score
            ):
                continue
            expanded += 1
            actions = generated_actions(state)
            generated += len(actions)
            for action in actions:
                child = reference.apply_action(state, action)
                if not child.remaining:
                    if (
                        child.g_score < best.g_score
                        or (
                            not best_is_window_valid
                            and child.g_score == best.g_score
                        )
                    ):
                        best = child
                        best_is_window_valid = True
                    continue
                if (
                    child.f_score >= best.g_score
                    if best_is_window_valid
                    else child.f_score > best.g_score
                ):
                    continue
                fingerprint = child.fingerprint()
                previous = layer.get(fingerprint)
                if previous is not None and previous.f_score <= child.f_score:
                    continue
                if fingerprint in seen and seen[fingerprint] <= child.f_score:
                    continue
                layer[fingerprint] = child
        if not layer:
            break
        candidates = list(layer.values())
        if rank_mode == "f_g":
            ordered = sorted(candidates)
        elif rank_mode == "completion":
            ordered = sorted(
                candidates,
                key=lambda state: (
                    reference.completion_estimate(state),
                    state.f_score,
                    state.g_score,
                ),
            )
        elif rank_mode == "cache":
            ordered = sorted(
                candidates,
                key=lambda state: (
                    reference._cache_aware_completion_estimate(state),
                    state.f_score,
                    state.g_score,
                ),
            )
        elif rank_mode == "lpt":
            ordered = sorted(
                candidates,
                key=lambda state: (
                    reference._lpt_completion_estimate(state),
                    state.f_score,
                    state.g_score,
                ),
            )
        elif rank_mode == "block":
            ordered = sorted(candidates, key=lambda state: block_rank(state, False))
        elif rank_mode == "block_cache":
            ordered = sorted(candidates, key=lambda state: block_rank(state, True))
        else:
            raise ValueError(f"unknown seed beam rank mode {rank_mode!r}")
        beam = reference._select_family_diverse_beam(ordered, beam_width)
        for child in beam:
            fingerprint = child.fingerprint()
            if fingerprint not in seen or child.f_score < seen[fingerprint]:
                seen[fingerprint] = child.f_score

    validated = reference.validate_schedule_history(best.history, scheduler.token_dist)
    if validated != best.g_score:
        raise RuntimeError(
            f"seed beam history replay {validated} != state score {best.g_score}"
        )
    return best, {
        "beam_width": beam_width,
        "rank_mode": rank_mode,
        "candidate_window": (
            None
            if candidate_window is None
            else {"top": candidate_window[0], "bottom": candidate_window[1]}
        ),
        "window_history_found": best_is_window_valid,
        "makespan_ticks": _ticks(best.g_score),
        "expanded": expanded,
        "generated": generated,
        "runtime_s": round(time.perf_counter() - started, 6),
    }, best_is_window_valid


def _history_matches_candidate_window(
    scheduler: reference.FourStageScheduler,
    history: tuple[reference.StageAction, ...],
    candidate_window: tuple[int, int],
) -> bool:
    """Replay whether every explicit issue/PF target is scheduler-visible.

    A previously issued prefetch remains resident scheduler state even when
    later rank changes move its target outside TOP+BOTTOM.  This mirrors the
    same exception used by ``seed_beam_incumbent.visible_remaining``.
    """
    top, bottom = candidate_window
    state = scheduler._initial_state()
    for action in history:
        entries = len(state.remaining)
        indices = list(range(min(top, entries)))
        if bottom:
            indices.extend(range(max(top, entries - bottom), entries))
        rank_by_eid = {
            eid: index for index, (eid, _count) in enumerate(state.remaining)
        }
        for snap in (state.c2, state.c3):
            if snap.pf_eid in rank_by_eid:
                indices.append(rank_by_eid[snap.pf_eid])
        visible_eids = {
            state.remaining[index][0] for index in dict.fromkeys(indices)
        }
        targets = {
            eid
            for eid in (action.c2_eid, action.c3_eid, action.pf_eid)
            if eid >= 0
        }
        if not targets.issubset(visible_eids):
            return False
        state = reference.apply_action(state, action)
    return not state.remaining


def prove_case(
    case: directed.DirectedCase,
    *,
    time_limit_s: float,
    max_expansions: int,
    seed_beam_widths: tuple[int, ...],
    seed_beam_modes: tuple[str, ...],
    seed_window: tuple[int, int] | None,
    prior_row: dict | None = None,
    target_decision: bool = False,
    target_rank_mode: str = "depth",
) -> dict:
    reference.clear_scheduler_caches()
    token_dist = {eid: ntok for eid, ntok in enumerate(case.counts)}
    started = time.perf_counter()
    mirror_result = policy.schedule_result(token_dist)
    scheduler = reference.FourStageScheduler(token_dist)
    initial = scheduler._initial_state()
    root_lb = initial.f_score
    prior_lb = (
        int(Fraction(prior_row["certified_lower_bound_ticks"]) * TICK_CC)
        if prior_row is not None
        else root_lb
    )
    known_lb = max(root_lb, prior_lb)
    if prior_row is not None:
        prior_history = tuple(
            deserialize_action(action) for action in prior_row["actions"]
        )
        lowered = scheduler._validated_incumbent_state(prior_history)
        lowering_mode = (
            "prior_replay_validated_history"
            if seed_window is None
            else "outside_window_prior_replay_bound"
        )
        lowering_log = tuple(prior_row.get("lowering", ()))
        lowering_error = prior_row.get("lowering_error")
    else:
        lowering_error = None
        lowering_variants = []
        lowering_errors = []
        for lowering_mode, proactive in (("late", False), ("proactive", True)):
            try:
                variant, variant_log = lower_policy_to_reference(
                    token_dist, proactive_prefetch=proactive
                )
                lowering_variants.append(
                    (variant.g_score, lowering_mode, variant, variant_log)
                )
            except RuntimeError as exc:
                lowering_errors.append(f"{lowering_mode}: {exc}")
        if lowering_variants:
            _score, lowering_mode, lowered, lowering_log = min(
                lowering_variants, key=lambda item: (item[0], item[1])
            )
            lowering_error = "; ".join(lowering_errors) or None
        else:
            # A mirror trace can depend on a ghost prefetch that has no legal
            # explicit-lane placement.  Fall back to an independently generated
            # feasible history; never use the mirror makespan as an incumbent.
            lowering_mode = "reference_greedy_fallback"
            lowering_error = "; ".join(lowering_errors)
            lowered = scheduler._greedy_incumbent(initial, target_gap=0.0)
            lowering_log = ()

    incumbent = lowered
    incumbent_matches_seed_window = (
        True
        if seed_window is None
        else _history_matches_candidate_window(
            scheduler, incumbent.history, seed_window
        )
    )
    seed_beam_trials = []
    for beam_width in seed_beam_widths:
        for rank_mode in seed_beam_modes:
            incumbent, trial, incumbent_matches_seed_window = seed_beam_incumbent(
                scheduler,
                incumbent,
                beam_width,
                rank_mode,
                candidate_window=seed_window,
                incumbent_is_window_valid=incumbent_matches_seed_window,
            )
            seed_beam_trials.append(trial)
            if incumbent.g_score == root_lb and incumbent_matches_seed_window:
                break
        if incumbent.g_score == root_lb and incumbent_matches_seed_window:
            break

    if incumbent.g_score == root_lb and incumbent_matches_seed_window:
        makespan = incumbent.g_score
        lower_bound = root_lb
        history = incumbent.history
        proven_optimal = True
        termination = (
            "seed_beam_history_equals_root_lb"
            if incumbent.g_score < lowered.g_score
            else "feasible_history_equals_root_lb"
        )
        expansions = 0
        generated = 0
        optimality_gap = 0.0
        search_runtime_s = 0.0
    elif target_decision:
        if time_limit_s <= 0:
            raise ValueError("target decision requires a positive time limit")
        quantum = reference.SCHEDULE_TIME_QUANTUM_CC
        # Search the first makespan lattice point not already excluded by the
        # certified lower bound.  A feasible history at this frontier is
        # globally optimal; exhaustive infeasibility advances the certified
        # lower bound by exactly one proven lattice quantum.
        target_makespan = ((known_lb + quantum - 1) // quantum) * quantum
        if target_makespan >= incumbent.g_score:
            makespan = incumbent.g_score
            history = incumbent.history
            lower_bound = incumbent.g_score
            proven_optimal = True
            termination = "lattice_lower_bound_equals_incumbent"
            expansions = 0
            generated = 0
            optimality_gap = 0.0
            search_runtime_s = 0.0
            decision = None
        else:
            decision = scheduler.run_target_feasibility(
                target_makespan,
                time_limit_s=time_limit_s,
                max_expansions=max_expansions,
                rank_mode=target_rank_mode,
            )
            expansions = decision.expansions
            generated = decision.generated
            search_runtime_s = decision.runtime_s
            if decision.feasible:
                decided_state = scheduler._validated_incumbent_state(
                    decision.history
                )
                makespan = decided_state.g_score
                history = decided_state.history
                if makespan != target_makespan:
                    raise AssertionError(
                        "frontier feasibility returned a non-frontier makespan"
                    )
                lower_bound = makespan
                proven_optimal = True
                termination = "target_frontier_feasible_optimal"
            elif decision.exhaustive:
                makespan = incumbent.g_score
                history = incumbent.history
                lower_bound = target_makespan + quantum
                proven_optimal = lower_bound >= makespan
                if proven_optimal:
                    lower_bound = makespan
                    termination = "target_frontier_infeasible_incumbent_optimal"
                else:
                    termination = "target_frontier_infeasible_lb_advanced"
            else:
                makespan = incumbent.g_score
                history = incumbent.history
                lower_bound = known_lb
                proven_optimal = False
                termination = f"target_decision_{decision.termination}"
            optimality_gap = (
                0.0
                if proven_optimal or lower_bound == 0
                else (makespan - lower_bound) / lower_bound
            )
    elif time_limit_s > 0:
        result = scheduler.run_anytime(
            time_limit_s=time_limit_s,
            max_expansions=max_expansions,
            target_gap=0.0,
            incumbent_history=incumbent.history,
        )
        makespan = result.makespan
        lower_bound = result.lower_bound
        history = result.history
        proven_optimal = result.proven_optimal
        termination = result.termination
        expansions = result.expansions
        generated = result.generated
        optimality_gap = result.optimality_gap
        search_runtime_s = result.runtime_s
    else:
        makespan = incumbent.g_score
        lower_bound = root_lb
        history = incumbent.history
        proven_optimal = False
        termination = "search_disabled"
        expansions = 0
        generated = 0
        optimality_gap = (
            0.0 if lower_bound >= makespan else (makespan - lower_bound) / lower_bound
        )
        search_runtime_s = 0.0

    validated = reference.validate_schedule_history(history, token_dist)
    if validated != makespan:
        raise RuntimeError(f"final history replay {validated} != result {makespan}")
    row = {
        "name": case.name,
        "origin": case.origin,
        "tier": case.tier,
        "family": case.family,
        "hot_experts": case.hot_experts,
        "profile": case.profile,
        "batch_tokens": case.batch_tokens,
        "active_experts": len(case.counts),
        "counts": list(case.counts),
        "root_lower_bound_ticks": _ticks(root_lb),
        "mirror_policy_ticks": _ticks(mirror_result.makespan_cc),
        "lowered_reference_ticks": _ticks(lowered.g_score),
        "best_reference_ticks": _ticks(makespan),
        "certified_lower_bound_ticks": _ticks(lower_bound),
        "optimality_gap": optimality_gap,
        "proven_optimal": proven_optimal,
        "termination": termination,
        "expansions": expansions,
        "generated_states": generated,
        "seed_beam_trials": seed_beam_trials,
        "search_runtime_s": round(search_runtime_s, 6),
        "total_runtime_s": round(time.perf_counter() - started, 6),
        "history_replay_valid": True,
        "lowering_mode": lowering_mode,
        "lowering_error": lowering_error,
        "actions": [serialize_action(action) for action in history],
        "lowering": list(lowering_log),
    }
    if target_decision and incumbent.g_score != root_lb:
        row["target_decision_ticks"] = _ticks(target_makespan)
        row["target_decision_rank_mode"] = target_rank_mode
        if decision is not None:
            row["target_decision_exhaustive"] = decision.exhaustive
            row["target_decision_feasible"] = decision.feasible
            row["target_decision_open_states"] = decision.open_states
            row["target_decision_closed_states"] = decision.closed_states
            row["target_decision_peak_open_states"] = decision.peak_open_states
            row["target_decision_pruned_by_bound"] = decision.pruned_by_bound
    return row


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_external_cases(path: Path) -> tuple[list[directed.DirectedCase], dict[str, dict]]:
    """Load a generated directed-case payload without changing the frozen suite."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("cases"), list):
        raise SystemExit(f"{path}: missing cases list")
    cases: list[directed.DirectedCase] = []
    metadata_by_name: dict[str, dict] = {}
    for row in payload["cases"]:
        try:
            name = str(row["name"])
            full_counts = tuple(int(value) for value in row["counts_64"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"{path}: malformed case row {row!r}") from exc
        if name in metadata_by_name:
            raise SystemExit(f"{path}: duplicate case name {name!r}")
        if len(full_counts) > 64 or any(value < 0 for value in full_counts):
            raise SystemExit(f"{path}: invalid counts for {name}")
        if tuple(sorted(full_counts, reverse=True)) != full_counts:
            raise SystemExit(f"{path}: counts for {name} must be sorted")
        active_counts = [value for value in full_counts if value > 0]
        case = directed._case(
            name,
            str(row.get("tier", "external_directed")),
            str(row.get("family", "external")),
            active_counts,
            batch_tokens=row.get("batch_tokens"),
            origin=str(row.get("origin", "external_case_input")),
            hot_experts=row.get("hot_experts"),
            medium_experts=row.get("medium_experts"),
            profile=row.get("profile"),
        )
        cases.append(case)
        metadata_by_name[name] = {
            "source_schema": payload.get("schema"),
            "source_contract": payload.get("source_contract"),
            "full_expert_count": len(full_counts),
            "full_counts": list(full_counts),
            "metrics": row.get("metrics"),
            "description": row.get("description"),
            "target_constraints": row.get("target_constraints"),
        }
    return cases, metadata_by_name


def _summary(rows: list[dict]) -> dict:
    return {
        "cases": len(rows),
        "proven_optimal": sum(row["proven_optimal"] for row in rows),
        "unproven": sum(not row["proven_optimal"] for row in rows),
        "mirror_equals_lowered": sum(
            row["mirror_policy_ticks"] == row["lowered_reference_ticks"]
            for row in rows
        ),
        "mirror_better_than_lowered_unverified": [
            row["name"]
            for row in rows
            if Fraction(row["mirror_policy_ticks"])
            < Fraction(row["lowered_reference_ticks"])
        ],
    }


def _payload(rows: list[dict], *, complete: bool) -> dict:
    return {
        "schema": "top4_bottom2_directed_proof_v1",
        "proof_model": "explicit_dma_lane_four_stage_anytime",
        "complete": complete,
        "summary": _summary(rows),
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-input",
        type=Path,
        help=(
            "optional generated directed-case JSON; when present it replaces "
            "the frozen built-in directed suite"
        ),
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--time-limit-s", type=float, default=0.0)
    parser.add_argument("--max-expansions", type=int, default=200_000)
    parser.add_argument(
        "--target-decision",
        action="store_true",
        help=(
            "run complete feasibility search at the first unexcluded schedule "
            "time lattice point"
        ),
    )
    parser.add_argument(
        "--target-rank-mode",
        choices=(
            "completion",
            "depth",
            "balance",
            "lpt",
            "cache",
            "hot_tail",
            "dma",
        ),
        default="depth",
        help="OPEN ordering only; does not change exact target state coverage",
    )
    parser.add_argument(
        "--seed-beam-widths",
        default="",
        help="comma-separated legal seed-action beam widths; empty disables",
    )
    parser.add_argument(
        "--seed-beam-modes",
        default="f_g",
        help="comma-separated seed beam ranks: f_g,completion,cache,lpt",
    )
    parser.add_argument(
        "--seed-window",
        default="",
        help=(
            "optional semantic candidate window TOP,BOTTOM for seed search; "
            "for example 4,2"
        ),
    )
    parser.add_argument(
        "--prior-proof",
        type=Path,
        help="reuse replay-validated histories from a previous proof payload",
    )
    parser.add_argument(
        "--classification-input",
        type=Path,
        help="optional directed_case_classification payload used to select cases",
    )
    parser.add_argument(
        "--classification",
        action="append",
        default=[],
        help="classification value to retain; may be repeated",
    )
    parser.add_argument("--only-unproven", action="store_true")
    parser.add_argument("--min-active", type=int, default=0)
    parser.add_argument("--max-active", type=int, default=0)
    parser.add_argument("--max-start-gap-ticks", type=int, default=-1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/policy_search/top4_bottom2_directed_proofs.json"),
    )
    args = parser.parse_args()
    try:
        seed_beam_widths = tuple(
            int(field.strip())
            for field in args.seed_beam_widths.split(",")
            if field.strip()
        )
    except ValueError as exc:
        raise SystemExit("--seed-beam-widths must contain integers") from exc
    seed_beam_modes = tuple(
        field.strip() for field in args.seed_beam_modes.split(",") if field.strip()
    )
    seed_window = None
    if args.seed_window:
        try:
            seed_window_fields = tuple(
                int(field.strip()) for field in args.seed_window.split(",")
            )
        except ValueError as exc:
            raise SystemExit("--seed-window must be TOP,BOTTOM") from exc
        if len(seed_window_fields) != 2:
            raise SystemExit("--seed-window must be TOP,BOTTOM")
        seed_window = seed_window_fields
    valid_seed_modes = {
        "f_g",
        "completion",
        "cache",
        "lpt",
        "block",
        "block_cache",
    }
    if (
        args.limit < 0
        or args.time_limit_s < 0
        or args.max_expansions <= 0
        or args.min_active < 0
        or args.max_active < 0
        or any(width <= 0 for width in seed_beam_widths)
        or any(mode not in valid_seed_modes for mode in seed_beam_modes)
        or (
            seed_window is not None
            and (seed_window[0] <= 0 or seed_window[1] < 0)
        )
    ):
        raise SystemExit("invalid non-positive limit")
    if args.target_decision and args.time_limit_s <= 0:
        raise SystemExit("--target-decision requires --time-limit-s > 0")

    source_metadata_by_name: dict[str, dict] = {}
    if args.case_input is None:
        cases = list(directed.directed_cases())
    else:
        cases, source_metadata_by_name = _load_external_cases(args.case_input)
    prior_by_name = {}
    if args.prior_proof is not None:
        prior_payload = json.loads(args.prior_proof.read_text(encoding="utf-8"))
        prior_by_name = {row["name"]: row for row in prior_payload["cases"]}
        if args.only_unproven:
            cases = [
                case
                for case in cases
                if case.name in prior_by_name
                and not prior_by_name[case.name]["proven_optimal"]
            ]
        if args.max_start_gap_ticks >= 0:
            cases = [
                case
                for case in cases
                if case.name in prior_by_name
                and Fraction(prior_by_name[case.name]["best_reference_ticks"])
                - Fraction(prior_by_name[case.name]["certified_lower_bound_ticks"])
                <= args.max_start_gap_ticks
            ]
    elif args.only_unproven or args.max_start_gap_ticks >= 0:
        raise SystemExit("proof-based selection requires --prior-proof")
    if args.classification:
        if args.classification_input is None:
            raise SystemExit("--classification requires --classification-input")
        classification_payload = json.loads(
            args.classification_input.read_text(encoding="utf-8")
        )
        selected_names = {
            row["name"]
            for row in classification_payload["cases"]
            if row["classification"] in set(args.classification)
        }
        cases = [case for case in cases if case.name in selected_names]
    if args.min_active:
        cases = [case for case in cases if len(case.counts) >= args.min_active]
    if args.max_active:
        cases = [case for case in cases if len(case.counts) <= args.max_active]
    if args.case:
        requested = set(args.case)
        cases = [case for case in cases if case.name in requested]
        missing = requested - {case.name for case in cases}
        if missing:
            raise SystemExit(f"unknown cases: {sorted(missing)}")
    if args.limit:
        cases = cases[: args.limit]

    rows = []
    for index, case in enumerate(cases, 1):
        row = prove_case(
            case,
            time_limit_s=args.time_limit_s,
            max_expansions=args.max_expansions,
            seed_beam_widths=seed_beam_widths,
            seed_beam_modes=seed_beam_modes,
            seed_window=seed_window,
            prior_row=prior_by_name.get(case.name),
            target_decision=args.target_decision,
            target_rank_mode=args.target_rank_mode,
        )
        if case.name in source_metadata_by_name:
            row["source_metadata"] = source_metadata_by_name[case.name]
        rows.append(row)
        print(
            f"[{index}/{len(cases)}] {case.name} "
            f"LB={row['root_lower_bound_ticks']} "
            f"mirror={row['mirror_policy_ticks']} "
            f"lowered={row['lowered_reference_ticks']} "
            f"best={row['best_reference_ticks']} "
            f"certLB={row['certified_lower_bound_ticks']} "
            f"proven={row['proven_optimal']} "
            f"term={row['termination']} "
            f"runtime={row['total_runtime_s']}s",
            flush=True,
        )
        _atomic_write(args.output, _payload(rows, complete=False))

    summary = _summary(rows)
    _atomic_write(args.output, _payload(rows, complete=True))
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
