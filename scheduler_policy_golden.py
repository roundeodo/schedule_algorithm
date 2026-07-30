#!/usr/bin/env python3
"""Deterministic golden model for the frozen bounded RTL scheduler policy.

Candidate construction is shared with ``derive_scheduler_policy.py`` so that
the audited direct generator and the deployable policy cannot drift.  This
file owns the final runtime constants, future score, tie-break and commit loop.
It performs no beam search, continuation rollout or fitted-model inference.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Callable

from four_stage_scheduler import (
    BeamState,
    DmaBinding,
    FourStageScheduler,
    PF_EID_GHOST,
    SHAPE_A,
    SHAPE_B,
    SHAPE_C,
    StageAction,
    apply_action,
    clear_scheduler_caches,
    validate_schedule_history,
)
from run_four_stage_reference import serialize_action


P5_CANDIDATE_REVISION = "direct-slot-conditional-cache-v9-rtl-order"
LEGACY_CANDIDATE_REVISION = "direct-slot-conditional-cache-v8"
CANDIDATE_REVISION = P5_CANDIDATE_REVISION
TIME_QUANTUM_CC = 11_264
HALF_QUANTUM_CC = TIME_QUANTUM_CC // 2
BOUNDED_HEAD_VISIBLE = 8
BOUNDED_TAIL_VISIBLE = 4
ACTION_FAMILIES = ("SINGLE", "PAIR", "SPLIT", "PREFETCH")


@dataclass(frozen=True)
class PolicyConfig:
    """Frozen generator and selection constants for one policy revision."""

    policy_id: str
    rank_limit: int
    bottom_count: int
    candidate_budget: int
    scorer_kind: str
    pathmax_kind: str
    candidate_revision: str


FROZEN_V1_CONFIG = PolicyConfig(
    policy_id="r4-b2-k32-direct-v8-lpt-rem-snap-v1",
    rank_limit=4,
    bottom_count=2,
    candidate_budget=32,
    scorer_kind="full-lpt",
    pathmax_kind="full-reference",
    candidate_revision=LEGACY_CANDIDATE_REVISION,
)
P5_CONFIG = PolicyConfig(
    policy_id="r8-k32-direct-v9-full-lpt-dma-pm-rem-snap-p5",
    rank_limit=8,
    bottom_count=0,
    candidate_budget=32,
    scorer_kind="full-lpt",
    pathmax_kind="dma-capacity",
    candidate_revision=P5_CANDIDATE_REVISION,
)
BOUNDED_S2_CONFIG = PolicyConfig(
    policy_id="r4-b2-k32-direct-v9-bounded-s2-dma-rem-snap-action-v1",
    rank_limit=4,
    bottom_count=2,
    candidate_budget=32,
    scorer_kind="bounded-s2",
    pathmax_kind="dma-capacity",
    candidate_revision=P5_CANDIDATE_REVISION,
)
POLICY_CONFIGS = {
    "r4-b2-v1": FROZEN_V1_CONFIG,
    "r8-p5": P5_CONFIG,
    "r4-b2-s2": BOUNDED_S2_CONFIG,
}

# Backward-compatible aliases for users importing the frozen v1 constants.
POLICY_ID = FROZEN_V1_CONFIG.policy_id
RANK_LIMIT = FROZEN_V1_CONFIG.rank_limit
BOTTOM_COUNT = FROZEN_V1_CONFIG.bottom_count
CANDIDATE_BUDGET = FROZEN_V1_CONFIG.candidate_budget


@dataclass(frozen=True)
class Decision:
    """One committed golden-policy decision and its auditable score key."""

    action: StageAction
    child: BeamState
    candidate_index: int
    candidate_count: int
    score_key: tuple
    next_pathmax_cc: int


CandidateGenerator = Callable[..., list[StageAction]]


def isolated_duration_cc(ntok: int) -> int:
    """Best isolated four-stage duration used by the integer LPT estimate.

    In the fixed timing model the best S1/S2 duration is
    ``ceil(ntok/2) * 2*Tq`` and the best S3/S4 duration is
    ``ceil(ntok/2) * Tq``.  The combined duration therefore needs only an
    increment, right shift, and shift-add multiplication by three.
    """
    if ntok <= 0:
        return 0
    half_token_blocks = (int(ntok) + 1) >> 1
    return 3 * TIME_QUANTUM_CC * half_token_blocks


def lpt_future_score_cc(state: BeamState) -> int:
    """Estimate final makespan by two-lane longest-processing-time placement."""
    return max(int(state.f_score), lpt_load_score_cc(state))


def lpt_load_score_cc(state: BeamState) -> int:
    """Two-lane LPT load estimate without any search-only lower bound."""
    end2 = int(state.c2.task_end)
    end3 = int(state.c3.task_end)
    # ``remaining`` is maintained in descending token order by the state
    # transition.  Sorting here makes that contract explicit and defensive;
    # equal-token experts have equal duration, so their ID order cannot change
    # the score.
    remaining = sorted(state.remaining, key=lambda item: (-item[1], item[0]))
    for _, ntok in remaining:
        duration = isolated_duration_cc(ntok)
        if end2 <= end3:
            end2 += duration
        else:
            end3 += duration
    return max(end2, end3)


def _dma_event_points(state: BeamState, start: int) -> list[int]:
    """Sorted committed DMA endpoints visible to the capacity sweep."""
    points = {int(start)}
    for snap in (state.c2, state.c3):
        if snap.cur_eid >= 0 and snap.bw_s1 > 0 and snap.task_start < snap.dma1_end:
            points.update((int(snap.task_start), int(snap.dma1_end)))
        if snap.s2pf_start >= 0 and snap.s2pf_bw > 0:
            points.update((int(snap.s2pf_start), int(snap.s2pf_end)))
        if snap.cur_eid >= 0 and snap.bw_s3 > 0 and snap.s2_end < snap.dma3_end:
            points.update((int(snap.s2_end), int(snap.dma3_end)))
        if snap.pf_start >= 0 and snap.pf_bw > 0:
            points.update((int(snap.pf_start), int(snap.pf_end)))
    return sorted(point for point in points if point >= start)


def dma_capacity_bound_cc(state: BeamState) -> int:
    """Hardware-oriented mandatory-DMA capacity lower bound.

    Work is measured in cycles on one 64-B/cc lane.  One missing S1 transfer
    costs 45,056 lane-cycles and one missing S3 transfer costs 22,528.
    Existing reservations may cover at most one remaining expert per cache
    slot.  The two committed DMA timelines are then swept by endpoint.
    """
    remaining_eids = {eid for eid, _ in state.remaining}
    n_experts = len(remaining_eids)
    if n_experts == 0:
        return max(int(state.c2.task_end), int(state.c3.task_end))

    snaps = (state.c2, state.c3)
    concrete_s1 = {
        snap.pf_eid
        for snap in snaps
        if snap.pf_eid in remaining_eids and snap.pf_end >= 0
    }
    ghost_s1 = sum(
        snap.pf_eid == PF_EID_GHOST and snap.pf_end >= 0 for snap in snaps
    )
    s1_slots = min(n_experts, len(concrete_s1) + ghost_s1)
    full_slots = len(
        {
            snap.pf_eid
            for snap in snaps
            if snap.pf_eid in remaining_eids
            and snap.pf_end >= 0
            and snap.pf_full
        }
    )
    remaining_lane_cycles = (
        (n_experts - s1_slots) * 45_056
        + (n_experts - full_slots) * 22_528
    )

    releases = [int(state.c2.task_end), int(state.c3.task_end)]
    for snap in snaps:
        if snap.cur_eid >= 0 and snap.pf_eid == -1:
            releases.append(int(snap.dma3_end))
    start = min(releases)
    points = _dma_event_points(state, start)

    for left, right in zip(points, points[1:]):
        if right <= left:
            continue
        used = DmaBinding(
            state.c2.active_dma_mask_at(left)
            | state.c3.active_dma_mask_at(left)
        )
        free_lanes = 2 - int(bool(used & DmaBinding.IDMA)) - int(
            bool(used & DmaBinding.XDMA)
        )
        if free_lanes == 0:
            continue
        capacity = (right - left) * free_lanes
        if remaining_lane_cycles <= capacity:
            finish = left + (
                remaining_lane_cycles + free_lanes - 1
            ) // free_lanes
            break
        remaining_lane_cycles -= capacity
    else:
        tail = points[-1]
        finish = tail + (remaining_lane_cycles + 1) // 2

    bound = max(int(state.c2.task_end), int(state.c3.task_end), finish)
    if bound % (TIME_QUANTUM_CC // 2) != 0:
        raise RuntimeError(f"DMA capacity bound is not Hq-aligned: {bound}")
    return bound


def _time_hq(value: int) -> int:
    if value < 0:
        return int(value)
    if value % HALF_QUANTUM_CC != 0:
        raise RuntimeError(f"time is not Hq-aligned: {value}")
    return int(value) // HALF_QUANTUM_CC


def _expert_work_hq(ntok: int) -> int:
    blocks = (int(ntok) + 1) >> 1
    return (blocks << 1) + (blocks << 2)


def _action_family(action: StageAction) -> str:
    if (
        action.pf_cluster in (2, 3)
        or action.c2_eid == -2
        or action.c3_eid == -2
        or action.tag.startswith("PF-")
    ):
        return "PREFETCH"
    if action.c2_eid >= 0 and action.c3_eid >= 0:
        return "SPLIT" if action.c2_eid == action.c3_eid else "PAIR"
    return "SINGLE"


def rtl_action_order_key(action: StageAction) -> tuple[int, ...]:
    """Frozen direct-v9 fixed-width numeric physical-action order."""
    shape_order = {None: -1, SHAPE_A: 0, SHAPE_B: 1, SHAPE_C: 2}
    return (
        int(action.c2_eid),
        int(action.c2_ntok),
        shape_order[action.c2_shape_s1],
        shape_order[action.c2_shape_s3],
        int(action.c2_start),
        int(action.c2_s1_cached),
        int(action.c2_s3_cached),
        int(action.c3_eid),
        int(action.c3_ntok),
        shape_order[action.c3_shape_s1],
        shape_order[action.c3_shape_s3],
        int(action.c3_start),
        int(action.c3_s1_cached),
        int(action.c3_s3_cached),
        int(action.pf_cluster),
        int(action.pf_eid),
        shape_order[action.pf_shape],
        int(action.pf_start),
        int(action.c2_s2pf_start),
        int(action.c3_s2pf_start),
        int(action.c2_dma_s1),
        int(action.c2_dma_s3),
        int(action.c2_s2pf_dma),
        int(action.c3_dma_s1),
        int(action.c3_dma_s3),
        int(action.c3_s2pf_dma),
        int(action.pf_dma),
    )


def _visible_ids(state: BeamState) -> set[int]:
    remaining_eids = {eid for eid, _ in state.remaining}
    visible = {eid for eid, _ in state.remaining[:BOUNDED_HEAD_VISIBLE]}
    visible.update(eid for eid, _ in state.remaining[-BOUNDED_TAIL_VISIBLE:])
    visible.update(
        snap.pf_eid
        for snap in (state.c2, state.c3)
        if snap.pf_eid in remaining_eids
    )
    return visible


def _visible_partition(
    visible_source: BeamState, state: BeamState
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    child_by_eid = dict(state.remaining)
    head_eids = [eid for eid, _ in visible_source.remaining[:BOUNDED_HEAD_VISIBLE]]
    head_set = set(head_eids)
    tail_eids = [
        eid
        for eid, _ in visible_source.remaining[-BOUNDED_TAIL_VISIBLE:]
        if eid not in head_set
    ]
    head = [(eid, child_by_eid[eid]) for eid in head_eids if eid in child_by_eid]
    tail = [(eid, child_by_eid[eid]) for eid in tail_eids if eid in child_by_eid]
    visible_work = sum(_expert_work_hq(ntok) for _, ntok in (*head, *tail))
    total_work = sum(_expert_work_hq(ntok) for _, ntok in state.remaining)
    middle_work = total_work - visible_work
    if middle_work < 0:
        raise RuntimeError("visible work exceeds total remaining work")
    return head, tail, middle_work


def _lpt_place_hq(loads: list[int], entries: list[tuple[int, int]]) -> None:
    for _, ntok in sorted(entries, key=lambda item: (-item[1], item[0])):
        target = 0 if loads[0] <= loads[1] else 1
        loads[target] += _expert_work_hq(ntok)


def _balance_middle_hq(loads: list[int], work: int) -> None:
    if work <= 0:
        return
    low = 0 if loads[0] <= loads[1] else 1
    high = 1 - low
    fill = min(loads[high] - loads[low], work)
    loads[low] += fill
    work -= fill
    if work:
        loads[low] += work // 2
        loads[high] += work - work // 2


def bounded_s1_score_hq(visible_source: BeamState, state: BeamState) -> int:
    loads = [_time_hq(state.c2.task_end), _time_hq(state.c3.task_end)]
    head, tail, middle_work = _visible_partition(visible_source, state)
    _lpt_place_hq(loads, head)
    _balance_middle_hq(loads, middle_work)
    _lpt_place_hq(loads, tail)
    return max(loads)


def _bounded_next_family_actions(
    visible_source: BeamState,
    child: BeamState,
    *,
    generator: CandidateGenerator,
    config: PolicyConfig,
) -> list[StageAction]:
    visible = _visible_ids(visible_source)
    visible_remaining = tuple(item for item in child.remaining if item[0] in visible)
    if not visible_remaining:
        return []
    restricted = replace(child, remaining=visible_remaining)
    generated = generator(
        restricted,
        rank_limit=config.rank_limit,
        bottom_count=config.bottom_count,
        budget=config.candidate_budget,
    )
    first_by_family = {}
    for action in generated:
        first_by_family.setdefault(_action_family(action), action)
    return [
        first_by_family[family]
        for family in ACTION_FAMILIES
        if family in first_by_family
    ]


def bounded_s2_score_hq(
    parent: BeamState,
    child: BeamState,
    *,
    generator: CandidateGenerator,
    config: PolicyConfig,
    inherited_pathmax_cc: int,
) -> tuple[int, int]:
    next_pathmax_cc = max(inherited_pathmax_cc, dma_capacity_bound_cc(child))
    current = max(
        bounded_s1_score_hq(parent, child), _time_hq(next_pathmax_cc)
    )
    if not child.remaining:
        return current, next_pathmax_cc
    next_actions = _bounded_next_family_actions(
        parent, child, generator=generator, config=config
    )
    if not next_actions:
        return current, next_pathmax_cc
    terminals = []
    for next_action in next_actions:
        grandchild = apply_action(child, next_action)
        grandchild_pathmax_cc = max(
            next_pathmax_cc, dma_capacity_bound_cc(grandchild)
        )
        terminals.append(
            max(
                bounded_s1_score_hq(parent, grandchild),
                _time_hq(grandchild_pathmax_cc),
            )
        )
    return max(current, min(terminals)), next_pathmax_cc


def _default_candidate_generator(config: PolicyConfig) -> CandidateGenerator:
    # Lazy import avoids a module cycle: the derivation tool calls this golden
    # loop with its already-loaded generator, while standalone golden-model use
    # resolves the same function here.
    from derive_scheduler_policy import (
        generate_direct_candidates,
        generate_direct_candidates_v8,
    )

    return (
        generate_direct_candidates_v8
        if config.candidate_revision == LEGACY_CANDIDATE_REVISION
        else generate_direct_candidates
    )


def select_action(
    state: BeamState,
    *,
    candidate_generator: CandidateGenerator | None = None,
    config: PolicyConfig = FROZEN_V1_CONFIG,
    inherited_pathmax_cc: int = 0,
) -> Decision:
    """Build, score and select one action under the frozen RTL policy."""
    generator = candidate_generator or _default_candidate_generator(config)
    actions = generator(
        state,
        rank_limit=config.rank_limit,
        bottom_count=config.bottom_count,
        budget=config.candidate_budget,
    )
    if not actions:
        raise RuntimeError("golden policy has no legal candidate")
    if len(actions) > config.candidate_budget:
        raise RuntimeError("golden policy exceeded the candidate budget")

    decisions = []
    for candidate_index, action in enumerate(actions):
        child = apply_action(state, action)
        if config.scorer_kind == "bounded-s2":
            if config.pathmax_kind != "dma-capacity":
                raise ValueError("bounded S2 requires DMA-capacity pathmax")
            score_hq, next_pathmax_cc = bounded_s2_score_hq(
                state,
                child,
                generator=generator,
                config=config,
                inherited_pathmax_cc=inherited_pathmax_cc,
            )
            score_key = (
                score_hq,
                len(child.remaining),
                max(child.c2.task_end, child.c3.task_end),
                rtl_action_order_key(action),
            )
        elif config.scorer_kind == "full-lpt":
            if config.pathmax_kind == "full-reference":
                next_pathmax_cc = int(child.f_score)
                future_score_cc = lpt_future_score_cc(child)
            elif config.pathmax_kind == "dma-capacity":
                next_pathmax_cc = max(
                    int(inherited_pathmax_cc), dma_capacity_bound_cc(child)
                )
                future_score_cc = max(
                    lpt_load_score_cc(child), next_pathmax_cc
                )
            else:
                raise ValueError(f"unsupported pathmax kind: {config.pathmax_kind}")
            score_key = (
                future_score_cc,
                len(child.remaining),
                max(child.c2.task_end, child.c3.task_end),
                candidate_index,
            )
        else:
            raise ValueError(f"unsupported scorer kind: {config.scorer_kind}")
        decisions.append(
            Decision(
                action=action,
                child=child,
                candidate_index=candidate_index,
                candidate_count=len(actions),
                score_key=score_key,
                next_pathmax_cc=next_pathmax_cc,
            )
        )
    return min(decisions, key=lambda decision: decision.score_key)


def run_policy(
    initial: BeamState,
    *,
    candidate_generator: CandidateGenerator | None = None,
    config: PolicyConfig = FROZEN_V1_CONFIG,
) -> tuple[BeamState, list[StageAction], int]:
    """Run the frozen policy round by round until every expert is consumed."""
    state = initial
    history = []
    max_candidates = 0
    max_decisions = 4 * len(initial.remaining) + 8
    pathmax_cc = (
        dma_capacity_bound_cc(initial)
        if config.pathmax_kind == "dma-capacity"
        else int(initial.f_score)
    )
    while state.remaining:
        decision = select_action(
            state,
            candidate_generator=candidate_generator,
            config=config,
            inherited_pathmax_cc=pathmax_cc,
        )
        history.append(decision.action)
        state = decision.child
        pathmax_cc = decision.next_pathmax_cc
        max_candidates = max(max_candidates, decision.candidate_count)
        if len(history) > max_decisions:
            raise RuntimeError("golden policy exceeded the progress guard")
    return state, history, max_candidates


def run_distribution(
    distribution: dict[int, int],
    *,
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
    config: PolicyConfig = FROZEN_V1_CONFIG,
) -> dict:
    """Convenience API that returns a serialized, independently validated run."""
    scheduler = FourStageScheduler(
        distribution,
        initial_cache_c2=initial_cache_c2,
        initial_cache_c3=initial_cache_c3,
    )
    final, history, max_candidates = run_policy(
        scheduler._initial_state(), config=config
    )
    validated = validate_schedule_history(
        tuple(history),
        scheduler.token_dist,
        initial_cache_c2=initial_cache_c2,
        initial_cache_c3=initial_cache_c3,
    )
    if validated != final.g_score:
        raise RuntimeError(
            f"golden replay makespan {validated} != state score {final.g_score}"
        )
    serialized = [serialize_action(action) for action in history]
    history_blob = json.dumps(
        serialized, sort_keys=True, separators=(",", ":")
    ).encode()
    result = {
        "schema": "scheduler_policy_golden_run_v1",
        "policy_id": config.policy_id,
        "candidate_revision": config.candidate_revision,
        "rank_limit": config.rank_limit,
        "bottom_count": config.bottom_count,
        "candidate_budget": config.candidate_budget,
        "scorer_kind": config.scorer_kind,
        "pathmax_kind": config.pathmax_kind,
        "makespan_cc": int(final.g_score),
        "decisions": len(history),
        "max_candidates": max_candidates,
        "history_sha256": hashlib.sha256(history_blob).hexdigest(),
        "actions": serialized,
    }
    # Candidate timing helpers are intentionally cached within one run. Clear
    # them at the public case boundary so long regressions do not retain state
    # from thousands of unrelated distributions.
    clear_scheduler_caches()
    from derive_scheduler_policy import _equal_finish_left, _release_target_left

    _equal_finish_left.cache_clear()
    _release_target_left.cache_clear()
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def _run_suite_case(case: dict, config: PolicyConfig) -> tuple[str, dict]:
    case_id = str(case["case_id"])
    result = run_distribution(
        {int(eid): int(ntok) for eid, ntok in case["dist"].items()},
        initial_cache_c2=int(case.get("c2", -1)),
        initial_cache_c3=int(case.get("c3", -1)),
        config=config,
    )
    result.update(
        {
            "case_id": int(case["case_id"]),
            "e_total": int(case["e_total"]),
            "dataset_split": case.get("dataset_split"),
            "analysis_eligible": bool(case.get("analysis_eligible", True)),
        }
    )
    return case_id, result


def run_suite(
    suite_path: Path,
    out_path: Path,
    *,
    seeded_suite_out: Path | None,
    config: PolicyConfig,
    resume: bool,
    workers: int,
    progress_every: int,
    limit: int | None,
) -> dict:
    """Run the frozen policy over a coverage suite with atomic resume."""
    payload = json.loads(suite_path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("--suite must contain a strategy-coverage cases list")
    input_hash = _file_sha256(suite_path)
    configuration = {
        "suite": str(suite_path.resolve()),
        "suite_sha256": input_hash,
        "policy_id": config.policy_id,
        "candidate_revision": config.candidate_revision,
        "rank_limit": config.rank_limit,
        "bottom_count": config.bottom_count,
        "candidate_budget": config.candidate_budget,
        "scorer_kind": config.scorer_kind,
        "pathmax_kind": config.pathmax_kind,
    }
    if out_path.exists():
        if not resume:
            raise FileExistsError(f"{out_path} exists; pass --resume")
        report = json.loads(out_path.read_text())
        if report.get("configuration") != configuration:
            raise ValueError("suite checkpoint configuration changed")
    else:
        report = {
            "schema": "scheduler_policy_golden_suite_v1",
            "provisional": True,
            "configuration": configuration,
            "results": {},
        }
    results = report["results"]
    eligible = [case for case in payload["cases"] if case.get("analysis_eligible", True)]
    pending = [case for case in eligible if str(case["case_id"]) not in results]
    if limit is not None:
        pending = pending[:limit]
    started = time.perf_counter()
    completed_this_run = 0

    def checkpoint(provisional: bool) -> None:
        report.update(
            {
                "provisional": provisional,
                "completed_cases": len(results),
                "eligible_cases": len(eligible),
                "runtime_s_this_invocation": time.perf_counter() - started,
            }
        )
        _atomic_write_json(out_path, report)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_suite_case, case, config): case for case in pending
        }
        for future in as_completed(futures):
            case = futures[future]
            case_id, result = future.result()
            results[case_id] = result
            completed_this_run += 1
            if progress_every > 0 and completed_this_run % progress_every == 0:
                checkpoint(True)
                print(
                    f"golden-suite completed={len(results)}/{len(eligible)} "
                    f"new={completed_this_run} elapsed_s={time.perf_counter()-started:.1f}",
                    flush=True,
                )
    complete = len(results) == len(eligible)
    checkpoint(not complete)
    if seeded_suite_out is not None:
        if not complete:
            raise RuntimeError("cannot write seeded suite from an incomplete report")
        for case in payload["cases"]:
            if not case.get("analysis_eligible", True):
                continue
            result = results[str(case["case_id"])]
            case["incumbent"] = {
                "source": config.policy_id,
                "makespan_cc": int(result["makespan_cc"]),
                "actions": result["actions"],
            }
        payload.setdefault("meta", {}).update(
            {
                "incumbent_policy_id": config.policy_id,
                "incumbent_report": str(out_path.resolve()),
                "incumbent_report_sha256": _file_sha256(out_path),
            }
        )
        _atomic_write_json(seeded_suite_out, payload)
    print(
        f"golden-suite finished completed={len(results)}/{len(eligible)} "
        f"provisional={not complete} "
        f"wrote {out_path}",
        flush=True,
    )
    return report


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(fraction * len(ordered) + 0.999999) - 1))
    return ordered[index]


def _comparison_summary(rows: list[dict]) -> dict:
    if not rows:
        return {
            "cases": 0,
            "exact_to_reference_upper_bound": 0,
            "ratio_mean": None,
            "ratio_p95": None,
            "ratio_max": None,
            "regret_mean_cc": None,
            "regret_p95_cc": None,
            "regret_max_cc": None,
        }
    ratios = [float(row["ratio_to_reference_upper_bound"]) for row in rows]
    regrets = [int(row["regret_to_reference_upper_bound_cc"]) for row in rows]
    return {
        "cases": len(rows),
        "exact_to_reference_upper_bound": sum(regret == 0 for regret in regrets),
        "ratio_mean": sum(ratios) / len(ratios),
        "ratio_p95": _percentile(ratios, 0.95),
        "ratio_max": max(ratios),
        "regret_mean_cc": sum(regrets) / len(regrets),
        "regret_p95_cc": _percentile(regrets, 0.95),
        "regret_max_cc": max(regrets),
    }


def compare_suite_reports(
    golden_paths: list[Path],
    reference_paths: list[Path],
    out_path: Path,
    *,
    quality: str,
    target_gap: float,
) -> dict:
    """Compare frozen-policy runs with independent anytime-reference results.

    The reference makespan is an achieved upper bound.  It is the certified
    optimum only for rows marked ``proven_optimal``.  The report therefore
    names every nonnegative delta "regret to reference upper bound" and keeps
    proof/gap fields on every row rather than overstating search quality.
    """
    if len(golden_paths) != len(reference_paths):
        raise ValueError("--golden-report and --reference-file counts differ")
    if not golden_paths:
        raise ValueError("at least one report pair is required")
    if target_gap < 0:
        raise ValueError("--target-gap must be nonnegative")

    rows = []
    source_pairs = []
    policy_ids = set()
    for golden_path, reference_path in zip(golden_paths, reference_paths):
        golden = json.loads(golden_path.read_text())
        reference = json.loads(reference_path.read_text())
        if golden.get("schema") != "scheduler_policy_golden_suite_v1":
            raise ValueError(f"unexpected golden schema: {golden_path}")
        if golden.get("provisional", True):
            raise ValueError(f"golden report is incomplete: {golden_path}")
        golden_results = golden.get("results")
        reference_results = reference.get("results")
        if not isinstance(golden_results, dict) or not isinstance(reference_results, dict):
            raise ValueError("comparison inputs must contain result dictionaries")
        policy_ids.add(golden["configuration"]["policy_id"])
        source_pairs.append(
            {
                "golden_report": str(golden_path.resolve()),
                "golden_sha256": _file_sha256(golden_path),
                "reference_file": str(reference_path.resolve()),
                "reference_sha256": _file_sha256(reference_path),
            }
        )
        for case_id, policy_result in golden_results.items():
            reference_result = reference_results.get(str(case_id))
            if reference_result is None:
                continue
            if reference_result.get("status") != "ok":
                continue
            if not reference_result.get("analysis_eligible", True):
                continue
            proven = bool(reference_result.get("proven_optimal"))
            gap = float(reference_result.get("optimality_gap", float("inf")))
            if quality == "proven" and not proven:
                continue
            if quality == "within-gap" and not (proven or gap <= target_gap):
                continue
            policy_cc = int(policy_result["makespan_cc"])
            reference_cc = int(reference_result["makespan_cc"])
            regret_cc = policy_cc - reference_cc
            if regret_cc < 0:
                raise RuntimeError(
                    f"policy beats seeded reference upper bound for case {case_id}: "
                    f"{policy_cc} < {reference_cc}"
                )
            rows.append(
                {
                    "case_id": int(case_id),
                    "e_total": int(policy_result["e_total"]),
                    "policy_makespan_cc": policy_cc,
                    "reference_upper_bound_cc": reference_cc,
                    "reference_lower_bound_cc": int(reference_result["lower_bound_cc"]),
                    "reference_proven_optimal": proven,
                    "reference_optimality_gap": gap,
                    "reference_termination": reference_result.get("termination"),
                    "regret_to_reference_upper_bound_cc": regret_cc,
                    "ratio_to_reference_upper_bound": policy_cc / reference_cc,
                }
            )
    if len(policy_ids) != 1:
        raise ValueError(f"golden reports contain different policies: {policy_ids}")
    rows.sort(key=lambda row: (row["e_total"], row["case_id"]))
    by_e = {
        str(e_total): _comparison_summary(
            [row for row in rows if row["e_total"] == e_total]
        )
        for e_total in sorted({row["e_total"] for row in rows})
    }
    report = {
        "schema": "scheduler_policy_reference_comparison_v1",
        "policy_id": next(iter(policy_ids)),
        "quality_filter": quality,
        "target_gap": target_gap if quality == "within-gap" else None,
        "interpretation": (
            "reference makespan is an achieved upper bound; regret is certified "
            "optimal only where reference_proven_optimal is true"
        ),
        "source_pairs": source_pairs,
        "summary": {
            "overall": _comparison_summary(rows),
            "by_e_total": by_e,
            "proven_optimal_cases": sum(
                row["reference_proven_optimal"] for row in rows
            ),
        },
        "worst_rows": sorted(
            rows,
            key=lambda row: (
                -row["ratio_to_reference_upper_bound"],
                -row["regret_to_reference_upper_bound_cc"],
            ),
        )[:100],
        "rows": rows,
    }
    _atomic_write_json(out_path, report)
    print(
        f"comparison finished cases={len(rows)} policy={next(iter(policy_ids))} "
        f"wrote {out_path}",
        flush=True,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dist-json",
        help='expert-token object, for example \'{"0": 16, "1": 8}\'',
    )
    source.add_argument("--suite", type=Path)
    source.add_argument("--golden-report", action="append", type=Path)
    parser.add_argument("--initial-cache-c2", type=int, default=-1)
    parser.add_argument("--initial-cache-c3", type=int, default=-1)
    parser.add_argument(
        "--policy",
        choices=tuple(POLICY_CONFIGS),
        default="r4-b2-v1",
        help="frozen policy configuration; default preserves the v1 baseline",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seeded-suite-out", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reference-file", action="append", type=Path, default=[])
    parser.add_argument(
        "--quality",
        choices=("proven", "within-gap", "all-ok"),
        default="proven",
        help="reference rows admitted by comparison mode",
    )
    parser.add_argument("--target-gap", type=float, default=0.03)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    config = POLICY_CONFIGS[args.policy]
    if args.golden_report is not None:
        if args.out is None:
            raise ValueError("--golden-report requires --out")
        compare_suite_reports(
            args.golden_report,
            args.reference_file,
            args.out,
            quality=args.quality,
            target_gap=args.target_gap,
        )
        return 0
    if args.suite is not None:
        if args.out is None:
            raise ValueError("--suite requires --out")
        run_suite(
            args.suite,
            args.out,
            seeded_suite_out=args.seeded_suite_out,
            config=config,
            resume=args.resume,
            workers=args.workers,
            progress_every=args.progress_every,
            limit=args.limit,
        )
        return 0
    raw = json.loads(args.dist_json)
    if not isinstance(raw, dict):
        raise ValueError("--dist-json must decode to an object")
    result = run_distribution(
        {int(eid): int(ntok) for eid, ntok in raw.items()},
        initial_cache_c2=args.initial_cache_c2,
        initial_cache_c3=args.initial_cache_c3,
        config=config,
    )
    text = json.dumps(result, indent=2)
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(text + "\n")
        temporary.replace(args.out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
