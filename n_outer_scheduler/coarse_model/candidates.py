#!/usr/bin/env python3
"""Candidate semantics and bounded-window generation for coarse N-outer.

This module owns only action structure and N-outer phase modes.  It neither
imports nor mutates the four-stage scheduler.  An external candidate can be
injected by spelling the same SINGLE/PAIR/SPLIT slices as a
``CandidateSkeleton``; N-outer then chooses its own shape and DMA modes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterable, Sequence

from .semantics import (
    ALL_SHAPES,
    ActionKind,
    ExpertSlice,
    MacroActionPlan,
    MacroPhaseSpec,
    MacroTaskPlan,
    PhasePlan,
    ShapeSpec,
    DmaBinding,
    LANE_BW_BYTES_PER_CC,
    compute_block_cc,
    default_phases,
    legal_bindings,
)


@dataclass(frozen=True)
class RemainingExpert:
    eid: int
    token_start: int
    ntokens: int

    def __post_init__(self) -> None:
        ExpertSlice(self.eid, self.token_start, self.ntokens)

    @property
    def expert_slice(self) -> ExpertSlice:
        return ExpertSlice(self.eid, self.token_start, self.ntokens)


@dataclass(frozen=True)
class SliceAssignment:
    cluster: int
    expert_slice: ExpertSlice

    def __post_init__(self) -> None:
        if self.cluster not in (0, 1):
            raise ValueError("cluster must be 0 or 1")


@dataclass(frozen=True)
class CandidateSkeleton:
    """Action structure shared at the candidate-semantics boundary."""

    kind: ActionKind
    assignments: tuple[SliceAssignment, ...]

    def __post_init__(self) -> None:
        clusters = [item.cluster for item in self.assignments]
        if not self.assignments or len(self.assignments) > 2:
            raise ValueError("candidate has one or two assignments")
        if len(clusters) != len(set(clusters)):
            raise ValueError("candidate cannot assign a cluster twice")
        if self.kind == ActionKind.SINGLE and len(self.assignments) != 1:
            raise ValueError("SINGLE requires one assignment")
        if self.kind in (ActionKind.PAIR, ActionKind.SPLIT) and len(self.assignments) != 2:
            raise ValueError("PAIR/SPLIT require two assignments")
        if self.kind == ActionKind.PAIR:
            if self.assignments[0].expert_slice.eid == self.assignments[1].expert_slice.eid:
                raise ValueError("PAIR requires different experts")
        if self.kind == ActionKind.SPLIT:
            left, right = (item.expert_slice for item in self.assignments)
            if left.eid != right.eid:
                raise ValueError("SPLIT requires one expert")
            if max(left.token_start, right.token_start) < min(
                left.token_end, right.token_end
            ):
                raise ValueError("SPLIT slices overlap")

    @property
    def label(self) -> str:
        body = ",".join(
            f"c{item.cluster}:e{item.expert_slice.eid}"
            f"[{item.expert_slice.token_start}:"
            f"{item.expert_slice.token_end}]"
            for item in self.assignments
        )
        return f"{self.kind.value}({body})"


@dataclass(frozen=True)
class WindowSpec:
    top: int
    bottom: int

    def __post_init__(self) -> None:
        if self.top < 0 or self.bottom < 0 or self.top + self.bottom <= 0:
            raise ValueError("window must contain at least one visible expert")


def visible_experts(
    remaining: Sequence[RemainingExpert], window: WindowSpec
) -> tuple[RemainingExpert, ...]:
    """Return a stable top+bottom union ranked by remaining token count."""

    ranked = sorted(remaining, key=lambda item: (-item.ntokens, item.eid))
    chosen = ranked[: window.top]
    if window.bottom:
        selected = {item.eid for item in chosen}
        chosen.extend(
            item
            for item in reversed(ranked)
            if item.eid not in selected
            and len(chosen) < min(len(ranked), window.top + window.bottom)
        )
    return tuple(chosen)


def balanced_split_cuts(ntokens: int) -> tuple[int, ...]:
    """Small RTL-oriented split bank; actual slices never overlap or pad."""

    if ntokens < 2:
        return ()
    cuts = {ntokens // 2, (ntokens + 1) // 2}
    for multiple in (2, 4, 8):
        lower = (ntokens // (2 * multiple)) * multiple
        upper = ((ntokens + 2 * multiple - 1) // (2 * multiple)) * multiple
        cuts.update((lower, upper))
    return tuple(sorted(cut for cut in cuts if 0 < cut < ntokens))


def generate_skeletons(
    remaining: Sequence[RemainingExpert],
    *,
    window: WindowSpec,
    split_cuts: str = "balanced",
) -> tuple[CandidateSkeleton, ...]:
    """Generate legal action structures without shape/DMA cross-products."""

    visible = visible_experts(remaining, window)
    result: list[CandidateSkeleton] = []
    for expert in visible:
        for cluster in (0, 1):
            result.append(
                CandidateSkeleton(
                    ActionKind.SINGLE,
                    (SliceAssignment(cluster, expert.expert_slice),),
                )
            )
    for left, right in combinations(visible, 2):
        result.extend(
            (
                CandidateSkeleton(
                    ActionKind.PAIR,
                    (
                        SliceAssignment(0, left.expert_slice),
                        SliceAssignment(1, right.expert_slice),
                    ),
                ),
                CandidateSkeleton(
                    ActionKind.PAIR,
                    (
                        SliceAssignment(0, right.expert_slice),
                        SliceAssignment(1, left.expert_slice),
                    ),
                ),
            )
        )
    for expert in visible:
        if split_cuts == "all":
            cuts: Iterable[int] = range(1, expert.ntokens)
        elif split_cuts == "balanced":
            cuts = balanced_split_cuts(expert.ntokens)
        else:
            raise ValueError("split_cuts must be 'balanced' or 'all'")
        for cut in cuts:
            left = ExpertSlice(expert.eid, expert.token_start, cut)
            right = ExpertSlice(
                expert.eid,
                expert.token_start + cut,
                expert.ntokens - cut,
            )
            result.extend(
                (
                    CandidateSkeleton(
                        ActionKind.SPLIT,
                        (SliceAssignment(0, left), SliceAssignment(1, right)),
                    ),
                    CandidateSkeleton(
                        ActionKind.SPLIT,
                        (SliceAssignment(0, right), SliceAssignment(1, left)),
                    ),
                )
            )
    return tuple(result)


def canonical_shape(ntokens: int, phase: MacroPhaseSpec) -> ShapeSpec:
    """Fastest shape, then the largest M tile among exact timing ties."""

    return min(
        ALL_SHAPES,
        key=lambda shape: (compute_block_cc(ntokens, shape, phase), -shape.m_dim),
    )


def phase_plan_options(
    ntokens: int, phase: MacroPhaseSpec
) -> tuple[PhasePlan, ...]:
    shape = canonical_shape(ntokens, phase)
    return tuple(
        PhasePlan(shape, binding)
        for binding in legal_bindings(ntokens, shape, phase)
    )


def materialize_modes(
    skeleton: CandidateSkeleton,
    *,
    phases: tuple[MacroPhaseSpec, MacroPhaseSpec] | None = None,
) -> tuple[MacroActionPlan, ...]:
    """Attach non-dominated N-outer phase modes to an action skeleton."""

    gate_up, down = phases or default_phases()
    task_options: list[tuple[MacroTaskPlan, ...]] = []
    for assignment in skeleton.assignments:
        ntokens = assignment.expert_slice.ntokens
        task_options.append(
            tuple(
                MacroTaskPlan(
                    assignment.cluster,
                    assignment.expert_slice,
                    gate_plan,
                    down_plan,
                )
                for gate_plan, down_plan in product(
                    phase_plan_options(ntokens, gate_up),
                    phase_plan_options(ntokens, down),
                )
            )
        )
    return tuple(
        MacroActionPlan(skeleton.kind, tuple(tasks))
        for tasks in product(*task_options)
    )


def bounded_joint_mode_bank(
    skeleton: CandidateSkeleton,
    *,
    budget: int = 8,
    phases: tuple[MacroPhaseSpec, MacroPhaseSpec] | None = None,
) -> tuple[MacroActionPlan, ...]:
    """Keep a small, category-diverse phase-mode bank for one skeleton."""

    if budget <= 0:
        raise ValueError("mode budget must be positive")
    phase_specs = phases or default_phases()
    plans = materialize_modes(skeleton, phases=phase_specs)

    def category(plan: MacroActionPlan) -> tuple[str, str]:
        labels = []
        for phase_name in ("gate_up", "down"):
            bindings = [getattr(task, phase_name).dma for task in plan.tasks]
            if len(bindings) == 1:
                labels.append(
                    "both" if bindings[0] == DmaBinding.BOTH else "single"
                )
            elif all(binding == DmaBinding.BOTH for binding in bindings):
                labels.append("both_both")
            elif all(
                binding in (DmaBinding.IDMA, DmaBinding.XDMA)
                for binding in bindings
            ) and bindings[0] != bindings[1]:
                labels.append("parallel_single")
            elif DmaBinding.BOTH in bindings:
                labels.append("mixed_both_single")
            else:
                labels.append("same_single")
        return tuple(labels)

    def plan_rank(plan: MacroActionPlan) -> tuple[object, ...]:
        exposed_stall = 0
        same_lane = 0
        for task in plan.tasks:
            for phase_plan, phase in zip(
                (task.gate_up, task.down), phase_specs
            ):
                compute_cc = compute_block_cc(
                    task.expert_slice.ntokens, phase_plan.shape, phase
                )
                if phase_plan.dma != DmaBinding.BOTH:
                    load_cc = math.ceil(
                        phase.weight_block_bytes / LANE_BW_BYTES_PER_CC
                    )
                    exposed_stall += max(0, load_cc - compute_cc)
        if len(plan.tasks) == 2:
            for phase_name in ("gate_up", "down"):
                left = getattr(plan.tasks[0], phase_name).dma
                right = getattr(plan.tasks[1], phase_name).dma
                same_lane += int(
                    left == right and left in (DmaBinding.IDMA, DmaBinding.XDMA)
                )
        return (
            exposed_stall,
            same_lane,
            tuple(
                (task.gate_up.dma.value, task.down.dma.value)
                for task in plan.tasks
            ),
        )

    ordered = sorted(plans, key=plan_rank)

    def is_fixed_lane_baseline(plan: MacroActionPlan) -> bool:
        return all(
            task.gate_up.dma
            == (DmaBinding.IDMA if task.cluster == 0 else DmaBinding.XDMA)
            and task.down.dma
            == (DmaBinding.IDMA if task.cluster == 0 else DmaBinding.XDMA)
            for task in plan.tasks
        )

    category_order = (
        ("both_both", "both_both"),
        ("parallel_single", "parallel_single"),
        ("mixed_both_single", "mixed_both_single"),
        ("both", "both"),
        ("single", "single"),
    )
    selected: list[MacroActionPlan] = []
    used: set[int] = set()
    # The first entry is the auditable no-contention fallback.  Keeping it in
    # every bounded bank makes a local guard implementable and gives a stable
    # bank-order tie break.  This is especially important for a cluster-1
    # SINGLE M2 action, where XDMA/XDMA was not guaranteed by category-only
    # truncation.
    for index, plan in enumerate(ordered):
        if is_fixed_lane_baseline(plan):
            selected.append(plan)
            used.add(index)
            break
    if not selected:
        raise AssertionError("bounded mode bank lost its fixed-lane baseline")
    for wanted in category_order:
        for index, plan in enumerate(ordered):
            if index not in used and category(plan) == wanted:
                selected.append(plan)
                used.add(index)
                break
    selected.extend(plan for index, plan in enumerate(ordered) if index not in used)
    return tuple(selected[:budget])


def rtl_symmetric_mode_bank(
    skeleton: CandidateSkeleton,
    *,
    phases: tuple[MacroPhaseSpec, MacroPhaseSpec] | None = None,
) -> tuple[MacroActionPlan, ...]:
    """Return the minimal directly-generatable RTL mode bank.

    Entry 0 is always the no-contention fixed-lane plan: cluster 0 uses iDMA
    and cluster 1 uses xDMA in both phases.  Entry 1 exists only when every
    task/phase can legally use BOTH, in which case all tasks use BOTH.  The
    bank deliberately contains no one-sided or one-phase BOTH combinations.

    This is a structural generator, not a truncation of the larger analysis
    bank.  A future RTL therefore needs to construct and compare at most two
    mode variants for a fixed SINGLE/PAIR skeleton.
    """

    gate_up, down = phases or default_phases()
    phase_specs = (gate_up, down)
    fixed_tasks: list[MacroTaskPlan] = []
    both_tasks: list[MacroTaskPlan] = []
    both_legal = True
    for assignment in skeleton.assignments:
        ntokens = assignment.expert_slice.ntokens
        shapes = tuple(canonical_shape(ntokens, phase) for phase in phase_specs)
        fixed_binding = (
            DmaBinding.IDMA if assignment.cluster == 0 else DmaBinding.XDMA
        )
        fixed_tasks.append(
            MacroTaskPlan(
                assignment.cluster,
                assignment.expert_slice,
                PhasePlan(shapes[0], fixed_binding),
                PhasePlan(shapes[1], fixed_binding),
            )
        )
        legal_both = all(
            DmaBinding.BOTH in legal_bindings(ntokens, shape, phase)
            for shape, phase in zip(shapes, phase_specs)
        )
        both_legal &= legal_both
        both_tasks.append(
            MacroTaskPlan(
                assignment.cluster,
                assignment.expert_slice,
                PhasePlan(shapes[0], DmaBinding.BOTH),
                PhasePlan(shapes[1], DmaBinding.BOTH),
            )
        )
    result = [MacroActionPlan(skeleton.kind, tuple(fixed_tasks))]
    if both_legal:
        result.append(MacroActionPlan(skeleton.kind, tuple(both_tasks)))
    return tuple(result)


def plan_matches_pair_mode_policy(
    skeleton: CandidateSkeleton,
    plan: MacroActionPlan,
    policy: str,
) -> bool:
    """Apply the static PAIR/SPLIT ablation policy to one materialized plan."""

    if policy not in ("all", "no_mixed", "fixed_only"):
        raise ValueError("policy must be all, no_mixed, or fixed_only")
    if skeleton.kind == ActionKind.SINGLE or policy == "all":
        return True
    fixed = all(
        task.gate_up.dma
        == (DmaBinding.IDMA if task.cluster == 0 else DmaBinding.XDMA)
        and task.down.dma
        == (DmaBinding.IDMA if task.cluster == 0 else DmaBinding.XDMA)
        for task in plan.tasks
    )
    if policy == "fixed_only":
        return fixed
    return all(
        sum(
            getattr(task, phase_name).dma == DmaBinding.BOTH
            for task in plan.tasks
        )
        != 1
        for phase_name in ("gate_up", "down")
    )


def consume_candidate(
    remaining: Sequence[RemainingExpert], skeleton: CandidateSkeleton
) -> tuple[RemainingExpert, ...]:
    """Remove exactly the experts completely covered by one action."""

    by_eid = {item.eid: item for item in remaining}
    covered: dict[int, list[ExpertSlice]] = {}
    for assignment in skeleton.assignments:
        covered.setdefault(assignment.expert_slice.eid, []).append(
            assignment.expert_slice
        )
    for eid, slices in covered.items():
        original = by_eid.get(eid)
        if original is None:
            raise ValueError(f"candidate references absent expert {eid}")
        ordered = sorted(slices, key=lambda item: item.token_start)
        cursor = original.token_start
        for item in ordered:
            if item.token_start != cursor:
                raise ValueError("candidate slices leave a gap or overlap")
            cursor = item.token_end
        if cursor != original.token_start + original.ntokens:
            raise ValueError("candidate does not cover the complete expert")
    consumed = set(covered)
    return tuple(item for item in remaining if item.eid not in consumed)
