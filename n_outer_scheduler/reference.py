#!/usr/bin/env python3
"""Exact DMA-grant search for one fixed N-outer candidate.

The outer candidate fixes expert-to-cluster assignment and expert order.  This
module then searches every non-dominated grant choice at DMA event boundaries:

* one pending load may use one lane or BOTH lanes;
* two pending loads may SPLIT the lanes or run serially with BOTH;
* a free lane may deliberately wait for the next running event when immediate
  service could block a better future BOTH grant.

Compute always starts as soon as its weight is ready.  This is safe because a
cluster's compute resource is private and earlier compute completion can only
release ping/pong buffers earlier.  States are normalized to remaining times,
so schedules that reach the same future-visible state share one exact suffix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .model import (
    LANE_BW_BYTES_PER_CC,
    CandidateResult,
    ComputeRecord,
    LoadRecord,
    NOuterSimulator,
    ScheduleCandidate,
    WorkItem,
    _dma_duration,
)


LaneGrant = tuple[int, int]
GrantAction = tuple[LaneGrant, ...]
RunningLoad = tuple[int, int]  # remaining cycles, lane mask


@dataclass(frozen=True)
class PlanStep:
    grants: GrantAction
    advance_cc: int


@dataclass(frozen=True)
class ExactDmaResult:
    candidate: ScheduleCandidate
    makespan_cc: int
    relaxed_lower_bound_cc: int
    heuristic_makespan_cc: int
    expanded_states: int
    memoized_states: int
    decisions: tuple[PlanStep, ...]
    schedule: CandidateResult
    proven_optimal: bool
    plan_validated: bool

    @property
    def heuristic_gap_cc(self) -> int:
        return self.heuristic_makespan_cc - self.makespan_cc


@dataclass(frozen=True)
class _State:
    load_launched: tuple[int, int]
    load_completed: tuple[int, int]
    load_running: tuple[RunningLoad | None, RunningLoad | None]
    compute_completed: tuple[int, int]
    compute_running: tuple[int | None, int | None]


@dataclass(frozen=True)
class _Suffix:
    remaining_cc: int
    decisions: tuple[PlanStep, ...]


class ExactDmaPlanner:
    """Prove the minimum makespan for one fixed cluster/order candidate."""

    def __init__(self, simulator: NOuterSimulator):
        self.simulator = simulator
        self._streams: tuple[tuple[WorkItem, ...], ...] = ((), ())
        self._memo: dict[_State, _Suffix] = {}
        self._expanded_states = 0

    def evaluate(self, candidate: ScheduleCandidate) -> ExactDmaResult:
        self.simulator._validate_group_capacity(candidate.group)
        self._streams = self.simulator.build_streams(candidate.group)
        self._memo = {}
        self._expanded_states = 0

        initial = self._prime_computes(
            _State(
                load_launched=(0, 0),
                load_completed=(0, 0),
                load_running=(None, None),
                compute_completed=(0, 0),
                compute_running=(None, None),
            )
        )
        lower_bound = self._lower_bound(initial)
        suffix = self._solve(initial)
        self._validate_plan(initial, suffix.decisions, suffix.remaining_cc)
        loads, computes = self._replay_history(initial, suffix.decisions)
        schedule = self.simulator.result_from_history(
            candidate,
            self._streams,
            loads,
            computes,
            expected_makespan=suffix.remaining_cc,
        )

        heuristic: CandidateResult = self.simulator.evaluate(candidate)
        if suffix.remaining_cc > heuristic.makespan_cc:
            raise AssertionError("exact DMA search is worse than its feasible heuristic UB")
        return ExactDmaResult(
            candidate=candidate,
            makespan_cc=suffix.remaining_cc,
            relaxed_lower_bound_cc=lower_bound,
            heuristic_makespan_cc=heuristic.makespan_cc,
            expanded_states=self._expanded_states,
            memoized_states=len(self._memo),
            decisions=suffix.decisions,
            schedule=schedule,
            proven_optimal=True,
            plan_validated=True,
        )

    def _solve(self, state: _State) -> _Suffix:
        cached = self._memo.get(state)
        if cached is not None:
            return cached
        if self._done(state):
            result = _Suffix(0, ())
            self._memo[state] = result
            return result

        self._expanded_states += 1
        best_cost = math.inf
        best_steps: tuple[PlanStep, ...] | None = None
        for action in self._ordered_actions(state):
            launched = self._launch(state, action)
            advanced, delta = self._advance(launched)
            if delta + self._lower_bound(advanced) >= best_cost:
                continue
            suffix = self._solve(advanced)
            cost = delta + suffix.remaining_cc
            if cost < best_cost:
                best_cost = cost
                best_steps = (PlanStep(action, delta), *suffix.decisions)

        if best_steps is None:
            raise RuntimeError("exact N-outer DMA search reached a dead end")
        result = _Suffix(int(best_cost), best_steps)
        self._memo[state] = result
        return result

    def _prime_computes(self, state: _State) -> _State:
        running = list(state.compute_running)
        changed = False
        for cluster in (0, 1):
            index = state.compute_completed[cluster]
            if running[cluster] is not None or index >= len(self._streams[cluster]):
                continue
            if state.load_completed[cluster] <= index:
                continue
            running[cluster] = self._streams[cluster][index].compute_cc
            changed = True
        if not changed:
            return state
        return _State(
            load_launched=state.load_launched,
            load_completed=state.load_completed,
            load_running=state.load_running,
            compute_completed=state.compute_completed,
            compute_running=tuple(running),
        )

    def _done(self, state: _State) -> bool:
        return all(
            state.compute_completed[cluster] == len(self._streams[cluster])
            and state.compute_running[cluster] is None
            for cluster in (0, 1)
        )

    def _eligible_clusters(self, state: _State) -> tuple[int, ...]:
        eligible: list[int] = []
        for cluster in (0, 1):
            index = state.load_launched[cluster]
            if state.load_running[cluster] is not None:
                continue
            if index >= len(self._streams[cluster]):
                continue
            if index >= 2 and state.compute_completed[cluster] < index - 1:
                continue
            eligible.append(cluster)
        return tuple(eligible)

    @staticmethod
    def _occupied_mask(state: _State) -> int:
        mask = 0
        for running in state.load_running:
            if running is not None:
                mask |= running[1]
        return mask

    def _ordered_actions(self, state: _State) -> tuple[GrantAction, ...]:
        eligible = self._eligible_clusters(state)
        free_mask = 0b11 ^ self._occupied_mask(state)
        has_event = any(run is not None for run in state.load_running) or any(
            run is not None for run in state.compute_running
        )

        if not eligible or free_mask == 0:
            if not has_event:
                raise RuntimeError("no DMA action and no future event")
            return ((),)

        initial = state.load_launched == (0, 0) and state.load_completed == (0, 0)
        if (
            initial
            and self.simulator.config.force_initial_split
            and len(self._streams[0]) > 0
            and len(self._streams[1]) > 0
        ):
            return (((0, 0b01), (1, 0b10)),)

        actions: list[GrantAction] = []
        if free_mask == 0b11 and len(eligible) == 2:
            first, second = eligible
            actions.extend(
                (
                    ((first, 0b01), (second, 0b10)),
                    ((first, 0b11),),
                    ((second, 0b11),),
                )
            )
        elif free_mask == 0b11:
            cluster = eligible[0]
            single = ((cluster, 0b01),)
            both = ((cluster, 0b11),)
            deadline = state.compute_running[cluster] or 0
            single_duration = _dma_duration(
                self._streams[cluster][state.load_launched[cluster]].weight_bytes, 1
            )
            actions.extend((both, single) if single_duration > deadline else (single, both))
            if has_event:
                actions.append(())
        else:
            cluster = eligible[0]
            actions.append(((cluster, free_mask),))
            if has_event:
                actions.append(())
        return tuple(actions)

    def _launch(self, state: _State, action: GrantAction) -> _State:
        occupied = self._occupied_mask(state)
        launched = list(state.load_launched)
        running = list(state.load_running)
        granted_mask = 0
        granted_clusters: set[int] = set()
        for cluster, lane_mask in action:
            if cluster in granted_clusters or cluster not in self._eligible_clusters(state):
                raise AssertionError("plan contains an ineligible or duplicate DMA grant")
            if lane_mask not in (0b01, 0b10, 0b11):
                raise AssertionError("plan contains an invalid DMA lane mask")
            if lane_mask & (occupied | granted_mask):
                raise AssertionError("plan overlaps DMA lane ownership")
            item = self._streams[cluster][launched[cluster]]
            running[cluster] = (
                _dma_duration(item.weight_bytes, lane_mask.bit_count()),
                lane_mask,
            )
            launched[cluster] += 1
            granted_mask |= lane_mask
            granted_clusters.add(cluster)
        return _State(
            load_launched=tuple(launched),
            load_completed=state.load_completed,
            load_running=tuple(running),
            compute_completed=state.compute_completed,
            compute_running=state.compute_running,
        )

    def _advance(self, state: _State) -> tuple[_State, int]:
        times = [run[0] for run in state.load_running if run is not None]
        times.extend(run for run in state.compute_running if run is not None)
        if not times:
            raise RuntimeError("cannot advance an event-free state")
        delta = min(times)

        load_completed = list(state.load_completed)
        load_running: list[RunningLoad | None] = []
        for cluster, running in enumerate(state.load_running):
            if running is None:
                load_running.append(None)
            elif running[0] == delta:
                load_completed[cluster] += 1
                load_running.append(None)
            else:
                load_running.append((running[0] - delta, running[1]))

        compute_completed = list(state.compute_completed)
        compute_running: list[int | None] = []
        for cluster, running in enumerate(state.compute_running):
            if running is None:
                compute_running.append(None)
            elif running == delta:
                compute_completed[cluster] += 1
                compute_running.append(None)
            else:
                compute_running.append(running - delta)

        advanced = _State(
            load_launched=state.load_launched,
            load_completed=tuple(load_completed),
            load_running=tuple(load_running),
            compute_completed=tuple(compute_completed),
            compute_running=tuple(compute_running),
        )
        return self._prime_computes(advanced), delta

    def _lower_bound(self, state: _State) -> int:
        compute_bounds: list[int] = []
        for cluster in (0, 1):
            running = state.compute_running[cluster]
            next_index = state.compute_completed[cluster]
            bound = 0
            if running is not None:
                bound += running
                next_index += 1
            bound += sum(
                item.compute_cc for item in self._streams[cluster][next_index:]
            )
            compute_bounds.append(bound)

        remaining_dma_bytes = 0
        for running in state.load_running:
            if running is not None:
                remaining_dma_bytes += (
                    running[0]
                    * LANE_BW_BYTES_PER_CC
                    * running[1].bit_count()
                )
        for cluster in (0, 1):
            remaining_dma_bytes += sum(
                item.weight_bytes
                for item in self._streams[cluster][state.load_launched[cluster]:]
            )
        dma_bound = math.ceil(remaining_dma_bytes / (2 * LANE_BW_BYTES_PER_CC))
        return max(*compute_bounds, dma_bound)

    def _validate_plan(
        self,
        initial: _State,
        decisions: tuple[PlanStep, ...],
        expected_makespan: int,
    ) -> None:
        state = initial
        elapsed = 0
        for step in decisions:
            if step.grants not in self._ordered_actions(state):
                raise AssertionError("exact plan contains an action outside the legal set")
            state, delta = self._advance(self._launch(state, step.grants))
            if delta != step.advance_cc:
                raise AssertionError("exact plan has an inconsistent event delta")
            elapsed += delta
        if not self._done(state) or elapsed != expected_makespan:
            raise AssertionError("exact plan replay did not reach the declared goal")

    def _replay_history(
        self,
        initial: _State,
        decisions: tuple[PlanStep, ...],
    ) -> tuple[tuple[LoadRecord, ...], tuple[ComputeRecord, ...]]:
        state = initial
        elapsed = 0
        loads: list[LoadRecord] = []
        computes: list[ComputeRecord] = []

        for step in decisions:
            for cluster, lane_mask in step.grants:
                item = self._streams[cluster][state.load_launched[cluster]]
                duration = _dma_duration(item.weight_bytes, lane_mask.bit_count())
                loads.append(
                    LoadRecord(
                        item=item,
                        start_cc=elapsed,
                        end_cc=elapsed + duration,
                        lanes=tuple(
                            lane for lane in (0, 1) if lane_mask & (1 << lane)
                        ),
                    )
                )

            launched = self._launch(state, step.grants)
            advanced, delta = self._advance(launched)
            start_cc = elapsed + delta
            for cluster in (0, 1):
                old_running = launched.compute_running[cluster]
                new_running = advanced.compute_running[cluster]
                started = new_running is not None and (
                    old_running is None or old_running == delta
                )
                if not started:
                    continue
                index = advanced.compute_completed[cluster]
                item = self._streams[cluster][index]
                computes.append(
                    ComputeRecord(
                        item=item,
                        start_cc=start_cc,
                        end_cc=start_cc + item.compute_cc,
                    )
                )
            state = advanced
            elapsed = start_cc

        return tuple(loads), tuple(computes)
