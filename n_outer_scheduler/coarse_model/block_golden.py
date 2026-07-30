#!/usr/bin/env python3
"""Independent block-level execution golden for coarse N-outer histories.

The main scheduler never generates block candidates.  This module expands a
selected macro history only for calibration and lowering verification.  Its
stream order is expert -> phase -> block, and the alternate ping/pong buffer
may contain only the unique next item in that stream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .search import SelectedStep
from .semantics import (
    DmaBinding,
    ExpertSlice,
    LANE_BW_BYTES_PER_CC,
    MacroPhaseSpec,
    ShapeSpec,
    compute_block_cc,
    default_phases,
    dma_duration,
)


class ArbitrationPolicy(str, Enum):
    DEADLINE = "deadline"
    BOTH_FIRST = "both_first"
    MACRO_ORDER = "macro_order"


@dataclass(frozen=True)
class BlockItem:
    cluster: int
    stream_index: int
    step_index: int
    eid: int
    token_start: int
    ntokens: int
    phase_name: str
    phase_index: int
    block_id: int
    block_count: int
    weight_bytes: int
    compute_cc: int
    shape: ShapeSpec
    binding: DmaBinding
    service_rank: int

    @property
    def buffer_slot(self) -> int:
        return self.stream_index & 1


@dataclass(frozen=True)
class GoldenLoad:
    item: BlockItem
    start_cc: int
    end_cc: int


@dataclass(frozen=True)
class GoldenCompute:
    item: BlockItem
    start_cc: int
    end_cc: int


@dataclass(frozen=True)
class GoldenResult:
    policy: ArbitrationPolicy
    streams: tuple[tuple[BlockItem, ...], tuple[BlockItem, ...]]
    loads: tuple[GoldenLoad, ...]
    computes: tuple[GoldenCompute, ...]
    makespan_cc: int
    initial_fill_cc: tuple[int, int]
    steady_stall_cc: tuple[int, int]
    history_validated: bool


@dataclass
class _ClusterState:
    items: tuple[BlockItem, ...]
    next_load: int = 0
    next_compute: int = 0
    running_load: GoldenLoad | None = None
    running_compute: GoldenCompute | None = None


def _operation_name(phase_name: str, block_id: int) -> str:
    prefix = "gate_up" if phase_name == "gate_up" else "down"
    return f"{prefix}_{'first' if block_id == 0 else 'stream'}"


def build_block_streams(
    history: Sequence[SelectedStep],
    *,
    phases: tuple[MacroPhaseSpec, MacroPhaseSpec] | None = None,
) -> tuple[tuple[BlockItem, ...], tuple[BlockItem, ...]]:
    phase_specs = phases or default_phases()
    streams: list[list[BlockItem]] = [[], []]
    for step_index, step in enumerate(history):
        order_rank = {operation: rank for rank, operation in enumerate(step.timing.service_order)}
        for task in step.plan.tasks:
            cluster = task.cluster
            for phase_index, (phase, phase_plan) in enumerate(
                zip(phase_specs, (task.gate_up, task.down))
            ):
                for block_id in range(phase.block_count):
                    operation = (cluster, _operation_name(phase.name, block_id))
                    streams[cluster].append(
                        BlockItem(
                            cluster=cluster,
                            stream_index=len(streams[cluster]),
                            step_index=step_index,
                            eid=task.expert_slice.eid,
                            token_start=task.expert_slice.token_start,
                            ntokens=task.expert_slice.ntokens,
                            phase_name=phase.name,
                            phase_index=phase_index,
                            block_id=block_id,
                            block_count=phase.block_count,
                            weight_bytes=phase.weight_block_bytes,
                            compute_cc=compute_block_cc(
                                task.expert_slice.ntokens, phase_plan.shape, phase
                            ),
                            shape=phase_plan.shape,
                            binding=phase_plan.dma,
                            service_rank=order_rank[operation],
                        )
                    )
    return tuple(streams[0]), tuple(streams[1])


def _required_lanes(binding: DmaBinding) -> tuple[int, ...]:
    lanes = []
    if binding & DmaBinding.IDMA:
        lanes.append(0)
    if binding & DmaBinding.XDMA:
        lanes.append(1)
    return tuple(lanes)


def _load_ready(state: _ClusterState, completed_compute: dict[int, GoldenCompute]) -> bool:
    index = state.next_load
    if state.running_load is not None or index >= len(state.items):
        return False
    return index < 2 or index - 2 in completed_compute


def _request_key(
    item: BlockItem,
    state: _ClusterState,
    now: int,
    policy: ArbitrationPolicy,
) -> tuple[int, ...]:
    if state.running_compute is not None:
        deadline = state.running_compute.end_cc
    elif state.next_compute < len(state.items):
        deadline = now
    else:
        deadline = now
    both = int(item.binding != DmaBinding.BOTH)
    if policy == ArbitrationPolicy.DEADLINE:
        return (deadline, item.service_rank, both, item.cluster)
    if policy == ArbitrationPolicy.BOTH_FIRST:
        return (both, deadline, item.service_rank, item.cluster)
    return (item.step_index, deadline, item.service_rank, both, item.cluster)


def replay_block_history(
    history: Sequence[SelectedStep],
    *,
    policy: ArbitrationPolicy = ArbitrationPolicy.DEADLINE,
    phases: tuple[MacroPhaseSpec, MacroPhaseSpec] | None = None,
) -> GoldenResult:
    streams = build_block_streams(history, phases=phases)
    return replay_block_streams(streams, policy=policy)


def replay_block_streams(
    streams: tuple[tuple[BlockItem, ...], tuple[BlockItem, ...]],
    *,
    policy: ArbitrationPolicy = ArbitrationPolicy.DEADLINE,
) -> GoldenResult:
    """Replay self-contained block parameters without a source macro history."""

    states = [_ClusterState(streams[0]), _ClusterState(streams[1])]
    completed_load: list[dict[int, GoldenLoad]] = [{}, {}]
    completed_compute: list[dict[int, GoldenCompute]] = [{}, {}]
    loads: list[GoldenLoad] = []
    computes: list[GoldenCompute] = []
    now = 0

    def all_done() -> bool:
        return all(
            state.next_compute == len(state.items)
            and state.running_compute is None
            and state.next_load == len(state.items)
            and state.running_load is None
            for state in states
        )

    while not all_done():
        for cluster, state in enumerate(states):
            if state.running_load is not None and state.running_load.end_cc == now:
                record = state.running_load
                completed_load[cluster][record.item.stream_index] = record
                state.running_load = None
            if state.running_compute is not None and state.running_compute.end_cc == now:
                record = state.running_compute
                completed_compute[cluster][record.item.stream_index] = record
                state.running_compute = None

        for cluster, state in enumerate(states):
            index = state.next_compute
            if state.running_compute is not None or index >= len(state.items):
                continue
            if index not in completed_load[cluster]:
                continue
            if index and index - 1 not in completed_compute[cluster]:
                continue
            item = state.items[index]
            record = GoldenCompute(item, now, now + item.compute_cc)
            computes.append(record)
            state.running_compute = record
            state.next_compute += 1

        occupied = {
            lane
            for state in states
            if state.running_load is not None
            for lane in _required_lanes(state.running_load.item.binding)
        }
        pending = [
            (state.items[state.next_load], state)
            for cluster, state in enumerate(states)
            if _load_ready(state, completed_compute[cluster])
        ]
        pending.sort(key=lambda pair: _request_key(pair[0], pair[1], now, policy))
        for item, state in pending:
            required = _required_lanes(item.binding)
            if any(lane in occupied for lane in required):
                continue
            duration = dma_duration(item.weight_bytes, item.binding)
            record = GoldenLoad(item, now, now + duration)
            loads.append(record)
            state.running_load = record
            state.next_load += 1
            occupied.update(required)

        if all_done():
            break
        events = [
            record.end_cc
            for state in states
            for record in (state.running_load, state.running_compute)
            if record is not None and record.end_cc > now
        ]
        if not events:
            raise RuntimeError("block golden deadlocked")
        now = min(events)

    result = _build_result(policy, streams, loads, computes)
    validate_block_result(result)
    return GoldenResult(**{**result.__dict__, "history_validated": True})


def replay_best_policy(
    history: Sequence[SelectedStep],
    *,
    phases: tuple[MacroPhaseSpec, MacroPhaseSpec] | None = None,
) -> GoldenResult:
    results = [
        replay_block_history(history, policy=policy, phases=phases)
        for policy in ArbitrationPolicy
    ]
    return min(
        results,
        key=lambda item: (
            item.makespan_cc,
            sum(item.steady_stall_cc),
            item.policy.value,
        ),
    )


def _build_result(
    policy: ArbitrationPolicy,
    streams: tuple[tuple[BlockItem, ...], tuple[BlockItem, ...]],
    loads: Sequence[GoldenLoad],
    computes: Sequence[GoldenCompute],
) -> GoldenResult:
    by_compute = {
        (record.item.cluster, record.item.stream_index): record
        for record in computes
    }
    initial: list[int] = []
    steady: list[int] = []
    for cluster, stream in enumerate(streams):
        if not stream:
            initial.append(0)
            steady.append(0)
            continue
        initial.append(by_compute[(cluster, 0)].start_cc)
        steady.append(
            sum(
                by_compute[(cluster, index)].start_cc
                - by_compute[(cluster, index - 1)].end_cc
                for index in range(1, len(stream))
            )
        )
    makespan = max(
        [record.end_cc for record in loads]
        + [record.end_cc for record in computes]
        + [0]
    )
    return GoldenResult(
        policy=policy,
        streams=streams,
        loads=tuple(loads),
        computes=tuple(computes),
        makespan_cc=makespan,
        initial_fill_cc=tuple(initial),
        steady_stall_cc=tuple(steady),
        history_validated=False,
    )


def validate_block_result(result: GoldenResult) -> None:
    expected = {
        (item.cluster, item.stream_index)
        for stream in result.streams
        for item in stream
    }
    loads = {(item.item.cluster, item.item.stream_index): item for item in result.loads}
    computes = {
        (item.item.cluster, item.item.stream_index): item
        for item in result.computes
    }
    if set(loads) != expected or set(computes) != expected:
        raise AssertionError("golden does not contain one load/compute per block")
    for cluster, stream in enumerate(result.streams):
        for index, item in enumerate(stream):
            load = loads[(cluster, index)]
            compute = computes[(cluster, index)]
            if load.end_cc > compute.start_cc:
                raise AssertionError("compute starts before load completion")
            if index and computes[(cluster, index - 1)].end_cc > compute.start_cc:
                raise AssertionError("cluster compute overlaps")
            if index >= 2 and computes[(cluster, index - 2)].end_cc > load.start_cc:
                raise AssertionError("load overwrites a live ping/pong slot")
            if item.buffer_slot != index % 2:
                raise AssertionError("buffer-slot mapping is not deterministic")
    for lane in (0, 1):
        records = sorted(
            (
                record
                for record in result.loads
                if lane in _required_lanes(record.item.binding)
            ),
            key=lambda record: (record.start_cc, record.end_cc),
        )
        for left, right in zip(records, records[1:]):
            if left.end_cc > right.start_cc:
                raise AssertionError(f"DMA lane {lane} overlaps")
