#!/usr/bin/env python3
"""Deterministic lowering from a selected macro history to Bingo parameters.

The output is a sequence of macro records, not one scheduler decision per
weight block.  A fixed Bingo worker expands each record over Gate/Up then Down
blocks and alternates two logical weight buffers.  Logical slot roles are kept
separate from physical slot numbers so this package does not modify or assume
the current HeMAiA/Bingo allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .block_golden import (
    ArbitrationPolicy,
    BlockItem,
    GoldenResult,
    replay_block_streams,
)
from .search import SearchNode, validate_history
from .semantics import (
    DmaBinding,
    MacroPhaseSpec,
    ShapeName,
    compute_block_cc,
    default_phases,
)


@dataclass(frozen=True)
class LogicalSlotRoles:
    load_worker: str
    compute_worker: str
    weight_ping: int = 0
    weight_pong: int = 1


@dataclass(frozen=True)
class BingoMacroRecord:
    step_index: int
    action_kind: str
    cluster: int
    cluster_sequence_index: int
    eid: int
    token_start: int
    ntokens: int
    gate_up_shape: ShapeName
    gate_up_dma_mask: int
    down_shape: ShapeName
    down_dma_mask: int
    gate_up_first_rank: int
    gate_up_stream_rank: int
    down_first_rank: int
    down_stream_rank: int

    @property
    def token_end(self) -> int:
        return self.token_start + self.ntokens


@dataclass(frozen=True)
class BingoMacroProgram:
    records: tuple[BingoMacroRecord, ...]
    cluster_records: tuple[
        tuple[BingoMacroRecord, ...], tuple[BingoMacroRecord, ...]
    ]
    logical_slots: tuple[LogicalSlotRoles, LogicalSlotRoles]
    source_makespan_cc: int
    history_validated: bool


def lower_history_to_bingo(
    distribution: Sequence[int], node: SearchNode
) -> BingoMacroProgram:
    """Lower only fields needed by a fixed N-outer Bingo worker."""

    validate_history(distribution, node)
    records: list[BingoMacroRecord] = []
    cluster_sequences = [0, 0]
    for step_index, step in enumerate(node.history):
        ranks = {
            operation: rank
            for rank, operation in enumerate(step.timing.service_order)
        }
        for task in sorted(step.plan.tasks, key=lambda item: item.cluster):
            cluster = task.cluster
            expert_slice = task.expert_slice
            records.append(
                BingoMacroRecord(
                    step_index=step_index,
                    action_kind=step.plan.kind.value,
                    cluster=cluster,
                    cluster_sequence_index=cluster_sequences[cluster],
                    eid=expert_slice.eid,
                    token_start=expert_slice.token_start,
                    ntokens=expert_slice.ntokens,
                    gate_up_shape=task.gate_up.shape.name,
                    gate_up_dma_mask=int(task.gate_up.dma),
                    down_shape=task.down.shape.name,
                    down_dma_mask=int(task.down.dma),
                    gate_up_first_rank=ranks[(cluster, "gate_up_first")],
                    gate_up_stream_rank=ranks[(cluster, "gate_up_stream")],
                    down_first_rank=ranks[(cluster, "down_first")],
                    down_stream_rank=ranks[(cluster, "down_stream")],
                )
            )
            cluster_sequences[cluster] += 1
    by_cluster = tuple(
        tuple(
            sorted(
                (record for record in records if record.cluster == cluster),
                key=lambda record: record.cluster_sequence_index,
            )
        )
        for cluster in (0, 1)
    )
    program = BingoMacroProgram(
        records=tuple(records),
        cluster_records=by_cluster,
        logical_slots=(
            LogicalSlotRoles("cluster0_weight_loader", "cluster0_nouter_compute"),
            LogicalSlotRoles("cluster1_weight_loader", "cluster1_nouter_compute"),
        ),
        source_makespan_cc=node.makespan_cc,
        history_validated=False,
    )
    validate_bingo_program(distribution, program)
    return BingoMacroProgram(**{**program.__dict__, "history_validated": True})


def _shape_by_name(name: ShapeName):
    from .semantics import ALL_SHAPES

    return next(shape for shape in ALL_SHAPES if shape.name == name)


def expand_bingo_program(
    program: BingoMacroProgram,
    *,
    phases: tuple[MacroPhaseSpec, MacroPhaseSpec] | None = None,
) -> tuple[tuple[BlockItem, ...], tuple[BlockItem, ...]]:
    """Fixed-worker expansion used only to verify the macro output contract."""

    phase_specs = phases or default_phases()
    streams: list[list[BlockItem]] = [[], []]
    for cluster in (0, 1):
        for record in program.cluster_records[cluster]:
            phase_fields = (
                (
                    phase_specs[0],
                    _shape_by_name(record.gate_up_shape),
                    DmaBinding(record.gate_up_dma_mask),
                    record.gate_up_first_rank,
                    record.gate_up_stream_rank,
                ),
                (
                    phase_specs[1],
                    _shape_by_name(record.down_shape),
                    DmaBinding(record.down_dma_mask),
                    record.down_first_rank,
                    record.down_stream_rank,
                ),
            )
            for phase_index, (
                phase,
                shape,
                binding,
                first_rank,
                stream_rank,
            ) in enumerate(phase_fields):
                for block_id in range(phase.block_count):
                    streams[cluster].append(
                        BlockItem(
                            cluster=cluster,
                            stream_index=len(streams[cluster]),
                            step_index=record.step_index,
                            eid=record.eid,
                            token_start=record.token_start,
                            ntokens=record.ntokens,
                            phase_name=phase.name,
                            phase_index=phase_index,
                            block_id=block_id,
                            block_count=phase.block_count,
                            weight_bytes=phase.weight_block_bytes,
                            compute_cc=compute_block_cc(
                                record.ntokens, shape, phase
                            ),
                            shape=shape,
                            binding=binding,
                            service_rank=(
                                first_rank if block_id == 0 else stream_rank
                            ),
                        )
                    )
    return tuple(streams[0]), tuple(streams[1])


def replay_bingo_program(
    program: BingoMacroProgram,
    *,
    policy: ArbitrationPolicy = ArbitrationPolicy.MACRO_ORDER,
) -> GoldenResult:
    if not program.history_validated:
        raise ValueError("only a validated Bingo macro program can be replayed")
    return replay_block_streams(expand_bingo_program(program), policy=policy)


def validate_bingo_program(
    distribution: Sequence[int], program: BingoMacroProgram
) -> None:
    for cluster in (0, 1):
        records = program.cluster_records[cluster]
        if [record.cluster_sequence_index for record in records] != list(
            range(len(records))
        ):
            raise AssertionError("cluster macro records are not contiguous")
        if any(record.cluster != cluster for record in records):
            raise AssertionError("record appears in the wrong cluster stream")
    coverage: dict[int, list[tuple[int, int]]] = {}
    for record in program.records:
        if record.ntokens <= 0 or record.token_start < 0:
            raise AssertionError("invalid token range")
        if record.gate_up_dma_mask not in (1, 2, 3) or record.down_dma_mask not in (1, 2, 3):
            raise AssertionError("invalid DMA mask")
        coverage.setdefault(record.eid, []).append(
            (record.token_start, record.token_end)
        )
    for eid, ntokens in enumerate(distribution):
        if ntokens <= 0:
            continue
        cursor = 0
        for start, end in sorted(coverage.get(eid, ())):
            if start != cursor:
                raise AssertionError("Bingo token slices overlap or leave a gap")
            cursor = end
        if cursor != ntokens:
            raise AssertionError("Bingo token coverage is incomplete")

