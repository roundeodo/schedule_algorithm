#!/usr/bin/env python3
"""Executable Bingo-style task stream for the N-outer timing model.

The timing simulator decides the cluster order and DMA-lane schedule.  This
module lowers that result into a self-contained task graph.  The graph is the
contract between the scheduler and Bingo: task arguments describe the work,
dependencies describe when it is legal to start, and ``issue_order`` provides
a deterministic topological order for filling the ready queues.

No absolute start cycle is required by the device.  ``model_start_cc`` is kept
only as audit metadata so the task graph can be replayed against the source
timing history.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable

from .model import CandidateResult, ComputeRecord, LoadRecord, WorkItem


class TaskKind(str, Enum):
    LOAD_WEIGHT = "load_weight"
    COMPUTE_BLOCK = "compute_block"


class StartupMode(str, Enum):
    COLD = "cold"
    PRELOADED_FIRST = "preloaded_first"


@dataclass(frozen=True)
class NOuterTask:
    task_id: int
    kind: TaskKind
    cluster: int
    stream_index: int
    phase_index: int
    phase_name: str
    block_id: int
    expert_sequence_index: int
    eid: int
    ntokens: int
    token_ref_start: int
    split_token_start: int
    weight_buffer_slot: int
    duration_cc: int
    depends_on: tuple[int, ...]
    dma_lane_mask: int = 0
    weight_bytes: int = 0
    m4_iters: int = 0
    m2_iters: int = 0
    m4_tail_valid_tokens: int = 0
    m2_valid_tokens: int = 0
    weight_preloaded: bool = False
    model_start_cc: int = 0
    model_end_cc: int = 0

    @property
    def resource_names(self) -> tuple[str, ...]:
        if self.kind == TaskKind.COMPUTE_BLOCK:
            return (f"compute_c{self.cluster}",)
        return tuple(
            f"dma_lane{lane}" for lane in (0, 1) if self.dma_lane_mask & (1 << lane)
        )


@dataclass(frozen=True)
class TaskExecution:
    task_id: int
    start_cc: int
    end_cc: int


@dataclass(frozen=True)
class NOuterTaskStream:
    startup_mode: StartupMode
    tasks: tuple[NOuterTask, ...]
    issue_order: tuple[int, ...]
    source_makespan_cc: int

    def task_by_id(self) -> dict[int, NOuterTask]:
        return {task.task_id: task for task in self.tasks}

    def as_dict(self) -> dict[str, object]:
        return {
            "startup_mode": self.startup_mode.value,
            "source_makespan_cc": self.source_makespan_cc,
            "issue_order": list(self.issue_order),
            "tasks": [
                {
                    **asdict(task),
                    "kind": task.kind.value,
                }
                for task in self.tasks
            ],
        }


@dataclass(frozen=True)
class TaskReplayResult:
    makespan_cc: int
    executions: tuple[TaskExecution, ...]
    dependencies_valid: bool
    resources_valid: bool


def _item_fields(item: WorkItem) -> dict[str, int | str]:
    expert = item.expert
    return {
        "cluster": item.cluster,
        "stream_index": item.stream_index,
        "phase_index": item.phase_index,
        "phase_name": item.phase_name,
        "block_id": item.block_id,
        "expert_sequence_index": item.sequence_index,
        "eid": expert.eid,
        "ntokens": expert.ntokens,
        "token_ref_start": expert.token_ref_start,
        "split_token_start": expert.split_token_start,
        "weight_buffer_slot": item.buffer_slot,
    }


def lower_schedule_to_tasks(
    schedule: CandidateResult,
    *,
    startup_mode: StartupMode = StartupMode.COLD,
) -> NOuterTaskStream:
    """Lower one validated timing history into executable LOAD/COMPUTE tasks.

    Natural correctness edges are always emitted:

    * compute ``i`` waits for load ``i`` and compute ``i-1``;
    * load ``i`` waits for compute ``i-2`` before reusing ping/pong;
    * every DMA lane follows the exact lane order selected by the scheduler.

    A deliberate scheduler WAIT is represented by an edge from an event that
    ends at the chosen grant point.  Thus the task graph reproduces the timing
    history without a global-cycle timer.
    """

    if not schedule.history_validated:
        raise ValueError("only a validated schedule can be lowered")

    load_records = {(r.item.cluster, r.item.stream_index): r for r in schedule.loads}
    compute_records = {
        (r.item.cluster, r.item.stream_index): r for r in schedule.computes
    }
    keys = sorted(compute_records)
    preloaded = {
        key for key in keys if startup_mode == StartupMode.PRELOADED_FIRST and key[1] == 0
    }

    next_id = 0
    load_id: dict[tuple[int, int], int] = {}
    compute_id: dict[tuple[int, int], int] = {}
    for key in keys:
        if key not in preloaded:
            load_id[key] = next_id
            next_id += 1
        compute_id[key] = next_id
        next_id += 1

    lane_predecessors: dict[tuple[int, int], set[int]] = {key: set() for key in keys}
    for lane in (0, 1):
        ordered = sorted(
            (record for record in schedule.loads if lane in record.lanes),
            key=lambda record: (record.start_cc, record.end_cc, record.item.cluster),
        )
        previous_key: tuple[int, int] | None = None
        for record in ordered:
            key = record.item.key
            if previous_key is not None and previous_key in load_id:
                lane_predecessors[key].add(load_id[previous_key])
            previous_key = key

    tasks: list[NOuterTask] = []
    for key in keys:
        compute = compute_records[key]
        item = compute.item
        common = _item_fields(item)

        if key in load_id:
            load = load_records[key]
            deps = set(lane_predecessors[key])
            if key[1] >= 2:
                deps.add(compute_id[(key[0], key[1] - 2)])
            tasks.append(
                NOuterTask(
                    task_id=load_id[key],
                    kind=TaskKind.LOAD_WEIGHT,
                    duration_cc=load.end_cc - load.start_cc,
                    depends_on=tuple(sorted(deps)),
                    dma_lane_mask=sum(1 << lane for lane in load.lanes),
                    weight_bytes=item.weight_bytes,
                    model_start_cc=load.start_cc,
                    model_end_cc=load.end_cc,
                    **common,
                )
            )

        deps = set()
        if key in load_id:
            deps.add(load_id[key])
        if key[1] > 0:
            deps.add(compute_id[(key[0], key[1] - 1)])
        expert = item.expert
        tasks.append(
            NOuterTask(
                task_id=compute_id[key],
                kind=TaskKind.COMPUTE_BLOCK,
                duration_cc=compute.end_cc - compute.start_cc,
                depends_on=tuple(sorted(deps)),
                m4_iters=expert.m4_iters,
                m2_iters=expert.m2_iters,
                m4_tail_valid_tokens=expert.m4_tail_valid_tokens,
                m2_valid_tokens=expert.m2_valid_tokens,
                weight_preloaded=key in preloaded,
                model_start_cc=compute.start_cc,
                model_end_cc=compute.end_cc,
                **common,
            )
        )

    if startup_mode == StartupMode.COLD:
        tasks = _encode_deliberate_waits(tasks)

    provisional = NOuterTaskStream(
        startup_mode=startup_mode,
        tasks=tuple(sorted(tasks, key=lambda task: task.task_id)),
        issue_order=(),
        source_makespan_cc=schedule.makespan_cc,
    )
    replay = replay_task_stream(provisional)
    starts = {execution.task_id: execution.start_cc for execution in replay.executions}
    issue_order = tuple(
        task.task_id
        for task in sorted(
            provisional.tasks,
            key=lambda task: (
                starts[task.task_id],
                0 if task.kind == TaskKind.LOAD_WEIGHT else 1,
                task.cluster,
                task.task_id,
            ),
        )
    )
    result = NOuterTaskStream(
        startup_mode=startup_mode,
        tasks=provisional.tasks,
        issue_order=issue_order,
        source_makespan_cc=schedule.makespan_cc,
    )
    validate_task_stream(result)
    if startup_mode == StartupMode.COLD:
        _validate_cold_replay_matches_source(result, replay)
    return result


def _encode_deliberate_waits(tasks: list[NOuterTask]) -> list[NOuterTask]:
    """Add an event dependency when natural edges would start a task too early."""

    by_id = {task.task_id: task for task in tasks}
    result: list[NOuterTask] = []
    for task in tasks:
        dependency_end = max(
            (by_id[dep].model_end_cc for dep in task.depends_on), default=0
        )
        if dependency_end < task.model_start_cc:
            guards = [
                candidate.task_id
                for candidate in tasks
                if candidate.task_id != task.task_id
                and candidate.model_end_cc == task.model_start_cc
                and candidate.model_start_cc < candidate.model_end_cc
            ]
            if not guards:
                raise AssertionError(
                    f"task {task.task_id} waits until {task.model_start_cc} without an event"
                )
            deps = tuple(sorted((*task.depends_on, min(guards))))
            task = NOuterTask(**{**task.__dict__, "depends_on": deps})
        result.append(task)
    return result


def replay_task_stream(stream: NOuterTaskStream) -> TaskReplayResult:
    """Run every task at the earliest cycle allowed by its dependency edges."""

    tasks = stream.task_by_id()
    completed: dict[int, TaskExecution] = {}
    remaining = set(tasks)
    while remaining:
        ready = [
            tasks[task_id]
            for task_id in remaining
            if all(dep in completed for dep in tasks[task_id].depends_on)
        ]
        if not ready:
            raise AssertionError("task graph contains a cycle or missing dependency")
        for task in sorted(ready, key=lambda value: value.task_id):
            start = max(
                (completed[dep].end_cc for dep in task.depends_on), default=0
            )
            completed[task.task_id] = TaskExecution(
                task_id=task.task_id,
                start_cc=start,
                end_cc=start + task.duration_cc,
            )
            remaining.remove(task.task_id)

    executions = tuple(sorted(completed.values(), key=lambda value: value.task_id))
    makespan = max((execution.end_cc for execution in executions), default=0)
    resources_valid = _resources_do_not_overlap(stream.tasks, executions)
    return TaskReplayResult(
        makespan_cc=makespan,
        executions=executions,
        dependencies_valid=True,
        resources_valid=resources_valid,
    )


def _resources_do_not_overlap(
    tasks: Iterable[NOuterTask], executions: Iterable[TaskExecution]
) -> bool:
    task_by_id = {task.task_id: task for task in tasks}
    execution_by_id = {execution.task_id: execution for execution in executions}
    resources: dict[str, list[TaskExecution]] = {}
    for task_id, task in task_by_id.items():
        for resource in task.resource_names:
            resources.setdefault(resource, []).append(execution_by_id[task_id])
    for records in resources.values():
        records.sort(key=lambda value: (value.start_cc, value.end_cc, value.task_id))
        if any(left.end_cc > right.start_cc for left, right in zip(records, records[1:])):
            return False
    return True


def validate_task_stream(stream: NOuterTaskStream) -> None:
    tasks = stream.task_by_id()
    if len(tasks) != len(stream.tasks):
        raise AssertionError("duplicate task id")
    if set(stream.issue_order) != set(tasks) or len(stream.issue_order) != len(tasks):
        raise AssertionError("issue order must contain every task exactly once")
    position = {task_id: index for index, task_id in enumerate(stream.issue_order)}
    for task in stream.tasks:
        if any(dep not in tasks for dep in task.depends_on):
            raise AssertionError(f"task {task.task_id} references an unknown dependency")
        if any(position[dep] >= position[task.task_id] for dep in task.depends_on):
            raise AssertionError("issue order is not topological")
        if task.kind == TaskKind.LOAD_WEIGHT and task.dma_lane_mask not in (1, 2, 3):
            raise AssertionError("load task has an invalid DMA lane mask")
        if task.kind == TaskKind.COMPUTE_BLOCK and task.dma_lane_mask != 0:
            raise AssertionError("compute task owns a DMA lane")
    replay = replay_task_stream(stream)
    if not replay.resources_valid:
        raise AssertionError("task dependencies permit a resource overlap")


def _validate_cold_replay_matches_source(
    stream: NOuterTaskStream, replay: TaskReplayResult
) -> None:
    executions = {execution.task_id: execution for execution in replay.executions}
    for task in stream.tasks:
        actual = executions[task.task_id]
        if (actual.start_cc, actual.end_cc) != (task.model_start_cc, task.model_end_cc):
            raise AssertionError(
                f"task {task.task_id} replay {actual.start_cc}:{actual.end_cc} "
                f"!= model {task.model_start_cc}:{task.model_end_cc}"
            )
    if replay.makespan_cc != stream.source_makespan_cc:
        raise AssertionError("lowered cold task stream changed the source makespan")
