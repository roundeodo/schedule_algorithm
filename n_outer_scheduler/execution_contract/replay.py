#!/usr/bin/env python3
"""Independent earliest-time replay of the fixed block-major runner program."""

from __future__ import annotations

from dataclasses import dataclass

from .lowering import StaticRunnerProgram, validate_runner_program
from .model import DmaMask, ItemKey


@dataclass(frozen=True)
class ReplayInterval:
    key: ItemKey
    start_tick: int
    end_tick: int


@dataclass(frozen=True)
class ReplayResult:
    loads: tuple[ReplayInterval, ...]
    computes: tuple[ReplayInterval, ...]
    makespan_ticks: int
    dependencies_valid: bool
    resources_valid: bool


def replay_static_runner(program: StaticRunnerProgram) -> ReplayResult:
    """Replay only emitted commands/dependencies, never oracle timestamps."""

    validate_runner_program(program)
    load_commands = {command.key: command for command in program.loads}
    compute_commands = {command.key: command for command in program.computes}
    pending_loads = set(load_commands)
    pending_computes = set(compute_commands)
    loads: dict[ItemKey, ReplayInterval] = {}
    computes: dict[ItemKey, ReplayInterval] = {}

    while pending_loads or pending_computes:
        changed = False
        for key in sorted(tuple(pending_loads)):
            command = load_commands[key]
            load_deps = [
                dependency
                for dependency in (
                    command.wait_local_load,
                    *command.wait_lane_load,
                )
                if dependency is not None
            ]
            if any(dependency not in loads for dependency in load_deps):
                continue
            if (
                command.wait_buffer_compute is not None
                and command.wait_buffer_compute not in computes
            ):
                continue
            start = max(
                [loads[dependency].end_tick for dependency in load_deps]
                + (
                    [computes[command.wait_buffer_compute].end_tick]
                    if command.wait_buffer_compute is not None
                    else []
                )
                + [0]
            )
            loads[key] = ReplayInterval(key, start, start + command.duration_ticks)
            pending_loads.remove(key)
            changed = True

        for key in sorted(tuple(pending_computes)):
            command = compute_commands[key]
            if command.wait_load not in loads:
                continue
            if (
                command.wait_previous_compute is not None
                and command.wait_previous_compute not in computes
            ):
                continue
            start = max(
                loads[command.wait_load].end_tick,
                computes[command.wait_previous_compute].end_tick
                if command.wait_previous_compute is not None
                else 0,
            )
            computes[key] = ReplayInterval(key, start, start + command.duration_ticks)
            pending_computes.remove(key)
            changed = True

        if not changed:
            raise RuntimeError("runner dependency graph is cyclic or incomplete")

    result = ReplayResult(
        loads=tuple(sorted(loads.values(), key=lambda event: (event.start_tick, event.key))),
        computes=tuple(
            sorted(computes.values(), key=lambda event: (event.start_tick, event.key))
        ),
        makespan_ticks=max(
            (event.end_tick for event in (*loads.values(), *computes.values())),
            default=0,
        ),
        dependencies_valid=True,
        resources_valid=True,
    )
    validate_replay(program, result)
    return result


def validate_replay(program: StaticRunnerProgram, result: ReplayResult) -> None:
    loads = {event.key: event for event in result.loads}
    computes = {event.key: event for event in result.computes}
    load_commands = {command.key: command for command in program.loads}
    compute_commands = {command.key: command for command in program.computes}

    for key, command in load_commands.items():
        event = loads[key]
        if command.wait_local_load is not None:
            if loads[command.wait_local_load].end_tick > event.start_tick:
                raise AssertionError("local LOAD stream overlaps")
        if command.wait_buffer_compute is not None:
            if computes[command.wait_buffer_compute].end_tick > event.start_tick:
                raise AssertionError("replay overwrites a live buffer")
        for dependency in command.wait_lane_load:
            if dependency is not None and loads[dependency].end_tick > event.start_tick:
                raise AssertionError("replay violates DMA lane order")

    for key, command in compute_commands.items():
        event = computes[key]
        if loads[key].end_tick > event.start_tick:
            raise AssertionError("replay computes before LOAD completion")
        if command.wait_previous_compute is not None:
            if computes[command.wait_previous_compute].end_tick > event.start_tick:
                raise AssertionError("VersaCore executes overlapping commands")

    for lane, mask in enumerate((DmaMask.IDMA, DmaMask.XDMA)):
        intervals = sorted(
            (loads[key] for key, command in load_commands.items() if command.lanes & mask),
            key=lambda event: (event.start_tick, event.end_tick, event.key),
        )
        for left, right in zip(intervals, intervals[1:]):
            if left.end_tick > right.start_tick:
                raise AssertionError(f"DMA lane {lane} overlaps in replay")
