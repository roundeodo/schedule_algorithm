#!/usr/bin/env python3
"""Generate an auditable N-outer task table for Bingo lowering."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from .model import ExpertDescriptor
from .scheduler import NOuterScheduler, SchedulerMode, SchedulerOptions
from .task_stream import StartupMode, lower_schedule_to_tasks, replay_task_stream


def _tokens(value: str) -> tuple[int, ...]:
    result = tuple(int(part) for part in value.split(",") if part.strip())
    if not result or any(value <= 0 for value in result):
        raise argparse.ArgumentTypeError("tokens must be comma-separated positive integers")
    return result


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True, type=_tokens)
    parser.add_argument(
        "--mode", choices=[mode.value for mode in SchedulerMode], default="fast"
    )
    parser.add_argument("--exact-top", type=int, default=10)
    parser.add_argument(
        "--startup",
        choices=[mode.value for mode in StartupMode],
        default=StartupMode.COLD.value,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    experts = tuple(
        ExpertDescriptor(eid=eid, ntokens=ntokens)
        for eid, ntokens in enumerate(args.tokens)
    )
    selected = NOuterScheduler(
        options=SchedulerOptions(
            mode=SchedulerMode(args.mode), exact_top_k=args.exact_top
        )
    ).schedule(experts)
    stream = lower_schedule_to_tasks(
        selected.schedule, startup_mode=StartupMode(args.startup)
    )
    replay = replay_task_stream(stream)
    payload = {
        "input_tokens": list(args.tokens),
        "scheduler_mode": args.mode,
        "candidate": selected.candidate.label,
        "selected_model_makespan_cc": selected.schedule.makespan_cc,
        "task_replay_makespan_cc": replay.makespan_cc,
        "task_count": len(stream.tasks),
        "task_stream": stream.as_dict(),
    }

    print(f"candidate={selected.candidate.label}")
    print(
        f"startup={stream.startup_mode.value} tasks={len(stream.tasks)} "
        f"model_cc={selected.schedule.makespan_cc} replay_cc={replay.makespan_cc}"
    )
    print(
        f"dependencies_valid={replay.dependencies_valid} "
        f"resources_valid={replay.resources_valid}"
    )
    if args.output is not None:
        _write_json_atomic(args.output, payload)
        print(f"result_written={args.output}")


if __name__ == "__main__":
    main()
