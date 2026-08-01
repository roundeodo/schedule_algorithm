#!/usr/bin/env python3
"""Evaluate replay-valid partition/order/DMA baselines on the frozen 65 cases.

The three policies form a causal ladder:

``BLOCK-LPT-FIFO``
    Greedily assign each whole expert to the stream with smaller accumulated
    ``ceil(tokens/2)`` compute-block load, then execute each stream in FIFO
    order.

``BLOCK-DP-FIFO``
    Use exact two-way block-load partitioning, but keep deterministic FIFO
    execution inside each stream.

``BLOCK-DP-DMA-GREEDY``
    Keep the same exact partition and, whenever a stream is ready, choose the
    remaining expert/shape/DMA realization with the locally earliest legal
    completion.  It has no continuation score, lookahead, or standalone S4PF.

Every policy emits concrete explicit-DMA ``StageAction`` objects and is replayed
by the four-stage validator.  Partition load alone is never reported as a
schedule result.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import astuple
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import four_stage_scheduler as reference  # noqa: E402
from run_four_stage_reference import serialize_action  # noqa: E402


DEFAULT_PROOF = (
    HERE / "results" / "policy_search" / "olmoe_top2_projection_65_optimal_v1.json"
)
DEFAULT_OUTPUT = (
    HERE / "results" / "policy_search" / "olmoe_65_partition_dma_baselines_v1.json"
)
POLICIES = (
    "BLOCK-LPT-FIFO",
    "BLOCK-DP-FIFO",
    "BLOCK-DP-DMA-GREEDY",
)
TICK_CC = reference.SCHEDULE_TIME_QUANTUM_CC


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ticks(cc: int) -> str:
    value = Fraction(int(cc), TICK_CC)
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _blocks(ntok: int) -> int:
    return (int(ntok) + 1) // 2


def _ordered_experts(counts: list[int]) -> list[int]:
    return sorted(
        (eid for eid, ntok in enumerate(counts) if int(ntok) > 0),
        key=lambda eid: (-_blocks(counts[eid]), -int(counts[eid]), eid),
    )


def _lpt_partition(counts: list[int]) -> tuple[list[int], list[int]]:
    streams = [[], []]
    loads = [0, 0]
    for eid in _ordered_experts(counts):
        stream = min(range(2), key=lambda index: (loads[index], index))
        streams[stream].append(eid)
        loads[stream] += _blocks(counts[eid])
    return streams[0], streams[1]


def _dp_partition(counts: list[int]) -> tuple[list[int], list[int]]:
    experts = _ordered_experts(counts)
    if len(experts) <= 1:
        return experts, []
    total = sum(_blocks(counts[eid]) for eid in experts)
    # At most roughly 70 blocks in the frozen corpus.  One deterministic
    # predecessor per reachable sum is sufficient for exact minimax balance.
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for eid in experts:
        weight = _blocks(counts[eid])
        updates = {}
        for subtotal, chosen in reachable.items():
            candidate = subtotal + weight
            selected = chosen + (eid,)
            previous = reachable.get(candidate, updates.get(candidate))
            if previous is None or selected < previous:
                updates[candidate] = selected
        for subtotal, chosen in updates.items():
            previous = reachable.get(subtotal)
            if previous is None or chosen < previous:
                reachable[subtotal] = chosen
    valid = [
        subtotal
        for subtotal, chosen in reachable.items()
        if chosen and len(chosen) < len(experts)
    ]
    best_sum = min(
        valid,
        key=lambda subtotal: (
            max(subtotal, total - subtotal),
            abs(total - 2 * subtotal),
            reachable[subtotal],
        ),
    )
    left = set(reachable[best_sum])
    return (
        [eid for eid in experts if eid in left],
        [eid for eid in experts if eid not in left],
    )


def _action_key(
    action: reference.StageAction,
    child: reference.BeamState,
) -> tuple:
    ends = (int(child.c2.task_end), int(child.c3.task_end))
    s2pf = sum(
        binding != reference.DmaBinding.NONE
        for binding in (action.c2_s2pf_dma, action.c3_s2pf_dma)
    )
    return (
        max(ends),
        abs(ends[0] - ends[1]),
        sum(ends),
        -s2pf,
        repr(astuple(action)),
    )


def _subset(
    state: reference.BeamState,
    eids: set[int],
) -> tuple[tuple[int, int], ...]:
    return tuple(item for item in state.remaining if item[0] in eids)


def _choose_action(
    state: reference.BeamState,
    c2_eid: int | None,
    c3_eid: int | None,
    *,
    wait_until: int | None = None,
) -> tuple[reference.StageAction, reference.BeamState, int]:
    selected = {eid for eid in (c2_eid, c3_eid) if eid is not None}
    actions = reference.gen_stage_actions(
        state.c2,
        state.c3,
        _subset(state, selected),
    )
    eligible = []
    for action in actions:
        if c2_eid is not None and c3_eid is not None:
            # The reference removes cluster/lane-symmetric future states and
            # may retain only the swapped concrete PAIR.  At a synchronized
            # boundary the two logical streams can exchange physical cluster
            # ownership after both FIFO heads have completed.
            if {action.c2_eid, action.c3_eid} != {c2_eid, c3_eid}:
                continue
        else:
            if c2_eid is not None and action.c2_eid != c2_eid:
                continue
            if c2_eid is None and action.c2_eid >= 0:
                continue
            if c3_eid is not None and action.c3_eid != c3_eid:
                continue
            if c3_eid is None and action.c3_eid >= 0:
                continue
        if action.c2_eid >= 0 and action.c2_eid == action.c3_eid:
            continue
        starts = [
            start
            for eid, start in (
                (action.c2_eid, action.c2_start),
                (action.c3_eid, action.c3_start),
            )
            if eid >= 0
        ]
        if wait_until is not None and any(start < wait_until for start in starts):
            continue
        child = reference.apply_action(state, action)
        eligible.append((action, child))
    if not eligible:
        raise RuntimeError(
            f"no physical action for c2={c2_eid}, c3={c3_eid}, "
            f"wait_until={wait_until}, ends="
            f"{(state.c2.task_end, state.c3.task_end)}"
        )
    action, child = min(eligible, key=lambda item: _action_key(*item))
    return action, child, len(eligible)


def _rank_in_remaining(state: reference.BeamState, eid: int) -> int:
    return next(index for index, item in enumerate(state.remaining) if item[0] == eid)


def _orient_equal_clusters(
    state: reference.BeamState,
    stream2: list[int],
    stream3: list[int],
) -> tuple[list[int], list[int]]:
    if state.c2.task_end == state.c3.task_end and not stream2 and stream3:
        # Both physical clusters are free.  Continue the sole remaining
        # logical stream on C2, which is the reference generator's canonical
        # single-cluster direction at a symmetric decision boundary.
        return stream3, stream2
    if (
        state.c2 == state.c3
        and stream2
        and stream3
        and _rank_in_remaining(state, stream2[0])
        > _rank_in_remaining(state, stream3[0])
    ):
        return stream3, stream2
    if state.c2 == state.c3 and not stream2 and stream3:
        return stream3, stream2
    return stream2, stream3


def _fifo_schedule(
    counts: list[int],
    partition: tuple[list[int], list[int]],
) -> dict:
    token_dist = {
        eid: int(ntok) for eid, ntok in enumerate(counts) if int(ntok) > 0
    }
    state = reference.FourStageScheduler(token_dist)._initial_state()
    stream2, stream3 = [list(part) for part in partition]
    decisions = []
    while state.remaining:
        stream2, stream3 = _orient_equal_clusters(state, stream2, stream3)
        t2, t3 = int(state.c2.task_end), int(state.c3.task_end)
        wait_until = None
        if t2 == t3:
            c2_eid = stream2[0] if stream2 else None
            c3_eid = stream3[0] if stream3 else None
        elif t2 < t3:
            if stream2:
                c2_eid, c3_eid = stream2[0], None
            else:
                # The first stream is empty.  Handoff the second FIFO only
                # after its previous task completes; this preserves order.
                stream2, stream3 = stream3, []
                c2_eid, c3_eid = stream2[0], None
                wait_until = t3
        else:
            if stream3:
                c2_eid, c3_eid = None, stream3[0]
            else:
                stream3, stream2 = stream2, []
                c2_eid, c3_eid = None, stream3[0]
                wait_until = t2
        action, child, variants = _choose_action(
            state, c2_eid, c3_eid, wait_until=wait_until
        )
        swapped_pair = (
            c2_eid is not None
            and c3_eid is not None
            and action.c2_eid == c3_eid
            and action.c3_eid == c2_eid
        )
        if c2_eid is not None:
            if not stream2 or stream2[0] != c2_eid:
                raise AssertionError("C2 FIFO order drift")
            stream2.pop(0)
        if c3_eid is not None:
            if not stream3 or stream3[0] != c3_eid:
                raise AssertionError("C3 FIFO order drift")
            stream3.pop(0)
        if swapped_pair:
            stream2, stream3 = stream3, stream2
        decisions.append(
            {
                "remaining_before": len(state.remaining),
                "physical_variants": variants,
                "selected": serialize_action(action),
                "ends_after_ticks": [
                    _ticks(child.c2.task_end),
                    _ticks(child.c3.task_end),
                ],
            }
        )
        state = child
    return _finalize(state, token_dist, decisions)


def _greedy_schedule(
    counts: list[int],
    partition: tuple[list[int], list[int]],
) -> dict:
    token_dist = {
        eid: int(ntok) for eid, ntok in enumerate(counts) if int(ntok) > 0
    }
    state = reference.FourStageScheduler(token_dist)._initial_state()
    stream2, stream3 = [list(part) for part in partition]
    decisions = []

    def representatives(stream: list[int]) -> list[int]:
        # Equal-token experts in the same partition are physically
        # interchangeable because this baseline never creates next-expert
        # residency.  One ID per load class preserves the complete set of
        # locally distinct timing/DMA choices.
        selected = []
        seen = set()
        for eid, ntok in state.remaining:
            if eid not in stream or ntok in seen:
                continue
            seen.add(ntok)
            selected.append(eid)
        return selected

    while state.remaining:
        stream2, stream3 = _orient_equal_clusters(state, stream2, stream3)
        t2, t3 = int(state.c2.task_end), int(state.c3.task_end)
        wait_until = None
        reps2 = representatives(stream2)
        reps3 = representatives(stream3)
        allowed2 = set(reps2)
        allowed3 = set(reps3)
        mode = ""
        if t2 == t3 and stream2 and stream3:
            mode = "PAIR"
        elif t2 <= t3 and stream2:
            mode = "C2_SINGLE"
        elif t3 <= t2 and stream3:
            mode = "C3_SINGLE"
        else:
            # One partition has drained.  Preserve the other stream's order
            # constraint only through its previous completion boundary, then
            # let the free physical cluster take any remaining expert.
            if stream2:
                wait_until = t2
                stream3, stream2 = stream2, []
                reps2 = []
                reps3 = representatives(stream3)
                allowed2, allowed3 = set(), set(reps3)
                mode = "C3_SINGLE"
            elif stream3:
                wait_until = t3
                stream2, stream3 = stream3, []
                reps2 = representatives(stream2)
                reps3 = []
                allowed2, allowed3 = set(reps2), set()
                mode = "C2_SINGLE"

        selected_eids = allowed2 | allowed3
        raw_actions = reference.gen_stage_actions(
            state.c2,
            state.c3,
            _subset(state, selected_eids),
        )
        eligible = []
        logical = set()
        for action in raw_actions:
            if action.c2_eid >= 0 and action.c2_eid == action.c3_eid:
                continue
            if mode == "PAIR":
                direct = action.c2_eid in allowed2 and action.c3_eid in allowed3
                swapped = action.c2_eid in allowed3 and action.c3_eid in allowed2
                if not (direct or swapped):
                    continue
                logical.add(frozenset((action.c2_eid, action.c3_eid)))
            elif mode == "C2_SINGLE":
                if action.c2_eid not in allowed2 or action.c3_eid >= 0:
                    continue
                logical.add((action.c2_eid,))
            elif mode == "C3_SINGLE":
                if action.c3_eid not in allowed3 or action.c2_eid >= 0:
                    continue
                logical.add((action.c3_eid,))
            else:
                continue
            starts = [
                start
                for eid, start in (
                    (action.c2_eid, action.c2_start),
                    (action.c3_eid, action.c3_start),
                )
                if eid >= 0
            ]
            if wait_until is not None and any(start < wait_until for start in starts):
                continue
            eligible.append((action, reference.apply_action(state, action)))
        if not eligible:
            raise RuntimeError("DMA-greedy partition scheduler has no legal action")
        action, child = min(
            eligible, key=lambda item: _action_key(item[0], item[1])
        )
        eid2 = action.c2_eid if action.c2_eid >= 0 else None
        eid3 = action.c3_eid if action.c3_eid >= 0 else None
        swapped_pair = (
            eid2 is not None
            and eid3 is not None
            and eid2 in allowed3
            and eid3 in allowed2
        )
        if swapped_pair:
            stream3.remove(eid2)
            stream2.remove(eid3)
            stream2, stream3 = stream3, stream2
        else:
            if eid2 is not None:
                stream2.remove(eid2)
            if eid3 is not None:
                stream3.remove(eid3)
        decisions.append(
            {
                "remaining_before": len(state.remaining),
                "logical_choices": len(logical),
                "physical_variants": len(eligible),
                "selected": serialize_action(action),
                "ends_after_ticks": [
                    _ticks(child.c2.task_end),
                    _ticks(child.c3.task_end),
                ],
            }
        )
        state = child
    return _finalize(state, token_dist, decisions)


def _finalize(
    state: reference.BeamState,
    token_dist: dict[int, int],
    decisions: list[dict],
) -> dict:
    validated = reference.validate_schedule_history(state.history, token_dist)
    if validated != state.g_score:
        raise RuntimeError(f"history replay {validated} != state {state.g_score}")
    return {
        "makespan_cc": int(state.g_score),
        "makespan_ticks": _ticks(state.g_score),
        "terminal_ticks": [_ticks(state.c2.task_end), _ticks(state.c3.task_end)],
        "action_count": len(state.history),
        "history_replay_valid": True,
        "actions": [serialize_action(action) for action in state.history],
        "decisions": decisions,
    }


def _partition_record(
    counts: list[int],
    partition: tuple[list[int], list[int]],
) -> dict:
    return {
        "stream_experts": [list(partition[0]), list(partition[1])],
        "block_loads": [
            sum(_blocks(counts[eid]) for eid in partition[0]),
            sum(_blocks(counts[eid]) for eid in partition[1]),
        ],
        "token_loads": [
            sum(counts[eid] for eid in partition[0]),
            sum(counts[eid] for eid in partition[1]),
        ],
    }


def _evaluate_case(payload: tuple[int, dict]) -> dict:
    index, case = payload
    counts = [int(value) for value in case["counts"]]
    lpt = _lpt_partition(counts)
    dp = _dp_partition(counts)
    row = {
        "index": index,
        "name": case["name"],
        "counts": counts,
        "optimal_ticks": str(case["best_reference_ticks"]),
        "partitions": {
            "BLOCK-LPT": _partition_record(counts, lpt),
            "BLOCK-DP": _partition_record(counts, dp),
        },
    }
    reference.clear_scheduler_caches()
    row["BLOCK-LPT-FIFO"] = _fifo_schedule(counts, lpt)
    reference.clear_scheduler_caches()
    row["BLOCK-DP-FIFO"] = _fifo_schedule(counts, dp)
    reference.clear_scheduler_caches()
    row["BLOCK-DP-DMA-GREEDY"] = _greedy_schedule(counts, dp)
    for policy in POLICIES:
        if Fraction(row[policy]["makespan_ticks"]) < Fraction(row["optimal_ticks"]):
            raise RuntimeError(f"{case['name']} {policy} beats certified optimum")
    return row


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def _summary(rows: list[dict], policy: str) -> dict:
    gaps = [
        float(Fraction(row[policy]["makespan_ticks"]) - Fraction(row["optimal_ticks"]))
        for row in rows
    ]
    ratios = [
        float(Fraction(row[policy]["makespan_ticks"]) / Fraction(row["optimal_ticks"]))
        for row in rows
    ]
    return {
        "cases": len(rows),
        "optimal_cases": sum(abs(gap) < 1e-12 for gap in gaps),
        "gap_ticks": {
            "sum": sum(gaps),
            "mean": statistics.mean(gaps),
            "p50": statistics.median(gaps),
            "p95": _percentile(gaps, 0.95),
            "max": max(gaps),
        },
        "ratio": {
            "mean": statistics.mean(ratios),
            "p50": statistics.median(ratios),
            "p95": _percentile(ratios, 0.95),
            "max": max(ratios),
        },
    }


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", type=Path, default=DEFAULT_PROOF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    proof = json.loads(args.proof.read_text(encoding="utf-8"))
    if not proof.get("complete") or proof.get("summary", {}).get("proven_optimal") != 65:
        raise SystemExit("proof must contain 65 certified optimal cases")
    cases = list(proof["cases"])
    if args.case:
        requested = set(args.case)
        cases = [case for case in cases if case["name"] in requested]
        if len(cases) != len(requested):
            raise SystemExit("one or more --case names are unknown")
    if args.limit >= 0:
        cases = cases[: args.limit]

    started = time.perf_counter()
    rows = []
    indexed = list(enumerate(cases, 1))
    if args.workers == 1:
        for completed, item in enumerate(indexed, 1):
            rows.append(_evaluate_case(item))
            if args.progress_every and completed % args.progress_every == 0:
                print(f"cases={completed}/{len(cases)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_evaluate_case, item): item[0] for item in indexed}
            for completed, future in enumerate(as_completed(futures), 1):
                rows.append(future.result())
                if args.progress_every and completed % args.progress_every == 0:
                    print(f"cases={completed}/{len(cases)}", flush=True)
    rows.sort(key=lambda row: row["index"])
    script = Path(__file__).resolve()
    reference_path = Path(reference.__file__).resolve()
    report = {
        "schema": "olmoe_partition_dma_baselines_v1",
        "complete": len(rows) == 65,
        "configuration": {
            "policies": list(POLICIES),
            "partition_weight": "ceil(tokens/2)",
            "whole_experts": True,
            "standalone_prefetch": False,
            "continuation_score": False,
            "lookahead": False,
            "physical_selection": "locally earliest explicit-DMA completion",
        },
        "evidence": {
            "proof": str(args.proof.resolve()),
            "proof_sha256": _sha256(args.proof),
            "reference": str(reference_path),
            "reference_sha256": _sha256(reference_path),
            "script": str(script),
            "script_sha256": _sha256(script),
        },
        "summary": {policy: _summary(rows, policy) for policy in POLICIES},
        "runtime_s": time.perf_counter() - started,
        "rows": rows,
    }
    _atomic_write(args.output, report)
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
