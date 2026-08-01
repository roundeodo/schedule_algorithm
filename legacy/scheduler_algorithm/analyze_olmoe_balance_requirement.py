#!/usr/bin/env python3
"""Quantify what load balancing alone can and cannot prove on the 65 OLMoE cases.

The report deliberately separates three objects that are easy to conflate:

* raw-token LPT: assign the next largest expert to the cluster with fewer tokens;
* compute-block LPT/exact partition: use ceil(tokens / 2) compute blocks;
* the replayed explicit-DMA four-stage optimum from the frozen certificate.

The first two are computation-only estimates, not executable schedules.  The
third includes action order, shapes, cache state, prefetch and DMA-lane legality.
"""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
import os
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import four_stage_scheduler as reference  # noqa: E402
from run_four_stage_reference import deserialize_action  # noqa: E402


DEFAULT_CERTIFICATE = (
    ROOT / "results" / "policy_search" / "olmoe_top2_projection_65_optimal_v1.json"
)
DEFAULT_OUT = (
    ROOT / "results" / "policy_search" / "olmoe_65_balance_requirement_v1.json"
)
DEFAULT_MARKDOWN_OUT = (
    ROOT / "results" / "policy_search" / "olmoe_65_distribution_catalog_v1.md"
)
TICK_CC = reference.SHAPE_C.T_s3


def _parse_ticks(value: str | int) -> Fraction:
    return Fraction(str(value))


def _ticks(value_cc: int) -> str:
    value = Fraction(int(value_cc), TICK_CC)
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _exact_partition(weights: list[int]) -> tuple[int, tuple[int, int]]:
    """Return the minimum two-bin maximum and one corresponding load pair."""
    total = sum(weights)
    reachable = 1
    for weight in weights:
        reachable |= reachable << weight
    half = total // 2
    left = max(value for value in range(half + 1) if (reachable >> value) & 1)
    right = total - left
    return max(left, right), (left, right)


def _lpt_assignment(
    counts: list[int], *, balance_by: str
) -> dict:
    """Greedily assign complete experts using raw tokens or compute blocks."""
    if balance_by not in {"tokens", "blocks"}:
        raise ValueError(balance_by)
    items = [
        {
            "eid": eid,
            "tokens": int(ntok),
            "blocks": (int(ntok) + 1) // 2,
        }
        for eid, ntok in enumerate(counts)
    ]
    sort_key = "tokens" if balance_by == "tokens" else "blocks"
    items.sort(
        key=lambda item: (
            -item[sort_key],
            -item["tokens"],
            item["eid"],
        )
    )
    token_loads = [0, 0]
    block_loads = [0, 0]
    assignments: list[list[int]] = [[], []]
    for item in items:
        loads = token_loads if balance_by == "tokens" else block_loads
        cluster = 0 if loads[0] <= loads[1] else 1
        token_loads[cluster] += item["tokens"]
        block_loads[cluster] += item["blocks"]
        assignments[cluster].append(item["eid"])
    return {
        "token_loads": token_loads,
        "block_loads": block_loads,
        "optimistic_compute_ticks": 3 * max(block_loads),
        "assignments": assignments,
    }


def _replay_optimum(case: dict) -> dict:
    token_dist = {eid: int(ntok) for eid, ntok in enumerate(case["counts"])}
    history = tuple(deserialize_action(action) for action in case["actions"])
    validated = reference.validate_schedule_history(history, token_dist)
    state = reference.FourStageScheduler(token_dist)._initial_state()
    for action in history:
        state = reference.apply_action(state, action)
    if state.remaining:
        raise RuntimeError(f"{case['name']}: optimal history did not terminate")
    if validated != state.g_score:
        raise RuntimeError(
            f"{case['name']}: validator {validated} != replay {state.g_score}"
        )
    if _ticks(validated) != str(case["best_reference_ticks"]):
        raise RuntimeError(
            f"{case['name']}: replay {_ticks(validated)} != certificate "
            f"{case['best_reference_ticks']}"
        )

    token_loads = [0, 0]
    segment_block_loads = [0, 0]
    action_families: Counter[str] = Counter()
    dma_bindings: Counter[str] = Counter()
    for action in case["actions"]:
        tag = str(action["tag"])
        if tag.startswith("PAIR"):
            action_families["PAIR"] += 1
        elif tag.startswith("SINGLE"):
            action_families["SINGLE"] += 1
        elif "SPLIT" in tag:
            action_families["SPLIT"] += 1
        elif tag.startswith("PF"):
            action_families["PREFETCH"] += 1
        else:
            action_families["OTHER"] += 1
        for cluster, prefix in enumerate(("c2", "c3")):
            eid = int(action[f"{prefix}_eid"])
            ntok = int(action[f"{prefix}_ntok"])
            if eid >= 0 and ntok > 0:
                token_loads[cluster] += ntok
                segment_block_loads[cluster] += (ntok + 1) // 2
            for suffix in ("dma_s1", "dma_s3", "s2pf_dma"):
                binding = str(action[f"{prefix}_{suffix}"])
                if binding != "NONE":
                    dma_bindings[binding] += 1
        pf_binding = str(action["pf_dma"])
        if pf_binding != "NONE":
            dma_bindings[pf_binding] += 1

    end_cc = [int(state.c2.task_end), int(state.c3.task_end)]
    return {
        "token_loads": token_loads,
        "token_imbalance": abs(token_loads[0] - token_loads[1]),
        "segment_block_loads": segment_block_loads,
        "terminal_ticks": [_ticks(value) for value in end_cc],
        "terminal_imbalance_ticks": _ticks(abs(end_cc[0] - end_cc[1])),
        "action_count": len(history),
        "action_families": dict(sorted(action_families.items())),
        "dma_bindings": dict(sorted(dma_bindings.items())),
    }


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def _summarize(rows: list[dict]) -> dict:
    opt = [float(_parse_ticks(row["optimal_ticks"])) for row in rows]
    partition = [row["exact_compute_partition_ticks"] for row in rows]
    raw = [row["raw_token_lpt"]["optimistic_compute_ticks"] for row in rows]
    block = [row["block_lpt"]["optimistic_compute_ticks"] for row in rows]
    raw_excess = [r - p for r, p in zip(raw, partition)]
    block_excess = [b - p for b, p in zip(block, partition)]
    physical_excess = [o - p for o, p in zip(opt, partition)]
    token_imbalance = [row["optimal_replay"]["token_imbalance"] for row in rows]
    terminal_imbalance = [
        float(_parse_ticks(row["optimal_replay"]["terminal_imbalance_ticks"]))
        for row in rows
    ]
    terminations = Counter(row["termination"] for row in rows)
    return {
        "cases": len(rows),
        "all_replay_valid_and_proven_optimal": all(
            row["history_replay_valid"] and row["proven_optimal"] for row in rows
        ),
        "compute_partition_equals_physical_optimum_cases": sum(
            abs(value) < 1e-12 for value in physical_excess
        ),
        "physical_optimum_excess_over_compute_partition_ticks": {
            "sum": sum(physical_excess),
            "mean": statistics.mean(physical_excess),
            "max": max(physical_excess),
        },
        "raw_token_lpt_compute_partition_exact_cases": sum(
            value == 0 for value in raw_excess
        ),
        "raw_token_lpt_excess_compute_ticks": {
            "sum": sum(raw_excess),
            "mean": statistics.mean(raw_excess),
            "p50": statistics.median(raw_excess),
            "p95": _percentile(raw_excess, 0.95),
            "max": max(raw_excess),
        },
        "block_lpt_compute_partition_exact_cases": sum(
            value == 0 for value in block_excess
        ),
        "block_lpt_excess_compute_ticks": {
            "sum": sum(block_excess),
            "mean": statistics.mean(block_excess),
            "p50": statistics.median(block_excess),
            "p95": _percentile(block_excess, 0.95),
            "max": max(block_excess),
        },
        "optimal_replay_token_imbalance": {
            "equal_cases": sum(value == 0 for value in token_imbalance),
            "le2_cases": sum(value <= 2 for value in token_imbalance),
            "median": statistics.median(token_imbalance),
            "max": max(token_imbalance),
        },
        "optimal_replay_terminal_imbalance_ticks": {
            "equal_cases": sum(abs(value) < 1e-12 for value in terminal_imbalance),
            "median": statistics.median(terminal_imbalance),
            "max": max(terminal_imbalance),
        },
        "certificate_termination_counts": dict(sorted(terminations.items())),
        "constructive_block_certificate_cases": terminations.get(
            "constructive_block_history_equals_certified_lb", 0
        ),
    }


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_markdown(path: Path, report: dict) -> None:
    summary = report["summary"]
    lines = [
        "# Frozen OLMoE 65-case distribution catalog",
        "",
        "All entries contain 140 Top-2 assignments. Only active experts are listed; "
        "unlisted experts have zero tokens.",
        "",
        "## Balance audit",
        "",
        f"- Replayed and proven optimal: {summary['cases']}/{summary['cases']}",
        "- Raw-token LPT reaches the best unsplit compute partition in "
        f"{summary['raw_token_lpt_compute_partition_exact_cases']}/{summary['cases']} cases.",
        "- Compute-block LPT reaches it in "
        f"{summary['block_lpt_compute_partition_exact_cases']}/{summary['cases']} cases.",
        "- The compute-only partition is also a legal physical optimum in "
        f"{summary['compute_partition_equals_physical_optimum_cases']}/{summary['cases']} cases.",
        "- Optimal histories have equal raw-token loads in only "
        f"{summary['optimal_replay_token_imbalance']['equal_cases']}/{summary['cases']} cases, "
        "but equal terminal completion times in "
        f"{summary['optimal_replay_terminal_imbalance_ticks']['equal_cases']}/{summary['cases']} cases.",
        "",
        "The token/block partitions are computation-only estimates. They do not "
        "constitute legal DMA schedules.",
        "",
        "## Distributions",
        "",
        "| # | Case | Active | Certified optimum (ticks) | Sorted nonzero counts |",
        "|---:|---|---:|---:|---|",
    ]
    for row in report["rows"]:
        counts = ",".join(str(value) for value in row["counts"])
        lines.append(
            f"| {row['index']} | `{row['name']}` | {row['active_experts']} | "
            f"{row['optimal_ticks']} | `[{counts}]` |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate", type=Path, default=DEFAULT_CERTIFICATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    args = parser.parse_args()

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    cases = certificate.get("cases", [])
    if len(cases) != 65:
        raise RuntimeError(f"expected frozen 65-case corpus, got {len(cases)}")
    rows = []
    for index, case in enumerate(cases, 1):
        if not case.get("proven_optimal") or not case.get("history_replay_valid"):
            raise RuntimeError(f"{case['name']}: certificate is not complete")
        counts = [int(value) for value in case["counts"]]
        weights = [(value + 1) // 2 for value in counts]
        partition_blocks, partition_loads = _exact_partition(weights)
        rows.append(
            {
                "index": index,
                "name": case["name"],
                "family": case["family"],
                "counts": counts,
                "token_sum": sum(counts),
                "active_experts": len(counts),
                "compute_blocks": sum(weights),
                "exact_compute_partition_block_loads": list(partition_loads),
                "exact_compute_partition_ticks": 3 * partition_blocks,
                "raw_token_lpt": _lpt_assignment(counts, balance_by="tokens"),
                "block_lpt": _lpt_assignment(counts, balance_by="blocks"),
                "optimal_ticks": str(case["best_reference_ticks"]),
                "root_lower_bound_ticks": str(case["root_lower_bound_ticks"]),
                "termination": str(case["termination"]),
                "proven_optimal": bool(case["proven_optimal"]),
                "history_replay_valid": bool(case["history_replay_valid"]),
                "optimal_replay": _replay_optimum(case),
            }
        )

    report = {
        "schema": "olmoe-balance-requirement-v1",
        "interpretation": {
            "raw_token_lpt": "computation-only; not a legal DMA schedule",
            "block_lpt": "computation-only; not a legal DMA schedule",
            "exact_compute_partition": "best unsplit two-bin compute partition; not a legal DMA schedule",
            "optimal_ticks": "replayed explicit-DMA four-stage certified optimum",
        },
        "certificate": str(args.certificate),
        "summary": _summarize(rows),
        "rows": rows,
    }
    _atomic_write(args.out, report)
    _write_markdown(args.markdown_out, report)
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
