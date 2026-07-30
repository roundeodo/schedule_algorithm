#!/usr/bin/env python3
"""Audit S2PF/S4PF DMA-binding use in stored four-stage reference histories.

This is a history audit, not a policy ablation.  It answers how often the
reference's best-known schedules selected a single lane or BOTH, and whether
the selected prefetch was fully hidden by its local compute window.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

from four_stage_scheduler import (
    DmaBinding,
    FourStageSnap,
    WEIGHT_BYTES_S1,
    WEIGHT_BYTES_S3,
    dma_duration,
    make_initial_snap,
)
from run_four_stage_reference import deserialize_action


ROOT = Path(__file__).resolve().parent
TICK_CC = 11_264
DEFAULT_REFERENCES = tuple(
    [
        ROOT / "results" / "final_reference" / f"scheduler_reference_E{e}.json"
        for e in (8, 32, 64)
    ]
    + [
        ROOT
        / "results"
        / "blind_v2"
        / "reference_merged"
        / f"scheduler_blind_v2_reference_E{e}_merged.json"
        for e in (8, 32, 64)
    ]
)
DEFAULT_OUT = (
    ROOT / "results" / "policy_search" / "prefetch_binding_usage_audit.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def source_group(path: Path) -> str:
    return "blind_v2" if "blind_v2" in path.parts else "final_30k"


def action_family(tag: str) -> str:
    if tag.startswith("PF-"):
        return "PREFETCH"
    if "PAIR" in tag:
        return "PAIR"
    if "SPLIT" in tag:
        return "SPLIT"
    if tag.startswith("SINGLE"):
        return "SINGLE"
    return "OTHER"


def binding_class(binding: DmaBinding) -> str:
    if binding == DmaBinding.BOTH:
        return "BOTH"
    if binding in (DmaBinding.IDMA, DmaBinding.XDMA):
        return "SINGLE"
    return "NONE"


class UsageCounters:
    def __init__(self) -> None:
        self.events = Counter()
        self.cases = Counter()
        self.window_ticks = Counter()
        self.slack_ticks = Counter()
        self.by_family = Counter()
        self.case_total = 0
        self.action_total = 0
        self.s2_both_single_would_not_fit = 0
        self.s4_both_single_would_not_fit = 0

    def record_event(
        self,
        *,
        kind: str,
        binding: DmaBinding,
        hidden: bool,
        window_cc: int,
        end_cc: int,
        compute_end_cc: int,
        family: str,
    ) -> None:
        mode = binding_class(binding)
        hidden_name = "hidden" if hidden else "partial"
        self.events[(kind, mode, binding.name, hidden_name)] += 1
        self.window_ticks[(kind, mode, window_cc // TICK_CC)] += 1
        self.slack_ticks[(kind, mode, (compute_end_cc - end_cc) // TICK_CC)] += 1
        self.by_family[(kind, mode, family)] += 1
        single_duration = dma_duration(
            WEIGHT_BYTES_S3 if kind == "S2PF" else WEIGHT_BYTES_S1,
            DmaBinding.IDMA,
        )
        if binding == DmaBinding.BOTH and window_cc < single_duration:
            if kind == "S2PF":
                self.s2_both_single_would_not_fit += 1
            else:
                self.s4_both_single_would_not_fit += 1

    def record_case_modes(self, seen: set[tuple[str, str]]) -> None:
        for kind, mode in seen:
            self.cases[(kind, mode)] += 1

    @staticmethod
    def _nested(counter: Counter, names: tuple[str, ...]) -> dict:
        root: dict = {}
        for key, count in sorted(counter.items()):
            keys = key if isinstance(key, tuple) else (key,)
            cursor = root
            for name, value in zip(names[:-1], keys[:-1]):
                del name
                cursor = cursor.setdefault(str(value), {})
            cursor[str(keys[-1])] = count
        return root

    def report(self) -> dict:
        totals = Counter()
        for (kind, mode, _binding, _hidden), count in self.events.items():
            totals[(kind, mode)] += count
        return {
            "cases": self.case_total,
            "actions": self.action_total,
            "event_totals": self._nested(totals, ("kind", "mode")),
            "event_detail": self._nested(
                self.events, ("kind", "mode", "binding", "visibility")
            ),
            "case_coverage": self._nested(self.cases, ("kind", "mode")),
            "window_ticks": self._nested(
                self.window_ticks, ("kind", "mode", "ticks")
            ),
            "slack_ticks": self._nested(
                self.slack_ticks, ("kind", "mode", "ticks")
            ),
            "action_family": self._nested(
                self.by_family, ("kind", "mode", "family")
            ),
            "both_events_where_single_would_not_fit_local_window": {
                "S2PF": self.s2_both_single_would_not_fit,
                "S4PF": self.s4_both_single_would_not_fit,
            },
        }


def assignment_snap(action, cluster: int) -> FourStageSnap:
    prefix = "c2" if cluster == 2 else "c3"
    return FourStageSnap.from_assign(
        getattr(action, f"{prefix}_start"),
        getattr(action, f"{prefix}_shape_s1"),
        getattr(action, f"{prefix}_shape_s3"),
        getattr(action, f"{prefix}_ntok"),
        getattr(action, f"{prefix}_eid"),
        getattr(action, f"{prefix}_s1_cached"),
        getattr(action, f"{prefix}_s3_cached"),
        getattr(action, f"{prefix}_s2pf_start"),
        getattr(action, f"{prefix}_dma_s1"),
        getattr(action, f"{prefix}_dma_s3"),
        getattr(action, f"{prefix}_s2pf_dma"),
    )


def audit_case(row: dict, counters: UsageCounters) -> None:
    actions = [deserialize_action(item) for item in row.get("actions", ())]
    snaps = [
        make_initial_snap(int(row.get("initial_cache_c2", -1))),
        make_initial_snap(int(row.get("initial_cache_c3", -1))),
    ]
    seen: set[tuple[str, str]] = set()
    counters.case_total += 1
    counters.action_total += len(actions)

    for action in actions:
        family = action_family(action.tag)
        for cluster, index in ((2, 0), (3, 1)):
            prefix = "c2" if cluster == 2 else "c3"
            if getattr(action, f"{prefix}_eid") < 0:
                continue
            snap = assignment_snap(action, cluster)
            snaps[index] = snap
            binding = getattr(action, f"{prefix}_s2pf_dma")
            if binding == DmaBinding.NONE:
                continue
            start = getattr(action, f"{prefix}_s2pf_start")
            end = start + dma_duration(WEIGHT_BYTES_S3, binding)
            mode = binding_class(binding)
            seen.add(("S2PF", mode))
            counters.record_event(
                kind="S2PF",
                binding=binding,
                hidden=end <= snap.s2_end,
                window_cc=snap.s2_end - start,
                end_cc=end,
                compute_end_cc=snap.s2_end,
                family=family,
            )

        if action.pf_cluster in (2, 3):
            index = action.pf_cluster - 2
            snap = snaps[index]
            binding = action.pf_dma
            end = action.pf_start + dma_duration(WEIGHT_BYTES_S1, binding)
            mode = binding_class(binding)
            seen.add(("S4PF", mode))
            counters.record_event(
                kind="S4PF",
                binding=binding,
                hidden=end <= snap.compute_end,
                window_cc=snap.compute_end - action.pf_start,
                end_cc=end,
                compute_end_cc=snap.compute_end,
                family=family,
            )
            snaps[index] = snap.with_prefetch(
                action.pf_eid, action.pf_shape, action.pf_start, binding
            )

    counters.record_case_modes(seen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="append", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    references = tuple(args.reference) if args.reference else DEFAULT_REFERENCES

    counters: dict[str, UsageCounters] = defaultdict(UsageCounters)
    inputs = []
    for path in references:
        payload = json.loads(path.read_text())
        group = source_group(path)
        inputs.append({"path": str(path.resolve()), "sha256": sha256(path)})
        for row in payload["results"].values():
            audit_case(row, counters[group])
            audit_case(row, counters["combined"])
            e_group = f"{group}:E{int(row['e_total'])}"
            audit_case(row, counters[e_group])

    report = {
        "schema": "prefetch_binding_usage_audit_v1",
        "method": {
            "source": "stored best-known explicit-DMA reference histories",
            "single": ["IDMA", "XDMA"],
            "both": "BOTH",
            "hidden_definition": "prefetch_end <= local_compute_end",
            "warning": "selection frequency is not a causal policy ablation",
        },
        "inputs": inputs,
        "summary": {name: counter.report() for name, counter in sorted(counters.items())},
    }
    atomic_write(args.out, report)
    print(json.dumps(report["summary"]["final_30k"], indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
