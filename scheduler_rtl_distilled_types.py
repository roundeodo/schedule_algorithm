#!/usr/bin/env python3
"""Shared immutable types and constants for the final bounded scheduler."""

from __future__ import annotations

from dataclasses import dataclass

import four_stage_scheduler as reference


POLICY_ID = "bounded-distilled-top5-bottom1-targeted-s4pf"
WINDOW = (5, 1)
MAX_PHYSICAL_CANDIDATES = 18
TICK_CC = reference.SCHEDULE_TIME_QUANTUM_CC

# The continuation relation is a deterministic fold, not an order-independent
# scalar minimum.  These mode-local IDs are therefore part of the policy ABI.
LOGICAL_ACTIONS_BY_MODE = {
    "SYNC": (
        ("PAIR", ("B0", "T0"), "NONE"),
        ("PAIR", ("T0", "T1"), "NONE"),
        ("PAIR", ("T0", "T4"), "NONE"),
        ("PAIR", ("T1", "T2"), "NONE"),
        ("PAIR", ("T2", "T3"), "NONE"),
        ("SPLIT", ("T0",), "HALF"),
    ),
    "ONE_IDLE": (
        ("SINGLE", ("B0",), "NONE"),
        ("SINGLE", ("T0",), "NONE"),
        ("SINGLE", ("T3",), "NONE"),
    ),
    "TERMINAL": (
        ("SINGLE", ("T0",), "NONE"),
        ("SPLIT", ("T0",), "BALANCED"),
    ),
}
LOGICAL_ACTION_PRIORITY = {
    (mode, family, selectors, split_rule): logical_id
    for mode, actions in LOGICAL_ACTIONS_BY_MODE.items()
    for logical_id, (family, selectors, split_rule) in enumerate(actions)
}


def logical_action_order_key(
    action_key: tuple[str, str, tuple[str, ...], str],
) -> tuple:
    """Fixed fold order, including canonical Top/Bottom overlap aliases.

    When fewer than six descriptors remain, B0 can alias a visible Top rank.
    The concrete action is canonicalized to that Top rank before local
    reduction, so the final fold order must cover those bounded aliases rather
    than assume that every runtime key is one of the static template keys.
    """
    mode, family, selectors, split_rule = action_key
    family_priority = {
        "SYNC": {"PAIR": 0, "SPLIT": 1},
        "ONE_IDLE": {"SINGLE": 0},
        "TERMINAL": {"SINGLE": 0, "SPLIT": 1},
    }

    def selector_priority(selector: str) -> tuple[int, int]:
        if selector.startswith("B"):
            return 0, int(selector[1:])
        if selector.startswith("T"):
            return 1, int(selector[1:])
        raise ValueError(f"unsupported logical selector {selector!r}")

    return (
        family_priority[mode][family],
        tuple(selector_priority(selector) for selector in selectors),
        split_rule,
    )


@dataclass(frozen=True, order=True)
class LogicalActionSpec:
    """One state-relative action selected from the bounded descriptor window."""

    mode: str
    family: str
    selectors: tuple[str, ...]
    split_rule: str = "NONE"


@dataclass(frozen=True, order=True)
class PhysicalProfile:
    """Fixed shapes, DMA bindings, prefetch fields and residency expectations."""

    c2_s1: str
    c2_s3: str
    c3_s1: str
    c3_s3: str
    c2_dma_s1: str
    c2_dma_s3: str
    c2_s2pf: str
    c3_dma_s1: str
    c3_dma_s3: str
    c3_s2pf: str
    s4pf_dma: str
    c2_s1_cached: bool
    c2_s3_cached: bool
    c3_s1_cached: bool
    c3_s3_cached: bool


@dataclass(frozen=True, order=True)
class CandidateProfile:
    """One statically encoded logical action and physical implementation."""

    logical: LogicalActionSpec
    physical: PhysicalProfile
