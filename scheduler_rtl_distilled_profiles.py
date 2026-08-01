#!/usr/bin/env python3
"""Hard-wired top5+bottom1 profiles for the bounded distilled scheduler.

``PROFILE_SPECS`` is a Python mirror of combinational decode cases.  It is not
runtime-programmable storage and does not require a ROM in RTL.
"""

from __future__ import annotations

from scheduler_rtl_distilled_types import (
    CandidateProfile,
    LogicalActionSpec,
    PhysicalProfile,
)


PROFILE_FIELD_ORDER = (
    "c2_s1", "c2_s3", "c3_s1", "c3_s3",
    "c2_dma_s1", "c2_dma_s3", "c2_s2pf",
    "c3_dma_s1", "c3_dma_s3", "c3_s2pf", "s4pf_dma",
    "c2_s1_cached", "c2_s3_cached", "c3_s1_cached", "c3_s3_cached",
)


PROFILE_SPECS = (
    (("ONE_IDLE", "SINGLE", ("B0",), "NONE"), ("NONE", "NONE", "C(M2,bw128)", "C(M2,bw128)", "NONE", "NONE", "NONE", "BOTH", "BOTH", "NONE", "NONE", False, False, False, False)),
    (("ONE_IDLE", "SINGLE", ("B0",), "NONE"), ("C(M2,bw128)", "C(M2,bw128)", "NONE", "NONE", "BOTH", "BOTH", "NONE", "NONE", "NONE", "NONE", "NONE", False, False, False, False)),
    (("ONE_IDLE", "SINGLE", ("T0",), "NONE"), ("B(M4,bw64)", "B(M4,bw64)", "NONE", "NONE", "BOTH", "NONE", "BOTH", "NONE", "NONE", "NONE", "NONE", False, True, False, False)),
    (("ONE_IDLE", "SINGLE", ("T0",), "NONE"), ("NONE", "NONE", "B(M4,bw64)", "B(M4,bw64)", "NONE", "NONE", "NONE", "BOTH", "NONE", "BOTH", "NONE", False, False, False, True)),
    (("ONE_IDLE", "SINGLE", ("T3",), "NONE"), ("B(M4,bw64)", "B(M4,bw64)", "NONE", "NONE", "BOTH", "NONE", "BOTH", "NONE", "NONE", "NONE", "NONE", False, True, False, False)),
    (("ONE_IDLE", "SINGLE", ("T3",), "NONE"), ("NONE", "NONE", "B(M4,bw64)", "B(M4,bw64)", "NONE", "NONE", "NONE", "BOTH", "NONE", "BOTH", "NONE", False, False, False, True)),
    (("SYNC", "PAIR", ("B0", "T0"), "NONE"), ("A(M8,bw32)", "B(M4,bw64)", "B(M4,bw64)", "B(M4,bw64)", "IDMA", "NONE", "XDMA", "XDMA", "IDMA", "NONE", "NONE", False, True, False, False)),
    (("SYNC", "PAIR", ("T0", "T1"), "NONE"), ("B(M4,bw64)", "B(M4,bw64)", "B(M4,bw64)", "B(M4,bw64)", "IDMA", "IDMA", "NONE", "XDMA", "XDMA", "NONE", "NONE", False, False, False, False)),
    (("SYNC", "PAIR", ("T0", "T1"), "NONE"), ("B(M4,bw64)", "B(M4,bw64)", "B(M4,bw64)", "B(M4,bw64)", "IDMA", "NONE", "IDMA", "XDMA", "IDMA", "NONE", "NONE", False, True, False, False)),
    (("SYNC", "PAIR", ("T0", "T4"), "NONE"), ("A(M8,bw32)", "B(M4,bw64)", "B(M4,bw64)", "B(M4,bw64)", "IDMA", "NONE", "XDMA", "XDMA", "IDMA", "NONE", "NONE", False, True, False, False)),
    (("SYNC", "PAIR", ("T1", "T2"), "NONE"), ("B(M4,bw64)", "B(M4,bw64)", "B(M4,bw64)", "B(M4,bw64)", "IDMA", "NONE", "IDMA", "XDMA", "NONE", "XDMA", "NONE", False, True, False, True)),
    (("SYNC", "PAIR", ("T2", "T3"), "NONE"), ("B(M4,bw64)", "B(M4,bw64)", "B(M4,bw64)", "B(M4,bw64)", "IDMA", "NONE", "IDMA", "XDMA", "NONE", "XDMA", "NONE", False, True, False, True)),
    (("TERMINAL", "SINGLE", ("T0",), "NONE"), ("C(M2,bw128)", "C(M2,bw128)", "NONE", "NONE", "BOTH", "BOTH", "NONE", "NONE", "NONE", "NONE", "NONE", False, False, False, False)),
    (("TERMINAL", "SINGLE", ("T0",), "NONE"), ("NONE", "NONE", "C(M2,bw128)", "C(M2,bw128)", "NONE", "NONE", "NONE", "BOTH", "BOTH", "NONE", "NONE", False, False, False, False)),
    (("TERMINAL", "SPLIT", ("T0",), "BALANCED"), ("B(M4,bw64)", "B(M4,bw64)", "B(M4,bw64)", "B(M4,bw64)", "IDMA", "IDMA", "NONE", "XDMA", "XDMA", "NONE", "NONE", False, False, False, False)),
    (("ONE_IDLE", "SINGLE", ("T0",), "NONE"), ("A(M8,bw32)", "B(M4,bw64)", "NONE", "NONE", "XDMA", "IDMA", "NONE", "NONE", "NONE", "NONE", "NONE", False, False, False, False)),
    (("ONE_IDLE", "SINGLE", ("T0",), "NONE"), ("B(M4,bw64)", "B(M4,bw64)", "NONE", "NONE", "XDMA", "IDMA", "NONE", "NONE", "NONE", "NONE", "NONE", False, False, False, False)),
    (("ONE_IDLE", "SINGLE", ("T0",), "NONE"), ("NONE", "NONE", "B(M4,bw64)", "B(M4,bw64)", "NONE", "NONE", "NONE", "XDMA", "XDMA", "NONE", "NONE", False, False, False, False)),
    (("ONE_IDLE", "SINGLE", ("T0",), "NONE"), ("NONE", "NONE", "C(M2,bw128)", "C(M2,bw128)", "NONE", "NONE", "NONE", "BOTH", "BOTH", "NONE", "NONE", False, False, False, False)),
    (("ONE_IDLE", "SINGLE", ("T0",), "NONE"), ("C(M2,bw128)", "C(M2,bw128)", "NONE", "NONE", "BOTH", "BOTH", "NONE", "NONE", "NONE", "NONE", "NONE", False, False, False, False)),
    (("ONE_IDLE", "SINGLE", ("T0",), "NONE"), ("B(M4,bw64)", "B(M4,bw64)", "NONE", "NONE", "XDMA", "XDMA", "NONE", "NONE", "NONE", "NONE", "NONE", False, False, False, False)),
    (("ONE_IDLE", "SINGLE", ("T0",), "NONE"), ("A(M8,bw32)", "B(M4,bw64)", "NONE", "NONE", "IDMA", "IDMA", "NONE", "NONE", "NONE", "NONE", "NONE", False, False, False, False)),
    (("ONE_IDLE", "SINGLE", ("T0",), "NONE"), ("B(M4,bw64)", "C(M2,bw128)", "NONE", "NONE", "XDMA", "BOTH", "NONE", "NONE", "NONE", "NONE", "NONE", False, False, False, False)),
    (("ONE_IDLE", "SINGLE", ("T0",), "NONE"), ("NONE", "NONE", "A(M8,bw32)", "B(M4,bw64)", "NONE", "NONE", "NONE", "IDMA", "IDMA", "NONE", "NONE", False, False, False, False)),
    (("TERMINAL", "SINGLE", ("T0",), "NONE"), ("NONE", "NONE", "B(M4,bw64)", "B(M4,bw64)", "NONE", "NONE", "NONE", "XDMA", "XDMA", "NONE", "NONE", False, False, False, False)),
    (("ONE_IDLE", "SINGLE", ("T0",), "NONE"), ("NONE", "NONE", "B(M4,bw64)", "C(M2,bw128)", "NONE", "NONE", "NONE", "XDMA", "BOTH", "NONE", "NONE", False, False, False, False)),
    (("ONE_IDLE", "SINGLE", ("T0",), "NONE"), ("NONE", "NONE", "B(M4,bw64)", "B(M4,bw64)", "NONE", "NONE", "NONE", "IDMA", "IDMA", "NONE", "NONE", False, False, False, False)),
    (("ONE_IDLE", "SINGLE", ("T0",), "NONE"), ("C(M2,bw128)", "B(M4,bw64)", "NONE", "NONE", "BOTH", "XDMA", "NONE", "NONE", "NONE", "NONE", "NONE", False, False, False, False)),
    (("ONE_IDLE", "SINGLE", ("T0",), "NONE"), ("B(M4,bw64)", "B(M4,bw64)", "NONE", "NONE", "IDMA", "XDMA", "NONE", "NONE", "NONE", "NONE", "NONE", False, False, False, False)),
    (("TERMINAL", "SINGLE", ("T0",), "NONE"), ("B(M4,bw64)", "C(M2,bw128)", "NONE", "NONE", "XDMA", "BOTH", "NONE", "NONE", "NONE", "NONE", "NONE", False, False, False, False)),
    (("SYNC", "SPLIT", ("T0",), "HALF"), ("A(M8,bw32)", "B(M4,bw64)", "A(M8,bw32)", "B(M4,bw64)", "IDMA", "IDMA", "NONE", "XDMA", "XDMA", "NONE", "NONE", False, False, False, False)),
    (("SYNC", "PAIR", ("T0", "T1"), "NONE"), ("C(M2,bw128)", "C(M2,bw128)", "C(M2,bw128)", "C(M2,bw128)", "NONE", "NONE", "NONE", "BOTH", "BOTH", "NONE", "NONE", True, True, False, False)),
)


def _compile_profiles() -> tuple[CandidateProfile, ...]:
    if tuple(PhysicalProfile.__dataclass_fields__) != PROFILE_FIELD_ORDER:
        raise AssertionError("PhysicalProfile field order changed")
    profiles = tuple(
        CandidateProfile(
            logical=LogicalActionSpec(
                mode=mode,
                family=family,
                selectors=selectors,
                split_rule=split_rule,
            ),
            physical=PhysicalProfile(*physical),
        )
        for (mode, family, selectors, split_rule), physical in PROFILE_SPECS
    )
    if len(profiles) != len(set(profiles)):
        raise AssertionError("duplicate distilled physical profile")
    if any(
        token.logical.mode == "SYNC" and token.logical.family == "SINGLE"
        for token in profiles
    ):
        raise AssertionError("dominated SYNC SINGLE family must remain absent")
    visible = {f"T{rank}" for rank in range(5)} | {"B0"}
    selectors = {
        selector for token in profiles for selector in token.logical.selectors
    }
    if selectors - visible:
        raise AssertionError(
            "profile bank references selectors outside top5+bottom1: "
            f"{sorted(selectors - visible)}"
        )
    return profiles


COMPILED_PROFILES = _compile_profiles()
