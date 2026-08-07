#!/usr/bin/env python3
"""Emit exhaustive vectors for the fixed distilled RTL profile decoder."""

from scheduler_rtl_distilled_profiles import COMPILED_PROFILES
from scheduler_rtl_distilled_types import LOGICAL_ACTION_PRIORITY

MODE = {"TERMINAL": 0, "SYNC": 1, "ONE_IDLE": 2}
FAMILY = {"SINGLE": 0, "PAIR": 1, "SPLIT": 2}
SELECTOR = {f"T{rank}": rank for rank in range(5)} | {"B0": 5}
SHAPE = {"NONE": 2, "A(M8,bw32)": 0, "B(M4,bw64)": 1, "C(M2,bw128)": 2}
DMA = {"NONE": 0, "IDMA": 1, "XDMA": 2, "BOTH": 3}


def logical_id(token) -> int:
    logical = token.logical
    return LOGICAL_ACTION_PRIORITY[(
        logical.mode, logical.family, logical.selectors, logical.split_rule,
    )]


def main() -> int:
    slots_by_mode = {
        mode: sorted(
            (slot for slot, token in enumerate(COMPILED_PROFILES)
             if token.logical.mode == mode),
            key=lambda slot: (logical_id(COMPILED_PROFILES[slot]), slot),
        )
        for mode in MODE
    }
    print(len(COMPILED_PROFILES))
    for slot, token in enumerate(COMPILED_PROFILES):
        logical = token.logical
        profile = token.physical
        index = slots_by_mode[logical.mode].index(slot)
        key = (logical.mode, logical.family, logical.selectors, logical.split_rule)
        selectors = list(logical.selectors) + [logical.selectors[0]]
        values = (
            MODE[logical.mode], index, slot, LOGICAL_ACTION_PRIORITY[key],
            FAMILY[logical.family], SELECTOR[selectors[0]], SELECTOR[selectors[1]],
            int(logical.split_rule == "BALANCED"),
            int(profile.c2_s1 != "NONE"), int(profile.c3_s1 != "NONE"),
            SHAPE[profile.c2_s1], SHAPE[profile.c2_s3],
            SHAPE[profile.c3_s1], SHAPE[profile.c3_s3],
            DMA[profile.c2_dma_s1], DMA[profile.c2_dma_s3], DMA[profile.c2_s2pf],
            DMA[profile.c3_dma_s1], DMA[profile.c3_dma_s3], DMA[profile.c3_s2pf],
            int(profile.c2_s1_cached), int(profile.c2_s3_cached),
            int(profile.c3_s1_cached), int(profile.c3_s3_cached),
        )
        print(*values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
