#!/usr/bin/env python3
"""Generate exhaustive profile/token endpoint vectors for distilled RTL."""

import four_stage_scheduler as reference
from scheduler_rtl_distilled_profiles import COMPILED_PROFILES

SHAPE = {
    "A(M8,bw32)": reference.SHAPE_A,
    "B(M4,bw64)": reference.SHAPE_B,
    "C(M2,bw128)": reference.SHAPE_C,
}
SHAPE_ID = {reference.SHAPE_A: 0, reference.SHAPE_B: 1, reference.SHAPE_C: 2}
DMA = {name: reference.DmaBinding[name] for name in ("NONE", "IDMA", "XDMA", "BOTH")}


def main() -> int:
    rows = []
    for slot, token in enumerate(COMPILED_PROFILES):
        profile = token.physical
        for cluster in (2, 3):
            prefix = f"c{cluster}"
            shape_s1_name = getattr(profile, f"{prefix}_s1")
            if shape_s1_name == "NONE":
                continue
            shape_s1 = SHAPE[shape_s1_name]
            shape_s3 = SHAPE[getattr(profile, f"{prefix}_s3")]
            dma_s1 = DMA[getattr(profile, f"{prefix}_dma_s1")]
            dma_s3 = DMA[getattr(profile, f"{prefix}_dma_s3")]
            s2pf_dma = DMA[getattr(profile, f"{prefix}_s2pf")]
            s1_cached = bool(getattr(profile, f"{prefix}_s1_cached"))
            s3_cached = bool(getattr(profile, f"{prefix}_s3_cached")) and (
                s2pf_dma == reference.DmaBinding.NONE
            )
            for ntok in range(1, 257):
                start_tick = slot * 7 + cluster + ntok % 5
                start = start_tick * reference.SCHEDULE_TIME_QUANTUM_CC
                snap = reference.FourStageSnap.from_assign(
                    start, shape_s1, shape_s3, ntok, slot,
                    s1_cached=s1_cached, s3_cached=s3_cached,
                    dma_s1=dma_s1, dma_s3=dma_s3,
                )
                if s2pf_dma != reference.DmaBinding.NONE:
                    snap = snap.with_s2_down_prefetch(
                        shape_s3, snap.dma1_end, s2pf_dma
                    )
                rows.append((
                    slot, cluster, ntok, start_tick,
                    SHAPE_ID[shape_s1], SHAPE_ID[shape_s3],
                    int(s1_cached), int(s3_cached),
                    int(dma_s1), int(dma_s3), int(s2pf_dma),
                    snap.task_end // reference.SCHEDULE_TIME_QUANTUM_CC,
                    snap.dma1_end // reference.SCHEDULE_TIME_QUANTUM_CC,
                    snap.s2_end // reference.SCHEDULE_TIME_QUANTUM_CC,
                    snap.dma3_end // reference.SCHEDULE_TIME_QUANTUM_CC,
                    snap.compute_end // reference.SCHEDULE_TIME_QUANTUM_CC,
                    int(snap.s2pf_start >= 0),
                    (snap.s2pf_start // reference.SCHEDULE_TIME_QUANTUM_CC
                     if snap.s2pf_start >= 0 else 0),
                    (snap.s2pf_end // reference.SCHEDULE_TIME_QUANTUM_CC
                     if snap.s2pf_end >= 0 else 0),
                    int(snap.dma_s1), int(snap.dma_s3), int(snap.s2pf_dma),
                ))
    print(len(rows))
    for row in rows:
        print(*row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
