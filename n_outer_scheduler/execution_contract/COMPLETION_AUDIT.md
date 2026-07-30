# Block-major contract completion audit

## Corrected after semantic failure

The former execution contract used `expert -> phase -> block` and treated each
PAIR/SINGLE/SPLIT action as an execution epoch.  That model was internally
consistent but did not represent the required N-outer dataflow.  Its timing
results, including the previously reported 416-tick example, are withdrawn.

The current contract uses:

```text
phase -> block -> complete ordered expert list -> token tile
```

No production RTL, Bingo, four-stage, or pre-existing N-outer source file was
modified by this correction.

## Directed evidence

Input:

```text
C0=[16]
C1=[4,2,2,2,2,2,2]
```

Deadline-aware result:

```text
makespan_ticks=196
lower_bound_ticks=196
compute_lower_bound_ticks=196
dma_lower_bound_ticks=192
initial_wait_ticks=(4,4)
steady_stall_ticks=(0,0)
compute_utilization=0.9795918367
dma_lane_utilization=0.9795918367
```

The directed schedule reaches the cold-model lower bound exactly.

SINGLE_ONLY causal ablation:

```text
makespan_ticks=337
steady_stall_ticks=(0,141)
```

The corrected stream executes C1 Gate/Up block 0 as
`4,2a,2b,2c,2d,2e,2f`, then returns to the same sequence for block 1.
During `C(block0,E4)=[4,8)`, `L(block0,E2a)=[4,8)` overlaps exactly; the six
small-expert transitions and every later block remain free of steady VC stall.

## Automated evidence

The focused suite checks:

- exact block-major stream order;
- initial split-lane prime;
- directed zero-steady-stall result;
- causal SINGLE_ONLY regression;
- ping/pong ownership and DMA lane exclusion;
- 25-bit descriptor pack/unpack;
- fixed topology independent of distribution;
- exact model/lowering/replay event equality;
- 40 deterministic random group replays;
- legal and overlapping SPLIT slices.

Current focused result: 13 tests pass.

## Remaining boundary

This completes the isolated Python execution/lowering contract.  Production
RTL and Bingo integration have not been implemented or physically measured.
Those are separate future steps and must not be described as verified.
