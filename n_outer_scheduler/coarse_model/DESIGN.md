# Coarse N-outer scheduler model

## Scope and isolation

This package is independent of the pre-existing N-outer prototype and the
four-stage scheduler. It does not modify `n_outer_scheduler/model.py`,
`scheduler.py`, `task_stream.py`, `four_stage_scheduler.py`, HeMAiA, Bingo, or
RTL. Those files and certified result files are read-only baselines.

The main model is phase-granular. Blocks are never candidate actions. A fixed
block recurrence exists inside one phase operator only to enforce ping/pong
ownership and ready-only DMA arbitration. `block_golden.py` independently
expands a selected history for calibration and lowering replay.

## Execution order

Each cluster executes:

```text
expert slice
  -> Gate/Up block 0..7
       -> every token tile for the resident block
  -> Down block 0..7
       -> every token tile for the resident block
  -> next expert slice
```

The selected shape belongs to one expert slice and one phase. It is reused for
all blocks of that phase. Weight bytes are independent of shape.

For phase `p`, shape `s`, and `n` real tokens, the per-block compute window is:

```text
C(n,s,p) = first_tile_time(s,p)
         + ceil(max(0,n-M(s))/2) * M2_tile_time(p)
```

Real token slices never overlap and never contain padding. Padding exists only
inside the selected compute shape. Therefore a three-token SPLIT is `1+2` or
`2+1`, not `2+2`.

## Double-buffer successor rule

With two weight buffers, the legal target is deterministic while an internal
phase successor exists:

1. block `b < last` can prefetch only block `b+1` of the same expert/phase;
2. the final Gate/Up block can prefetch Down block 0 of the same expert;
3. the final Down block can prefetch Gate/Up block 0 of the next scheduled
   expert slice;
4. otherwise the alternate buffer remains unused.

The first block is always charged as a real DMA transfer. Whether its transfer
finishes inside the predecessor's final compute window is derived from lane
availability. There is no caller-controlled `first_block_ready` flag.

## Shape and DMA modes

Available shapes are M8/bw32, M4/bw64, and M2/bw128. Among equal compute times,
the largest M tile is canonical. For one block of `W` bytes:

```text
L_single = ceil(W / 64)
L_both   = ceil(W / 128)
```

If `L_single <= C`, BOTH is locally dominated and is not generated. Otherwise
BOTH is the hidden-latency mode and IDMA/XDMA remain explicit exposed-stall
fallbacks. This produces the expected rules without hard-coding token counts:

- a normal M8 phase uses one lane;
- a normal M4 phase uses one lane;
- a two-token M2 phase may use BOTH, IDMA, or XDMA.

The analysis bank preserves four joint categories: parallel singles,
BOTH+BOTH, C0-BOTH/C1-single, and C0-single/C1-BOTH. K4 and K8 remain software
ablations. The frozen RTL-oriented bank is smaller and structural:

1. mode 0 always uses IDMA for cluster 0 and XDMA for cluster 1;
2. mode 1 uses BOTH for every task and phase, and exists only when every use of
   BOTH is legal;
3. one-sided BOTH and phase-mixed BOTH are not generated.

Thus a SINGLE or PAIR skeleton requires at most two timespan evaluations. Mode
0 is both the no-contention baseline and the deterministic tie winner.

## Macro state and timespan

The state carried between actions is:

```text
cluster_free_cc[2]
prefetch_release_cc[2]
idma_free_cc
xdma_free_cc
```

`prefetch_release_cc[c]` is the start of cluster `c`'s final Down compute
block. It is a release time, not a promise that the next block is resident.

The reference operator can evaluate 18 finite C0/C1 phase-priority orders for
calibration. The RTL-oriented policy does not search them. It uses one
deterministic `binding_chain` order: the task with more BOTH-bound phases runs
its four phase operations first; ties prioritize the larger total resident-
block compute window and then the lower cluster ID. Inside that one order, a
compile-time recurrence processes the fixed block count. A load can start only
when:

- its unique successor relation is known;
- its ping/pong slot has been released;
- its chosen DMA lane set is free;
- the request is currently ready.

The recurrence uses counters, addition, maximum, comparison, and a fixed
priority choice. It never reserves a future DMA interval before the request is
ready. Resource waits and compute stalls increase the candidate makespan.

## Candidate boundary

`candidate_adapter.py` accepts only:

```text
kind = SINGLE | PAIR | SPLIT
(cluster, expert ID, token start, real token count)
```

No four-stage shape, S1/S2/S3/S4 state, cache metadata, prefetch metadata, or
score crosses the adapter. N-outer materializes and scores its own modes.

## Current policy family

The simple baseline and the bounded policy are deliberately separate:

1. `fixed_lane_lpt` assigns experts by their measured cold single-lane service
   cost and fixes IDMA to cluster 0 and XDMA to cluster 1.
2. `paired_lpt_mode_search` keeps the LPT partition/order, pairs the two cluster
   heads, evaluates the structural two-mode bank with one fixed service-order
   rule, and greedily selects:

   ```text
   primary   = max(C0_end, C1_end, iDMA_free, xDMA_free)
   secondary = max(C0_end, C1_end)
   tie       = mode_id, where fixed-lane mode 0 precedes all-BOTH mode 1
   ```

   Stall and the mode signature are later deterministic ties only. This score
   does not expand future rounds and has no trained coefficients.
3. `split_hot_lpt_mode_search` compares complete schedules for aligned legal
   cuts of only the hottest expert. The duplicated weight traffic is charged
   normally. Since cut selection currently observes full-history results, it
   is an offline upper-bound ablation, not the RTL policy.

The frozen RTL-oriented policy is no-SPLIT LPT + paired heads + local
`rtl_symmetric2` + fixed-first ties. The full 65-case analysis reports
fixed-lane and offline top-1 SPLIT separately. It does not claim global
N-outer optimality.

## Final 65-case evidence

The authoritative run is
`results/policy_search/n_outer_coarse_final_policy_65_symmetric2_fixed_first_final.json`.
Its embedded source hashes match the current model sources, all 65 histories
and task-ABI replays validate, and the analyzed report has zero policy-selection
regret inside the reported three-policy bank.

- Relative to fixed-lane LPT, the frozen main policy is better in 26 cases,
  equal in 39, and worse in 0. Its mean saving is 8,383 cc per case.
- Across 1,167 selected actions, 1,136 use the fixed-lane mode and only 31 use
  all-BOTH (5 PAIR and 26 SINGLE actions). The second mode is therefore rarely
  selected but produces measurable gains without introducing regressions.
- Relative to the pre-existing certified four-stage result on identical atomic
  work, its executable ratio has mean 1.213359, median 1.208333, and maximum
  1.282895. N-outer is slower on all 65 cases; this is a measured dataflow
  comparison, not an optimality claim.
- The local fixed-first scorer has better mean and median than the projected
  future-cost scorer. Projected fixed-first obtains mean 1.213652 and median
  1.210417, while requiring future per-cluster work accumulators.
- Projected BOTH-first improves the maximum ratio slightly to 1.280702 but has
  a worse mean (1.213488) and adds future-state cost. It is not selected.
- Local BOTH-first is rejected: it is worse than fixed-lane in 31 of 65 cases
  and raises the mean ratio to 1.229556. The 27 prefix-level executable-oracle
  ties therefore do not justify choosing BOTH without continuation state.
- The deterministic `binding_chain` service rule is equal to the best-of-18
  macro policy in 63 cases, better after executable replay in 2, and worse in
  0. The 18-order search is not part of the frozen policy.
- Full-history macro timing differs from dependency-only task replay in only 2
  cases, with maximum absolute error 14,080 cc. Reported performance always
  uses the executable task replay, never the macro prediction.
- 42 focused coarse-model tests and all 14 pre-existing N-outer tests pass.
- `bingo_task_abi.py` lowers a selected macro history into self-contained
  LOAD/COMPUTE task arguments and dependency edges. The production 344-byte
  S1/S2/S3/S4 record is not reused; the remaining physical integration boundary
  is recorded in `BINGO_ABI_MAPPING.md`.
