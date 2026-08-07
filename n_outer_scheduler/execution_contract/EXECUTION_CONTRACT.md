# Frozen block-major N-outer semantics

## 1. Scope

This contract begins after a candidate has produced ordered slots.  Each slot
contains one complete ordered expert-slice list for C0 and one for C1.
Candidate construction may use PAIR, SINGLE, or SPLIT; the selected slices and
slot boundaries are preserved, while candidate-search bookkeeping is not.

The evaluator returns the resource-legal makespan of the complete group.  It
does not define the candidate bank, a lookahead scorer, or a search width.

## 2. Dynamic group input

For each slot and cluster, the descriptor contains an ordered list of:

```text
(expert_id, token_start, real_token_count)
```

Disjoint slices of one expert may reside on different clusters.  Padding never
changes the real token interval.  CVA6 lowering derives:

```text
m4_iters
m2_iters
m4_tail_valid_tokens
m2_valid_tokens
```

The default decomposition uses M4 for the bulk, a padded M4 for a three-token
tail, and one M2 for a one/two-token tail.

## 3. Static loop nest

Each cluster executes independently:

```text
for local_slot in schedule order:
    for phase in [Gate/Up, Down]:
        for block in phase.block_count:
            for slice in local_slot.cluster_order:
                wait_weight_ready(local_slot, phase, block, slice)
                compute_all_token_tiles(slice)
                release_weight_buffer()
```

The LOAD worker walks the identical `(slot, phase, block, slice)` stream.  Therefore
the next item after `(block, slice[i])` is `(block, slice[i+1])`; after the last
slice it is `(block+1, slice[0])`; after the final Gate/Up item it is the first
Down item; after the final Down item it is the first Gate/Up item of the next
non-empty local slot.

There is no global slot barrier.  C0 may enter its next slot while C1 remains
in the prior slot.  Only compute/DMA/buffer dependencies constrain progress.

An implementation that completes all phases/blocks of one expert before the
next expert is not this contract.

## 4. Tick model

One scheduler tick is 1408 accelerator cycles.  Default atomic costs are:

| Operation per block | Gate/Up | Down |
|---|---:|---:|
| single-lane LOAD | 4 | 2 |
| BOTH LOAD | 2 | 1 |
| M4 tile compute | 4 | 2 |
| M2 tile compute | 2 | 1 |

All state and event times are integer ticks.

## 5. Double buffering

The local stream index does not reset at a slot boundary.  Work item `i` uses
buffer `i & 1`.  LOAD `i` may start only after COMPUTE
`i-2` has released the same slot.  COMPUTE `i` waits for LOAD `i` and COMPUTE
`i-1`.  Thus the producer can be one item ahead without overwriting resident
weights.

The cold default charges both initial loads.  At tick zero, C0 receives iDMA
and C1 receives xDMA, each at 64 B/cc.  No compute starts before its complete
first block is resident.

## 6. DMA arbitration

iDMA and xDMA are independent non-preemptive 64 B/cc lanes.  A BOTH transfer
occupies both lanes and completes at 128 B/cc aggregate.

At every event boundary the deterministic arbiter considers the first
buffer-eligible load of each cluster.  If both lanes and both requests are
available it evaluates:

```text
two parallel single-lane loads
C0 request with BOTH, then C1
C1 request with BOTH, then C0
```

The lexicographic decision minimizes maximum deadline lateness, total
lateness, completion time, then prefers parallel singles.  With one pending
request it uses a single lane if that meets the next-compute deadline and BOTH
otherwise.  The arbiter never reserves unavailable bandwidth.

This rule keeps the directed `[16]` versus `[4,2,2,2,2,2,2]` stream free of
steady compute stalls.  The package also retains a SINGLE_ONLY ablation to
prove the causal contribution of BOTH grants.

## 7. Makespan and validity

The schedule is valid only if:

- every item has exactly one LOAD and one COMPUTE;
- LOAD completes before its COMPUTE;
- each cluster runs at most one COMPUTE at a time;
- each cluster has at most one logical LOAD outstanding;
- neither ping/pong slot is overwritten while live;
- no DMA lane has overlapping transfers.

Makespan is the maximum completion tick of both compute streams and both DMA
lanes.  Contention is represented as waiting, not as a post-hoc legal flag.

## 8. Proof boundary

The current evidence proves that the deterministic single/multi-slot model,
compact 64-bit stream, CVA6-style runtime tables, lowered audit dependencies,
and independent replay agree exactly.  It does
not yet prove physical cycle accuracy against production RTL/Bingo.  Launch,
CSR, polling, unrelated traffic, and physical L3 bank effects remain outside
this ideal accelerator-local contract until calibrated symmetrically.
