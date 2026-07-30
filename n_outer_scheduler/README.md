# N-outer scheduler model

This directory is independent of `four_stage_scheduler.py` and the resident
M-outer lineage.  It models a block-major static execution template for a small
TCDM with exactly two weight buffers per cluster.

## Static contract

For each cluster and phase, the dynamic expert list is flattened as:

```text
(block0, expert0), ..., (block0, expertN-1),
(block1, expert0), ..., (block1, expertN-1), ...
```

The fixed DMA worker produces this stream and the fixed compute worker consumes
it.  Work item `k` uses buffer `k & 1`; load `k` cannot overwrite the buffer
until compute `k-2` completes.  The initial work item must be loaded before any
compute.  Thereafter the producer tries to overlap load `k+1` with compute `k`.

Gate/Up and Down are separate phase specifications but form one continuous
item stream, so the first Down block may be loaded during the final Gate/Up
compute when the alternate buffer is free.  This is fixed one-item lookahead,
not the resident model's S2PF/S4PF candidate.  It is still prefetch: the
scheduler chooses when the eligible lookahead load starts and whether it uses
one DMA lane, both lanes, or waits for a better grant point.

## Dynamic parameters

The static workers do not contain a fixed `16` or `4+2+...` schedule.  A group
descriptor supplies the ordered expert list for each cluster.  Each expert
descriptor supplies its expert ID, token count and token offsets.  The current
micro-tiling rule uses M4 for the bulk, one padded M4 for a three-token tail,
or one M2 for a one/two-token tail.  CVA6 lowering can materialize `m4_iters`,
`m2_iters` and valid-tail metadata without putting division logic in the
hardware scheduler.

Phase block counts, bytes, expert/block strides and ping/pong addresses belong
to the static model context.  There are no per-round dynamic arguments: the
workers derive the round and successor from their loop indices.

## Executable task-stream contract

`task_stream.py` is the boundary between the scheduler model and Bingo.  It
lowers every selected weight-block item into two explicit operations:

```text
LOAD_WEIGHT(eid, phase, block, bytes, ping/pong slot, DMA lane mask)
COMPUTE_BLOCK(eid, phase, block, token range, M4/M2 shape iterations)
```

Every operation carries task IDs on which it depends.  The mandatory edges
are load-before-compute, previous-compute-before-next-compute, ping/pong slot
release (`compute[i-2] -> load[i]`), and the scheduler-selected order on each
DMA lane.  An intentional DMA wait is encoded as another event dependency,
not as an absolute start cycle.  Consequently the same table is both a timing
model and a device-execution description.

`issue_order` is a deterministic topological order for filling Bingo ready
queues.  It is not a command to serialize task completion globally.  Bingo (or
the CVA6 producer) must release a task only after all `depends_on` task IDs are
complete; otherwise a FIFO order alone cannot preserve the modeled two-cluster
overlap.

Two startup contracts are explicit:

- `cold`: both first weight blocks are ordinary DMA tasks and replay exactly
  matches the source scheduler history;
- `preloaded_first`: the first block of each non-empty cluster is declared
  resident, those two LOAD tasks are omitted, and the replay reports the
  steady-state makespan.  The removed cost must be paid by an earlier group or
  by end-to-end initialization and must not be silently discarded.

Generate a concrete JSON task table from `Idea_Model` with:

```bash
python3 -m n_outer_scheduler.run_task_stream \
  --tokens 16,4,2,2,2,2,2,2 \
  --mode fast \
  --startup cold \
  --output results/n_outer_task_stream.json
```

The JSON contains scheduler-selected expert order, every task argument,
dependency IDs, DMA lane ownership, issue order, and both source/replayed
makespans.  This is the input format to map onto the revised Bingo task word
and argument records; the existing S1/S2/S3/S4 record is not assumed.

## Candidate evaluation

`model.py` runs an event simulation with:

- two cluster-local sequential compute consumers;
- two cluster-local sequential DMA producers;
- two global non-preemptive 64 B/cc DMA lanes;
- single-lane and BOTH transfers;
- exact load-before-compute and two-buffer ownership checks.

Every candidate remains executable by waiting, so bandwidth is not represented
as a post-hoc legal/illegal flag.  The evaluator reports real makespan, initial
prime time, steady-state stalls, compute/DMA utilization and a relaxed lower
bound.  Candidates are ranked lexicographically by makespan, steady stall and
lower-bound overhead.  No arbitrary weighted score is used.

The deadline-aware lane policy is a deterministic fast implementation policy,
not an optimality certificate.  `reference.py` supplies a separate exact
fixed-candidate solver without changing the static descriptor contract.  It
explicitly branches over legal SINGLE, SPLIT, BOTH and deliberate WAIT grants, memoizes
future-equivalent states and prunes only when an admissible residual lower
bound cannot improve an already solved branch.

## Relation to the M-outer candidate flow

The outer search structure remains useful, but its action meanings change:

- a candidate chooses the cluster assignment and within-cluster expert order;
- M4/M2 lowering is deterministic from each expert's token count;
- one-item lookahead is part of the static double-buffer worker, not a
  separately enumerated S2PF/S4PF action;
- DMA feasibility is enforced at every transfer by lane ownership and buffer
  release times, then audited by `validate_history`;
- the objective is minimum simulated makespan, with steady stall and
  lower-bound overhead used only as deterministic tie breakers.

Consequently there is no weighted "highest score" in this reference model.
Every accepted candidate is resource-legal; a candidate that requests too
much simultaneous bandwidth waits in the event model and receives a longer
makespan.  A later lightweight hardware scheduler may approximate this ranking,
but it should be distilled against these measured candidate results rather
than changing the reference objective.

## Scheduler policy

`scheduler.py` selects the two ordered expert lists and their DMA-prefetch
schedule.  It exposes three modes:

- `fast`: evaluate every generated mapping with the deadline-aware DMA policy;
- `hybrid`: fast-rank every mapping, then exactly solve the first `K` mappings;
- `reference`: exactly solve every mapping in the generated candidate bank.

The primary cost is the final compute makespan.  Steady compute stall and
lower-bound overhead are deterministic tie breakers, not weighted terms.  A
candidate lower bound is the maximum of its two cluster-local compute chains
and total remaining weight bytes divided by 128 B/cc.

Bandwidth is enforced at event level.  Each transfer owns lane 0, lane 1, or
both lanes for its full non-preemptive interval.  `validate_history` checks
per-lane overlap and ping/pong ownership; `audit_bandwidth` independently
sweeps all transfer boundaries and requires peak bandwidth to remain at or
below 128 B/cc.  Contention delays a load and increases makespan instead of
being hidden in a feasibility score.

Run the scheduler directly with:

```bash
python3 -m n_outer_scheduler.run_scheduler \
  --tokens 16,4,2,2,2,2,2,2 \
  --mode hybrid \
  --exact-top 10
```

The output reports mapping, makespan/LB/stall, bandwidth validity, prefetch
overlap, and the exactness scope.  `dma_optimal=True` proves the selected
candidate's DMA plan; only `bank_optimal=True` proves all generated candidates
were exactly evaluated.  `permutation_complete=False` remains explicit.

## Run the directed example

From `Idea_Model`:

```bash
python3 -m n_outer_scheduler.run_search \
  --tokens 16,4,2,2,2,2,2,2 \
  --top 10
```

The search exhaustively enumerates cluster partitions for up to 16 active
experts after removing cluster symmetry.  It evaluates original, descending
and ascending token-count orders on each side.  Without `--exact-top`, it does
not claim exhaustive permutation or DMA-grant optimality.

Prove the DMA-grant optimum for the first ten fast-ranked candidates with:

```bash
python3 -m n_outer_scheduler.run_search \
  --tokens 16,4,2,2,2,2,2,2 \
  --top 10 \
  --exact-top 10
```

Use `--exact-top -1` to prove every candidate in the generated bank.  This
certifies the best DMA schedule inside that bank, but the bank still contains
only the three documented order variants; it is not a proof over every expert
permutation.  The CLI prints these two scopes separately so a bounded result
cannot be reported as a global optimum.

Add `--output results/n_outer_directed_reference.json` to save the input,
matrix/block configuration, source SHA-256 values, candidate-space scope,
ranked metrics and every exact DMA grant plan.  The file is written atomically;
the final `result_written=...` line confirms completion.

Run the focused tests with:

```bash
python3 -m unittest discover -s n_outer_scheduler/tests -v
```

## Paired four-stage diagnostic

Before interpreting a loop-order comparison, run the paired small-case audit:

```bash
python3 -m n_outer_scheduler.compare_four_stage \
  --output results/n_outer_vs_four_stage_small.json
```

For N-outer this enumerates every ordered, no-SPLIT two-cluster partition and
exactly solves every fixed candidate's DMA grants.  The four-stage side uses
its certified anytime search and validates the returned history.  Both sides
must first agree on total compute work and transferred weight bytes.  The
output explicitly marks a case as not directly search-space comparable when
the four-stage optimum uses SPLIT, because the current N-outer candidate bank
does not yet support splitting one expert across both clusters.
