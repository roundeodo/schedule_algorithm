# N-outer Bingo ABI mapping audit

## Scope

This is a read-only audit of the current HeMAiA/Bingo implementation and the
contract implemented by `bingo_task_abi.py`. No HeMAiA, Bingo, M-outer, or RTL
file is modified.

## Why the current slot record cannot be reused as-is

The current dynamic expert record is semantically M-outer:

- `device_kernel_args.h:842-898` defines `ctrl` bits for `skip_s1..skip_s4`,
  `shape_s1`, `shape_s3`, `dma_s1`, and `dma_s3`, followed by S1/S2/S3/S4 call
  records.
- `device_kernel_args.h:900-905` fixes that record to 344 bytes inside a
  384-byte dynamic slot.
- `host_moe_sw_path.h:27-102` lowers one scheduler task into exactly those four
  stage calls and derives S2/S4 tail work from S1/S3 shapes.
- `moe_dynamic_slot_dfg.py:197-289` constructs an S1 -> S2 -> S3 -> S4 ->
  store task chain.

N-outer instead executes, for one expert slice:

```text
Gate/Up block 0..B-1, all token tiles per resident block
Down    block 0..B-1, all token tiles per resident block
```

Reinterpreting the current S1/S2/S3/S4 fields would silently change their
meaning and would not describe the required expert -> phase -> block order.
Therefore the Python model does not patch or alias the existing ABI.

## What is reusable

The following mechanisms are independent of M-outer stage semantics:

1. a fixed DFG created before runtime;
2. one cluster-local slot sequence per compute cluster;
3. a dynamic argument base, static argument base, and pipeline-control base;
4. host lowering from a compact scheduler result into complete device
   arguments;
5. task dependencies instead of absolute start-cycle commands.

The current generator already keeps C2 and C3 slot sequences independent
(`main_bingo.py:1263-1280`). N-outer preserves this property.

## Proposed independent N-outer ABI

`bingo_task_abi.py` has two levels.

### Macro record

One selected expert slice carries:

- action and cluster-local slot order;
- cluster, expert ID, real token start, and real token count;
- Gate/Up shape and DMA mask;
- Down shape and DMA mask;
- the finite service-priority ranks derived from the deterministic
  `binding_chain` rule. A physical ABI may derive these instead of storing
  them.

No block descriptor is generated or scored by the scheduler.

### Lowered fixed-task arguments

The host lowering pass expands each macro record into fixed tasks. A load task
contains:

- cluster and macro-slot selector;
- expert, phase, block, and ping/pong selector;
- weight byte count and DMA lane mask;
- dependency IDs encoding lane order and buffer release.

A compute task contains:

- cluster and macro-slot selector;
- expert, phase, block, shape, and ping/pong selector;
- real token start and count;
- dependency IDs for the matching load and previous compute.

Physical addresses remain in static context. Device code derives a weight
source from `(eid, phase, block)` and a destination from
`(cluster, ping_pong)`. This prevents duplicated physical addresses in every
dynamic task and keeps the model independent of the current L1 allocation.

## Dependency contract

The lowering emits no absolute start timestamp.

1. compute `i` depends on load `i` and compute `i-1`;
2. load `i` depends on load `i-1` because one cluster-local DMA worker walks
   that cluster's stream;
3. load `i` depends on compute `i-2` before reusing the same ping/pong slot;
4. every load depends on the preceding load on each DMA lane it uses;
5. a BOTH transfer therefore depends on both preceding lane chains and blocks
   both successor chains until completion.

These edges are sufficient to reproduce the selected non-preemptive DMA
order. They also avoid a runtime search and avoid a shared multi-writer RMW
counter.

## Current validation boundary

Implemented and automatically checked:

- macro history and SPLIT token coverage;
- deterministic macro-to-task expansion;
- task DAG completeness and acyclicity;
- exact replay of the independent block golden makespan;
- no iDMA/xDMA overlap;
- cluster compute order;
- no ping/pong overwrite before compute release.

Not implemented in the existing HeMAiA tree:

- C structs for this new N-outer record;
- N-outer device LOAD/COMPUTE kernels;
- fixed DFG nodes using the new arguments;
- physical address formulas and a dynamic integration simulation.

Those are downstream implementation work. The current result is a verified
software ABI and replay specification, not a claim that the resident M-outer
Bingo graph already executes N-outer.
