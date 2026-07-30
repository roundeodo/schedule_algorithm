# Bingo static runner for block-major N-outer

## Fixed graph

One group uses six fixed nodes:

```text
group_start
  |-- C0_LOAD_WORKER    (DM core)
  |-- C0_COMPUTE_WORKER (VC core)
  |-- C1_LOAD_WORKER    (DM core)
  `-- C1_COMPUTE_WORKER (VC core)
group_join
```

LOAD and COMPUTE must remain separate nodes so DMA and VersaCore execution can
overlap.  No weight block or expert becomes a dynamically created Bingo node.

## Dynamic descriptor

The RTL/host descriptor is a complete ordered list per cluster:

```text
group_last       1 bit
cluster_last     1 bit
cluster          1 bit
expert_id        6 bits
token_start      8 bits
ntokens_minus_1  8 bits
```

The packed record occupies 25 low bits.  Phase sizes, block counts, weight
strides, and ping/pong addresses are static model context.  CVA6 derives M4/M2
iteration counts and valid-tail metadata from each record.

## Worker loop

Both workers derive the same linear item index:

```text
for phase:
  for block:
    for descriptor in cluster_list:
      item_index++
```

`buffer_slot = item_index & 1`.

LOAD worker:

```text
wait COMPUTE(item_index-2) released buffer_slot
obtain deterministic iDMA/xDMA/BOTH grant
load weight block
publish LOAD_DONE(item_index)
```

COMPUTE worker:

```text
wait LOAD_DONE(item_index)
wait COMPUTE_DONE(item_index-1)
run every M4/M2 token tile for this descriptor
publish COMPUTE_DONE(item_index)
```

Completion words are single-writer: only a LOAD worker writes its LOAD_DONE
event and only a COMPUTE worker writes its COMPUTE_DONE event.  Consumers read
with volatile/acquire semantics and producers publish with release/fence
semantics.  Shared multi-writer counters are forbidden.

## DMA plan reproduction

The Python timespan engine applies a deterministic ready-only arbiter.  Host
lowering reruns the same recurrence and emits each LOAD's lane mask and the
predecessor on every occupied lane.  The fixed worker can therefore reproduce
the selected schedule without absolute start timestamps.

An equivalent implementation may place the same deterministic arbiter beside
the two LOAD workers.  It must produce the same grant sequence for identical
descriptors; otherwise Python/RTL/Bingo timing equivalence is not established.

## Runtime versus model time

Runner commands contain durations and dependencies, not absolute start ticks.
The tick trace is audit metadata.  Earliest execution under the emitted
dependencies must reproduce every model LOAD/COMPUTE interval.
