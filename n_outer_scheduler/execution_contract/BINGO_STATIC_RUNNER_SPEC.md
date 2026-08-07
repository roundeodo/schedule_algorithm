# Bingo static runner for multi-slot N-outer

## Fixed graph

The graph is schedule-level and independent of slot, block, and expert count:

```text
host_prepare_schedule
  |-- C0_DMA_SLOT_WORKER
  |-- C0_COMPUTE_SLOT_WORKER
  |-- C1_DMA_SLOT_WORKER
  `-- C1_COMPUTE_SLOT_WORKER
all workers -> host_schedule_join
```

All four device workers start concurrently.  LOAD and COMPUTE remain separate
because they use different cores and must overlap.  A DFG edge from an entire
DMA worker to an entire COMPUTE worker is forbidden.

## Compact dynamic input

The RTL stream uses 64-bit records:

```text
SCHEDULE header
SLOT header
  C0 SLICE records in local order
  C1 SLICE records in local order
SLOT header
  ...
```

The semantic dynamic fields are:

```text
slot/group id
cluster-local slice counts
expert id
token_start
real ntokens
```

The stream contains no phase/block command, timestamp, address, buffer index,
or Bingo node identifier.  Exact field packing is implemented in
`protocol.py`.

## CVA6 lowering

CVA6 builds a schedule header, slot table, and slice table.  From the compact
records and static layout it derives:

```text
token-reference position
L1 input/intermediate/output offsets
M4/M2 iteration counts
valid-tail metadata
physical base/stride context
```

The worker ABI is schedule-level:

```text
NOuterWorkerArgs {
  schedule_header_addr
  static_context_addr
  runtime_sync_addr
  cluster_id
  worker_role
}
```

## Worker loop

Each cluster independently executes:

```text
for local_slot in schedule order:
  for phase in [Gate/Up, Down]:
    for block in phase:
      for slice in local_slot.cluster_list:
        process every M4/M2 tile
```

There is no cross-cluster slot barrier.  Stream indices and ping/pong
generations continue across slot boundaries.  The first weight of the next
local slot may therefore be prefetched before the current slot's final compute
ends when the buffer and DMA grant are legal.

DMA worker:

```text
wait buffer_released(item-2)
request the single global DMA arbiter
load the selected weight block
publish weight_ready(item)
```

COMPUTE worker:

```text
wait weight_ready(item)
wait previous local compute
execute all token tiles for the slice
publish buffer_released(item)
```

Completion generations are single-writer.  Two cluster DM workers never
decide BOTH ownership independently; one logical arbiter owns all iDMA/xDMA
grants.

## Audit trace versus runtime ABI

`lowering.py` expands the compact plan into `RunnerLoadCommand` and
`RunnerComputeCommand` records only for verification.  `replay.py` consumes
their dependencies to prove agreement with the timing recurrence.  These
records are not RTL words, public SW calls, or arrays required by the device
worker ABI.
