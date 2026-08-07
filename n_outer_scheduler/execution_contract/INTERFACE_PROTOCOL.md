# N-outer RTL-to-Bingo interface protocol v1

This is the frozen interface for the isolated Python/RTL co-design model.
Production RTL/C/Bingo integration is not implemented yet.

## 1. RTL output stream

Every record is one 64-bit word.  Records appear as:

```text
SCHEDULE
SLOT 0
  C0 SLICE 0..N0-1
  C1 SLICE 0..N1-1
SLOT 1
  ...
```

### SCHEDULE record (`kind=00`)

| Bits | Field |
|---:|---|
| 63:62 | kind = `00` |
| 61:58 | protocol version = 1 |
| 57:50 | slot count |
| 49:40 | total slice count |
| 39:36 | DMA policy version = 1 |
| 35:28 | flags |
| 27:12 | schedule id |
| 11:0 | reserved = 0 |

Flags:

```text
bit 0: no global slot barrier
bit 1: fixed M4/M2 token lowering
```

### SLOT record (`kind=01`)

| Bits | Field |
|---:|---|
| 63:62 | kind = `01` |
| 61:54 | slot ordinal |
| 53:46 | group/slot id |
| 45:38 | C0 slice count |
| 37:30 | C1 slice count |
| 29 | final slot |
| 28:0 | reserved = 0 |

### SLICE record (`kind=10`)

| Bits | Field |
|---:|---|
| 63:62 | kind = `10` |
| 61 | cluster |
| 60:55 | expert id |
| 54:47 | token start within this expert |
| 46:39 | real token count minus one |
| 38:31 | slot ordinal |
| 30:23 | local slice index |
| 22:0 | reserved = 0 |

The redundant slot ordinal and local index allow CVA6 to reject a corrupted or
misordered stream.  The transport never carries phase/block events.

## 2. What the scheduler decides

The scheduler owns only:

```text
slot partition
cluster assignment
slice order in each local slot
expert id
token start
real token count
```

SPLIT is represented by disjoint slices.  Runtime addresses, token-reference
positions, M4/M2 loops, ping/pong indices, and DMA command addresses are not
scheduler outputs.

## 3. CVA6 runtime tables

CVA6 validates the record stream and creates:

```text
RuntimeScheduleHeader
RuntimeSlotDesc[slot_count]
RuntimeSliceDesc[total_slice_count]
```

`RuntimeSlotDesc` identifies the contiguous C0 and C1 ranges in the slice
table and records the total real tokens on each side.

`RuntimeSliceDesc` preserves `eid/token_start/ntokens` and derives:

```text
token_ref_start = eid * max_tokens_per_expert + token_start
input_l1_offset
intermediate_l1_offset
output_l1_offset
m4_iters
m2_iters
valid-tail fields
```

Workspace offsets restart at zero for each cluster-local slot because that
cluster reuses its slot workspace only after completing its prior local slot.
The two clusters have independent workspaces and may therefore occupy
different slot ordinals concurrently.

## 4. Public Bingo worker ABI

Every persistent worker receives:

```c
struct NOuterWorkerArgs {
    uint32_t schedule_header_addr;
    uint32_t static_context_addr;
    uint32_t runtime_sync_addr;
    uint32_t cluster_id;
    uint32_t worker_role;  // DMA_SLOT_WORKER or COMPUTE_SLOT_WORKER
};
```

The static context supplies:

```text
Gate/Up and Down weight bases
expert and block strides
input, token-reference, intermediate, and output bases/strides
phase block counts and block byte counts
two weight ping/pong buffer addresses per cluster
slot workspace base and token capacity per cluster
iDMA/xDMA endpoint configuration
```

The public ABI contains no block id, absolute tick, lane predecessor, or
expanded command pointer.

## 5. Runtime synchronization

The four long-lived workers use generation words, not DFG edges, for internal
pipeline synchronization:

```text
dma_request[C0/C1]       written by the corresponding DMA worker
dma_grant[C0/C1]         written only by the global arbiter owner
weight_ready[C0/C1][2]  written by the corresponding DMA worker
buffer_released[C0/C1][2] written by the corresponding compute worker
local_compute_seq[C0/C1] written by the corresponding compute worker
local_schedule_done[C0/C1] written by the corresponding worker owner
```

The C0 DMA worker is the default logical arbiter owner.  It remains active
until both local DMA streams finish, even if C0 has no remaining loads.  C1
publishes requests and consumes grants.  Each transfer is still issued by the
cluster-local DMA worker that owns its destination.

The arbiter follows the same policy-version-1 integer recurrence used by the
Python/RTL coarse model.  This avoids exporting a block-level grant script.

## 6. Execution reconstruction

Both DMA and COMPUTE workers reconstruct the same local item order directly
from the compact tables:

```text
for slot ordinal in schedule order:
  if this cluster has no slices: continue
  for phase:
    for block:
      for local slice:
        item_index++
```

`buffer_slot = item_index & 1`, and `item_index` never resets between slots.
This is the only legal reconstruction of the coarse-model execution stream.
