# N-outer multi-slot coarse model and Bingo interface proposal

Status: the isolated multi-slot Python model, compact protocol, runtime tables,
and audit replay are implemented and tested.  The production RTL datapath and
Bingo kernels remain a design proposal and have not been implemented.

## 1. Frozen hierarchy

The scheduler produces a sequence of slots.  Each slot contains one ordered
slice list for C0 and one for C1:

```text
SchedulePlan
  SlotPlan[0]
    C0: ExpertSlice[]
    C1: ExpertSlice[]
  SlotPlan[1]
    C0: ExpertSlice[]
    C1: ExpertSlice[]
  ...
```

For each cluster, the static execution nest is:

```text
for local_slot in schedule order:
  for phase in [Gate/Up, Down]:
    for weight_block in phase:
      for expert_slice in local_slot.cluster_list:
        for token_tile in expert_slice:
          compute
```

Therefore slot is above phase.  The two clusters do not share a global slot
barrier: when C0 finishes its side of slot `s`, it may advance to its side of
slot `s+1` while C1 is still completing slot `s`.  This preserves utilization.
The global iDMA/xDMA resource is the only mandatory cross-cluster ordering
point.

A slot with an empty side creates no local execution on that cluster.  PAIR,
SINGLE, and SPLIT are candidate-construction operations; the selected result
is represented completely by the two ordered slice lists.

## 2. Slice and slot semantics

One real slice is:

```text
ExpertSlice = (eid, token_start, ntokens)
```

Slices of the same expert must cover disjoint real-token intervals.  Padding
is lowering metadata and never changes this interval.

One slot is:

```text
SlotPlan = {
  slot_id,
  c0_slices[],
  c1_slices[]
}
```

The sum of resident input, Gate/Up intermediate, and Down output storage for
each side must fit that cluster's slot workspace.  This capacity check is part
of candidate legality, even when the ideal timing mode assigns zero time to
activation gather/store.

The default token lowering is deterministic:

```text
M4 for each complete four-token tile
one padded M4 for a three-token tail
one M2 for a one/two-token tail
```

Consequently shape is not a scheduler output in protocol version 1.  If a
later model searches alternative shape decompositions, an explicit per-slice
shape policy must be added; it must not be inferred from an audit trace.

## 3. Python and RTL common coarse model

### 3.1 Static resources

- two independent VersaCore compute timelines, C0 and C1;
- two non-preemptive global DMA lanes, iDMA and xDMA, each 64 B/cc;
- BOTH occupies both lanes and transfers at 128 B/cc aggregate;
- two ping/pong weight buffers per cluster;
- one logical weight LOAD outstanding per cluster;
- one logical COMPUTE outstanding per cluster.

### 3.2 Virtual work stream

The model internally expands every non-empty local slot into virtual items:

```text
(slot, phase, block, slice)
```

This expansion is required to score resource contention.  It is not an RTL
output stream and it is not a software-call sequence.

For local item `i`:

```text
buffer_slot(i) = i & 1
LOAD(i) waits for COMPUTE(i-2) to release that slot
COMPUTE(i) waits for LOAD(i) and COMPUTE(i-1)
```

The recurrence does not reset DMA-lane or compute timelines between slots.
The next local item after the final Down item of slot `s` is the first
Gate/Up item of the next non-empty local slot.  Its weight may be prefetched
when the buffer and DMA lane are legal.  Its compute waits for that cluster's
previous slot completion and activation-workspace readiness.

### 3.3 Slot boundary events

The model exposes two optional boundary costs per cluster:

```text
slot_input_ready_tick
slot_output_release_tick
```

Ideal scheduler mode sets activation gather/store cost to zero but still
checks workspace capacity and ordering.  Calibrated mode supplies measured or
analytical gather/store durations and places their DMA occupancy in the same
resource model.  Reports must distinguish core-only makespan from calibrated
end-to-end makespan.

### 3.4 DMA decision

There is one logical arbiter owner.  At each event boundary it sees at most
the first buffer-eligible weight request from C0 and C1.  It selects a legal
combination of:

```text
C0 single + C1 single
C0 BOTH
C1 BOTH
one available single request
```

Python and RTL must implement the same integer recurrence and tie-break.  RTL
keeps only finish-time/resource state while evaluating candidates; it does not
materialize an array of LOAD/COMPUTE events.

### 3.5 Model result

The selected candidate result contains:

```text
SchedulePlan slots
makespan_ticks
per-cluster completion ticks
DMA-lane completion ticks
stall/utilization audit
optional virtual event trace for verification only
```

The score is the final resource-legal makespan.  Virtual start/end ticks and
lane predecessors are verification evidence, not execution ABI fields.

## 4. RTL scheduler output boundary

The RTL output must describe the selected semantic plan only:

```text
Schedule header:
  protocol_version
  slot_count
  total_slice_count
  dma_policy_version
  flags

For each slot:
  slot_id
  c0_slice_count
  c1_slice_count

For each slice in cluster order:
  cluster
  eid
  token_start
  ntokens
```

The stream order provides the local ordering.  `token_start` is an index into
that expert's token-reference row, not a physical address.

RTL does not output:

- phase or block records;
- one command per LOAD/COMPUTE;
- absolute timestamps;
- ping/pong indices;
- L1/L3 addresses;
- M4/M2 iteration counts when using the fixed lowering;
- Bingo node identifiers.

The exact 64-bit schedule/slot/slice packing is implemented in `protocol.py`.
It remains isolated from the production scheduler transport until the RTL and
host integration step.

## 5. Host lowering boundary

CVA6 drains the selected RTL plan and constructs compact tables.  It does not
create block-level tasks.

Logical runtime tables:

```text
NOuterScheduleHeader {
  version
  slot_count
  total_slice_count
  flags
  slot_table_addr
  slice_table_addr
  static_context_addr
  runtime_sync_addr
}

NOuterSlotDesc {
  c0_first_slice
  c0_slice_count
  c1_first_slice
  c1_slice_count
  c0_workspace_offset
  c1_workspace_offset
}

NOuterSliceDesc {
  eid
  token_start
  ntokens
  token_ref_start
  input_l1_offset
  intermediate_l1_offset
  output_l1_offset
  m4_iters
  m2_iters
  tail_valid_tokens
}
```

Only `eid/token_start/ntokens` and list placement come from the scheduler.
CVA6 derives token-reference positions, workspace offsets, M4/M2 loop counts,
valid-tail fields, and all physical addresses from the static layout.

## 6. Static Bingo execution model

The production graph should contain schedule-level persistent workers, not a
node per block or per expert:

```text
host_prepare_schedule
  |-- C0_DMA_SLOT_WORKER       (cluster C0 DM core)
  |-- C0_COMPUTE_SLOT_WORKER   (cluster C0 GEMM core)
  |-- C1_DMA_SLOT_WORKER       (cluster C1 DM core)
  `-- C1_COMPUTE_SLOT_WORKER   (cluster C1 GEMM core)
all workers -> host_schedule_join
```

The four device workers start concurrently.  There must be no DFG edge from a
whole DMA worker to its COMPUTE worker, because that would serialize the
complete schedule.  Per-item readiness uses shared generation words in local
TCDM.

DMA worker responsibilities:

```text
for local_slot:
  gather/prepare slot input when required
  for phase/block/slice:
    wait buffer generation
    request global lane grant
    load the selected expert weight block
    publish weight_ready generation
  drain/store slot output when required
```

COMPUTE worker responsibilities:

```text
for local_slot:
  wait slot input ready
  for phase/block/slice:
    wait weight_ready generation
    execute every M4/M2 token tile for the slice
    publish buffer_released generation
  publish local_slot_compute_done
```

The arbiter has one logical writer.  A physical implementation may use a
dedicated coordinator or one designated DM worker plus request/grant words.
The choice requires checking which core can legally program both iDMA and
xDMA endpoints.  Two independent DM workers must not decide BOTH ownership
without a single arbitration owner.

## 7. Public worker arguments

Each persistent worker needs a small fixed argument record:

```text
NOuterWorkerArgs {
  schedule_header_addr
  static_context_addr
  runtime_sync_addr
  cluster_id
  worker_role
}
```

The shared static context supplies:

```text
Gate/Up and Down weight bases
expert and block strides
input/token-reference/output bases and strides
phase block counts
two weight-buffer addresses per cluster
slot workspace bases and capacity
DMA endpoint configuration
```

No worker argument contains an expanded LOAD/COMPUTE command array.
`load_one_weight_block()` and `compute_one_slice_block()` are internal helper
operations inside the persistent workers, not public Bingo APIs.

## 8. Required validation before freezing bit packing

1. One-slot execution must reproduce every existing focused test.
2. Two-slot execution must prove `slot -> phase -> block -> slice` locally.
3. C0 must be able to enter its next slot while C1 remains in the prior slot.
4. Cross-slot first-weight prefetch must not overwrite a live ping/pong slot.
5. Every DMA interval must satisfy exclusive lane ownership.
6. Compact schedule tables must reconstruct the same virtual streams.
7. Dependency-only worker replay must reproduce the coarse-model makespan.
8. A focused production API test must confirm which worker owns global DMA
   arbitration before the runtime protocol is frozen.

The isolated 64-bit fields and logical runtime tables are now frozen for
Python/RTL co-design.  Production C structs remain to be added only when the
actual DMA arbitration owner and address spaces have been verified.
