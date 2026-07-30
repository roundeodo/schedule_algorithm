# RTL handoff checklist for protected18-v4

Status: implementation handoff only. This document does not claim that
`Scheduler_hw` has been modified or synthesized.

Python oracle:

```text
/esat/studscratch/r1015673/Thesis/Idea_Model/
  scheduler_rtl_unified_policy.py
policy_id = rtl-unified-t6b2-protected18-v4
```

Current RTL resource baseline supplied by the user:

```text
LUT = 6000
FF  = 1590
```

No synthesis report containing those two values is present in
`Scheduler_hw`; they are treated as the external baseline, not independently
reproduced evidence.

## 1. Non-negotiable architecture

The implementation must remain sequential and resource-reused:

```text
one candidate generator
  -> one four-stage timing evaluator
  -> one DMA legality checker
  -> optional recovery-SINGLE local reducer
  -> one multi-cycle continuation scorer
  -> one global winner reducer
  -> replay one compact winning token
```

Forbidden implementations:

- parallel copies of the timing evaluator for 18 candidates;
- a 30-entry runtime ROM/RAM of wide candidate records;
- storing full child snapshots for every candidate;
- beam search, SIM1, rollout, or future-candidate expansion;
- floating-point arithmetic or trained coefficient memory;
- a second full global-winner record;
- retaining current S4PF policy state while claiming exact v4 lockstep.

## 2. What must remain unchanged

- M-outer four-stage S1/S2/S3/S4 task semantics;
- shape A/B/C timing equations;
- explicit IDMA/XDMA/BOTH resource legality;
- S2PF semantics;
- one committed action per round;
- compact candidate token and winner replay;
- dense one-task FIFO and current 64-bit task-word field positions;
- MMIO blocking-read, atomic-pop, ready/backpressure, and one-write refill
  transaction semantics.

The window payload meaning and refill direction metadata must change as
described below. Preserving the handshake protocol does not mean pretending
that the current one-sided `head6+reserve6` refill implementation already
maintains `top6+bottom2`.

The S4PF descriptor bits in the 64-bit task word remain for ABI compatibility,
but strict v4 emits `S4PF_DESC_OP_NONE`.

## 3. Window/register reuse

The Python policy observes `top6+bottom2`. Do not add another descriptor RAM.

Reuse the current twelve 16-bit descriptor registers as:

```text
head[0:5]      = T0..T5      6 entries
bottom[0:1]    = B0..B1      2 entries
reserve[0:3]                  4 refill entries
```

The descriptor storage remains `12 * 16 = 192` bits, identical to the current
`head6+reserve6` capacity. Use this initial-write layout:

```text
WINDOW0       = T0,T1,T2,T3
WINDOW1       = T4,T5,B0,B1
WINDOW2_START = reserve0,reserve1,reserve2,reserve3
```

`B0` is the coldest remaining expert and `B1` the second coldest. Head/bottom
aliases in tails of eight or fewer experts are one descriptor, not two
experts; valid/selector logic must deduplicate them before enumeration and
removal.

Software keeps two monotone cursors over the full sorted list:

- `head_cursor` supplies the next hot-side reserve descriptors;
- `tail_cursor` supplies a replacement when `B0` or `B1` is consumed.

The wrapper must expose separate `head_refill_count` and
`bottom_refill_count` in the existing refill event word. Use two write
offsets, not a direction bit that reduces the four-descriptor payload:

```text
0x20 REFILL_HEAD_QUAD    append 1..4 head-side reserve entries
0x38 REFILL_BOTTOM_PAIR  install 1..2 next cold-side entries
```

Keep each response as one 64-bit transaction; do not introduce a
descriptor-at-a-time protocol. A compatible event-word assignment is:

```text
bit 0       batch complete
bit 1       head refill request
bits 4:2    head refill count
bit 5       bottom refill request
bits 7:6    bottom refill count
bits 11:8   task FIFO count
```

A round may commit only when both the next `T0..T5` and `B0..B1` views are
complete. Refill latency may add backpressure but must not change the
candidate set.

The commit token must decode into `head_remove_mask[5:0]` and
`bottom_remove_mask[1:0]`. Implement only the fixed mask patterns emitted by
the profile bank; do not add an eight-entry general compactor. These masks
are internal window-control metadata and do not change the task word.

## 4. Candidate token and generator

Use a 5-bit profile index plus a small phase/family field. Do not silently
encode all 15 base and 30 recovery source profiles in one 5-bit global ID.
The maximum physical stream count is 18, so its counter is also 5 bits. IDs
may be sparse if that reduces decode logic.

Implement four fixed enumeration phases:

1. protected base;
2. recovery `T0 SINGLE` profiles;
3. the one recovery `T0 HALF-SPLIT` profile;
4. the one recovery cached `PAIR(T0,T1)` profile.

There are 30 recovery source profiles in the Python oracle, but only the legal
mode/cache/cluster subset materializes. Encode them as `unique case` decode or
factored shape/lane cases. Do not infer a writable ROM.

For a recovery profile, check event-aligned starts in ascending release order
and accept the first legal start. This is priority-first control; it does not
need a per-profile finish comparator or provisional child register.

Measured bounds over 478,477 rounds:

| Stream | Maximum |
|---|---:|
| base physical candidates | 12 |
| all physical candidates | 18 |
| recovery-family winners | 3 |
| global scorer entries | 15 |

Add synthesis-time and simulation assertions for all four bounds.

## 5. Recovery-family reducer

Only recovery `SINGLE` has more than one physical profile. The recovery SPLIT
and PAIR profiles bypass local reduction.

The SINGLE reducer key is:

```text
late_end       : T_W bits
sum_ends       : T_W+1 bits
latest_start   : T_W bits
profile_id     : 5 bits, fixed priority only
valid          : 1 bit
```

For `T_W=16`, this is 55 bits. The profile ID is replayed after the group ends;
no task plan or full cluster snapshot is stored.

The numeric comparator is lexicographic over `late_end`, `sum_ends`, and
`latest_start`. Exact equality retains the earlier profile ID. Attempts to
remove any numeric field caused regressions and are not permitted.

## 6. Continuation scorer

Replace the current top4 aggregate projection with the exact v4 bounded
sequence:

1. calculate the child monotone `f`;
2. place child `T0..T4` sequentially on the earlier assigned load;
3. place histogram buckets four down to one;
4. balance aggregate overflow work;
5. retain compute-capacity and DMA-capacity bounds as separate tie fields;
6. apply the fixed pairwise regime rules.

Reuse the current scorer's two assigned-load registers, unassigned-work
register, adder, and comparator. Extend the FSM; do not unroll five head
placements or four histogram buckets in parallel.

Repurpose two current aggregate registers before adding state:

- replace `total_parallel_work_q` with `parent_f_q`; v4 does not consume the
  former aggregate;
- replace scaled `total_serial_work_q` with unscaled
  `remaining_block_sum_q = sum(ceil(ntok/2))`;
- derive `best_work = 3 * remaining_block_sum_q` by shift plus add.

This makes the compute-capacity block sum directly available and avoids both
a duplicate register and a divide-by-three datapath. Initialize `parent_f_q`
with one bounded-state pass before the first candidate is evaluated.

Required new persistent aggregate state beyond those two reused registers:

| Counter | Width at E_MAX=64 |
|---|---:|
| remaining token sum | 15 |
| odd-token count | 7 |
| one-block count | 7 |
| two-block count | 7 |
| three-block count | 7 |
| four-block count | 7 |

Total additional aggregate state is 50 bits. Reuse `active_count`; do not add
separate remaining-count or best-work registers.

The initial values do not fit in the current window payload. Add one packed
configuration write before `WINDOW2_START`:

```text
0x40 CONFIG_EXT
  bits 14:0   remaining token sum
  bits 21:15  odd-token count
  bits 28:22  one-block count
  bits 35:29  two-block count
  bits 42:36  three-block count
  bits 49:43  four-block count
  bit 50       C2 initial cache eid is still remaining
  bit 51       C3 initial cache eid is still remaining
```

The existing CONFIG serial-work field changes meaning to
`remaining_block_sum`; its parallel-work field is no longer an input policy
aggregate. Extend the local address decode to cover `0x40`; do not change the
blocking read or task-stream transactions.

All counters are initialized once and decremented from the committed token(s).
Do not recompute them by scanning reserve or software memory.

Strict v4 has no future-S1 S4PF, so a cluster cache record is only the initial
full-residency entry until that cluster accepts its first task. Replace each
current `{pf_eid, pf_end, pf_full}` 24-bit cache record with:

```text
cache_eid             : 7 bits (existing tagged encoding)
cache_eid_is_remaining: 1 bit
```

`pf_end=0` and `pf_full=1` are implicit while that record is valid. Clear a
record when its cluster commits any task, or when its cache eid is consumed
on either cluster. S2PF fetches the current task's down weight and does not
create a future cache record.

The two remaining flags are sufficient for the mandatory-DMA lower bound;
the scorer must not scan hidden expert IDs. In tick-domain lane-work units:

```text
cached_slots = number of distinct valid cache_eid values still remaining
mandatory_lane_ticks = 4 * (remaining_count - cached_slots)
                     + 2 * (remaining_count - cached_slots)
```

Both initial entries are full-residency, and equal cache eids count once.
Integrate this lane work over the free IDMA/XDMA intervals using the same
sequential interval datapath used for legality. Do not instantiate a second
wide DMA-capacity engine or carry byte-valued products through the scorer.

The other primary-bound components use only `parent_f`, child endpoints,
child `T0`, and `remaining_block_sum`. The compute-capacity crossing checks
floor/ceil around division by six; implement the constant division
sequentially with scorer scratch registers rather than a parallel divider.

The following scorer blocks are required by causal ablation and must not be
deleted: one-progress, sync-hot, plateau, tail-plateau, slack-fill, hist4,
compute tie field, and DMA tie field.

## 7. Compact global winner state

Do not instantiate two complete best-candidate records.

During base enumeration, update one global winner summary. At the base/recovery
boundary, snapshot only:

```text
base_token       : valid + mode + 4-bit base-local profile ID
base_primary_f   : T_W bits
```

The base phase is implicit in this fallback snapshot. The active global
winner token, unlike the base-only fallback, also carries the bank/family
field and a 5-bit local profile ID.

Then continue the same global fold with at most three recovery-family winners.
The complete current winner summary needs only fields that cannot be decoded
again from the compact token and stable window:

```text
f, lpt, compute_lb, dma_lb, early_end, late_end
```

Selected token load/rank, remaining count, and S2PF count are reconstructed
from the compact token and stable current-round inputs. Do not store them in
the winner record. The same token also reconstructs the two fixed removal
masks used by the wrapper.

At the end, a recovery token may commit only if:

```text
recovery_f + 1 <= base_primary_f
```

Otherwise replay `base_token`. The comparison is in tick units.

## 8. S4PF removal and ABI preservation

The v4 Python oracle never generates S4PF. To claim lockstep, remove or bypass:

- per-cluster pending S4PF task records;
- commit-time S4PF SINGLE/BOTH search states;
- the commit-side bandwidth checker used only by S4PF;
- S4PF residency updates in the scheduling state.

Keep the existing task-word S4PF descriptor byte and drive operation `NONE`.
This avoids a software ABI change while removing policy state that v4 does not
use.

If S4PF is retained as an experimental feature, that design is a different
policy and must receive a separate Python mirror and complete 65/29,928/11,928
validation. It must not be called protected18-v4.

## 9. Conservative FF budget

The following is a source-level implementation budget, not a post-synthesis
result:

| Incremental state | Conservative FF bound |
|---|---:|
| token/window/control width changes | 16 |
| new aggregate counters after register reuse | 50 |
| recovery-SINGLE reducer | 55 |
| expanded current-winner summary over current 46-bit record | 60 |
| compact base fallback snapshot | 24 |
| added scorer/group FSM indices | 12 |
| **gross addition** | **217** |

Strict v4 also permits removal of at least:

| Removed current state | Bits documented by current RTL |
|---|---:|
| per-cluster pending task, valid, emit index | 79 |
| commit-side pointer-only BW checker | 8 |
| persistent S4PF fields in two timelines | at least 6 |
| two cache records, 24 bits to 8 bits each | 32 |
| **identified offset** | **at least 125** |

The resulting conservative net bound is approximately `+92 FF`, before
further synthesis sharing. Against the supplied 1,590-FF baseline, that is
about `+5.8%`.

Do not present this estimate as measured utilization. The handoff acceptance
limits are:

```text
target:       LUT <= 6600, FF <= 1750      (+10%)
hard ceiling: LUT <= 6900, FF <= 1830      (+15%)
```

Exceeding the target requires a module-level utilization explanation.
Exceeding the hard ceiling rejects the implementation unless the user accepts
a measured performance/area trade-off.

## 10. Expected decision latency

Area is prioritized over single-round scheduler latency.

- at most 18 candidates pass through the timing evaluator;
- at most 15 entries pass through the continuation/global scorer;
- base candidates are never evaluated twice;
- only the winning compact token is replayed;
- recovery SPLIT/PAIR bypass the local reducer.

Report cycles per candidate, worst-case cycles per round, and complete-batch
scheduler cycles after RTL implementation. Do not infer them from Python wall
time.

## 11. Required lockstep verification

Verification order:

1. compare mode and visible `T0..T5,B0..B1` after every commit;
2. compare every physical candidate token and first legal start;
3. compare candidate count and assert `physical <= 18`;
4. compare recovery-SINGLE local key and winner;
5. compare `f`, LPT, compute LB, DMA LB, and pairwise decision for each global
   scorer entry;
6. compare the saved base token/primary, one-tick guard, and committed token;
7. replay the RTL-selected action in the explicit-DMA Python checker;
8. compare the complete selected history and makespan on proof65;
9. compare per-case makespan on coverage30k and post-freeze;
10. synthesize with the same FPGA part, constraints, tool version, and seed as
    the 6000-LUT/1590-FF baseline.

Passing simulation proves lockstep behavior. Only the final synthesis report
proves LUT/FF cost.
