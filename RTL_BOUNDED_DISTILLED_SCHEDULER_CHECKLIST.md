# RTL handoff checklist for the bounded distilled scheduler

Status: implementation checklist only.  The Python mirror is verified; RTL
modification, RTL/Python lockstep, synthesis, and timing closure are pending.

Normative contract: `BOUNDED_DISTILLED_SCHEDULER.md`.

Python policy ID: `bounded-distilled-top5-bottom1`.

## 1. Preserve unchanged behavior

- M-outer S1/S2/S3/S4 timing equations;
- shape A/B/C definitions;
- explicit IDMA/XDMA/BOTH legality;
- S2PF residency and timing semantics;
- top5+bottom1 descriptor window and deterministic rank ordering;
- aggregate remaining-work counters and monotone parent bound;
- one committed action per round;
- compact winner replay through the shared timing datapath;
- scheduler-slave behavior and existing reader/MMIO/FIFO protocol.

Do not change task-word transport or add an autonomous DMA requester as part of
this scheduler-policy update.

## 2. Delete v4-only control

Remove all logic whose only purpose is the protected18 organization:

- base-bank versus recovery-bank FSM branches;
- `base_best` and recovery/global protected winner records;
- base/recovery candidate-origin metadata;
- recovery-family-only reducer enables;
- `RECOVERY_MARGIN_CC` and its one-tick adder/comparison;
- protected arbitration and fallback mux;
- synchronous `SINGLE(T0)` profile cases;
- archived SIM1 or standalone S4PF paths, if still present.

The final RTL must not preserve these signals under new names.

## 3. Hard-wire the 32 physical profile cases

Use `scheduler_rtl_distilled_profiles.py` as the exact field-level source.
Implement the constants with combinational `case` decode or equivalent control
logic.  Do not infer a 32-entry register array, ROM, RAM, CSR-programmable
table, or software-loaded policy memory.

Static case counts by mode are:

- `ONE_IDLE`: 19;
- `SYNC`: 8;
- `TERMINAL`: 5.

A 5-bit profile index is sufficient.  Mode legality must suppress profiles
from the other two modes before timing evaluation.

For each profile, generate only selectors visible in `T0..T4,B0`, resolve
cache residency, and retain the earliest-finish legal event-aligned start.

## 4. Assign logical action IDs

Logical IDs are mode-local; no mode evaluates more than six:

### `SYNC`

| ID | Action |
|---:|---|
| 0 | `PAIR(B0,T0)` |
| 1 | `PAIR(T0,T1)` |
| 2 | `PAIR(T0,T4)` |
| 3 | `PAIR(T1,T2)` |
| 4 | `PAIR(T2,T3)` |
| 5 | `SPLIT(T0,HALF)` |

### `ONE_IDLE`

| ID | Action |
|---:|---|
| 0 | `SINGLE(B0)` |
| 1 | `SINGLE(T0)` |
| 2 | `SINGLE(T3)` |

### `TERMINAL`

| ID | Action |
|---:|---|
| 0 | `SINGLE(T0)` |
| 1 | `SPLIT(T0,BALANCED)` |

A 3-bit logical-action ID and six valid bits are sufficient.  `SYNC SINGLE`
must not be generated.

## 5. Implement the physical local reducer

All physical profiles with the same mode-local logical ID share one local
winner accumulator.  Compare the following fields in order:

```text
1. max(c2_task_end, c3_task_end)       smaller wins
2. c2_task_end + c3_task_end           smaller wins
3. latest selected task start          smaller wins
4. number of completed S2 prefetches   larger wins
5. fixed physical profile index        smaller wins
```

The S2PF count requires two bits for values 0, 1, or 2.  Profile index is the
last exact-tie field; it must not override a useful S2PF.

Deduplicate equal child states after local reduction.  No candidate-bank
origin bit is part of this key.

## 6. Resource-reused sequencing

The preferred FSM is:

```text
LOAD_STATE
  -> ENUMERATE_PHYSICAL_PROFILES
  -> UPDATE_MODE_LOCAL_WINNER[logical_id]
  -> ENUMERATE_VALID_LOGICAL_WINNERS
  -> UPDATE_GLOBAL_CONTINUATION_WINNER
  -> REPLAY_AND_COMMIT
  -> DONE_OR_NEXT_ROUND
```

Reuse one four-stage timing evaluator and one DMA legality checker.  Retain at
most six compact local-winner records; do not retain one full child snapshot
per physical profile.  Replay each compact local winner through the shared
datapath when its continuation fields are evaluated.

The complete 30K run observed a maximum of 14 materialized physical candidates
and six logical candidates.  Keep the Python contract's conservative physical
budget assertion of 18 until an all-reachable-state proof justifies reducing
the hardware guard.

## 7. Preserve one continuation comparator

All valid logical winners enter the same scorer and the same global winner
fold.  Preserve the existing bounded arithmetic:

- monotone lower bound `F`;
- head-5 plus four-bin tail histogram LPT estimate `H`;
- compute lower bound `C`;
- DMA-capacity lower bound `D`;
- selected-load, cluster-release, S2PF, window-rank, and aggregate-counter
  fields required by the frozen state predicates.

The exact behavioral reference is `select_continuation_winner` in
`scheduler_rtl_distilled_scoring.py`.  Logical candidates are folded in the
fixed order emitted by `scheduler_rtl_distilled_policy.py`; RTL must preserve
that order unless order invariance is separately proven.

Do not implement:

- separate base and recovery scoring passes;
- a candidate-origin-dependent threshold;
- floating-point or learned coefficient storage;
- a scalar-only fallback scorer;
- future child-action expansion or beam search.

The structural ablation shows that scalar-only scoring loses 14 of 65 exact
certificates.

## 8. Commit path and counter update

After the global winner is selected:

1. replay its compact token through the existing transition datapath;
2. assert that the replayed child fields equal the fields used by the scorer;
3. remove the selected expert load or split portion;
4. update remaining count, token sum, odd count, block histogram, and work sum;
5. update C2/C3 task, DMA, cache, and S2PF snapshots;
6. update the monotone parent bound;
7. request only the number of descriptor refills required by the bounded
   window protocol;
8. start the next round or publish completion.

## 9. Required assertions

Add simulation assertions for:

- profile index in `0..31`;
- no duplicate static profile specification;
- no legal `SYNC SINGLE` output;
- logical ID within the active mode's table;
- physical candidate count no greater than 18;
- logical valid count no greater than 6;
- local-winner logical ID matches the candidate logical ID;
- selected global slot is valid;
- exactly one commit per nonterminal round;
- committed action removes positive remaining work;
- explicit DMA bandwidth legality;
- selected child replay equality;
- terminal history makespan equality.

## 10. Python/RTL lockstep gates

Do not claim RTL equivalence until all gates pass:

1. unit tests for every logical ID and invalid-selector condition;
2. physical local-reducer tie tests, including reversed profile enumeration;
3. cache-hit and S2PF profile tests on both clusters;
4. exact per-round lockstep on all 65 certificate cases;
5. directed showcase lockstep at 129 tick;
6. randomized lockstep with initial-cache and no-cache cases;
7. full 29,928-case aggregate equality to
   `bounded_top5_bottom1_random_validation.json`;
8. synthesis and timing report against the current 6000-LUT/1590-FF external
   baseline.

Passing simulation without per-round action/score equality is insufficient.

## 11. Expected structural resource change

Relative to v4, the final organization removes:

- the second protected winner record;
- recovery margin arithmetic;
- candidate provenance and arbitration muxing;
- up-to-twelve-entry global scorer input stream.

It adds or makes explicit:

- up to six local-winner valid bits and compact records;
- one two-bit S2PF-count comparison field;
- one 5-bit fixed profile tie-break field.

The global scorer fan-in is bounded at six.  This is a structural expectation,
not a post-synthesis resource claim.
