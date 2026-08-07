# RTL handoff checklist for the bounded distilled scheduler

Status: RTL, MMIO protocol, Python/RTL lockstep, and the HeMAiA software path
are implemented and verified. The final 40 MHz out-of-context synthesis uses
7117 LUT and 1707 FF with zero LUTRAM/SRL/BRAM/DSP and +14.036 ns WNS. Full-SoC
placement, routing, and timing closure remain pending.

Normative contract: `BOUNDED_DISTILLED_SCHEDULER.md`.

Python policy ID: `bounded-distilled-top5-bottom1-targeted-s4pf`.

## 1. Preserve unchanged behavior

- M-outer S1/S2/S3/S4 timing equations;
- shape A/B/C definitions;
- explicit IDMA/XDMA/BOTH bandwidth legality;
- S2PF residency and timing semantics;
- top5+bottom1 descriptor window and deterministic rank ordering;
- aggregate remaining-work counters and monotone parent bound;
- one committed logical action per round;
- compact winner replay through the shared timing datapath;
- scheduler-slave behavior, task-word encoding, and blocking-read semantics.

Do not add an autonomous DMA requester as part of this policy update.

## 2. Remove superseded control only

Remove logic whose only purpose is the protected base/recovery organization:

- base-bank versus recovery-bank FSM branches;
- `base_best` and recovery/global protected winner records;
- candidate-origin metadata;
- recovery-family-only reducer enables;
- recovery margin adders/comparators and protected arbitration;
- synchronous `SINGLE(T0)` cases;
- SIM1 and standalone or wildcard post-winner S4PF arbitration.

Do not remove the S4PF descriptor protocol, pending-task mechanism, bandwidth
checker, or S4PF timing fields. They are reused by the target-aware lowering.

## 3. Implement the bounded state

Expose only `T0..T4` and `B0` to candidate generation. Deduplicate aliases
when the remaining list is short. Maintain these aggregate counters:

- remaining expert count;
- remaining token sum and odd-token count;
- serial-work or M2-block sum;
- one-, two-, three-, and four-block histogram;
- monotone parent lower bound;
- C2/C3 task, cache, S2PF, and DMA timestamps.

Initialize the counters with the software-provided window metadata and update
them by subtracting committed work. Descriptor refills remain software-driven.

The MMIO window protocol uses `hot9 + cold5`: visible `top5 + bottom1` plus four
local reserves on each side. Software sends at most 14 initial descriptors in
four fixed writes. A refill is requested only with hidden work and a depleted
reserve; each side requests at most four descriptors and one transaction requests
at most six. Software sends top descriptors first and bottom descriptors second
through one or two consecutive writes to the same quad register. RTL latches the
credits until the final write, so no refill ACK or polling register is added.

Use one blocking event for refill, task-FIFO watermark, and batch completion.
Use an eight-entry compact FF FIFO with no LUTRAM or SRL inference. Keep the
existing 64-bit task word and `TASK_STREAM` pop-on-successful-read behavior.

## 4. Hard-wire 28 canonical physical profiles

Use `scheduler_rtl_distilled_profiles.py` as the exact field-level source. Use
combinational `case` decode or equivalent fixed control. Do not infer a
28-entry register array, ROM, RAM, CSR-programmable table, or software-loaded
policy memory.

Static case counts are:

- `ONE_IDLE`: 15;
- `SYNC`: 8;
- `TERMINAL`: 5.

A 5-bit physical-profile index is sufficient. All single-lane S1, S2PF, and S3
bindings are canonical: C2 uses IDMA and C3 uses XDMA. Explicit `BOTH` bindings
in the profile source remain legal physical choices.

Do not delete S2PF `BOTH` merely because the existing compact word represents
only the local single lane. The current constrained ablation reached only
33/65 certificates without those choices. Either extend the physical task
metadata to represent the retained `BOTH` cases or run and approve a new
single-only reference search before changing the Python contract.

For each fixed profile, resolve only visible selectors, calculate residency,
enumerate legal event-aligned starts, and retain the earliest-finish concrete
transition.

## 5. Assign mode-local logical action IDs

No mode evaluates more than six logical actions.

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

A 3-bit logical-action ID and six valid bits are sufficient. `SYNC SINGLE`
must not be generated.

## 6. Lower target-aware S4PF locally

S4PF is a physical realization of a known consumer; it is not a new logical
candidate. For each concrete physical transition:

1. calculate and retain the OFF baseline;
2. disable targeted S4PF when remaining expert count is below nine;
3. visit C2 and then C3;
4. require an active, non-S1-cached consumer on that cluster;
5. require a previous same-cluster task and no existing S4 prefetch;
6. target the concrete consumer EID;
7. start at the previous task's `dma3_end`;
8. reject intervals older than the bounded peer snapshot can certify;
9. try local `SINGLE`, then `BOTH`, then retain `OFF`;
10. require completion no later than the previous task's compute end;
11. run the exact two-lane bandwidth check;
12. rematerialize the consumer as an S1-cache hit;
13. accept the targeted realization only if
    `baseline_max_end - targeted_max_end >= 1 tick`.

Local `SINGLE` means C2/IDMA and C3/XDMA. `BOTH` uses both lanes. Never commit a
wildcard target or a feasibility-only S4PF.

Retain the existing compatible S4PF output representation and pending-target
semantics. The task that owns the S4 compute window must remain pending until
the next same-cluster consumer is known; then write that consumer EID and the
selected NONE/SINGLE/BOTH descriptor into the owning task word.

## 7. Implement the physical local reducer

All physical profiles with the same mode-local logical ID share one local
winner accumulator. First reduce OFF realizations with these fields:

```text
1. max(c2_task_end, c3_task_end)       smaller wins
2. c2_task_end + c3_task_end           smaller wins
3. latest selected task start          smaller wins
4. number of completed S2 prefetches   larger wins
5. number of targeted S4 prefetches    larger wins
6. fixed physical profile index        smaller wins
```

Reduce targeted realizations with the same key. A targeted winner replaces the
OFF winner only after the one-tick gain guard in Section 6. Deduplicate equal
child states after local reduction. No candidate origin bit participates.

## 8. Use a resource-reused sequence

The recommended FSM is:

```text
LOAD_STATE
  -> ENUMERATE_PHYSICAL_PROFILES
  -> UPDATE_OFF_LOCAL_WINNER[logical_id]
  -> EVALUATE_TARGETED_S4PF_FOR_LOCAL_WINNER
  -> APPLY_S4PF_GAIN_GUARD
  -> ENUMERATE_VALID_LOGICAL_WINNERS
  -> UPDATE_GLOBAL_CONTINUATION_WINNER
  -> REPLAY_AND_COMMIT
  -> DONE_OR_NEXT_ROUND
```

Reuse one four-stage timing evaluator and one bandwidth checker. Retain at most
six compact local-winner records. The targeted S4PF trial reuses the local
accumulator; it must not double the global candidate bank or require two wide
child-state arrays.

The 29,928-case run observed at most 13 top-level physical candidates and six
logical candidates. Per-mode physical maxima are `SYNC=13`, `ONE_IDLE=8`, and
`TERMINAL=5`. Keep the Python assertion budget of 18 until an all-reachable
state proof justifies a smaller guard.

## 9. Preserve one continuation comparator

All valid logical winners enter the same scorer and global winner fold.
Preserve:

- monotone lower bound `F`;
- head-5 plus four-bin tail LPT estimate `H`;
- compute lower bound `C`;
- DMA-capacity lower bound `D`;
- selected-load, release-time, S2PF, window-rank, and aggregate-counter fields;
- the five state predicates in `scheduler_rtl_distilled_scoring.py`;
- the fixed logical enumeration order.

The exact behavioral reference is `select_continuation_winner`. Do not add a
second S4PF scorer, base/recovery passes, candidate-origin threshold, learned
coefficient memory, or child-round beam expansion. The scalar-only ablation
reaches only 51/65 certificates.

## 10. Commit path and counters

After global selection:

1. attach any target-aware S4PF descriptor to the pending preceding task;
2. replay the compact consumer token through the transition datapath;
3. assert that the replayed child equals the child used by the scorer;
4. subtract selected or split work from remaining counters;
5. update C2/C3 task, DMA, cache, S2PF, and S4PF snapshots;
6. update the monotone parent bound;
7. request only the descriptor refills required by the bounded window;
8. continue or publish completion.

## 11. Required assertions

Add simulation assertions for:

- profile index in `0..27`;
- no duplicate canonical profile;
- no legal `SYNC SINGLE` output;
- logical ID within the active mode table;
- top-level physical count no greater than 18;
- logical valid count no greater than 6;
- selected global slot is valid;
- exactly one consumer commit per nonterminal round;
- committed action removes positive work;
- explicit two-lane bandwidth legality;
- single S4PF uses C2/IDMA or C3/XDMA;
- S4PF target equals the next same-cluster consumer;
- S4PF interval lies within the preceding compute window;
- targeted realization passes the one-tick gain guard;
- selected child replay equality;
- terminal history makespan equality.

## 12. Python/RTL lockstep gates

Do not claim RTL equivalence until all gates pass:

1. unit tests for every logical ID and invalid-selector condition;
2. fixed-lane and explicit-BOTH profile tests;
3. local-reducer tie tests with reversed profile enumeration;
4. target-aware S4PF SINGLE/BOTH/OFF tests on both clusters;
5. exact per-round lockstep on all 65 certificate cases;
6. directed showcase lockstep at 129 ticks;
7. randomized lockstep with initial-cache and no-cache cases;
8. full 29,928-case aggregate equality to
   `bounded_top5_bottom1_fixed_lane_targeted_s4pf_random_validation.json`;
9. synthesis and timing comparison against the current external
   6000-LUT/1590-FF baseline.

Passing simulation without per-round action, DMA descriptor, and child-state
equality is insufficient.

## 13. Expected structural resource change

Compared with the protected base/recovery structure, remove:

- one protected winner record;
- one recovery winner record;
- origin metadata;
- recovery margin arithmetic;
- the protected arbitration mux;
- duplicated global scoring passes.

Retain or add:

- 28 fixed profile decode cases;
- at most six compact local-winner records;
- one local S4PF trial controller;
- one pending target EID and S4PF mode per cluster task record;
- the existing shared bandwidth checker and timing datapath;
- one global continuation winner record.

Target-aware S4PF adds control cycles if the timing datapath is fully reused,
but it does not increase the number of logical candidates or require a second
scorer. The exact LUT/FF and scheduler-latency cost must be measured after RTL
implementation; this checklist makes no synthesis claim.
