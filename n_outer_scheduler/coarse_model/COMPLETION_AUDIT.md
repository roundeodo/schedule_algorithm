# N-outer software-model completion audit

This audit covers the software-model objective only. `PROVED` means verified by
the isolated Python model and its tests. It does not mean that production RTL or
the physical Bingo C ABI has been implemented.

## 1. Isolation and scope

Status: **PROVED**

- All model source, tests, documentation, and launchers are under
  `Idea_Model/n_outer_scheduler/coarse_model/`.
- Evaluation artifacts use independent filenames. The production M-outer,
  HeMAiA, Bingo, and RTL trees were read only.
- The current 344-byte M-outer S1/S2/S3/S4 record is not modified or claimed to
  implement N-outer.

## 2. N-outer execution semantics

Status: **PROVED**

- Stream order is expert slice -> Gate/Up blocks -> Down blocks -> next slice.
- A phase shape is selected once and reused across its fixed blocks.
- SPLIT uses disjoint real token ranges; padding never creates duplicate work.
- First-block DMA traffic is charged.
- SINGLE, PAIR, SPLIT, phase order, next-expert overlap, and token coverage are
  covered by focused tests.

## 3. Double-buffer successor rules

Status: **PROVED**

- Before a phase tail, the only legal target is the same expert/phase's next
  block.
- Final Gate/Up may target the same expert's Down block zero.
- Final Down may target the next scheduled slice's Gate/Up block zero.
- Load `i` waits for compute `i-2` before reusing ping/pong.
- Blocks remain fixed recurrence counters, not scheduler candidates.

## 4. Shape and DMA modes

Status: **PROVED**

- M8, M4, and M2 are N-outer phase shapes.
- iDMA and xDMA are independent 64 B/cc resources; BOTH occupies both.
- `L_single <= C` suppresses locally dominated BOTH.
- The deployable `rtl_symmetric2` bank contains fixed lanes and, when legal for
  every task/phase, all-BOTH. It never generates one-sided or phase-mixed BOTH.
- K4/K8 remain analysis banks and are not required by the frozen main policy.

## 5. Macro timespan, resources, and stall

Status: **PROVED for the specified coarse model**

- State contains two cluster ends, two boundary-prefetch releases, and two DMA
  lane ends.
- The fixed-trip recurrence uses counters, max/add/compare, ready-only issue,
  ping/pong release, and non-overlapping lane reservations.
- On the final 65 cases, macro versus dependency-only task replay differs in 2
  cases; the maximum absolute error is 14,080 cc.
- All reported performance uses dependency-only task replay. Macro timing is a
  candidate scorer, not execution ground truth.

## 6. Candidate generation and scorer

Status: **PROVED and frozen**

- The adapter imports only SINGLE/PAIR/SPLIT cluster, expert, and real-token
  slice semantics.
- History validation proves every positive-count expert is covered exactly
  once.
- The frozen main path is sorted LPT assignment, paired cluster heads, no
  SPLIT, `rtl_symmetric2`, deterministic `binding_chain`, beam 1, and local
  fixed-first scoring.
- The score is lexicographic
  `(max_all_resources, max_cluster_end, mode_id, stall, signature)`, where mode
  0 is the fixed-lane fallback.
- The main path is better than fixed-lane LPT in 26 cases, equal in 39, and
  worse in 0.
- Of 1,167 selected actions, 1,136 use fixed lanes and 31 use all-BOTH (5 PAIR
  and 26 SINGLE actions). The second mode is sparse but useful.
- Local BOTH-first is rejected: it regresses 31 cases versus fixed-lane and
  raises the four-stage ratio mean from 1.213359 to 1.229556.
- Projected fixed-first is rejected because it adds future work state while
  worsening mean/median to 1.213652/1.210417.
- Projected BOTH-first improves maximum ratio from 1.282895 to 1.280702, but
  worsens mean/median to 1.213488/1.210417 and requires future accumulators.
  The small worst-case gain does not justify its RTL cost.
- Top-1 SPLIT is an offline full-history ablation only; it is not described as
  deployable.

## 7. Service-order complexity

Status: **PROVED and frozen**

- The analysis best-of-18 bank is a calibration policy, not an RTL candidate
  multiplier.
- Across 65 cases, one deterministic `binding_chain` is equal after executable
  replay in 63 cases, better in 2, and worse in 0 relative to the best-of-18
  macro-selection policy.
- The frozen RTL-oriented path evaluates one service order per DMA mode.

## 8. Independent block golden

Status: **PROVED**

- Golden expansion is used only after selecting a macro history.
- It independently checks load readiness, compute dependencies, lane
  ownership, ping/pong reuse, initial fill, and steady stall.
- The symmetric-bank history calibration reports 27 prefix rounds where the
  executable prefix oracle prefers BOTH on an equal macro score. Full-history
  ablation proves that changing all such ties to BOTH is harmful; this metric
  is therefore not treated as continuation-aware policy regret.

## 9. Bingo lowering and replay

Status: **PROVED as a software ABI; physical integration is downstream**

- A macro history lowers deterministically into cluster-local macro slots and
  fixed LOAD/COMPUTE task arguments.
- Dependencies encode prior compute, ping/pong release, and both DMA lane
  chains; absolute model timestamps are not ABI fields.
- Dependency-only replay equals independent block-golden macro-order replay.
- Tests reject token ranges that diverge from the macro slot.
- Existing M-outer device kernels and structs are not claimed compatible.

## 10. Same-input four-stage comparison

Status: **PROVED for the 65-case catalog**

- `_work_signature` checks identical atomic compute work and weight bytes for
  every compared distribution.
- The frozen N-outer main policy divided by the certified four-stage result has
  mean 1.213359, median 1.208333, minimum 1.019006, and maximum 1.282895.
- N-outer is slower in all 65 cases. This is an honest measured dataflow result,
  not evidence that either policy is globally optimal.
- The offline best among fixed-lane, symmetric2, and top-1 SPLIT has mean
  1.209764, median 1.206019, and maximum 1.274225.

## 11. Tests and authoritative artifacts

Status: **PROVED**

- 42 isolated coarse-model tests pass.
- All 14 pre-existing N-outer regression tests pass unchanged.
- The authoritative full result is
  `Idea_Model/results/policy_search/n_outer_coarse_final_policy_65_symmetric2_fixed_first_final.json`.
- Its embedded source hashes match the current model, every task ABI validates,
  and the analyzer reports zero policy-selection regret.
- The authoritative summary is
  `Idea_Model/results/policy_search/n_outer_coarse_final_policy_65_symmetric2_fixed_first_final_summary.json`.

## Completion boundary

The no-RTL software-model goal is complete. Remaining work is a new downstream
implementation task: define physical N-outer C records and device kernels,
instantiate the fixed task graph, implement the two-mode recurrence in RTL,
and compare RTL outputs against the frozen Python records.
