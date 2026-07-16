# RTL Scheduler Policy Derivation Contract

Status: the R4+bottom2/LPT policy is a frozen evaluated baseline.  Active P5
development replaces its software-visible tail selectors with an R8/K32
ordered window and trains a new RTL-oriented future-value ranker.  P5 is not
an RTL implementation target until its validation and fresh-blind gates pass.

## Active P5: R8/K32 future-value ranker

### Fixed architecture boundary

- Candidate experts are the ordered top eight remaining entries plus concrete
  named residency/prefetch identities; no bottom selector is maintained.
- Every decision still emits at most 32 direct-v8 candidates and applies the
  exact four-stage child transition before scoring.
- RTL owns the eight-entry window and compaction.  CVA6 owns the sorted L3
  stream and advances only a refill cursor; software does not mirror the RTL
  window or decide whether a refill is a top or bottom entry.
- The old R4+bottom2 report and golden model remain immutable comparison
  evidence.  Their opened blind partition is not reused as an unbiased P5
  test.

### Evidence that opened P5

On all 4,739 proven validation cases, pure R8/K32 with the old LPT scorer had
mean ratio 1.013365, p95 1.040816 and 4,125 exact cases.  It matched R4+bottom2
on 4,565 cases, improved 43 and regressed 131.  Mean failed the predeclared
90%-gain retention gate even though p95 and exact-count passed.

First-divergence analysis found 111 cases where the R4-selected child was also
present under R8 but an added rank4--7 candidate won an indistinguishable LPT
score.  Another 63 selected a bottom child absent from pure R8.  Forced
continuation on three severe missing-bottom states showed one exact R8
continuation and two R8 upper bounds only 33,792 cycles above reference, while
their greedy R8 rollouts lost 1.37M--1.91M cycles.  The dominant open problem
is therefore multi-round scorer error; pure-R8 structural loss remains bounded
evidence rather than being assumed zero.

### Runtime score family

P5 does not scan the complete E64 remaining list.  Its persistent scheduling
summary is limited to top8 plus incrementally maintained remaining count,
total isolated compute blocks, odd-token count and small-expert count.  The
base uses the two absolute cluster release times, total remaining work, top0
work and pathmax:

```text
average      = ceil((load2 + load3 + remaining_work) / 2)
single_chain = min(load2, load3) + top0_work
rtl_base     = max(pathmax, load2, load3, average, single_chain)
score        = 16 * rtl_base + sum(weight[mode,term] * term)
```

Timing uses a 5,632-cycle unit because legal DMA/pathmax boundaries may be
half of the 11,264-cycle compute quantum.  Coefficients are restricted to
`{-16,-8,-4,-2,-1,0,1,2,4,8,16}` so every product is a sign, shift or bypass.
The only coefficient-bank selector is the existing decision mode
`BOTH_IDLE`, `ONE_IDLE` or `LAST_EXPERT`.

Predeclared correction terms cover release/load imbalance, outstanding DMA
tail, pathmax gap, useful and duplicate named residency work, odd/small expert
counts, selected top8 ranks and action-family flags.  Calibration may select
only among the already declared `rtl-base`, `rtl-timing`,
`rtl-timing-cache` and `rtl-full` profiles; it may not invent features.

### Counterfactual data and objective

The formal v1 dataset contains 1,024 discovery states: 512 deterministic
reference-path states and 512 R8/LPT on-policy states, stratified across
E8/E32/E64, mode, remaining-count band, action family and whether the rollout
has positive regret.  Case hash, not individual state, assigns 80% fit and 20%
calibration roles.

For every state, all K32 candidates receive an independent forced-continuation
interval `[Q_lower,Q_upper]`.  A legal bounded R4/R8 completion seeds the upper
bound; anytime unrestricted search tightens the interval.  Training directly
minimizes selected-candidate regret against the best candidate upper bound,
with certified regret (`selected_lower > best_upper`) taking priority over
feasible upper-bound regret.  It does not regress absolute makespan and does
not treat a loose interval midpoint as truth.

### P5 acceptance gates

- exactly one complete K32 group per sampled state; deterministic resume and
  no duplicate entry IDs;
- no candidate continuation undercuts a proven root reference;
- calibration chooses only a predeclared profile and mode coefficient banks;
- full validation mean ratio at most 1.012534, p95 at most 1.042375 and at
  least 4,057 exact cases, retaining at least 90% of the R4 baseline gain over
  current hardware;
- validation must also improve the untrained R8 `rtl-base` and report paired
  results against both R8/LPT and frozen R4+bottom2/LPT;
- after implementation and constants are frozen, generate and solve a fresh
  independent blind-v2 distribution.  The previously opened blind split is
  diagnostic only for P5.

## Frozen v1 deliverable and evidence (historical baseline)

The frozen v1 algorithm is a bounded, round-by-round scheduler:

1. Build at most `K` legal actions from the current remaining experts and the
   two cluster/DMA/cache snapshots.
2. Evaluate the exact four-stage timing transition for every action.
3. Estimate the final makespan of the complete remaining workload with the
   fixed integer two-cluster LPT function.
4. Commit the minimum-score action and repeat until no expert remains.

The deliverable includes:

- a fixed candidate generator with `R <= 8` and `K <= 32`;
- a fixed scorer using integer add/subtract, compare, absolute value and shifts;
- a deterministic Python golden model;
- full discovery and validation rollout reports against the four-stage
  reference;
- a hardware contract covering state, candidate latency and arithmetic cost;
- one final blind-test report generated only after the design is locked.

## Non-goals

- The runtime algorithm does not run beam search or complete future rollouts.
- It does not store a flat table of every concrete action.
- It does not use a neural network or an unrestricted regression model.
- A decision tree, if ever justified, may select a small coefficient bank but
  may not bypass legal candidate generation.

## Data contract

Inputs:

- `scheduler_strategy_coverage_E{8,32,64}.json`: 30,000 distributions.
- `results/final_reference/scheduler_reference_E{8,32,64}.json`: reference
  makespans, proof gaps and complete action histories.

Usage:

- `discovery`: candidate and scorer design.
- `validation`: select among already defined alternatives; no new features or
  templates may be invented from validation failures.
- `blind_test`: opened once after the policy and all constants are frozen.

Proven-optimal cases provide exact path evidence. Cases with a nonzero gap are
handled as bounded evidence and are never relabeled as exact optima.

## Error decomposition

For state `s`, action `a`, and bounded generator `G(s)`:

```text
V(s,a)   = best final makespan after forcing a
V*(s)    = unrestricted reference completion
VG(s)    = min(V(s,a) for a in G(s))
a_hat    = argmin(score(s,a) for a in G(s))
```

The analysis reports two separate losses:

```text
candidate loss = VG(s) - V*(s)
scorer loss    = V(s,a_hat) - VG(s)
```

End-to-end rollout regret remains the final acceptance metric; state losses
are diagnostics and are not assumed to add linearly.

## Candidate design contract

The ordinary expert pool compares `R in {4, 8}` and always adds concrete
cluster-resident or prefetched expert IDs.  Reference census may nominate a
bounded number of derived selectors (for example adjacent tail representatives)
when top-rank coverage misses a materially distinct action.  Derived selectors
must survive continuation-value ablation and count against the same candidate
budget. Candidate budgets compare `K in {16, 24, 32}`.

Concrete actions are generated from templates over:

- action family: `SINGLE`, `PAIR`, `SPLIT`, `PREFETCH`;
- expert rank or resident/prefetch role;
- cluster orientation;
- split-cut timing class;
- S1/S3 shape profile;
- DMA binding pattern;
- S2/S4 prefetch mode;
- event-relative start class.

Template selection uses, in order:

1. exact symmetry/equivalence and strict-dominance removal;
2. reference-path coverage over every discovery state;
3. forced-action continuation bounds and candidate-oracle regret;
4. backward ablation under the `R` and `K` limits;
5. hardware scan and state cost.

Frequency alone may not delete a rare template with material oracle value.

## Frozen future-cost contract

Every candidate first receives the exact one-round four-stage transition. For
the resulting child, initialize two LPT loads with its absolute `c2.task_end`
and `c3.task_end`. Scan remaining experts in descending token rank order. The
isolated duration of an expert with `n` tokens is:

```text
blocks   = (n + 1) >> 1
duration = 3 * 11,264 * blocks
```

Add each duration to the currently smaller load; equal loads choose C2. The
primary score is:

```text
score = max(child.f_score, lpt_load_c2, lpt_load_c3)
```

Candidates are compared lexicographically by:

```text
(score, child.remaining_count,
 max(child.c2.task_end, child.c3.task_end), candidate_index)
```

The duration multiplication is a shift-add by three in units of 11,264 cycles.
No fitted features, coefficient banks, multiplier, divider, beam search or
future child expansion are present in the frozen runtime scorer.

## Acceptance gates

The final policy must satisfy all of the following:

- 100% legal four-stage/DMA transitions on every evaluated case;
- deterministic replay and output;
- `R <= 8`, `K <= 32` for every runtime state;
- no general multiplier or divider in the scorer;
- mean and p95 validation regret both improve over the current hardware mirror;
- candidate-oracle and scorer losses are reported separately;
- no blind-test access before the implementation and constants are frozen.

If a gate fails, only the component identified by the error decomposition is
revised. A scorer failure does not reopen the candidate generator unless the
candidate-oracle result also fails.

## Persistent files

To prevent workspace growth, policy derivation may add only:

- `analyze_scheduler_candidates.py`;
- `derive_scheduler_policy.py`, the single continuation/scorer derivation tool;
- `results/policy_search/candidate_census.json`;
- one resumable future-value dataset;
- one scorer/full-rollout report;
- the final Python golden model and RTL contract.

Temporary probes must use `/tmp` and are not retained.

## Locked progress

### P1 candidate census (complete, 2026-07-15)

Canonical report:
`results/policy_search/candidate_census.json`.

- 14,118 proven-optimal discovery cases replayed without a history, remaining
  set, action-count or final-makespan mismatch.
- 116,443 decisions: 82,572 `SINGLE`, 25,313 `PAIR`, 8,468 `SPLIT`, and 90
  explicit `PREFETCH` actions.
- Top-rank/reference-action equivalence coverage:
  - `R4 + residency`: 95.5188%.
  - `R8 + residency`: 97.8685%.
  - `R4 + bottom2 + residency`: 98.0926%.
  - `R8 + bottom2 + residency`: 99.2614%.
  - `R8 + bottom4 + residency`: 99.5629%.
- High-rank pair actions are often adjacent rank/tail pairs.  Therefore P2
  must evaluate a bounded tail/adjacent selector; top-R-only generation is not
  accepted without candidate-oracle evidence.
- Census coverage is not candidate-oracle quality.  No `R`, tail selector or
  template has been selected yet.

The four-stage search now accepts an already replayed `initial_state` in
`run_anytime()`.  Prefix preservation, continuation completion, final history
validation, and the root-incumbent mutual-exclusion guard have been checked on
a proven reference case.  This is the fixed entry point for P2 forced-action
continuation bounds.

### P2 structural constraints (complete)

- Full legal-action enumeration is not a deployable candidate generator.  At
  representative active-8 roots it creates 17,008 to 32,537 actions, dominated
  by SPLIT cut/shape/DMA/prefetch products.
- The three task profiles `A/B`, `B/B`, and `C/C` cover 115,842 of 116,443
  reference decisions (99.4839%).  They form the initial micro-profile bank;
  the remaining 601 decisions are retained as an ablation set, not silently
  discarded.
- A half cut covers 55.3968% of reference SPLIT decisions.  Recomputing the
  locally minimum completion cut under the selected cache/shape/binding fields
  covers 70.8550%.  Therefore neither half-only nor a single local equal-finish
  rule is accepted as the SPLIT generator.
- `half +/- 1` plus front/tail cuts at 1, 2, 4 and 8 covers 89.8914% of
  reference SPLIT decisions.  These are an initial cut-rule bank for oracle
  ablation; their frequency does not yet authorize all of them in RTL.

The next P2 implementation is a two-level bounded generator: macro slots choose
expert role/family/orientation/cut rule, while each concrete micro action uses
one of the three initial task profiles and a bounded prefetch mode.  Every
micro action counts against `K`; local enumeration is not hidden from the
hardware budget.

### P2 bounded-generator evidence (complete, 2026-07-15)

The current provisional generator uses `R8 + bottom2 + concrete residency` and
`K32`.  Its two-level slots are:

- mode-conditioned family quotas;
- macro round-robin over expert role, orientation and split rule;
- token/cache-conditioned profile priority: cache-ready or one token selects
  `C/C`, two through seven tokens selects `B/B`, and eight or more selects
  `A/B` first;
- S2PF mode as a distinct micro choice;
- split cuts at half +/- 1, four one-future-expert release targets, local
  equal-finish, and front/tail 1, 2, 4 and 8.

The release-target cut is not a stored token ratio.  For future expert rank
`r`, it chooses the split whose cluster-release imbalance is closest to the
isolated work bound of future expert `r`.  This recovered a proven 78/118 split
where a local equal-finish cut was 98/98: the earlier release permits useful
work on the freed cluster, so local balance was not the correct objective.

Forced-action continuation on 16 deterministic proven discovery states with
`remaining <= 6` reached zero candidate-oracle loss on all 16 states.  This
includes E8, E32 and E64; `BOTH_IDLE` PAIR/SPLIT/SINGLE, `ONE_IDLE` SINGLE,
`LAST_EXPERT` SINGLE, and an explicit PREFETCH state.  The zero-loss candidate
was found after evaluating one to five unique candidate children per state.

The offline local-completion physical allocator was then replaced by a fixed,
RTL-oriented rule.  Noncached `A/B` profiles prefer C2 on iDMA and C3 on xDMA;
`C/C` prefers both lanes.  S2PF and explicit prefetch use the same cluster lane
preference.  The allocator chooses the earliest legal foreground start, then
the fixed lane priority, then the earliest background start.  It does not read
a child lower bound, completion estimate, or continuation value.

The fixed allocator passed three deterministic `R8/K32` gates:

- 16 late states (`remaining <= 6`): 16/16 zero candidate-oracle loss;
- 16 ordinary early/mid states: 16/16 zero candidate-oracle loss;
- 16 states selected specifically because the reference action misses
  `R4+bottom2` but hits `R8+bottom2`: 16/16 zero candidate-oracle loss.

Thus fixed-lane `R8/K32` is 48/48 on the current continuation sample.  The
fixed allocator also increased direct reference-child hits versus the offline
local allocator on the late and ordinary early/mid gates.  The local allocator
branch has been removed from the derivation tool.

`R4/K32` also reached zero candidate-oracle loss on all 16 targeted R4-miss
states even though none of their reference macros were representable under
R4.  This proves that reference-history rank coverage alone does not establish
R8 necessity.  R4 and R8 remain rollout/area alternatives.  A provisional K16
generator reached zero loss on the 16 late states, but required up to 10 of 16
candidates versus the smaller search pressure seen with K32; K remains open.

The deployable generator is now instantiated directly.  It never calls the
full `gen_stage_actions()` or `gen_prefetch_actions()` path.  Its runtime work
is bounded by the selected expert pool, family quotas, `K`, three base profile
classes, cache-conditional profile forms, four S2PF patterns, eight fixed
SINGLE lane modes and two PAIR/SPLIT lane modes.  Every emitted action is
checked for token conservation, residency flags, reserved-prefetch ownership,
event-relative start time and two-lane DMA feasibility.

Two corrections found by the larger audit are part of the locked generator:

- a one-stage cache hit canonicalizes only that stage, so the bank includes
  conditional `C/B`, `A/C` and `B/C` forms in addition to `A/B`, `B/B`, `C/C`;
- `ONE_IDLE` covers balanced SPLITs from several expert ranks and prioritizes
  a two-lane explicit prefetch when the peer is idle.

`EQUAL_FINISH` and `RELEASE_R0..R3` cuts do not scan every token.  Task end
times are monotone in the left split size, so each target uses binary crossing
search plus a fixed +/-4 neighborhood.  On 308 retained SPLIT actions this
matched exhaustive cut selection exactly.

The direct generator passed the original 48-state continuation suite at zero
candidate-oracle loss.  A separate deterministic 96-state discovery suite was
then stratified across E8/E32/E64, control mode, remaining-count band and
reference family.  Results were:

- `R8 + bottom2`, `K32`: 96/96 zero loss;
- `R4 + bottom2`, `K32`: 96/96 zero loss, with at most four forced
  continuations evaluated in any state;
- `R8 + bottom2`, `K24`: 90/96 zero loss; the six losses were 11,264 to
  33,792 cycles;
- `K16` was rejected after already missing states also missed by `K24`.

The locked P2 point is therefore `R4 + bottom2 + concrete residency`, `K32`.
It has the same oracle result as R8 on the controlled suite while requiring
four fewer leading-rank selectors.  `K24` and `K16` are not carried into P3;
scorer failures will not be repaired by silently reopening this choice.

### P3 scorer and full rollout (complete, 2026-07-16)

Fitted residual scorers were rejected by end-to-end validation. The selected
scorer is the fixed LPT estimate followed by the `rem-snap` tie-break above.
It is implemented once in `scheduler_policy_golden.py`; the derivation rollout
command delegates the frozen configuration to that implementation.

On every proven-optimal discovery case (14,118 cases):

- exact reference makespan: 12,525 cases (88.72%);
- ratio mean: 1.010770;
- ratio p95: 1.032680;
- ratio max: 1.931373;
- no illegal transition, candidate-budget overflow or reference undercut;
- versus the current hardware mirror: 5,147 wins, 8,341 ties and 630 losses,
  with 269,862,912 fewer aggregate cycles.

On every proven-optimal validation case (4,739 cases):

- exact reference makespan: 4,207 cases (88.77%);
- ratio mean: 1.011394;
- ratio p95: 1.034483;
- ratio max: 1.928571;
- versus the current hardware mirror: 1,761 wins, 2,774 ties and 204 losses,
  with 95,496,192 fewer aggregate cycles.

The current hardware mirror validation ratios are mean 1.022788 and p95
1.113402. The acceptance gate requiring improvement in both metrics therefore
passes. Rare maximum-regret cases remain concentrated in explicit-prefetch
tail decisions; a tested one-step special case worsened mean and p95 and was
not retained.

### P4 one-time blind test (complete, 2026-07-16)

Before opening the blind partition, SHA-256 values were recorded for the golden
model, direct generator, four-stage timing model, policy specification and RTL
contract. All five matched immediately after the blind run. No algorithm,
constant, candidate order or acceptance gate changed in response to blind data.

All 4,732 proven-optimal blind cases completed successfully:

- exact reference makespan: 4,200 cases (88.76%);
- ratio mean: 1.010404;
- ratio p95: 1.032258;
- ratio max: 1.928571;
- regret mean: 10,223.77 cycles;
- regret p95: 45,056 cycles;
- maximum candidate count: 32;
- zero illegal reference undercuts.

The current hardware mirror on the identical case set has ratio mean 1.023345
and p95 1.117647. Head-to-head, the frozen policy has 1,732 wins, 2,797 ties
and 203 losses, saving 97,320,960 aggregate cycles. It improves aggregate
cycles separately for E8, E32 and E64.

Blind mean and p95 are within the discovery/validation range and slightly
better than validation, so the improvement generalizes without post-blind
tuning. The maximum-ratio tail is still real: 23 blind cases exceed 1.5x and
the worst case is a small 157,696-cycle reference schedule that becomes
304,128 cycles. This limitation is recorded rather than repaired after opening
blind data. The contractual mean, p95, legality, determinism and K32 gates all
pass, so the policy is accepted for RTL implementation.
