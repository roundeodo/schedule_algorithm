# RTL Scheduler Policy Re-derivation Contract

> Archived baseline. This document preserves HW-v2 and earlier derivation
> evidence. Do not implement it as the current target. The controlling policy
> is `scheduler_rtl_unified_policy.py`, specified by
> `OLMOE_BOUNDED_SCHEDULER_IMPLEMENTATION.md`.

## Controlling HW-v2 decision (2026-07-20)

This section supersedes the older R4+bottom2/K32 and staged derivation text
retained below as experiment history.  The deployable algorithmic target is
now `scheduler_hw_fixed_policy.py::hw_v2_schedule`; RTL implementation has not
yet started.

The runtime boundary is one scheduling round per invocation.  HW-v2 never
runs beam search, generates a child action, or unfolds the remaining batch.
It keeps the deployed fixed candidate bank and adds only one alternative shape
profile at each existing `ONE_IDLE` release point:

- `ntok >= 7`: `A/B`;
- `3 <= ntok < 7`: `B/B`;
- `ntok <= 2`: `C/C`.

Thus `BOTH_IDLE` remains bounded by five candidates, `ONE_IDLE` increases from
at most three to at most six, and the final one-expert round retains its fixed
candidate IDs.  No standalone prefetch, arbitrary-rank candidate, resident
expert candidate, or variable candidate budget is added.  Resident candidates
were rejected because their additional full-30K gain over the shape-only
policy was only 0.032 percentage points while requiring software-visible
maintenance of experts outside the ordered head window.

The selected continuation score is

```text
min(aggregate_greedy, LPT_top4_then_balance_aggregate_tail)
```

For LPT, the first four remaining experts are placed in descending order onto
the currently earlier cluster using `best_task(ntok)`.  Work after those four
experts is represented only by a scalar total and balanced between the two
loads.  `remaining <= 2` uses the aggregate greedy expression.  There is no
SIM1 continuation expansion.  The final expert is still executed through the
ordinary fixed-candidate pipeline when it becomes the current round; this is
not a nested lookahead.

An RTL implementation needs a six-entry ordered head window so every possible
removal of up to two among the leading four still exposes the next four items
to the scorer, plus scalar `total_task` and existing aggregate state.  A second
six-entry refill buffer is permitted.  HW compacts its own window using the
winning remove mask and requests only the consumed count from software;
software streams the next unseen sorted descriptors and does not mirror the
window identities.  No bottom-side cursor is required.

The frozen 29,928-case comparison is
`results/policy_search/scheduler_hw_v2_30k_comparison.json`:

- aggregate makespan versus deployed HW: -1.014485%;
- wins/ties/losses versus deployed HW: 9,206 / 19,170 / 1,552;
- on 23,589 proven-optimal cases, deployed-HW gap: 1.447406%;
- on the same proven cases, HW-v2 gap: 0.690060%;
- exact proven cases: 17,500 for HW-v2 versus 13,575 for deployed HW.

The failure attribution uses a 132-case stratified candidate-oracle audit.  On
the 100 cases for which no beam state was pruned, the residual over the proven
four-stage optimum is split exactly into 32.18% candidate-space loss and
67.82% scorer/control loss.  Therefore the remaining limitation is primarily
the score, not candidate coverage, but the tested shift/add release-gap,
S4-prefetch reward, short-tail guard, and old-SIM1 variants all degraded the
full closed-loop objective.  They are rejected rather than added after the
fact.

A subsequent fixed-candidate scorer audit explicitly excluded one-round
lookahead and tested two additional families on all 29,928 cases.  The evidence
is `results/policy_search/scheduler_hw_scorer_non_lpt_full.json`.

- A maximum of release-aware workload, largest-expert critical-chain and
  mandatory-DMA-capacity lower bounds regressed aggregate makespan by 2.32% to
  2.41% versus deployed HW.  The relaxed estimates did not exceed the reference
  makespan at any of the 23,589 proven-optimal initial states, but they are too
  insensitive to candidate-specific ordering to serve as the candidate score.
- A cache-aware list estimate for the first four remaining experts, combined
  with `aggregate_greedy` by `min`, improved aggregate makespan by 0.0320
  percentage points over the selected LPT scorer.  Adding committed-DMA
  conflict checks raised that improvement to 0.0356 points.  The latter reduced
  the proven-reference gap from 0.690060% to 0.644945% and increased exact
  proven cases from 17,500 to 17,964.
- The cache-aware estimate requires up to 72 shape/lane finish evaluations per
  candidate for four visible experts; the DMA-aware form additionally scans
  committed transfer intervals.  This is substantially more control and
  arithmetic than four LPT placements for only a 0.0356-point aggregate gain.
  Therefore both are recorded as diagnostic upper-complexity alternatives and
  are not selected for RTL.  The controlling score remains
  `min(aggregate_greedy, LPT_top4_then_balance_aggregate_tail)`.

Historical status snapshot (superseded by the controlling decision above):
the bounded generator, physical
scorer, constants and numeric tie-break were frozen under the confirmed
slave-only interface before blind-v2 generation.  Canonical validation and
the fresh blind-v2 gate both passed without any post-blind policy change.  The
next task is the bounded RTL microarchitecture and C-golden equivalence, not
another scheduler search.  Historical P5 evidence is retained below, but its
full-descriptor/full-LPT implementation is not deployable under the current
storage and software-interface constraints.

## Historical bounded-window derivation plan

This section records the earlier plan and is not controlling after the
2026-07-20 HW-v2 decision above.

### Fixed runtime boundary

- One invocation selects exactly one scheduling action.  RTL never expands a
  complete batch and never runs offline beam search.
- The scheduler is a slave.  It cannot fetch descriptors or initiate a refill.
- RTL may hold an active `top4 + bottom2` window and one equally partitioned
  refill buffer: at most eight head-side and four tail-side descriptors, or
  twelve descriptors total.  It may also maintain scalar aggregates.
- Software supplies the initially sorted descriptor stream and refills the two
  monotone sides.  Hardware reports separate head-side and tail-side consume
  counts, so software advances two indices; it does not mirror every internal
  window identity after each decision.
- The score may inspect the exact current C2/C3/DMA/cache/prefetch snapshots,
  the visible descriptors, `remaining_count`, and an incrementally maintained
  `total_remaining_work`.  It may not inspect an unseen middle descriptor
  individually.
- Candidate actions still use the exact four-stage child transition.  Any
  approximate reasoning begins only after that exact child has been formed.

### Predeclared candidate alternatives

The deployment comparison is between complete `(generator, scorer)` pairs:

1. `R4`: leading four remaining experts plus concrete named
   residency/prefetch identities;
2. `R4+B2`: the same pool plus the trailing two remaining experts.

Both use the existing bounded physical-action templates and `K <= 32`.
Candidate coverage is reported separately from scorer loss.  Prior reference
coverage motivates testing `B2`, but does not predetermine that it is worth the
extra two-sided refill interface.

### Predeclared scorer alternatives

All score arithmetic is integer half-quantum arithmetic (`Hq = 5,632` cycles).
For every candidate, first apply the exact four-stage transition and then
evaluate one of these fixed alternatives:

- `S0 aggregate`: reproduce the current hardware-oriented aggregate estimate
  from the two child release times, total remaining work, and largest visible
  work item.  This is the control baseline.
- `S1 ordered-middle`: place the visible head descriptors by two-bin LPT,
  balance the unseen middle only through its total-work aggregate, then place
  the visible bottom descriptors by LPT.  This preserves the known descending
  order without pretending the unseen middle is individually stored.
- `S2 monotone one-round lookahead`: for each current candidate, form at most
  one legal next-round physical representative from each action family
  `SINGLE/PAIR/SPLIT/PREFETCH`, apply each exact transition, and use
  `max(S1(current_child), min S1(next_child))`.  Thus at most four next
  children, not a future tree, are evaluated per current candidate.  The
  monotone envelope is mandatory: the lookahead may expose and penalize an S1
  underestimate, but may not lower the current S1 score and repeatedly defer a
  large expert beyond the finite horizon.

Each scorer is evaluated both without pathmax and with only the mandatory-DMA
capacity pathmax.  This creates exactly six baseline combinations:

```text
S0, S0+DMA, S1, S1+DMA, S2, S2+DMA
```

No fitted residual, decision tree, LUT, or depth correction is eligible until
these six physical baselines have been compared.  If a correction is later
needed, it must use only the fixed runtime features above, have depth at most
three, and have output clamped to `[-2*Tq, +2*Tq]`.

### Evaluation objective and gates

For a state `s` and generated candidate `a`, the offline label is the best
forced continuation value `Q(s,a)`.  The primary statewise diagnostic is
candidate-selection regret:

```text
Q(s, selected_by_score) - min_a Q(s,a)
```

Absolute-score MSE is not a selection objective.  A scorer is accepted only
after both of the following are reported:

- statewise candidate ranking: zero-regret rate, mean/p95/max regret and first
  error by action family and remaining-count band;
- complete round-by-round rollout: exact cases, mean/p95/max makespan ratio,
  paired wins/losses, candidate count and decision count.

The final choice between `R4` and `R4+B2` is made with the selected scorer in
closed loop.  A generator is not accepted from reference-history coverage
alone, and a scorer is not accepted from isolated-state fitting alone.

### Locked execution order

1. Freeze and unit-check the bounded runtime feature contract.
2. Implement `S0/S1/S2` and their DMA-pathmax variants in the existing
   derivation script; do not create another parallel pipeline.
3. Reconstruct the existing complete R4+bottom2 counterfactual groups and use
   them for an initial ranking audit.
4. If coverage is insufficient, generate one stratified counterfactual dataset
   for the final candidate alternatives, with case-level fit/calibration
   separation and resumable entry IDs.
5. Compare all six scorers statewise, then run paired closed-loop `R4` versus
   `R4+B2` validation with the best physical scorer.
6. Only if the physical scorer misses the predeclared gate, fit the bounded
   correction and repeat the same closed-loop comparison.
7. Freeze generator, scorer, constants and numeric tie-break; then generate a
   fresh blind-v2 set exactly once.  RTL work begins only after that result.

Any command estimated to exceed 30 minutes is handed to the user as a complete
background command with an explicit log, result path, PID/status check and
resume behavior.  Short checks and implementation tests are run directly.

### Execution status (2026-07-18)

Steps 1--4 are complete.  The single bounded v1 dataset contains 384 mixed
reference, R4/S2+DMA and R4+bottom2/S2+DMA states and 13,687 unique physical
candidates.  On all 105 states whose complete R4/R4+bottom2 union was proven,
both generators had zero exact candidate loss.  This separates the remaining
error from candidate coverage: it is scorer or receding-horizon error.

The first non-monotone S2 repeatedly postponed a large expert.  It was rejected
and replaced by the predeclared monotone envelope.  With the canonical
`rem-snap-action` tie, R4+bottom2/S2+DMA has 99/108 zero-regret fully proven
states, 2,294.52-cycle mean exact regret and 22,528-cycle p95 exact regret.
Among the six fixed physical scorers it has the best fully proven mean regret.

Complete 4,739-case validation with the former candidate-index tie gave:

| Policy | Exact | Mean ratio | p95 ratio | Mean regret | p95 regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| R4/S1+DMA | 4,046 | 1.017019 | 1.060606 | 17,379.69 cc | 67,584 cc |
| R4/S2+DMA | 4,135 | 1.009942 | 1.040000 | 13,543.42 cc | 45,056 cc |
| R4+B2/S2+DMA | 4,158 | 1.008698 | 1.037037 | 11,710.85 cc | 33,792 cc |

S2 versus S1 was better on 331 cases and worse on 45, reducing mean makespan
by 3,836.27 cycles/case.  Adding bottom2 under S2 was better on 98 cases and
worse on 79, reducing mean makespan by 1,832.57 cycles/case and increasing
exact cases by 23.  These closed-loop results confirm the previously locked
`R4 + bottom2 + concrete residency, K32` generator under the bounded scorer.

The final tie no longer uses a generator-dependent candidate index.  A
fixed-width direct-v9 physical-action key had zero identity collisions on all
384 dataset states.  On a paired 180-case validation pilot, replacing the
index with this key while retaining `remaining -> snap` produced 163 exact
cases versus 160, was better on nine cases and worse on one, and reduced total
makespan by 1,869,824 cycles.

A Q-directed `snap -> remaining` alternative improved isolated proven-state
ranking but failed closed loop: only 107/180 exact cases, 66 regressions versus
two improvements, and 52,918,272 additional cycles relative to
`remaining -> snap`.  It is rejected.  This also rejects a PREFETCH-favoring
tie or fitted correction based only on isolated-state Q.  No residual model,
decision tree or LUT is selected.

The complete canonical-key validation subsequently finished all 4,739 proven
validation cases:

| Group | Cases | Exact | Mean ratio | p95 ratio | Mean regret | p95 regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E8 | 1,912 | 1,711 | 1.004669 | 1.020408 | 5,307.98 cc | 33,792 cc |
| E32 | 1,541 | 1,388 | 1.006986 | 1.023810 | 7,536.13 cc | 33,792 cc |
| E64 | 1,286 | 1,133 | 1.011263 | 1.041667 | 17,404.02 cc | 33,792 cc |
| Overall | 4,739 | 4,232 | 1.007212 | 1.025000 | 9,314.96 cc | 33,792 cc |

Relative to the former candidate-index tie on identical cases, the canonical
tie was better on 160 cases, worse on 48 and identical on 4,531.  It added 74
exact cases and reduced total makespan by 11,354,112 cycles.  Every E group
improved in mean makespan.  Relative to the historical P5 R8/full-LPT point,
it added 104 exact cases and improved overall mean ratio from 1.013338 to
1.007212 and p95 from 1.040816 to 1.025000.

An independent implementation in `scheduler_policy_golden.py` reproduced the
complete action history, history SHA-256, makespan, decision count and maximum
candidate count on five high-risk E8/E32/E64 cases, including the largest
paired improvements and regressions.  The frozen validation report is
`results/policy_search/bounded_r4_b2_s2_dma_canonical_validation_v1.json`,
SHA-256
`cc573c077763842d7095ff720fd13a763915bb6df2a164ceb5bd8ba7b6a040c0`.

The frozen policy is therefore R4+bottom2/K32 with monotone S2,
mandatory-DMA pathmax and the canonical `score-rem-snap-action` key.  No
formula, template, constant, tie-break or fitted model may change after the
fresh blind-v2 inputs are generated.  Only that blind evaluation remains.
The source and constant freeze manifest is
`results/policy_search/bounded_policy_freeze_v1.json`, SHA-256
`45771ae84f84edc90683556f1fc9d50681efac6b4cbba2c5ce0613b71460c93f`.

### Blind-v2 final decision (complete, 2026-07-19)

Blind-v2 was generated once from the frozen manifest with seed 20260718.  It
contains 976 analysis-eligible cases for each of E8, E32 and E64, or 2,928 in
total.  The independent golden model completed every case with the frozen
policy ID and no illegal history.

The reference used a 60-second/K16 first pass followed only on the 343
high-gap cases by termination-aware refinement.  Expansion-limited cases used
K256/180 seconds; time-limited cases retained K16 and used 180 seconds.  Final
certificates merge all passes with `UB = min(UB_i)` and `LB = max(LB_i)`.
Every merged result passed `LB <= UB`, gap recomputation and history-source
checks.  Refinement improved 20 upper bounds by 596,992 cycles in total,
strengthened nine lower bounds and added 19 optimality proofs.

Final reference quality and frozen-policy results are:

| Evidence set | Cases | Policy exact/equal-UB | Mean ratio | p95 ratio | Max ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Proven optimum | 2,395 | 2,369 | 1.002485 | 1.000000 | 1.625000 |
| Certified within 3% | 2,605 | 2,578 | 1.002305 vs UB | 1.000000 vs UB | 1.625000 vs UB |
| All best-known UB | 2,928 | 2,900 | 1.002057 | 1.000000 | 1.625000 |

On the certified-within-3% set, the policy-to-LB certified upper ratio has
mean 1.003490 and p95 1.013468.  Of the 323 cases whose reference gap remains
above 3%, 322 use exactly the same makespan as the best reference UB.  These
323 rows remain bounded evidence and are not called optimal.  Their loose
certificates are dominated by reference lower-bound/search difficulty; they
do not justify changing a policy after opening blind-v2.

The frozen policy therefore passes the blind gate and is accepted as the RTL
target.  The canonical summary is
`results/blind_v2/final_evaluation_summary.json`, SHA-256
`0651733726d4c119d50eab5044fb3241a4ded6dcdd5e5f669fd784c5152467b3`.
The row-level comparison is
`results/blind_v2/frozen_policy_vs_reference_final.json`, SHA-256
`e6445b247a23c909026577af24c5fc449856295666772f962d6e80db9f62a192`.

## Historical P5: R8/K32 full-descriptor experiment

The following P5 section is retained as experimental evidence.  Its E64 local
descriptor store and full-list LPT scan violate the active bounded-window
runtime boundary and must not be treated as the current RTL target.

### Fixed architecture boundary

- Candidate experts are the ordered top eight remaining entries plus concrete
  named residency/prefetch identities; no bottom selector is maintained.
- Every decision still emits at most 32 direct-v9 candidates and applies the
  exact four-stage child transition before scoring.
- CVA6 writes the sorted descriptor list once at batch initialization.  RTL
  owns its valid mask, extracts the first eight valid ranks for candidate
  generation and scans all valid entries for full-LPT.  There is no per-round
  software refill, bottom selector or mirrored hardware window.
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

### Runtime score alternatives

Candidate generation is frozen.  The remaining architecture decision is the
amount of distribution state used by the future estimate:

- `window-LPT`: run two-bin integer LPT over the current top8 and concrete
  named residency entries, then add the unseen tail through its incrementally
  maintained total-work aggregate;
- `full-LPT`: store every remaining descriptor locally and run the same LPT
  update over all remaining experts for every candidate.

`window-LPT` requires only the ordered top8 window, refill cursor, remaining
count and total remaining isolated work.  `full-LPT` removes the tail
approximation but needs up to E64 descriptors and up to `K32 * E64` entry
visits per decision.  Both use the exact four-stage child transition before
the future estimate.  There are no learned coefficients in either deployable
alternative.

The rejected aggregate base was:

```text
average      = ceil((load2 + load3 + remaining_work) / 2)
single_chain = min(load2, load3) + top0_work
rtl_base     = max(pathmax, load2, load3, average, single_chain)
score        = 16 * rtl_base + sum(weight[mode,term] * term)
```

Timing uses a 5,632-cycle unit because legal DMA/pathmax boundaries may be
half of the 11,264-cycle compute quantum.  The aggregate base and its
mode-selected shift/add corrections remain diagnostic experiments only.

### Counterfactual diagnostic data

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

This objective improved held-out single-state ranking but did not predict
closed-loop behavior.  On the fixed 180-case validation pilot,
`full-LPT-base` achieved 160 exact cases, mean ratio 1.015466 and p95 1.028571;
adding the fitted timing correction fell to 97 exact cases, mean 1.048476 and
p95 1.375.  The selected aggregate correction had already fallen to 100 exact
cases.  Consequently no fitted residual is eligible for P5 deployment.  The
Q dataset is retained to explain candidate-level errors, not to override
end-to-end rollout selection.

### P5 acceptance gates

- exactly one complete, consecutively indexed group of at most K32 candidates
  per sampled state; deterministic resume and no duplicate entry IDs;
- no candidate continuation undercuts a proven root reference;
- full validation reports paired results for `window-LPT`, `full-LPT`, and the
  frozen R4+bottom2/LPT baseline on identical cases;
- prefer `window-LPT` only if, relative to `full-LPT`, its paired mean
  makespan increase is at most 5,632 cycles/case, p95 ratio increases by at
  most 0.005, and exact count decreases by at most 1% of evaluated cases;
- otherwise select `full-LPT`; fitted residual profiles are ineligible even if
  their single-state calibration objective is better;
- after implementation and constants are frozen, generate and solve a fresh
  independent blind-v2 distribution.  The previously opened blind split is
  diagnostic only for P5.

### P5 validation decision (complete, 2026-07-17)

Both alternatives completed the same 4,739 proven validation cases.  Paired
results were 4,665 identical schedules, 74 cases favoring full-LPT and zero
favoring window-LPT.  Window-LPT increased mean makespan by 549.06 cycles per
case; overall p95 ratio remained 1.040816 and its maximum paired increase was
67,584 cycles.  It therefore passed the mean and p95 simplification limits.

Window-LPT produced 4,052 exact cases versus 4,125 for full-LPT, a loss of 73
or 1.54% of the evaluated set.  This exceeds the predeclared maximum loss of
1% (47 cases).  P5 therefore selects `full-LPT` exactly as required by the
gate.  The frozen deployment point is `R8 + concrete residency`, `K32`, exact
four-stage child transition, full-list integer LPT and `rem-snap` tie-break.
No fitted coefficient bank is part of the selected algorithm.

### P5 pathmax reduction

RTL mapping exposed that the original full-LPT score also inherited the
reference search `f_score`.  A controlled 180-case ablation separated its
compute, release-chain, critical-chain and mandatory-DMA components.  Removing
all pathmax terms reduced exact cases from 160 to 153 and worsened p95 from
1.028571 to 1.111111, so pathmax cannot be silently dropped.  The three
compute/chain components produced exactly the same histories as no pathmax;
DMA capacity alone reproduced the complete `f_score` policy with zero
makespan or history-hash mismatches.

The deployable P5 score is therefore full-list LPT plus one inherited
mandatory-DMA-capacity pathmax.  Its independent golden implementation counts
8 `Hq` lane units for every uncovered S1 transfer and 4 for every uncovered
S3 transfer, then sweeps the committed two-lane DMA endpoints.  Full validation
produced zero summary, makespan, decision-count or history-hash mismatches
against the already recorded full-`f_score` report across all 4,739 cases.
The reduction is therefore frozen.

### P5 RTL-order correction

RTL mapping found that direct-v8 used Python `repr(action_key)` as the last
local-action tie-break.  That compares decimal integers as strings and is not a
valid hardware contract.  Direct-v9 replaces only this last tie-break with the
same fields in fixed-width numeric order.  The archived direct-v8 path remains
available for exact reproduction of prior reports.

On a paired 180-case validation pilot, direct-v9 changed one action history and
zero final makespans; there were no better or worse cases.  The subsequent
complete 4,739-case independent DMA-pathmax run changed 17 histories and four
makespans relative to direct-v8.  All four changes were improvements, none were
regressions, and the maximum improvement was 33,792 cycles.  Direct-v9 reached
4,128 exact cases, mean ratio 1.013338 and p95 1.040816, versus 4,125 exact and
mean 1.013365 for direct-v8.  The numeric ordering is therefore frozen.  This
correction does not reopen R8, K32, full-LPT, the family quotas or the
DMA-pathmax equation.

The frozen report is
`results/policy_search/r8_p5_v9_dma_pathmax_validation_full.json`, SHA-256
`14f1756dd59d3ef82933e61afd3fa9f801db949e066868d3ef7a77d0db11c5b4`.

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

## Frozen bounded future-cost contract

Every current candidate first receives the exact four-stage transition.  S1
starts two integer `Hq` loads from the child C2/C3 task endpoints, places the
visible head descriptors by LPT, balances the unseen middle through only its
aggregate work, and then places the visible bottom descriptors by LPT.

For monotone S2, generate at most one next physical representative from each
of `SINGLE`, `PAIR`, `SPLIT` and `PREFETCH`, using only descriptors visible in
the current invocation.  Apply each exact transition and compute:

```text
pm_child   = max(pm_parent, dma_capacity_bound(child))
current    = max(S1(parent, child), pm_child)
next_best  = min(max(S1(parent, grandchild), pm_grandchild))
score_q    = max(current, next_best)
```

If there is no legal next representative, `score_q = current`.  The pathmax
register contains only the mandatory-DMA capacity bound.  No hidden refill,
unseen-middle descriptor scan, fitted coefficient, tree, LUT, general
multiplier or divider is used.

Candidates are compared lexicographically by:

```text
(score_q,
 child.remaining_count,
 max(child.c2.task_end, child.c3.task_end),
 rtl_action_order_key(action))
```

The final action key is the direct-v9 fixed-width numeric encoding of the
physical action fields.  Candidate slot index remains report metadata and does
not affect selection.

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
