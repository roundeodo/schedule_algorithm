# Bounded distilled scheduler contract

Status: final RTL-oriented Python mirror contract.  RTL implementation and
synthesis are not claimed by this document.

Policy ID: `bounded-distilled-top5-bottom1`.

Normative Python entry point: `scheduler_rtl_distilled_policy.py`.

## 1. Final one-round architecture

Every scheduling round follows one path:

```text
T0..T4 + B0 + cluster/DMA state
  -> materialize one fixed physical-profile bank
  -> local reduction within each logical action
  -> exact four-stage child-state calculation
  -> one bounded continuation comparator
  -> one global winner
  -> commit one action and update the bounded state
```

There is no base/recovery split, protected winner, recovery margin,
candidate-origin bit, distribution classifier, beam expansion, child-round
rollout, SIM1, standalone S4PF action, floating-point coefficient, or runtime
policy table.

The M-outer four-stage timing and explicit IDMA/XDMA/BOTH legality model are
unchanged.  This work changes candidate organization and arbitration; it does
not change S1/S2/S3/S4 execution semantics.

## 2. Offline derivation and distillation method

The final policy is the result of offline structural distillation, not a
gradient-trained model:

1. the four-stage reference search supplies optimal paths for 65 directed
   OLMoE-style distributions;
2. the frozen v4 profile union supplies candidate physical implementations;
3. closed-loop ablation separates logical-action coverage, physical-profile
   reduction, and continuation selection;
4. redundant action families and dominated physical profiles are removed on
   discovery data;
5. the compact policy is frozen and independently evaluated on validation,
   blind-test, directed, and certificate inputs.

The structural ablation is reproducible with
`ablate_scheduler_rtl_distilled_structure.py`:

| Variant | Exact certificates | Total target gap | Meaning |
|---|---:|---:|---|
| frozen v4 | 65/65 | 0 tick | frozen quality baseline |
| naive single-bank union | 6/65 | +870 tick | deleting protected arbitration alone is invalid |
| source-order local reduction | 65/65 | 0 tick | inherited profile order hides a physical tie-break |
| reversed source order, no semantic tie-break | 34/65 | +131 tick | profile-order dependence is not an acceptable RTL contract |
| S2PF-aware local reduction | 65/65 | 0 tick | explicit physical tie-break |
| S2PF-aware reduction with reversed source order | 65/65 | 0 tick | quality is invariant to profile enumeration |
| fallback scalar continuation key | 51/65 | +100 tick | a pure fallback lexicographic key is insufficient |
| final 32-profile policy | 65/65 | 0 tick | final distilled structure |

This evidence fixes the causal interpretation:

- synchronous one-cluster issue is removed because both clusters are available
  in `SYNC`; retaining it wastes an issue opportunity and caused the naive
  union to select locally attractive but globally poor actions;
- physical profiles must be reduced before global scoring, otherwise one
  logical action receives multiple votes;
- an S2 prefetch that changes no current task-end time must be retained because
  it reduces future DMA work;
- the bounded state-conditioned continuation comparator is retained because
  the scalar fallback fails in closed loop.

## 3. Bounded observation state

Remaining experts are sorted by descending token count, with expert ID as the
deterministic equal-load tie-break.  The visible descriptor window is:

- `T0..T4`: five hottest remaining experts;
- `B0`: the coldest remaining expert.

Overlapping head and bottom aliases are deduplicated.  An unavailable selector
invalidates only the corresponding logical action.

The continuation comparator also consumes bounded aggregate state:

- remaining expert count;
- remaining token sum;
- remaining odd-token count;
- remaining best serial work or equivalent M2-block sum;
- counts of one-, two-, three-, and four-M2-block experts;
- monotone parent lower bound;
- current C2/C3 task, cache, S2PF, and DMA timestamps.

The Python mirror recomputes aggregate counters.  RTL must initialize them once
and subtract the committed expert contributions each round.

## 4. Logical actions and physical profiles

The final hard-wired bank contains 32 physical profiles:

- 19 `ONE_IDLE` profiles;
- 8 `SYNC` profiles;
- 5 `TERMINAL` profiles.

These are combinational decode cases, not a 32-entry runtime ROM or RAM.  The
exact field constants are in `scheduler_rtl_distilled_profiles.py`.

The 32 profiles implement eleven logical action templates:

| State mode | Logical actions |
|---|---|
| `SYNC` | `PAIR(B0,T0)`, `PAIR(T0,T1)`, `PAIR(T0,T4)`, `PAIR(T1,T2)`, `PAIR(T2,T3)`, `SPLIT(T0,HALF)` |
| `ONE_IDLE` | `SINGLE(B0)`, `SINGLE(T0)`, `SINGLE(T3)` |
| `TERMINAL` | `SINGLE(T0)`, `SPLIT(T0,BALANCED)` |

`SYNC SINGLE` is absent by construction.  Fewer than eleven templates are ever
simultaneously active because mode is exclusive; the global scorer sees at
most six logical candidates.

For each physical profile, the lowering logic considers legal event-aligned
starts and retains the realization with the earliest current-round finish.
Shape, DMA binding, cache-hit, and S2PF legality remain explicit.

## 5. Physical-profile local reduction

Profiles are grouped by the complete logical-action identity:

```text
(mode, family, visible selectors, split rule)
```

Within each group, the unique physical winner minimizes:

```text
Rphysical = (
  max(c2_task_end, c3_task_end),
  c2_task_end + c3_task_end,
  latest_selected_task_start,
  -number_of_S2_prefetches,
  fixed_profile_slot
)
```

`fixed_profile_slot` is only the final exact-tie priority.  No Python string or
candidate-origin field participates in the decision.  Equal child states are
deduplicated after local reduction.

The five physical profiles removed during the final pruning never became a
local winner in the discovery split.  Re-running the unchanged policy on
validation and blind-test inputs confirmed bit-identical final makespans.

## 6. Single bounded continuation comparator

Every locally reduced logical action enters the same public selector:

```text
select_bounded_continuation_winner(state, logical_candidates)
```

Candidate provenance is not an input.  The comparator first calculates four
bounded continuation quantities:

- monotone combined lower bound `F`;
- head-5 plus four-bin tail-histogram LPT estimate `H`;
- compute-capacity lower bound `C`;
- DMA-capacity lower bound `D`.

The fallback priorities are:

```text
SYNC:
  (F, H, C, D, -largest_selected_load,
   selected_load_sum, committed_makespan, -S2PF_count)

ONE_IDLE:
  (F, H, C, D, later_release, earlier_release,
   selected_load_sum, committed_makespan, -S2PF_count,
   remaining_count)
```

One deterministic `better(lhs, rhs, state)` comparator folds the candidate
stream once.  Five bounded state predicates change field priority when the
fallback bound is unable to distinguish future progress:

- sparse-hot synchronization;
- low-work one-idle progress;
- mid-plateau prefetch progress;
- short-tail plateau fill;
- large-slack fill.

These are subconditions inside one comparator, not separate candidate paths or
separate scorers.  Their exact integer predicates and dominance guards remain
frozen in `select_practical_probe_candidate`, called only through the public
`select_bounded_continuation_winner` entry point.  They use window ranks,
aggregate counters, integer comparisons, additions, subtractions, and shifts;
there is no multiply-accumulate coefficient array.

The scalar-only ablation reached only 51/65 optimal certificates.  Therefore
replacing the comparator with the fallback tuple is explicitly rejected.

## 7. Commit and state update

Exactly one winner is committed per round.  The selected compact action is
replayed through the four-stage transition and updates:

- remaining experts and aggregate counters;
- C2/C3 task and DMA snapshots;
- S1/S3 residency and S2PF state;
- the monotone lower bound;
- the software-visible bounded window refill count.

The scheduler remains a slave.  It does not fetch expert descriptors or move
expert data by itself.

## 8. Complexity contract

The complete 29,928-case run observed:

- maximum materialized physical candidates: 14;
- per-mode maxima: `SYNC=14`, `ONE_IDLE=11`, `TERMINAL=5`;
- maximum logical candidates entering the continuation comparator: 6;
- maximum selected logical slot: 5.

The mirror retains a conservative assertion budget of 18 physical candidates.
The observed value 14 is validation evidence, not a formal all-reachable-state
proof.

A resource-reused RTL does not store 32 wide child states.  It can iterate the
hard-wired profile cases, retain at most six compact local-winner records, and
then replay those records through the shared timing/scoring datapath.  Relative
to v4, the recovery winner record, protected winner record, origin bit,
one-tick margin adder/comparator, and up-to-twelve-entry global scorer stream
are removed.  The only new local comparison field is the S2PF count, followed
by the existing fixed profile index.

No LUT, FF, timing, or power improvement is claimed until RTL synthesis and
lockstep simulation are complete.

## 9. Validation evidence

### 65 optimal certificates

- exact: 65/65;
- total target gap: 0 tick;
- equal to frozen v4: 65/65;
- explicit-DMA history replay and per-round slot/action/score regeneration:
  passed.

### 29,928 random cases

| Comparison | Better | Equal | Worse | Aggregate delta |
|---|---:|---:|---:|---:|
| distilled vs v4 | 2,163 | 25,975 | 1,790 | −5,090 tick, −0.0943% |
| distilled vs adaptive | 6,965 | 18,576 | 4,387 | −28,734 tick, −0.5301% |
| v4 vs adaptive | 6,960 | 18,348 | 4,620 | −23,644 tick, −0.4362% |

The improvement over v4 is present in every split:

| Split | Better | Equal | Worse | Delta vs v4 |
|---|---:|---:|---:|---:|
| discovery | 1,291 | 15,600 | 1,068 | −2,917 tick |
| validation | 441 | 5,193 | 353 | −1,099 tick |
| blind test | 431 | 5,182 | 369 | −1,074 tick |

For the strict OLMoE-style subset, the distilled policy is 86/167/54 against
v4 and reduces the aggregate by 507 tick (−1.0127%).  Against adaptive it
reduces the aggregate by 3,896 tick (−7.2886%).

The E8 subset remains a limitation relative to adaptive: its aggregate is
4,649 tick higher (+0.2714%).  The policy still improves E8 over v4 by 1,137
tick.  The intended high-expert-count regime is stronger: E64 improves v4 by
2,823 tick and adaptive by 27,516 tick.

### Window-normalization comparison

The prior result mislabeled the same 32-profile/head5 policy as top6+bottom2.
On the identical 29,928 inputs, the explicit top5+bottom1 normalization is
bit-identical on 29,922 cases.  Five cases improve, one case regresses, and the
aggregate decreases by another 12 ticks.  All six differences occur at a
six-expert tail where the old window aliased `B0` as `T5`; the narrower window
keeps `B0` distinct and therefore changes a final candidate-order tie-break.
Every changed history passes explicit-DMA four-stage replay.

### Directed showcase

Distribution:

```text
22, 18, 14,
3 x 19,
2 x 8,
1 x 13
```

This is 140 assignments over 43 active experts in a conceptual 64-expert
layer.  All four methods use workload-encodable fixed DMA bindings and the same
four-stage timing model.

| Method | Makespan |
|---|---:|
| descending issue | 163 tick |
| ascending issue | 165 tick |
| ends-inward issue | 168 tick |
| distilled scheduler | 129 tick |
| certified optimum | 129 tick |

The distilled scheduler reduces elapsed time by 20.86%–23.21% relative to the
three fixed-order baselines.

## 10. Normative files

- final mirror: `scheduler_rtl_distilled_policy.py`;
- hard-wired profiles: `scheduler_rtl_distilled_profiles.py`;
- continuation arithmetic and public selector:
  `evaluate_olmoe_fixed_token_banks.py`;
- certificate/30K validator: `verify_scheduler_rtl_distilled_policy.py`;
- directed validator: `verify_scheduler_rtl_distilled_showcase.py`;
- structural ablation: `ablate_scheduler_rtl_distilled_structure.py`;
- 65 result:
  `results/policy_search/bounded_top5_bottom1_certificate_validation.json`;
- 30K result:
  `results/policy_search/bounded_top5_bottom1_random_validation.json`;
- directed result:
  `results/policy_search/scheduler_rtl_distilled_showcase.json`;
- ablation result:
  `results/policy_search/scheduler_rtl_distilled_structure_ablation.json`.

The archived protected18-v4 files remain baseline evidence only and are not the
final implementation contract.
