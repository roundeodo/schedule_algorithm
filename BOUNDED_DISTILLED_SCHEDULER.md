# Bounded distilled scheduler contract

Status: final Python policy contract with a direct C translation. RTL
implementation, Python/RTL lockstep, synthesis, and timing closure are not
claimed here.

Policy ID: `bounded-distilled-top5-bottom1-targeted-s4pf`.

Normative Python entry point: `scheduler_rtl_distilled_policy.py`.

## 1. One-round architecture

Every scheduling round follows one unified path:

```text
T0..T4 + B0 + bounded cluster/DMA state
  -> decode the fixed physical-profile bank
  -> retain the earliest legal transition per runtime physical profile
  -> deduplicate exact future-equivalent physical children
  -> retain a no-S4PF realization for every logical action
  -> jointly lower legal S4PF with its concrete next consumer
  -> locally reduce physical realizations of each logical action
  -> calculate the exact four-stage child state
  -> evaluate one bounded continuation comparator
  -> select one global winner
  -> commit the attached S4PF, the consumer action, and the bounded state
```

There is no base/recovery split, protected winner, recovery margin,
candidate-origin bit, distribution classifier, beam expansion, child-round
rollout, SIM1, standalone S4PF candidate, floating-point coefficient, or
runtime policy table.

The M-outer S1/S2/S3/S4 timing equations and explicit DMA-bandwidth model are
unchanged. The policy changes candidate organization, physical lowering, and
continuation arbitration; it does not change the four-stage execution model.

## 2. Offline derivation and distillation

The final policy is a structurally distilled algorithm, not a
gradient-trained model:

1. the four-stage reference search supplies certified best paths for 65
   directed OLMoE-style distributions;
2. the frozen comparison baseline supplies candidate physical profiles;
3. closed-loop ablation separates logical-action coverage, physical-profile
   selection, and continuation selection;
4. equivalent single-DMA lane swaps and dominated action families are removed;
5. S4PF is reintroduced as a physical realization of a known next-consumer
   action rather than as an independent or wildcard action;
6. the frozen policy is evaluated on the 65 certificates, the partitioned
   29,928-case corpus, and the directed showcase.

The 29,928-case partition labels are retained for reporting. They are not
claimed as a never-observed blind test because aggregate results were inspected
during development.

The important S4PF causal ablation is:

- a post-winner wildcard S4PF passed current-state bandwidth checks but changed
  64 of the 65 certified schedules for the worse;
- a target-aware S4PF realization keeps the no-S4PF child as the local
  baseline, names the concrete next same-cluster expert, and is accepted only
  after the complete consumer child is available;
- the final target-aware policy leaves all 65 certificate makespans unchanged
  and reduces the aggregate 29,928-case makespan by 485 ticks relative to the
  same policy with S4PF disabled.

This distinction is necessary because `BW-OK` is evaluated against DMA
intervals already represented in the bounded state. A wildcard prefetch can
still occupy bandwidth needed by a future task whose interval has not yet been
generated. Therefore current-state bandwidth legality is necessary but not
sufficient for closed-loop benefit.

## 3. Bounded observation state

Remaining experts are sorted by descending token count, with expert ID as the
deterministic equal-load tie-break. The visible descriptor window is:

- `T0..T4`: five hottest remaining experts;
- `B0`: the coldest remaining expert.

Overlapping head and bottom aliases are deduplicated. An unavailable selector
invalidates only the corresponding logical action.

The continuation comparator also consumes bounded aggregate state:

- remaining expert count, token sum, and odd-token count;
- remaining serial work or equivalent M2-block sum;
- counts of one-, two-, three-, and four-M2-block experts;
- monotone parent lower bound;
- current C2/C3 task, cache, S2PF, and DMA timestamps.

The Python mirror recomputes aggregate counters. RTL should initialize them
once and subtract committed expert contributions each round.

## 4. Logical actions and fixed physical profiles

The hard-wired bank contains 28 canonical physical profiles:

- 15 `ONE_IDLE` profiles;
- 8 `SYNC` profiles;
- 5 `TERMINAL` profiles.

They are combinational decode cases, not a runtime ROM, RAM, register table, or
software-loaded policy memory. The exact constants are frozen in
`scheduler_rtl_distilled_profiles.py`.

Every single-lane transfer is canonicalized consistently across S1, S2PF, and
S3: C2 uses IDMA and C3 uses XDMA. Explicit `BOTH` profiles remain physical
choices. Deleting only the S2PF `BOTH` choices reduced the certified coverage
to 33/65, so a single-only S2PF policy is not part of the frozen model without
a new constrained reference search.

The profiles implement eleven logical action templates:

| State mode | Logical actions |
|---|---|
| `SYNC` | `PAIR(B0,T0)`, `PAIR(T0,T1)`, `PAIR(T0,T4)`, `PAIR(T1,T2)`, `PAIR(T2,T3)`, `SPLIT(T0,HALF)` |
| `ONE_IDLE` | `SINGLE(B0)`, `SINGLE(T0)`, `SINGLE(T3)` |
| `TERMINAL` | `SINGLE(T0)`, `SPLIT(T0,BALANCED)` |

`SYNC SINGLE` is absent by construction. Mode is exclusive, so the global
continuation scorer sees at most six logical candidates in one round.

## 5. Target-aware S4PF lowering

S4PF is evaluated only as a physical realization of a concrete consumer
action. For each active cluster, the lowering follows this sequence:

1. retain the no-S4PF consumer as the baseline;
2. require at least nine remaining experts;
3. require a previous task on the same cluster and no existing S4 prefetch;
4. set the target EID to the concrete consumer selected by the profile;
5. set `pf_start` to the previous task's `dma3_end`;
6. reject a retroactive interval that cannot be certified from the bounded
   peer snapshot;
7. try the cluster-local single lane first: C2/IDMA or C3/XDMA;
8. if single does not fit, try `BOTH`;
9. require `pf_end <= previous_compute_end` and exact `bw_feasible`;
10. rematerialize the consumer as an S1-cache hit and calculate its complete
    child state;
11. replace the no-S4PF local baseline only if the current global task end is
    improved by at least one tick.

Cluster trial order is C2 then C3. Per-cluster DMA trial order is local
`SINGLE`, `BOTH`, then implicit `OFF`. The OFF realization is never removed by
the feasibility check alone.

The Python history emits the targeted S4PF action immediately before its
consumer so explicit-DMA replay can validate it. In RTL, the preceding task
record can remain pending until the next same-cluster consumer EID is known;
the S4PF descriptor is then attached to that preceding task. No wildcard cache
state is committed.

## 6. Physical-profile local reduction

Profiles are grouped by the complete logical-action identity:

```text
(mode, family, visible selectors, split rule)
```

Before logical grouping, every runtime physical profile retains its earliest
legal transition and exact future-equivalent children are deduplicated in
runtime-profile order. This order is part of the deterministic policy because
two cluster-swapped actions can have identical timing and child state while
emitting different task records.

Then the unique OFF baseline of each logical action minimizes:

```text
Rphysical = (
  max(c2_task_end, c3_task_end),
  c2_task_end + c3_task_end,
  latest_selected_task_start,
  -number_of_S2_prefetches,
  -number_of_targeted_S4_prefetches,
  fixed_profile_slot
)
```

The best target-aware realization uses the same key. It replaces the OFF
baseline only after the one-tick current-gain guard in Section 5 passes.
`fixed_profile_slot` is only the final explicit tie priority. Runtime physical
profile order resolves a remaining exact tie. Equal child states are
deduplicated once more after local reduction.

## 7. Single bounded continuation comparator

Every locally reduced logical action enters the same public selector:

```text
select_continuation_winner(state, logical_candidates)
```

Candidate provenance is not an input. The comparator calculates:

- monotone combined lower bound `F`;
- head-5 plus four-bin tail-histogram LPT estimate `H`;
- compute-capacity lower bound `C`;
- DMA-capacity lower bound `D`.

The common fallback priorities are:

```text
SYNC:
  (F, H, C, D, -largest_selected_load,
   selected_load_sum, committed_makespan, -S2PF_count)

ONE_IDLE:
  (F, H, C, D, later_release, earlier_release,
   selected_load_sum, committed_makespan, -S2PF_count,
   remaining_count)
```

One deterministic `better(lhs, rhs, state)` comparator folds the fixed-order
logical stream once. Five bounded state predicates adjust field priority when
the fallback bound cannot distinguish future progress:

- sparse-hot synchronization;
- low-work one-idle progress;
- mid-plateau prefetch progress;
- short-tail plateau fill;
- large-slack fill.

These predicates are subconditions inside one comparator, not separate policy
paths. Their exact integer conditions are frozen in
`scheduler_rtl_distilled_scoring.py`. A scalar-only fallback reached only
51/65 optimal certificates and is therefore rejected.

## 8. Commit and state update

Exactly one logical winner is committed per round. If its selected physical
realization contains S4PF, the targeted prefetch is replayed before the
consumer transition. The commit updates:

- remaining experts and aggregate counters;
- C2/C3 task and DMA snapshots;
- S1/S3 residency and S2PF state;
- the monotone lower bound;
- the software-visible bounded-window refill count.

The scheduler remains a slave. It does not fetch expert descriptors or move
expert data by itself.

## 9. Complexity contract

The complete 29,928-case run observed:

- 28 hard-wired canonical profile cases;
- maximum materialized top-level physical candidates: 13;
- per-mode maxima: `SYNC=13`, `ONE_IDLE=8`, `TERMINAL=5`;
- maximum logical candidates entering the continuation comparator: 6;
- 768 cases with committed S4PF and 2,738 committed S4PF events.

The mirror retains a conservative assertion budget of 18 top-level physical
candidates. The observed value 13 is validation evidence, not a formal proof
over every reachable state.

Targeted S4PF is a local physical trial, not a new top-level logical action. A
resource-reused RTL can evaluate the OFF baseline first, then evaluate the
targeted trial against the same logical-winner accumulator. It does not need to
store both wide child states or double the global candidate stream.

The RTL can iterate fixed decode cases, retain at most six compact local-winner
records, and replay them through shared timing/scoring logic. A 5-bit physical
profile index and a 3-bit logical-action ID are sufficient. No LUT, FF, timing,
or power improvement is claimed until RTL synthesis and lockstep simulation
are complete.

## 10. Validation evidence

### 65 optimal certificates

- exact: 65/65;
- total target gap: 0 tick;
- target-aware S4PF versus disabled S4PF: 0 better, 65 equal, 0 worse;
- final policy versus frozen comparison baseline: 0 better, 65 equal, 0 worse;
- explicit-DMA history replay and per-round regeneration: passed.

### 29,928 random cases

| Comparison | Better | Equal | Worse | Aggregate delta |
|---|---:|---:|---:|---:|
| S4PF on vs the same policy with S4PF off | 330 | 29,524 | 74 | -485 tick, -0.0090% |
| final policy vs frozen comparison baseline | 2,575 | 25,935 | 1,418 | -10,109 tick, -0.1873% |
| final policy vs adaptive policy | 7,484 | 18,506 | 3,938 | -33,753 tick, -0.6227% |
| frozen comparison baseline vs adaptive | 6,960 | 18,348 | 4,620 | -23,644 tick, -0.4362% |

The S4PF OFF/ON comparison is the direct measurement of the reintroduced
feature. It preserves every certified case and improves total random-corpus
time. It is not per-case monotonic: the one-round local improvement changes the
later heuristic trajectory in 74 cases. Eliminating all such changes would
require a stronger continuation predictor or a corpus-specific classifier;
neither is silently added to this bounded RTL policy.

The improvement over the frozen comparison baseline is present in every
partition:

| Split | Better | Equal | Worse | Delta vs baseline |
|---|---:|---:|---:|---:|
| discovery | 1,553 | 15,548 | 858 | -5,913 tick |
| validation | 513 | 5,199 | 275 | -2,131 tick |
| blind test | 509 | 5,188 | 285 | -2,065 tick |

On the strict OLMoE-style subset, the final policy is 120/158/29 against the
frozen comparison baseline and reduces the aggregate by 1,135 ticks. S4PF
itself is 1/306/0 on this subset and reduces three ticks.

The E8 subset remains a limitation relative to adaptive: the final aggregate
is 4,369 ticks higher. It still improves the frozen comparison baseline by
1,417 ticks. The intended high-expert-count regime is stronger: E64 improves
the frozen comparison baseline by 6,471 ticks and adaptive by 31,164 ticks.

### Directed showcases

The thesis comparison separates dynamic physical arguments from dynamic
logical scheduling. `STATIC_DESC` fixes B/B shapes, local single DMA lanes,
and disables prefetch. `DYNAMIC_DESC` keeps the same global descending queue
but dynamically chooses legal shape/DMA/S2PF/S4PF parameters.
`DYNAMIC_TWO_ENDED` uses the identical physical selector while C2 consumes the
hot end and C3 the cold end without a global barrier. `FULL_SCHEDULER` also
chooses PAIR/SINGLE/SPLIT, expert order, cluster mapping, and the bounded
continuation score.

| Distribution | Static desc. | Dynamic desc. | Dynamic two-ended | Full |
|---|---:|---:|---:|---:|
| certified OLMoE triple-hot | 162 | 159 | 137 | **129** |
| `M70: 28x3,6x4,2x16` | 132 | 126 | 127 | **105** |
| `M92: 76,40,2x32,1x4` | 198 | 168 | 172 | **144** |
| `M60: 36,22,13,6,2x17,1x9` | 138 | 133 | 111 | **99** |

The first full schedule matches its certified optimum. The M70 multi-hot case
gives 1.200x over dynamic descending and 1.210x over dynamic two-ended. The
skewed M92 case has 38 active and 26 inactive experts. It shows a 1.179x
physical-parameter benefit, a further 1.167x full-scheduling benefit over
dynamic descending, and a 1.194x benefit over dynamic two-ended. The M60 OLMoE-style high-skew
stress case contains 30 active experts and 60 conceptual experts at load at
most two. Full scheduling is 1.394x faster than static descending, 1.343x
faster than dynamic descending, and 1.121x faster than dynamic two-ended. This
is a structured stress case, not a measured router window. No representative
distributed OLMoE-style case reaching 1.3x against dynamic two-ended was found;
the largest selected full-versus-dynamic-two-ended ratio is 1.210x, while full
versus the no-scheduler static baseline reaches 1.394x.

## 11. Normative files

- final mirror: `scheduler_rtl_distilled_policy.py`;
- hard-wired profiles: `scheduler_rtl_distilled_profiles.py`;
- shared constants and types: `scheduler_rtl_distilled_types.py`;
- candidate and targeted-S4PF lowering:
  `scheduler_rtl_distilled_lowering.py`;
- continuation comparator: `scheduler_rtl_distilled_scoring.py`;
- certificate/30K validator: `verify_scheduler_rtl_distilled_policy.py`;
- four-policy evaluator: `evaluate_scheduler_thesis_four_policy.py`;
- deployed direct C translation:
  `../HeMAiA/target/sw/host/apps/offload_bingo_hw/single_chip/workloads/multi_cluster_MoE/moe_scheduler.c`;
- Python/C task, DMA, and tick lockstep validator:
  `verify_moe_scheduler_c_lockstep.py`;
- 65 result:
  `results/policy_search/bounded_top5_bottom1_fixed_lane_targeted_s4pf_certificate_validation.json`;
- 30K result:
  `results/policy_search/bounded_top5_bottom1_fixed_lane_targeted_s4pf_random_validation.json`;
- four-policy result:
  `results/policy_search/scheduler_thesis_four_policy_showcases.json`.

Archived protected-arbitration files and earlier targeted-S4PF result names are
baseline or ablation evidence only; they are not the final implementation
contract.

## 12. Python-to-C translation boundary

The implementation direction is strictly:

```text
scheduler_rtl_distilled_policy.py
  -> profiles + lowering + scoring semantics
  -> moe_scheduler.c
  -> public C task/DMA schedule
```

The existing RTL does not define C behavior. `moe_make_hw_plan()` is only an
optional compact-format export for later comparison; the production
`moe_schedule()` path lowers the Python-derived internal plan directly.

The C translation uses the same 11,264-cycle lattice tick internally and does
not use floating point, dynamic allocation, recursion, beam search, or a
runtime policy table. The implementation keeps one 2,304-byte static plan
scratch buffer. A native `-O2 -fstack-usage` build reports a 9,824-byte maximum
estimated stack frame after exact physical-child deduplication; this is below
the earlier approximately 19 KiB translation but still must be checked against
the final CVA6 stack allocation.

Current direct evidence is:

- strict native C99 build with `-Wall -Wextra -Werror`: passed;
- HeMAiA RV64 production build with `MOE_HW_SCHEDULER=0`: passed;
- 65 optimal certificates plus three directed/API cases: exact lockstep;
- ten targeted-S4PF cases with initial residency: exact lockstep;
- complete E8/E32/E64 30K coverage corpus: exact lockstep;
- independent 100-case random regression plus three directed/API cases: exact
  lockstep;
- the earlier four ranked-EID showcase inputs: exact task/DMA/tick lockstep;
  the new four-policy workload set remains Python-replay-validated only.

Lockstep means equality of makespan tick, ordered task fields, shape and DMA
bindings, S2PF operations, and concrete S4PF target operations. It is not only
an equality of final makespan.

The completed 30K evidence is:

```text
checkpoint: next_coverage_index=30000, coverage_total=30000, complete=true
log: PASS Python/C lockstep cases=30078
source SHA-256: 3f5f5e758f0f9554fb4bafbe25df887548eaeedc549dbda4cee5fb418cac6446
```

The full run remains checkpointed because it exceeds the short interactive-test
budget.  A fresh run uses:

```bash
cd /esat/studscratch/r1015673/Thesis/Idea_Model
nohup python3 -u verify_moe_scheduler_c_lockstep.py \
  --coverage-all \
  --checkpoint results/policy_search/python_c_lockstep_30k_checkpoint.json \
  --progress-every 100 \
  > results/policy_search/python_c_lockstep_30k.log 2>&1 < /dev/null &
```

The checkpoint embeds a SHA-256 digest of the Python policy, C source/header,
and validator. A changed source invalidates resume instead of silently mixing
two implementations. Progress can be inspected with:

```bash
tail -n 20 results/policy_search/python_c_lockstep_30k.log
```

For the eventual hardware-speedup claim, use the same workload and input in
both builds. The software path now records only the `moe_schedule()` call as
`[MOE_PHASE_MCYCLE] SW_SCHED`; the hardware path records the corresponding
MMIO scheduling interval as `HW_SCHED`. The surrounding
`BINGO_TRACE_HOST_MOE_SCHED_START/END` markers remain available for waveform
cross-checking. Scheduler model ticks measure predicted workload completion;
they are not CVA6 algorithm-execution cycles and must not be reported as the
software scheduler runtime.
