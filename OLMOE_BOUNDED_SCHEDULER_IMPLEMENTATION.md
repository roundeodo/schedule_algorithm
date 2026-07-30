# Unified top6+bottom2 protected18 scheduler contract

Status: controlling Python policy contract, frozen on 2026-07-30.

Policy ID: `rtl-unified-t6b2-protected18-v4`.

The normative Python entry point is `scheduler_rtl_unified_policy.py`. It
keeps the M-outer four-stage execution and explicit-DMA semantics. It does not
contain the separate N-outer work and it does not modify `Scheduler_hw`.

## 1. One-round decision flow

One invocation commits exactly one current-round action:

1. read the C2/C3 snapshots, maintained aggregate counters, and the
   `T0..T5,B0..B1` descriptor window;
2. enumerate the protected base profiles and evaluate each through the
   unchanged four-stage timing engine;
3. retain `base_best`;
4. enumerate the fixed recovery profiles, accepting the first legal
   event-aligned start for each physical profile;
5. reduce recovery profiles by logical family with the current-round finish
   key;
6. continue the global pairwise fold from `base_best`, evaluating only the
   recovery-family winners;
7. accept a recovery winner only if its primary bound improves by at least
   one tick; otherwise commit `base_best`;
8. update the cluster snapshots, counters, and visible window and start the
   next round.

There is one committed path. There is no JSON/ROM lookup, distribution mode
bit, beam search, child-round expansion, rollout, SIM1, floating-point
coefficient, or standalone S4PF action.

## 2. State and visible descriptors

Remaining experts are ordered by descending token count with deterministic
expert-ID tie-breaking. The visible descriptor set is:

- `T0..T5`: the hottest six remaining experts;
- `B0`: the coldest remaining expert;
- `B1`: the second-coldest remaining expert.

Head/bottom aliases are deduplicated. The scheduler remains a slave and does
not fetch descriptors. Software owns the full ordered stream and refills the
bounded window.

The bounded scorer uses the following maintained scalar state:

- remaining expert count;
- remaining token sum;
- remaining odd-token count;
- total best serial work;
- counts of one-, two-, three-, and four-M2-block experts;
- monotone parent path bound.

The Python model recomputes these values as an oracle. RTL initializes them
once and subtracts the selected expert contribution after every commit.

For RTL, repurpose the current scaled `total_serial_work` register as the
unscaled `remaining_block_sum = sum(ceil(ntok/2))`; derive `best_work` with a
shift plus add. Repurpose the unused-in-v4 `total_parallel_work` register as
the monotone parent path bound. This keeps both quantities without adding
either register or implementing division by three.

## 3. Candidate representation

A candidate has a logical part and a physical profile.

The logical part selects:

- mode: `SYNC`, `ONE_IDLE`, or `TERMINAL`;
- family: `SINGLE`, `PAIR`, or `SPLIT`;
- visible selectors such as `T0`, `T1`, or `B0`;
- split rule, when applicable.

The physical profile fixes:

- C2/C3 binding;
- S1 and S3 shapes;
- IDMA/XDMA/BOTH binding;
- cache residency;
- S2PF realization.

The profile tuples in Python are compile-time decode descriptions. They do
not imply runtime-programmable storage.

### 3.1 Protected base bank

The former fixed13-v2 base remains protected. It contains:

- `ONE_IDLE`: `B0`, `T0`, and `T3` singles;
- `SYNC`: hot+cold, adjacent-hot, and middle-adjacent pairs;
- `TERMINAL`: C2 single, C3 single, and balanced split;
- the associated cache-aware and S2PF physical variants.

Base profiles use the existing bounded-release behavior.

### 3.2 Recovery bank

The v4 recovery bank contains 30 fixed source profiles:

- 28 `T0 SINGLE` profiles: 17 `ONE_IDLE`, eight `SYNC`, and three
  `TERMINAL`;
- one `SYNC T0 HALF-SPLIT` profile;
- one cached `SYNC PAIR(T0,T1)` profile.

Mode, cluster, cache, lane, and descriptor legality filter this source set.
Across all 41,921 validation cases and 478,477 scheduling rounds, the observed
bounds are:

| Quantity | Maximum |
|---|---:|
| protected base candidates | 12 |
| physical base plus recovery candidates | 18 |
| recovery-family winners | 3 |
| global scorer stream | 15 |

The RTL hard limit is therefore 18 physical timing evaluations per round. A
5-bit counter is sufficient. The timing engine is sequentially reused; there
must not be 18 parallel copies.

## 4. Recovery hierarchy

### 4.1 First legal start

For each recovery profile, event-aligned starts are checked in ascending
release order. The first legal start is accepted. This replaces v3's
per-profile earliest-finish comparison and maps directly to a priority-first
FSM.

This change was checked on all 41,921 cases together with the final profile
bank: it introduces no regression relative to v3.

### 4.2 Family reducer

Profiles are reduced by logical group with:

```text
(max(child_c2_end, child_c3_end),
 child_c2_end + child_c3_end,
 latest_selected_start,
 fixed_profile_priority)
```

The numeric reducer stores only the first three fields. Exact ties retain the
earlier fixed profile ID; Python's deterministic action ordering is only the
oracle for that decode order.

In the final bank, only the recovery `SINGLE` group has multiple physical
profiles. `SPLIT` and `PAIR(T0,T1)` each have one profile and bypass the local
reducer.

Removing the family reducer changed 79 of 1,865 sampled cases and worsened two.
Reducing its key to only latest finish, latest finish plus sum, or latest
finish plus start also caused regressions. The reducer and all three numeric
fields are therefore retained.

## 5. Fixed scorer

Every candidate is first applied through the unchanged explicit-DMA
four-stage transition. Token conservation, cache residency, release times,
and same-lane DMA interval legality are checked before scoring.

The monotone primary bound is:

```text
child.f = max(parent.f,
              committed_finish,
              compute_capacity_LB,
              hottest_release_chain_LB,
              hottest_critical_chain_LB,
              mandatory_DMA_capacity_LB)
```

The fixed scorer is `HEAD5_HIST4_PAIRWISE_SCORER`. Its lexicographic base key
contains:

1. monotone `f`;
2. head5 plus four-bin cold-tail LPT estimate;
3. compute-capacity lower bound;
4. mandatory-DMA-capacity lower bound;
5. mode-specific current-round tie fields.

The head5/hist4 estimate uses five visible descriptors, four maintained block
histogram counters, and the existing aggregate serial work. It never scans
hidden descriptors.

The pairwise rules are fixed integer comparisons for:

- one-progress selection in `ONE_IDLE`;
- hot-expert preservation in `SYNC`;
- medium-load plateau handling;
- terminal plateau handling;
- large-slack filling.

These are parts of one fixed scorer, not runtime-selected scoring functions.

## 6. Protected arbitration

The conceptual result is:

```text
base_best  = fold(base candidates)
union_best = continue_fold(base_best, recovery-family winners)
```

The base candidates are not scored twice. RTL retains the complete current
winner summary and snapshots only the compact `base_token + base_primary_f`
before entering recovery. A second complete winner record is unnecessary.

If the final winner is a recovery candidate, it is committed only when:

```text
recovery_primary_f + 1_tick <= base_primary_f
```

Otherwise the compact base token is replayed and committed.

The one-tick margin is essential. Setting it to zero worsened 545 of 1,865
sampled cases; increasing it to two ticks worsened 103.

## 7. Causal simplification audit

The following tests used the same closed-loop scheduler rather than static
candidate-frequency counts.

### 7.1 Removed safely

| Change | Full-set result relative to protected24-v3 |
|---|---:|
| recovery source profiles `42 -> 30` | 5 better / 41,916 equal / 0 worse |
| physical candidate peak `22 -> 18` | included above |
| first-legal start per recovery profile | no additional difference |
| fold base once, then continue with recovery | exact structural equivalence |
| string/action tie field in numeric reducer | replaced by fixed profile order |

The full paired comparison is:

| Suite | Better / equal / worse for v4 versus v3 | Aggregate change |
|---|---:|---:|
| certified OLMoE | 0 / 65 / 0 | 0 ticks |
| coverage30k | 4 / 29,924 / 0 | -125 ticks |
| post-freeze | 1 / 11,927 / 0 | -32 ticks |

### 7.2 Tested and retained

Results below are from the 1,865-case stratified ablation set.

| Removed/simplified logic | Worse cases |
|---|---:|
| one-progress gate | 66 |
| sync-hot gate | 13 |
| plateau gate | 18 |
| tail-plateau gate | 1 |
| slack-fill gate | 6 |
| all pairwise gates | 78 |
| family reducer | 2 |
| four-bin histogram, replaced by head5 aggregate | 100 |
| compute field removed from global key | 26 |
| DMA field removed from global key | 78 |
| compute and DMA fields both removed | 104 |

These blocks cannot be called redundant. They remain in v4.

## 8. Frozen validation

Authoritative outputs are:

- `results/policy_search/scheduler_rtl_unified_65_v4.json`;
- `results/policy_search/scheduler_rtl_unified_30k_v4.json`;
- `results/policy_search/scheduler_rtl_unified_postfreeze_v4.json`.

Each result records source/input SHA-256 hashes. The proof65 result also stores
and regenerates every selected slot and action.

| Suite/bucket | Cases | Better / equal / worse vs adaptive | Aggregate delta | Max physical candidates |
|---|---:|---:|---:|---:|
| certified OLMoE | 65 | 64 / 1 / 0 | -15.5871% | 12 |
| coverage30k overall | 29,928 | 6,960 / 18,348 / 4,620 | -0.4362% | 18 |
| coverage30k E8 | 9,976 | 1,009 / 7,706 / 1,261 | +0.3378% | 18 |
| coverage30k E32 | 9,976 | 2,542 / 5,706 / 1,728 | -0.2654% | 18 |
| coverage30k E64 | 9,976 | 3,409 / 4,936 / 1,631 | -1.2845% | 18 |
| coverage30k strict OLMoE | 307 | 241 / 28 / 38 | -6.3401% | 17 |
| post-freeze overall | 11,928 | 2,558 / 7,618 / 1,752 | -0.3835% | 18 |
| post-freeze E8 | 3,976 | 359 / 3,131 / 486 | +0.3810% | 18 |
| post-freeze E64 | 3,976 | 1,255 / 2,107 / 614 | -1.1935% | 18 |
| post-freeze strict OLMoE | 88 | 68 / 8 / 12 | -6.3191% | 18 |

Negative aggregate delta is an improvement. The E8 aggregate regression
remains a measured policy boundary and is not hidden.

Relative to fixed13-v2, v4 gives:

- coverage30k: 4,093 better, 25,090 equal, 745 worse, and 183,144 ticks less;
- post-freeze: 1,661 better, 9,987 equal, 280 worse, and 67,021 ticks less.

The 29,928 and post-freeze sets do not have per-case optimum certificates.
Their results are paired policy comparisons. Only the 65-case set proves
`LB=UB`, and v4 reaches all 65 targets.

## 9. Reproduction

```bash
cd /esat/studscratch/r1015673/Thesis/Idea_Model

python3 -m py_compile \
  four_stage_scheduler.py \
  evaluate_olmoe_fixed_token_banks.py \
  scheduler_rtl_unified_policy.py \
  verify_scheduler_rtl_unified_policy.py

python3 verify_scheduler_rtl_unified_policy.py \
  --suite proof65 --workers 24 --checkpoint-every 20

python3 verify_scheduler_rtl_unified_policy.py \
  --suite coverage30k --workers 24 --checkpoint-every 1000

python3 verify_scheduler_rtl_unified_policy.py \
  --suite postfreeze --workers 24 --checkpoint-every 1000
```

The separate RTL handoff and resource acceptance checklist is
`RTL_PROTECTED18_V4_MODIFICATION_CHECKLIST.md`.
