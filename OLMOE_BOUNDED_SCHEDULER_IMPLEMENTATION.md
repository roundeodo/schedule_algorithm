# Unified top6+bottom2 RTL scheduler contract

Status: controlling Python-to-RTL contract, frozen on 2026-07-30.

Policy ID: `rtl-unified-t6b2-protected24-v3`.

The normative Python entry point is `scheduler_rtl_unified_policy.py`. It
keeps the existing M-outer four-stage and explicit-DMA semantics. It does not
contain the separate N-outer work.

## 1. Final decision flow

One invocation makes exactly one current-round decision:

1. read the C2/C3 snapshots and the `T0..T5,B0..B1` window;
2. materialize the protected base profiles at bounded release points;
3. materialize the fixed recovery profiles and select the earliest-finishing
   event-aligned start within each profile;
4. reduce recovery profiles locally by logical family;
5. score the base stream and the base-plus-recovery stream with the same
   bounded integer scorer;
6. accept a recovery winner only when its primary score improves by at least
   one tick;
7. commit one exact action and update state for the next round.

There is no JSON/ROM lookup, initial-distribution classifier, OLMoE mode bit,
beam search, rollout, child-round expansion, SIM1, or standalone S4PF action.
The Python token tuples are source-level descriptions of hard-wired decode
branches. They require no runtime-programmable candidate storage.

The asserted hard limit is 24 concrete physical candidates in a round. The
formal 29,928-case run observed 22; the post-freeze run observed 21.

## 2. Runtime-visible state

Remaining experts are sorted by descending token count with deterministic
expert-ID tie-breaking. The descriptor window is:

- `T0..T5`: hottest six remaining experts;
- `B0`: coldest remaining expert;
- `B1`: second-coldest remaining expert.

Overlapping head/bottom aliases are deduplicated. The scheduler is a slave and
cannot fetch descriptors. Deployment therefore still needs the previously
specified two-sided refill mechanism with monotone head and tail cursors.

The scorer uses maintained scalar state rather than hidden descriptor scans:

- remaining expert count and token sum;
- odd-token count;
- total M2-block count and best-work sum;
- four counters for one-, two-, three-, and four-block experts;
- monotone path lower bound.

These counters are initialized once and decremented after each committed
expert or split. Widths must be derived from `E_MAX`, `NTOK_MAX`, and
`TOTAL_TOKEN_MAX` without silent saturation.

## 3. Candidate organization

### 3.1 Protected base bank

The base bank is the former unified-v2 bank and remains a strict subset of the
new scorer stream. It retains:

- `ONE_IDLE`: cold `B0`, hot `T0`, and plateau `T3` singles;
- `SYNC`: hot+cold, adjacent hot, and middle adjacent pairs;
- `TERMINAL`: C2 single, C3 single, and balanced split.

Base candidates use bounded release points. Cache-valid overlays mask the
corresponding DMA transfer without changing the logical candidate.

### 3.2 Recovery bank

The recovery bank contains fixed physical profiles selected from the frozen
discovery split and then checked independently on validation, blind, and
post-freeze inputs. It restores candidate behavior removed by fixed13-v2:

- 36 `T0` SINGLE physical profiles covering A/B, B/B, B/C, C/B, and C/C,
  legal C2/C3 lane bindings, cache hits, and S2PF realizations;
- three `SYNC T0` HALF-SPLIT profiles;
- three `SYNC` PAIR profiles for `T0+T1` or `T2+T3`.

The 42 source profiles are filtered by mode, cluster, cache, shape, and lane
legality; only the legal subset materializes in a round. They are therefore
not 42 simultaneous RTL slots. Together with the base bank, the implementation
asserts a limit of 24; the largest stream seen in the two full validation sets
is 22.

For each fixed profile, the lowering logic considers only event-aligned start
timestamps already present in the four-stage state. It selects the realization
with the lexicographic key:

```text
(latest_child_end,
 sum_child_ends,
 latest_selected_start,
 deterministic_action_order)
```

The deterministic action-order field exists only to make Python ties stable;
RTL uses fixed decode priority. This local operation never looks at a child
round or final makespan.

After per-profile start selection, recovery candidates are locally reduced by
logical group: one SINGLE winner, one SPLIT winner, and one winner for each
PAIR selector tuple. The global scorer therefore receives the protected base
stream plus at most five recovery-family winners.

## 4. Score and protected arbitration

Every action is first applied through the unchanged explicit-DMA four-stage
transition. Token counts are conserved, cache flags must match residency,
starts cannot precede a legal release, and overlapping same-lane DMA intervals
are rejected.

The monotone child bound is:

```text
child.f = max(parent.f,
              committed_finish,
              compute_capacity_LB,
              hottest_release_chain_LB,
              hottest_critical_chain_LB,
              mandatory_DMA_capacity_LB)
```

The frozen scorer is `HEAD5_HIST4_PAIRWISE_SCORER`. Its base fields are the
path bound, head5/hist4 LPT continuation estimate, compute/DMA lower bounds,
mode-specific current-round fields, and deterministic slot order. Its
pairwise overrides use only the current mode, the visible window, maintained
counters, exact child endpoints, and S2PF count. No coefficient is trained at
runtime and no future action is generated.

Two global winners are retained:

- `base_best`: best action from the protected base bank;
- `union_best`: best action after adding recovery-family winners.

If `union_best` is a recovery action, it is accepted only when:

```text
union_best.primary_score + 1_tick <= base_best.primary_score
```

Otherwise `base_best` is committed. This is a bounded current-state guard, not
a distribution gate. One tick is 11,264 model cycles.

## 5. Why the hard limit is 24 rather than 13 or 32

The old adaptive trajectory audit over 29,928 cases showed that the adaptive
bank itself reaches 13 concrete candidates. Therefore 13 cannot simultaneously
preserve that behavior and hold new hot/cold profiles.

The frozen budget sweep gave:

| Retained recovery source profiles | Actual max candidates | Better / equal / worse vs adaptive | Aggregate delta |
|---:|---:|---:|---:|
| 16 SINGLE | 16 | 6,395 / 18,149 / 5,384 | -0.2149% |
| 24 SINGLE | 18 | 6,692 / 18,074 / 5,162 | -0.3452% |
| 36 SINGLE | 18 | 6,778 / 18,073 / 5,077 | -0.3778% |
| 48 SINGLE | 21 | 6,782 / 18,117 / 5,029 | -0.3801% |

Adding the three SPLIT and three PAIR profiles improved the final 36-profile
design to -0.4339% with a maximum of 22 candidates. Keeping more than three
PAIR profiles did not improve the pilot but increased the maximum toward 31.
Thus 24 is the smallest rounded hardware budget covering the selected design;
32 adds no measured benefit.

## 6. Frozen validation

Authoritative outputs are:

- `results/policy_search/scheduler_rtl_unified_65_v3.json`;
- `results/policy_search/scheduler_rtl_unified_30k_v3.json`;
- `results/policy_search/scheduler_rtl_unified_postfreeze_v3.json`.

The verifier records source/input SHA-256 hashes and independently replays
every selected history through the explicit-DMA reference. The 65-case output
also stores and regenerates every selected slot/action trace.

| Suite/bucket | Cases | Better / equal / worse vs adaptive | Aggregate delta | Max candidates |
|---|---:|---:|---:|---:|
| certified OLMoE | 65 | 64 / 1 / 0 | -15.5871% | 14 |
| coverage30k overall | 29,928 | 6,960 / 18,348 / 4,620 | -0.4339% | 22 |
| coverage30k E8 | 9,976 | 1,009 / 7,706 / 1,261 | +0.3398% | 22 |
| coverage30k E32 | 9,976 | 2,542 / 5,706 / 1,728 | -0.2627% | 21 |
| coverage30k E64 | 9,976 | 3,409 / 4,936 / 1,631 | -1.2824% | 21 |
| coverage30k strict OLMoE | 307 | 241 / 28 / 38 | -6.3401% | 19 |
| post-freeze overall | 11,928 | 2,558 / 7,618 / 1,752 | -0.3819% | 21 |
| post-freeze E8 | 3,976 | 359 / 3,131 / 486 | +0.3859% | 21 |
| post-freeze E64 | 3,976 | 1,255 / 2,107 / 614 | -1.1935% | 20 |
| post-freeze strict OLMoE | 88 | 68 / 8 / 12 | -6.3191% | 20 |

Negative delta is an improvement. The E8 regression is a measured remaining
boundary and must not be hidden by the aggregate result. The intended E64
hot-head/cold-tail class improves substantially, and all 65 certified cases
reach their `LB=UB` target.

The 29,928 and post-freeze sets do not have per-case optimum certificates.
Their results are paired policy comparisons, not proofs of optimum or exact
candidate/scoring-regret decomposition.

Relative to fixed13-v2, protected24-v3 improves 4,093 cases, equals 25,088,
and worsens 747, for an aggregate improvement of 183,019 ticks. Of the 4,620
remaining regressions versus adaptive, 4,256 were already regressions in
fixed13-v2. This supports the conclusion that recovery candidates repair most
of the replacement damage, while the residual difference is not solely caused
by the newly admitted profiles.

## 7. RTL cost boundary

Relative to the fixed13-v2 mirror, RTL needs:

- a 5-bit physical-candidate counter with an assertion at 24;
- hard-wired decode for the recovery physical profiles;
- local finish comparators for SINGLE, SPLIT, and up to three PAIR groups;
- one `base_best` score/action register and one `union_best` score/action
  register if evaluation is sequential;
- one subtract/compare condition for the 1-tick recovery margin.

The existing four-stage timing engine, explicit DMA legality checks,
head5/hist4 scorer, window, and aggregate counters are reused. No new search
FSM, descriptor RAM, model coefficient RAM, or candidate ROM is required.
Area, Fmax, and decision latency remain unclaimed until RTL lockstep and
synthesis are complete.

## 8. RTL verification order

1. Compare legal physical-candidate count and assert it never exceeds 24.
2. Compare every profile's selected start, exact child endpoints, and DMA
   legality.
3. Compare local recovery-family winners.
4. Compare `base_best`, `union_best`, primary-score margin, and final commit.
5. Replay the selected RTL history in the explicit-DMA Python checker.
6. Run the 65, 29,928, and post-freeze inputs with matching hashes.
7. Synthesize and report area, Fmax, decision latency, and refill storage
   separately.

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
