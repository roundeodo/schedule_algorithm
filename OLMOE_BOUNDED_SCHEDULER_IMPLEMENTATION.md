# Unified top6+bottom2 RTL scheduler contract

Status: controlling Python-to-RTL contract, frozen on 2026-07-30.

Policy ID: `rtl-unified-t6b2-fixed13-v2`.

The normative Python entry point is `scheduler_rtl_unified_policy.py`.  The
policy keeps the existing M-outer four-stage transition semantics and changes
only the observable window, candidate cases, and continuation comparator.  It
does not implement the separate N-outer experiment.

## 1. Design boundary

One invocation makes one current-round decision:

1. read the C2/C3 four-stage snapshots and `T0..T5,B0..B1` window;
2. decode legal candidate cases for the current mode;
3. lower every candidate to an exact S1/S2/S3/S4 and explicit-DMA action;
4. apply the exact child transition;
5. compare candidates with one fixed integer comparison policy;
6. commit one winner, update counters/window, and repeat in the next round.

The hard limit is 13 concrete candidate slots per state.  A sequential RTL
implementation needs one current-candidate register and one best-candidate
register; it does not need 13 copies of the timing/scoring datapath.

The final path contains none of the following:

- runtime JSON/ROM candidate tables;
- an initial-distribution classifier or OLMoE-only mode bit;
- a legacy-policy fallback path;
- beam search, rollout, child-round expansion, or SIM1;
- standalone S4 prefetch candidates;
- training coefficients or experimental scorer switches.

The Python tuple constants are source-level encodings of combinational `case`
branches.  They do not imply a hardware ROM and require zero ROM data bits.

## 2. Runtime state

### 2.1 Descriptor window

Remaining experts stay sorted by descending token count and then deterministic
expert-ID order.  The observable window is:

- `T0..T5`: first six remaining descriptors;
- `B0`: coldest remaining descriptor;
- `B1`: second-coldest remaining descriptor.

When the head and bottom overlap, duplicate expert IDs are suppressed before
candidate lowering.  Candidate slot order after legality filtering and child
deduplication is architectural: `candidate_slot = 0..candidate_count-1`.
`ScheduleStep.candidate_slot` records that index for RTL lockstep.

The scheduler is a slave and cannot fetch descriptors.  A deployment therefore
still needs a two-sided refill protocol.  The minimum compatible protocol keeps
monotone head and tail cursors, reports separate consumed head/bottom counts,
and suppresses duplicate physical indices when the cursors meet.  Software
does not need to mirror every current window identity, but it must supply the
correct next descriptors for each side.  This refill protocol is outside the
current Python policy and must be verified separately before claiming an
end-to-end RTL deployment.

### 2.2 Aggregate counters

The scorer does not scan unseen descriptors.  It uses counters initialized
once and decremented on every committed expert or split:

- `remaining_count`;
- `remaining_token_sum`;
- `remaining_odd_count`;
- `remaining_block_sum`, where `blocks = ceil(ntok/2)`;
- `remaining_best_work`;
- four histogram counts for one-, two-, three-, and four-block experts.

Any tail work above four blocks is represented by
`remaining_best_work - visible_head_work - histogram_work`.  It is balanced as
one aggregate scalar, so no hidden middle descriptor is read.

Widths must be derived from the configured `E_MAX`, `NTOK_MAX`, and
`TOTAL_TOKEN_MAX`; they must not silently saturate.  For example,
`remaining_count` needs `ceil(log2(E_MAX+1))` bits and each histogram counter
uses the same width.

## 3. Candidate decode cases

Notation below is `S1/S3 ; DMA-S1/DMA-S3 ; S2PF`.  `A`, `B`, and `C` are the
existing M8/bw32, M4/bw64, and M2/bw128 shapes.  `I`, `X`, and `BOTH` denote
iDMA, xDMA, and both DMA lanes.

Only cases belonging to the current mode are decoded.  Cluster-reflected
single cases share one logical branch in RTL even though the Python constants
contain C2 and C3 endpoint forms.

### 3.1 `ONE_IDLE`

The legal idle cluster receives one of three selector/profile families:

| Selector | Profile | Purpose |
|---|---|---|
| `B0` | `C/C ; BOTH/BOTH ; OFF` | fit one cold expert into available slack |
| `T0` | `B/B ; BOTH/OFF ; BOTH` | advance the hottest expert with S2PF |
| `T3` | `B/B ; BOTH/OFF ; BOTH` | preserve plateau/parity alternatives |

The `T0/T3` S3 weight is marked resident only when the S2PF interval completes
legally.  No eager S4PF is inserted.

### 3.2 `SYNC`

Six pair families are decoded:

| Family | C2 profile | C3 profile |
|---|---|---|
| `B0 + T0` | `A/B ; I/OFF ; X` | `B/B ; X/I ; OFF` |
| `T0 + T1`, no PF | `B/B ; I/I ; OFF` | `B/B ; X/X ; OFF` |
| `T0 + T1`, C2 PF | `B/B ; I/OFF ; I` | `B/B ; X/I ; OFF` |
| `T0 + T4` | `A/B ; I/OFF ; X` | `B/B ; X/I ; OFF` |
| `T1 + T2` | `B/B ; I/OFF ; I` | `B/B ; X/OFF ; X` |
| `T2 + T3` | `B/B ; I/OFF ; I` | `B/B ; X/OFF ; X` |

Selector aliases, illegal DMA overlaps, impossible cache flags, and identical
child states are removed.  Cache/residency observations can create distinct
legal physical realizations; every realization consumes a concrete slot.

### 3.3 `TERMINAL`

The remaining expert has three possible concrete cases:

- one C2 `C/C ; BOTH/BOTH` single;
- one C3 `C/C ; BOTH/BOTH` single;
- one balanced split using `B/B`, with floor/ceil token halves and dedicated
  C2-iDMA/C3-xDMA lanes.

The balanced split is an exact current-round action.  It is not SIM1 or
lookahead and is legal for odd token counts because the two halves still sum
to the original count.

## 4. Exact transition and score

Every candidate first passes the unchanged explicit-DMA four-stage legality
model.  In particular, selected token counts are conserved; starts cannot
precede cluster release; cache flags must match residency at the proposed
start; S2PF is legal only for the selected Down weight; and intervals using
the same DMA lane cannot overlap.

The child path bound is monotone:

```text
child.f = max(parent.f,
              committed_finish,
              compute_capacity_LB,
              hottest_release_chain_LB,
              hottest_critical_chain_LB,
              mandatory_DMA_capacity_LB)
```

The base integer key is formed from:

```text
(child.f,
 head5_hist4_LPT(child),
 compute_capacity_LB,
 mandatory_DMA_capacity_LB,
 mode_specific_current_round_fields,
 candidate_slot)
```

`head5_hist4_LPT` places `T0..T4` in descending order onto the currently
earlier cluster, then places histogram bins 4,3,2,1, and finally balances the
aggregate overflow work.  It estimates continuation work but never generates
a future action.

The comparator has one frozen deterministic implementation,
`HEAD5_HIST4_PAIRWISE_SCORER`.  Its local tie overrides use only the current
mode, `T0..T5,B0..B1`, the counters above, exact child endpoints, S2PF count,
and lower-bound fields.  The five override classes are:

- `ONE_IDLE` progress under a small remaining-work/odd-tail condition;
- hot-head preservation in `SYNC` when `T0 >= 2*T1` and the cold population is
  large;
- long plateau progress when cluster release skew is three ticks;
- short-tail plateau progress when release skew is six ticks;
- large-slack head fill for 8--16 remaining experts.

These are current-state tie rules, not an initial-distribution gate and not a
second scheduler.  Exact normative comparisons are in
`evaluate_olmoe_fixed_token_banks.py::select_practical_probe_candidate`; the
RTL implementation must reproduce them in source order because the relation
is a sequential pairwise reducer, not an unordered scalar sort.

All timing and score arithmetic is integer.  One tick is 11,264 cycles in the
current model.  No multiplier is required by the fixed thresholds: constants
such as 2, 3, 6, 11, 32, 84, 102, and 115 can be implemented with shifts,
adds, and constant comparisons.

## 5. Why there is no generic safety candidate

An earlier prototype exposed one apparent safety slot, but constructed that
slot by running the complete reference `gen_stage_actions` menu and selecting
its earliest-finishing SINGLE internally.  Although the resulting list never
exceeded 13 entries, the hidden micro-action search was not a truthful fixed
RTL candidate budget.  That prototype is rejected and is not present in the
controlling source.

Removing it preserves 65/65 certified OLMoE optima and the strict OLMoE gain,
but reduces performance on broad random/cache-heavy inputs.  This is an
intentional, measured complexity/performance choice rather than an omitted
optimization.  Reintroducing that behavior would require either counting its
shape/lane alternatives as real candidate slots or specifying and costing a
separate local selector.

## 6. Verification and claims

Authoritative validation outputs are:

- `results/policy_search/scheduler_rtl_unified_65_v1.json`;
- `results/policy_search/scheduler_rtl_unified_30k_v1.json`;
- `results/policy_search/scheduler_rtl_unified_postfreeze_v1.json`.

The verifier records input and source SHA-256 hashes, supports checkpoint
resume, and stores a complete selected-slot/action trace for the 65 proof
cases.  Every final history is independently replayed through the explicit-DMA
reference.

The 65-case claim has two parts:

1. every closed-loop action came from the bounded candidate stream, so the
   stream contains an optimal path for all 65 cases;
2. the frozen comparator selected that path and reached the certified
   `LB=UB` makespan for all 65 cases.

Thus both candidate loss and scoring loss are zero on the certified set.  The
selected optimum need not be action-for-action identical to an older stored
certificate; equality is established by legal replay and the certified lower
bound.

The 29,928 and post-freeze sets do not contain per-case four-stage optimum
certificates.  Their deltas against the adaptive policy are therefore paired
policy comparisons, not a decomposition into candidate and scoring regret.
The final report must keep this distinction.

Final same-input results versus the adaptive Python baseline are:

| Suite/bucket | Cases | Better / equal / worse | Aggregate delta | Max candidates |
|---|---:|---:|---:|---:|
| certified OLMoE | 65 | 64 / 1 / 0 | -15.5871% | 6 |
| coverage30k overall | 29,928 | 5,867 / 17,758 / 6,303 | +2.9428% | 12 |
| coverage30k strict OLMoE | 307 | 241 / 11 / 55 | -5.9884% | 10 |
| post-freeze overall | 11,928 | 2,088 / 7,369 / 2,471 | +2.8175% | 12 |
| post-freeze strict OLMoE | 88 | 68 / 1 / 19 | -5.4978% | 10 |

Negative delta is an improvement.  The conclusion is therefore workload
specific: this fixed bank is strong on the intended E64 hot-head/cold-tail
class and reaches every certified target, but it is not a drop-in dominance
replacement for the adaptive policy on broad random/cache-heavy traffic.  An
RTL or thesis claim must state that boundary explicitly.

## 7. RTL cost statement

Structural additions relative to the existing four-stage timing engine are:

- storage/refill support for `top6+bottom2` physical descriptors;
- the aggregate counters and four histogram counters;
- combinational decoding for the mode-specific cases above;
- head5/hist4 continuation arithmetic and the frozen pairwise comparisons;
- a 4-bit candidate-slot counter, one best-score register, and one best-action
  register.

The hard slot budget is 13.  The final full validations observed at most 12;
the remaining code point is headroom, not a hidden action.  A sequential
implementation therefore needs at most 12 candidate-evaluation iterations on
the validated data and must assert if an unsupported state tries to emit more
than 13.

This policy has fewer bounded candidate slots than an unconstrained reference
search, but its scorer is more complex than the deployed adaptive heuristic.
No area, Fmax, or cycle count should be claimed until synthesis and RTL
lockstep are complete.

## 8. Required RTL verification order

1. Compare Python and RTL candidate count, slot ID, and serialized action for
   every round of all 65 traces.
2. Compare every candidate's exact child endpoints and DMA legality, not only
   the selected winner.
3. Compare score fields and pairwise winner updates slot by slot.
4. Replay the selected RTL history in the Python explicit-DMA checker.
5. Run the 29,928 and post-freeze suites with identical input/cache manifests.
6. Synthesize only after lockstep passes; report area, Fmax, decision latency,
   and the two-sided refill storage separately.

## 9. Reproduction

```bash
cd /esat/studscratch/r1015673/Thesis/Idea_Model

python3 -m py_compile \
  four_stage_scheduler.py \
  evaluate_olmoe_fixed_token_banks.py \
  scheduler_rtl_unified_policy.py \
  verify_scheduler_rtl_unified_policy.py

python3 verify_scheduler_rtl_unified_policy.py \
  --suite proof65 --workers 24 --checkpoint-every 65

python3 verify_scheduler_rtl_unified_policy.py \
  --suite coverage30k --workers 24 --checkpoint-every 1000

# Recreate the deterministic post-freeze inputs if /tmp was cleared.
python3 generate_scheduler_strategy_coverage.py \
  --seed 20260730 \
  --cases-per-e 4000 \
  --directed-cases-per-e 800 \
  --corner-cases-per-e 24 \
  --e-total 8 32 64 \
  --out-dir /tmp/scheduler_t6b2_postfreeze_v4 \
  --prefix scheduler_t6b2_postfreeze \
  --split-label postfreeze \
  --policy-freeze-manifest \
    results/policy_search/scheduler_adaptive_t6b2_joint_union_30k_v4.json

python3 verify_scheduler_rtl_unified_policy.py \
  --suite postfreeze --workers 24 --checkpoint-every 1000
```

Completion of a process alone is not acceptance.  The result must have
`complete=true`, the expected case count, matching source/input hashes,
`candidate_count_max <= 13`, explicit-DMA replay success, and 65/65 certified
optima.
