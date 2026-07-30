# Top6+bottom2 joint scheduler implementation contract

## 1. Frozen v4 decision

The final Python policy ID is
`rtl-adaptive-t6b2-protected-b0-certified-fixed14-union-v4`.

The observable remaining-expert window is fixed as:

- `T0..T5`: the six hottest remaining experts;
- `B0..B1`: the two coldest remaining experts, with `B0` the coldest;
- duplicate identities are suppressed when fewer than eight experts remain.

The execution model remains the existing M-outer four-stage model. One policy
invocation selects one current-round action. The implementation does not run
beam search, recursively generate a child round, execute SIM1, scan all
remaining descriptors, or use a trained model.

Every mode preserves the existing sequential RTL dataflow:

1. read the current `T0..T5,B0,B1` state;
2. emit one fixed candidate at a time;
3. calculate the exact S1/S2/S3/S4 and DMA timing for that candidate;
4. calculate its integer score;
5. update one best-candidate register;
6. commit one winner and update the state/window.

Authoritative files are:

- final entry point: `scheduler_rtl_adaptive_olmoe_policy.py`;
- adaptive timing and protected selection: `scheduler_rtl_adaptive_prefetch_policy.py`;
- fixed RTL-style candidate generator: `scheduler_hw_fixed_policy.py`;
- bounded OLMoE scorer/control: `scheduler_olmoe_bounded_policy.py`;
- fixed14 OLMoE token ROM:
  `results/policy_search/olmoe_t5b1_hist4_bounded14_token_bank_v1.json`;
- 65-case audit:
  `results/policy_search/scheduler_adaptive_t6b2_joint_union_65_v4.json`;
- 29,928-case paired audit:
  `results/policy_search/scheduler_adaptive_t6b2_joint_union_30k_v4.json`;
- first-divergence causal audit:
  `results/policy_search/scheduler_adaptive_t6b2_first_divergence_30k_v1.json`;
- post-freeze 11,928-case audit:
  `results/policy_search/scheduler_adaptive_t6b2_postfreeze_11928_v4.json`;
- post-freeze generation manifest:
  `results/policy_search/scheduler_adaptive_t6b2_postfreeze_inputs_v4.json`.

The ROM filename contains `t5b1` because it records the minimum selector set
of that earlier ablation. The hardware interface is uniformly `top6+bottom2`;
each mode may leave a visible selector unused.

## 2. Why a larger window did not initially improve every case

Candidate-space monotonicity and closed-loop policy monotonicity are different
claims.

If the old candidate set is a subset of the new set and every candidate is
ranked by its exact final cost, the best reachable cost cannot increase. The
current RTL scorer is a finite continuation estimate. A new candidate can
therefore receive a better approximate score and still produce a worse final
schedule.

The complete 29,928-case ablations demonstrated this directly:

| T6+B2 control | Better | Equal | Worse |
|---|---:|---:|---:|
| globally use legacy control | 1,202 | 22,955 | 5,771 |
| globally rescore with continuation | 3,009 | 25,339 | 1,580 |
| preserve old winner, SYNC addition only | 191 | 29,737 | 0 |
| preserve old winner, initial head-critical rule | 299 | 29,606 | 23 |
| final protected B0 rule | 295 | 29,633 | 0 |

All 23 losses of the initial head-critical rule occurred in `ONE_IDLE` states
with exactly three remaining experts. Twenty-two selected `B1@release0`; one
selected `B0@release1`. In the same audit, protected `B0@release0` produced 104
improvements and no loss, while protected `SYNC T0+B0` produced 191
improvements and no loss. The final rule therefore uses fixed candidate-valid
bits, not token-count thresholds fitted to individual distributions.

## 3. Runtime mode selection

Mode selection is fixed at initialization and does not change between rounds.

### 3.1 Certified OLMoE mode

Use the fixed14/head5-hist4 policy when all conditions hold:

- total configured experts is 64;
- both initial cache IDs are invalid;
- total routed assignments is 140;
- active expert count is in `[29,57]`;
- experts with at most two assignments, including zero-load experts, is in
  `[40,49]`;
- `T0.ntok <= 34`;
- every expert below the first five has at most seven assignments;
- the head5+hist4 scorer contract is valid.

These comparisons form the inclusive feature envelope of the frozen 65-case
proof set. They are not a distribution LUT or trained classifier. The
global-optimality claim applies to the audited 65 cases only.

This mode uses the fixed14 candidate bank, head5+hist4 scorer, bounded local
lowering, and explicit `S4PF=OFF` profiles. It sends at most six candidates to
the bounded scorer on the 65 audited cases.

### 3.2 Generic protected T6+B2 mode

Every non-certified input uses the protected path. There is no longer an
initial `T0<=4` gate and no bit-exact fallback region.

At every round the policy first reproduces the old adaptive winner exactly.
The final generic candidate bank then exposes at most one additional candidate:

| Current mode | Additional candidate | Physical path |
|---|---|---|
| `SYNC` | pair `T0+B0` | existing adaptive pair/shape/S2PF/DMA lowering |
| `ONE_IDLE` | single `B0` at release point 0 | existing C/C single lowering and eager S2PF/S4PF rules |

`B1` and later release points were retained during ablation, but the final
generic valid mask rejects them unconditionally. They are therefore removed
from `generate_top6_bottom2_protected_successors` and consume no final generic
scoring slots. `B1` remains visible because the certified fixed14 mode uses it.

The additional candidate replaces the old winner only when its fixed
acceptance rule succeeds. Otherwise the exact old transition is committed.

## 4. Generic score and acceptance contract

The existing integer continuation function is:

```text
continuation(child) =
  min(aggregate_greedy(child),
      LPT(first four remaining experts) + balanced aggregate tail)
```

The first four visible jobs are placed in descending token order onto the
currently earlier cluster. Work below the first four is represented by one
integer sum and divided between the two loads. This calculation does not
generate another action.

For `SYNC T0+B0`, accept the additional candidate only if:

```text
continuation(old_winner) - continuation(added) >= 1 tick
```

For `ONE_IDLE B0@release0`, all conditions must hold:

```text
max(added_child.c2.task_end, added_child.c3.task_end)
    <= current_busy_cluster.task_end

min(added_child.c2.task_end, added_child.c3.task_end)
    + best_task(added_child.remaining.T0)
    >= continuation(added_child)

continuation(old_winner) - continuation(added_child) >= 1 tick
```

The first condition prevents the cold fill from extending the existing busy
interval. The second prevents postponing the remaining hot head when that head
lies on the estimated critical completion chain. The third requires a strict
one-tick continuation advantage.

Tie-breaks never allow an added candidate to replace the old winner. This is
the source of the finite-set zero-regression property; it is not a proof over
all possible distributions.

## 5. Certified fixed14 candidate bank

Notation is `S1/S3 ; DMA-S1/DMA-S3 ; S2PF`. `-` means that the cluster is idle
for that entry.

| ID | Mode | Selector | C2 profile | C3 profile |
|---:|---|---|---|---|
| 0 | `ONE_IDLE` | `B0` | - | `C/C; BOTH/BOTH; OFF` |
| 1 | `ONE_IDLE` | `B0` | `C/C; BOTH/BOTH; OFF` | - |
| 2 | `ONE_IDLE` | `T0` | `B/B; BOTH/OFF; BOTH` | - |
| 3 | `ONE_IDLE` | `T0` | - | `B/B; BOTH/OFF; BOTH` |
| 4 | `ONE_IDLE` | `T3` | `B/B; BOTH/OFF; BOTH` | - |
| 5 | `ONE_IDLE` | `T3` | - | `B/B; BOTH/OFF; BOTH` |
| 6 | `SYNC` | `B0,T0` | `A/B; IDMA/OFF; XDMA` | `B/B; XDMA/IDMA; OFF` |
| 7 | `SYNC` | `T0,T1` | `B/B; IDMA/IDMA; OFF` | `B/B; XDMA/XDMA; OFF` |
| 8 | `SYNC` | `T0,T1` | `B/B; IDMA/OFF; IDMA` | `B/B; XDMA/IDMA; OFF` |
| 9 | `SYNC` | `T0,T4` | `A/B; IDMA/OFF; XDMA` | `B/B; XDMA/IDMA; OFF` |
| 10 | `SYNC` | `T1,T2` | `B/B; IDMA/OFF; IDMA` | `B/B; XDMA/OFF; XDMA` |
| 11 | `SYNC` | `T2,T3` | `B/B; IDMA/OFF; IDMA` | `B/B; XDMA/OFF; XDMA` |
| 12 | `TERMINAL` | `T0` | `C/C; BOTH/BOTH; OFF` | - |
| 13 | `TERMINAL` | `T0` | - | `C/C; BOTH/BOTH; OFF` |

Shape A/B/C are M8/bw32, M4/bw64, and M2/bw128. The bank contains no
standalone prefetch action, S4PF action, WAIT-PAIR, SIM1, runtime rank loop, or
dynamic candidate expansion.

The certified scorer maintains `remaining_count`, `remaining_token_sum`,
`remaining_odd_count`, `remaining_shape_c_block_sum`, four histogram counters,
and the existing pathmax/lower-bound state. It places visible `T0..T4` by LPT,
then drains histogram bins 4,3,2,1. The fixed pairwise conditions use only the
current mode, visible window, maintained counters, child score fields, and
integer thresholds.

## 6. RTL impact

Required common state/interface changes are:

- maintain `T0..T5,B0,B1` descriptors;
- suppress duplicate selectors in short tails;
- select and retain one initialization mode bit;
- add selector encodings for `B0/B1`.

Required generic-control changes are:

- one fixed mode-specific candidate-valid slot;
- one register for the old winner's continuation score;
- comparisons for busy-slack fit, head-finish, and one-tick advantage;
- reuse the existing continuation arithmetic in `ONE_IDLE` states.

The generic candidate bank is structurally the old bank plus at most one
candidate. The observed maximum materialized count is 13 on both the 29,928
and post-freeze 11,928 sets. With a sequential scorer, the worst scheduling
latency increment is at most one candidate-scoring iteration per decision.
No parallel evaluator, second reducer, second commit path, beam queue,
multiplier, learned coefficient, or new timing model is required.

The certified mode additionally requires the fixed14 physical profile ROM,
four small histogram counters, and its fixed comparison control. The low-level
union generator materializes at most 15 endpoint-distinct transitions in the
65-case audit; the bounded certified scorer sees at most six.

Area, Fmax, and cycle-accurate RTL latency have not yet been synthesized. The
complexity statements above are structural bounds from the frozen Python
model, not post-synthesis measurements.

The scheduler remains a slave. Software or the existing upstream controller
must refill the `top6+bottom2` window. The Python policy does not define or
validate that refill protocol.

## 7. Verification evidence

### 7.1 Certified 65-case OLMoE set

| Metric | Result |
|---|---:|
| certified reference with `LB=UB` | 65/65 |
| final policy reaches certified optimum | 65/65 |
| low-level best-history coverage | 65/65 |
| better / equal / worse than old adaptive | 64 / 1 / 0 |
| old adaptive optimum cases | 1/65 |
| old adaptive cumulative gap | 1,427 ticks |
| final cumulative gap | 0 ticks |
| maximum certified scorer candidates | 6 |
| maximum low-level union transitions | 15 |

The audit checks both final closed-loop makespan and replay of the certified
best history through the RTL-style generator. A matching makespan alone is not
used as candidate-coverage proof.

### 7.2 Original complete 29,928 paired set

| Partition label | Cases | Better | Equal | Worse | Aggregate delta |
|---|---:|---:|---:|---:|---:|
| discovery | 17,959 | 188 | 17,771 | 0 | -0.016775% |
| validation | 5,987 | 54 | 5,933 | 0 | -0.014313% |
| blind_test | 5,982 | 53 | 5,929 | 0 | -0.012380% |
| total | 29,928 | 295 | 29,633 | 0 | -0.015406% |

The final rule was derived after examining affected cases across this dataset.
The `blind_test` label must therefore not be presented as an untouched blind
result for v4. This audit is a complete same-input regression test.

The dataset contains no case selected by the 140-assignment certified gate;
all 29,928 cases exercise the generic protected path.

The separate first-divergence audit forces each selected child to finish under
the old policy. It confirms that all 295 final first actions are beneficial,
with 191 `SYNC T0+B0` actions and 104 `ONE_IDLE B0@release0` actions; no final
first action has positive rollout delta. It also reproduces all 23 rejected
head-critical losses and their exact selector classification.

### 7.3 Post-freeze independent set

After source and the 29,928 result were frozen, a new coverage-balanced set was
generated once with seed `20260730` and bound to the frozen result SHA-256. No
policy parameter was changed after observing it.

| Expert count | Cases | Better | Equal | Worse | Aggregate delta |
|---|---:|---:|---:|---:|---:|
| 8 | 3,976 | 31 | 3,945 | 0 | -0.014073% |
| 32 | 3,976 | 56 | 3,920 | 0 | -0.021899% |
| 64 | 3,976 | 54 | 3,922 | 0 | -0.021250% |
| total | 11,928 | 141 | 11,787 | 0 | -0.019200% |

This is the untouched post-freeze validation of the generic rule. It remains a
finite constrained-random coverage set, not a measured router probability
distribution and not a mathematical proof for all inputs.

## 8. Reproduction

```bash
cd /esat/studscratch/r1015673/Thesis/Idea_Model

python3 -m py_compile \
  scheduler_hw_fixed_policy.py \
  scheduler_rtl_adaptive_prefetch_policy.py \
  scheduler_rtl_adaptive_olmoe_policy.py \
  audit_scheduler_t6b2_certified_union.py \
  compare_scheduler_adaptive_olmoe_30k.py

python3 audit_scheduler_t6b2_certified_union.py

python3 analyze_t6b2_protected_first_divergence.py --workers 24

python3 compare_scheduler_adaptive_olmoe_30k.py \
  --workers 24 \
  --checkpoint-every 1000 \
  --progress-every 1000

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

python3 compare_scheduler_adaptive_olmoe_30k.py \
  --input /tmp/scheduler_t6b2_postfreeze_v4/scheduler_t6b2_postfreeze_E8.json \
  --input /tmp/scheduler_t6b2_postfreeze_v4/scheduler_t6b2_postfreeze_E32.json \
  --input /tmp/scheduler_t6b2_postfreeze_v4/scheduler_t6b2_postfreeze_E64.json \
  --expected-cases 11928 \
  --workers 24 \
  --out results/policy_search/scheduler_adaptive_t6b2_postfreeze_11928_v4.json
```

The JSON audits record input and source SHA-256 values. Any edit to a hashed
source invalidates the corresponding manifest and requires the audits to be
rerun.
