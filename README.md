# Scheduler algorithm workspace

The controlling RTL-oriented Python policy is
`scheduler_rtl_distilled_policy.py`
(`bounded-distilled-top5-bottom1`).  Its normative method contract and RTL
handoff checklist are `BOUNDED_DISTILLED_SCHEDULER.md` and
`RTL_BOUNDED_DISTILLED_SCHEDULER_CHECKLIST.md`.  It uses a top5+bottom1
descriptor window, 32 hard-wired physical profiles, at most six logical
candidates per round, one continuation comparator, and no base/recovery,
beam/SIM1, standalone S4PF, or runtime policy table.

The workflow below is retained as derivation history and proof provenance.  It
is not the current RTL implementation specification.

## Active workflow

1. `generate_olmoe_top2_projection_cases.py` generates the exact-70-token,
   64-expert, Top-2 suite.  The core is stratified by top1/mean hotness
   (`6--8x`, `8--12x`, `12--14x`), two/three/four local hotspots, active-expert
   count, and cold-expert count.
2. `run_olmoe_optimal_path_pass1.sh` runs unrestricted four-stage beam seeds to
   obtain replay-valid upper bounds.  This pass does not impose a hardware
   candidate window and does not call a time-limited result optimal.
3. `prove_top4_bottom2_directed.py` and
   `run_isolated_directed_proofs.py` perform selective branch-and-bound and
   history replay.  Only a closed lower/upper-bound gap is an optimality
   certificate.
4. After the target histories are certified, the generic directed-case tools
   evaluate candidate-window coverage, candidate actions, and scorer quality in
   that order.  Window and scorer conclusions from the deleted pre-v3 suites
   are not carried forward.
5. `run_olmoe_min2_supplement_pass1.sh` covers the separate supplement in
   which zero-load experts are allowed but every active expert has at least two
   tokens.  It is run after the active v3 base pass, not concurrently with it.

## Current inputs and results

- `results/policy_search/olmoe_top2_projection_cases_v3.json`: current target
  distributions.
- `results/policy_search/olmoe_top2_projection_v3_pass1_full_seed_w8_w16.*`:
  current resumable pass-1 output, log, and PID.
- `results/policy_search/.proof_fragments/olmoe_v3_pass1_full_seed_w8_w16/`:
  current per-case checkpoints.
- `results/policy_search/olmoe_top2_projection_v2_best_known.json`: temporary
  prior used only to seed matching v3 anchor cases.  Remove it after pass 1 has
  completed and the v3 result is self-contained.
- `results/policy_search/olmoe_top2_projection_min2_supplement_v1.json`: 22
  additional exact-total cases with minimum positive expert load equal to two.

## Core models and reusable tools

- `four_stage_scheduler.py`: complete explicit-DMA four-stage reference model.
- `scheduler_rtl_unified_policy.py`: controlling bounded RTL mirror.
- `verify_scheduler_rtl_unified_policy.py`: proof65/30K/post-freeze validator.
- `run_four_stage_reference.py`: reference runner and action serialization.
- `evaluate_top4_bottom2_directed.py`: directed case and lower-bound utilities.
- `scheduler_hw_fixed_policy.py`: current hardware-oriented policy state model.
- `scheduler_top4_bottom2_policy.py`: candidate policy used as a lowering hint;
  it is not an optimality oracle.
- `scheduler_rtl_adaptive_prefetch_policy.py`: adaptive-prefetch policy mirror.
- `evaluate_directed_window_grid.py`: candidate-window coverage evaluator.
- `evaluate_directed_candidate_score_ablation.py`: scorer ablation evaluator.
- `analyze_directed_case_classification.py`: certificate/history classifier.
- `merge_top4_bottom2_proofs.py`: proof-result merger.

## Retained baselines

The remaining large JSON files in `results/policy_search/` are intentional:

- current-HW and scheduler-strategy 30K comparisons;
- current-HW tail and candidate/scorer evidence;
- S2/S4 prefetch causal ablations;
- the final bounded-policy freeze and canonical-validation records.

`SCHEDULER_POLICY_SPEC.md` and `RTL_POLICY_CONTRACT.md` document historical
policies and prior RTL decisions.  They are retained as comparison evidence,
not as the specification of the active v3 search.

## Evidence rules

- A replay-valid schedule without a closed bound is an upper bound only.
- A timeout or expansion limit is not proof of optimality or infeasibility.
- Candidate-window sufficiency is tested only after a reference history is
  available and is separate from scorer accuracy.
- All policy comparisons use identical input distributions and preserve the
  current-HW and four-stage baselines.
