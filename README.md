# Scheduler algorithm workspace

This directory contains the frozen R4+bottom2/LPT baseline, the reference
assets used to derive it, and the active P5 R8/K32 future-ranker derivation.

Retained code:

- `four_stage_scheduler.py`: complete four-stage reference search model.
- `SCHEDULER_POLICY_SPEC.md`: locked policy-derivation and acceptance contract.
- `analyze_scheduler_candidates.py`: deterministic reference-history replay and candidate census.
- `derive_scheduler_policy.py`: bounded-candidate and forced-continuation audits under the design contract.
- `scheduler_policy_golden.py`: deterministic runtime golden model for the frozen policy.
- `RTL_POLICY_CONTRACT.md`: state, candidate, scorer and sequencing contract for RTL.
- `generate_scheduler_strategy_coverage.py`: deterministic 30K distribution generator.
- `run_four_stage_reference.py`: reference-search runner and action serializer.
- `verify_scheduler_lower_bounds.py`: lower-bound checks.
- `eval_c_mirror_v2.py`: current C scheduler mirror.
- `eval_hw_mirror_s2pf_lite.py`: current hardware-oriented baseline mirror.
- `evaluate_scheduler_baselines.py`: baseline-versus-reference evaluator.

Retained data:

- `scheduler_strategy_coverage_E{8,32,64}.json`: 30K input distributions.
- `results/final_reference/scheduler_reference_E{8,32,64}.json`: full reference results and action histories.
- `results/final_reference/scheduler_reference_E{8,32,64}_compact.json`: compact result views.
- `results/final_reference/scheduler_reference_manifest.json`: reference provenance and summary.

Frozen evaluated baseline:

- candidate generator: `direct-slot-conditional-cache-v8`;
- expert pool: top 4, bottom 2 and concrete resident/prefetched IDs;
- maximum candidates per decision: 32;
- score: exact child transition followed by integer two-cluster LPT;
- tie-break: remaining count, later cluster completion, candidate index.

`derive_scheduler_policy.py` retains alternative scorer code only as derivation
evidence. The selected runtime path is owned by `scheduler_policy_golden.py`
and does not read a fitted model or coefficient file.

Evaluation status:

- discovery: 14,118 proven cases, mean ratio 1.010770, p95 1.032680;
- validation: 4,739 proven cases, mean ratio 1.011394, p95 1.034483;
- one-time blind test: 4,732 proven cases, mean ratio 1.010404, p95 1.032258;
- blind current-hardware baseline: mean ratio 1.023345, p95 1.117647.

The blind test was opened only after implementation hashes were recorded. The
policy passed without post-blind changes and was accepted as the v1 RTL target
before the active P5 R8/window revision was opened.

Active P5 work does not modify that baseline report or golden model.  It uses
an ordered top8/K32 generator and a mode-selected shift/add ranker trained from
interval-valued forced continuations.  Its formal dataset is
`results/policy_search/r8_future_rank_dataset_v1.jsonl`; no P5 model becomes an
RTL target before the gates in `SCHEDULER_POLICY_SPEC.md` pass.
