# Scheduler algorithm workspace

This directory contains one active RTL-oriented Python mirror and the reference
models required to validate it.  Historical search campaigns and superseded
policies live under `legacy/scheduler_algorithm/`; they are evidence, not
active implementation specifications.

## Active policy

- `scheduler_rtl_distilled_policy.py`: top-level round-by-round scheduler.
- `scheduler_rtl_distilled_profiles.py`: 32 hard-wired physical profiles and
  their logical-action grouping.
- `scheduler_rtl_distilled_types.py`: shared policy constants and fixed-profile
  types.
- `scheduler_rtl_distilled_lowering.py`: bounded selector resolution,
  residency, DMA legality and direct four-stage action lowering.
- `scheduler_rtl_distilled_scoring.py`: maintained counters, F/H/C/D bounds,
  regime predicates and the continuation comparator.
- `BOUNDED_DISTILLED_SCHEDULER.md`: normative algorithm and methodology.
- `RTL_BOUNDED_DISTILLED_SCHEDULER_CHECKLIST.md`: RTL implementation handoff.

The frozen policy identifier is `bounded-distilled-top5-bottom1`.  Each round
observes a top5+bottom1 descriptor window, emits at most 18 distinct physical
candidates from the 32 static profiles, locally reduces equivalent physical
profiles to at most six logical actions, scores every logical action with one
continuation comparator, and selects one global winner.  It has no
base/recovery arbitration, beam/SIM1, standalone S4 prefetch, or runtime policy
table.

The active policy modules do not import the historical experiment driver
`evaluate_olmoe_fixed_token_banks.py`.  That file remains only for archived
experiments and the frozen comparison baseline.

## Validation entry points

- `verify_scheduler_rtl_distilled_policy.py`: same-input proof65 and 30K
  validation with checkpoints.
- `verify_scheduler_rtl_distilled_showcase.py`: directed showcase traces.
- `ablate_scheduler_rtl_distilled_structure.py`: structural ablations.
- `four_stage_scheduler.py` and `run_four_stage_reference.py`: unrestricted
  four-stage reference semantics.

Frozen result records:

- `results/policy_search/bounded_top5_bottom1_certificate_validation.json`
  (65/65 certified cases match the optimal makespan).
- `results/policy_search/bounded_top5_bottom1_random_validation.json`
  (29,928 complete same-input random cases).
- `results/policy_search/olmoe_top2_projection_65_optimal_v1.json`
  (the 65-case optimality certificate set).

Quick validation commands:

```bash
python3 -m py_compile \
  scheduler_rtl_distilled_types.py \
  scheduler_rtl_distilled_lowering.py \
  scheduler_rtl_distilled_scoring.py \
  scheduler_rtl_distilled_policy.py \
  scheduler_rtl_distilled_profiles.py \
  verify_scheduler_rtl_distilled_policy.py

python3 verify_scheduler_rtl_distilled_policy.py \
  --suite proof65 \
  --out /tmp/bounded_top5_bottom1_proof65.json
```

## Supporting baselines

Some older-looking modules remain at repository root because the active
validators import them to reproduce the current-HW, adaptive, unified, and
four-stage same-input baselines.  They are support code, not alternative active
policies.  Do not move them without also removing those baseline comparisons.

## Historical material

See `legacy/scheduler_algorithm/README.md`.  To run a historical script while
keeping imports pointed at the active repository root, use:

```bash
PYTHONPATH="$PWD" python3 legacy/scheduler_algorithm/<script>.py --help
```

## Evidence rules

- A replay-valid schedule without a closed lower/upper-bound gap is only an
  upper bound.
- A timeout or expansion limit is not proof of optimality or infeasibility.
- Candidate-window coverage and scorer accuracy are separate questions.
- Policy comparisons must use identical input distributions and preserve the
  hardware and four-stage baselines.
