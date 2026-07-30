#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

jobs="${JOBS:-2}"
results_dir="results/policy_search"

python3 -u run_isolated_directed_proofs.py \
  --case-input "${results_dir}/olmoe_top2_projection_cases_v3.json" \
  --prior-proof "${results_dir}/olmoe_top2_projection_base_best_known_v1.json" \
  --only-unproven \
  --jobs "${jobs}" \
  --time-limit-s 0 \
  --seed-beam-widths 64 \
  --seed-beam-modes completion,cache,lpt,f_g \
  --keep-fragments \
  --output "${results_dir}/olmoe_base_unproven_seed_w64.json"

python3 merge_top4_bottom2_proofs.py \
  "${results_dir}/olmoe_top2_projection_base_best_known_v1.json" \
  "${results_dir}/olmoe_base_unproven_seed_w64.json" \
  --output "${results_dir}/olmoe_top2_projection_base_best_known_w64.json"

python3 -u run_isolated_directed_proofs.py \
  --case-input "${results_dir}/olmoe_top2_projection_min2_supplement_v1.json" \
  --prior-proof "${results_dir}/olmoe_top2_projection_min2_best_known_v1.json" \
  --only-unproven \
  --jobs "${jobs}" \
  --time-limit-s 0 \
  --seed-beam-widths 64 \
  --seed-beam-modes completion,cache,lpt,f_g \
  --keep-fragments \
  --output "${results_dir}/olmoe_min2_unproven_seed_w64.json"

python3 merge_top4_bottom2_proofs.py \
  "${results_dir}/olmoe_top2_projection_min2_best_known_v1.json" \
  "${results_dir}/olmoe_min2_unproven_seed_w64.json" \
  --output "${results_dir}/olmoe_top2_projection_min2_best_known_w64.json"
