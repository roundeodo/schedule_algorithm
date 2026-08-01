#!/usr/bin/env bash
set -euo pipefail

# Full-action-space seed pass for the minimum-positive-load=2 supplement.
# This is kept separate from the already-running frozen v3 base pass.  A zero
# time limit disables branch-and-bound; non-LB results remain upper bounds.

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

case_input="results/policy_search/olmoe_top2_projection_min2_supplement_v1.json"
output="results/policy_search/olmoe_top2_projection_min2_pass1_full_seed_w8_w16.json"
work_dir="results/policy_search/.proof_fragments/olmoe_min2_pass1_full_seed_w8_w16"

test -f "$case_input"

exec python3 -u run_isolated_directed_proofs.py \
  --case-input "$case_input" \
  --seed-beam-widths 8,16 \
  --seed-beam-modes completion,cache,lpt,f_g \
  --time-limit-s 0 \
  --work-dir "$work_dir" \
  --output "$output"
