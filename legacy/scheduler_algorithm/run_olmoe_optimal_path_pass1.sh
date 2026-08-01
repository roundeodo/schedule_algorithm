#!/usr/bin/env bash
set -euo pipefail

# Pass 1 of the stratified exact-70 OLMoE-like optimal-path campaign.
#
# The case set covers 6--8x, 8--12x, and 12--14x top1/mean bands with two,
# three, and four local hotspots.  This pass deliberately uses unrestricted
# full-action-space beam seeds; it does not impose a hardware candidate window.
# --time-limit-s 0 disables branch-and-bound, so this pass primarily obtains
# replay-valid upper bounds.  A case is labelled optimal only if a seed history
# reaches the admissible full-model root lower bound.  Unclosed cases require a
# later positive-time branch-and-bound pass and are not window failures.

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

case_input="results/policy_search/olmoe_top2_projection_cases_v3.json"
prior_proof="results/policy_search/olmoe_top2_projection_v2_best_known.json"
output="results/policy_search/olmoe_top2_projection_v3_pass1_full_seed_w8_w16.json"
work_dir="results/policy_search/.proof_fragments/olmoe_v3_pass1_full_seed_w8_w16"

test -f "$case_input"
test -f "$prior_proof"

exec python3 -u run_isolated_directed_proofs.py \
  --case-input "$case_input" \
  --prior-proof "$prior_proof" \
  --seed-beam-widths 8,16 \
  --seed-beam-modes completion,cache,lpt,f_g \
  --time-limit-s 0 \
  --work-dir "$work_dir" \
  --output "$output"
