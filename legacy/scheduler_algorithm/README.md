# Legacy scheduler-algorithm material

This directory is a read-only historical archive of superseded scheduler
policies, exploratory analyses, search launchers, and old RTL specifications.
Nothing here defines the current RTL-oriented Python mirror.

The active model is documented by:

- `../../BOUNDED_DISTILLED_SCHEDULER.md`
- `../../RTL_BOUNDED_DISTILLED_SCHEDULER_CHECKLIST.md`
- `../../scheduler_rtl_distilled_policy.py`

The archived files are retained for three reasons:

1. reproduce the derivation path and ablation history;
2. audit old 30K and directed-case result files;
3. recover a discarded hypothesis without relying only on Git history.

Large historical outputs are archived separately under
`../../results/legacy_scheduler_algorithm/`.  The active proof65, 30K,
showcase, and structural-ablation outputs remain directly under
`../../results/policy_search/`.

Historical scripts still import modules from the repository root.  Run them
from the repository root with the root on `PYTHONPATH`, for example:

```bash
PYTHONPATH="$PWD" python3 legacy/scheduler_algorithm/evaluate_directed_window_grid.py --help
```

Names containing `v1`, `v2`, `v3`, `v4`, `top4+bottom2`, `top6+bottom2`,
`protected`, `adaptive`, or `unified` describe historical experiments only.
Do not use those names as the current algorithm contract.
