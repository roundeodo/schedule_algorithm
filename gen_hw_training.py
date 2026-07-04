"""
gen_hw_training.py
Generate (state → optimal_action) training pairs for the hardware scheduler DT.

For each random MoE distribution:
  1. Run beam search (beam_width=16) to get optimal schedule.
  2. Walk the schedule step-by-step and record the feature vector at
     each decision point alongside the action label.

Output: hw_training_data.csv   (in Idea_Model/)
"""

import sys, random, csv, math, time

sys.path.insert(0, ".")

from four_stage_scheduler import (
    FourStageScheduler,
    FourStageSnap,
    BeamState,
    lb_remaining,
    make_initial_snap,
    apply_action,
    SHAPE_A,
    SHAPE_B,
    SHAPE_C,
    WEIGHT_BYTES_S1,
    WEIGHT_BYTES_S3,
    MAX_BW,
)

# ── reproducibility ──────────────────────────────────────────────────────────
random.seed(42)

# ── constants ─────────────────────────────────────────────────────────────────
N_SAMPLES = 10000
BEAM_WIDTH = 16  # quality vs speed tradeoff for training data
SHAPE_TO_IDX = {8: 0, 4: 1, 2: 2}  # A→0, B→1, C→2

# ── helpers ───────────────────────────────────────────────────────────────────


def random_dist():
    """Return a random token distribution dict {eid: ntok, ...}."""
    M = random.choice([4, 8, 12, 16, 24, 32, 48, 64])
    n = random.randint(1, min(8, M))

    dist_type = random.choices(
        ["uniform", "zipf", "hot", "bimodal", "single"], weights=[2, 4, 3, 2, 1]
    )[0]

    if dist_type == "single":
        return {0: M}

    if dist_type == "uniform":
        base = M // n
        toks = [base] * n
        toks[0] += M - sum(toks)

    elif dist_type == "zipf":
        alpha = random.uniform(0.6, 2.5)
        weights = [1.0 / (i + 1) ** alpha for i in range(n)]
        s = sum(weights)
        toks = [max(1, round(w / s * M)) for w in weights]
        toks.sort(reverse=True)
        toks[0] += M - sum(toks)

    elif dist_type == "hot":
        hot_frac = random.uniform(0.35, 0.80)
        hot = max(1, round(M * hot_frac))
        rest = M - hot
        toks = [hot] + ([max(1, rest // max(1, n - 1))] * (n - 1) if n > 1 else [])
        toks[0] += M - sum(toks)

    else:  # bimodal
        h = max(1, n // 2)
        c = n - h
        hot_v = max(1, round(M * 0.7 / max(1, h)))
        cold_v = max(1, round(M * 0.3 / max(1, c)))
        toks = [hot_v] * h + [cold_v] * c
        toks.sort(reverse=True)
        toks[0] += M - sum(toks)

    toks = sorted([max(1, t) for t in toks], reverse=True)
    return {i: t for i, t in enumerate(toks) if t > 0}


def get_cache_rank(snap: FourStageSnap, remaining: tuple) -> int:
    """Return the rank (0-based) of the snap's cached expert in remaining.
    Returns 7 if not cached or not present."""
    eid = snap.pf_eid
    if eid < 0:
        return 7
    # pf_end == 0 → pre-loaded (initial cache); pf_end > 0 → prefetched
    if snap.pf_end < 0:
        return 7
    for i, (e, _) in enumerate(remaining):
        if e == eid:
            return i
    return 7  # cached expert consumed already


def encode_action(action, remaining: tuple) -> int:
    """Encode beam action as integer label.
    0=PAIR, 1=SPLIT, 2=SINGLE_C2, 3=SINGLE_C3, 4=PREFETCH/WAIT
    """
    if action.pf_cluster > 0:
        return 4  # PREFETCH
    c2 = action.c2_eid
    c3 = action.c3_eid
    if c2 >= 0 and c3 >= 0:
        return 1 if c2 == c3 else 0  # SPLIT or PAIR
    if c2 >= 0:
        return 2  # SINGLE_C2
    if c3 >= 0:
        return 3  # SINGLE_C3
    return 4  # WAIT / other


def encode_shape(shape_obj) -> int:
    if shape_obj is None:
        return -1
    return SHAPE_TO_IDX.get(shape_obj.M_dim, -1)


# ── main data generation loop ─────────────────────────────────────────────────

rows = []
skipped = 0
t0 = time.time()

for sample_i in range(N_SAMPLES):
    if (sample_i + 1) % 200 == 0:
        elapsed = time.time() - t0
        rate = (sample_i + 1) / elapsed
        eta = (N_SAMPLES - sample_i - 1) / rate
        print(
            f"  [{sample_i+1}/{N_SAMPLES}]  rows={len(rows)}  "
            f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
            flush=True,
        )

    dist = random_dist()
    n = len(dist)

    # Random initial cache (25% chance for C2, 12% for C3)
    cache_c2 = random.choice([-1, -1, -1, 0]) if n >= 1 else -1
    cache_c3_opts = [-1, -1, -1]
    if n >= 2 and cache_c2 >= 0:
        cache_c3_opts.append(1)
    cache_c3 = random.choice(cache_c3_opts)

    try:
        ms, hist = FourStageScheduler(
            dist,
            beam_width=BEAM_WIDTH,
            initial_cache_c2=cache_c2,
            initial_cache_c3=cache_c3,
        ).run()
    except Exception as exc:
        skipped += 1
        continue

    if not hist:
        skipped += 1
        continue

    # Replay history and collect decision-point features
    c2_snap = make_initial_snap(cache_c2)
    c3_snap = make_initial_snap(cache_c3)
    remaining = tuple(sorted(dist.items(), key=lambda x: -x[1]))
    state = BeamState(
        c2=c2_snap, c3=c3_snap, remaining=remaining, history=(), g_score=0, f_score=0
    )

    for action in hist:
        if not state.remaining:
            break

        tok = [nt for _, nt in state.remaining] + [0] * (8 - len(state.remaining))
        n_act = len(state.remaining)
        t2, t3 = state.c2.task_end, state.c3.task_end
        both_idle = int(t2 == t3)

        c2r = get_cache_rank(state.c2, state.remaining)
        c3r = get_cache_rank(state.c3, state.remaining)

        act_lbl = encode_action(action, state.remaining)

        c2_s1 = encode_shape(action.c2_shape_s1)
        c2_s3 = encode_shape(action.c2_shape_s3)
        c3_s1 = encode_shape(action.c3_shape_s1)
        c3_s3 = encode_shape(action.c3_shape_s3)

        rows.append(
            {
                "tok0": tok[0],
                "tok1": tok[1],
                "tok2": tok[2],
                "n_active": n_act,
                "both_idle": both_idle,
                "c2_rank": c2r,
                "c3_rank": c3r,
                "act": act_lbl,
                "c2_s1": c2_s1,
                "c2_s3": c2_s3,
                "c3_s1": c3_s1,
                "c3_s3": c3_s3,
            }
        )

        try:
            state = apply_action(state, action)
        except Exception:
            break

# ── save CSV ───────────────────────────────────────────────────────────────────
FIELDS = [
    "tok0",
    "tok1",
    "tok2",
    "n_active",
    "both_idle",
    "c2_rank",
    "c3_rank",
    "act",
    "c2_s1",
    "c2_s3",
    "c3_s1",
    "c3_s3",
]

out_path = "Idea_Model/hw_training_data.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)

elapsed = time.time() - t0
print(
    f"\nDone in {elapsed:.1f}s: {len(rows)} decision points from "
    f"{N_SAMPLES - skipped}/{N_SAMPLES} samples  (skipped={skipped})"
)
print(f"Saved → {out_path}")

# ── quick action-distribution summary ────────────────────────────────────────
from collections import Counter

act_names = {0: "PAIR", 1: "SPLIT", 2: "SINGLE_C2", 3: "SINGLE_C3", 4: "PREFETCH/WAIT"}
cnt = Counter(r["act"] for r in rows)
total = sum(cnt.values())
print("\nAction distribution:")
for k in sorted(cnt):
    print(f"  {act_names[k]:12s}: {cnt[k]:5d}  ({100*cnt[k]/total:.1f}%)")
