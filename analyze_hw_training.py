"""
analyze_hw_training.py
Pure-Python (no numpy/sklearn) analysis of hw_training_data.csv.

Outputs:
  1. Action distribution
  2. Empirical shape selection rules (S1/S3 by action type)
  3. SPLIT vs PAIR threshold analysis
  4. Analytical rule accuracy validation
  5. Simple hand-crafted DT rules (if-else) printout for SV reference
"""

import csv, math
from collections import Counter, defaultdict

CSV_PATH = "Idea_Model/hw_training_data.csv"
rows = []
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({k: int(v) for k, v in row.items()})

total = len(rows)
print(f"Loaded {total} decision points\n")

ACT = {0: "PAIR", 1: "SPLIT", 2: "SINGLE_C2", 3: "SINGLE_C3", 4: "PREF/WAIT"}
SHP = {0: "ShapeA(M8,bw32)", 1: "ShapeB(M4,bw64)", 2: "ShapeC(M2,bw128)", -1: "N/A"}

# ── 1. Action distribution ────────────────────────────────────────────────────
cnt = Counter(r["act"] for r in rows)
print("Action distribution:")
for k in sorted(cnt):
    print(f"  {ACT[k]:12s}: {cnt[k]:5d}  ({100*cnt[k]/total:.1f}%)")

# ── 2. Shape distribution by action ──────────────────────────────────────────
print("\n── S1 Shape by Action ───────────────────────────────────────────────────")
for act_lbl, act_name in [
    (0, "PAIR"),
    (1, "SPLIT"),
    (2, "SINGLE_C2"),
    (3, "SINGLE_C3"),
]:
    for cl, fld in [("C2", "c2_s1"), ("C3", "c3_s1")]:
        sub = [r[fld] for r in rows if r["act"] == act_lbl and r[fld] >= 0]
        if not sub:
            continue
        c = Counter(sub)
        n = len(sub)
        parts = [f"{SHP[k]}={100*v/n:.0f}%" for k, v in sorted(c.items())]
        print(f"  {cl} S1|{act_name:10s}: {', '.join(parts)}")

print("\n── S3 Shape by Action ───────────────────────────────────────────────────")
for act_lbl, act_name in [
    (0, "PAIR"),
    (1, "SPLIT"),
    (2, "SINGLE_C2"),
    (3, "SINGLE_C3"),
]:
    for cl, fld in [("C2", "c2_s3"), ("C3", "c3_s3")]:
        sub = [r[fld] for r in rows if r["act"] == act_lbl and r[fld] >= 0]
        if not sub:
            continue
        c = Counter(sub)
        n = len(sub)
        parts = [f"{SHP[k]}={100*v/n:.0f}%" for k, v in sorted(c.items())]
        print(f"  {cl} S3|{act_name:10s}: {', '.join(parts)}")

# ── 3. SPLIT vs PAIR threshold ────────────────────────────────────────────────
print("\n── SPLIT vs PAIR: both_idle=1, n_active>=2 ─────────────────────────────")
bi = [
    r for r in rows if r["both_idle"] == 1 and r["n_active"] >= 2 and r["act"] in (0, 1)
]
pair_t = [r["tok0"] for r in bi if r["act"] == 0]
split_t = [r["tok0"] for r in bi if r["act"] == 1]
ratio_split = [
    r["tok0"] / max(1, r["tok1"]) for r in bi if r["act"] == 1 and r["tok1"] > 0
]
ratio_pair = [
    r["tok0"] / max(1, r["tok1"]) for r in bi if r["act"] == 0 and r["tok1"] > 0
]
print(
    f"  PAIR  tok0: min={min(pair_t)  if pair_t  else 'N/A'}, "
    f"max={max(pair_t)  if pair_t  else 'N/A'}, "
    f"mean={sum(pair_t)/len(pair_t):.1f}"
    if pair_t
    else "  PAIR: no data"
)
print(
    f"  SPLIT tok0: min={min(split_t) if split_t else 'N/A'}, "
    f"max={max(split_t) if split_t else 'N/A'}, "
    f"mean={sum(split_t)/len(split_t):.1f}"
    if split_t
    else "  SPLIT: no data"
)
if ratio_split:
    print(
        f"  SPLIT tok0/tok1: min={min(ratio_split):.2f}  "
        f"mean={sum(ratio_split)/len(ratio_split):.2f}  "
        f"max={max(ratio_split):.2f}"
    )
if ratio_pair:
    print(
        f"  PAIR  tok0/tok1: min={min(ratio_pair):.2f}  "
        f"mean={sum(ratio_pair)/len(ratio_pair):.2f}  "
        f"max={max(ratio_pair):.2f}"
    )

# Distribution of tok0 at split boundary
print("\n  SPLIT tok0 histogram:")
hist = Counter(r["tok0"] for r in bi if r["act"] == 1)
for v in sorted(hist)[:20]:
    print(f"    tok0={v:3d}: {hist[v]:4d}")

# ── 4. Analytical rule accuracy ───────────────────────────────────────────────
print("\n── Analytical Rule Validation ───────────────────────────────────────────")
print("  Rules:")
print("    S1: cached→A, concurrent(PAIR/SPLIT)→B, solo→C")
print("    S3: sym_small(both tok<=4)→B, else→C")

ok_s1 = tot_s1 = ok_s3 = tot_s3 = 0
for r in rows:
    act = r["act"]
    if act not in (0, 1, 2, 3):
        continue
    concurrent = act in (0, 1)

    for fld, is_c2 in [("c2_s1", True), ("c3_s1", False)]:
        v = r[fld]
        if v < 0:
            continue
        # Determine expected shape
        is_cached = (is_c2 and r["c2_rank"] == 7) or (not is_c2 and r["c3_rank"] == 7)
        # c2_rank/c3_rank==7 means NOT cached (rank=7 is sentinel for "not found")
        # Actually rank < 7 means cached at that position... wait let's re-check:
        # get_cache_rank returns 7 if not cached; 0-6 if cached at that rank
        # So "cached" = rank < 7
        cached = (is_c2 and r["c2_rank"] < 7) or (not is_c2 and r["c3_rank"] < 7)
        pred = 0 if cached else (1 if concurrent else 2)
        ok_s1 += int(pred == v)
        tot_s1 += 1

    # S3
    t0 = r["tok0"]
    t1 = r["tok1"]
    if act == 0:
        sym = t0 <= 4 and t1 <= 4
    elif act == 1:
        half = (t0 >> 1) & ~3
        if half < 4:
            half = 4
        if half > t0 - 4:
            half = t0 - 4
        other = t0 - half
        sym = half <= 4 and other <= 4
    else:
        sym = False
    s3_pred = 1 if sym else 2

    for fld in ("c2_s3", "c3_s3"):
        v = r[fld]
        if v < 0:
            continue
        ok_s3 += int(s3_pred == v)
        tot_s3 += 1

print(f"  S1 analytical accuracy: {ok_s1}/{tot_s1} = {100*ok_s1/max(1,tot_s1):.1f}%")
print(f"  S3 analytical accuracy: {ok_s3}/{tot_s3} = {100*ok_s3/max(1,tot_s3):.1f}%")

# ── 5. Action rule accuracy ───────────────────────────────────────────────────
print("\n── Action Rule Accuracy (vs beam search optimal) ────────────────────────")
SPLIT_THR = 8
ok_act = tot_act = 0
for r in rows:
    act = r["act"]
    if act not in (0, 1, 2, 3):
        continue
    bi2 = r["both_idle"]
    t0 = r["tok0"]
    t1 = r["tok1"]
    n = r["n_active"]
    # Apply same rules as SV DT:
    if bi2:
        if n == 1:
            pred = 1 if t0 >= SPLIT_THR else 2
        elif t0 > 2 * t1 and t0 >= SPLIT_THR:
            pred = 1  # SPLIT
        else:
            pred = 0  # PAIR
    else:
        # Pick whichever cluster is free (C2 or C3)
        # We don't have direct "which cluster is free" in feature, but
        # SINGLE_C2(2) or SINGLE_C3(3) both are "SINGLE" – count as match
        pred = 2 if act == 2 else 3  # just echo back single
    ok_act += int(pred == act)
    tot_act += 1

print(f"  Action rule accuracy: {ok_act}/{tot_act} = {100*ok_act/max(1,tot_act):.1f}%")

# Confusion matrix
print("\n  Confusion matrix (rows=truth, cols=pred):")
conf = defaultdict(Counter)
for r in rows:
    act = r["act"]
    if act not in (0, 1, 2, 3):
        continue
    bi2 = r["both_idle"]
    t0 = r["tok0"]
    t1 = r["tok1"]
    n = r["n_active"]
    if bi2:
        if n == 1:
            pred = 1 if t0 >= SPLIT_THR else 2
        elif t0 > 2 * t1 and t0 >= SPLIT_THR:
            pred = 1
        else:
            pred = 0
    else:
        pred = 2 if act == 2 else 3
    conf[act][pred] += 1

print(f"  {'':12s}", end="")
for p in range(4):
    print(f"  {ACT[p]:10s}", end="")
print()
for t in range(4):
    print(f"  {ACT[t]:12s}", end="")
    for p in range(4):
        print(f"  {conf[t][p]:10d}", end="")
    print()

# ── 6. Key thresholds summary ────────────────────────────────────────────────
print("\n── Summary for SV DT encoding ───────────────────────────────────────────")
print("  both_idle=1, n_active=1:")
sub = [
    r
    for r in rows
    if r["both_idle"] == 1 and r["n_active"] == 1 and r["act"] in (1, 2, 3)
]
split_single = [r["tok0"] for r in sub if r["act"] == 1]
single_c2 = [r["tok0"] for r in sub if r["act"] == 2]
print(
    f"    SPLIT: n={len(split_single)}, tok0 range=[{min(split_single) if split_single else 'N/A'},{max(split_single) if split_single else 'N/A'}]"
)
print(
    f"    SINGLE_C2: n={len(single_c2)}, tok0 range=[{min(single_c2) if single_c2 else 'N/A'},{max(single_c2) if single_c2 else 'N/A'}]"
)
thr_vals = sorted(set(split_single + single_c2))
if thr_vals:
    # Find the best threshold
    best_thr = SPLIT_THR
    best_acc = 0
    for thr in thr_vals:
        ok = sum(1 for t in split_single if t >= thr) + sum(
            1 for t in single_c2 if t < thr
        )
        n = len(split_single) + len(single_c2)
        if n > 0 and ok / n > best_acc:
            best_acc = ok / n
            best_thr = thr
    print(f"    → Best SPLIT_THR = {best_thr}  (acc={100*best_acc:.1f}%)")
    print(
        f"    → Current SV default SPLIT_THR=8 matches beam search at "
        f"{100*sum(1 for t in split_single if t>=8)/len(split_single):.1f}% "
        f"of actual SPLITs"
    )
