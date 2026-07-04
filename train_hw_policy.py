"""
train_hw_policy.py
Load the training CSV, train a sklearn DT for action classification,
and print a human-readable rule summary for verification / SV derivation.

Also derives shape-selection statistics and validates analytical rules.
"""

import sys, csv, math
from collections import Counter, defaultdict
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Load data ────────────────────────────────────────────────────────────────
CSV_PATH = 'Idea_Model/hw_training_data.csv'

rows = []
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({k: int(v) for k, v in row.items()})

print(f"Loaded {len(rows)} decision points\n")

ACT_NAMES = {0: 'PAIR', 1: 'SPLIT', 2: 'SINGLE_C2', 3: 'SINGLE_C3', 4: 'PREF/WAIT'}
SHP_NAMES = {0: 'ShapeA(M8)', 1: 'ShapeB(M4)', 2: 'ShapeC(M2)', -1: 'N/A'}

# ── Action distribution ───────────────────────────────────────────────────────
cnt = Counter(r['act'] for r in rows)
total = sum(cnt.values())
print("Action distribution:")
for k in sorted(cnt):
    print(f"  {ACT_NAMES[k]:12s}: {cnt[k]:5d}  ({100*cnt[k]/total:.1f}%)")
print()

# ── Filter to PAIR/SPLIT/SINGLE only (exclude PREF/WAIT for DT training) ────
train_rows = [r for r in rows if r['act'] in (0, 1, 2, 3)]
print(f"Training rows (excl. PREF/WAIT): {len(train_rows)}\n")

FEAT_COLS = ['tok0', 'tok1', 'tok2', 'n_active', 'both_idle', 'c2_rank', 'c3_rank']

X = np.array([[r[c] for c in FEAT_COLS] for r in train_rows])
y = np.array([r['act'] for r in train_rows])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Train DT ─────────────────────────────────────────────────────────────────
dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=42)
dt.fit(X_tr, y_tr)

print(f"DT accuracy on test set: {dt.score(X_te, y_te)*100:.1f}%\n")
print(classification_report(y_te, dt.predict(X_te),
      target_names=[ACT_NAMES[k] for k in sorted(set(y))]))

print("\n── Decision Tree Rules ──────────────────────────────────────────────────")
print(export_text(dt, feature_names=FEAT_COLS, max_depth=8))

# ── Shape analysis: S1 shapes by action type ─────────────────────────────────
print("\n── S1 Shape Distribution by Action ─────────────────────────────────────")
for act_lbl, act_name in [(0, 'PAIR'), (1, 'SPLIT'), (2, 'SINGLE_C2'), (3, 'SINGLE_C3')]:
    sub = [r for r in rows if r['act'] == act_lbl and r['c2_s1'] >= 0]
    if not sub: continue
    cnt2 = Counter(r['c2_s1'] for r in sub)
    n = len(sub)
    parts = [f"{SHP_NAMES[k]}={100*v/n:.0f}%" for k, v in sorted(cnt2.items())]
    print(f"  C2 S1 | {act_name:10s}: {', '.join(parts)}")

print()
for act_lbl, act_name in [(0, 'PAIR'), (1, 'SPLIT'), (2, 'SINGLE_C2'), (3, 'SINGLE_C3')]:
    sub = [r for r in rows if r['act'] == act_lbl and r['c3_s1'] >= 0]
    if not sub: continue
    cnt2 = Counter(r['c3_s1'] for r in sub)
    n = len(sub)
    parts = [f"{SHP_NAMES[k]}={100*v/n:.0f}%" for k, v in sorted(cnt2.items())]
    print(f"  C3 S1 | {act_name:10s}: {', '.join(parts)}")

# ── Shape analysis: S3 shapes by action type ─────────────────────────────────
print("\n── S3 Shape Distribution by Action ─────────────────────────────────────")
for act_lbl, act_name in [(0, 'PAIR'), (1, 'SPLIT'), (2, 'SINGLE_C2'), (3, 'SINGLE_C3')]:
    sub = [r for r in rows if r['act'] == act_lbl and r['c2_s3'] >= 0]
    if not sub: continue
    cnt2 = Counter(r['c2_s3'] for r in sub)
    n = len(sub)
    parts = [f"{SHP_NAMES[k]}={100*v/n:.0f}%" for k, v in sorted(cnt2.items())]
    print(f"  C2 S3 | {act_name:10s}: {', '.join(parts)}")

# ── Analytical shape rule validation ─────────────────────────────────────────
print("\n── Analytical Rule Validation ───────────────────────────────────────────")
print("Rule: S1_solo→ShapeC, S1_concurrent→ShapeB, S3→ShapeC (sym-small→ShapeB)")
correct_s1 = total_s1 = 0
correct_s3 = total_s3 = 0
SMALL_THR  = 4  # tok <= 4 → symmetric small

for r in rows:
    act = r['act']
    if act not in (0, 1, 2, 3):
        continue
    concurrent = (act in (0, 1))   # PAIR or SPLIT → both fetching S1

    # C2 S1 prediction
    if r['c2_s1'] >= 0:
        pred = 1 if concurrent else 2  # concurrent→B(1), solo→C(2)
        correct_s1 += int(pred == r['c2_s1'])
        total_s1 += 1

    # C3 S1 prediction
    if r['c3_s1'] >= 0:
        pred = 1 if concurrent else 2
        correct_s1 += int(pred == r['c3_s1'])
        total_s1 += 1

    # S3 prediction (both C2 and C3 if both assigned)
    sym_small = (act == 0 and r['tok0'] <= SMALL_THR and r['tok1'] <= SMALL_THR)
    if act == 1:  # SPLIT
        sym_small = (r['tok0'] // 2 <= SMALL_THR and
                     (r['tok0'] - r['tok0'] // 2) <= SMALL_THR)
    s3_pred = 1 if sym_small else 2  # sym_small→B(1), else→C(2)
    if r['c2_s3'] >= 0:
        correct_s3 += int(s3_pred == r['c2_s3'])
        total_s3 += 1
    if r['c3_s3'] >= 0:
        correct_s3 += int(s3_pred == r['c3_s3'])
        total_s3 += 1

print(f"  S1 analytical accuracy: {correct_s1}/{total_s1} = {100*correct_s1/max(1,total_s1):.1f}%")
print(f"  S3 analytical accuracy: {correct_s3}/{total_s3} = {100*correct_s3/max(1,total_s3):.1f}%")

# ── Simple threshold analysis: when does SPLIT happen? ────────────────────────
print("\n── SPLIT vs PAIR: threshold analysis (both_idle=1, n_active>=2) ─────────")
both_idle_rows = [r for r in rows if r['both_idle'] == 1 and r['n_active'] >= 2
                  and r['act'] in (0, 1)]
print(f"  Total PAIR/SPLIT rows with both_idle=1, n_active>=2: {len(both_idle_rows)}")
pair_tok0  = [r['tok0'] for r in both_idle_rows if r['act'] == 0]
split_tok0 = [r['tok0'] for r in both_idle_rows if r['act'] == 1]

if pair_tok0:
    print(f"  PAIR  tok0: min={min(pair_tok0)}, max={max(pair_tok0)}, "
          f"mean={sum(pair_tok0)/len(pair_tok0):.1f}")
if split_tok0:
    print(f"  SPLIT tok0: min={min(split_tok0)}, max={max(split_tok0)}, "
          f"mean={sum(split_tok0)/len(split_tok0):.1f}")

# Ratio analysis
both_idle_with_ratio = [(r['tok0'], r['tok1'], r['act'])
                        for r in both_idle_rows if r['tok1'] > 0]
split_ratios = [t / s for t, s, a in both_idle_with_ratio if a == 1]
pair_ratios  = [t / s for t, s, a in both_idle_with_ratio if a == 0]
if split_ratios:
    print(f"  SPLIT tok0/tok1: min={min(split_ratios):.2f}, mean={sum(split_ratios)/len(split_ratios):.2f}")
if pair_ratios:
    print(f"  PAIR  tok0/tok1: min={min(pair_ratios):.2f}, mean={sum(pair_ratios)/len(pair_ratios):.2f}")
