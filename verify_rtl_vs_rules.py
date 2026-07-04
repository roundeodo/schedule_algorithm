"""
verify_rtl_vs_rules.py
验证 moe_hw_scheduler.sv 的决策树实现与 hw_rules.py 的 Python 参考完全一致。

方法：穷举/采样 (tok0, tok1, tok2, q_rem, both_idle, c2_rank, c3_rank) 的典型值，
      分别用 hw_rules.py 和 RTL 的 Python 仿真模型计算，比对输出。

注意：这是 "RTL vs hw_rules.py" 的等价性验证，
      与 "hw_rules.py vs beam-search" 的质量评估是完全不同的测试。
"""

import sys, importlib.util, math
from typing import Dict

# ── 加载 hw_rules.py ──────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("hw_rules", "Idea_Model/hw_rules.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

ACT_NAMES  = {0: "PAIR", 1: "SPLIT", 2: "SNGL_C2", 3: "SNGL_C3", 4: "PREFETCH"}
SHAPE_NAMES = {0: "SH_A", 1: "SH_B", 2: "SH_C", -1: "N/A"}

# ── RTL 的 Python 仿真模型（与 moe_hw_scheduler.sv always_comb 完全一致）──────
# 每次修改 RTL 后，手动同步这里的逻辑，然后跑本脚本验证等价性

def rtl_action(tok0, tok1, tok2, q_rem, both_idle, c2_rank, c3_rank) -> int:
    """直接翻译自 moe_hw_scheduler.sv always_comb action block"""
    tok_diff = tok0 - tok1
    if tok_diff <= 5:
        if tok1 <= 0:
            if tok0 <= 3:
                if both_idle <= 0:
                    return 2  # SNGL_C2
                else:
                    if tok0 <= 2:
                        return 2  # SNGL_C2
                    else:
                        return 1  # SPLIT
            else:
                if both_idle <= 0:
                    if tok0 <= 4:
                        return 2  # SNGL_C2
                    else:
                        return 1  # SPLIT
                else:
                    return 1  # SPLIT
        else:
            if tok2 <= 4:
                if both_idle <= 0:
                    if tok2 <= 0:
                        return 0  # PAIR
                    else:
                        return 3  # SNGL_C3
                else:
                    if tok0 <= 4:
                        return 0  # PAIR
                    else:
                        return 0  # PAIR
            else:
                if both_idle <= 0:
                    if tok1 <= 8:
                        return 3  # SNGL_C3
                    else:
                        return 1  # SPLIT
                else:
                    if c3_rank <= 0:
                        return 2  # SNGL_C2
                    else:
                        return 0  # PAIR
    else:
        if tok2 <= 0:
            if both_idle <= 0:
                if tok_diff <= 6:
                    if tok0 <= 6:
                        return 1  # SPLIT
                    else:
                        return 2  # SNGL_C2
                else:
                    return 1  # SPLIT
            else:
                if tok_diff <= 6:
                    return 1  # SPLIT
                else:
                    if tok1 <= 0:
                        return 1  # SPLIT
                    else:
                        return 1  # SPLIT
        else:
            if both_idle <= 0:
                if c2_rank <= 3:
                    return 1  # SPLIT
                else:
                    if q_rem <= 4:
                        return 3  # SNGL_C3
                    else:
                        return 3  # SNGL_C3
            else:
                if q_rem <= 3:
                    if tok0 <= 8:
                        return 2  # SNGL_C2
                    else:
                        return 1  # SPLIT
                else:
                    if q_rem <= 5:
                        return 2  # SNGL_C2
                    else:
                        return 0  # PAIR


def rtl_c2_s1(tok0, tok1, tok2, q_rem, both_idle, c2_rank, c3_rank) -> int:
    tok_diff = tok0 - tok1
    if tok0 <= 12:
        if tok1 <= 6:
            if tok_diff <= 0:
                return 2 if c2_rank <= 3 else 1  # SH_C / SH_B
            else:
                return 2 if tok0 <= 1 else 1  # SH_C / SH_B
        else:
            if tok2 <= 1:
                return 0  # SH_A
            else:
                return 1  # SH_B
    else:
        if q_rem <= 3:
            if tok2 <= 0:
                return 0  # SH_A (both branches)
            else:
                return 2 if tok1 <= 2 else 0  # SH_C / SH_A
        else:
            if tok2 <= 4:
                return 2 if q_rem <= 6 else 1  # SH_C / SH_B
            else:
                return 1 if tok2 <= 6 else 0  # SH_B / SH_A


def rtl_c2_s3(tok0, tok1, tok2, q_rem, both_idle, c2_rank, c3_rank) -> int:
    tok_diff = tok0 - tok1
    if tok0 <= 8:
        if tok1 <= 0:
            return 2 if tok0 <= 2 else 1  # SH_C / SH_B
        else:
            if tok1 <= 6:
                return 1  # SH_B
            else:
                return 0 if tok2 <= 1 else 1  # SH_A / SH_B
    else:
        if q_rem <= 3:
            if tok0 <= 12:
                return 0 if tok_diff <= 4 else 1  # SH_A / SH_B
            else:
                return 0  # SH_A (both branches)
        else:
            if tok2 <= 2:
                return 2 if q_rem <= 6 else 1  # SH_C / SH_B
            else:
                return 1 if tok2 <= 8 else 0  # SH_B / SH_A


def rtl_c3_s1(tok0, tok1, tok2, q_rem, both_idle, c2_rank, c3_rank) -> int:
    tok_diff = tok0 - tok1
    if tok0 <= 12:
        if tok1 <= 4:
            if c2_rank <= 3:
                return 2 if tok0 <= 4 else 1  # SH_C / SH_B
            else:
                return 1  # SH_B (both idle/not)
        else:
            if tok2 <= 0:
                return 1 if tok1 <= 6 else 0  # SH_B / SH_A
            else:
                return 1  # SH_B
    else:
        if q_rem <= 3:
            # both branches return SH_A
            return 0
        else:
            if not both_idle:  # DT: both_idle<=0.5 (NOT idle)
                return 2 if tok2 <= 2 else 1  # SH_C / SH_B
            else:
                return 0 if q_rem <= 5 else 1  # SH_A / SH_B


def rtl_c3_s3(tok0, tok1, tok2, q_rem, both_idle, c2_rank, c3_rank) -> int:
    tok_diff = tok0 - tok1
    if tok0 <= 12:
        if not both_idle:  # DT: both_idle<=0.5 (NOT idle)
            if c3_rank <= 0:
                return 0 if tok2 <= 2 else 2  # SH_A / SH_C
            else:
                return 2 if tok0 <= 2 else 1  # SH_C / SH_B
        else:
            if tok1 <= 6:
                return 1  # SH_B (both tok_diff branches)
            else:
                return 0 if tok2 <= 2 else 1  # SH_A / SH_B
    else:
        if tok2 <= 0:
            if tok1 <= 2:
                return 0  # SH_A (both tok_diff branches)
            else:
                return 1 if tok1 <= 4 else 0  # SH_B / SH_A
        else:
            if tok2 <= 2:
                return 1 if both_idle else 2  # SH_B / SH_C  (RTL: both_idle → SH_B, !idle → SH_C)
            else:
                return 0 if q_rem <= 3 else 1  # SH_A / SH_B


# ── 穷举测试 ───────────────────────────────────────────────────────────────────
tok_range   = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32]
q_rem_range = [1, 2, 3, 4, 6, 8]
rank_range  = [0, 1, 3, 4, 7]  # typical cache rank values
both_vals   = [0, 1]

total = mismatches_act = mismatches_c2s1 = mismatches_c2s3 = 0
mismatches_c3s1 = mismatches_c3s3 = 0
errors = []

for tok0 in tok_range:
    for tok1 in tok_range:
        if tok1 > tok0:
            continue  # descending order constraint
        for tok2 in tok_range:
            if tok2 > tok1:
                continue
            for q_rem in q_rem_range:
                for both_idle in both_vals:
                    for c2r in rank_range:
                        for c3r in rank_range:
                            total += 1
                            args = (tok0, tok1, tok2, q_rem, both_idle, c2r, c3r)

                            # hw_rules.py 预测
                            py_act   = mod.predict_action(*args)
                            py_c2s1  = mod.predict_c2_s1(*args)
                            py_c2s3  = mod.predict_c2_s3(*args)
                            py_c3s1  = mod.predict_c3_s1(*args)
                            py_c3s3  = mod.predict_c3_s3(*args)

                            # RTL Python 模型预测
                            sv_act   = rtl_action(*args)
                            sv_c2s1  = rtl_c2_s1(*args)
                            sv_c2s3  = rtl_c2_s3(*args)
                            sv_c3s1  = rtl_c3_s1(*args)
                            sv_c3s3  = rtl_c3_s3(*args)

                            diffs = []
                            if py_act  != sv_act:   mismatches_act  += 1; diffs.append(f"act: py={ACT_NAMES[py_act]} sv={ACT_NAMES[sv_act]}")
                            if py_c2s1 != sv_c2s1:  mismatches_c2s1 += 1; diffs.append(f"c2s1: py={SHAPE_NAMES[py_c2s1]} sv={SHAPE_NAMES[sv_c2s1]}")
                            if py_c2s3 != sv_c2s3:  mismatches_c2s3 += 1; diffs.append(f"c2s3: py={SHAPE_NAMES[py_c2s3]} sv={SHAPE_NAMES[sv_c2s3]}")
                            if py_c3s1 != sv_c3s1:  mismatches_c3s1 += 1; diffs.append(f"c3s1: py={SHAPE_NAMES[py_c3s1]} sv={SHAPE_NAMES[sv_c3s1]}")
                            if py_c3s3 != sv_c3s3:  mismatches_c3s3 += 1; diffs.append(f"c3s3: py={SHAPE_NAMES[py_c3s3]} sv={SHAPE_NAMES[sv_c3s3]}")

                            if diffs and len(errors) < 20:
                                errors.append(
                                    f"  tok={tok0},{tok1},{tok2} q={q_rem} bi={both_idle} "
                                    f"c2r={c2r} c3r={c3r} → {'; '.join(diffs)}"
                                )

print(f"总测试点: {total}")
print(f"动作  不一致: {mismatches_act:5d}  ({mismatches_act/total*100:.1f}%)")
print(f"c2_s1 不一致: {mismatches_c2s1:5d}  ({mismatches_c2s1/total*100:.1f}%)")
print(f"c2_s3 不一致: {mismatches_c2s3:5d}  ({mismatches_c2s3/total*100:.1f}%)")
print(f"c3_s1 不一致: {mismatches_c3s1:5d}  ({mismatches_c3s1/total*100:.1f}%)")
print(f"c3_s3 不一致: {mismatches_c3s3:5d}  ({mismatches_c3s3/total*100:.1f}%)")

if errors:
    print(f"\n前 {len(errors)} 个不一致示例:")
    for e in errors:
        print(e)
else:
    print("\n✓ RTL Python 模型与 hw_rules.py 完全一致")
