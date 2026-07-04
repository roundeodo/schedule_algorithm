"""
train_dt.py  ─  纯 Python CART 决策树，从 hw_training_data.csv 训练 RTL 调度规则

无需 sklearn / numpy / pandas，只依赖 Python 3 标准库。

训练三棵树：
  1. act_tree   : 预测动作（0=PAIR 1=SPLIT 2=SINGLE_C2 3=SINGLE_C3 4=PREFETCH）
  2. shape_tree : 预测两个 cluster 的 S1/S3 shape（ShapeA/B/C）
                  — 实际上训练 4 棵独立子树（c2_s1, c2_s3, c3_s1, c3_s3）

输出：
  - 终端打印 if-else 规则（可直接翻译为 SystemVerilog）
  - Idea_Model/hw_rules.py   (Python if-else，用于对照仿真)
"""

import csv, math, sys
from collections import Counter
from typing import List, Dict, Optional, Tuple, Any

# ─────────────────────────────────────────────────────────────────────────────
# CART 决策树（最小化加权 Gini 不纯度）
# ─────────────────────────────────────────────────────────────────────────────

FEATURES = [
    "tok0",
    "tok1",
    "tok2",
    "tok_diff",
    "tok_ratio4",
    "n_active",
    "both_idle",
    "c2_rank",
    "c3_rank",
]
ACT_NAMES = {0: "PAIR", 1: "SPLIT", 2: "SINGLE_C2", 3: "SINGLE_C3", 4: "PREFETCH"}
SHAPE_NAMES = {0: "ShapeA", 1: "ShapeB", 2: "ShapeC", -1: "N/A"}


class Node:
    """决策树节点"""

    __slots__ = ("feat", "thresh", "left", "right", "label", "dist", "n_samples")

    def __init__(self):
        self.feat: Optional[str] = None
        self.thresh: Optional[float] = None
        self.left: Optional["Node"] = None
        self.right: Optional["Node"] = None
        self.label: Optional[int] = None  # 叶节点：预测标签
        self.dist: Optional[Dict] = None  # 叶节点：类分布
        self.n_samples: int = 0


def gini(labels: List[int]) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    cnt = Counter(labels)
    return 1.0 - sum((c / n) ** 2 for c in cnt.values())


def weighted_gini(left: List[int], right: List[int]) -> float:
    n = len(left) + len(right)
    if n == 0:
        return 0.0
    return (len(left) * gini(left) + len(right) * gini(right)) / n


def best_split(
    X: List[Dict], y: List[int], features: List[str]
) -> Tuple[Optional[str], Optional[float], float]:
    """返回 (feature, threshold, best_weighted_gini)"""
    best_feat, best_thresh, best_g = None, None, float("inf")
    n = len(y)
    if n == 0:
        return None, None, float("inf")

    for feat in features:
        vals = sorted(set(row[feat] for row in X))
        if len(vals) <= 1:
            continue
        # 候选分割点：相邻唯一值的中点
        thresholds = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
        for thr in thresholds:
            left_y = [y[i] for i, row in enumerate(X) if row[feat] <= thr]
            right_y = [y[i] for i, row in enumerate(X) if row[feat] > thr]
            if len(left_y) < 1 or len(right_y) < 1:
                continue
            g = weighted_gini(left_y, right_y)
            if g < best_g:
                best_g = g
                best_feat = feat
                best_thresh = thr

    return best_feat, best_thresh, best_g


def build_tree(
    X: List[Dict],
    y: List[int],
    features: List[str],
    max_depth: int,
    min_samples_split: int = 10,
    depth: int = 0,
) -> Node:
    node = Node()
    node.n_samples = len(y)
    cnt = Counter(y)
    node.label = cnt.most_common(1)[0][0]
    node.dist = dict(cnt)

    # 终止条件
    if depth >= max_depth or len(set(y)) == 1 or len(y) < min_samples_split:
        return node

    feat, thresh, g = best_split(X, y, features)
    if feat is None or g >= gini(y) - 1e-9:
        return node

    left_mask = [row[feat] <= thresh for row in X]
    Xl = [X[i] for i, m in enumerate(left_mask) if m]
    yl = [y[i] for i, m in enumerate(left_mask) if m]
    Xr = [X[i] for i, m in enumerate(left_mask) if not m]
    yr = [y[i] for i, m in enumerate(left_mask) if not m]

    if not Xl or not Xr:
        return node

    node.feat = feat
    node.thresh = thresh
    node.left = build_tree(Xl, yl, features, max_depth, min_samples_split, depth + 1)
    node.right = build_tree(Xr, yr, features, max_depth, min_samples_split, depth + 1)
    return node


def predict_one(node: Node, row: Dict) -> int:
    while node.feat is not None:
        if row[node.feat] <= node.thresh:
            node = node.left
        else:
            node = node.right
    return node.label


def accuracy(node: Node, X: List[Dict], y: List[int]) -> float:
    correct = sum(predict_one(node, row) == label for row, label in zip(X, y))
    return correct / len(y) if y else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# if-else 规则打印（支持 SV 风格注释）
# ─────────────────────────────────────────────────────────────────────────────


def _thresh_int(thr: float, feat: str) -> str:
    """输出整数阈值（tok/n_active 均为整数）"""
    v = int(thr) if thr == int(thr) else thr
    return str(v)


def print_tree(
    node: Node,
    label_map: Dict[int, str],
    feat_map: Optional[Dict[str, str]] = None,
    indent: str = "",
    prefix: str = "",
    file=sys.stdout,
):
    """递归打印 if-else 规则树"""
    if feat_map is None:
        feat_map = {}
    fname = feat_map.get(node.feat, node.feat) if node.feat else None

    if node.feat is None:
        # 叶节点
        lbl = label_map.get(node.label, str(node.label))
        dist_str = ", ".join(
            f"{label_map.get(k,k)}:{v}"
            for k, v in sorted(node.dist.items(), key=lambda x: -x[1])
        )
        pct = node.dist.get(node.label, 0) / node.n_samples * 100
        print(
            f"{indent}{prefix}→ {lbl}  "
            f"[n={node.n_samples}, {pct:.0f}%]  ({dist_str})",
            file=file,
        )
        return

    thr_s = _thresh_int(node.thresh, node.feat)
    print(f"{indent}{prefix}if {fname} <= {thr_s}:", file=file)
    print_tree(node.left, label_map, feat_map, indent + "    ", "", file=file)
    print(f"{indent}else:  # {fname} > {thr_s}", file=file)
    print_tree(node.right, label_map, feat_map, indent + "    ", "", file=file)


def emit_sv_case(
    node: Node,
    label_map: Dict[int, str],
    sv_feat: Dict[str, str],
    assign_var: str,
    sv_vals: Dict[int, str],
    indent: str = "        ",
    file=sys.stdout,
):
    """生成 SystemVerilog always_comb 可综合代码段（嵌套 if-else）"""

    def _sv(n: Node, ind: str):
        if n.feat is None:
            sv_val = sv_vals.get(n.label, str(n.label))
            pct = n.dist.get(n.label, 0) / n.n_samples * 100
            print(
                f"{ind}{assign_var} = {sv_val};  // n={n.n_samples}, {pct:.0f}%",
                file=file,
            )
            return
        sf = sv_feat.get(n.feat, n.feat)
        thr = int(n.thresh) if n.thresh == int(n.thresh) else n.thresh
        print(f"{ind}if ({sf} <= {thr}) begin", file=file)
        _sv(n.left, ind + "    ")
        print(f"{ind}end else begin", file=file)
        _sv(n.right, ind + "    ")
        print(f"{ind}end", file=file)

    _sv(node, indent)


# ─────────────────────────────────────────────────────────────────────────────
# 加载数据
# ─────────────────────────────────────────────────────────────────────────────

CSV_PATH = "Idea_Model/hw_training_data.csv"

print(f"Loading {CSV_PATH} ...")
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    raw = list(reader)

print(f"  Loaded {len(raw)} rows")

# 转为整数
INT_COLS = [
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
data: List[Dict] = []
for row in raw:
    r = {k: int(row[k]) for k in INT_COLS}
    # 衍生特征：tok 差值和 tok0 是否 ≥4（SPLIT 阈值）
    r["tok_diff"] = r["tok0"] - r["tok1"]  # 正：tok0 更热
    r["tok_ratio4"] = 1 if r["tok0"] >= 4 else 0  # 1=tok0 达到 SPLIT 阈值
    data.append(r)

# ── action distribution ───────────────────────────────────────────────────────
act_cnt = Counter(r["act"] for r in data)
total = len(data)
print("\n=== 动作分布 ===")
for a, nm in ACT_NAMES.items():
    print(f"  {nm:12s}: {act_cnt[a]:6d}  ({act_cnt[a]/total*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# Tree 1: 动作预测（过滤 PREFETCH/WAIT，因 RTL 不做 pf 决策）
# ─────────────────────────────────────────────────────────────────────────────

# 先看看是否需要过滤 n_active==0 或 PREFETCH（act==4）
# RTL 只在 both_idle==1 时做 action 决策，所以训练时可以只取 both_idle==1 的子集
# 但 n_active==1 时也会有 action 决策（SINGLE_C2/SPLIT），所以不过滤 n_active

# 过滤 PREFETCH（act==4）：RTL 不需要 pf 预测，这里专注 0-3
act_data = [r for r in data if r["act"] != 4]
print(f"\n过滤 PREFETCH 后: {len(act_data)} 行（用于动作预测树）")

X_act = [{f: r[f] for f in FEATURES} for r in act_data]
y_act = [r["act"] for r in act_data]

print("\n训练动作决策树 (max_depth=5) ...")
act_tree = build_tree(X_act, y_act, FEATURES, max_depth=5, min_samples_split=5)
acc_act = accuracy(act_tree, X_act, y_act)
print(f"  训练集准确率: {acc_act*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# Tree 2: shape 预测（只取 shape != -1 的行，即该 cluster 确实有任务）
# ─────────────────────────────────────────────────────────────────────────────

shape_trees = {}
shape_acc = {}
for col in ["c2_s1", "c2_s3", "c3_s1", "c3_s3"]:
    sub = [r for r in data if r[col] != -1]
    Xs = [{f: r[f] for f in FEATURES} for r in sub]
    ys = [r[col] for r in sub]
    print(f"\n训练 {col} shape 树 (max_depth=4, n={len(sub)}) ...")
    t = build_tree(Xs, ys, FEATURES, max_depth=4, min_samples_split=5)
    acc = accuracy(t, Xs, ys)
    shape_trees[col] = t
    shape_acc[col] = acc
    print(f"  训练集准确率: {acc*100:.1f}%")
    dist_cnt = Counter(ys)
    for v, nm in SHAPE_NAMES.items():
        if v in dist_cnt:
            print(f"    {nm}: {dist_cnt[v]/len(ys)*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 打印规则
# ─────────────────────────────────────────────────────────────────────────────

FEAT_DISPLAY = {
    "tok0": "tok0",
    "tok1": "tok1",
    "tok2": "tok2",
    "n_active": "n_active",
    "both_idle": "both_idle",
    "c2_rank": "c2_rank",
    "c3_rank": "c3_rank",
}

print("\n" + "=" * 72)
print("  动作决策树 if-else 规则")
print("=" * 72)
print_tree(act_tree, ACT_NAMES, FEAT_DISPLAY)

print("\n" + "=" * 72)
print("  C2 S1 Shape 决策树 if-else 规则")
print("=" * 72)
print_tree(shape_trees["c2_s1"], SHAPE_NAMES, FEAT_DISPLAY)

print("\n" + "=" * 72)
print("  C2 S3 Shape 决策树 if-else 规则")
print("=" * 72)
print_tree(shape_trees["c2_s3"], SHAPE_NAMES, FEAT_DISPLAY)

print("\n" + "=" * 72)
print("  C3 S1 Shape 决策树 if-else 规则")
print("=" * 72)
print_tree(shape_trees["c3_s1"], SHAPE_NAMES, FEAT_DISPLAY)

print("\n" + "=" * 72)
print("  C3 S3 Shape 决策树 if-else 规则")
print("=" * 72)
print_tree(shape_trees["c3_s3"], SHAPE_NAMES, FEAT_DISPLAY)

# ─────────────────────────────────────────────────────────────────────────────
# 生成 SystemVerilog 片段（打印到终端，方便复制进 moe_hw_scheduler.sv）
# ─────────────────────────────────────────────────────────────────────────────

SV_FEAT = {
    "tok0": "top0_ntok",
    "tok1": "top1_ntok",
    "tok2": "top2_ntok",
    "tok_diff": "(top0_ntok - top1_ntok)",
    "tok_ratio4": "(top0_ntok >= 4)",
    "n_active": "q_rem",
    "both_idle": "both_idle",
    "c2_rank": "c2_cache_rank",
    "c3_rank": "c3_cache_rank",
}

ACT_SV = {
    0: "ACT_PAIR",
    1: "ACT_SPLIT",
    2: "ACT_SNGL_C2",
    3: "ACT_SNGL_C3",
    4: "ACT_WAIT",
}

SHAPE_SV = {0: "SH_A", 1: "SH_B", 2: "SH_C"}

print("\n" + "=" * 72)
print("  SystemVerilog: 动作决策（替换 moe_hw_scheduler.sv 中的手写规则）")
print("=" * 72)
print("always_comb begin")
print("    action = ACT_WAIT;  // default")
emit_sv_case(act_tree, ACT_NAMES, SV_FEAT, "action", ACT_SV, indent="    ")
print("end")

print("\n" + "=" * 72)
print("  SystemVerilog: C2 S1 shape 选择")
print("=" * 72)
print("always_comb begin")
print("    c2_s1_shape = SH_B;  // default")
emit_sv_case(
    shape_trees["c2_s1"], SHAPE_NAMES, SV_FEAT, "c2_s1_shape", SHAPE_SV, indent="    "
)
print("end")

print("\n" + "=" * 72)
print("  SystemVerilog: C2 S3 shape 选择")
print("=" * 72)
print("always_comb begin")
print("    c2_s3_shape = SH_B;  // default")
emit_sv_case(
    shape_trees["c2_s3"], SHAPE_NAMES, SV_FEAT, "c2_s3_shape", SHAPE_SV, indent="    "
)
print("end")

print("\n" + "=" * 72)
print("  SystemVerilog: C3 S1 shape 选择")
print("=" * 72)
print("always_comb begin")
print("    c3_s1_shape = SH_B;  // default")
emit_sv_case(
    shape_trees["c3_s1"], SHAPE_NAMES, SV_FEAT, "c3_s1_shape", SHAPE_SV, indent="    "
)
print("end")

print("\n" + "=" * 72)
print("  SystemVerilog: C3 S3 shape 选择")
print("=" * 72)
print("always_comb begin")
print("    c3_s3_shape = SH_B;  // default")
emit_sv_case(
    shape_trees["c3_s3"], SHAPE_NAMES, SV_FEAT, "c3_s3_shape", SHAPE_SV, indent="    "
)
print("end")

# ─────────────────────────────────────────────────────────────────────────────
# 保存 Python if-else 到 hw_rules.py 供验证
# ─────────────────────────────────────────────────────────────────────────────

import io

buf = io.StringIO()
print(
    '"""hw_rules.py — auto-generated from train_dt.py, do not edit manually"""',
    file=buf,
)
print("", file=buf)
print(
    "def predict_action(tok0, tok1, tok2, n_active, both_idle, c2_rank, c3_rank):",
    file=buf,
)
print(
    '    """Return action label: 0=PAIR 1=SPLIT 2=SINGLE_C2 3=SINGLE_C3 4=PREFETCH"""',
    file=buf,
)
print("    tok_diff = tok0 - tok1", file=buf)
print("    tok_ratio4 = 1 if tok0 >= 4 else 0", file=buf)
print("    row = dict(tok0=tok0, tok1=tok1, tok2=tok2, tok_diff=tok_diff,", file=buf)
print("               tok_ratio4=tok_ratio4, n_active=n_active,", file=buf)
print("               both_idle=both_idle, c2_rank=c2_rank, c3_rank=c3_rank)", file=buf)


def node_to_py(n: Node, label_map, indent="    "):
    lines = []
    if n.feat is None:
        lbl = label_map.get(n.label, str(n.label))
        lines.append(f"{indent}return {n.label}  # {lbl} n={n.n_samples}")
    else:
        thr = int(n.thresh) if n.thresh == int(n.thresh) else n.thresh
        lines.append(f"{indent}if row['{n.feat}'] <= {thr}:")
        lines.extend(node_to_py(n.left, label_map, indent + "    "))
        lines.append(f"{indent}else:")
        lines.extend(node_to_py(n.right, label_map, indent + "    "))
    return lines


for ln in node_to_py(act_tree, ACT_NAMES, "    "):
    print(ln, file=buf)

for col in ["c2_s1", "c2_s3", "c3_s1", "c3_s3"]:
    param_names = [
        "tok0",
        "tok1",
        "tok2",
        "n_active",
        "both_idle",
        "c2_rank",
        "c3_rank",
    ]
    print(f"\ndef predict_{col}({', '.join(param_names)}):", file=buf)
    print(f'    """Return shape label: 0=ShapeA 1=ShapeB 2=ShapeC"""', file=buf)
    print("    tok_diff = tok0 - tok1", file=buf)
    print("    tok_ratio4 = 1 if tok0 >= 4 else 0", file=buf)
    print(
        "    row = dict(tok0=tok0, tok1=tok1, tok2=tok2, tok_diff=tok_diff,", file=buf
    )
    print("               tok_ratio4=tok_ratio4, n_active=n_active,", file=buf)
    print(
        "               both_idle=both_idle, c2_rank=c2_rank, c3_rank=c3_rank)",
        file=buf,
    )
    for ln in node_to_py(shape_trees[col], SHAPE_NAMES, "    "):
        print(ln, file=buf)

rules_path = "Idea_Model/hw_rules.py"
with open(rules_path, "w") as f:
    f.write(buf.getvalue())
print(f"\n✓ hw_rules.py 已写入: {rules_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 快速验证: 用 hw_rules.py 预测，对比 training 标签
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 快速验证（training set）===")
import importlib.util, os

spec = importlib.util.spec_from_file_location("hw_rules", rules_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

correct_act = sum(
    mod.predict_action(
        r["tok0"],
        r["tok1"],
        r["tok2"],
        r["n_active"],
        r["both_idle"],
        r["c2_rank"],
        r["c3_rank"],
    )
    == r["act"]
    for r in act_data
)
print(
    f"  动作准确率: {correct_act}/{len(act_data)} = {correct_act/len(act_data)*100:.1f}%"
)

for col in ["c2_s1", "c2_s3", "c3_s1", "c3_s3"]:
    sub = [r for r in data if r[col] != -1]
    fn = getattr(mod, f"predict_{col}")
    correct = sum(
        fn(
            r["tok0"],
            r["tok1"],
            r["tok2"],
            r["n_active"],
            r["both_idle"],
            r["c2_rank"],
            r["c3_rank"],
        )
        == r[col]
        for r in sub
    )
    print(f"  {col} 准确率: {correct}/{len(sub)} = {correct/len(sub)*100:.1f}%")

print("\n✓ 训练完成")
