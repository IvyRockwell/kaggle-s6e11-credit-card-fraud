"""产物一致性校验：用 OOF 文件重算 v19–v23 的折均值与整体 AUC，
并检查提交文件的表头、行数、id 对齐与概率范围。"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import config  # noqa: E402

MEMBERS = PROJECT / "outputs" / "ensemble" / "members"
TOL_LOG = 1e-6
TOL_NOTE = 5e-7
failures = []


def check(cond, label, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(label)


def load_oof(path, col="oof_prob"):
    df = pd.read_csv(path)
    check(df[config.ID_COL].equals(train[config.ID_COL]),
          f"id 对齐: {Path(path).name}")
    p = df[col].to_numpy(dtype=float)
    check(np.isfinite(p).all(), f"无 NaN: {Path(path).name}")
    check((p > 0).all() and (p < 1).all(), f"概率∈(0,1): {Path(path).name}")
    return p


def fold_aucs(p, seed):
    folds = list(StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y))
    return np.array([roc_auc_score(y[va], p[va]) for _, va in folds])


train = pd.read_csv(config.TRAIN_PATH)
test = pd.read_csv(config.TEST_PATH)
y = train[config.TARGET].to_numpy()
log = pd.read_csv(config.EXPERIMENT_LOG_PATH)
print(f"experiment_log rows: {len(log)}; ids={list(log['experiment_id'])}")

for vid in ("v19", "v20", "v21"):
    rows = log[log["experiment_id"] == vid]
    check(len(rows) == 1, f"日志行唯一: {vid}", f"count={len(rows)}")

# ---------- v18 基线重建 ----------
p_d1_42 = load_oof(PROJECT / "outputs" / "v17_oof.csv")
p_d2_42 = load_oof(PROJECT / "outputs" / "ensemble" / "d2_seed42_oof.csv",
                   col="d2_oof")
v18_42 = 0.5 * p_d1_42 + 0.5 * p_d2_42
check(abs(roc_auc_score(y, v18_42) - 0.948114) < TOL_NOTE,
      "v18 seed42 OOF=0.948114", f"recomputed={roc_auc_score(y, v18_42):.6f}")

# ---------- v19 / v20：OOF 与日志 ----------
MEMBER_OF = {"v19": "cb", "v20": "lg", "v22": "xgb"}
for vid, oof_path in (
    ("v19", PROJECT / "outputs" / "v19_oof.csv"),
    ("v20", PROJECT / "outputs" / "v20_oof.csv"),
    ("v22", PROJECT / "outputs" / "v22_oof.csv"),
    ("v23", PROJECT / "outputs" / "v23_oof.csv"),
):
    row = log[log["experiment_id"] == vid].iloc[0]
    p = load_oof(oof_path)
    fa = fold_aucs(p, 42)
    oof_auc = float(roc_auc_score(y, p))
    check(abs(fa.mean() - float(row["mean_val_auc"])) < TOL_LOG,
          f"{vid} mean_val_auc 与日志一致",
          f"{fa.mean():.6f} vs {row['mean_val_auc']}")
    m = re.search(r"OOF=([0-9.]+)", str(row["notes"]))
    check(m is not None and abs(oof_auc - float(m.group(1))) < TOL_NOTE,
          f"{vid} OOF AUC 与 notes 一致", f"{oof_auc:.6f}")
    std = float(fa.std(ddof=0))
    check(abs(std - float(row["val_auc_std"])) < TOL_LOG,
          f"{vid} val_auc_std 与日志一致", f"{std:.6f}")
    # 成员 OOF（seed 42）必须与正式 OOF 完全相同（v23 是多种子平均，无单一成员）
    if vid in MEMBER_OF:
        member = MEMBERS / f"oof_{MEMBER_OF[vid]}_seed42.csv"
        p_m = load_oof(member)
        check(np.max(np.abs(p_m - p)) < 1e-12, f"{vid} 成员 OOF=正式 OOF")

# ---------- v21 嵌套 OOF 与日志 ----------
row21 = log[log["experiment_id"] == "v21"].iloc[0]
p21 = load_oof(PROJECT / "outputs" / "v21_oof.csv")
fa21 = fold_aucs(p21, 42)
check(abs(fa21.mean() - float(row21["mean_val_auc"])) < TOL_LOG,
      "v21 mean_val_auc 与日志一致", f"{fa21.mean():.6f} vs {row21['mean_val_auc']}")
print(f"       v21 OOF={roc_auc_score(y, p21):.6f} (notes 记录)")
print(f"       v21 notes: {row21['notes'][:200]}...")

# ---------- 成员 OOF 覆盖矩阵 ----------
print("\nmember OOF matrix:")
MEMBER_FILES = {
    ("d1", 42): (PROJECT / "outputs" / "v17_oof.csv", "oof_prob"),
    ("d2", 42): (PROJECT / "outputs" / "ensemble" / "d2_seed42_oof.csv", "d2_oof"),
}
for m in ("d1", "d2", "cb", "lg", "xgb"):
    aucs = {}
    for seed in (42, 123):
        path, col = MEMBER_FILES.get(
            (m, seed), (MEMBERS / f"oof_{m}_seed{seed}.csv", "oof_prob"))
        check(path.exists(), f"成员文件存在: {path.name}")
        p = load_oof(path, col=col)
        aucs[seed] = roc_auc_score(y, p)
    print(f"  {m}: seed42={aucs[42]:.6f} seed123={aucs[123]:.6f}")

# ---------- 提交文件 ----------
print("\nsubmissions:")
for name in ("v19_catboost_d2.csv", "v19_catboost_d3.csv", "v19_catboost_d4.csv",
             "v20_additive_logit_b64.csv", "v20_additive_logit_s16.csv",
             "v20_additive_logit_s32.csv", "v21_stack_level2.csv",
             "v22_xgboost_d2.csv", "v23_v18_seedavg_42_666_789.csv"):
    path = PROJECT / "submissions" / name
    if not path.exists():
        continue
    df = pd.read_csv(path)
    check(list(df.columns) == ["id", "is_fraud"], f"表头: {name}")
    check(len(df) == 299984, f"行数: {name}", f"{len(df)}")
    check(df["id"].equals(test[config.ID_COL]), f"id 逐行: {name}")
    check(df["is_fraud"].between(0, 1).all(), f"概率∈[0,1]: {name}")
    check(not df["is_fraud"].isin([0.0, 1.0]).any(), f"未取整: {name}")
    check(df["is_fraud"].nunique() > 1000, f"连续概率: {name}",
          f"nunique={df['is_fraud'].nunique()}")

print("\n" + ("ALL CHECKS PASSED" if not failures else
              f"FAILURES ({len(failures)}): {failures}"))
sys.exit(1 if failures else 0)
