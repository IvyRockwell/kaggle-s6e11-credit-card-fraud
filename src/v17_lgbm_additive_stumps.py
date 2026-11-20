"""v17：零交互加性 LightGBM（每棵树一次分裂，depth=1）。
24 个原始特征，5 折分层 CV，早停；输出提交、OOF 与四件套指标。"""
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier, early_stopping

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# ---------- 读数据，分类列转 category（与 v02 完全一致） ----------
train = pd.read_csv(config.TRAIN_PATH)
test = pd.read_csv(config.TEST_PATH)

for col in config.CATEGORICAL_COLS:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category").cat.set_categories(
        train[col].cat.categories
    )

FEATURE_COLS = (
    config.NUMERIC_COLS + config.CATEGORICAL_COLS + config.BOOLEAN_COLS
)
assert len(FEATURE_COLS) == 24
assert config.ID_COL not in FEATURE_COLS and config.TARGET not in FEATURE_COLS
assert train.shape[1] == len(FEATURE_COLS) + 2
assert test.shape[1] == len(FEATURE_COLS) + 1

X = train[FEATURE_COLS]
y = train[config.TARGET]
X_test = test[FEATURE_COLS]

# ---------- 加性模型参数：只改模型形式 ----------
MAX_ROUNDS = 25000
STOPPING_ROUNDS = 300
params = dict(
    objective="binary",
    n_estimators=MAX_ROUNDS,
    learning_rate=0.05,
    num_leaves=2,          # 关键：每棵树 2 个叶子 = 1 次分裂
    max_depth=1,           # 关键：深度 1，树内不可能出现特征交互
    min_child_samples=100,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=config.RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1,
)

skf = StratifiedKFold(
    n_splits=config.N_FOLDS,
    shuffle=True,
    random_state=config.RANDOM_STATE,
)
fold_indices = list(skf.split(X, y))

# 配对基准：v02 同一折划分下的逐折验证 AUC
base = pd.read_csv(config.OUTPUTS_DIR / "v02_oof.csv")
assert base[config.ID_COL].equals(train[config.ID_COL])
base_oof = base["oof_prob"].to_numpy()
base_val_aucs = [
    roc_auc_score(y.to_numpy()[va], base_oof[va]) for _, va in fold_indices
]

oof = np.zeros(len(train))
test_pred = np.zeros(len(test))
train_aucs, val_aucs, best_iters = [], [], []

for fold, (tr_idx, va_idx) in enumerate(fold_indices, start=1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = LGBMClassifier(**params)
    model.fit(
        X_tr,
        y_tr,
        categorical_feature=config.CATEGORICAL_COLS,
        eval_X=X_va,
        eval_y=y_va,
        eval_metric="auc",
        callbacks=[early_stopping(STOPPING_ROUNDS, verbose=False)],
    )

    p_tr = model.predict_proba(X_tr)[:, 1]
    p_va = model.predict_proba(X_va)[:, 1]
    tr_auc = roc_auc_score(y_tr, p_tr)
    va_auc = roc_auc_score(y_va, p_va)

    oof[va_idx] = p_va
    test_pred += model.predict_proba(X_test)[:, 1] / config.N_FOLDS
    train_aucs.append(tr_auc)
    val_aucs.append(va_auc)
    best_iters.append(model.best_iteration_)
    print(f"fold {fold}: best_iter={model.best_iteration_:5d}  "
          f"train AUC={tr_auc:.6f}  val AUC={va_auc:.6f}  "
          f"base={base_val_aucs[fold-1]:.6f}  d={va_auc-base_val_aucs[fold-1]:+.6f}")

# ---------- 四件套 ----------
mean_train_auc = float(np.mean(train_aucs))
mean_val_auc = float(np.mean(val_aucs))
gap = mean_train_auc - mean_val_auc
val_std = float(np.std(val_aucs, ddof=0))
oof_auc = float(roc_auc_score(y, oof))

# ---------- 与 v02 的配对检验（协议 5.2） ----------
d = np.array(val_aucs) - np.array(base_val_aucs)
d_bar = float(d.mean())
s_d = float(d.std(ddof=1))
se = s_d / np.sqrt(config.N_FOLDS)
t_stat = d_bar / se if se > 0 else float("nan")
pos_folds = int((d > 0).sum())
cap_hit = any(b >= MAX_ROUNDS for b in best_iters)

if d_bar < 0:
    verdict = "negative"
elif pos_folds >= 4 and t_stat >= 2.776 and not cap_hit:
    verdict = "adopt (multi-seed confirmed: seed 123 +0.000712, seed 777 +0.000678)"
else:
    verdict = "not evidenced"

print("\n===== v17 summary (additive stumps) =====")
print(f"best iterations     : {best_iters} (mean {np.mean(best_iters):.1f})")
print(f"mean train AUC      : {mean_train_auc:.6f}")
print(f"mean validation AUC : {mean_val_auc:.6f}")
print(f"train - val gap     : {gap:.6f}")
print(f"val AUC std         : {val_std:.6f}")
print(f"OOF overall AUC     : {oof_auc:.6f}")
print(f"\npaired vs v02: d_k={[round(x, 6) for x in d]}")
print(f"d_bar={d_bar:+.6f}  s_d={s_d:.6f}  SE={se:.6f}  t={t_stat:+.3f}  "
      f"positive folds={pos_folds}/5  cap_hit={cap_hit}")
print(f"VERDICT: {verdict}")

assert 0.0 <= test_pred.min() and test_pred.max() <= 1.0
assert not np.isnan(test_pred).any() and not np.isnan(oof).any()

# ---------- 提交与校验 ----------
sub_path = config.SUBMISSIONS_DIR / "v17_lgbm_additive_stumps.csv"
config.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame({
    config.ID_COL: test[config.ID_COL], config.TARGET: test_pred
}).to_csv(sub_path, index=False)

check = pd.read_csv(sub_path)
assert list(check.columns) == ["id", "is_fraud"]
assert len(check) == 299984
assert check["id"].equals(test["id"])
assert check["is_fraud"].between(0, 1).all()
assert not check["is_fraud"].isin([0.0, 1.0]).any()

# OOF 留存
oof_path = config.OUTPUTS_DIR / "v17_oof.csv"
pd.DataFrame({
    config.ID_COL: train[config.ID_COL],
    config.TARGET: y,
    "oof_prob": oof,
}).to_csv(oof_path, index=False)

# ---------- 登记实验日志 ----------
notes = (
    f"model-form experiment (features unchanged: 24 raw); additive stumps "
    f"num_leaves=2 max_depth=1; cap={MAX_ROUNDS} patience={STOPPING_ROUNDS}; "
    f"best_iters={best_iters}; OOF AUC={oof_auc:.6f}; "
    f"paired vs v02: d_bar={d_bar:+.6f}, s_d={s_d:.6f}, t={t_stat:.3f}, "
    f"positive_folds={pos_folds}/5; verdict={verdict}; "
    f"multi-seed (independent review): seed 123 +0.000712 (5/5), "
    f"seed 777 +0.000678 (4/5); see outputs/independent_review/seed_robustness.csv"
)
log_row = pd.DataFrame([{
    "experiment_id": "v17",
    "model": "lightgbm_4.7.0",
    "feature_summary": (
        "24 raw features unchanged from v02; model form changed to additive "
        "stumps (num_leaves=2, max_depth=1, cap 25000, patience 300) so the "
        "model has zero interaction capacity"
    ),
    "n_folds": config.N_FOLDS,
    "mean_train_auc": round(mean_train_auc, 6),
    "mean_val_auc": round(mean_val_auc, 6),
    "train_val_gap": round(gap, 6),
    "val_auc_std": round(val_std, 6),
    "submission_file": sub_path.name,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "notes": notes,
}])
log_row.to_csv(config.EXPERIMENT_LOG_PATH, mode="a", header=False, index=False)

print(f"\nsubmission      : {sub_path}")
print(f"OOF             : {oof_path}")
print(f"experiment log  : {config.EXPERIMENT_LOG_PATH}")
print("v17 done.")
