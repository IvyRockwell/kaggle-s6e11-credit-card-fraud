"""历史实验 v07：四个风险布尔标志求和。"""
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

# ---------- 唯一变量：新增 risk_flag_count（四个原始布尔列保留） ----------
RISK_FLAGS = [
    "ip_country_mismatch",
    "is_ai_generated_scam_attempt",
    "billing_shipping_mismatch",
    "is_foreign_transaction",
]
for df in (train, test):
    df["risk_flag_count"] = sum(df[c].astype(int) for c in RISK_FLAGS)

base_cols = (
    config.NUMERIC_COLS + config.CATEGORICAL_COLS + config.BOOLEAN_COLS
)
FEATURE_COLS = base_cols + ["risk_flag_count"]
assert len(FEATURE_COLS) == 25
for c in RISK_FLAGS:
    assert c in FEATURE_COLS  # 四个原始布尔列必须保留
assert set(train["risk_flag_count"].unique()).issubset({0, 1, 2, 3, 4})
assert set(test["risk_flag_count"].unique()).issubset({0, 1, 2, 3, 4})
assert config.ID_COL not in FEATURE_COLS and config.TARGET not in FEATURE_COLS
assert not test[config.CATEGORICAL_COLS].isna().any().any()

X = train[FEATURE_COLS]
y = train[config.TARGET]
X_test = test[FEATURE_COLS]

# ---------- 协议参数：与 v02 完全一致，n_estimators 为上限 1000 ----------
MAX_ROUNDS = 1000
STOPPING_ROUNDS = 50
params = dict(
    objective="binary",
    n_estimators=MAX_ROUNDS,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
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

oof = np.zeros(len(train))
test_pred = np.zeros(len(test))
train_aucs, val_aucs, best_iters = [], [], []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
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
    print(f"fold {fold}: best_iter={model.best_iteration_:4d}  "
          f"train AUC={tr_auc:.6f}  val AUC={va_auc:.6f}")

# ---------- 四件套 ----------
mean_train_auc = float(np.mean(train_aucs))
mean_val_auc = float(np.mean(val_aucs))
gap = mean_train_auc - mean_val_auc
val_std = float(np.std(val_aucs, ddof=0))
oof_auc = float(roc_auc_score(y, oof))

print("\n===== v07 四件套（新增 risk_flag_count） =====")
print(f"best iterations     : {best_iters}（均值 {np.mean(best_iters):.1f}）")
print(f"mean train AUC      : {mean_train_auc:.6f}")
print(f"mean validation AUC : {mean_val_auc:.6f}")
print(f"train - val gap     : {gap:.6f}")
print(f"val AUC std         : {val_std:.6f}")
print(f"OOF overall AUC     : {oof_auc:.6f}")

assert 0.0 <= test_pred.min() and test_pred.max() <= 1.0
assert not np.isnan(test_pred).any() and not np.isnan(oof).any()

# ---------- 提交与校验 ----------
config.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
sub_path = config.SUBMISSIONS_DIR / "v07_lgbm_risk_flag_count.csv"
pd.DataFrame({
    config.ID_COL: test[config.ID_COL], config.TARGET: test_pred
}).to_csv(sub_path, index=False)

check = pd.read_csv(sub_path)
assert list(check.columns) == ["id", "is_fraud"]
assert len(check) == 299984
assert check["id"].equals(test["id"])
assert check["is_fraud"].between(0, 1).all()
assert not (check["is_fraud"].isin([0.0, 1.0])).any()  # 连续概率，禁止取整

# OOF 留存
oof_path = config.OUTPUTS_DIR / "v07_oof.csv"
pd.DataFrame({
    config.ID_COL: train[config.ID_COL],
    config.TARGET: y,
    "oof_prob": oof,
}).to_csv(oof_path, index=False)

# ---------- 登记实验日志（11 列，追加） ----------
log_row = pd.DataFrame([{
    "experiment_id": "v07",
    "model": "lightgbm_4.7.0",
    "feature_summary": "25 features: 24 raw v02 features plus risk_flag_count=sum of 4 boolean flags (ip_country_mismatch, is_ai_generated_scam_attempt, billing_shipping_mismatch, is_foreign_transaction), values 0-4; source booleans kept; G1/G2 rejected so built on v02; early stopping per frozen protocol",
    "n_folds": config.N_FOLDS,
    "mean_train_auc": round(mean_train_auc, 6),
    "mean_val_auc": round(mean_val_auc, 6),
    "train_val_gap": round(gap, 6),
    "val_auc_std": round(val_std, 6),
    "submission_file": sub_path.name,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "notes": f"G3 risk_flag_count only; best_iters={best_iters}; "
             f"OOF AUC={oof_auc:.6f}; label-free deterministic integer sum, no leakage",
}])
log_row.to_csv(config.EXPERIMENT_LOG_PATH, mode="a", header=False, index=False)

print(f"\n提交文件: {sub_path}")
print(f"OOF 文件: {oof_path}")
print(f"实验日志: {config.EXPERIMENT_LOG_PATH}")
print("v07 完成。")
