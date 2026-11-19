"""加性逻辑回归的编码网格：quantile 分箱与 B 样条（seed 42 选型，seed 123 确认）。"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import (
    KBinsDiscretizer, OneHotEncoder, SplineTransformer, StandardScaler,
)

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import config  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUT = PROJECT / "outputs" / "additive_logit"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN = pd.read_csv(config.TRAIN_PATH)
y = TRAIN[config.TARGET].to_numpy()

SMOOTH = ["amount_usd", "hours_since_last_txn", "distance_from_home_km",
          "account_balance_usd", "velocity_score", "merchant_risk_score",
          "customer_age", "card_age_months"]
DISCRETE = ["txn_count_last_24h", "cvv_retry_count", "prior_disputes"]
CYCLIC = ["time_of_day_hour", "day_of_week"]
CAT = config.CATEGORICAL_COLS
BOOL = config.BOOLEAN_COLS
FEATURES = SMOOTH + DISCRETE + CYCLIC + CAT + BOOL
X = TRAIN[FEATURES]
assert len(FEATURES) == 24

ENCODINGS = {
    # name: (kind, resolution)
    "B64": ("bin", 64),
    "S16": ("spline", 16),
    "S32": ("spline", 32),
}
PARAM_COUNT = {"S16": 1, "S32": 2, "B64": 3}  # 简约优先顺序（参数越少越靠前）
PRIMARY_C = 0.1
SENSITIVITY_C = (0.03, 0.3)


def make_transformer(kind, res):
    if kind == "bin":
        first = ("smooth", KBinsDiscretizer(
            n_bins=res, encode="onehot-dense", strategy="quantile",
            subsample=None), SMOOTH)
    else:
        first = ("smooth", SplineTransformer(
            n_knots=res, degree=3, knots="quantile",
            extrapolation="linear", include_bias=False), SMOOTH)
    ct = ColumnTransformer([
        first,
        ("disc", OneHotEncoder(handle_unknown="ignore",
                               sparse_output=False), DISCRETE),
        ("cyc", OneHotEncoder(handle_unknown="ignore",
                              sparse_output=False), CYCLIC),
        ("cat", OneHotEncoder(handle_unknown="ignore",
                              sparse_output=False), CAT),
        ("bool", "passthrough", BOOL),
    ], sparse_threshold=0.0)
    return ct, StandardScaler()


def run_config(encoding, C, seed, save_oof=True):
    kind, res = ENCODINGS[encoding]
    t0 = time.time()
    folds = list(
        StratifiedKFold(
            n_splits=config.N_FOLDS, shuffle=True, random_state=seed
        ).split(X, y)
    )
    oof = np.zeros(len(y))
    train_aucs, val_aucs, n_iters, n_features = [], [], [], None
    for k, (tr_idx, va_idx) in enumerate(folds, start=1):
        ct, scaler = make_transformer(kind, res)
        Z_tr = scaler.fit_transform(ct.fit_transform(X.iloc[tr_idx]))
        Z_va = scaler.transform(ct.transform(X.iloc[va_idx]))
        Z_tr = np.asarray(Z_tr, dtype=np.float32)
        Z_va = np.asarray(Z_va, dtype=np.float32)
        n_features = Z_tr.shape[1]
        model = LogisticRegression(C=C, solver="lbfgs", max_iter=500, tol=1e-4)
        model.fit(Z_tr, y[tr_idx])
        p_tr = model.predict_proba(Z_tr)[:, 1]
        p_va = model.predict_proba(Z_va)[:, 1]
        oof[va_idx] = p_va
        train_aucs.append(roc_auc_score(y[tr_idx], p_tr))
        val_aucs.append(roc_auc_score(y[va_idx], p_va))
        n_iters.append(int(model.n_iter_[0]))
        print(f"  [{encoding} C={C} seed={seed}] fold {k}: "
              f"p={n_features} n_iter={n_iters[-1]} val AUC={val_aucs[-1]:.6f}",
              flush=True)
    row = dict(
        encoding=encoding, C=C, seed=seed, n_design_cols=n_features,
        mean_train_auc=float(np.mean(train_aucs)),
        mean_val_auc=float(np.mean(val_aucs)),
        gap=float(np.mean(train_aucs) - np.mean(val_aucs)),
        val_std=float(np.std(val_aucs, ddof=0)),
        oof_auc=float(roc_auc_score(y, oof)),
        n_iters=str(n_iters), secs=round(time.time() - t0, 1),
    )
    if save_oof:
        pd.DataFrame({
            config.ID_COL: TRAIN[config.ID_COL],
            config.TARGET: y,
            "oof_prob": oof,
        }).to_csv(OUT / f"oof_lg_{encoding}_seed{seed}.csv", index=False)
    return row, np.array(val_aucs)


summary_rows, fold_rows = [], []
t_all = time.time()

print("=" * 70)
print("seed 42 grid: {B64,S16,S32} x C=0.1")
print("=" * 70)
for enc in ENCODINGS:
    row, val_aucs = run_config(enc, PRIMARY_C, 42)
    summary_rows.append(row)
    for k in range(config.N_FOLDS):
        fold_rows.append(dict(stage="grid", seed=42, encoding=enc,
                              C=PRIMARY_C, fold=k + 1, val_auc=val_aucs[k]))
    pd.DataFrame(summary_rows).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "per_fold.csv", index=False)
    print(f"grid {enc} C=0.1: val={row['mean_val_auc']:.6f} "
          f"gap={row['gap']:.6f} std={row['val_std']:.6f} "
          f"OOF={row['oof_auc']:.6f} p={row['n_design_cols']} "
          f"({row['secs']:.0f}s)", flush=True)

grid = pd.DataFrame([r for r in summary_rows if r["seed"] == 42])
# 预注册选择：mean_val_auc 最高；差距 <0.0003 取参数更少者
grid = grid.sort_values(["mean_val_auc"], ascending=False,
                        key=lambda s: s.round(6))
best_val = float(grid["mean_val_auc"].max())
tied = grid[best_val - grid["mean_val_auc"] < 0.0003]
chosen = min(tied["encoding"], key=lambda e: PARAM_COUNT[e])
print(f"\ngrid ranking (seed 42): "
      f"{dict(zip(grid['encoding'], grid['mean_val_auc'].round(6)))}")
print(f"best={grid.iloc[0]['encoding']} ({best_val:.6f}); "
      f"chosen={chosen} (parsimony band 0.0003)")

print("\n" + "=" * 70)
print(f"seed 123 confirmation: {chosen} C=0.1")
print("=" * 70)
row, val_aucs = run_config(chosen, PRIMARY_C, 123)
summary_rows.append(row)
for k in range(config.N_FOLDS):
    fold_rows.append(dict(stage="confirm", seed=123, encoding=chosen,
                          C=PRIMARY_C, fold=k + 1, val_auc=val_aucs[k]))
pd.DataFrame(summary_rows).to_csv(OUT / "summary.csv", index=False)
pd.DataFrame(fold_rows).to_csv(OUT / "per_fold.csv", index=False)
print(f"confirm {chosen} C=0.1 seed=123: val={row['mean_val_auc']:.6f} "
      f"OOF={row['oof_auc']:.6f}")

print("\n" + "=" * 70)
print(f"sensitivity (seed 42, {chosen}, not used for selection)")
print("=" * 70)
for C in SENSITIVITY_C:
    row, _ = run_config(chosen, C, 42, save_oof=False)
    summary_rows.append(row)
    print(f"sensitivity {chosen} C={C}: val={row['mean_val_auc']:.6f} "
          f"OOF={row['oof_auc']:.6f}", flush=True)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "summary.csv", index=False)
selection = pd.DataFrame([{
    "chosen_encoding": chosen, "primary_C": PRIMARY_C,
    "grid_mean_val": str(dict(zip(grid["encoding"],
                                  grid["mean_val_auc"].round(6)))),
    "rule": "max mean_val_auc on seed42; tie<0.0003 -> fewer params",
}])
selection.to_csv(OUT / "selection.csv", index=False)
print(f"\nchosen encoding={chosen}; files under {OUT}; "
      f"done in {time.time()-t_all:.0f}s")
