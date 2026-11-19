"""v20：分箱/样条加性逻辑回归成员实验（模型族对照）。"""
import sys
import time
import warnings
from datetime import datetime
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

OUT_MEMBERS = PROJECT / "outputs" / "ensemble" / "members"
OUT_MEMBERS.mkdir(parents=True, exist_ok=True)

# ===== 由 additive_logit_grid.py 的预注册规则选定（由编码网格确定） =====
CHOSEN_ENCODING = "S32"  # additive_logit_grid.py 预注册规则选定（简约带内取参数更少者）
assert CHOSEN_ENCODING in ("B64", "S16", "S32"), \
    "先跑 additive_logit_grid.py 并填入选中编码"
PRIMARY_C = 0.1

ENCODINGS = {"B64": ("bin", 64), "S16": ("spline", 16), "S32": ("spline", 32)}
SMOOTH = ["amount_usd", "hours_since_last_txn", "distance_from_home_km",
          "account_balance_usd", "velocity_score", "merchant_risk_score",
          "customer_age", "card_age_months"]
DISCRETE = ["txn_count_last_24h", "cvv_retry_count", "prior_disputes"]
CYCLIC = ["time_of_day_hour", "day_of_week"]
FEATURES = (SMOOTH + DISCRETE + CYCLIC + config.CATEGORICAL_COLS
            + config.BOOLEAN_COLS)


def make_transformer():
    kind, res = ENCODINGS[CHOSEN_ENCODING]
    if kind == "bin":
        first = ("smooth", KBinsDiscretizer(
            n_bins=res, encode="onehot-dense", strategy="quantile",
            subsample=None), SMOOTH)
    else:
        first = ("smooth", SplineTransformer(
            n_knots=res, degree=3, knots="quantile",
            extrapolation="linear", include_bias=False), SMOOTH)
    return ColumnTransformer([
        first,
        ("disc", OneHotEncoder(handle_unknown="ignore",
                               sparse_output=False), DISCRETE),
        ("cyc", OneHotEncoder(handle_unknown="ignore",
                              sparse_output=False), CYCLIC),
        ("cat", OneHotEncoder(handle_unknown="ignore",
                              sparse_output=False), config.CATEGORICAL_COLS),
        ("bool", "passthrough", config.BOOLEAN_COLS),
    ], sparse_threshold=0.0), StandardScaler()


train = pd.read_csv(config.TRAIN_PATH)
test = pd.read_csv(config.TEST_PATH)
y = train[config.TARGET].to_numpy()
X = train[FEATURES]
X_test = test[FEATURES]
assert len(FEATURES) == 24

CAP = None  # 无早停：逻辑回归按 lbfgs 收敛


def run_cv(seed, want_test):
    t0 = time.time()
    folds = list(StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=seed).split(X, y))
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(test)) if want_test else None
    train_aucs, val_aucs, n_iters, n_cols = [], [], [], None
    for k, (tr_idx, va_idx) in enumerate(folds, start=1):
        ct, scaler = make_transformer()
        Z_tr = scaler.fit_transform(ct.fit_transform(X.iloc[tr_idx]))
        Z_va = scaler.transform(ct.transform(X.iloc[va_idx]))
        Z_tr = np.asarray(Z_tr, dtype=np.float32)
        Z_va = np.asarray(Z_va, dtype=np.float32)
        n_cols = Z_tr.shape[1]
        model = LogisticRegression(C=PRIMARY_C, solver="lbfgs",
                                   max_iter=500, tol=1e-4)
        model.fit(Z_tr, y[tr_idx])
        p_tr = model.predict_proba(Z_tr)[:, 1]
        p_va = model.predict_proba(Z_va)[:, 1]
        oof[va_idx] = p_va
        train_aucs.append(roc_auc_score(y[tr_idx], p_tr))
        val_aucs.append(roc_auc_score(y[va_idx], p_va))
        n_iters.append(int(model.n_iter_[0]))
        if want_test:
            Z_te = np.asarray(scaler.transform(ct.transform(X_test)),
                              dtype=np.float32)
            test_pred += model.predict_proba(Z_te)[:, 1] / config.N_FOLDS
        print(f"  [seed={seed}] fold {k}: p={n_cols} n_iter={n_iters[-1]} "
              f"train={train_aucs[-1]:.6f} val={val_aucs[-1]:.6f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return dict(oof=oof, test_pred=test_pred, train_aucs=train_aucs,
                val_aucs=val_aucs, n_iters=n_iters, n_cols=n_cols,
                secs=round(time.time() - t0, 1))


def four_piece(r):
    mt, mv = float(np.mean(r["train_aucs"])), float(np.mean(r["val_aucs"]))
    return dict(mean_train=mt, mean_val=mv, gap=mt - mv,
                val_std=float(np.std(r["val_aucs"], ddof=0)),
                oof=float(roc_auc_score(y, r["oof"])))


def paired(a_aucs, b_aucs):
    d = np.asarray(a_aucs) - np.asarray(b_aucs)
    sd = float(d.std(ddof=1))
    t = float(d.mean() / (sd / np.sqrt(config.N_FOLDS))) if sd > 0 else float("nan")
    return float(d.mean()), sd, t, int((d > 0).sum())


def baseline_aucs(seed):
    folds = list(StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=seed).split(X, y))
    if seed == 42:
        d1 = pd.read_csv(PROJECT / "outputs" / "v17_oof.csv")
        d2 = pd.read_csv(PROJECT / "outputs" / "ensemble" / "d2_seed42_oof.csv")
        p_d2 = d2["d2_oof"].to_numpy()
    else:
        d1 = pd.read_csv(OUT_MEMBERS / f"oof_d1_seed{seed}.csv")
        d2 = pd.read_csv(OUT_MEMBERS / f"oof_d2_seed{seed}.csv")
        p_d2 = d2["oof_prob"].to_numpy()
    assert d1[config.ID_COL].equals(train[config.ID_COL])
    p_d1 = d1["oof_prob"].to_numpy()
    p_v18 = 0.5 * p_d1 + 0.5 * p_d2
    return (
        [roc_auc_score(y[va], p_d1[va]) for _, va in folds],
        [roc_auc_score(y[va], p_v18[va]) for _, va in folds],
    )


results = {}
print("=" * 70)
print(f"v20 additive logit enc={CHOSEN_ENCODING} C={PRIMARY_C} seed 42 (official)")
print("=" * 70)
r42 = run_cv(42, want_test=True)
fp42 = four_piece(r42)
results[42] = (r42, fp42)

sub_name = f"v20_additive_logit_{CHOSEN_ENCODING.lower()}.csv"
sub_path = PROJECT / "submissions" / sub_name
assert r42["test_pred"] is not None
assert 0.0 < r42["test_pred"].min() and r42["test_pred"].max() < 1.0
pd.DataFrame({config.ID_COL: test[config.ID_COL],
              config.TARGET: r42["test_pred"]}).to_csv(sub_path, index=False)
chk = pd.read_csv(sub_path)
assert list(chk.columns) == ["id", "is_fraud"]
assert len(chk) == 299984
assert chk["id"].equals(test[config.ID_COL])
assert chk["is_fraud"].between(0, 1).all()
assert not chk["is_fraud"].isin([0.0, 1.0]).any()

pd.DataFrame({
    config.ID_COL: train[config.ID_COL], config.TARGET: y,
    "oof_prob": r42["oof"],
}).to_csv(PROJECT / "outputs" / "v20_oof.csv", index=False)
pd.DataFrame({
    config.ID_COL: train[config.ID_COL], config.TARGET: y,
    "oof_prob": r42["oof"],
}).to_csv(OUT_MEMBERS / "oof_lg_seed42.csv", index=False)
pd.DataFrame({
    config.ID_COL: test[config.ID_COL], config.TARGET: r42["test_pred"],
}).to_csv(OUT_MEMBERS / "test_lg_seed42.csv", index=False)

print("\n" + "=" * 70)
print(f"v20 additive logit enc={CHOSEN_ENCODING} C={PRIMARY_C} seed 123 (confirmation)")
print("=" * 70)
r123 = run_cv(123, want_test=False)
fp123 = four_piece(r123)
results[123] = (r123, fp123)
pd.DataFrame({
    config.ID_COL: train[config.ID_COL], config.TARGET: y,
    "oof_prob": r123["oof"],
}).to_csv(OUT_MEMBERS / "oof_lg_seed123.csv", index=False)

print("\n===== v20 four-piece =====")
print(f"seed 42 : train={fp42['mean_train']:.6f} val={fp42['mean_val']:.6f} "
      f"gap={fp42['gap']:.6f} std={fp42['val_std']:.6f} OOF={fp42['oof']:.6f}")
print(f"seed 123: train={fp123['mean_train']:.6f} val={fp123['mean_val']:.6f} "
      f"gap={fp123['gap']:.6f} std={fp123['val_std']:.6f} OOF={fp123['oof']:.6f}")

pair_notes = []
for seed, (r, fp) in results.items():
    v17, v18 = baseline_aucs(seed)
    for label, base in (("v17", v17), ("v18", v18)):
        d_bar, sd, t, pos = paired(r["val_aucs"], base)
        print(f"seed {seed} vs {label}: d_bar={d_bar:+.6f} s_d={sd:.6f} "
              f"t={t:+.3f} positive={pos}/5")
        pair_notes.append(f"seed{seed} vs {label}: d={d_bar:+.6f} t={t:+.2f} "
                          f"pos={pos}/5")

notes = (
    f"member candidate (family=additive logistic regression); 24 raw features; "
    f"encoding={CHOSEN_ENCODING}, C={PRIMARY_C} (L2, lbfgs), design cols={r42['n_cols']}, "
    f"all preprocessing fit inside each fold (label-free); seed42 four-piece: "
    f"train={fp42['mean_train']:.6f} val={fp42['mean_val']:.6f} gap={fp42['gap']:.6f} "
    f"std={fp42['val_std']:.6f} OOF={fp42['oof']:.6f}; seed123 OOF={fp123['oof']:.6f}; "
    f"n_iters={r42['n_iters']} / {r123['n_iters']}; pairs: {'; '.join(pair_notes)}; "
    f"single model weaker than v17 by design, enters level-2 stack only; does not "
    f"change model baseline"
)
row = {
    "experiment_id": "v20",
    "model": "logreg_additive_1.9.1",
    "feature_summary": (
        f"24 raw features; additive logistic regression, continuous columns "
        f"encoded as {CHOSEN_ENCODING}, discrete/cyclic/categorical one-hot, "
        f"booleans raw; family member for v21"
    ),
    "n_folds": config.N_FOLDS,
    "mean_train_auc": round(fp42["mean_train"], 6),
    "mean_val_auc": round(fp42["mean_val"], 6),
    "train_val_gap": round(fp42["gap"], 6),
    "val_auc_std": round(fp42["val_std"], 6),
    "submission_file": sub_name,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "notes": notes,
}
pd.DataFrame([row]).to_csv(config.EXPERIMENT_LOG_PATH, mode="a",
                           header=False, index=False)

print(f"\nsubmission -> {sub_path}")
print(f"members -> {OUT_MEMBERS}")
print("v20 done.")
