"""v22：XGBoost depth2 成员实验（模型族对照）。"""
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import config  # noqa: E402

OUT_MEMBERS = PROJECT / "outputs" / "ensemble" / "members"
OUT_MEMBERS.mkdir(parents=True, exist_ok=True)

# ===== 由 xgboost_curve.py 的事先登记规则选定（诊断结果写入） =====
CHOSEN_DEPTH = 2  # xgboost_curve.py 事先登记规则选定（两 seed 均值 0.947568）
assert CHOSEN_DEPTH in (2, 3, 4), "先跑 xgboost_curve.py 并填入选定 depth"

CAP = 20000
PATIENCE = 500
PARAMS = dict(
    n_estimators=CAP, max_depth=CHOSEN_DEPTH, learning_rate=0.1,
    tree_method="hist", enable_categorical=True, min_child_weight=1.0,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    objective="binary:logistic", eval_metric="auc",
    early_stopping_rounds=PATIENCE, random_state=config.RANDOM_STATE,
    n_jobs=-1, verbosity=0,
)

train = pd.read_csv(config.TRAIN_PATH)
test = pd.read_csv(config.TEST_PATH)
y = train[config.TARGET].to_numpy()
FEATURES = config.NUMERIC_COLS + config.CATEGORICAL_COLS + config.BOOLEAN_COLS
X = train[FEATURES].copy()
X_test = test[FEATURES].copy()
for c in config.CATEGORICAL_COLS:
    X[c] = X[c].astype("category")
    X_test[c] = X_test[c].astype("category").cat.set_categories(
        X[c].cat.categories)


def run_cv(seed, want_test):
    t0 = time.time()
    folds = list(StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=seed).split(X, y))
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(test)) if want_test else None
    train_aucs, val_aucs, best_iters = [], [], []
    for k, (tr_idx, va_idx) in enumerate(folds, start=1):
        model = xgb.XGBClassifier(**PARAMS)
        model.fit(X.iloc[tr_idx], y[tr_idx],
                  eval_set=[(X.iloc[va_idx], y[va_idx])], verbose=False)
        p_tr = model.predict_proba(X.iloc[tr_idx])[:, 1]
        p_va = model.predict_proba(X.iloc[va_idx])[:, 1]
        oof[va_idx] = p_va
        train_aucs.append(roc_auc_score(y[tr_idx], p_tr))
        val_aucs.append(roc_auc_score(y[va_idx], p_va))
        best_iters.append(int(model.best_iteration))
        if want_test:
            test_pred += model.predict_proba(X_test)[:, 1] / config.N_FOLDS
        print(f"  [seed={seed}] fold {k}: best_iter={best_iters[-1]:5d} "
              f"train={train_aucs[-1]:.6f} val={val_aucs[-1]:.6f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return dict(oof=oof, test_pred=test_pred, train_aucs=train_aucs,
                val_aucs=val_aucs, best_iters=best_iters,
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
print(f"v22 XGBoost depth={CHOSEN_DEPTH} seed 42 (official)")
print("=" * 70)
r42 = run_cv(42, want_test=True)
fp42 = four_piece(r42)
results[42] = (r42, fp42)

sub_name = f"v22_xgboost_d{CHOSEN_DEPTH}.csv"
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
}).to_csv(PROJECT / "outputs" / "v22_oof.csv", index=False)
pd.DataFrame({
    config.ID_COL: train[config.ID_COL], config.TARGET: y,
    "oof_prob": r42["oof"],
}).to_csv(OUT_MEMBERS / "oof_xgb_seed42.csv", index=False)
pd.DataFrame({
    config.ID_COL: test[config.ID_COL], config.TARGET: r42["test_pred"],
}).to_csv(OUT_MEMBERS / "test_xgb_seed42.csv", index=False)

print("\n" + "=" * 70)
print(f"v22 XGBoost depth={CHOSEN_DEPTH} seed 123 (confirmation)")
print("=" * 70)
r123 = run_cv(123, want_test=False)
fp123 = four_piece(r123)
results[123] = (r123, fp123)
pd.DataFrame({
    config.ID_COL: train[config.ID_COL], config.TARGET: y,
    "oof_prob": r123["oof"],
}).to_csv(OUT_MEMBERS / "oof_xgb_seed123.csv", index=False)

print("\n===== v22 four-piece =====")
print(f"seed 42 : train={fp42['mean_train']:.6f} val={fp42['mean_val']:.6f} "
      f"gap={fp42['gap']:.6f} std={fp42['val_std']:.6f} OOF={fp42['oof']:.6f}")
print(f"seed 123: train={fp123['mean_train']:.6f} val={fp123['mean_val']:.6f} "
      f"gap={fp123['gap']:.6f} std={fp123['val_std']:.6f} OOF={fp123['oof']:.6f}")

pair_notes = []
verdicts = {}
for seed, (r, fp) in results.items():
    v17, v18 = baseline_aucs(seed)
    for label, base in (("v17", v17), ("v18", v18)):
        d_bar, sd, t, pos = paired(r["val_aucs"], base)
        passed = d_bar > 0 and pos >= 4 and t >= 2.776
        verdicts[(seed, label)] = passed
        print(f"seed {seed} vs {label}: d_bar={d_bar:+.6f} s_d={sd:.6f} "
              f"t={t:+.3f} positive={pos}/5 passed={passed}")
        pair_notes.append(f"seed{seed} vs {label}: d={d_bar:+.6f} t={t:+.2f} "
                          f"pos={pos}/5 pass={passed}")

cap_hit = any(b >= CAP for b in r42["best_iters"] + r123["best_iters"])
beats_v18_both = all(verdicts[(s, "v18")] for s in (42, 123))
beats_v17_both = all(verdicts[(s, "v17")] for s in (42, 123))
notes = (
    f"model-form experiment (XGBoost 3.4.1, family member); 24 raw features; "
    f"native categoricals, depth={CHOSEN_DEPTH}, eta=0.1, cap={CAP}, "
    f"patience={PATIENCE}; seed42 four-piece: train={fp42['mean_train']:.6f} "
    f"val={fp42['mean_val']:.6f} gap={fp42['gap']:.6f} std={fp42['val_std']:.6f} "
    f"OOF={fp42['oof']:.6f}; seed123 OOF={fp123['oof']:.6f}; best_iters="
    f"{r42['best_iters']} / {r123['best_iters']}; cap_hit={cap_hit}; "
    f"pairs: {'; '.join(pair_notes)}; beats v18 on both seeds={beats_v18_both}, "
    f"beats v17 on both={beats_v17_both}; candidate submission stays v18 unless "
    f"v18 comparison passes on both seeds"
)
row = {
    "experiment_id": "v22",
    "model": "xgboost_3.4.1",
    "feature_summary": (
        f"24 raw features (13 numeric + 5 categorical native hist + 6 boolean); "
        f"XGBoost hist depth={CHOSEN_DEPTH}; model-form experiment"
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
print(f"beats v18 both seeds={beats_v18_both}; beats v17 both={beats_v17_both}")
print("v22 done.")
