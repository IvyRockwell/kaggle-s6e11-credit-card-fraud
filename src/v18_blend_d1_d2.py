"""v18：depth1 stumps 与 depth2 LightGBM 的 0.5/0.5 概率平均。
两成员使用相同折划分，权重固定，不在 OOF 上调参。"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import config  # noqa: E402

OUT = PROJECT / "outputs" / "ensemble"
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(config.TRAIN_PATH)
test = pd.read_csv(config.TEST_PATH)
for c in config.CATEGORICAL_COLS:
    train[c] = train[c].astype("category")
    test[c] = test[c].astype("category").cat.set_categories(train[c].cat.categories)

y = train[config.TARGET].to_numpy()
BASE = config.NUMERIC_COLS + config.CATEGORICAL_COLS + config.BOOLEAN_COLS
X = train[BASE]
X_test = test[BASE]

BASE_PARAMS = dict(
    objective="binary", learning_rate=0.05, min_child_samples=100,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
    random_state=config.RANDOM_STATE, n_jobs=-1, verbosity=-1,
)
DEPTH = {
    "d1": dict(BASE_PARAMS, n_estimators=30000, num_leaves=2, max_depth=1),
    "d2": dict(BASE_PARAMS, n_estimators=10000, num_leaves=4, max_depth=2),
}
PATIENCE = 300


def run_cv(tag, params, seed, want_test=False):
    t0 = time.time()
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=seed
    )
    oof = np.zeros(len(y))
    aucs, iters = [], []
    test_pred = np.zeros(len(test)) if want_test else None
    for k, (tr, va) in enumerate(skf.split(X, y), start=1):
        m = LGBMClassifier(**params)
        m.fit(
            X.iloc[tr], y[tr],
            categorical_feature=config.CATEGORICAL_COLS,
            eval_X=X.iloc[va], eval_y=y[va], eval_metric="auc",
            callbacks=[early_stopping(PATIENCE, verbose=False)],
        )
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        aucs.append(roc_auc_score(y[va], p))
        iters.append(m.best_iteration_)
        if want_test:
            test_pred += m.predict_proba(X_test)[:, 1] / config.N_FOLDS
    aucs = np.array(aucs)
    print(f"[{tag} seed={seed}] OOF={roc_auc_score(y, oof):.6f} "
          f"mean_fold={aucs.mean():.6f} iters={iters} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return dict(tag=tag, seed=seed, oof=oof, aucs=aucs,
                test_pred=test_pred, iters=iters)


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def evaluate(seed, oof1, oof2, tag1="d1", tag2="d2"):
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=seed
    )
    folds = list(skf.split(X, y))
    # global rank (AUC is rank-based, so global ranks are fine)
    r1 = rankdata(oof1) / len(oof1)
    r2 = rankdata(oof2) / len(oof2)
    blends = {
        "prob50": 0.5 * oof1 + 0.5 * oof2,
        "logit50": sigmoid(0.5 * logit(oof1) + 0.5 * logit(oof2)),
        "rank50": 0.5 * r1 + 0.5 * r2,
    }
    rows = []
    for name, p in [("d1", oof1), ("d2", oof2), *blends.items()]:
        aucs = np.array([roc_auc_score(y[va], p[va]) for _, va in folds])
        rows.append(dict(seed=seed, model=name, oof_auc=roc_auc_score(y, p),
                         mean_fold_auc=aucs.mean(), aucs=aucs))
    for w in (0.3, 0.4, 0.5, 0.6, 0.7):
        p = w * oof1 + (1 - w) * oof2
        rows.append(dict(seed=seed, model=f"prob_w{w}", oof_auc=roc_auc_score(y, p),
                         mean_fold_auc=np.nan, aucs=None))
    res = pd.DataFrame([{k: v for k, v in r.items() if k != "aucs"} for r in rows])
    print(f"\n[{seed}] single vs blends (OOF AUC):")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    return rows


print("=" * 70)
print("seed 42: reuse v17 depth1; run depth2")
print("=" * 70)
v17 = pd.read_csv(PROJECT / "outputs" / "v17_oof.csv")
assert v17[config.ID_COL].equals(train[config.ID_COL])
oof_d1_42 = v17["oof_prob"].to_numpy()
d2_42 = run_cv("d2", DEPTH["d2"], 42, want_test=True)

print("\n" + "=" * 70)
print("seed 123: run both depth1 and depth2")
print("=" * 70)
d1_123 = run_cv("d1", DEPTH["d1"], 123)
d2_123 = run_cv("d2", DEPTH["d2"], 123)

rows42 = evaluate(42, oof_d1_42, d2_42["oof"])
rows123 = evaluate(123, d1_123["oof"], d2_123["oof"])

# paired deltas for the 50/50 prob blend
def paired(p_new, p_old, seed, label):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    folds = list(skf.split(X, y))
    a = np.array([roc_auc_score(y[va], p_new[va]) for _, va in folds])
    b = np.array([roc_auc_score(y[va], p_old[va]) for _, va in folds])
    d = a - b
    sd = d.std(ddof=1)
    t = d.mean() / (sd / np.sqrt(5)) if sd > 0 else np.nan
    print(f"[seed {seed}] {label}: d_bar={d.mean():+.6f} s_d={sd:.6f} "
          f"t={t:+.3f} positive={int((d>0).sum())}/5")


p50_42 = 0.5 * oof_d1_42 + 0.5 * d2_42["oof"]
p50_123 = 0.5 * d1_123["oof"] + 0.5 * d2_123["oof"]
print()
paired(p50_42, oof_d1_42, 42, "prob50 vs d1")
paired(p50_42, d2_42["oof"], 42, "prob50 vs d2")
paired(p50_123, d1_123["oof"], 123, "prob50 vs d1")
paired(p50_123, d2_123["oof"], 123, "prob50 vs d2")

# ------- candidate submission: seed 42 blend of v17 test preds + d2 test preds -------
v17_sub = pd.read_csv(PROJECT / "submissions" / "v17_lgbm_additive_stumps.csv")
assert v17_sub["id"].equals(test[config.ID_COL])
test_blend = 0.5 * v17_sub["is_fraud"].to_numpy() + 0.5 * d2_42["test_pred"]
assert test_blend.min() > 0 and test_blend.max() < 1
sub_path = PROJECT / "submissions" / "v18_lgbm_blend_d1_d2.csv"
pd.DataFrame({"id": test[config.ID_COL], "is_fraud": test_blend}).to_csv(
    sub_path, index=False
)
print(f"\ncandidate submission -> {sub_path}")

summary = pd.DataFrame([{k: v for k, v in r.items() if k != "aucs"}
                        for r in rows42 + rows123])
summary.to_csv(OUT / "blend_results.csv", index=False)
print("Done.")
