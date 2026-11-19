"""成员 OOF 缓存：为二级集成与多种子实验补齐同折 OOF。"""
from datetime import datetime
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import config  # noqa: E402

OUT = PROJECT / "outputs" / "ensemble" / "members"
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(config.TRAIN_PATH)
test = pd.read_csv(config.TEST_PATH)
for col in config.CATEGORICAL_COLS:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category").cat.set_categories(
        train[col].cat.categories
    )

y = train[config.TARGET].to_numpy()
FEATURES = config.NUMERIC_COLS + config.CATEGORICAL_COLS + config.BOOLEAN_COLS
X = train[FEATURES]
X_test = test[FEATURES]
assert len(FEATURES) == 24

BASE_PARAMS = dict(
    objective="binary", learning_rate=0.05, min_child_samples=100,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
    random_state=config.RANDOM_STATE, n_jobs=-1, verbosity=-1,
)
MEMBERS = {
    "d1": dict(BASE_PARAMS, n_estimators=25000, num_leaves=2, max_depth=1),
    "d2": dict(BASE_PARAMS, n_estimators=10000, num_leaves=4, max_depth=2),
}
PATIENCE = 300
CAP = {"d1": 25000, "d2": 10000}


def run_member(member, seed, want_test):
    """同一成员的完整 5 折运行；返回 OOF、逐折 AUC、best_iter、测试预测。"""
    t0 = time.time()
    folds = list(
        StratifiedKFold(
            n_splits=config.N_FOLDS, shuffle=True, random_state=seed
        ).split(X, y)
    )
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(test)) if want_test else None
    train_aucs, val_aucs, best_iters = [], [], []
    for k, (tr_idx, va_idx) in enumerate(folds, start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        model = LGBMClassifier(**MEMBERS[member])
        model.fit(
            X_tr, y_tr,
            categorical_feature=config.CATEGORICAL_COLS,
            eval_X=X_va, eval_y=y_va, eval_metric="auc",
            callbacks=[early_stopping(PATIENCE, verbose=False)],
        )
        p_tr = model.predict_proba(X_tr)[:, 1]
        p_va = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = p_va
        train_aucs.append(roc_auc_score(y_tr, p_tr))
        val_aucs.append(roc_auc_score(y_va, p_va))
        best_iters.append(model.best_iteration_)
        if want_test:
            test_pred += model.predict_proba(X_test)[:, 1] / config.N_FOLDS
        print(f"  [{member} seed={seed}] fold {k}: best_iter={model.best_iteration_:5d}"
              f"  val AUC={val_aucs[-1]:.6f}", flush=True)
    oof_auc = float(roc_auc_score(y, oof))
    print(f"[{member} seed={seed}] OOF={oof_auc:.6f} "
          f"mean_val={np.mean(val_aucs):.6f} gap={np.mean(train_aucs)-np.mean(val_aucs):.6f}"
          f" iters={best_iters} ({time.time()-t0:.0f}s)", flush=True)
    return dict(oof=oof, test_pred=test_pred, train_aucs=train_aucs,
                val_aucs=val_aucs, best_iters=best_iters, oof_auc=oof_auc,
                secs=round(time.time() - t0, 1))


def save_oof(member, seed, r):
    path = OUT / f"oof_{member}_seed{seed}.csv"
    pd.DataFrame({
        config.ID_COL: train[config.ID_COL],
        config.TARGET: y,
        "oof_prob": r["oof"],
    }).to_csv(path, index=False)
    return path


meta_rows = []


def add_meta(member, seed, r, note):
    meta_rows.append(dict(
        member=member, seed=seed, n_folds=config.N_FOLDS,
        mean_train_auc=float(np.mean(r["train_aucs"])),
        mean_val_auc=float(np.mean(r["val_aucs"])),
        gap=float(np.mean(r["train_aucs"]) - np.mean(r["val_aucs"])),
        val_std=float(np.std(r["val_aucs"], ddof=0)),
        oof_auc=r["oof_auc"], best_iters=str(r["best_iters"]),
        secs=r["secs"], note=note,
    ))
    pd.DataFrame(meta_rows).to_csv(OUT / "cache_meta.csv", index=False)


# ---------- 任务 1：d2 seed 42（测试预测 + 复现性校验） ----------
print("=" * 70)
print("task 1: d2 seed 42 with test predictions (reproducibility check)")
print("=" * 70)
r = run_member("d2", 42, want_test=True)
prev = pd.read_csv(PROJECT / "outputs" / "ensemble" / "d2_seed42_oof.csv")
assert prev[config.ID_COL].equals(train[config.ID_COL])
diff = float(np.max(np.abs(prev["d2_oof"].to_numpy() - r["oof"])))
print(f"[check] d2 seed42 recompute vs saved OOF: max|diff|={diff:.3e}")
assert diff < 1e-4, "d2 seed42 重跑与既有 OOF 不一致，环境/参数漂移"
pd.DataFrame({
    config.ID_COL: test[config.ID_COL],
    config.TARGET: r["test_pred"],
}).to_csv(OUT / "test_d2_seed42.csv", index=False)
add_meta("d2", 42, r, f"reused for blend (v18); recompute max|diff|={diff:.2e}")

# ---------- 任务 2：d1 seed 123 OOF ----------
print("\n" + "=" * 70)
print("task 2: d1 seed 123 OOF")
print("=" * 70)
r = run_member("d1", 123, want_test=False)
save_oof("d1", 123, r)
cap_hit = any(b >= CAP["d1"] for b in r["best_iters"])
add_meta("d1", 123, r, f"cap_hit={cap_hit} (cap {CAP['d1']}); used by stack seed123")

# ---------- 任务 3：d2 seed 123 OOF ----------
print("\n" + "=" * 70)
print("task 3: d2 seed 123 OOF")
print("=" * 70)
r = run_member("d2", 123, want_test=False)
save_oof("d2", 123, r)
add_meta("d2", 123, r, "used by stack seed123")

print(f"\nmembers dir: {OUT}")
print("cache_meta.csv written; done.")
