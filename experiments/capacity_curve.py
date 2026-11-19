"""交互容量曲线：LightGBM depth 1–4 与 31 叶参照的对比诊断（seed 42/123）。"""
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

OUT = PROJECT / "outputs" / "capacity_curve"
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(config.TRAIN_PATH)
for c in config.CATEGORICAL_COLS:
    train[c] = train[c].astype("category")
y = train[config.TARGET].to_numpy()
BASE = config.NUMERIC_COLS + config.CATEGORICAL_COLS + config.BOOLEAN_COLS
X = train[BASE]

MAX_ROUNDS = 30000
PATIENCE = 300
BASE_PARAMS = dict(
    objective="binary", n_estimators=MAX_ROUNDS, learning_rate=0.05,
    min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.0, reg_lambda=1.0,
    random_state=config.RANDOM_STATE, n_jobs=-1, verbosity=-1,
)

# (max_depth, num_leaves, 显示标签)
CONFIGS = [
    (1, 2, "1"),
    (2, 4, "2"),
    (3, 8, "3"),
    (4, 16, "4"),
    (-1, 31, "full_31"),
]

summary_rows, fold_rows = [], []
t_all = time.time()

for seed in (42, 123):
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=seed
    )
    folds = list(skf.split(X, y))
    for max_depth, num_leaves, label in CONFIGS:
        t0 = time.time()
        aucs, tr_aucs, iters = [], [], []
        oof = np.zeros(len(y))
        for k, (tr, va) in enumerate(folds, start=1):
            params = dict(BASE_PARAMS, num_leaves=num_leaves, max_depth=max_depth)
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
            tr_aucs.append(roc_auc_score(y[tr], m.predict_proba(X.iloc[tr])[:, 1]))
            iters.append(m.best_iteration_)

        row = dict(
            seed=seed, depth=label, num_leaves=num_leaves,
            mean_train_auc=float(np.mean(tr_aucs)),
            mean_val_auc=float(np.mean(aucs)),
            gap=float(np.mean(tr_aucs) - np.mean(aucs)),
            val_std=float(np.std(aucs)),
            oof_auc=float(roc_auc_score(y, oof)),
            best_iters=iters,
            secs=round(time.time() - t0, 1),
        )
        summary_rows.append(row)
        for k in range(config.N_FOLDS):
            fold_rows.append(dict(
                seed=seed, depth=label, fold=k + 1,
                val_auc=aucs[k], train_auc=tr_aucs[k], best_iter=iters[k],
            ))
        pd.DataFrame(summary_rows).to_csv(OUT / "summary.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(OUT / "per_fold.csv", index=False)
        print(
            f"seed={seed} depth={label:>7s}: val={row['mean_val_auc']:.6f} "
            f"train={row['mean_train_auc']:.6f} gap={row['gap']:.6f} "
            f"std={row['val_std']:.6f} OOF={row['oof_auc']:.6f} "
            f"iters={iters} ({row['secs']:.0f}s, total {time.time()-t_all:.0f}s)",
            flush=True,
        )

print("\n===== capacity curve summary =====")
summary = pd.DataFrame(summary_rows)
pivot = summary.pivot(index="depth", columns="seed", values="mean_val_auc")
print(pivot.to_string(float_format=lambda x: f"{x:.6f}"))
print(f"\nDone in {time.time()-t_all:.0f}s")
