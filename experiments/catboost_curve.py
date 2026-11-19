"""CatBoost 深度 2–4 的容量诊断（分类特征 one-hot，seed 42/123）。"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import config  # noqa: E402

OUT = PROJECT / "outputs" / "catboost_curve"
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(config.TRAIN_PATH)
y = train[config.TARGET].to_numpy()
FEATURES = config.NUMERIC_COLS + config.CATEGORICAL_COLS + config.BOOLEAN_COLS
X = train[FEATURES].copy()
for c in config.CATEGORICAL_COLS:
    X[c] = X[c].astype(str)
CAT_IDX = [X.columns.get_loc(c) for c in config.CATEGORICAL_COLS]

DEPTHS = (2, 3, 4)
SEEDS = (42, 123)
CAP = 12000
PATIENCE = 500


def run_config(depth, seed):
    t0 = time.time()
    folds = list(
        StratifiedKFold(
            n_splits=config.N_FOLDS, shuffle=True, random_state=seed
        ).split(X, y)
    )
    oof = np.zeros(len(y))
    train_aucs, val_aucs, best_iters = [], [], []
    for k, (tr_idx, va_idx) in enumerate(folds, start=1):
        model = CatBoostClassifier(
            iterations=CAP, depth=depth, learning_rate=0.1,
            l2_leaf_reg=1.0, bootstrap_type="Bernoulli", subsample=0.8,
            rsm=0.8, border_count=128, one_hot_max_size=16,
            max_ctr_complexity=1, loss_function="Logloss", eval_metric="AUC",
            random_seed=config.RANDOM_STATE, thread_count=-1,
            early_stopping_rounds=PATIENCE, use_best_model=True,
            allow_writing_files=False, verbose=False,
        )
        model.fit(
            X.iloc[tr_idx], y[tr_idx],
            cat_features=CAT_IDX,
            eval_set=(X.iloc[va_idx], y[va_idx]),
            verbose=False,
        )
        p_tr = model.predict_proba(X.iloc[tr_idx])[:, 1]
        p_va = model.predict_proba(X.iloc[va_idx])[:, 1]
        oof[va_idx] = p_va
        train_aucs.append(roc_auc_score(y[tr_idx], p_tr))
        val_aucs.append(roc_auc_score(y[va_idx], p_va))
        best_iters.append(int(model.get_best_iteration()))
        print(f"  [depth={depth} seed={seed}] fold {k}: "
              f"best_iter={best_iters[-1]:5d}  val AUC={val_aucs[-1]:.6f}",
              flush=True)
    row = dict(
        seed=seed, depth=depth,
        mean_train_auc=float(np.mean(train_aucs)),
        mean_val_auc=float(np.mean(val_aucs)),
        gap=float(np.mean(train_aucs) - np.mean(val_aucs)),
        val_std=float(np.std(val_aucs, ddof=0)),
        oof_auc=float(roc_auc_score(y, oof)),
        best_iters=str(best_iters), secs=round(time.time() - t0, 1),
        cap_hit=bool(any(b >= CAP for b in best_iters)),
    )
    pd.DataFrame({
        config.ID_COL: train[config.ID_COL],
        config.TARGET: y,
        "oof_prob": oof,
    }).to_csv(OUT / f"oof_cb_d{depth}_seed{seed}.csv", index=False)
    return row, np.array(val_aucs), best_iters


summary_rows, fold_rows = [], []
t_all = time.time()
for seed in SEEDS:
    for depth in DEPTHS:
        row, val_aucs, best_iters = run_config(depth, seed)
        summary_rows.append(row)
        for k in range(config.N_FOLDS):
            fold_rows.append(dict(seed=seed, depth=depth, fold=k + 1,
                                  val_auc=val_aucs[k],
                                  best_iter=best_iters[k]))
        pd.DataFrame(summary_rows).to_csv(OUT / "summary.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(OUT / "per_fold.csv", index=False)
        print(f"seed={seed} depth={depth}: val={row['mean_val_auc']:.6f} "
              f"train={row['mean_train_auc']:.6f} gap={row['gap']:.6f} "
              f"std={row['val_std']:.6f} OOF={row['oof_auc']:.6f} "
              f"iters={best_iters} cap_hit={row['cap_hit']} "
              f"({row['secs']:.0f}s, total {time.time()-t_all:.0f}s)", flush=True)

summary = pd.DataFrame(summary_rows)
print("\n===== CatBoost capacity summary (mean_val_auc) =====")
print(summary.pivot(index="depth", columns="seed", values="mean_val_auc")
      .to_string(float_format=lambda x: f"{x:.6f}"))

# ---------- 预注册选深规则（协议 §5.7 / §5.9） ----------
per_depth = summary.groupby("depth")["mean_val_auc"].mean().sort_index()
best_depth = int(per_depth.idxmax())
best_val = float(per_depth.max())
chosen = best_depth
for d in sorted(per_depth.index):
    if d < best_depth and best_val - float(per_depth[d]) < 0.0003:
        chosen = int(d)
        break
print(f"\nper-depth two-seed mean: {per_depth.round(6).to_dict()}")
print(f"nominal peak depth={best_depth} ({best_val:.6f}); "
      f"chosen depth={chosen} (parsimony band 0.0003)")
pd.DataFrame([{"nominal_peak_depth": best_depth, "chosen_depth": chosen,
               "per_depth_two_seed_mean": str(per_depth.round(6).to_dict())}]
             ).to_csv(OUT / "selection.csv", index=False)
print(f"\nDone in {time.time()-t_all:.0f}s")
