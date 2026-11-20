"""阳性对照：向特征注入已知信号，检验流程的检出能力。"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier, early_stopping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config  # noqa: E402

# ---------- 数据与分类列处理（与冻结协议一致） ----------
train = pd.read_csv(config.TRAIN_PATH)
for col in config.CATEGORICAL_COLS:
    train[col] = train[col].astype("category")

FEATURE_COLS = (
    config.NUMERIC_COLS + config.CATEGORICAL_COLS + config.BOOLEAN_COLS
)
assert len(FEATURE_COLS) == 24
X = train[FEATURE_COLS]
y_real = train[config.TARGET].to_numpy()

# ---------- 独立随机特征 z（与现有特征无关） ----------
rng = np.random.default_rng(config.RANDOM_STATE)
z = rng.standard_normal(len(train))
X_treat = X.copy()
X_treat["synthetic_z"] = z

# ---------- 三个等级的标签改造（内存） ----------
neg_positions = np.where(y_real == 0)[0]
neg_sorted = neg_positions[np.argsort(-z[neg_positions])]  # 负样本按 z 降序

levels = {
    "placebo": 0.0,
    "weak_0.5pct": 0.005,
    "medium_1.5pct": 0.015,
}
y_variants = {}
for name, q in levels.items():
    y_mod = y_real.copy()
    n_flip = int(round(q * len(neg_sorted)))
    if n_flip > 0:
        y_mod[neg_sorted[:n_flip]] = 1
    y_variants[name] = y_mod
    print(f"{name:14s}: 注入正例 {n_flip:6d} 个，改造后欺诈率 "
          f"{y_mod.mean()*100:.3f}%")

# ---------- 协议参数（与 v02 完全一致） ----------
params = dict(
    objective="binary",
    n_estimators=1000,
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

# 折划分按原始 y 分层，一次计算、全程复用，保证严格配对
skf = StratifiedKFold(
    n_splits=config.N_FOLDS,
    shuffle=True,
    random_state=config.RANDOM_STATE,
)
fold_indices = list(skf.split(X, y_real))


def run_cv(feature_frame, target):
    """同一套折划分训练 5 个模型，返回逐折验证 AUC 与 best_iter。"""
    val_aucs, best_iters = [], []
    for tr_idx, va_idx in fold_indices:
        X_tr, X_va = feature_frame.iloc[tr_idx], feature_frame.iloc[va_idx]
        y_tr, y_va = target[tr_idx], target[va_idx]
        model = LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            categorical_feature=config.CATEGORICAL_COLS,
            eval_X=X_va,
            eval_y=y_va,
            eval_metric="auc",
            callbacks=[early_stopping(50, verbose=False)],
        )
        p_va = model.predict_proba(X_va)[:, 1]
        val_aucs.append(roc_auc_score(y_va, p_va))
        best_iters.append(model.best_iteration_)
    return np.array(val_aucs), best_iters


T_CRIT = 2.776  # t_{0.975, df=4}
summary_rows, detail_rows = [], []

for name, y_mod in y_variants.items():
    base_aucs, base_iters = run_cv(X, y_mod)
    treat_aucs, treat_iters = run_cv(X_treat, y_mod)
    deltas = treat_aucs - base_aucs

    d_bar = float(deltas.mean())
    s_d = float(deltas.std(ddof=1))
    se = s_d / np.sqrt(len(deltas))
    t_stat = d_bar / se if se > 0 else np.nan
    positive_folds = int((deltas > 0).sum())
    significant = bool(t_stat >= T_CRIT) if not np.isnan(t_stat) else False
    verdict = "adopt" if (significant and positive_folds >= 4) else "not evidenced"

    summary_rows.append({
        "level": name,
        "base_mean_val_auc": round(base_aucs.mean(), 6),
        "treat_mean_val_auc": round(treat_aucs.mean(), 6),
        "mean_delta": round(d_bar, 6),
        "delta_std_sd": round(s_d, 6),
        "paired_t": round(float(t_stat), 3),
        "positive_folds": f"{positive_folds}/5",
        "base_best_iters": base_iters,
        "treat_best_iters": treat_iters,
        "mechanism_verdict": verdict,
    })
    for k, d in enumerate(deltas, start=1):
        detail_rows.append({
            "level": name, "fold": k,
            "base_val_auc": round(base_aucs[k - 1], 6),
            "treat_val_auc": round(treat_aucs[k - 1], 6),
            "delta": round(float(d), 6),
        })

    print(f"\n===== {name} =====")
    print(f"base val AUCs : {np.round(base_aucs, 6)}")
    print(f"treat val AUCs: {np.round(treat_aucs, 6)}")
    print(f"deltas        : {np.round(deltas, 6)}")
    print(f"d̄={d_bar:.6f}  s_d={s_d:.6f}  SE={se:.6f}  t={t_stat:.3f}  "
          f"positive folds={positive_folds}/5  verdict={verdict}")

summary = pd.DataFrame(summary_rows)
details = pd.DataFrame(detail_rows)

out_path = config.OUTPUTS_DIR / "positive_control_results.csv"
detail_path = config.OUTPUTS_DIR / "positive_control_fold_details.csv"
summary.to_csv(out_path, index=False)
details.to_csv(detail_path, index=False)

print("\n===== 灵敏性体检汇总 =====")
print(summary[[
    "level", "base_mean_val_auc", "treat_mean_val_auc", "mean_delta",
    "paired_t", "positive_folds", "mechanism_verdict",
]].to_string(index=False))
print(f"\n汇总: {out_path}")
print(f"逐折明细: {detail_path}")
print("阳性对照完成。")
