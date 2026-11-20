"""SHAP 交互筛查：在折内模型上计算特征对交互强度。"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier, early_stopping
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config  # noqa: E402

# ---------- 读数据，分类列处理（与冻结协议完全一致） ----------
train = pd.read_csv(config.TRAIN_PATH)
for col in config.CATEGORICAL_COLS:
    train[col] = train[col].astype("category")

FEATURE_COLS = (
    config.NUMERIC_COLS + config.CATEGORICAL_COLS + config.BOOLEAN_COLS
)
assert len(FEATURE_COLS) == 24
X = train[FEATURE_COLS]
y = train[config.TARGET].to_numpy()

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

CANDIDATE_PAIRS = {
    "G4_velocity_x_merchant_risk": ("velocity_score", "merchant_risk_score"),
    "G5_hours_x_txn_count": ("hours_since_last_txn", "txn_count_last_24h"),
    "G6_amount_x_balance": ("amount_usd", "account_balance_usd"),
    "G11_distance_x_foreign": ("distance_from_home_km", "is_foreign_transaction"),
}
SAMPLE_SIZE = 2000
SCREEN_FOLDS = 2

skf = StratifiedKFold(
    n_splits=config.N_FOLDS,
    shuffle=True,
    random_state=config.RANDOM_STATE,
)

fold_pair_tables = []
fold_main_effects = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
    if fold > SCREEN_FOLDS:
        break
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

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
    print(f"fold {fold}: best_iter={model.best_iteration_}")

    # 分层抽样陌生验证样本
    rng = np.random.default_rng(config.RANDOM_STATE)
    sample_idx = np.concatenate([
        rng.choice(np.where(y_va == c)[0], size=SAMPLE_SIZE // 2, replace=False)
        for c in (0, 1)
    ])
    rng.shuffle(sample_idx)
    # shap 0.54 要求数值矩阵：分类列转 category codes（与 LightGBM
    # 内部使用的编码一致），其余列原样，特征顺序不变
    X_s = X_va.iloc[sample_idx].copy()
    for col in config.CATEGORICAL_COLS:
        X_s[col] = X_s[col].cat.codes

    explainer = shap.TreeExplainer(model)
    siv = explainer.shap_interaction_values(X_s)
    # 兼容二分类返回 list 的版本：取正类数组
    if isinstance(siv, list):
        siv = siv[1]
    siv = np.asarray(siv)
    assert siv.shape == (len(X_s), len(FEATURE_COLS), len(FEATURE_COLS))

    # 平均绝对交互矩阵（对称）与主效应（对角）
    mean_abs = np.abs(siv).mean(axis=0)
    np.fill_diagonal(mean_abs, 0.0)  # 只看特征对，主效应单独留存
    main_effect = np.abs(np.diagonal(np.asarray(siv).mean(axis=0)))
    fold_main_effects.append(main_effect)

    rows = []
    for i in range(len(FEATURE_COLS)):
        for j in range(i + 1, len(FEATURE_COLS)):
            rows.append({
                "feature_a": FEATURE_COLS[i],
                "feature_b": FEATURE_COLS[j],
                "interaction_strength": mean_abs[i, j],
            })
    table = pd.DataFrame(rows).sort_values(
        "interaction_strength", ascending=False, ignore_index=True
    )
    table["rank"] = table.index + 1
    table["total_pairs"] = len(table)
    fold_pair_tables.append(table)
    print(f"fold {fold}: 已计算 {len(table)} 个特征对，样本 {len(X_s)} 行")

# ---------- 两折平均：按特征对合并 ----------
merged = fold_pair_tables[0].merge(
    fold_pair_tables[1], on=["feature_a", "feature_b"],
    suffixes=("_f1", "_f2"),
)
merged["strength_mean"] = merged[
    ["interaction_strength_f1", "interaction_strength_f2"]
].mean(axis=1)
merged = merged.sort_values("strength_mean", ascending=False, ignore_index=True)
merged["rank_mean"] = merged.index + 1
merged["percentile"] = 1.0 - (merged["rank_mean"] - 1) / len(merged)

main_effect_mean = np.mean(fold_main_effects, axis=0)
main_series = pd.Series(main_effect_mean, index=FEATURE_COLS)

out_path = config.OUTPUTS_DIR / "eda" / "shap_interaction_screen.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(out_path, index=False)

# ---------- 候选对结论 ----------
print("\n===== 候选交互对 SHAP 筛选结果（2 折平均） =====")
median_strength = merged["strength_mean"].median()
for label, (a, b) in CANDIDATE_PAIRS.items():
    hit = merged[
        ((merged["feature_a"] == a) & (merged["feature_b"] == b))
        | ((merged["feature_a"] == b) & (merged["feature_b"] == a))
    ]
    assert len(hit) == 1
    r = hit.iloc[0]
    ratio_to_median = r["strength_mean"] / median_strength
    print(
        f"{label:32s} strength={r['strength_mean']:.6f}  "
        f"rank={int(r['rank_mean']):3d}/{len(merged)}  "
        f"percentile={r['percentile']*100:5.1f}%  "
        f"vs_median={ratio_to_median:.2f}x"
    )

print("\n===== 头部 10 个交互对（参照） =====")
print(merged.head(10)[
    ["feature_a", "feature_b", "strength_mean", "rank_mean", "percentile"]
].to_string(index=False))

print("\n===== 主效应 Top 10（参照） =====")
print(main_series.sort_values(ascending=False).head(10).to_string())
print(f"\n全量结果已保存: {out_path}")
print("SHAP 交互诊断完成。")
