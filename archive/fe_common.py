"""共享实验运行器：统一的 5 折训练、早停、四件套与配对检验流程。"""
from datetime import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier, early_stopping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config  # noqa: E402

T_CRIT = 2.776  # 双侧 5%，df=4
BASE_COLS = (
    config.NUMERIC_COLS + config.CATEGORICAL_COLS + config.BOOLEAN_COLS
)
STOPPING_ROUNDS = 50
MAX_ROUNDS = 1000
TE_SMOOTHING = 20.0  # G12 折内目标编码平滑强度，仅用于 v16

PARAMS = dict(
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


def _load_data():
    """读入原始数据并把 5 个分类列转 category、test 对齐 train 类别域。"""
    train = pd.read_csv(config.TRAIN_PATH)
    test = pd.read_csv(config.TEST_PATH)
    for col in config.CATEGORICAL_COLS:
        train[col] = train[col].astype("category")
        test[col] = test[col].astype("category").cat.set_categories(
            train[col].cat.categories
        )
    return train, test


def _baseline_fold_aucs(y, skf):
    """从 v02 OOF 概率按相同折索引复算每折基线验证 AUC（配对基准）。"""
    base = pd.read_csv(config.OUTPUTS_DIR / "v02_oof.csv")
    train_ids = pd.read_csv(config.TRAIN_PATH, usecols=[config.ID_COL])[
        config.ID_COL
    ]
    # 配对前提：v02 OOF 行序与 train 完全一致，折索引才能直接复用
    assert base[config.ID_COL].equals(train_ids)
    oof_arr = base["oof_prob"].to_numpy()
    y_arr = y.to_numpy()
    aucs = []
    for _, va_idx in skf.split(np.zeros(len(y)), y_arr):
        aucs.append(roc_auc_score(y_arr[va_idx], oof_arr[va_idx]))
    return aucs


def _te_fold_frames(X_tr, X_va, X_test, y_tr):
    """G12 专用：仅用当前折训练部分 fit 平滑目标编码并 transform 三处。"""
    X_tr, X_va, X_test = X_tr.copy(), X_va.copy(), X_test.copy()
    global_mean = float(y_tr.mean())
    for col in config.CATEGORICAL_COLS:
        stats = X_tr.assign(_y=y_tr.to_numpy()).groupby(
            col, observed=True
        )["_y"].agg(["mean", "count"])
        smoothed = (
            stats["count"] * stats["mean"] + TE_SMOOTHING * global_mean
        ) / (stats["count"] + TE_SMOOTHING)
        mapping = smoothed.to_dict()
        te_col = col + "_te"
        X_tr[te_col] = X_tr[col].map(mapping).astype(float)
        X_va[te_col] = X_va[col].map(mapping).astype(float)
        X_test[te_col] = X_test[col].map(mapping).astype(float)
        X_tr[te_col] = X_tr[te_col].fillna(global_mean)
        X_va[te_col] = X_va[te_col].fillna(global_mean)
        X_test[te_col] = X_test[te_col].fillna(global_mean)
    return X_tr, X_va, X_test


def run_experiment(*, version, sub_filename, feature_summary,
                   build_fn=None, expected_features, te=False,
                   transform_note="label-free deterministic transform, no leakage"):
    train, test = _load_data()
    y = train[config.TARGET]

    if te:
        new_cols = [c + "_te" for c in config.CATEGORICAL_COLS]
        feature_cols = BASE_COLS + new_cols
    else:
        new_cols = build_fn(train, test)
        feature_cols = BASE_COLS + new_cols
    assert len(feature_cols) == expected_features
    assert config.ID_COL not in feature_cols and config.TARGET not in feature_cols

    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True,
        random_state=config.RANDOM_STATE,
    )
    base_val_aucs = _baseline_fold_aucs(y, skf)

    # TE 列在折内 fit 后才存在，折前只取 24 基础列；常规模式取全量特征
    pre_cols = BASE_COLS if te else feature_cols
    X = train[pre_cols]
    X_test = test[pre_cols]
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    train_aucs, val_aucs, best_iters = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        X_te_fold = X_test
        if te:
            X_tr, X_va, X_te_fold = _te_fold_frames(
                X_tr, X_va, X_test, y_tr
            )

        model = LGBMClassifier(**PARAMS)
        model.fit(
            X_tr, y_tr,
            categorical_feature=config.CATEGORICAL_COLS,
            eval_X=X_va, eval_y=y_va, eval_metric="auc",
            callbacks=[early_stopping(STOPPING_ROUNDS, verbose=False)],
        )
        p_tr = model.predict_proba(X_tr)[:, 1]
        p_va = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = p_va
        test_pred += model.predict_proba(X_te_fold)[:, 1] / config.N_FOLDS
        train_aucs.append(roc_auc_score(y_tr, p_tr))
        val_aucs.append(roc_auc_score(y_va, p_va))
        best_iters.append(model.best_iteration_)

    mean_train = float(np.mean(train_aucs))
    mean_val = float(np.mean(val_aucs))
    gap = mean_train - mean_val
    val_std = float(np.std(val_aucs, ddof=0))
    oof_auc = float(roc_auc_score(y, oof))

    d = np.array(val_aucs) - np.array(base_val_aucs)
    d_bar = float(d.mean())
    s_d = float(d.std(ddof=1))
    se = s_d / np.sqrt(config.N_FOLDS)
    t_stat = d_bar / se if se > 0 else np.nan
    pos_folds = int((d > 0).sum())
    # 早停需要 patience 轮的余地；best_iter 落在距上限 patience 轮以内说明
    # 是被上限截断，增益只能算下界（不能当"结论无效"）。
    cap_hit = any(b >= MAX_ROUNDS - STOPPING_ROUNDS for b in best_iters)

    if d_bar < 0:
        verdict = "negative"
    elif pos_folds >= 4 and t_stat >= T_CRIT:
        verdict = ("adopt (cap reached: gain is a lower bound)" if cap_hit
                   else "passes-seed42-pending-multiseed-confirmation")
    else:
        verdict = "not evidenced"

    print(f"\n===== {version} 逐折对照（基线 v02） =====")
    for k in range(config.N_FOLDS):
        print(f"fold {k+1}: best_iter={best_iters[k]:4d}  "
              f"val AUC={val_aucs[k]:.6f}  base={base_val_aucs[k]:.6f}  "
              f"d={d[k]:+.6f}")
    print(f"\nmean train AUC={mean_train:.6f}  mean val AUC={mean_val:.6f}  "
          f"gap={gap:.6f}  val std={val_std:.6f}  OOF AUC={oof_auc:.6f}")
    print(f"paired deltas d_k={[round(x, 6) for x in d]}")
    print(f"d_bar={d_bar:+.6f}  s_d={s_d:.6f}  SE={se:.6f}  "
          f"t={t_stat:.3f} (crit {T_CRIT})  positive folds={pos_folds}/5  "
          f"cap_hit={cap_hit}")
    print(f"VERDICT: {verdict}")

    assert 0.0 <= test_pred.min() and test_pred.max() <= 1.0
    assert not np.isnan(test_pred).any() and not np.isnan(oof).any()

    sub_path = config.SUBMISSIONS_DIR / sub_filename
    pd.DataFrame({
        config.ID_COL: test[config.ID_COL], config.TARGET: test_pred
    }).to_csv(sub_path, index=False)
    check = pd.read_csv(sub_path)
    assert list(check.columns) == ["id", "is_fraud"]
    assert len(check) == 299984
    assert check["id"].equals(test["id"])
    assert check["is_fraud"].between(0, 1).all()
    assert not check["is_fraud"].isin([0.0, 1.0]).any()

    oof_path = config.OUTPUTS_DIR / f"{version}_oof.csv"
    pd.DataFrame({
        config.ID_COL: train[config.ID_COL], config.TARGET: y,
        "oof_prob": oof,
    }).to_csv(oof_path, index=False)

    notes = (
        f"new cols={new_cols}; best_iters={best_iters}; OOF AUC={oof_auc:.6f}; "
        f"paired vs v02: d_bar={d_bar:+.6f}, s_d={s_d:.6f}, t={t_stat:.3f}, "
        f"positive_folds={pos_folds}/5; verdict={verdict}; {transform_note}"
    )
    pd.DataFrame([{
        "experiment_id": version,
        "model": "lightgbm_4.7.0",
        "feature_summary": feature_summary,
        "n_folds": config.N_FOLDS,
        "mean_train_auc": round(mean_train, 6),
        "mean_val_auc": round(mean_val, 6),
        "train_val_gap": round(gap, 6),
        "val_auc_std": round(val_std, 6),
        "submission_file": sub_path.name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": notes,
    }]).to_csv(config.EXPERIMENT_LOG_PATH, mode="a", header=False, index=False)

    print(f"提交: {sub_path} | OOF: {oof_path} | 日志已追加")
    return dict(version=version, mean_train=mean_train, mean_val=mean_val,
                gap=gap, val_std=val_std, oof_auc=oof_auc, d_bar=d_bar,
                t_stat=t_stat, pos_folds=pos_folds, verdict=verdict,
                best_iters=best_iters)
