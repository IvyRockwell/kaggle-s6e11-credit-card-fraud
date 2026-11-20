"""嵌套早停审计：给 v18 一个无偏口径的数字。

问题：v17/v18 的 OOF 都用"外层验证折"做早停。同一个折既决定何时停手、
又用来打分，分数因此虚高——v17 实测虚高 +5.4e−4（见 results/nested_es_audit.csv），
比 v18 相对 v17 的增益（+3.99e−4）还大，所以"v18 更好"可能只是协议偏差。

做法：外层 5 折不变；在每个外层训练折内再切 15% 做内层验证（只用于选轮数），
外层验证折只打分、不参与任何训练决策。代价是每个成员只能用 68% 的数据。

同时跑 d1（depth1 stumps）与 d2（depth2），按 v18 的 0.5/0.5 融合，
得到嵌套口径下的 v17 与 v18，并对二者的差值做配对 bootstrap。

用法：
    python experiments/nested_es_v18.py            # 约 15 分钟
    python experiments/nested_es_v18.py --boot 0   # 跳过 bootstrap

输出：
    outputs/nested/oof_d1.csv, oof_d2.csv     （逐行 OOF，供复核）
    results/nested_v18.csv                    （汇总，纳入版本控制）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import config  # noqa: E402

OUT = PROJECT / "outputs" / "nested"
RES = PROJECT / "results" / "nested_v18.csv"
OUTER_SEED = 42
INNER_FRAC = 0.15
INNER_SEED = 42
PATIENCE = 300

BASE_PARAMS = dict(
    objective="binary", learning_rate=0.05, min_child_samples=100,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
    random_state=OUTER_SEED, n_jobs=-1, verbosity=-1,
)
MEMBERS = {
    "d1": dict(BASE_PARAMS, n_estimators=25000, num_leaves=2, max_depth=1),
    "d2": dict(BASE_PARAMS, n_estimators=10000, num_leaves=4, max_depth=2),
}

# 非嵌套参考值（results/experiment_log.csv）
REF_NON_NESTED = {"d1": 0.947746, "d2": 0.947955, "blend": 0.948142}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=400,
                    help="嵌套口径 blend-vs-d1 的 bootstrap 次数；0 表示跳过")
    args = ap.parse_args()

    train = pd.read_csv(config.TRAIN_PATH)
    test = pd.read_csv(config.TEST_PATH)  # noqa: F841  (只用于确认数据版本)
    for col in config.CATEGORICAL_COLS:
        train[col] = train[col].astype("category")
    y = train[config.TARGET].to_numpy()
    FEATURES = config.NUMERIC_COLS + config.CATEGORICAL_COLS + config.BOOLEAN_COLS
    X = train[FEATURES]

    outer = list(StratifiedKFold(
        config.N_FOLDS, shuffle=True, random_state=OUTER_SEED).split(X, y))

    oof = {m: np.zeros(len(y)) for m in MEMBERS}
    iters = {m: [] for m in MEMBERS}
    fold_aucs = {m: [] for m in MEMBERS}

    t0 = time.time()
    for k, (tr_idx, va_idx) in enumerate(outer, start=1):
        # 内层再切 15%，只用它决定停手；外层验证折不参与任何训练决策
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=INNER_FRAC, random_state=INNER_SEED)
        inner_tr, inner_va = next(sss.split(X.iloc[tr_idx], y[tr_idx]))
        tr_pos = tr_idx[inner_tr]
        in_pos = tr_idx[inner_va]

        for name, params in MEMBERS.items():
            model = LGBMClassifier(**params)
            model.fit(
                X.iloc[tr_pos], y[tr_pos],
                categorical_feature=config.CATEGORICAL_COLS,
                eval_X=X.iloc[in_pos], eval_y=y[in_pos], eval_metric="auc",
                callbacks=[early_stopping(PATIENCE, verbose=False)],
            )
            p = model.predict_proba(X.iloc[va_idx])[:, 1]
            oof[name][va_idx] = p
            iters[name].append(int(model.best_iteration_))
            fold_aucs[name].append(float(roc_auc_score(y[va_idx], p)))

        print(f"fold {k}: d1 val={fold_aucs['d1'][-1]:.6f} "
              f"(iter={iters['d1'][-1]})  d2 val={fold_aucs['d2'][-1]:.6f} "
              f"(iter={iters['d2'][-1]})  [{time.time() - t0:.0f}s]", flush=True)

    blend = 0.5 * oof["d1"] + 0.5 * oof["d2"]
    blend_fold = np.array([
        roc_auc_score(y[va], blend[va]) for _, va in outer])
    fold_aucs["blend"] = blend_fold.tolist()

    summary = {
        "d1": (float(np.mean(fold_aucs["d1"])), float(roc_auc_score(y, oof["d1"]))),
        "d2": (float(np.mean(fold_aucs["d2"])), float(roc_auc_score(y, oof["d2"]))),
        "blend": (float(blend_fold.mean()), float(roc_auc_score(y, blend))),
    }

    print("\n===== nested (outer seed 42, inner 15%) =====")
    for name, (mf, oof_auc) in summary.items():
        print(f"{name:6s} mean_fold={mf:.6f}  OOF={oof_auc:.6f}  "
              f"iters={iters.get(name, '-')}")
    print(f"reference non-nested: d1={REF_NON_NESTED['d1']:.6f} "
          f"d2={REF_NON_NESTED['d2']:.6f} blend={REF_NON_NESTED['blend']:.6f}")

    delta = summary["blend"][0] - summary["d1"][0]
    lo = hi = float("nan")
    if args.boot > 0:
        from significance_test import bootstrap_delta
        d, lo, hi, p = bootstrap_delta(y, blend, oof["d1"], args.boot, OUTER_SEED)
        print(f"nested blend - d1: {delta:+.6f}  "
              f"bootstrap 95% CI [{lo:+.6f}, {hi:+.6f}]  (p={p:.4f}, B={args.boot})")

    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("d1", "d2"):
        pd.DataFrame({config.ID_COL: train[config.ID_COL],
                      "oof_prob": oof[name]}).to_csv(
            OUT / f"oof_{name}.csv", index=False)

    rows = [
        dict(run="nested_v17_d1", mean_fold=summary["d1"][0],
             oof_auc=summary["d1"][1], best_iters=iters["d1"],
             reference_non_nested=REF_NON_NESTED["d1"],
             delta_vs_nested_v17=0.0, ci95_low=np.nan, ci95_high=np.nan),
        dict(run="nested_v18_blend", mean_fold=summary["blend"][0],
             oof_auc=summary["blend"][1], best_iters="d1+d2",
             reference_non_nested=REF_NON_NESTED["blend"],
             delta_vs_nested_v17=delta, ci95_low=lo, ci95_high=hi),
        dict(run="nested_d2_member", mean_fold=summary["d2"][0],
             oof_auc=summary["d2"][1], best_iters=iters["d2"],
             reference_non_nested=REF_NON_NESTED["d2"],
             delta_vs_nested_v17=summary["d2"][0] - summary["d1"][0],
             ci95_low=np.nan, ci95_high=np.nan),
    ]
    pd.DataFrame(rows).to_csv(RES, index=False)
    print(f"\nwritten: {RES}")
    print(f"OOF files: {OUT}")


if __name__ == "__main__":
    main()
