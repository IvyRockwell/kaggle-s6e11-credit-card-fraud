"""v23：v18 在 CV 种子 42/666/789 上的等权 logit 平均。
用于降低对单次折划分的依赖；权重固定为 1/3。"""
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import config  # noqa: E402

OUT = PROJECT / "outputs" / "seed_average"
OUT.mkdir(parents=True, exist_ok=True)
MEMBERS = PROJECT / "outputs" / "ensemble" / "members"
MEMBERS.mkdir(parents=True, exist_ok=True)
EPS = 1e-9

train = pd.read_csv(config.TRAIN_PATH)
test = pd.read_csv(config.TEST_PATH)
for col in config.CATEGORICAL_COLS:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category").cat.set_categories(
        train[col].cat.categories)

y = train[config.TARGET].to_numpy()
FEATURES = config.NUMERIC_COLS + config.CATEGORICAL_COLS + config.BOOLEAN_COLS
X = train[FEATURES]
X_test = test[FEATURES]

BASE_PARAMS = dict(
    objective="binary", learning_rate=0.05, min_child_samples=100,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
    random_state=config.RANDOM_STATE, n_jobs=-1, verbosity=-1,
)
MEMBER_PARAMS = {
    "d1": dict(BASE_PARAMS, n_estimators=25000, num_leaves=2, max_depth=1),
    "d2": dict(BASE_PARAMS, n_estimators=10000, num_leaves=4, max_depth=2),
}
PATIENCE = 300
SEEDS = (42, 666, 789)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def folds_of(seed):
    return list(StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=seed).split(X, y))


def fold_aucs(p, seed):
    return np.array([roc_auc_score(y[va], p[va]) for _, va in folds_of(seed)])


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    sd = float(d.std(ddof=1))
    t = float(d.mean() / (sd / np.sqrt(config.N_FOLDS))) if sd > 0 else float("nan")
    return float(d.mean()), sd, t, int((d > 0).sum())


def run_member(member, seed):
    """补跑某成员某折种子的 5 折，返回 OOF 与 5 折平均测试预测并落盘。"""
    t0 = time.time()
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(test))
    train_aucs, val_aucs, best_iters = [], [], []
    for k, (tr_idx, va_idx) in enumerate(folds_of(seed), start=1):
        model = LGBMClassifier(**MEMBER_PARAMS[member])
        model.fit(
            X.iloc[tr_idx], y[tr_idx],
            categorical_feature=config.CATEGORICAL_COLS,
            eval_X=X.iloc[va_idx], eval_y=y[va_idx], eval_metric="auc",
            callbacks=[early_stopping(PATIENCE, verbose=False)],
        )
        p_tr = model.predict_proba(X.iloc[tr_idx])[:, 1]
        p_va = model.predict_proba(X.iloc[va_idx])[:, 1]
        oof[va_idx] = p_va
        test_pred += model.predict_proba(X_test)[:, 1] / config.N_FOLDS
        train_aucs.append(roc_auc_score(y[tr_idx], p_tr))
        val_aucs.append(roc_auc_score(y[va_idx], p_va))
        best_iters.append(model.best_iteration_)
        print(f"  [{member} seed={seed}] fold {k}: best_iter={best_iters[-1]:5d} "
              f"val={val_aucs[-1]:.6f} ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame({config.ID_COL: train[config.ID_COL], config.TARGET: y,
                  "oof_prob": oof}).to_csv(
        MEMBERS / f"oof_{member}_seed{seed}.csv", index=False)
    pd.DataFrame({config.ID_COL: test[config.ID_COL],
                  config.TARGET: test_pred}).to_csv(
        MEMBERS / f"test_{member}_seed{seed}.csv", index=False)
    print(f"[{member} seed={seed}] OOF={roc_auc_score(y, oof):.6f} "
          f"mean_val={np.mean(val_aucs):.6f} iters={best_iters} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return dict(oof=oof, test=test_pred, best_iters=best_iters,
                secs=round(time.time() - t0, 1))


def main():
    # ---------- 装载/补跑各成员 ----------
    members = {seed: {} for seed in SEEDS}

    # seed 42：d1 复用 v17 产物；d2 重跑（与其它种子同一路径，避免依赖额外缓存）
    d1_42 = pd.read_csv(PROJECT / "outputs" / "v17_oof.csv")
    d1_test_42 = pd.read_csv(PROJECT / "submissions" /
                             "v17_lgbm_additive_stumps.csv")
    assert d1_42[config.ID_COL].equals(train[config.ID_COL])
    members[42]["d1"] = dict(oof=d1_42["oof_prob"].to_numpy(),
                             test=d1_test_42[config.TARGET].to_numpy())
    print("[seed 42] d1 reused from v17; running d2")
    members[42]["d2"] = run_member("d2", 42)

    for seed in (666, 789):
        print("=" * 70)
        print(f"seed {seed}: run d1 + d2 (with test predictions)")
        print("=" * 70)
        for member in ("d1", "d2"):
            members[seed][member] = run_member(member, seed)

    # ---------- 每个种子的 v18_s ----------
    v18 = {}
    for seed in SEEDS:
        oof = 0.5 * members[seed]["d1"]["oof"] + 0.5 * members[seed]["d2"]["oof"]
        tst = 0.5 * members[seed]["d1"]["test"] + 0.5 * members[seed]["d2"]["test"]
        v18[seed] = dict(oof=oof, test=tst)

    seed_rows = []
    for seed in SEEDS:
        fa = fold_aucs(v18[seed]["oof"], seed)
        seed_rows.append(dict(seed=seed, mean_fold=float(fa.mean()),
                              val_std=float(fa.std(ddof=0)),
                              oof=float(roc_auc_score(y, v18[seed]["oof"]))))
    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(OUT / "seed_summary.csv", index=False)
    print("\n===== per-seed v18 (own partition, clean) =====")
    print(seed_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # ---------- 三 seed 平均 ----------
    logit_avg_oof = sigmoid(np.mean([logit(v18[s]["oof"]) for s in SEEDS], axis=0))
    prob_avg_oof = np.mean([v18[s]["oof"] for s in SEEDS], axis=0)
    logit_avg_test = sigmoid(np.mean([logit(v18[s]["test"]) for s in SEEDS], axis=0))
    prob_avg_test = np.mean([v18[s]["test"] for s in SEEDS], axis=0)

    # 2-seed 参照
    combos = {
        "avg_42_666_logit": (42, 666),
        "avg_42_789_logit": (42, 789),
        "avg_666_789_logit": (666, 789),
    }
    cand_oof = {"v18_42": v18[42]["oof"], "avg3_prob": prob_avg_oof,
                "avg3_logit": logit_avg_oof}
    for name, ss in combos.items():
        cand_oof[name] = sigmoid(np.mean([logit(v18[s]["oof"]) for s in ss], axis=0))

    # 在 seed 42 划分上评估（配对 vs v18_42，仅作参考）
    folds42 = folds_of(42)
    aucs42 = {name: np.array([roc_auc_score(y[va], p[va]) for _, va in folds42])
              for name, p in cand_oof.items()}
    rows = []
    for name, a in aucs42.items():
        d, sd, t, pos = paired(a, aucs42["v18_42"])
        rows.append(dict(candidate=name, mean_fold_42=float(a.mean()),
                         oof_global=float(roc_auc_score(y, cand_oof[name])),
                         d_vs_v18_42=d, t=t, positive_folds=pos))
    # 干净口径：每个种子本划分的均值；平均值 = 三种子均值再平均
    clean_mean = float(seed_summary["mean_fold"].mean())
    rows.append(dict(candidate="clean_mean_of_3_seeds", mean_fold_42=np.nan,
                     oof_global=np.nan, d_vs_v18_42=np.nan, t=np.nan,
                     positive_folds=np.nan))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "average_results.csv", index=False)
    print("\n===== seed-averaged predictions (evaluated on seed42 folds) =====")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nclean per-seed fold mean: {clean_mean:.6f}")

    # ---------- 提交 ----------
    sub_path = PROJECT / "submissions" / "v23_v18_seedavg_42_666_789.csv"
    assert np.isfinite(logit_avg_test).all()
    assert 0.0 < logit_avg_test.min() and logit_avg_test.max() < 1.0
    pd.DataFrame({config.ID_COL: test[config.ID_COL],
                  config.TARGET: logit_avg_test}).to_csv(sub_path, index=False)
    chk = pd.read_csv(sub_path)
    assert list(chk.columns) == ["id", "is_fraud"]
    assert len(chk) == 299984
    assert chk["id"].equals(test[config.ID_COL])
    assert chk["is_fraud"].between(0, 1).all()
    assert not chk["is_fraud"].isin([0.0, 1.0]).any()
    # 敏感性版本（概率平均）只落盘备查
    pd.DataFrame({config.ID_COL: test[config.ID_COL],
                  config.TARGET: prob_avg_test}).to_csv(
        OUT / "v23_test_probavg.csv", index=False)

    pd.DataFrame({config.ID_COL: train[config.ID_COL], config.TARGET: y,
                  "oof_prob": logit_avg_oof}).to_csv(
        PROJECT / "outputs" / "v23_oof.csv", index=False)

    # ---------- 日志 ----------
    r = res.set_index("candidate")
    d_logit = float(r.loc["avg3_logit", "d_vs_v18_42"])
    t_logit = float(r.loc["avg3_logit", "t"])
    pos_logit = int(r.loc["avg3_logit", "positive_folds"])
    fold_mean_logit = float(r.loc["avg3_logit", "mean_fold_42"])
    oof_logit = float(r.loc["avg3_logit", "oof_global"])
    seed_txt = "; ".join(
        f"seed{int(row.seed)}: fold={row.mean_fold:.6f} oof={row.oof:.6f} "
        f"std={row.val_std:.6f}" for row in seed_summary.itertuples())
    notes = (
        f"seed-averaged v18 (seeds 42/666/789, equal weights 1/3, primary=logit "
        f"space); members configs identical to v18; per-seed clean results: {seed_txt}; "
        f"avg3_logit on seed42 folds: fold_mean={fold_mean_logit:.6f} "
        f"OOF={oof_logit:.6f}, paired vs v18_42: d={d_logit:+.6f} t={t_logit:+.2f} "
        f"pos={pos_logit}/5; avg3_prob OOF={float(r.loc['avg3_prob', 'oof_global']):.6f}; "
        f"clean mean of 3 seeds={clean_mean:.6f}; caveat: cross-seed OOF evaluation is "
        f"mildly optimistic for the average (other seeds' models may have trained on "
        f"other rows of the same fold), so the clean per-seed means and global OOF are "
        f"primary; submission uses logit-space equal-weight average of the 3 test "
        f"predictions"
    )
    row = {
        "experiment_id": "v23",
        "model": "lgbm_v18_seedavg_3",
        "feature_summary": (
            "24 raw features; v18 (0.5*LGB stumps + 0.5*LGB depth2) repeated on CV "
            "seeds 42/666/789; equal-weight logit-space seed average (prob average "
            "reported as sensitivity)"
        ),
        "n_folds": config.N_FOLDS,
        "mean_train_auc": "",
        "mean_val_auc": round(fold_mean_logit, 6),
        "train_val_gap": "",
        "val_auc_std": round(float(aucs42["avg3_logit"].std(ddof=0)), 6),
        "submission_file": sub_path.name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": notes,
    }
    pd.DataFrame([row]).to_csv(config.EXPERIMENT_LOG_PATH, mode="a",
                               header=False, index=False)

    print(f"\nsubmission -> {sub_path}")
    print(f"aux prob-avg test -> {OUT / 'v23_test_probavg.csv'}")
    print(f"v23 OOF -> {PROJECT / 'outputs' / 'v23_oof.csv'}")
    print("v23 done.")


if __name__ == "__main__":
    main()
