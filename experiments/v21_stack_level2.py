"""v21：二级集成实验，比较固定权重与嵌套 5 折学习权重。"""
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import config  # noqa: E402

OUT = PROJECT / "outputs" / "ensemble2"
OUT.mkdir(parents=True, exist_ok=True)
MEMBERS_DIR = PROJECT / "outputs" / "ensemble" / "members"
T_CRIT = 2.776
SEEDS = (42, 123)
STACK_C = 1.0
EPS = 1e-9

train = pd.read_csv(config.TRAIN_PATH)
test = pd.read_csv(config.TEST_PATH)
y = train[config.TARGET].to_numpy()
IDs = train[config.ID_COL]
TEST_IDS = test[config.ID_COL]

# ---------- 成员 OOF / 测试预测 ----------
OOF_PATHS = {
    42: {
        "d1": (PROJECT / "outputs" / "v17_oof.csv", "oof_prob"),
        "d2": (PROJECT / "outputs" / "ensemble" / "d2_seed42_oof.csv", "d2_oof"),
        "cb": (MEMBERS_DIR / "oof_cb_seed42.csv", "oof_prob"),
        "lg": (MEMBERS_DIR / "oof_lg_seed42.csv", "oof_prob"),
    },
    123: {
        "d1": (MEMBERS_DIR / "oof_d1_seed123.csv", "oof_prob"),
        "d2": (MEMBERS_DIR / "oof_d2_seed123.csv", "oof_prob"),
        "cb": (MEMBERS_DIR / "oof_cb_seed123.csv", "oof_prob"),
        "lg": (MEMBERS_DIR / "oof_lg_seed123.csv", "oof_prob"),
    },
}
TEST_PATHS = {
    "d1": PROJECT / "submissions" / "v17_lgbm_additive_stumps.csv",
    "d2": MEMBERS_DIR / "test_d2_seed42.csv",
    "cb": MEMBERS_DIR / "test_cb_seed42.csv",
    "lg": MEMBERS_DIR / "test_lg_seed42.csv",
}
MEMBER_NAMES = ["d1", "d2", "cb", "lg"]


def load_oof(seed):
    mats = {}
    for m in MEMBER_NAMES:
        path, col = OOF_PATHS[seed][m]
        df = pd.read_csv(path)
        assert df[config.ID_COL].equals(IDs), f"{path} id 与 train 不一致"
        p = df[col].to_numpy(dtype=float)
        assert np.isfinite(p).all(), f"{path} 含 NaN"
        assert (p > 0).all() and (p < 1).all(), f"{path} 概率不在 (0,1)"
        mats[m] = p
    return mats


def load_test():
    mats = {}
    for m in MEMBER_NAMES:
        df = pd.read_csv(TEST_PATHS[m])
        assert df[config.ID_COL].equals(TEST_IDS), f"{TEST_PATHS[m]} id 不一致"
        p = df[config.TARGET].to_numpy(dtype=float)
        assert np.isfinite(p).all()
        assert (p > 0).all() and (p < 1).all()
        mats[m] = p
    return mats


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fold_aucs(p, seed):
    folds = list(StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y))
    return np.array([roc_auc_score(y[va], p[va]) for _, va in folds])


def paired(a_aucs, b_aucs):
    d = np.asarray(a_aucs) - np.asarray(b_aucs)
    sd = float(d.std(ddof=1))
    t = float(d.mean() / (sd / np.sqrt(config.N_FOLDS))) if sd > 0 else float("nan")
    return float(d.mean()), sd, t, int((d > 0).sum())


def nested_stack(z, seed):
    """B2 嵌套评估：第 k 折的堆叠器只在其余折 OOF 行上拟合。"""
    folds = list(StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y))
    Z = np.column_stack(z)
    oof = np.zeros(len(y))
    coefs = []
    for tr_idx, va_idx in folds:
        model = LogisticRegression(C=STACK_C, solver="lbfgs", max_iter=500)
        model.fit(Z[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict_proba(Z[va_idx])[:, 1]
        coefs.append(model.coef_[0].copy())
    return oof, coefs


rows, fold_rows = [], []
adopted = None
evaluations, z42, oof42, p_b2_by_seed, p_b1_by_seed = {}, {}, {}, {}, {}
t0 = time.time()
for seed in SEEDS:
    print("=" * 70)
    print(f"seed {seed}")
    print("=" * 70)
    oof = load_oof(seed)
    z = {}
    for m in MEMBER_NAMES:
        z[m] = (logit(oof[m]) - logit(oof[m]).mean()) / logit(oof[m]).std(ddof=0)
        print(f"  member {m}: OOF AUC={roc_auc_score(y, oof[m]):.6f}")
    p_b0 = 0.5 * oof["d1"] + 0.5 * oof["d2"]
    p_b1 = sigmoid(np.mean([z[m] for m in MEMBER_NAMES], axis=0))
    p_b2, coefs = nested_stack([z[m] for m in MEMBER_NAMES], seed)
    p_b1_by_seed[seed], p_b2_by_seed[seed] = p_b1, p_b2
    if seed == 42:
        z42, oof42 = z, oof

    names = {"v18": p_b0, "B1_fixed": p_b1, "B2_learned": p_b2, **{
        f"member_{m}": oof[m] for m in MEMBER_NAMES}}
    aucs = {k: fold_aucs(p, seed) for k, p in names.items()}
    print("  fold AUCs:")
    for k, a in aucs.items():
        print(f"    {k:12s}: OOF={roc_auc_score(y, names[k]):.6f} "
              f"mean_fold={a.mean():.6f}")
    evaluations[seed] = dict(aucs=aucs, oof=names, oof_prob=dict(
        {k: roc_auc_score(y, p) for k, p in names.items()}))

    for label, cand in (("B1_fixed", "v18"), ("B2_learned", "v18"),
                        ("B2_learned", "B1_fixed")):
        d_bar, s_d, t, pos = paired(aucs[label], aucs[cand])
        print(f"  paired {label} - {cand}: d_bar={d_bar:+.6f} s_d={s_d:.6f} "
              f"t={t:+.3f} positive={pos}/5")
        rows.append(dict(seed=seed, comparison=f"{label}_vs_{cand}",
                         d_bar=d_bar, s_d=s_d, t=t, positive_folds=pos))
    for m in MEMBER_NAMES:
        d_bar, s_d, t, pos = paired(aucs["B2_learned"], aucs[f"member_{m}"])
        rows.append(dict(seed=seed, comparison=f"B2_learned_vs_{m}",
                         d_bar=d_bar, s_d=s_d, t=t, positive_folds=pos))
    for k in range(config.N_FOLDS):
        fold_rows.append(dict(seed=seed, fold=k + 1,
                              **{key: float(aucs[key][k]) for key in aucs}))
    print(f"  B2 stacker coefs (mean over folds): "
          f"{np.mean(coefs, axis=0).round(4)}")

# ---------- 成员相关性（seed 42 OOF logits） ----------
corr = pd.DataFrame(
    {a: {b: spearmanr(logit(oof42[a]), logit(oof42[b])).statistic
         for b in MEMBER_NAMES} for a in MEMBER_NAMES}
).loc[MEMBER_NAMES, MEMBER_NAMES]
corr.to_csv(OUT / "member_corr.csv")
print("\nmember Spearman corr (seed42 OOF logits):")
print(corr.round(4).to_string())

# ---------- 预注册采纳规则 ----------
def qualifies(cand):
    stats = {s: dict(zip(("d_bar", "s_d", "t", "pos"),
                         paired(evaluations[s]["aucs"][cand],
                                evaluations[s]["aucs"]["v18"])))
             for s in SEEDS}
    pos_total = sum(st["pos"] for st in stats.values())
    both_pos = all(st["d_bar"] > 0 for st in stats.values())
    t_ok = any(st["t"] >= T_CRIT for st in stats.values())
    mean_d = float(np.mean([st["d_bar"] for st in stats.values()]))
    return dict(stats=stats, pos_total=pos_total, both_pos=both_pos,
                t_ok=t_ok, mean_d=mean_d), (both_pos and pos_total >= 8 and t_ok)


q1, ok1 = qualifies("B1_fixed")
q2, ok2 = qualifies("B2_learned")
print("\n===== adoption rule (protocol 5.9) =====")
for name, q, ok in (("B1_fixed", q1, ok1), ("B2_learned", q2, ok2)):
    print(f"{name}: mean_d={q['mean_d']:+.6f} pos_folds={q['pos_total']}/10 "
          f"both_seeds_positive={q['both_pos']} t_ok={q['t_ok']} -> {ok}")

if ok2 and q2["mean_d"] >= q1["mean_d"] + 0.0001:
    adopted = "B2_learned"
elif ok1:
    adopted = "B1_fixed"
else:
    adopted = None
    signs = [q2["stats"][s]["d_bar"] > 0 for s in SEEDS]
    if any(signs) and not all(signs):
        print("NOTE: seeds disagree in sign for B2 -> seed 777 tie-break "
              "required before any adoption")

print(f"\nADOPTED: {adopted if adopted else 'none (keep v18, not evidenced)'}")

# ---------- 最终提交（seed 42 成员测试预测） ----------
test_preds = load_test()
z_test = {}
for m in MEMBER_NAMES:
    lm = logit(oof42[m])
    z_test[m] = (logit(test_preds[m]) - lm.mean()) / lm.std(ddof=0)
Z_test = np.column_stack([z_test[m] for m in MEMBER_NAMES])
p_test_b1 = sigmoid(np.mean([z_test[m] for m in MEMBER_NAMES], axis=0))
stack_model = LogisticRegression(C=STACK_C, solver="lbfgs", max_iter=500)
stack_model.fit(np.column_stack([z42[m] for m in MEMBER_NAMES]), y)
p_test_b2 = stack_model.predict_proba(Z_test)[:, 1]
p_test_v18 = 0.5 * test_preds["d1"] + 0.5 * test_preds["d2"]

final_name = adopted if adopted else "B2_learned"
test_final = {"v18": p_test_v18, "B1_fixed": p_test_b1,
              "B2_learned": p_test_b2}[final_name]
assert np.isfinite(test_final).all()
assert (test_final > 0).all() and (test_final < 1).all()
sub_path = PROJECT / "submissions" / "v21_stack_level2.csv"
pd.DataFrame({config.ID_COL: TEST_IDS, config.TARGET: test_final}
             ).to_csv(sub_path, index=False)

chk = pd.read_csv(sub_path)
assert list(chk.columns) == ["id", "is_fraud"]
assert len(chk) == 299984
assert chk["id"].equals(test[config.ID_COL])
assert chk["is_fraud"].between(0, 1).all()
assert not chk["is_fraud"].isin([0.0, 1.0]).any()

# ---------- v21 OOF（seed 42 嵌套预测；采纳 B1 时写 B1） ----------
oof_for_v21 = (p_b1_by_seed[42] if adopted == "B1_fixed"
               else p_b2_by_seed[42])
pd.DataFrame({
    config.ID_COL: IDs, config.TARGET: y,
    "oof_prob": oof_for_v21,
}).to_csv(PROJECT / "outputs" / "v21_oof.csv", index=False)

# ---------- 结果表 ----------
pd.DataFrame(rows).to_csv(OUT / "stack_results.csv", index=False)
pd.DataFrame(fold_rows).to_csv(OUT / "stack_fold_details.csv", index=False)

# ---------- 实验日志 ----------
v18_oof = roc_auc_score(y, 0.5 * oof42["d1"] + 0.5 * oof42["d2"])
b1_oof = roc_auc_score(y, p_b1_by_seed[42])
b2_mean_fold = float(evaluations[42]["aucs"]["B2_learned"].mean())
b1_mean_fold = float(evaluations[42]["aucs"]["B1_fixed"].mean())
pairs_txt = "; ".join(
    f"seed{r['seed']} {r['comparison']}: d={r['d_bar']:+.6f} t={r['t']:+.2f} "
    f"pos={r['positive_folds']}/5" for r in rows
    if r["comparison"] in ("B1_fixed_vs_v18", "B2_learned_vs_v18",
                           "B2_learned_vs_B1_fixed"))
notes = (
    f"level-2 ensemble; members d1(v17),d2(lgb depth2),cb(v19 catboost),"
    f"lg(v20 additive logit); B1=fixed equal-weight z-scored logit avg, "
    f"B2=L2 logistic stacker C={STACK_C} nested 5-fold; seeds {SEEDS}; "
    f"v18 OOF={v18_oof:.6f}; B1 OOF={b1_oof:.6f} mean_fold={b1_mean_fold:.6f}; "
    f"B2 nested mean_fold={b2_mean_fold:.6f}; adopted={adopted or 'none (keep v18)'}; "
    f"pairs: {pairs_txt}; leakage caveat: nested stacker trained on member OOF "
    f"rows only, 4 coefs, C fixed; test blend from seed42 members"
)
row = {
    "experiment_id": "v21",
    "model": "stack_level2_logit",
    "feature_summary": (
        "24 raw features via 4 base members: v17 depth1 stumps, LGB depth2, "
        "CatBoost (v19), additive logistic regression (v20); level-2 on member "
        "OOF logits (fixed equal weights / nested L2 logistic stacker)"
    ),
    "n_folds": config.N_FOLDS,
    "mean_train_auc": "",
    "mean_val_auc": round(evaluations[42]["aucs"][final_name].mean(), 6),
    "train_val_gap": "",
    "val_auc_std": round(float(evaluations[42]["aucs"][final_name].std(ddof=0)),
                         6),
    "submission_file": sub_path.name,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "notes": notes,
}
pd.DataFrame([row]).to_csv(config.EXPERIMENT_LOG_PATH, mode="a",
                           header=False, index=False)

print(f"\nB1 OOF={b1_oof:.6f} | v18 OOF={v18_oof:.6f}")
print(f"submission ({final_name if adopted else 'best candidate ' + final_name})"
      f" -> {sub_path}")
print(f"outputs -> {OUT}")
print(f"done in {time.time()-t0:.0f}s")
