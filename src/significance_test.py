"""显著性检验：DeLong 检验 + 分层配对 bootstrap，用于比较两个模型的 OOF AUC。

背景：仓库早期的"采纳门槛"用 5 折配对 t 检验（t >= 2.776）。5 个折级 AUC 来自同一批
数据、训练集互相重叠，不满足 t 检验的独立性假设，这个 t 没有有效的零分布，只能当
"方向一致性"的启发式指标。本脚本用两个标准工具替换它：

  1. DeLong 检验（Sun & Xu 2014 的快速实现）——同一批样本上两个 AUC 的差异检验；
  2. 分层配对 bootstrap——正负例分别有放回重抽，同一份重抽同时用于两个模型，
     给出 AUC 差值的 95% 百分位置信区间。

两项都直接用 70 万行 OOF，而不是 5 个折级数字。

用法：
    python src/significance_test.py                 # B=1000, seed=42
    python src/significance_test.py --boot 5000     # 更稳的 CI，约 1 分钟

输出：results/significance.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import config  # noqa: E402

OOF_DIR = PROJECT / "outputs"
OUT_CSV = PROJECT / "results" / "significance.csv"
EXPECTED_V18_OOF = 0.948114
TOL_V18 = 1e-6


# --------------------------------------------------------------------------
# DeLong
# --------------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    """平均秩（ties 取平均），O(n log n) 向量化实现。"""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = xs.size
    ranks = np.arange(1, n + 1, dtype=float)
    new_group = np.r_[True, xs[1:] != xs[:-1]]
    group_id = np.cumsum(new_group) - 1
    sums = np.bincount(group_id, weights=ranks)
    counts = np.bincount(group_id)
    avg = (sums / counts)[group_id]
    out = np.empty(n, dtype=float)
    out[order] = avg
    return out


def delong_paired(y: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    """返回 (auc1, auc2, p_value)，两个模型在同一批样本上的配对 DeLong 检验。

    记 tx 为全样本平均秩、tz 为正例内部平均秩、ty 为负例内部平均秩，则
        V10_i = (tx[pos_i] - tz_i) / n      # 正例的 placement value
        V01_j = 1 - (tx[neg_j] - ty_j) / m  # 负例的 placement value
    mean(V10) 即 AUC，Var(AUC) = S10/m + S01/n。
    """
    pos = y == 1
    neg = ~pos
    m = int(pos.sum())
    n = int(neg.sum())

    tx = np.vstack([_midrank(p1), _midrank(p2)])
    tz = np.vstack([_midrank(p1[pos]), _midrank(p2[pos])])
    ty = np.vstack([_midrank(p1[neg]), _midrank(p2[neg])])

    v10 = (tx[:, pos] - tz) / n            # (2, m)
    v01 = 1.0 - (tx[:, neg] - ty) / m      # (2, n)

    aucs = v10.mean(axis=1)
    cov = np.cov(v10) / m + np.cov(v01) / n   # 2x2

    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return float(aucs[0]), float(aucs[1]), float(p)


# --------------------------------------------------------------------------
# 分层配对 bootstrap（排序一次 + 加权 AUC）
# --------------------------------------------------------------------------
def _weighted_auc(y_sorted: np.ndarray, w_sorted: np.ndarray) -> float:
    """按分数升序排好的标签与权重，O(n) 算加权 AUC。"""
    wn = w_sorted * (1.0 - y_sorted)
    wp = w_sorted * y_sorted
    cum_neg = np.cumsum(wn) - wn
    denom = wp.sum() * wn.sum()
    return float(np.dot(wp, cum_neg) / denom)


def bootstrap_delta(y, p1, p2, n_boot, seed, alpha=0.05):
    """分层配对 bootstrap：返回 (delta_point_estimate, lo, hi, p_two_sided)。

    delta 用原始样本上的 AUC 差（点估计）；bootstrap 只负责给 95% CI 与 p。
    """
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)

    prep = []
    for p in (p1, p2):
        order = np.argsort(p, kind="mergesort")
        prep.append((order, y[order]))

    deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        w = np.zeros(y.size, dtype=float)
        w += np.bincount(rng.choice(pos_idx, size=pos_idx.size, replace=True),
                         minlength=y.size)
        w += np.bincount(rng.choice(neg_idx, size=neg_idx.size, replace=True),
                         minlength=y.size)
        aucs = [_weighted_auc(ys, w[order]) for order, ys in prep]
        deltas[b] = aucs[0] - aucs[1]

    delta = float(roc_auc_score(y, p1) - roc_auc_score(y, p2))  # 点估计
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # 双侧 bootstrap p：差值分布中与 0 相反的一侧占比的两倍
    frac = (np.sum(deltas <= 0) + 1) / (n_boot + 1) if delta > 0 else \
           (np.sum(deltas >= 0) + 1) / (n_boot + 1)
    p = float(min(1.0, 2 * frac))
    return delta, float(lo), float(hi), p


# --------------------------------------------------------------------------
def load_oof(path: Path, col: str = "oof_prob") -> np.ndarray:
    if not path.exists():
        sys.exit(f"missing artifact: {path}\nrun `python src/run_all.py` first.")
    return pd.read_csv(path)[col].to_numpy(dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    train = pd.read_csv(config.TRAIN_PATH, usecols=[config.ID_COL, config.TARGET])
    y = train[config.TARGET].to_numpy()
    n_pos = int(y.sum())
    print(f"rows={y.size} positives={n_pos} prevalence={y.mean():.5f}")

    p02 = load_oof(OOF_DIR / "v02_oof.csv")
    p17 = load_oof(OOF_DIR / "v17_oof.csv")
    p23 = load_oof(OOF_DIR / "v23_oof.csv")
    d2 = load_oof(OOF_DIR / "ensemble" / "d2_seed42_oof.csv", col="d2_oof")

    v18 = 0.5 * p17 + 0.5 * d2
    got = float(roc_auc_score(y, v18))
    if abs(got - EXPECTED_V18_OOF) > TOL_V18:
        sys.exit(f"v18 rebuilt from members gives {got:.6f}, "
                 f"expected {EXPECTED_V18_OOF:.6f}; artifacts may be stale.")
    print(f"v18 rebuilt from v17 + depth2: OOF AUC = {got:.6f}")

    models = {"v02": p02, "v17": p17, "v18": v18, "v23": p23}
    pairs = [("v17", "v02"), ("v18", "v17"), ("v18", "v02"), ("v23", "v18")]

    rows = []
    print(f"\nbootstrap B={args.boot} seed={args.seed} alpha={args.alpha}")
    print(f"{'comparison':<12s} {'delta':>10s} {'95% CI':>22s} "
          f"{'p_delong':>10s} {'p_boot':>9s}  verdict")
    for a, b in pairs:
        pa, pb = models[a], models[b]
        auc_a, auc_b, p_delong = delong_paired(y, pa, pb)
        delta, lo, hi, p_boot = bootstrap_delta(
            y, pa, pb, args.boot, args.seed, args.alpha)
        sig = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0"
        print(f"{a + '-' + b:<12s} {delta:+10.6f} "
              f"[{lo:+.6f}, {hi:+.6f}]  {p_delong:10.4f} {p_boot:9.4f}  {sig}")
        rows.append(dict(comparison=f"{a}-{b}", delta_auc=delta,
                         ci95_low=lo, ci95_high=hi, p_delong=p_delong,
                         p_bootstrap=p_boot, n_boot=args.boot, seed=args.seed))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nwritten: {OUT_CSV}")


if __name__ == "__main__":
    main()
