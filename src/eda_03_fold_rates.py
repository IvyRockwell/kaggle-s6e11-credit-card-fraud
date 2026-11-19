"""折内欺诈率统计：在每折训练部分内计算各特征分箱或取值的欺诈率，仅用于指导建模，不作为特征。"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

EDA_DIR = config.OUTPUTS_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.size"] = 9
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

train = pd.read_csv(config.TRAIN_PATH)
y = train[config.TARGET].to_numpy()
base_rate = y.mean()
print(f"全量基准欺诈率 = {base_rate:.4%}（仅作参照，折内统计不使用全量标签）")

skf = StratifiedKFold(
    n_splits=config.N_FOLDS,
    shuffle=True,
    random_state=config.RANDOM_STATE,
)
fold_ids = list(skf.split(train, y))

# ========== 一、分类列 + 布尔列 ==========
cat_bool_cols = config.CATEGORICAL_COLS + config.BOOLEAN_COLS
rows = []
# 结构：fold_rates[col] = {value: [5 折欺诈率]}，fold_counts 同理
fold_rates: dict[str, dict[str, list]] = {}
fold_counts: dict[str, dict[str, list]] = {}

for fold, (tr_idx, _va_idx) in enumerate(fold_ids):
    tr = train.iloc[tr_idx]
    for col in cat_bool_cols:
        if col in config.BOOLEAN_COLS:
            values = tr[col].map({True: "True", False: "False"})
        else:
            values = tr[col]
        stats = tr[config.TARGET].groupby(values).agg(["mean", "count"])
        fr = fold_rates.setdefault(col, {})
        fc = fold_counts.setdefault(col, {})
        for val, r in stats["mean"].items():
            fr.setdefault(str(val), []).append(r)
            fc.setdefault(str(val), []).append(int(stats.loc[val, "count"]))

for col in cat_bool_cols:
    for val, rates in fold_rates[col].items():
        counts = fold_counts[col][val]
        rows.append({
            "feature": col,
            "value": val,
            "mean_fraud_rate": np.mean(rates),
            "std_fraud_rate": np.std(rates, ddof=0),
            "min_fraud_rate": np.min(rates),
            "max_fraud_rate": np.max(rates),
            "mean_count": np.mean(counts),
            "n_folds_present": len(rates),
        })

cat_tbl = pd.DataFrame(rows).sort_values(
    ["feature", "mean_fraud_rate"], ascending=[True, False])
cat_tbl.to_csv(EDA_DIR / "fold_rates_categorical_boolean.csv",
               index=False, encoding="utf-8-sig")
print("已保存 fold_rates_categorical_boolean.csv")

# 每列一张图：均值欺诈率柱 + min/max 波动线，红色虚线为基准率。
for col in cat_bool_cols:
    sub = cat_tbl[cat_tbl["feature"] == col].sort_values(
        "mean_fraud_rate", ascending=False)
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x, sub["mean_fraud_rate"], color="#4C72B0", alpha=0.85)
    ax.errorbar(x, sub["mean_fraud_rate"],
                yerr=[sub["mean_fraud_rate"] - sub["min_fraud_rate"],
                      sub["max_fraud_rate"] - sub["mean_fraud_rate"]],
                fmt="none", ecolor="black", capsize=4, lw=1)
    ax.axhline(base_rate, color="red", ls="--", lw=1,
               label=f"基准率 {base_rate:.3%}")
    for i, (_, r) in enumerate(sub.iterrows()):
        ax.text(i, r["max_fraud_rate"],
                f"{r['mean_fraud_rate']:.3%}\nn={r['mean_count']:.0f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["value"], rotation=30, ha="right")
    ax.set_ylabel("折内欺诈率（5 折均值，误差线=min~max）")
    ax.set_title(f"{col} 折内欺诈率")
    ax.legend()
    fig.tight_layout()
    out = EDA_DIR / f"foldrate_{col}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out.name}")

# ========== 二、数值列：折内分位数分箱 ==========
N_BINS = 10
num_rows = []
# per_col_fold[col] = list of (edges, rates_by_bin_index)
per_col: dict[str, list] = {}

for fold, (tr_idx, _va_idx) in enumerate(fold_ids):
    tr = train.iloc[tr_idx]
    for col in config.NUMERIC_COLS:
        # qcut 仅用该折训练部分的特征值，不接触标签；重复边界自动丢弃。
        binned = pd.qcut(tr[col], q=N_BINS, duplicates="drop",
                         labels=False)
        edges = pd.qcut(tr[col], q=N_BINS, duplicates="drop").cat.categories
        stats = tr[config.TARGET].groupby(binned).mean()
        edge_list = [(iv.left, iv.right) for iv in edges]
        per_col.setdefault(col, []).append((edge_list, stats.to_dict()))

for col, fold_data in per_col.items():
    # 各折箱数可能不同（离散列），按箱位索引汇总。
    max_bins = max(len(e) for e, _ in fold_data)
    for b in range(max_bins):
        rates = [rdict[b] for _, rdict in fold_data if b in rdict]
        lefts = [e[b][0] for e, _ in fold_data if b < len(e)]
        rights = [e[b][1] for e, _ in fold_data if b < len(e)]
        num_rows.append({
            "feature": col,
            "bin_index": b,
            "mean_left_edge": np.mean(lefts),
            "mean_right_edge": np.mean(rights),
            "mean_fraud_rate": np.mean(rates),
            "std_fraud_rate": np.std(rates, ddof=0),
            "min_fraud_rate": np.min(rates),
            "max_fraud_rate": np.max(rates),
            "n_folds_present": len(rates),
        })

num_tbl = pd.DataFrame(num_rows)
num_tbl.to_csv(EDA_DIR / "fold_rates_numeric_binned.csv",
               index=False, encoding="utf-8-sig")
print("已保存 fold_rates_numeric_binned.csv")

# 每列一张图：各分位箱均值欺诈率折线 + min/max 阴影，x 轴标注箱区间。
for col in config.NUMERIC_COLS:
    sub = num_tbl[num_tbl["feature"] == col].sort_values("bin_index")
    x = sub["bin_index"].to_numpy()
    labels = [f"{r.mean_left_edge:.2g}~{r.mean_right_edge:.2g}"
              for r in sub.itertuples()]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, sub["mean_fraud_rate"], "o-", color="#C44E52")
    ax.fill_between(x, sub["min_fraud_rate"], sub["max_fraud_rate"],
                    color="#C44E52", alpha=0.2, label="5 折 min~max")
    ax.axhline(base_rate, color="red", ls="--", lw=1,
               label=f"基准率 {base_rate:.3%}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("折内分箱欺诈率")
    ax.set_title(f"{col} 折内十分位分箱欺诈率（分箱边界不依赖标签）")
    ax.legend()
    fig.tight_layout()
    out = EDA_DIR / f"foldrate_binned_{col}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out.name}")

print("\n声明：以上统计仅用于 EDA 理解信号；建模时目标编码/聚合必须在"
      "各自 CV 折内重新 fit，禁止复用本结果。")
print("EDA-03 完成。")
