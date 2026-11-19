"""分布分析：输出数值、分类与布尔列的统计表和分布图到 outputs/eda/。"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

EDA_DIR = config.OUTPUTS_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.size"] = 9
plt.rcParams["figure.dpi"] = 110
# 指定 Windows 自带中文字体，避免中文标题显示为方块；axes.unicode_minus 保证负号正常。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

train = pd.read_csv(config.TRAIN_PATH)
test = pd.read_csv(config.TEST_PATH)

# ---------- 数值列 ----------
# describe 全量统计（count/mean/std/分位数/极值），train/test 各一份并合并落盘。
desc_tr = train[config.NUMERIC_COLS].describe().T
desc_te = test[config.NUMERIC_COLS].describe().T
desc = desc_tr.join(desc_te, suffixes=("_train", "_test"))
desc.to_csv(EDA_DIR / "numeric_describe.csv", encoding="utf-8-sig")
print("数值 describe 已保存 numeric_describe.csv")

# 每列一张图，左为 train/test 直方图（y 轴对数，便于看长尾），右为箱线图。
for col in config.NUMERIC_COLS:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    tr_v = train[col].to_numpy()
    te_v = test[col].to_numpy()
    lo = min(np.nanmin(tr_v), np.nanmin(te_v))
    hi = max(np.nanmax(tr_v), np.nanmax(te_v))
    bins = np.linspace(lo, hi, 60)
    axes[0].hist(tr_v, bins=bins, alpha=0.55, label="train", density=True)
    axes[0].hist(te_v, bins=bins, alpha=0.55, label="test", density=True)
    axes[0].set_yscale("log")
    axes[0].set_title(f"{col} 直方图（密度，log y）")
    axes[0].legend()
    axes[1].boxplot(tr_v, orientation="vertical", showfliers=True, widths=0.5)
    axes[1].set_title(f"{col} 箱线图（train，含离群点）")
    fig.tight_layout()
    out = EDA_DIR / f"numeric_{col}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out.name}")

# ---------- 分类列 ----------
# 按 train 频次排序，画 train/test 占比分组条形图，标签上附 train 计数。
for col in config.CATEGORICAL_COLS:
    tr_counts = train[col].value_counts()
    te_counts = test[col].value_counts()
    categories = tr_counts.index.tolist()
    tr_prop = tr_counts / len(train)
    te_prop = te_counts.reindex(categories, fill_value=0) / len(test)
    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 0.2, tr_prop.values, width=0.4, label="train 占比")
    ax.bar(x + 0.2, te_prop.values, width=0.4, label="test 占比")
    for i, c in enumerate(categories):
        ax.text(i - 0.2, tr_prop.values[i], f"n={tr_counts[c]}",
                ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right")
    ax.set_ylabel("占比")
    ax.set_title(f"{col} 取值频次/占比（train n={len(train)} vs test n={len(test)}）")
    ax.legend()
    fig.tight_layout()
    out = EDA_DIR / f"categorical_{col}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out.name}")

# ---------- 布尔列 ----------
# 6 个布尔列合成一张 2x3 图，每柱显示 True/False 占比（train/test 对照）。
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, col in zip(axes.flat, config.BOOLEAN_COLS):
    tr_true = train[col].mean()
    te_true = test[col].mean()
    vals = ["False", "True"]
    tr_p = [1 - tr_true, tr_true]
    te_p = [1 - te_true, te_true]
    x = np.arange(2)
    ax.bar(x - 0.2, tr_p, width=0.4, label="train")
    ax.bar(x + 0.2, te_p, width=0.4, label="test")
    for i, p in enumerate(tr_p):
        ax.text(i - 0.2, p, f"{p:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(vals)
    ax.set_ylim(0, 1.08)
    ax.set_title(col)
axes.flat[0].legend()
fig.suptitle("布尔列取值占比（train vs test）")
fig.tight_layout()
out = EDA_DIR / "boolean_proportions.png"
fig.savefig(out)
plt.close(fig)
print(f"saved {out.name}")

# 同时输出布尔列精确计数表，便于报告引用。
bool_tbl = pd.DataFrame({
    "train_true": [int(train[c].sum()) for c in config.BOOLEAN_COLS],
    "train_true_pct": [train[c].mean() for c in config.BOOLEAN_COLS],
    "test_true": [int(test[c].sum()) for c in config.BOOLEAN_COLS],
    "test_true_pct": [test[c].mean() for c in config.BOOLEAN_COLS],
}, index=config.BOOLEAN_COLS)
bool_tbl.to_csv(EDA_DIR / "boolean_counts.csv", encoding="utf-8-sig")

# ---------- 标签 ----------
# 画目标计数/占比，量化 1.67% 的不平衡。
counts = train[config.TARGET].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(["0 非欺诈", "1 欺诈"], counts.values, color=["#4C72B0", "#C44E52"])
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v,
            f"n={v}\n({v / len(train):.4%})", ha="center", va="bottom")
ax.set_ylabel("计数")
ax.set_title("is_fraud 标签分布（train）")
fig.tight_layout()
out = EDA_DIR / "target_distribution.png"
fig.savefig(out)
plt.close(fig)
print(f"saved {out.name}")
print("EDA-02 完成。")
