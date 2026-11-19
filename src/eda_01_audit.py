"""数据审计：核对行列数、dtype、缺失值、重复行与 id 唯一性，输出审计表到 outputs/eda/。"""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

EDA_DIR = config.OUTPUTS_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    lines.append(msg)


log("=" * 70)
log("EDA-01 数据核对")
log("=" * 70)

# 读取原始 train/test（只读，不修改原文件）。
train = pd.read_csv(config.TRAIN_PATH)
test = pd.read_csv(config.TEST_PATH)

# --- 1. shape ---
log("\n[1] shape 核对")
log(f"train shape = {train.shape}  期望 (699960, 26)  -> "
    f"{train.shape == (699960, 26)}")
log(f"test  shape = {test.shape}   期望 (299984, 25)  -> "
    f"{test.shape == (299984, 25)}")

# --- 2. 列名与 config 分组 ---
log("\n[2] 列名与 config 列分组核对")
expected_train_cols = (
    [config.ID_COL] + config.NUMERIC_COLS
    + config.CATEGORICAL_COLS + config.BOOLEAN_COLS + [config.TARGET]
)
expected_test_cols = (
    [config.ID_COL] + config.NUMERIC_COLS
    + config.CATEGORICAL_COLS + config.BOOLEAN_COLS
)
log(f"train 列与 config 一致 -> {list(train.columns) == expected_train_cols}")
log(f"test  列与 config 一致 -> {list(test.columns) == expected_test_cols}")
log(f"train 列数分组计数: 数值 {len(config.NUMERIC_COLS)}、"
    f"分类 {len(config.CATEGORICAL_COLS)}、布尔 {len(config.BOOLEAN_COLS)}、"
    f"id 1、target 1，合计 {len(expected_train_cols)}")
extra_test = set(train.columns) - set(test.columns)
log(f"test 比 train 少的列: {sorted(extra_test)}")

# --- 3. dtype ---
log("\n[3] 各列 dtype（train / test 并置）")
dtype_df = pd.DataFrame({
    "train_dtype": train.dtypes.astype(str),
    "test_dtype": test.dtypes.astype(str),
})
log("\n" + dtype_df.to_string())
dtype_df.to_csv(EDA_DIR / "audit_dtypes.csv", encoding="utf-8-sig")

# --- 4. 内存占用 ---
log("\n[4] 内存占用")
train_mem = train.memory_usage(deep=True).sum()
test_mem = test.memory_usage(deep=True).sum()
log(f"train 内存 = {train_mem / 1024**2:.2f} MB")
log(f"test  内存 = {test_mem / 1024**2:.2f} MB")

# --- 5. id 唯一性与 train/test id 交叠 ---
log("\n[5] id 唯一性")
log(f"train id 唯一 -> {train[config.ID_COL].is_unique}")
log(f"test  id 唯一 -> {test[config.ID_COL].is_unique}")
id_overlap = set(train[config.ID_COL]) & set(test[config.ID_COL])
log(f"train/test id 交叠数量 = {len(id_overlap)}（期望 0）")
log(f"train id 范围 = {train[config.ID_COL].min()} ~ "
    f"{train[config.ID_COL].max()}")
log(f"test  id 范围 = {test[config.ID_COL].min()} ~ "
    f"{test[config.ID_COL].max()}")

# --- 6. 重复行 ---
log("\n[6] 重复行检查（全部列）")
log(f"train 完全重复行数 = {int(train.duplicated().sum())}")
log(f"test  完全重复行数 = {int(test.duplicated().sum())}")
feat_cols = config.NUMERIC_COLS + config.CATEGORICAL_COLS + config.BOOLEAN_COLS
log(f"train 忽略 id/target 后特征重复行数 = "
    f"{int(train.duplicated(subset=feat_cols).sum())}")
log(f"test  忽略 id 后特征重复行数 = "
    f"{int(test.duplicated(subset=feat_cols).sum())}")

# --- 7. 缺失值 ---
log("\n[7] 缺失值数量")
miss = pd.DataFrame({
    "train_missing": train.isna().sum(),
    "test_missing": test.isna().sum(),
})
log("\n" + miss.to_string())
log(f"train 缺失总数 = {int(miss['train_missing'].sum())}；"
    f"test 缺失总数 = {int(miss['test_missing'].sum())}")

# --- 8. 布尔列合法性 ---
log("\n[8] 布尔列核对（取值必须恰为 True/False 且无缺失）")
for col in config.BOOLEAN_COLS:
    tr_vals = set(train[col].dropna().unique())
    te_vals = set(test[col].dropna().unique())
    log(f"{col}: dtype={train[col].dtype}, train 取值={tr_vals}, "
        f"test 取值={te_vals}, train 缺失={int(train[col].isna().sum())}, "
        f"test 缺失={int(test[col].isna().sum())}")

# --- 9. 分类列真实唯一值 ---
log("\n[9] 分类列真实唯一值集合")
for col in config.CATEGORICAL_COLS:
    tr_uv = sorted(train[col].dropna().unique().tolist())
    te_uv = sorted(test[col].dropna().unique().tolist())
    log(f"\n{col} ({train[col].dtype})")
    log(f"  train 唯一值 ({len(tr_uv)}): {tr_uv}")
    log(f"  test  唯一值 ({len(te_uv)}): {te_uv}")
    log(f"  两集合一致 -> {set(tr_uv) == set(te_uv)}")

# --- 10. 目标列合法性 ---
log("\n[10] 目标列 is_fraud 核对")
tvals = sorted(train[config.TARGET].unique().tolist())
log(f"is_fraud 取值 = {tvals}；缺失 = {int(train[config.TARGET].isna().sum())}")
vc = train[config.TARGET].value_counts().sort_index()
log(f"计数: {vc.to_dict()}")

(EDA_DIR / "audit_summary.txt").write_text(
    "\n".join(lines), encoding="utf-8")
log(f"\n核对结果已保存: {EDA_DIR / 'audit_summary.txt'}")
