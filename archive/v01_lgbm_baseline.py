"""历史实验 v01：LightGBM 原始特征基线，固定 500 轮。"""
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config  # noqa: E402

# ---------- 第 1 步：读数据，分类列转 category（折外无标签操作，不接触 y） ----------
train = pd.read_csv(config.TRAIN_PATH)
test = pd.read_csv(config.TEST_PATH)

for col in config.CATEGORICAL_COLS:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category").cat.set_categories(
        train[col].cat.categories
    )

FEATURE_COLS = (
    config.NUMERIC_COLS + config.CATEGORICAL_COLS + config.BOOLEAN_COLS
)

# 安全检查：防列错位、防泄露、防陌生类别
assert len(FEATURE_COLS) == 24
assert config.ID_COL not in FEATURE_COLS and config.TARGET not in FEATURE_COLS
assert train.shape[1] == len(FEATURE_COLS) + 2
assert test.shape[1] == len(FEATURE_COLS) + 1
assert not test[config.CATEGORICAL_COLS].isna().any().any()

X = train[FEATURE_COLS]
y = train[config.TARGET]
X_test = test[FEATURE_COLS]

# ---------- 第 2 步：基线超参数（本阶段不调参；AUC 阈值无关，不默认加 scale_pos_weight） ----------
params = dict(
    objective="binary",          # 二分类，输出 sigmoid 概率
    n_estimators=500,            # 提升树总棵数（固定轮数，不做早停）
    learning_rate=0.05,          # 每棵树贡献的收缩步长，小步走更稳
    num_leaves=31,               # 每棵树最多叶子数，控制复杂度的主旋钮
    max_depth=-1,                # 不硬性限制树深，由 num_leaves 控制
    min_child_samples=100,       # 一个叶子至少包含的样本数，防小叶子过拟合
    subsample=0.8,               # 每棵树随机抽 80% 行，增加随机性
    colsample_bytree=0.8,        # 每棵树随机抽 80% 列，增加随机性
    reg_alpha=0.0,               # L1 正则，基线先关
    reg_lambda=1.0,              # L2 正则，LightGBM 默认值
    random_state=config.RANDOM_STATE,
    n_jobs=-1,                   # 用满 CPU
    verbosity=-1,                # 关训练日志
)

# ---------- 第 3 步：5 折分层 CV，折内训练，验证折不参与任何 fit ----------
skf = StratifiedKFold(
    n_splits=config.N_FOLDS,
    shuffle=True,
    random_state=config.RANDOM_STATE,
)

oof = np.zeros(len(train))                 # 全量 out-of-fold 验证概率
test_pred = np.zeros(len(test))            # 测试集累计概率，最后取平均
train_aucs, val_aucs = [], []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = LGBMClassifier(**params)
    model.fit(X_tr, y_tr, categorical_feature=config.CATEGORICAL_COLS)

    # 训练 AUC：用当前折训练部分自评；验证 AUC：模型没见过的验证折
    p_tr = model.predict_proba(X_tr)[:, 1]
    p_va = model.predict_proba(X_va)[:, 1]
    tr_auc = roc_auc_score(y_tr, p_tr)
    va_auc = roc_auc_score(y_va, p_va)

    oof[va_idx] = p_va
    test_pred += model.predict_proba(X_test)[:, 1] / config.N_FOLDS
    train_aucs.append(tr_auc)
    val_aucs.append(va_auc)
    print(f"fold {fold}: train AUC={tr_auc:.6f}  val AUC={va_auc:.6f}")

# ---------- 第 4 步：四件套 + 概率合法性校验 ----------
mean_train_auc = float(np.mean(train_aucs))
mean_val_auc = float(np.mean(val_aucs))
gap = mean_train_auc - mean_val_auc
val_std = float(np.std(val_aucs, ddof=0))
oof_auc = float(roc_auc_score(y, oof))   # 全量 OOF 整体 AUC，交叉参照

print("\n===== v01 四件套 =====")
print(f"mean train AUC      : {mean_train_auc:.6f}")
print(f"mean validation AUC : {mean_val_auc:.6f}")
print(f"train - val gap     : {gap:.6f}")
print(f"val AUC std         : {val_std:.6f}")
print(f"OOF overall AUC     : {oof_auc:.6f}")

assert 0.0 <= test_pred.min() and test_pred.max() <= 1.0
assert not np.isnan(test_pred).any() and not np.isnan(oof).any()

# ---------- 提交文件：表头 id,is_fraud，行数与 id 严格对齐 test ----------
sub = pd.DataFrame({config.ID_COL: test[config.ID_COL], config.TARGET: test_pred})
sub_path = config.SUBMISSIONS_DIR / "v01_lgbm_baseline.csv"
config.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
sub.to_csv(sub_path, index=False)

# 提交格式校验
check = pd.read_csv(sub_path)
assert list(check.columns) == ["id", "is_fraud"]
assert len(check) == 299984
assert check["id"].equals(test["id"])
assert check["is_fraud"].between(0, 1).all()

# 保留全量 OOF 概率，供后续分析/集成
oof_path = config.OUTPUTS_DIR / "v01_oof.csv"
pd.DataFrame({
    config.ID_COL: train[config.ID_COL],
    config.TARGET: y,
    "oof_prob": oof,
}).to_csv(oof_path, index=False)

# ---------- 第 5 步：登记实验日志（沿用现有 11 列表头） ----------
log_row = pd.DataFrame([{
    "experiment_id": "v01",
    "model": "lightgbm_4.7.0",
    "feature_summary": "24 raw features (13 numeric + 5 categorical native + 6 boolean), no FE",
    "n_folds": config.N_FOLDS,
    "mean_train_auc": round(mean_train_auc, 6),
    "mean_val_auc": round(mean_val_auc, 6),
    "train_val_gap": round(gap, 6),
    "val_auc_std": round(val_std, 6),
    "submission_file": sub_path.name,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "notes": f"baseline fixed 500 rounds lr=0.05 leaves=31; OOF AUC={oof_auc:.6f}",
}])
if config.EXPERIMENT_LOG_PATH.exists():
    log_row.to_csv(
        config.EXPERIMENT_LOG_PATH, mode="a", header=False, index=False
    )
else:
    log_row.to_csv(config.EXPERIMENT_LOG_PATH, index=False)

print(f"\n提交文件: {sub_path}")
print(f"OOF 文件: {oof_path}")
print(f"实验日志: {config.EXPERIMENT_LOG_PATH}")
print("v01 完成。")
