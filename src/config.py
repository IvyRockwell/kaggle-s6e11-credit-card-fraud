"""全局配置：集中管理路径、列名与实验常量，避免在各脚本中硬编码。

所有路径均基于项目根目录解析，保证从任意工作目录运行都能定位到正确文件。
"""
from pathlib import Path

# ---------- 路径 ----------
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
SUBMISSIONS_DIR = PROJECT_DIR / "submissions"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

# ---------- 任务常量 ----------
ID_COL = "id"
TARGET = "is_fraud"
RANDOM_STATE = 42
N_FOLDS = 5  # 分层 K 折，少数类仅 1.67%，分层保证每折欺诈比例一致

# ---------- 列分组（依据赛题文件，首次 EDA 时用真实数据核对） ----------
NUMERIC_COLS = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h",
    "distance_from_home_km", "card_age_months", "customer_age",
    "account_balance_usd", "cvv_retry_count", "velocity_score",
    "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
]

CATEGORICAL_COLS = [
    "merchant_category", "card_type", "auth_method", "channel", "device_type",
]

BOOLEAN_COLS = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]

# ---------- 实验日志 ----------
EXPERIMENT_LOG_PATH = OUTPUTS_DIR / "experiment_log.csv"
