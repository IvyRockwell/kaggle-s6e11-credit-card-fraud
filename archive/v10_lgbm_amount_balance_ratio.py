"""历史实验 v10：amount_usd / account_balance_usd 比值。"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fe_common  # noqa: E402


def build_fn(train, test):
    for df in (train, test):
        df["amount_to_balance_ratio"] = (
            df["amount_usd"] / df["account_balance_usd"]
        )
        assert np.isfinite(df["amount_to_balance_ratio"]).all()
    return ["amount_to_balance_ratio"]


if __name__ == "__main__":
    fe_common.run_experiment(
        version="v10",
        sub_filename="v10_lgbm_amount_balance_ratio.csv",
        feature_summary="25 features: 24 raw v02 features plus amount_to_balance_ratio=amount_usd/account_balance_usd; source columns kept; built on v02; early stopping per frozen protocol",
        build_fn=build_fn,
        expected_features=25,
    )
