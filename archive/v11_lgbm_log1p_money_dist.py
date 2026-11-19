"""历史实验 v11：金额/距离列的 log1p 变换。"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fe_common  # noqa: E402

SRC = ["amount_usd", "account_balance_usd", "distance_from_home_km"]
NEW = [c + "_log1p" for c in SRC]


def build_fn(train, test):
    for df in (train, test):
        for src, new in zip(SRC, NEW):
            df[new] = np.log1p(df[src])
    return NEW


if __name__ == "__main__":
    fe_common.run_experiment(
        version="v11",
        sub_filename="v11_lgbm_log1p_money_dist.csv",
        feature_summary="27 features: 24 raw v02 features plus log1p versions of amount_usd, account_balance_usd, distance_from_home_km (originals kept); one tightly-coupled monotonic transform group; built on v02; early stopping per frozen protocol",
        build_fn=build_fn,
        expected_features=27,
    )
