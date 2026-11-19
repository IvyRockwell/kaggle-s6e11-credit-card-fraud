"""历史实验 v14：金额最高十分位标志。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fe_common  # noqa: E402


def build_fn(train, test):
    for df in (train, test):
        df["amount_top_decile"] = (df["amount_usd"] > 89.87).astype(int)
    return ["amount_top_decile"]


if __name__ == "__main__":
    fe_common.run_experiment(
        version="v14",
        sub_filename="v14_lgbm_amount_top_decile.csv",
        feature_summary="25 features: 24 raw v02 features plus amount_top_decile=(amount_usd>89.87) binary flag; original amount_usd kept; 89.87 is label-free top-decile boundary; built on v02; early stopping per frozen protocol",
        build_fn=build_fn,
        expected_features=25,
    )
