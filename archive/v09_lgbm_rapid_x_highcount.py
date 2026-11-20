"""历史实验 v09：快速交易 × 高频交易。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import fe_common  # noqa: E402


def build_fn(train, test):
    for df in (train, test):
        rapid = (df["hours_since_last_txn"] < 0.85).astype(int)
        df["rapid_x_highcount"] = rapid * df["txn_count_last_24h"]
    return ["rapid_x_highcount"]


if __name__ == "__main__":
    fe_common.run_experiment(
        version="v09",
        sub_filename="v09_lgbm_rapid_x_highcount.csv",
        feature_summary="25 features: 24 raw v02 features plus rapid_x_highcount=(hours_since_last_txn<0.85)*txn_count_last_24h; rapid flag not separately included; SHAP interaction rank 20/276; built on v02; early stopping per frozen protocol",
        build_fn=build_fn,
        expected_features=25,
    )
