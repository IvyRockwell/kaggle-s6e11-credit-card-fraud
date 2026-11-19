"""历史实验 v15：距离 × 境外交易。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fe_common  # noqa: E402


def build_fn(train, test):
    for df in (train, test):
        df["distance_x_foreign"] = (
            df["distance_from_home_km"]
            * df["is_foreign_transaction"].astype(int)
        )
    return ["distance_x_foreign"]


if __name__ == "__main__":
    fe_common.run_experiment(
        version="v15",
        sub_filename="v15_lgbm_distance_foreign.csv",
        feature_summary="25 features: 24 raw v02 features plus distance_x_foreign=distance_from_home_km*is_foreign_transaction; source columns kept; previously SHAP rank 197/276 but formally tested under revised mechanism; built on v02; early stopping per frozen protocol",
        build_fn=build_fn,
        expected_features=25,
    )
