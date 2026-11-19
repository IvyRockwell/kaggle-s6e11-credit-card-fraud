"""历史实验 v13：高风险类别与无认证标志。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fe_common  # noqa: E402


def build_fn(train, test):
    for df in (train, test):
        df["mc_high_risk"] = df["merchant_category"].isin(
            ["Gift Cards", "Crypto Exchange", "Gaming"]
        ).astype(int)
        df["auth_none"] = (
            df["auth_method"] == "No Authentication"
        ).astype(int)
    return ["mc_high_risk", "auth_none"]


if __name__ == "__main__":
    fe_common.run_experiment(
        version="v13",
        sub_filename="v13_lgbm_cat_risk_flags.csv",
        feature_summary="26 features: 24 raw v02 features plus mc_high_risk (merchant_category in Gift Cards/Crypto Exchange/Gaming) and auth_none (auth_method == No Authentication) binary flags; source categoricals kept; one risk-flag encoding group; built on v02; early stopping per frozen protocol",
        build_fn=build_fn,
        expected_features=26,
    )
