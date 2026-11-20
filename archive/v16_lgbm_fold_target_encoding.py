"""历史实验 v16：折内目标编码。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import fe_common  # noqa: E402

if __name__ == "__main__":
    fe_common.run_experiment(
        version="v16",
        sub_filename="v16_lgbm_fold_target_encoding.csv",
        feature_summary="29 features: 24 raw v02 features plus smoothed (smoothing=20) fold-internal target encodings for all 5 categorical columns; fitted only on each fold's training split then applied to validation and test; original categoricals kept; built on v02; early stopping per frozen protocol",
        build_fn=None,
        expected_features=29,
        te=True,
        transform_note="fold-internal smoothed target encoding, fit on training split only, no leakage",
    )
