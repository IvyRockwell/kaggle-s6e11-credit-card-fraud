"""Artifact checks for the main pipeline (v02 -> v17 -> v18 -> v23).

Run after `python src/run_all.py --through v23`. It re-computes the recorded
numbers from the OOF files and validates the submission files, then exits
non-zero if anything disagrees.

Checks:
  1. every OOF / member file has aligned ids, no NaN, probabilities in (0, 1);
  2. v02, v17, v23 fold-mean and OOF AUC match outputs/experiment_log.csv;
  3. v18 can be rebuilt as 0.5 * v17 + 0.5 * depth2 and lands on 0.948114;
  4. submissions have the right header, row count, id order and value range.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import config  # noqa: E402

TOL_LOG = 1e-6
TOL_V18 = 1e-5
EXPECTED_V18_OOF = 0.948114
failures = []


def check(cond, label, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(label)


def load_oof(path, col="oof_prob"):
    check(path.exists(), f"file exists: {Path(path).name}")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    check(df[config.ID_COL].equals(train[config.ID_COL]),
          f"id alignment: {Path(path).name}")
    p = df[col].to_numpy(dtype=float)
    check(np.isfinite(p).all(), f"no NaN: {Path(path).name}")
    check((p > 0).all() and (p < 1).all(), f"probabilities in (0,1): {Path(path).name}")
    return p


def fold_aucs(p, seed):
    folds = list(StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y))
    return np.array([roc_auc_score(y[va], p[va]) for _, va in folds])


train = pd.read_csv(config.TRAIN_PATH)
test = pd.read_csv(config.TEST_PATH)
y = train[config.TARGET].to_numpy()

log = pd.read_csv(config.EXPERIMENT_LOG_PATH)
print(f"experiment_log rows: {len(log)}; ids={list(log['experiment_id'])}")

# ---------- 1. OOF artifacts ----------
print("\nOOF artifacts")
p02 = load_oof(PROJECT / "outputs" / "v02_oof.csv")
p17 = load_oof(PROJECT / "outputs" / "v17_oof.csv")
p23 = load_oof(PROJECT / "outputs" / "v23_oof.csv")
d2_42 = load_oof(PROJECT / "outputs" / "ensemble" / "d2_seed42_oof.csv", col="d2_oof")

# ---------- 2. v18 rebuilt from its two members ----------
if p17 is not None and d2_42 is not None:
    v18 = 0.5 * p17 + 0.5 * d2_42
    got = float(roc_auc_score(y, v18))
    check(abs(got - EXPECTED_V18_OOF) < TOL_V18,
          f"v18 rebuilt from members = {EXPECTED_V18_OOF:.6f}", f"got {got:.6f}")

# ---------- 3. recorded numbers ----------
print("\nrecorded numbers vs experiment_log.csv")
for vid, p in (("v02", p02), ("v17", p17), ("v23", p23)):
    if p is None:
        continue
    rows = log[log["experiment_id"] == vid]
    if len(rows) != 1:
        print(f"[ skip] {vid}: expected exactly 1 log row, found {len(rows)}")
        continue
    row = rows.iloc[0]
    got_fold = float(fold_aucs(p, 42).mean())
    check(abs(got_fold - float(row["mean_val_auc"])) < TOL_LOG,
          f"{vid} mean_val_auc matches log",
          f"{got_fold:.6f} vs {float(row['mean_val_auc']):.6f}")
    m = re.search(r"OOF(?: AUC)?=([0-9.]+)", str(row["notes"]))
    got_oof = float(roc_auc_score(y, p))
    check(m is not None and abs(got_oof - float(m.group(1))) < TOL_LOG,
          f"{vid} OOF AUC matches log",
          f"{got_oof:.6f} vs {m.group(1) if m else 'n/a'}")

# ---------- 4. submissions ----------
print("\nsubmissions")
SUBMISSIONS = (
    "v17_lgbm_additive_stumps.csv",
    "v18_lgbm_blend_d1_d2.csv",
    "v23_v18_seedavg_42_666_789.csv",
)
for name in SUBMISSIONS:
    path = PROJECT / "submissions" / name
    if not path.exists():
        print(f"[ skip] {name}: not generated yet (run run_all.py)")
        continue
    df = pd.read_csv(path)
    check(list(df.columns) == ["id", config.TARGET], f"header: {name}")
    check(len(df) == len(test), f"row count: {name}", f"{len(df)}")
    check(df["id"].equals(test[config.ID_COL]), f"id order: {name}")
    check(df[config.TARGET].between(0, 1).all(), f"probabilities in [0,1]: {name}")
    check(not df[config.TARGET].isin([0.0, 1.0]).any(), f"not rounded: {name}")
    check(df[config.TARGET].nunique() > 1000, f"continuous probabilities: {name}",
          f"nunique={df[config.TARGET].nunique()}")

print("\n" + ("ALL CHECKS PASSED" if not failures
              else f"FAILURES ({len(failures)}): {failures}"))
sys.exit(1 if failures else 0)
