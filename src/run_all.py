"""一键复现：从原始数据跑到最终提交。

用法：
    python src/run_all.py                 # 全链路 v02 → v17 → v18 → v23（约 60 分钟，8 核）
    python src/run_all.py --through v17   # 只到单模型基线（约 16 分钟）
    python src/run_all.py --skip-existing # 产物已存在则跳过，用于断点续跑

依赖链：
    v02 (archive) → v17 → v18 → v23
    v17 的配对基准是 v02 的 OOF；v18 复用 v17 的 OOF 与提交文件；v23 复用 v18 的成员配置。

run_all 只负责：跑之前把目录和数据准备好、按依赖顺序调用脚本、每阶段结束做产物校验。
每个阶段脚本仍可单独运行。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")

# (阶段名, 脚本, 该阶段必需的输入产物, 该阶段应产出的文件)
STAGES = [
    ("v02", "archive/v02_lgbm_earlystop.py",
     [], ["outputs/v02_oof.csv"]),
    ("v17", "src/v17_lgbm_additive_stumps.py",
     ["outputs/v02_oof.csv"],
     ["outputs/v17_oof.csv", "submissions/v17_lgbm_additive_stumps.csv"]),
    ("v18", "src/v18_blend_d1_d2.py",
     ["outputs/v17_oof.csv", "submissions/v17_lgbm_additive_stumps.csv"],
     ["submissions/v18_lgbm_blend_d1_d2.csv"]),
    ("v23", "src/v23_seed_average.py",
     ["outputs/v17_oof.csv", "submissions/v17_lgbm_additive_stumps.csv"],
     ["submissions/v23_v18_seedavg_42_666_789.csv"]),
]

REQUIRED_DATA = {
    "data/train.csv": 26,
    "data/test.csv": 25,
    "data/sample_submission.csv": 2,
}


def preflight() -> None:
    """把能提前发现的问题都发现掉：数据、依赖、目录。"""
    print("=" * 72)
    print("PREFLIGHT")
    print("=" * 72)

    missing = [p for p in REQUIRED_DATA if not (PROJECT / p).exists()]
    if missing:
        sys.exit(
            "缺少数据文件：" + ", ".join(missing) +
            "\n请从 Kaggle 比赛页下载 train.csv / test.csv / "
            "sample_submission.csv 放到 data/ 下（见 README「复现」）。"
        )
    for rel, ncol in REQUIRED_DATA.items():
        with open(PROJECT / rel, "rb") as f:
            header = f.readline().decode("utf-8").strip().split(",")
        if len(header) != ncol:
            sys.exit(f"{rel} 列数为 {len(header)}，期望 {ncol}；数据版本不对。")
        print(f"  [ok] {rel:34s} {ncol} 列")

    try:
        import lightgbm, numpy, pandas, sklearn  # noqa: F401
    except ImportError as e:  # pragma: no cover
        sys.exit(f"依赖缺失：{e}。先执行 pip install -r requirements-lock.txt")

    for d in ("outputs", "submissions"):
        (PROJECT / d).mkdir(parents=True, exist_ok=True)
    print("  [ok] 目录就绪（outputs/, submissions/）")

    # 各阶段脚本以 mode="a" 追加日志且不写表头。全链路若从 v02 起跑
    # （而不是从 v01 起跑），日志首行会是数据行，后续读日志的脚本会解析错。
    log_path = PROJECT / "outputs" / "experiment_log.csv"
    if not log_path.exists():
        log_path.write_text(
            "experiment_id,model,feature_summary,n_folds,mean_train_auc,"
            "mean_val_auc,train_val_gap,val_auc_std,submission_file,"
            "timestamp,notes\n",
            encoding="utf-8",
        )
        print("  [ok] outputs/experiment_log.csv 已建表头")

    print(f"  [ok] python {sys.version.split()[0]}")


def run_stage(name: str, script: str) -> None:
    t0 = time.time()
    print("\n" + "=" * 72)
    print(f"STAGE {name}: {script}")
    print("=" * 72, flush=True)
    rc = subprocess.call([PY, script], cwd=PROJECT, env=ENV)
    dt = time.time() - t0
    if rc != 0:
        sys.exit(f"阶段 {name} 失败（exit={rc}），已耗时 {dt / 60:.1f} 分钟。")
    print(f"STAGE {name} done in {dt / 60:.1f} min")


def check_outputs(outs: list[str]) -> None:
    for rel in outs:
        p = PROJECT / rel
        if not p.exists() or p.stat().st_size == 0:
            sys.exit(f"阶段产物缺失：{rel}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--through", default="v23", choices=[s[0] for s in STAGES],
                    help="跑到哪个阶段为止")
    ap.add_argument("--skip-existing", action="store_true",
                    help="产物已存在则跳过该阶段")
    args = ap.parse_args()

    preflight()
    t_all = time.time()

    for name, script, needs, outs in STAGES:
        if args.skip_existing and all((PROJECT / o).exists() for o in outs):
            print(f"[skip] {name}: 产物已存在")
        else:
            lacking = [p for p in needs if not (PROJECT / p).exists()]
            if lacking:
                sys.exit(
                    f"阶段 {name} 缺少上游产物：{lacking}\n"
                    f"请先运行更早的阶段（阶段不能跳过）。"
                )
            run_stage(name, script)
            check_outputs(outs)
        if name == args.through:
            break

    print("\n" + "=" * 72)
    print(f"全部完成，总耗时 {(time.time() - t_all) / 60:.1f} 分钟")
    print(f"提交文件：{PROJECT / 'submissions'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
