# Results

- `experiment_log.csv`: 23 个实验（v01–v23）的四件套指标与结论。
- `public_scores.csv`: 两次 Kaggle 提交的 Public 分数。
- `capacity_curve.csv` / `catboost_curve.csv` / `xgboost_curve.csv` / `additive_logit_grid.csv`: 模型容量与编码的对照结果。
- `seed_summary.csv` / `seed_average_results.csv`: 多种子平均的逐种子与融合结果。
- `stack_results.csv` / `stack_ablation.csv` / `v18_fold_details.csv` / `blend_results.csv`: 集成与逐折明细。
- `nested_es_audit.csv` / `seed_robustness.csv` / `model_checks.csv` / `positive_control.csv`: 早停偏差、种子稳健性、阳性对照与模型形式审计。
- `eda_fold_rates_categorical_boolean.csv`: 分类/布尔特征的折内欺诈率。

说明：仓库只保留最终两个提交（v18、v23）；v01–v22 的历史提交归档在本地，未纳入版本控制。
