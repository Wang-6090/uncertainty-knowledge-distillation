# 齐誉涵任务修改日志（2026-11-26）

## 修改目标

本轮按以下顺序补齐冻结候选池聚类审计：

1. 对全部真实未知样本增加 evaluation-only oracle-K 诊断；
2. 修正表征和方法选择口径，避免用回顾性测试标签结果描述“稳定表征”；
3. 增加匹配 L2 归一化的 raw feature 对照；
4. 为冻结池、数组、分析报告和运行记录补齐 SHA-256 完整性信息。

## 代码和协议变化

- `scripts/analyze_frozen_candidate_pool.py`
  - 校验 schema v2 的 `artifact_sha256` 和逐数组 `array_sha256`；
  - 对全部 4000 个真实未知样本运行 oracle K=40 的独立诊断；
  - 单独保存 `all_unknown_oracle_assignments.npz`；
  - 明确该诊断不参与候选、表征或方法选择；
  - 提供 `--skip-all-unknown-oracle`，供重复对照跳过昂贵诊断；
  - 聚类与特征准备实现保持离线自包含，不依赖未提交的训练管线扩展。
- `scripts/summarize_frozen_candidate_pool.py`
  - 要求六个单因素对照和一致的候选池 artifact 哈希；
  - 将 `feature_pca` 标为 `diagnostic_anchor_not_label_selected`；
  - 将算法比较标为 `exploratory_post_hoc_not_confirmatory`；
  - 明确区分 raw feature、L2-normalized raw feature 和 L2+PCA/whitening+L2；
  - auto K 暂缓，不再声称 K 估计不是瓶颈。
- `scripts/finalize_frozen_candidate_pool_records.py`
  - 为既有冻结池回填 schema v2 哈希；
  - 补齐根级 `command.txt`、`metrics.json` 和 `summary.md`；
  - 对重建命令明确标记 `reconstructed_from_saved_config`。
- `scripts/run_qiyuhan_frozen_pool_controls.ps1`
  - 对 seed 42、123、3407 统一运行六个对照。
- 两个冻结池测试文件覆盖 artifact 篡改拒绝、全部真实未知诊断、归一化匹配和汇总口径。

## 三种子结果

- 全部真实未知样本 oracle-K：ACC `0.2496 +/- 0.0026`，NMI `0.3336 +/- 0.0056`，ARI `0.1236 +/- 0.0014`。
- 冻结候选池中的真实未知子集：ACC `0.3407 +/- 0.0186`，NMI `0.6151 +/- 0.0280`，ARI `0.1008 +/- 0.0346`。
- 候选池纯度：`0.3482 +/- 0.0080`；未知召回：`0.0404 +/- 0.0033`。
- `feature_pca` 与匹配 L2 raw 对照分别为：
  - ACC：`0.3407 +/- 0.0186` vs `0.3292 +/- 0.0295`；
  - NMI：`0.6151 +/- 0.0280` vs `0.6011 +/- 0.0137`；
  - ARI：`0.1008 +/- 0.0346` vs `0.0985 +/- 0.0208`。
- 逐种子和逐指标胜负不完全一致，因此不把 PCA 或某个聚类方法升级为稳定默认方案。
- 当前直接瓶颈是候选池低召回和高已知污染，但结果不能排除表征或 K 估计误差。

完整汇总位于 `analysis/qiyuhan_frozen_candidate_pool_multiseed/`。

## Artifact 哈希

- seed 42：`044be2bbd95ffe26072e5ab70122740ab5e087fc160fe826210a3ab2fc6fdf2e`
- seed 123：`a51b16239ee18ae6b4b9dd339de1575318507848ba2a11e9db4f242291e2133a`
- seed 3407：`974020cf76eea8858927f63e5af0ee44a6ddbee4b94543e21b92be8c67edbc7b`

`runs/` 中的 NPZ 和逐运行 assignments 受仓库忽略规则保护，不随本次代码提交上传；上述哈希用于绑定本地冻结产物与已提交的聚合报告。

## 验证

```text
python -m pytest tests/test_frozen_candidate_pool_analysis.py tests/test_frozen_candidate_pool_summary.py -q
11 passed
```

同时通过 `py_compile` 和 `git diff --check`。
