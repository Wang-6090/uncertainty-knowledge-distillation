# Prototype-boundary weighted rejector fresh replication: seed 2119

## 1. 研究问题与预注册条件

本实验检验固定的 `prototype_boundary` 候选加权是否能改善当前
`feature_rejector` 的未知类拒识。候选样本来自已知类原型支持区域边界附近的
无标签 discovery pool；该规则的目的，是让 NNPU 拒识器更多看到可能处于已知/未知
边界的样本，而不是用测试标签直接挑选未知样本。

实验在新的 semantic-hard CIFAR-100 60/40 split（seed=`2119`）上完成。两臂
只训练一份 split-matched teacher 和一份 shared-backbone student，然后在同一个
冻结 student checkpoint 上运行两个 rejector arm：

| 条件 | `all_pool` control | `prototype_boundary_weighted` treatment |
|---|---|---|
| discovery pool | matched mixed pool，5,400，known fraction=0.2 | 与 control 完全相同 |
| rejector | feature rejector + NNPU | feature rejector + NNPU |
| 候选池规则 | `all` | `weighted_boundary` + `prototype_boundary` |
| 候选比例 / 权重 | 配置保留为 0.25 / 2.0，但不执行边界加权 | 固定为 0.25 / 2.0 |
| threshold | known validation-only，目标已知覆盖率 95% | 完全相同 |
| clustering | disabled | disabled |

因此本实验只评价 detector/rejector，不把结果写成端到端新类聚类提升。两臂
配置审计确认除输出目录、`rejector_pool_selection` 和
`rejector_candidate_mode` 外没有差异；两臂均引用同一个 student checkpoint。
测试集和 mixed pool 的真值标签没有进入训练、checkpoint 选择、校准、候选选择
或方法选择，仅用于最终回顾性指标和标签流审计。

预注册联合门槛为：unknown rejection 至少提升 2 pp；known acceptance 损失不
超过 1 pp；FPR95 不恶化超过 1 pp；OSCR 不下降超过 0.005；accepted-known
accuracy 不下降超过 0.5 pp。该门槛是联合判据，不能用单个排序指标改善替代。

## 2. 机器结果与逐项差值

测试集共有 6,000 个 known 和 4,000 个 unknown。以下均为 treatment − control；
百分比指标的变化量以百分点（pp）表示。

| 指标 | all-pool control | prototype-boundary treatment | 变化 | 是否符合门槛 |
|---|---:|---:|---:|---|
| AUROC | 0.778810 | 0.789709 | +0.010899 | 排序改善 |
| AUPR | 0.671914 | 0.681410 | +0.009497 | 排序改善 |
| FPR95 | 64.717% | 58.750% | −5.967 pp | 通过 |
| OSCR | 0.583718 | 0.594945 | +0.011227 | 通过 |
| known acceptance | 94.383% | 94.350% | −0.033 pp | 通过 |
| unknown rejection | 25.325% | 25.050% | **−0.275 pp** | **失败：未达到 +2 pp** |
| accepted-known accuracy | 71.499% | 71.754% | +0.255 pp | 通过 |
| accepted correct fraction of all known | 67.483% | 67.700% | +0.217 pp | 通过 |
| known rejected | 337 / 6,000 | 339 / 6,000 | +2 | 工作点基本不变 |
| correctly rejected unknown | 1,013 / 4,000 | 1,002 / 4,000 | −11 | 未改善 |
| unknown false accept | 2,987 / 4,000 | 2,998 / 4,000 | +11 | 变差 |

分数分布的辅助审计也显示，treatment 的平均 unknown-minus-known 分数间隔从
`4.073937` 增加到 `5.152587`（+`1.078650`），KS distance 从 `0.421333` 增加
到 `0.442750`；但在实际 95% known-coverage 工作点，unknown below known-q95
从 `76.225%` 增加到 `76.900%`（+`0.675 pp`），说明整体排序分离改善没有转化为
可用的未知覆盖率。两臂 calibration known acceptance 都为 `0.949384`，校准
过程本身一致。

## 3. 与此前 prototype-boundary 结果的关系

此前固定配方在同一 semantic-hard 类别划分的三个训练 seed 上，工作点 unknown
rejection 相对各自 all-pool 的变化为：

| seed | unknown rejection 变化 |
|---:|---:|
| 2026 | +2.600 pp |
| 2027 | +0.625 pp |
| 2028 | −0.425 pp |
| 2119（本次新 split） | −0.275 pp |

因此 AUROC/FPR95/OSCR 的排序收益在本次仍然出现，但 calibrated unknown
rejection 的变化已经从早期的正向点估计变为负向，且在四次复核中没有稳定达到
预设的 +2 pp。2119 也没有显示通过明显降低 known acceptance 换取 rejection
收益：known acceptance 只下降 0.033 pp，但 unknown rejection 同时下降 0.275 pp。

## 4. 结论与后续决策

1. **该固定 prototype-boundary 加权在 seed=2119 上没有通过预注册联合门槛。**
   它改善了 AUROC、AUPR、FPR95、OSCR 和 accepted-known accuracy，但最重要的
   validation-calibrated unknown rejection 没有改善，反而下降 0.275 pp。
2. **不能将该结果称为完整新类发现方法有效。** clustering 已关闭，且候选规则
   的排序改善没有转化为未知类覆盖改善；因此不报告 Auto-K、NMI/ARI 作为该实验
   的成功证据。
3. **停止 prototype-boundary 的 ratio/weight 扫描和当前 split 上的继续试探。**
   继续调参只能增加测试集导向的选择风险，不能解决 semantic-near overlap。
4. 当前默认配置不变。后续优先级回到训练期、可由 known train/validation 合法
   驱动的 class-conditional support/outlier 表征目标，并继续同时报告排序、固定
   coverage 拒识、known utility 和分数重叠；若没有新的可验证机制，不再堆叠
   rejector 后处理变体。

## 5. 可追溯产物与检查

- 预注册协议：`analysis/prototype_boundary_fresh_replication_protocol_s2119_20261025.md`
- 机器结果：`analysis/prototype_boundary_fresh_replication_s2119.json`
- runner：`scripts/run_prototype_boundary_fresh_replication.py`
- 两臂均为 mixed unlabeled NNPU；JSON 中 `test_and_mixed_labels_used_for_training_selection_or_calibration=false`，并明确 `test_labels_used_only_for_retrospective_metrics=true`。
- `python -m py_compile scripts/run_prototype_boundary_fresh_replication.py` 通过。
- `pytest -q`：`317 passed, 2 warnings`。
- `git diff --check` 通过；仅报告和 README 更新属于本轮文档变更，未据此失败结果修改默认算法。
