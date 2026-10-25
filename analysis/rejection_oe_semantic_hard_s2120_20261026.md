# 外部 Outlier Exposure 对语义近邻新类发现的影响：seed=2120

## 1. 实验目的

本实验检验：在当前独立 rejection encoder 已使用外部 CIFAR-10 特征间隔约束的基础上，增加 rejection logits 的 uniform-logit Outlier Exposure（OE），是否能够改善 CIFAR-100 semantic-hard 场景下的未知类拒识和新类发现。

该实验针对当前已确认的核心瓶颈——语义相近未知类与已知类在 rejection 表征和分数空间中重叠——引入训练期外部未知样本代理。CIFAR-10 只作为无标签外部暴露数据使用；其类别标签没有进入训练目标。

需要明确的是，本实验验证的是“外部 OE 是否有助于 rejection 分支”，不是对训练期不确定性知识蒸馏本身有效性的直接验证。因此，即使结果为正，也不能据此宣称 uncertainty-KD 已经有效。

## 2. 对照设计与实验条件

### 2.1 数据与划分

- 数据集：CIFAR-100。
- 划分协议：fresh semantic-hard 60/40 split，seed=`2120`；60 个已知类、40 个潜在新类。
- discovery mixed pool：5,400 个样本，已知类先验为 0.2。
- 外部 OE 数据：CIFAR-10，仅用于训练期 rejection exposure。
- 两个实验臂使用完全相同的类别划分、teacher、student 初始化、训练预算、候选池、NNPU 目标、checkpoint 选择、阈值校准和聚类配置。

### 2.2 两个实验臂

| 实验臂 | rejection 训练项 | 变化因素 |
|---|---|---|
| control：`margin_only` | CIFAR-10 rejection feature-margin，权重 `0.05`，margin=`0.2` | `alpha_outlier_rejection_uniform=0` |
| treatment：`margin_plus_uniform` | control 全部训练项 + CIFAR-10 rejection uniform-logit OE，权重 `0.05` | `alpha_outlier_rejection_uniform=0.05` |

两臂均使用 standard logits KD、shared-backbone independent rejection encoder、mixed-pool NNPU、known-validation-only checkpoint selection、feature-rejector score 以及 known-validation-only 的 95% known coverage calibration。除 `alpha_outlier_rejection_uniform` 外，没有改变算法因素。

### 2.3 算法作用与修改理由

当前 rejection 分支通过特征间隔项使外部样本远离已知类 rejection 支持区域。新增的 uniform-logit OE 不把 CIFAR-10 映射到某个具体未知类别，而是要求 rejection logits 在外部样本上保持接近均匀分布，从而降低模型对外部样本的已知类别偏置。选择该项的理由是：它可以在不使用 mixed/test 真值的前提下，为 rejection 分支提供训练期的“非已知类”代理信号。

该修改没有改变分类 head 的标准 logits KD，也没有把 CIFAR-10 类别标签作为新类标签使用。由于外部数据与 CIFAR-100 semantic-near unknown 并不完全同分布，这一方法只能被视为代理暴露机制，必须通过固定 known coverage 下的未知拒识和已知类效用联合判断。

## 3. 实验完整性检查

检查结果如下：

- control 的 `outlier_rejection_uniform` 四个 epoch 均为 `0.0`。
- treatment 的该项四个 epoch 为 `5.494114、5.494877、5.420017、5.350533`，训练信号真实参与优化。
- 两臂均使用 `outlier_dataset=cifar10`、`independent_rejection_encoder=true`、`independent_rejection_encoder_init=shared_backbone`。
- 两臂校准 known acceptance 均为 `0.9490358`，说明阈值校准条件一致。
- `test_labels_used_for_training_or_calibration=false`。
- `mixed_pool_labels_used_for_training_or_calibration=false`。
- mixed pool 作为 unlabeled mixed PU 数据使用，`hard_labels_used=false`；未知真值仅用于最终回顾性指标。

因此，该结果可以作为有效的单因素配对实验结果使用。

## 4. 结果

测试集包含 6,000 个 known 样本和 4,000 个 unknown 样本。表中“变化”均为 treatment-control；百分比指标的变化以百分点（pp）表示。

| 指标 | control | treatment | 变化 |
|---|---:|---:|---:|
| AUROC | 0.812349 | 0.816958 | +0.004608 |
| AUPR | 0.718763 | 0.720796 | +0.002033 |
| FPR95 | 57.183% | 57.233% | +0.050 pp |
| OSCR | 0.605984 | 0.609894 | +0.003910 |
| known acceptance | 93.383% | 94.050% | +0.667 pp |
| calibrated known acceptance（validation） | 94.904% | 94.904% | 0 |
| unknown rejection | 33.825% | 32.350% | **-1.475 pp** |
| accepted-known accuracy | 72.461% | 72.461% | +0.0003 pp |
| accepted correct fraction of all known | 67.667% | 68.150% | +0.483 pp |
| candidate count | 1,750 | 1,651 | -99 |
| candidate purity | 77.314% | 78.377% | +1.063 pp |
| automatic K（true K=40） | 37 | 35 | 下降 2 |
| K absolute error | 3 | 5 | +2 |
| all-unknown NMI / ARI | 0.451151 / 0.211137 | 0.441755 / 0.209432 | -0.009396 / -0.001705 |
| candidate NMI / ARI | 0.512887 / 0.241362 | 0.503467 / 0.221576 | -0.009421 / -0.019786 |
| unknown-only NMI / ARI | 0.504150 / 0.247425 | 0.499955 / 0.262051 | -0.004195 / +0.014625 |

### 4.1 分数空间变化

| 指标 | control | treatment | 变化 |
|---|---:|---:|---:|
| KS distance | 0.470833 | 0.488250 | +0.017417 |
| unknown below known q95 | 71.000% | 71.050% | +0.050 pp |
| known above unknown q05 | 57.183% | 57.250% | +0.067 pp |
| unknown-known mean gap | 5.206753 | 5.193979 | -0.012773 |
| unknown-known median gap | 4.951496 | 5.079601 | +0.128104 |

KS distance和中位数间隔有所改善，但在本实验真正关心的 known-validation 95% coverage 工作点，unknown rejection 反而下降。unknown below known q95 几乎没有改善，说明全局排序分离没有转化为更有效的固定覆盖率拒识。

## 5. 与预设门槛的比较

协议要求 treatment 同时满足：unknown rejection 至少提升 2 pp；AUROC 不下降；FPR95 不恶化超过 1 pp；OSCR 和 accepted-known accuracy 不下降超过 0.005；聚类质量不能出现明显损害。

本实验中：

1. AUROC、AUPR 和 OSCR 小幅上升。
2. FPR95 仅恶化 0.050 pp，仍在容许范围内。
3. accepted-known accuracy 基本不变，known acceptance 上升 0.667 pp。
4. 但 unknown rejection 下降 1.475 pp，未达到至少提升 2 pp 的核心要求。
5. 自动 K 从 37 变为 35，K 绝对误差从 3 增加到 5；all-unknown NMI 下降，candidate NMI/ARI 下降，聚类结果没有形成一致改善。

因此，该 treatment 未通过联合门槛。

## 6. 与已有研究结论的关系

当前历史结果已经显示：若只看排序指标，某些 rejection 表征或后处理方法可以改善 AUROC、AUPR、FPR95 或 OSCR；但这些改善可能不转化为固定 known coverage 下的 unknown rejection。此次 OE 实验再次出现同样现象：AUROC 提升 `0.004608`，OSCR 提升 `0.003910`，但 unknown rejection 下降 `1.475 pp`。

该结果与此前 semantic-hard 场景中 prototype-boundary、feature-KD 和部分训练预算实验暴露出的“排序—工作点—聚类”不一致问题相吻合。它没有证明外部 OE 完全无效，但证明了当前固定配置下的 uniform-logit OE 不能作为完整新类发现方法的有效改进。

## 7. 决策

- 不修改默认配置。
- 不继续在 seed=`2120` 上扫描 uniform-logit OE 权重、margin 或阈值。
- 不把该结果写成 uncertainty-KD 成功证据。
- 保留该实验作为外部 OE 的正式负结果和可追溯 ablation。
- 后续优先研究 semantic-near unknown 的训练期覆盖和 support-aware rejection 表征；如果重新研究 OE，应使用新的预注册问题和新的 fresh split，并继续采用 unknown rejection、known utility、分数重叠、Auto-K 和聚类质量的联合门槛。

## 8. 可复现文件

- 协议：[`rejection_oe_semantic_hard_protocol_s2120_20261025.md`](rejection_oe_semantic_hard_protocol_s2120_20261025.md)
- 机器结果：[`rejection_oe_semantic_hard_s2120.json`](rejection_oe_semantic_hard_s2120.json)
- student 配置：[`rejection_oe_semantic_hard_s2120_student_config.json`](rejection_oe_semantic_hard_s2120_student_config.json)
- runner：[`../scripts/run_rejection_oe_semantic_hard_pair.py`](../scripts/run_rejection_oe_semantic_hard_pair.py)
- control discovery：`runs/rejection_oe_semantic_hard_s2120_margin_only_discover/discovery_report.json`
- treatment discovery：`runs/rejection_oe_semantic_hard_s2120_margin_plus_uniform_discover/discovery_report.json`

