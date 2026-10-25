# 基于不确定性知识蒸馏的新类发现方法

## 中期研究总结

**总结日期：** 2026-10-25  
**研究状态：** 第一阶段实验已阶段性结束，当前没有正在运行的 Python/GPU 实验进程。  
**主要数据协议：** CIFAR-100 semantic-hard 60/40 新类发现划分。  
**文档定位：** 本文件按照常见中期汇报结构整理，可直接作为汇报初稿；其中所有数值均来自已保存的实验报告、机器可读结果或协议审计文件。

## 一、研究背景与目标

### 1. 研究问题

本项目研究的是开放环境下的新类发现问题：训练阶段只拥有一部分已知类别监督，另有未标注的 mixed pool；模型需要在测试阶段同时完成：

1. 判断样本属于已知类还是潜在未知类；
2. 对已知类进行分类；
3. 对被拒识的未知样本进行自动聚类，发现潜在新类。

研究主题是“基于不确定性知识蒸馏的新类发现方法”。原始设想是利用 teacher 的不确定性指导 student 的知识蒸馏、拒识表征学习和候选新类发现。但经过系统审计，目前不能把 uncertainty-KD 宣称为已经验证有效的核心方法；当前最可靠的研究结论是：不确定性信号在 evaluation-only 的拒识融合中表现出一定互补性，但训练期 uncertainty-KD 尚未稳定转化为固定已知覆盖率下的未知拒识收益。

### 2. 核心科学问题

实验反复暴露出的主要瓶颈不是单一排序指标不足，而是：

> semantic-near novel 与 known 在共享分类表征、rejection 表征和最终拒识分数上存在明显重叠。

这会造成两个后果：

- AUROC、AUPR、FPR95、KS 等全局排序指标改善时，固定 known-coverage 工作点的 unknown rejection 仍可能下降；
- 未知样本候选池的总体纯度或自动聚类指标有所改善时，semantic-near 类别的覆盖和已知类保护仍可能不足。

因此，本项目始终把以下指标联合看待，而不是只看 AUROC：固定 known-validation calibration 下的 unknown rejection、known acceptance、accepted-known accuracy、OSCR、自动 K、ACC/NMI/ARI，以及 unknown below known q95 的分数重叠指标。

## 二、当前实验协议与真实性保障

### 1. 数据与训练设置

- 主要使用 CIFAR-100 semantic-hard 60/40 split：60 个 known 类、40 个 novel 类。
- 许多正式 fresh split 使用独立 seed，例如 2045、2046、2051、2052、2061-2065、2071-2079、2084-2103。
- mixed discovery pool 通常为 5,400 个样本，known prior 为 0.2；mixed pool 的未知身份不作为训练 hard label 使用。
- student 训练通常使用四个 epoch 的固定预算，checkpoint 通过 known validation 规则选择。
- 拒识阈值只使用 known validation 进行约 95% known-coverage calibration。
- test labels 只用于最终回顾性评估；不得用于训练、checkpoint 选择、阈值、融合权重、候选规则或方法选择。

### 2. 已修复的协议问题

整体代码审计发现并修复了会影响实验解释的实现问题：

1. `calibration_overlap_control` 原先可以出现在配置中，但在 `train_student` 构造数据时没有实际传递；现已统一通过 `build_bundle_from_args` 传递 split、open-validation、calibration、mixed-pool 和 transform 协议。
2. mixed pool 中不能把所有未标注样本直接当作 hard unknown 训练。当前 `mixed + rejector-training=hard + rejector-pool-selection=all` 会直接报错；mixed 协议必须使用 soft PU/nnPU，或显式声明候选伪标签语义。
3. uncertainty-KD 曾经是 CLI 隐式默认值，现已改为 `kd_mode=standard`；uncertainty-KD 只能通过显式 `--kd-mode uncertainty` opt-in。
4. discovery report 现在记录 `hard_labels_used`、training interpretation、score mode 和 test-label isolation，避免将 pure-unknown hard control、mixed PU 和伪未知候选实验混为一类。
5. rejection representation 审计工具已修复为按 checkpoint 配置构造 independent rejection encoder，避免错误地用 shared-feature 模型复核独立分支。
6. rejection-feature KD 实验新增 `teacher_rejection_feature_dim`，保证 teacher rejection checkpoint 在 control/treatment 间严格使用相同架构。

### 3. 当前验证状态

截至本阶段收束：

- `python -m pytest -q`：`317 passed, 2 warnings`。
- `python -m compileall -q train.py novel_discovery scripts tests`：通过。
- `git diff --check`：通过；仅有换行符 CRLF 提示，没有 whitespace error。
- 主要实验均保存了 protocol、配置、checkpoint、训练历史、calibration report、discovery report 和分析文档。
- 已确认没有正在运行的 Python/GPU 实验进程。

## 三、方法与系统组成

当前系统可分为四个部分：

1. **已知类分类与蒸馏：** teacher-student logits KD；standard KD 是当前默认对照，uncertainty-weighted KD 为显式消融。
2. **rejection representation：** 使用独立 rejection 分支或 shared-pretrained rejection backbone，产生拒识表征和 feature-rejector score。
3. **未知拒识与校准：** 由 rejection score、validation-only uncertainty-to-error calibration 和 known-validation 95% coverage threshold 构成可选拒识机制。
4. **新类发现：** 对候选未知样本进行自动聚类，报告自动 K 以及 ACC/NMI/ARI；这些指标只用于最终评估，不能使用 test labels 选择方法。

当前默认研究主线已收敛为：可靠的 shared-pretrained rejection representation + standard KD 对照；validation-calibrated uncertainty fusion 保留为显式 opt-in 候选，而不是默认分数；训练期 uncertainty-KD 不作为已验证增益。

## 四、关键实验结果

### 1. rejection encoder 初始化与表征质量

早期 seed=2042-2044 的结果曾被解释为“shared_backbone 初始化有效”，但审计发现 scratch CNN 与 shared ResNet18 同时改变了架构、容量和初始化来源，因此不能作为纯初始化因果证据。

随后完成了同架构 `shared_backbone` vs `random_backbone` 的正确对照：

| 指标 | seed=2045 copied - random | seed=2046 copied - random |
|---|---:|---:|
| AUROC | +0.01893 | +0.02273 |
| FPR95 | -3.85 pp | -5.82 pp |
| OSCR | +0.01477 | +0.01916 |
| Score KS | +0.01550 | +0.03142 |
| Unknown rejection | -1.25 pp | +6.10 pp |
| Accepted-known accuracy | -0.18 pp | +1.00 pp |

结论：shared-pretrained initialization 在两个 split 上稳定改善排序和分数分离，但固定工作点的 unknown rejection 方向不一致，且 seed=2045 存在 known acceptance trade-off。因此它是有依据的表征初始化候选，不是已经证明的完整新类发现方法，当前不改默认为“已解决”。

known-only 表征诊断进一步显示，scratch CNN 和 random ResNet18 的 rejection 分支存在明显容量/优化不足：seed=2040 independent scratch CNN 的 train-to-validation known-class probe accuracy 仅 `20.59%`，而 seed=2051 shared-pretrained ResNet18 为 `70.96%`。这说明先解决 rejection branch 的有效表征能力，比继续堆叠复杂拒识损失更重要。

### 2. 训练期拒识目标与表示约束

这些实验均使用单因素 paired protocol，主要结果如下：

| 方向 | 代表实验 | 结果 | 当前决定 |
|---|---|---|---|
| synthetic unknown 上的 VOS + rejection BCE | seeds 44, 123 | unknown rejection 分别下降 `2.37 pp`、`5.45 pp` | 不设为默认，停止继续扫 BCE 权重 |
| hard nnPU vs soft nnPU | seeds 44, 123 | loss 机制确实改变，但 AUROC、固定工作点拒识和 overlap 没有一致改善 | 不宣称 soft nnPU 有效 |
| PU class-conditional boundary | seed 2052 | AUROC `+0.00719`、FPR95 `-2.05 pp`、OSCR `+0.00622`，但 unknown rejection `-0.725 pp` | 排序正向、工作点失败，停止在该 split 调权重 |
| known CE late-decay | seeds 2054-2057, 2061-2063 | AUROC/KS/overlap 七次方向一致改善；unknown rejection 仅五次改善，seed=2063 下降 `2.78 pp` 且 accepted-known accuracy 下降 `1.10 pp` | 仅保留为 ranking-oriented candidate，不改默认 |
| variance floor | seed 2093 | loss/梯度真实非零，但 AUROC `-0.00146`、AUPR `-0.00357`、unknown rejection 仅 `+0.325 pp`，KS/overlap无改善 | 停止该方向 |
| rejection embedding 128d vs 64d | seed 2099 | AUROC `-0.00089`、FPR95 恶化 `1.32 pp`、unknown rejection `-1.58 pp` | 停止 dimension search |

其中 late-decay 是最明显的“排序改善不能等同工作点收益”案例：在七个 paired split 上 AUROC、KS 和 overlap proxy 全部改善，但 calibrated unknown rejection 仍有两个 split 回退。因此后续汇报不能只展示平均 AUROC。

### 3. uncertainty-KD 训练实验

#### 3.1 uncertainty-weighted logits KD：seeds 2071-2073

在相同 teacher、split、训练预算、mixed PU 和 validation-only calibration 下，将 standard logits KD 与 uncertainty-weighted logits KD 配对比较。训练审计确认 uncertainty 臂的非均匀权重真实生效，KD loss 非零，feature KD 为零。

三 split 平均差值（uncertainty KD - standard KD）为：

| 指标 | 平均变化 |
|---|---:|
| AUROC | +0.0009 |
| AUPR | -0.0032 |
| FPR95 | -0.84 pp |
| OSCR | +0.0099 |
| Unknown rejection | -3.32 pp |
| Auto-cluster NMI | -0.0037 |
| Auto-cluster ARI | -0.0058 |

结论：该训练期 uncertainty weighting 没有稳定改善未知拒识或聚类质量，不能恢复为默认，也不继续在原 split 上搜索 `exp(-u)` 或其他 weighting function。

#### 3.2 CE-only / standard-KD / uncertainty-KD：seed 2079

这是对当前完整辅助损失管线的三臂探索性消融：

| 指标 | CE-only | Standard KD | Uncertainty KD |
|---|---:|---:|---:|
| AUROC | 0.703847 | 0.708586 | 0.709437 |
| Unknown rejection | 14.30% | 10.05% | 11.95% |
| Auto NMI | 0.436954 | 0.485926 | 0.464595 |

uncertainty-KD 相对 standard KD 的 unknown rejection 提升 `1.90 pp`，但相对 CE-only 仍下降 `2.35 pp`，并且 NMI 下降 `0.021331`。这只是单 seed 探索性证据，不支持把 uncertainty-KD 作为稳定核心增益。

#### 3.3 直接 teacher uncertainty transfer：seed 2101

该实验将 frozen teacher uncertainty head 的标量作为 student rejection-uncertainty head 的直接回归目标。treatment 的 uncertainty regression loss 四个 epoch 均非零：`0.10910, 0.09345, 0.08793, 0.08549`，说明路径真实执行。

| 指标 | Confidence target | Teacher target | 变化 |
|---|---:|---:|---:|
| AUROC | 0.79122 | 0.79483 | +0.00361 |
| AUPR | 0.68374 | 0.67594 | -0.00780 |
| FPR95 | 0.60633 | 0.59400 | -1.23 pp |
| OSCR | 0.59700 | 0.60812 | +0.01112 |
| Unknown rejection | 26.15% | 19.90% | -6.25 pp |
| Accepted-known accuracy | 72.89% | 73.52% | +0.62 pp |

结论：student 确实学到了 teacher 的标量 uncertainty，但标量一致性没有转化为固定工作点 unknown rejection，停止该直接 target 变体。

#### 3.4 rejection-feature KD：seed 2102

为避免只匹配标量，进一步比较 `alpha_rejection_feat_kd=0` 与 `0.1`。treatment 的 feature KD loss 为 `0.227385, 0.165860, 0.144463, 0.133811`，对应 rejection gradient norm 为 `0.056047, 0.037592, 0.032083, 0.028235`；control 全程为零，说明 feature-KD 路径和梯度真实生效。但后续审计发现 teacher head uncertainty 在该 split 饱和为常数，因此不能将该实验解释为有效的样本级 uncertainty weighting 对照。

| 指标 | Control | Feature KD | 变化 |
|---|---:|---:|---:|
| AUROC | 0.800266 | 0.806794 | +0.006528 |
| AUPR | 0.697525 | 0.704745 | +0.007220 |
| FPR95 | 0.590333 | 0.586333 | -0.40 pp |
| OSCR | 0.612721 | 0.624250 | +0.011530 |
| Unknown rejection | 31.275% | 25.025% | -6.25 pp |
| Accepted-known accuracy | 74.491% | 74.630% | +0.14 pp |

结论：feature KD 同样改善排序和部分表征诊断，但核心 fixed-coverage unknown rejection 下降 `6.25 pp`，因此停止该 coefficient/target variant，不改默认。后续权重审计确认 teacher head uncertainty 输出饱和为 `1.0`，`exp(-u)` 权重标准差约 `2e-8`，所以该对照实质上是近似常数缩放，而非有信息的样本级重加权。fresh seed=2105 改用非退化 MC epistemic uncertainty 后，weight 标准差约 `0.0145`；unknown rejection 相对 uniform 仍下降 `1.25 pp`，AUROC/AUPR 仅小幅改善且 NMI/ARI 下降。MC weighting 也未通过联合标准，但这是可解释的真实机制结果。详见 [`rejection_feature_kd_mc_epistemic_s2105_20261025.md`](rejection_feature_kd_mc_epistemic_s2105_20261025.md)。

### 4. validation-only uncertainty fusion：当前最有希望但仍未晋升默认的方向

与训练期 uncertainty-KD 不同，validation-only fusion 不改变 frozen student 和 checkpoint，只在 discovery 阶段把 feature-rejector score 与 known-validation 校准的 uncertainty risk 按固定 `0.5/0.5` 融合。没有在测试集上调融合权重。

在七个 fresh split（2084、2085、2086、2089、2092、2098、2103）上，paired audit 确认除 `score_mode` 外协议字段一致，threshold policy 一致，mixed pool 使用 nnPU，`hard_labels_used=false`，test labels 未用于训练或校准。

| 指标 | 平均 treatment - control | 标准差 |
|---|---:|---:|
| AUROC | +0.011677 | 0.006555 |
| AUPR | +0.025344 | 0.015468 |
| FPR95 | -0.733 pp | 1.227 pp |
| OSCR | +0.006075 | 0.003008 |
| Unknown rejection | +4.393 pp | 3.424 pp |
| Known acceptance | -0.286 pp | 0.636 pp |
| Accepted-known accuracy | -0.087 pp | 0.188 pp |

预注册联合标准要求：unknown rejection 至少提升 2 pp、FPR95 不恶化超过 1 pp、OSCR 下降不超过 0.005、accepted-known accuracy 下降不超过 0.005。结果为 `5/7` split 通过。

重要反例是 seed=2103：

- AUROC：`0.804584 → 0.812403`；
- AUPR：`0.696878 → 0.707485`；
- FPR95：`0.582667 → 0.563167`；
- OSCR：`0.608177 → 0.612898`；
- unknown rejection：`28.475% → 28.075%`，下降 `0.40 pp`。

因此当前准确表述是：fixed validation-only uncertainty fusion 是有平均收益、但存在 split-dependent operating-point trade-off 的强 opt-in 候选；它不能被表述为已稳定解决 semantic-near overlap，也不能据此证明训练期 uncertainty-KD 有效。当前不改默认，不继续搜索融合权重或 conjunction 变体。

另一个预注册 conjunction 变体（seed=2090）将两个风险信号取 `min`，结果 AUROC `0.813643 → 0.802724`，unknown rejection `32.925% → 28.975%`，下降 `3.95 pp`，因此停止。该负结果说明要求两个相关信号同时高会损失未知覆盖。

### 5. hard-unknown 候选与 selector 方向

候选池审计显示，当前 label-free selector 的总体 purity 不能代表所有 novel 类覆盖：在冻结 checkpoint 上，entropy+uncertainty selector 的 novel recall 约为 `28.43%`，但最低 novel class recall 只有 `2.63%`；class-balanced 配额虽将最低 recall 提高到 `4.39%`，总体 novel recall 从 `28.43%` 降到 `27.50%`，known contamination 从 `9.04%` 升至 `12.00%`。因此不能仅依据总体候选纯度推进候选选择器。

MC selector 的 fresh replication 也不稳定：seed=2100 的 raw-sum MC selector 相比 entropy control，AUROC `0.78725 → 0.77700`，AUPR `0.67502 → 0.66572`，FPR95 恶化 `1.35 pp`，unknown rejection 下降 `1.15 pp`。结合 seeds 2096-2100 的混合证据，停止 candidate ratio、MC 次数和 score weight 搜索。

这些结果把问题进一步定位到“hard-unknown 信号覆盖不足”，尤其是 semantic-near novel 类别的尾部覆盖，而不是简单增加 uncertainty 来源或 selector 复杂度。

## 五、已确认无效或不应继续的方向

以下方向已有有效负结果或跨 split 不稳定证据，当前阶段不再继续扫参：

- 训练期 scalar uncertainty transfer 与 uncertainty-weighted logits KD；
- rejection-feature KD 的当前 teacher target/coefficient；补充的 seed=2106
  uniform feature-KD vs no-KD 归因复核仍为负，未支持 feature-KD 本身；
- VOS、直接 rejection BCE、额外 PU/边界损失的组合搜索；
- variance-floor 和 rejection embedding dimension 搜索；
- MC-dropout/raw-sum selector、候选比例和 MC 次数搜索；
- conjunction 规则、fusion 权重和后处理 score mode 搜索；
- 未经同架构控制的 scratch CNN vs shared-backbone 因果解释；
- 使用 test/mixed-pool label 选择候选、阈值、checkpoint 或融合规则。

历史负结果没有作为“无效文件”永久删除，而是保留在 `analysis/` 和归档目录中，便于追溯和防止重复实验。已经明确错误 wiring、无法形成有效对照的产物移入 `runs/_archive_confirmed_invalid_20261024/`，不参与主表和结论。

## 六、当前阶段性结论

### 已经得到的可靠结论

1. 实验协议和代码链路已经经过系统审计，数据划分、训练、校准、发现和最终评估的边界比早期版本清晰，主要高风险协议问题已修复。
2. 当前主要困难是 semantic-near novel/known 的表征和分数重叠，而不是单纯缺少一个 uncertainty score。
3. shared-pretrained 同架构 rejection encoder 在排序和分数分离上有重复正向证据，但 fixed-coverage unknown rejection 仍有 split-dependent trade-off。
4. 训练期 uncertainty-KD 的 loss、权重、teacher target 和 feature target 均可以真实执行，但目前没有稳定的核心拒识收益；seed=2106 的 uniform feature-KD vs no-KD 配对还显示 feature-KD 本身在该 fresh split 上全面退化。
5. 共享 rejection 分支由 4 epoch 延长到 8 epoch，在 seed=2107–2109 三个 fresh split 上稳定改善 AUROC/AUPR/FPR95/OSCR 和已知类接受后准确率；但固定覆盖 unknown rejection 仅 2/3 split 改善，NMI/ARI 三个 split 均下降，因此 8 epoch 不能据此升为默认的新类发现方法。
5. validation-only uncertainty fusion 是当前最有希望的 rejection opt-in 候选：七 split 平均指标改善、5/7 通过联合标准，但仍存在 seed=2103 这样的反例，因此不应晋升为默认。
6. 候选池整体 purity 或平均排序指标不能替代 semantic-near 类别覆盖、固定工作点 unknown rejection 和 known utility 的联合验证。

### 尚未证明的结论

- 尚未证明“基于不确定性知识蒸馏”本身能稳定提升新类发现。
- 尚未证明 uncertainty-KD 能解决 semantic-near novel/known overlap。
- 尚未证明当前自动聚类质量的提升能够稳定转化为新类发现收益。
- 尚未证明 validation-only fusion 在不同数据集、不同预算或更长训练 schedule 下仍保持收益。

## 七、当前方法状态建议

建议在中期汇报中将方法状态表述为：

> 本研究已完成从代码协议审计、拒识表征诊断、训练期不确定性蒸馏消融到 validation-only 不确定性融合复核的系统实验。结果表明，训练期 uncertainty-KD 尚未形成稳定的新类发现增益；validation-only uncertainty fusion 在 7 个 semantic-hard fresh split 上取得平均 AUROC `+0.01168`、AUPR `+0.02534`、unknown rejection `+4.39 pp`，其中 5/7 通过预注册联合标准，但仍存在 split-dependent operating-point trade-off。因此当前工作应被定位为“已经完成严格实验验证、明确核心瓶颈、获得一个有前景但尚未定型的拒识候选机制”，而不是“核心方法已经被证明有效”。

## 八、下一阶段优先级

### P0：保持协议稳定

- 不改变默认 `kd_mode=standard`。
- 不使用测试标签反向调整 fusion、threshold、checkpoint 或候选选择。
- 新实验必须预注册单一变化因素，并保存逐样本 validation score、sample id、split 和 checkpoint 元数据。
- 每次报告必须同时给出排序、固定工作点拒识、known utility、聚类和 overlap 指标。

### P1：直接研究 hard-unknown 信号覆盖

优先使用训练数据、known validation、CIFAR-100 coarse taxonomy 和模型预测等不泄漏信息，研究 semantic-near novel 类的 label-free coverage。先做冻结模型的候选覆盖审计，再决定是否进入完整训练配对；不能直接使用 mixed/test 标签选择规则。

### P2：改善 rejection representation 的容量与保持已知类能力

如果重新研究 rejection encoder，应只选择一个因素，例如同架构容量、初始化、归一化或训练时长，并加入 known-only probe、有效秩、participation、known-class retention 等机制检查。不要同时叠加 VOS、OE、额外 PU、candidate gating 和 uncertainty-KD。

### P3：校准迁移稳定性

validation-only fusion 的主要剩余风险是 validation-to-test coverage shift。下一步应保存 validation score 的逐样本产物，进行 validation-only bootstrap/OOB 的阈值稳定性审计，再在全新 split 上一次性评估 coverage transfer、unknown rejection 和 known utility。bootstrap 只能描述 validation 抽样稳定性，不能被解释为 test coverage 或 unknown rejection 的置信区间。

## 九、可复现材料索引

### 总体审计与清理

- [`full_program_research_theme_audit_20261024.md`](full_program_research_theme_audit_20261024.md)
- [`cleanup_confirmed_invalid_20261024.md`](cleanup_confirmed_invalid_20261024.md)

### 当前最重要的汇总

- [`shared_rejection_fusion_fresh_2084_2103_summary_20261024.md`](shared_rejection_fusion_fresh_2084_2103_summary_20261024.md)
- [`shared_rejection_fusion_fresh_s2103_20261024.md`](shared_rejection_fusion_fresh_s2103_20261024.md)
- [`rejection_feature_kd_s2102_20261024.md`](rejection_feature_kd_s2102_20261024.md)
- [`rejection_feature_kd_uniform_vs_none_protocol_s2106_20261025.md`](rejection_feature_kd_uniform_vs_none_protocol_s2106_20261025.md)
- [`rejection_epoch_budget_protocol_s2107_20261025.md`](rejection_epoch_budget_protocol_s2107_20261025.md)
- [`rejection_epoch_budget_protocol_s2108_20261025.md`](rejection_epoch_budget_protocol_s2108_20261025.md)
- [`rejection_epoch_budget_protocol_s2109_20261025.md`](rejection_epoch_budget_protocol_s2109_20261025.md)
- [`teacher_uncertainty_transfer_s2101_20261024.md`](teacher_uncertainty_transfer_s2101_20261024.md)
- [`rejection_embedding_compression_s2099_20261024.md`](rejection_embedding_compression_s2099_20261024.md)
- [`mc_rawsum_selector_pair_s2100_20261024.md`](mc_rawsum_selector_pair_s2100_20261024.md)

### 表征与训练机制

- [`rejection_encoder_init_attribution_audit_20261020.md`](rejection_encoder_init_attribution_audit_20261020.md)
- [`rejection_encoder_same_arch_s2045_20261020.md`](rejection_encoder_same_arch_s2045_20261020.md)
- [`rejection_encoder_same_arch_s2046_20261020.md`](rejection_encoder_same_arch_s2046_20261020.md)
- [`rejection_ce_decay_four_seed_summary_20261021.md`](rejection_ce_decay_four_seed_summary_20261021.md)
- [`rejection_ce_decay_s2063_replication_20261022.md`](rejection_ce_decay_s2063_replication_20261022.md)
- [`rejection_variance_floor_s2093_20261024.md`](rejection_variance_floor_s2093_20261024.md)

## 十、最终摘要

本阶段已经完成了研究问题的严格化和实验协议的清理，证明了多条看似合理的改进路径在固定 known coverage 下并不可靠。当前最有价值的发现不是某个单一参数带来的提升，而是明确了评价重点：必须同时解决 semantic-near unknown 的信号覆盖、rejection representation 的有效容量、校准到测试工作点的迁移，以及自动聚类的长尾覆盖。

在现有证据下，validation-only uncertainty fusion 可以作为后续研究的强 opt-in 候选，但不能替代训练期 uncertainty-KD，也不能被写成已验证的核心方法。下一阶段应围绕 hard-unknown coverage 和 support-aware rejection representation 做更直接、更可解释的单因素实验，并继续保持当前的标签隔离和配对审计标准。
