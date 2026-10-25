# 中期进展报告实验部分填写稿

适用模板：`附件2-北京交通大学大学生创新创业训练计划项目中期进展报告模板.docx`

本文只处理模板中的研究、实验、进展、困难和阶段成果，不填写项目基本信息、教师意见、成员分工和签字内容。粘贴到 Word 后，应删除模板中的“填写说明”文字，并根据实际项目起止时间补充月份。

## 一、栏目对应关系

| 模板栏目 | 建议填写内容 | 说明 |
|---|---|---|
| 三（一）主要研究内容 | 研究问题、方法组成、实验指标 | 写“研究什么、怎么验证”，不要写本阶段结果 |
| 三（二）实施进度安排 | 前期、中期、后期的实验安排 | 写计划和阶段目标，不虚构未完成成果 |
| 四（一）已开展的工作 | 已完成的代码审计、协议设计、训练和消融实验 | 这是实验部分的主体 |
| 四（二）困难及解决办法 | semantic-near 重叠、工作点收益不稳定、候选覆盖不足 | 重点写问题和已经采取的解决措施 |
| 五、阶段性成果 | 100 字以内的最重要结论 | 只保留阶段结论，详细数据可作为附件 |

## 二、三（一）主要研究内容

本项目面向开放环境下的新类发现问题，研究在仅具有部分已知类别标注、并存在未标注混合样本的条件下，如何同时完成已知类分类、未知样本拒识和未知样本聚类。主要研究内容包括：

1. 构建 teacher-student 新类发现框架，比较标准 logits 知识蒸馏与显式不确定性知识蒸馏，分析教师预测不确定性对学生分类和拒识表征的影响。
2. 设计独立 rejection representation 分支，研究 shared-backbone 初始化、拒识特征蒸馏和混合池 nnPU 训练对已知类保持能力、未知类分数分离和固定覆盖率拒识的影响。
3. 研究基于 known validation 的不确定性校准与拒识分数融合方法，在不使用测试标签选择规则的前提下，评价未知类拒识和已知类识别之间的平衡。
4. 对拒识得到的候选未知样本进行自动聚类，使用自动聚类类别数、ACC、NMI、ARI 及候选池纯度和类别覆盖评价新类发现质量。
5. 建立严格的成对消融实验协议，统一数据划分、训练预算、checkpoint 选择、阈值校准和标签隔离，重点考察 semantic-near novel 与 known 的表征和分数重叠问题。

## 三、三（二）实施进度安排

可按项目实际起止月份调整下列“第 1 阶段”等表述：

**第 1 阶段：问题分析与系统搭建。** 完成新类发现、开放集拒识、不确定性估计和知识蒸馏相关文献调研；完成 CIFAR-100 semantic-hard 60/40 数据划分、teacher-student 训练流程、rejection 分支、混合池 nnPU 训练、known-validation 阈值校准和自动聚类评价代码。

**第 2 阶段：实验协议审计与基线建立。** 固定 known validation 约 95% coverage 的工作点，明确 mixed pool 不使用未知真值 hard label，禁止测试标签参与训练、checkpoint 选择、阈值和方法选择；建立 standard KD、shared-pretrained rejection representation 等对照，并完成代码、配置、数据边界和结果保存审计。

**第 3 阶段：不确定性蒸馏与拒识表征消融。** 完成 uncertainty-weighted logits KD、teacher uncertainty transfer、rejection-feature KD、rejection encoder 初始化和容量等成对实验；同时记录 loss、梯度、排序指标、固定工作点拒识、known utility 和聚类指标，判断算法路径是否真实生效及是否转化为端到端收益。

**第 4 阶段：不确定性融合与新类候选覆盖研究。** 在冻结 student 的条件下，开展 validation-only uncertainty fusion 和候选选择器复核；重点分析 semantic-near novel 的候选覆盖、类别均衡性和 known contamination，停止没有跨 split 支持的权重、阈值和 selector 搜索。

**下一阶段：** 保持实验协议和默认配置稳定，优先研究 label-free hard-unknown 信号覆盖及 rejection representation 对 semantic-near 类别的区分能力；对 validation-only fusion 进行 validation-to-test coverage transfer 和阈值稳定性审计，再决定是否扩大验证。

## 四、四（一）已开展的工作

项目已完成以下实验工作：

1. **完成实验链路和真实性审计。** 统一了数据划分、训练、mixed-pool nnPU、known-validation checkpoint 选择、95% known-coverage 阈值校准、未知样本发现和自动聚类流程。当前实验报告记录了 split、checkpoint、训练历史、校准报告、逐项配置和 discovery report。审计确认 mixed-pool/test 标签未用于训练、checkpoint、阈值或方法选择；当前回归测试为 317 passed。

2. **建立并复核 rejection representation 基线。** 完成 shared-pretrained 与 random initialization 的同架构对照。两个 fresh split 上 shared-pretrained 分支的 AUROC 分别提高 0.01893 和 0.02273，FPR95 分别改善 3.85 和 5.82 个百分点，说明其具有较稳定的排序和分数分离价值；但固定工作点 unknown rejection 在不同 split 上方向不完全一致，因此暂作为基线候选，不表述为已经解决新类发现。

3. **完成训练期 uncertainty-KD 消融。** 在 seeds 2071--2073 上，uncertainty-weighted logits KD 相比 standard KD 的平均 AUROC 仅提高 0.0009，而 unknown rejection 平均下降 3.32 个百分点；在 seed 2101 上，student 对 teacher uncertainty 的回归 loss 四个 epoch 均非零，但固定工作点 unknown rejection 下降 6.25 个百分点。这说明蒸馏路径真实执行，但数值一致性没有稳定转化为拒识收益。

4. **完成 rejection-feature KD 及机制拆分。** seed 2102 中，feature KD loss 和 rejection gradient 均真实非零，AUROC 从 0.800266 提升至 0.806794、OSCR 从 0.612721 提升至 0.624250，但 fixed-coverage unknown rejection 从 31.275% 降至 25.025%。为区分 feature KD 与 uncertainty weighting 的影响，seed 2104 固定系数 0.1，比较 uniform 和 teacher-uncertainty weighting；uncertainty weighting 使 unknown rejection 从 26.50% 降至 21.95%，AUROC 从 0.792513 降至 0.781977，FPR95 恶化 1.33 个百分点。该加权变体因此停止继续搜索。

5. **完成 validation-only uncertainty fusion 复核。** 在 7 个 semantic-hard fresh split 上，固定 0.5/0.5 融合平均带来 AUROC +0.01168、AUPR +0.02534、unknown rejection +4.39 个百分点，5/7 个 split 通过预注册联合标准。但 seed 2103 的 unknown rejection 下降 0.40 个百分点，说明仍存在 split-dependent operating-point trade-off，当前仅保留为 opt-in 候选，不改默认配置。

6. **完成候选池和自动聚类诊断。** 结果表明候选池总体纯度不能代表所有 novel 类的覆盖，semantic-near 类别仍存在明显尾部漏检；MC selector、conjunction、额外 rejection 损失和简单维度扩容均未形成稳定联合收益。因此当前研究重点已从继续堆叠 uncertainty 来源，转向 hard-unknown 信号覆盖和 known 类保护。

## 五、四（二）项目实施存在的困难及解决办法

**困难一：semantic-near novel 与 known 在共享表征、rejection 表征和最终分数上重叠。** 这导致 AUROC、AUPR 或聚类指标改善时，固定 known coverage 工作点的 unknown rejection 仍可能下降。项目已将 fixed-coverage unknown rejection、known acceptance、accepted-known accuracy、OSCR、score overlap、自动 K、NMI 和 ARI 作为联合指标，不再只依据单一排序指标判断方法有效性。

**困难二：训练期不确定性蒸馏虽能真实传递 loss 和梯度，但收益不稳定。** 已通过 standard KD 对照、teacher uncertainty target、rejection-feature KD 和 weighting mechanism split 分离问题来源；目前停止没有通过联合标准的 uncertainty-KD 目标、系数和权重搜索，保留 standard KD 为默认对照。

**困难三：拒识阈值从 validation 向 test 工作点迁移存在波动。** 已统一使用 known-validation-only 约 95% coverage 校准，禁止测试标签参与阈值或融合规则选择；下一阶段将保存 validation score 的逐样本结果，开展 bootstrap/OOB 稳定性和 coverage transfer 审计。

**困难四：候选池整体 purity 不能保证 novel 类别均衡覆盖。** 已增加 novel class recall、最低类别 recall、known contamination 和候选池审计，停止仅依据总体 purity 推进 selector；下一阶段优先研究不依赖未知标签的 hard-unknown 覆盖信号。

**困难五：实验结果和实现复杂度较高，早期对照存在协议混淆风险。** 已修复 mixed-pool hard-label 语义、uncertainty-KD 默认开关、独立 rejection encoder 构造和 teacher rejection feature dimension 等问题，并要求每个新实验保存协议、配置、checkpoint、训练历史和机器结果。

## 六、五、已取得的阶段性成果（100 字以内建议稿）

已完成系统与实验协议审计，建立标准KD和拒识表征基线。训练期不确定性蒸馏尚无稳定增益；验证集融合在7个新划分上平均提升AUROC 0.01168、未知拒识4.39个百分点，但仍存在波动。

## 七、建议作为附件保留的实验依据

模板正文不适合放入全部表格，建议将以下材料作为电子附件或汇报 PPT 数据依据：

- `analysis/midterm_summary_20261025.md`：完整阶段总结；
- `analysis/full_program_research_theme_audit_20261024.md`：研究主题和协议审计；
- `analysis/shared_rejection_fusion_fresh_2084_2103_summary_20261024.md`：7 个 fresh split 融合汇总；
- `analysis/rejection_feature_kd_s2102_20261024.md`：rejection-feature KD 结果；
- `analysis/rejection_feature_kd_weighting_s2104_20261025.md`：weighting 机制拆分结果；
- `analysis/teacher_uncertainty_transfer_s2101_20261024.md`：teacher uncertainty transfer 结果。

## 八、填写时的真实性边界

- 不要写“已解决 semantic-near overlap”或“uncertainty-KD 已证明有效”。
- 不要把 validation-only fusion 的平均收益写成所有数据集、所有 split 都有效。
- 不要把 seed=2104 的 uniform weighting 单个 split 优势写成最终方法收益。
- 不要把自动聚类 NMI/ARI 的改善直接等同于新类发现成功，必须同时报告固定工作点 unknown rejection 和 known utility。
- 不要填写无法从已有记录追溯的经费金额、成员投入小时、论文/专利或产品成果；这些内容应由团队按实际情况补充。
