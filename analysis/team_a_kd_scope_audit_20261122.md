# 王笑颜负责方向阶段审计（2026-11-22）

本阶段只审计分类表征与不确定性知识蒸馏，回答四个问题：

1. standard-KD 是否改善学生分类表征；
2. uncertainty-KD 是否比 standard-KD 有稳定额外收益；
3. 不确定性权重是否进入 KD、产生梯度并更新学生；
4. 表征变化是否传递到未知检测和新类发现。

拒识器结构、风险加权、候选筛选、评分融合和聚类算法不属于本阶段的改动范围。

## 一、协议兼容性审计

仓库中存在 `42/123/3407` 的 A、B、C checkpoint，但不能直接把它们合并成当前口径的正式三种子主结果。

- `42/123` 的 A、B、C 训练配置在已保存字段中主要只改变 `alpha_kd/kd_mode`，可以作为历史表征对照；但对应的历史检测产物使用旧的评测/发现口径，不能与当前严格 CE-only 结果混合。
- `3407` 的 A checkpoint 是近期保存的完整配置，而 B/C 是较早的兼容配置；A 显式记录了 known-validation checkpoint、matched pool 和 `discovery_pool_mode=unknown` 等字段，B/C 缺少同样的字段，且历史报告声明了 oracle-filtered novel pool 语义。因此 3407 的 A/B/C 不能作为同一正式协议的主结果。
- 旧结果全部保留，没有覆盖、重写或把缺失字段补写成已验证事实。

因此，本阶段将历史结果分为“机制/趋势证据”和“严格主结果”，并准备在新的输出目录中用当前 `ce_only` profile 重新完成正式 A/B/C。

## 二、冻结 checkpoint 的学生表征审计

为直接回答表征问题，对已有 A/B/C checkpoint 做了冻结的 known-validation-only 审计。每个 seed 使用 3000 个已知验证样本和 60 个已知类别；未知标签没有进入该分析，也没有参与 checkpoint 或方法选择。指标包括共享分类特征的类内方差、类间方差、between/total 方差比和已知类线性 probe 准确率。

| 训练方案 | between/total | 已知类 probe | 解释 |
|---|---:|---:|---|
| CE-only | `0.468696 ± 0.009671` | `0.621481 ± 0.006120` | 基线 |
| standard-KD | `0.473258 ± 0.005197` | `0.626667 ± 0.009876` | 表征指标均值略升，但方向不是每个 seed 都一致 |
| uncertainty-KD | `0.458129 ± 0.018792` | `0.634444 ± 0.011706` | probe 三个 seed 都高于 standard-KD，但 between/total 两个 seed 下降 |

配对变化为：

- standard-KD 相对 CE-only 的 between/total 平均 `+0.004563`，probe 平均 `+0.005185`；三 seed 的 probe 变化为 `-0.012222/+0.014444/+0.013333`，所以不能称为稳定改善。
- uncertainty-KD 相对 standard-KD 的 probe 变化为 `+0.007778/+0.003333/+0.012222`，方向一致；但 between/total 变化为 `-0.036188/+0.005846/-0.015045`，表征几何并未一致改善。

当前更准确的判断是：standard-KD 对已知类表征有弱的、seed 敏感的正向信号；uncertainty-KD 可能提高线性可读性，但没有证明它让表征的类间分离稳定变好。该审计结果保存在 `analysis/team_a_representation/formal_shared_features_20261122.json`。

## 三、严格 CE-only 三臂结果

现有最干净的三臂实验是 semantic-hard 60/40、seed 2079。三臂关闭 discovery、outlier、PU、rejection 和其他数值 `alpha_*`，共享 split-matched teacher、训练预算、known-validation checkpoint 选择和评测流程。

| 指标 | CE-only | standard-KD | uncertainty-KD |
|---|---:|---:|---:|
| AUROC | 0.703847 | 0.708586 | 0.709437 |
| AUPR | 0.569262 | 0.571229 | 0.574108 |
| FPR95 | 0.723333 | 0.723333 | 0.723333 |
| OSCR | 0.556445 | 0.552395 | 0.556223 |
| known acceptance | 0.947500 | 0.959833 | 0.954167 |
| unknown rejection | 0.143000 | 0.100500 | 0.119500 |
| all-known accuracy | 0.689000 | 0.679333 | 0.686667 |
| Auto-K absolute error | 2 | 5 | 6 |
| Auto-NMI | 0.436954 | 0.485926 | 0.464595 |

standard-KD 相对 CE-only 只在 AUROC、AUPR 和 Auto-NMI 上有局部改善，同时 unknown rejection 下降 `4.25` 个百分点、all-known accuracy 下降约 `0.97` 个百分点。uncertainty-KD 相对 standard-KD 的 AUROC 变化只有 `+0.000851`，unknown rejection `+1.90` 个百分点，但 Auto-NMI 下降 `0.021331`；相对 CE-only 的 unknown rejection 仍低 `2.35` 个百分点。这是单 seed 探索性证据，不能支持把 uncertainty-KD 设为默认。

2074–2076 的三臂结果可以用于补充趋势，但其中 `no_kd` 保留了其他辅助目标，不是纯 CE-only。该协议下 standard-KD 相对 no-KD 的 AUROC 平均变化接近零，uncertainty-KD 相对 standard-KD 的 AUROC 平均 `-0.004897`、AUPR `-0.008241`、FPR95 `+1.611` 个百分点、unknown rejection `-2.208` 个百分点；聚类 ACC/NMI/ARI 均值虽有上升，但不稳定，不能抵消检测端退化。

## 四、蒸馏机制审计

代码调用链已核对：教师 logits 和教师不确定性在 `torch.no_grad()` 中生成；`uncertainty_weights` 对教师不确定性执行 `detach`，按 `exp(-u)` 生成样本权重并可做 batch mean normalization；权重乘到逐样本 logits KL 后，KD 项与 CE 及其他启用目标共同进入总损失。学生不确定性头、分类熵和最终拒识分数没有被当作该 KD 权重。

机制证据分为两层：

- 2071–2079 的历史运行记录显示 uncertainty 臂每轮使用非均匀权重、KD loss 非零、feature KD 关闭，说明路径被执行；但这些旧 JSON 没有逐 epoch 的 KD 梯度范数和学生参数更新范数，因此不能把它们写成“运行级梯度已完整记录”。
- 当前单元测试覆盖 detached teacher、非均匀权重、KD loss、学生梯度和更新路径；针对性测试结果为 `78 passed, 260 deselected`。quality-aware fresh confirmation 的运行级审计还确认 KD 梯度和学生分类头更新均非零、教师参数保持冻结，但这是质量感知变体的直接证据，不能反向替代旧 uncertainty-KD 运行日志。

因此，机制层面的结论是“代码路径和单测已确认，历史 uncertainty-KD 运行级梯度记录不完整”；任务层面的结论仍由独立的 A/B/C 结果决定。

## 五、阶段结论和下一步

当前没有足够证据证明：

- standard-KD 在跨 seed 上稳定改善学生表征；
- uncertainty-KD 比 standard-KD 稳定提高未知检测；
- uncertainty-KD 的局部排序变化能够稳定传递到 Auto-K、NMI/ARI 和完整新类发现。

下一步只做一件必要的工作：在新的、互不覆盖的输出目录中，用当前 `ce_only` runner 对 `42/123/3407` 重新完成严格 A/B/C。每个 seed 共享 teacher、split、学生初始化、训练预算和 known-only 选择规则；运行日志必须补齐教师不确定性来源、权重均值/标准差/ESS、KD loss、KD 梯度范数、学生分类头更新和教师冻结状态。若三种子仍不能同时支持 B 相对 A、C 相对 B，则停止继续扫描 uncertainty weighting，保留 standard-KD 为对照，uncertainty-KD 为显式消融。

验证命令和正式运行命令均使用新 tag，避免复用或覆盖旧的 A/B/C 产物；未知标签只进入最终测试评价。
