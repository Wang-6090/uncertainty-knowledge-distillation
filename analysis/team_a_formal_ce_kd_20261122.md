# 王笑颜负责方向正式 CE/KD 三臂复核（2026-11-22）

本阶段只回答四个问题：

1. standard-KD 是否改善学生分类表征；
2. uncertainty-KD 是否比 standard-KD 有稳定额外收益；
3. 教师不确定性权重是否进入 KD、产生梯度并更新学生；
4. 表征变化是否传递到未知检测和新类发现。

拒识器结构、风险加权、候选筛选、评分融合和聚类算法没有改动。

## 协议与运行完整性

旧的 `42/123/3407` A/B/C 产物配置和评测口径不完全兼容，因此没有混合旧结果。本阶段使用新的 `team_a_formal_ce_kd_20261122` 输出目录，semantic-hard CIFAR-100 60/40 split，seed `42/123/3407`，每个 seed 使用同一个 split-matched teacher 和相同训练预算。

三臂定义为：

- `ce_only`：所有数值 `alpha_*` 为 0，`alpha_kd=0`；
- `standard`：仅启用普通 logits-KD，`kd_mode=standard`；
- `uncertainty`：仅把教师 uncertainty 权重应用于 logits-KD，`kd_mode=uncertainty`。

三臂均关闭 feature KD、rejection-branch KD、discovery、outlier 和 PU 训练目标，checkpoint 按 known-validation accuracy 选择，发现阶段使用相同的 max-softmax、known-validation 95% coverage、matched pool 和 Auto-K/KMeans 流程。测试标签没有用于训练、checkpoint 选择或阈值校准，只用于最终检测和聚类指标。

正式 runner 汇总时发现一个门禁 bug：CE-only 控制臂必然将 `alpha_kd` 从 1 改为 0，但旧检查把这个声明的处理差异当作意外配置差异。已在 `scripts/run_uncertainty_kd_core_pair.py` 中加入仅对 `ce_only` 控制臂的显式白名单；没有放宽其他配置差异。修复后没有重训 seed 42，runner 复用完整产物并完成 seed 123/3407，三种子汇总状态为 `completed`。

## 学生表征

对 9 个正式 checkpoint 做了冻结的 test-set representation audit。test labels 只用于事后描述性 probe 和方差分解，没有进入训练、选择或校准。known-class probe 是在 60 个已知类上训练的描述性线性分类器；`between/total` 是共享分类特征的类间方差占总方差比例。

| 训练方案 | between/total 均值 ± 样本 SD | 已知类 probe 均值 ± 样本 SD | 相对 CE probe 的配对均值变化 |
|---|---:|---:|---:|
| CE-only | `0.456080 ± 0.008818` | `0.704630 ± 0.011130` | — |
| standard-KD | `0.468368 ± 0.015858` | `0.707963 ± 0.004243` | `+0.003333` |
| uncertainty-KD | `0.465367 ± 0.018059` | `0.701296 ± 0.005702` | `-0.003333` |

standard-KD 相对 CE-only 的 probe 变化为 seed `42/123/3407` 上的 `+0.015556/-0.013889/+0.008333`，只有 2/3 个 seed 上升；between/total 变化为 `+0.030919/+0.017671/-0.011724`。因此 standard-KD 有弱的表征信号，但没有跨 seed 稳定改善。

uncertainty-KD 相对 standard-KD 的 probe 变化为 `-0.008889/-0.008333/-0.002778`，三组均下降；between/total 变化为 `+0.000201/-0.005963/-0.003243`，三组均没有改善。当前没有证据支持 uncertainty-KD 带来额外稳定表征收益。

机器审计结果保存在被忽略的本地文件 `analysis/team_a_formal_ce_kd_20261122_representation.json`；病例定义为 `analysis/team_a_formal_ce_kd_20261122_representation_cases.json`。

## 未知检测

下表是测试集原始指标，均值和 SD 使用三个 seed 的样本标准差；检测指标和聚类指标分开报告。

| 臂 | AUROC | AUPR | FPR95 | OSCR | known acceptance | unknown rejection |
|---|---:|---:|---:|---:|---:|---:|
| CE-only | `0.693465 ± 0.004977` | `0.551462 ± 0.007979` | `0.734944 ± 0.010304` | `0.568637 ± 0.008168` | `0.948778 ± 0.010636` | `0.116750 ± 0.025161` |
| standard-KD | `0.685469 ± 0.001823` | `0.542830 ± 0.003923` | `0.745722 ± 0.004131` | `0.541006 ± 0.006436` | `0.949944 ± 0.007001` | `0.105167 ± 0.018759` |
| uncertainty-KD | `0.690256 ± 0.004921` | `0.548927 ± 0.008745` | `0.740111 ± 0.014166` | `0.546733 ± 0.013155` | `0.955389 ± 0.005779` | `0.102750 ± 0.020930` |

standard-KD 相对 CE-only 的均值变化为 AUROC `-0.007996`、AUPR `-0.008632`、FPR95 `+0.010778`、OSCR `-0.027630`、unknown rejection `-1.158` 个百分点。它没有改善未知检测。

uncertainty-KD 相对 standard-KD 的均值变化为 AUROC `+0.004787`、AUPR `+0.006098`、FPR95 `-0.005611`、OSCR `+0.005727`，但 unknown rejection `-0.242` 个百分点。三组 seed 的 AUROC 变化为 `+0.003334/+0.008364/+0.002663`，AUPR 变化为 `+0.004140/+0.011604/+0.002549`，而 unknown rejection 变化为 `-0.975/+0.225/+0.025` 个百分点。排序指标有一致的轻微上升，固定 coverage 下的拒识没有同步改善，因此不能称为稳定未知检测收益。

## 新类发现

| 臂 | Cluster ACC | Cluster NMI | Cluster ARI |
|---|---:|---:|---:|
| CE-only | `0.169319 ± 0.017900` | `0.437508 ± 0.025089` | `0.047261 ± 0.007302` |
| standard-KD | `0.176173 ± 0.008458` | `0.440403 ± 0.017002` | `0.044854 ± 0.004340` |
| uncertainty-KD | `0.166327 ± 0.015793` | `0.433850 ± 0.037173` | `0.047904 ± 0.006001` |

standard-KD 相对 CE-only 的 ACC/NMI/ARI 均值变化为 `+0.006854/+0.002895/-0.002407`；uncertainty-KD 相对 standard-KD 的变化为 `-0.009846/-0.006553/+0.003050`。因此 standard-KD 没有形成一致聚类改善，uncertainty-KD 的轻微 ARI 上升也伴随 ACC/NMI 下降，不能认为收益已传递到端到端新类发现。

## 蒸馏机制审计

代码路径和运行记录共同支持以下结论：

- `teacher_kd_uncertainty` 使用 `@torch.no_grad()`，训练 batch 中又在 `torch.no_grad()` 上下文调用教师；optimizer 的参数组只来自 student 和可选的 student novel head，没有教师参数。因此教师没有参与更新。
- `kd_mode=standard` 和 `kd_mode=uncertainty` 的 `alpha_kd` 都为 `1.0`，`alpha_feat_kd=0`。总损失显式包含 `alpha_kd * loss_kd`；CE-only 的 `alpha_kd=0` 是唯一允许的控制差异。
- standard 臂 12 个 epoch 记录中 `kd_uncertainty_weight_applied` 均为 false；uncertainty 臂 12 个 epoch 均为 true。uncertainty 臂的 batch 权重均值约 `1.000000`，标准差跨 epoch/seed 均值 `0.200781`，最小 ESS fraction `0.953333`，说明权重不是常数且进行了 mean normalization。
- uncertainty 臂所有 epoch 的 KD loss、KD encoder gradient norm 和 student classifier update norm 都大于 0。跨三种子和 epoch 的均值分别为 `1.449459`、`13.512339` 和 `0.029831`；最小值分别为 `0.981682`、`10.606258` 和 `0.025887`。这证明权重进入实际 KD 项，并产生了进入学生优化过程的梯度和参数更新。
- 教师权重、学生 uncertainty head、分类熵和最终 discovery reject score 在实现中是不同变量和损失项；本实验没有使用学生 uncertainty head 或最终拒识分数替代教师 KD 权重。

## 阶段结论

本次正式三种子复核没有支持把 standard-KD 或 uncertainty-KD 写成稳定的端到端改进：standard-KD 的表征 probe 只有弱且 seed 敏感的信号，检测端整体下降；uncertainty-KD 确实被正确执行并产生梯度，但相对 standard-KD 只有轻微排序改善，固定 coverage 的 unknown rejection 和聚类 ACC/NMI 没有改善。

因此当前决策是：保留 standard-KD 作为可比较基线，保留 uncertainty-KD 作为机制已验证但性能未确认的显式消融；不修改默认配置，不继续在同一协议上扫描 weighting function、阈值或其他拒识/聚类组件。后续若继续研究，只能提出新的、预先固定的学生表征机制，并重新使用相同的四项判据验证。
