# 类别协议审计与 mixed-pool 联合目标复核（2026-09-29）

## 1. 目的

本轮先检查未知检测失败是否主要由 CIFAR-100 的类别语义重叠造成，再检查
当前 mixed-pool 联合新类发现目标是否把已知样本错误地纳入 100 类平衡分配。

所有检测均使用 CIFAR-100 60/40、seed=42、分层 known train/val 切分、
预训练 ResNet-34 teacher/ResNet-18 student、3 epochs、1200/300/1000 样本、
`normalized_entropy_mahalanobis`、95% known coverage 和跳过聚类。

## 2. 类别协议审计

| 协议 | AUROC | FPR95 | OSCR | Known acc | Unknown reject |
|---|---:|---:|---:|---:|---:|
| random | 0.5577 | 0.8826 | 0.1476 | 0.2083 | 7.09% |
| semantic_hard | 0.4925 | 0.9436 | 0.0826 | 0.1282 | 6.51% |
| semantic_isolated | 0.5973 | 0.9035 | 0.1618 | 0.2130 | 11.78% |

语义相近的 known/novel 划分明显恶化特征和分数分离，语义隔离划分有所改善，
但未知拒绝率仍然很低。因此类别语义重叠是重要因素，但不能解释全部问题；
模型训练目标没有学到稳定的 known/novel 分配仍是主要瓶颈。

## 3. 已知残差加权尝试

新增 `--joint-mixed-residual`。其原理是使用
`1 - max softmax(known logits)` 作为 mixed pool 样本权重，并让新类原型学习
只在 novel 空间中进行，避免 known 与 novel 一起进入同一个 Sinkhorn 平衡分配。

单元测试为 `87 passed`，最小 smoke 能正常执行，日志能够记录
`joint_residual_weight`。

配对训练条件完全相同，仅改变该开关：

| 方法 | AUROC | FPR95 | OSCR | Known acc | Unknown reject |
|---|---:|---:|---:|---:|---:|
| 原统一 100 类联合目标 | 0.5976 | 0.8595 | 0.2049 | 0.2628 | 8.35% |
| 已知残差加权 | 0.5793 | 0.8612 | 0.1951 | 0.2579 | 8.10% |

结论：本次改动未有效解决未知检测，不能作为当前主方法。原因是该权重衡量
的是已知类别内部不确定性，不是 known 分支与 novel 分支之间的相对证据；
当前分类器未校准时，大量样本都会得到偏高残差。

## 4. 下一步

下一轮改用 known/novel 联合概率中的 `novel mass` 作为候选权重，并加入双视图
一致性约束。权重应来自 `softmax([known logits, novel logits])` 中 novel 类的
总概率，而不是只使用 known logits 内部的最大概率。仍然先做 smoke 和单 seed
配对实验，只有 AUROC、FPR95、OSCR 和 unknown rejection 同时不恶化，才考虑
扩大训练规模和增加随机种子。

