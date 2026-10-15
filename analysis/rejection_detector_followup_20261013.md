# 2026-10-13：独立 rejection detector 的 OE、分数融合与 MLP 复核

## 实验目的

核心问题仍是已知/未知特征与分数分布重叠。上一轮 seed=44 的独立 rejection branch 结果较有希望，但不能据单个 seed 认定有效。本轮先在完全固定的 seed=123 协议上复核三类后处理/辅助方法：

1. 外部 CIFAR-10 uniform-logit OE 是否进一步改善 rejection branch；
2. 将 rejection 分数与 kNN/原型分数融合是否改善整体检测；
3. 将线性 nnPU rejector 换成 MLP 是否能建模非线性边界。

训练、数据划分、学生 checkpoint、混合发现池、known prior、MC=4、open validation 和 95% known-coverage 校准均保持固定。测试标签只用于事后报告与配对 bootstrap，不用于训练、阈值或方法选择。

## 1. rejection feature-margin + uniform OE

| 配置 | AUROC | AUPR | FPR95 | OSCR | known acc | known accept | unknown reject | KS | unknown below known q95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| margin only | 0.87986 | 0.81461 | 0.43817 | 0.58793 | 0.63833 | 95.67% | 41.58% | 0.60817 | 54.85% |
| margin + uniform OE | 0.88071 | 0.82253 | 0.43467 | 0.57784 | 0.62667 | 94.98% | 44.63% | 0.59975 | 55.55% |

表面上 AUROC/AUPR/FPR95 有小幅改善，但 OSCR、已知准确率、KS 和 overlap 变差。严格固定 95% 已知覆盖率的配对 bootstrap 给出 AUROC 差异 `+0.00087`，95% CI `[-0.00383, +0.00550]`；未知拒绝率差异 `-0.00316`，95% CI `[-0.03375, +0.02800]`。因此不能称为稳定收益，继续作为关闭状态的 ablation。

## 2. rejection score 与 kNN/原型 score 融合

固定 margin-only student，只改变后处理融合配置：

| 配置 | AUROC | FPR95 | OSCR | known acc | unknown reject | KS | overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| rejector only | 0.87986 | 0.43817 | 0.58793 | 0.63833 | 41.58% | 0.60817 | 54.85% |
| kNN + rejector, w=0.25 | 0.84240 | 0.51183 | 0.58362 | 0.63833 | 29.38% | 0.53792 | 67.98% |
| kNN + rejector, w=0.50 | 0.86280 | 0.45633 | 0.59159 | 0.63833 | 34.43% | 0.56817 | 62.13% |
| kNN + rejector, w=0.75 | 0.88064 | 0.41733 | 0.59619 | 0.63833 | 40.98% | 0.60383 | 57.05% |
| prototype + rejector, w=0.50 | 0.86074 | 0.46633 | 0.59055 | 0.63833 | 35.70% | 0.56775 | 62.20% |

`w=0.75` 的普通报告看起来最强，但严格精确 95% known coverage 的 bootstrap 中，AUROC 差异为 `+0.00078`，95% CI `[-0.00276, +0.00433]`；未知拒绝率差异为 `-0.02398`，95% CI `[-0.04926, +0.00050]`。因此这是阈值/校准协议敏感的工作点变化，不足以证明分布重叠改善，不进入默认流程。

## 3. 线性 nnPU 与 MLP nnPU

固定同一 student、同一发现池和校准协议，只改变 rejector 函数族：

| rejector | AUROC | AUPR | FPR95 | OSCR | known accept | unknown reject | KS | overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | 0.87986 | 0.81461 | 0.43817 | 0.58793 | 95.67% | 41.58% | 0.60817 | 54.85% |
| MLP | 0.87249 | 0.81398 | 0.50533 | 0.58021 | 96.03% | 39.80% | 0.60858 | 55.53% |

MLP 没有改善排序或固定工作点，反而使 FPR95、OSCR 和未知拒绝率变差；当前 linear nnPU rejector 更可靠。

## 阶段结论

- uniform OE 已确认正确接入，但在 seed=123 没有稳定收益，不能与现有方法叠加进入主线。
- 后处理融合可以改善某些报告指标，但严格 matched-coverage 复核不支持收益，不能把阈值变化写成特征分离改善。
- MLP rejector 在当前样本量和特征上不如线性 rejector，停止继续增加 rejector 容量。
- 当前更值得保留的是独立 rejection representation + linear `nu_corrected` nnPU，以及训练端 `encoder_lr_scale=0.1`；但即使该分支的未知拒绝率约 40%，score overlap 仍约 55%，核心问题尚未解决。

## 下一步

停止继续堆叠 post-hoc detector，回到共享 encoder 的训练目标。下一轮应只引入一个直接作用于共享特征几何的目标，并与当前 linear rejection baseline 严格配对；优先检查该目标的有效梯度、已知/未知代理分布和 q95 overlap，再决定是否恢复聚类评估。所有新方法继续遵守：固定 open-val 选模、测试集只做最终评估、同时报告排序指标、固定覆盖率工作点和 overlap。

## 补充：按真实类别的重叠诊断

为了避免把 40 个未知类误当作同一种困难，新增了 `scripts/analyze_class_overlap.py`。注意开放集报告中的 `true_label` 会把未知样本统一编码为 `-1`，脚本已改用保留原始 CIFAR 类别 id 的 `raw_labels`；这是诊断字段语义的修正，不改变训练或测试指标。

在 seed=123 的 margin-only checkpoint 上，未知类的报告阈值拒绝率从 `12%`（class 40 lamp）到 `71%`（class 97 wolf）差异很大；class 13 bus、33 forest、22 clock、93 turtle 等也只有约 `18%--21%`。同时已知类也存在高分长尾，例如 class 80 squirrel 的拒绝率为 `32%`。这说明整体 overlap 主要由部分困难未知类与部分困难已知类共同造成，不能只用一个全局阈值或全局 uniform OE 解释。

该结果支持下一步优先研究“类条件 support / 局部边界”的训练或评分目标，并要求只使用训练集已知类统计构造类条件目标；不能使用测试集真实未知类别来定制方法。诊断产物见 [`analysis/class_overlap_rejection_s123_20261013.json`](class_overlap_rejection_s123_20261013.json)。
