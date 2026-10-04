# 表征结构与 BatchNorm 稳定化尝试（2026-09-30）

## 目标

围绕“已知/未知特征与分数分布重叠、未知拒绝率低”测试两个不改变损失和检测器的表征稳定化方向：CIFAR-style ResNet stem，以及冻结 BatchNorm running statistics。

## CIFAR-style stem

新增可选 `--cifar-stem`：ResNet 首层改为 `3×3、stride=1`，移除初始 max-pooling；默认关闭。预训练时将原首层权重插值到 `3×3` 初始化。

条件：CIFAR-100 60/40、seed=42、预训练 ResNet-34/ResNet-18、3 epochs、1200/300/1000、相同 CE + uncertainty KD + feature KD + prototype + SupCon、`normalized_entropy_mahalanobis`、MC=4、known-only 95% coverage。

| 方法 | AUROC | FPR95 | OSCR | known accuracy | 报告 unknown reject | 测试集精确95%覆盖率 unknown reject |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 ImageNet stem | 0.5153 | 0.9223 | 0.1177 | 0.1818 | 0.0430 | 0.0430 |
| CIFAR stem | 0.4879 | 0.9107 | 0.0928 | 0.1471 | 0.0506 | 0.0329 |

CIFAR stem 的 FPR95 表面下降，但 AUROC、OSCR、known accuracy 和 matched-coverage unknown rejection 均变差；同时教师/学生验证准确率较低。因此不能作为主方法。

## BatchNorm running statistics

新增可选 `--freeze-bn-stats`：训练时把 BatchNorm 置于 eval 状态，固定 running mean/variance，但保留 affine weight/bias 的梯度更新；默认关闭。

条件与 baseline 相同，且教师和学生都重新训练。

| 方法 | AUROC | FPR95 | OSCR | known accuracy | 报告 unknown reject | 测试集精确95%覆盖率 unknown reject |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 baseline | 0.5153 | 0.9223 | 0.1177 | 0.1818 | 0.0430 | 0.0430 |
| freeze BN stats | 0.4836 | 0.9603 | 0.0031 | 0.0083 | 0.0228 | 0.0405 |

冻结 BN 使教师 best validation accuracy 约为 `0.0267`，学生 best validation accuracy 约为 `0.0267`，明显破坏已知分类。因此停止该方向，不调节其权重或默认开启。

## 代码验证

- `--cifar-stem` 和 `--freeze-bn-stats` 均已接入教师、学生和 discover 的模型构建/训练调用链；
- toy 前向与 toy 教师/学生 smoke test 通过；
- 项目测试：`97 passed`；
- 两个选项默认关闭，历史配置不受影响。

## 当前判断

这两次尝试都没有解决核心问题，说明当前瓶颈不是简单的 ResNet 首层下采样或 BatchNorm running statistics。后续应减少网络结构层面的试探，转向能明确利用可靠未知/辅助异常证据的表征目标，并继续用 matched known coverage、AUROC、FPR95、OSCR、known accuracy 和分数分布联合判断。
