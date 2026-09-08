# 不确定性感知知识蒸馏算法框架

Uncertainty-Aware Knowledge Distillation for Novel Class Discovery

## 项目概述

本仓库记录了“基于不确定性知识蒸馏的新类发现方法”的代码实现、实验脚本和阶段性结果。
项目目标是把知识蒸馏、不确定性建模、开放集识别和未知样本聚类组合到一套可复现实验流程里。

当前实现已经支持：

- CIFAR-100 和 ImageFolder 风格数据集的开放集划分
- 教师/学生训练
- 标准 KL 蒸馏和不确定性加权蒸馏
- MC Dropout 不确定性估计
- 原型距离、Entropy、Mahalanobis 等开放集打分
- 未知样本聚类与结果分析

## 当前状态

目前的系统是一个可复现的开放集识别与未知聚类流程，还不是完整的端到端新类发现系统。
我们已经完成了 CIFAR-100 60/40 划分上的一组严格消融，并得到以下单种子结果。

| Run | AUROC | FPR95 | Known acc | Unknown reject | Cluster ACC | NMI | ARI |
|---|---:|---:|---:|---:|---:|---:|---:|
| CE | 0.5939 | 0.8812 | 0.4527 | 0.0673 | 0.1820 | 0.5127 | 0.0535 |
| Standard KD | 0.5955 | 0.8632 | 0.4093 | 0.0628 | 0.2022 | 0.5190 | 0.0571 |
| Uncertainty KD | 0.5898 | 0.8613 | 0.4290 | 0.0508 | 0.2117 | 0.5387 | 0.0628 |
| Feature KD | 0.5846 | 0.8790 | 0.4113 | 0.0653 | 0.1865 | 0.5083 | 0.0483 |
| Full model | 0.5977 | 0.8713 | 0.4375 | 0.0747 | 0.1768 | 0.4759 | 0.0398 |

## 目前结论

- 普通 KD 比 CE 只带来很小的检测收益。
- 不确定性加权 KD 更偏向改善聚类结构，而不是直接提升 AUROC。
- 当前特征蒸馏版本没有带来稳定提升。
- 完整模型在这次单种子实验里 AUROC 和未知拒识率最好，但聚类指标最差。
- 这说明检测与聚类目标仍然存在明显冲突，后续还需要多 seed 和真正的新类发现训练。

## 项目成员

- 项目负责人：王笑颜
- 开发成员：李柯颖
- 开发成员：齐誉涵

## 主要文件

- `train.py`：训练和发现入口
- `novel_discovery/`：模型、损失、数据、指标和流程代码
- `scripts/`：可直接运行的实验脚本
- `analysis/revised_ablation/`：正式消融的结果摘要
- `docs/revised_method.md`：当前方法说明
- `docs/revised_experiment_plan.md`：当前实验方案

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

查看数据划分：

```bash
python train.py inspect_data --dataset cifar100 --data-root ./data --download
```

运行正式消融：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_revised_ablation.ps1
```

运行轻量学生实验：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_mobilenet_compression.ps1
```

## 说明

- 仓库默认不包含 `data/`、`runs/` 和缓存权重等大文件。
- 结果摘要保存在 `analysis/` 下的 Markdown 和 JSON 文件中。
- 当前结果是单种子结果，后续还需要补多 seed 才能形成最终结论。
