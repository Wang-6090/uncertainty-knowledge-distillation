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
- AUPR、FPR95、OSCR 以及已知/未知分离后的聚类指标
- 可选的无标注 discovery pool 双视图一致性训练和 NT-Xent 对比训练

## 当前状态

目前的系统是一个可复现的开放集识别与未知聚类流程，还不是完整的端到端新类发现系统。
默认训练协议保持不变；discovery pool 通过显式参数开启，便于和原有 CE/KD 基线公平对照。
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

## 本次改进

为回应当前文档中提到的主要问题，代码已经做了以下增强：

- 修正 `supervised_contrastive_loss`：只对 batch 内存在同类正样本的 anchor 计算监督对比损失，避免单样本类别稀释 SupCon。
- 增加 `--uncertainty-weight-mode`：支持 `raw` 和 `mean_normalized`，便于公平比较标准 KD 与不确定性加权 KD。
- 增强 discovery pool 训练：支持 `--discovery-loss consistency|nt_xent`、`--alpha-discovery-unknown` 和 `--discovery-pool-mode unknown|mixed`。
- 增强聚类评估：`discover` 现在可选择 KMeans、Agglomerative、Spectral，可选择 projection/feature/PCA 特征，并记录 auto-K 诊断。
- 增加候选池污染诊断：`discovery_report.json` 会记录 candidate count、true unknown count、false reject count、candidate purity 和 K 估计误差。
- 改进分析脚本：`analyze_results.py` 的 Markdown 汇总会展示 candidate purity、estimated K 和更清晰的缺失文件错误。

## 推荐实验协议

短期不要继续叠加复杂模块，建议先固定协议验证当前最有价值的方向：

1. 固定 CIFAR-100 60/40 split、backbone、epoch、batch size、score mode 和 cluster config。
2. 至少跑 3 个 seed，并报告 mean ± std。
3. 主对照保持 `CE`、`standard KD`、`uncertainty KD`、`full representation`、`full + discovery pool`。
4. 主检测分数优先比较 `entropy_proto` 与 `normalized_entropy_mahalanobis`。
5. 聚类同时报告 oracle-K 和 auto-K；oracle-K 只作为上限分析，不作为最终无监督结论。
6. 每次报告 candidate purity，先判断候选池是否被误拒已知样本污染，再解释聚类指标。

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
- `docs/group_progress.md`：组内进度说明

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

启用无标注 discovery pool 一致性训练：

```powershell
python train.py train_student `
  --dataset toy `
  --num-known 10 `
  --teacher-ckpt .\runs\teacher\teacher.pt `
  --student-ckpt .\runs\student_discovery\student.pt `
  --work-dir .\runs\student_discovery `
  --discovery-pool `
  --limit-discovery 256 `
  --alpha-discovery 0.1 `
  --device cpu
```

该开关只使用未知类别训练图像的两个随机增强视图，不把标签传给损失函数。`--limit-discovery` 用于控制实验规模；`--alpha-discovery 0` 或不传
`--discovery-pool` 时，行为与旧版学生训练一致。

启用 discovery pool NT-Xent 和候选池诊断：

```powershell
python train.py train_student `
  --dataset cifar100 `
  --data-root .\data `
  --num-known 60 `
  --split-path .\splits_cifar100_60_40.json `
  --teacher-ckpt .\runs\teacher\teacher.pt `
  --student-ckpt .\runs\student_discovery\student.pt `
  --work-dir .\runs\student_discovery `
  --discovery-pool `
  --discovery-pool-mode unknown `
  --alpha-discovery 0.1 `
  --discovery-loss nt_xent `
  --uncertainty-weight-mode mean_normalized `
  --device auto

python train.py discover `
  --dataset cifar100 `
  --data-root .\data `
  --num-known 60 `
  --num-novel 40 `
  --split-path .\splits_cifar100_60_40.json `
  --student-ckpt .\runs\student_discovery\student.pt `
  --work-dir .\runs\student_discovery_detect `
  --score-mode normalized_entropy_mahalanobis `
  --cluster-k auto `
  --cluster-method kmeans `
  --cluster-feature projection_pca `
  --cluster-selection composite `
  --cluster-normalize `
  --device auto
```

运行轻量测试：

```bash
python -m unittest discover -s tests
```

## 说明

- 仓库默认不包含 `data/`、`runs/` 和缓存权重等大文件。
- 结果摘要保存在 `analysis/` 下的 Markdown 和 JSON 文件中。
- `discovery_report.json` 现在额外记录 AUPR、OSCR、估计 K 误差、candidate purity，以及包含误拒已知样本和仅真实未知样本的两组聚类指标。
- 当前结果是单种子结果，后续还需要补多 seed 才能形成最终结论。
