# 不确定性感知知识蒸馏算法框架

Uncertainty-Aware Knowledge Distillation for Novel Class Discovery

## 项目概述

本仓库把知识蒸馏、不确定性建模、开放集识别和未知样本聚类组合到一套可复现实验流程里。
当前默认数据是本地 **ImageFolder 版 CIFAR-100**：

```text
CIFAR-100-dataset-main/
├── train/   # 100 个类别目录
└── test/    # 100 个类别目录
```

默认命令使用：

- `--dataset imagefolder`
- `--data-root ./CIFAR-100-dataset-main`
- 60 个已知类 / 40 个未知类
- 类别划分保存在 `splits_cifar100_60_40_imagefolder.json`

当前系统仍是“先未知检测、再聚类”的两阶段流程，不是严格端到端新类发现。

## 当前训练状态

这是截至当前工作区的真实状态，不是预期结果。

| 项目 | 状态 |
|---|---|
| 本地 CIFAR-100 ImageFolder 数据 | 已确认可用：train 27000，val 3000，test 10000，60/40 划分 |
| 数据加载 | 已支持 `train/`、`test/` 双目录 ImageFolder，不会把这两个文件夹当成类别 |
| 冒烟训练 | 已完成 1 epoch、128 张训练样本，checkpoint 在 `runs/cifar100_imagefolder_smoke/teacher.pt`，known acc = 0.0781 |
| 正式 15 epoch 教师训练 | **未完成**。进程仍在 CPU 上运行，输出目录是 `runs/cifar100_imagefolder_teacher/`，目前只有 `train.log`，**还没有** `teacher.pt` |
| 正式学生训练 / discover | 尚未开始，因为教师 checkpoint 还没生成 |
| 3 个随机种子正式实验 | 脚本已准备，结果尚未产出 |

当前 CPU 上每个 batch 大约 5 秒，15 epoch × 422 batch 预计需要很多小时。训练完成后应出现：

```text
runs/cifar100_imagefolder_teacher/teacher.pt
runs/cifar100_imagefolder_teacher/config.json
```

日志里应出现类似：

```text
saved to runs\cifar100_imagefolder_teacher\teacher.pt (best_epoch=..., best_val_known_acc=...)
```

## 历史单种子结果

下面这组数字来自更早的 CIFAR-100 torchvision 协议单种子消融，**不是**本次 `CIFAR-100-dataset-main` 正式训练结果，不能当作当前实验结论。

| Run | AUROC | FPR95 | Known acc | Unknown reject | Cluster ACC | NMI | ARI |
|---|---:|---:|---:|---:|---:|---:|---:|
| CE | 0.5939 | 0.8812 | 0.4527 | 0.0673 | 0.1820 | 0.5127 | 0.0535 |
| Standard KD | 0.5955 | 0.8632 | 0.4093 | 0.0628 | 0.2022 | 0.5190 | 0.0571 |
| Uncertainty KD | 0.5898 | 0.8613 | 0.4290 | 0.0508 | 0.2117 | 0.5387 | 0.0628 |
| Feature KD | 0.5846 | 0.8790 | 0.4113 | 0.0653 | 0.1865 | 0.5083 | 0.0483 |
| Full model | 0.5977 | 0.8713 | 0.4375 | 0.0747 | 0.1768 | 0.4759 | 0.0398 |

当时的观察：普通 KD 对检测只有很小收益；不确定性加权 KD 更偏向聚类结构；检测和聚类目标仍有冲突。最终结论需要至少 3 个 seed 的 mean ± std。

## 代码已实现的评估协议

- 未知检测可比较 MSP、Energy、熵、原型距离和 Mahalanobis，并用已知验证集 z-score 归一化后融合。
- `discover --temperature-calibration` 会在已知验证集上拟合温度，并保存 ECE、可靠性分箱、不确定性与分类错误相关性。
- 报告 AUROC、AUPR、FPR95、OSCR、已知准确率和未知拒识率。
- 聚类同时报告：候选池整体结果、候选池中真实未知子集、全部真实未知样本的 oracle-K 上限，以及 auto-K 诊断。
- 蒸馏对照保持 CE、Standard KD、Uncertainty KD；Uncertainty KD 可比较 `raw/mean_normalized` 权重和 `confidence/classification_error/margin` 目标。

## 推荐实验协议

1. 固定 `CIFAR-100-dataset-main`、60/40 划分、ResNet-34 teacher、ResNet-18 student、image size 64、15 epoch、batch size 64。
2. 固定 score mode 为 `normalized_entropy_mahalanobis`，阈值用已知验证集 percentile。
3. 至少 3 个 seed：`42, 43, 44`，报告 mean ± std、参数量和推理成本。
4. 主对照：CE、Standard KD、Uncertainty KD。
5. 聚类同时看 oracle-K 和 auto-K；oracle-K 只作为表征上限，不作为无监督最终结论。
6. 先看 candidate purity，再解释聚类指标。训练阶段不使用未知测试标签。

## 项目成员

- 项目负责人：王笑颜
- 开发成员：李柯颖
- 开发成员：齐誉涵

## 主要文件

- `train.py`：训练和发现入口
- `novel_discovery/`：模型、损失、数据、指标和流程代码
- `CIFAR-100-dataset-main/`：本地 ImageFolder CIFAR-100
- `splits_cifar100_60_40_imagefolder.json`：当前 ImageFolder 60/40 类别名划分
- `scripts/run_cifar_fixed_3seeds.ps1`：3 seed 正式对照脚本
- `scripts/run_revised_ablation.ps1`：历史消融脚本
- `analysis/`：结果摘要

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

检查本地 CIFAR-100 ImageFolder 划分：

```powershell
python train.py inspect_data `
  --dataset imagefolder `
  --data-root .\CIFAR-100-dataset-main `
  --num-known 60 `
  --split-path .\splits_cifar100_60_40.json `
  --image-size 32 `
  --num-workers 0
```

正常输出应类似：

```text
dataset: imagefolder
known classes: 60
novel classes: 40
train size: 27000
val size: 3000
test size: 10000
```

### 继续 / 启动正式教师训练

当前 CPU 教师训练如果仍在运行，不要重复启动第二个进程。完成后直接使用：

```text
.\runs\cifar100_imagefolder_teacher\teacher.pt
```

如果进程已经退出且没有 checkpoint，重新运行：

```powershell
python train.py train_teacher `
  --dataset imagefolder `
  --data-root .\CIFAR-100-dataset-main `
  --num-known 60 `
  --seed 42 `
  --split-path .\splits_cifar100_60_40.json `
  --image-size 64 `
  --batch-size 64 `
  --num-workers 0 `
  --backbone resnet34 `
  --teacher-backbone resnet34 `
  --pretrained `
  --epochs 15 `
  --alpha-unc 0.1 `
  --work-dir .\runs\cifar100_imagefolder_teacher `
  --teacher-ckpt .\runs\cifar100_imagefolder_teacher\teacher.pt `
  --device auto
```

### 教师完成后训练学生并发现

```powershell
python train.py train_student `
  --dataset imagefolder `
  --data-root .\CIFAR-100-dataset-main `
  --num-known 60 `
  --seed 42 `
  --split-path .\splits_cifar100_60_40.json `
  --image-size 64 `
  --batch-size 64 `
  --num-workers 0 `
  --backbone resnet18 `
  --teacher-backbone resnet34 `
  --student-backbone resnet18 `
  --pretrained `
  --epochs 15 `
  --alpha-kd 1.0 `
  --kd-mode uncertainty `
  --uncertainty-weight-mode mean_normalized `
  --uncertainty-target-mode classification_error `
  --work-dir .\runs\cifar100_imagefolder_student `
  --teacher-ckpt .\runs\cifar100_imagefolder_teacher\teacher.pt `
  --student-ckpt .\runs\cifar100_imagefolder_student\student.pt `
  --device auto

python train.py discover `
  --dataset imagefolder `
  --data-root .\CIFAR-100-dataset-main `
  --num-known 60 `
  --num-novel 40 `
  --seed 42 `
  --split-path .\splits_cifar100_60_40.json `
  --image-size 64 `
  --batch-size 64 `
  --num-workers 0 `
  --backbone resnet18 `
  --student-backbone resnet18 `
  --pretrained `
  --score-mode normalized_entropy_mahalanobis `
  --temperature-calibration `
  --cluster-k auto `
  --cluster-method kmeans `
  --cluster-feature projection_pca `
  --cluster-selection composite `
  --cluster-normalize `
  --work-dir .\runs\cifar100_imagefolder_detect `
  --student-ckpt .\runs\cifar100_imagefolder_student\student.pt `
  --device auto
```

运行 3 个 seed 的正式对照：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_cifar_fixed_3seeds.ps1
```

运行测试：

```bash
python -m unittest discover -s tests
```

## 说明

- 仓库可以包含 `CIFAR-100-dataset-main/`，但默认不提交 `runs/` 和缓存权重。
- ImageFolder 模式会把 `--split-path splits_cifar100_60_40.json` 转成类别名划分文件 `splits_cifar100_60_40_imagefolder.json`。
- `discovery_report.json` 记录 AUPR、OSCR、candidate purity、K 估计误差、候选池/真实未知 oracle 聚类，以及参数量。
- `calibration_report.json` 记录温度、ECE 和不确定性-错误相关性。
- 完整 15 epoch 教师训练完成前，不要把冒烟实验的 `known_acc=0.0781` 当成正式结果。
