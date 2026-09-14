# 基于不确定性知识蒸馏的新类发现方法

本仓库是大创项目“基于不确定性知识蒸馏的新类发现方法”的实验代码、实验脚本和阶段性记录。

项目面向开放场景的三个任务：已知类别分类、已知/未知样本检测，以及对未知候选样本进行无监督新类聚类。

## 当前阶段

目前已经完成一套可运行、可复现、可进行消融比较的两阶段流程：

```text
已知类别数据
    ├─ 教师模型：CE + 不确定性相关损失
    └─ 学生模型：CE + 知识蒸馏 + 不确定性对齐
                 + 可选特征蒸馏、监督对比学习、原型约束
                 + 可选 discovery pool 双视图对比学习
    └─ 测试阶段：开放分数 → 已知/未知检测 → 未知候选聚类
```

当前程序还不是已经充分验证的最终论文方法，更准确的定位是：已知类别监督分类、开放集检测和未知候选聚类实验框架，并包含不确定性感知知识蒸馏与 discovery pool 表征学习的可选实现。

最大问题仍是未知检测能力不足，FPR95 偏高；聚类也会受到错误候选样本和聚类数量估计误差影响。现有结果用于说明流程和比较改进方向，不能直接宣称最终方法已经优于现有方法。

## 已实现内容

### 数据和环境

- 支持 `toy`、`fake`、CIFAR-100 和一般 `imagefolder` 数据集。
- CIFAR-100 主要采用固定的 60 个已知类别、40 个未知类别划分：`splits_cifar100_60_40.json`。
- 支持训练/验证/测试集、discovery pool、缩放、标准化、增强和样本数量限制。
- 使用 PyTorch、torchvision、NumPy、SciPy 和 scikit-learn。
- `--device auto` 会在 CUDA 可用时使用 GPU，也可指定 `cpu` 或 `cuda`。

### 模型和损失

教师模型在已知类别上监督训练，学生模型在教师模型基础上训练。当前可选组件包括：CE、标准温度缩放 KL 蒸馏、不确定性感知蒸馏、特征蒸馏、不确定性对齐、监督对比学习、类别原型约束、伪未知损失、discovery pool 双视图一致性/NT-Xent，以及 discovery unknown loss。

不确定性感知蒸馏的基本形式为：

```text
L_unc-KD = weight(u_teacher) × KL(p_teacher || p_student)
```

教师越不确定，蒸馏权重越小；教师越可靠，蒸馏权重越大。但该设计是否稳定有效仍需固定协议、多随机种子和统计检验确认。

### 开放集检测

测试阶段用模型输出计算开放分数，并使用已知验证集分位数确定阈值。支持 MSP、Energy、熵、预测熵、偶然/认知不确定性、原型距离、Mahalanobis 距离及其组合分数，也支持已知验证集归一化组合。

手动指定 `score-mode` 时，测试集未知标签不参与阈值或分数选择。`--score-mode auto` 或 `--auto-calibrate-score` 配合 `--open-val-ratio` 属于带未知验证数据的校准/分析实验，不能和完全无未知标签的最终设置混淆。

### 未知样本聚类

聚类只处理被检测为未知的候选样本。当前支持 `kmeans`、`agglomerative`、`spectral`，支持 `projection`、`feature`、`projection_pca`、`feature_pca`，并支持 L2 归一化、PCA 降维和白化。

`oracle K` 使用真实未知类别数，作为聚类能力上限/对照；`auto K` 不使用未知标签，根据候选池内部结构估计聚类数。自动 K 会记录 Silhouette、Calinski-Harabasz、Davies-Bouldin、不同初始化之间的稳定性 NMI 和每个候选 K 的指标，写入 `discovery_report.json` 和 `discovery_detail.json`。

## 已有进步与结果

已经完成教师/学生训练、开放检测、未知候选聚类、消融、分数比较、误差分析和部分多随机种子实验。

在 CIFAR-100 60/40 设置下，最新一组 discovery pool 多种子汇总为：

| 方法 | AUROC | FPR95 | OSCR | 已知准确率 | Cluster ACC |
|---|---:|---:|---:|---:|---:|
| E：原完整模型 | 0.5823 | 0.8829 | 0.3118 | 0.4068 | 0.1873 |
| F：E + discovery pool NT-Xent | 0.6397 | 0.8413 | 0.4084 | 0.5221 | 0.2058 |

阶段性结论：discovery pool 双视图对比学习对未知检测和已知分类有积极作用，但聚类提升较小；`auto K` 仍不可靠；不确定性感知 KD 尚未显示稳定优于标准 KD 的证据；单次或小样本实验不能支持最终论文结论。

主要结果文件：

- `analysis/revised_multiseed/multiseed_summary.md`
- `analysis/revised_multiseed_2seed/multiseed_summary.md`
- `analysis/discovery_pool_multiseed_comparison.md`
- `analysis/uncertainty_weight_compare/`
- `analysis/score_sweep_s123_E/`
- `analysis/open_set_error_analysis.md`

## 当前问题与解决方向

### 1. 未知检测能力不足

AUROC 约 0.6、FPR95 偏高，说明未知和已知样本的开放分数重叠严重。可能原因是模型只见过已知类别、已知分类准确率不足、不确定性未校准、分数尺度不一致以及阈值策略有限。

建议固定训练协议，系统比较 MSP、Energy、熵、原型距离和 Mahalanobis；使用已知验证集统计量进行 z-score 归一化后融合；检查温度校准、可靠性图和 ECE；同时报告 AUROC、AUPR、FPR95、OSCR、已知准确率和未知拒识率。确认基础模型后，再尝试 Outlier Exposure、Energy loss 或更明确的伪未知训练。可参考 Energy-based OOD Detection、Outlier Exposure 和 Open Set Recognition: A Good Closed-Set Classifier is All You Need?。

### 2. 聚类候选池混入错误接受的已知样本

聚类对象是“被检测为未知”的样本，不是全部真实未知样本；因此聚类差不一定是聚类算法本身的问题。

建议同时报告候选池整体结果和真实未知子集结果；用 `oracle K` 判断特征本身是否可聚类；用 `auto K` 时保存候选 K 指标；比较 projection、feature、PCA、L2 归一化特征以及三种聚类方法。oracle K 也差时优先改进表征；oracle K 好而 auto K 差时优先改进 K 估计。可参考 Deep Embedded Clustering、Supervised Contrastive Learning、Deep Transfer Clustering 和 Relational Knowledge Distillation。

### 3. 不确定性感知蒸馏尚未证明有效

代码已经实现不确定性调节 KD 权重，但结果没有稳定证明其优于标准 KD。可能原因是教师模型不够准确、MC Dropout 未校准或权重尺度不合适。

建议在完全相同条件下比较 CE、Standard KD 和 Uncertainty KD；比较 `raw`/`mean_normalized` 权重及 confidence、classification_error、margin 目标；分析教师不确定性与分类错误的相关性，并用至少 3 个随机种子报告均值和标准差。可参考 Hinton 的 Knowledge Distillation、Kendall and Gal、Gal and Ghahramani 和 Deep Ensembles。

### 4. 尚不是严格的端到端新类发现

当前 discovery pool 主要用于学生模型表征学习，测试时仍是“先检测、后聚类”的两阶段流程。建议先验证 discovery pool 的稳定收益，再加入高置信候选伪标签、周期更新和一致性约束；训练阶段不使用未知测试标签，并明确两阶段方法的适用边界。可参考 Deep Transfer Clustering、Deep Embedded Clustering 和对比学习中的伪标签更新思路。

### 5. 实验统计可靠性不足

smoke test 和单随机种子只适合检查代码。正式实验应固定数据划分、backbone、预训练、训练轮数、学习率、batch size、阈值、score mode 和聚类配置，至少使用 3 个随机种子，并报告均值、标准差、参数量和推理成本。

## 运行方法

### 检查数据

```powershell
python train.py inspect_data `
  --dataset cifar100 --data-root .\data --download `
  --num-known 60 --seed 42 `
  --split-path .\splits_cifar100_60_40.json
```

### 训练教师和学生

```powershell
python train.py train_teacher `
  --dataset cifar100 --data-root .\data --num-known 60 `
  --split-path .\splits_cifar100_60_40.json --image-size 64 `
  --batch-size 128 --epochs 30 --device auto `
  --work-dir .\runs\teacher

python train.py train_student `
  --dataset cifar100 --data-root .\data --num-known 60 `
  --split-path .\splits_cifar100_60_40.json --image-size 64 `
  --batch-size 128 --epochs 30 --device auto `
  --teacher-ckpt .\runs\teacher\teacher.pt `
  --student-ckpt .\runs\student\student.pt `
  --work-dir .\runs\student
```

如需启用 discovery pool，可增加：`--discovery-pool --alpha-discovery 0.1 --discovery-loss nt_xent`。

### 改进后的自动 K 聚类

```powershell
python train.py discover `
  --dataset cifar100 --data-root .\data --num-known 60 --num-novel 40 `
  --seed 42 --split-path .\splits_cifar100_60_40.json `
  --student-ckpt .\runs\student\student.pt `
  --cluster-k auto --cluster-method kmeans `
  --cluster-feature projection_pca --cluster-selection composite `
  --cluster-pca-dim 32 --cluster-normalize --mc-samples 4 `
  --device auto --work-dir .\runs\discover_auto_pca
```

新增聚类选项已经写入代码，但尚未正式实验验证，不能提前宣称一定提升指标。已有长实验脚本：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_revised_multiseed.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_discovery_pool_multiseed.ps1
```

## 目录说明

```text
train.py                         训练、检测和聚类入口
novel_discovery/data.py          数据加载、划分和 discovery pool
novel_discovery/models.py        教师/学生模型
novel_discovery/losses.py        CE、KD、不确定性、对比和原型损失
novel_discovery/pipeline.py      训练、开放检测和聚类
novel_discovery/metrics.py       检测与聚类指标
scripts/                         对比、消融和多种子脚本
analysis/                        阶段性结果和误差分析
docs/                            方法说明、实验计划和组内记录
splits_*.json                    固定类别划分
data/                            本地数据，不上传仓库
runs/                            权重和运行结果，不上传仓库
```

## 指标说明

- `AUROC`：开放分数区分未知/已知的整体排序能力，越高越好。
- `AUPR`：未知样本为正类时的平均精确率。
- `FPR95`：未知召回率达到 95% 时的已知误接受率，越低越好。
- `OSCR`：同时考虑已知分类正确率和未知拒识能力，越高越好。
- `unknown_reject_rate`：真实未知样本被拒绝的比例，越高越好。
- `cluster_acc/nmi/ari`：未知候选样本聚类指标，必须结合候选池大小和错误接受率分析。

## 注意事项

- `oracle K` 使用真实未知类别数，只能作为分析对照。
- 不能使用测试集未知标签选择阈值、score mode 或聚类方法后，把结果当作无监督最终结果。
- `toy` 和带 `limit-*` 的实验只用于调试，不能替代 CIFAR-100 正式实验。
- 本仓库不包含 CIFAR-100 数据、模型权重和大型运行结果，组员应按命令重新生成或使用外部共享路径。
