# Uncertainty-Aware Knowledge Distillation for Novel Class Discovery

本仓库实现一个面向开放环境图像识别的实验框架，研究不确定性估计、知识蒸馏、未知样本拒识与新类别聚类之间的衔接关系。代码以 CIFAR-100 为主要验证数据集，同时保留 `imagefolder`、`toy` 和 `fake` 数据接口，便于扩展和快速测试。

## 研究任务

给定训练阶段可获得标注的已知类别，模型需要完成以下连续任务：

1. 学习已知类别的分类表征；
2. 对偏离已知类别支持范围的样本进行风险评分和未知拒识；
3. 将通过拒识的候选样本映射到特征空间，并组织为若干候选新类别簇。

实验协议中，未知类别不参与模型训练、模型选择或阈值校准；未知标签只在最终评测阶段用于计算开放集指标和聚类指标。这样可以区分模型真正学到的拒识能力与使用未知标签进行调参的结果。

## 方法流程

当前代码将流程拆成训练、拒识和聚类三个阶段：

- 教师模型学习已知类别分类知识；
- 学生模型可以使用标准知识蒸馏，或进一步使用教师不确定性对蒸馏强度进行加权；
- 学生模型输出分类 logits、分类特征、不确定性信号以及可选的拒识特征；
- 拒识阶段在已知验证数据上校准工作点，再对发现池计算开放风险；
- 阈值以上的样本进入未知候选池，候选池可按实验设置进行净化或重新排序；
- 聚类阶段默认对分类特征执行 PCA，再使用 KMeans 进行候选新类别划分；
- 评估同时报告分类、未知检测、候选池和聚类结果，以定位误差来自检测还是聚类。

拒识编码器的两种主要配置通过 `--independent-rejection-encoder-init` 选择：`scratch_cnn` 使用独立轻量 CNN 从随机参数开始，`shared_backbone` 使用学生分类主干的参数初始化独立拒识编码器。它们是拒识分支的配置，不是最终分类器的名称。

## Team A：分类表征与不确定性知识蒸馏

该实验线固定 CIFAR-100 semantic-hard 60/40 开放环境协议，在三组独立类别划分上比较标准 logits-KD 与 MC-epistemic uncertainty-KD。两种方案使用相同的教师模型、学生模型、训练目标、拒识配置、阈值策略和自动 K 聚类流程；唯一改变的是 logits-KD 是否按教师四次随机前向结果得到的不确定性进行样本级加权。

结果审计确认不确定性权重实际参与了优化，但当前权重变化幅度较小，且三组划分上的端到端收益不稳定。因此，standard-KD 保持为正式对照基线，uncertainty-KD 保留为显式消融与诊断方案，暂不作为默认训练方式。实验协议、逐划分结果和训练审计见：

- [实验协议](analysis/team_a_representation/protocol.md)
- [结果总结](analysis/team_a_representation/summary.md)
- [训练与梯度审计](analysis/team_a_representation/audit.md)
- [逐划分结果](analysis/team_a_representation/results.csv)
- [可复现配置](configs/reproducibility/team_a/)

## 环境安装

建议使用 Python 3.10 或更高版本，并在独立虚拟环境中安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

GPU 版本的 PyTorch 请按照本机 CUDA 版本先安装对应官方包，再安装其余依赖。

## 数据与类别划分

首次运行 CIFAR-100 时，可以让 torchvision 自动下载数据。仓库内提供了可复用的类别划分文件：

- `splits/cifar100_60_40.json`：60 个已知类别、40 个未知类别的基础划分；
- `splits/semantic_hard/`：用于检验语义相近类别下拒识稳定性的划分；
- `splits/semantic_isolated/`：用于补充协议敏感性分析的划分；
- `splits/default.json`：快速检查和默认示例使用的划分。

训练、校准、测试和发现阶段应使用同一个 `--split-path`。不要在阈值校准阶段读取测试标签。

## 基本复现流程

### 1. 检查数据协议

```powershell
python train.py inspect_data --dataset cifar100 --data-root ./data --split-path ./splits/cifar100_60_40.json --download
```

### 2. 训练教师模型

```powershell
python train.py train_teacher --dataset cifar100 --data-root ./data --split-path ./splits/cifar100_60_40.json --backbone resnet34 --work-dir ./runs/teacher --download
```

默认保存为 `runs/teacher/teacher.pt`；实际训练输出目录由 `--work-dir` 决定。

### 3. 训练学生模型

```powershell
python train.py train_student --dataset cifar100 --data-root ./data --split-path ./splits/cifar100_60_40.json --teacher-ckpt ./runs/teacher/teacher.pt --backbone resnet18 --work-dir ./runs/student --download
```

若要进行拒识编码器配置对照，可在保持其余参数一致时分别使用：

```powershell
python train.py train_student --dataset cifar100 --data-root ./data --split-path ./splits/cifar100_60_40.json --teacher-ckpt ./runs/teacher/teacher.pt --backbone resnet18 --independent-rejection-encoder --independent-rejection-encoder-init scratch_cnn --work-dir ./runs/student_scratch --download
python train.py train_student --dataset cifar100 --data-root ./data --split-path ./splits/cifar100_60_40.json --teacher-ckpt ./runs/teacher/teacher.pt --backbone resnet18 --independent-rejection-encoder --independent-rejection-encoder-init shared_backbone --work-dir ./runs/student_shared --download
```

### 4. 运行未知拒识和新类聚类

```powershell
python train.py discover --dataset cifar100 --data-root ./data --split-path ./splits/cifar100_60_40.json --student-ckpt ./runs/student/student.pt --work-dir ./runs/discovery --download
```

发现阶段的关键参数包括：

- `--threshold-policy`：选择全局阈值、已知覆盖率阈值或开放验证工作点；
- `--threshold-percentile`：已知验证分数的百分位阈值，默认值为 95；
- `--cluster-feature`：选择聚类表征，默认是 `feature_pca`；
- `--cluster-method`：选择 `kmeans`、`agglomerative` 或 `spectral`；
- `--cluster-k`：使用已知类别数作为评测用 oracle，或启用自动类别数搜索；
- `--candidate-purify`：在拒识之后对候选池进行可选净化；
- `--cluster-candidate-mode`：选择检测器候选或拒识特征排序候选。

### 5. 扫描开放风险分数

`score_sweep.py` 在同一次 Monte Carlo 特征提取结果上比较多种开放风险分数，减少不同分数重复抽样造成的随机差异：

```powershell
python score_sweep.py --dataset cifar100 --data-root ./data --split-path ./splits/cifar100_60_40.json --student-ckpt ./runs/student/student.pt --work-dir ./runs/score_sweep --device auto
```

## 评价指标

- 分类准确率：衡量已知类别分类是否正确；
- AUROC：衡量未知风险分数对已知和未知样本的整体排序能力；
- 未知拒识率：未知样本被判定为候选的比例；
- 已知污染率：候选池中实际属于已知类别的比例；
- 候选池未知召回率和纯度：分别衡量未知样本覆盖范围与候选池可靠性；
- ACC、NMI、ARI：衡量候选样本聚类与真实未知类别之间的一致程度；
- 自动 K 误差：衡量自动估计的新类别数量与评测类别数量之间的差异。

测试标签只用于这些最终统计，不应反过来用于训练、阈值选择或方法选择。

## 测试与代码检查

提交前可运行：

```powershell
python -m compileall train.py novel_discovery tests
python -m unittest discover -s tests
```

代码中的实验脚本位于 `scripts/`，聚合和诊断工具位于仓库根目录及 `scripts/`。运行脚本产生的 checkpoint、运行目录和分析输出默认不纳入版本控制。

## 目录结构

```text
train.py                  # 教师、学生、发现和数据检查入口
score_sweep.py            # 开放风险分数对照
novel_discovery/          # 数据、模型、损失、指标和发现流程
tests/                    # 单元测试
scripts/                  # 实验运行、审计和结果分析脚本
splits/                   # 可复用的类别划分协议
docs/                     # 研究方案及必要说明
requirements.txt          # Python 依赖
EXPERIMENT_LOG.md         # 原有实验记录，内容原样保留
```

## 可复现注意事项

1. 固定 `--split-path`、`--seed`、`--model-seed`、模型结构和训练轮数；
2. 训练、校准和发现阶段使用一致的图像尺寸、归一化方式和 batch 设置；
3. 比较单一因素时只改变一个参数，其余参数、数据划分和随机种子保持一致；
4. 记录完整命令、checkpoint 路径和输出目录，避免仅保存最终指标；
5. 不要把测试标签用于阈值校准或模型选择。

## 当前边界

当前实现提供的是开放集识别与候选新类聚类的研究框架，不保证在所有类别划分上同时取得稳定的完整新类发现收益。语义相近类别之间的特征重叠、候选池边界样本的稳定性以及自动类别数估计仍需要通过独立对照实验继续验证。

## 实验记录

旧 README 中的实验记录已完整迁移到 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)，记录正文未做删改或重写。新的 README 只负责说明项目用途、代码结构和复现方法。
