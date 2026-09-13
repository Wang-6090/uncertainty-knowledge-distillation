# 基于不确定性知识蒸馏的新类发现方法

本仓库是大创项目“基于不确定性知识蒸馏的新类发现方法”的实验代码和阶段性记录。

项目希望解决一个开放场景问题：模型训练时只见过已知类别，测试或部署时可能出现没有训练过的新类别。系统需要同时完成：

1. 已知类别分类；
2. 已知样本与未知样本检测；
3. 对未知样本进行无监督的新类聚类。

当前代码已经可以完成一套可复现的“两阶段开放学习流程”，但还不是最终意义上的端到端新类发现系统。当前结果主要用于验证流程和比较模块，不能直接作为最终论文结论。

## 一、当前已经完成的工作

### 1. 数据和实验环境

- 支持 `toy` 快速测试数据集；
- 支持 CIFAR-100；
- CIFAR-100 当前固定划分为 60 个已知类别、40 个未知类别；
- 已保存划分文件：`splits_cifar100_60_40.json`；
- 支持图像缩放、标准化、数据增强和训练/验证/测试划分；
- 使用 PyTorch、Torchvision、NumPy、SciPy 和 scikit-learn。

### 2. 教师模型和学生模型

当前实验流程为：

```text
已知类别训练集
       │
       ▼
教师模型：学习已知类别分类和特征表示
       │
       │ logits、特征、不确定性
       ▼
学生模型：CE + 知识蒸馏 + 特征约束 + 对比/原型约束
       │
       ▼
测试样本的分类、未知检测和新类聚类
```

当前代码支持：

- 已知类别交叉熵训练；
- 标准温度缩放 KL 蒸馏；
- 教师不确定性感知的加权 KL 蒸馏；
- 教师/学生特征蒸馏；
- MC Dropout 不确定性估计；
- 监督对比损失；
- 类别原型约束；
- MobileNetV3-Small 轻量模型实验。

主要代码位于：

- `train.py`：训练和评测命令入口；
- `novel_discovery/models.py`：教师、学生模型及输出特征；
- `novel_discovery/losses.py`：CE、KL、特征蒸馏、不确定性和对比损失；
- `novel_discovery/pipeline.py`：不确定性计算、开放集打分和未知聚类；
- `novel_discovery/data.py`：数据集加载和已知/未知划分；
- `novel_discovery/metrics.py`：检测和聚类指标。

### 3. 开放集检测和新类发现

当前支持以下未知分数：

- 最大 Softmax 概率 MSP；
- Energy；
- 预测熵；
- 原型距离；
- 对角协方差 Mahalanobis 距离；
- 熵与原型距离组合；
- 熵与 Mahalanobis 距离组合；
- MC Dropout 得到的 epistemic、aleatoric 等不确定性量。

阈值当前使用已知验证集的固定分位数确定，测试集标签不参与阈值选择。

未知样本过滤后，代码支持两种聚类数量设置：

- `oracle`：使用真实未知类别数量，属于上限或对照实验；
- `auto`：使用轮廓系数自动估计聚类数量，更接近真实开放场景。

聚类算法当前使用 K-Means，特征主要使用归一化 projection 特征。

## 二、目前已有结果及其含义

当前 CIFAR-100 60/40 单次消融实验结果如下：

| 方法 | AUROC | FPR95 | Known acc | Unknown reject | Cluster ACC | NMI | ARI |
|---|---:|---:|---:|---:|---:|---:|---:|
| CE | 0.5939 | 0.8812 | 0.4527 | 0.0673 | 0.1820 | 0.5127 | 0.0535 |
| Standard KD | 0.5955 | 0.8632 | 0.4093 | 0.0628 | 0.2022 | 0.5190 | 0.0571 |
| Uncertainty KD | 0.5898 | 0.8613 | 0.4290 | 0.0508 | 0.2117 | 0.5387 | 0.0628 |
| Feature KD | 0.5846 | 0.8790 | 0.4113 | 0.0653 | 0.1865 | 0.5083 | 0.0483 |
| Full model | 0.5977 | 0.8713 | 0.4375 | 0.0747 | 0.1768 | 0.4759 | 0.0398 |

这些结果说明：

- 当前流程确实可以运行并输出分类、检测和聚类结果；
- 标准 KD 相比 CE 的提升很小；
- 不确定性加权 KD 暂时没有稳定提升 AUROC；
- 完整模型没有同时改善所有指标；
- FPR95 较高，未知样本拒识率较低，开放集检测仍然较弱；
- 聚类 ACC 较低，说明未知样本的特征表示和未知样本筛选仍需改进。

因此当前最严谨的表述是：

> 我们已经完成了不确定性知识蒸馏开放学习原型和初步消融实验，但尚未证明提出的方法稳定优于标准 KD 或其他基线。

上述结果只有一个主要随机种子，且部分早期实验使用了样本限制参数。正式结论必须重新进行多随机种子和统一训练规模实验。

## 三、最近吸收的实验脚本修改

`scripts/run_cifar_long_compare.ps1` 已吸收同学分支中值得保留的实验配置：

- 教师和学生训练轮数从 20 增加到 30；
- 增加 `--download`，数据不存在时自动下载；
- 发现阶段显式使用 `--cluster-k auto`；
- 关闭 tqdm 进度条，减少日志干扰。

没有修改 `train.py` 的全局默认参数。已有命令仍然可以运行；第四大点对应的 discovery pool、温度缩放、分数对比和 auto-K 改进需要显式打开新参数。不确定性加权 KD 现在默认对 `exp(-u)` 做裁剪和均值归一化。

## 四、当前问题、推荐解决方法与代码改动

第四大点里的 A–F 已经落到代码上。旧的两阶段命令仍然可用；新功能默认关闭，需要显式打开。

### 问题 A：当前还不是真正的端到端新类发现

当前默认流程仍是“已知类别监督训练 → 测试阶段未知检测 → 对未知样本进行后处理聚类”。未知样本没有参与特征学习，因此更准确地说是开放集检测加聚类原型。

代码改动：增加独立的无标注 `discovery pool`。训练集中的未知类图像不带标签，每张图生成两个增强视图；学生在已知类 CE / KD 之外，对这两个视图做 cosine consistency 和 NT-Xent 对比学习。可以周期性用 K-Means 生成伪标签，只用离簇中心较近的高置信样本，再用多视图一致性更新学生。

默认第一版只开双视图一致性、固定 `K=8`；确认特征聚类改善后再开 `--alpha-cluster` 和 `--discovery-cluster-k auto`。

```powershell
python train.py train_student `
  --include-discovery-pool --discovery-epochs 8 `
  --alpha-consistency 0.5 --alpha-contrast 0.1 --alpha-cluster 0.0 `
  --discovery-cluster-k fixed8
```

也可以在已有学生权重上单独做发现训练：

```powershell
python train.py train_discovery --student-ckpt .\runs\student.pt --include-discovery-pool
```

文献参考：Han、Vedaldi、Zisserman 的 [Learning to Discover Novel Visual Categories via Deep Transfer Clustering](https://arxiv.org/abs/1908.09884) 可参考“已知类监督信息如何迁移到未知类聚类”的整体思路；Khosla 等人的 [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) 可参考方法部分的多视图对比损失和实验中的消融设置；Xie、Girshick、Farhadi 的 [Unsupervised Deep Embedding for Clustering Analysis](https://arxiv.org/abs/1511.06335) 可参考“表示学习与聚类目标联合优化”的思路。我们不必照搬网络结构，重点借鉴 discovery pool、伪标签更新和特征空间优化方式。

### 问题 B：未知检测 AUROC 较低、FPR95 较高

这说明已知和未知样本的分数分布重叠较严重，当前模型容易把未知样本当成已知类别。

代码改动：`--compare-scores` 会在同一测试集上比较 MSP、Energy、预测熵、原型距离、Mahalanobis 以及它们的组合；`--score-mode zscore_fusion` 只使用已知验证集均值/方差做 z-score 融合；`--temperature-scaling` 在已知验证集上拟合标量温度。发现报告现在同时给出 AUROC、AUPR、FPR95、OSCR 和已知类 ECE。阈值仍然只由已知验证集分位数决定。

```powershell
python train.py discover --score-mode entropy_mahalanobis --temperature-scaling --compare-scores
```

如果仍然没有改善：检查数据划分、预训练权重、教师模型收敛情况、prototype 计算和 MC Dropout 是否正确；减少复杂分数融合，选择验证集上最稳定的单一分数，并报告检测能力有限这一结果。

文献参考：Vaze、Han、Vedaldi 的 [Open-Set Recognition: A Good Closed-Set Classifier is All You Need?](https://arxiv.org/abs/2110.06207) 可参考开放集评测中对已知分类能力和未知检测能力的分离分析；Hendrycks、Mazeika、Dietterich 的 [Deep Anomaly Detection with Outlier Exposure](https://arxiv.org/abs/1812.04606) 可参考利用额外异常样本改善未知检测的思路；Energy 分数可结合 Liu 等人的 [Energy-based Out-of-distribution Detection](https://arxiv.org/abs/2010.03759) 进一步阅读。重点看这些论文的方法定义、阈值/评分实验和不同 OOD 数据集上的对比，而不是只比较单一 AUROC。

### 问题 C：不确定性知识蒸馏暂时没有证明有效

当前形式是：

```text
L_unc-KD = mean(exp(-u_teacher) × KL(p_teacher || p_student))
```

它的假设是教师越可靠，学生越应该学习教师输出；教师越不确定，蒸馏权重越小。但当前不确定性加权 KD 没有稳定优于标准 KD。

代码改动：不确定性加权改为

```text
w = clip(exp(-u_teacher), 0.05, 1.0)
w = w / mean(w)
L_unc-KD = mean(w × KL(p_teacher || p_student))
```

这样平均蒸馏强度与标准 KD 可比，同时仍然下调不可靠教师样本。学生训练后会写出 `teacher_uncertainty_diagnostics.json`，记录不确定性与分类错误、置信度的相关性和 ECE。三种不确定性目标仍然可用：`--uncertainty-target-mode confidence|classification_error|margin`。固定协议脚本见 `scripts/run_protocol_ablation.ps1`，默认只跑 A/B/C 和 3 个 seed。

如果仍然没有改善：不要继续堆叠损失函数，也不要宣称该模块有效。可以将“不确定性蒸馏未带来稳定提升”作为负结果，同时保留标准 KD 作为基线，并进一步尝试 deep ensemble 或温度校准后的教师不确定性。

文献参考：Hinton、Vinyals、Dean 的 [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) 可参考温度缩放 KL 蒸馏的基本形式；Gal、Ghahramani 的 [Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142) 可参考 MC Dropout 近似认知不确定性的方法；Kendall、Gal 的 [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977) 可参考偶然不确定性与认知不确定性的区分；Lakshminarayanan 等人的 [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/abs/1612.01474) 可作为 deep ensemble 对照。我们主要借鉴蒸馏公式、不确定性分解和对照实验设计。

### 问题 D：模型特征不够适合未知类聚类

当前完整模型的聚类指标没有稳定提升，说明未知样本特征可能不够紧凑，或者错误接受的已知样本混入了聚类集合。

代码改动：聚类一律使用 L2 归一化 projection 特征；发现训练加入双视图对比；`--cluster-confidence-percentile 50` 只把高未知分数样本送去聚类。报告拆开 `unknown_detection_miss_rate`（真未知没被检出）和 `cluster_error_on_filtered`（检出后的聚类错误）。`--cluster-k both` 会同时给出 oracle K 和 auto K，便于判断是特征差还是 K 估差。

如果仍然没有改善：先使用 oracle K 判断特征本身是否可聚类；若 oracle K 也很差，优先改进特征学习；若 oracle K 较好而 auto K 较差，优先改进聚类数量估计，而不是继续修改模型损失。

文献参考：Chen 等人的 [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) 可参考数据增强、投影头和特征归一化；Khosla 等人的 [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) 可参考同类聚合、不同类分离和温度参数实验；Park 等人的 [Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068) 可参考蒸馏特征关系而不只蒸馏分类 logits 的思路。重点检查论文的 representation ablation 和 t-SNE/聚类可视化实验。

### 问题 E：`auto K` 仍然部分使用未知类别信息

当前自动聚类使用轮廓系数，但 `--num-novel 40` 仍然作为最大聚类数上限。因此它比 oracle K 更接近真实场景，但还不是完全未知类别数的设置。

代码改动：`--cluster-k auto` 不再把 `--num-novel` 当作搜索上限。候选范围只由样本数决定，大约是 `[2, min(30, sqrt(N))]`。K 的选择综合 silhouette、Calinski-Harabasz 和 Davies-Bouldin。报告中会写出 `true_k`、`estimated_k` 和 `k_abs_error`。`--cluster-k both` 把 oracle 和 auto 分开保存，避免混进同一张结论表。

如果自动 K 仍然不稳定：保留 oracle K 作为特征聚类上限实验，保留 auto K 作为真实场景实验，明确说明两者用途不同，不能混在同一张结论表中。

文献参考：Xie、Girshick、Farhadi 的 [Unsupervised Deep Embedding for Clustering Analysis](https://arxiv.org/abs/1511.06335) 可参考聚类目标与表示学习结合的做法；可进一步比较 K-Means、层次聚类和基于密度的方法，并采用 silhouette、Calinski-Harabasz、Davies-Bouldin 等内部指标选择候选 K。这里应重点借鉴“如何在没有真实标签时选择聚类数量”的实验设计，不能用测试集真实 K 选择最终方法。

### 问题 F：实验协议和结论可靠性不足

当前部分结果来自单个随机种子或受限样本量，不能代表最终性能；同时多个损失一起启用时，很难判断是哪一个模块产生了影响。

推荐方法：固定 CIFAR-100 60/40 划分、backbone、预训练设置、训练轮数、学习率、阈值策略和 score mode；使用至少 3 个随机种子；按以下顺序进行消融：

```text
A: CE
B: CE + Standard KD
C: CE + Uncertainty KD
D: C + Feature KD
E: D + SupCon + Prototype
F: E + discovery pool 一致性训练
```

代码改动：`scripts/run_protocol_ablation.ps1` 固定划分、backbone、轮数、阈值和 score mode，默认跑 A/B/C 三个设置和 3 个 seed。`discover --report-efficiency` 记录参数量和推理时间。`analyze_results.py` 汇总 AUROC、AUPR、FPR95、OSCR、已知准确率、未知拒识率、聚类 ACC/NMI/ARI 和估计 K。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_protocol_ablation.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_discovery_pool.ps1
```

如果资源或时间不足：优先完成 A、B、C 三组和 3 个 seed；暂时不加入更多模块。若 C 相比 B 没有稳定改善，应如实报告，不用完整模型结果掩盖该结论。

文献参考：Guo 等人的 [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) 可参考温度缩放、可靠性图和 ECE 评价；Zhao、Cui、Song 的 [Decoupled Knowledge Distillation](https://arxiv.org/abs/2203.08679) 可参考把分类性能和知识迁移效果分开分析的消融思路；Hinton 等人的蒸馏论文可作为标准 KD 基线。我们应借鉴这些论文的固定协议、基线和消融原则，避免同时改变模型、损失、阈值和数据划分。

## 五、如何运行当前代码

### 安装依赖

```powershell
pip install -r requirements.txt
```

### 最小数据检查

```powershell
python train.py inspect_data `
  --dataset toy `
  --num-known 10 `
  --limit-train 32 `
  --limit-val 16 `
  --limit-test 32 `
  --image-size 64
```

### 运行长实验脚本

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_cifar_long_compare.ps1
```

该脚本会运行：

- CIFAR-100 教师模型 30 轮训练；
- CIFAR-100 学生模型 30 轮蒸馏；
- `full + auto K` 发现实验；
- `entropy_proto + auto K` 发现实验；
- 结果分析。

### 单独运行发现阶段

```powershell
python train.py discover `
  --dataset cifar100 `
  --data-root .\data `
  --num-known 60 `
  --num-novel 40 `
  --seed 42 `
  --split-path .\splits_cifar100_60_40.json `
  --student-ckpt .\runs\cifar_long_train\student.pt `
  --score-mode entropy_mahalanobis `
  --cluster-k both `
  --temperature-scaling `
  --compare-scores `
  --mc-samples 4 `
  --device auto `
  --work-dir .\runs\manual_discover
```

`--device auto` 会自动使用 CUDA；也可以明确指定 `--device cpu` 或 `--device cuda`。

## 六、目录说明

```text
train.py                         训练、检测和聚类入口
novel_discovery/                 模型、损失、数据、指标和流程
scripts/run_protocol_ablation.ps1  A/B/C 多 seed 固定协议
scripts/run_discovery_pool.ps1     discovery pool 一致性训练
analysis/                        阶段性实验结果和误差分析
docs/revised_method.md           当前方法说明
docs/revised_experiment_plan.md  消融实验计划
docs/group_progress.md           组内进度记录
splits_cifar100_60_40.json       CIFAR-100 固定类别划分
data/                            本地数据，不上传仓库
runs/                            模型权重和运行结果，不上传仓库
```

## 七、当前阶段给组员的结论

目前项目已经从“能否运行”进入“建立可靠实验和改进方法”的阶段。组员接下来最重要的工作不是继续随意增加损失函数，而是：

1. 按统一协议完成多 seed 基线；
2. 检查和校准不确定性；
3. 让不确定性加权 KD 与标准 KD 进行清晰、公平的对照；
4. 增加真正的无标注 discovery pool 和伪标签一致性训练；
5. 分开报告 `oracle K` 与 `auto K`；
6. 对所有结论保留配置、日志、指标和可视化结果。

只有完成这些工作后，才能判断“基于不确定性知识蒸馏的新类发现方法”是否真的优于现有基线。
