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

没有修改 `train.py` 的全局默认参数，也没有改变核心模型、蒸馏损失或不确定性算法。已有命令仍然可以复现旧实验。

## 四、当前存在的主要问题

### 问题 A：目前还不是真正的端到端 NCD

当前流程本质上是：

```text
已知类别监督训练 → 测试阶段未知检测 → 对筛出的未知样本聚类
```

未知样本没有参与表示学习，聚类发生在训练完成之后。因此当前更准确的名称是“开放集检测加后处理聚类”，还不能完全声称实现了无监督新类发现训练。

后续需要增加独立的无标注 discovery pool，并使用多视图一致性、伪标签或对比学习优化未知样本的表示。

### 问题 B：未知检测指标不够好

目前 AUROC 大约在 0.59 左右，FPR95 较高。这表明模型对已知和未知样本的分数分布重叠严重。

需要重点检查：

- 教师模型和学生模型是否充分收敛；
- 训练集、验证集、测试集是否严格隔离；
- 阈值是否只用已知验证集确定；
- prototype 是否由训练集统计并正确归一化；
- `full` 分数中不同分量是否量纲不一致；
- MC Dropout 是否真的处于随机 dropout 状态；
- 不确定性是否经过校准。

### 问题 C：不确定性蒸馏的有效性尚未验证

当前教师不确定性通过样本权重影响 KL 蒸馏，基本形式为：

```text
L_unc-KD = mean(exp(-u_teacher) × KL(p_teacher || p_student))
```

这个设计有明确直觉：教师越可靠，蒸馏越强；教师越不确定，蒸馏越弱。但目前实验中它没有稳定优于标准 KD，可能原因包括：

- 教师模型本身准确率和校准不足；
- 不确定性量的范围没有统一；
- `exp(-u)` 的权重可能过于接近 1，无法产生有效差异；
- 不确定性目标和真实分类错误的对应关系较弱；
- KL、特征、对比、原型等损失同时使用，难以判断究竟是哪一项有效。

下一轮必须保留清晰的 B/C 对照：

```text
B: CE + 标准 KL
C: CE + 不确定性加权 KL
```

只有 C 在固定协议和多个随机种子下稳定优于 B，才能支持“不确定性蒸馏有效”的结论。

### 问题 D：自动聚类数量仍不是完全未知

`auto K` 使用轮廓系数选择聚类数，但当前仍通过 `--num-novel 40` 提供搜索上限。因此它比 `oracle K` 更合理，但还不能说完全不使用未知类别信息。

报告时必须区分：

- `oracle K`：知道真实未知类别数的上限实验；
- `auto K`：不知道真实类别数的自动估计实验。

后续可以报告估计的 K、真实 K、K 的绝对误差，并研究不提供真实未知类别数时的搜索范围设置。

### 问题 E：实验协议还需要统一

正式实验必须固定：

- 数据集和类别划分；
- 随机种子；
- backbone 和是否使用预训练；
- 训练轮数；
- batch size 和学习率；
- 阈值策略；
- score mode；
- `oracle K` 或 `auto K`；
- 是否使用伪未知样本；
- 是否使用真实未知标签参与任何选择。

否则不同实验之间无法公平比较。

## 五、推荐的后续实现路线

不要一次加入很多新技术。建议按照下面顺序逐步实现，每一步都保留对照实验。

### 第 1 阶段：先建立可靠基线

目标是确认训练和评测本身没有问题。

需要完成：

1. CIFAR-100 60/40 使用完整训练集；
2. 教师和学生训练 30 轮；
3. 使用固定 seed，例如 42、123、3407；
4. 保存验证集表现最好的 checkpoint；
5. 统一比较 CE、Standard KD、Uncertainty KD；
6. 同时报告 `oracle K` 和 `auto K`；
7. 保存每次运行的配置和 JSON 结果。

建议命令：

```powershell
python train.py inspect_data `
  --dataset cifar100 `
  --data-root .\data `
  --download `
  --num-known 60 `
  --seed 42 `
  --split-path .\splits_cifar100_60_40.json

powershell -ExecutionPolicy Bypass -File .\scripts\run_cifar_long_compare.ps1
```

长实验脚本主要用于验证 30 轮训练和自动 K。正式消融仍应使用 `scripts/run_revised_ablation.ps1`，并确保每个实验使用相同的训练协议。

### 第 2 阶段：改进未知检测分数

优先推荐以下组合，而不是继续堆叠更多分数：

1. 使用预测熵或 Energy 作为简单基线；
2. 使用归一化后的 prototype distance；
3. 使用 diagonal Mahalanobis distance；
4. 通过已知验证集统计量进行 z-score 归一化后再融合；
5. 通过温度缩放校准分类 logits；
6. 使用 AUROC、AUPR、FPR95 和 OSCR 共同评价。

分数融合必须先归一化，否则熵和距离可能处于不同数值范围，某一个分量会无意中支配总分。

### 第 3 阶段：实现真正的无标注新类发现

建议新增 discovery 阶段，而不是只在测试阶段聚类：

```text
已知标注集：训练分类和蒸馏
无标注 discovery pool：只提供图像，不提供类别标签
        │
        ├─ 两种增强视图
        ├─ 教师不确定性筛选可靠未知样本
        ├─ projection 特征对比学习
        ├─ K-Means/层次聚类产生候选伪标签
        └─ 多视图伪标签一致性训练
```

第一版建议使用以下简单方案：

- 两个增强视图输入同一学生模型；
- 对两个 projection 做 cosine consistency loss；
- 只对高置信度、低预测熵或稳定 MC 预测样本生成伪标签；
- 周期性使用 K-Means 更新伪标签；
- 伪标签置信度低的样本不参与分类损失；
- 通过 `SupCon` 或伪标签对比损失拉近同类、推远异类。

先实现固定 K 版本验证训练机制，再实现 auto K。这样可以区分“表示学习是否有效”和“自动估计 K 是否有效”。

### 第 4 阶段：完善不确定性建模

当前 MC Dropout 可以作为认知不确定性近似，但还需要验证和校准。

建议增加：

- ECE、Brier score 和可靠性图；
- MC 次数 1、4、8、16 的比较；
- 标准 KD 与不确定性 KD 的权重分布可视化；
- 对 `exp(-u)` 做裁剪或归一化，例如限制在 `[0.2, 1.0]`；
- 比较 confidence、classification error、margin 三种不确定性目标；
- 资源允许时增加 3 个独立教师模型组成 deep ensemble 对照。

推荐的实验逻辑是：先证明不确定性可以识别错误预测，再研究它是否能改善蒸馏。不能直接假设不确定性量一定有效。

### 第 5 阶段：最终评测和论文材料

正式表格至少应包含：

- 已知类别准确率；
- AUROC；
- AUPR；
- FPR95；
- 已知样本接受率；
- 未知样本拒识率；
- 未知类聚类 ACC、NMI、ARI；
- 真实 K、估计 K 和 K 误差；
- 参数量、模型大小和推理时间；
- 3 个以上随机种子的均值和标准差。

必须做的消融顺序：

```text
A: CE
B: CE + Standard KD
C: CE + Uncertainty KD
D: C + Feature KD
E: D + SupCon + Prototype
F: E + Discovery-pool pseudo-label consistency
```

每次只增加一个主要模块，才能说明模块是否真正有效。

## 六、如何运行当前代码

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
  --score-mode entropy_proto `
  --cluster-k auto `
  --mc-samples 4 `
  --device auto `
  --work-dir .\runs\manual_discover
```

`--device auto` 会自动使用 CUDA；也可以明确指定 `--device cpu` 或 `--device cuda`。

## 七、目录说明

```text
train.py                         训练、检测和聚类入口
novel_discovery/                  模型、损失、数据、指标和流程
scripts/                          实验运行脚本
analysis/                         阶段性实验结果和误差分析
docs/revised_method.md            当前方法说明
docs/revised_experiment_plan.md   消融实验计划
docs/group_progress.md             组内进度记录
splits_cifar100_60_40.json        CIFAR-100 固定类别划分
data/                             本地数据，不上传仓库
runs/                             模型权重和运行结果，不上传仓库
```

## 八、当前阶段给组员的结论

目前项目已经从“能否运行”进入“建立可靠实验和改进方法”的阶段。组员接下来最重要的工作不是继续随意增加损失函数，而是：

1. 按统一协议完成多 seed 基线；
2. 检查和校准不确定性；
3. 让不确定性加权 KD 与标准 KD 进行清晰、公平的对照；
4. 增加真正的无标注 discovery pool 和伪标签一致性训练；
5. 分开报告 `oracle K` 与 `auto K`；
6. 对所有结论保留配置、日志、指标和可视化结果。

只有完成这些工作后，才能判断“基于不确定性知识蒸馏的新类发现方法”是否真的优于现有基线。
