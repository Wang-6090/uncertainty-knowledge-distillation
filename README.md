# 基于不确定性知识蒸馏的新类发现方法

本项目面向开放场景下的图像识别问题，研究如何同时完成已知类别分类、未知样本检测和未知样本的新类聚类。

当前仓库已经实现了一套可运行的实验框架，但还不能直接宣称已经得到最终有效的新类发现方法。现阶段重点是建立公平、可复现的基线，验证不确定性知识蒸馏和 discovery pool 学习是否真正改善开放集检测与新类聚类。

## 当前流程

```text
已知类别图像
    |
    +--> 教师模型：CE + 不确定性辅助损失
    |
    +--> 学生模型：CE + KL知识蒸馏
                 + 可选不确定性加权蒸馏
                 + 可选特征蒸馏、SupCon、原型约束
                 + 可选 discovery pool 双视图学习
    |
    +--> 开放集打分：熵、MSP、ODIN、Energy、原型距离、Mahalanobis 等
    |
    +--> 阈值判断：已知样本 / 未知候选样本
    |
    +--> 未知候选样本聚类：KMeans、层次聚类或谱聚类
```

目前的测试阶段仍然是“先未知检测，再对候选样本聚类”的两阶段流程，不是完整的端到端新类发现模型。

## 已实现功能

### 数据与环境

- 支持 `CIFAR-100`、`ImageFolder`、`toy` 和 `fake` 数据集。
- `ImageFolder` 既支持单目录数据，也支持 `root/train`、`root/test` 双目录数据；双目录模式会自动使用训练目录划分 train/val/discovery pool，测试目录作为开放测试集。
- 支持固定已知类 / 未知类划分，例如 `splits_cifar100_60_40.json`。
- 支持训练集、验证集、开放测试集和 discovery pool。
- 支持数据增强、归一化、样本数量限制和 `--device auto` 自动选择 CUDA 或 CPU。
- 默认将 torchvision 预训练权重缓存到项目下的 `.torch_cache`。

### 模型与损失

- 教师模型和学生模型支持 ResNet-18、ResNet-34、MobileNetV3-Small。
- 模型同时输出分类 logits、特征、投影向量和辅助不确定性。
- 支持标准温度 KL 蒸馏：

  ```text
  L_KD = KL(teacher || student) * temperature^2
  ```

- 支持不确定性加权蒸馏：

  ```text
  weight = exp(-teacher_uncertainty)
  ```

  教师越确定，蒸馏信号越强；教师越不确定，蒸馏信号越弱。

- 支持 `raw` 和 `mean_normalized` 两种蒸馏权重模式。
- 支持特征蒸馏、监督对比学习、原型约束、proxy contrastive loss 和伪未知样本损失。
- 支持可选 Energy 分离损失：约束已知样本和伪未知样本的 Energy 分数拉开间隔；通过 `--alpha-energy` 开启。
- SupCon 只对 batch 内确实存在同类正样本的 anchor 计算损失，避免单样本类别干扰训练。
- 新增 `--alpha-proxy`：参考 Proxy-NCA / Proxy Anchor / normalized softmax 思路，把分类器权重作为类别代理，让样本特征靠近自己的类代理并远离其他类代理。它不依赖 batch 内同类正样本，适合 CIFAR-100 这类类别多、batch 内正样本稀疏的快速实验。目前只完成 toy smoke test，尚未在 CIFAR 正式消融验证。

### Discovery pool 学习

- 支持双视图 cosine consistency。
- 支持带负样本的 NT-Xent 对比损失。
- `--discovery-pool-mode unknown` 使用纯未知池，适合上限或受控实验。
- `--discovery-pool-mode mixed` 使用已知类和未知类混合池，更接近真实开放环境。
- `discovery_unknown_loss` 只允许用于纯未知池，避免把“所有 discovery 样本都是未知”的假设错误用于混合数据。
- 支持 `--alpha-discovery-energy`：在纯未知 discovery pool 上施加 Energy 间隔约束，用于验证 Outlier Exposure 风格训练是否能改善未知检测。
- 支持 mixed discovery pool 的选择性未知约束：`--alpha-discovery-selective-unknown` / `--alpha-discovery-selective-energy` 会根据熵、MSP、Energy 或熵+辅助不确定性，从无标签池中选取 top-ratio 高风险样本，再只对这些样本施加未知约束。这样不要求 mixed pool 中所有样本都是未知类，更接近真实开放场景。
- 新增 `--discovery-select-mode consensus`：分别用熵、1-MSP、Energy 和辅助不确定性给 discovery 样本投票，只有多个风险信号同时认为样本像未知时才施加 selective unknown / energy 约束。它比单一 top-ratio 筛选更保守，目标是减少 mixed discovery pool 中已知样本被错当未知样本训练的问题。
- 新增 `--discovery-selection-model ema`：参考 Mean Teacher 和 FixMatch 的 teacher-student 稳定伪标签思想，维护一个学生模型的指数滑动平均副本，只用于 discovery 候选筛选。原有固定教师模型仍用于知识蒸馏，EMA 模型不参与反向传播，也不替代最终推理模型。
- 新增选择性 discovery 预热/渐进启用：`--discovery-selective-warmup-epochs` 会在前若干轮关闭选择性未知约束，`--discovery-selective-ramp-epochs` 会在预热后线性增加约束权重。这个设计参考半监督学习和广义新类发现中“先学稳定表征，再逐步信任伪标签/高风险样本”的训练思路，用来缓解过早把混合无标签样本当作未知而损伤已知分类的问题。

### 未知检测与聚类

- 开放集分数支持 MSP、ODIN-style MSP、Energy、predictive entropy、epistemic uncertainty、prototype distance、diagonal / shared-covariance Mahalanobis distance 及组合分数。
- ODIN-style MSP 使用温度缩放和输入微扰，在不重新训练模型的情况下作为未知检测强基线。
- MC Dropout 用于估计预测熵、数据不确定性代理和认知不确定性。
- 阈值默认只根据已知验证集分位数确定，不使用测试集未知标签。
- `discover --temperature-calibration` 支持在已知验证集上拟合温度，并输出 ECE、可靠性分箱、辅助不确定性与分类错误相关性到 `calibration_report.json`。
- 聚类支持 KMeans、Agglomerative 和 Spectral。
- 支持 projection / feature 特征、PCA、whitening 和 L2 归一化。
- auto-K 支持 silhouette、内部指标组合和稳定性分析。
- 结果中记录候选池纯度、误拒已知样本数量、真实 K、估计 K 以及 K 误差。
- 结果额外记录 `cluster_all_unknown_oracle_*` 和 `cluster_candidate_unknown_oracle_*`，用于区分“检测没筛出未知”和“真实未知特征本身不可聚类”。

## 当前阶段性结论

现有 CIFAR-100 60/40 实验表明：

- 普通 KD 相比 CE 通常只有小幅变化。
- 不确定性加权 KD 已经正确接入代码，但目前还没有通过多随机种子实验稳定证明它优于普通 KD。
- 特征蒸馏和完整表示学习可能改善特征结构，但不一定改善未知检测。
- 未知检测仍然是最大问题：AUROC 大约在 `0.59–0.61`，FPR95 大约在 `0.86–0.89`。
- 最新 Energy 消融中，伪未知 Energy 训练使 AUROC 从 `0.5975` 小幅升至 `0.6036`，AUPR 从 `0.4595` 升至 `0.4677`，但 FPR95、unknown reject rate 和已知分类准确率没有改善；因此只能说明方向有信号，不能说明已经解决未知检测问题。
- 5 epoch 快速对比中，纯未知 discovery pool + Energy 约束使 AUROC 从 `0.5259` 升至 `0.5800`，FPR95 从 `0.9125` 降至 `0.8897`，known accuracy 从 `0.2719` 升至 `0.3746`，unknown reject rate 从 `0.0456` 升至 `0.0714`，候选池纯度从 `0.3426` 升至 `0.5000`。这说明“真实无标签未知样本参与训练”比伪未知样本更值得继续验证。
- 新增 ODIN-style MSP 检测后，在同一 discovery-energy 快速模型、1000 张 CIFAR 测试子集上扫描 `epsilon`：`0.0002/0.0005` 的 FPR95 从 `0.9276` 降到 `0.8964`，unknown reject rate 从 `0.0561` 升到 `0.0791`，候选池纯度从 `0.4000` 升到 `0.4306`，但 AUROC 从 `0.5505` 降到约 `0.541`。`0.005` 的 AUROC 最高约 `0.5520`，但 FPR95 仍高达 `0.9227`。因此 ODIN 目前只能作为值得纳入的检测基线，还不能单独说明未知检测已经解决。
- 参考 Lee 等人的 Mahalanobis detector，代码新增了 shared-covariance Mahalanobis 分数；但在同一快速模型和 1000 张 CIFAR 测试子集上，`normalized_entropy_mahalanobis_shared` 的 AUROC 约 `0.5207`、FPR95 约 `0.9178`、unknown reject rate 约 `0.0536`，不如原 `normalized_entropy_mahalanobis` 和 ODIN。因此它目前只作为可选对比基线，不作为下一阶段主线。
- mixed discovery pool 3 epoch 快速消融中，选择性 Energy 相比 mixed NT-Xent baseline 的 FPR95 从 `0.9276` 降到 `0.9095`，候选池纯度从 `0.4590` 升到 `0.4727`；但 AUROC 从 `0.5693` 降到 `0.5545`，known accuracy 从 `0.3158` 降到 `0.2878`，unknown reject rate 从 `0.0714` 降到 `0.0663`。因此 selective energy 有部分信号，但当前权重 `0.1` 可能损伤已知分类，不能作为默认主方法。
- 针对上述问题，代码新增了 selective discovery warmup/ramp，并完成 3 epoch 快速消融。结果显示 warmup/ramp 版本 AUROC `0.5358`、FPR95 `0.9128`、known accuracy `0.2895`、unknown reject rate `0.0587`、candidate purity `0.4107`，没有优于 mixed baseline，也没有优于不加 warmup 的 selective energy。因此该策略目前只能作为可选稳定化机制，不能作为主改进结论。
- 新增更保守的 consensus 选择后，3 epoch 快速消融结果为 AUROC `0.5575`、FPR95 `0.9095`、known accuracy `0.3372`、unknown reject rate `0.0663`、candidate purity `0.5200`。相比 mixed baseline，它牺牲少量 AUROC，但提高了 known accuracy、降低了 FPR95，并把候选池纯度从 `0.4590` 提高到 `0.5200`；相比原 selective energy，它显著缓解了已知分类下降问题。因此 consensus 筛选比 warmup/ramp 更值得继续验证。
- 在 consensus 基础上加入 EMA selection model 后，3 epoch 快速消融结果为 AUROC `0.5712`、FPR95 `0.8832`、known accuracy `0.2928`、unknown reject rate `0.0561`、candidate purity `0.5500`。它在 AUROC、FPR95 和候选池纯度上是当前 mixed discovery 相关实验中最好的，但 known accuracy 和 unknown reject rate 下降，说明 EMA 筛选方向有价值，但 selective energy 权重或筛选比例还需要降低。
 - 在 consensus 基础上加入 EMA selection model 后，3 epoch 快速消融结果为 AUROC `0.5712`、FPR95 `0.8832`、known accuracy `0.2928`、unknown reject rate `0.0561`、candidate purity `0.5500`。它在 AUROC、FPR95 和候选池纯度上是当前 mixed discovery 相关实验中最好的，但 known accuracy 和 unknown reject rate 下降。
 - 后续参数搜索显示：`alpha=0.03, ratio=0.15, decay=0.99` 的 AUROC `0.5276`、candidate purity `0.3864`；`alpha=0.03, ratio=0.25, decay=0.99` 的 AUROC `0.5569`、candidate purity `0.3529`；`alpha=0.05, ratio=0.25, decay=0.95` 的 AUROC `0.5656`、FPR95 `0.8947`、candidate purity `0.4902`。因此简单降低 selective energy 权重或降低筛选比例没有解决问题，`decay=0.95` 有一定折中但不如原 EMA 0.99 的候选池纯度。
- 候选池纯度大约为 `0.43–0.50`，说明候选池中混入了较多被误拒的已知样本。
 - 候选池纯度大约为 `0.35–0.55`，不同筛选策略波动较大，说明候选池仍会混入较多被误拒的已知样本，且参数选择对聚类前候选质量影响明显。
- auto-K 在当前特征空间中可能估计为 `2`，而真实未知类别数为 `40`，说明自动类别数估计仍不可靠。
- 已吸收 `lky` 分支中较有价值的诊断功能：ImageFolder 双目录支持、温度校准 / ECE、uncertainty-error correlation、真实未知 oracle 聚类诊断和多 run 均值方差汇总；未合并其中会删除 Energy / discovery-energy 的回退改动。

这些结果只能作为阶段性实验记录，不能作为最终论文结论。正式结论应在固定协议下至少运行 3 个随机种子，并报告均值和标准差。

## 主要问题与处理方向

### 1. 未知检测能力不足

原因可能包括已知分类准确率不足、已知与未知分数分布重叠、不确定性没有校准、特征空间类间分离不足以及阈值策略较简单。

建议先固定训练协议，系统比较 MSP、ODIN、Energy、熵、原型距离和 Mahalanobis 分数，并检查温度校准、ECE、可靠性图和已知 / 未知分数分布。当前代码已经提供两类 Energy 训练：一类基于已知样本生成的伪未知样本，另一类基于纯未知 discovery pool 的 `discovery_energy`。前者已经验证只有小幅 AUROC/AUPR 提升，后者更接近 Outlier Exposure 和新类发现训练设定，需要继续跑对比实验确认是否真正提升未知拒绝率。本次新增的 ODIN 分数参考 Liang 等人的 ODIN 思路，通过温度缩放和输入微扰放大已知 / 未知置信度差异，适合作为当前未知检测短板的低成本强基线。

可参考 Liang 等人的 ODIN、Lee 等人的 Mahalanobis OOD Detection、Liu 等人的 Energy-based OOD Detection、Hendrycks 等人的 Outlier Exposure，以及 Vaze 等人的 Open-Set Recognition 工作。当前实验说明：单纯替换检测分数收益有限，后续更应优先改善特征空间和训练协议。

### 2. 候选池污染严重

当前聚类对象是“被检测为未知的所有样本”，其中包括误拒的已知样本。因此聚类指标低不一定完全是聚类算法的问题。

代码已经记录 candidate purity，并同时报告整个候选池和真实未知子集的聚类结果。后续应先提升候选池纯度，再解释聚类指标；同时固定报告 oracle-K 和 auto-K，不能只报告 oracle-K。

### 3. auto-K 估计不可靠

当前 `novel_discovery/pipeline.py` 中候选 K 的搜索上限是 20，而 CIFAR-100 实验有 40 个未知类。因此当前 auto-K 不可能估计出 40。

后续需要把最大 K 改成可配置参数，并在不使用未知标签的前提下比较 silhouette、CH、DB、稳定性和密度聚类方法。oracle-K 只能作为聚类能力上限，不能作为最终无监督结果。

### 4. 不确定性蒸馏尚未被充分验证

目前训练时使用辅助不确定性头的输出对 KL 和特征蒸馏进行加权；测试时还使用 MC Dropout 的预测熵和认知不确定性。两者是不同的不确定性信号，不能简单当作同一个量。

后续应在完全相同的设置下比较 CE、标准 KD、不确定性 KD，并比较 `raw` / `mean_normalized` 权重和 confidence / classification-error / margin 目标，至少运行 3 个随机种子。如果不确定性 KD 没有稳定提升，应如实记录为未验证，而不是强行宣称有效。

可参考 Hinton 等人的 Knowledge Distillation、Gal and Ghahramani 的 MC Dropout，以及 Kendall and Gal 关于不确定性分解的工作。

### 5. 特征空间仍不够适合开放集检测

当前检测端的 ODIN、Energy、Mahalanobis 等方法只能带来有限变化，说明已知/未知特征本身仍高度重叠。代码已经新增 proxy contrastive loss 作为特征改进方向：相比 SupCon，它不要求一个 batch 内出现同类正样本，而是使用类别代理提供稳定的类间负样本。

后续应在固定 CIFAR-100 协议下比较 `--alpha-proxy 0`、`0.05`、`0.1`、`0.2`，并同时观察 known accuracy、AUROC、FPR95、candidate purity。如果 proxy loss 只提升分类准确率但不提升未知检测，也应如实记录。

### 6. mixed discovery pool 不能直接全部当作未知

纯未知 discovery pool 上的 Energy 约束是受控上限实验，但真实无标签池可能同时包含已知和未知样本。代码现在增加了选择性未知训练：每个 discovery batch 根据当前模型的高熵 / 低置信度 / 高 Energy 样本筛选 top-ratio，只对筛出的候选施加 unknown 或 Energy 约束，并记录 `discovery_selected_ratio`。该方法参考 Outlier Exposure 的异常暴露思想、FixMatch/半监督学习的高置信筛选思想，以及 GCD 的无标签池建模思路，已通过 toy smoke test，并完成 CIFAR 快速消融；当前结果显示它能略降 FPR95、略提候选池纯度，但会降低已知分类准确率和 AUROC。

为处理“选择性样本早期不可靠”的问题，代码进一步加入了 warmup/ramp：训练前期先不启用 selective unknown/energy，等已知分类和特征空间相对稳定后再逐步增加权重。一次 CIFAR-100 快速消融表明它没有带来改善，说明当前主要瓶颈不只是“启用太早”，还包括选择规则本身不能稳定筛出未知样本。

当前更有价值的改动是 `--discovery-select-mode consensus`：它不只看一个分数，而是让熵、1-MSP、Energy 和辅助不确定性共同投票筛选疑似未知样本。快速实验中，consensus 的实际选择比例约 `0.17–0.18`，低于设定的 `0.25`，说明它确实更保守；最终 candidate purity 提升到 `0.5200`，known accuracy 提升到 `0.3372`。

进一步参考 Mean Teacher / FixMatch 的稳定教师思想后，代码新增 `--discovery-selection-model ema`。EMA+consensus 的实际选择比例约 `0.11–0.14`，候选池更小、更纯，candidate purity 提升到 `0.5500`，FPR95 降到 `0.8832`，AUROC 达到 `0.5712`；但 known accuracy 降到 `0.2928`，unknown reject rate 降到 `0.0561`。后续小网格实验表明，降低 `--alpha-discovery-selective-energy` 到 `0.03` 或把 `--discovery-select-ratio` 降到 `0.15` 并没有带来稳定提升；`--discovery-ema-decay 0.95` 可以把 known accuracy 略微拉回，但 candidate purity 降到 `0.4902`。因此下一步不应继续盲目调小权重，而应考虑更精细的动态权重、按置信度加权的 selective energy，或者加入 kNN 邻域一致性筛选。

### 7. 还不是完整端到端新类发现

目前 discovery pool 主要用于双视图表示学习，测试时仍然是检测后聚类。还没有把聚类伪标签、未知类分类头和周期性伪标签更新纳入统一训练。

后续可以参考 Deep Embedded Clustering、Deep Transfer Clustering 和 UNO 等方法，逐步加入高置信伪标签、聚类一致性损失和周期性更新，但必须先验证 discovery pool 本身确实改善检测和聚类，避免一次加入过多模块。

## 文献依据与当前实现边界

下面的文献用于确定实验设计和改进方向，不表示当前代码已经完整复现这些论文。

- Hinton et al., Distilling the Knowledge in a Neural Network (2015)：采用软化 logits、温度系数和 KL 散度作为标准知识蒸馏基线。当前代码的 distillation_loss 已实现标准 KL，并可用教师不确定性对每个样本的蒸馏权重进行调节。
- Gal and Ghahramani, Dropout as a Bayesian Approximation (ICML 2016)：使用 MC Dropout 的多次随机前向估计预测熵和认知不确定性。当前代码将它作为检测端的不确定性基线，但它和训练时的辅助不确定性头不是同一个信号，需要分开做消融。
- Kendall and Gal, What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? (NeurIPS 2017)：区分数据噪声导致的偶然不确定性和模型未知导致的认知不确定性。当前代码有辅助 uncertainty head 和 MC Dropout 估计，但还没有完成严格的偶然 / 认知不确定性分解实验。
- Liang et al., Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks (ODIN, ICLR 2018)：温度缩放加输入微扰可以作为低成本 OOD 检测基线。当前代码已支持 odin_msp；已有快速实验显示收益有限，因此它目前是对照方法，不是主改进方向。
- Lee et al., A Simple Unified Framework for Detecting OOD Samples and Adversarial Attacks (NeurIPS 2018)：使用类别条件高斯分布和 Mahalanobis 距离建模特征空间。当前代码实现 diagonal/shared covariance 两种形式，但 shared covariance 快速实验效果较差，后续重点仍应放在特征学习和协方差估计是否可靠。
- Hendrycks et al., Deep Anomaly Detection with Outlier Exposure (ICLR 2019)：通过辅助异常数据训练模型降低异常样本置信度。它支持纯未知 discovery pool 的 Energy 约束，但不能直接证明 mixed pool 的每个样本都是未知，因此当前代码只允许对纯 unknown 池使用全量 unknown/Energy 约束。
- Han et al., Learning to Discover Novel Visual Categories via Deep Transfer Clustering (ICCV 2019)：联合利用已知类监督信号和未知类聚类结构。它说明仅在最终阶段对候选特征做 KMeans 不足以完成新类发现，后续应考虑训练期的聚类一致性和伪标签更新。
- Van Gansbeke et al., SCAN (ECCV 2020)：利用近邻一致性约束学习类别结构。当前新增的 kNN 邻域筛选直接借鉴这一思路：样本本身被多个风险指标选中后，只有其邻域也有足够候选样本时才保留。
- Han et al., AutoNovel (ICLR 2020)：强调样本对关系、排序统计和新类结构，而不是只依赖单样本置信度。它提示当前的 consensus/EMA 仍只是候选筛选机制，后续可增加样本间关系学习。
- Sohn et al., FixMatch (NeurIPS 2020)：只对高置信伪标签使用无监督损失，避免把不可靠样本强行加入训练。当前代码的 selective unknown/energy、consensus 和 EMA selection model 均属于这一思想的开放集改写，但尚未实现按置信度连续加权的 loss。
- Fini et al., A Unified Objective for Novel Class Discovery (UNO, ICCV 2021)：将已知分类、未知类伪标签学习和类别均衡放在统一目标中。当前代码还没有 UNO 式未知分类头、均衡分配和周期性伪标签更新，这也是从两阶段流程走向统一新类发现的主要缺口。
- Caron et al., Emerging Properties in Self-Supervised Vision Transformers (DINO, ICCV 2021)：teacher-student 自蒸馏可以产生更有类别结构的表征。当前项目暂不替换 backbone，仅吸收其 teacher-student 稳定表征的思路，避免一次引入过大的结构变化。
- Vaze et al., Generalized Category Discovery (CVPR 2022)：无标签池可能同时含已知类和未知类，不能把 discovery pool 全部视为未知。它是当前 mixed pool、candidate purity、consensus 和 EMA 筛选设计的主要依据。
- Wen et al., SimGCD (ICCV 2023)：通过自蒸馏、伪标签和统一分类空间在训练阶段学习 novel structure。当前代码还未实现完整 SimGCD/UNO 目标，只把 discovery pool 的双视图学习作为过渡模块。

因此，当前方法的选择依据是：KL 蒸馏用于建立可解释的 KD 基线；Energy/MSP/ODIN/Mahalanobis 用于检测对比；consensus、EMA 和 kNN 用于降低 mixed pool 的候选污染；KMeans 主要作为简单可复现的聚类基线。选择这些技术是为了逐步验证单个假设，而不是声称它们天然优于所有替代方法。

## 本轮代码修改：kNN 邻域一致性筛选

本轮新增可选参数：

    --discovery-neighbor-filter
    --discovery-neighbor-k 5
    --discovery-neighbor-min-votes 2

启用后，流程变为：

    EMA 或 student 输出
        -> consensus 多指标初筛
        -> projection 特征上的 kNN 邻域一致性过滤
        -> selective unknown / selective energy

它针对的具体问题是：单个样本可能因分类器失误而被判为高风险，但真正的未知类通常会在特征空间形成局部结构。若一个初筛候选的近邻中几乎没有其他候选，则暂不把它用于 unknown/Energy 训练。该机制参考 SCAN 的近邻一致性和 AutoNovel 的样本关系建模，但目前只用于候选筛选，不使用标签，也不等于已经完成聚类伪标签学习。

训练日志新增：

- discovery_raw_selected_ratio：kNN 过滤前的候选比例；
- discovery_selected_ratio：过滤后的候选比例；
- discovery_neighbor_agreement：初筛候选的平均邻域候选比例。

该开关默认关闭，因此不改变已有实验。下一步应先用 toy 和小规模 CIFAR 做 smoke test，再比较“EMA + consensus”和“EMA + consensus + kNN”的 AUROC、FPR95、known accuracy、unknown reject rate 及 candidate purity。若 kNN 只减少候选数量却没有提高 purity，应调整 k / min-votes 或暂停该方向，而不能仅凭候选变少就判定有效。

本轮已完成 toy smoke test：1 epoch toy 教师训练、带 EMA + consensus + kNN 的学生训练、以及 discover 检测流程均可运行。学生训练日志中，discovery_raw_selected_ratio 为 0.15625，经过 kNN 过滤后的 discovery_selected_ratio 为 0.046875，说明新筛选逻辑确实生效；但 toy 数据和 1 epoch 训练不能证明指标提升，正式有效性仍需在 CIFAR-100 小规模对比和多 seed 实验中验证。

补充的小规模 CIFAR-100 对比实验设置为：known/novel = 60/40，seed = 42，limit-train = 1000，limit-val = 300，limit-test = 1000，limit-discovery = 1000，epochs = 2，teacher 使用已有 ResNet-34 checkpoint，student 为 ResNet-18，检测分数为 normalized_entropy_mahalanobis_diag，聚类特征为 projection_pca + L2 normalize。结果如下：

| 方法 | AUROC | FPR95 | known acc all known | unknown reject | candidate count | candidate purity | auto-K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EMA + consensus | 0.4854 | 0.9490 | 0.0954 | 0.0434 | 55 | 0.3091 | 20 |
| EMA + consensus + kNN | 0.4932 | 0.9441 | 0.0855 | 0.0306 | 39 | 0.3077 | 17 |

这次快速实验说明：kNN 确实让训练期 selective 样本更少，第二轮训练中 discovery_raw_selected_ratio 约为 0.115，过滤后 discovery_selected_ratio 约为 0.018；测试时候选池也从 55 个降到 39 个。但是它没有提升 candidate purity，unknown reject rate 和 known accuracy 反而下降。因此当前 kNN 版本只能说明“更保守”，不能说明“更有效”。后续若继续尝试，应优先调整 k / min-votes 或改成按邻域一致性连续加权的 selective energy；如果多组设置仍不能提升 purity，就应暂停 kNN 作为主线。

## 下一步代码分工建议

下面的分工只使用“同学 1 / 同学 2”表示，目的是让两个人都能改代码，但尽量不修改同一批文件，减少冲突。这个分工直接对应当前实验暴露出来的问题：未知检测仍然较弱、候选池污染较重、kNN hard filter 只让候选变少但没有提高 candidate purity、不确定性知识蒸馏还没有被充分验证。

### 同学 1：candidate weighting 与 discovery 筛选

负责问题：

- mixed discovery pool 中不能把所有样本都当未知；
- 当前 consensus / EMA / kNN 仍然会选入较多误拒的已知样本；
- 最新 kNN hard filter 实验中，候选数从 55 降到 39，但 candidate purity 从 0.3091 变为 0.3077，说明“硬过滤”没有真正改善候选质量；
- 下一步应从“选中 / 不选中”的二值判断，改为“不同候选样本具有不同可信权重”的 soft weighting。

建议实现方向：

- 对 discovery sample 先计算 entropy / MSP / Energy / uncertainty 等风险信号；
- 使用 consensus 得到初筛风险分数；
- 使用 kNN 邻域一致性判断该样本附近是否也有疑似未知候选；
- 使用 EMA/student 预测一致性判断当前模型判断是否稳定；
- 最终输出 candidate_weight，范围建议控制在 0 到 1。

建议权重来源：

- consensus_score：熵、1-MSP、Energy、辅助不确定性等风险信号越一致，权重越高；
- neighbor_agreement：近邻中也被认为像未知的样本越多，权重越高；
- ema_student_agreement：EMA 模型和当前 student 对样本风险判断越一致，权重越高。

可参考文献与对应思路：

- Vaze et al., Generalized Category Discovery, CVPR 2022：无标签池同时包含已知类和未知类，不能把 discovery pool 全部当未知；当前 mixed pool 和 candidate purity 记录就是基于这个问题。
- Sohn et al., FixMatch, NeurIPS 2020：只使用高置信伪标签，不可靠样本不应强行训练；这里对应“candidate_weight 低的样本减少训练强度”。
- Tarvainen and Valpola, Mean Teachers are Better Role Models, NeurIPS 2017：EMA teacher 的预测更稳定；这里对应 discovery selection model = ema 和 EMA/student agreement。
- Van Gansbeke et al., SCAN, ECCV 2020：类别结构可以通过近邻一致性学习；这里对应 kNN neighbor agreement。
- Han et al., AutoNovel, ICLR 2020：新类发现不能只看单样本置信度，还要利用样本间关系；这里对应“样本自己像未知，并且邻居也像未知”。
- Wen et al., SimGCD, ICCV 2023：训练阶段应利用无标签样本结构，而不是只在测试后聚类；这里支持把 candidate weighting 放到训练阶段。

建议主要修改文件：

- novel_discovery/pipeline.py
- train.py
- 可新增 tests/test_discovery_selection.py

尽量不要修改：

- novel_discovery/losses.py
- README.md 的实验结论部分

建议测试内容：

- 未被初筛选中的样本权重为 0；
- 邻域一致性越高，权重越高；
- EMA/student 判断越一致，权重越高；
- k 大于 batch size 时不报错；
- 权重不出现 NaN，范围保持在 0 到 1。

### 同学 2：weighted loss 与不确定性蒸馏

负责问题：

- 当前 selective energy 对所有被选中的候选样本几乎一视同仁；
- 但候选池里仍混有误拒的已知样本，如果它们以同等强度参与 unknown / energy 训练，会继续污染模型；
- 当前不确定性知识蒸馏已经接入代码，但还没有通过严格消融证明优于标准 KD；
- 下一步需要让 loss 支持 per-sample weight，方便同学 1 输出的 candidate_weight 真正进入训练目标。

建议实现方向：

- teacher uncertainty 用于 KD weight，再进入 uncertainty-weighted KL / feature KD；
- candidate_weight 用于 weighted selective energy；
- 更可靠的候选样本权重大，不可靠候选样本权重小；
- 新增 per-sample energy margin loss 和 weighted energy margin loss，同时保持旧的 energy_margin_loss 接口不被破坏。

建议同时整理不确定性蒸馏权重：

- raw：保持当前 exp(-uncertainty) 行为；
- mean_normalized：让 batch 平均权重接近 1，避免只是整体改变 KD 强度；
- clamp：限制最小 / 最大权重，避免少数样本权重过大或过小；
- 后续消融比较 CE、标准 KD、不确定性 KD、特征 KD 和完整方法。

可参考文献与对应思路：

- Hinton et al., Distilling the Knowledge in a Neural Network, 2015：标准 KD 使用温度 soft logits 和 KL 散度；当前 KL 蒸馏基线来自这里。
- Kendall and Gal, What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?, NeurIPS 2017：区分偶然不确定性和认知不确定性；当前项目需要明确不同不确定性信号在训练和检测中的作用。
- Gal and Ghahramani, Dropout as a Bayesian Approximation, ICML 2016：MC Dropout 可估计模型不确定性；当前检测端的 epistemic uncertainty 和 predictive entropy 可参考该思路。
- Liu et al., Energy-based Out-of-distribution Detection, NeurIPS 2020：Energy 分数可用于 OOD 检测和训练约束；当前 selective energy 和 weighted energy loss 主要参考这个方向。
- Hendrycks et al., Deep Anomaly Detection with Outlier Exposure, ICLR 2019：辅助异常样本可以降低模型对异常输入的置信度；但 mixed pool 不保证全是异常，因此需要 weighted selective loss，而不是全量 OE。
- Sohn et al., FixMatch, NeurIPS 2020：伪标签样本应该按可靠程度筛选或加权；这里对应对 candidate_weight 使用不同 loss 强度。

建议主要修改文件：

- novel_discovery/losses.py
- 可新增 novel_discovery/uncertainty_kd.py
- 可新增 tests/test_uncertainty_kd.py

尽量不要修改：

- novel_discovery/pipeline.py 的训练主循环
- train.py 的参数入口
- README.md 的实验结论部分

建议测试内容：

- mean_normalized 权重平均值接近 1；
- clamp 后权重在指定范围内；
- weighted energy loss 在空 outlier 输入时返回 0；
- weighted energy loss 可以正常反向传播；
- 样本权重越高，对最终 loss 的影响越大；
- 旧的 energy_margin_loss 和 distillation_loss 原接口仍然可用。

### 两部分如何衔接

同学 1 输出 candidate_weight；同学 2 提供 weighted_energy_margin_loss。最终主流程可以变成：mixed discovery sample 先根据风险、邻域和 EMA/student 一致性计算 candidate_weight，再根据 candidate_weight 计算 weighted selective energy，最后训练学生模型。

这个衔接正好对应当前最大问题链条：未知检测不可靠导致候选池污染，错误候选又以同等权重参与训练，进一步损伤已知分类和未知检测。因此，下一步不是继续单纯调小 alpha 或 ratio，而是把“候选是否可靠”和“候选以多大强度参与训练”拆开处理。

## 推荐实验协议

1. 固定 CIFAR-100 60/40 划分、backbone、输入尺寸、epoch、batch size、优化器、阈值策略和评分方式。
2. 使用同一个教师模型比较 CE、标准 KD、不确定性 KD、特征 KD 和完整模型。
3. 先进行单模块消融，再测试 discovery pool；不要同时改变多个模块。
4. 每个主要实验至少运行 3 个 seed，报告 mean ± std。
5. 检测报告 AUROC、AUPR、FPR95、OSCR、known accuracy、known accept rate 和 unknown reject rate。
6. 聚类同时报告候选池纯度、candidate pool 全体聚类结果、真实未知子集结果、oracle-K 和 auto-K。
7. 记录模型参数量、推理时间；比较 ResNet-18 与 MobileNetV3-Small 时保持训练和检测协议一致。

## 运行方法

安装依赖：

```bash
pip install -r requirements.txt
```

检查数据划分：

```powershell
python train.py inspect_data --dataset cifar100 --data-root .\data --download `
  --num-known 60 --seed 42 --split-path .\splits_cifar100_60_40.json
```

如果使用本地 ImageFolder 版 CIFAR-100，可采用如下结构：

```text
CIFAR-100-dataset-main/
  train/
  test/
```

检查 ImageFolder 双目录划分：

```powershell
python train.py inspect_data --dataset imagefolder `
  --data-root .\CIFAR-100-dataset-main `
  --num-known 60 --seed 42 `
  --split-path .\splits_cifar100_60_40.json `
  --image-size 64 --num-workers 0
```

训练教师模型：

```powershell
python train.py train_teacher --dataset cifar100 --data-root .\data `
  --num-known 60 --seed 42 --split-path .\splits_cifar100_60_40.json `
  --backbone resnet34 --pretrained --image-size 64 --batch-size 64 `
  --epochs 15 --work-dir .\runs\teacher `
  --teacher-ckpt .\runs\teacher\teacher.pt --device auto
```

训练学生模型：

```powershell
python train.py train_student --dataset cifar100 --data-root .\data `
  --num-known 60 --seed 42 --split-path .\splits_cifar100_60_40.json `
  --backbone resnet18 --teacher-backbone resnet34 --student-backbone resnet18 `
  --pretrained --image-size 64 --batch-size 64 --epochs 15 `
  --teacher-ckpt .\runs\teacher\teacher.pt `
  --student-ckpt .\runs\student\student.pt `
  --work-dir .\runs\student --device auto
```

在已知数据生成的伪未知样本上启用 Energy 分离损失：

```powershell
python train.py train_student --dataset cifar100 --data-root .\data `
  --num-known 60 --seed 42 --split-path .\splits_cifar100_60_40.json `
  --backbone resnet18 --teacher-backbone resnet34 --student-backbone resnet18 `
  --pretrained --alpha-energy 0.1 --energy-margin 1.0 `
  --teacher-ckpt .\runs\teacher\teacher.pt `
  --student-ckpt .\runs\student_energy\student.pt `
  --work-dir .\runs\student_energy --device auto
```

`--alpha-energy 0` 时保持原有训练行为。该版本使用的是伪未知样本，后续还需要增加真正的辅助异常数据协议，才能严格复现 Outlier Exposure。

启用 proxy contrastive loss 改善已知类特征结构：

```powershell
python train.py train_student --dataset cifar100 --data-root .\data `
  --num-known 60 --seed 42 --split-path .\splits_cifar100_60_40.json `
  --backbone resnet18 --teacher-backbone resnet34 --student-backbone resnet18 `
  --pretrained --alpha-proxy 0.1 --proxy-temperature 0.1 `
  --teacher-ckpt .\runs\teacher\teacher.pt `
  --student-ckpt .\runs\student_proxy\student.pt `
  --work-dir .\runs\student_proxy --device auto
```

`--alpha-proxy 0` 时保持原有训练行为。该功能目前只是可选特征学习模块，需要和 CE / KD / SupCon 在同一协议下做消融对比后才能判断是否有效。

启用 discovery pool 的 NT-Xent 表示学习：

```powershell
python train.py train_student --dataset cifar100 --data-root .\data `
  --num-known 60 --seed 42 --split-path .\splits_cifar100_60_40.json `
  --backbone resnet18 --teacher-backbone resnet34 --student-backbone resnet18 `
  --pretrained --discovery-pool --discovery-pool-mode mixed `
  --limit-discovery 5000 --alpha-discovery 0.1 --discovery-loss nt_xent `
  --teacher-ckpt .\runs\teacher\teacher.pt `
  --student-ckpt .\runs\student_discovery\student.pt `
  --work-dir .\runs\student_discovery --device auto
```

注意：混合 discovery pool 只能使用双视图一致性或 NT-Xent，不能启用 `--alpha-discovery-unknown` 或 `--alpha-discovery-energy`。纯未知池可以用于受控上限实验，但不能直接代表完全真实的无标签场景。

在 mixed discovery pool 上启用选择性未知 Energy 约束：

```powershell
python train.py train_student --dataset cifar100 --data-root .\data `
  --num-known 60 --seed 42 --split-path .\splits_cifar100_60_40.json `
  --backbone resnet18 --teacher-backbone resnet34 --student-backbone resnet18 `
  --pretrained --discovery-pool --discovery-pool-mode mixed `
  --limit-discovery 4000 --alpha-discovery 0.05 `
  --alpha-discovery-selective-energy 0.1 `
  --discovery-selective-warmup-epochs 1 `
  --discovery-selective-ramp-epochs 2 `
  --discovery-select-ratio 0.25 --discovery-select-mode entropy_uncertainty `
  --teacher-ckpt .\runs\teacher\teacher.pt `
  --student-ckpt .\runs\student_selective_discovery\student.pt `
  --work-dir .\runs\student_selective_discovery --device auto
```

选择性版本不会把 mixed discovery pool 的全部样本都当作未知，只会对当前模型认为最像未知的一部分样本施加约束。`warmup=1, ramp=2` 表示第 1 轮不使用 selective loss，第 2、3 轮逐步增加到完整权重。正式实验时需要和只使用 `--alpha-discovery` 的 mixed baseline，以及不加 warmup/ramp 的 selective 版本对比。

如果重点是提高候选池纯度，可以使用 consensus 筛选：

```powershell
python train.py train_student --dataset cifar100 --data-root .\data `
  --num-known 60 --seed 42 --split-path .\splits_cifar100_60_40.json `
  --backbone resnet18 --teacher-backbone resnet34 --student-backbone resnet18 `
  --pretrained --discovery-pool --discovery-pool-mode mixed `
  --limit-discovery 4000 --alpha-discovery 0.05 `
  --alpha-discovery-selective-energy 0.05 `
  --discovery-select-ratio 0.25 --discovery-select-mode consensus `
  --teacher-ckpt .\runs\teacher\teacher.pt `
  --student-ckpt .\runs\student_consensus\student.pt `
  --work-dir .\runs\student_consensus --device auto
```

consensus 不保证固定选取 ratio 比例的样本；它会根据多个风险信号的一致程度实际选取更少的候选，因此更适合优先控制候选池污染。

如果希望用更稳定的 EMA 学生模型做候选筛选，可以加入：

```powershell
python train.py train_student --dataset cifar100 --data-root .\data `
  --num-known 60 --seed 42 --split-path .\splits_cifar100_60_40.json `
  --backbone resnet18 --teacher-backbone resnet34 --student-backbone resnet18 `
  --pretrained --discovery-pool --discovery-pool-mode mixed `
  --limit-discovery 4000 --alpha-discovery 0.05 `
  --alpha-discovery-selective-energy 0.05 `
  --discovery-select-ratio 0.25 --discovery-select-mode consensus `
  --discovery-selection-model ema --discovery-ema-decay 0.99 `
  --teacher-ckpt .\runs\teacher\teacher.pt `
  --student-ckpt .\runs\student_ema_consensus\student.pt `
  --work-dir .\runs\student_ema_consensus --device auto
```

EMA+consensus 目前更适合提高候选池纯度和降低 FPR95，但会损伤已知分类准确率；正式实验应优先尝试更小的 selective energy 权重。

在纯未知 discovery pool 上启用 Energy 分离约束：

```powershell
python train.py train_student --dataset cifar100 --data-root .\data `
  --num-known 60 --seed 42 --split-path .\splits_cifar100_60_40.json `
  --backbone resnet18 --teacher-backbone resnet34 --student-backbone resnet18 `
  --pretrained --discovery-pool --discovery-pool-mode unknown `
  --limit-discovery 4000 --alpha-discovery 0.05 `
  --alpha-discovery-energy 0.1 --discovery-loss nt_xent `
  --teacher-ckpt .\runs\teacher\teacher.pt `
  --student-ckpt .\runs\student_discovery_energy\student.pt `
  --work-dir .\runs\student_discovery_energy --device auto
```

运行未知检测和 auto-K 聚类：

```powershell
python train.py discover --dataset cifar100 --data-root .\data `
  --num-known 60 --num-novel 40 --seed 42 `
  --split-path .\splits_cifar100_60_40.json `
  --backbone resnet18 --student-backbone resnet18 --pretrained `
  --student-ckpt .\runs\student\student.pt `
  --work-dir .\runs\discover_auto `
  --score-mode normalized_entropy_mahalanobis `
  --temperature-calibration `
  --cluster-k auto --cluster-method kmeans `
  --cluster-feature projection_pca --cluster-selection composite `
  --cluster-pca-dim 32 --cluster-normalize `
  --mc-samples 8 --device auto
```

运行 ODIN-style MSP 检测基线：

```powershell
python train.py discover --dataset cifar100 --data-root .\data `
  --num-known 60 --num-novel 40 --seed 42 `
  --split-path .\splits_cifar100_60_40.json `
  --backbone resnet18 --student-backbone resnet18 --pretrained `
  --student-ckpt .\runs\student_discovery_energy\student.pt `
  --work-dir .\runs\discover_odin `
  --score-mode odin_msp --odin-epsilon 0.0005 --odin-temperature 1000 `
  --cluster-k auto --cluster-method kmeans `
  --cluster-feature projection_pca --cluster-selection composite `
  --cluster-pca-dim 32 --cluster-normalize `
  --mc-samples 8 --device auto
```

运行核心测试：

```bash
python -m unittest discover -s tests
```

运行已有对比脚本：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_revised_ablation.ps1
```

比较是否加入 Energy 分离损失：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_energy_compare.ps1
```

该脚本只改变 `--alpha-energy`，并复用同一个教师模型。正式实验前应确认教师权重已经存在；首次测试可以把脚本中的 epoch 和数据量改小，正式结果再恢复完整设置。

比较是否加入纯未知 discovery pool 的 Energy 约束：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_discovery_energy_compare.ps1
```

汇总多个 run 的均值和标准差：

```powershell
python aggregate_multiseed.py --root .\runs `
  --runs run_seed42_detect run_seed43_detect run_seed44_detect `
  --out .\analysis\multiseed_summary.json
```

## 文件说明

- `train.py`：训练、未知检测和聚类入口。
- `novel_discovery/data.py`：数据集、类别划分和 discovery pool。
- `novel_discovery/models.py`：教师 / 学生模型和 MC Dropout 推理。
- `novel_discovery/losses.py`：分类、蒸馏、不确定性、对比和原型损失。
- `novel_discovery/pipeline.py`：训练流程、开放集打分、阈值和聚类。
- `novel_discovery/metrics.py`：AUROC、AUPR、FPR95、OSCR 和聚类指标。
- `analyze_results.py`：多组实验结果和误差分析。
- `aggregate_multiseed.py`：把多个 `discovery_report.json` 汇总为 mean/std。
- `tests/`：核心行为测试。
- `scripts/`：消融、对比和多配置实验脚本。
- `analysis/`：阶段性实验结果和分析文档。
- `docs/`：方法说明、实验计划和组内进度。

`data/`、`runs/`、`.torch_cache/` 和模型权重不提交到仓库。正式结果应保留对应的 `config.json`、模型配置、随机种子和报告文件。

## 结果解释原则

- toy 数据和带有 `limit-*` 参数的实验只用于检查代码是否可运行。
- oracle-K 只用于聚类上限分析。
- 使用测试集未知标签选择分数、阈值或聚类配置的结果不能作为严格无监督结果。
- 目前结果属于阶段性结果，后续必须补充多随机种子实验和统一协议后，才能形成论文结论。
