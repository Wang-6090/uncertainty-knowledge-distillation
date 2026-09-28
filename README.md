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
- `--open-val-ratio` 会从训练划分中单独保留已知/未知 open-validation 样本，用于选择 OOD 分数；最终测试集不参与选择。
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

### 联合新类发现（实验版）

- 新增 NovelPrototypeHead：在学生模型特征上学习一组未知类原型，不改变原有已知分类头。
- 新增 UNO 风格 balanced assignment，缓解所有未知样本塌缩到少数 prototype 的问题。
- 新增 SimGCD 风格双视图 novel consistency：用一个增强视图生成伪标签，约束另一个视图的 novel prototype 预测。
- 新增 SCAN 风格 neighbor consistency：约束特征空间近邻具有相似的 novel prototype 分布。
- 通过 --joint-discovery 显式开启，默认关闭；训练后额外保存 novel_head.pt。
- `--joint-head-temperature` 与 `--joint-assignment-temperature` 分开：前者控制 cosine prototype logits，默认 `0.2`；后者控制均衡分配，默认 `1.0`。
- --joint-confidence-threshold 使用相对均匀分布的置信度，即 max softmax probability 乘以 novel 类数量。默认值 1.1，适用于 CIFAR-100 的 40 个未知类。
- `discover` 可通过 `--novel-head-ckpt` 加载 `novel_head.pt`，使用 `--score-mode novel_msp` / `novel_entropy` 做检测，或使用 `--cluster-feature novel` / `novel_pca` 评估 prototype 表示。
- 新增可选 `--joint-space unified`：把已知分类 logits 与 novel prototype logits 拼成统一的 known+novel 空间，并支持 `unified_novel_mass` 检测分数和 `unified` / `unified_pca` 聚类特征。

这一版是从两阶段“检测后聚类”走向联合新类发现的最小实验模块，还没有实现完整 UNO/SimGCD 的周期性伪标签更新、类别均衡分配优化和未知类分类评测。它目前用于验证训练期 novel prototype 是否能学习结构，不应直接作为最终论文方法。

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

- 新增可选 soft candidate weighting：--discovery-soft-weighting 不再让所有初筛候选以同等强度参与 selective unknown / Energy loss，而是结合风险强度、kNN 邻域一致性和 EMA/student 一致性生成连续权重；默认关闭，以保持旧实验可复现。
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
 - 联合新类发现模块已经完成 toy smoke test 和小规模 CIFAR-100 GPU smoke test。CIFAR smoke 设置为 known/novel = 60/40、训练样本 1000、discovery 样本 1000、1 epoch，训练日志中 `joint_discovery=3.6891`、`joint_consistency=3.6890`、`joint_balance=0.00017`、`joint_neighbor=0.000095`，并成功保存 `novel_head.pt`。这只能证明联合损失确实参与了反向传播且代码可运行，不能证明未知检测或聚类性能已经提升。
 - 在纯未知 discovery pool、2 epoch、prototype temperature `0.2` 的快速实验中，`joint_information` 从第 1 轮约 `-0.0054` 变为第 2 轮约 `-0.0428`，说明 prototype 分配开始脱离均匀状态。使用 `novel_msp` 检测时 AUROC `0.5127`、unknown reject rate `0.0689`、candidate purity `0.4737`、candidate ARI `0.0697`；使用 `novel_entropy` 时 AUROC `0.5130`、FPR95 `0.9490`、unknown-only ARI `0.3268`。相对同规模 baseline 的 AUROC `0.4685`、unknown reject rate `0.0255`、candidate purity `0.2632`，有初步正向信号，但 FPR95 仍很差，且实验只有单 seed、2 epoch，不能作为最终结论。
 - 为控制训练轮数和 discovery pool 的影响，补充了同样为纯未知池、2 epoch 的 baseline：AUROC `0.4946`、FPR95 `0.9490`、unknown reject rate `0.0332`、candidate purity `0.3171`。因此在更公平的对照下，novel head 的 `novel_msp` / `novel_entropy` 仍有初步收益，但提升幅度有限，必须进行多 seed 和更长训练验证。
 - 目前 mixed discovery pool 上的联合 head 没有表现出同样收益，原因是当前 head 只建模 40 个未知 prototype，却把已知和未知混合样本全部送入该 head；这与 GCD/SimGCD 的统一 known+novel 类别空间设定不完全一致。后续应优先改成统一类别空间或加入可靠的未知候选门控，再做正式对比。
- 已实现统一 known+novel 空间并完成 smoke test。在 mixed discovery pool、2 epoch、60/40 划分下，`unified_novel_mass` 的 AUROC `0.5378`、FPR95 `0.9260`、unknown reject rate `0.0663`、candidate purity `0.4643`、candidate ARI `-0.0118`。相同思路在纯未知 pool 上 AUROC 仅 `0.4874`，说明统一空间更适合 mixed unlabeled pool；但 ARI 仍接近 0，prototype 尚未稳定对应真实新类，不能作为最终方法结论。
 - 新增 joint candidate gating，支持 EMA + consensus 的 hard gate 和 EMA + entropy/uncertainty 的 soft weighting。mixed pool、2 epoch 快速结果中，hard gate 为 AUROC `0.5180`、FPR95 `0.9293`、unknown reject `0.0255`、candidate purity `0.4000`；soft gate 为 AUROC `0.5157`、FPR95 `0.9424`、unknown reject `0.0816`、candidate purity `0.3902`。两者都没有超过无门控统一空间，说明当前风险分数与 novel structure 的对应关系仍弱，门控暂不作为主方法。
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

当前 auto-K 的搜索上限已经改为命令行参数 `--max-auto-clusters`，默认值为 50，因此在 CIFAR-100 的 40 个未知类设置下，搜索范围不再被固定的 20 截断。代码和测试已验证搜索可以超过 20。

但“搜索范围允许到 40”不等于模型能够正确估计出 40。当前 auto-K 仍只依据候选池几何结构选择 K，且候选池本身可能混入误拒的已知样本；因此仍需在不使用未知标签的前提下比较 silhouette、CH、DB、稳定性和密度聚类方法。oracle-K 只能作为聚类能力上限，不能作为最终无监督结果。正式实验必须同时报告 `--cluster-k oracle` 和 `--cluster-k auto`，并记录估计 K。

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

soft candidate weighting 已进一步实现为可选训练机制。它参考 FixMatch 的置信度加权、Mean Teacher 的稳定 teacher/student 预测、SCAN 的近邻一致性和 AutoNovel 的样本关系思想：初筛候选仍由风险信号决定，但 candidate_weight 会随风险、邻域一致性和 EMA/student 一致性连续变化；同时新增 weighted Energy-margin loss 和 weighted discovery unknown loss。CIFAR-100 快速实验表明它能减少候选污染，但检测排序指标没有同步提升，说明当前权重公式可能过于保守，后续需要调节邻域平滑温度或加入权重下限，不能直接作为默认方法。

soft candidate weighting 的小规模 CIFAR-100 结果为：candidate purity 从 0.3091 提升到 0.3913，unknown reject rate 从 0.0434 提升到 0.0459，但 AUROC 从 0.4854 降到 0.4805，FPR95 从 0.9490 变差到 0.9589。该结果只能说明候选池纯度有所改善，不能说明开放集检测整体改善。

## 当前三人并行算法分工

三个人同时做算法，但分别负责不同的机制，避免直接修改同一核心文件。每个人从最新 `main` 创建自己的分支，完成单元测试和轻量 smoke test 后再合并。

| 角色 | 算法方向 | 主要文件范围 | 直接对应的问题 |
| --- | --- | --- | --- |
| 负责人 | 联合新类发现与原型分配 | `novel_discovery/joint_discovery.py`、可新增 `novel_head.py`、`train.py` 的接入、独立测试 | 目前主要是检测后聚类，未知类结构没有在训练期学习 |
| 同学 1 | 不确定性感知知识蒸馏 | `novel_discovery/losses.py`、可新增 `uncertainty_kd.py`、独立测试 | 不确定性 KD 尚未证明稳定优于普通 KD，蒸馏权重还需校准 |
| 同学 2 | 未知检测与候选筛选/聚类 | 可新增 `novel_discovery/discovery_selection.py`、检测和聚类评测脚本、独立测试 | AUROC/FPR95 较差，候选池污染严重，auto-K 不可靠 |

### 负责人：联合新类发现与原型分配

负责人负责把未知类结构学习真正放进训练目标，而不是只做最后的 KMeans：

- 维护 `NovelPrototypeHead`、balanced assignment、双视图一致性和近邻一致性；
- 参考 UNO 的类别均衡分配、SimGCD 的自蒸馏、SCAN 的近邻一致性，逐步加入周期性伪标签更新和 prototype 稳定化；
- 通过 `train.py` 接入已知分类、蒸馏与 novel discovery loss，并保留 `--joint-discovery` 作为可关闭开关；
- 训练阶段只能使用 train/discovery pool，不能使用测试集未知标签；
- 对比“检测后 KMeans”与“训练期联合 novel head”，报告 AUROC、FPR95、known accuracy、candidate purity、NMI 和 ARI。

参考方法：Fini et al., UNO (ICCV 2021) 的 balanced assignment；Wen et al., SimGCD (ICCV 2023) 的 self-distillation 与统一分类空间；Van Gansbeke et al., SCAN (ECCV 2020) 的 nearest-neighbor consistency；Han et al., Deep Transfer Clustering (ICCV 2019) 的监督特征迁移；Vaze et al., GCD (CVPR 2022) 的 known/novel 混合评测协议。

当前已完成最小版本：`NovelPrototypeHead`、UNO 风格近似均衡分配、SimGCD 风格双视图 consistency、SCAN 风格 neighbor consistency，并已通过 toy 与小规模 CIFAR smoke test。尚未完成完整 UNO/SimGCD 的周期性伪标签更新和严格的 novel head 评测。

### 同学 1：不确定性感知知识蒸馏

- 在 `losses.py` 或独立 `uncertainty_kd.py` 中实现并比较标准 KL、教师不确定性加权 KL、特征蒸馏和不确定性校准；
- 参考 Hinton et al. 的温度 KL 蒸馏、Kendall and Gal 的 aleatoric/epistemic 不确定性区分、Gal and Ghahramani 的 MC Dropout、Liu et al. 的 Energy 分数；
- 检查不确定性权重的范围、归一化、截断、空 batch 和梯度传播，不改变未知检测和聚类主流程；
- 通过 CE、普通 KD、不确定性 KD 三组消融，报告 known accuracy、ECE、AUROC、FPR95 和多 seed 均值。

建议分支：`lky-uncertainty-losses`。只提交损失函数、独立模块和对应测试，不修改 `joint_discovery.py`。

### 同学 2：未知检测与候选筛选/聚类

- 新增独立的 `discovery_selection.py` 或评测脚本，改进 consensus、EMA、kNN、soft weighting 和 auto-K；
- 参考 Vaze et al. 的 GCD 混合无标签池设定、Tarvainen and Valpola 的 Mean Teacher、Sohn et al. 的 FixMatch 置信度筛选、SCAN 的邻域一致性、AutoNovel 的样本关系建模；
- 重点比较 hard filter 与 soft weight，分析候选池纯度、unknown reject rate、AUROC、FPR95、silhouette、NMI 和 ARI；
- 检查空候选、小 batch、`k > batch size` 和 auto-K 边界，默认参数保持当前行为，所有新策略都用显式开关开启；
- 不修改 `losses.py` 和 `joint_discovery.py`，如果需要训练接入，先提交独立接口，由负责人统一合并。

建议分支：`qiyuhan-discovery-selection`。只提交候选筛选、检测/聚类评测和对应测试。

### 并行协作规则

- 三个人都从最新 `main` 建分支，不提交数据集、`.pt` 权重和大型运行目录；
- 负责人拥有 `train.py` 的最终接入权；同学 1 不改 `joint_discovery.py`，同学 2 不改 `losses.py` 或 `joint_discovery.py`；
- 每个方向先提供独立函数和测试，再做主流程接入；
- 合并后统一运行 `compileall`、全量单元测试和固定协议的多 seed 消融，不能仅凭单次 smoke test 宣称有效。

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

启用负责人实现的联合新类发现实验版：

```powershell
python train.py train_student --dataset cifar100 --data-root .\data `
  --num-known 60 --seed 42 --split-path .\splits_cifar100_60_40.json `
  --backbone resnet18 --teacher-backbone resnet34 --student-backbone resnet18 `
  --pretrained --discovery-pool --discovery-pool-mode mixed `
  --limit-discovery 1000 --joint-discovery --joint-num-novel 40 `
  --joint-head-temperature 0.2 `
  --alpha-joint-discovery 0.2 --joint-confidence-threshold 1.0 `
  --alpha-joint-consistency 1.0 --alpha-joint-balance 0.1 `
  --alpha-joint-neighbor 0.1 --joint-neighbor-k 5 `
  --teacher-ckpt .\runs\teacher\teacher.pt `
  --student-ckpt .\runs\student_joint\student.pt `
  --work-dir .\runs\student_joint --device auto
```

该命令必须带 `--discovery-pool`，因为 novel prototype 只在训练期的无标签 discovery pool 上学习。`--joint-confidence-threshold` 是相对均匀分布的置信度阈值，不是普通的最大 softmax 概率；当前模块只用于验证训练期 novel prototype 是否能学习结构，不能替代最终的检测和聚类评测。

如果使用 GCD/SimGCD 风格的混合无标签池，可将训练命令中的联合空间改为：

```text
--discovery-pool-mode mixed --joint-space unified
```

该模式同时学习已知和未知的统一 logits，但仍需通过消融实验确认它是否优于只学习 novel prototype 的模式。

候选门控为实验开关，当前不建议直接作为默认方法：

```text
--joint-candidate-gating
--joint-candidate-mode entropy_uncertainty
--joint-candidate-soft-weighting
--joint-candidate-weight-floor 0.05
--discovery-selection-model ema
```

hard gate 会丢弃非候选样本，soft weighting 会保留全部样本但按风险连续加权；两者都需要和无门控版本进行同协议比较。

加载联合 head 并评估 prototype 分数：

```powershell
python train.py discover --dataset cifar100 --data-root .\data `
  --num-known 60 --num-novel 40 --split-path .\splits_cifar100_60_40.json `
  --student-ckpt .\runs\student_joint\student.pt `
  --novel-head-ckpt .\runs\student_joint\novel_head.pt `
  --score-mode novel_entropy --cluster-feature novel_pca `
  --cluster-k oracle --cluster-pca-dim 16 --cluster-normalize `
  --work-dir .\runs\student_joint_detect --device auto
```

`novel_msp` 和 `novel_entropy` 只在联合 head 已经形成稳定 prototype 结构时有意义；当前它们仍是实验性分数，必须与原有 Energy、Mahalanobis 等分数做同协议消融。

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

如果希望让候选样本按可信度参与训练，而不是所有候选等权，可以在上述命令中增加：

    --discovery-soft-weighting
    --discovery-neighbor-k 5
    --discovery-neighbor-temperature 0.5

该选项目前是实验功能。它可能提高 candidate purity，但快速实验尚未证明能同步提高 AUROC 和 FPR95。

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
## Latest local validation (2026-09-20)

- Full CIFAR-100 60/40 run: ImageNet-pretrained ResNet-34 teacher, pretrained ResNet-18 student, seed 42, 10 teacher epochs and 5 unified-joint student epochs.
- Best teacher validation accuracy: 40.43%; best student validation accuracy: 49.50%.
- Full 10,000-image open-test result with `unified_novel_mass`: AUROC 0.6698, AUPR 0.5580, FPR95 0.8147, known accuracy 48.93%, unknown reject rate 14.13%.
- Candidate-pool purity was 66.00%; clustering remains the main weakness: unknown-only NMI 0.421 and ARI 0.088.
- The full run is a meaningful baseline improvement over the earlier small-data smoke experiments, but it is still one seed and is not a final paper result.

## Recent code changes

- Added `--joint-prototype-init kmeans_candidates` as an experimental option. It is not the default because the current small-data comparison did not improve AUROC.
- Fixed `--novel-known-temperature` so an explicit command-line value overrides the checkpoint value.
- Added `unified_novel_mass` to automatic score selection when a novel head is loaded.
- Added `--auto-score-fast` to skip high-dimensional Mahalanobis calibration during lightweight score comparison. This is useful for smoke tests and multi-seed iteration; it does not change the default full calibration path.
- Added the optional `classwise_unified_novel_mass` score. It calibrates the unified novel probability mass separately for each predicted known class using known validation samples, which can reduce class-dependent score bias. It is not enabled by default.
- The first controlled 2,000-image comparison did not support the classwise score: `unified_novel_mass` reached AUROC `0.6939` and FPR95 `0.8051`, while `classwise_unified_novel_mass` reached AUROC `0.6483` and FPR95 `0.8903`. It remains an explicit ablation only and is no longer included in automatic score selection.
- Added `--skip-clustering` for detection-only experiments. It avoids running KMeans and clustering diagnostics when the goal is only AUROC/FPR95 comparison.
- Fixed an efficiency bug: non-Mahalanobis scores no longer compute the high-dimensional Mahalanobis distance, and classwise novel-mass calibration no longer fits unnecessary Mahalanobis statistics.
- The current priority remains improving unknown-class feature structure and clustering, followed by 3-seed ablation experiments.

## Latest local validation update (2026-09-20)

- The discovery pipeline now reuses the final candidate clustering result when writing per-sample assignments. This removes a duplicate PCA/KMeans pass without changing metric definitions.
- Historical note (protocol corrected below): this run used 20% of the CIFAR-100 open test pool for score selection. Its reported test metrics are not a clean held-out estimate and must not be used as a final generalization claim.
- On the same checkpoint, `projection_pca` was the strongest tested clustering representation. In a 2,000-image comparison, unknown-only ARI was `0.1639`, compared with `0.1405` for `feature_pca`, `0.0945` for `novel_pca`, and `0.0727` for `unified_pca`. On the full open-validation run, `projection_pca` reached unknown-only ARI `0.0990`, versus `0.0781` for `unified_pca`.
- With `projection_pca` fixed, KMeans still exceeded Agglomerative clustering in the 2,000-image comparison (`0.1639` vs. `0.1242` unknown-only ARI), so KMeans remains the default clustering baseline.
- The command-line default for `--cluster-feature` is now `projection_pca`. This is a provisional single-seed choice and must be checked with multiple seeds before being treated as a final method conclusion.
- The open-validation result is only a small improvement over the earlier fixed-score baseline. Unknown detection and clustering remain the main research problems; the next formal step is a controlled multi-seed ablation.
- The classwise score passed unit-level numerical checks but performed worse in the first controlled comparison. No performance claim is made for it.

## 本次吸收 qiyuhan 分支的内容（2026-09-21）

本次没有整体合并 qiyuhan 分支，因为该分支基于较早版本，整体合并会覆盖当前 main 已有的统一 known+novel 空间、`--skip-clustering`、Mahalanobis 延迟计算和聚类结果复用等改进。当前只吸收了三个可独立验证的功能，并保持默认行为不变：

1. 不确定性感知蒸馏权重裁剪
   - 新增 `--uncertainty-weight-min` 和 `--uncertainty-weight-max`。
   - 作用于 logits KL 蒸馏和 projection 特征蒸馏，限制 `exp(-teacher_uncertainty)` 的极端值，避免少量样本过度放大或削弱蒸馏梯度。
   - 默认值为 `None`，不改变原有 `raw` / `mean_normalized` 蒸馏行为。

2. 类别条件未知检测阈值
   - `discover` 新增 `--threshold-policy {global,class_conditional}` 和 `--threshold-min-class-samples`。
   - `class_conditional` 只使用已知验证集，按照模型预测的已知类别分别计算分位数阈值；样本不足的类别回退到全局阈值。
   - 默认仍为 `global`，避免历史实验结果因阈值定义变化而失去可比性。阈值类型和具体阈值会保存到 `calibration_report.json`。

3. 聚类前候选池纯化
   - `discover` 新增 `--candidate-purify {none,open_score,entropy,head_uncertainty,uncertainty_consensus}` 和 `--candidate-keep-ratio`。
   - 纯化只改变进入聚类的候选池，不改变 AUROC、AUPR、FPR95 或 known/unknown 检测判断，因此不能把候选池纯度提升宣称为未知检测提升。
   - `discovery_report.json` 和 `discovery_detail.json` 会记录纯化前后数量、删除数量、候选池纯度和真实未知召回率。
   - 默认是 `none`、`1.0`；建议把纯化作为聚类前处理消融，而不是默认主方法。

### qiyuhan 分支已有实验的正确解读

该分支三 seed 对比中，Standard KD 的 AUROC/FPR95/NMI/ARI 为 `0.5883/0.8697/0.4677/0.0529`，Uncertainty KD 为 `0.6012/0.8720/0.4923/0.0507`。这说明不确定性蒸馏对 AUROC、AUPR 和 NMI 有初步收益，但 FPR95 略差、ARI 没有提高，不能声称它全面有效。候选纯化示例中候选纯度约从 `0.4486` 提升到 `0.4707`，同时会删除候选样本，因此必须同步报告未知召回率。

### 当前建议的验证命令

```powershell
# 保持旧实验定义：全局阈值、不过滤候选池
python train.py discover --dataset cifar100 --threshold-policy global --candidate-purify none

# 只比较类别条件阈值，不改变聚类流程
python train.py discover --dataset cifar100 --threshold-policy class_conditional --threshold-min-class-samples 5

# 只比较聚类前候选池纯化；检测指标仍按未纯化的 open score 计算
python train.py discover --dataset cifar100 --candidate-purify uncertainty_consensus --candidate-keep-ratio 0.75
```

候选纯化实验应至少同时查看 AUROC、FPR95、known accuracy、unknown reject rate、candidate purity 和 candidate unknown recall；如果只是候选数量减少而纯度或聚类 NMI/ARI没有稳定提升，就不应继续把它作为主线方法。

## 本轮新增：ReAct 特征裁剪检测基线（2026-09-26）

当前未知检测的核心问题仍是已知/未知分数分布重叠。代码新增可选 `--react-percentile` 和 `--score-mode react_energy`，参考 Sun et al., *ReAct: Out-of-distribution Detection With Rectified Activations*（NeurIPS 2021）的做法：

1. 只用已知训练集提取特征，按指定分位点估计一个激活上限；
2. 对验证集、测试集和 open-validation 样本的特征做上限裁剪；
3. 使用裁剪后的特征重新通过已训练分类器，计算 Energy 分数；
4. 仍然用已知验证集确定阈值，不使用测试集未知标签拟合裁剪值或阈值。

该方法不重新训练模型，适合作为低成本未知检测消融。默认 `--react-percentile 0`，不会改变历史实验。建议先比较：

```powershell
python train.py discover --dataset cifar100 --data-root .\data `
  --num-known 60 --seed 42 --split-path .\splits_cifar100_60_40.json `
  --student-ckpt .\runs\cifar_full_pretrained_joint_e5\student.pt `
  --score-mode react_energy --react-percentile 99.5 `
  --limit-val 300 --limit-test 1000 --skip-clustering `
  --work-dir .\runs\react_energy_smoke --device cpu
```

需要与原始 `energy` 在相同 checkpoint、相同数据子集、相同 MC 设置下比较 AUROC、AUPR、FPR95、known accuracy 和 unknown reject rate。ReAct 只改善检测分数，不应直接宣称改善蒸馏或新类聚类；若检测分数改善但候选池聚类没有改善，下一步仍应处理候选污染和未知特征结构。

相关依据还包括 Liu et al., *Energy-based Out-of-Distribution Detection*（NeurIPS 2020），其核心是用能量而非最大 softmax 概率进行 OOD 排序；ReAct 的裁剪步骤用于抑制已知类别特征中的异常高响应。两者结合适合作为当前项目的检测侧基线，但不是对不确定性知识蒸馏本身有效性的证明。

本地小规模 smoke test（同一 checkpoint、128 张测试样本、120 张验证样本、2 次 MC 采样）中，原始 Energy 的 AUROC 为 `0.4884`、FPR95 为 `0.9398`；ReAct-99.5% Energy 的 AUROC 为 `0.4916`、FPR95 仍为 `0.9398`。这只是代码和评估路径验证，提升极小且不是正式结论；正式实验需要固定完整数据规模并至少运行多个 seed。

## 本轮新增：组合分数标准化（2026-09-26）

原有 `full` 分数直接相加熵、认知不确定性、偶然不确定性和原型距离。这些量的数值尺度不同，可能导致某一个分量主导未知检测排序，因而不能公平体现各模块的贡献。现新增可选 `--score-mode normalized_full`：

- 只使用已知验证集估计每个分量的均值和标准差；
- 对 entropy、epistemic、aleatoric 和 prototype distance 分别做 z-score；
- 再按现有权重组合，默认权重仍为 `(0.5, 0.3, 0.2)`，原型距离保持单独加入；
- 不使用测试集未知标签拟合标准化参数，避免评估泄漏；
- 默认分数模式不变，因此不会破坏历史实验的可复现性。

示例：

```powershell
python train.py discover --dataset cifar100 --score-mode normalized_full `
  --student-ckpt .\runs\cifar_full_pretrained_joint_e5\student.pt `
  --limit-val 300 --limit-test 1000 --skip-clustering `
  --work-dir .\runs\normalized_full_smoke --device cpu
```

该方法解决的是组合分数的尺度不一致问题，不等同于已经解决未知检测或证明蒸馏有效。应在同一 checkpoint、数据划分和 MC 设置下，与 `full`、`entropy_proto`、`energy` 一起比较 AUROC、FPR95、AUPR、known accuracy 和 unknown reject rate。

本地轻量对比（CIFAR-100，60/40 划分，固定已有 checkpoint，验证集 120、测试集 128、MC=2、CPU、跳过聚类）如下：

| score mode | AUROC | AUPR | FPR95 | known accuracy | unknown reject rate |
|---|---:|---:|---:|---:|---:|
| `full` | 0.4597 | 0.3194 | 0.9398 | 0.4337 | 0.0444 |
| `entropy_proto` | 0.4576 | 0.3181 | 0.9157 | 0.4337 | 0.0444 |
| `normalized_full` | 0.5012 | 0.3481 | 0.9036 | 0.4578 | 0.0444 |

在这次小样本测试中，标准化组合分数相对两个基线有改善，但未知拒绝率没有改善，且样本量太小，不能作为正式结论。下一步仍应在完整测试集和多个随机种子上确认；若收益不稳定，应优先改进训练得到的特征空间，而不是继续堆叠检测分数。

## 本轮修正：open-validation 与测试集隔离（2026-09-26）

审查发现旧实现的 `--open-val-ratio` 从开放测试集拆出一部分，用于自动选择 OOD 分数，再在剩余测试样本上报告结果。虽然未直接按测试标签调阈值，但 score-mode 的选择仍接触测试分布，会使最终测试估计偏乐观。现已改为：已知 open-validation 样本从训练划分中的已知验证子集保留，未知 open-validation 样本从训练划分中的 novel pool 保留；这些样本从相应的 validation / discovery pool 中扣除。CIFAR-100 的官方 test split 完整保留作最终评估。单元测试验证了 calibration、open-validation、discovery pool 和 test 的样本 ID 互不重叠。

GPU smoke（固定旧 checkpoint，`open-val-ratio=0.2`、自动快速评分选择、验证/测试各受限为 128）中，自动选择 `novel_msp`，open-validation AUROC `0.6013`；独立测试 AUROC `0.5569`、FPR95 `0.9157`、unknown reject rate `0.1333`。本次数据量很小，仅证明新协议能运行。旧版从测试集挑选评分模式的结果（包括此前记录的 `0.6743`）不是严格 held-out 指标，不能作为最终性能结论；应按新协议重新跑正式实验。

## 本轮尝试：纯未知池的特征原型排斥（2026-09-26）

当前已有 Energy separation 和未知置信度约束，但它们主要作用在分类 logits 上；为直接检验“未知样本靠近已知特征中心”是否是检测失败原因，新增可选 `--alpha-discovery-feature-margin` 与 `--discovery-feature-margin`。做法是将 discovery 特征和已知分类器权重归一化，计算其与最近已知原型的余弦相似度，并惩罚超过 margin 的样本：

```text
L_feature_unknown = mean(relu(max_c cosine(z_unknown, w_c) - margin))
```

该损失只允许用于 `--discovery-pool-mode unknown`，不允许用于 mixed pool；默认权重为 0，不改变旧实验。实现是基于类原型/角度间隔的实验性特征排斥，不是对某篇论文算法的原样复现。相关方法背景可参考：Khosla et al., *Supervised Contrastive Learning*（NeurIPS 2020）的类内/类间对比表征思路；Hendrycks et al., *Deep Anomaly Detection with Outlier Exposure*（ICLR 2019）的辅助异常样本训练协议；Liu et al., *Energy-based Out-of-Distribution Detection*（NeurIPS 2020）的已知/异常分离目标。本文新增项需要单独消融，不能据此宣称复现了这些论文。

小规模 GPU smoke 对比（CIFAR-100，512 张已知训练样本、512 张纯未知 discovery 样本、1 epoch、同一教师权重和 seed；测试 128 张，Energy 检测、跳过聚类）中，基线 AUROC/FPR95 为 `0.4750/0.9759`，加特征排斥后为 `0.4043/0.9157`；unknown reject rate 分别为 `0.0667/0.0444`。known accuracy 只有约 `0.084/0.048`。因此目前不能说该方法有效：它降低了 FPR95，但 AUROC 和未知拒绝率变差，而且 1 epoch/小样本导致分类能力太弱，结果只用于确认代码可训练，不足以定论。**不要将该损失作为默认配置。**

后续判断顺序：先用相同 seed、完整已知训练数据和足够 epoch 做 baseline / feature-margin 配对实验，再至少用 3 个 seed 比较 AUROC、AUPR、FPR95、known accuracy、unknown reject rate 及已知/未知特征到最近原型的距离分布。若检测收益仍不稳定或损害已知分类，应放弃该项；优先回到更可靠的预训练特征和训练协议，再评估 Energy、Mahalanobis 与特征表征学习的独立贡献。

## 本轮继续尝试：KNN-OOD、open-validation 阈值和 Gaussian NLL（2026-09-26）

在上一版协议修复基础上，本轮尝试三种不同方向：

1. **KNN-OOD**：参考 Sun et al., *Out-of-Distribution Detection with Deep Nearest Neighbors*（ICML 2022），以已知训练样本的特征库为参考，按 k 近邻余弦距离评分。新增 `--knn-ood`、`--knn-k`、`--knn-bank-size`、`--knn-feature features|proj`。当前小规模同 checkpoint 测试中，projection KNN AUROC `0.4744`、FPR95 `1.0000`；backbone feature KNN AUROC `0.3360`、FPR95 `1.0000`。不支持将 KNN 作为当前模型的改进方案，说明当前表示空间的局部近邻距离也不可靠。
2. **open-validation 阈值工作点**：新增 `--threshold-policy open_balanced|open_f1`，仅用独立 open-validation 选择工作阈值。Energy + `open_balanced` 在小测试上的未知拒绝率达到 `0.7778`，但已知接受率仅 `0.2651`，AUROC 仍为 `0.4843`。这验证提高拒绝率可以靠牺牲已知覆盖率做到，但没有改善排序能力；默认仍使用已知验证阈值。该策略要求 `--open-val-ratio > 0`，只能作为已知 open-validation 标签可得时的 operating-point 校准，不是纯无监督部署方案。
3. **Gaussian NLL**：参考 Mukhoti et al., *Deep Deterministic Uncertainty: A Simple Baseline*（ECCV 2020）的类条件高斯密度思想，新增对角 Gaussian negative log-likelihood：用已知训练类均值、方差，计算到最近已知类的 NLL，加入 `--score-mode gaussian_nll` 及 `normalized_entropy_gaussian_nll`。同一 checkpoint、checkpoint 内已有 Gaussian 统计量（该 checkpoint 训练配置使用完整已知训练集）、128 张独立测试子集时，Gaussian NLL AUROC `0.6046`、AUPR `0.4456`、FPR95 `0.8554`、unknown reject rate `0.0667`；Energy 对照为 AUROC `0.4843`、AUPR `0.3392`、FPR95 `0.9398`、unknown reject rate `0.0889`。这是本轮最有希望的排序基线，但测试集只有 128 张，且单个 seed，必须用完整测试集和至少 3 seeds 复验，暂不能宣称有效。

因此本轮的判断是：KNN 和直接特征排斥未显示收益；open-validation 阈值能调工作点但会改变已知/未知错误权衡；Gaussian NLL 是当前最值得正式复验的方向。下一步优先固定 `gaussian_nll` 与 Energy / Mahalanobis 对照，完整训练特征建高斯统计量、保持测试集不参与选分数和选阈值，并报告多 seed 均值与标准差。若 Gaussian NLL 复验失败，再考虑更稳健的协方差估计（shrinkage / low-rank）或增强 backbone 预训练与分类精度，不继续盲加损失。

## 本轮后续验证：完整集对照与其他评分器（2026-09-26）

为避免只看 128 张 smoke 子集，本轮在 CIFAR-100 全部 10,000 张开放测试样本上，用同一 `cifar_full_pretrained_joint_e5/student.pt`、60/40 类别划分、`open-val-ratio=0.2`、MC=2 和 known-validation 95 分位阈值复核 Gaussian NLL 与 Energy：

| scorer | AUROC | AUPR | FPR95 | known accuracy | unknown reject rate |
|---|---:|---:|---:|---:|---:|
| Gaussian NLL | 0.6379 | 0.4963 | 0.8557 | 0.4863 | 0.0675 |
| Energy | 0.6422 | 0.5185 | 0.8310 | 0.4912 | 0.1090 |
| logit margin | 0.5954 | 0.4629 | 0.8508 | 0.4868 | 0.0858 |

结论：全量测试上 Energy 略优于 Gaussian NLL，后者未能验证为改进；logit margin 也低于 Energy。三个检测器的 unknown reject rate 都不高，当前阈值仍偏重已知接受率。另加了 `max_logit`、`logit_margin` 两种 logits 基线；小样本中 max-logit 接近随机，logit-margin 有些信号，但全量结果没有超过 Energy。

本轮还实现了 KNN-OOD（参考 Sun et al., *Out-of-Distribution Detection with Deep Nearest Neighbors*, ICML 2022）和 VIM-style 主子空间残差（参考 Ming et al., * აკეთ?* VIM: Out-of-Distribution with Virtual-logit Matching, CVPR 2023）。KNN 在 projection 与 backbone features 上的 smoke AUROC 分别为 `0.4744`、`0.3360`；VIM residual AUROC `0.4744`。它们在此 checkpoint 上都没有效果，不进入推荐主配置。Gaussian NLL 是 DDU 类条件密度建模启发的对角高斯基线，不是对 DDU 完整方法的复现。

全量 `auto` 候选扫描在高维 Mahalanobis 候选上运行数分钟并占用约 2.5 GB 内存，已中止；单个评分器全量评估均顺利完成。后续不要用昂贵的 auto 扫描代替严谨比较；先固定 Energy / Gaussian NLL / logit margin 三个候选，再在至少 3 个 seed 上做完整评估。要提高未知拒绝率而保持已知覆盖，必须改善分数排序/表征，而不是单纯移动阈值；`open_balanced` 的 smoke test 拒绝了 77.8% 未知，但只接受 26.5% 已知且 AUROC 不变，展示了这项权衡。

继续检查题目核心的不确定性分支：新增 `head_uncertainty` 和 `margin_uncertainty` 检测评分，前者直接使用训练出的 uncertainty head，后者结合 logit margin。全量 CIFAR-100 测试中，`head_uncertainty` AUROC `0.5073`、FPR95 `0.9465`、unknown reject rate `0.0660`；`margin_uncertainty` AUROC `0.5903`、FPR95 `0.8530`、unknown reject rate `0.0713`。这说明当前 auxiliary uncertainty head 单独检测未知的能力弱；margin+head 略有信号但没有超过 Energy。结论是不能声称当前不确定性分支已解决未知检测，后续应重新审视 uncertainty target 是否只学到已知分类置信度/错误，而不是未知概率。

## 本轮尝试：纯未知池监督不确定性分离（2026-09-26）

为验证上述问题，新增 `--alpha-discovery-uncertainty-separation`。在纯未知 discovery pool 上，将已知训练样本的 uncertainty 目标设为 0，将 discovery 未知样本目标设为 1，使用 BCE 直接训练 uncertainty head；mixed pool 禁止使用该损失，默认权重为 0。

同一教师模型、CIFAR-100、seed=42、512 known-train、512 pure-unknown discovery、1 epoch 的 GPU smoke 对比，使用 `head_uncertainty` 检测：普通 discovery-energy baseline 的 AUROC/FPR95/unknown reject rate 为 `0.4731/0.9759/0.0000`；加入 uncertainty separation 后为 `0.5264/0.9759/0.1111`。说明该损失确实能让 uncertainty head 对未知样本产生一些区分信号，但已知分类准确率仍很低，且 FPR95 没有改善；当前只能作为受控实验方向，不能作为最终改进结论，也不能用于真实 mixed unlabeled 场景。

下一步应使用完整训练数据、足够 epoch 和至少 3 个 seed 复验，并同时报告 uncertainty head 的已知/未知分布、AUROC、FPR95、known accuracy。若 separation 持续提高未知拒绝却损害已知分类，应降低其权重或采用 warmup/ramp；若仍不稳定，应停止把 uncertainty head 当作独立未知检测器，改为只作为蒸馏权重和组合评分的辅助信号。
## 本轮尝试：Outlier Exposure 均匀 logits 约束（2026-09-26）

为直接降低纯未知样本被某个已知类别高置信接收的问题，新增可选参数 `--alpha-discovery-uniform`。该损失参考 Hendrycks et al., *Deep Anomaly Detection with Outlier Exposure*（ICLR 2019）的异常暴露思想，但这里只实现了其中的均匀预测约束：对纯未知 discovery 样本的已知分类 logits，令 softmax 输出接近所有已知类别上的均匀分布。

实现的目标是：未知样本不应强烈偏向任何一个已知类别。该损失与 Energy 间隔、uncertainty separation 分开统计，因此可以单独做消融；只允许用于 `--discovery-pool-mode unknown`，默认权重为 0，不改变历史实验。

小规模 GPU 对照使用相同 CIFAR-100 60/40 划分、seed=42、512 个已知训练样本、512 个纯未知 discovery 样本、120 个验证样本、128 个测试样本、3 epoch、Energy 检测：

| 方法 | AUROC | FPR95 | known acc（验证集） | unknown reject rate |
| --- | ---: | ---: | ---: | ---: |
| Energy baseline | 0.409 | 0.976 | 0.308 | 0.022 |
| Energy + uncertainty separation（直接启用） | 0.506 | 0.940 | 0.233 | 0.067 |
| Energy + uncertainty separation（warmup/ramp） | 0.440 | 0.964 | 0.292 | 0.089 |
| Energy + uniform logits | **0.531** | **0.916** | 0.275 | 0.022 |

这次结果说明 uniform-logit 约束在该 smoke 配置下改善了未知/已知分数排序和 FPR95，优于本轮的 uncertainty separation 及其 warmup/ramp 版本；但阈值下 unknown reject rate 没有提升，验证集分类准确率也略有下降。因此它只能作为有希望的可选训练机制，不能据此宣称未知检测问题已经解决。下一步应在更大训练规模、完整测试集和至少 3 个随机种子上复核，并同时尝试降低权重或对该损失做渐进调度，观察能否保留 AUROC/FPR95 收益而减少已知分类损失。
补充的中等规模复核使用 1200 个已知训练样本、1200 个纯未知 discovery 样本、300 个验证样本、1000 个测试样本和 5 epoch。Energy baseline 的最佳验证 known_acc 为 `0.360`，uniform 版本为 `0.343`；在 1000 张测试子集上，baseline/uniform 的 AUROC 分别为 `0.497/0.534`，FPR95 分别为 `0.957/0.949`，unknown reject rate 分别为 `0.082/0.046`。因此 uniform 的排序收益在更大子集上仍保留，但默认 known-only 阈值下的拒绝率没有提升。对 uniform 使用独立 open-validation 的 `open_balanced` 阈值后，unknown reject rate 为 `0.240`、known accept rate 为 `0.794`，说明阈值工作点仍是独立问题。

新增 `--discovery-uniform-warmup-epochs` 和 `--discovery-uniform-ramp-epochs` 做训练时机消融。在 512/512/128、3 epoch smoke 中，warmup=1、ramp=2 的 uniform 版本 AUROC `0.501`、FPR95 `0.940`、unknown reject rate `0.044`，低于直接启用 uniform 的 AUROC `0.531`。因此 warmup/ramp 暂不推荐作为 uniform 的默认配置；后续应优先搜索较小 uniform 权重，并在多 seed 上确认排序收益是否稳定。
### 3-seed 配对复核（2026-09-27）

为检查 uniform-logit 收益是否只来自单个随机种子，在同一训练配置下补跑 seed 43、44，并与 seed 42 组成三组配对对照。每组均为 CIFAR-100 60/40、1200 已知训练样本、1200 纯未知 discovery 样本、300 验证样本、1000 测试样本、5 epoch；Energy loss 固定为 0.1，仅切换 uniform loss（0 或 0.1）。检测使用独立 open-validation 协议选择分数/阈值设置，表中是最终 1000 张测试子集指标。

| seed | 方法 | AUROC | FPR95 | known accuracy | unknown reject rate |
| ---: | --- | ---: | ---: | ---: | ---: |
| 42 | Energy baseline | 0.497 | 0.957 | 0.028 | 0.082 |
| 42 | Energy + uniform | 0.534 | 0.949 | 0.023 | 0.046 |
| 43 | Energy baseline | 0.479 | 0.955 | 0.074 | 0.045 |
| 43 | Energy + uniform | 0.533 | 0.948 | 0.029 | 0.077 |
| 44 | Energy baseline | 0.468 | 0.962 | 0.035 | 0.052 |
| 44 | Energy + uniform | 0.503 | 0.941 | 0.040 | 0.028 |
| 平均 | Energy baseline | 0.481 | 0.958 | 0.045 | 0.059 |
| 平均 | Energy + uniform | **0.523** | **0.946** | 0.030 | 0.050 |

三组中 AUROC 都有提升，平均增加约 `0.042`；FPR95 平均下降约 `0.012`。但 known accuracy 平均下降约 `0.015`，unknown reject rate 平均下降约 `0.009`，且每组拒绝率方向不一致。因此这项结果支持“uniform logits 可能改善排序”，却没有证明它改善默认阈值下的未知拒绝能力，更不能宣称整体开放识别已经改善。样本规模仍为 1000 测试图、训练仅 1200 样本且只有 3 个 seed，需在完整训练规模、完整官方测试集和更多 seed 上复核。

方法依据仍是 Hendrycks et al., *Deep Anomaly Detection with Outlier Exposure*（ICLR 2019）的辅助异常数据暴露思想；本实现是纯未知 discovery pool 上的均匀已知类预测约束，并非该论文完整复现。对于当前“AUROC 上升但阈值拒绝率未升”的情况，应分别报告排序指标与工作点指标；阈值可用独立 open-validation 做风险—覆盖率/已知接受率权衡，不能用测试集未知标签挑选阈值。下一步建议先做较小权重（例如 0.025、0.05）的配对多 seed 消融，若已知准确率与拒绝率仍不改善，则不把 uniform 作为主方法，转而优先提升表征质量与 discovery pool 的候选筛选可靠性。
## 本轮继续：较小 uniform 权重消融（2026-09-27）

在上轮 uniform 权重 `0.1` 的三 seed 结果显示“AUROC 有提升但 known accuracy / unknown reject rate 有代价”后，进一步测试权重 `0.05`，其余训练协议完全不变：CIFAR-100 60/40、seed 42/43/44、1200 known train、1200 pure-unknown discovery、300 validation、1000 test、5 epochs，Energy 权重 `0.1`。检测使用 Energy、MC=2 和从训练划分预留的独立 open-validation；最终测试集只用于报告指标。

| seed | 权重 | AUROC | FPR95 | known accuracy | unknown reject rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.05 | 0.487 | 0.954 | 0.044 | 0.043 |
| 43 | 0.05 | 0.538 | 0.950 | 0.040 | 0.040 |
| 44 | 0.05 | 0.506 | 0.951 | 0.056 | 0.035 |
| 平均 | 0.05 | 0.510 | 0.952 | 0.047 | 0.039 |

对照上轮权重 `0` / `0.1` 的三 seed 平均，权重 `0.05` 的 AUROC `0.510` 高于 baseline `0.481`，但低于 `0.1` 的 `0.523`；FPR95 `0.952` 略优于 baseline `0.958` 和 `0.1` 的 `0.946`；known accuracy `0.047` 接近 baseline `0.045`，而 unknown reject rate `0.039` 低于 baseline `0.059` 与 `0.1` 的 `0.050`。因此 `0.05` 减轻了部分已知分类代价，却没有改善默认阈值下的未知拒绝率，不是整体更优配置。

另在 seed 42 单独测试权重 `0.025`：AUROC `0.476`、FPR95 `0.957`、known accuracy `0.043`、unknown reject rate `0.061`。相较同 seed baseline `0.497/0.957/0.028/0.082`，AUROC 与拒绝率都没有改善；只有单 seed，作为探索结果，不扩成正式对比。

当前判断：不继续盲目细扫 uniform 权重。`0.1` 可保留为提高 AUROC 的实验性候选，`0.05` 可作为较温和的分类折中，但二者都没有解决 unknown reject rate 偏低的问题。下一步应将排序指标（AUROC/AUPR/FPR95）与工作点指标（known accept / unknown reject）分开优化：只用训练划分预留的 known+unknown open-validation 选择目标工作点，并报告 known coverage 与 unknown rejection 的完整权衡；同时检查 validation/test 分布是否一致。参考依据为 Saito et al., *OpenMatch: Open-set Consistency Regularization for Semi-supervised Learning*（NeurIPS 2021）的开放集阈值/未知样本识别思路，以及 Geifman and El-Yaniv, *Selective Classification for Deep Neural Networks*（NeurIPS 2017）的风险—覆盖率分析。当前代码已有 `open_balanced` / `open_f1` 阈值选项，但不代表已复现上述方法；必须固定 open-validation 目标和报告方式，且绝不使用测试未知标签调阈值。

## 本轮新增：固定已知覆盖率工作点验证（2026-09-27）

为避免通过“拒绝更多已知样本”制造未知检测提升，新增阈值策略：

```powershell
python train.py discover ... `
  --threshold-policy known_coverage `
  --target-known-coverage 0.95
```

该策略只使用已知验证集分数的 95% 分位数确定阈值，不使用未知测试标签。`calibration_report.json` 会记录验证集上的目标覆盖率和实际校准覆盖率；`discovery_report.json` 还会记录 `threshold_policy`、`target_known_coverage` 和 `calibration_known_accept_rate`。其中 `threshold_type: global` 仅表示最终阈值是一个标量，不能代替阈值策略名称。

在 CIFAR-100 60/40 划分、seed=42、相同 1200 张已知训练样本、300 张验证样本、1000 张测试样本、Energy、MC=2、跳过聚类的条件下，固定验证集已知覆盖率为 95%，结果为：

| 方法 | AUROC | FPR95 | 测试已知接受率 | 测试未知拒绝率 |
| --- | ---: | ---: | ---: | ---: |
| Energy baseline | 0.497 | 0.957 | 0.970 | 0.048 |
| Energy + uniform=0.05 | 0.487 | 0.954 | 0.933 | **0.087** |
| Energy + uniform=0.10 | **0.521** | **0.903** | 0.928 | 0.054 |

这里测试集已知接受率不一定恰好等于 95%，因为阈值只在验证集上校准，测试集存在分布和抽样波动；公平比较的原则是三种模型均使用同一校准规则，而不是用测试标签重新调阈值。结果说明：uniform=0.10 的排序指标最好，但未知拒绝率只略高于 baseline；uniform=0.05 在这次单 seed 工作点上未知拒绝率最高，却伴随较低 AUROC 和较低已知接受率，不能据此认定它是整体最优。

因此，uniform logits 当前只能作为“可能改善分数排序”的辅助训练消融，不能宣称已经解决未知检测。后续应报告完整的 known-coverage / unknown-rejection 曲线，并在完整训练规模和多个 seed 上复核；若模型在 90%--95% 已知覆盖率范围内仍不能稳定提高未知拒绝，就应停止继续细调 uniform 权重，转向改善 backbone 特征空间和未知候选池质量。固定覆盖率策略参考 selective classification 的风险—覆盖率评估思想；本项目代码实现的是评估协议，不是对该文方法的完整复现。

## 本轮新增：ArcFace 风格角度间隔表征约束（2026-09-27）

针对已知类和未知类特征重叠的问题，新增可选参数：

```powershell
--alpha-angular 0.1 --angular-margin 0.2 --angular-scale 16
```

该损失参考 Deng et al., *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*（CVPR 2019）的核心思路：将特征和分类器权重归一化，在真实类别的角度上增加 margin，使已知类别形成更紧凑、更有间隔的特征簇。当前实现是附加在普通交叉熵、知识蒸馏和对比损失上的辅助项，不改变推理阶段的分类 logits，也不是 ArcFace 的完整复现。默认 `--alpha-angular 0`，因此历史实验仍可复现。

在 CIFAR-100 60/40、seed=42、预训练 ResNet34 教师和 ResNet18 学生、1200 张已知训练样本、1200 张纯未知 discovery 样本、5 epochs、Energy、MC=2 的配对实验中，使用固定验证集已知覆盖率 95% 的阈值策略，结果如下：

| 方法 | AUROC | FPR95 | 测试已知接受率 | 未知拒绝率 | 最佳验证 known accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原有 Energy + KD | 0.497 | 0.957 | 0.970 | 0.048 | 0.360 |
| 加入 angular margin | **0.535** | **0.901** | 0.949 | **0.056** | 0.353 |

这次结果初步支持“增强已知类角度间隔有助于未知检测排序”的判断，并且未知拒绝率有小幅提高；但已知分类准确率略降，且只有一个 seed、训练规模仍受限，不能据此确定方法有效。后续应至少用 seed=42/43/44、完整训练规模和完整测试集复验，同时报告 known coverage、unknown rejection、AUROC、FPR95、已知分类准确率以及到最近类原型的距离分布。如果多 seed 后收益不稳定，应该降低 angular loss 权重或只在训练前期使用，而不是继续叠加更多检测分数。

该方法解决的是已知特征紧凑性和类间间隔问题，与 Hendrycks et al. 的 Outlier Exposure 不同：它不直接把未知样本推向均匀 logits，也不使用未知标签，因此可以作为当前项目“已知表征增强”和“不确定性感知蒸馏”的独立消融。后续还应检查角度间隔是否会改变教师—学生 logits 的蒸馏难度，必要时分别比较 standard KD、uncertainty KD 和 angular + uncertainty KD。

## 本轮新增：OpenMax 风格 Weibull 尾部评分（2026-09-27）

在 angular checkpoint 上继续比较距离型未知检测器。新增 `--score-mode openmax`，参考 Bendale and Boult, *Towards Open Set Deep Networks*（CVPR 2016）的 OpenMax 思路：

1. 使用已知训练特征计算各类别的 mean activation vector（这里对应类中心）；
2. 统计每个已知类样本到类中心的距离；
3. 对距离尾部拟合 Weibull 分布；
4. 对测试样本计算到每个已知类中心的尾部 CDF，并以所有类别中的最小尾部概率作为未知分数。

这只是 OpenMax 的距离/极值理论部分的简化实现，没有复现原论文的完整 logit 重校准和 top-k activation 机制。它只使用已知训练数据拟合尾部分布，不使用未知测试标签，默认不改变训练流程。

同一 angular checkpoint、CIFAR-100 60/40、seed=42、1200 已知训练样本、300 验证样本、1000 测试样本、MC=2、验证集已知覆盖率 95% 的对比结果如下：

| 评分器 | AUROC | FPR95 | 测试已知接受率 | 未知拒绝率 |
| --- | ---: | ---: | ---: | ---: |
| Energy | 0.535 | 0.901 | 0.949 | 0.056 |
| 原型距离 | 0.559 | 0.893 | 0.967 | 0.036 |
| Gaussian NLL | 0.578 | 0.911 | 0.962 | 0.048 |
| OpenMax-Weibull | **0.580** | **0.891** | 0.975 | **0.064** |

在这次单 seed 实验中，OpenMax-Weibull 的排序和未知拒绝率略好于 Energy、原型距离和 Gaussian NLL，但提升幅度很小，测试已知接受率也高于目标 95%。因此它目前只能作为有希望的检测基线，不能证明核心特征重叠问题已经解决。下一步应在 3 个 seed 和完整数据规模上复验，并检查 Weibull 尾部拟合是否受每类样本数影响；如果收益消失，应优先继续改进特征学习，而不是继续修改 EVT 拟合细节。

随后用相同 angular 训练配置在 seed=43、44 上复验 OpenMax，三 seed 汇总为：AUROC `0.525 ± 0.042`、FPR95 `0.924 ± 0.023`、测试已知接受率 `0.957 ± 0.013`、未知拒绝率 `0.056 ± 0.012`。seed=42 的 AUROC 为 `0.580`，但 seed=43 只有 `0.479`，说明随机波动仍然明显。因此 OpenMax-Weibull 目前保留为检测侧候选基线，不作为核心创新或默认评分器；下一步重点仍应放在稳定改善特征空间，并用多 seed 对比 standard KD、uncertainty KD 和 angular + uncertainty KD。

### angular margin 的严格三 seed 配对结论

为排除随机波动，使用相同 teacher、数据划分、训练轮数和 discovery pool，对原有模型与 angular margin 模型分别在 seed=42/43/44 上训练，并统一用 OpenMax 和固定验证集已知覆盖率 95% 评估：

| 方法 | AUROC | FPR95 | 测试已知接受率 | 未知拒绝率 | 已知分类准确率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原有模型平均 | 0.525 | 0.937 | 0.955 | **0.067** | 0.279 |
| angular margin 平均 | 0.525 | **0.924** | 0.957 | 0.056 | **0.301** |

因此 angular margin 只稳定改善了 FPR95 和已知分类准确率，但 AUROC 基本不变，未知拒绝率反而下降。它不能作为当前主方法，保留为表征学习消融；这也说明单纯增强已知类间隔不足以解决未知特征与已知类重叠。

## 本轮尝试：Proxy Anchor 表征损失（2026-09-27）

参考 Kim et al., *Proxy Anchor Loss for Deep Metric Learning*（CVPR 2020），新增可选参数：

```powershell
--alpha-proxy-anchor 0.1 --proxy-anchor-alpha 32 --proxy-anchor-margin 0.1
```

Proxy Anchor 按类别代理聚合正样本和负样本，理论上比当前 Proxy-NCA 风格损失更适合一个 batch 中每类样本较少的情况。它只用于改善已知表征，不直接使用未知标签，默认权重为 0。

在 seed=42、其余配置与 angular/baseline 一致的 5 epoch 实验中，OpenMax + 固定 95% 已知覆盖率结果为：AUROC `0.553`、FPR95 `0.962`、测试已知接受率 `0.956`、未知拒绝率 `0.059`、已知分类准确率 `0.263`。AUROC 比同 seed 原有 OpenMax baseline 的 `0.524` 高，但 FPR95 明显变差，已知分类准确率也偏低，因此不能认定 Proxy Anchor 有效，也暂不扩展多 seed。

当前判断是：ArcFace 和 Proxy Anchor 都能改变已知特征结构，但尚未稳定改善未知检测工作点。下一步应优先研究显式未知空间建模方法（例如 ARPL 的 reciprocal points），而不是继续堆叠多个已知类度量损失。

## 本轮尝试：ARPL-inspired reciprocal points（2026-09-27）

为检验“显式建模未知空间”是否优于仅根据已知类原型/能量打分，新增可选 reciprocal points 训练项和检测分数，思路借鉴 Chen et al., *Adversarial Reciprocal Points Learning for Open Set Recognition*（ECCV 2020，ARPL）。它不是 ARPL 的完整复现：当前实现使用一组可学习向量，令已知特征远离 reciprocal points、纯未知 discovery 特征靠近 reciprocal points，同时保持其与已知分类器原型的间隔；推理阶段以特征对 reciprocal points 的最大余弦相似度作为未知分数。

训练需显式提供纯未知 discovery pool；若 discovery pool 混有已知类，不能把其中样本全部当成未知来训练。可用 `--reciprocal-points 8 --alpha-reciprocal 0.1 --reciprocal-margin 0.2` 开启训练项，再用 `--score-mode reciprocal --threshold-policy known_coverage --target-known-coverage 0.95` 检测。默认关闭，尚未验证前不建议设为默认选项。

在 CIFAR-100 60/40、seed=42/43/44、1200 个已知训练样本、1200 个纯未知 discovery 样本、5 epoch 的快速实验中，对同一 reciprocal 训练模型比较不同检测分数；阈值均按已知验证集 95% 覆盖率校准：

| 分数 | AUROC（均值±标准差） | FPR95（均值±标准差） | 测试已知接受率 | 未知拒绝率 |
| --- | ---: | ---: | ---: | ---: |
| Energy | 0.555 ± 0.010 | **0.894 ± 0.024** | 0.954 ± 0.011 | 0.053 ± 0.016 |
| OpenMax-Weibull | 0.521 ± 0.032 | 0.951 ± 0.009 | 0.956 ± 0.008 | 0.038 ± 0.009 |
| Reciprocal similarity | **0.568 ± 0.011** | 0.918 ± 0.007 | 0.947 ± 0.014 | **0.100 ± 0.024** |

Reciprocal similarity 相对同一 checkpoint 的 Energy，平均 AUROC 提高约 0.014、未知拒绝率提高约 0.047，但 FPR95 恶化约 0.024；所以它改善了部分排序/工作点指标，却没有全面优于 Energy，更不能称为解决了未知检测。三个 seed 的 reciprocal 未知拒绝率为 0.128、0.087、0.087，存在波动。不同 seed 的测试已知接受率在 0.930–0.963 之间，与校准目标 0.95 有小幅偏差；这是有限验证样本分位数阈值在独立测试样本上的泛化误差，应同时报告校准覆盖率和测试覆盖率，不应利用测试标签调阈值。

这组结果支持继续做受控消融，而不是直接扩展为完整大规模训练：先固定训练配置和 seed，比较 reciprocal points 数量（例如 1/4/8/16）与 `alpha-reciprocal` 权重，并同时报告 AUROC、AUPR、FPR95、OSCR、已知接受率及未知拒绝率；若只在某个阈值工作点有提升而 AUROC/FPR95 不稳，应保留为辅助分数/消融，不作为默认检测器。之后再用完整训练数据和至少 3 个 seed 复验。聚类评估还应分开报告候选池纯度和真实未知样本子集的聚类结果，避免检测漏检掩盖聚类本身的能力。

## 全量数据检测器复验（2026-09-27）

为验证小样本筛选结果能否复现，使用 CIFAR-100 60/40、seed=42、完整已知训练集、完整验证/测试集、ResNet-34 已有教师、ResNet-18 学生、5 epoch 训练了一个 reciprocal + discovery-energy 模型；训练使用纯未知 discovery pool。所有评分器在同一个学生 checkpoint 上比较，阈值只由已知验证样本按 95% 覆盖率校准，不用测试标签挑阈值。KNN 参考 Sun et al., *Out-of-Distribution Detection with Deep Nearest Neighbors*（ICML 2022）；ReAct 参考 Sun et al., *React: Out-of-Distribution Detection With Rectified Activations*（NeurIPS 2021）；ViM 参考 Wang et al., *ViM: Out-of-Distribution with Virtual-logit Matching*（CVPR 2022）。当前 KNN/ViM/ReAct 都是项目现有评分器的实现与对照，不代表完整复现上述论文所有设定。

| 同一全量 checkpoint 的评分器 | AUROC | AUPR | FPR95 | 测试已知接受率 | 未知拒绝率 | OSCR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Energy | 0.6491 | 0.5291 | **0.7910** | **0.9488** | 0.1070 | 0.3718 |
| Reciprocal similarity | 0.6139 | 0.5373 | 0.9047 | 0.9525 | **0.1460** | 0.3104 |
| KNN feature distance, k=10 | 0.5692 | 0.4596 | 0.9507 | 0.9495 | 0.0828 | 0.3340 |
| Entropy + KNN distance, k=10 | 0.5866 | 0.4666 | 0.8840 | 0.9485 | 0.0838 | 0.3535 |
| ReAct energy, 90th-percentile cap | **0.6544** | **0.5437** | 0.8233 | 0.9438 | 0.1198 | **0.4021** |
| ViM residual, rank=64（小样本筛查） | 0.5448 | 0.4148 | 0.9161 | 0.9145 | 0.0663 | 0.1582 |

注意：ViM 一行来自同一 seed 的 1200/300/1000 小样本筛查，而不是全量模型；仅为记录筛选结果，不能和上方全量数值直接作严格横向结论。其余行来自同一全量 checkpoint 和完整测试集。全量结果显示，KNN 在小样本上的较好表现没有复现；熵+KNN 也未超过 Energy。ReAct 在 AUROC、AUPR、OSCR 上仅有小幅优势（AUROC +0.005），但 FPR95 更差（+0.032），且测试已知接受率低于目标约 0.6 个百分点，因此目前只能作为检测侧候选，不足以替换 Energy 默认配置。reciprocal 明显提高了未知拒绝率，却损害排序与 FPR95，继续保留作训练/评分器消融，不宜单独作为主方案。

全量聚类报告使用 ReAct 分数和 composite auto-K：候选池 816 个，其中真正未知样本 479 个、被错拒的已知样本 337 个，候选池纯度 `0.587`；自动估计 `K=20`，但真实未知类别数是 40（误差 20）。候选池真实未知子集的 NMI 为 `0.429`，ARI 为 `0.196`。同一模型的 Energy + silhouette auto-K 也估计 `K=20`，候选纯度 `0.582`，未知子集 NMI `0.498`、ARI `0.237`。因此 auto-K 和混杂候选池是新类发现的主要瓶颈：不能只报告聚类准确率或 oracle-K；需分开报告候选纯度、未知召回、已知误拒、auto-K 误差、未知子集聚类，以及在 oracle-K=40 下的聚类上限。oracle-K 只能作诊断上界，不能冒充无监督真实部署结果。

当前建议：暂时保留 Energy 为强基线，并将 ReAct 加入检测消融；不要继续优先堆叠更多 OOD 分数。下一步优先改进自动估计新类数和未知候选净化/拒识边界，并确保任何 auto-K 选择只使用无标签候选特征，测试真值仅用于最终评估。对 reciprocal 和 ReAct 的结论仍是单 seed，需要至少 3 个 seed 的全量配对复验后才能确定是否稳定有效。

## auto-K 上限 bug 修复与候选净化诊断（2026-09-27）

复核上面的全量聚类诊断后发现，`evaluate_cluster_candidates` 曾把搜索上限硬编码为 `K<=20`，而 CIFAR-100 实验的未知类别数是 40。之前 silhouette/composite auto-K 都返回 20，受这个隐藏上限直接影响；因此旧版结果不能用来判断自然簇数为 20。现已删除硬编码限制，新增 `--max-auto-clusters`（默认 50），并把它和评估配置一起记录。该上限是无标签搜索的计算边界，不应根据测试集真实类数调节；实际使用时需要在独立验证/开发划分上预先设定合理范围。

相同全量 ReAct checkpoint、相同 816 个未过滤候选的重跑结果：silhouette 在 `K=2..50` 中选择 `K=38`，与评估真值 `K=40` 相差 2；candidate-pool purity 仍为 `0.587`，未知子集 NMI/ARI 为 `0.475/0.186`。对照 Rousseeuw (1987) 的 silhouette 准则，放宽搜索空间确实消除了旧版“卡在上限”的假象，但 `K=38` 只是这一候选集合和特征表示下的估计，不能据一次实验宣称 auto-K 已解决。新增对角协方差 GMM-BIC 选择（BIC 思路源自 Schwarz, 1978）在同一批候选上选 `K=21`，表现较差；这说明简单的单高斯混合假设不适配当前高维、多形状簇数据，暂时不作为默认方法。

另对 `uncertainty_consensus` 候选净化预先固定保留比例 75% 做了诊断：候选从 816 减为 612，但真正未知召回率由 `1.000` 降到 `0.691`，候选纯度反而由 `0.587` 降至 `0.541`；auto-K 变为 27，未知子集 NMI/ARI 为 `0.487/0.167`。所以当前 consensus 排名没有把未知样本可靠地排在前面，简单 top-ratio 删除不仅没净化，反而丢失约 31% 已检测出的未知样本。该策略应继续关闭，不可根据这次测试标签结果再调保留比例。

当前更合理的改进顺序是：先保留修复后的上限控制，避免人为截断；再在训练/验证开发划分上对比 silhouette、stability 和多个聚类表示，冻结选择策略后仅在测试集报告一次；候选净化需要先提升未知/误拒已知的排序能力（例如基于局部邻域一致性与已知类密度联合建模），不能靠降低保留比例制造表面纯度；若现实设定中未知类别数完全未知，应报告不同预设搜索上限的敏感性，而非用真实 `K=40` 作为算法输入。Silhouette：Rousseeuw, *A Graphical Aid to the Interpretation and Validation of Cluster Analysis*, J. Comput. Appl. Math., 1987；BIC：Schwarz, *Estimating the Dimension of a Model*, Ann. Statist., 1978。当前 GMM-BIC 只借鉴准则作对比，不是新类发现论文算法的完整复现。

## 相对密度评分与开放验证阈值尝试（2026-09-27）

针对候选池混入误拒已知样本，继续检验两条思路。第一条借鉴 Ren et al., *A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection*（NeurIPS 2021）的 relative Mahalanobis distance（RMD）：从类条件 Mahalanobis 距离中减去全局背景密度距离，避免仅因样本处在全局低密度区域就判为未知。当前实现以已知训练特征拟合类别统计，因 dense precision 求逆在本机出现明显计算阻塞，改用对角全局背景方差作为可运行的轻量近似；因此它是借鉴 RMD 机制的对角近似，不是原论文完整实现。

在同一全量 reciprocal checkpoint、CIFAR-100 完整测试集、已知验证集 95% 覆盖率下，该近似 RMD 得到 AUROC `0.5795`、FPR95 `0.9907`、已知接受率 `0.9508`、未知拒绝率 `0.0540`；相同 checkpoint 的 Energy 为 `0.6491 / 0.7910 / 0.9488 / 0.1070`。结果明显更差，因此不纳入主线；说明背景密度相减并不能自动修复当前表征重叠，协方差/距离假设也需与表示匹配。

第二条检验类条件阈值（按预测已知类分别用已知验证分数分位数校准），参考 conformal prediction 中按组校准/coverage 控制的思想（例如 Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*, 2023）。同一全量模型的 AUROC 不变（`0.6491`），测试已知接受率 `0.9275`、未知拒绝率 `0.1278`；相比全局 Energy 的 `0.9488 / 0.1070`，它只是把更多已知样本拒掉换取稍高未知拒绝，未形成更好的已知覆盖率控制。该小样本每类分位数阈值受有限校准样本和组间分布偏移影响，不能声称有严格 conformal 保证；保留为可选对照，不作为修复方案。

据此新增 `--threshold-policy open_known_coverage`：在 `--open-val-ratio` 预留的已知/未知开放验证子集上，选择满足目标已知接受率约束、同时最大化未知拒绝率的阈值。它明确需要一小份带 known/unknown 身份标签的校准数据，不属于纯无监督开放集检测；AUROC、AUPR、FPR95 等排序指标不因此改变。Energy、seed=42 的完整测试结果为 AUROC `0.6491`、FPR95 `0.7910`、测试已知接受率 `0.9463`、未知拒绝率 `0.1125`；而 open balanced 阈值曾把已知接受率压到 `0.5145` 才得到未知拒绝率 `0.7160`。在 95% known coverage 约束下，开放校准仅带来与 known-only 分位数相近的工作点，没有显著抬升未知拒绝率。这能用于有标注校准样本的部署场景，但不是模型表征本身改善。

这些尝试的文献启发与判断：RMD 通过对照背景密度来消除 Mahalanobis 的低密度偏置，但本任务当前近似结果失败；按组/类分位数有助于处理类别间置信度尺度差异，但有限样本时阈值方差较高；开放验证阈值直接利用未知校准样本，必须与不使用未知标签的主设定分开报告。核心工作仍应回到提高表示质量和候选池局部结构，而不是把阈值校准误当成模型能力提升。下一步宜在冻结 auto-K 修复后，比较候选样本的 kNN 局部密度/邻域一致性与类条件距离联合排序，并用独立开发划分选定机制，再在测试集报告候选纯度、未知召回、误拒已知及聚类指标。

## 审计后第一轮复核（2026-09-27）

本轮没有重新启动最慢的完整训练，而是使用已经完成的、相同数据划分和检测协议下的 E/F 多 seed 结果重新汇总，确认审计修改没有改变历史结果解释：

| 方法 | seed | AUROC | FPR95 | known acc | unknown reject | auto-K |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E：完整表征 | 42, 123 | 0.6027 ± 0.0169 | 0.8655 ± 0.0255 | 0.4218 ± 0.0040 | 0.0660 ± 0.0141 | 15.0 ± 1.4 |
| F：E + 纯未知 discovery pool NT-Xent | 42, 123 | **0.6397 ± 0.0031** | **0.8413 ± 0.0016** | **0.5221 ± 0.0107** | **0.0864 ± 0.0062** | 11.5 ± 0.7 |

F 相对 E 的方向在两个 seed 上保持一致：AUROC、FPR95、已知分类准确率和未知拒绝率均改善；但这仍只是 2 个 seed，且 F 使用的是由类别划分得到的纯未知 discovery pool，只能视为受控上限实验，不能直接等同于真实无标签混合场景。F 的 oracle-K 聚类 ACC/NMI/ARI 为 `0.2058/0.5053/0.0679`，auto-K 下为 `0.1258/0.3389/0.0426`，说明检测和表征有一定改善，但聚类仍弱，且自动 K 平均只估计到约 12 个。

本轮修正的 auto-K 说明也已同步更新：搜索上限现在由 `--max-auto-clusters` 控制，默认 50；这只是消除了旧版 `K<=20` 的程序限制，不代表自动估计新类数已经可靠。当前仍应把 `oracle K` 作为诊断上限，把 `auto K` 作为真实无标签结果单独报告。

下一轮只做一个受控问题：在相同 seed、训练轮数、学生模型、检测评分器和数据规模下，对比 `discovery-pool-mode unknown` 与 `discovery-pool-mode mixed`。前者回答“纯未知池学习的上限是否真实存在”，后者回答“当无标签池含有已知/未知混合样本时，这个损失是否仍然安全”。在该对照完成前，不把 F 的收益归因于不确定性蒸馏，也不把纯未知池结果写成真实开放场景结论。

该最小对照已经完成：CIFAR-100 60/40、seed=42、预训练 ResNet-34 教师/ResNet-18 学生、2 epochs、512 张已知训练样本、128 张验证样本、512 张测试样本、512 张 discovery pool、同一 `normalized_entropy_mahalanobis` 检测器和 oracle-K 聚类。

| discovery pool | AUROC | AUPR | FPR95 | known acc | unknown reject | candidate purity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `unknown`（纯未知） | **0.5712** | **0.4187** | **0.9233** | 0.1779 | **0.0538** | 0.4545 |
| `mixed`（已知/未知混合） | 0.5158 | 0.3829 | 0.9509 | **0.1902** | 0.0484 | 0.4500 |

这次结果与已有较大规模 E/F 复核的方向一致：纯未知池学习对未知检测更有利，但它使用了类别标签筛出的未知样本，属于受控上限；mixed 池更接近真实场景，却没有在本轮小实验中获得同样收益。两种设置的候选池纯度都只有约 0.45，说明候选池污染仍是聚类瓶颈。该实验只有一个 seed、训练轮数很少，不能作为正式性能结论；它的作用是确认后续必须把 `unknown` 和 `mixed` 分开报告，并优先研究混合池中的可靠候选筛选，而不是继续把纯未知池收益直接归因于整体方法。

对应目录为 `runs/audit_pool_unknown_s42`、`runs/audit_pool_unknown_s42_detect`、`runs/audit_pool_mixed_s42` 和 `runs/audit_pool_mixed_s42_detect`。

## 蒸馏独立验证：标准 KD 与不确定性 KD（小规模，2026-09-27）

为避免把 discovery pool 的影响混入蒸馏结论，本轮关闭 discovery pool、特征蒸馏、SupCon 和原型约束，只保留 CE + KL；教师模型、数据划分、学生结构、训练轮数和检测器完全相同。实验使用 seed=42、2 epochs、512/128/512 数据规模，检测使用 `normalized_entropy_mahalanobis`，跳过聚类。

| 方法 | AUROC | AUPR | FPR95 | known acc | unknown reject | OSCR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 标准 KL KD | **0.4831** | **0.3575** | 0.9509 | 0.1626 | 0.0161 | 0.0968 |
| 不确定性加权 KL KD | 0.4658 | 0.3319 | **0.9080** | **0.1810** | **0.0376** | **0.1184** |

这次单 seed、短训练结果不能证明不确定性 KD 整体优于标准 KD：它改善了 FPR95、已知分类准确率和未知拒绝率，但 AUROC、AUPR 下降。它说明不确定性加权可能改变排序与阈值工作点之间的权衡，不能只看某一个指标；也说明此前“不确定性蒸馏有效”的说法仍未被严格验证。正式结论需要在修正后的固定协议下至少做 3 个 seed，并同时报告排序指标与工作点指标。

当前阶段性结论因此进一步明确：

- 纯未知 discovery pool 学习有较稳定的初步正向信号，但属于上限实验；
- mixed discovery pool 在小实验中没有复现同样收益；
- 不确定性 KD 目前只能作为待验证模块，不能直接称为有效创新；
- 下一步应做固定协议下的 3-seed `standard KD vs uncertainty KD`，然后再决定是否把不确定性权重纳入主模型。

## 严格三 seed 蒸馏复核：Standard KD vs Uncertainty KD（2026-09-27）

本轮补齐了完整规模 CIFAR-100 60/40 划分下的第三个 seed（`3407`），并与已有的 `42`、`123` 结果组成严格配对比较。两种方法使用相同教师模型、ResNet-18 学生模型、预训练初始化、15 个 epoch、相同数据划分和 `normalized_entropy_mahalanobis` 检测器；训练中关闭 discovery pool、特征蒸馏、SupCon、原型、VOS 等其他模块，只改变 `kd-mode`。

oracle-K 检测结果如下，数值为三个 seed 的均值 ± 样本标准差：

| 方法 | AUROC | FPR95 | known accuracy | unknown reject |
| --- | ---: | ---: | ---: | ---: |
| Standard KD | **0.5764 ± 0.0233** | **0.8826 ± 0.0174** | **0.4261 ± 0.0055** | 0.0561 ± 0.0064 |
| Uncertainty KD | 0.5737 ± 0.0340 | 0.8833 ± 0.0165 | 0.4143 ± 0.0042 | **0.0564 ± 0.0114** |

这组正式复核没有支持“不确定性 KD 整体优于标准 KD”：Uncertainty KD 的 unknown reject rate 只高 `0.0003`，同时 AUROC、FPR95、known accuracy 的均值略差，且 AUROC 和 unknown reject 的跨 seed 波动更大。因此当前更严谨的结论是：不确定性加权 KD 已经完成实现，但尚未证明能稳定解决已知/未知特征重叠，也不能直接作为默认主方法或论文创新结论。该结论与当前核心瓶颈一致：未知拒绝率仍约为 `5.6%`，大量未知样本仍被接受为已知；问题不是简单调阈值，而是学生特征和不确定性分数没有形成足够分离。

本轮还有两个实验记录问题需要修正：42/123 的旧版检测报告没有保存 AUPR 和 OSCR，只有 3407 新报告包含这两个字段，因此不能把 `0.4252` 和 `0.2962` 误写成三 seed 均值。后续补跑时必须统一保存 AUROC、AUPR、FPR95、OSCR、known accuracy、known accept rate 和 unknown reject rate。

3407 的新校准诊断中，auxiliary uncertainty head 与已知分类错误的 Pearson 相关系数为 `0.0213`，已知验证集 ECE 为 `0.2452`。但必须注意：本轮为了隔离蒸馏因素，Standard KD 与 Uncertainty KD 都设置了 `alpha_unc=0`，因此该 head 没有参与训练；这组数值不能被解读为 uncertainty target 本身失败。它只说明蒸馏独立对照没有验证 uncertainty head。下一步应单独固定 `alpha_unc>0`，再比较 confidence、classification-error、margin 等 target，并用独立验证集检查 uncertainty-error correlation、AUROC 和可靠性曲线。仅把 `exp(-u_teacher)` 乘到 KL 损失上，也不能自动得到有效的未知检测器。

对应结果目录为 `analysis/audit_kd_standard_full3`、`analysis/audit_kd_uncertainty_full3`，以及 `runs/revised_ms_s3407_B_standard_kd*` 和 `runs/revised_ms_s3407_C_uncertainty_kd*`。本轮还暴露出全量检测包含多次 MC 特征提取和 KMeans 稳定性分析，单次耗时较长；后续应增加特征缓存或快速评估模式，但这属于实验工程优化，不应改变正式评估协议。

当前决策：保留 Uncertainty KD 作为待研究模块和消融项，暂不默认启用；Standard KD 作为更稳定的蒸馏基线。下一步不再盲目调蒸馏权重，而是检查不确定性目标是否真正表示未知风险，并在统一报告协议下研究显式未知空间建模、可靠候选筛选和 mixed discovery pool。

## 本轮新增：VOS 风格虚拟未知特征实验（2026-09-27）

为直接缓解已知/未知特征重叠，新增可选的 VOS-inspired 虚拟未知训练项。当前支持两种生成方式：`batch_center` 从当前已知 batch 的类条件特征中心生成尾部特征，`gaussian` 从已知训练特征拟合的类条件对角高斯尾部采样；随后对虚拟特征施加 Energy 间隔和可选的均匀 logits 约束。该实现是轻量近似，不是 Du et al., *VOS: Learning What You Don't Know by Virtual Outlier Synthesis* 的完整复现；默认 `--alpha-vos 0`，不会改变历史实验。

可用参数：

```powershell
--alpha-vos 0.01 --vos-mode gaussian --vos-tail-scale 2.0 --vos-noise-scale 0.1 --vos-uniform-weight 0.1
```

本轮小规模 CIFAR-100 对照使用相同 seed=42、2 epochs、512/128/512 数据规模、预训练 ResNet-34 教师和 ResNet-18 学生、CE + 不确定性 KD，并使用 `normalized_entropy_mahalanobis` 检测：

| 方法 | AUROC | FPR95 | known acc | unknown reject |
| --- | ---: | ---: | ---: | ---: |
| 不确定性 KD baseline | 0.4658 | **0.9080** | **0.1810** | **0.0376** |
| 初版代理外推 VOS，alpha=0.05 | 0.4613 | 0.9417 | 0.1503 | 0.0538 |
| batch 类中心 VOS，alpha=0.01 | **0.4930** | 0.9479 | 0.1687 | 0.0215 |
| Gaussian 尾部 VOS，alpha=0.01 | 0.4589 | 0.9479 | 0.0798 | 0.0376 |

batch 类中心版本的 AUROC 有小幅提升，说明虚拟尾部特征可能包含未知分离信号；但 FPR95 和未知拒绝率没有改善，不能说明核心问题已经解决。初版代理外推的 `vos` 损失约为 19，训练扰动过强；改为类中心后损失约为 1.5，数值更稳定，但当前目标仍存在排序指标与工作点指标冲突。

当前判断：VOS 只保留为研究消融，不纳入默认主模型，也不继续盲目搜索权重。Gaussian 版本虽然实现了类条件统计量和每轮刷新，但在小规模实验中损害了已知分类，说明当前简单对角高斯尾部与学生特征空间并不匹配。除非后续有新的理论或实现依据，否则不再沿 VOS 方向扩展；下一步回到真实 mixed discovery pool 的候选质量、已知表征学习和不确定性 KD 的严格多 seed 验证。

## 数据协议审计与下一步计划（2026-09-28）

针对“未知拒绝率约为 5.6%，是否由训练集和测试集划分造成”的问题，先审计了当前 CIFAR-100 数据协议。当前流程使用官方训练集和官方测试集：训练集中的 90% 已知样本用于训练，10% 已知样本用于验证；测试集保持独立，包含 60 个已知类和 40 个未知类，每个类别各 100 张测试图像。CIFAR-100 路径没有发现训练/测试样本泄漏，因此目前不能把未知检测较弱直接归因于数据泄漏或样本划分错误。

当前 `splits_cifar100_60_40.json` 是固定的 60/40 类别划分，不是按照 coarse superclass 构造的纯随机难度控制协议。根据 CIFAR-100 的 coarse superclass 标签，40 个未知类中有 35 个与至少一个已知类属于同一 coarse superclass，另外 5 个未知类 `bed`、`chair`、`couch`、`table`、`wardrobe` 来自已知类没有覆盖的 coarse superclass。因此当前测试集混合了细粒度相似未知类和语义上未覆盖的未知类，整体指标会掩盖不同未知类型的差异。

在 `seed=3407` 的 Standard KD 全量测试中，使用相同检测器对未知类别做子集分析：与已知类共享 coarse superclass 的未知子集 AUROC 约为 `0.561`，未共享 coarse superclass 的未知子集 AUROC 约为 `0.492`。这说明类别划分会影响结果，必须报告划分敏感性；但它也没有证明重新划分就能解决特征重叠问题。当前主问题仍然是模型学习到的已知表征和开放集分数没有形成稳定分离。

因此不立即删除现有 CIFAR-100 实验，也不立即用新数据替换它。现有划分继续作为主基准，新增以下补充协议：

1. **当前混合划分**：保留现有固定 60/40 划分，用于保证历史结果连续，并作为主实验协议。
2. **语义困难划分**：尽量让每个 coarse superclass 同时包含已知类和未知类，例如每组 3 个已知类、2 个未知类，共 60/40，用于集中测试细粒度相似未知类的检测能力。
3. **语义隔离划分**：选择 12 个 coarse superclass 作为已知类、8 个 coarse superclass 作为未知类，共 60/40，只作为补充实验，不能替代困难划分或主协议。
4. **跨数据集验证**：在 CIFAR-100 协议稳定后，再使用项目计划中的 CUB-200-2011 验证细粒度新类发现；后续可再考虑 Tiny-ImageNet 或 ImageNet-100。

新的类别划分必须在训练前固定，不能根据测试结果挑选最有利的类别。每个协议应保持相同的模型、训练轮数、检测器和阈值校准规则，并至少运行 3 个随机种子。除总体 AUROC、FPR95、known accuracy 和 unknown reject rate 外，还要按未知类别类型报告结果，并分别报告 oracle-K 和 auto-K 聚类指标。

### 后续执行顺序

1. 生成并审查语义困难划分和语义隔离划分，保存为独立 JSON，不修改现有主划分。
2. 在三个划分上先做小规模 smoke 实验，检查训练、检测、聚类和指标保存是否正常。
3. 如果不同划分下的趋势一致，再进行固定协议的三 seed 完整实验。
4. 对已知/未知特征距离、分数分布、每类误拒与误接收进行分组分析，判断问题来自类别相似性、表征能力还是不确定性估计。
5. 在协议稳定后，将表现最稳定的方案迁移到 CUB-200-2011，而不是用新数据集掩盖 CIFAR-100 上尚未解释的问题。

当前判断是：需要重新划分已有数据集进行敏感性分析，但不应抛弃现有数据；暂时不需要马上下载新数据。数据协议审计是为了确认方法是否稳健，不是为了寻找一个更容易得到高指标的划分。

## 本轮代码更新：可复现的数据协议生成与校验（2026-09-28）

为把上面的数据协议审计落到代码中，本轮新增了：

- `scripts/make_cifar100_splits.py`：读取官方 CIFAR-100 的 fine/coarse 标签，生成三套固定 60/40 类别协议：
  - `random`：带随机种子的随机划分基线；
  - `semantic_hard`：每个 coarse superclass 选择 3 个已知类、2 个未知类，集中测试细粒度相似未知类；
  - `semantic_isolated`：12 个完整 coarse superclass 作为已知类、其余 8 个作为未知类，作为语义隔离补充协议。
- `novel_discovery/data.py`：加载 split JSON 时校验类别是否重复、是否交叉、是否覆盖全部类别，以及 `num_known` 是否匹配。错误协议会在训练前直接报错。
- `tests/test_core_behaviors.py`：增加协议结构和非法 split 校验测试。

生成命令：

```powershell
python scripts/make_cifar100_splits.py --data-root .\data --output-dir .\splits_cifar100_protocols --seed 42
```

使用示例：

```powershell
python train.py inspect_data --dataset cifar100 --data-root .\data `
  --num-known 60 `
  --split-path .\splits_cifar100_protocols\cifar100_60_40_semantic_hard.json `
  --limit-train 4 --limit-val 4 --limit-test 8 --image-size 64 --num-workers 0 --device cpu
```

本轮验证结果：三套协议均成功生成并被 `inspect_data` 读取，均为 60 个已知类和 40 个未知类；Python 编译通过，测试为 `65 passed`。这一步没有声称提升模型指标，它解决的是实验协议可控、可复现和不易误配的问题。下一步应在三套协议上运行相同设置的小规模 teacher/student/discover smoke 实验，然后再决定是否进行三 seed 正式比较。

## 本轮代码更新：已知类分层切分（2026-09-28）

审计发现，原流程按全部已知样本随机切分 train/val。在完整 CIFAR-100 上通常不会漏掉类别，但在 `limit-train`、toy smoke 或迁移到样本更少的数据集时，某些已知类可能只出现在训练集或验证集，导致实验结果混入类别覆盖差异。

现已新增：

- `--known-split-mode random`：历史默认行为，保持旧实验可复现；
- `--known-split-mode stratified`：按已知类别分层切分，尽量保证每个有多个样本的已知类同时出现在 train 和 val。

教师训练、学生训练、discover 和 `inspect_data` 都使用同一个参数，避免不同阶段读取出不一致的训练/验证协议。正式的新协议实验建议显式使用：

```powershell
--known-split-mode stratified
```

该修改只减少数据切分噪声，不直接提升未知检测 AUROC；它的作用是让后续比较更能反映算法本身，而不是某个类别是否偶然进入验证集。本轮新增测试后共 `68 passed`。下一步应在三套 CIFAR-100 类别协议上用相同参数进行小规模对照，并同时固定 `known-split-mode stratified`。

另外，`limit-train` 和 `limit-val` 的限量抽样也已接入同一分层逻辑：当样本预算不少于类别数时，会先为每个已知类保留一个样本，再补足剩余预算。若预算小于类别数，则无法保证所有类别都出现，程序会按预算尽可能保持类别覆盖。实际检查中，`limit-train 64` 在 60 个已知类协议下仍覆盖 60 类；`limit-val 32` 覆盖 32 类，符合预算约束。

标签统计对 `Subset` 和 `ConcatDataset` 使用底层索引递归获取，不读取图像、不触发随机增强，避免分层采样过程污染训练前的随机状态。

## 本轮代码更新：已知类别原型间隔约束（2026-09-28）

针对已知类特征重叠问题，在已有 prototype alignment 之外新增了可选的 `prototype_repulsion_loss`。它计算分类器原型之间的余弦相似度，只惩罚超过指定 margin 的类别对：

```text
L_repulsion = mean(ReLU(cos(w_i, w_j) - margin)), i != j
```

该方法只约束已知类别原型，不把未标注样本错误当作未知样本，因此适合作为第一步表征改进。教师和学生均支持：

```powershell
--alpha-proto-repulsion 0.05 --proto-repulsion-margin 0.0
```

默认权重仍为 `0`，历史基线不变。toy smoke 中教师和学生日志均出现非零 `proto_repulsion`，并成功保存 checkpoint；discover 入口也正常运行。但该 smoke 只有 1 个 epoch、64 个样本，不能证明 AUROC 提升。下一步应在固定 CIFAR-100 协议和相同训练设置下，与不加该项的 baseline 做小规模配对实验，再决定是否进入正式多 seed 实验。

### CIFAR-100 配对实验结果（seed=3407）

在 CIFAR-100 60/40、预训练 ResNet-34/ResNet-18、3 epochs、1200/300/1000 样本和相同 `normalized_entropy_mahalanobis` 检测器下，完成了 baseline 与 `alpha_proto_repulsion=0.05` 的配对实验：

| 方法 | AUROC | FPR95 | known accuracy | known accept | unknown reject | candidate purity | cluster ACC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.5335 | 0.9289 | 0.2264 | 0.9074 | 0.0608 | 0.3000 | 0.5500 |
| prototype repulsion, alpha=0.01 | **0.5536** | **0.9008** | **0.2529** | 0.9455 | 0.0582 | **0.4107** | 0.6250 |
| prototype repulsion | **0.5459** | **0.9041** | 0.2116 | **0.9504** | 0.0481 | **0.3878** | **0.7143** |

该结果说明原型间隔约束确实改变了特征空间。`alpha=0.01` 在本次小实验中比 `0.05` 更平衡：AUROC、FPR95、known accuracy 和候选纯度均优于 baseline，unknown reject rate 仅略低于 baseline；`alpha=0.05` 的聚类 ACC 更高，但 known accuracy 和 unknown reject rate 下降更明显。因此只能判定为“排序和聚类的局部改进”，不能宣称已经解决未知检测。完整记录见 `analysis/proto_repulsion_compare_s3407.md`。

下一步不立即继续增大权重，而是先在相同已知覆盖率下比较三种模型，再用至少两个额外 seed 优先验证 `alpha=0.01` 的稳定性。

下一步不立即继续增大权重，而是：

1. 在相同已知覆盖率下比较两种模型，区分真正的排序改善和阈值尺度变化；
2. 在相同设置下补跑至少两个 seed，并测试 `alpha=0.01/0.05`，确认收益是否稳定且不继续损害 known accuracy。

## 本轮继续审计：mixed 池 rejector 与 MC 不确定性增强（2026-09-28）

本轮围绕核心问题“已知/未知特征与分数分布重叠、未知拒绝率低”继续进行公平验证。重点不是继续移动阈值，而是检查独立 rejector 是否能够在更接近真实开放环境的 mixed 无标签池上工作。

### 1. 标准 KD + support-augmented rejector

在完整 CIFAR-100 60/40 测试协议下，使用标准 KD 学生模型、mixed discovery pool、留出的 known open-validation 样本训练 rejector，并把已知特征支持边界分数加入 rejector 输入。结果为：

| 协议 | AUROC | FPR95 | known accept | unknown reject |
| --- | ---: | ---: | ---: | ---: |
| 纯未知池，support-augmented | 0.7199 | 0.7288 | 0.9472 | 0.1638 |
| mixed 池，严格 open-validation support-augmented | 0.5108 | 0.9377 | 0.9598 | 0.0365 |

这说明纯未知池上的较好结果不能直接代表真实 mixed 场景。mixed 池中已知样本会污染 rejector 的“未知”训练侧，且当前学生特征本身没有形成稳定的已知/未知边界；继续只调阈值或只扩大 rejector 并不能解决这个问题。

### 2. MC Dropout 不确定性增强 rejector

新增了 `uncertainty_augmented` 和 `support_uncertainty_augmented` 两种特征模式，并新增 `--rejector-mc-samples`。新模式将以下信号联合输入 rejector：

- 学生特征与分类 logits；
- 辅助 uncertainty head；
- MC Dropout 的 epistemic uncertainty；
- expected entropy 与 aleatoric uncertainty；
- 可选的 class-conditional support score。

该设计参考 Gal and Ghahramani 的 MC Dropout 不确定性估计，以及 Kendall and Gal 对 epistemic / aleatoric uncertainty 的区分。它用于检验“不确定性信号是否能补足特征重叠”，并不声称已经完整实现贝叶斯模型。

在同一 mixed 协议下，`--rejector-mc-samples 4` 的结果为：

| 方法 | AUROC | FPR95 | known accept | unknown reject |
| --- | ---: | ---: | ---: | ---: |
| support-augmented | 0.5108 | 0.9377 | 0.9598 | 0.0365 |
| support + MC uncertainty augmented | 0.5084 | 0.9388 | 0.9577 | 0.0395 |

结果没有改善整体排序，未知拒绝率仅有很小变化。因此当前 MC 不确定性信号不能单独解决特征重叠问题，也不能据此宣称“不确定性建模有效提升了未知检测”。

### 3. 当前判断与修改方向

当前代码检查通过：`83 passed`，并完成 `compileall` 检查。今天的实验进一步确定：

1. 纯未知 discovery pool 是受控上限，不应作为真实 mixed 开放环境的最终结果；
2. 独立 Logistic rejector 在纯未知池有效，但在 mixed 池明显失效；
3. support boundary、虚拟未知、PU、简单伪未知筛选和 MC 不确定性增强都没有稳定解决 mixed 场景；
4. 核心瓶颈仍是学生特征空间没有把未知样本推到已知分布之外，而不是阈值或单一检测分数选择错误；
5. Standard KD 目前仍是较稳定的表示学习基线，不确定性 KD 只能保留为消融项，不能提前宣称优于 Standard KD。

后续优先方向：

- **先改训练协议**：固定 `stratified` 已知类划分，使用完整训练集、至少 3 个 seed，并分开报告 pure-unknown 与 mixed 两种协议；
- **再改表征学习**：在 Standard KD 基础上逐一验证 supervised contrastive、Proxy Anchor、prototype repulsion、特征蒸馏和 angular margin，观察最近已知原型距离与已知/未知分布重叠是否真的改善；
- **改造 mixed 学习方式**：参考 Vaze 的 GCD、Fini 的 UNO、Wen 的 SimGCD 和 FixMatch 的高置信伪标签思想，不再把 mixed 池全部当作未知，而是使用统一 known+novel 空间、周期性伪标签更新、类别均衡和双视图一致性；
- **再考虑 rejector**：若表征空间仍重叠，应停止扩展 rejector；只有在表征改善后，再比较 PU、可靠负样本、energy、Mahalanobis 和 OpenMax 等检测器；
- **严格记录失败实验**：AUROC、AUPR、FPR95、OSCR、known accuracy、known accept、unknown reject、候选纯度和 auto-K 必须同时报告，不能只挑选单个变好的指标。

本轮新增代码主要位于 `novel_discovery/pipeline.py` 和 `train.py`，默认模式保持不变；新 rejector 特征模式只通过命令行显式启用。
