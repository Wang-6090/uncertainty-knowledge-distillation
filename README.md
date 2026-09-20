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
- A full open-validation run used 20% of the CIFAR-100 open test pool for score selection and evaluated the remaining 8,000 images. Automatic score selection chose `unified_novel_mass`: AUROC `0.6743`, AUPR `0.5607`, FPR95 `0.8115`, known accuracy `0.4892`, and unknown reject rate `0.1474`.
- On the same checkpoint, `projection_pca` was the strongest tested clustering representation. In a 2,000-image comparison, unknown-only ARI was `0.1639`, compared with `0.1405` for `feature_pca`, `0.0945` for `novel_pca`, and `0.0727` for `unified_pca`. On the full open-validation run, `projection_pca` reached unknown-only ARI `0.0990`, versus `0.0781` for `unified_pca`.
- With `projection_pca` fixed, KMeans still exceeded Agglomerative clustering in the 2,000-image comparison (`0.1639` vs. `0.1242` unknown-only ARI), so KMeans remains the default clustering baseline.
- The command-line default for `--cluster-feature` is now `projection_pca`. This is a provisional single-seed choice and must be checked with multiple seeds before being treated as a final method conclusion.
- The open-validation result is only a small improvement over the earlier fixed-score baseline. Unknown detection and clustering remain the main research problems; the next formal step is a controlled multi-seed ablation.
- The classwise score passed unit-level numerical checks but performed worse in the first controlled comparison. No performance claim is made for it.
