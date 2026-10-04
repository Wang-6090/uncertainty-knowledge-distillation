# 方法实现忠实度审计与 Sinkhorn 配对复核（2026-09-29）

## 本轮要回答的问题

之前报告把部分实验简称为“novel-mass + weighted Sinkhorn”，但方法名称和运行代码是否一致？邻域支持、novel-mass 与检测指标是否确实测量了同一机制？本轮先检查调用链和保存配置，再做单因素重训。评估标签只用于事后诊断，不进入训练、阈值拟合或模型选择。

## 实现审计：发现的解释偏差

### 1. 原所谓 Sinkhorn 对照并非真正的加权/不加权对照

`joint_novel_mass=True` 会构造 `joint_sample_weights`，随后此前的 `joint_discovery_loss` 将这些权重同时传入一致性损失中的 Sinkhorn target、balance / information / neighbor loss 和最终逐样本 loss reduction。代码里没有独立的 weighted-Sinkhorn 开关。因此旧报告里“原始等权 Sinkhorn vs 加权样本边际”的命名没有被配置文件证明；两边可能都用了加权 Sinkhorn。旧实验的 Sinkhorn 因果结论应撤回，旧数字只能视作不同 checkpoint 的探索性结果，不能用于断言加权 Sinkhorn 负面或正面。

### 2. Novel-mass 训练权重不等于实际检测分数

训练权重实际为：

`softmax(concat(known_logits / known_temperature, novel_logits)).sum(novel classes)`

然后乘跨增强一致性；启用邻域支持时，再乘当前 mixed-pool batch 中 k 个最近邻的平均 novel mass。它是软样本权重，不是已校准的 unknown posterior。原先比较的主检测分数是 `normalized_entropy_mahalanobis`，所以这些检测差异只能说明训练正则改变了学生表示/分数，不能单独证明 novel-mass 这个量有好的 unknown-detection 能力。

另外 known classifier logits 和 novel prototype logits 的尺度没有单独校准；把它们拼到同一 softmax 后，概率质量未必有严格的概率解释。

### 3. “邻域支持”具体是 batch-local，不是全局邻域图

实现使用每个训练 minibatch 当前增强特征的余弦相似度，排除自身后取 top-k；支持度为邻居 novel-mass 均值。它没有 memory bank / 全数据近邻图，也不是直接照搬 SCAN 的全局邻居图一致性优化。因此旧结论只适用于 batch-local reweighting，不能扩展为“SCAN式全局邻域方法有效/无效”。

### 4. 加权 Sinkhorn 的旧选项耦合了两件事

此前 `sample_weights` 会同时改变 Sinkhorn transport 的样本边际和伪标签 loss 的加权平均。为拆分假设，新代码增加 `--joint-weighted-sinkhorn` / `--no-joint-weighted-sinkhorn`：该开关只控制 Sinkhorn 伪标签是否使用软样本边际，loss reduction 继续使用 novel-mass 样本权重。这样才可单独检验“加权 transport target”的作用。

## 新的受控实验

- CIFAR-100，固定 random 60/40 类划分，seed 42；同一个 ResNet-34 teacher、预训练 ResNet-18 student。
- mixed discovery pool，known discovery 子集与监督训练集分离，known pool ratio 0.2。
- 两组均为 1200/300/1000 train/known-validation/test 子集、1200 discovery pool、3 epochs、batch 64、同一 KMeans 原型初始化和相同损失。
- 两组都启用 novel-mass + batch-local kNN 支持（k=5）；**唯一意图变量**是 Sinkhorn transport target 是否使用样本权重。
- 测试集固定为 605 known / 395 novel。Novel-mass 直接检测的阈值只由 300 个 known validation 样本的 novel-mass 95% 分位数确定。
- 另行使用同一 `normalized_entropy_mahalanobis` 检测器与 known-only 95% validation coverage 阈值，复核项目此前主检测协议。
- 测试标签用于 AUROC/FPR95、逐类错误和事后精确覆盖率统计；不用于训练/阈值拟合。表中 test-known 精确 95% coverage 是事后诊断，不是可部署阈值结果。

## 结果：Novel-mass 直接检测

| Sinkhorn transport | AUROC | FPR95 | known acc | validation阈值下 test known accept | validation阈值下 unknown reject | test标签事后精确95% coverage下 unknown reject | 候选纯度（事后精确95%） |
|---|---:|---:|---:|---:|---:|---:|---:|
| 等权 target（`--no-joint-weighted-sinkhorn`） | 0.5664 | 0.8711 | 0.2645 | 94.21% | 6.84% | 6.08% | 43.64% |
| 加权 target（`--joint-weighted-sinkhorn`） | 0.5749 | 0.8165 | 0.2727 | 92.40% | 7.34% | 6.08% | 43.64% |

单 seed 下，加权 target 的 AUROC 增加 0.0085、FPR95 降低 0.0545，known accuracy 增加约 0.83 个百分点；但在事后近似匹配 95% known coverage 后，unknown rejection 与候选纯度相同。有限样本且规则使用 `score <= threshold`，项目统一线性分位数校准在测试集上约达到 94.88% 而非数学上的精确 95%。validation 阈值下，加权组 test known acceptance 只有 92.40%，提示校准迁移/小验证集波动会影响工作点指标。因此这个信号是“排序略有改善的候选”，不是未知召回改善，也远不足以证明普遍有效。

两组都有明显分数重叠：等权组 known/unknown 中位 novel mass 分别为 0.199/0.226；加权组为 0.403/0.457。分布尺度发生明显变化，进一步说明直接用拼接 logits 的 softmax mass 对校准及训练路径敏感。多数高误接受 novel 类仍全部被接收，说明类间差异也未消失。

## 结果：原项目主检测器复核

| Sinkhorn transport | AUROC | FPR95 | known accuracy（all） | test known accept | unknown reject |
|---|---:|---:|---:|---:|---:|
| 等权 target | 0.5546 | 0.9074 | 0.2231 | 95.04% | 6.58% |
| 加权 target | 0.5357 | 0.8777 | 0.2545 | 94.71% | 4.56% |

在 Mahalanobis 主分数下，加权组的 FPR95 改善，但 AUROC 和未知拒绝率变差，且 known accuracy 上升。不是一致胜出。最终判断：加权 Sinkhorn 对当前低拒绝率核心问题**尚未证明有效**；该单 seed 小预算结果只能用于筛选方向。

## 对早先尝试的结论如何修正

- 旧 `weighted_sinkhorn_recheck_s42_20260929.md` 中“等权 vs 加权 Sinkhorn”的对照定义与当前代码调用关系不符，不能再作为有效因果实验引用。
- 旧邻域支持对照的实现确实是 batch-local kNN 乘 novel-mass 权重，但其检测分数仍是 Mahalanobis；它检验的是正则对表示/主分数的间接影响，不是 novel-mass 检测能力。
- 直接 novel-mass 检测之前没有对这几组 checkpoint 做过同协议评估；本报告首次补上这项检验。
- 这次只在 Sinkhorn assignment 的目标权重机制上做了隔离；总训练仍有随机增强和优化随机性。需要多 seed 才能判断小幅差异是否稳定。

## 代码与验证

- 新增 `--joint-weighted-sinkhorn` / `--no-joint-weighted-sinkhorn`，将 target transport 与 loss sample weighting 解耦。
- 新增 `scripts/diagnose_novel_mass.py`，对齐已知验证阈值并直接测 novel-mass；输出包括候选纯度、召回、分数分位数和误接受最多的 novel 类。
- 报告数据：`analysis/sinkhorn_weighting_direct_eval_s42.json`；运行目录为 `runs/audit_sinkhorn_unweighted_pretrained_s42_e3`、`runs/audit_sinkhorn_weighted_pretrained_s42_e3` 及各自 detect 子目录。
- 初次针对性测试为 17 passed；后续增加数值稳定性测试、统一诊断脚本阈值和 FPR95 实现后，全套测试为 94 passed，编译及 diff 检查通过。

## 本轮额外交叉检查

- 旧 neighbor-support 的三组 config 确认：novel-mass 与 batch-local neighbor 组除该开关外主要训练参数一致；EMA 组额外把 discovery selection model 从 student 改为 EMA，因此它检验的是 EMA 权重来源替换，而不是独立的 EMA 模型整体收益。相应结果仍是单 seed 诊断。
- 旧报告中的“等权/加权 Sinkhorn”文件已在开头标注因果对照无效；“neighbor/EMA”记录已补充 detector 使用 Mahalanobis、邻域只在 batch 内计算的限定。保留历史数值，但禁止继续沿用错误因果表述。
- 新诊断脚本现在使用项目统一的 `calibrate_coverage_threshold`、`compute_fpr95`、`compute_oscr`，修正了最初版本使用 `higher` 分位数及 sklearn ROC 首点的实现差异。重新生成的 JSON 是最终可引用数值。
- Sinkhorn 指数核现在先对温度缩放后的 logits 做最大值平移；新增低温度/大 logits 数值测试。
- README 对 UNO/SimGCD/SCAN 的表述改为“借鉴局部机制”，明确当前实现不是论文完整复现。

## 尚未核实到的边界

- 单 seed 训练中的随机增强、DataLoader 顺序、原型初始化与优化过程仍会造成随机差异。命令在两组中完全一致不代表每个随机操作会逐 batch 相同（训练开关可能改变随机数消耗）。故这次是控制配置下的近似单因素对照，不是多 seed 因果证明。
- 历史 `runs/*/config.json` 记录了大部分配置，但没有完整记录源代码 git commit/hash、有效的每 epoch minibatch 样本 ID 和所有数据变换随机状态。对早期产物只能核实保存配置与本地实现，无法重建确切训练执行轨迹。
- 当前 FPR95 由测试标签计算，是标准的事后 benchmark 排序指标，不参与训练/阈值选择；known-only validation coverage 是可部署校准。额外用 test-known 标签重新校准的诊断指标不可部署，已在字段名中标注 diagnostic-only。
- 因此，“所有历史实验均已完全复现”不成立。本审计修正了目前最直接影响近期 joint-discovery 结论的实现/指标错配；更早的 README 历史记录若没有 run directory、config 和 per-sample scores 支撑，仍需逐项追溯后才能提升为正式结论。

## 方法文献定位与边界

- Caron et al., *Unsupervised Learning of Visual Features by Contrasting Cluster Assignments* (SwAV, NeurIPS 2020)：在线聚类分配及 Sinkhorn-Knopp 平衡；可参考其跨视图预测集群分配机制，但不能据此宣称当前启发式实现等同 SwAV。
- Asano et al., *Self-labelling via Simultaneous Clustering and Representation Learning* (ECCV 2020)：均衡伪标签与表示联合优化；当前代码只借鉴平衡分配思想。
- Van Gansbeke et al., *SCAN: Learning to Classify Images without Labels* (ECCV 2020)：利用近邻图关系做聚类一致性；当前 batch-local top-k 权重是更简化的局部启发式，不等于 SCAN 的全局邻居一致性方法。
- Vaze et al., *Generalized Category Discovery* (CVPR 2022)：已知类监督与未知类别发现共存的评测/目标背景；支持区分 known classification 与 novel clustering，但不能直接验证本项目的检测边界。

## 下一步

1. 暂不调 Sinkhorn iterations、邻居 k 或权重 floor；先对这一真正单因素对照做 seed 123、3407 复核，仍保持 smoke 预算。
2. 同时报 novel-mass 和固定 Mahalanobis 的 AUROC、FPR95、validation-calibrated known coverage 下 unknown rejection、OSCR/known accuracy；不以单一 AUROC 判胜。
3. 若只有分数尺度/阈值变化而 matched-coverage recall 不升，停止把 weighted Sinkhorn 当作解决核心检测问题的方向，转向 mixed-pool candidate purity 和表征诊断。
4. 对 known/novel logits 做独立尺度诊断；未来任何校准参数只能在训练或 known validation 上确定，不能根据 test novel labels 调整。
5. 之后才用 held-out labels 事后计算候选纯度、邻域 known/novel 混合率、类条件召回与簇纯度，决定主要瓶颈是候选污染还是表征重叠。
