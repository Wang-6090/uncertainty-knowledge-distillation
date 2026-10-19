# 基于不确定性知识蒸馏的新类发现方法

## 2026-10-18：边界 VOS + rejection BCE 配对复核——未获得稳定收益

为验证“困难边界虚拟未知上的 rejection-head BCE”是否真的改善核心的
已知/未知分数分离，在 seed=44 和 seed=123 分别做了配对对照。两臂均固定
CIFAR-100 semantic-isolated 60/40、4 epochs、nearest-class VOS、
`alpha_vos=0.01`、边界比例 25%、相同训练/数据协议、检测器和验证集
95% known-coverage 校准；唯一训练变量是 `alpha_vos_rejection`（0 vs 0.05）。
配置审计确认除损失权重、输出路径和随训练产生的最佳验证准确率元数据外，
其余训练参数相同；treatment 训练历史中 `vos_rejection` 非零，说明损失实际生效。

| seed | BCE | AUROC | FPR95 | OSCR | KS | unknown below known-q95 | known above unknown-q05 | 已知接受率 | 未知拒绝率 | 接受后已知准确率 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 44 | 关 | 0.89524 | 0.38033 | 0.65273 | 0.64292 | 0.54375 | 0.38033 | 93.92% | 51.45% | 72.64% |
| 44 | 0.05 | 0.90118 | 0.34583 | 0.65685 | 0.64700 | 0.51500 | 0.34583 | 94.92% | 49.08% | 72.36% |
| 123 | 关 | 0.89518 | 0.40750 | 0.64017 | 0.63950 | 0.50900 | 0.40750 | 95.18% | 48.20% | 71.00% |
| 123 | 0.05 | 0.89078 | 0.41417 | 0.63212 | 0.63025 | 0.53700 | 0.41417 | 95.73% | 42.75% | 70.40% |

seed=44 的排序与 overlap 有小幅改善，但 seed=123 反向，未知拒绝率在两个 seed
都下降（分别 −2.37、−5.45 个百分点）。因此这不是稳定有效的拒识改进，不能只
凭 seed=44 的 AUROC 或已知接受率得出正面结论。**不将该 BCE 设为默认，也不继续
对它扫权重。**保留 boundary VOS 与 rejection BCE 的区别：此前 boundary VOS
本身较有希望，但仍需独立复核；本次结果否定的是将 BCE 直接施加到合成虚拟未知上
能稳定帮助 rejector 的假设。接下来按未知细类及其最近已知类分析误接收，再设计
有可靠非测试监督依据的类条件拒识信号；所有实验仍联合考察排序、分数重叠和已知效用。

此前高风险伪未知筛选方向也仍判为无效：拒绝率提升主要来自误拒已知样本，而非分数
分离改善。 :codex-annotation{index="1"}

完整设计、逐 seed 差值和限制见
[`analysis/vos_boundary_rejection_matched_s44_s123_e4_20261018.md`](analysis/vos_boundary_rejection_matched_s44_s123_e4_20261018.md)。

### 配对实验后的误差邻域诊断（seed=123）

进一步复核了 control 与 BCE=0.05 treatment 的测试误差邻域。测试标签仅用于推理后的
描述性分组，没有参与训练、校准或方法选择。control 有 `2,072/4,000` 个未知样本被误接收
（51.80%），treatment 为 `2,290/4,000`（57.25%）；误接收样本的平均最近已知类 cosine
距离分别为 0.2090 和 0.2230。高误接收集中在 lamp、snake、clock、kangaroo 等少数未知类，
常见近邻包括 squirrel、shrew、rabbit、mouse、shark 等。也就是说，错误有局部类别/语义集中性，
但单纯更远的最近邻距离并没有对应更强拒识，不能据此直接把 kNN 距离加进默认检测分数。

这项诊断不挽救 rejection BCE：treatment 比 control 多误接收 218 个未知样本，与两 seed 配对
实验中未知拒绝率均下降的结论一致。下一步先在冻结 checkpoint 上，用**已知验证集**检查预测类
条件分数校准和局部支持量与已知误拒之间的关系；任何类别条件映射都要报告按类样本数、校准误差、
排名/重叠指标及已知效用，并与全局校准对照。验证通过后才值得设计训练期算法实验；不能按测试集
未知类别拟合阈值，也不能把测试集分组结果当作监督信号。详细统计、协议和限制见
[`analysis/vos_boundary_rejection_error_neighborhoods_s123_20261018.md`](analysis/vos_boundary_rejection_error_neighborhoods_s123_20261018.md)。

### 冻结模型的类条件阈值诊断（seed=123）

为检查类别间 rejector 分数尺度差异是否导致未知误接收，固定上述 BCE-off checkpoint、
`feature_rejector` 分数、数据划分与模型，只将全局阈值换成按预测已知类分别取已知验证分数
95 分位数的阈值（每类至少 5 个验证样本；不使用未知验证/测试标签校准）。这是校准策略诊断，
不是训练算法实验；阈值不改变分数排序，因此 AUROC 等排序指标应保持不变。

| 阈值策略 | AUROC | FPR95 | OSCR | 测试已知接受率 | 测试未知拒绝率 |
|---|---:|---:|---:|---:|---:|
| 全局 95% known-coverage（原配对基线） | 0.89518 | 0.40750 | 0.64017 | 95.18% | 48.20% |
| 按预测类条件 95 分位数 | 0.89518 | 0.40750 | 0.64017 | 88.45% | 49.65% |

类条件阈值只多拒绝 1.45 个百分点未知样本，却多误拒 6.73 个百分点已知样本；实际已知覆盖
也明显偏离名义 95%。因此它没有解决特征/分数重叠，不设为默认，也不继续用测试集选阈值或
调收缩系数。后续若要研究类条件校准，先在已知验证数据内部做独立校准/评估拆分和阈值稳定性
诊断；算法主线仍应回到能改善排序与分数重叠、同时维持已知效用的训练期方法。复现实验命令与
限制见 [`analysis/class_conditional_threshold_pilot_s123_20261011.md`](analysis/class_conditional_threshold_pilot_s123_20261011.md)
（历史阈值试验）以及本轮运行配置 `runs/vos_boundary_rejection_classcond_s123_e4_feature_rejector/`。

## 2026-10-18：候选筛选后的有效先验校正——实现正确，但未改善核心分离

上一轮审计发现：如果 kNN 边界损失只作用于候选筛选后的 mixed pool，直接使用原始已知先验可能使 PU 风险参考分布不匹配。因此新增可选开关 `--discovery-knn-selection-conditional`，按

`pi_selected = pi × P(S|known) / P(S|mixed)`

计算筛选后的有效先验，并对已知样本使用与 mixed pool 相同的单视图候选规则。该开关默认关闭，只在 mixed pool、candidate gating、`hard_nnpu/soft_nnpu`、kNN 边界损失开启且 cross-view gating 关闭时生效。训练日志新增 mixed/known selection rate 和 effective prior，合成数据审计恢复误差约为 `1.98e-7`，说明公式和 wiring 正确。

随后完成了 seed=44 的严格配对实验：CIFAR-100 semantic-isolated 60/40、4 epochs、mixed pool=5400、known prior=0.2、shared-boundary 候选空间、hard-NNPU、同一检测器和验证集 95% known-coverage 校准，唯一变量是该开关。

| 条件 | AUROC | FPR95 | OSCR | 已知接受率 | 未知拒绝率 | KS | unknown below known-q95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原始先验 | 0.90048 | 0.35550 | 0.65797 | 95.783% | 42.450% | 0.65717 | 53.600% |
| 筛选条件先验 | 0.89906 | 0.36317 | 0.65904 | 95.350% | 45.750% | 0.64725 | 51.325% |

有效先验确实恢复到 `0.2`，但 AUROC、FPR95、KS 没有改善；未知拒绝率上升伴随已知接受率下降，主要是工作点变化，而不是已知/未知分布真正分离。该方法不设为默认，也不再继续扫描先验或阈值。完整记录见 [`analysis/selection_conditional_pu_wiring_20261018.md`](analysis/selection_conditional_pu_wiring_20261018.md)，汇总见 [`analysis/selection_conditional_pu_s44_e4_summary.json`](analysis/selection_conditional_pu_s44_e4_summary.json)。

本轮还修复了实验汇总脚本在 Windows 下无法序列化 `Path` 的问题，并补充了报告字段映射，避免实验实际完成后因记录失败而被误判为未完成。当前核心问题仍是 semantic-near unknown 与已知类在共享表示和 rejector 分数上的重叠；下一步应停止调 PU 先验，转向直接的类条件拒识监督或针对 hard unknown 的表示几何约束，并继续用 AUROC、FPR95、OSCR、KS、overlap 和已知效用共同判断。

## 2026-10-18：困难边界虚拟未知（VOS）seed 复核——收益未复现

前面对 inter-class VOS 的审计表明，对所有已知样本生成虚拟未知可能扰动正常已知区域。因此本轮只使用已知分类 margin 最低的 25% 样本生成 nearest-class 虚拟未知，固定 `alpha_vos=0.01`、`alpha_vos_rejection=0`，与同脚本的 no-VOS control 配对。

| 条件 | AUROC | FPR95 | OSCR | 已知接受率 | 未知拒绝率 | KS | unknown below known-q95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| no VOS control | 0.89218 | 0.38883 | 0.65732 | 95.833% | 43.600% | 0.63058 | 53.325% |
| nearest-class + boundary 25% | **0.89919** | **0.36667** | **0.65976** | 94.133% | **52.850%** | **0.64617** | **52.800%** |

这次 AUROC、FPR95、OSCR、KS 和重叠代理均向好的方向移动，比“筛选条件先验校正”更有希望；但已知接受率下降 1.70 个百分点，仍不能只凭未知拒绝率上升判定解决核心问题。完整结果见 [`analysis/vos_boundary_fraction_control_s44_e4_20261018.md`](analysis/vos_boundary_fraction_control_s44_e4_20261018.md)。

同时修复了 VOS 的一个 wiring 问题：`vos_boundary_fraction` 原来只作用于 Energy VOS，若开启 `alpha_vos_rejection`，拒识头仍会对全部虚拟样本施加 unknown 监督。新增 `weighted_unknown_bce`，使两条分支使用相同边界 mask；默认 `alpha_vos_rejection=0`，因此不改变默认流程。对应测试验证了被屏蔽样本没有梯度。

随后在 seed=123 完成同协议复现（4 epochs、nearest-class VOS、boundary fraction=0.25、`alpha_vos=0.01`、`alpha_vos_rejection=0`，仅比较 no-VOS 与 boundary VOS）：

| seed | 条件 | AUROC | FPR95 | OSCR | 已知接受率 | 未知拒绝率 | KS | unknown below known-q95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | no VOS control | 0.90069 | 0.37983 | 0.64926 | 96.20% | 44.18% | 0.64883 | 49.65% |
| 123 | nearest-class + boundary 25% | 0.89193 | 0.40950 | 0.63662 | 96.08% | 43.08% | 0.63350 | 50.85% |

seed=123 中 treatment 相对 control 的 AUROC 下降 0.00876、FPR95 恶化 0.02967、OSCR 下降
0.01264、KS 下降 0.01533，未知拒绝率下降 1.10 个百分点。seed=44 的正向变化因此没有复现，
目前不把 boundary VOS 认定为稳定有效方法，不扩大 VOS 权重/比例搜索；它也不应与 BCE 叠加，
因为 rejection BCE 的双 seed 配对结果同样不稳定。接下来回到表示/拒识机制层：检查已知分类
margin 所选边界样本是否真对应 semantic-near unknown 的局部困难区域，以及 VOS 梯度对已知分类
边界和 rejector 分数的影响，再据此设计新的一因素训练试验。完整结果见
[`analysis/vos_boundary_vos_replication_s123_20261018.md`](analysis/vos_boundary_vos_replication_s123_20261018.md)。

### VOS source-gradient 配对实验（seed=44/123；方向未复现，停止此分支）

在梯度路径机制 smoke test 确认 detach 只切断虚拟特征构造路径之后，完成 seed=123 配对并对 seed=44 复核。两 seed 均固定相同 teacher/初始化协议、semantic-isolated 60/40 划分、nearest-class VOS、boundary 25%、`alpha_vos=0.01`、`alpha_vos_rejection=0`、mixed discovery pool、检测器与验证集 95% known-coverage 校准；唯一训练变量是是否让合成虚拟特征对源特征回传梯度。seed=123 的 detached → source-gradient：AUROC `0.88736→0.89541`、FPR95 `0.39533→0.41783`（变差）、OSCR `0.63657→0.64061`、KS `0.61783→0.63600`、unknown-below-known-q95 overlap `56.40%→47.80%`、已知接受率 `95.42%→95.77%`、未知拒绝率 `41.70%→46.80%`。seed=44 则 AUROC `0.89494→0.89261`、FPR95 `0.38033→0.37400`、OSCR `0.65610→0.65129`、KS `0.63742→0.63783`、unknown-below-known-q95 overlap `52.43%→53.58%`、已知接受率 `94.37%→95.07%`、未知拒绝率 `49.48%→46.20%`。排序、overlap 与工作点指标方向跨 seed 不一致，因此不构成可靠改进，核心分数重叠仍在。

梯度审计（每臂每 epoch 338 个 batch，跨 epoch 均值）在两 seed 都确认：detached 的 VOS 总 encoder 梯度约 `0.217–0.219`，虚拟特征构造路径为 `0`；source-gradient 的总量约 `0.209`，新增虚拟路径约 `0.286`，两路径 cosine 约 `0.66`。source-gradient 新增路径与真实已知 logits 分支存在较强部分抵消；`alpha_vos_rejection=0` 时 rejector 梯度为 `0`。机制稳定存在，但检测收益不稳定。**不改变默认 detach，停止该 source-gradient VOS 分支，不做权重扫描或与 rejection BCE 组合。**下一步回到真实 hard unknown 的类别级错误结构和训练协议可学习信号，不再把合成边界点当作主要突破口。

实验过程中发现短跑 driver 曾继承源配置中的绝对 checkpoint 输出路径，导致覆盖 seed-123 VOS treatment checkpoint；该 1-epoch/512 样本产物未用作性能证据。随后修正输出隔离，并在独立目录完成完整配对和 seed-44 复核。旧的 discovery 指标记录仍是历史结果，但原 checkpoint 文件需要重生成后才能做 checkpoint 级复分析。梯度机制 smoke-test 脚本为 [`scripts/run_vos_gradient_path_audit.py`](scripts/run_vos_gradient_path_audit.py)，完整配对脚本为 [`scripts/run_vos_source_gradient_compare.py`](scripts/run_vos_source_gradient_compare.py)；协议、逐 seed 结果、限制及事故说明见 [`analysis/vos_source_gradient_compare_s123_e4_20261018.md`](analysis/vos_source_gradient_compare_s123_e4_20261018.md)。

### VOS 梯度路径核查（2026-10-18；机制审计，非性能实验）

复查实现后确认，`detach_source=True` 只切断“虚拟特征由真实特征合成”这一条反向路径，并不会让 VOS 完全绕开 encoder：Energy margin 仍以真实已知样本 logits 作为一侧，因此 `loss_vos` 仍能经已知 logits 更新 encoder 和 classifier。此前将该项概括为“只训练分类头”不准确，现已补上训练期分模块梯度审计：记录实际加权 VOS 对 encoder、分类头的梯度范数；另以 detach 的已知 logits 单独记录虚拟特征构造路径梯度；若启用 `alpha_vos_rejection`，也单独审计 rejection 分支梯度。`alpha_vos_rejection=0` 时该拒识头项应为零。

新增确定性单元测试验证：source detach 默认确实消除虚拟构造路径的特征梯度，但完整 energy-margin 仍保留真实已知分支梯度；开启 source-gradient 时虚拟构造路径重新连通。测试通过（`224 passed`，`git diff --check` 通过）。这只是计算图机制核实，没有新训练、没有新性能数据，因此不据此声称 VOS 有效；seed=44/123 的结论仍是收益未稳定复现。下一步如再评估 VOS，先用 `--audit-loss-gradients` 在训练 batch 上查看各路径梯度范数及相对主任务的尺度，再决定是否值得做 detach/source-gradient 配对训练；对比中除梯度开关外固定训练 seed、初始化、数据与校准协议，并联合检查 AUROC、FPR95、OSCR、overlap、固定覆盖拒绝率和已知效用。

## 2026-10-17：support score 与候选池审计补充

在严格 PU-kNN wiring 复核之后，又对同一个 seed=44 学生 checkpoint 做了两项不重训审计：

1. 固定数据划分、混合池、验证集 95% known-coverage 校准和 checkpoint，只切换最终检测分数；当前 `feature_rejector` 的 AUROC 为 `0.9008`，直接 `known_support` 为 `0.7717`，`known_conformal` 为 `0.7161`。这说明简单支持半径或共形支持分数不能替代当前 mixed-pool nnPU rejector，不能把它们作为核心问题的解决方案。
2. 审计 5,400 样本混合发现池的候选质量。当前 `entropy_uncertainty` 候选池已知污染约 `4.96%`、未知纯度 `95.04%`、未知召回 `29.70%`；`distance_consensus` 的纯度最高约 `96.59%`，但未知召回仅 `30.19%`。因此候选池不是“已知污染严重”，反复更换候选 gate 预计不能单独解决分布重叠。

本轮新增报告见 [`analysis/support_score_and_candidate_audit_s44_20261017.md`](analysis/support_score_and_candidate_audit_s44_20261017.md)，审计脚本见 [`scripts/audit_discovery_candidate_pool.py`](scripts/audit_discovery_candidate_pool.py)。`train.py` 现在会把 support radius 的类数、分位数及半径统计写入 `calibration_report.json`，仅用于诊断，不改变默认训练或检测结果。

另外，按预测已知类分别校准 95% 阈值后，未知拒绝率仅由 `47.08%` 提高到 `48.53%`，但已知接受率降至 `87.95%`，AUROC 不变；这进一步说明不能继续把核心问题归因于全局阈值或简单 calibration。

按未知细类的事后统计显示，`clock`、`lamp`、`beaver`、`porcupine`、`possum` 等类别拒绝率仅约 `16%–25%`，失败集中在少数 hard/semantic-near unknown，而不是所有未知类均匀失败。后续算法实验必须增加逐未知类指标和 accepted-unknown 的最近已知类统计。

当前下一步应转向：按未知细类统计拒绝率，定位最容易被接受的 semantic-near unknown；再针对这些样本设计表示学习/类条件边界改动，并用固定验证协议做配对实验。不要把本轮 support score 或候选 gate 审计误写成算法效果。

## 2026-10-17：PU-kNN 边界损失严格 wiring 复核

本轮先审计上一轮实验的 wiring，再进行严格配对复核。原实验配置同时启用了 `alpha_discovery_rejection_boundary=0.1`，因此不能把结果归因于 pure/hard-NNPU kNN 风险形式。本轮修正脚本，显式关闭独立 rejection boundary 和 rejection-kNN boundary，只保留 `alpha_discovery_knn_boundary=0.1`。比较因素仍只有 `discovery_knn_risk=pure` 与 `hard_nnpu`；两臂固定 seed=44、CIFAR-100 semantic-isolated 60/40、4 epochs、混合池 5,400（已知先验 0.2）、共享边界候选空间、同一检测器和验证集 95% known-coverage 校准。

严格复核已完成，报告目录为 `runs/knn_pu_boundary_strict_s44_e4_pure_discover/` 和 `runs/knn_pu_boundary_strict_s44_e4_hard_nnpu_discover/`。

| seed | kNN risk | AUROC | FPR95 | OSCR | 已知接受率 | 未知拒绝率 | KS | unknown below known-q95 | 接受后已知准确率 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 44 | pure | 0.90078 | 0.36233 | 0.65830 | 95.63% | 47.08% | 0.64942 | 50.75% | 72.34% |
| 44 | hard_nnpu | 0.89380 | 0.38633 | 0.66304 | 94.33% | 51.65% | 0.63392 | 51.05% | 73.96% |

`hard_nnpu` 的未知拒绝率提高 4.58 个百分点，但已知接受率下降 1.30 个百分点；同时 AUROC、FPR95、KS 变差，overlap 也略变差，只有 OSCR 和接受后已知准确率上升。因此这不是稳定的分布分离改进，更像是工作点与选择性分类效用的 trade-off。训练日志还显示 hard-NNPU 的 `discovery_knn_boundary` 末轮约为 `0.000405`，而 pure 为 `0.023561`，说明硬截断使该项大部分时间几乎没有有效梯度；不能把更高拒绝率解释为更强的 kNN 边界学习。

代码修改位于 `novel_discovery/losses.py`、`novel_discovery/pipeline.py`、`train.py`、`tests/test_core_behaviors.py` 和 `scripts/run_knn_pu_boundary_control.py`。严格复核脚本现在会在训练和 discover 配置中显式记录四个 boundary 权重，避免历史配置污染。单元测试结果为 `206 passed`。

本轮结论：`hard_nnpu` 暂不设为默认，也不补第二个 seed 来放大一个已出现负向排序信号的方法。下一步优先审计 support radius、候选池的已知污染率/未知召回率，以及 semantic-near unknown 的误接收；若继续研究 PU-kNN，应先改进风险项的梯度有效性并用验证集预注册选择规则，而不是继续调阈值或叠加 gating。

核心判据仍是已知/未知特征与分数分布是否真正分开，而非拒绝率单项升高；当前尚无证据表明本轮 PU-kNN 改善了该问题。

## 2026-10-16：支持集上限修复（已进入默认配置）

上一轮 paired 实验发现，feature rejector 的已知支持最多使用 5,000 个样本时，类条件 support 覆盖不足会加重已知/未知分数重叠。本轮固定 student checkpoint、CIFAR-100 semantic-isolated 60/40、mixed pool=5400、`nu_corrected`、`support_augmented`、linear nnPU、MC=4 和 95% known-coverage 校准，只比较分层 5,000 样本与全部支持样本：

| seed | support | AUROC | FPR95 | OSCR | 已知接受率 | 未知拒绝率 | KS | unknown below known q95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 44 | stratified 5000 → all | 0.88389 → **0.90342** | 0.4027 → **0.3590** | 0.65048 → 0.66667 | 93.28% → **95.37%** | 50.15% → 49.48% | 0.6098 → **0.6512** | 56.35% → **48.40%** |
| 123 | stratified 5000 → all | 0.88964 → **0.90315** | 0.4070 → **0.3812** | 0.63537 → 0.64405 | 96.65% → 95.72% | 39.40% → **50.05%** | 0.6279 → **0.6571** | 51.88% → **45.95%** |

两组 seed 都支持“使用全部已知支持样本”这一修复：排序分离、KS、重叠代理和未知拒绝率整体改善，说明这不是简单阈值调整，而是修复了 rejector 支持集覆盖不足。它仍不能说明共享表征已经完全解决重叠问题，OSCR 和已知分类效用仍需持续监控。因此 `train.py` 的 `--rejector-max-samples` 默认值已从 `5000` 改为 `0`（使用全部可用样本）；显式设置正数仍可复现旧实验或进行容量消融。完整协议与原始结果见 [`analysis/nnpu_support_size_control_20261016.md`](analysis/nnpu_support_size_control_20261016.md)，复现实验见 [`scripts/run_nnpu_support_size_control.py`](scripts/run_nnpu_support_size_control.py)。

同时完成了支持规模参数的 wiring 审计：`hard`、`soft_pu` 和 `nnpu` 三种 rejector 都统一使用 `0/None=全部支持`，并且 hard 分支现在会真正接收 `--rejector-max-samples`；此前 hard 分支忽略了该命令行参数，可能造成实验协议误判。

## 2026-10-16：类条件 support 距离不是充分检测器

为确认 full-support 的收益是否只是来自原型/支持集距离，本轮在同一 student checkpoint、同一 CIFAR-100 semantic-isolated 60/40、同一 mixed pool、同一 95% known-coverage 校准下，将最终分数改为只使用已知类条件 support 距离：

| detector | AUROC | FPR95 | OSCR | 已知接受率 | 未知拒绝率 | KS | unknown below known q95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| full-support `support_augmented` nnPU | 0.90342 | 0.3590 | 0.66667 | 95.37% | 49.48% | 0.6512 | 48.40% |
| known-only class support | 0.76023 | 0.6763 | 0.58271 | 93.82% | 23.23% | 0.3876 | 79.45% |

结论是：类条件 support 距离单独不能解决重叠问题；当前有效因素是 mixed-pool nnPU 的学习，而 `known_support_score` 更适合作为 rejector 输入或诊断量，不应替代最终检测器。完整记录见 [`analysis/known_support_vs_nnpu_s44_20261016.md`](analysis/known_support_vs_nnpu_s44_20261016.md)。下一步固定 full-support、`nu_corrected` 和其他协议，只测试 `rejection_balanced_support_augmented`，检查高维 logit 块是否造成尺度/维度捷径。

## 2026-10-16：平衡高维 logit 特征的复核

为检查 rejector 是否被 60 维分类 logits 的尺度或维度主导，本轮只将 `support_augmented` 换成已有的 `rejection_balanced_support_augmented`，对 logits 逐样本归一化；full-support、`nu_corrected`、linear nnPU、student checkpoint、mixed pool、MC=4 和 95% known-coverage 校准全部固定：

| seed | feature mode | AUROC | FPR95 | OSCR | 已知接受率 | 未知拒绝率 | KS | unknown below known q95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 44 | support → balanced | 0.90342 → **0.90863** | 0.3590 → **0.3318** | 0.66667 → **0.67036** | 95.37% → **96.37%** | 49.48% → 42.40% | 0.6512 → **0.6685** | 48.40% → 49.75% |
| 123 | support → balanced | 0.90315 → **0.90582** | 0.3812 → **0.3488** | 0.64405 → **0.64791** | 95.72% → 94.90% | 50.05% → **52.03%** | 0.6571 → **0.6636** | 45.95% → 48.13% |

AUROC、FPR95、OSCR 和 KS 在两个 seed 都改善，说明特征尺度确实影响 rejector；但 overlap 代理在两个 seed 都变差，且 seed=44 的工作点未知拒绝率下降。因此暂不修改默认 feature mode，只保留该模式作为候选消融，避免把排序指标改善误写为核心重叠问题已经解决。完整记录见 [`analysis/nnpu_balanced_support_control_20261016.md`](analysis/nnpu_balanced_support_control_20261016.md)。

## 2026-10-16：已知支持分层抽样（历史候选，优先级已被 full-support 复核覆盖）

在 nnPU rejector 的已知支持最多抽取 5,000 个样本时，随机抽样可能使不同已知类的支持数量不均，进而加重类条件 support 覆盖不一致。本轮只打开已有的 `--rejector-stratified-known`，其余固定为 `nu_corrected`、`support_augmented`、linear nnPU、同一 mixed pool、同一 student checkpoint、MC=4 和验证集 95% known-coverage 校准。

| seed | 抽样 | AUROC | FPR95 | OSCR | 已知接受率 | 未知拒绝率 | KS | unknown below known q95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 44 | random → stratified | 0.88225 → 0.88389 | 0.4070 → 0.4027 | 0.64918 → 0.65048 | 95.15% → 93.28% | 42.40% → 50.15% | 0.6046 → 0.6098 | 56.73% → 56.35% |
| 123 | random → stratified | 0.88326 → 0.88964 | 0.4362 → 0.4070 | 0.62922 → 0.63537 | 96.37% → 96.65% | 40.18% → 39.40% | 0.6038 → 0.6279 | 52.45% → 51.88% |

两 seed 的排序与分布分离指标均同向改善，说明该方向可能确实减少了已知支持集合的类偏差；但 seed=44 的已知接受率下降约 1.9 个百分点，seed=123 的未知拒绝率略降，因此当时没有直接设为无条件默认。随后完成的 full-support paired 复核在两个 seed 上进一步改善 AUROC、FPR95、KS 和重叠代理，因此当前优先使用全部支持样本，分层 5,000 仅保留为历史 ablation。详细记录见 [`analysis/nnpu_stratified_support_control_20261016.md`](analysis/nnpu_stratified_support_control_20261016.md)，复现实验见 [`scripts/run_nnpu_stratified_support_control.py`](scripts/run_nnpu_stratified_support_control.py)。

## 2026-10-16：nnPU 风险归一化审计（两 seed，结论为候选而非默认）

本轮先核对了 feature rejector 的 nnPU 风险实现，再做严格 paired control。修正内容是去掉历史实现中对 mixed-pool 正类风险的额外 `1/(1-已知先验)` 缩放；已知样本仍作为可靠负类，mixed pool 仍作为未标注样本，不能把这次修正误写成交换类别标签。新增 `nnpu_binary_risk` 公共函数、公式单元测试和复现实验脚本 [`scripts/run_nnpu_formula_control.py`](scripts/run_nnpu_formula_control.py)。

固定学生 checkpoint、CIFAR-100 semantic-isolated 60/40、mixed pool=5400、已知先验=0.2、`support_augmented`、linear rejector、MC=4 和验证集 95% known-coverage 校准，仅比较 `legacy` 与 `nu_corrected`：

| seed | risk | AUROC | FPR95 | OSCR | 已知接受率 | 未知拒绝率 | KS | unknown below known q95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 44 | legacy → nu_corrected | 0.89385 → 0.89999 | 0.3720 → 0.3578 | 0.65765 → 0.66143 | 95.18% → 94.98% | 46.43% → 47.48% | 0.6283 → 0.6445 | 52.88% → 52.53% |
| 123 | legacy → nu_corrected | 0.88511 → 0.88326 | 0.4543 → 0.4362 | 0.63460 → 0.62922 | 96.13% → 96.37% | 42.60% → 40.18% | 0.6145 → 0.6038 | 52.18% → 52.45% |

结论：公式修正是有效的代码和目标审计，seed=44 的整体分离指标有改善，但 seed=123 的方向不一致。因此 `nu_corrected` 暂时作为明确的可复现实验选项保留，不能单独设为“已经解决核心重叠问题”的默认方案，也不应继续只调这个缩放项。详细公式、协议、结果和限制见 [`analysis/nnpu_formula_control_20261016.md`](analysis/nnpu_formula_control_20261016.md)。下一步优先检查 mixed pool 中已知 support 的类均衡与覆盖，而不是继续做阈值扫描。

## 2026-10-15：soft-NNPU treatment 已完成与判断

上轮中断的 `soft_nnpu` treatment 已在严格同协议下补完：服务器、数据划分、教师和学生、混合池、已知先验、4 epochs、检测器和校准方式均保持一致，唯一算法变量是 `hard_nnpu -> soft_nnpu`，边界权重为 `0.1`。训练和 discover 均已完成，完整审计见 [`analysis/rejection_boundary_soft_nnpu_audit_s44_20261015.md`](analysis/rejection_boundary_soft_nnpu_audit_s44_20261015.md)。

soft-NNPU 相对 baseline 的排序指标小幅改善：AUROC `0.8959 -> 0.9000`，FPR95 `0.3667 -> 0.3578`，OSCR `0.6534 -> 0.6614`，已知准确率 `0.7088 -> 0.7155`；但报告阈值下未知拒绝率从 `0.5100` 降至 `0.4748`。严格 95% 已知覆盖率下，paired bootstrap 的 AUROC 差异 CI 为 `[-0.00043,+0.00857]`，未知拒绝率差异 CI 为 `[-3.15,+4.35]` 个百分点，因此不足以支持稳定的总体改进。结论：实现和 wiring 正确，但 soft-NNPU 不能作为已解决核心重叠问题的证据，默认仍保持 `hard_nnpu`，停止继续扫描 risk form。

下一步回到共享 encoder 的类条件 support 几何：先用已知标签估计类内半径和边界，再只用高纯度未知候选施加局部边界约束；每次只改一个损失项，并同时检查 AUROC、FPR95、OSCR、KS、overlap、已知效用和固定覆盖率拒绝率。

## 2026-10-14：soft-NNPU rejection-boundary 复核已暂停

### 本次目的与唯一变化

此前的 PU rejection-boundary loss 使用 `hard_nnpu`：当混合池校正后的风险为负时，直接用 `clamp_min(0)` 截断，可能使部分 batch 的边界损失没有有效梯度。本次只准备比较同一 PU 边界损失的两种风险形式：

- `hard_nnpu`：非负风险硬截断，当前默认行为；
- `soft_nnpu`：用 softplus 平滑硬截断，保留较小的连续梯度。

实验脚本 [`scripts/run_rejection_boundary_epoch_control.py`](scripts/run_rejection_boundary_epoch_control.py) 新增了 `--risk-mode` 参数。除该参数外，计划固定 seed=44、CIFAR-100 semantic-isolated 60/40、相同 teacher、mixed discovery pool、known prior、4 epochs、boundary weight=0.1、相同 rejection 表征、检测器和验证校准协议。测试标签不参与训练、阈值或模型选择。

### 当前状态与可用结论

本轮在 soft-NNPU treatment 完成前被暂停：脚本只重新生成了一个 soft-NNPU 协议下的 baseline，目录为
`runs/rejection_boundary_epoch_control_s44_e4_soft_nnpu_audit_baseline/`；该 baseline 不是新的算法处理组，不能与历史 hard-NNPU 结果直接拼接，也没有产生 soft-versus-hard 的有效实验结论。未完成的 treatment 运行目录和中间 checkpoint 仅作为中止记录保留，不能用于报告或调参。

因此截至今日：

1. 没有证据表明 `soft_nnpu` 比 `hard_nnpu` 更好；默认仍保持 `hard_nnpu`。
2. 没有修改训练主流程的默认配置，也没有根据未完成实验修改 boundary 权重、阈值或 checkpoint 选择。
3. 现有最可靠的方向仍是：PU boundary loss 在严格 4-epoch、两个 seed 上改善了 AUROC/FPR95/KS 和重叠代理，但 5-epoch 延长实验未稳定复现，未知拒绝率的跨 seed 置信区间仍不稳定。因此它是候选消融，不是已经解决核心问题的方法。
4. 核心问题仍是已知/未知表征和分数分布重叠；单独提高拒绝率不能证明问题解决，必须同时观察 AUROC、FPR95、OSCR、已知准确率/接受率、KS 和 overlap。

下一次继续实验时，应先补齐同协议的 soft-NNPU treatment，并与同一 soft 协议下的 baseline 比较；如果 soft 只提高未知拒绝率而损害 AUROC、FPR95、OSCR、KS 或已知效用，就停止该方向。若实验未补齐，不应把本节写成 soft-NNPU 的正面或负面算法结果。

本次代码检查：实验脚本已通过 Python 语法检查；未改变默认训练行为。此前完整测试结果为 `236 passed`，当前新增参数只位于实验脚本，主程序默认参数仍保持兼容。

## 2026-10-13：边界损失停止调度控制实验

为检查 4 epoch 的边界损失收益是否因 5 epoch 后期继续优化而退化，新增
`--discovery-rejection-boundary-stop-epoch`，并比较了 seed=44、5 epoch 下的
baseline 与“前 4 轮使用 `alpha=0.05`、第 5 轮关闭”的版本。调度实际生效，
训练历史记录的权重为 `1,1,1,1,0`，但结果没有改善：AUROC
`0.9051→0.9024`、FPR95 `0.3395→0.3565`、OSCR `0.6649→0.6604`、已知准确率
`0.7152→0.7105`。未知拒绝率虽为 `0.5408→0.5495`，但 KS 和分布重叠代理变差，
因此不能把该调度作为默认方法，也不能把拒绝率的单点上升解读为核心问题已解决。
完整记录见 [`analysis/rejection_boundary_stop_schedule_20261013.md`](analysis/rejection_boundary_stop_schedule_20261013.md)。

本次负结果进一步说明，下一步应优先检查开放验证集上的 checkpoint 选择：当前训练
可以建立 open validation，但历史协议主要按已知准确率保存模型。应在不使用测试标签的
前提下，对比 `known_acc`、`open_auroc` 和 `open_ks` 选择规则，再决定边界损失是否
值得继续深入；暂不继续堆叠新的分数融合或阈值调整。

## 2026-10-13：PU rejection-boundary loss 的公平复核

此前 boundary-loss pilot 的两组 checkpoint 轮数不同，不能直接把差异归因于方法。本轮重新固定两组均训练 4 个 epoch，seed=123 和 seed=44 分别使用相同的数据划分、teacher、mixed discovery pool、known prior、MC 和验证/校准协议；处理组唯一新增 `alpha_discovery_rejection_boundary=0.1`，测试标签只用于事后指标和 paired bootstrap。

- seed=123：AUROC `0.8833→0.8910`，FPR95 `0.4362→0.4127`，OSCR `0.6292→0.6392`，已知准确率 `0.6938→0.6992`，KS `0.6038→0.6263`，重叠代理 `0.5245→0.5135`。
- seed=44：AUROC `0.8822→0.8982`，FPR95 `0.4070→0.3797`，OSCR `0.6492→0.6576`，已知准确率 `0.7150→0.7112`，KS `0.6046→0.6493`，重叠代理 `0.5673→0.5320`。
- 配对 bootstrap 的 AUROC 差异 CI 分别为 `[+0.00311,+0.01259]` 和 `[+0.01114,+0.02084]`；固定精确 95% 已知覆盖率下的未知拒绝率差异 CI 分别为 `[-2.05,+4.30]` 和 `[+0.57,+6.45]` 个百分点。

这次是目前最有希望的训练期改动之一：两个 seed 的 AUROC、FPR95、KS 和重叠代理都同向改善，说明它不只是调整阈值。但 seed=123 的拒绝率区间仍跨 0，不能宣称已经稳定解决核心问题，也不能立即把它写成默认方法。实现的是混合池 PU 边界损失：用已知先验校正混合池风险，并在独立 rejection 表征上约束未知样本远离类条件已知 support 边界。下一步只做预先固定的权重敏感性 `0.05/0.2`，随后再考虑更长训练和多 seed；暂不叠加新的分数融合或用测试集调阈值。完整记录见 [`analysis/rejection_boundary_epoch_control_20261013.md`](analysis/rejection_boundary_epoch_control_20261013.md)。

### boundary 权重敏感性补充

为检查 `0.1` 是否只是偶然工作点，在相同 4-epoch 协议下又测试了 `0.05` 和 `0.2`。seed=44 上，`0.05` 的 AUROC 为 `0.9007`、FPR95 为 `0.3552`、OSCR 为 `0.6638`；相对同 seed baseline 的 paired bootstrap AUROC 差异为 `+0.01846`，95% CI `[+0.01413,+0.02310]`，固定 95% known coverage 下未知拒绝率差异为 `+3.77` 个百分点，CI `[+0.63,+6.80]`。`0.2` 仍有正向信号，但 AUROC 差异为 `+0.01318`，低于 `0.05`。

seed=123 的 `0.05` 复核 AUROC 为 `0.8918`、FPR95 为 `0.4063`、OSCR 为 `0.6395`；paired bootstrap AUROC 差异为 `+0.00859`，95% CI `[+0.00344,+0.01415]`，但未知拒绝率差异 CI `[-2.87,+3.83]` 个百分点仍跨 0。因此当前结论是：`0.05` 是值得进一步研究的候选范围，且比 `0.1/0.2` 更有希望；但这些权重是用测试结果做事后比较，不能直接据此改生产默认值。下一步应在 open validation 上预先固定权重选择，或增加新的 seed 后再决定默认设置。对照产物见 `analysis/rejection_boundary_weight_s44_w005_vs_base_20261013.json`、`analysis/rejection_boundary_weight_s44_w02_vs_base_20261013.json` 和 `analysis/rejection_boundary_weight_s123_w005_vs_base_20261013.json`。

随后在 seed=44 上把 baseline 和 `boundary=0.05` 同时延长到 5 epochs，检查短训练收益是否能保持。结果为：AUROC `0.8995→0.8970`，FPR95 `0.3832→0.3740`，OSCR `0.6558→0.6628`，已知准确率 `0.7095→0.7175`；固定精确 95% known coverage 的未知拒绝率 `49.50%→48.50%`，AUROC 配对 bootstrap CI `[-0.00653,+0.00142]`，拒绝率差异 CI `[-3.73,+1.80]` 个百分点。结论是该损失存在训练预算敏感性：4 epoch 的 AUROC 正向信号不能直接推广到 5 epoch。它继续保留为候选消融，不修改默认配置；后续必须使用验证集预先选定权重和 checkpoint，并增加 seed，不能根据测试集结果继续挑选 epoch 或权重。详见 [`analysis/rejection_boundary_epoch_control_s44_e5_20261013.md`](analysis/rejection_boundary_epoch_control_s44_e5_20261013.md)。

## 2026-10-13：rejection detector 后续复核结论

本日围绕“已知/未知特征与分数重叠”继续做了三组严格对照：外部 CIFAR-10 uniform OE、rejector 与 kNN/原型分数融合、线性 nnPU 与 MLP nnPU。所有比较固定 seed=123、学生 checkpoint、CIFAR-100 semantic-isolated 60/40、mixed pool、known prior、MC=4、open validation 和 95% known-coverage 校准；测试标签只用于事后指标和 paired bootstrap。

- uniform OE 的 AUROC `0.87986→0.88071`、FPR95 `0.43817→0.43467`，但 OSCR `0.58793→0.57784`、已知准确率 `0.63833→0.62667`、KS/overlap 变差；配对 bootstrap 不支持稳定收益，保留为关闭状态消融。
- kNN+rejector 的 `w=0.75` 在普通报告中达到 AUROC `0.88064`、FPR95 `0.41733`、OSCR `0.59619`，但严格精确 95% 已知覆盖率下未知拒绝率比 rejector-only 低 `2.40` 个百分点，AUROC CI 跨 0；这是校准/工作点变化，不进入默认流程。
- MLP nnPU 的 AUROC `0.87249`、FPR95 `0.50533`，均不如 linear nnPU 的 `0.87986/0.43817`，停止增加 rejector 容量。

详细数据和判断见 [`analysis/rejection_detector_followup_20261013.md`](analysis/rejection_detector_followup_20261013.md)。当前继续保留 independent rejection representation + linear `nu_corrected` nnPU 和 `encoder_lr_scale=0.1` 作为候选；下一步停止堆叠后处理检测器，回到直接改善共享 encoder 几何的单一训练目标，并继续同时报告 AUROC、FPR95、OSCR、已知效用、KS 和 overlap。

本轮代码检查：`python -m pytest -q` 为 `228 passed`，`compileall` 和 `git diff --check` 均通过。融合脚本初次运行曾因漏传 `--knn-ood` 暴露配置错误，已修正并重新运行；该失败不是算法结果，脚本现已通过语法检查。

补充的类别级诊断发现，未知类难度并不均匀：在 seed=123 rejection baseline 上，单个未知类的拒绝率约为 `12%--71%`，同时部分已知类也有较高误拒率。这支持下一步从类条件 support / 局部边界角度改善表征，而不是继续做全局阈值或全局 OE 调整。诊断时已确认开放集字段 `true_label=-1` 代表所有未知样本，真实类别必须读取 `raw_labels`；脚本和结果见 [`scripts/analyze_class_overlap.py`](scripts/analyze_class_overlap.py) 与 [`analysis/class_overlap_rejection_s123_20261013.json`](analysis/class_overlap_rejection_s123_20261013.json)。该诊断只用于事后分析，不参与训练或调参。

# 2026-10-13：uncertainty score 融合权重的配对复核

本轮固定学生 checkpoint、CIFAR-100 semantic-isolated 60/40 测试协议、kNN 特征库、MC=4 和验证拆分，仅比较检测分数中 head uncertainty 修正权重 `0.25 / 0.50 / 0.75`。seed 44、123 各自使用完整测试集；测试标签只用于事后指标与配对 bootstrap，不参与阈值或权重选择。

结果不支持改动默认权重：0.25 在 seed 123 的 AUROC 仅增加 `0.00054`（配对 95% CI `[-0.00039, +0.00151]`），在 seed 44 则降低 `0.00141`（CI `[-0.00226, -0.00056]`）。0.75 的 AUROC 在两个 seed 都下降；固定精确 95% 已知覆盖率下的未知拒绝率差异，四个比较的 bootstrap 区间均跨 0。部分设置下拒绝率上升不等于分布分离改善，完整指标也显示 AUROC、FPR95、OSCR、KS 与重叠率并未一致变好。**保留当前默认权重 0.50，不宣称核心问题得到解决。**

详细对照条件、六组完整测试指标、bootstrap 解释及后续方向见 [`analysis/uncertainty_score_weight_recheck_20261013.md`](analysis/uncertainty_score_weight_recheck_20261013.md)。本轮是检测器分数诊断，不改训练程序和默认算法；下一步应优先验证训练阶段对共享表征几何的改变，并继续采用新 seed、预先固定的验证协议，避免用测试集调参。

## 2026-10-12：开放验证集选 checkpoint 的严格同分数复核

本轮验证“按 open-validation AUROC 选择学生 checkpoint”是否能改善未知检测。此前一次快速比较把 `max_softmax` 选模与 kNN 最终检测混在一起，不能用于判断该方法；该比较已作废。本轮固定 CIFAR-100 semantic-hard 60/40、seed=44、ResNet-34/18、完整已知训练划分、验证集 300、发现池 1200、测试集 1000、2 epochs、`open_val_ratio=0.2`、95% 已知覆盖率阈值和跳过聚类，只比较 checkpoint 选择规则，并让选模与最终测试都使用同一个 `max_softmax` 分数。

| 选择规则 | 选中 epoch | AUROC | AUPR | FPR95 | OSCR | 已知全体准确率 | 已知接受率 | 未知拒绝率 | KS | 未知低于已知 q95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| open-val AUROC | 1 | 0.5351 | 0.4515 | 0.9350 | 0.1267 | 0.1829 | 0.9504 | 0.0530 | 0.0847 | 93.98% |
| known-val accuracy | 2 | 0.5447 | 0.4426 | 0.8752 | 0.2344 | 0.3128 | 0.9368 | 0.0723 | 0.1003 | 93.73% |

在这组严格配对实验中，known-accuracy checkpoint 在 AUROC、FPR95、OSCR、已知分类准确率、未知拒绝率和 KS 上都更好。因此开放验证集选模没有显示收益，不能作为解决“已知/未知特征与分数重叠”的算法方案；默认仍保留 `known_acc`。`open_auroc`/`open_ks` 只保留作冻结检测器后的诊断工具，并且必须同时报告已知性能保护指标。详细记录见 [`analysis/checkpoint_selection_openval_matched_s44_20261012.md`](analysis/checkpoint_selection_openval_matched_s44_20261012.md)。

随后在同一个 known-accuracy checkpoint 上比较距离型分数，固定测试集、阈值协议和 95% 已知覆盖率，只把 `max_softmax` 换成 `normalized_entropy_mahalanobis_diag`。Mahalanobis 结果为 AUROC `0.5424`、FPR95 `0.9214`、OSCR `0.2149`、已知接受率 `95.56%`、未知拒绝率 `4.34%`、KS `0.0753`；`max_softmax` 对照为 AUROC `0.5447`、FPR95 `0.8752`、OSCR `0.2344`、已知接受率 `93.68%`、未知拒绝率 `7.23%`、KS `0.1003`。因此当前表示空间下距离分数没有解决重叠，暂不继续堆叠检测器，后续优先回到直接改善共享 encoder 几何的训练目标。详细记录见 [`analysis/mahalanobis_vs_msp_same_checkpoint_s44_20261012.md`](analysis/mahalanobis_vs_msp_same_checkpoint_s44_20261012.md)。

## 2026-10-12：共享 encoder 的 nnPU 梯度机制审计（seed 44，5 epochs）

为避免继续盲目增大 mixed-pool uncertainty nnPU 权重，本轮先审计它是否真的给共享学生 encoder 提供了足够、方向稳定的更新信号。实验固定 CIFAR-100 semantic-isolated 60/40 划分、seed=44、CIFAR transform、预训练 ResNet-34/18、完整数据、5 epochs、matched mixed pool=5400、known prior=0.2、`nu_corrected` nnPU、系数 0.1，以及原有 CE/KD/feature KD/prototype/SupCon；唯一新增 `--audit-loss-gradients`，不改变优化更新。

每个 epoch 记录 raw loss 对共享 encoder 的梯度范数和余弦方向。nnPU 的实际有效范数需要再乘训练系数 0.1：

| epoch | CE | KD | SupCon | nnPU raw | nnPU 实际 | 实际 nnPU/CE | cos(CE,nnPU) | cos(KD,nnPU) | cos(SupCon,nnPU) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.8289 | 8.5832 | 10.7239 | 1.7659 | 0.1766 | 1.80% | -0.0068 | 0.0025 | 0.0106 |
| 2 | 8.1097 | 6.2273 | 10.5547 | 3.8440 | 0.3844 | 4.74% | -0.0046 | 0.0023 | 0.0062 |
| 3 | 7.2380 | 5.4682 | 10.5484 | 3.8091 | 0.3809 | 5.26% | -0.0017 | -0.0014 | 0.0029 |
| 4 | 6.6409 | 5.0226 | 10.8721 | 3.7258 | 0.3726 | 5.61% | 0.0020 | -0.0034 | 0.0063 |
| 5 | 6.2604 | 4.8000 | 11.2437 | 3.6593 | 0.3659 | 5.85% | 0.0028 | -0.0051 | 0.0015 |

结论：nnPU 确实连接到了 encoder，但在当前系数下有效梯度明显小于 CE/KD/SupCon；其方向长期接近正交，没有发现强烈、稳定的梯度冲突。因此当前瓶颈更像是“不确定性 head 的信号没有直接形成主检测器需要的表征分离”，而不是简单的 nnPU 与分类目标互相抵消。该结果不是性能提升证据，也不支持直接把 nnPU 权重调大。历史 `0.3/0.5` 权重实验使用了不同的 teacher、数据限制和 rejector，不能与本轮 full-data 诊断直接拼接比较。

本轮还修复了诊断工具的真实边界缺陷：某个 loss 未连接某些 encoder 参数时，必须按参数位置补零；否则不同 loss 的梯度向量长度不同，余弦诊断会崩溃或失真。新增了正常梯度、未连接参数、零梯度和空参数测试；全量测试 `225 passed`，compileall 与 `git diff --check` 通过。详细记录见 [`analysis/encoder_gradient_audit_s44_20261012.md`](analysis/encoder_gradient_audit_s44_20261012.md)，原始产物在 `runs/gradient_audit_cifar_s44_20261012_5ep/`。

### feature-margin 诊断接线复核

随后对实际启用的 `discovery_uncertainty_feature_margin` 做了 1 epoch 梯度审计，修正了此前把诊断字段接到已禁用的 `discovery_feature_margin` 上的问题。修复前日志中的 feature-margin 梯度为 0 不能解释为训练没有梯度；修复后实际 raw 梯度范数为 `0.1520`，损失系数为 `0.05`，因此有效梯度约为 `0.0076`，约为 CE 梯度的 `0.08%`，与 CE 的余弦约为 `0.0025`。这证明该损失确实连接到 encoder，但当前权重下信号很弱且没有形成稳定的分离方向。详细记录见 [`analysis/feature_margin_gradient_audit_s44_20261012.md`](analysis/feature_margin_gradient_audit_s44_20261012.md)。

五轮 feature-margin 对照中，未知拒绝率由 `23.68%` 升至 `30.70%`，但 AUROC 仅由 `0.80917` 升至 `0.80981`，AUPR 反而由 `0.70519` 降至 `0.70466`；三种特征距离的 AUROC 也没有同向改善。因此这更像工作点变化，而不是已知/未知特征分布被可靠拉开。该方法保留为显式 ablation，不设为默认，也不与其他未经验证的边界损失继续叠加。

下一步不把该诊断开关或 encoder-gradient 路由设为默认，而是按同一实验协议逐项检验直接作用于共享特征几何的单一目标。每次同时报告 AUROC、AUPR、FPR95、OSCR、已知准确率/接受率、未知拒绝率、KS/overlap 和特征距离分布；只看未知拒绝率上升不能判定核心重叠问题解决。已有 feature-margin、OE、boundary、rejector 等历史方案只有在日志确认损失非零且对照协议匹配时才可继续复核，不能把多个未证实模块重新叠加。

## 2026-10-12：mixed-pool nnPU 梯度是否应进入共享 encoder（两 seed 配对复核）

### 问题与对照设计

此前的混合池 uncertainty nnPU 经由 uncertainty head 训练，但主检测分数并不直接读取该 head；它只有在梯度回传到共享 encoder 时，才可能改变检测器使用的表征。本轮因此只检验一个问题：保留相同 nnPU 目标时，是否应允许其梯度更新 encoder。对照组启用 `--discovery-uncertainty-pu-detach-features`（只训练 uncertainty head），实验组不启用（梯度可回传 encoder）。

每个 seed 内，两臂使用相同 CIFAR-100 semantic-isolated 60/40 划分、教师、预训练 ResNet-18 学生、5 epochs、matched mixed pool=5,400、已知先验 0.2、`nu_corrected` nnPU（权重 0.1）、encoder LR scale=0.1、相同的检测器 `normalized_entropy_min_class_knn`、MC=4 和独立 calibration/selection 的 95% known-coverage 阈值。官方测试标签只用于最终评估。检测阶段使用各自学生 checkpoint，但同一 seed 内校准与测试流程一致。

seed=44 复用了当日建立的 CIFAR transform、disjoint-calibration teacher；seed=123 的旧 teacher 缺少相同预处理配置，因此先按当前协议重新训练 teacher，再完成两臂复核。两 seed 学生配置核验为 `transform_profile=cifar`、`calibration_ratio=0.25`，nnPU 先验、损失权重和池大小匹配。seed=44 的 `transform_profile_requested=auto` 最终解析为 `cifar`，与 seed=123 实际使用的 `cifar` 一致。

### 官方测试结果

| seed | nnPU 梯度路由 | AUROC | AUPR | FPR95 | OSCR | 已知准确率 | 已知接受率 | 未知拒绝率 | KS | unknown below known-q95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 44 | 进入 encoder | 0.8135 | 0.7108 | 0.5978 | 0.5315 | 58.97% | 94.60% | 28.95% | 0.4871 | 72.75% |
| 44 | 仅更新 uncertainty head | 0.8079 | 0.7023 | 0.6005 | 0.5252 | 58.80% | 94.43% | 26.80% | 0.4806 | 74.95% |
| 123 | 进入 encoder | 0.8106 | 0.7124 | 0.5955 | 0.5624 | 62.70% | 95.03% | 26.25% | 0.4861 | 73.68% |
| 123 | 仅更新 uncertainty head | 0.8154 | 0.7138 | 0.5785 | 0.5632 | 62.55% | 95.40% | 24.33% | 0.5048 | 74.23% |

差值（encoder-gradient 减 head-only）：

| seed | AUROC | AUPR | FPR95 | OSCR | 已知准确率 | 已知接受率 | 未知拒绝率 | KS | overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 44 | +0.0056 | +0.0085 | -0.0027 | +0.0063 | +0.17 pp | +0.17 pp | +2.15 pp | +0.0065 | -2.20 pp |
| 123 | -0.0048 | -0.0014 | +0.0170 | -0.0008 | +0.15 pp | -0.37 pp | +1.93 pp | -0.0188 | -0.55 pp |

95% known-coverage 校准集的接受率分别为 seed 44 的 94.92% 和 seed 123 的 94.92%，两臂各自完全相同；因此测试差异不是两臂采用了不同校准集或覆盖目标造成的。测试集未知拒绝率在两 seed 都提高，但更全面的排序/分离指标没有一致改善：seed 44 略偏正向，seed 123 的 AUROC、FPR95、OSCR 和 KS 均偏向 head-only。不能把拒绝率单项上升解读成未知/已知分布已经分开。

### 结论与后续

- 允许 nnPU 梯度回传 encoder 有一个可复现但幅度有限的工作点信号：两 seed 的未知拒绝率都增加约 2 个百分点；同时已知准确率几乎不变。
- 该增益没有稳定改善整体区分能力；两 seed 的 AUROC 差异异号，FPR95、OSCR、KS 也不一致。因此目前证据不足以将 encoder-gradient 设为默认配置，更不能声称解决了核心分数/特征重叠。
- 暂时保留该参数作为可选诊断开关，不继续盲目增大 nnPU 权重。下一步优先核对训练期间 nnPU 相对分类/KD 梯度在共享 encoder 上的范数与方向（梯度冲突/余弦），以及每 epoch 的 known/unknown 代理分布指标；再决定是调整梯度系数/路由，还是停止这条支路，转向直接优化并验证表征几何。
- 这是两个 seed 的有限复核，不能据此推断广泛泛化收益。两 seed 使用相同预处理，但各自独立训练教师与学生；需要更大 seed 数和预注册的主指标后才适合做强结论。

报告与原始指标：`runs/nnpu_encoder_gradient_s44_encoder_grad_discover/discovery_report.json`、`runs/nnpu_encoder_gradient_s44_head_only_discover/discovery_report.json`、`runs/nnpu_encoder_gradient_s123_encoder_grad_discover/discovery_report.json`、`runs/nnpu_encoder_gradient_s123_head_only_discover/discovery_report.json`。复现脚本：`scripts/run_nnpu_encoder_gradient_s44.ps1` 和 `scripts/run_nnpu_encoder_gradient_s123.ps1`。

## 2026-10-12：open-validation checkpoint 选模审计

本轮没有改训练损失，而是检查“只按已知类验证准确率保存/选择 checkpoint”是否会漏掉开放集排序更好的 epoch。固定 seed=44 的 CIFAR-100 semantic-isolated 60/40 五轮训练及检测配置，逐轮计算预留 open-validation 指标；每个 epoch 都用该 epoch 自己的模型重新构建 kNN 特征库，校准子集与 checkpoint-selection 子集不重叠。

已知验证准确率选择 epoch 2（0.5741），open-val AUROC 选择 epoch 5（0.7984）。相对 epoch 2，epoch 5 的 open-val AUROC +0.0102、AUPR +0.0042、KS +0.0127，未知分数落在已知 q95 以下的比例下降 4.75 个百分点；但 FPR95 从 0.6000 升至 0.6133，open-val 已知准确率从 0.5767 降至 0.5500。因此这是有取舍的选模信号，不是整体性能已改善，也不是阈值技巧解决了特征重叠。单 seed 不足以更改默认选模规则。

本次审计还纠正了脚本中的特征库复用错误：不同 epoch 的 embedding 空间不同，必须为每个 checkpoint 单独建立 kNN bank。随后 seed=123 诊断中，已知准确率和 open-val AUROC 都选择 epoch 4（AUROC 0.7969、FPR95 0.6033、已知准确率 0.5967、KS 0.4807、未知低于已知 q95 的比例 78.55%）；并未复现 seed=44 上“两个选择规则指向不同 epoch”的现象。两个 seed 的教师 checkpoint 选择协议也不完全相同（seed44 teacher 使用 disjoint calibration，seed123 teacher 未使用），因此这只是方向性诊断，不能作为严格多 seed 结论。当前不改默认选模规则。

限制：seed44 五个 epoch 的官方 test discovery 报告在审计前已被探索性查看，因此此处不把 test 的逐 epoch 对比作为独立证据，也不据此挑选 checkpoint。seed123 的 test 未用于该选模审计。后续若研究 checkpoint 选择，应先统一教师 checkpoint 协议、冻结 open-val 目标和已知准确率保护条件，再做新 seed 复核；每个 seed 仅用 open-val 选模型，然后只评估所选模型的 official test。当前核心仍是特征/分数重叠，未知低于已知 q95 的比例约 74–79%，尚未解决。

seed44 与 seed123 的详细逐 epoch 指标、审计脚本修正过程和协议限制见 [`analysis/checkpoint_selection_openval_s44_20261012.md`](analysis/checkpoint_selection_openval_s44_20261012.md)；机器可读结果分别见 [`analysis/checkpoint_selection_openval_s44_20261012.json`](analysis/checkpoint_selection_openval_s44_20261012.json) 和 [`analysis/checkpoint_selection_openval_s123_20261012.json`](analysis/checkpoint_selection_openval_s123_20261012.json)。

## 2026-10-12：hard vs soft nnPU 拒识特征边界配对实验

核心问题仍是已知与未知分数分布重叠。此前 hard nnPU 特征边界损失在训练中持续变小，
因此本轮在其他设置固定时，仅比较非负 PU 风险的硬截断与 softplus 平滑形式。两组使用
CIFAR-100 semantic-isolated 60/40、seed=44、同一教师初始化、完整训练集、5 epochs、
`encoder_lr_scale=0.1`、5400 个 mixed discovery pool（known prior=0.2）、相同检测器与
验证集 95% known-coverage 校准；特征边界损失权重均为 0.1、softness 均为 0.05。

| 指标 | hard nnPU | soft nnPU | soft - hard |
|---|---:|---:|---:|
| AUROC | 0.8706 | 0.8742 | +0.0036，配对 95% CI [-0.0020, +0.0093] |
| AUPR | 0.7927 | 0.8013 | +0.0087 |
| FPR95 | 0.4213 | 0.4203 | -0.0010 |
| OSCR | 0.5597 | 0.5635 | +0.0037 |
| 全部已知样本分类准确率 | 0.6135 | 0.6142 | +0.0007 |
| 精确 95% 已知覆盖下未知拒绝率 | 0.3750 | 0.4008 | +0.0258，配对 95% CI [-0.0063, +0.0580] |
| KS 分数距离 | 0.5882 | 0.5935 | +0.0053 |
| 未知分数低于已知 q95 的比例 | 0.6245 | 0.5958 | -0.0288（改善）|

soft nnPU 确实使第 5 epoch 的特征边界 PU 损失保持在 0.0536，而 hard nnPU 降到
0.0181，说明损失衰减机制得到缓解。但 AUROC 和固定覆盖率未知拒绝率的配对置信区间都
跨过零；这是单 seed 的小幅、方向混合结果，不能称为已验证提升。报告中各自验证阈值在
测试集上的已知覆盖率分别是 94.10% 和 95.95%，所以其对应未知拒绝率 41.65% vs 35.40%
不能直接比较。bootstrap 的精确 95% 覆盖操作点是事后诊断，不参与阈值选择或模型选择。

随后按预先设定的 seed=123 做了第二个 seed 复核：AUROC 从 0.8827 到 0.8855
（配对 95% CI [-0.0016, +0.0071]），但 AUPR 从 0.8212 降到 0.8170；精确 95% 已知
覆盖下未知拒绝率从 0.4538 降到 0.4350（配对 95% CI [-0.0478, +0.0123]），未知尾部
重叠也变差。两个 seed 的 AUROC 点估计方向都略正，但置信区间均跨零；FPR95、AUPR、
固定覆盖拒绝率和尾部重叠并无一致方向。因此 soft nnPU 的损失机制变化确实存在，却没有
证明能可靠改善已知/未知分离。第 5 epoch 的边界 PU 损失衰减模式在两个 seed 中均被缓解
（soft 为 0.0536；hard 分别为 0.0181、0.0190），但检测收益不稳。当前保留 hard nnPU
默认，停止 softness/阈值扫描，优先转向表征几何训练信号，并保持检测协议不变。

详细协议与解释见 [`analysis/rejection_pu_margin_risk_s44_20261012.md`](analysis/rejection_pu_margin_risk_s44_20261012.md)，
seed=44/123 配对 bootstrap 分别见
`analysis/rejection_pu_margin_s44_soft_vs_hard_bootstrap_20261012.json` 和
`analysis/rejection_pu_margin_s123_soft_vs_hard_bootstrap_20261012.json`。

## 2026-10-12：拒识器已知支持来源与数据协议审计

为确认当前较高的拒识 AUROC 是否受支持样本选择或数据协议影响，本轮先审计 CIFAR-100
semantic-isolated seed=44 的实际样本索引，再在不重训模型的前提下比较拒识器支持样本。
核对了 supervised train、known/novel discovery pool、checkpoint-selection validation、阈值
calibration validation、open validation 和官方 test：样本两两互斥，matched discovery pool
共 5,400 张且 known prior 精确为 0.2；校准集与 checkpoint-selection 集不重叠，test 未用于
训练或阈值校准。

比较固定相同学生 checkpoint、测试样本、混合 PU pool、线性 `nu_corrected` nnPU 拒识器和
验证集 95% known-coverage 校准，仅改变拒识器已知支持来源；为消除先前“训练集最多 5,000
张 vs open-val 仅 300 张”的样本量混杂，所有组限制为 300 known + 300 unlabeled。另单独
比较训练样本使用训练增强还是确定性 eval transform。

| checkpoint | known 支持方式 | AUROC | FPR95 | OSCR | test known accept | test unknown reject | KS | unknown below known q95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| margin only | train + train augmentation | 0.7929 | 0.7422 | 0.4966 | 94.53% | 33.23% | 0.4513 | 68.58% |
| margin only | train + eval transform | 0.7959 | 0.6808 | 0.5040 | 94.93% | 30.48% | 0.4464 | 70.03% |
| margin only | open-val known | 0.7763 | 0.7462 | 0.4905 | 96.08% | 27.18% | 0.4133 | 70.05% |
| margin + uniform OE | train + train augmentation | 0.7840 | 0.7315 | 0.4968 | 94.65% | 29.10% | 0.4345 | 71.85% |
| margin + uniform OE | train + eval transform | 0.7778 | 0.7192 | 0.4959 | 93.83% | 30.43% | 0.4178 | 72.42% |
| margin + uniform OE | open-val known | 0.7770 | 0.7180 | 0.4920 | 94.72% | 29.83% | 0.4086 | 71.13% |

同为 300 支持样本时，open-val 支持没有带来提升；margin-only 组 AUROC 配对 bootstrap
差值为 -0.0165（95% CI [-0.0261, -0.0069]），uniform-OE 组差值 -0.0069（95% CI
[-0.0177, +0.0044]）。训练支持改用 eval transform 也没有一致改善。因此暂时保留训练集
known 支持方案，不推广 strict-mixed 支持；同时注意 300 样本诊断不能用于估计原先最多
5,000 支持样本设置的绝对效果。结论不是核心重叠已解决，而是数据泄漏/校准重叠并非当前
结果的原因，小规模独立支持可能带来较大方差。后续优先固定 `encoder_lr_scale=0.1`，继续
比较针对表征或 PU 目标的单因素修改，使用相同测试样本、固定验证 known coverage，并联合
报告 AUROC、FPR95、OSCR、分数重叠、已知分类与覆盖率；有希望的设置再用预先指定的第二个
seed 复核。完整协议、指标说明和 artifacts 见
[`analysis/rejector_support_source_s44_20261012.md`](analysis/rejector_support_source_s44_20261012.md)。

训练来源固定时的配对 bootstrap 补充显示，改用 eval transform 对 margin-only checkpoint
的 AUROC 差值 +0.0031（95% CI [-0.0025, +0.0090]），对 uniform-OE checkpoint 为
-0.0062（95% CI [-0.0120, -0.0001]）；未知拒绝率在精确 95% 已知覆盖率下均未可靠提升。
这说明特征提取变换会影响后处理拒识器，但没有证据支持它是核心重叠的主要成因。

## 2026-10-11：encoder 学习率缩放的第二个 seed 复核

前一轮 seed=42 的结果显示，将学生模型预训练 encoder 的学习率从完整值缩放到
`0.1`，可能在不改变 nnPU、蒸馏和检测器的情况下减轻已知/未知分数重叠。本轮在
seed=123 上严格复核：CIFAR-100 semantic-isolated 60/40、同一 ResNet-34 teacher、
ResNet-18 student、CIFAR transform、完整数据、5 epochs、matched mixed pool=5400、
known prior=0.2、audited nnPU、`normalized_entropy_min_class_knn`、MC=4 和验证集
95% known coverage 全部固定，只改变 `encoder_lr_scale: 1.0 -> 0.1`。

| 指标 | nnPU baseline | encoder LR 0.1 | 变化 |
|---|---:|---:|---:|
| AUROC | 0.7933 | 0.8170 | +0.0237 |
| AUPR | 0.6701 | 0.7226 | +0.0525 |
| FPR95 | 0.6088 | 0.5910 | -0.0178 |
| OSCR | 0.5349 | 0.5603 | +0.0254 |
| 已知分类准确率 | 60.73% | 62.10% | +1.37 个百分点 |
| 已知接受率 | 95.68% | 94.55% | -1.13 个百分点 |
| 未知拒绝率 | 19.78% | 29.53% | +9.75 个百分点 |
| score KS | 0.4547 | 0.4974 | +0.0427 |

seed=42 和 seed=123 的 AUROC、AUPR、FPR95、OSCR、KS 及固定工作点大体同向改善，
因此这是目前最有希望的候选方向；它改变了训练得到的连续分数分布，不是简单调阈值。
但它仍不是核心问题的最终解决方案：seed=123 仍有约 70.47% 的未知样本被接受，且已知
接受率下降约 1.1 个百分点。详细协议和限制见
[`analysis/cifar_profile_encoder_lr01_s123_20261011.md`](analysis/cifar_profile_encoder_lr01_s123_20261011.md)。

第三个 seed 的严格配对复核已经完成，结果见下方最新记录。当前暂不把
boundary loss、angular margin、distance-consensus/support 加权 PU 等已表现不稳定的
方案与 encoder 学习率缩放叠加；下一步应先在固定 `0.1` 主候选上做小型尺度消融
（`1.0/0.1/0.01`）或继续做跨 seed 稳定性与表征几何诊断。

## 2026-10-12：encoder 学习率缩放的第三个 seed 严格配对复核

为避免把旧的非配对实验误当成证据，本轮重新训练了 seed=43 的 nnPU baseline，
只比较 `encoder_lr_scale=1.0` 与 `0.1`。两组使用相同 teacher、数据划分、CIFAR
transform、5 epochs、matched mixed pool=5400、known prior=0.2、KD、prototype loss、
nnPU、检测器和验证集 95% known coverage；旧的 seed=43 baseline/treatment 目录没有
用于归因，因为它们还同时改变了 feature-margin 或检测协议。

| 指标 | scale 1.0 | scale 0.1 | 变化 |
|---|---:|---:|---:|
| AUROC | 0.7901 | 0.8228 | +0.0327 |
| AUPR | 0.6773 | 0.7241 | +0.0468 |
| FPR95 | 0.6400 | 0.5560 | -0.0840 |
| OSCR | 0.5324 | 0.5570 | +0.0246 |
| 已知分类准确率 | 60.55% | 62.12% | +1.57 个百分点 |
| 已知接受率 | 95.08% | 95.45% | +0.37 个百分点 |
| 未知拒绝率 | 21.45% | 26.25% | +4.80 个百分点 |
| score KS | 0.4527 | 0.5113 | +0.0587 |
| score 均值间隔 | 2.1084 | 2.3149 | +0.2065 |
| 未知低于已知 q95 | 0.7850 | 0.7173 | -0.0678 |

seed=42、123、43 的严格配对结果均显示同向改善。三 seed 平均值如下：

| 指标 | scale 1.0 平均 | scale 0.1 平均 | 平均变化 |
|---|---:|---:|---:|
| AUROC | 0.7913 | 0.8175 | +0.0262 |
| AUPR | 0.6744 | 0.7196 | +0.0453 |
| FPR95 | 0.6258 | 0.5727 | -0.0530 |
| OSCR | 0.5403 | 0.5570 | +0.0167 |
| 已知分类准确率 | 61.45% | 62.10% | +0.65 个百分点 |
| 已知接受率 | 95.19% | 95.04% | -0.15 个百分点 |
| 未知拒绝率 | 21.41% | 27.52% | +6.11 个百分点 |
| score KS | 0.4543 | 0.4975 | +0.0432 |

这说明 `encoder_lr_scale=0.1` 是目前最值得保留的训练主候选，且收益不只是阈值
变化：连续分数的 KS、均值间隔和 q95 overlap 同时改善。不过未知拒绝率平均仍约
27.5%，大多数未知样本仍被接受，因此不能宣称核心重叠问题已经解决。详细 seed=43
协议和结果见 [`analysis/cifar_profile_encoder_lr01_s43_20261012.md`](analysis/cifar_profile_encoder_lr01_s43_20261012.md)。

## 2026-10-12：encoder 学习率尺度 `0.01` 消融

为判断是否“越小的 encoder 学习率越能保护预训练特征”，在同一 seed=43、同一
teacher、数据、5 epochs、matched mixed pool、nnPU、检测器和验证校准下，只把
`encoder_lr_scale` 从 `0.1` 改为 `0.01`。

| 指标 | scale 0.1 | scale 0.01 | 变化 |
|---|---:|---:|---:|
| AUROC | 0.8228 | 0.7971 | -0.0257 |
| AUPR | 0.7241 | 0.6872 | -0.0369 |
| FPR95 | 0.5560 | 0.6027 | +0.0467 |
| OSCR | 0.5570 | 0.4762 | -0.0808 |
| 已知分类准确率 | 62.12% | 53.62% | -8.50 个百分点 |
| 已知接受率 | 95.45% | 94.68% | -0.77 个百分点 |
| 未知拒绝率 | 26.25% | 23.98% | -2.28 个百分点 |
| score KS | 0.5113 | 0.4584 | -0.0529 |
| score 均值间隔 | 2.3149 | 1.9600 | -0.3549 |
| 未知低于已知 q95 | 0.7173 | 0.7695 | +0.0523 |

`0.01` 全面差于 `0.1`，说明当前收益不是来自简单地冻结 encoder；`0.1` 是保护
预训练结构和适应已知类别之间更合适的折中。现在停止继续缩小统一学习率，保留
`0.1` 为主候选。若后续测试 layer-wise 或 scheduled learning rate，必须以固定
`0.1` 为单因素 baseline。详见
[`analysis/cifar_profile_encoder_lr_scale_ablation_s43_20261012.md`](analysis/cifar_profile_encoder_lr_scale_ablation_s43_20261012.md)。

## 2026-10-12：layer-wise encoder 学习率衰减的严格复核

在不改变默认训练行为的前提下，代码新增 `--encoder-lr-decay`。默认值 `1.0` 时，
encoder 仍使用原来的单一学习率；小于 `1.0` 时，按 backbone 的有序 feature stage
给浅层更小、深层相对更大的学习率，并与 `--encoder-lr-scale` 相乘。该设计用于在
当前 `scale=0.1` 的基础上进一步保护低层预训练特征，同时保留高层适应已知类别的能力。

本轮按照该单因素条件完成了 CIFAR-100 seed=43 的严格复核：固定
`encoder_lr_scale=0.1`，只比较 `encoder_lr_decay=1.0` 与 `0.8`。数据协议、teacher、
student、5 epochs、matched mixed pool=5400、nnPU、检测器和验证集 95% known coverage
均不变。

| 指标 | decay 1.0 | decay 0.8 | 变化 |
|---|---:|---:|---:|
| AUROC | 0.8228 | 0.8223 | -0.0006 |
| AUPR | 0.7241 | 0.7257 | +0.0016 |
| FPR95 | 0.5560 | 0.5823 | +0.0263 |
| OSCR | 0.5570 | 0.5440 | -0.0130 |
| 已知分类准确率 | 62.12% | 60.78% | -1.34 个百分点 |
| 已知接受率 | 95.45% | 94.95% | -0.50 个百分点 |
| 未知拒绝率 | 26.25% | 28.18% | +1.53 个百分点 |
| score KS | 0.5113 | 0.5075 | -0.0038 |

未知拒绝率虽然提高了 1.53 个百分点，但同时已知接受率下降，且 FPR95、OSCR、KS
和均值间隔没有改善。因此这主要是工作点变得更激进，不是连续分数分布真正分离，不能
认为 layer-wise decay 解决了核心的已知/未知重叠问题。当前保留
`--encoder-lr-decay` 作为可复现的可选消融开关，主线仍使用
`encoder_lr_scale=0.1, encoder_lr_decay=1.0`，停止继续扫描 decay。完整协议和产物见
[`analysis/cifar_profile_encoder_lr_decay08_s43_20261012.md`](analysis/cifar_profile_encoder_lr_decay08_s43_20261012.md)。

### 今日阶段性总结

今天完成了三项相互衔接的工作：

1. 用三个严格配对 seed 复核 `encoder_lr_scale=0.1`。相较 scale=1.0，平均 AUROC
   从 0.7913 提升到 0.8175，FPR95 从 0.6258 降到 0.5727，OSCR 从 0.5403 提升到
   0.5570，未知拒绝率从 21.41% 提升到 27.52%；这是目前最有价值的训练改动，但仍
   有多数未知样本被接受。
2. 用 seed=43 做 `encoder_lr_scale=0.01` 消融。该设置全面差于 0.1，说明简单地
   更强地冻结 encoder 不是解决方案。
3. 新增并严格测试 layer-wise `encoder_lr_decay=0.8`。它没有改善排序和分布分离，
   因此不进入默认主线。

当前结论是：应停止在统一或分层 encoder 学习率上继续盲目搜索；保留
`audited nnPU + encoder_lr_scale=0.1 + normalized_entropy_min_class_knn` 作为稳定
基线。未知/已知重叠仍是首要问题，后续若继续研究，应优先从未知证据、开放集表征
目标和数据协议审计入手，并继续使用配对 seed、固定 known coverage 以及 AUROC、
FPR95、OSCR、KS、score overlap 的联合判断。上述判断和本轮数字均记录在
[`analysis/cifar_profile_encoder_lr_decay08_s43_20261012.md`](analysis/cifar_profile_encoder_lr_decay08_s43_20261012.md)。

## 2026-10-11：按预测类别校准阈值的小实验

前一轮按类误差显示，不同未知类别会被吸收到不同已知预测类。代码已有基于已知验证样本
的 `class_conditional` 阈值策略；本轮固定同一 seed=123 的 5/10 epoch checkpoint、
检测分数与数据，只比较该策略和原有全局 95% known-coverage 阈值。类别阈值仅用已知验证
样本按“模型预测类别”估计，样本不足时回退到全局阈值，未知测试标签只用于最终报告。

5 epoch 下，按类阈值将未知拒绝率从 13.53% 提到 16.45%，但已知接受率从 95.92%
降到 94.30%；检测平衡准确率（已知接受率与未知拒绝率的平均值）增加 0.65 个百分点。
在 seed=42 的 5 epoch 模型上也复核到相同方向：未知拒绝率从 14.95% 提到 17.28%，
已知接受率从 94.43% 降到 93.48%，平衡准确率增加约 0.69 个百分点。10 epoch 下，
拒绝率从 13.60% 提到 14.98%，已知接受率从 95.07% 降到 93.68%，平衡准确率基本不变。
各比较的 AUROC/AUPR/FPR95/OSCR 均不变，因为只改变了二元决策阈值，
没有改变连续分数或特征。进一步把每个样本分数减去其预测类别的验证阈值，尝试按类中心化
后统一排序，seed=42/123 的 AUROC/AUPR/FPR95 均变差，因此拒绝该分数重排方案。
按类阈值最多是有待更多 seed 复核的可选工作点策略，不是核心重叠问题
的解决方案，也不能据此声称表征改善。测试集仍显示皮卡车、电视/公共汽车等未知类难以拒绝，
并会流向语义相近的已知类。详细协议、指标和局限见
[`analysis/class_conditional_threshold_pilot_s123_20261011.md`](analysis/class_conditional_threshold_pilot_s123_20261011.md)。

本轮也修正了按类误差脚本：现在能正确应用全局标量阈值或类别阈值数组，并输出未知类被接收后
对应的已知预测类别，避免审计 class-conditional 实验时错误地把阈值数组转成单个数。

## 2026-10-11：几何诊断与训练时长复核

本轮针对已知/未知重叠，使用 CIFAR-100 semantic-isolated 60/40、seed=123
和完整数据做了按样本配对的特征距离诊断，并在相同数据、教师、无权重
audited nnPU、检测器及验证校准下比较 5 与 10 epochs。几何诊断分别考察分类器
原型距离、经验类中心距离、最近已知训练样本距离；测试标签只用于事后分组统计。

结果没有支持“某个 PU 梯度路由全面改善表征”：直接加权 PU 的三个距离指标都变差；
detached PU 只改善原型距离，类中心和最近样本距离反而变差。延长到 10 epochs
使全部已知分类准确率从 46.58% 升至 52.13%、OSCR 从 0.3914 升至 0.4358，
但 AUROC 从 0.7180 降至 0.7156、AUPR 从 0.5938 降至 0.5824、FPR95 从
0.7635 升至 0.7685、未知拒绝率仅从 13.53% 变为 13.60%，KS 也略降。
因此，更多训练对已知分类有帮助，但没有解决开集排序和分布重叠；拒绝率的微小变化
不构成有效提升。当前不把 detached PU 或 10-epoch 训练升级为开集主方案。

按类事后审计进一步显示，误接收不是由少数未知类单独造成：5 epoch 接受了
3459/4000 个未知样本，10 epoch 接受了 3456/4000 个。皮卡车（CIFAR-100 类别 58）
在两者中都最难拒绝（误接收率均为 98%），但大量其他未知类别也有高误接收；误接收
还分散流入多个已知预测类别（5 epoch 主要流向仓鼠、沙发、摩天大楼、蘑菇、衣柜；
10 epoch 主要流向鲨鱼、沙发、仓鼠、蘑菇、松鼠）。这说明问题不只是某一个未知类别，
也可能同时受类别表征/分类混淆影响。此结果只作测试集事后诊断，不据此设置类阈值或训练。
详见 [`analysis/feature_geometry_and_training_budget_20261011.md`](analysis/feature_geometry_and_training_budget_20261011.md)
及 [`analysis/classwise_open_errors_unweighted_5vs10_s123_20261011.json`](analysis/classwise_open_errors_unweighted_5vs10_s123_20261011.json)。

按类误差审计已完成：误接收分散于多个未知类，并流入多个已知预测类；下一步应做
逐样本近邻诊断，检查已知误分类与未知误接收是否共享局部邻域，再据此区分语义相似、
类别多模态或数据/分辨率问题。测试标签只用于事后分析，不能参与训练、阈值选择或调参。
协议、全部距离结果和限制见
[`analysis/feature_geometry_and_training_budget_20261011.md`](analysis/feature_geometry_and_training_budget_20261011.md)。

## 2026-10-11：分离 nnPU 梯度路由的全量复核

前一轮 seed=123 发现，`distance_consensus` 加权 nnPU 能提高 uncertainty
head 自身排序，却损害 prototype distance、entropy 与最终检测器。本轮检验
“PU 梯度干扰共享特征表示”这一具体假设：新增默认关闭的
`--discovery-uncertainty-pu-detach-features`，让 PU loss 在 detach 后的特征上
更新 uncertainty/rejection head，但不让该项梯度回到 backbone；其余训练损失
仍按原路径更新表示。定向单测验证 head 有梯度、输入特征无梯度。

本轮在 CIFAR-100 semantic-isolated 60/40、seed=123、全量数据、同一教师、预训练
ResNet-18 学生、5 epochs、matched mixed pool=5400、known prior=0.2 下比较加权 PU
直接回传与 detached 特征；两组只差 detach 开关。检测固定为
`normalized_entropy_proto`、MC=4、验证集 95% known coverage；另列此前无权重 nnPU
模型作为参照。

| 指标 | 无权重 nnPU | 加权 PU：直接梯度 | 加权 PU：detached features |
|---|---:|---:|---:|
| AUROC | 0.7180 | 0.7012 | 0.6997 |
| AUPR | 0.5938 | 0.5716 | 0.5615 |
| FPR95 | 0.7635 | 0.7800 | 0.7832 |
| OSCR | 0.3914 | 0.3720 | 0.3639 |
| 已知接受率 | 95.92% | 94.75% | 95.47% |
| 未知拒绝率 | 13.53% | 13.18% | 11.73% |
| 全部已知分类准确率 | 46.58% | 44.78% | 43.68% |
| score KS | 0.3348 | 0.3041 | 0.3103 |

**结论：这个梯度隔离方案没有改善总体开放集表现，不应设为默认。**它比直接梯度略
多接受已知样本，但 AUROC/AUPR/OSCR 和未知拒绝率更差于无权重参照；不能把已知接受率
单项变化误判为解决了已知/未知重叠。保留为可选消融，不继续对同一开关做参数扫描。
实验协议、限制与后续诊断方向见
[`analysis/pu_detached_features_full_recheck_20261011.md`](analysis/pu_detached_features_full_recheck_20261011.md)。

## 2026-10-11：distance-consensus 全量独立 seed 复核

上一轮 seed=42 的全量实验中，`distance_consensus` 加权 mixed-pool uncertainty
nnPU 曾显示小幅正向信号，但只有一个完整 seed，不能据此设为主线。本轮按相同
协议补做 seed=123 的严格配对复核：只改变
`discovery_uncertainty_pu_weight_source`（`none` 对 `distance_consensus`），其余
教师、数据划分、ResNet-34/18、全量数据、5 epoch、matched pool=5400、known
prior=0.2、`normalized_entropy_proto` 检测器、MC=4 和 validation-only 95%
known coverage 均保持一致。

| 指标 | none 对照 | distance-consensus | treatment - baseline |
|---|---:|---:|---:|
| AUROC | 0.7180 | 0.7012 | -0.0167 |
| AUPR | 0.5938 | 0.5716 | -0.0222 |
| FPR95 | 0.7635 | 0.7800 | +0.0165 |
| OSCR | 0.3914 | 0.3720 | -0.0193 |
| 已知接受率 | 95.92% | 94.75% | -1.17 个百分点 |
| 未知拒绝率 | 13.53% | 13.18% | -0.35 个百分点 |
| score KS | 0.3348 | 0.3041 | -0.0308 |

结论：seed=42 的正向结果没有在 seed=123 复现，且 treatment 的排序指标、固定
已知覆盖率工作点和分布分离指标均变差。因此该方法只能保留为可复现实验开关和
消融/诊断项，不能设为默认，也不再继续盲目搜索 temperature 或融合比例。当前
主线应回到不加该权重的 audited nnPU，并转向直接作用于特征空间的表征学习和
数据协议审计。详细记录见
`analysis/distance_consensus_full_recheck_20261011.md`；产物见
`runs/distance_consensus_full_s123_baseline_discover/` 和
`runs/distance_consensus_full_s123_treatment_discover/`。

这次复核也再次说明：unknown reject 单独上升不能证明算法有效，必须同时检查
AUROC、FPR95、OSCR、known accept 和 score KS；如果只改变阈值附近工作点而没有
降低整体 overlap，就不能宣称核心问题得到解决。

进一步的组件诊断显示，seed=123 treatment 并非所有信号都变差：最终
`normalized_entropy_proto` 的 AUROC/KS 从 `0.7180/0.3348` 降到
`0.7012/0.3041`，prototype distance 从 `0.7042/0.3053` 降到
`0.6851/0.2812`，entropy 从 `0.6994/0.2993` 降到 `0.6863/0.2785`；只有
uncertainty head 自身从 `0.6087/0.1741` 升到 `0.6415/0.2072`。这说明该 PU
权重确实强化了 uncertainty head 的排序，却损害了最终检测器依赖的原型距离和
熵几何，属于训练目标冲突，而不是简单阈值问题。后续应优先研究如何解耦或协调
uncertainty head 与已知类表征目标，而不是继续调同一个权重来源。

## 2026-10-11：局部已知支持几何的成对复核

本轮针对“已知/未知分布重叠”测试了训练端局部支持几何。新增的
`known_support` 使用已知训练样本建立每类 leave-one-out kNN cosine 支持半径，
将混合 discovery pool 中相对远离已知支持的样本赋予更高的 nnPU 未标注项权重；
`support_consensus` 将该信号与 prototype distance、entropy、MSP 等信号等权融合；
`support_blended` 保留原 `distance_consensus` 的 75% 权重，只加入 25% 局部支持排名。
这些权重仍是启发式样本权重，不是严格无偏 nnPU 估计，也没有使用未知标签训练。

代码审计方面，支持库构建现在通过 `torch.random.fork_rng` 恢复 CPU/CUDA RNG，
避免启用支持库后改变训练增强和 batch 顺序；候选选择器 CLI 不接受这些仅属于
PU 权重的模式。全量测试为 `201 passed`。

检测协议固定为 CIFAR-100 random 60/40、seed、预训练 teacher、1200/300/1000
数据限制、2 epochs、matched mixed pool 400、known prior 0.2、
`normalized_entropy_proto`、验证集 95% known coverage。只改变 PU 权重来源：

| seed | 方法 | AUROC | FPR95 | OSCR | known accept | unknown reject | KS distance |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | distance_consensus | 0.6331 | 0.8430 | 0.1915 | 93.22% | 8.86% | 0.2223 |
| 42 | known_support | 0.6223 | 0.8777 | 0.1732 | 96.69% | 7.34% | 0.2079 |
| 42 | support_consensus | 0.6232 | 0.8165 | 0.2111 | 93.22% | 6.84% | 0.1959 |
| 42 | support_blended | 0.6317 | 0.8132 | 0.1929 | 95.04% | 7.85% | 0.2277 |
| 123 | distance_consensus | 0.5772 | 0.8844 | 0.2588 | 95.48% | 7.94% | 0.1131 |
| 123 | support_blended | 0.5755 | 0.8878 | 0.2311 | 93.97% | 7.94% | 0.1338 |

判断：`known_support` 单独使用无效；`support_consensus` 只改善 seed=42 的
FPR95/OSCR 工作点，整体排序和未知拒绝率没有改善；`support_blended` 在 seed=42
的工作点较好，但 seed=123 未复现，两个 seed 的 AUROC 均低于对照。局部支持信号
在两 seed 上提高了 KS distance，却没有稳定提高 AUROC，说明它可能只改变分数的
局部形状或校准，而没有可靠解决已知/未知特征重叠。因此当前主线仍是
`distance_consensus`，支持几何保留为可选消融与诊断，不纳入默认配置，也不继续
盲目搜索更多融合比例。

下一步应转向更根本的表征学习/协议审计：检查 uncertainty head 的 PU 目标是否与
分类蒸馏目标冲突，并测试能直接作用于特征空间的、严格单变量的未知代理或类条件
边界目标；同时继续报告 AUROC、OSCR、known accept、unknown reject 与 score KS，
不能只依据某个阈值工作点的拒绝率判断有效。

## 2026-10-11：围绕核心重叠问题的继续审计与组合尝试

本轮继续围绕“已知/未知特征与分数分布重叠，未知拒绝率低”做小步迭代。为了避免只看 unknown reject 而误判，代码新增了 score_distribution 诊断：discover 报告现在会输出 known/unknown 分数均值、分位数、KS distance、unknown 落在 known 95% 分位以下的比例、阈值处 known/unknown 的实际拒绝比例。这个诊断不改变模型，只帮助判断方法是否真的拉开分布。

对昨天最有希望的 distance_consensus weighted nnPU 做了分布复核。结论是：weighted PU 能提高 unknown 分数均值、降低一部分阈值附近重叠，但 KS distance 没有同步明显变好，说明它还没有真正整体解决分布重叠，更像是在高分尾部和工作点附近改善。

本轮尝试的组合方法中，distance_consensus PU + consensus feature-margin 得到 AUROC 0.5956、FPR95 0.8793、unknown reject 9.11%，差于 PU alone 的 AUROC 0.6034、FPR95 0.8496、unknown reject 9.37%，所以暂不保留。teacher known-anchor consistency 的普通门控得到 AUROC 0.5708、unknown reject 6.33%，训练日志中锚点真实已知比例约 52%；更严格门控得到 AUROC 0.5954、FPR95 0.8281、unknown reject 8.10%，但锚点真实已知比例反而约 37%，说明伪已知锚点筛选仍不可靠。candidate-gated KNN boundary 修复维度错误后可以运行，但 AUROC 0.5630、AUPR 0.4230、unknown reject 4.81%，也暂不纳入主线。

本轮还发现一个代码问题：candidate-gated KNN boundary 的 soft weighting 分支在 distance_consensus 模式下错误使用了 projection 特征与 classifier prototype 相乘，出现 64x128 与 512x60 维度不匹配。现在已改为使用 backbone features，并通过测试。这个修复是必要的代码健壮性改动，但 KNN boundary 组合本身的短实验结果仍不支持纳入主线。

检测分数复核显示，同一个 distance_consensus weighted PU 模型下，不同 score-mode 差异明显。normalized_entropy_mahalanobis 为 AUROC 0.6034、AUPR 0.4808、FPR95 0.8496、OSCR 0.1732、unknown reject 9.37%；normalized_entropy_proto 提升到 AUROC 0.6331、AUPR 0.4908、FPR95 0.8430、OSCR 0.1915，是当前最强 AUROC/OSCR 检测信号；normalized_entropy_relative_mahalanobis 也达到 AUROC 0.6261、OSCR 0.1916。新增的 normalized_entropy_proto_relative_mahalanobis 为 AUROC 0.6327、AUPR 0.4913、FPR95 0.8463，与 proto 接近但没有明显超过 proto。prototype_distance 训练配合 normalized_entropy_proto 检测得到 AUROC 0.6083、FPR95 0.8083、unknown reject 10.38%，工作点更激进，但排序不如 consensus 训练。

阶段判断：训练端目前仍以 distance_consensus weighted nnPU 最值得继续；检测端应把 normalized_entropy_proto 加入主报告，因为它更能利用当前特征中的几何信号。prototype_distance 训练能提高 unknown reject/FPR95，但会牺牲 known accept 和 AUROC；更适合当作偏激进工作点的对照，而不是替代主线。

下一步建议：优先做 distance_consensus weighted nnPU + normalized_entropy_proto 的 3 seed 完整复核，并同步报告 score_distribution。如果 proto score 在多 seed 上稳定更好，后续应考虑把 prototype-distance 信号更温和地反馈到训练，而不是继续叠加当前已经验证无效的 feature-margin、known-anchor 或 KNN boundary。

## 2026-10-11：语义距离加权 mixed-pool uncertainty nnPU

本轮继续围绕核心问题“已知/未知特征与分数分布重叠，未知拒绝率低”进行修改。上一轮审计确认：直接修正 mixed-pool uncertainty nnPU 公式后，相比 baseline 有稳定提升，但三种子完整协议的 unknown reject 平均仍只有约 11.19%，说明不能继续只调阈值或简单扫全局 prior。

本轮新增一个可关闭的训练开关：--discovery-uncertainty-pu-weight-source，支持 none、prototype_distance、msp、entropy、distance_consensus。它只作用于 mixed-pool uncertainty nnPU 的 unlabeled 项：对更像未知的混合池样本赋更高软权重，对更像已知的样本赋更低权重；损失内部会把权重归一化到均值 1，避免单纯改变 loss 尺度。该方法不是严格无偏 nnPU 的新公式，而是一个受 prototype distance、MSP/entropy OOD 信号启发的语义加权启发式，用于缓解 mixed pool 中已知样本污染未知训练信号的问题。

代码验证已经完成：python -m pytest -q 通过，结果为 197 passed, 2 warnings；python -m compileall -q train.py novel_discovery tests 通过。新增测试保证全 1 unlabeled 权重与原 nnPU 等价，并检查权重形状错误会报错。

短实验协议固定为：CIFAR-100 random 60/40，seed=42，limit-train/val/test=1200/300/1000，预训练 ResNet-34 teacher / ResNet-18 student，2 epochs，matched mixed pool=400，known ratio/prior=0.2，检测器固定为 normalized_entropy_mahalanobis，阈值固定为 validation-only 95% known coverage，跳过聚类。只改变 uncertainty nnPU 的 unlabeled 权重来源。

| 配置 | AUROC | AUPR | FPR95 | OSCR | known acc all known | known accept | unknown reject | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| unweighted 当前对照 | 0.5982 | 0.4597 | 0.9273 | 0.1591 | 22.64% | 95.54% | 3.80% | 当前短实验基线 |
| prototype_distance | 0.6004 | 0.4758 | 0.8579 | 0.1704 | 25.62% | 92.73% | 12.66% | 未知拒绝提升最大，但已知接受下降明显 |
| msp | 0.6008 | 0.4759 | 0.8777 | 0.1650 | 23.80% | 94.55% | 9.87% | 有提升，但 FPR95 不如 consensus |
| distance_consensus, temperature=0.5 | 0.6034 | 0.4808 | 0.8496 | 0.1732 | 24.46% | 95.21% | 9.37% | 当前最平衡，优先保留继续验证 |
| distance_consensus, temperature=0.25 | 0.5991 | 0.4706 | 0.8595 | 0.1547 | 21.65% | 96.36% | 6.84% | 权重过尖锐后效果下降 |

阶段结论：语义加权 PU 是目前比盲目扫 prior 更有价值的方向。distance_consensus 在保住 known accept 的同时，改善 AUROC/AUPR/FPR95/OSCR 和 unknown reject；单独 prototype_distance 能更强地拒绝未知，但会误拒更多已知。下一步不建议继续扩大无目的方法搜索，而是围绕 distance_consensus 做多 seed 和更长训练验证，并检查它是否真的拉开特征/分数分布，而不只是改变阈值工作点。

后续优先事项：1）用完整测试集和 seed=42/123/3407 复核 distance_consensus；2）增加特征/分数分布诊断，比较加权前后已知/未知 overlap 是否下降；3）如果多 seed 仍有效，再考虑和已有的 uncertainty feature-margin 或 KNN support-boundary 组合；4）如果只提升 unknown reject 但不提升 AUROC/OSCR，则说明它只是改变工作点，需要转向更强的表征学习或数据划分/语义难度审计。

## 2026-10-11：不确定性 nnPU 公式审计与 warmup 小实验（重要修正）

本轮先审计了 mixed-pool 不确定性损失的标签定义，而不是继续堆叠检测分数。`nnpu_known_uncertainty_loss` 中 `u` 表示未知概率，`z = log((1-u)/u)` 表示已知性 logit；已知训练样本是正类，mixed discovery pool 是包含已知/未知样本的无标注混合分布。因此标准混合池风险中的负类项应为 `softplus(z)`。

审计发现，旧的 `nu_corrected` uncertainty-head 分支错误地对 mixed pool 使用了 `softplus(-z)`，这与当前“已知为正类”的定义相反。现在代码已修正：`legacy` 和 `nu_corrected` 两个名称都使用数学一致的风险表达式；两个名称保留是为了兼容历史配置和复现实验，不再把它们视为两个不同的 uncertainty-head 算法。注意：这项修正只针对训练时的 uncertainty head；post-hoc feature rejector 使用的是相反的“未知为正类”操作约定，未随本次修改改变。

在 CIFAR-100 random 60/40、seed=42、预训练 ResNet-34/18、matched mixed pool=400、known prior=0.2、1200/300/1000 数据限制、2 epochs、相同 `normalized_entropy_mahalanobis` 检测器和 validation-only 95% known-coverage 下进行了小规模审计：

| 配置 | AUROC | FPR95 | OSCR | known acceptance | unknown rejection |
|---|---:|---:|---:|---:|---:|
| 直接 nnPU（legacy 名称） | 0.6003 | 0.8843 | 0.1781 | 93.72% | 8.86% |
| 审计后的公式（nu_corrected 名称） | 0.6119 | 0.8760 | 0.1637 | 94.88% | 6.33% |
| 审计公式 + 1 epoch warmup/1 epoch ramp | 0.5753 | 0.8959 | 0.1652 | 95.70% | 7.09% |

这轮结果不能证明公式名称之间存在算法增益；两次训练均为独立 smoke run，修正后两个名称对应同一目标。warmup/ramp 在短训练预算下没有改善排序，暂不设为默认方案。核心问题“已知/未知特征与分数分布重叠、未知拒绝率低”仍未解决。完整公式推导、命令和结果见 `analysis/uncertainty_nnpu_formula_audit_20261011.md`。

随后完成了 CIFAR-100 random 60/40、seed=42/123/3407、完整测试集、5 epochs、预训练 ResNet-34/18、matched mixed pool、相同 `normalized_entropy_mahalanobis` 检测器和 validation-only 95% known-coverage 的三种子复核。对比变量只看训练端是否使用审计后的 mixed-pool uncertainty nnPU；旧 `treatment` 使用修正前公式，因此只能作为历史/部分审计参考。

| 方法 | AUROC | AUPR | FPR95 | OSCR | known acc | known accept | unknown reject |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline（三种子均值） | 0.5977 | 0.4623 | 0.8632 | 0.3052 | 41.28% | 94.61% | 7.11% |
| 旧 treatment（历史/部分审计） | 0.6491 | 0.5226 | 0.8272 | 0.3718 | 48.11% | 95.24% | 10.44% |
| 审计后 direct nnPU | 0.6518 | 0.5209 | 0.8160 | 0.3767 | 50.07% | 94.54% | 11.19% |

结论：审计后的 direct nnPU 相对 baseline 在三种子上稳定改善 AUROC、FPR95、OSCR、已知分类准确率和未知拒绝率，说明这个方向有继续深入价值；但未知拒绝率均值仍只有约 11.19%，大量未知样本仍被接受，不能宣称核心问题已解决。相对旧 treatment，审计版平均 FPR95、OSCR、known acc 和 unknown reject 更好，但 seed=3407 的 AUROC/AUPR 低于旧 treatment，说明仍有 seed 敏感性。

下一步不应继续只调阈值或堆叠 score fusion。更合理的改进方向是围绕 mixed-pool nnPU 本身做单变量深入：检查 known prior 与 mixed-known-pool-ratio 是否匹配真实池分布；尝试类条件/语义距离感知的 prior 或权重；把 uncertainty-head 训练诊断和特征重叠统计绑定报告；必要时重新审查 discovery pool 划分是否过于接近已知类，导致已知/未知天然难分。

补充做了一个轻量 prior sensitivity pilot：保持 seed=42、CIFAR-100 random 60/40、1200/300/1000 限制、matched pool=400、2 epochs、同一 teacher 和同一检测器，只把 matched mixed pool 的 known ratio / nnPU known prior 从 0.2 改到 0.1 或 0.3。0.1 得到 AUROC 0.6075、FPR95 0.8876、unknown reject 3.80%；0.3 得到 AUROC 0.5520、FPR95 0.9190、unknown reject 2.78%。二者都差于 0.2 smoke（AUROC 0.6119、FPR95 0.8760、unknown reject 6.33%）。因此当前不建议继续盲扫全局 prior；下一步应尝试类条件/语义距离感知的样本权重或 prior，而不是把所有 mixed-pool 样本用同一个已知比例解释。

## 2026-10-10：PU 能量边界配对实验（未纳入主线）

针对 mixed discovery pool 中“不能把全部样本都当作未知”的问题，本轮新增
`--alpha-discovery-energy-pu`。它使用已知样本提供能量参考，并按照
`known_prior=0.2` 对混合池风险进行 PU 校正；与原有的纯未知能量损失不同，
不会给 mixed pool 中每个样本硬贴未知标签。

代码接线先经过 toy smoke 检查。最初发现训练循环条件漏掉了新参数，导致参数
虽然可解析但损失没有执行；已修复并重新验证。正式实验中训练日志的
`discovery_energy_pu` 非零，说明损失确实进入反向传播。

严格配对条件为 CIFAR-100 semantic-isolated 60/40、seed=42、完整数据、
预训练 ResNet-34/ResNet-18、3 epochs、5400 张 matched mixed pool、known
prior=0.2，以及相同的 support-augmented nnPU rejector 和 95% known-coverage
校准。唯一变量是 PU-energy 权重 `0 -> 0.1`。

| 指标 | baseline | PU-energy | 变化 |
|---|---:|---:|---:|
| AUROC | 0.7703 | 0.7637 | -0.0066 |
| FPR95 | 0.6423 | 0.6483 | 变差 |
| OSCR | 0.3544 | 0.3508 | 变差 |
| 已知接受率 | 94.35% | 95.23% | +0.88 个百分点 |
| 未知拒绝率 | 20.25% | 20.43% | +0.18 个百分点 |

结论：未知拒绝率只有很小的工作点变化，但 AUROC、FPR95 和 OSCR 均变差，
不能认为该方法有效，也不进入主线。它保留为可复现实验开关，不继续盲目扫权重。
分数审计还显示 known 分布整体略向 unknown 方向移动，而 unknown 分布没有同步远离，
因此这不是形成了新的已知/未知间隔，而是检测排序变差。
详细记录见 `analysis/pu_energy_margin_paired_s42_e3_20261010.md`。

### 2026-10-12：PU-energy 在当前 LR=0.1 配方上的双 seed 复核

为确认此前 seed-42 的负向 PU-energy 结果是否依赖训练配方，本次在当前
CIFAR-100 semantic-isolated 完整数据配方上复核：固定 seed=44、5 epochs、
encoder LR scale=0.1、matched mixed pool=5400/known prior=0.2、已有
uncertainty nnPU 权重=0.1、相同检测器/MC=4/独立验证集 95% known-coverage
校准，只额外启用 `alpha_discovery_energy_pu=0.1`。保存配置核对显示有效训练变量
只有这一项；训练历史的 `discovery_energy_pu` 每个 epoch 均非零，确认损失确实接入。

| 指标 | baseline | PU-energy | 变化 |
|---|---:|---:|---:|
| AUROC / AUPR | 0.8089 / 0.7037 | 0.8152 / 0.7169 | 小幅提升 |
| FPR95 / OSCR | 0.5888 / 0.5217 | 0.5942 / 0.5204 | 均变差 |
| 已知分类准确率 | 58.10% | 57.77% | -0.33 pp |
| 已知接受率 | 93.88% | 94.82% | +0.93 pp |
| 未知拒绝率 | 30.50% | 27.85% | -2.65 pp |
| KS / unknown-below-known-q95 | 0.4807 / 0.7385 | 0.4899 / 0.7273 | 混合信号 |

结论：不能判定有效。排序 AUROC 与部分分布指标略有改善，但 FPR95、OSCR、已知分类和未知拒绝率没有共同改善，平均分数间隔也几乎不变；这更像权衡/小幅扰动，而不是稳健分离已知与未知。结合 seed-42 旧协议下 AUROC、FPR95、OSCR 均变差的结果，不把 PU-energy 纳入默认主线，也不继续盲扫其权重。后续先审查混合池 prior 假设与能量梯度作用，再做至少双 seed 配对复核，并同时报告排序、操作点和已知类代价。

随后在 seed=123 上按同样条件复核，AUROC `0.8191 -> 0.8265`、AUPR `0.7255 -> 0.7383`，但 FPR95 `0.5797 -> 0.5845`，OSCR `0.5602 -> 0.5603`（基本不变），known accuracy `62.20% -> 61.82%`，unknown rejection `25.10% -> 29.58%`。两 seed 的排序指标小幅同向，但 FPR95 都变差，unknown rejection 在 seed 44 下降 2.65 pp、seed 123 上升 4.48 pp，平均分数间隔只变化约 `+0.01`。因此双 seed 证据仍为混合结果，PU-energy 不进入默认主线，也停止权重扫描。优先回到 PU 先验/批采样假设审计及损失梯度几何诊断；不要将工作点变化表述为分布重叠已解决。

完整条件、双 seed 数值和 artifacts 见 `analysis/pu_energy_paired_s44_20261012.md`。


## 2026-10-10：rejection-PU 特征间隔配对实验结论

本轮针对“已知/未知特征和分数仍然重叠”的核心问题，在独立
`rejection_features` 分支上加入了混合池 PU 特征间隔损失。为了避免把
结构、数据或阈值差异误判成算法收益，进行了严格配对：CIFAR-100
semantic-isolated 60/40、seed=42、相同教师/学生、相同 3 epoch、相同
5400 样本 mixed discovery pool（known prior=0.2），只改变
`alpha_discovery_rejection_pu_feature_margin`（0 vs 0.1）。检测也固定为
support-augmented linear `nu_corrected` nnPU rejector、95% known-coverage
校准和 `feature_rejector` 分数。

| 指标 | baseline | PU feature-margin | 差值 |
|---|---:|---:|---:|
| AUROC | 0.7823 | 0.7762 | -0.0062 |
| FPR95（越低越好） | 0.6318 | 0.6263 | -0.0055 |
| OSCR | 0.3723 | 0.3701 | -0.0022 |
| 已知接受率 | 94.57% | 95.25% | +0.68 个百分点 |
| 未知拒绝率 | 23.15% | 21.08% | -2.08 个百分点 |

训练日志中 baseline 的新损失始终为 0，treatment 在三轮约为
`0.002105 / 0.0000045 / 0.0000050`，因此代码路径和梯度连接是有效的，
但损失很快饱和。综合指标不能支持该方法有效：它只带来很小的 FPR95
改善，同时降低 AUROC、OSCR 和未知拒绝率。本方法暂不纳入主线，也不再
仅通过调大权重或调阈值继续试验。详细记录见
`analysis/rejection_pu_feature_margin_paired_s42_e3_20261010.md`。

下一步应检查并改造 PU 特征风险本身：当前实现使用非负风险的硬
`clamp_min(0)`，混合风险减去已知校正项后可能迅速归零，导致没有持续的
未知分离梯度。后续先实现一个可记录、非饱和的 PU 风险变体，再保持同一
配对协议验证；如果仍无改善，就停止这条 rejection-margin 路线，转向
更直接的已知支持建模或训练时的能量/密度边界学习，而不是继续阈值搜索。

## 2026-10-09: uncertainty loss 2x2 component screen

To avoid attributing the previous positive signal to an inseparable loss
combination, a paired 2x2 pilot was added. It independently toggles mixed-pool
uncertainty nnPU and uncertainty-weighted feature margin while keeping the
teacher, split, matched discovery pool, rejector, and 95% known-coverage
calibration fixed. The 3-epoch seed-42 results were:

| arm | AUROC | FPR95 | OSCR | known accept | unknown reject |
|---|---:|---:|---:|---:|---:|
| baseline | 0.6758 | 0.8020 | 0.1189 | 95.51% | 8.27% |
| nnPU only | 0.7191 | 0.7072 | 0.2170 | 94.68% | 17.79% |
| feature margin only | 0.7232 | 0.7488 | 0.1821 | 97.50% | 9.52% |
| combined | 0.7126 | 0.7404 | 0.1811 | 96.01% | 13.28% |

The current evidence favors nnPU as the more useful component for the fixed
operating point: unknown rejection increases by 9.52 percentage points over
baseline. Feature margin mainly improves ranking in this pilot, while the
combined arm does not show synergy after only three epochs. This is still a
single-seed, limited-data pilot and does not prove that feature overlap is
solved. A 10-epoch paired run is being used to distinguish a short-training
optimization conflict from a genuinely weak combination. Details are in
`analysis/uncertainty_2x2_pilot_s42_20261009.md`; the runner is
`scripts/run_uncertainty_2x2_pilot.ps1`.

## 2026-10-09: 独立 rejection 表征上的 PU 特征间隔损失

### 本轮修改

针对“已知/未知共享特征和分数仍然重叠”的核心问题，本轮增加了一个独立的
`rejection_features` 分支，并在该分支上加入混合 discovery pool 的 PU 特征间隔
损失。它不把混合池全部当成未知，而是使用已知先验做风险修正：先计算混合池中
样本到已知 rejection prototypes 的相似度间隔损失，再减去估计的已知贡献，得到
未知部分的近似风险。主分类分支的 CE、KD 和原有损失保持不变。

相关实现位于 `novel_discovery/losses.py` 和 `novel_discovery/pipeline.py`，命令行入口
是 `--rejection-feature-dim`、`--alpha-discovery-rejection-pu-feature-margin` 和
`--discovery-rejection-feature-margin`。该项要求 `--discovery-pool-mode mixed`，并且
要求显式设置 `--rejection-feature-dim > 0`；默认权重为 0，因此不会改变历史配置的
行为。训练日志会记录 `discovery_rejection_pu_feature_margin`，便于确认该项是否真的
激活。

### 本轮验证与结论

已完成一轮 toy smoke：1 个 epoch、32 个训练样本、80 个 mixed discovery pool 样本、
独立 rejection 表征维度 128。训练正常保存 checkpoint，日志中的新损失为
`0.301456`，说明在该构造下损失非零并进入训练计算；此前间隔过大时损失为 0，已据此
增加日志和函数级梯度测试。该结果只证明代码路径和梯度路径可运行，不能证明未知检测
效果，也不能与 CIFAR-100 的完整实验直接比较。

当前仍没有证据表明该方法解决了核心问题。下一步应在固定 CIFAR-100 语义隔离
60/40、相同教师/学生、相同 discovery pool、训练轮数、checkpoint 选择和独立阈值校准
协议下，做 baseline 与该 rejection-PU treatment 的配对对照；至少比较 AUROC、FPR95、
OSCR、known accuracy/acceptance、unknown rejection，以及特征/分数分布重叠。只有在
多个 seed 上提升排序和未知拒绝，且不明显增加已知误拒后，才考虑保留到主线。

这一路线的思路来自 PU learning 的混合池风险分解和 open-set recognition 中显式拒绝
空间的做法，但当前实现不是某篇论文的完整复现。它暂时属于待验证实验分支，不应把
toy smoke 的非零 loss 写成算法有效性结论。

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

## 2026-10-07 当前状态总览

这几天的代码和实验已经完成一次较严格的归因审计。下面的结论按“完整协议结果”和“短 pilot / 消融结果”分开，避免把不同数据规模、split、训练轮数或检测器下的数值直接混比。

### 当前保留的主线

- 训练主线：mixed discovery pool 上的 `nnPU` uncertainty loss，加上原始的 uncertainty-weighted feature-margin。
- 检测主线：学生模型的 `normalized_entropy_min_class_knn`；support-only nnPU rejector 和 `0.75` score fusion 作为可选检测器，不默认替换。
- 聚类主线：先检测未知候选，再使用 `feature_pca` / `projection_pca` 与 KMeans 等方法聚类；自动估计 K 仍单独报告，不能用 oracle K 的结果代替真实部署结果。

完整 matched 三 seed 的 nnPU + uncertainty-margin treatment 平均结果为：AUROC `0.6902`、FPR95 `0.7675`、未知拒绝率 `14.03%`。这说明该训练方向有可复现的正向信号，但未知样本仍大量被接受，不能表述为已经解决开放集检测。

在冻结学生特征上训练的 support-only nnPU rejector / score fusion 能进一步改善排序；已有三 seed 对照中 support-only AUROC 约 `0.7531`，fusion AUROC 约 `0.7595`。这属于解耦的后处理检测模块，不能误写成学生 backbone 已经完全分离了已知和未知特征。

### 最近确认的核心问题

- 已知与未知特征仍明显重叠。seed=42 的后验几何审计中，nnPU + margin 相比无 discovery treatment 的三种距离未知性 AUROC 从约 `0.569/0.534/0.575` 提升到 `0.637/0.605/0.621`，但直方图重叠仍约 `0.79–0.84`。
- 当前训练方法改善了部分表征和 uncertainty ranking，但提升没有稳定传递为高未知拒绝率；不能继续把问题归结为阈值选择。
- 当前联合 novel head、批内伪标签、uniform OE、独立 rejection branch 等方向没有形成稳定正向证据，不能因为某个指标或某个 seed 变好就吸收到主流程。

### 最近已判定为负向或暂不主推的尝试

严格对照下，uniform-only OE、Angular margin、Reciprocal Points、独立 rejection branch、soft support separation、独立 rejection + OE 组合，以及 PU-corrected uncertainty feature-margin 均未改善整体开放集效用；它们保留为关闭状态的消融记录。详细数字见对应的 `analysis/` 文件。

### 当前下一步

独立 rejection head 上的 uniform OE 已完成完整数据三随机种子配对复核。seed=42 有改善，seed=43 基本持平/部分指标下降，seed=44 的 AUROC、FPR95、OSCR 和 known accuracy 变差；因此它是效果混合的 opt-in 消融，不是已验证有效的主方法。完整结果见
`analysis/rejection_uniform_oe_full_data_s42_20261007.md`。

历史主线实验中，checkpoint 选择与阈值校准曾共用 validation split；2026-10-08 的 seed=42 配对复核已改用互斥子集，但只验证了这一组模型，不能覆盖旧实验结论。接下来应先保存并核对校准/模型选择样本索引，再用至少 3 个配对 seed 复核该训练改动；同时报告 AUROC、FPR95、OSCR、known accuracy/acceptance、unknown rejection 和 feature-overlap，聚类指标单独报告，不能根据测试集标签调参。

## 2026-10-08 独立校准集配对复核

为检查之前的训练改动是否依赖验证集阈值校准，本次在 CIFAR-100 语义隔离 60/40、seed=42 上，对 baseline 与 treatment 使用同一检测协议重新评估。检测配置固定为：完整测试集（6000 known / 4000 unknown）、matched discovery pool 5400（已知先验 0.2）、nnPU feature rejector（`support_augmented`）、独立校准比例 0.4、目标已知覆盖率 95%；聚类跳过，因此本次只评价分类与未知检测，不评价新类聚类。

训练对照为同 seed、同教师/学生架构与训练预算的配对模型。baseline 不启用 discovery uncertainty nnPU loss 与 uncertainty-weighted feature margin；treatment 启用 `alpha_discovery_uncertainty_pu=0.1`、known prior `0.2`、`nu_corrected` risk，以及 `alpha_discovery_uncertainty_feature_margin=0.05`、margin `0.2`。两者按相同验证集已知分类准确率规则选 checkpoint，baseline 选到 epoch 10，treatment 选到 epoch 8；treatment 最佳验证准确率为 0.5167，baseline 为 0.4701。检测校准集与 checkpoint 选择子集在报告中标记为互斥（1823/1177 个样本）。

| 指标 | baseline | treatment | treatment - baseline |
|---|---:|---:|---:|
| AUROC | 0.7664 | 0.7743 | +0.0079 |
| AUPR | 0.6425 | 0.6496 | +0.0070 |
| FPR95（越低越好） | 0.6357 | 0.6203 | -0.0153 |
| OSCR | 0.3842 | 0.4296 | +0.0454 |
| 已知分类准确率（全体 known） | 0.4433 | 0.4987 | +0.0553 |
| 已知接受率 | 0.9587 | 0.9542 | -0.0045 |
| 未知拒绝率 | 0.1663 | 0.1753 | +0.0090 |
| 已知误拒数量 / 6000 | 248 | 275 | +27 |
| 未知正确拒绝数量 / 4000 | 665 | 701 | +36 |

分数诊断显示，treatment 的 entropy AUROC 从 0.6746 升至 0.7010，prototype distance AUROC 从 0.6716 升至 0.7125，epistemic AUROC 从 0.6260 升至 0.6536；但单独的模型 uncertainty head AUROC 从 0.6664 变为 0.6620，没有改善。这个模式更像是分类置信度/原型几何有所改善，而不是不确定性分支本身提供了更好的未知排序。最终拒绝率虽增加约 0.9 个百分点，已知误拒也增加 27 个，因此不能把拒绝率增量单独解释为有效分离。

**结论与限制：** 这是一个完整测试集上的单 seed pilot，只有小幅正向信号，不能证明方法稳定有效，也没有验证聚类效果。未知拒绝率仍只有 17.53%，仍有 3299/4000 个未知样本被接受，核心问题没有解决。历史 JSON 只记录样本数和互斥标志；事后用保存的配置与当前确定性划分函数重建 seed=42 两臂数据划分，模型选择/校准分别为 1823/1177 个样本，实际索引交集均为 0。为后续实验增加了索引审计：新报告保存底层索引空间、数量、SHA-256 和交集计数。若多 seed 仍能提升排序且不以明显增加已知误拒为代价，再保留该训练改动；否则应回到表征/数据协议诊断，而不是继续调阈值。实验产物位于 `runs/semantic_isolated_cal40_s42_{baseline,treatment}_detect/`，训练配置与历史位于对应的 `runs/semantic_isolated_cal40_s42_{baseline,treatment}/`。

### seed=43 配对复核

沿用 seed=42 的唯一对照变量和评估协议，使用 seed=43 独立训练 baseline/treatment，并在完整 CIFAR-100 测试集上比较。两组均使用同一 seed、数据划分、教师模型、网络、训练预算、matched discovery pool（5400，known fraction/prior 均为 0.2）和独立校准比例 0.4；treatment 额外启用 uncertainty nnPU（权重 0.1、`nu_corrected`、prior 0.2）和 uncertainty-weighted feature margin（权重 0.05、margin 0.2）。两组配置差异还包括预期的输出路径和根据验证准确率选出的 checkpoint epoch（baseline 6、treatment 10）；baseline 未启用 nnPU 时其 prior 设置为 `auto`，该项不参与损失计算。校准与 checkpoint 选择子集分别为 1175/1825 个样本，报告标记为互斥，但没有保存样本 ID，因此仍未完成索引级互斥审计。

| 指标 | baseline | treatment | treatment - baseline |
|---|---:|---:|---:|
| AUROC | 0.7722 | 0.7942 | +0.0220 |
| AUPR | 0.6503 | 0.6728 | +0.0225 |
| FPR95（越低越好） | 0.6343 | 0.5717 | -0.0627 |
| OSCR | 0.3877 | 0.4363 | +0.0486 |
| 已知分类准确率（全体 known） | 0.4525 | 0.4982 | +0.0457 |
| 已知接受率 | 0.9478 | 0.9505 | +0.0027 |
| 未知拒绝率 | 0.2003 | 0.2140 | +0.0138 |
| 已知误拒数量 / 6000 | 313 | 297 | -16 |
| 未知正确拒绝数量 / 4000 | 801 | 856 | +55 |

seed=43 在排序指标、OSCR、known accuracy 和拒绝计数上均优于 baseline，方向与 seed=42 一致，是值得继续复核的正向信号。特别是已知误拒数没有随未知拒绝增加而变多，说明这次拒绝增益不只是把更多已知样本错误拒掉。但仍有 3144/4000 个未知样本被接受，AUROC/FPR95 也离理想水平较远；两个 seed 仍不足以证明稳定收益。用相同配置重建两臂的 seed=43 数据划分后，校准/选择集数量为 1175/1825，实际底层索引交集为 0。新代码会在后续运行报告里直接写入索引哈希与交集计数。该结果只支持“继续做多 seed 配对验证”，不支持把 treatment 宣称为已解决核心问题或默认主方法。下一步完成第三个配对 seed，并汇总逐 seed 差值和波动。产物位于 `runs/semantic_isolated_cal40_s43_{baseline,treatment}/` 与对应的 `_detect/` 目录。

### 校准切分审计实现

开放集评估报告新增 `validation_protocol.sample_index_audit`，解析嵌套 `Subset` 与 `ConcatDataset` 的底层位置索引，保存模型选择/阈值校准的索引数量、排序索引集合 SHA-256、交集数量及 `disjoint` 布尔值；不保存样本内容或逐条索引。该字段是审计证据，不改变训练或阈值算法。旧 seed=42/43 报告没有此字段，已用相同保存配置重建划分并确认交集为 0；未来报告可直接验证。新增单元测试覆盖嵌套子集及拼接数据集索引映射。

第三个配对随机种子可从保存配置复现：`python scripts/run_paired_seed.py --seed 44`。脚本以 seed=43 的已保存 baseline/treatment 配置为模板，生成同一个 seed=44 教师、两臂学生和两臂完整检测报告，所有产物写入新的 `runs/semantic_isolated_cal40_s44_*` 目录。当前 treatment 同时增加 nnPU 与 feature-margin 两项损失；即使第三 seed 方向一致，也只能支持“这组损失组合值得进一步验证”，不能区分两项各自的贡献。下一阶段应在固定 seed 和评估协议下做小型 2×2 消融（两项分别关闭/开启），再决定是否保留或简化组合。

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

- 教师模型和学生模型支持 ResNet-18、ResNet-34、MobileNetV3-Small；ResNet 还支持显式启用 `--cifar-stem`，使用适合 CIFAR 小图像的 3×3、stride=1 首层并移除初始最大池化。
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
- 新增借鉴 Sinkhorn-Knopp 平衡分配思路的 balanced assignment，目标是缓解 prototype 分配塌缩；这只是局部机制借鉴，不是 UNO 完整复现。
- 新增借鉴 SimGCD/SwAV 跨视图伪标签思路的双视图 novel consistency：使用平衡分配目标约束另一增强视图的 novel prototype 预测；训练流程与 SimGCD 并不等同。
- 新增借鉴 SCAN 邻域一致性动机的 batch 内 neighbor consistency：约束当前 minibatch 特征近邻具有相似 novel prototype 分布；没有构造 SCAN 式全局近邻图。
- 通过 --joint-discovery 显式开启，默认关闭；训练后额外保存 novel_head.pt。
- `--joint-head-temperature` 与 `--joint-assignment-temperature` 分开：前者控制 cosine prototype logits，默认 `0.2`；后者控制均衡分配，默认 `1.0`。
- --joint-confidence-threshold 使用相对均匀分布的置信度，即 max softmax probability 乘以 novel 类数量。默认值 1.1，适用于 CIFAR-100 的 40 个未知类。
- `discover` 可通过 `--novel-head-ckpt` 加载 `novel_head.pt`，使用 `--score-mode novel_msp` / `novel_entropy` 做检测，或使用 `--cluster-feature novel` / `novel_pca` 评估 prototype 表示。
- 新增可选 `--joint-space unified`：把已知分类 logits 与 novel prototype logits 拼成统一的 known+novel 空间，并支持 `unified_novel_mass` 检测分数和 `unified` / `unified_pca` 聚类特征。

这一版是从两阶段“检测后聚类”走向联合新类发现的最小实验模块，还没有实现完整 UNO/SimGCD 的周期性伪标签更新、类别均衡分配优化和未知类分类评测。它目前用于验证训练期 novel prototype 是否能学习结构，不应直接作为最终论文方法。

方法实现审计补充：`--joint-novel-mass` 产生的是训练用软样本权重，公式为把 known logits 和 novel prototype logits 拼接后 softmax 并求 novel 概率总和，再乘跨视图一致性；它不是校准好的 unknown posterior。`--joint-novel-neighbor-support` 当前只在每个训练 minibatch 内做特征 top-k，不是全局 memory bank/SCAN 图。历史检测对照使用 Mahalanobis 分数时，只能说明训练变化对 Mahalanobis 检测的间接影响。此前 weighted-Sinkhorn 对照未独立隔离 transport 与 loss reweighting，因果结论已撤回；现加入 `--joint-weighted-sinkhorn` 开关进行独立对照。完整审计与修正结果见 `analysis/method_fidelity_and_sinkhorn_audit_20260929.md`。

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
- 早期 baseline 和短 pilot 的未知检测 AUROC 约为 `0.59–0.61`、FPR95 约为 `0.86–0.89`；在完整 matched 协议下，当前 nnPU + uncertainty-margin treatment 平均达到 AUROC `0.6902`、FPR95 `0.7675`，support/fusion rejector 还能进一步改善排序，但这些结果仍不能说明特征重叠已解决。
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

## 2026-09-30 今日实验总结

今天停止了新的完整训练，集中复核已有尝试，并用统一的 CIFAR-100 semantic-hard 60/40 协议检查哪些方向真正改变了表征，而不是只改变了阈值。除特别说明外，测试标签只用于最终描述性统计，没有用于训练、阈值选择或模型选择。详细原始记录见 `analysis/` 目录中的同名实验报告。

### 今天尝试了什么

| 方向 | 思路来源 | 主要结果 | 判断 |
| --- | --- | --- | --- |
| CIFAR-style ResNet stem、冻结 BatchNorm running statistics | CIFAR 小图像常用网络结构和 BN 稳定化思路 | CIFAR stem：AUROC `0.5153 -> 0.4879`；冻结 BN：AUROC `0.5153 -> 0.4836`，known accuracy 明显下降 | 无效，停止作为主线 |
| 直接对 backbone feature 使用 SupCon | Khosla et al. 的 Supervised Contrastive Learning | AUROC `0.5669 -> 0.5011`，unknown rejection `6.27% -> 4.82%`，接受已知准确率下降 | 说明普通类内紧凑目标不能直接解决开放集重叠 |
| MC-Dropout 认知不确定性蒸馏 | Gal and Ghahramani 的 MC Dropout；与已有 uncertainty-head KD 对比 | MC-epistemic 加权 KD 的 AUROC、OSCR 和接受已知准确率变差 | 保留为消融，不作为默认方法；当前不确定性 KD 尚未证明优于标准 KD |
| mixed-pool nnPU 不确定性约束 | Kiryo et al. 的 nnPU 风险估计 | 单次试验 AUROC `0.5445 -> 0.5819`，但 calibrated unknown rejection 和 OSCR 没有稳定提升；干净三 seed 结果仍是混合 | 有排序信号，但没有解决工作点上的未知拒绝 |
| Gaussian class-conditional NLL、kNN/预测类局部距离等检测分数 | Lee et al. 的 Mahalanobis/OOD 建模和局部 support 思路 | NLL AUROC `0.4785 -> 0.4911`；若干 kNN 组合改善 AUROC 或候选纯度，但 FPR95、known coverage 与 unknown rejection 不稳定 | 仅改变分数，不能替代表征学习；保留为对照 |
| hybrid/local feature boundary、candidate-gated feature separation | ARPL/VOS 的类边界与异常特征动机；可靠候选后再施加表征约束 | seed 42、123、3407 的结果方向不完全一致；部分实验改善 AUROC/特征距离，但 unknown rejection 仍约 `4%–6%` | 有候选价值，暂不升为默认方法 |
| full-data 与训练轮数检查 | 排除数据量不足和欠训练造成的假象 | 使用完整已知训练集比只用 1200 张更好；但 5 -> 10 epoch 的完整测试中，KNN AUROC `0.5732 -> 0.5641`，unknown rejection `0.1140 -> 0.0585`，Mahalanobis unknown rejection `0.0903 -> 0.0588` | 数据规模重要，单纯增加 epoch 无效 |
| class-wise KNN support-boundary loss + warm-up/ramp | SCAN 的邻域一致性、局部 support 约束，以及半监督学习中先稳定表征再逐步使用伪标签的思想 | 三个 seed 的完整 10000 张测试均改善 AUROC/FPR95/OSCR；但 seed 45 的 unknown rejection 略降 | 目前最有希望的候选，但仍不是已解决方案 |

### 当前最有价值的候选结果

候选方法只相对 matched baseline 增加 `alpha_discovery_knn_boundary=0.1`、两轮 warm-up 和两轮线性 ramp，其余 teacher、student、数据划分、检测器、校准规则和测试集固定。完整测试结果如下：

| Seed | AUROC baseline -> candidate | FPR95 baseline -> candidate | Known acceptance baseline -> candidate | Unknown rejection baseline -> candidate |
| --- | --- | --- | --- | --- |
| 43 | `0.5652 -> 0.5988` | `0.8910 -> 0.8570` | `0.9212 -> 0.9458` | `0.0875 -> 0.0918` |
| 44 | `0.5584 -> 0.6016` | `0.8790 -> 0.8357` | `0.9483 -> 0.9510` | `0.0598 -> 0.0700` |
| 45 | `0.5928 -> 0.6036` | `0.8672 -> 0.8480` | `0.9255 -> 0.9327` | `0.1058 -> 0.1020` |
| mean | `0.5721 -> 0.6013` | `0.8791 -> 0.8469` | `0.9317 -> 0.9432` | `0.0844 -> 0.0879` |

这组结果的可取之处是：改进不只来自重新调阈值。全测试表征诊断中，最近已知训练样本距离和分类器原型距离的 AUROC/重叠度在多数 seed 上向正确方向变化，说明 support-boundary loss 确实影响了特征空间。它仍然不能被表述为“解决了未知检测”：AUROC 只有约 `0.60`，特征直方图重叠仍约 `0.83–0.86`，且未知拒绝率在 seed 45 没有提升。

### 今天的最终判断

1. **已经排除的方向**：只换 stem、冻结 BN、直接把 SupCon 加到 backbone、单纯增加 epoch，以及只替换 ODIN/Mahalanobis/NLL/kNN 分数，都不能稳定解决已知/未知重叠。
2. **有研究价值但不能夸大的方向**：nnPU、candidate-gated feature separation、纯未知 Energy、consensus/EMA 和局部 kNN 分数有局部信号，但跨 seed 或工作点指标不稳定；它们应继续作为消融或辅助机制。
3. **当前首选候选**：KNN support-boundary loss 的 warm-up/ramp 版本。下一步应在固定第三方 split 或更多 seed 上复核，并优先检查其对已知类误拒、类别条件 support 半径和未知类别分布的影响，而不是继续调阈值。
4. **项目状态**：代码框架、标准 KD、可选不确定性 KD、开放集检测、候选筛选和聚类流程均可运行；但“基于不确定性知识蒸馏的新类发现”仍是研究中的实验框架，不应把当前 KNN 边界候选误写成已经完成的最终算法。蒸馏和不确定性模块仍需在相同协议下做 CE / 标准 KD / uncertainty KD 的多 seed 消融，联合新类发现模块也还没有完成完整 UNO/SimGCD 式周期伪标签训练。

本节对应的详细记录：

- `analysis/knn_boundary_followup_s43_20260930.md`
- `analysis/raw_feature_supcon_pilot_s42_20260930.md`
- `analysis/mc_epistemic_kd_pilot_s42_20260930.md`
- `analysis/mixed_pool_nnpu_uncertainty_pilot_s42_20260930.md`
- `analysis/semantic_hard_gaussian_score_and_data_scale_20260930.md`
- `analysis/stem_and_bn_representation_pilots_s42_20260930.md`
- `analysis/candidate_feature_sep_recheck_s123_20260930.md`
- `analysis/candidate_feature_sep_recheck_s3407_20260930.md`

## 主要问题与处理方向

### 2026-10-01：KNN support-boundary 连续权重尝试

为处理 mixed discovery pool 中候选污染的问题，新增可选参数
`--discovery-knn-soft-weighting`。它参考 FixMatch 的置信度加权、Mean
Teacher 的稳定模型和 SCAN 的邻域一致性，只对已经通过候选门控的样本按风险、邻域一致性和 EMA/student 一致性连续加权；默认关闭，旧实验不受影响。

在 CIFAR-100 semantic-hard 60/40、seed=43、完整 10000 张测试和与现有
hard KNN warm-up/ramp 完全相同的协议下，soft 版本结果为：AUROC `0.5991`、
FPR95 `0.8582`、OSCR `0.3351`、known acceptance `0.9385`、unknown rejection
`0.0938`。hard 版本为 `0.5988 / 0.8570 / 0.3308 / 0.9458 / 0.0918`。

特征诊断显示，soft 版本的 classifier-prototype 距离 AUROC/重叠从
`0.5956/0.8555` 改善到 `0.6105/0.8187`，但最近已知训练样本距离从
`0.5990/0.8482` 变为 `0.5940/0.8490`，最终检测 AUROC 几乎不变。因此它只
改善了部分全局原型几何，未解决局部 support overlap；当前保留为可选消融，
不设为默认主方法。详细记录见
`analysis/knn_boundary_soft_weighting_pilot_s43_20261001.md`。

### 2026-10-01：所有已知类局部 support 的 kNN 检测对照

此前 `predicted_class_knn_distance` 只计算模型预测类别内的 kNN 距离；当未知样本被
高置信度错分到某个已知类时，可能漏掉它在另一个已知类 support 附近的情况。本轮新增
所有已知类别逐类计算 kNN 距离、再取最小值的检测信号：`normalized_entropy_min_class_knn`。
它只改变检测评分，不改变学生 checkpoint、数据划分、训练损失或阈值校准协议。

在 seed=43/44 的 CIFAR-100 semantic-hard 60/40、完整测试集、MC=4、95% known coverage
协议下，旧预测类 kNN 到新方法的结果如下：

| seed | AUROC | FPR95 | OSCR | known acceptance | unknown rejection | accepted-known accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 43 | 0.5991 -> **0.6042** | 0.8582 -> **0.8560** | 0.3351 -> **0.3405** | 0.9385 -> 0.9070 | 0.0938 -> **0.1468** | 0.4729 -> **0.4859** |
| 44 | 0.6016 -> **0.6061** | 0.8357 -> **0.8315** | 0.3475 -> **0.3528** | 0.9510 -> 0.9418 | 0.0700 -> **0.0878** | 0.4791 -> **0.4843** |

两个 seed 的排序指标和未知拒绝率方向一致改善，说明“只看预测类 support”确实可能造成漏检；
但提升幅度有限，seed=43 的工作点改善部分受到验证集阈值尺度变化影响，不能认为核心问题已经解决。
该方法目前作为可选评分器保留，并在 `--knn-ood` 时进入自动评分候选，不替换默认评分器。详细记录见
`analysis/min_class_knn_support_score_pilot_20261001.md`。后续重点应回到训练阶段的已知/未知表征分离
或专门开放集检测头，而不是继续堆叠检测后处理分数。

### 2026-10-01：联合新类发现权重稳定化尝试

针对 mixed discovery pool 中 novel prototype 权重快速衰减和自举错误的问题，本轮比较了
四种训练权重：原始 `joint_novel_mass`、增加 `--joint-novel-weight-floor 0.05`、使用
EMA 模型计算 novel mass，以及只使用已知分类 residual `1-max softmax(known logits)`。
floor 参数和日志 `joint_novel_weight_min` 已加入代码，但默认值仍为 `0`，不会改变历史配置。

实验固定为 CIFAR-100 semantic-hard 60/40、seed=43、预训练 ResNet-34/ResNet-18、1200/300/1200/1000
样本、2 epoch、mixed pool、相同 `unified_novel_mass` 检测和 95% known coverage 校准：

| 方法 | AUROC | FPR95 | OSCR | known acceptance | unknown rejection | accepted-known accuracy |
|---|---:|---:|---:|---:|---:|---:|
| novel mass, floor=0 | 0.5503 | 0.8976 | 0.1733 | 0.9480 | 0.0727 | 0.2521 |
| novel mass, floor=0.05 | 0.5258 | 0.8927 | 0.1729 | 0.9642 | 0.0156 | 0.2513 |
| EMA novel mass | 0.5439 | 0.9138 | 0.1658 | 0.9528 | 0.0857 | 0.2423 |
| known residual | **0.5700** | 0.9122 | 0.1642 | 0.9659 | 0.0416 | 0.2306 |

这组 2026-10-01 结果是早期探索记录，不应直接作为当前权重实现的结论：后续梯度图审计发现
当时的 residual 权重没有 detach；其中 residual 行及其结论属于旧梯度语义，不能代表修正后的
detached residual。novel-mass 的早期数值仍可作为当时配置下的历史结果，但不证明它适合与
其他损失组合。修正后的 residual 已在 2026-10-12 用两个 seed 重新配对复核，结论见本 README
后面的“Joint residual re-audit”。floor/EMA 旧结果不进入默认主方法。详细记录见
`analysis/joint_weight_stability_pilot_20261001.md`。下一步应停止叠加 novel 权重变体，转向显式的
known-vs-unknown 表征或拒绝头，并把未知检测和新类聚类分开验证。

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

## 2026-09-29：CIFAR-style ResNet stem 尝试

### 修改内容

当前代码新增了可选参数：

```powershell
--cifar-stem
```

启用后，ResNet 使用 `3×3、stride=1、padding=1` 的首层，并移除 ImageNet ResNet 默认的初始 max-pooling。对于 CIFAR-100 的 `32×32` 原始图像，原来的 ImageNet 风格首层和池化会较早降低空间分辨率；CIFAR 风格 stem 的目的，是尽量保留局部结构，从表征学习角度缓解已知类与未知类特征重叠。

该实现仍然支持预训练权重：启用预训练时，将原首层权重插值到 `3×3` 后初始化新首层。这个初始化方式是工程适配，不是某篇论文的完整复现；原有默认配置不变，因此历史实验仍可复现。

### 已完成的检查

- `python -m compileall -q novel_discovery train.py`：通过；
- toy 数据检查和 CIFAR stem 模型前向：通过；
- 项目测试：`96 passed`；
- CIFAR-100 60/40、seed=42、预训练 ResNet-34 教师、3 epochs、1200/300/1000 样本的 CIFAR-stem 教师训练已正常完成并保存 checkpoint，best validation accuracy 为 `0.1233`。

学生训练和最终 discover 对照尚未完成，因此目前只能确认代码可运行，不能声称 CIFAR stem 已改善未知检测。下一步应在相同教师/学生结构、相同 seed、数据预算、损失、评分器和 known-only 阈值校准下，与原 ImageNet stem 做配对比较，并同时报告 AUROC、FPR95、OSCR、known accuracy、known acceptance、unknown rejection，以及 known/unknown 分数分位数。

### 相关先行对照与当前判断

在继续修改结构前，已先复核了预训练 encoder 学习率方向：

- seed=42 的 `encoder_lr_scale=0.1` pilot，相对于原始 ImageNet stem baseline，AUROC 从 `0.6026` 到 `0.6194`，FPR95 从 `0.8683` 到 `0.8512`，固定测试集 95% known coverage 的事后 unknown rejection 从 `6.65%` 到 `9.03%`；但 known accuracy 从 `0.3987` 降到 `0.3678`，OSCR 也下降；
- seed=123 的 `encoder_lr_scale=0.1` 对照，AUROC 为 `0.6309`，baseline 为 `0.6479`；FPR95 为 `0.8375`，baseline 为 `0.8268`；固定 95% known coverage 的 unknown rejection 为 `10.85%`，baseline 为 `10.60%`；known accuracy 从 `0.4750` 降到 `0.4428`；
- 完全冻结 encoder 的 seed=123 对照进一步下降到 AUROC `0.5826`、FPR95 `0.8890`、unknown rejection `6.48%`、known accuracy `0.2768`。

因此，降低或冻结 encoder 学习率目前没有稳定、无代价地解决特征重叠，不能修改为默认方案。CIFAR stem 也必须经过学生训练和检测对照后再决定是否保留为有效方法；如果它只提高已知分类而不提高固定 known coverage 下的未知拒绝，就应作为结构消融，而不是项目主方法。

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

## Objectosphere 严格复核更新（2026-10-01）

- 已完成 seed=42 和 seed=123 的严格一因素对照：CIFAR-100 随机 60/40、完整数据、预训练 ResNet-34/18、5 epoch、纯未知 discovery pool、相同检测器和 95% known coverage 阈值；唯一变量是 `alpha_discovery_objectosphere=0` 或 `0.01`。
- seed=42：AUROC `0.6108 -> 0.6283`，FPR95 `0.8540 -> 0.8343`，OSCR `0.2845 -> 0.3306`，unknown rejection `7.48% -> 7.98%`。
- seed=123：AUROC `0.6065 -> 0.6245`，FPR95 `0.8598 -> 0.8268`，OSCR `0.2950 -> 0.3563`，unknown rejection `8.43% -> 9.18%`，接收后 known 分类准确率 `41.93% -> 50.35%`。
- 两个 seed 的 feature-norm histogram overlap 都下降（seed=42：`0.8202 -> 0.7853`；seed=123：`0.8148 -> 0.7714`），说明该损失确实改变了表示分布；但未知拒绝率仍然很低，不能说已经解决 known/unknown overlap。
- 当前结论：Objectosphere 有可复现的小幅正向信号，保留为可选表示消融，不设为默认方法，也不继续只调它的权重。

### Mixed pool 路径更新

- 训练代码现在支持 `--discovery-pool-mode mixed` 与 Objectosphere 联用，但必须同时开启 `--discovery-feature-candidate-gating`。
- known 项只使用有标签 known batch；unknown 项只使用两视图共同选出的高风险候选，避免把 mixed pool 中的已知样本整体当作 unknown。
- 未开启 candidate gating 时仍会主动报错，防止污染训练目标。
- toy smoke 已验证该路径能运行、候选比例和 `discovery_objectosphere` 损失均非零；这只是代码路径验证，不代表性能提升。
- 尚未完成 CIFAR-100 mixed-pool 严格对照。下一步应固定 seed、teacher、backbone、epoch、检测器和阈值，只比较 mixed baseline 与 candidate-gated Objectosphere。

### Mixed-pool pilot 结果（2026-10-01）

- 已按预先定义的协议完成 CIFAR-100 mixed-pool 对照：seed=123、相同 teacher/split/backbone、1200 train、300 val、1000 test、3 epoch、MC=4、95% known coverage；唯一训练差异是 candidate-gated Objectosphere，权重 `0.01`。
- mixed baseline -> Objectosphere：AUROC `0.5726 -> 0.5357`，FPR95 `0.8693 -> 0.9313`，OSCR `0.2360 -> 0.2038`，unknown rejection `9.43% -> 4.22%`，accepted-known accuracy `34.09% -> 34.05%`。
- feature-norm histogram overlap 虽然从 `0.8281` 降到 `0.7722`，但开放集指标全面变差，说明“范数分布变开”不等于“检测边界有效”。
- 结论：Objectosphere 的纯未知实验有小幅正向信号，但 candidate-gated mixed-pool 版本当前无效；不继续调该方向权重。代码保留为受保护的消融选项，后续优先研究更可靠的候选筛选或直接建模 mixed pool 的 PU/rejector 方法。

### Mixed candidate / feature recheck（2026-10-01）

- 新增可选 `--discovery-cross-view-gating`：mixed pool 中要求两个增强视图都独立选中候选，才施加 feature/Objectosphere 约束；默认关闭。
- 新增可选候选模式 `prototype_distance` 和 `distance_consensus`：使用候选到最近已知分类原型的余弦距离，后者再融合 entropy、max-softmax risk 和 uncertainty 的秩分数；默认仍为 `entropy_uncertainty`。
- 在相同 CIFAR-100 mixed pilot（seed=123、1200 train、300 val、1000 test、3 epoch、20% known pool、同一 teacher/检测器/95% known coverage）中：baseline 的 AUROC/FPR95/OSCR/unknown rejection 为 `0.5726/0.8693/0.2360/9.43%`；entropy + Objectosphere 为 `0.5357/0.9313/0.2038/4.22%`；entropy + cross-view 为 `0.5579/0.9229/0.2267/5.46%`；distance-consensus + cross-view + Objectosphere 为 `0.5528/0.9045/0.2285/7.20%`；distance-consensus + cross-view + feature separation 为 `0.5702/0.8710/0.2311/2.23%`。
- 结论：cross-view 和 prototype-distance 能减少候选噪声造成的损害，但没有超过 mixed baseline；Objectosphere 和局部 feature repulsion 不应作为 mixed 默认方法。完整记录见 `analysis/mixed_candidate_representation_recheck_20261001.md`。
- 有一组 feature-margin 实验的损失始终为 `0`，即没有真正参与训练，已明确标为无效证据；不能据此宣称该方法有效或无效。
- 下一步不再继续调这些候选排斥损失，先做 full-data matched protocol，判断低预算是否掩盖了表征学习效果；若 full-data 仍失败，再优先研究直接 mixed-PU/rejector 目标或更强的预训练/度量表征。

### Full-data mixed nnPU 复核（2026-10-01）

- 已完成完整训练数据的严格对照：CIFAR-100 随机 60/40、seed=123、同一 teacher、预训练 ResNet-18、5 epoch、mixed pool 实际 known fraction `0.212598`、同一检测器和 95% known coverage；唯一训练变量是 `alpha_discovery_uncertainty_pu: 0 -> 0.1`。
- baseline -> nnPU：AUROC `0.6077 -> 0.6533`，AUPR `0.4695 -> 0.5322`，FPR95 `0.8497 -> 0.8167`，OSCR `0.3137 -> 0.3728`，unknown rejection `7.88% -> 12.23%`，accepted-known accuracy `44.47% -> 50.61%`，known acceptance `94.22% -> 94.90%`。
- 这是目前 mixed pool 完整训练预算下最有希望的结果，且比 1200 样本 pilot 更好；但目前只有一个 seed，不能作为最终论文结论。
- 分布诊断并非全部改善：主 score overlap `0.8303 -> 0.7796`、entropy `0.7873 -> 0.7618`、epistemic `0.8282 -> 0.7969`，但 feature-norm overlap `0.7809 -> 0.8093` 变差，Mahalanobis overlap 仍为 `0.9165`。准确表述应是 nnPU 改善了 uncertainty/entropy 与综合 score 的排序和工作点，不是已经完全解决 embedding overlap。
- 当前决策：暂不与 Objectosphere、candidate-only feature repulsion 叠加；先用 matching teacher 重复 seed=42 的 full-data baseline/nnPU 对照。若方向重复，再做三 seed 确认并单独评估聚类。

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

## 次日复核：prototype repulsion 是否稳定（2026-09-29）

昨天的待办是用额外随机种子复核 `alpha_proto_repulsion=0.01`。今天在 seed=42、123 上补跑 baseline / repulsion 配对训练与检测，并纳入昨天已有的 seed=3407 同协议小规模结果。所有配对均使用 CIFAR-100 60/40、Standard KD、相同教师、相同 1200/300/1000 数据预算、3 epochs 和 `normalized_entropy_mahalanobis`。

| seed | 方法 | AUROC | FPR95 | known accept | unknown reject | known accuracy |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | baseline | 0.4698 | 0.9408 | 0.9457 | 0.0485 | 0.1464 |
| 42 | repulsion 0.01 | 0.4690 | 0.9474 | 0.9572 | 0.0281 | 0.1612 |
| 123 | baseline | 0.5210 | 0.9224 | 0.9439 | 0.0457 | 0.1766 |
| 123 | repulsion 0.01 | 0.5438 | 0.8944 | 0.9538 | 0.0330 | 0.2228 |
| 3407 | baseline | 0.5335 | 0.9289 | 0.9074 | 0.0608 | 0.2264 |
| 3407 | repulsion 0.01 | 0.5536 | 0.9008 | 0.9455 | 0.0582 | 0.2529 |

三组平均 AUROC 从 `0.5081` 到 `0.5221`，FPR95 从 `0.9307` 到 `0.9142`，有一定排序改善；但未知拒绝率在三个 seed 中都下降，平均从 `0.0516` 降至 `0.0398`。seed=42 的 AUROC 基本不变且 FPR95 变差，正向排序结果主要来自 seed=123/3407。因此它不是稳定解决核心问题的方法，不能默认启用或宣称已有效；只保留为消融候选。不同 seed 实际 known coverage 有差异，工作点均值只能作方向性参考。

这轮结果进一步说明：原型分离可能改善已知类几何结构或检测排序，但没有带来更高未知拒绝率；核心重叠仍在。完整记录见 `analysis/proto_repulsion_recheck_20260929.md`。下一步应使用完整训练数据和 stratified split 做严格多 seed 配对，报告 matched-known-coverage 下的未知拒绝率，并检查最近已知原型距离分布；若重叠仍明显，就停止搜索排斥权重，转向更强的预训练/度量表征或规范的 mixed-pool GCD 目标。

### 完整数据、相同分层协议的复核

为避免历史 baseline 的 random split 与 stratified split 不一致，补训了完整 CIFAR-100 训练集上的 Standard KD baseline，并与 `alpha_proto_repulsion=0.01` 模型进行同协议评估。两者使用相同 seed=42、教师权重、15 epoch、完整 10,000 张测试集、`normalized_entropy_mahalanobis`，并在 known coverage 95% 处比较：

| 方法 | AUROC | AUPR | FPR95 | 测试 known accept | unknown reject | known accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard KD baseline | **0.5309** | **0.4119** | **0.8987** | 0.9537 | **0.0433** | 0.4017 |
| Prototype repulsion 0.01 | 0.5167 | 0.3999 | 0.9047 | 0.9493 | 0.0380 | **0.4105** |

完整数据结果没有复现小样本实验中的排序改善：虽然 repulsion 的 known accuracy 略高，但 AUROC、AUPR、FPR95 和 unknown reject 均变差。分数中位数仍高度重叠：baseline known/unknown 为 `0.059/0.165`，repulsion 为 `0.024/0.097`；known 分数 90 分位与 unknown 分数 90 分位仍接近。这说明当前排斥项没有形成有效的已知/未知检测间隔，应保留为可选消融，不再继续搜索权重，也不作为默认方案。

完整分析见 `analysis/full_proto_repulsion_stratified_s42_20260929.md`。下一步应在固定 checkpoint 上检查最近类原型距离和逐类误差，定位哪些未知类别落入已知类簇；随后比较更强预训练表示或规范的 GCD mixed-pool 统一已知/新类目标。新的方法需使用同 split、matched known coverage，并至少三 seed 验证，避免再以单个小样本 seed 作结论。

## 方法忠实度复核：weighted Sinkhorn 与 novel-mass（2026-09-29）

重新检查训练调用链后，发现旧报告的 weighted-Sinkhorn 配对实验没有独立控制 Sinkhorn 伪标签边际：novel-mass 权重此前同时进入 Sinkhorn assignment 和 loss reduction。因此旧的“加权 vs 等权 Sinkhorn”结论撤回，只保留旧 checkpoint 数字作为探索性记录。代码现在用 `--joint-weighted-sinkhorn` / `--no-joint-weighted-sinkhorn` 将这两种作用拆开。

新增 `scripts/diagnose_novel_mass.py` 直接评估 novel-mass 分数。CIFAR-100 60/40、seed=42、小样本 3 epoch 配对实验中，等权/加权 Sinkhorn 的 novel-mass AUROC 分别为 `0.5664/0.5749`，FPR95 为 `0.8711/0.8165`；但 test-known 标签只用于事后覆盖率诊断时，两组未知拒绝率均约 `0.0608`。项目原 Mahalanobis 检测器下，加权组 AUROC 和未知拒绝率反而下降。该单 seed 结果指标冲突，尚未证明加权 Sinkhorn 能解决低拒绝率问题。

诊断脚本复用项目统一的阈值校准、FPR95 和 OSCR 实现；测试集标签只用于事后指标。测试集 known coverage 重校准是不可部署的诊断结果，不能用于模型选择。完整审计、协议和指标见 `analysis/method_fidelity_and_sinkhorn_audit_20260929.md`。下一步先用其他 seeds 复核真正单因素设置，再决定保留还是停止该方向。

## 本轮审计与 Objectosphere 表征试验（2026-10-01）

本轮先修正了 mixed discovery pool 的 nnPU 先验协议。由于 `limit-discovery` 会在 known/novel 拼接池上再次抽样，请求的 `mixed-known-pool-ratio` 不一定等于训练时实际看到的 known 比例。新增 `known_proportion(dataset)`，并支持 `--discovery-uncertainty-known-prior auto`，将抽样后的实际比例写入 `config.json` 后传入 nnPU；显式数字仍保留给真实部署场景。toy smoke 中请求比例为 `0.2`，实际抽样比例为 `0.01`；CIFAR-100 小规模协议的实际比例为 `0.210833`。

固定 CIFAR-100 60/40、seed=42、同一 teacher/student、1200/300/1000、1200 mixed discovery、3 epoch 和 `normalized_entropy_mahalanobis` 后，手动先验 `0.2` 到自动先验 `0.210833` 的变化为：AUROC `0.5337 -> 0.5492`，FPR95 `0.9391 -> 0.9112`，但 unknown reject `0.0536 -> 0.0434`，OSCR 略降。因此这是实验协议修正和有限排序信号，不是核心问题的解决方案。

随后新增可选的 `--alpha-discovery-objectosphere`。参考 Objectosphere 思路，纯未知 discovery 特征被拉向低范数，已知特征保持最小范数；同时新增 `feature_norm` 与 `normalized_entropy_feature_norm` 检测分数。该损失禁止用于 mixed pool，默认权重仍为 `0`。CIFAR-100 单 seed、3 epoch、`alpha=0.01` 的小规模结果显示，主 `normalized_entropy_mahalanobis` 的 AUROC `0.4901 -> 0.5870`、FPR95 `0.9589 -> 0.8865`，但 unknown reject `0.0510 -> 0.0281`；新范数分数有局部改善但工作点指标不一致。因此 Objectosphere 目前只保留为纯未知、可选表征消融，不能默认启用或宣称已解决已知/未知重叠。

完整协议、命令、结果和下一步要求见 `analysis/protocol_and_objectosphere_pilot_20261001.md`。后续必须使用 full-data、matched known coverage 和至少 3 个 seed 复核，同时报告分数分布重叠和 unknown reject；不能只依据单 seed 的 AUROC/FPR95 调大损失权重。
## 2026-10-01 strict Objectosphere recheck

本轮先检查了实验配置，而不是直接相信已有数字。第一次 full-data Objectosphere 运行遗漏了 baseline 的 `alpha_feat_kd=0.1` 和 `alpha_proto=0.1`，不能作为单因素对比；第二次 treatment 又与旧 baseline 的 discovery-pool 设置不同。这两次结果均保留为审计记录，但不作为方法证据。

随后重新训练了严格配对的 control/treatment：CIFAR-100 random 60/40、seed=42、完整 known 训练集、预训练 ResNet-34/ResNet-18、5 epochs、同一教师和同一 pure-unknown discovery pool，只改变 `alpha_discovery_objectosphere=0` 与 `0.01`。完整结果见 `analysis/objectosphere_full_strict_recheck_20261001.md`。

| 指标 | control | Objectosphere | 变化 |
| --- | ---: | ---: | ---: |
| AUROC | 0.6108 | 0.6283 | +0.0175 |
| FPR95 | 0.8540 | 0.8343 | -0.0197 |
| OSCR | 0.2845 | 0.3306 | +0.0461 |
| known acceptance | 95.12% | 94.65% | -0.47 pp |
| unknown rejection | 7.475% | 7.975% | +0.50 pp |
| accepted-known accuracy | 40.00% | 45.89% | +5.89 pp |

修正后的特征诊断显示 score histogram overlap 从 `0.8202` 降至 `0.7940`，feature-norm overlap 从 `0.8202` 降至 `0.7853`。因此本轮有小幅一致的正向信号，但 unknown rejection 仍然很低，而且只有一个 seed、pure-unknown discovery pool；Objectosphere 目前只能作为可选消融，不能作为已经解决核心问题的主方法。

本轮还修复了一个实验记录问题：检测过程计算了 `feature_norm`，但 `discovery_detail.json` 之前没有正确保存，导致旧的范数诊断被零值污染。现在两条 `run_discovery` 路径都会保存真实范数；toy smoke 已验证非零输出，测试为 `130 passed`。

下一步应优先在第二个固定 seed 和 mixed discovery pool 上复核 Objectosphere。如果 pure-unknown 的信号无法迁移到 mixed 场景，就停止继续调它的权重，转向可靠的 mixed-pool known/novel 统一空间训练目标。不得把本轮 pure-unknown 结果直接写成真实开放环境结论。
## 2026-10-01 full mixed-pool nnPU recheck (seed 42)

本轮严格复核 mixed discovery pool 中的 nnPU 不确定性约束。实验目的不是继续调阈值，而是验证前一轮 seed=123 的正向结果能否在独立 seed 上复现。两组实验使用相同的 CIFAR-100 random 60/40 划分、相同的 ResNet-34 teacher、预训练 ResNet-18 student、完整训练集、5 epochs、mixed pool、`normalized_entropy_mahalanobis` 检测分数、MC=4，以及只用 known validation 校准到 95% known coverage 的阈值；唯一训练变量是 `alpha_discovery_uncertainty_pu` 从 `0` 改为 `0.1`。完整记录见 `analysis/full_mixed_nnpu_recheck_s42_20261001.md`。

| 指标 | mixed baseline | mixed nnPU | 变化 |
| --- | ---: | ---: | ---: |
| AUROC | 0.5866 | 0.6498 | +0.0633 |
| AUPR | 0.4486 | 0.5189 | +0.0703 |
| FPR95 | 0.8667 | 0.8283 | -0.0383 |
| OSCR | 0.2921 | 0.3735 | +0.0814 |
| known acceptance | 94.50% | 95.45% | +0.95 pp |
| unknown rejection | 6.33% | 9.45% | +3.13 pp |
| accepted-known accuracy | 42.01% | 50.50% | +8.49 pp |

这次 seed=42 与 seed=123 的完整 mixed-pool 结果方向一致：AUROC、FPR95、OSCR、未知拒绝率和接受后已知准确率均改善，而且已知接受率没有下降。treatment 日志中 `discovery_uncertainty_pu` 为非零，说明该损失确实参与了训练；两组检测都使用相同的验证集阈值校准规则，因此不能把结果解释为单纯移动测试阈值的收益。

但目前仍不能宣称 nnPU 已经解决核心问题。当前只有两个 seed，未知拒绝率绝对值仍低，且该结果主要说明不确定性/检测排序改善，并不等价于已知和未知 embedding 已完全分离。下一步应使用第三个固定 seed 和匹配 teacher 做同样的 control/treatment 配对，报告三 seed 均值和标准差；若趋势保持，再恢复聚类评估，单独报告聚类 ACC/NMI/ARI、候选池纯度和估计类别数误差。下一轮仍不要把 nnPU 与 Objectosphere、feature repulsion、selective energy 或 candidate gating 叠加，以保持单因素归因。
## 2026-10-01：mixed-pool nnPU 三 seed 复核与组合方向

在 seed=42、123、3407 上完成了相同协议的 full-data mixed-pool baseline / nnPU 配对实验。每个 seed 使用匹配的 ResNet-34 teacher、预训练 ResNet-18 student、CIFAR-100 random 60/40 split、完整训练集、5 epochs、MC=4、`normalized_entropy_mahalanobis`，并只用 known validation 在 95% known coverage 处校准阈值；聚类暂时跳过。唯一训练变量是 `alpha_discovery_uncertainty_pu: 0 -> 0.1`，实际 mixed known prior 由程序自动测量并记录。完整复盘见 `analysis/full_mixed_nnpu_three_seed_summary_20261001.md`。

| 指标 | baseline 均值 +/- 标准差 | nnPU 均值 +/- 标准差 | 平均变化 |
| --- | ---: | ---: | ---: |
| AUROC | 0.5977 +/- 0.0106 | 0.6491 +/- 0.0047 | +0.0514 |
| FPR95 | 0.8632 +/- 0.0121 | 0.8272 +/- 0.0100 | -0.0360 |
| OSCR | 0.3052 +/- 0.0115 | 0.3718 +/- 0.0023 | +0.0666 |
| known acceptance | 94.61% +/- 0.46 pp | 95.24% +/- 0.30 pp | +0.63 pp |
| unknown rejection | 7.11% +/- 0.78 pp | 10.44% +/- 1.55 pp | +3.33 pp |
| accepted-known accuracy | 43.63% +/- 1.41 pp | 50.51% +/- 0.09 pp | +6.88 pp |

三个 seed 的所有主要指标变化方向一致，说明 nnPU 已经是当前最有价值的 mixed-pool 主候选，而不是单个 seed 的偶然结果。它仍没有解决核心重叠：平均未知拒绝率只有 10.44%，之前的表征诊断也显示 feature / Mahalanobis 分布仍有较大重叠。因此准确结论是“不确定性排序和工作点改善”，不是“已知未知 embedding 已完全分离”。

复盘后，下一种值得尝试的组合是 `nnPU + class-wise KNN support-boundary warm-up/ramp`：nnPU 作用于不确定性 head，KNN support-boundary 作用于已知类局部特征支持，两者作用位置互补。组合实验不会加入 Objectosphere、feature repulsion、selective energy 或 EMA candidate gating，因为这些方向在 mixed 场景中结果不稳定或容易受到候选污染影响。组合只与 nnPU-only 做单因素增量比较；若组合无效，就保留 nnPU 单项并停止继续堆叠损失；若有效，再做三 seed 确认。
## 2026-10-01：nnPU 与 KNN 组合复盘、min-class kNN 检测改进

在 nnPU 三 seed 复核后，先尝试了 `nnPU + class-wise KNN support-boundary warm-up/ramp`。该组合在 seed=42 中 AUROC `0.6498 -> 0.6349`、FPR95 `0.8283 -> 0.8340`、OSCR `0.3735 -> 0.3476`，未知拒绝率只从 `9.45%` 增至 `10.33%`，但接受后已知准确率从 `50.50%` 降到 `47.83%`。因此这是误伤已知样本的失败组合，不再继续叠加或调大 KNN 训练损失；完整记录见 `analysis/nnpu_knn_combo_recheck_s42_20261001.md`。

随后保持三个 nnPU student checkpoint 完全不变，只把检测分数从 `normalized_entropy_mahalanobis` 换成 `normalized_entropy_min_class_knn`：对每个样本计算到所有 60 个已知类局部支持的最小 kNN 距离，而不是只计算预测类别距离。三 seed 结果如下：

| 指标 | nnPU + Mahalanobis 均值 | nnPU + min-class kNN 均值 | 平均变化 |
| --- | ---: | ---: | ---: |
| AUROC | 0.6491 | 0.6903 | +0.0413 |
| FPR95 | 0.8272 | 0.7681 | -0.0591 |
| OSCR | 0.3718 | 0.4103 | +0.0385 |
| known acceptance | 95.24% | 95.24% | -0.01 pp |
| unknown rejection | 10.44% | 13.03% | +2.58 pp |
| accepted-known accuracy | 50.51% | 51.53% | +1.02 pp |

三个 seed 的方向全部一致，且已知接受率没有下降，说明该改进不是简单通过误拒已知样本换来的。它支持“未知样本可能被错误分类到某个已知类，预测类局部距离会漏检”的诊断。完整记录见 `analysis/full_mixed_nnpu_minclass_knn_recheck_20261001.md`。

程序现在把 raw min-class kNN 加入 `--score-mode auto` 候选，并增加了单元测试；但当前 auto 选择逻辑会利用 open validation 的已知/未知标签按 AUROC 选分数，因此只能作为分析工具，不是无标签部署规则。本次 auto 实际选中了 `full`，没有自动选到 min-class kNN。当前可复现实验应显式使用 `--score-mode normalized_entropy_min_class_knn --knn-ood`，后续再单独设计不使用未知标签的自动校准策略。
## 2026-10-01：聚类空间复核与当前推荐流程

在三个 seed 上固定 nnPU student、显式 min-class kNN 检测、oracle `K=40`、KMeans 和 normalization，只比较 `projection_pca` 与 `feature_pca`。feature_pca 的候选聚类结果在三个 seed 上全部更好：

| 指标 | projection_pca 均值 | feature_pca 均值 | 变化 |
| --- | ---: | ---: | ---: |
| 候选聚类 ACC | 0.1819 | 0.2006 | +0.0187 |
| 候选聚类 NMI | 0.3698 | 0.3975 | +0.0278 |
| 候选聚类 ARI | 0.0307 | 0.0521 | +0.0214 |
| 候选池纯度 | 0.6470 | 0.6470 | 0 |

候选池纯度不变而聚类指标提升，说明改进来自聚类表示本身。当前推荐实验流程更新为：训练使用 nnPU，检测显式使用 `--score-mode normalized_entropy_min_class_knn --knn-ood`，聚类显式使用 `--cluster-feature feature_pca`；`projection_pca` 保留为消融。完整记录见 `analysis/full_mixed_nnpu_cluster_feature_space_recheck_20261001.md`。

这仍不是最终解决方案：即使 oracle K=40，平均 NMI 只有 0.3975、ARI 只有 0.0521；auto-K 在同一 seed 上因 silhouette / composite 分别选择 K=48 / K=29，说明类别数估计不稳定。下一步应转向真正的 known+novel GCD 训练目标、周期性伪标签更新和类别均衡分配，而不是继续堆叠阈值、KMeans 或后处理分数。

## 2026-10-01：joint memory-bank 跨 batch 一致性尝试

为解决当前 joint discovery 只使用批内邻域、批间伪标签不稳定的问题，新增了可选的 FIFO memory-bank 邻域一致性损失。它保存历史 batch 的 detached feature 和 novel logits，当前样本只与全局近邻的停梯度预测对齐；mixed pool 中还会按 residual novel weight 过滤样本。该机制默认关闭，不影响历史流程。相关参数为 `--alpha-joint-memory-neighbor`、`--joint-memory-size`、`--joint-memory-k`、`--joint-memory-temperature` 和 `--joint-memory-warmup-size`。

在 CIFAR-100 random 60/40、seed=42、1200/300/1200 小规模配对实验中，固定 nnPU、`joint_mixed_residual`、预训练 ResNet、oracle `K=40`、min-class kNN 检测和 `feature_pca` 聚类，只改变 memory-bank 方案：

| 方法 | AUROC | FPR95 | OSCR | unknown reject | candidate purity | candidate NMI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| joint residual baseline | 0.59118 | 0.89091 | 0.15992 | 7.09% | 49.12% | 0.86994 |
| memory, alpha=0.1 | 0.58799 | 0.88760 | 0.15454 | 6.08% | 44.44% | 0.86889 |
| memory, alpha=0.05 + warm-up | 0.59261 | 0.88595 | 0.16142 | 6.84% | 47.37% | 0.84345 |

结果说明 memory-bank 已正确参与训练，但没有改善核心的已知/未知重叠；warm-up 版本只带来可以忽略的 AUROC/FPR95 变化，同时降低候选纯度和 NMI。因此该方向保留为默认关闭的消融选项，不再继续调大权重或继续堆叠。完整记录见 `analysis/joint_memory_bank_pilot_20261001.md`。

当前应回到真正的全局表征学习：参考 GCD/UNO/SimGCD 的 momentum teacher、周期性伪标签刷新、全局 memory queue 和类别均衡分配，但必须先实现与当前 nnPU 主流程的单因素对比。下一步不能把 stale mixed-pool 预测直接当成可靠伪标签，也不能继续只靠阈值、KMeans 或后处理分数解决核心问题。

## 2026-10-01 Angular 复盘与协议审计

当前复盘保留的主流程是 `nnPU + normalized_entropy_min_class_knn + feature_pca`。
nnPU 和 min-class kNN 已在三个 seed 上表现出一致的正向变化；直接 Angular
`alpha=0.05` 只完成了小规模单 seed 验证，暂时属于待复核候选，不应直接写成最终创新结论。

本轮新增了可选的 `--angular-correct-only`，只对当前已正确分类的已知样本施加 Angular
margin。CIFAR-100 random 60/40、seed=42 的对照中，该方法 AUROC 为 `0.5872`，
FPR95 为 `0.8959`，未知拒绝率为 `5.06%`，候选纯度为 `0.4444`；直接 Angular
对照分别为 `0.6084`、`0.8694`、`8.61%`、`0.6071`。因此该选项只保留为消融，默认关闭。

新增的 `--angular-warmup-epochs`、`--angular-ramp-epochs` 以及 uncertainty 修正分数
也已经完成小规模对照，但没有稳定改善未知检测，不能替代主流程。训练日志现在会记录
`angular_active_ratio`，用于确认 Angular 实际作用于多少样本，避免把未生效或样本覆盖率
不同的实验误判为算法改进。

seed=123 的 Angular 复核必须使用与 seed=42 匹配的 teacher：相同 backbone、训练轮数、
训练数据规模和数据协议。当前本地已有的 seed=123 teacher 是另一套 3 epoch/1200 样本
配置，不能直接用于严格对照；在匹配 teacher 生成前，不报告 seed=123 的 Angular 结论。
完整记录见 `analysis/angular_correct_only_pilot_20261001.md` 和
`analysis/angular_schedule_and_uncertainty_score_20261001.md`。

### seed=123 严格复核结果

已使用与 seed=42 相同协议重新训练匹配的 ResNet-34 teacher，并在同一
`nnPU + normalized_entropy_min_class_knn + feature_pca` 流程下比较
`alpha_angular=0` 与 `alpha_angular=0.05`。两组只改变 Angular loss 权重，
其余训练、检测和聚类条件完全一致。

| 指标 | nnPU 基线 | nnPU + Angular | 变化 |
| --- | ---: | ---: | ---: |
| AUROC | 0.5672 | 0.5470 | -0.0202 |
| AUPR | 0.4607 | 0.4379 | -0.0228 |
| FPR95 | 0.8945 | 0.8928 | -0.0017 |
| OSCR | 0.1523 | 0.1420 | -0.0103 |
| known acceptance | 94.64% | 92.46% | -2.18 pp |
| unknown rejection | 7.94% | 8.93% | +0.99 pp |
| accepted-known accuracy | 22.48% | 22.28% | -0.20 pp |
| candidate purity | 0.5000 | 0.4444 | -0.0556 |
| candidate NMI | 0.8494 | 0.8184 | -0.0310 |

这次复核说明 Angular 的拒绝率小幅增加来自更激进的拒绝，并没有形成更好的
已知/未知排序或更纯的未知候选池。因此 Angular 不进入主流程，只保留为消融。
当前更需要验证的是训练充分性：快速对比使用了 1200 个训练样本和 2 个 student
epoch，学生特征可能尚未稳定。下一步应固定 nnPU 算法，增加训练样本或 epoch，
保持检测器、阈值校准和聚类设置不变，再判断特征重叠是否主要来自欠训练。
完整记录见 `analysis/angular_strict_recheck_s123_20261001.md`。

## 2026-10-01：冻结 teacher 候选筛选器复核

为减少 mixed discovery pool 中“高熵但其实是已知类”的候选污染，新增了
`--discovery-selection-model teacher`。该模式只用冻结的 teacher 选择
candidate-gated feature separation 的候选，不参与 student 反向传播，也不会被
错误更新为 EMA；原有 `student` 和 `ema` 模式保持不变。

在 CIFAR-100 random 60/40、seed=123、完整 known training partition、5 个
student epoch、同一 teacher、同一 mixed pool、同一 nnPU 和 feature-separation
权重下，只比较候选筛选器来源。检测固定为
`normalized_entropy_min_class_knn + feature_pca`：

| 指标 | student selector | frozen teacher selector | teacher - student |
| --- | ---: | ---: | ---: |
| AUROC | 0.6899 | 0.6696 | -0.0203 |
| FPR95 | 0.7806 | 0.7722 | -0.0084 |
| OSCR | 0.4160 | 0.4226 | +0.0066 |
| known acceptance | 92.46% | 94.97% | +2.51 pp |
| unknown rejection | 19.85% | 16.38% | -3.47 pp |
| accepted-known accuracy | 53.44% | 54.32% | +0.88 pp |
| candidate purity | 0.6400 | 0.6875 | +0.0475 |
| candidate unknown NMI | 0.6941 | 0.7691 | +0.0750 |

teacher selector 确实提高了候选池纯度和聚类质量，但 validation 阈值直接迁移到
有限测试集时，student selector 的 known acceptance 只有 92.46%，因此不能直接把
它的 19.85% unknown rejection 与 teacher 的 16.38% 比较。仅作为事后诊断，将两组
测试 known score 分别校准到约 95% known acceptance 后，student selector 的未知
拒绝率为 12.16%，teacher selector 为 16.38%，接受后已知准确率分别为 52.73% 和
54.32%。该诊断使用测试 known 标签，不可部署，也不能用于选择模型，但说明 teacher
selector 在公平工作点上有进一步研究价值。

逐新类分析中，40 个未知类只有 9 个误接受率下降，16 个上升，15 个不变，平均
误接受率变化为 `+0.0357`，所以它仍没有让所有未知类整体远离已知特征空间。
该选项目前保留为有希望的候选纯度/工作点消融，不设为默认方案；需要在另一个
seed、独立 known calibration set 和独立 open-validation set 上复核后，才考虑进入
主流程。完整记录见
`analysis/candidate_selector_teacher_vs_student_classwise_s123_20261001.json`
和 `analysis/candidate_selector_teacher_vs_student_s123_20261001.md`。

### seed=42 复核与两 seed 结论

随后用完全相同的 full-data 协议在 seed=42 上复核。teacher selector 的
AUROC 从 `0.6709` 略升到 `0.6753`，FPR95 从 `0.8132` 降到 `0.8000`，但候选
纯度从 `0.7093` 降到 `0.6706`，raw unknown rejection 从 `15.44%` 降到
`14.43%`。在测试 known 标签仅用于事后诊断的 matched 95% known coverage
工作点上，student / teacher 的 unknown rejection 分别为 `16.71%` / `14.68%`。
这与 seed=123 的 teacher 优势不一致，说明 selector 来源的收益具有 seed 敏感性。

两个 seed 的 matched 工作点平均结果为：teacher selector 的 unknown rejection
`15.53%`，student selector `14.44%`；候选纯度为 `0.6790` / `0.6747`，候选
NMI 为 `0.7851` / `0.7389`。teacher 主要在候选池聚类质量上显示出较稳定的局部
价值，但 AUROC 和未知拒绝率并未稳定占优。因此：

- `teacher` 保留为稳定目标候选筛选和聚类质量消融，不设为默认主流程；
- 当前主流程仍使用 `student` selector，以保持与三 seed nnPU 主比较的一致性；
- 不再继续优先调 selector 来源或阈值，下一步应回到更强的表征学习目标，或严格
  隔离的 mixed-pool GCD 目标，直接处理已知/未知特征重叠。

完整 seed=42 记录见
`analysis/candidate_selector_teacher_vs_student_s42_20261001.md`，两 seed 综合见
`analysis/candidate_selector_teacher_vs_student_two_seed_20261001.md`。

## 2026-10-01：固定 checkpoint 的 OpenMax 检测复核

为确认问题是否只是检测器选择，保持
`candidate_sep_student_s42_full/student.pt`、完整 known train bank、random
60/40 split、验证集阈值协议和测试集完全不变，只把当前主分数
`normalized_entropy_min_class_knn` 换成已有的 OpenMax-Weibull 分数。OpenMax
参考 Bendale and Boult 的 OpenMax 思路，但当前实现是基于已知类特征距离尾部的
轻量 Weibull 评分，不是原论文完整的 top-k activation 重校准。

| 指标 | 主分数 | OpenMax-Weibull | 变化 |
| --- | ---: | ---: | ---: |
| AUROC | 0.6709 | 0.5668 | -0.1040 |
| AUPR | 0.5639 | 0.4557 | -0.1081 |
| FPR95 | 0.8132 | 0.9273 | +0.1140 |
| OSCR | 0.3702 | 0.2823 | -0.0879 |
| raw unknown rejection | 15.44% | 4.81% | -10.63 pp |

在测试 known 标签仅用于事后诊断的 matched 95% known coverage 工作点上，主分数
和 OpenMax 的 unknown rejection 分别为 `16.71%` 和 `8.86%`。因此 OpenMax
没有改善当前表征，说明继续替换 post-hoc 检测分数不能解决已知/未知重叠。OpenMax
保留为文献基线，不再作为下一轮主改进方向；后续代码修改应直接作用于表征学习或
严格隔离的 mixed-pool GCD 训练目标。

完整记录见 `analysis/openmax_fixed_checkpoint_recheck_s42_20261001.md`。

## 2026-10-01 今日尝试总结与后续方向

今天的所有对比都围绕同一个核心问题：已知与未知样本在特征和开放集分数上
重叠严重，导致未知拒绝率低。实验过程中优先检查协议一致性，raw unknown
rejection 不再单独作为结论，必要时同时报告 AUROC、FPR95、OSCR、known
acceptance、accepted-known accuracy，以及 matched known coverage 的事后诊断。

### 1. 冻结 teacher 候选筛选器

思路来自 Mean Teacher / teacher-guided pseudo-labeling：在 mixed discovery pool
中，用冻结 teacher 替代不断变化的 student 选择高风险候选，降低候选污染，再把
候选交给 feature-separation 训练。该改动已落实为
`--discovery-selection-model teacher`，并确认 teacher 不参与 student 反向传播、
也不会被错误更新为 EMA。

在 seed=123 和 seed=42 的 full-data 配对实验中，teacher selector 没有稳定改善
AUROC 或未知拒绝率。两个 seed 的 matched 工作点平均 unknown rejection 为
teacher `15.53%`、student `14.44%`，候选纯度为 `0.6790` / `0.6747`，候选 NMI
为 `0.7851` / `0.7389`。seed=123 teacher 的工作点拒绝率更好，但 seed=42
反而更差；逐类分析也显示它只改善部分未知类。

判断：该方法对候选池质量和聚类有局部价值，可以保留为稳定目标消融；不能把它
当作已知/未知特征已经分离的证据，也不设为默认主流程。下一步不再优先搜索
student/EMA/teacher selector 或继续移动阈值。

### 2. 固定 checkpoint 的 OpenMax-Weibull 检测器

思路来自 Bendale and Boult 的 OpenMax：用已知类特征到类中心的距离拟合 Weibull
尾部，作为 post-hoc 未知评分。此次严格固定同一个 student checkpoint、完整
known train bank、random 60/40 split、验证集阈值协议和测试集，只替换检测分数，
避免把训练变化误认为 OpenMax 效果。

结果为主分数 `normalized_entropy_min_class_knn` 对比 OpenMax：AUROC
`0.6709 -> 0.5668`，FPR95 `0.8132 -> 0.9273`，OSCR `0.3702 -> 0.2823`，
raw unknown rejection `15.44% -> 4.81%`；matched 95% known coverage 的
unknown rejection 为 `16.71% -> 8.86%`。

判断：当前表征不适合仅靠 EVT/Weibull 尾部建模，OpenMax 没有缓解核心重叠。
保留为文献检测基线，不再继续调 OpenMax 参数，也不纳入默认流程。

### 3. 今天的代码与质量检查

- 新增并复核 frozen teacher candidate selector 及其单元测试；
- 新增 seed=42 selector 复核、两 seed 汇总和 OpenMax 固定 checkpoint 分析文档；
- `137 passed`；`compileall` 通过；`git diff --check` 通过；
- 中途发现一次实验命令把 `limit-train=1200` 和错误 split 路径带入 OpenMax 对照，
  已将该结果作废并按 checkpoint 原始协议重跑，README 只记录修正后的结果。

### 后续推荐

1. 保持当前主流程 `nnPU + normalized_entropy_min_class_knn + feature_pca`，不要
   再叠加新的 post-hoc 分数或阈值规则。
2. 下一轮优先实现一个与当前主流程单因素隔离的 mixed-pool GCD 表征目标：明确区分
   known anchor、novel prototype 空间和未知候选权重，避免已知和新类再次被迫放入同一
   个错误的平衡 Softmax；伪标签应周期性刷新，并记录候选纯度、类别占用和跨 batch
   稳定性。
3. 对新目标先做 toy smoke 和小规模 CIFAR 配对实验，再用至少两个固定 seed、相同
   teacher、相同数据预算和 matched known coverage 判断是否有效；若没有同时改善
   排序和工作点指标，就停止该方向，不继续堆叠损失。
4. teacher selector 只作为候选池/聚类消融保留；OpenMax 只作为文献检测基线保留。

详细记录：
`analysis/candidate_selector_teacher_vs_student_s42_20261001.md`、
`analysis/candidate_selector_teacher_vs_student_two_seed_20261001.md`、
`analysis/openmax_fixed_checkpoint_recheck_s42_20261001.md`。

## 2026-10-01：轻量整体审计与下一步重点

在继续增加算法之前，对当前程序、实验协议和已有结果做了一轮轻量审计。审计原则是：
“代码里有开关”不等于“方法已经有效”，单个指标变化也不能代替 matched known
coverage 和多 seed 对照。

### 当前可以确认的内容

1. 三 seed full-data 实验支持 `nnPU` 是目前最可靠的 mixed-pool 改进：AUROC、FPR95、
   OSCR、known acceptance 和 accepted-known accuracy 均同方向改善。但平均 unknown
   rejection 仍只有 `10.44%`，所以它改善的是不确定性排序和工作点，不是完整的已知/未知
   特征分离。
2. `normalized_entropy_min_class_knn` 和 `feature_pca` 分别改善了检测排序和聚类表示，
   但二者都不能单独解决特征重叠。
3. frozen teacher selector 对候选聚类有局部价值，但两 seed 的 AUROC 和未知拒绝率不
   稳定，继续作为消融，不进入默认主流程。
4. OpenMax、Angular、memory bank、各种 candidate-only boundary loss 没有形成稳定的
   matched 工作点收益，暂不继续叠加。

### 当前最核心的四个问题

1. **已知表征和分类能力仍不足。** 当前最强 full-data nnPU 结果中，accepted-known
   accuracy 约为 `50%`。这说明很多被接受为已知的样本仍然分类错误；在已知支持没有建好
   之前，未知检测也很难可靠。
2. **mixed pool 的未知伪标签仍然会污染。** 现有 joint 代码有 novel-only、unified、
   residual、novel-mass、candidate gating 和 memory-bank 等选项，但还不是完整的
   UNO/SimGCD 训练协议；novel 权重和 Sinkhorn 分配仍可能依赖同一个正在变化的学生模型。
3. **原型和伪标签缺少稳定的全局更新。** 当前 KMeans 初始化主要是一次性的，批内邻域和
   evolving head 可能造成原型漂移、类别空置和伪标签不稳定。
4. **数据协议需要分层验证。** random 60/40 是当前主协议，semantic-hard 和
   semantic-isolated 只能作为不同协议分别报告，不能混合比较或用单一 split 宣称泛化。

### 重新确定的下一步重点

下一步只优先做“表征学习和 mixed-pool GCD 目标”，暂时停止继续寻找新的 post-hoc
检测分数。目标是形成如下可审计的单因素实验：

- known head 只用已知标签训练；
- novel prototype head 与 known head 解耦，不把所有样本强行放进同一个平衡 Softmax；
- 用 EMA 或冻结 teacher 产生伪标签目标；
- 周期性刷新 novel prototypes / 伪标签；
- 同时记录 prototype occupancy、伪标签跨 epoch 稳定性、候选纯度、accepted-known
  accuracy、AUROC、FPR95、OSCR 和 matched unknown rejection。

本轮新增的 `--joint-prototype-refresh-epochs` 是这个方向的最小实现，默认关闭。toy
smoke 已确认它会按计划执行，但目前没有 CIFAR 性能证据，不能宣称有效。

### 下一轮实验判定标准

先在相同 CIFAR split、teacher、训练数据量、epoch、检测器和阈值协议下，对比“一次性
KMeans 初始化”和“周期性 prototype refresh”。只有同时改善至少一个排序指标和一个
matched 工作点指标，且 accepted-known accuracy 没有明显下降，才保留该方向；如果只
改变 loss 或候选数量，就判定为无效。若失败，则停止继续堆叠 prototype/memory 变体，
转向完整、文献对齐的 UNO/SimGCD 式 mixed-pool GCD 目标。

详细审计见 `analysis/lightweight_project_audit_20261001.md`。

## 2026-10-05：实验协议审计与 matched split 复核

### 发现的协议问题

之前部分 `audit_current_*` 实验不能作为算法效果证据。student 使用的是
`splits_cifar100_60_40.json`，但 teacher checkpoint 来自
`splits_cifar100_protocols/cifar100_60_40_random.json`。两份 split 的 known 类别不相同，
而 KD 按 logit 下标对齐，因此 teacher 的第 `i` 个 logit 与 student 的第 `i` 个 logit
可能对应不同的真实类别。这不是随机波动，而是实验协议错误。相关历史结果保留用于诊断，
但不再用于证明某个算法有效。

同学的 `quick_lky` 结果也不能直接与当前结果比较，原因包括：训练/验证/测试样本数量不同、
`discovery_pool_mode` 分别为 `unknown` 和 `mixed`、MC 次数不同，以及纯未知池属于受控上限协议。
后续比较必须同时固定 split 文件、teacher checkpoint、数据规模、训练轮数、discovery pool、
MC 次数、评分器和阈值策略。

### 代码修复

- teacher 和 student checkpoint 现在保存 `known_classes` 与 `novel_classes`。
- 加载 checkpoint 时会校验当前 split；known 类别不匹配会直接报错，避免错误语义的 logits
  静默进入 KD。
- 旧 checkpoint 如果只有 `known_classes` 仍可兼容加载，但新实验必须使用包含完整类别元数据的
  checkpoint。
- `tests/test_core_behaviors.py` 新增 split 不匹配检查；当前核心测试和编译检查已通过。

### matched split 实验

本轮重新用当前主 split `splits_cifar100_60_40.json` 训练同一个 teacher，再固定所有条件，
只改变 `alpha_discovery_uncertainty_pu`：baseline 为 `0`，nnPU 为 `0.1`。两组均为 seed 42、
完整 CIFAR-100 数据、pretrained backbone、5 个 student epochs、mixed discovery pool、
known pool ratio `0.2`、同一 teacher、同一 `normalized_entropy_mahalanobis` 评分器、MC=8、
known-only validation 95 分位阈值。

| 方法 | AUROC | FPR95 | OSCR | known acceptance | unknown rejection | accepted-known accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| matched baseline | 0.5453 | 0.8963 | 0.2478 | 94.32% | 6.18% | 38.59% |
| matched nnPU | 0.5809 | 0.8908 | 0.3399 | 95.32% | 5.90% | 50.57% |

这组结果是有效的单因素对照。nnPU 对未知/已知排序、OSCR 和已知分类有明显帮助，但
unknown rejection 从 `6.18%` 降为 `5.90%`，因此不能声称已经解决核心问题。当前核心问题仍是
已知与未知特征/分数重叠；高拒绝率如果伴随大量 known 误拒绝，也不能算真正改进。

### 下一步验证

下一步固定上述两个 checkpoint、数据协议、阈值和 MC 次数，只把检测器换成
`normalized_entropy_min_class_knn --knn-ood`。这个实验回答：当前问题主要来自训练表征，还是
Mahalanobis/entropy 评分器没有利用好已有表征。若两个 checkpoint 的 kNN 结果都同步提升，优先
改检测器；若 nnPU 仍不能超过 baseline，优先改表征学习和 mixed-pool GCD 目标；若只有拒绝率提升
而 accepted-known accuracy 明显下降，则判定为误拒绝，不保留为主方案。

本次 kNN 对照结果如下。除评分器外，其他条件与上一表完全相同：

| 方法 | AUROC | FPR95 | OSCR | known acceptance | unknown rejection | accepted-known accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| matched baseline + min-class kNN | 0.5925 | 0.8600 | 0.2908 | 95.20% | 7.53% | 39.81% |
| matched nnPU + min-class kNN | 0.6183 | 0.8430 | 0.3821 | 94.57% | 9.05% | 51.97% |

这说明 min-class kNN 比当前 Mahalanobis/entropy 评分器更能利用已知类特征库，nnPU 与该评分器
结合后是目前最有希望的工作点。相对同一 checkpoint 的 Mahalanobis/entropy，nnPU+kNN 的
AUROC 从 `0.5809` 提升到 `0.6183`，FPR95 从 `0.8908` 降到 `0.8430`，unknown rejection
从 `5.90%` 提升到 `9.05%`。但是约九成未知样本仍被接受，说明特征空间重叠依然严重；这组
结果应表述为“有效改善但未解决核心问题”。后续不能只继续调 percentile，应优先研究让未知
样本远离 known prototypes / class support 的表征目标，并继续报告 matched known coverage、
accepted-known accuracy 和多 seed 方差。

## 2026-10-05：joint discovery 与候选特征分离复核

本轮继续围绕同一个核心问题做严格单因素复核：已知与未知样本在特征和开放集分数上重叠，
导致未知拒绝率低。实验固定 CIFAR-100 random 60/40 split、teacher/student 协议、训练
数据预算、检测器、阈值校准和聚类设置；只改变 joint discovery 或候选特征分离机制。结果
不能与早期 split 不匹配的历史实验混合使用。

### 1. 无门控 `joint_mixed_residual`

该尝试的动机是让混合 discovery pool 中的样本都参与 novel head，并用 residual novel
weight 减弱低风险样本的影响。结果为：AUROC `0.5630`、FPR95 `0.8880`、OSCR `0.2257`、
unknown rejection `5.60%`。相对于当前严格基线 `nnPU + projection kNN` 的 AUROC
`0.6198`、FPR95 `0.8407`、OSCR `0.3836`、unknown rejection `9.10%`，无门控 joint
明显退化。

历史结论（旧梯度语义）：当时的实验显示该配置退化，但后续方法保真审计发现
`known_residual_weights` 实际保留了梯度，与“停梯度样本权重”的设计不符。因此这组旧实验
只能说明旧实现/旧配置表现不佳，不能证明修正后的 detached residual 无效。修正后方法已经在
2026-10-12 做了 seed 43/44 配对复核；应以本 README 后面的“Joint residual re-audit”作为
当前证据。该方法仍不进入默认主流程，因为修正后的多指标结果依然是混合而非稳定胜出。

### 2. 完整数据候选门控 joint discovery

该尝试只允许候选门控后的一部分高风险样本参与联合 novel discovery，目的是减少 known
污染。结果为：AUROC `0.6320`、FPR95 `0.8198`、OSCR `0.3735`、unknown rejection
`7.50%`、accepted-known accuracy `49.85%`。它对排序指标有局部改善，但 OSCR、未知
拒绝率和已知分类工作点仍不优于严格基线，因此不能称为核心问题已解决。

结论：候选门控可能改善候选池组成，但“高风险”并不等于“可靠未知”。它适合作为候选
纯度消融，不能直接作为 mixed-pool GCD 的完整训练目标。后续若重用，应配合冻结/EMA
教师、跨 batch 稳定性和独立 open-validation，而不是把一次筛选结果当成真伪标签。

### 3. 候选门控 feature separation

本尝试只对 mixed pool 中最高风险约 25% 的样本施加 feature separation loss，分别检查
projection 空间和实际用于 kNN 的 backbone feature 空间。结果如下：

| 方法 | AUROC | FPR95 | OSCR | unknown rejection |
| --- | ---: | ---: | ---: | ---: |
| projection kNN 基线 | 0.6198 | 0.8407 | 0.3836 | 9.10% |
| candidate separation + projection kNN | 0.6208 | 0.8543 | 0.3870 | 8.33% |
| candidate separation + feature kNN | 0.6185 | 0.8668 | 0.3846 | 8.00% |
| features kNN 基线 | 0.6183 | 0.8430 | 未单独记录 | 9.05% |

projection 结果的 AUROC/OSCR 只有极小变化，FPR95 和 unknown rejection 反而变差；
features 空间也没有改善。说明当前 separation loss 没有把未知样本整体推离 known
support，可能只是改变了少量候选样本或分数尺度，不能作为有效解决方案。

### 4. 本轮代码审计工具

训练过程新增了已知验证集表征几何诊断：`feature_within_mean`、`feature_within_q95`、
`nearest_center_distance` 和 `center_margin`。这些量只使用 known validation labels，
用于观察类内紧凑度与类中心间隔，不使用测试未知标签，也不替代开放集指标。新增可选参数
`--save-epoch-checkpoints`，可保存 `teacher_epoch_N.pt` 和 `student_epoch_N.pt`，用于
核对“最终 checkpoint 表现差”究竟来自训练不足、表征退化还是检测器问题；默认关闭，
不改变原有训练流程和模型选择。

### 当前判断与后续方向

本轮没有找到可以进入默认主流程的新方法。当前应保留的主线仍是：

```text
mixed nnPU uncertainty
+ normalized_entropy_min_class_knn
+ projection kNN
+ known-only validation threshold
```

下一步应停止继续堆叠阈值、OpenMax、残差权重和候选后处理，转向真正隔离的 mixed-pool
GCD 表征目标：known classification head 只由已知标签监督，novel prototype 空间与
known head 解耦；由 EMA/冻结 teacher 产生伪标签；周期性刷新 prototype 和伪标签；
记录 prototype occupancy、伪标签跨 epoch 稳定性、候选纯度和 matched known coverage。
先做 toy smoke 和小规模 CIFAR 单因素配对，只有排序指标与 matched 工作点至少各有一项
改善且 accepted-known accuracy 不明显下降，才进入多 seed 完整实验。

本轮三组结果均应作为“已验证但未采用”的负结果保存，不能从小幅 AUROC 变化推断未知
特征已经分离。详细实验日志见对应的 `analysis/` 记录；后续新增实验必须同时注明 split、
teacher checkpoint、数据预算、epoch、评分器、阈值协议和唯一改变因素。

### 代码验证

- `pytest -q`：`155 passed`；
- `python -m compileall -q train.py novel_discovery`：通过；
- `git diff --check`：通过；
- toy teacher/student smoke：通过，确认几何指标会输出，且 `--save-epoch-checkpoints`
  会保存每轮 checkpoint；该 smoke 只验证可运行性，不作为性能结论。

## 2026-10-05：EMA target 与 KMeans 原型同步复核

本轮发现并修复了一个会影响实验有效性的实现问题：启用
`--joint-prototype-init kmeans` 和 `--joint-pseudo-ema-target` 时，KMeans
只更新在线 `novel_head`，原 EMA novel head 仍可能保留随机原型。现在每次
初始或周期性 KMeans refresh 后，都会同步 EMA novel head；之后的参数更新仍
使用无梯度 EMA。修复后 `pytest -q` 为 `155 passed`，toy KMeans + EMA smoke
和 CUDA CIFAR smoke 均通过。

在 CIFAR-100 random 60/40、seed=42、ResNet-34 teacher / ResNet-18 student、
1200/300/1000/1200、3 epochs、mixed pool、K=40、同一检测和聚类协议下，
只比较 `joint-prototype-init=random` 与修复后的 `kmeans`：

| 方法 | AUROC | FPR95 | OSCR | known acceptance | unknown rejection | candidate purity | auto-K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random + EMA soft target | 0.5071 | 0.9260 | 0.0682 | 94.57% | 5.10% | 0.3774 | 20 |
| KMeans + EMA soft target | 0.5247 | 0.9260 | 0.0621 | 95.72% | 5.10% | 0.4348 | 29 |

结论是：KMeans 初始化能改善候选纯度、AUROC 和自动类别数估计，但没有提高
固定 95% known coverage 下的 unknown rejection，OSCR 还下降。因此它只能作为
novel 聚类初始化的可选消融，不能作为解决 known/unknown 分数重叠的主方法；不再
继续盲目增加 KMeans refresh 或调阈值。完整记录见
`analysis/ema_kmeans_sync_recheck_s42_20261005.md`。

## 2026-10-05：prototype refresh 复核

本轮先修复了 prototype refresh 的工程问题：`train.py` 的 KMeans 拟合现在使用
`threadpool_limits(limits=1)`，并支持显式 `n_init`，避免 Windows/CPU 线程过度争用导致
训练进程无输出或占用异常。修复后 `pytest -q` 为 `150 passed`，toy 函数级和 CIFAR
smoke 均能完成，日志确认 `initial` 与周期性 refresh 事件真实执行。

随后在 CIFAR-100 random 60/40、seed=42、同一 teacher、3 epoch、1200/300/1000 数据预算、
K=40 和同一 discovery 评估协议下，对比 joint random prototype、全 mixed pool KMeans
refresh、以及高风险候选子集 KMeans refresh。结果如下：

| 方法 | AUROC | FPR95 | OSCR | known acceptance | unknown rejection | candidate purity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| joint random baseline | 0.5051 | 0.9359 | 0.0648 | 97.04% | 5.61% | 55.00% |
| full-pool KMeans refresh | 0.5013 | 0.9326 | 0.0463 | 93.59% | 6.89% | 40.91% |
| candidate-only KMeans refresh | 0.4993 | 0.9326 | 0.0516 | 92.43% | 9.44% | 44.58% |

结论是：候选子集能比全池 refresh 减少一部分污染，但两种 refresh 都没有稳定改善
AUROC/OSCR，且 known acceptance 明显下降；因此暂不把它设为默认方法，也不把未知拒绝率
单独上升解释为核心问题解决。完整日志见
`analysis/prototype_refresh_recheck_s42_20261005.md`。

随后复核了 joint candidate gating。固定同一 teacher、split、seed、3 epoch、数据预算、
K=40 和检测协议，只比较 student hard gate、student soft gate、以及 frozen teacher hard
gate：

| 方法 | AUROC | FPR95 | OSCR | known acceptance | unknown rejection | candidate purity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| joint random baseline | 0.5051 | 0.9359 | 0.0648 | 97.04% | 5.61% | 55.00% |
| student hard gate | 0.4882 | 0.9326 | 0.0477 | 93.42% | 3.83% | 27.27% |
| student soft gate | 0.5031 | 0.9490 | 0.0758 | 93.91% | 6.38% | 40.32% |
| frozen teacher hard gate | 0.5046 | 0.9276 | 0.0547 | 96.55% | 4.34% | 44.74% |

student hard gate 会造成 joint 信号稀疏；soft gate 虽然 OSCR 有小幅改善，但 FPR95 和
known acceptance 变差；teacher gate 能提高候选纯度，却没有改善 AUROC 或 unknown rejection。
因此 gating 仍不能解决核心的 known/unknown 表征重叠问题。完整记录见
`analysis/joint_candidate_gate_recheck_s42_20261005.md`。结合前面的 prototype refresh 和
residual weighting 结果，下一步应停止继续堆叠 KMeans、阈值和候选门控，转向重新学习
可分的表示：known head 只接受已知监督，novel 分支使用 EMA/冻结 teacher、全局 memory
bank 和稳定伪标签更新，并记录候选纯度、prototype occupancy 和跨 epoch 一致性。

本轮还确认了两个必须遵守的实验协议：teacher backbone 必须与 checkpoint 匹配；novel head
的 `num_novel` 必须与训练 checkpoint 的 K 一致。违反这两点的命令被程序拒绝，不能作为实验
结果。mixed-pool soft residual weighting 的早期版本曾呈现误拒绝倾向，但其权重梯度语义后来被
发现不符合设计；该旧结论已由修正后审计限定，不能泛化到 detached residual。当前 detached
版本仍不进入默认流程，原因是两 seed 的排序、固定工作点与候选聚类指标没有一致改善。后续应
停止只调阈值和候选门控，继续改进 known/novel 解耦表征目标。

补充的 residual weighting 对照属于旧梯度实现。它按已知分类器的 residual weight 对 joint loss
软加权，训练日志中的平均权重从 `0.886` 降到 `0.797`，说明实现确实生效；但 AUROC 为
`0.4952`、FPR95 为 `0.9375`、OSCR 为 `0.0629`，相对 joint-random baseline 的
`0.5051/0.9359/0.0648` 没有改善，unknown rejection 的上升伴随 known acceptance 从
`97.04%` 降到 `94.24%`，属于误拒绝倾向。该方法不进入默认流程。详细记录见
`analysis/prototype_refresh_recheck_s42_20261005.md`。后续审计发现 residual weight 未 detach；
因此该结果是历史实现结果，不是当前 detached residual 的有效性证据。修正后配对实验见
`analysis/joint_residual_fidelity_recheck_20261012.md`。

补充的 assignment 诊断显示，单纯增大 discovery batch 也不能解决 novel head 未形成
稳定分配的问题：batch 64 对照的 AUROC/FPR95/OSCR/unknown rejection 为
`0.5051/0.9359/0.0648/5.61%`，batch 512 为 `0.4973/0.9161/0.0631/4.59%`，joint
loss 仍约为 `3.685`，接近 `log(40)=3.6889`。toy K=4 smoke 中 assignment entropy
为 `1.3772`（均匀上限 `1.3863`）、max probability 为 `0.2896`（均匀值 `0.25`），
active prototypes 约为 `2.75/4`，说明分配仍接近均匀且存在 prototype 使用不足。后续
应实现跨 batch 的全局 assignment/memory bank，并在实验中报告 occupancy、entropy 和
跨 epoch 稳定性，而不是继续调整 batch 或阈值。详见
`analysis/joint_assignment_diagnostic_s42_20261005.md`。
### 2026-10-05：全局 assignment 与 soft pseudo-label 复核

本轮针对 joint novel head 的两个可检验原因做了单因素实验：批内 Sinkhorn 样本太少，可能导致原型分配近似均匀；同时将 Sinkhorn 概率直接 `argmax`，可能把早期随机差异放大成 prototype collapse。所有实验固定 CIFAR-100 random 60/40、seed=42、同一 ResNet-34 teacher、ResNet-18 student、1200/300/1000/1200 数据预算、3 epoch、batch=64、mixed pool、K=40 和同一检测/聚类协议。

| 方法 | AUROC | FPR95 | OSCR | known acceptance | unknown rejection | 候选纯度 | 末轮活跃原型数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| joint random baseline | 0.5051 | 0.9359 | 0.0648 | 97.04% | 5.61% | 55.00% | 约 5 |
| global assignment，memory weight=1.0 | 0.5001 | 0.9293 | 0.0658 | 93.91% | 7.14% | 43.08% | 约 5 |
| global assignment，memory weight=0.1 | 0.5120 | 0.9359 | 0.0681 | 96.22% | 5.61% | 48.89% | 约 17 |
| hard pseudo，balance=0.1 | 0.4971 | 0.9293 | 0.0683 | 88.65% | 10.46% | 37.27% | 约 9 |
| hard pseudo，balance=1.0 | 0.4958 | 0.9095 | 0.0659 | 93.09% | 8.16% | 43.24% | 约 8 |
| soft pseudo，balance=1.0 | 0.5014 | 0.9457 | 0.0632 | 91.94% | 6.12% | 32.88% | 约 25 |

global assignment 确实改变了跨 batch 伪标签，但 memory weight=1.0 时历史队列压制当前 batch，出现原型集中；weight=0.1 能减轻集中，却没有提升 unknown rejection 或候选纯度。hard pseudo-label 会让分配变得更尖锐，但主要是误拒绝已知样本换来的；增大 balance 权重也无法恢复 40 个原型。soft pseudo-label 在训练诊断上保留了更多活跃原型，说明 `argmax` 确实是塌缩原因之一，但检测和候选纯度仍变差，表明“分配更均匀”不等于“学到了真实未知语义”。

因此这些 joint assignment/pseudo-label 选项全部保持默认关闭，不再继续在低预算 mixed pool 上调标量权重。它们只作为诊断性消融保留。完整协议和原始结果见 `analysis/joint_global_assignment_and_soft_pseudo_s42_20261005.md`。当前主候选仍是 full-data mixed-pool nnPU + min-class kNN；joint discovery 后续若继续，应加入 EMA/frozen target、prototype occupancy、跨 epoch assignment stability，并明确把 known support 与 novel partition 解耦，不能只靠 memory bank 传播受污染预测。

### 2026-10-05：nnPU warm-up/ramp 复核

为检查混合池 nnPU 是否因过早施加而放大初期错误排序，新增了默认关闭的
`--discovery-uncertainty-pu-warmup-epochs` 和
`--discovery-uncertainty-pu-ramp-epochs`。本次只改变 nnPU 的启用时序：立即启用
的 `alpha=0.1` 对比第 1 个 epoch 关闭、第 2 个 epoch 使用 0.05、第 3 个
epoch 起使用 0.1；其它训练、数据 split、teacher、检测器和阈值保持一致。

完整 CIFAR-100 seed=42 对照结果如下：

| 方法 | AUROC | FPR95 | OSCR | known acceptance | unknown rejection | accepted-known accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| immediate nnPU | 0.7009 | 0.7608 | 0.4134 | 95.83% | 12.78% | 51.32% |
| nnPU warm-up/ramp | 0.6970 | 0.7452 | 0.3987 | 94.12% | 14.40% | 50.15% |

该策略确实提高了 FPR95 和当前工作点的未知拒绝率，但 AUROC、OSCR、已知接受率
和接受后已知准确率下降，因此不能称为全面改进，也不能据此宣称表征重叠已解决。
它保留为可选消融，当前主流程仍是立即 mixed-pool nnPU + 显式 min-class kNN。
详细记录见 `analysis/nnpu_warmup_ramp_recheck_s42_20261005.md`。

## 2026-10-05：已知类几何紧凑性复核

针对已知/未知特征重叠，本轮只在可靠的已知标签上测试了三种辅助表征目标：批内 Center Loss、类内半径 hinge，以及样本到自身类中心和最近错误类中心的相对 center-margin。三组实验均固定 CIFAR-100 random 60/40、seed=42、匹配 teacher、完整数据、5 epochs、mixed-pool immediate nnPU、min-class kNN 检测和 95% known coverage 阈值。

| 方法 | AUROC | FPR95 | OSCR | known acceptance | unknown rejection | accepted-known accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| immediate nnPU baseline | 0.7009 | 0.7608 | 0.4134 | 95.83% | 12.78% | 51.32% |
| nnPU + center loss, alpha=0.05 | 0.6962 | 0.7757 | 0.4014 | 94.78% | 13.40% | 50.11% |
| nnPU + radius hinge, alpha=0.05 | 0.6875 | 0.7777 | 0.3902 | 94.72% | 14.73% | 49.39% |
| nnPU + center-margin, alpha=0.05 | 0.6941 | 0.7500 | 0.4187 | 94.00% | 14.98% | 52.84% |

结论：Center Loss 和 radius hinge 没有改善开放集排序，未知拒绝率的增加伴随已知误拒绝，因此默认关闭，仅保留消融。center-margin 改善 FPR95、OSCR 和接受后已知准确率，但 AUROC 与 known acceptance 下降，属于有局部价值但尚未稳定有效的方向，不能作为主流程。完整记录见 `analysis/known_geometry_compactness_recheck_s42_20261005.md`。

代码新增可选参数 `--alpha-center`、`--alpha-radius`、`--known-feature-radius`、`--alpha-center-margin` 和 `--center-margin`，默认均不启用；相关损失和梯度测试已加入，当前测试为 `133 passed`。下一步只做低权重 center-margin 复核；若仍是“工作点改善、整体排序下降”，就停止堆叠已知类几何损失，回到 mixed-pool 的解耦表征学习。

## 2026-10-05：不确定性加权 discovery feature-margin 三 seed 复核

针对核心的已知/未知表征重叠，本轮新增可选损失：对 mixed discovery pool 两个增强视图，按学生不确定性（detach 后作为软权重）惩罚样本与最近已知类分类器原型过于相似；没有给 discovery 样本分配硬未知标签。训练损失权重为 `--alpha-discovery-uncertainty-feature-margin 0.05`，相似度间隔为 `--discovery-uncertainty-feature-margin 0.2`，默认权重为 0。该设计是本项目的实验假设，借鉴不确定性软加权和 margin 表征学习的一般思路，不是对某篇论文算法的直接复现。

在 CIFAR-100 random 60/40、完整数据、seed 42/123、匹配预训练 ResNet-34 teacher / ResNet-18 student、5 epochs、mixed pool、`nnPU=0.1` 的配对实验中，只增加上述损失。baseline 与 treatment 均重新使用完全相同的 `normalized_entropy_min_class_knn`、feature kNN `k=10`、MC=8 和 known-validation 95% coverage 阈值评估，聚类关闭以隔离检测结果：

| Seed | 方法 | AUROC | FPR95 | OSCR | known acceptance | unknown rejection | accepted-known accuracy |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | nnPU baseline | 0.7012 | 0.7590 | 0.4110 | 95.55% | 13.40% | 51.19% |
| 42 | + uncertainty feature-margin | 0.7108 | 0.7348 | 0.4132 | 95.30% | 13.90% | 50.87% |
| 123 | nnPU baseline | 0.6830 | 0.7693 | 0.4071 | 94.48% | 14.08% | 51.91% |
| 123 | + uncertainty feature-margin | 0.7091 | 0.7282 | 0.4449 | 94.62% | 14.88% | 55.33% |
| 3407 | nnPU baseline | 0.6868 | 0.7685 | 0.4115 | 95.42% | 12.70% | 51.56% |
| 3407 | + uncertainty feature-margin | 0.6969 | 0.7595 | 0.4066 | 94.95% | 13.35% | 50.80% |

三 seed 的平均 AUROC 为 0.6903→0.7056，FPR95 为 0.7656→0.7408，OSCR 为 0.4099→0.4216，未知拒绝率为 13.39%→14.04%；AUROC、FPR95、未知拒绝率三个方向在三 seed 一致，但 OSCR 在 seed 3407 回退，且 known acceptance 平均略降（95.15%→94.96%）。全量特征诊断（27,000 张已知训练图像、10,000 张开放测试图像）中，三种距离的排序 AUROC 在每个 seed 都改善，九个直方图 overlap 都下降，支持该损失确实改变了表征几何，而不只是移动阈值；但 overlap 仍约为 0.68–0.77，核心问题并未解决。当前是值得深入验证的候选，而不是已证明最终有效的方法：仍默认关闭。随后已完成预先限定的低权重 `alpha=0.025` 对照，固定间隔 0.2 与其它全部设置，结果见下文。

低权重 `alpha=0.025` 的 seed-42 复核现已完成：AUROC/FPR95/OSCR/unknown rejection 为 `0.6934/0.7275/0.3978/12.28%`，相较 nnPU baseline 的 `0.7012/0.7590/0.4110/13.40%`，只有 FPR95 改善；known acceptance 和 accepted-known accuracy 也下降。三种全量距离几何 AUROC 均轻微下降，原型与质心 overlap 变差。因此降低权重没有保留 alpha=0.05 的几何/排序收益，停止盲目扫权重；`alpha=0.025` 记为负向消融，`alpha=0.05` 仍是有三 seed 正向排序和几何证据、但任务效用有 seed 间波动的候选。下一步优先审计不确定性软权重本身的分布及其未知纯度/已知污染，再决定是否改软门控或渐进启用。

实验完整记录、seed3407结果、低权重负向消融及协议陷阱说明见 `analysis/uncertainty_feature_margin_recheck_20261005.md`。注意：重叠诊断脚本默认是小样本预算，正式比较必须显式传 `--limit-train 0 --limit-val 0 --limit-test 0 --limit-discovery 0`；本轮误用默认值的输出只作为 pilot，不纳入全量结论。有效训练必须显式传 `--discovery-pool --teacher-backbone resnet34`；训练前被程序拒绝的命令未计入结果。当前完整测试为 `158 passed`（2 条环境/测试写法 warning）。

## 2026-10-05：不确定性软门控替代方案复核

上一轮审计发现，原 uncertainty head 在 mixed discovery pool 上的未知排序 AUROC（seed 42/123/3407 为 0.6444/0.6623/0.5876）弱于同一模型的 `1-MSP`（0.6822/0.7046/0.6891），且 uncertainty 权重 ESS 很高，接近近似均匀加权。本轮新增可选参数 `--discovery-feature-margin-weight-source`，比较原 uncertainty、`1-MSP` 以及二者乘积门控；默认仍为 `uncertainty`，不影响旧实验。

固定 CIFAR-100 seed=42、完整数据、同一 teacher、5 epochs、mixed pool、nnPU=0.1、feature-margin 权重 0.05/间隔 0.2、同一 kNN 检测器和 95% known coverage 阈值，仅改变门控来源：

| 方法 | AUROC | FPR95 | OSCR | known acceptance | unknown rejection | accepted-known accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nnPU baseline | 0.7012 | 0.7590 | 0.4110 | 95.55% | 13.40% | 51.19% |
| uncertainty gate | 0.7108 | 0.7348 | 0.4132 | 95.30% | 13.90% | 50.87% |
| `1-MSP` gate | 0.6874 | 0.7603 | 0.3856 | 94.72% | 13.35% | 48.64% |
| uncertainty × `1-MSP` gate | 0.7010 | 0.7477 | 0.4083 | 94.92% | 13.08% | 51.01% |

结论：直接用 `1-MSP` 作为 feature-margin 权重无效，整体低于原 uncertainty gate；乘积门控能部分恢复性能，但仍未超过原方法。因此停止继续盲目尝试门控来源或权重扫描。当前更值得尝试的是用 EMA/冻结模型产生稳定门控，并配合 warm-up/ramp，检查是否是 batch 内自反馈权重导致表征更新噪声；如果仍无效，就停止堆叠 mixed-pool feature penalty，转向解耦的表征学习/拒识器目标。完整记录见 `analysis/uncertainty_gate_alternatives_s42_20261005.md`。

随后继续验证了两个时序方案。EMA uncertainty gate（decay=0.99）结果为 `AUROC/FPR95/OSCR/unknown rejection = 0.6943/0.7563/0.3888/13.73%`，低于原 uncertainty gate；这不支持“当前 student 自反馈是主要原因”的假设。当前 student 的 warmup/ramp（前 1 epoch 关闭，第 2 epoch 线性增加，第 3 epoch 达到完整权重）结果为 `0.7007/0.7550/0.4276/14.30%`，OSCR、unknown rejection 和 accepted-known accuracy（53.65%）改善，但 AUROC、FPR95 和 known acceptance 变差。全量几何诊断也是混合结果：classifier prototype 变差，centroid 和 nearest-sample 改善，不能称为稳定减少特征重叠。因此停止继续盲调 EMA decay 或 warmup/ramp；原 uncertainty gate 仍是这一族方法中最有希望的候选，但后续应转向解耦表征/拒识器目标或更可靠的不确定性监督。

本轮新增的可选参数包括 `--discovery-feature-margin-weight-source` 的 `ema_uncertainty`、`ema_msp`、`ema_product`，以及 `--discovery-uncertainty-feature-margin-warmup-epochs`、`--discovery-uncertainty-feature-margin-ramp-epochs`；默认行为不变。EMA smoke、warmup/ramp 完整训练与同协议检测均已完成，当前测试仍为 `158 passed`。完整记录见 `analysis/uncertainty_gate_alternatives_s42_20261005.md`。
## 2026-10-05: mixed-pool nnPU risk formula recheck

This round first rechecked the feature scaling protocol. The target
`support_uncertainty_augmented + nnPU` experiment already used joint
standardization inside `fit_nnpu_feature_rejector`; its recheck reproduced the
previous result exactly (AUROC `0.70094`, FPR95 `0.74933`, OSCR `0.33865`,
known acceptance `94.58%`, unknown rejection `16.23%`). It is therefore not
promoted over the simpler support-only representation.

The next code change added the opt-in `--rejector-nnpu-risk nu_corrected`.
It uses the direct non-negative negative-unlabeled risk decomposition for a
mixed pool, while `legacy` remains the default for compatibility. The change
is motivated by PU/NU risk-estimation work such as du Plessis et al. and Kiryo
et al.; this project uses a small linear rejector and does not claim to
reproduce those complete methods.

Under the same full-data CIFAR-100 random 60/40 protocol, matched checkpoint,
support-augmented features, known prior `0.2125984252`, support quantile
`0.95`, and 95% known-coverage threshold, the three-seed means were:

| Risk | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy | 0.71125 | 0.73956 | 0.35414 | 95.07% | 16.19% |
| nu_corrected | 0.71411 | 0.73839 | 0.35478 | 95.02% | 16.57% |

The corrected formula is worth retaining as a candidate: AUROC and FPR95
improve in all three seeds, and the average known-acceptance change is only
`-0.04` percentage points. However, the improvement is small, OSCR and the
operating-point unknown rejection are not better in every seed, and the
unknown rejection remains low. This does not solve known/unknown feature
overlap. The main unresolved direction remains a representation or mixed-pool
rejector objective that creates a stronger boundary, rather than more
threshold tuning. Full details are in
`analysis/nnpu_risk_formula_recheck_20261005.md`.

An interaction check combined `nu_corrected` with the previously weaker
`support_uncertainty_augmented` features on seed 42. It produced
AUROC/FPR95/OSCR/known acceptance/unknown rejection of
`0.70482/0.74917/0.34007/94.52%/16.60%`, versus
`0.70697/0.74517/0.34184/94.77%/16.40%` for support-only corrected risk.
The uncertainty coordinates therefore do not complement the corrected risk;
this combination is rejected and will not become a default.

The nonlinear follow-up added an opt-in two-layer MLP nnPU rejector. It
improved AUROC/FPR95 on seeds 42 and 123, but failed severely on seed 3407
(AUROC `0.50472`, known acceptance `100%`, unknown rejection `0%`). The
three-seed MLP mean (`AUROC 0.65642`, `FPR95 0.78722`, `OSCR 0.31221`, unknown
rejection `12.30%`) is worse than corrected linear nnPU, so MLP is rejected as
a default. This exposes substantial optimization/seed sensitivity in the
nonlinear mixed-pool risk fit. Any ensemble/regularization follow-up must be
selected using a predeclared validation-only protocol, not the test set. See
`analysis/nnpu_risk_formula_recheck_20261005.md`.

## 2026-10-05: review and selective absorption of the `lky` branch

Reviewed branch head `4e0cbec` (8 commits ahead of the then-current GitHub
`main`, 19 behind; therefore not a safe whole-branch merge). The branch adds
an older standalone `discovery_selection.py`, experiment runners/reports,
ImageFolder protocol changes, and detection/clustering diagnostics. The
standalone selector overlaps with the more developed candidate gating,
neighbor filtering, EMA weighting, and multi-criterion auto-K code already in
this checkout, so it was not copied as a second implementation. The peer
branch's ImageFolder warning did expose an incomplete local fix: `ImageFolder`
has integer `targets` as well as class-name folders, and the generic metadata
helpers had been checking integer targets against a name-keyed known-class
map. `_known_labels`, `_known_flags`, and `unknown_subset` now explicitly map
ImageFolder samples by class name. A regression test checks known/unknown
labels, unknown-pool membership, known-only stratified labels, and the
mixed-pool known fraction.

### Recheck of the branch's discovery-pool NT-Xent result

Before training another model, the checked-in local E/F run artifacts were
re-analyzed from their per-run configs and detection reports. This is an audit
of existing experiments, not a fresh training run. Across seeds 42 and 123,
both used CIFAR-100 60/40, 64px inputs, 15 epochs, the same recorded KD,
feature-KD, SupCon, and prototype settings. F additionally enabled
`alpha_discovery=0.05`, NT-Xent, and `discovery_pool_mode=unknown`; this is an
unknown-only pool selected by the known/novel split, not a fully mixed,
unfiltered open-world pool. Detection used the saved oracle-K reports.

| Local run artifacts (mean ± sample std, n=2) | AUROC | FPR95 | OSCR | known acc (all known) | known acceptance | unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E: full representation | 0.5823 ± 0.0119 | 0.8829 ± 0.0008 | 0.3118 ± 0.0124 | 0.4068 ± 0.0172 | 95.35% ± 0.59pp | 5.89% ± 0.41pp |
| F: E + unknown-only discovery NT-Xent | 0.6397 ± 0.0031 | 0.8413 ± 0.0016 | 0.4084 ± 0.0069 | 0.5221 ± 0.0107 | 95.03% ± 0.11pp | 8.64% ± 0.62pp |
| F − E | +0.0574 | −0.0416 | +0.0966 | +0.1153 | −0.32pp | +2.75pp |

The direction is favorable for all listed detection/known-class metrics in
both seeds, so discovery-pool contrastive learning merits a controlled
follow-up. It does not establish that the main mixed-pool problem is solved:
the training pool was prefiltered to unknown classes, the comparison has only
two seeds, and rejection is still low. Oracle-K candidate clustering NMI was
nearly unchanged (`0.5046` to `0.5053`); ARI rose from `0.0398` to `0.0679`.
The branch's committed aggregate reports E AUROC `0.6027 ± 0.0169`, whereas
the current local E reports recompute to `0.5823 ± 0.0119`; F agrees at
`0.6397 ± 0.0031`. Since the saved configs do not include checkpoint/source
hashes, the exact discrepancy cannot be resolved from the reports alone. Do
not quote the branch's `F − E` effect as a fully reproducible result until
both rows are rerun from hashed checkpoints and a recorded code revision.

### Verification performed

- The newly added ImageFolder regression test initially failed on the
  integer-target/name-key mismatch, then passed after the helper fix.
- Re-analysis outputs are in `analysis/lky_branch_recheck_20261005/` and are
  generated from the existing E/F reports; they do not represent new training.
- Other branch additions (temperature/ECE summaries, per-class error reports,
  and multi-seed aggregation) are already present locally in more extensive
  forms. The old selector module and its standalone relation-affinity helper
  were not absorbed because they are not wired into training and duplicate or
  lag behind current implementations.
- Next verification: run the full test suite; for a new algorithmic claim,
  compare unknown-only versus mixed-pool training on the same teacher, split,
  seeds, and budget, then report auto-K separately from oracle-K. The
  unknown-only result is an upper-bound/control setting, not the deployment
  protocol.

## 2026-10-05: sharpened uncertainty gate pilot

### Hypothesis and controlled comparison

The uncertainty-margin gate had nearly uniform weights (ESS fraction about
`0.96–0.97` in the prior three-seed audit). This pilot tested whether raising
the detached uncertainty to the fourth power focuses the margin loss on more
novel-looking samples without changing candidate order. The only intended
algorithmic change was `u -> u^4` (`power=1` remains the default and exactly
preserves the previous formula). CIFAR-100 random 60/40, seed 42, full data,
same ResNet-34 teacher, ResNet-18 student, five epochs, mixed pool,
`nnPU=0.1`, feature-margin coefficient `0.05`, margin `0.2`, optimizer and
augmentation protocol were held fixed. Detection was rerun with the same
normalized-entropy + min-class kNN (`k=10`), MC=8, full test set, and
known-validation 95% coverage policy. Clustering was skipped. The test set
was not used to select or tune the exponent.

| Seed 42 | AUROC | AUPR | FPR95 | OSCR | known acc (all known) | known acceptance | unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing linear gate (`power=1`) | 0.7108 | 0.5799 | 0.7348 | 0.4132 | 0.4848 | 95.30% | 13.90% |
| Sharpened gate (`power=4`) | 0.6891 | 0.5622 | 0.7770 | 0.3983 | 0.4767 | 94.30% | 15.10% |
| Change | -0.0217 | -0.0177 | +0.0422 | -0.0149 | -0.0082 | -1.00pp | +1.20pp |

Decision: reject `power=4` as an improvement. The higher unknown rejection
came with lower known acceptance, worse AUROC/FPR95/OSCR, and lower known
accuracy; it is not evidence that the feature distributions separated
better. Keep the option for reproducible ablations, but do not use it by
default and do not sweep more exponents based on this test set. Artifacts are
in `runs/uncertainty_margin_power4_s42/` and
`runs/uncertainty_margin_power4_s42_detect/`.

### Next distinct hypothesis

Rather than make the weight sharper, test whether augmentation instability is
causing false high-uncertainty weights: use the smaller uncertainty from the
two views as a conservative cross-view gate, so a sample receives a strong
margin penalty only when both views signal uncertainty. This is motivated by
consistency-based semi-supervised learning, but is a project-specific
ablation, not a direct reproduction of a paper. Keep all other settings and
the seed-42 protocol fixed; if AUROC/OSCR or known utility degrades again,
stop adding uncertainty-gated margins and return to the stable nnPU +
min-class kNN baseline rather than tuning the test operating point.

### Cross-view gate result

The paired full-data seed-42 run completed with only the gate source changed
from single-view uncertainty to the minimum uncertainty across the two
training augmentations (`power=1`). Detection was rerun with the same full
known training bank, test set, score, MC=8, and validation-only 95% known
coverage calibration:

| Seed 42 | AUROC | AUPR | FPR95 | OSCR | known acc (all known) | known acceptance | unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single-view uncertainty gate | 0.7108 | 0.5799 | 0.7348 | 0.4132 | 0.4848 | 95.30% | 13.90% |
| Cross-view minimum gate | 0.6894 | 0.5621 | 0.7552 | 0.4012 | 0.4768 | 95.37% | 13.03% |
| Change | -0.0214 | -0.0177 | +0.0203 | -0.0120 | -0.0080 | +0.07pp | -0.87pp |

This second gate hypothesis is also not supported: it did not reduce false
uncertainty enough to help, and worsened ranking and the unknown-rejection
operating point. Stop this family of uncertainty-gated feature-margin
variants; retain the default single-view option only as a documented
ablation, not as a promoted solution. The power-4 run likewise remains an
explicit negative ablation. Both are single-seed comparisons against a
three-seed prior candidate, so neither justifies replacing the current
full-data mixed-pool nnPU + min-class kNN baseline. Do not spend more compute
sweeping gate exponents or combining these two failed variants. Next work
should return to the best-supported representation candidate (`nnPU` plus the
original single-view uncertainty feature-margin) and assess it on an
independent class split or additional seeds with checkpoint/config hashes.
Also keep the known/unknown feature-overlap diagnostics fixed and compare them
to the operating metrics. Do not add another local-kNN score: radius and
neighborhood-vote variants have already failed replication in earlier
records, and score-only changes are not the current bottleneck.

Artifacts are in `runs/uncertainty_margin_crossview_s42/` and
`runs/uncertainty_margin_crossview_s42_detect/`.
Full protocol and both pilot tables are also recorded in
`analysis/uncertainty_margin_gate_pilots_20261005.md`.

### Independent training-seed replication (seed 2026)

After the two negative gate variants above, we returned to the only candidate
with a positive three-seed signal: the original single-view uncertainty
feature-margin (`alpha=0.05`, cosine margin `0.2`). This was a paired
replication on the *same fixed CIFAR-100 random 60/40 class split*, adding a
new training seed (2026); it is not an independent class-split test.

Both arms used the same seed-2026 ResNet-34 teacher, ResNet-18 student,
pretrained initialization, full data, five epochs, mixed unlabeled discovery
pool (known prior `0.212598`), and immediate nnPU (`alpha=0.1`). The treatment
added only the original uncertainty-weighted feature-margin loss
(`alpha=0.05`, margin `0.2`, source `uncertainty`, power `1`). Both detection
runs used the complete open test set, normalized entropy + min-class kNN
(`k=10`), MC=8, and a threshold calibrated on known validation examples only
for 95% target known coverage; clustering was skipped.

| Seed 2026 | AUROC | AUPR | FPR95 | OSCR | Known accuracy (all known) | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nnPU baseline | 0.7001 | 0.5739 | 0.7405 | 0.4377 | 0.5217 | 95.55% | 12.70% |
| + uncertainty feature-margin | 0.7033 | 0.5822 | 0.7312 | 0.4297 | 0.5103 | 95.75% | 13.88% |
| Change | +0.0032 | +0.0083 | -0.0093 | -0.0080 | -0.0113 | +0.20pp | +1.18pp |

Interpretation: the added seed weakly supports better ranking (AUROC/AUPR and
FPR95 move favorably) and yields about 1.18pp more unknown rejection without
lowering measured known acceptance. However, OSCR and known classification
accuracy regress, and the AUROC gain is only `0.0032`. This is mixed, small
evidence—not a resolution of the known/unknown overlap and not sufficient to
claim a robust improvement. Across the four paired seeds now documented,
mean AUROC gain is approximately `+0.0123`, mean FPR95 reduction `0.0209`,
mean OSCR gain `+0.0068`, and mean unknown-rejection gain `+0.78pp`; the
direction is not uniformly favorable across utility metrics. For seed 2026,
all-known classification accuracy fell by `1.13pp`. Full-data geometry also
gives a mixed result: classifier-prototype AUROC/overlap changed
`0.6807/0.7211 → 0.6723/0.7402`, centroid `0.6731/0.7464 →
0.6740/0.7544`, and nearest-sample `0.6974/0.6974 → 0.7071/0.6979`.
There is no consistent overlap reduction, so this seed does not support a
general representation-separation effect.

The teacher command was initially launched with the parser's default
ResNet-18 and failed strict checkpoint loading before student training. That
invalid attempt is excluded; a matched ResNet-34 teacher was trained and used
for both valid arms. Artifacts: `runs/uncertainty_margin_replication_s2026_*`.
Full protocol, geometry details, and execution caveat are documented in
`analysis/uncertainty_margin_replication_s2026_20261006.md`. Pause gate and
exponent tuning; the next useful validation is a separately generated class
split. Do not tune detector thresholds on test labels.

### 2026-10-06: combining mixed-pool NT-Xent with feature-margin

We tested whether two apparently complementary ideas could be combined:
two-view NT-Xent consistency on the mixed unlabeled pool, inspired by the
earlier unknown-only discovery experiment, plus the original uncertainty
feature-margin. The earlier branch result was not directly reusable because it
used a different split, teacher, epoch budget and small test subset, so the
combination was rerun under the current full-data protocol. A matched
NT-Xent-only arm was also run to identify whether any effect came from NT-Xent
itself or from an interaction.

All new arms used the fixed CIFAR-100 random 60/40 split, seed 42 or 2026,
full data, the same ResNet-34 teacher/ResNet-18 student setup, five epochs,
mixed pool, nnPU `0.1`, normalized entropy + min-class kNN (`k=10`), MC=8,
and a validation-only 95% known-coverage threshold. Only the discovery loss
was changed. Test labels were not used for training, threshold fitting or
model selection.

| Seed / arm | AUROC | FPR95 | OSCR | Known accuracy | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42: original uncertainty margin | 0.7108 | 0.7348 | 0.4132 | 48.48% | 95.30% | 13.90% |
| 42: NT-Xent only | 0.6988 | 0.7513 | 0.4139 | 49.03% | 94.98% | 14.60% |
| 42: NT-Xent + margin | 0.7025 | 0.7500 | 0.4117 | 49.13% | 96.03% | 12.83% |
| 2026: original uncertainty margin | 0.7033 | 0.7312 | 0.4297 | 51.03% | 95.75% | 13.88% |
| 2026: NT-Xent + margin | 0.7075 | 0.7442 | 0.4308 | 51.38% | 95.82% | 14.75% |

The seed-2026 combination looked locally better than its treatment, but seed
42 did not reproduce it: AUROC, FPR95, OSCR and unknown rejection all moved
unfavorably relative to the original margin. NT-Xent-only also reduced AUROC
and worsened FPR95 on seed 42 despite increasing operating-point rejection.
Therefore direct loss addition is not a stable solution to feature overlap.
The mixed pool contains known samples, so uniform NT-Xent consistency does not
provide a reliable direction away from the known manifold. Keep both variants
as ablations, stop coefficient stacking, and prefer a separate class-split
validation or a decoupled representation/rejector objective next. Full
protocol and artifacts are recorded in
`analysis/mixed_ntxent_margin_combination_recheck_20261006.md`.

### 2026-10-06: direction audit with a pure-unknown pool

The preceding matched 10-epoch mixed-pool test showed that uncertainty-weighted
feature-margin did not improve over the nnPU baseline. We tested whether the
same soft-weighted loss has a signal when the discovery pool is oracle-filtered
to novel classes, and also included the distinct unweighted feature-margin as
a separate arm. All arms used CIFAR-100 random 60/40, seed 2026, the same
ResNet-34 teacher / ResNet-18 student, full data, 10 epochs, and the same
detector and known-validation 95%-coverage threshold. The pure pool is selected
using training labels and is an upper-bound diagnostic, not the deployment
protocol.

| Pure-unknown arm | AUROC | FPR95 | OSCR | Known acc | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.6891 | 0.7638 | 0.4182 | 50.48% | 95.65% | 11.18% |
| Fixed feature-margin, `0.05/0.2` | 0.6984 | 0.7565 | 0.4425 | 52.83% | 94.87% | 14.40% |
| Uncertainty-weighted margin, `0.05/0.2` | 0.7043 | 0.7332 | 0.4593 | 54.80% | 95.62% | 12.75% |

On this one seed, the uncertainty-weighted arm also improved all three
post-hoc feature-distance AUROCs and reduced histogram overlap for classifier
prototypes, empirical centroids, and nearest known training examples. This is
a meaningful signal that the loss can affect representation geometry under a
clean pool; it is not a resolution of the core problem, since unknown rejection
is still only 12.75% and the overlap remains high. Keep the loss disabled by
default pending multi-seed replication.

Important audit caveat: pure vs. mixed is not yet an isolated test of
contamination. The mixed runs use nnPU uncertainty training and reserve some
known training examples into the unlabeled pool, while the pure-pool run
correctly disables nnPU and retains a different supervised-data count. The
within-pure comparison is paired; cross-protocol differences must not be
attributed solely to pool contamination. The next experiment should match the
known supervised subset and pool sizes, compare pure vs. controlled-mixed pools
with nnPU disabled in both, and only then add nnPU as a separate factor.
Detailed protocol, results, caveats and artifacts are in
`analysis/direction_audit_pure_unknown_margin_20261006.md`.

The central issue remains overlapping known/unknown representations and score
distributions, not simply threshold selection. Continue judging any change by
ranking metrics, fixed-known-coverage utility, and independent feature-overlap
diagnostics together; a higher unknown rejection rate by itself may only mean
more known samples were rejected.

### 2026-10-06: matched 2×2 pool-composition × margin diagnostic

To resolve the protocol confound noted above, we ran a matched 2×2 diagnostic:
pure-unknown vs. 20%-known mixed discovery pools, each with a baseline and an
uncertainty-weighted feature-margin arm. All four arms used seed 2026, the same
random CIFAR-100 60/40 split and ResNet-34 teacher, pretrained ResNet-18
student, full data, 10 epochs, 25,650 known supervised examples, and a 5,400
sample discovery pool. nnPU was disabled in all arms. Treatment changed only
the feature-margin coefficient (`0.05`, cosine margin `0.2`). The pure pool is
an oracle-filtered diagnostic, not a deployable setup. Detection was identical
across arms: normalized entropy + min-class feature kNN (`k=10`), MC=8,
validation-only 95% known-coverage threshold; clustering was skipped.

| Pool | Arm | AUROC | FPR95 | OSCR | Known accuracy | Known acceptance | Unknown rejection |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure | Baseline | 0.6779 | 0.7693 | 0.3865 | 46.32% | 95.55% | 10.98% |
| Pure | + uncertainty margin | 0.7063 | 0.7373 | 0.4580 | 54.67% | 94.95% | 13.88% |
| Mixed | Baseline | 0.6718 | 0.7783 | 0.3725 | 45.47% | 95.00% | 13.88% |
| Mixed | + uncertainty margin | 0.7095 | 0.7182 | 0.4658 | 54.97% | 95.25% | 12.88% |

The margin arm improved AUROC, FPR95, OSCR, and known classification in both
pool conditions. Full-data post-hoc feature geometry also improved in all six
distance-reference comparisons (prototype, empirical centroid, nearest known
training sample; AUROC increased and histogram overlap decreased). This is a
promising single-seed representation signal, but it does not solve the core
problem: treatment overlap remains about `0.68–0.74`. Crucially, unknown
rejection increased by `2.90pp` in the pure pool but decreased by `1.00pp` in
the mixed pool, despite better ranking. Thus better AUROC/geometry does not
guarantee a better chosen operating point; do not infer success from rejection
rate alone or blame pool contamination from this one seed.

Decision: keep uncertainty feature-margin optional and default-off. Before
another loss or threshold change, replicate this 2×2 comparison on at least
two more training seeds, with paired shared initialization and controlled
data-order/augmentation RNG streams, and report the pool×loss interaction.
If geometry/ranking gains replicate but the fixed-coverage rejection effect
does not, investigate score calibration on a separate validation protocol;
otherwise deprioritize the margin and evaluate stronger GCD representation or
a separate rejector. Full protocol, metrics, geometry and caveats are in
`analysis/matched_pool_factorial_margin_s2026_20261006.md`.

### 2026-10-06: mixed-pool nnPU plus uncertainty-margin combination

The preceding matched 2x2 study disabled nnPU in every arm. This follow-up
kept the mixed discovery pool fixed and compared nnPU uncertainty learning and
uncertainty-weighted feature-margin as two factors. All four arms used the
same CIFAR-100 random 60/40 split, seed 2026, pretrained ResNet-34 teacher,
pretrained ResNet-18 student, full data, 10 epochs, a 5400-image mixed pool
with known prior 0.2, and the same normalized-entropy plus feature-kNN
detector. The threshold was fitted only on known validation data for 95%
known coverage; clustering was skipped.

| PU | Margin | AUROC | FPR95 | OSCR | Known accuracy | Known acceptance | Unknown rejection |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | off | 0.6718 | 0.7783 | 0.3725 | 45.47% | 95.00% | 13.88% |
| off | on | 0.7095 | 0.7182 | 0.4658 | 54.97% | 95.25% | 12.88% |
| on | off | 0.7103 | 0.7293 | 0.4647 | 55.28% | 95.25% | 13.13% |
| on | on | 0.7205 | 0.7170 | 0.4764 | 56.25% | 95.70% | 13.15% |

Within the PU-on pair, the margin improves AUROC by 1.02 percentage points,
FPR95 by 1.23 points, OSCR by 1.18 points, and known accuracy by 0.97
points. Feature diagnostics also improve: classifier-prototype distance
AUROC/overlap changes from `0.6860/0.7107` to `0.7047/0.6898`, empirical
centroid from `0.6912/0.7232` to `0.7014/0.7004`, and nearest known sample
from `0.7138/0.6743` to `0.7277/0.6534`. This is a positive representation
signal, but unknown rejection is essentially unchanged, so the core problem
is not solved. The result is single-seed evidence only; the combined method
must be replicated before being promoted.

The training log now records
`discovery_uncertainty_feature_margin_weight_mean`, which verifies the actual
average soft novelty weight used by the margin term. The full protocol,
artifacts, caveats, and next decision criteria are in
`analysis/matched_nnpu_margin_factorial_s2026_20261006.md`.

### 2026-10-06: nnPU + uncertainty-margin replication on seed 42

To test whether the promising seed-2026 combination was reproducible, the
same matched full-data protocol was repeated with seed 42. The second seed
also included a PU-only arm, allowing the incremental contribution of the
uncertainty-weighted feature-margin to be separated from nnPU itself. All
detector settings, pool size, known prior, teacher/student architecture,
training budget and validation-only threshold policy were fixed.

| Seed | PU | Margin | AUROC | FPR95 | OSCR | Known accuracy | Known acceptance | Unknown rejection |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026 | off | off | 0.6718 | 0.7783 | 0.3725 | 45.47% | 95.00% | 13.88% |
| 2026 | on | off | 0.7103 | 0.7293 | 0.4647 | 55.28% | 95.25% | 13.13% |
| 2026 | on | on | 0.7205 | 0.7170 | 0.4764 | 56.25% | 95.70% | 13.15% |
| 42 | off | off | 0.6872 | 0.7478 | 0.4319 | 51.85% | 93.73% | 13.10% |
| 42 | on | off | 0.7152 | 0.7575 | 0.4637 | 55.40% | 95.20% | 15.75% |
| 42 | on | on | 0.7250 | 0.7233 | 0.4802 | 56.82% | 95.03% | 16.33% |

Within the PU-on condition, adding the margin improved both seeds. The mean
increment was AUROC `+0.0100`, FPR95 `-0.0233`, OSCR `+0.0141`, known
accuracy `+1.19pp`, and unknown rejection `+0.30pp`. The seed-42 treatment
also improved unknown rejection by `0.58pp` over PU-only while slightly
reducing known acceptance by `0.17pp`, so the gain is not explained by simply
rejecting many more known examples.

Post-hoc feature diagnostics on seed 42 also improved for every reference:
classifier-prototype distance AUROC/overlap changed from `0.6946/0.7016` to
`0.7002/0.6938`, empirical class centroid from `0.6918/0.7169` to
`0.7090/0.6930`, and nearest known sample from `0.7161/0.6820` to
`0.7270/0.6626`. This supports a representation effect, but large overlap
remains and the method does not yet solve the core problem. The combination
is now the leading candidate, with the third-seed replication completed. An
independent class split is still required before becoming the default. Full details are in
`analysis/nnpu_uncertainty_margin_replication_s42_s2026_20261006.md`.

### 2026-10-06: frozen-feature rejector recheck

在当前最有希望的 `mixed-pool nnPU + uncertainty-weighted feature-margin`
学生 checkpoint 上，进一步比较了两个独立 rejector。学生模型、mixed pool、
known open-validation、数据划分、阈值策略和测试集完全固定；rejector 只使用冻结
特征训练，不参与学生反向传播。详细记录见
`analysis/nnpu_margin_rejector_recheck_s42_20261006.md`。

| 检测器 | AUROC | FPR95 | OSCR | known acc | known accept | unknown reject |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原学生检测器 | 0.7250 | 0.7233 | 0.4802 | 56.82% | 95.03% | 16.33% |
| support-only nnPU rejector | 0.7529 | 0.6853 | 0.4796 | 56.00% | 94.85% | 19.73% |
| support + MC uncertainty rejector | 0.7470 | 0.6943 | 0.4739 | 55.98% | 94.93% | 20.75% |

support-only rejector 的 AUROC、FPR95 和未知拒绝率有明显改善，但 OSCR 和已知
准确率略降，因此保留为可选检测器，不替换主检测器。加入 MC uncertainty 后未知
拒绝率更高，但整体排序和 OSCR 变差，暂不吸收。后续实验继续以学生表征为主，
第三个 seed 的 PU-only / PU+margin 配对复现已完成；独立类别划分仍待验证。

### 2026-10-06: 第三个 seed、完整 overlap 复核与 rejector 复现

第三个 seed `3407` 已按与 `42/2026` 相同的 full-data、10 epoch 协议完成
PU-only / PU+margin 配对。详细记录见
`analysis/nnpu_margin_third_seed_and_rejector_20261006.md`。

| Seed | Arm | AUROC | FPR95 | OSCR | known acc | known accept | unknown reject |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026 | nnPU | 0.7103 | 0.7293 | 0.4647 | 55.28% | 95.25% | 13.13% |
| 2026 | nnPU + margin | 0.7205 | 0.7170 | 0.4764 | 56.25% | 95.70% | 13.15% |
| 42 | nnPU | 0.7152 | 0.7575 | 0.4637 | 55.40% | 95.20% | 15.75% |
| 42 | nnPU + margin | 0.7250 | 0.7233 | 0.4802 | 56.82% | 95.03% | 16.33% |
| 3407 | nnPU | 0.7122 | 0.7357 | 0.4331 | 50.48% | 95.35% | 14.45% |
| 3407 | nnPU + margin | 0.7105 | 0.7295 | 0.4593 | 54.42% | 95.57% | 12.68% |

margin 在三个 seed 上都改善 FPR95、OSCR 和 known accuracy，AUROC 在两个
seed 上改善，三 seed 平均 AUROC 变化约 `+0.0061`；但 seed 3407 的未知拒绝
下降，三 seed 平均 unknown rejection 反而下降约 `0.39pp`。因此它是有希望的
表征/效用候选，但还不能宣称解决未知拒绝问题，也不应只根据单一指标启用为
默认方法。

完整 feature-overlap 诊断必须显式使用 `--limit-train 0 --limit-val 0
--limit-test 0`。一次初始运行使用了脚本默认的有限样本协议，输出只有 589/411
测试样本，已丢弃；纠正后的 full-data 结果显示 margin 在 3407 上只改善
classifier-prototype 距离，empirical centroid 与 nearest known support 略变差：

| 距离参考 | nnPU AUROC / overlap | nnPU + margin AUROC / overlap |
| --- | ---: | ---: |
| classifier prototype | 0.6782 / 0.7224 | 0.6857 / 0.7072 |
| empirical centroid | 0.7051 / 0.6965 | 0.7024 / 0.7056 |
| nearest known sample | 0.7212 / 0.6712 | 0.7190 / 0.6734 |

这进一步说明表征改善不是所有参考距离上的一致分离，核心重叠仍然存在。

在相同的 3407 margin checkpoint 上复现 support-only nnPU rejector 后，结果如下：

| Seed | Detector | AUROC | FPR95 | OSCR | known acc | known accept | unknown reject |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 原学生检测器 | 0.7250 | 0.7233 | 0.4802 | 56.82% | 95.03% | 16.33% |
| 42 | support-only nnPU rejector | 0.7529 | 0.6853 | 0.4796 | 56.00% | 94.85% | 19.73% |
| 3407 | 原学生检测器 | 0.7105 | 0.7295 | 0.4593 | 54.42% | 95.57% | 12.68% |
| 3407 | support-only nnPU rejector | 0.7467 | 0.6683 | 0.4607 | 53.78% | 94.75% | 19.18% |

support-only rejector 在两个 seed 上都改善 AUROC/FPR95，并将未知拒绝提升
`3.40pp/6.50pp`；代价是不到 1 个百分点的 known accuracy 和 known acceptance。
因此保留为首选的可选检测器，但必须与 student training contribution 分开报告。
support + MC uncertainty 在 seed 42 上不如 support-only，暂不继续堆叠。

The overlap diagnostic script now defaults all limit parameters to `0` for
full-data evaluation. Positive limits must be supplied explicitly for smoke
tests, so limited-sample diagnostics are not mistaken for final evidence.

另一次 rejector 命令因漏写 `--discovery-pool-mode mixed` 被程序保护性拒绝，未产生
指标，不能当作负结果；补齐参数后才得到上表结果。后续所有 full-data 实验都要从
保存的 `config.json` 回读，并显式写出数据规模参数。

## 2026-10-06: semantic-hard independent split audit

To test whether the recent detector gains depend on the random class split, an
existing semantic-hard checkpoint was evaluated with the same detector family
and fixed settings. The baseline, support-only rejector, and fixed 0.75 fusion
were evaluated on the same checkpoint; no test labels were used for fitting or
selection. The checkpoint is an older 5-epoch, limited-test artifact, so this
is an audit rather than a final paper comparison.

| Detector | AUROC | FPR95 | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: |
| Student baseline | 0.5877 | 0.8618 | 93.33% | 8.31% |
| Support-only rejector | 0.6510 | 0.8179 | 92.52% | 14.29% |
| Fusion 0.75 | 0.6536 | 0.8065 | 93.01% | 11.69% |

The random-split fusion gains do not transfer as a uniformly better operating
point: fusion has the best ranking but support-only has higher unknown
rejection. The semantic-hard baseline is also much weaker than the recent
random-split baseline, confirming that class split and protocol materially
affect the result. The next fair comparison is a current full-data semantic-
hard student trained with the same nnPU + uncertainty-margin protocol and a
disjoint calibration subset. Full details are in
`analysis/semantic_hard_detector_audit_s43_20261006.md`.

## 2026-10-06: detector operating-point audit and metric correction

This round changed the audit protocol and reporting correctness, not the
student model. The new `scripts/audit_detector_operating_points.py` compares
the saved Student detector, support-only nnPU rejector, and 0.75-weight fusion
at common 90%, 95%, and 97% known-coverage operating points. Test labels are
used only for post-hoc diagnostics; they are not used for fitting or method
selection. Full details are in
`analysis/detector_operating_points_audit_20261006.md`.

The audit confirms that fusion improves AUROC, FPR95, and score-distribution
overlap on all three seeds, but its fixed-coverage unknown rejection is not
uniformly better than support-only nnPU: seed 3407 is slightly worse. The
remaining histogram overlap is about `0.61` for fusion. Therefore the fusion
is a promising optional ranking detector, not a solution to the known/unknown
representation-overlap problem. Further fusion-weight tuning on the final
test set is stopped.

The audit also found and fixed a metric-definition bug. Older reports stored
accepted-and-correct known samples divided by all known samples under the name
`known_class_accuracy_all_known`. New reports distinguish:

- `known_class_accuracy_all_known`: classifier accuracy over every known sample;
- `known_class_accuracy_after_accept`: classifier accuracy among accepted known samples;
- `accepted_correct_fraction_of_all_known`: accepted-and-correct known samples divided by all known samples.

New reports also expose `overall_accept_rate`; the historical `known_ratio`
field is retained only for compatibility and means the same overall accepted-
sample fraction, not known coverage.

Existing historical JSON files were not rewritten. New experiments must be
rerun after the fix before their corrected classification metrics are used in
a final table. The next high-value experiment is an independent semantic-hard
or semantic-isolated class split with a disjoint calibration subset. If the
ranking and overlap improvements replicate, restore clustering and evaluate
candidate purity, NMI, and ARI; otherwise return to the representation/data
protocol instead of adding more detector losses.

### 2026-10-06: rejector score fusion recheck

为检验 support-only rejector 与 student detector 是否提供互补信息，新增了可选的
`feature_rejector_fusion` 检测模式。它默认把当前
`normalized_entropy_min_class_knn` student 分数与 support-augmented nnPU rejector
分数分别在 known validation 上标准化，再按权重融合。测试集标签不参与拟合、标准化或
阈值选择；默认权重为 `0.5`，历史 score mode 不变。

在相同的 CIFAR-100 60/40、matched mixed pool、10 epoch、三 seed 和 95% known-coverage
协议下，进一步测试 rejector 权重 `0.75`：

| Detector | AUROC | FPR95 | OSCR | Known accuracy | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Student base | 0.7187 | 0.7233 | 0.4720 | 55.83% | 95.43% | 14.05% |
| Support-only nnPU | 0.7531 | 0.6797 | 0.4726 | 55.08% | 94.74% | 20.52% |
| Fusion, rejector weight 0.75 | 0.7595 | 0.6568 | 0.4863 | 55.65% | 95.22% | 20.38% |

融合在三 seed 平均 AUROC、FPR95、OSCR、known accuracy 和 known acceptance 上优于
support-only rejector，说明两路分数存在互补性；但未知拒绝率略低 `0.14pp`，且 seed 3407
上低于 support-only。因此它是当前最有希望的可选检测器，不是已经解决未知检测的证据，
也暂不改为默认方法。等权融合在 seed 3407 上较弱，说明支持边界信号不能被平均权重过度
稀释。

详细命令、逐 seed 结果、限制和后续验证要求见
`analysis/rejector_score_fusion_recheck_20261006.md`。下一步应固定 `0.75` 权重，在
独立类别划分和独立校准集上复核；通过后再恢复新类聚类，报告 candidate purity、NMI 和
ARI。核心问题仍是已知/未知表征重叠，融合主要改善排序层，不能替代训练阶段的表征分离。
### 2026-10-06: semantic-hard nnPU + uncertainty-margin paired replication

A strict seed-42 paired test compared the mixed-pool nnPU uncertainty loss plus
uncertainty-weighted feature margin against the same student with both terms
disabled. The split, teacher, pretrained backbones, full data, 10 epochs, 5,400
image mixed pool, known prior, detector, MC samples, and 95% known-coverage
calibration were fixed. Test labels were used only for final reporting.

| Arm | AUROC | FPR95 | OSCR | Known accuracy | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.5735 | 0.8627 | 0.3170 | 42.23% | 95.35% | 6.43% |
| nnPU + uncertainty margin | **0.6319** | **0.8430** | **0.3955** | **51.48%** | 95.62% | **9.03%** |

The post-hoc feature audit also found lower known/unknown histogram overlap for
classifier-prototype distance (`0.8805 -> 0.7079`), empirical centroid distance
(`0.9293 -> 0.8395`), and nearest-known-sample distance (`0.8765 -> 0.8208`).
This supports a real representation change rather than a threshold-only gain,
but the remaining overlap and low unknown rejection show that the core problem
is not solved. The result is currently a promising single-seed candidate, not a
final method claim. Full details are in
`analysis/semantic_hard_nnpu_margin_paired_s42_20261006.md`; seed-43 replication
is running under the same protocol before this is made a default.
### 2026-10-07: nnPU + uncertainty-margin two-seed replication

The seed-43 paired experiment reproduced the direction of the seed-42 result.
Both seeds used the same semantic-hard 60/40 protocol, full data, 10 epochs,
matched 5,400-image mixed pool, detector, and 95% known-coverage calibration.
Only the nnPU uncertainty loss and uncertainty-weighted feature margin were
enabled in the treatment.

| Seed | Arm | AUROC | FPR95 | OSCR | Known accuracy | Known acceptance | Unknown rejection |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | Baseline | 0.5735 | 0.8627 | 0.3170 | 42.23% | 95.35% | 6.43% |
| 42 | Treatment | **0.6319** | **0.8430** | **0.3955** | **51.48%** | 95.62% | **9.03%** |
| 43 | Baseline | 0.5837 | 0.8602 | 0.2955 | 39.20% | 94.93% | 6.40% |
| 43 | Treatment | **0.6260** | **0.8113** | **0.3999** | **51.88%** | 95.13% | **8.60%** |
| Mean | Baseline | 0.5786 | 0.8614 | 0.3062 | 40.72% | 95.14% | 6.41% |
| Mean | Treatment | **0.6290** | **0.8272** | **0.3977** | **51.68%** | 95.38% | **8.81%** |

The mean treatment improvement is +0.0503 AUROC, -0.0343 FPR95, +0.0915 OSCR,
and +2.40 percentage points unknown rejection, while known acceptance changes
only +0.23 points. Feature-overlap audits also improve in all six paired
comparisons across classifier-prototype, centroid, and nearest-known distances.
This is the first direction in the current project with repeated evidence of a
real representation change, not merely threshold movement. The absolute
unknown rejection remains low, so this is a partial improvement rather than a
solution to the core problem.

Decision: retain this combination as the leading optional training candidate;
do not add more detector losses or threshold variants yet. The next check is
clustering with fixed oracle K on the same two seeds, followed by candidate
purity/NMI/ARI analysis. Full details are in
`analysis/semantic_hard_nnpu_margin_two_seed_20261007.md`.

## 2026-10-07: 当前最有希望的方向——独立 feature rejector

本轮重新审查了核心问题：当前熵加 min-class kNN 主分数没有充分利用冻结
backbone 特征中的已知/未知信息。于是增加了一个不改 student checkpoint 的
后处理 rejector：已知训练特征作为可靠已知样本，mixed discovery pool 作为
无标签混合池，使用 `nnPU` 风险训练一个冻结特征上的线性 rejector。测试集
标签只用于最后统计，阈值仍只在 known validation 上按 95% known coverage
校准。

初步结果显示，这比继续调阈值或继续堆共享 backbone 损失更有价值：

| Seed | 方法 | AUROC | FPR95 | Unknown rejection |
| ---: | --- | ---: | ---: | ---: |
| 42 | 原有 entropy + min-class kNN | 0.5735 | 0.8627 | 6.43% |
| 42 | embedding feature rejector | 0.6847 | 0.7860 | 13.60% |
| 42 | uncertainty-augmented rejector | **0.6910** | **0.7743** | **15.60%** |
| 43 | 原有 entropy + min-class kNN | 0.6260 | 0.8113 | 8.60% |
| 43 | embedding feature rejector | 0.6799 | 0.7825 | 13.05% |
| 43 | uncertainty-augmented rejector | **0.6937** | **0.7670** | **13.63%** |

固定 oracle `K=40` 聚类时，uncertainty-augmented rejector 的候选池纯度为
seed=42 的 `66.88%`、seed=43 的 `65.43%`；未知-only ARI 分别为 `0.1216`
和 `0.1745`。这说明候选门控和检测排序都有稳定改善，但未知拒绝率仍然
不高，且 oracle K 不能作为无标签部署结果。完整记录见
`analysis/feature_rejector_uncertainty_augmented_two_seed_20261007.md`。

当前应区分三件事：

1. `nnPU + uncertainty margin` 仍是训练 backbone 的候选方案，但修正后的
   `nu_corrected` 风险在 seed=43 上没有优于旧 `legacy` 实现，因此暂不替换
   默认训练流程，只保留 `--discovery-uncertainty-pu-risk nu_corrected` 作对照。
2. backbone/projection consistency 只部分修复聚类，且会轻微损伤检测，保留
   为消融，不作为主方法。
3. 当前最有希望的是冻结特征上的
   `feature_rejector + uncertainty_augmented + nu_corrected`。它是解耦的
   后处理模块，还不能表述为已经完成端到端联合训练。

该方向仍需补充第三个 seed、非 oracle K、先验变化实验和独立校准协议；在
这些检查完成前，不应把测试集指标用于选阈值、选方法或调参。

### 2026-10-07: backbone feature consistency follow-up

The nnPU plus uncertainty-margin treatment improved unknown-candidate purity
but reduced novel-class NMI/ARI on seed 43. To test whether this was caused by
augmentation instability, the code now provides the opt-in
`--alpha-discovery-feature-consistency` loss. It aligns the two discovery views
in backbone feature space and does not assign pseudo labels. The default is
`0`, so existing experiments are unchanged.

Under the same seed-43 semantic-hard protocol, the new term with weight `0.1`
gave the following paired result:

| Arm | AUROC | FPR95 | OSCR | Unknown rejection | Candidate purity | Unknown-only NMI | Unknown-only ARI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nnPU + uncertainty margin | 0.6260 | 0.8113 | 0.3999 | 8.60% | 52.86% | 0.4403 | 0.0346 |
| + backbone consistency | 0.6241 | 0.8183 | 0.3928 | 8.13% | **54.17%** | **0.4854** | **0.0565** |

The result is mixed: consistency partially restores clustering structure and
raises candidate purity, but slightly worsens detection and remains below the
baseline on NMI/ARI. It is therefore retained as an ablation option, not as a
default improvement. The broader conclusion is that the project has a real
trade-off between rejection geometry and novel-class geometry. The next major
direction should decouple the rejection representation from the generalized
category-discovery representation instead of stacking another global loss.
See `analysis/semantic_hard_feature_consistency_recheck_s43_20261007.md` for
the full protocol and artifacts.

### 2026-10-07: strict seed audit, automatic K, and nonlinear nnPU rejection

The uncertainty-augmented feature rejector was rechecked on a third seed under
the same semantic-hard CIFAR-100 protocol. An initial seed-44 command omitted
`--pretrained` and did not explicitly match the teacher backbone; that run is
marked invalid and is not used as an algorithm result. The corrected rerun used
pretrained ResNet-34/ResNet-18, full data, 10 epochs, a matched 5,400-image
mixed pool, known prior 0.2, MC=8, and 95% known-validation coverage.

| Seed | AUROC | FPR95 | Known acceptance | Unknown rejection |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.6910 | 0.7743 | 94.85% | 15.60% |
| 43 | 0.6937 | 0.7670 | 95.20% | 13.63% |
| 44, corrected | 0.6593 | 0.7937 | 94.97% | 10.18% |

The third valid seed remains directionally better than the historical baseline,
but is weaker than seeds 42/43 and has lower validation accuracy. The current
method is therefore promising but seed-sensitive; it is not yet a stable final
claim. Full details are in
`analysis/strict_seed44_and_nnpu_mlp_audit_20261007.md`.

The same seed-44 candidate pool was also evaluated with oracle K=40 and
label-free automatic K. Automatic silhouette selection chose K=36. Candidate
purity was 57.40% for both settings, with unknown-only NMI 0.5074 (oracle) and
0.5034 (automatic). This indicates that K selection is not the main bottleneck;
candidate contamination and student representation quality are more important.

Finally, a small MLP nnPU rejector was tested while keeping the checkpoint,
pool, prior, feature mode, calibration, and test protocol fixed. It collapsed
to an almost non-rejecting score range: AUROC 0.5268, FPR95 0.9010, and 0%
unknown rejection, compared with the same seed-42 linear nnPU result (0.6910,
0.7743, and 15.60%). The MLP variant is therefore disabled as a candidate
improvement. We will not tune its threshold on test labels. A future nonlinear
rejector would require a separately validated objective and collapse checks.

Current decision: retain linear uncertainty-augmented nnPU as an optional
detector, but prioritize stabilizing the student representation and measuring
seed variance before adding more detector losses. Oracle K remains an upper
bound; automatic K is the deployment-style result.

## 2026-10-07 protocol audit correction

The recent seed comparison contained a protocol mismatch that is now recorded
explicitly in `analysis/protocol_audit_matched_pool_20261007.md`. The saved
seed-42 and seed-43 detection runs used `matched_discovery_pool=false` and an
approximately 25,400-sample mixed pool, while seed 44 used a matched 5,400-
sample pool. In addition, the saved seed-44 student-training history had both
the nnPU and uncertainty-margin training weights set to zero, so it was not a
valid training-time treatment run. These historical results are retained, but
they must not be presented as a strict three-seed replication.

The corrected detection-only matched-pool reruns used the same semantic-hard
split, 5,400-sample pool, known fraction 0.2, linear `nu_corrected` nnPU
rejector, uncertainty-augmented features, MC=8, and 95% known-validation
coverage:

| Seed | AUROC | FPR95 | Unknown rejection |
| ---: | ---: | ---: | ---: |
| 42, matched | 0.6899 | 0.7558 | 14.58% |
| 43, matched | 0.6985 | 0.7587 | 14.45% |

The result is close to the earlier unmatched runs, so pool composition is not
the main cause of the known/unknown feature overlap. `discover` now records
the actual pool size and known fraction in `calibration_report.json` and the
run configuration, and warns when an explicit nnPU prior differs materially
from the measured pool fraction. The previously required matched seed-44
training-time treatment and third-seed detector evaluation are now complete;
their corrected results are recorded below.

An additional opt-in training option `--student-ema-decay` is now available to
test whether epoch-level exponential moving-average student weights reduce
seed-sensitive representation oscillation. Its default is `0`, so existing
experiments are unchanged. This is a stability experiment, not yet a claimed
improvement; it must use the same checkpoint, pool, detector, and calibration
protocol as the corrected treatment.

The corrected seed-44 training-time treatment is now complete. Its nnPU and
uncertainty-margin loss meters were non-zero, and the best checkpoint was epoch
8 with known validation accuracy `0.4990`. Under the same matched detector
protocol, the valid three-seed treatment results are:

| Seed | AUROC | FPR95 | Unknown rejection |
| ---: | ---: | ---: | ---: |
| 42 | 0.6899 | 0.7558 | 14.58% |
| 43 | 0.6985 | 0.7587 | 14.45% |
| 44, corrected | 0.6822 | 0.7880 | 13.08% |
| Mean | 0.6902 | 0.7675 | 14.03% |

This is valid evidence that the treatment is directionally reproducible, but
the weaker third seed confirms seed sensitivity and the low absolute unknown
rejection confirms that feature overlap remains the main unsolved problem.
Details are in `analysis/protocol_audit_matched_pool_20261007.md`.

The EMA path also passed a toy one-epoch smoke test after creating a fresh toy
teacher checkpoint (`runs/audit_ema_smoke_teacher/` and
`runs/audit_ema_smoke/`). This only verifies training and checkpoint saving;
no claim about detection improvement is made until a matched CIFAR-100 paired
EMA experiment is run. The detailed record is in
`analysis/student_ema_smoke_20261007.md`.

## 2026-10-07: 独立拒识表征的最小实现

由于当前核心问题仍是已知/未知表征重叠，本轮没有继续堆叠阈值或后处理
分数，而是实现了一个默认关闭的独立拒识分支。`UKDNet` 现在可以通过
`--rejection-feature-dim` 增加 rejection projection、已知类辅助分类头和
基于 rejection embedding 的 uncertainty head。分类器继续使用原始
backbone feature；未知候选的 feature-margin 可以通过
`--alpha-discovery-rejection-feature-margin` 作用在 rejection embedding
上。后处理 rejector 可以使用
`--rejector-feature-mode rejection_embedding`。

这次修改的目标是检验“分类表征和拒识表征解耦”是否比继续修改共享
backbone 更有希望。所有新参数默认关闭，不改变历史流程。教师和学生
必须使用相同的 `--rejection-feature-dim`，否则 checkpoint 参数不匹配，
这是有意保留的结构一致性检查。

验证结果：`py_compile` 通过，完整测试为 `172 passed, 2 warnings`；toy
训练、保存 checkpoint 和使用 `rejection_embedding` 的 discover smoke 均
通过。toy 的 AUROC `0.6339`、unknown rejection `3.57%` 只说明代码可运行，
不能说明算法有效。当前还没有 CIFAR-100 结果，因此不能把该分支列为已
验证的改进，也不替换当前 matched 三 seed 的线性 uncertainty-augmented
nnPU 结果。

随后按固定 pilot 条件完成了独立拒识分支和 Outlier Exposure 对照：
semantic-hard CIFAR-100 60/40、seed=42、pretrained ResNet-34/ResNet-18、
1200/300/1000 train/val/test、400 mixed pool、2 epochs、线性
`nu_corrected` nnPU、MC=4 和 95% known-validation calibration。结果如下，
只能作为方向筛选，不能替代完整三 seed 实验：

| 方法 | AUROC | FPR95 | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: |
| baseline embedding rejector | 0.5723 | 0.9094 | 96.07% | 6.51% |
| independent rejection branch, margin 0.2 | 0.5319 | 0.9111 | 97.44% | 4.82% |
| independent branch, margin 0.0 | 0.5776 | 0.9607 | 95.38% | 8.19% |
| OE uniform + uncertainty + feature margin | 0.5344 | 0.9385 | 93.33% | 10.36% |
| OE uniform only | **0.5788** | 0.9128 | 95.38% | **9.64%** |

结论：独立拒识分支的思想仍有研究价值，但当前实现没有稳定改善；
margin `0.2/0.8` 时 loss 为零，原因是当前 hinge 为
`ReLU(similarity - margin)`，增大 margin 反而减少梯度。margin `0.0` 能产生
短暂梯度，但第二个 epoch 基本饱和，不能继续靠调 margin 解决。三项 OE
组合主要通过误拒已知样本提高 unknown rejection，AUROC 反而下降，应舍弃
该组合。uniform-only OE 是本轮最值得保留的候选，AUROC 和 unknown
rejection 有小幅方向改善，但 FPR95 没有改善，暂不替换主流程。

因此当前主结论仍是：已知/未知表征重叠没有被解决，不能把任何本轮 pilot
称为最终改进。下一步优先做 uniform-only OE 的完整 matched 多 seed 验证；
独立 rejection branch 暂作为次要消融，并改用不会快速饱和的 support-based
或非饱和 rejection objective，不再继续盲目调 hinge margin。详细记录见
`analysis/rejection_branch_smoke_20261007.md`。

### 严格 Angular 复核：不进入主流程

在严格 baseline 三 seed 协议下加入 ArcFace 风格角度间隔：
`alpha_angular=0.05`、margin `0.2`、scale `16`，其余 mixed-pool 辅助损失全部关闭。

| 指标 | 严格 baseline 均值 | Angular 均值 | 变化 |
| --- | ---: | ---: | ---: |
| AUROC | 0.5715 | 0.5468 | -0.0247 |
| FPR95 | 0.9206 | 0.9172 | -0.0034 |
| 已知接收率 | 95.83% | 95.32% | -0.50 个百分点 |
| 未知拒绝率 | 7.44% | 5.83% | -1.61 个百分点 |
| 候选纯度 | 55.50% | 48.47% | -7.03 个百分点 |
| 候选未知子集 NMI | 0.8420 | 0.6615 | -0.1805 |

Angular 只略微降低 FPR95，却损害 AUROC、未知拒绝率、候选纯度和聚类质量，
不解决核心表征重叠问题，仅保留为负向消融。

### 严格 Reciprocal Points 复核：有聚类信号但检测失败

使用 8 个 reciprocal points、margin `0.2`、权重 `0.1` 和纯未知 discovery pool
进行 seed 42 短协议实验。这是显式建模未知空间的方向，但不是只改变已知类几何或
logits。

| 指标 | 严格 baseline seed 42 | Reciprocal seed 42 | 变化 |
| --- | ---: | ---: | ---: |
| AUROC | 0.5725 | 0.5164 | -0.0561 |
| FPR95 | 0.9077 | 0.9487 | +0.0410 |
| 已知接收率 | 94.87% | 96.58% | +1.71 个百分点 |
| 未知拒绝率 | 6.99% | 4.82% | -2.17 个百分点 |
| 候选纯度 | 49.15% | 50.00% | +0.85 个百分点 |
| 候选未知子集 NMI | 0.8593 | 0.8927 | +0.0334 |

该方法在候选内部聚类上有小信号，但未知检测明显恶化，不能进入主流程。旧实验中
出现的 reciprocal 正向结果具有不同训练规模或协议，不能和这次严格审计直接合并。
完整结果见 `analysis/rejection_branch_smoke_20261007.md`。

### 特征重叠诊断：uniform OE 没有改善几何结构

为确认 OE 的影响不是单纯分数校准变化，使用严格 seed 42 baseline 与严格
uniform-only checkpoint 做了后验特征诊断。测试标签只用于分层统计，不参与
训练、阈值选择或模型拟合。

| 距离 | baseline 未知性 AUROC | uniform-only 未知性 AUROC | baseline 重叠度 | uniform-only 重叠度 |
| --- | ---: | ---: | ---: | ---: |
| 分类器原型距离 | 0.5636 | 0.5127 | 0.8197 | 0.8518 |
| 经验类中心距离 | 0.5006 | 0.5075 | 0.8541 | 0.8627 |
| 最近已知训练样本距离 | 0.5093 | 0.5133 | 0.8464 | 0.8385 |

这说明 uniform OE 没有改善特征几何，分类器原型距离反而变得更不可分；类中心
和最近邻距离仍接近随机。当前瓶颈确实是已知/未知表征重叠，而不是单纯阈值问题。
同时，2 epoch、1200 样本的短 pilot 本身会产生较弱表征，后续必须把短实验的
方向性结论与完整训练预算的结果分开。诊断原始数据见
`analysis/strict_uniform_feature_overlap_s42.json`。

### 重要更正：严格 uniform-only 归因复核

此前标为“uniform-only OE”的 seed 42 配置实际仍开启了
`alpha_discovery_uncertainty_feature_margin=0.05` 和
`alpha_discovery_uncertainty_pu=0.1`；此前 baseline 也开启了这两项。因此旧表只能作为历史 pilot，不能证明 uniform OE 的独立效果。

为纠正这一点，重新进行了严格三 seed 配对实验：semantic-hard CIFAR-100
60/40、seed 42/43/44、预训练 ResNet-34/18、1200/300/1000 数据、400 样本
matched mixed pool、2 epochs、MC=4、同一 `nu_corrected` nnPU rejector、95%
known-validation coverage。两组都关闭 mixed-pool uncertainty-feature-margin、
uncertainty-PU 和 independent rejection branch，唯一差异是 CIFAR-10
uniform OE 权重 0.1。

| 指标 | 严格 baseline 均值 | 严格 uniform-only 均值 | 变化 |
| --- | ---: | ---: | ---: |
| AUROC | 0.5715 | 0.5378 | -0.0337 |
| FPR95 | 0.9206 | 0.9327 | +0.0122 |
| 已知接收率 | 95.83% | 94.91% | -0.91 个百分点 |
| 未知拒绝率 | 7.44% | 5.65% | -1.78 个百分点 |
| 候选纯度 | 55.50% | 43.28% | -12.22 个百分点 |

严格配对结果否定了 uniform-only OE 作为当前主改进方向。之前的正向判断
来自实验定义混杂，而不是已确认的 uniform OE 增益；该方法保留为负向消融记录，
不再与其他拒识损失继续叠加。完整数据见
`analysis/rejection_branch_smoke_20261007.md`。

### Cleaned combination recheck

The follow-up removed both mixed-pool uncertainty terms from the combination;
the two corresponding loss meters were zero in both training epochs. The
matched detector result was:

| Method | AUROC | FPR95 | Known acceptance | Unknown rejection | Candidate purity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent rejection branch + uniform-only OE, cleaned | 0.5664 | 0.9487 | 95.38% | 7.23% | 52.63% |

This recovered part of the previous combination's degradation (`0.5445 ->
0.5664` AUROC and `42.62% -> 52.63%` purity), confirming that the mixed-pool
terms conflicted with the combined objective. It still underperformed both
single-factor references, so the combination direction is closed rather than
further tuned. The next focus is a matched multi-seed validation of
uniform-only OE by itself, with the independent branch retained only as an
ablation.
## 2026-10-07 soft support separation recheck

The new smooth-max plus softplus support-separation loss on the independent
rejection embedding was evaluated under the same semantic-hard CIFAR-100
pilot protocol. Its training loss was non-zero (`0.5609` in epoch 1 and
`0.4331` in epoch 2), confirming that the path received gradients.

| Method | AUROC | FPR95 | Known acceptance | Unknown rejection | Candidate purity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent rejection branch + soft separation | 0.5686 | 0.9128 | 96.07% | 7.95% | 58.93% |

This is below the independent-branch margin-0 result (`0.5776`) and
uniform-only OE (`0.5788`). The higher rejection rate is not a useful gain by
itself: 23 of 56 rejected candidates were known samples, so candidate purity
was only `58.93%`. This treatment is not retained as a leading method, and we
will stop tuning its margin, temperature, or coefficient. The next controlled
test combines the independent rejection representation with uniform-only OE.
Details are in `analysis/rejection_branch_smoke_20261007.md`.

## 2026-10-07: independent rejection plus uniform-only OE

The controlled combination was trained under the same semantic-hard CIFAR-100
pilot protocol. It kept the independent rejection branch and added only
CIFAR-10 uniform-logit OE (`alpha_outlier_uniform=0.1`); soft separation,
Energy OE, and uncertainty OE were disabled.

| Method | AUROC | FPR95 | Known acceptance | Unknown rejection | Candidate purity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent rejection branch + uniform-only OE | 0.5445 | 0.9453 | 94.02% | 6.27% | 42.62% |

This is worse than both single-factor references: independent branch margin-0
(`0.5776` AUROC, `8.19%` unknown rejection) and uniform-only OE (`0.5788`,
`9.64%`). The result argues against simple loss stacking. The next diagnostic
removes the mixed-pool uncertainty-feature-margin and uncertainty-PU terms while
retaining the two factors, to check whether those objectives caused the
conflict. If it also fails, this combination will be abandoned rather than
tuned further. Full details are in
`analysis/rejection_branch_smoke_20261007.md`.

## 2026-10-07: PU-corrected uncertainty feature-margin recheck

为检验 mixed discovery pool 中已知样本污染的问题，新增了可选的
`--discovery-uncertainty-feature-margin-mode pu_corrected`。它借鉴 PU 风险分解，
从 mixed-pool feature-margin 风险中扣除已知标注 batch 的估计贡献；已知校正项
detach，不直接产生排斥已知特征的梯度。默认仍为 `soft_weighted`，不会改变已有主流程。

在相同的 semantic-hard CIFAR-100 60/40、seed=42、预训练 ResNet-34/18、
1200/300/1000 数据、400 mixed pool、2 epochs、nnPU=0.1、margin=0.05、
cosine margin=0.2、min-class kNN、MC=4 和 95% known-validation coverage 条件下，
只改变 feature-margin estimator：

| 方法 | AUROC | FPR95 | OSCR | 已知接受率 | 未知拒绝率 | 候选纯度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| soft-weighted | 0.5222 | 0.9316 | 0.1280 | 95.38% | 6.51% | 50.00% |
| PU-corrected | 0.4804 | 0.9573 | 0.0915 | 93.33% | 6.02% | 39.06% |

两轮训练中的 PU margin loss 均非零，说明该分支真实参与训练；但所有主要指标均变差，
因此该方法降级为关闭状态的负向消融，不替换原 uncertainty-weighted margin，
也不继续调它的先验或系数。完整记录见
`analysis/pu_feature_margin_recheck_20261007.md`。

## 2026-10-07: semantic-isolated transfer pilot

To check whether the recent positive signal depended on the semantic-hard
class split, a matched pilot was run on the independent
`cifar100_60_40_semantic_isolated.json` split. The shared teacher, student
architecture, optimizer, pool size, mixed-pool known fraction (`0.2`), and
95% known-validation coverage calibration were fixed. The treatment alone
enabled mixed-pool `nu_corrected` nnPU uncertainty learning and the original
uncertainty-weighted feature-margin; the baseline disabled both terms. Both
training meters were checked to be nonzero in the treatment.

With the basic normalized-entropy plus min-class-kNN detector, the baseline
and treatment obtained AUROC/FPR95/unknown rejection of
`0.5928/0.8785/6.02%` and `0.6178/0.8386/10.53%`. A post-hoc support-only
linear nnPU rejector on frozen features gave `0.7004/0.7770/14.04%` for the
baseline checkpoint and `0.7469/0.7105/23.31%` for the treatment checkpoint.
This is a useful isolated-split positive signal, but the treatment's known
acceptance was `92.51%`, so the gain has a rejection trade-off and is not a
complete solution.

The feature audit remained mixed: nearest-known-sample distance AUROC changed
from `0.5816` to `0.6287` and centroid distance from `0.5796` to `0.6040`,
but classifier-prototype histogram overlap increased from `0.7813` to
`0.7908`. Thus local support geometry improved while global overlap remains.
On the treatment checkpoint, support-only rejector features were better than
support plus MC uncertainty (`0.7469` vs `0.7334` AUROC), and the 0.75 score
fusion was worse (`0.7089`); do not stack fusion or extra uncertainty features
without a new matched hypothesis.

The leading candidate is therefore the pair `nnPU + uncertainty-margin`
training plus a support-only nnPU rejector, pending multi-seed validation on
the isolated split. The exact reproducible pilot is in
`scripts/run_semantic_isolated_pilot.ps1`; full details and limitations are in
`analysis/semantic_isolated_pilot_s42_20261007.md`.

## 2026-10-07: isolated seed-43 replication and calibration check

The same short isolated pilot was rerun with only the training seed changed
to 43. With the support-only nnPU rejector, baseline versus treatment was
AUROC/FPR95/known acceptance/unknown rejection
`0.6939/0.7634/93.79%/17.57%` versus
`0.7048/0.7383/94.13%/13.86%`. Ranking still improved, but the fixed
95%-coverage work point did not. This is seed-sensitive operating behavior,
not evidence that the treatment is useless.

The feature audit was more consistent: classifier-prototype AUROC/overlap
changed `0.5156/0.8368 -> 0.5702/0.8180`, centroid distance changed
`0.4665/0.8538 -> 0.5944/0.7894`, and nearest-known-sample distance changed
`0.4984/0.8759 -> 0.6170/0.7515`. The training treatment therefore appears to
improve local geometry, while score calibration and class overlap still make
the final operating point unstable.

An isolated class-conditional 95%-coverage threshold diagnostic kept AUROC at
`0.7048`, raised unknown rejection to `22.03%`, but reduced known acceptance to
`88.93%`; it is not a valid improvement. The next priority is validation-only
rejector calibration with enough known validation data, followed by additional
independent seeds. Do not promote classwise thresholds or tune test labels.

## 2026-10-07: isolated three-seed conclusion

The isolated pilot was completed for seeds 42, 43, and 44 with identical
protocols. The mean support-only nnPU results were:

| Student | AUROC | FPR95 | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.7008 | 0.7799 | 94.16% | 17.17% |
| treatment | **0.7311** | **0.7186** | 94.38% | 17.50% |

The treatment reliably improves ranking, and all three feature audits show
improved local support geometry, but the mean unknown rejection gain is only
0.33 percentage points. Therefore the method has a real but incomplete signal:
it reduces some feature overlap without producing a stable fixed-coverage
operating-point improvement. The seed-43 class-conditional threshold test was
also rejected because it raised unknown rejection to `22.03%` only by reducing
known acceptance to `88.93%`.

Do not add more global losses or continue threshold sweeps at this stage. The
next controlled experiment is a full-data semantic-isolated paired run with
the same treatment and support-only rejector. Its purpose is to determine
whether the small-pilot instability comes from only 300 validation samples and
limited rejector training data. Full details are in
`analysis/semantic_isolated_pilot_s42_20261007.md`, and the exact reproducible
script is `scripts/run_semantic_isolated_pilot.ps1` (use `-Seed 42`, `43`, or
`44`).

## 2026-10-07: full-data isolated audit and rejector support sampling

The full-data seed-42 paired run used the same semantic-isolated split,
matched mixed pool, teacher, five epochs, and 95% known-coverage calibration.
With the existing support-augmented `nu_corrected` nnPU rejector, baseline
versus treatment was:

| Student | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.7541 | 0.6722 | 0.3293 | 95.22% | 18.90% |
| treatment | **0.7642** | **0.6485** | **0.3796** | 94.90% | 19.08% |

The full-data feature audit confirms a genuine representation effect:
classifier-prototype AUROC/overlap changed `0.6577/0.7552 -> 0.6792/0.7352`,
empirical-centroid distance changed `0.6253/0.8097 -> 0.6573/0.7591`, and
nearest-known-sample distance changed `0.6704/0.7440 -> 0.6954/0.6959`.
However, that geometry improvement still transfers only weakly to the fixed
coverage unknown-rejection rate. The core overlap problem remains open.

Two rejector-support controls were added. `--rejector-max-samples` controls
the known and mixed-pool fitting budget, and `--rejector-stratified-known`
selects the known side approximately equally across the 60 known classes.
On the same full-data checkpoints, stratified 5000-sample support improved
both arms relative to random 5000 sampling:

| Student | Support sampling | AUROC | FPR95 | OSCR | Unknown rejection |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | random 5000 | 0.7541 | 0.6722 | 0.3293 | 18.90% |
| baseline | stratified 5000 | **0.7677** | **0.6547** | **0.3364** | **19.70%** |
| treatment | random 5000 | 0.7642 | 0.6485 | 0.3796 | 19.08% |
| treatment | stratified 5000 | **0.7705** | **0.6070** | 0.3804 | **19.83%** |

This is a useful low-cost stability improvement, but not a solution: rejection
remains about 20%, and the single-seed result must be replicated before making
stratified support the default. Increasing the limit to 20000 was not stable:
it improved treatment to `20.05%` unknown rejection but reduced baseline to
`16.95%`. Keep the larger limit as an ablation, not a default.

A class-conditional empirical conformal support score was also implemented as
an optional negative control. On the pilot it obtained AUROC `0.5632` for
baseline and `0.5295` for treatment, well below the nnPU rejector. It should
not replace the current rejector or motivate more calibration tuning. Full
protocols and all support-sampling results are in
`analysis/semantic_isolated_full_support_sampling_20261007.md`.

The immediate next check is a second isolated seed using stratified support
sampling, with the same full-data paired protocol. If the ranking/FPR95 gain
replicates but unknown rejection remains near 20%, the next research change
should target the rejector objective or a more explicit class-boundary
representation loss, not another global threshold or score transform.

## 2026-10-07: full-data seed-43 replication

The seed-43 replication used exactly the same full-data semantic-isolated
protocol as seed 42: CIFAR-100 semantic-isolated 60/40 split, five epochs,
the same teacher and student architectures, matched mixed discovery pool of
5400 samples with known prior 0.2, support-augmented `nu_corrected` nnPU, and
known-validation calibration at 95% known coverage. The only changed factor
was the random seed.

| Student | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.7707 | 0.6055 | 0.3814 | 94.87% | 19.53% |
| treatment | **0.7836** | **0.5972** | **0.3977** | 94.80% | **21.35%** |

Relative to the same-seed baseline, treatment improves AUROC by `+0.0130`,
FPR95 by `-0.0083`, OSCR by `+0.0163`, and unknown rejection by `+1.83` percentage
points. Together with seed 42, stratified support sampling improves both
baseline and treatment relative to random support sampling, so it is a
promising rejector-stability option. It should still remain opt-in until a
third full-data seed is checked.

This result does not solve the core problem. Unknown rejection is still only
about 20%, and the treatment's gain is partly a better score ranking rather
than a clean separation of known and unknown features. The training histories
also show that the discovery-specific uncertainty losses are active, while
the joint novel-head losses remain zero in this support-only audit; therefore
this experiment validates the detector/training treatment pair, not the full
end-to-end novel-class discovery claim.

The exact seed-43 reports are stored under
`runs/semantic_isolated_full_s43_baseline_rejector_support/` and
`runs/semantic_isolated_full_s43_treatment_rejector_support/`. The detailed
comparison is in `analysis/semantic_isolated_full_s43_support_sampling_20261007.md`.

## 2026-10-07: full-data seed-44 support-sampling replication

The third independent seed used the same semantic-isolated CIFAR-100 60/40
split, five-epoch teacher/student protocol, matched mixed discovery pool,
support-augmented `nu_corrected` nnPU rejector, class-stratified known support
sampling, and validation-only calibration at 95% known coverage.

| Seed 44 | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.7738 | 0.6397 | 0.3575 | 95.83% | 16.70% |
| treatment | **0.7901** | **0.6118** | **0.4196** | 95.25% | **21.83%** |

The treatment changes are AUROC `+0.0163`, FPR95 `-0.0278`, OSCR `+0.0621`,
known acceptance `-0.58pp`, and unknown rejection `+5.13pp`. Across seeds
42/43/44, the mean changes are AUROC `+0.0131`, FPR95 `-0.0199`, OSCR
`+0.0429`, known acceptance `-0.32pp`, and unknown rejection `+2.38pp`.

This confirms that class-stratified support sampling is a useful stability
improvement for the decoupled rejector. It does not solve the main problem:
unknown rejection is still only `16.7%–21.8%`, so the known and unknown
representations remain substantially overlapped. We will therefore stop
tuning support size and global thresholds, and check the separate possibility
that the five-epoch student has not learned a sufficiently discriminative
representation. The next paired experiment changes only the training budget
and uses an explicit run tag; all detector and calibration settings remain
fixed.

The exact reports are under
`runs/semantic_isolated_full_s44_baseline_rejector_support/` and
`runs/semantic_isolated_full_s44_treatment_rejector_support/`. Details are in
`analysis/semantic_isolated_full_s44_support_sampling_20261007.md`.

## 2026-10-07: training-budget audit, 5 versus 10 epochs

The three-seed support audit showed a stable ranking improvement but unknown
rejection remained low. Before adding another loss, we checked whether the
five-epoch protocol itself was under-training the student. This paired seed-42
audit changed only the teacher/student training budget from 5 to 10 epochs;
the semantic-isolated split, mixed pool, uncertainty-margin treatment,
stratified 5000-sample support, `nu_corrected` nnPU rejector, detector, and
validation-only 95% known-coverage calibration were fixed.

| Budget / arm | AUROC | FPR95 | OSCR | Known accuracy | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 ep baseline | 0.7541 | 0.6722 | 0.3293 | 39.07% | 95.22% | 18.90% |
| 5 ep treatment | 0.7642 | 0.6485 | 0.3796 | 45.18% | 94.90% | 19.08% |
| 10 ep baseline | 0.7866 | 0.6027 | 0.3795 | 43.63% | 95.35% | 21.85% |
| 10 ep treatment | **0.8008** | **0.5518** | **0.4629** | **52.45%** | 94.18% | **23.65%** |

Ten epochs improves the baseline and treatment. Relative to five epochs, the
treatment gains AUROC `+0.0365`, reduces FPR95 by `0.0967`, improves OSCR by
`0.0834`, and raises unknown rejection by `4.57pp`. At the matched ten-epoch
budget, treatment still beats baseline by AUROC `+0.0142`, FPR95 `-0.0508`,
OSCR `+0.0834`, and unknown rejection `+1.80pp`.

This identifies under-training as a real confounding factor and makes 10
epochs the more credible budget for the next ablations. It does not solve the
core overlap: the ten-epoch treatment still accepts `76.35%` of unknown test
samples at the fixed operating point. We should therefore stop treating
training longer as the complete solution and next test one explicit
class-boundary representation objective at this fixed budget. The five-epoch
results remain useful historical comparisons but should not be compared to
ten-epoch results as if they were matched experiments.

Full details are in
`analysis/semantic_isolated_epoch_budget_s42_20261007.md`; exact artifacts are
under `runs/semantic_isolated_full_e10_s42_*`.

## 2026-10-07: known center-margin pilot

Because the core issue is class-boundary overlap, we tested a supervised
center-margin objective that uses known labels only. It compares each known
sample with its own batch class center and the nearest wrong class center; it
does not assign unknown pseudo-labels to the mixed pool. The three arms used
the same semantic-isolated seed-42 pilot protocol, mixed pool, stratified
support rejector, and validation-only calibration.

| Arm | AUROC | FPR95 | OSCR | Known accuracy | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.6691 | 0.8286 | 0.1571 | 20.13% | 97.17% | 7.27% |
| uncertainty treatment | **0.7206** | **0.6889** | **0.2153** | **26.62%** | 92.01% | **19.05%** |
| center-margin-only | 0.6983 | 0.8170 | 0.1673 | 20.63% | 95.34% | 12.03% |

Center-margin-only improves AUROC by `+0.0292` and unknown rejection by
`+4.76pp` over baseline in this short pilot, but its FPR95/OSCR improvement
is small and it is clearly weaker than the uncertainty treatment. The ten-
epoch confirmation subsequently scored below baseline on AUROC, FPR95, OSCR,
and unknown rejection. Center-margin is therefore retained only as a
negative boundary ablation; it is not a main component.

Details are in `analysis/center_margin_pilot_s42_20261007.md`; the script now
supports the isolated arm through `-KnownCenterMarginTreatment`.
## 2026-10-07: ten-epoch boundary and rejector audit

The three-epoch center-margin pilot was confirmed at ten epochs under the
same semantic-isolated split, matched mixed pool, stratified support, linear
`nu_corrected` rejector, and validation-only 95% known-coverage calibration.
The center-margin-only arm scored AUROC/FPR95/OSCR/unknown rejection
`0.7751/0.6375/0.3716/21.25%`, versus baseline
`0.7866/0.6027/0.3795/21.85%` and uncertainty treatment
`0.8008/0.5518/0.4629/23.65%`. The earlier short-budget positive signal did
not survive, so center-margin is now a negative ablation rather than a main
direction.

Two post-hoc rejector representations were then tested on the same ten-epoch
checkpoints. Appending per-known-class prototype similarities produced
baseline `0.7832/0.6130/0.3771/20.00%` and treatment
`0.8000/0.5490/0.4633/22.33%`; it does not improve the fixed operating point.
Replacing the linear nnPU rejector with the existing MLP version collapsed for
both arms to AUROC about `0.516`, FPR95 about `0.916`, and `0%` unknown
rejection. These are negative controls, not evidence that the underlying
student is worse.

The code now supports opt-in `prototype_augmented` and
`prototype_support_augmented` rejector features. During implementation,
compile/test checks caught and fixed detached-tensor and support-initialization
integration bugs. Verification after the fixes: `178 passed, 2 warnings`.

The main protocol remains ten-epoch uncertainty treatment with class-stratified
support and a linear `nu_corrected` rejector. The core problem is still real:
unknown rejection is only about 20%--24% at the fixed known-coverage point.
Further post-hoc feature stacking and MLP rejectors are paused. The next
controlled direction should change the training target for a dedicated
rejection representation or reformulate the PU risk with explicit score
regularization; it should not be another threshold sweep.

Full records:

- `analysis/center_margin_e10_s42_20261007.md`
- `analysis/rejector_representation_audit_s42_20261007.md`

## 2026-10-07: nnPU score-margin audit

To test whether the rejector was reaching a weakly separated score solution,
an opt-in regularizer was added. It penalizes
`relu(margin - mean(unlabeled_score) + mean(known_score))`; the default weight
is zero. With the fixed setting `weight=0.05`, `margin=0.5`, baseline changed
from `0.7866/0.6027/0.3795/21.85%` to
`0.7816/0.6165/0.3780/21.03%` for AUROC/FPR95/OSCR/unknown rejection. The
treatment changed from `0.8008/0.5518/0.4629/23.65%` to
`0.7988/0.5553/0.4634/23.00%`. Known acceptance was essentially unchanged,
but the main unknown-rejection and ranking metrics did not improve.

The scalar score-gap regularizer is therefore retained only as an ablation and
is not enabled by default. The next method must target the representation or
the class-conditional PU assumption; another global score constraint is not a
credible solution to the overlap problem. Details are in
`analysis/nnpu_score_margin_audit_s42_20261007.md`.

## 2026-10-07: rejection-feature margin wiring recheck

代码审计发现，`alpha_discovery_rejection_feature_margin` 原本只准备了
rejection 特征，真正的损失计算却错误地嵌套在
`alpha_discovery_rejection_feature_separation` 分支内。因此过去只开启
rejection margin 的实验不能作为有效结论。现在两个损失已经独立接线，并
新增回归测试确认：关闭 separation 时，单独开启 margin 会产生非零损失。

修复后的严格 pilot 固定使用 semantic-hard CIFAR-100 60/40、seed=42、
预训练 ResNet-34/18、1200/300/1000 训练/验证/测试样本、400 样本 mixed
pool（known prior=0.2）、2 epoch、rejection embedding 维度 128、同一教师
模型、同一 `nu_corrected` 线性 nnPU rejector、95% known-coverage 校准，
只改变 rejection margin 的目标余弦相似度：

| 实验 | margin 目标 | AUROC | FPR95 | OSCR | known acceptance | unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 控制组 | 关闭 | 0.5893 | 0.8923 | 0.1388 | 96.07% | 4.34% |
| 修复后 margin | 0.2 | 0.5610 | 0.9590 | 0.1286 | 93.33% | 7.71% |
| 修复后 margin | 0.0 | 0.5208 | 0.9521 | 0.1062 | 97.61% | 6.27% |

结论是：修复后的损失确实被执行，但它提高的拒绝率伴随着已知样本误拒和
排序指标下降，不能解决已知/未知表征重叠。因此保留这次代码修复，不把
rejection-feature margin 加入主流程，也不继续围绕它调阈值或调 margin。
过去相关 margin 结果统一降级为“接线未完全核实的历史探索”，以后引用时
应以本次修复后的配对结果为准。详细记录见
`analysis/rejection_margin_wiring_recheck_s42_20261007.md`。

本轮验证：`179 passed, 2 warnings`，compileall 和 `git diff --check`
均通过。下一步应回到 mixed-pool 的 PU 假设、候选污染和类条件 known prior
审计，寻找真正改变表征学习目标的方法，而不是继续增加全局拒识约束。

## 2026-10-07：prototype similarity rejector recheck

The recently added classifier-prototype cosine similarities were tested as an
additional input to the current support-only nnPU rejector. The comparison used
the same semantic-isolated split, seed, student checkpoint, 5,400-sample mixed
pool, stratified known support, `nu_corrected` risk, and validation-only 95%
known-coverage calibration. Only the rejector feature input changed.

| Rejector input | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: |
| `support_augmented` | 0.80078 | 0.55183 | 0.46295 | 94.18% | 23.65% |
| `prototype_support_augmented` | 0.80002 | 0.54900 | 0.46332 | 94.57% | 22.33% |

The prototype features did not improve the main ranking or rejection result;
the small FPR95/OSCR changes are not sufficient evidence of better separation.
This direction is retained only as an optional diagnostic, and is not promoted
to the main pipeline. The detailed record is
`analysis/prototype_support_rejector_recheck_s42_20261007.md`.

## 2026-10-07: candidate-gated feature-separation recheck

This paired test enabled a candidate-gated feature-separation loss on the
mixed discovery pool. The purpose was to test whether pushing selected
candidate features away from the known representation would reduce the
known/unknown overlap. Both arms used the same semantic-isolated CIFAR-100
split, seed 42, pretrained teacher, full data, three epochs, mixed pool,
support-augmented linear `nu_corrected` rejector, and validation-only 95%
known-coverage calibration. The treatment only added
`alpha_discovery_feature_separation=0.05`, candidate gating, margin `0`, and
temperature `0.1`. The training log confirmed that the separation loss was
non-zero only in the treatment.

| Student arm | AUROC | AUPR | FPR95 | OSCR | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.7665 | 0.6434 | 0.6630 | 0.3693 | 94.92% | 18.55% |
| candidate separation | 0.7617 | 0.6431 | 0.6497 | 0.3660 | 95.15% | 19.08% |

The treatment improved FPR95 by `0.0133` and unknown rejection by only
`0.53pp`, but reduced AUROC, AUPR, and OSCR. This is an insufficient/negative
result, not evidence that feature separation solved the core overlap. The
candidate-gated loss remains an opt-in ablation and is not added to the main
recipe. The detailed record is
`analysis/semantic_isolated_candidate_separation_recheck_s42_20261007.md`.

The next change should target the source of the representation signal rather
than add another noisy mixed-pool margin: use cleaner known-support geometry,
explicitly separate the rejector representation from the classification
embedding, or reformulate the PU objective with a validated class-conditional
assumption. Any new method must be compared against the fixed current
reference and judged jointly by AUROC, FPR95, OSCR, known acceptance, unknown
rejection, and feature-overlap diagnostics.

### Independent rejection branch plus uncertainty-PU recheck

On 2026-10-07 we completed a strict paired test of the independent rejection
branch with and without mixed-pool uncertainty nnPU training. Both arms used
the semantic-isolated CIFAR-100 60/40 split, seed 42, full data, the same
pretrained teacher, three student epochs, a matched 5,400-image mixed pool,
known prior `0.2`, a linear `nu_corrected` rejector on `rejection_embedding`,
MC=4, and validation-only 95% known-coverage calibration. The treatment only
enabled `alpha_discovery_uncertainty_pu=0.1` with the `nu_corrected` risk.

| Metric | Independent branch | Branch + uncertainty-PU |
| --- | ---: | ---: |
| AUROC | 0.7627 | 0.7178 |
| AUPR | 0.6280 | 0.5803 |
| FPR95 | 0.6327 | 0.7045 |
| OSCR | 0.3485 | 0.3141 |
| Known acceptance | 94.48% | 93.83% |
| Unknown rejection | 18.63% | 15.33% |

The PU loss was active in the treatment, but every main detection metric
became worse. This is a negative result: mixed-pool uncertainty is currently
too noisy to supervise the rejection embedding, so this loss remains opt-in
and is not promoted to the main recipe. The complete record is
`analysis/semantic_isolated_rejection_branch_nnpu_recheck_s42_20261007.md`.

The next check reuses the same checkpoints and compares the rejection embedding
with the existing uncertainty-augmented rejector feature set. That experiment
tests whether the remaining issue is the learned representation or simply
failure to use the available uncertainty signals; it changes no student
weights or evaluation protocol.

### Rejection and uncertainty feature-mode recheck

The next controlled comparison added two opt-in rejector modes:
`rejection_uncertainty_augmented` uses the independent rejection embedding
plus classifier and uncertainty summaries, while
`rejection_support_uncertainty_augmented` additionally appends the known-class
support score. The student checkpoints, mixed pool, known prior, nnPU risk,
MC=4 extraction, and validation-only 95% known-coverage calibration were all
fixed; only the rejector feature mode changed.

| Checkpoint | Feature mode | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| independent branch | `rejection_embedding` | 0.7627 | 0.6327 | 0.3485 | 94.48% | 18.63% |
| independent branch | `rejection_uncertainty_augmented` | 0.7658 | 0.6150 | 0.3563 | 94.73% | 17.78% |
| independent branch | `rejection_support_uncertainty_augmented` | 0.7654 | 0.6168 | 0.3582 | 94.28% | 19.25% |
| branch + uncertainty-PU | `rejection_embedding` | 0.7178 | 0.7045 | 0.3141 | 93.83% | 15.33% |
| branch + uncertainty-PU | `rejection_uncertainty_augmented` | 0.7607 | 0.6442 | 0.3405 | 93.92% | 21.48% |
| branch + uncertainty-PU | `rejection_support_uncertainty_augmented` | 0.7636 | 0.6315 | 0.3441 | 93.73% | 21.95% |

The hybrid modes are a useful opt-in post-hoc detector ablation, but no mode
dominates all metrics and unknown rejection remains below 22%. They do not
prove that the learned feature space is separated. The feature-stacking
exploration is therefore closed for now; further work should target cleaner
training supervision or protocol validity rather than adding more correlated
post-hoc features. Full details are in
`analysis/rejection_feature_hybrid_recheck_s42_20261007.md`.

### 2026-10-07：冻结教师候选权重与 CIFAR stem 复核

本轮继续围绕“已知/未知表征重叠、未知拒绝率低”做严格单变量检查。

首先修复了一个实验入口问题：`--discovery-feature-margin-weight-source`
使用 `ema_*` 时，参数校验原来只允许 `--discovery-selection-model ema`，
导致冻结 teacher 不能作为选择模型。现在同时允许 `ema` 和 `teacher`，并新增回归测试。
修复后测试为 `182 passed`。

在 CIFAR-100 semantic-isolated 60/40、seed=42、3 epochs、1200/300/1000
训练/验证/测试限制、matched mixed pool=5400、known prior=0.2 的完全配对协议中，
冻结 teacher 的 `uncertainty × MSP-novelty` 权重得到 AUROC `0.7255`、FPR95
`0.7155`、OSCR `0.2240`、unknown rejection `16.29%`；teacher 的 MSP-novelty
单信号得到 AUROC `0.7331`、FPR95 `0.7055`、OSCR `0.2411`、unknown rejection
`13.78%`，均未超过原 student-weighted treatment（AUROC `0.7469`、OSCR
`0.2372`、unknown rejection `23.31%`）。因此“改用冻结 teacher 产生候选权重”暂不作为主方法。

随后测试了 CIFAR 风格 ResNet stem（首层 `3x3/stride=1`、取消首个 max-pool）。
其 AUROC `0.7199`、FPR95 `0.7488`、unknown rejection `18.55%` 比默认 stem pilot
baseline 的 `0.7004`、`0.7770`、`14.04%` 有所改善，但 all-known accuracy 从
`19.30%` 降到 `14.64%`，OSCR 从 `0.1543` 降到 `0.1151`。这说明它可能改变了
排序，却没有可靠提高已知分类与开放识别的综合质量，继续保留为结构消融，不设为默认。

本轮结论与完整条件记录见
`analysis/teacher_selection_and_cifar_stem_recheck_s42_20261007.md`。下一步应停止继续
堆叠 teacher selector、后处理分数或 stem 变体，转向固定检测器下的联合表征目标：同时提高
已知类闭集表示质量，并让 mixed pool 的 novel prototype 学习使用稳定的 teacher/EMA 目标和
周期性刷新；所有新方案先做单变量 pilot，再决定是否进行完整多 seed 实验。
## 2026-10-07: rejection-branch uniform OE recheck

本轮继续围绕“已知/未知表征重叠、unknown rejection 偏低”进行严格配对实验。上一轮的 rejection feature margin OE 能改善 seed=42 的排序，但固定工作点拒绝率没有提高，因此新增了 `--alpha-outlier-rejection-uniform`：对外部 CIFAR-10 OOD 样本的独立 `rejection_logits` 施加 uniform-logit OE，使其不集中到任何已知类别；原有 rejection feature margin 保持不变。该开关默认值为 `0`，不会改变历史流程。

实验协议固定为 semantic-isolated CIFAR-100 60/40、预训练 ResNet-34/18、1200/300/1000 train/val/test、2 个 student epoch、rejection embedding=128、5400 样本 mixed pool、known prior=0.2、`nu_corrected` rejector、`support_augmented` 特征和 validation-only 95% known-coverage 校准。

| Seed | 对照 | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | rejection margin OE | 0.7030 | 0.7720 | 0.1090 | 94.51% | 11.53% |
| 42 | margin OE + rejection uniform OE | 0.7144 | 0.7255 | 0.1203 | 95.34% | 16.29% |
| 43 | rejection margin OE | 0.6619 | 0.8188 | 0.1090 | 94.97% | 7.92% |
| 43 | margin OE + rejection uniform OE | 0.6742 | 0.8356 | 0.0969 | 95.97% | 12.62% |
| 44 | rejection margin OE | 0.6916 | 0.8073 | 0.1505 | 89.97% | 24.73% |
| 44 | margin OE + rejection uniform OE | 0.7375 | 0.6768 | 0.1743 | 93.95% | 16.67% |

两 个 seed 上 AUROC 和 unknown rejection 都提升，unknown rejection 约提升 4.7 个百分点，known acceptance 也提高约 1 个百分点；但 FPR95 与 OSCR 一好一坏，说明这是目前较有希望的候选组合，仍不能宣称解决核心问题，也暂不改为默认配置。下一步先做 `uniform-only` 消融，确认收益来自 uniform 项本身还是与 feature margin 的交互；若继续有效，再用第三个 seed 或更大训练预算复核。详细记录见 `analysis/rejection_oe_uniform_recheck_s42_s43_20261007.md`。首次 seed=42 漏传 `--alpha-rejection-known-ce 1.0` 的 smoke run 已判为无效，不纳入结果。

补充的 seed=42 `uniform-only` 消融（关闭 feature margin，只保留 uniform OE）为 AUROC `0.7180`、FPR95 `0.7438`、OSCR `0.1285`、known acceptance `96.67%`、unknown rejection `11.03%`；相同无 OE 基线为 `0.6474/0.8220/0.0800/94.01%/10.53%`。这说明 uniform OE 单独主要改善排序，几乎没有提高工作点拒绝率；目前更有价值的是 uniform OE 与 rejection feature margin 的组合，但需要第三个 seed 复核后才能考虑进入主实验配方。

补充的 seed=44 结果显示，当前 validation-calibrated unknown rejection 从 `24.73%` 降到 `16.67%`，因此不能只看验证阈值工作点；但在仅用于事后诊断的 exact-95% test-known coverage 下，三组 seed 的 margin-only -> 组合未知拒绝率分别为 `9.02% -> 16.54%`、`8.42% -> 15.35%`、`12.10% -> 15.05%`。该诊断使用测试标签，不能用于模型选择，但说明组合在匹配覆盖率的排序比较上三组均有改善。当前结论是“有希望的候选，仍不设为默认”，下一步应使用更大训练预算和独立 open-validation 校准复核。

## 2026-10-07: rejection geometry and fixed-score audit

本轮继续围绕“已知/未知表征重叠、unknown rejection 偏低”做严格小规模验证。
新增两个默认关闭的独立 rejection embedding 训练开关：

- `--alpha-rejection-supcon`：只使用已知标签，在 rejection embedding 上施加监督对比损失；
- `--alpha-rejection-center`：只使用已知标签，在 rejection embedding 上施加 batch center compactness 损失。

这两个损失都没有把 mixed discovery pool 的伪标签当作真值，因此用于检查“已知类几何质量不足”是否是主要原因。两种方法均完成 seed=42/43 的 paired pilot。SupCon 在 seed=42 的 AUROC 从 `0.6474` 提高到 `0.7317`，但 seed=43 从 `0.7287` 降到 `0.7086`；center loss 在 seed=42 从 `0.6474` 提高到 `0.6757`，但 seed=43 从 `0.7287` 降到 `0.6894`。因此两者都没有跨 seed 稳定改善，暂时只保留为消融，不改变默认配置。

同一十 epoch treatment checkpoint 的固定评分器审计也已完成：diagonal Mahalanobis 是四个评分器中最好的，AUROC `0.6924`、FPR95 `0.7820`、unknown rejection `9.77%`；classwise Mahalanobis AUROC `0.5997`，normalized classwise 版本 AUROC `0.6745`。评分器替换不能把 unknown rejection 从低水平提升起来，说明当前主要瓶颈仍在表征学习，而不是阈值或 classwise 标准化。

完整数据和结论见 `analysis/rejection_geometry_and_score_audit_s42_s43_20261007.md`。下一步应停止继续堆叠 score/threshold 或仅靠 known-only compactness，转向能在表示学习阶段提供可靠未知结构的方案：稳定的 mixed-pool target/refresh、或协议明确的外部 OOD 监督，并继续报告 feature-overlap、AUROC、FPR95、OSCR、known acceptance 和 unknown rejection。

## 2026-10-07: rejection-logit detector ablation

本轮先补齐了两个 opt-in rejector feature mode：`rejection_logit_augmented` 和 `rejection_support_logit_augmented`。它们把独立 rejection head 的 logits、softmax entropy、最大概率和 top-1/top-2 概率差纳入 detector；支持版另外加入 known-support score。由于训练时 rejection head 已接受外部 CIFAR-10 OOD 的 uniform-logit OE，原本 detector 没有直接使用这一路 logits。

代码验证通过：`python -m compileall -q train.py novel_discovery`，`python -m pytest -q` 为 `188 passed`，`git diff --check` 通过。新增 `rejection_support_augmented` 作为严格匹配的 post-hoc 对照：它与 `rejection_support_logit_augmented` 使用相同的 rejection embedding、主分类统计和 known-support score，唯一差异是是否追加 rejection logits 及其统计量。

实验目的：检查 rejection logits 是否包含超出 rejection embedding、主分类统计与 support score 的可泛化未知信息。变量仅为 rejector feature mode，不重新训练 student。固定项为 CIFAR-100 semantic-isolated 60/40、每 seed 原有 student checkpoint、1200/300/1000 train/val/test、5400 mixed pool（known prior 0.2）、linear `nu_corrected` nnPU、MC=4 和 validation-only 95% known-coverage calibration；聚类关闭。seed=42/43 checkpoint 使用 feature-margin OE，seed=44 使用 feature-margin + uniform OE，因此结论是每个既定 checkpoint 内的配对检测器消融，不把 seed 间差异归因于单一训练方法。

| Seed | Rejector features | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | rejection embedding + support | 0.7312 | 0.7571 | 0.1121 | 96.51% | 14.29% |
| 42 | same + rejection logits | 0.7332 | 0.7621 | 0.1114 | 95.34% | 15.54% |
| 43 | rejection embedding + support | 0.7101 | 0.7651 | 0.1193 | 95.30% | 10.89% |
| 43 | same + rejection logits | 0.7047 | 0.7869 | 0.1186 | 96.48% | 9.16% |
| 44 | rejection embedding + support | 0.7150 | 0.7484 | 0.1737 | 95.70% | 11.83% |
| 44 | same + rejection logits | 0.7132 | 0.7691 | 0.1729 | 94.43% | 14.25% |

结论：增加 rejection logits 没有跨 seed 稳定改善。seed=42/44 的 unknown rejection 有所提高，但伴随 known acceptance、FPR95 或 OSCR 的退化；seed=43 的主要指标整体变差。该后处理方向不提升为默认方法，也不支持“只要把 logits 喂给 detector 就能解决重叠”的判断。较早把新模式与旧 `support_augmented` 直接比较的 exploratory runs 同时改变了底层 embedding，属于混杂比较，不用于证明 logits 的独立贡献。

因此当前更重要的问题不是继续堆叠 rejection score，而是验证 rejection head 学到的 logits 对未知类别是否有稳定语义信息，以及训练监督是否足以让 semantic-near unknown 离开已知区域。下一步优先在固定训练 checkpoint 上分析 rejection logits/embedding 的 known-vs-unknown 分布、按语义相近类别分层的误接收率与相关性；若 logits 与 rejection embedding 高度冗余或只在少数 seed 有效，就停止该支路的特征堆叠，回到训练阶段设计有真实未知结构且不污染验证/测试的监督或 mixed-pool 目标。完整实验记录见 `analysis/rejection_logits_detector_ablation_20261007.md`。

## 2026-10-07: full-data rejection uniform-OE recheck

为判断之前的短训练/限量数据是否低估 rejection uniform-OE，先在
semantic-isolated CIFAR-100 60/40、完整数据、ResNet-34/18、5 epochs 上完成 seed=42
配对实验，再固定数据划分 seed=42，将 student 训练随机种子扩展到 43、44。三组共用
同一教师；每个 seed 内结构化比较训练配置确认，唯一算法参数差异是
`alpha_outlier_rejection_uniform: 0 -> 0.05`，rejection feature-margin OE 保持为
`0.05`。新增项在外部 CIFAR-10 样本的 rejection logits 上最小化到均匀类别分布的
交叉熵，属于 Outlier Exposure 思路在独立拒绝头上的应用。

检测端使用完全相同的完整数据协议、5400 mixed pool（known prior=0.2）、linear
`nu_corrected` nnPU、`rejection_support_augmented`、MC=4、validation-only 95%
known-coverage 校准，并关闭聚类；仅替换学生 checkpoint。

| Model seed | Arm | Known accuracy | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | margin only | 0.3965 | 0.7485 | 0.6780 | 0.3392 | 95.00% | 17.15% |
| 42 | margin + uniform OE | 0.4433 | 0.7623 | 0.6323 | 0.3790 | 94.42% | 20.10% |
| 43 | margin only | 0.3905 | 0.7703 | 0.6158 | 0.3402 | 94.92% | 19.50% |
| 43 | margin + uniform OE | 0.4303 | 0.7698 | 0.6010 | 0.3742 | 94.58% | 19.13% |
| 44 | margin only | 0.4855 | 0.7808 | 0.5968 | 0.4216 | 94.38% | 19.93% |
| 44 | margin + uniform OE | 0.4307 | 0.7743 | 0.6277 | 0.3720 | 94.72% | 20.33% |

三 seed 配对差值（treatment 减 control）的均值为：AUROC `+0.0023`、FPR95
`-0.0099`、OSCR `+0.0081`、known accuracy `+0.0106`、known acceptance `-0.19` 个
百分点、unknown rejection `+0.99` 个百分点。seed=42 上未知误接收由 3314/4000 降到
3196/4000；按 40 个未知原始类别做的事后描述中，27 类接受率下降、2 类不变、11 类
上升。这一逐类分析只针对 seed=42，用于解释错误，不用于调参。

总体结论从 seed=42 的“有希望信号”更新为“三 seed 效果混合且不稳定”：seed=43 的
unknown rejection 下降，seed=44 的 AUROC、FPR95、OSCR 和 known accuracy 变差；三 seed
平均 AUROC 仅增加 `0.0023`、unknown rejection 仅增加 `0.99` 个百分点，不能认为方法
稳定有效或核心问题已解决。unknown rejection 仍约 19%--20%。因此 uniform OE 继续保持
opt-in 消融，不升为默认训练配置。完整结果见
`analysis/rejection_uniform_oe_full_data_s42_20261007.md`。

校准阈值未使用测试标签，但同一验证集既用于挑选最佳 checkpoint，也用于阈值校准，
两者并不独立。下一步先划出独立 calibration 子集、固定模型选择规则，再检查不同 seed
下 known/unknown 特征与分数分布是否一致变化；只有训练期诊断出现稳定趋势，才值得预先
设定 OE 权重做下一轮对照。不要根据测试结果选择权重或继续做阈值调整。
## 2026-10-11: rejection branch audit and balanced detector fusion

本轮先纠正了一个实验归因风险：之前的 rejection 分支 baseline 与 treatment
使用了不同 student checkpoint，不能把差异归因于 rejection 分支。重新控制
teacher、student 训练协议、mixed pool、nnPU rejector、`rejection_support_augmented`
和 95% known-coverage 校准后，只增加 `alpha_rejection_known_ce=1.0` 的处理组为
AUROC `0.7626`、FPR95 `0.6417`、OSCR `0.3351`、unknown rejection `17.23%`，低于
baseline 的 `0.7652/0.6472/0.3552/17.70%`。再加入
`alpha_rejection_supcon=0.1` 后为 `0.7496/0.6235/0.3253/16.15%`，同样没有改善。
两项损失在训练日志中均非零，因此不是参数未接入；当前不继续叠加 rejection
分支的 CE/SupCon。

随后审计发现，之前所谓的“representation-only”对照命令虽然改变了
`rejector_feature_mode`，但仍使用 `score_mode=entropy_proto`；rejector 分数没有参与
最终检测，因此那组 `0.7700/0.6332/0.3570` 结果不能归因于 balanced 特征，也不能作为
有效的算法对照。代码现在会拒绝这种“配置了非默认 rejector 特征、但最终分数完全不用
rejector”的静默错误；同时修复了 `skip_clustering` 时中间 rejector/support 分数被写成
全零的问题。

在真正使用 `score_mode=feature_rejector`、同一 seed=43 checkpoint 和同一 mixed-pool
协议下重新对照：

| rejector 特征 | AUROC | FPR95 | OSCR | known acceptance | unknown rejection |
|---|---:|---:|---:|---:|---:|
| `rejection_support_augmented` | 0.7509 | 0.6557 | 0.3151 | 94.58% | 16.43% |
| `rejection_balanced_support_augmented` | 0.7519 | 0.6512 | 0.3161 | 95.30% | 15.03% |
| balanced - support | +0.0010 | -0.0045 | +0.0011 | +0.72pp | -1.40pp |

balanced 的排序指标只有很小改善，而且固定 95% known-coverage operating point 的未知
拒绝率下降；因此它暂不升级为默认方案，只保留为轻量 detector ablation。当前结论仍是
已知/未知表征重叠没有被解决，后续应转向语义相近未知类别的分层误差与训练表征审计，
而不是继续堆叠特征拼接或阈值调节。

注意：一次较早的 SupCon 检测命令漏写 `--threshold-policy known_coverage`，实际使用
了默认 global policy；该结果不用于因果结论。完整记录见
`analysis/rejection_branch_audit_and_balanced_fusion_20261011.md`。

## 2026-10-11：seed-44 分层 known 采样复核与类别级误差审计

在 seed=44 上重新进行了严格对照。两组使用同一个 student checkpoint，
semantic-isolated CIFAR-100 60/40、5400 样本 mixed pool、known prior=0.2、
linear `nu_corrected` nnPU、MC=4、验证集 95% known-coverage 校准和关闭聚类
均保持不变；唯一变量是 rejector 拟合 known 支持样本是否按类别分层采样。
这次明确使用 `score_mode=feature_rejector`，因此 rejector 分数确实参与了检测。

| known 支持采样 | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
|---|---:|---:|---:|---:|---:|
| uniform | 0.7681 | 0.6137 | 0.3859 | 94.22% | 22.03% |
| class-stratified | 0.7700 | 0.6115 | 0.3837 | 94.58% | 20.25% |

seed=43 的复核也呈现相同趋势：分层采样只能带来极小排序改善，工作点未知
拒绝率反而下降。因此该选项不设为默认方法，只保留为 ablation。类别级事后
分析显示误接收并非集中在单一未知类别：uniform 支持下多个未知类别的误接收率
达到 0.87--0.97；分层采样对不同类别的影响方向相反，最大改善约 10 个百分点，
最大退化约 14 个百分点，说明它没有形成稳定的语义分离边界。

本轮进一步确认：当前核心问题仍是训练得到的已知/未知表征重叠，不能继续依靠
rejector 特征拼接、采样微调或阈值搜索解决。下一步应回到训练阶段，优先检查并
改进两点：一是提高已知类闭集表征质量（当前审计 checkpoint 的 known 分类准确率
仍偏低），二是对 mixed pool 使用不把伪标签直接当真值的稳定 teacher/EMA 目标、
周期性刷新和类别均衡约束，避免 rejector 只学习当前错误分数。新增的逐类别报告为
`analysis/classwise_rejection_stratified_s44_20261011.json`，完整记录见
`analysis/rejection_branch_audit_and_balanced_fusion_20261011.md`。

## 2026-10-11: mixed-pool PU boundary pilot

The current code now includes an opt-in `PU boundary` loss for mixed discovery
pools. The historical boundary loss treated the whole pool as pure unknown;
the new mode subtracts the estimated known contribution using the configured
known prior and uses detached class-centroid/prototype support statistics.
This directly targets feature overlap while avoiding hard unknown labels for
the mixed pool. It is enabled with `--alpha-discovery-boundary 0.05`
`--discovery-boundary-mode pu`; the default remains unchanged.

The controlled semantic-isolated CIFAR-100 pilot used the same teacher,
pretrained ResNet-18 student, 1200/300/1000 limits, 2 epochs, matched
400-sample mixed pool, known prior 0.2, `normalized_entropy_proto`, MC=4 and
validation-only 95% known-coverage calibration. Only the PU boundary term
changed:

| Seed | Arm | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection | KS |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | baseline -> PU boundary | 0.6420 -> 0.6581 | 0.8436 -> 0.8802 | 0.1722 -> 0.1963 | 93.51% -> 95.67% | 10.53% -> 9.27% | 0.2667 -> 0.2801 |
| 43 | baseline -> PU boundary | 0.5886 -> 0.6251 | 0.9161 -> 0.8725 | 0.1861 -> 0.1904 | 96.14% -> 92.11% | 5.20% -> 10.89% | 0.1714 -> 0.2232 |
| 44 | baseline -> PU boundary | 0.5975 -> 0.6000 | 0.8774 -> 0.8519 | 0.1501 -> 0.1484 | 92.04% -> 94.27% | 13.98% -> 10.22% | 0.1835 -> 0.1548 |
| 3-seed mean | baseline -> PU boundary | 0.6094 -> 0.6277 | 0.8790 -> 0.8682 | 0.1695 -> 0.1784 | 93.90% -> 94.02% | 9.90% -> 10.13% | 0.2072 -> 0.2194 |

This is a small positive ranking signal, not a solution: the mean unknown
rejection gain is only 0.23 percentage points and seed variation remains.
Keep the method as an ablation, do not make it default, and do not stack more
losses before a full-data, longer-budget confirmation. Detailed commands and
limitations are in `analysis/pu_boundary_pilot_20261011.md`.

On the same seed-42 protocol, replacing `hard_nnpu` with `soft_nnpu` at
softness `0.05` gave AUROC `0.6457`, FPR95 `0.8552`, OSCR `0.1961`, known
acceptance `93.84%`, unknown rejection `14.54%`, and KS `0.2235`. The hard
variant gave `0.6581/0.8802/0.1963/95.67%/9.27%/0.2801`. Soft-PU is therefore
more aggressive but worse for ranking, feature separation, and known
acceptance; hard-PU remains the preferred optional variant.

## Full-data PU boundary confirmation (2026-10-11)

The new mixed-pool PU boundary loss was checked with a matched full-data
experiment. Both arms used semantic-isolated CIFAR-100 60/40, seed 42, the
same pretrained ResNet-34 teacher and ResNet-18 student, all training data,
five student epochs, a matched mixed discovery pool of 5400 samples, known
prior 0.2, the same uncertainty nnPU term, and validation-only 95% known
coverage. The only intended training difference was the PU boundary term:
weight 0 for baseline versus weight 0.05, `mode=pu`, `hard_nnpu`, margin
0.2, and prototype weight 0.5 for treatment.

| Metric | Baseline | PU boundary | Difference |
|---|---:|---:|---:|
| AUROC | 0.6993 | 0.7059 | +0.0066 |
| AUPR | 0.5666 | 0.5761 | +0.0095 |
| FPR95 | 0.8023 | 0.7715 | -0.0308 |
| OSCR | 0.3840 | 0.3808 | -0.0031 |
| Known acceptance | 94.43% | 94.95% | +0.52 pp |
| Unknown rejection | 13.58% | 14.43% | +0.85 pp |
| Score KS distance | 0.3079 | 0.3043 | -0.0036 |

The result is a small positive signal for ranking and the selected operating
point, but it does not reduce global score overlap and still accepts 3423 of
4000 unknown test samples. PU boundary therefore remains an optional ablation,
not the main solution. Further work should prioritize direct feature-space
separation, calibrated rejection modeling, and semantic split difficulty
audits instead of repeatedly tuning this boundary loss. Full reports are in
`runs/pu_boundary_full_s42_baseline_discover/discovery_report.json` and
`runs/pu_boundary_full_s42_treatment_discover/discovery_report.json`.
### Same-checkpoint detector audit

The same full-data baseline and PU-boundary checkpoints were also evaluated
with the existing class-wise kNN support detector (`normalized_entropy_min_class_knn`,
feature bank from known training data, `k=10`, MC=4). The kNN score was more
informative than the prototype score, and the boundary model improved ranking
and score KS, but not the fixed-coverage unknown-rejection rate:

| Metric | Baseline + kNN | PU boundary + kNN | Difference |
|---|---:|---:|---:|
| AUROC | 0.7053 | 0.7185 | +0.0132 |
| FPR95 | 0.7743 | 0.7473 | -0.0270 |
| OSCR | 0.3857 | 0.3838 | -0.0019 |
| Known acceptance | 94.52% | 95.25% | +0.73 pp |
| Unknown rejection | 14.05% | 13.53% | -0.53 pp |
| Score KS distance | 0.3190 | 0.3339 | +0.0149 |

This confirms that local known-support geometry is useful for detection, but
does not by itself prove that the student embedding is separated. The next
training comparison therefore returns to semantic distance-consensus weighting
inside mixed-pool nnPU, which directly addresses known-sample contamination.

## 2026-10-11 follow-up: training budget is not the main fix

The completed seed-123 5-vs-10 epoch comparison shows that longer training
improves closed-set known accuracy (`46.58% -> 52.13%`) and OSCR
(`0.3914 -> 0.4358`), but does not improve open-set score ranking: AUROC
`0.7180 -> 0.7156`, AUPR `0.5938 -> 0.5824`, FPR95
`0.7635 -> 0.7685`, and unknown rejection at the validation-calibrated
operating point `13.53% -> 13.60%`. Known acceptance also falls slightly
(`95.92% -> 95.07%`). Thus this is a known-class learning gain, not a solution
to known/unknown overlap.

A paired, stratified bootstrap over the same test examples (2,000 replicates)
estimated the AUROC change as `-0.00233`, 95% interval `[-0.01126, 0.00673]`.
The interval includes zero; this single fixed split does not establish a
reliable AUROC difference. The result is conditional on the split/checkpoints
and is not multi-seed uncertainty. A 20-epoch extension was stopped before
checkpoint creation because each full-data epoch takes about two minutes and
the 5-vs-10 comparison gives little reason to expect longer training alone to
fix rejection. The interrupted attempt is not a result.

The per-class retrospective audit already shows false accepts spread across
many unknown classes and several known predicted-class sinks; it does not
support solving the issue with one class-specific threshold. A follow-up kNN
audit on the same seed-123 checkpoints exactly reconstructed the supervised
known training subset (21,600 samples; 1,080 known samples were reserved
unlabeled in the matched 5,400-sample discovery pool), used it as a
normalized-embedding bank (`k=10`), and reused the existing validation
thresholds. Unknown false accepts were closer to known training support than
rejected unknowns (nearest cosine distance `0.0846` vs `0.1267` at 5 epochs;
`0.0936` vs `0.1367` at 10 epochs), and their top-10 predicted-class neighbor
purity was `33.3%` / `39.6%`, compared with `11.7%` / `13.3%` for rejected
unknowns. This indicates both local feature absorption and acceptance of
weak-support examples; it does not show that one threshold or one class is the
whole cause. In particular, pickup truck (class 58) is 98% falsely accepted
in both runs, but its nearest known training class differs from the most
frequent classifier prediction, so local geometry and classifier decisions
are not interchangeable.

The next step is to create a genuinely disjoint open-validation subset from
the CIFAR-100 training partition (`--open-val-ratio > 0`) and retrain under
that protocol, because the existing checkpoints did not reserve such a subset.
Only then should a hybrid detector score be selected on open-validation and
evaluated once on untouched test data. Do not tune the fusion weight on the
test results. If ranking and known-coverage operating points do not improve
together, do not turn the diagnostic into a training loss. Keep test labels
strictly post-hoc and do not tune per-class thresholds from the test class
breakdown.
Full protocol and class-level evidence are in
[`analysis/error_neighborhood_audit_s123_20261011.md`](analysis/error_neighborhood_audit_s123_20261011.md)
and [`analysis/error_neighborhoods_s123_20261011.json`](analysis/error_neighborhoods_s123_20261011.json).
# 2026-10-11 严格配对消融审计：feature KD 与 prototype loss

本轮完成了一个严格的一因素对照，用于确认“关闭 feature KD 和 prototype loss 后未知检测变好”的结论是否可靠。实验固定 CIFAR-100 semantic-isolated 60/40、seed=123、预训练 ResNet-34/18、完整数据、5 epochs、matched mixed pool=5400、known prior=0.2、MC=4、`normalized_entropy_min_class_knn`（k=10）检测器、仅用 known validation 校准到 95% known coverage，并跳过聚类。唯一改变的是训练端 `alpha_feat_kd/alpha_proto`：主线为 `0.1/0.1`，消融为 `0/0`。

结果如下：主线 AUROC/FPR95/OSCR/known acceptance/unknown rejection/known accuracy 为 `0.7273/0.7455/0.3950/95.43%/13.63%/46.58%`；关闭两项损失后为 `0.7261/0.7327/0.3955/95.47%/14.70%/46.63%`。消融的 FPR95 和未知拒绝率小幅改善，但 AUROC 略降、OSCR 基本不变，不能认定这两项损失已经解决核心问题，也不足以支持直接删除默认损失。特征几何审计显示，关闭损失后最近已知训练样本距离的 unknownness AUROC 从 `0.7203` 到 `0.7250`，直方图重叠从 `0.6678` 到 `0.6545`，但该局部几何改善没有稳定转化为最终检测收益。

因此当前结论更新为：`feature KD + prototype loss` 不是核心瓶颈的可靠解决方案，但也没有证据证明应立即删除。后续停止盲目微调这两个权重，优先设计能够跨 seed 改善已知支持边界的表征学习或独立拒绝表征，并继续使用完全相同的检测协议复核。完整记录见 [`analysis/strict_feature_proto_ablation_s123_20261011.md`](analysis/strict_feature_proto_ablation_s123_20261011.md)，原始几何数据见 [`analysis/feature_geometry_strict_pair_s123_20261011.json`](analysis/feature_geometry_strict_pair_s123_20261011.json)。

## 2026-10-11 update: stricter ablation audit and CIFAR preprocessing pilot

This round focused on the same core problem: known and unknown features/scores
still overlap heavily, so the unknown rejection rate remains low. The first
change is methodological: scripts/audit_paired_runs.py now compares two saved
training configs and lists every non-ignored difference. It caught that an
older seed-42 feature-KD/prototype ablation was not strict, because many
discovery-loss and default fields differed. Future ablations should run this
audit before being interpreted.

After retraining a strict seed-42 pair, the only differences were
alpha_feat_kd=0.1 to 0.0 and alpha_proto=0.1 to 0.0. With the same
semantic-isolated CIFAR-100 60/40 split, full data, 5 epochs, matched mixed
pool, normalized_entropy_min_class_knn, k=10, MC=4, validation-only 95%
known coverage and clustering skipped, the result was:

| Variant | AUROC | FPR95 | OSCR | known acc | known accept | unknown reject |
|---|---:|---:|---:|---:|---:|---:|
| feature KD + prototype | 0.6939 | 0.7622 | 0.3447 | 41.37% | 95.22% | 11.75% |
| both disabled | 0.6829 | 0.7680 | 0.3214 | 38.43% | 94.92% | 10.83% |

Conclusion: feature KD and prototype loss should stay in the current recipe,
but they are supporting terms rather than a solution to the overlap problem.
Do not spend more time blindly tuning their small weights. Details are in
analysis/strict_feature_proto_ablation_s42_20261011.md.

The second change adds an opt-in preprocessing switch:
--transform-profile imagenet keeps historical behavior, while
--transform-profile cifar uses CIFAR-100 mean/std and CIFAR-style
resize/crop/flip transforms. This matters because CIFAR-100 images are small,
and weak known-class representation can make unknown samples easier to absorb
into known classes. The default remains imagenet for compatibility.

A first quick check with a CIFAR-profile student and an old ImageNet-profile
teacher was rejected as a KD protocol mismatch. A stricter short paired pilot
then trained matching teachers and students for each profile, using the same
seed, split, 1,200/300 train/validation limit, 2 teacher epochs, 2 student
epochs, and full 10,000-image open test evaluation:

| Profile | teacher acc | student acc | AUROC | FPR95 | OSCR | known accept | unknown reject | score KS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ImageNet profile | 13.67% | 20.70% | 0.6062 | 0.8875 | 0.1521 | 95.27% | 7.98% | 0.1617 |
| CIFAR profile | 18.00% | 27.68% | 0.6502 | 0.8528 | 0.2253 | 94.42% | 10.23% | 0.2389 |

Conclusion: CIFAR-style preprocessing is a promising new direction because it
improves known accuracy, AUROC, FPR95, OSCR, unknown rejection and score KS in
a matched short pilot, although known acceptance drops slightly. It should be
rechecked with longer/full training and matching teachers before becoming the
recommended CIFAR-100 protocol. Details are in
analysis/cifar_transform_profile_pilot_s42_20261011.md.

## 2026-10-11 update: full CIFAR preprocessing recheck

The short CIFAR preprocessing pilot was followed by a full seed-42 recheck.
This is now the most important result from this round. The comparison uses the
same semantic-isolated CIFAR-100 60/40 split, pretrained ResNet-34/ResNet-18,
5 student epochs, matched mixed pool of 5,400 samples, feature KD + prototype
loss, the same normalized_entropy_min_class_knn detector, k=10, MC=4,
validation-only 95% known coverage and clustering skipped. The CIFAR-profile
arm retrains its own teacher, so KD is not affected by input-normalization
mismatch.

| Profile | teacher val acc | student val acc | test known acc | AUROC | FPR95 | OSCR | known accept | unknown reject | score KS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ImageNet profile | 43.87% | 40.80% | 41.37% | 0.6939 | 0.7622 | 0.3447 | 95.22% | 11.75% | 0.2893 |
| CIFAR profile | 57.33% | 56.30% | 56.65% | 0.7710 | 0.6547 | 0.4934 | 95.08% | 19.35% | 0.4173 |

Conclusion: this is a real direction change. The current bottleneck is not only
which open-set loss is used; the CIFAR-100 input/data protocol was weakening
the representation. CIFAR-style preprocessing substantially improves known
accuracy and known/unknown score separation while keeping known acceptance
almost unchanged. It still does not solve the task, because unknown rejection
is only 19.35%, but it is much more effective than the recent small loss
tweaks. The next priority should be multi-seed CIFAR-profile rechecks, not
adding another new loss immediately. Full record:
analysis/cifar_transform_profile_full_s42_20261011.md.

## 2026-10-11 update: strict seed-123 CIFAR preprocessing confirmation

After the seed-42 full recheck, we ran a stricter seed-123 comparison to avoid
mixing preprocessing effects with discovery-pool or PU-loss differences. The
student training config audit reports only one substantive difference:
`transform_profile: imagenet -> cifar`. The discovery config audit differs
only in checkpoint/output paths and `transform_profile`. This makes the result
much cleaner than several earlier historical comparisons.

Protocol: CIFAR-100 semantic-isolated 60/40, seed 123, full data, pretrained
ResNet-34 teacher / ResNet-18 student, 5 epochs, matched discovery pool 5400,
feature KD + prototype loss kept on, PU discovery losses disabled, same
`normalized_entropy_min_class_knn` detector with k=10, MC=4, validation-only
95% known coverage, clustering skipped.

| Profile | teacher val acc | student val acc | test known acc | AUROC | AUPR | FPR95 | OSCR | known accept | unknown reject | score KS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ImageNet profile | 42.00% | 42.00% | 42.37% | 0.6789 | 0.5412 | 0.7643 | 0.3511 | 95.35% | 10.73% | 0.2578 |
| CIFAR profile | 58.70% | 54.83% | 54.53% | 0.7550 | 0.6314 | 0.6793 | 0.4726 | 95.20% | 16.08% | 0.3972 |

Conclusion: CIFAR-style preprocessing is now a validated high-value direction,
not just another small detector tweak. It improves known-class representation,
AUROC/AUPR/FPR95/OSCR, unknown rejection, score KS, and score-gap diagnostics
while keeping known acceptance near 95%. The core problem is still not solved:
even the better profile rejects only about 16% of unknown samples at this
operating point. But the result shows that part of the known/unknown overlap
came from a weak CIFAR-100 input protocol.

Code update: `--transform-profile` now supports `auto` and uses it by default.
For CIFAR-100, `auto` resolves to `cifar`; for other datasets it resolves to
`imagenet`. Historical runs remain reproducible by explicitly passing
`--transform-profile imagenet`. `inspect_data` now prints both the requested
and resolved transform profile so group members can see which protocol is
actually active.

Next direction: use the CIFAR profile as the recommended CIFAR-100 protocol and
re-evaluate the most promising training-side methods under this stronger
representation baseline. Avoid continuing to tune thresholds or score fusions
on the older ImageNet-profile CIFAR runs. Full record:
analysis/cifar_transform_profile_strict_s123_20261011.md.

## 2026-10-11 update: CIFAR profile plus audited mixed-pool nnPU

After validating CIFAR-style preprocessing, we rechecked whether the audited
mixed-pool uncertainty nnPU objective still adds value on top of that stronger
representation baseline. This experiment directly targets the current core
problem: known/unknown scores still overlap, and unknown rejection remains low.

The comparison fixes CIFAR-100 semantic-isolated 60/40, seed 123, CIFAR
preprocessing, the same CIFAR-profile ResNet-34 teacher, pretrained ResNet-18
student, 5 epochs, the same `normalized_entropy_min_class_knn` detector with
k=10, MC=4, validation-only 95% known coverage, and clustering skipped. The
treatment changes the student training protocol by enabling
`alpha_discovery_uncertainty_pu=0.1` and using a matched mixed discovery pool
with known prior 0.2. The discover config audit confirms the detector settings
are unchanged; only the student checkpoint/output path changes.

| Student | val known acc | test known acc | AUROC | AUPR | FPR95 | OSCR | known accept | unknown reject | score KS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CIFAR profile baseline | 54.83% | 54.53% | 0.7550 | 0.6314 | 0.6793 | 0.4726 | 95.20% | 16.08% | 0.3972 |
| CIFAR profile + nnPU | 59.03% | 60.73% | 0.7933 | 0.6701 | 0.6088 | 0.5349 | 95.68% | 19.78% | 0.4547 |

This is a meaningful positive result. The gain is not just a threshold effect:
known accuracy, AUROC, AUPR, FPR95, OSCR, unknown rejection, score KS, and
overlap diagnostics all improve together. The mean unknown-minus-known score
gap improves from 1.7302 to 2.1143, and the unknown-below-known-q95 overlap
drops from 0.8303 to 0.7805.

Limitation: the task is still not solved. Even after this improvement, only
about 19.78% of unknown samples are rejected at the 95% known-coverage operating
point. The next step is not to tune thresholds on the test set; it is to repeat
this exact comparison on more seeds. If the gain repeats, CIFAR profile +
audited mixed-pool nnPU should become the recommended CIFAR-100 recipe, and
future feature-margin or semantic-weight variants should be tested as single
changes on top of it. Full record:
analysis/cifar_profile_nnpu_s123_20261011.md.

## 2026-10-11 update: PU class-boundary loss recheck

We tested the existing class-conditional prototype/local-centroid boundary
loss in its mixed-pool PU-corrected form. The strict baseline was CIFAR-style
preprocessing plus audited mixed-pool uncertainty nnPU (`alpha=0.1`) on the
semantic-isolated CIFAR-100 60/40 split with seed 42. The treatment changed
only `alpha_discovery_boundary=0.1` and `discovery_boundary_mode=pu`; all
training, detector, and validation-only 95% known-coverage settings were held
fixed.

| Variant | AUROC | FPR95 | OSCR | known acc | known accept | unknown reject | score KS |
|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR + nnPU | 0.7906 | 0.6285 | 0.5536 | 63.08% | 94.80% | 23.00% | 0.4554 |
| + PU boundary | 0.7829 | 0.6570 | 0.5387 | 61.65% | 94.42% | 23.30% | 0.4412 |

The boundary term is confirmed to be active, but it does not improve the
distribution-overlap problem: most ranking and separation metrics decline,
while unknown rejection rises only 0.30 percentage points. It is not part of
the recommended recipe. Full details are in
`analysis/cifar_profile_boundary_pu_s42_20261011.md`.

The current main recipe remains CIFAR preprocessing plus unweighted audited
mixed-pool nnPU at `alpha=0.1`. Angular margin has now been tested and is not
enabled in the main recipe; the next controlled experiment changes only the
pretrained encoder fine-tuning scale, with threshold tuning and score fusion
unchanged.

## 2026-10-12 update: disjoint-calibration replication of encoder LR scale

The encoder-learning-rate result was rechecked with a stricter validation
protocol. On CIFAR-100 semantic-isolated 60/40, seed 44, the known validation
set was split into a model-selection subset and a separate threshold-calibration
subset (`--calibration-ratio 0.25`). The same pretrained teacher, matched mixed
pool, audited nnPU/KD/prototype settings, detector, and 95% known-coverage rule
were used for both students. Only `encoder_lr_scale` changed.

| Variant | AUROC | AUPR | FPR95 | OSCR | known acc | known accept | unknown reject | score KS | overlap unknown below known q95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scale 1.0 | 0.7738 | 0.6620 | 0.6870 | 0.5087 | 58.55% | 95.18% | 21.40% | 0.4147 | 0.7793 |
| scale 0.1 | 0.8089 | 0.7037 | 0.5888 | 0.5217 | 58.10% | 93.88% | 30.50% | 0.4807 | 0.7385 |

This replicates the positive direction: the continuous score ranking and score
distribution separation improve together, so the gain is not only threshold
tuning. However, 69.5% of unknown test samples are still accepted, and known
acceptance is slightly below 95% on the test distribution. Therefore
`encoder_lr_scale=0.1` is a stronger training baseline, not a solution to the
core overlap problem. Smaller uniform scales and the tested layer-wise decay
remain unpromising. The full protocol and artifacts are recorded in
[`analysis/cifar_profile_encoder_lr01_disjoint_calibration_s44_20261012.md`](analysis/cifar_profile_encoder_lr01_disjoint_calibration_s44_20261012.md).

The reproducible command sequence is in
[`scripts/run_encoder_lr_disjoint_calibration_s44.ps1`](scripts/run_encoder_lr_disjoint_calibration_s44.ps1).
The next method should act on the training-time representation/unknown signal,
use this same disjoint-calibration protocol, and change one algorithmic factor
at a time. Post-hoc threshold tuning, score fusion, and previously negative
independent-rejection-branch losses are not the next priority.

### Seed-123 strict replication: score-separation direction repeats, rejection gain does not

An earlier seed-123 LR comparison was confounded: its arms also changed the
mixed-vs-novel-only discovery pool and nnPU/prior settings, so it is excluded.
We reran seed 123 with the same teacher, full-data CIFAR-100
semantic-isolated protocol, five epochs, mixed pool size 5,400/prior 0.2,
nnPU/KD/loss settings, min-class kNN detector, and a disjoint 728-sample
known-validation threshold set; only encoder LR scale differed.

| LR scale | AUROC | AUPR | FPR95 | OSCR | Known accuracy | Test known accept | Unknown reject | KS | Unknown below known q95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.7945 | 0.6871 | 0.6403 | 0.5331 | 60.35% | 94.58% | 25.18% | 0.4590 | 76.35% |
| 0.1 | **0.8191** | **0.7255** | **0.5797** | **0.5602** | **62.20%** | 95.85% | 25.10% | **0.5000** | **70.30%** |

The second seed replicates better ranking and score separation, and known
accuracy also rises; however, unknown rejection at the validation-calibrated
operating point is unchanged. Together with seed 44, this makes 0.1 a
promising working baseline, not a proven solution: continuous geometry gains
are more reproducible than thresholded rejection gains, and most unknowns are
still accepted. Next, hold LR scale 0.1 fixed and test one training-time
geometry intervention using the same disjoint calibration protocol; report
known acceptance/classification and score-tail overlap alongside AUROC. Full
protocol, artifacts, and the excluded confounded comparison are in
[`analysis/encoder_lr_scale_s123_disjoint_calibration_20261012.md`](analysis/encoder_lr_scale_s123_disjoint_calibration_20261012.md);
the exact command is
[`scripts/run_encoder_lr_disjoint_calibration_s123.ps1`](scripts/run_encoder_lr_disjoint_calibration_s123.ps1).

## 2026-10-11 update: angular-margin recheck

We also tested an ArcFace-style known-class angular margin on top of the strict
CIFAR + nnPU baseline. Only `alpha_angular=0.05` was enabled (margin `0.2`,
scale `16`); the data split, teacher, mixed pool, nnPU coefficient, detector,
and validation-only threshold rule were fixed.

| Variant | AUROC | FPR95 | OSCR | known acc | known accept | unknown reject | score KS |
|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR + nnPU | 0.7906 | 0.6285 | 0.5536 | 63.08% | 94.80% | 23.00% | 0.4554 |
| + angular margin | 0.7806 | 0.6518 | 0.5424 | 61.95% | 93.92% | 25.13% | 0.4347 |

The rejection-rate increase is accompanied by lower AUROC, AUPR, OSCR, known
accuracy, known acceptance, and score separation. It is therefore not a valid
core improvement and is not enabled in the main recipe. Details are in
`analysis/cifar_profile_angular_margin_s42_20261011.md`.

## 2026-10-11 update: preserving pretrained encoder features

The next controlled test changed only the student encoder learning-rate scale
from `1.0` to `0.1` on top of the CIFAR + audited mixed-pool nnPU recipe. The
motivation is to preserve pretrained visual structure that may help separate
unknowns from known support.

| Variant | AUROC | AUPR | FPR95 | OSCR | known acc | known accept | unknown reject | score KS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CIFAR + nnPU | 0.7906 | 0.6757 | 0.6285 | 0.5536 | 63.08% | 94.80% | 23.00% | 0.4554 |
| encoder LR scale 0.1 | 0.8128 | 0.7121 | 0.5712 | 0.5536 | 62.08% | 95.12% | 26.78% | 0.4837 |

The result is promising: ranking, FPR95, score KS, score gap, and fixed
known-coverage unknown rejection improve together, while known acceptance stays
near 95%. Known accuracy drops by about one percentage point and OSCR is
essentially unchanged, so this is not yet a final solution. The correct next
step is strict replication on seed 123, not immediate stacking of another
loss. Full details are in
`analysis/cifar_profile_encoder_lr01_s42_20261011.md`.

## 2026-10-12 update: prior-aware unified Sinkhorn pilot

The unified joint head concatenates 60 known classes and 40 novel prototypes.
The old Sinkhorn path assigned equal mass to all 100 outputs, which implies
60% total mass for known classes and 40% for novel classes. The matched mixed
discovery pool in this experiment has a known prior of 20% and a novel prior of
80%. A new opt-in flag, `--joint-prior-aware-assignments`, now assigns per-class
mass `[0.2/60]` to each known class and `[0.8/40]` to each novel prototype. The
legacy equal-mass behavior remains the default.

在 seed=44、3-epoch matched cluster-diagnostic 命令下，等质量分配与 20/80
先验分配结果分别为：AUROC `0.6977 -> 0.7159`，FPR95 `0.7962 -> 0.7627`，OSCR
`0.2434 -> 0.2748`，未知拒绝率 `6.45% -> 13.44%`。已知接受率
`98.41% -> 95.70%`，所以一部分收益也伴随更多已知误拒。候选 purity
`70.59% -> 64.94%`，候选未知-only ARI `0.1825 -> 0.0294`，聚类端恶化。
这是单 seed 的初步信号，未知拒绝率仍然很低，且不能据此宣称核心问题已解决。

Implementation audit caught two bugs before valid results were obtained: the
first pilot did not pass the marginal through the ordinary cross-view call
chain; the second had incorrect unweighted Sinkhorn normalization. Both outputs
are excluded. After fixing both, the matched seed-44 diagnostic run showed a
modest detection gain but worse candidate purity and candidate-only ARI, so the
option remains experimental and disabled by default. A follow-up attempt to
combine it with soft residual weighting was incompatible with the unified
assignment path and failed before producing a result; it is not an algorithm
ablation. Details and valid/invalid artifact paths are in
[`analysis/joint_prior_aware_sinkhorn_pilot_s44_20261012.md`](analysis/joint_prior_aware_sinkhorn_pilot_s44_20261012.md)
including the hypothesis, protocol, audit trail, results, and next step.
The paired pilot command is recorded in
[`scripts/run_joint_prior_aware_pilot_s44.ps1`](scripts/run_joint_prior_aware_pilot_s44.ps1).

### Seed-43 replication: prior-aware Sinkhorn did not robustly help

We repeated the equal-mass vs 20% known / 80% novel Sinkhorn comparison on
seed 43, holding the teacher, semantic-isolated split, student training,
matched mixed pool, detector, and validation-only 95%-known-coverage
calibration fixed. Only `--joint-prior-aware-assignments` changed. The
comparison is a limited 3-epoch/1200-example pilot, not a full-scale result.

| Metric | Equal mass | Prior-aware 20/80 |
|---|---:|---:|
| AUROC | 0.7491 | 0.7519 |
| FPR95 | 0.6963 | 0.6795 |
| OSCR | 0.3061 | 0.3045 |
| Known accuracy (all known) | 34.73% | 34.23% |
| Test known accept | 92.11% | 92.45% |
| Unknown reject | 27.97% | 24.50% |
| Score KS separation | 0.3789 | 0.3737 |
| Candidate purity | 70.63% | 68.75% |
| Candidate unknown-only ARI | 0.0455 | 0.0450 |

The small AUROC/AUPR gain is not accompanied by better OSCR, score
separation, unknown rejection, or candidate quality. About 81% of unknown
scores remain below the known-score 95th percentile. Together with the
seed-44 pilot (which had a mixed detector/clustering outcome), this makes
prior-aware group mass inconsistent and not suitable as the default; it is
not a solution to known/unknown overlap. Do not stack additional losses on it
yet. The
reproduction protocol, full metrics, calibration caveat, and artifact paths
are recorded in
[`analysis/joint_prior_aware_sinkhorn_pilot_s43_20261012.md`](analysis/joint_prior_aware_sinkhorn_pilot_s43_20261012.md),
with the command in
[`scripts/run_joint_prior_aware_pilot_s43.ps1`](scripts/run_joint_prior_aware_pilot_s43.ps1).

The current strongest independently replicated direction remains preserving
pretrained encoder features with a smaller encoder learning-rate scale, but
it still leaves substantial score overlap. Continue from that representation
finding and audit known/unknown separation directly; a threshold adjustment
or different Sinkhorn group prior is not enough. The core criterion remains
whether continuous ranking, fixed-coverage rejection, and distribution
separation improve together, without degrading known acceptance or candidate
purity.

### Seed-43 encoder LR scale 0.05 follow-up

Per-score diagnostics showed feature kNN distance was nearly as discriminative
as the fused detector score, while uncertainty-head scores were weak. We
therefore tested a representation-training change rather than adding another
detector score. Relative to the matched seed-43 equal-mass scale-0.1 student,
only `encoder_lr_scale` changed to 0.05.

| Metric | scale 0.1 | scale 0.05 |
|---|---:|---:|
| AUROC | 0.7491 | 0.7303 |
| FPR95 | 0.6963 | 0.6980 |
| Known accuracy (all known) | 34.73% | 30.70% |
| Test known accept | 92.11% | 96.64% |
| Unknown reject | 27.97% | 16.09% |
| Score KS | 0.3789 | 0.3579 |
| Candidate purity | 70.63% | 76.47% |
| Candidate unknown-only ARI | 0.0455 | 0.0227 |

The higher candidate purity is accompanied by a much smaller rejection set,
lower unknown rejection, worse ranking/separation, and lower known accuracy.
Scale 0.05 is not an improvement. Keep 0.1 as the better-supported working
point, not a proven optimum, and avoid blind LR sweeps. The per-score
diagnostic and full protocol/results are recorded in
[`analysis/encoder_lr_scale_s43_005_pilot_20261012.md`](analysis/encoder_lr_scale_s43_005_pilot_20261012.md)
and reproduced by
[`scripts/run_joint_encoder_lr_s43_lr005.ps1`](scripts/run_joint_encoder_lr_s43_lr005.ps1).

Taken together, prior-aware assignment and lower encoder LR do not remove the
large overlap. The next algorithmic investigation should target training-time
known/unknown feature geometry and evaluate multiple seeds; do not infer
success from a higher rejection rate or purity alone.

### Joint residual re-audit: detached-weight bug and corrected pilot

Revisiting the older `--joint-mixed-residual` idea under the newer matched
seed-43 recipe exposed an implementation-fidelity bug: the residual weight
was documented as detached, but previously retained gradients into known
logits. `known_residual_weights` now returns a detached tensor, and a
regression test checks that no gradient can flow through the weight. Results
from residual runs before this correction are historical exploratory results,
not validation of the detached-weight method.

| Seed / variant | AUROC | FPR95 | OSCR | Test known accept | Unknown reject | Candidate purity | Candidate unknown ARI |
|---|---:|---:|---:|---:|---:|---:|---:|
| 43 equal-mass baseline | 0.7491 | 0.6963 | 0.3061 | 92.11% | 27.97% | 70.63% | 0.0455 |
| 43 residual, detached | 0.7527 | 0.6913 | 0.3080 | 93.12% | 25.00% | 71.13% | 0.0313 |
| 44 equal-mass baseline | 0.6977 | 0.7962 | 0.2434 | 98.41% | 6.45% | 70.59% | 0.1825 |
| 44 residual, detached | 0.7208 | 0.7420 | 0.2693 | 95.38% | 15.32% | 66.28% | -0.0290 |
| 44 residual + view agreement | 0.7203 | 0.7452 | 0.2764 | 96.18% | 13.98% | 68.42% | 0.0001 |

The corrected seed-43 run has only a small ranking gain; unknown rejection and
candidate ARI decline. Corrected seed 44 improves AUROC/AUPR and FPR95, but
known accuracy, OSCR, known acceptance, candidate purity, and candidate ARI
are worse than equal-mass. The two seeds therefore show mixed tradeoffs, not
a robust overall gain. A view-agreement follow-up on seed 44 raises candidate
purity (66.28% to 68.42%) and OSCR (0.2693 to 0.2764) versus residual alone,
but lowers AUROC slightly and reduces unknown rejection (15.32% to 13.98%).
A residual-weight floor of 0.1 also failed to provide a consistent tradeoff
(AUROC 0.7502, OSCR 0.3006, unknown rejection 22.28%). Thus residual weighting
and its view-agreement variant remain experimental and off by default. The
pre-detach seed-44 run is historical only and must not be cited as evidence
for the corrected algorithm. Full protocol, metrics, artifacts, and the
gradient-graph audit are in
[`analysis/joint_residual_fidelity_recheck_20261012.md`](analysis/joint_residual_fidelity_recheck_20261012.md).

This audit changes the next step: the residual-weight gradient defect is fixed
and the corrected residual method has been evaluated on two seeds. A follow-up
autograd audit also found that novel-mass weights retained gradients when used
by `unknown_feature_margin_loss`; novel-mass and bounded weights now detach,
and that loss also detaches any supplied sample weight as a defensive boundary.
Regression tests enforce this sample-weight semantics. The stored
`joint_novel_mass_s42_e3` run does not document an active joint novel-margin
term, so this correction is not evidence that its reported metrics were wrong
or that performance improved. Re-evaluate only historical runs that used a
nonzero weighted joint-margin objective. Details are in
[`analysis/sample_weight_gradient_fidelity_audit_20261012.md`](analysis/sample_weight_gradient_fidelity_audit_20261012.md).

Do not stack the inconsistent prior-aware Sinkhorn, residual floor, or extra
candidate losses. The success criterion remains concurrent
improvement in score ranking, validation-calibrated operation, and candidate
quality, with known-class performance reported.

### Class-wise audit of the remaining overlap

After the disjoint-calibration LR comparison, I grouped the existing official
test errors by CIFAR-100 class for seeds 44 and 123. This is retrospective
error analysis only: test labels were not used to train the model, select a
score, or fit thresholds. The run `split.pt` files were checked and match the
repository's semantic-isolated class map; a separate hidden split JSON in the
workspace has a different class list and was deliberately not used.

At LR scale 0.1, seven of the ten highest-false-accept novel classes recur in
both seeds. Television is falsely accepted at 97%/94%, telephone at 94%/94%,
lamp at 92%/93%, clock at 90%/95%, bus at 89%/92%, and pickup truck at
89%/93% (seed 44 / seed 123). Yet their most frequent predicted known classes
are heterogeneous: e.g. television tends toward wardrobe, bus toward
streetcar, and pickup truck toward lawn mower/couch. This is evidence of
repeatable class-structured difficulty, not justification for a test-informed
class-specific loss or threshold. Aggregate unknown false-accept rates remain
69.50% and 74.90%, respectively.

The models used `open_val_ratio=0`; the exposed discovery pool cannot be
retrospectively split and reused as independent validation. Therefore the next
valid comparison must reserve disjoint known and novel open-validation examples
from the training partition before pool construction, freeze the selection
protocol, and then compare one training-time boundary intervention against
the LR=0.1 baseline. Open-validation can support prespecified model/score
selection, but does not itself prove better representation geometry. The
method must improve ranking/overlap and retain known performance on the
untouched official test set; a threshold-only or test-class-targeted gain is
not acceptable. Full protocol and class breakdown are in
[`analysis/classwise_encoder_lr_disjoint_20261012.md`](analysis/classwise_encoder_lr_disjoint_20261012.md),
with machine-readable outputs in
[`analysis/classwise_encoder_lr_disjoint_s44_20261012.json`](analysis/classwise_encoder_lr_disjoint_s44_20261012.json)
and
[`analysis/classwise_encoder_lr_disjoint_s123_20261012.json`](analysis/classwise_encoder_lr_disjoint_s123_20261012.json).

### Rejection-branch uniform OE on the LR-0.1 baseline

**Matched-coverage follow-up:** a paired stratified bootstrap (2,000
resamples, retrospective test-label diagnostic only) recomputed each arm's
threshold at approximately 95% test-known coverage. Unknown rejection was
`42.80%` for margin-only and `40.40%` for margin + uniform OE. The
treatment-minus-baseline mean difference was `-1.84` percentage points (95%
percentile CI `[-5.05, +1.38]` pp); only `13.25%` of paired resamples favored
the treatment. AUROC difference was `-0.00060` (95% CI `[-0.00579,
+0.00476]`). Thus the `+7.70` pp unknown-rejection change at separately
calibrated thresholds is not evidence of better ranking; at matched known
coverage it does not help. See the bootstrap report
[`analysis/rejection_uniform_lr01_openval_s44_paired_bootstrap_20261012.json`](analysis/rejection_uniform_lr01_openval_s44_paired_bootstrap_20261012.json)
and implementation
[`scripts/paired_open_score_bootstrap.py`](scripts/paired_open_score_bootstrap.py).

为了跟进此前“外部 outlier rejection-feature margin + uniform rejection logits”
在小规模多 seed pilot 中的正向信号，本轮在 seed 44 的 LR-scale `0.1`
基线上只改变 `alpha_outlier_rejection_uniform`（`0` vs `0.05`）。其余训练与
检测配置固定，并在 discovery-pool 构造前从 CIFAR-100 训练划分预留
`open_val_ratio=0.1`。外部数据使用 CIFAR-10，存在潜在语义重叠，因此只能视为
受控 OE 探针，不能当作新类泛化结论。

结果：AUROC `0.8805 -> 0.8800`、AUPR `0.8108 -> 0.8049`，基本没有提升；FPR95
略好 `0.4162 -> 0.4113`，OSCR 仅微升 `0.5592 -> 0.5616`。validation calibration
subset 的 known acceptance 两臂都是 `94.96%`，但官方 test known acceptance
从 `95.80%` 降至 `93.65%`，unknown rejection 从 `39.05%` 升至 `46.75%`。
KS 仅 `0.6008 -> 0.6041`，而 unknown 落在 known q95 以下的比例反而从
`57.15%` 增至 `59.35%`。因此拒绝率提高伴随更多已知误拒，排序和分布重叠没有
可靠改善；本轮判为“不确定/不晋升”，不作为核心算法改进，也不设为默认。

第一次检测命令因缺少显式 `--discovery-pool-mode mixed` 在拟合 rejector
前报错，没有产出有效指标；修正脚本后复用了已训练 checkpoint，最终两臂检测
均正常完成。完整协议、结果与判断见
[`analysis/rejection_uniform_lr01_openval_s44_20261012.md`](analysis/rejection_uniform_lr01_openval_s44_20261012.md)，
可复现脚本见
[`scripts/run_rejection_uniform_lr01_openval_s44.ps1`](scripts/run_rejection_uniform_lr01_openval_s44.ps1)。
如要继续此方向，应先在预先指定的第二 seed 上做同协议复核，并按共同 known
coverage 比较；否则优先探索真正改变表征的训练干预，而不是继续调 uniform 权重。

## 2026-10-12: nnPU encoder-gradient paired audit (seed 44 and 123)

This paired audit tested one mechanism: whether the mixed-pool uncertainty nnPU loss should backpropagate through the shared encoder. The head-only control used `--discovery-uncertainty-pu-detach-features`; the treatment left this flag off. Within each seed, the teacher, CIFAR semantic-isolated 60/40 split, CIFAR transform, five-epoch student training, matched 5,400-sample mixed pool, known prior 0.2, nnPU risk, detector, MC=4, and validation-only 95% known-coverage calibration were fixed.

| seed | arm | AUROC | AUPR | FPR95 | OSCR | known acc | known accept | unknown reject | KS | unknown below known-q95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 44 | encoder gradient | 0.8135 | 0.7108 | 0.5978 | 0.5315 | 58.97% | 94.60% | 28.95% | 0.4871 | 72.75% |
| 44 | head only | 0.8079 | 0.7023 | 0.6005 | 0.5252 | 58.80% | 94.43% | 26.80% | 0.4806 | 74.95% |
| 123 | encoder gradient | 0.8106 | 0.7124 | 0.5955 | 0.5624 | 62.70% | 95.03% | 26.25% | 0.4861 | 73.68% |
| 123 | head only | 0.8154 | 0.7138 | 0.5785 | 0.5632 | 62.55% | 95.40% | 24.33% | 0.5048 | 74.23% |

The encoder-gradient arm increased unknown rejection by 2.15 and 1.93 percentage points, but AUROC, FPR95, OSCR, and KS did not improve consistently: the AUROC difference was +0.0056 for seed 44 and -0.0048 for seed 123. Calibration known acceptance was 94.92% for both arms within each seed, so this is not a threshold-calibration artifact. The result is a small, seed-sensitive diagnostic signal, not evidence that the core known/unknown overlap is solved. Keep the flag optional and do not make encoder-gradient the default yet.

The fifth-epoch raw nnPU losses were 0.1501/0.1739 (seed 44, encoder/head-only) and 0.1488/0.1783 (seed 123). After the 0.1 coefficient they are about 0.015--0.018, but scalar loss values do not determine encoder gradient norms. The next mechanism audit should record per-loss encoder gradient norms and cosine conflicts for CE, KD, SupCon, and nnPU before changing the nnPU weight or routing.

Reproduction scripts: `scripts/run_nnpu_encoder_gradient_s44.ps1` and `scripts/run_nnpu_encoder_gradient_s123.ps1`. Reports are in the four corresponding `runs/nnpu_encoder_gradient_s*_*/discovery_report.json` files.

### Uncertainty-head score-weight sweep (detector-only, two seeds)

With four nnPU checkpoints held fixed (seed 44/123, encoder-gradient/head-only),
we compared uncertainty correction weights `0.25` and `0.75` against the
existing `0.5` score. Only the detector score changed; validation normalization,
MC=4, feature bank, and validation-only 95%-known-coverage calibration were
retained. The score is entropy-z + min-class-kNN-distance-z +
weight × uncertainty-head-z.

This is not a stable fix for known/unknown overlap. Weight `0.75` raises
unknown rejection in some arms, but seed 123 encoder-gradient AUROC decreases
`0.8131 -> 0.8097`, FPR95 worsens `0.6012 -> 0.6137`, and KS decreases
`0.4886 -> 0.4819` versus weight `0.25`; across all arms, `unknown below
known-q95` remains roughly 71.5%-74.0%. The best weight depends on seed and
metric, so the default is unchanged and no weight is promoted. Full paired
table, interpretation, and artifacts are in
[`analysis/uncertainty_score_weight_s44_s123_20261012.md`](analysis/uncertainty_score_weight_s44_s123_20261012.md).

The weight is exposed for controlled ablation, and a regression test verifies
zero-weight equivalence with the original entropy+kNN detector and linear
weight behavior. Validation: `218 passed, 2 warnings`, `compileall`, and
`git diff --check` pass. The next priority remains training-time representation
separation; do not spend more runs tuning this detector weight without a
prespecified validation-only selection rule.

### MC sample-count stability check (4 vs 8)

To check whether noisy MC uncertainty estimates explain the overlap, four
student checkpoints were held fixed and detector MC samples were changed from
4 to 8; score, kNN bank, calibration protocol, and test data stayed the same.
AUROC changed by at most `0.0004` and AUPR by at most `0.0012`. FPR95 worsened
in three of four arms, and unknown rejection declined in three of four. The
unknown-below-known-q95 overlap remained about `71.5%-73.1%`, with no consistent
improvement. This does not support increasing MC count as a solution to the
core overlap; keep MC=4 for efficiency. Because dropout masks were not
explicitly paired between runs, treat tiny differences as a stability check,
not a precise causal estimate of sample-count effects. Full results and
limitations: [`analysis/mc_samples_4_vs_8_s44_s123_20261012.md`](analysis/mc_samples_4_vs_8_s44_s123_20261012.md);
reproduce with [`scripts/run_mc8_recheck_s44_s123.ps1`](scripts/run_mc8_recheck_s44_s123.ps1).

### Discovery-loader preflight wiring audit (2026-10-12)

The student-training preflight previously omitted several objectives that
consume discovery batches (`alpha-discovery`, feature/projection consistency,
rejection-branch feature losses, and reciprocal-point loss). It also did not
classify `pu_corrected` uncertainty feature-margin as requiring a mixed pool.
These checks are now centralized and covered by parameterized tests for every
currently exposed discovery objective and PU mode.

A one-epoch toy smoke test with feature consistency confirmed the loss was
actually active (`discovery_feature_consistency=0.5708`) and the checkpoint was
saved. A second pure-unknown toy smoke test confirmed reciprocal-point loss was
active (`reciprocal=0.9266`). These are loader/loss wiring checks only, not an
AUROC improvement or evidence that known/unknown overlap has been reduced. The
finding and exact verification details are in
[`analysis/discovery_loader_wiring_audit_20261012.md`](analysis/discovery_loader_wiring_audit_20261012.md).

The main unresolved problem remains known/unknown feature and score overlap.
Keep algorithm comparisons protocol-matched, confirm non-zero loss meters and
the intended active config before interpreting a result, and judge AUROC,
FPR95, OSCR, overlap, known performance, and unknown rejection together.

### 2026-10-13: strict candidate-gating check (seed 123)

The first gating run was audited and found not to be a valid comparison: it
disabled pretrained initialization, feature KD, and prototype loss, and used
legacy instead of `nu_corrected` PU risk. Its low validation accuracy cannot
be attributed to gating, so that run is labeled invalid evidence.

A corrected run changed only `discovery_feature_candidate_gating` from the
seed-123 LR-scale-0.1 baseline. Training and detection configs were checked;
the split, initialization, losses, pool, seed, epochs, score, kNN, MC samples,
and disjoint 95%-known-coverage calibration were matched. Results were mixed
in the wrong direction: AUROC `0.8191 -> 0.7985`, AUPR `0.7255 -> 0.6879`,
FPR95 `0.5797 -> 0.6328`, OSCR `0.5602 -> 0.5368`, known accuracy
`62.20% -> 60.62%`, while unknown rejection increased only
`25.10% -> 25.70%`. KS separation fell `0.5000 -> 0.4642`, and unknown-below-
known-q95 overlap increased `70.30% -> 74.93%`. Therefore gating did not
improve the core overlap and remains off by default; do not treat the small
rejection-rate increase as a success. This is a single-seed screening result,
not a universal claim. Full protocol, audit, values, and artifact paths are in
[`analysis/candidate_gating_strict_s123_20261013.md`](analysis/candidate_gating_strict_s123_20261013.md).

Next, stop tuning this gating switch and return to the training supervision
and representation protocol: audit the class-wise source and quality of
unknown candidates, preserve disjoint open validation, and require gains in
ranking/OSCR/overlap without sacrificing known utility before promoting a
method.

Final code verification for this update: `python -m pytest -q` reports
`226 passed, 2 warnings`; `python -m compileall -q train.py novel_discovery
tests` and `git diff --check` pass. The warnings are non-failing PyTorch test
and Windows CPU-core detection warnings.

### 2026-10-13: proxy-contrastive representation pilot

To address the unresolved feature-space overlap, a controlled training-time
pilot added the existing `proxy_contrastive_loss` to the student with
`alpha_proxy=0.05`. This is a normalized-softmax/Proxy-NCA-style objective:
features are pulled toward their labeled classifier-weight proxy and pushed
away from other known proxies, without requiring same-class pairs in a batch.
It changes representation learning, not the detector threshold.

The experiment retrained baseline (`alpha_proxy=0`) and treatment on seeds 123
and 44 from the same teacher, semantic-isolated split, five-epoch recipe,
matched mixed discovery pool, kNN bank, MC=4 detector, and validation-only
95%-known-coverage calibration. Results:

| seed | arm | AUROC | FPR95 | OSCR | known acc | unknown reject | KS | unknown below known-q95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | baseline | 0.8167 | 0.5865 | 0.5560 | 61.88% | 28.58% | 0.4939 | 70.98% |
| 123 | proxy 0.05 | 0.8200 | 0.5907 | 0.5628 | 62.32% | 26.78% | 0.5046 | 72.78% |
| 44 | baseline | 0.8114 | 0.5827 | 0.5276 | 58.87% | 27.40% | 0.4868 | 73.60% |
| 44 | proxy 0.05 | 0.8107 | 0.5873 | 0.5260 | 58.72% | 25.18% | 0.4885 | 73.90% |

The paired bootstrap is retrospective test-label diagnosis only. Seed 123 has
AUROC `+0.0033`, but unknown rejection at exact 95% known coverage decreases
`1.84` percentage points (95% CI `[-3.43, -0.10]` pp). Seed 44 has AUROC
difference `-0.0006` (95% CI `[-0.32, +0.19]` pp) and no reliable rejection
change. Therefore this method is mixed/negative and does not solve the core
known/unknown overlap. Keep the implementation as an ablation with default
`alpha_proxy=0`; do not stack it into the main method. Full protocol and
artifacts are in [`analysis/proxy_representation_pair_20261013.md`](analysis/proxy_representation_pair_20261013.md).

### 2026-10-13: uncertainty-weighted feature-margin recheck

The uncertainty-weighted feature-margin loss was rechecked with newly trained,
matched baseline/treatment pairs on seeds 44 and 123. The baseline was the
audited mixed-pool nnPU recipe (`alpha_discovery_uncertainty_pu=0.1`) with the
margin coefficient set to zero; the treatment added weight `0.05` and cosine
margin `0.2`. Initialization, teacher, semantic-isolated 60/40 split, five
epochs, mixed-pool prior, MC=4 detector, kNN bank, and validation-only 95%
known-coverage calibration were held fixed.

The treatment was verified to be active: its final history contains non-zero
`discovery_uncertainty_feature_margin` and uncertainty-weight means, while the
baseline contains zero. It nevertheless did not consistently improve the
target problem:

| seed | arm | AUROC | FPR95 | OSCR | known acc | unknown reject | KS | unknown below known-q95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 44 | baseline | 0.8120 | 0.5965 | 0.5214 | 57.90% | 29.85% | 0.4843 | 73.93% |
| 44 | margin 0.05 | 0.8083 | 0.5943 | 0.5235 | 58.40% | 27.53% | 0.4869 | 74.68% |
| 123 | baseline | 0.8183 | 0.5850 | 0.5626 | 62.33% | 25.60% | 0.4962 | 72.18% |
| 123 | margin 0.05 | 0.8179 | 0.5850 | 0.5620 | 61.75% | 26.20% | 0.4971 | 70.45% |

The paired bootstrap confirms the mixed outcome: seed 44 AUROC difference is
`-0.00373` (95% CI `[-0.00620,-0.00124]`), while seed 123 difference is
`-0.00045` (95% CI `[-0.00270,0.00177]`). Unknown rejection at exact 95%
known coverage changes by `-0.00918` and `+0.01474`, respectively, with CIs
crossing zero. Therefore the loss is implemented and active, but it is not a
reliable solution to known/unknown overlap. Keep it as an ablation with
default coefficient zero; do not stack it into the main recipe.

This narrows the next direction: another local margin around known prototypes
does not supply a reliable unknown boundary. The next experiment should add a
separately routed boundary signal, such as a disjoint class-balanced open pool
or carefully controlled outlier-exposure/objectosphere supervision, while
keeping known CE/KD and the detector fixed. Candidate purity, active loss
meters, and the matched data protocol must be audited before interpreting
results. Full details and artifacts are in
[`analysis/nnpu_uncertainty_margin_recheck_20261013.md`](analysis/nnpu_uncertainty_margin_recheck_20261013.md).

### 2026-10-13: uniform Outlier Exposure pilot and preprocessing audit

After the uncertainty-margin recheck, a different boundary signal was tested:
an auxiliary CIFAR-10 stream with uniform predictive logits
(`alpha_outlier_uniform=0.1`). The matched baselines were the same mixed-pool
nnPU students on seeds 44 and 123; teacher, semantic-isolated 60/40 split,
pretrained initialization, five epochs, detector, kNN bank, MC=4, and
validation-only 95%-known-coverage calibration were fixed.

The outlier loss was verified to be active (`outlier_uniform` about 4.53 and
4.42 in the final epochs). During the audit, CIFAR-10 images were found to be
using CIFAR-100 normalization under the CIFAR transform profile. This is now
corrected in `build_outlier_dataset`, with regression tests for both CIFAR-10
and CIFAR-100 statistics.

| seed | arm | AUROC | FPR95 | OSCR | known acc | unknown reject | KS | unknown below known-q95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 44 | nnPU baseline | 0.8120 | 0.5965 | 0.5214 | 57.90% | 29.85% | 0.4843 | 73.93% |
| 44 | + uniform OE | 0.8144 | 0.5880 | 0.5498 | 61.50% | 30.05% | 0.4923 | 74.48% |
| 123 | nnPU baseline | 0.8183 | 0.5850 | 0.5626 | 62.33% | 25.60% | 0.4962 | 72.18% |
| 123 | + uniform OE | 0.8191 | 0.5908 | 0.5448 | 60.75% | 28.80% | 0.4976 | 72.15% |

This is a mixed result: uniform OE is correctly implemented and gives a small
ranking signal, but it does not consistently reduce score overlap and it
reduces known utility/OSCR on seed 123. The exact-95%-known-coverage paired
bootstrap changes in unknown rejection are `-0.45` and `+0.07` percentage
points for seeds 44 and 123, respectively, with intervals crossing zero.
Therefore keep uniform OE as an ablation, not as the default or a component to
stack blindly. Full details are in
[`analysis/nnpu_oe_uniform_pilot_s44_s123_20261013.md`](analysis/nnpu_oe_uniform_pilot_s44_s123_20261013.md).

The next boundary experiment should use a genuinely disjoint, visually
diverse outlier source or a class-held-out CIFAR-100 boundary protocol, and
prefer routing the signal to the independent rejection representation. Input
normalization, candidate purity, active loss meters, ranking/overlap, and
known utility must all be audited before promotion.

### 2026-10-13: open-validation checkpoint audit and split-identity correction

本轮先审计 checkpoint 选择协议，没有修改默认算法。历史审计用相对位置判断 calibration 与 open-validation 是否重叠，这对 CIFAR-100 的独立 train/test 数据视图不够可靠。现在新增 dataset_source_keys()，同时记录数据源、train/test 视图和底层样本索引；修正后的真实交集为 0。该修正只影响审计证据，不改变训练和测试。

在 seed=44 的已保存五轮 checkpoint 上，用当前正式检测分数 normalized_entropy_min_class_knn、MC=4 和每轮独立 kNN bank 回放 open validation：

| epoch | open AUROC | FPR95 | known accuracy | KS | unknown below known-q95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.7677 | 0.6700 | 0.5767 | 0.4108 | 0.7485 |
| 2 | 0.7882 | 0.6000 | 0.5767 | 0.4687 | 0.7860 |
| 3 | 0.7872 | 0.5933 | 0.5333 | 0.4460 | 0.7425 |
| 4 | 0.7926 | 0.6067 | 0.5733 | 0.4665 | 0.8205 |
| 5 | 0.7984 | 0.6133 | 0.5500 | 0.4813 | 0.7385 |

### 2026-10-13: rejection-space candidate gating and direct rejection-space scoring audit

This round tested whether the independent rejection representation can reduce
known/unknown score overlap. The paired training protocol fixed seed, CIFAR-100
semantic-isolated 60/40 split, teacher, mixed discovery pool (5400 samples,
known prior 0.2), rejection dimension 128, boundary-loss weight 0.1,
cross-view gating, one epoch, and the 95% known-coverage evaluation rule.

The first intervention changed only the candidate-selection space used by the
rejection kNN boundary loss: `shared` versus `rejection_support`. It was not a
stable improvement:

| seed | candidate space | AUROC | FPR95 | OSCR | unknown reject |
|---:|---|---:|---:|---:|---:|
| 123 | shared | 0.7976 | 0.6275 | 0.5126 | 0.2128 |
| 123 | rejection_support | 0.8002 | 0.6190 | 0.5120 | 0.2303 |
| 44 | shared | 0.7901 | 0.6335 | 0.4946 | 0.2548 |
| 44 | rejection_support | 0.7896 | 0.6422 | 0.4926 | 0.2560 |

The direction changed across seeds, so `rejection_support` remains an ablation
option and is not the default. The change also exposed an important audit
point: changing the training loss space does not help if final detection still
uses the shared feature space.

The second intervention enabled `--knn-feature rejection_features` and used it
directly in the same normalized entropy + classwise kNN detector. On the seed
123 rejection-support checkpoint, direct rejection-space scoring reached
AUROC 0.7892, FPR95 0.6517, OSCR 0.5066, and unknown reject 0.2240, versus
shared-space scoring on the same checkpoint (AUROC 0.8002, FPR95 0.6190,
OSCR 0.5120, unknown reject 0.2303). This does not support promoting the
rejection branch as the detector yet: it currently receives training signals,
but its geometry is not more separable than the shared representation.

Code changes are limited and backward-compatible: KNN-OOD now accepts
`--knn-feature rejection_features`; the default remains `features`.
`py_compile` passed and all 232 tests passed. The next priority is therefore
not another threshold or candidate-ratio sweep. We should audit and improve
the rejection branch's actual known-vs-unknown supervision (including outlier
semantic coverage and class-conditional boundary quality), then re-test with
the same paired multi-seed protocol. These results are diagnostic and should
not be reported as a solved core problem.

因此当前检测器在 open validation 上选择 epoch 5，而历史 known_acc 选择 epoch 2。但这是单 seed、带已知/未知标签的验证诊断，不能证明它解决了核心的特征重叠问题，也不足以修改默认选择规则。当前仍保留 known_acc 默认值；后续若研究 open-set checkpoint selection，必须使用完全一致的最终检测器、预先规定规则，并在新 seed 和独立 test 上复核。

详细记录见 analysis/checkpoint_selection_openval_reaudit_20261013.md。代码检查：191 passed、py_compile 和 git diff --check 通过。
### 2026-10-13: rejection 分支梯度审计与已知 CE 配比实验

为判断独立 rejection 分支是否确实收到有效的未知监督，新增了逐损失梯度审计。小规模 CIFAR-100 smoke 中，rejection 分支梯度范数约为：已知 rejection CE `1.9179`、混合池 uncertainty PU `0.0770`、外部 outlier uncertainty `0.1227`、outlier feature margin `0.0346`。这说明当前已知 CE 更新明显更强，但这只是机制证据，不等于性能提升。

随后进行严格单因素实验：固定语义隔离 CIFAR-100 60/40 划分、teacher、预训练 ResNet、独立 rejection 表示、混合 discovery pool、5 epoch、detector、MC 和 validation-only calibration，只将 `alpha_rejection_known_ce` 从 `1.0` 降为 `0.25`。seed 123 和 seed 44 均完成了对照复核：

| seed | arm | AUROC | FPR95 | OSCR | KS | unknown below known-q95 | known accept | unknown reject |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | CE=1.0 | 0.8865 | 0.4380 | 0.6389 | 0.6202 | 54.30% | 95.08% | 45.30% |
| 123 | CE=0.25 | 0.8895 | 0.4438 | 0.6463 | 0.6277 | 49.43% | 96.72% | 41.88% |
| 44 | CE=1.0 | 0.8995 | 0.3832 | 0.6558 | 0.6453 | 50.45% | 93.90% | 54.50% |
| 44 | CE=0.25 | 0.9033 | 0.3923 | 0.6609 | 0.6528 | 46.48% | 93.78% | 58.38% |

结论：CE 降权在两个 seed 上都使 KS、AUROC、OSCR 和重叠代理略有改善；seed 44 的未知拒绝率也提高，而 seed 123 的未知拒绝率下降。FPR95 在两个 seed 上都略变差，收益幅度较小，且存在指标权衡。因此它是“有希望的损失配比候选”，不是已经解决未知检测的证据；默认配置仍不修改，`0.25` 只保留作消融。实验脚本为 `scripts/run_rejection_ce_balance_pilot.py`，结果目录为 `runs/rejection_ce_balance_s123_ce0p25` 和 `runs/rejection_ce_balance_s44_ce0p25`。

下一步先检查中间权重 `0.5` 是否呈现稳定趋势；若 FPR95 和未知拒绝率仍无改善，则停止继续微调 CE，转向更直接的类条件未知边界监督，并继续使用多 seed、分数重叠和已知 utility 联合判断。任何仅提高未知拒绝率但增加已知误拒绝的方案都不作为改进。
### 2026-10-14: rejection CE 分阶段衰减实验

静态 rejection CE 权重在两个 seed 上显示了轻微但相互冲突的信号，因此新增了可选的分阶段调度：前期保持已知 rejection CE，后期线性衰减，让未知边界监督获得更大的相对影响。新增参数为 `--rejection-known-ce-decay-start-epoch`、`--rejection-known-ce-decay-epochs` 和 `--rejection-known-ce-final-scale`；默认参数下 scale 恒为 `1.0`，不会改变历史流程。

本轮固定 seed=44 的语义隔离 CIFAR-100 60/40 划分、teacher、预训练 student、rejection 表示、混合池、5 epochs、detector、MC 和 validation-only calibration，只将 CE 从 epoch 3 开始由 `1.0` 衰减到 `0.25`，实际记录的 scale 为 `[1.0, 1.0, 0.625, 0.25, 0.25]`：

| 方法 | AUROC | FPR95 | OSCR | KS | 未知低于已知 q95 | 已知接受率 | 未知拒绝率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 静态 CE=1.0 | 0.8995 | 0.3832 | 0.6558 | 0.6453 | 50.45% | 93.90% | 54.50% |
| 静态 CE=0.5 | 0.8966 | 0.3777 | 0.6599 | 0.6357 | 51.53% | 95.10% | 48.18% |
| 静态 CE=0.25 | 0.9033 | 0.3923 | 0.6609 | 0.6528 | 46.48% | 93.78% | 58.38% |
| 分阶段 1.0→0.25 | 0.8969 | 0.3968 | 0.6687 | 0.6475 | 52.63% | 94.98% | 47.45% |

结论：分阶段方案虽然 OSCR 最高，但 FPR95、分数重叠和未知拒绝率均没有改善，不能作为默认方法，也不能说明核心问题已解决。CE 调度方向暂时停止，保留为消融功能。下一步回到更直接的类条件未知边界监督，重点审查 PU rejection-boundary loss 的权重、训练 epoch 和 checkpoint 选择是否共同影响结果。完整记录见 [`analysis/rejection_ce_schedule_pilot_20261014.md`](analysis/rejection_ce_schedule_pilot_20261014.md)。
## 2026-10-15：候选空间实验的 wiring 审计与修正

上一轮 `shared` 与 `rejection_support` 候选空间实验不能作为有效算法结论。审计训练历史发现，两组每个 epoch 的 `feature_candidate_ratio` 都是 `0.0`，但 `discovery_rejection_boundary` 非零。原因是实验脚本使用了 `alpha_discovery_rejection_boundary`（完整混合池的 PU boundary loss），而当前代码中真正读取 `feature_candidate_masks` 的 rejection 边界路径是 `alpha_discovery_rejection_knn_boundary`。因此旧实验只是保留作 wiring 审计记录，不能解释为候选空间方法有效或无效。

已修正 [`scripts/run_rejection_boundary_candidate_space_control.py`](scripts/run_rejection_boundary_candidate_space_control.py)：现在只启用 `alpha_discovery_rejection_knn_boundary=0.1`，关闭独立的 PU boundary 项，并使用新的输出目录重新实验。下一次比较只改变 `discovery_feature_candidate_space`，固定 seed=44、4 epochs、CIFAR-100 semantic-isolated 60/40、matched mixed pool、known prior=0.2、teacher/student 初始化、检测器和 95% known-coverage 校准。实验必须先确认两组的 `feature_candidate_ratio` 与 `discovery_rejection_knn_boundary` 均非零，再比较 AUROC、FPR95、OSCR、已知效用、固定覆盖率未知拒绝率、KS 和 score overlap。

详细审计见 [`analysis/candidate_space_wiring_audit_20261015.md`](analysis/candidate_space_wiring_audit_20261015.md)。本次检查说明：参数出现在配置文件中不等于对应损失实际生效；以后每个算法实验都要同时核对训练历史中的激活比例和损失值。

### 2026-10-16：修正 wiring 后的候选空间配对实验

修正后的候选空间实验已经完成。两组均实际启用了 `alpha_discovery_rejection_knn_boundary=0.1`，训练历史中的 `feature_candidate_ratio` 和 `discovery_rejection_knn_boundary` 均非零，因此这次结果可以用于判断 `shared` 与 `rejection_support`，不再属于上一轮的 wiring 失效实验。

| 候选空间 | AUROC | FPR95 | OSCR | known accept | unknown reject | KS | unknown below known-q95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| shared | 0.89230 | 0.38083 | 0.65442 | 95.83% | 42.38% | 0.62450 | 52.75% |
| rejection_support | 0.89266 | 0.39783 | 0.65808 | 94.57% | 47.80% | 0.63875 | 55.13% |

`rejection_support` 虽然让报告阈值下未知拒绝率提高 5.43 个百分点，但已知接受率下降 1.27 个百分点；AUROC 几乎不变，FPR95 和 score overlap 变差，均值/中位数分数间隔缩小。因此不能把它判定为解决核心重叠问题的有效方法，也不修改默认候选空间。它最多说明 rejection-support 具有少量互补信号，后续若继续研究必须先固定独立验证集上的 support 半径和边界权重，并用多 seed 同时检查排序、分离、已知 utility 和固定覆盖率拒绝率。

本轮结论和完整协议见 [`analysis/rejection_knn_candidate_space_control_s44_20261016.md`](analysis/rejection_knn_candidate_space_control_s44_20261016.md)。当前候选空间方向暂时停止继续扫描，下一步回到共享 encoder 的类条件局部 support 几何，重点分析 semantic-near unknown 的误接收和 support 半径是否可靠。

### 2026-10-16：共享边界候选与分类共识硬交集复核

为降低 `shared_boundary` 候选中的已知样本污染，新增 `shared_boundary_consensus`：将共享特征的 kNN support 边界候选与原分类器/不确定性共识候选取交集。先在 seed=44 看机制，再使用相同 soft-NNPU 配置模板对 seed=123 做配对复核；三组均固定四轮训练、混合池（5400，已知先验 0.2）、共享特征 kNN 边界损失权重 0.1、跨视图 gate、检测器和仅用 validation 校准到 95% 已知覆盖率。不同 seed=123 的既有 hard-NNPU 记录没有混入对照。

| seed | 候选空间 | AUROC | FPR95 | OSCR | 已知接受率 | 未知拒绝率 | KS | 未知低于 known-q95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 44 | shared | 0.89753 | 0.35867 | 0.66378 | 95.67% | 43.78% | 0.64692 | 52.20% |
| 44 | shared_boundary | 0.89467 | 0.39017 | 0.65287 | 94.18% | 51.73% | 0.63750 | 52.85% |
| 44 | shared_boundary_consensus | 0.90097 | 0.35417 | 0.65818 | 96.02% | 44.28% | 0.64617 | 50.13% |
| 123 | shared | 0.89468 | 0.40833 | 0.64609 | 95.97% | 45.03% | 0.63333 | 48.78% |
| 123 | shared_boundary | 0.89328 | 0.40050 | 0.64483 | 94.88% | 48.45% | 0.63208 | 52.80% |
| 123 | shared_boundary_consensus | 0.89436 | 0.40983 | 0.64446 | 95.80% | 46.53% | 0.63817 | 50.70% |

机制审计确认 gate 和边界损失均实际生效，但硬交集把未知候选召回从约 11%–15% 压到约 1%–5%；已知污染降低是以候选覆盖骤降为代价。seed=44 的检测改善幅度小且 OSCR 下降，seed=123 没有复现，AUROC/FPR95/OSCR 均略差于 `shared`。结论是存在 precision–recall 取舍，不能认定解决了已知/未知重叠，也不改默认配置、不再针对该硬交集扫参数。

后续候选仅作为研究假设：保留边界候选覆盖，把分类共识作为有界软损失权重，而非硬删除样本；先用预先规定的 validation 规则与 boundary-only 对照。如果排序和重叠没有跨 seed 稳定改善，停止继续叠加候选 gate，转向 support 半径校准、开放集数据划分审计及拒识分支的直接监督。完整协议、机制指标和结果见 [`analysis/shared_boundary_consensus_recheck_20261016.md`](analysis/shared_boundary_consensus_recheck_20261016.md)。

### 2026-10-16：共享边界候选的软共识权重复核

针对硬共识交集会丢失未知候选的问题，新增 `shared_boundary_soft_consensus`。它保留 `shared_boundary` 的全部跨视图候选，只用原分类器/不确定性风险产生 `[0.25, 1.0]` 的无标签损失权重。seed=44 和 seed=123 均使用同一 soft-NNPU 配置模板，固定四轮训练、混合池、kNN 边界损失、检测器和 validation-only 95% known-coverage 校准；每个 seed 只比较 boundary-only 与 soft-consensus。

| seed | 方法 | AUROC | FPR95 | OSCR | 已知接受率 | 未知拒绝率 | KS | 未知低于 known-q95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 44 | boundary-only | 0.89964 | 0.35500 | 0.66006 | 95.48% | 46.50% | 0.64492 | 51.20% |
| 44 | soft consensus | **0.90447** | **0.34467** | **0.66855** | 94.70% | **50.95%** | **0.66783** | **50.30%** |
| 123 | boundary-only | 0.89788 | 0.39217 | 0.64421 | 95.98% | 44.95% | 0.64058 | 50.43% |
| 123 | soft consensus | **0.89914** | 0.39967 | 0.64257 | **96.68%** | 42.58% | 0.63775 | **48.00%** |

两组都确认权重路径实际生效，候选覆盖和未知召回没有像硬交集一样塌缩。但 seed=44 的多指标改善没有在 seed=123 完整复现：后者 FPR95、OSCR、KS 和固定工作点未知拒绝率变差。结论是该方向比硬交集更有价值，但仍属于有条件消融，不改默认配置，也不根据测试集挑选权重。

结果进一步提示：当前 kNN 边界损失可能把边界内的已知污染样本与真正位于已知 support 外的未知样本用相同的“向外推离”目标处理。下一步优先审计该损失的 signed/class-conditional 几何，并在保持相同多 seed 协议下尝试区分 support 内外样本，而不是继续堆叠候选 gate。完整记录见 [`analysis/shared_boundary_soft_consensus_recheck_20261016.md`](analysis/shared_boundary_soft_consensus_recheck_20261016.md)。
## 2026-10-17：soft-NNPU 对照与 kNN 边界梯度审计

本轮完成了 `pure`、`hard_nnpu`、`soft_nnpu` 三臂严格配对实验，并补齐了共享 encoder 上 `discovery_knn_boundary` 的实际梯度审计。实验固定 seed=44、CIFAR-100 semantic-isolated 60/40、mixed pool=5400、known prior=0.2、4 epochs、shared-boundary 候选空间、边界权重 0.1、同一检测器和 validation-only 95% known-coverage 校准，只有 `discovery_knn_risk` 不同。

| risk | AUROC | FPR95 | OSCR | 已知接受率 | 未知拒绝率 | KS | unknown below known-q95 | 接受后已知准确率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pure | 0.900255 | 0.364833 | 0.658245 | 94.517% | 50.800% | 0.653000 | 51.775% | 72.827% |
| hard_nnpu | 0.903022 | 0.347667 | 0.665626 | 94.367% | 52.500% | 0.656583 | 50.625% | 73.384% |
| soft_nnpu | 0.898656 | 0.360833 | 0.659466 | 93.917% | 52.125% | 0.642917 | 52.025% | 73.203% |

soft-NNPU 相对 pure 的 AUROC、KS 和 overlap 变差，因此不能判定为改善；未知拒绝率上升不能单独作为成功证据。hard-NNPU 在本次运行中较好，但与此前严格复核的负向结果不一致，暂不设为默认。

此前梯度诊断漏记了共享 encoder 上的 kNN 边界项，本轮已修正。1 epoch 机制审计显示 pure/soft 的 kNN encoder 梯度范数分别为 1.557455/1.046466，与 CE 的余弦分别为 -0.035698/-0.047631；因此该项确实参与表征学习，但方向略有冲突，soft 形式的有效梯度更弱。这解释了为什么 soft-NNPU 没有带来稳定开放集收益，但不能代替正式性能实验。

详细记录见 [`analysis/knn_pu_boundary_soft_and_gradient_audit_s44_20261017.md`](analysis/knn_pu_boundary_soft_and_gradient_audit_s44_20261017.md)，机制审计脚本为 [`scripts/run_knn_gradient_audit.py`](scripts/run_knn_gradient_audit.py)。下一步停止继续调 NNPU softness，先按未知细类、预测已知类和 kNN 支持距离审计误接收样本，再据此设计类条件表征改动。

### 2026-10-17：inter-class virtual outlier 配对策略审计

针对已知/未知分数重叠，本轮新增可选的 inter-class virtual outlier。第一轮随机选择异类样本进行特征插值，未知拒绝率提高但整体排序指标混合；随后新增 `nearest_class`，按当前 batch 中最近的异类原型选择配对，试图生成更贴近已知决策边界的虚拟未知样本。新增参数为 `--vos-interclass-pairing {random,nearest_class}`，默认仍为 `random`，不会改变历史流程。

严格匹配的 seed=44、4 epoch 对照结果如下：

| 配对策略 | AUROC | FPR95 | OSCR | 已知接受率 | 未知拒绝率 | KS |
|---|---:|---:|---:|---:|---:|---:|
| random 对照 | 0.90062 | 0.36033 | 0.66090 | 93.02% | **58.73%** | 0.65250 |
| nearest_class | **0.90495** | **0.33167** | 0.65971 | **94.90%** | 48.78% | **0.66383** |

`nearest_class` 改善了 AUROC、FPR95 和 KS，说明分数排序和整体分布分离有潜在收益；但 validation-only 校准后的测试未知拒绝率下降 9.95 个百分点，OSCR 略降，已知全体准确率也从 0.7155 降到 0.7067。因此它只能作为“有潜力但未稳定”的可选消融，不能替换默认策略，也不能仅凭未知拒绝率上升或下降判断成败。随机配对的上一轮结果与本轮绝对数值不完全一致，原因是当前 GPU 训练存在非位级确定性，已在分析文档中分开记录，不能跨运行直接合并。

完整记录见 [`analysis/interclass_virtual_outlier_pairing_s44_20261017.md`](analysis/interclass_virtual_outlier_pairing_s44_20261017.md)。下一步不继续盲目扫描配对策略，而是验证“只对高风险已知边界样本施加虚拟未知损失”是否能减少工作点误拒绝；同时保留已知分类约束，并继续使用 AUROC、FPR95、OSCR、已知接受率、未知拒绝率和 score overlap 联合判断。

### 2026-10-18: boundary candidate and class-conditional support feature recheck

本轮仍针对核心问题“已知/未知特征与分数分布重叠”，没有调整最终阈值。首先新增 `confident_boundary` 候选模式，将预测已知类的支持边界接近度与 Softmax 置信度组合，用于检验高置信 semantic-near unknown 是否能改善 rejector 训练。随后新增 `predicted_support_augmented` rejector 特征，把“最近已知支持类距离”和“模型预测类支持距离”同时输入 nnPU rejector。

代码验证：`python -m pytest -q tests/test_core_behaviors.py` 为 `218 passed`，`compileall` 通过。新增可复现实验脚本：`scripts/run_confident_boundary_compare.py` 和 `scripts/run_predicted_support_feature_compare.py`。

seed=44 的严格 paired 条件：CIFAR-100 semantic-isolated 60/40、固定 student checkpoint、mixed pool=5400、known prior=0.2、`nu_corrected` nnPU、support-augmented 主配置、MC=4、full-support、validation-only 95% known-coverage；候选实验只改变 `rejector_candidate_mode`，特征实验只改变 `rejector_feature_mode`。

| 实验 | AUROC | FPR95 | OSCR | 已知接受率 | 未知拒绝率 |
|---|---:|---:|---:|---:|---:|
| `prototype_boundary` | 0.9089 | 0.3323 | 0.6679 | 95.78% | 48.03% |
| `confident_boundary` | 0.8832 | 0.4352 | 0.6442 | 94.82% | 45.80% |
| `support_augmented` | 0.8983 | 0.3623 | 0.6617 | 94.88% | 48.33% |
| `predicted_support_augmented` | 0.8988 | 0.3682 | 0.6614 | 95.35% | 47.78% |

结论：`confident_boundary` 明显无效，不能继续采用，也不能用“已知拒绝率变化”解释为成功；简单加入置信度反而削弱了边界候选的训练方向。`predicted_support_augmented` 与现有特征几乎相当但 FPR95、OSCR 和未知拒绝率略差，当前不足以替换 `support_augmented`。这两轮结果共同说明，继续堆叠后处理候选或单个支持距离特征不是解决重叠问题的主要方向。

后续应优先转向训练阶段的类条件表示/拒绝分支监督：针对被错误接受的 semantic-near unknown 建立稳定的类条件困难负样本，同时保留已知类分类约束，避免把边界内已知样本一起推向 unknown。下一轮应先做小规模梯度与数据协议审计，再进行单因素训练实验；不能只依据未知拒绝率上升判断有效。

### 2026-10-18：候选门控后 PU 先验与风险分布审计

进一步沿训练调用链检查后发现，当前 PU-kNN 边界项可能存在先验/分布不匹配：训练先用共享边界和跨视图一致性从混合池筛出候选，再将筛选后的风险当作 `R_mix`；但修正项仍使用原始 `known_prior=0.2` 和未经相同门控筛选的整批已知样本风险。这不满足 PU 风险校正中“先验与混合风险来自同一采样分布”的前提。训练历史的标签诊断字段显示，候选中已知占比由第 1 轮的约 17% 降到后几轮约 6–9%；hard-NNPU 边界损失同期从 0.01095 降至 0.00051。该现象与过度扣除、非负截断后梯度趋近于零相符，但尚不能单独证明它解释了所有检测错误。

| 风险臂 | epoch 1 候选已知占比 | epoch 4 候选已知占比 | epoch 4 kNN loss |
|---|---:|---:|---:|
| pure | 16.67% | 6.19% | 0.02309 |
| hard-NNPU | 17.21% | 7.11% | 0.00051 |

以上候选占比仅来自训练期间的事后诊断标签，**没有进入 loss、模型选择或校准**，不能用它直接调先验。下一步应实现不使用混合池标签的 selection-conditional PU 修正：用同一候选规则测量已知训练样本的选中率与混合池整体选中率，推导筛选后的有效先验，并只用同一规则筛出的已知训练特征计算已知风险。先做已知真先验的合成混合测试，检查风险和梯度；通过后再做 seed=44、4 epoch 的 pure 对 corrected-PU 配对训练，固定其余配置和 validation-only 95% 已知覆盖率校准，联合比较 AUROC、FPR95、OSCR、KS、重叠和已知效用。

详细机制、日志数值、假设边界和参考文献见 [`analysis/selected_candidate_pu_prior_audit_s44_20261018.md`](analysis/selected_candidate_pu_prior_audit_s44_20261018.md)。这轮是机制审计和实验设计，不是性能提升结论；未改默认算法配置，也未上传仓库。
### 2026-10-18：无门控 mixed-pool PU-kNN 双 seed 配对复核

为验证“hard-NNPU 本身是否能改善已知/未知分布重叠”，本轮去掉两组实验中的候选门控和跨视图 gate，避免把候选选择造成的先验错配与 NNPU 风险形式混在一起。固定 CIFAR-100 semantic-isolated 60/40、mixed pool=5400、known prior=0.2、4 epochs、`alpha_discovery_knn_boundary=0.1`、其它 rejection-boundary 项为 0、同一检测器和 validation-only 95% known-coverage 校准；唯一改变 `discovery_knn_risk`，比较显式污染池基线 `naive_mixed` 与 `hard_nnpu`，并在 seed=44、123 各做一轮。

| seed | 风险形式 | AUROC | FPR95 | OSCR | 已知接受率 | 未知拒绝率 |
|---:|---|---:|---:|---:|---:|---:|
| 44 | `naive_mixed` | 0.902374 | 0.346500 | 0.659630 | 94.767% | 49.500% |
| 44 | `hard_nnpu` | 0.901560 | 0.347667 | 0.663014 | 94.617% | 50.700% |
| 123 | `naive_mixed` | 0.889590 | 0.420167 | 0.638726 | 95.500% | 45.475% |
| 123 | `hard_nnpu` | 0.889536 | 0.412500 | 0.634528 | 96.867% | 40.450% |

结果没有跨 seed 一致方向：seed=44 的 hard-NNPU 仅使未知拒绝率提高 1.20 个百分点，seed=123 反而下降 5.03 个百分点；AUROC 几乎不变，OSCR 一升一降。因此 hard-NNPU 没有证据能够稳定拉开已知/未知分数分布，不能设为默认，也不能继续只调先验或阈值。该结果同时说明：此前发现的“候选门控后 PU 先验/风险分布可能错配”仍是 gated 路径的真实代码问题，但不是全部重叠问题的唯一原因；即使取消门控，NNPU 本身也没有解决核心问题。

本轮完整记录见 [`analysis/knn_pu_ungated_control_s44_s123_20261018.md`](analysis/knn_pu_ungated_control_s44_s123_20261018.md)，汇总文件为 [`analysis/knn_pu_ungated_s44_e4_summary.json`](analysis/knn_pu_ungated_s44_e4_summary.json) 和 [`analysis/knn_pu_ungated_s123_e4_summary.json`](analysis/knn_pu_ungated_s123_e4_summary.json)。下一步不再扫描 NNPU softness/先验；先用合成混合数据验证 selection-conditional PU 推导和梯度，再决定是否修正 gated 路径。若推导验证通过，只做一次严格 gated paired control；若不通过，停止这条 PU 分支，转向直接的类条件表示/拒识分支监督。

随后增加了独立的 selection-conditional 先验计算函数和合成审计，但没有接入默认训练。公式为 `pi_selected = pi * s_known / s_mixed`，其中 `s_known` 是已知训练样本在同一候选规则下的选中率，`s_mixed` 是混合池整体选中率。两种不同选择率的 500,000 样本合成测试中，修正后的 PU 风险恢复未知风险的最大绝对误差约为 `1.98e-7`；这只证明公式和实现正确，不证明真实 gate 的选择率估计可靠。函数、脚本和报告分别见 [`novel_discovery/losses.py`](novel_discovery/losses.py)、[`scripts/audit_selection_conditional_pu.py`](scripts/audit_selection_conditional_pu.py) 和 [`analysis/selection_conditional_pu_synthetic_audit_20261018.json`](analysis/selection_conditional_pu_synthetic_audit_20261018.json)。在真实训练前仍需用同一无标签规则测量两种选择率、记录采样噪声和无效估计，并做一次 gated 配对实验；本轮不改变默认风险形式。
### 2026-10-18：selection-conditional PU-kNN 已严格接入（尚无性能结论）

针对前面发现的候选 gate 与 PU-kNN 先验可能不匹配的问题，已完成一个
**默认关闭的可选消融**：`--discovery-knn-selection-conditional`。它使用

```text
pi_selected = pi * P(S|known) / P(S|mixed)
```

估计候选筛选后的有效已知先验，并且只在以下条件同时满足时允许使用：混合
发现池、候选 gate、`hard_nnpu/soft_nnpu` 风险、kNN 边界损失开启，且关闭
cross-view 交集。为避免把不同采样事件混在一起，该模式将混合池和已知 batch
都按同一个单视图候选规则计算选择率；如果使用 EMA/teacher selection model，
两边的选择率都使用该模型，而实际 PU 已知风险仍取 student 的对应特征。双视图
交集模式会直接报错，不做静默近似。

这次修改的作用是修正候选筛选后先验与已知风险参考分布的不一致，**不是**直接
拉开已知/未知特征分布的算法证明。当前只完成代码和公式验证，没有把它设为默认，
也没有宣称性能提升：

- 核心测试 `221 passed`，`compileall` 和 `git diff --check` 通过；
- 合成 selection-conditional PU 审计最大风险恢复误差约 `1.98e-7`；
- 默认开关为关闭，显式参数可正确解析；
- 训练日志新增 `discovery_knn_mixed_selection_rate`、
  `discovery_knn_known_selection_rate` 和 `discovery_knn_effective_prior`，用于
  检查真实 gate 的采样率和有效先验，而不是只看未知拒绝率。

详细记录见 [`analysis/selection_conditional_pu_wiring_20261018.md`](analysis/selection_conditional_pu_wiring_20261018.md)。下一步只做一次严格配对的
gated 实验：固定 seed、数据划分、混合池、训练轮数、检测器和 validation-only
95% 已知覆盖率校准，只切换该开关，并联合比较 AUROC、FPR95、OSCR、KS、score
overlap、已知接受率、接受后已知准确率、未知拒绝率及三个选择率诊断。若只改变
工作点或跨 seed 不稳定，就停止继续调先验，转向直接的类条件拒绝分支监督。

### 2026-10-18：边界 VOS 上增加 rejection BCE 的配对实验暂停记录

本轮计划检验：在已有较有希望的 nearest-class、困难边界 25% VOS 上，增加小权重
`alpha_vos_rejection=0.05` 是否能进一步拉开已知/未知分数分布。复核后发现，最初运行的
control 是完全无 VOS，而 treatment 同时加入 VOS 和 rejection BCE，属于多因素同时变化，
因此其 `discovery_report.json` 数字只能作为探索观察，不能归因于 rejection BCE，也不能
与 boundary-only 实验作严格配对结论。

已修正 [`scripts/run_interclass_vos_control.py`](scripts/run_interclass_vos_control.py)，允许
明确配置 control 侧 VOS。修正后的设计两组都固定 seed=44、相同数据划分和训练配置、4 epochs、
nearest-class VOS、`alpha_vos=0.01`、boundary fraction=0.25、同一检测器和 validation-only
95% 已知覆盖率校准；唯一差异为 rejection BCE 权重（control=0，treatment=0.05）。

按“快速结束”要求，配对重跑已在 control 训练中途停止，treatment 尚未启动，没有完整模型或
有效比较指标。**目前不能判断 rejection BCE 有效或无效。**不得引用中间 loss 作为性能证据。
核心测试为 `222 passed`，`git diff --check` 通过。下午继续前先核查是否有完整 control checkpoint；
随后运行：

```powershell
python scripts/run_interclass_vos_control.py --tag vos_boundary_rejection_matched_s44_e4 --pairing-strategy nearest_class --boundary-fraction 0.25 --alpha-vos-rejection 0.05 --control-alpha-vos 0.01 --control-pairing-strategy nearest_class --control-boundary-fraction 0.25 --control-alpha-vos-rejection 0.0
```

完成后联合分析 AUROC、FPR95、OSCR、KS、score overlap、已知接受率、接受后已知准确率和未知拒绝率；
只有拒绝率上升不算核心问题改善。过程记录见
[`analysis/vos_boundary_rejection_matched_s44_e4_status_20261018.md`](analysis/vos_boundary_rejection_matched_s44_e4_status_20261018.md)。本轮未上传 GitHub。
### 2026-10-18：source-gradient VOS 逐未知类别误接收复核

对已有 seed=44/123、detached/source-gradient 四组 discovery detail 做了
**仅事后描述**的逐未知类别误接收审计；没有使用测试标签训练、选模型、校准或调阈值。
每组 4,000 个未知测试样本（每个未知类 100 个）。source-gradient 的未知误接收率
在 seed=123 从 58.33% 降到 53.20%，但在 seed=44 从 50.53% 升到 53.80%，仍是
seed-sensitive。四组 top-10 高误接收类别的交集只有 `bus`；类别的排名和梯度开关
带来的变化方向大多不稳定。因此不能用这些测试错误直接构造类条件训练权重，也没有
证据认为少数固定类别足以解释整体重叠。source-gradient VOS 的第二 seed 复核已完成，
不再继续该分支或扫权重，默认保留 detached 路径。

后续应只从训练/验证可合法获得的信号出发，先审计类条件边界风险代理及已知污染，再做
单因素配对实验；同时检查 mixed-pool 协议和表示学习目标，不把单点未知拒绝率作为成功标准。
完整方法、数字、限制和下一步见
[`analysis/vos_source_gradient_classwise_audit_s44_s123_20261018.md`](analysis/vos_source_gradient_classwise_audit_s44_s123_20261018.md)
及 [`analysis/vos_source_gradient_compare_s123_e4_20261018.md`](analysis/vos_source_gradient_compare_s123_e4_20261018.md)。
### 2026-10-18：匹配 teacher 的 split-protocol 对比（seed=44/123 方向复现）

前一轮因 semantic-isolated teacher 的类别头/原型不能用于 semantic-hard 而中止；之后使用同一 teacher 配置模板分别重训两种 split 的 teacher，再用同一 student/discovery 配置完成配对对比。现在已在 seed=44 和 seed=123 完成两次配对复核；各次均固定 teacher 5 epochs、student 4 epochs、同一 detached-VOS 配方和 discovery/calibration 配置；两个协议均为 60 known / 40 novel，已知验证集单独校准到约 95% coverage。每个 seed 内除类别 split 及其匹配 teacher 训练外，student/discovery 配置一致，测试标签未用于训练、选择或校准。估计量是“类别划分 + 对应 teacher 重训”的端到端协议敏感性，不是只改变 split 的因果效应；短训练预算且只有两个 seed，仍属于诊断性复核，不是最终性能结论。

| CIFAR-100 划分 | AUROC | FPR95 | OSCR | 测试已知接受率 | 测试未知拒绝率 | 接受后已知准确率 | score KS | unknown below known-q95 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| semantic-isolated（coarse superclass 不共享） | 0.89186 | 0.38717 | 0.65143 | 94.20% | 49.45% | 72.61% | 0.63267 | 53.53% |
| semantic-hard（20 个 superclass 全共享） | 0.79051 | 0.60667 | 0.59306 | 93.68% | 28.23% | 72.14% | 0.43450 | 75.25% |

seed=123 的独立配对复核（相同 5/4 epoch 预算及验证集 95% known-coverage 校准）：

| CIFAR-100 划分 | AUROC | FPR95 | OSCR | 测试已知接受率 | 测试未知拒绝率 | 接受后已知准确率 | score KS | unknown below known-q95 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| semantic-isolated（coarse superclass 不共享） | 0.89048 | 0.40883 | 0.63757 | 95.62% | 43.75% | 70.82% | 0.62350 | 51.98% |
| semantic-hard（20 个 superclass 全共享） | 0.78111 | 0.64683 | 0.59025 | 95.00% | 24.53% | 72.68% | 0.41825 | 75.45% |

两次配对中，semantic-hard 相较 isolated 的方向完全一致：AUROC 分别低 `0.10135/0.10937`，FPR95 高 `0.21950/0.23800`，KS 低 `0.19817/0.20525`，unknown-below-known-q95 overlap 高 `21.72/23.47` 个百分点，95% 校准工作点的未知拒绝率低 `21.22/19.23` 个百分点；接受后已知准确率没有相应下降（hard 分别高 `0.47/1.86` 个百分点）。这支持一个重要诊断：**数据协议及 known/novel 类别的语义关系显著影响检测难度，semantic-isolated 的结果不能代表同 superclass 内细粒度未知类别的难题。** 两 seed 的方向复现增加了该诊断的可信度，但这不是算法改进，也不能据短预算、两个 seed 推断普遍效应大小。

对 semantic-hard 的 discovery detail 做了额外事后错误分解（仅描述测试集，不参与训练/校准/选择）：4,000 个未知测试样本中有 2,871 个被接受为已知；40 个未知细类全部都出现误接收，各类误接收率范围为 39%–94%，中位数 74.5%。最高的几个细类也不是唯一来源，误接收分散到所有 novel 类别。误接收样本的已知预测较集中但非单一类别：排名前 5 的已知预测类覆盖 24.2%，前 10 覆盖 38.3%。因此这不只是少数异常未知类或一个已知输出类造成的局部问题，而是 semantic-hard 下较普遍的分数/表示重叠，同时存在细类难度差异。该汇总使用保存的 `raw_labels` 还原未知细类；报告中的 open-set `true_label=-1` 是未知标签占位符，不能拿它做细类分析。

seed=123 的事后细类审计与 seed=44 一致（只作测试集描述，不参与训练/校准/选择）：semantic-isolated 误接收 `2,250/4,000` 个未知样本（56.25%），semantic-hard 为 `3,019/4,000`（75.48%）；两种 split 的 40 个未知细类都全部出现误接收。各类误接收率中位数分别为 60.0% 和 75.5%（范围 17%–86%、49%–94%）。hard 条件下 top-5/top-10 已知预测类只覆盖 19.4%/33.8% 的误接收，进一步说明不是少数未知细类或单一已知输出类导致。seed=44 的对应 hard 审计为 40/40 类均有误接收，中位数 74.5%，top-5/top-10 覆盖 24.2%/38.3%。

核心问题仍未解决：semantic-hard 的已知与未知分数分布重叠明显，且在已知验证集约 95% 覆盖校准下测试未知拒绝率只有 28.23%（seed=44）/24.53%（seed=123），同时接受后已知准确率并未显著下降。因此差异主要不能用“阈值设得保守”解释；更符合细粒度语义相近未知类广泛进入已知类决策区域的表现。不要据此调阈值，也不要用测试标签选择损失。后续优先级：

1. 若需把协议效应量化为稳健估计，增加预先固定的 seed 配对复核，保持各臂训练配方、校准和预算匹配；否则不要继续重复同一类短跑。
2. 在不使用测试标签训练或选择的前提下，对 semantic-hard 的 validation/train 误差代理按未知细类语义、最近已知类及特征邻域做分析，检查模型是否学到局部近邻混淆；测试细类审计仅作最后的描述性确认。
3. 再在 semantic-hard 上开展一次只变一个因素的类条件拒识/表示间隔实验，监督信号必须来自训练集或合规验证集；按 AUROC、FPR95、OSCR、score overlap、95% known coverage 下未知拒绝率及已知效用联合评估，semantic-isolated 保留为较容易协议对照。

结果文件：seed=44 [`analysis/split_protocol_matched_teacher_s44_e4_20261018_v2_summary.json`](analysis/split_protocol_matched_teacher_s44_e4_20261018_v2_summary.json)，seed=123 [`analysis/split_protocol_matched_teacher_s123_e4_20261018_summary.json`](analysis/split_protocol_matched_teacher_s123_e4_20261018_summary.json)；配对运行脚本：[`scripts/run_split_protocol_sensitivity.py`](scripts/run_split_protocol_sensitivity.py)。可复跑的只读细类误接收审计为 [`scripts/analyze_split_protocol_errors.py`](scripts/analyze_split_protocol_errors.py)，seed=44/123 的四个输出分别为 [`analysis/split_protocol_semantic_isolated_errors_s44_20261018.json`](analysis/split_protocol_semantic_isolated_errors_s44_20261018.json)、[`analysis/split_protocol_semantic_hard_errors_s44_20261018.json`](analysis/split_protocol_semantic_hard_errors_s44_20261018.json)、[`analysis/split_protocol_semantic_isolated_errors_s123_20261018.json`](analysis/split_protocol_semantic_isolated_errors_s123_20261018.json)、[`analysis/split_protocol_semantic_hard_errors_s123_20261018.json`](analysis/split_protocol_semantic_hard_errors_s123_20261018.json)。运行示例：`python scripts/analyze_split_protocol_errors.py --detail runs/<protocol>_discover/discovery_detail.json --split splits_cifar100_protocols/cifar100_60_40_<protocol>.json --output analysis/<report>.json`。上一轮中止原因与无效的部分结果仍记录在 [`analysis/split_protocol_sensitivity_pilot_20261018.md`](analysis/split_protocol_sensitivity_pilot_20261018.md)，不得将其单臂结果当作协议比较证据。

### semantic-hard 冻结 checkpoint 分数族对照（seed=123；无 score 替代方案胜出）

为避免因 semantic-isolated 上的历史结果外推到 hard split，本轮固定 seed=123 semantic-hard student checkpoint、mixed discovery pool、测试样本与聚类关闭配置，只比较现有分数族，并用已知验证集约 95% coverage 校准；没有重训 student，测试标签没有用于校准或选分数。当前 `feature_rejector` 最好：AUROC/FPR95/OSCR=`0.78111/0.64683/0.59025`，未知拒绝率 `24.53%`；normalized entropy + prototype=`0.69923/0.73950/0.55937`、拒绝率 `9.80%`；normalized entropy + diagonal Mahalanobis=`0.67227/0.82433/0.52084`、拒绝率 `12.25%`；known support=`0.69601/0.75100/0.54051`、拒绝率 `12.90%`。因此没有证据支持替换或叠加这些弱分数，继续堆分数/调阈值不是当前优先方向；但该冻结 checkpoint 分数审计也不是训练算法比较，不能据此断言其他表示或训练目标无效。semantic-hard 未知拒绝仍很低，下一步应回到训练/验证可获得信号上的局部表示几何与污染控制，再做单因素训练配对实验。完整条件和局限见 [`analysis/semantic_hard_score_audit_s123_20261018.md`](analysis/semantic_hard_score_audit_s123_20261018.md)，运行入口为 [`scripts/run_score_mode_audit.py`](scripts/run_score_mode_audit.py)，各模式结果位于 `runs/split_hard_score_audit_s123_20261018_<score>/`。

随后在 semantic-hard 的 seed=44/123 冻结 checkpoint 上探索拒识器候选加权：保持完整 mixed pool、候选比例 25%、候选权重 2、known prior 和其余设置相同，只比较 `weighted_boundary` 下 `prototype_boundary` 与 `confident_boundary`。prototype-boundary 两 seed 的 AUROC/FPR95/OSCR 均优于 confident-boundary；相对原全池 rejector，AUROC 分别 `+0.00541/+0.01290`、FPR95 分别改善 `0.01567/0.03467`，但 validation-calibrated 未知拒绝率变化为 `+2.35/-0.33` 个百分点，未稳定提升。这只是冻结 checkpoint 的 detector/rejector 对照，不是 student 表示改进或核心问题解决。

方法学自检发现：最初只覆写 `rejector_candidate_mode` 的运行保留了 `rejector_pool_selection=all`，因此候选选择逻辑完全未执行，结果与基线相同；该次运行是惰性参数 wiring 检查，不是有效实验。之后才用两臂均为 `weighted_boundary` 的配对对照。另需降低证据等级：候选池审计报告了候选后 pool-label 污染率/召回，虽然这些标签没有进入模型或自动筛选，但其结果影响了人工优先测试哪一种候选规则；因此本次测试比较属于探索性、存在人工选择路径，不能作为未触碰测试集的确认结论，也不能据此改默认。配对 bootstrap 的 AUROC 区间仅描述固定 checkpoint 的测试样本不确定性；其未知拒绝区间使用每次重采样的测试已知标签重算 95% 阈值，不代表 validation-only 部署校准，不据此声称拒识收益。

结论：prototype-boundary 是较值得独立复核的方向，但目前最多是排序指标的弱正向信号，未知拒绝率没有跨 seed 稳定改善。暂不更改默认策略，也不继续用当前测试集扫候选权重/模式。若继续，应预先固定新的 seed/split 和一项候选比较，仅依据训练/验证可获得且不含隐藏 pool 标签的规则定义实验；报告排序、验证校准工作点、known utility，并把测试集留作最终一次确认。详见 [`analysis/semantic_hard_score_audit_s123_20261018.md`](analysis/semantic_hard_score_audit_s123_20261018.md)，池审计及 bootstrap 文件链接见该报告。

### 历史记录：数据划分协议敏感性实验最初被初始化 checkpoint 阻断（后续已完成配对）

本轮想检验核心重叠是否依赖 CIFAR-100 的类别划分：对比 `semantic_isolated`
（12 个完整 coarse superclass 为已知、8 个为未知）和 `semantic_hard`
（每个 superclass 内 3 个已知细类、2 个未知细类），固定 seed=44、4 epochs、
detached-VOS 配方、mixed discovery pool、检测器和 validation-only 95% known-coverage
校准。测试标签不用于训练、选模型或校准。

在最初这次尝试中，该对比**没有完成，不能作协议效果结论**。isolated 臂完成了独立运行，但 hard 臂加载
已有 `semantic_isolated_full_s123_disjointcal_teacher/teacher.pt` 时，训练代码正确发现其
60 类分类头及原型的 `known_classes/novel_classes` 与 semantic-hard split 不匹配并报错。
绕过检查或复用该 teacher 都会造成错误的类别头映射，不能接受。虽然本地有
`semantic_hard_teacher_s44/teacher.pt`，但它的教师训练 batch size、验证/测试限制及
calibration ratio 与原 isolated teacher 不同，直接替换会引入训练处方混杂。isolated 单臂
结果不作为跨协议证据，hard 臂没有有效训练结果。

当时增加了 [`scripts/run_split_protocol_sensitivity.py`](scripts/run_split_protocol_sensitivity.py)，
用 checkpoint 元数据检查阻止了不兼容 teacher 被错误复用。随后已按同一配置模板分别重新训练
两种协议的 teacher，并完成两臂配对；最新数据及解释以本节之前的“匹配 teacher 的 split-protocol
对比完成”记录为准。该对比衡量“协议 + 相匹配 teacher 训练”的端到端差异；若要单独识别
split 的因果效应，需要预先定义不依赖类别编号的 split-agnostic 初始化。详细记录见
[`analysis/split_protocol_sensitivity_pilot_20261018.md`](analysis/split_protocol_sensitivity_pilot_20261018.md)。
最初中止轮次的部分结果不用于协议结论；后续完整配对在 seed=44/123 各完成一次，但仍是短训练预算下的两 seed 诊断结果，不据此宣称 loss 有效或算法问题已解决。

### 2026-10-19：seed=2026 prototype-boundary rejector 复核完成

为检查此前探索性 `prototype_boundary + weighted_boundary` 的排序指标改善能否在新训练 seed 上复现，预先规划了 semantic-hard seed=2026 的单模型实验：5 epoch teacher、4 epoch student；同一冻结 student、mixed discovery pool、known prior 和 validation-only 95% known-coverage 校准下，只比较 all-pool 与 prototype-boundary weighted rejector（候选比例 0.25、权重 2.0）。联合观察 AUROC、FPR95、OSCR、校准工作点未知拒绝、已知效用和分数重叠；测试集不参与训练、校准或方法选择。

此前一次运行曾在 teacher 阶段中断，未产生结果；本次从头完成 5 epoch teacher、4 epoch student，并在同一冻结 student 上运行两臂。检查保存配置后确认两臂共用 seed=2026、semantic-hard split、mixed pool（5400，known prior=0.2）、student checkpoint、`feature_rejector`/NNPU、`support_augmented` 特征和 validation-only 95% known-coverage 校准。唯一有意改变的因素是 rejector pool treatment：all pool 对比 weighted prototype-boundary（候选比例 0.25、权重 2.0）。测试集没有用于训练、校准或调参；由于候选规则曾由旧探索性测试诊断优先选出，本次仍是探索性复核，而非独立确认。

| Rejector | AUROC | FPR95 | OSCR | 已知接受率 | 未知拒绝率 | 接受后已知准确率 | KS | unknown below known-q95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all-pool | 0.78433 | 0.63783 | 0.58743 | 95.28% | 23.60% | 71.58% | 0.43133 | 75.35% |
| prototype-boundary weighted | **0.79406** | **0.59983** | **0.59312** | 95.18% | **26.20%** | 71.67% | 0.43533 | **73.05%** |

该 seed 上 prototype-boundary 方向有一致但幅度有限的正向点估计：AUROC `+0.00973`、FPR95 改善 `0.03800`、OSCR `+0.00569`、工作点未知拒绝率 `+2.60` 个百分点；已知接受率只下降 `0.10` 个百分点，接受后已知准确率 `+0.09` 个百分点。因此这次拒绝率提升不像是通过大幅误拒已知样本换来的。分数分布重叠也仅小幅缓解：KS `+0.00400`，unknown-below-known-q95 比例 `−2.30` 个百分点；仍有 73.05% 的未知分数低于已知 q95，分布仍显著重叠。校准集实际已知接受率约 94.94%，测试已知接受率约 95.18%，符合预设工作点附近的预期。

判断：这是比此前冻结 checkpoint 上的单次弱信号更有希望的复核，但只在一个新训练 seed、一个固定类别 split 上运行一次，且候选规则有历史测试诊断导致的人工优先选择，证据仍不足以改变默认设置或宣称核心问题已解决。核心的已知/未知重叠依然严重；不要继续在这份测试集上扫候选比例/权重。下一步应预注册新的训练 seed（最好再增加独立类别 split），保持相同训练/评测协议，只确认 all-pool 与此固定 prototype-boundary 配方；报告完整排序、validation-calibrated 工作点、known utility 和 overlap。若不能在独立 seed/split 复现，再停止该 rejector 方向，转向训练期 class-conditional support/outlier 表征目标，而不是再调阈值。

结果明细见 [`analysis/semantic_hard_rejector_seed2026_interrupted_20261019.md`](analysis/semantic_hard_rejector_seed2026_interrupted_20261019.md) 和机器可读汇总 [`analysis/semantic_hard_rejector_seed2026_summary.json`](analysis/semantic_hard_rejector_seed2026_summary.json)；运行入口为 [`scripts/run_semantic_hard_rejector_seed_recheck.py`](scripts/run_semantic_hard_rejector_seed_recheck.py)。

随后使用完全相同的固定配方在 seed=2027 重复一次（5/4 epoch，语义划分、pool 规格和拒识器设置不变；两臂仍共用各自 seed 的 student）。prototype-boundary 相对各 seed 内 all-pool 的差值如下：

| Seed | AUROC Δ | FPR95 Δ（负向为改善） | OSCR Δ | 未知拒绝率 Δ | 已知接受率 Δ | KS Δ | unknown below known-q95 Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | +0.00973 | −0.03800 | +0.00569 | +2.60 pp | −0.10 pp | +0.00400 | −2.30 pp |
| 2027 | +0.00705 | −0.03150 | +0.00634 | +0.625 pp | −0.083 pp | +0.01000 | −0.525 pp |
| 2028 | +0.00805 | −0.02783 | +0.01258 | −0.425 pp | −0.383 pp | +0.01258 | +2.075 pp |
seed 2028 的配对条件、完整指标和后续判断见 [`analysis/semantic_hard_rejector_seed2028_recheck_20261019.md`](analysis/semantic_hard_rejector_seed2028_recheck_20261019.md)；机器可读汇总见 [`analysis/semantic_hard_rejector_seed2028_summary.json`](analysis/semantic_hard_rejector_seed2028_summary.json)。

**三 seed 复核后的修订结论：** AUROC/FPR95/OSCR 排序指标均同向小幅改善，但 validation-calibrated 未知拒绝变化依次为 `+2.60/+0.625/−0.425 pp`；unknown-below-known-q95 overlap 在 seed 2028 反而恶化 `+2.075 pp`。所以排序收益没有稳定转化为工作点未知拒绝收益，不能认为核心重叠已解决。三次共用 semantic-hard 类别划分，且候选方向由历史探索引出，证据仍属探索性；停止在该测试集扫参，不改默认。下一步优先做训练期 class-conditional support/outlier 表征实验：只用 known train labels 建立类别条件支持区域，以预先固定的训练/验证代理确定 hard-boundary 样本；mixed pool 伪未知须 label-free 保守筛选并报告候选污染敏感性，不得用测试标签或事后诊断挑样本、权重。训练目标与 rejector 后处理分开配对评估，联合考察局部邻域、排序、固定 coverage 下拒识、known utility。

两个 seed 的排序指标方向一致，已知接受率均约 95%，接受后已知分类准确率没有明显损失；但工作点未知拒绝率的收益从 `+2.60 pp` 缩小到 `+0.625 pp`，且两个 seed 中未知分数仍分别有 `73.05%` 和 `74.225%` 低于已知 q95。故结论是“值得继续做一次预注册多 seed/独立 split 确认，但收益幅度有限，核心重叠未解决”，而非稳健有效或可改默认。此两次复核共用固定 semantic-hard 类别划分，且方向曾被历史测试集诊断影响；它们不是完全独立的确认性证据。下一步若继续，只做固定配置的额外独立 seed/split 复核，不扫参数；若实际目标是减少重叠，应另立训练期 class-conditional support/outlier 表征实验并与此拒识器后处理分开评估。

### 2026-10-19：训练期 shared-feature kNN support-boundary 单因素筛查

为执行上述训练期 class-conditional support 方向，固定 semantic-hard split、seed=2029、4 epochs、同一 split-compatible teacher、mixed pool 5,400（known prior 0.2）、候选与 cross-view gating、feature rejector 及 known-validation-only 95% coverage 校准，只比较 `alpha_discovery_knn_boundary=0` 与 `0.1`。两臂均显式关闭独立 rejection-boundary 和 rejection-kNN loss。训练历史确认 treatment 的 kNN loss 和对应梯度均非零，因此这次确实执行了目标函数，而非仅配置开关。

结果是混合且幅度有限：AUROC `0.79636 -> 0.79578`（−0.00058），FPR95 `0.61367 -> 0.60883`（改善 0.00483），OSCR `0.58346 -> 0.59401`（+0.01055），95% validation-calibrated 工作点的未知拒绝率 `26.50% -> 27.68%`（+1.18 pp），接受后已知准确率 `70.13% -> 71.74%`（+1.61 pp）。但 KS 几乎不变（−0.00058），unknown-below-known-q95 overlap 从 `72.500%` 到 `72.525%`，没有改善；测试已知接受率为 95.23%/94.88%。因此不能把工作点拒绝率上升解读为分布重叠解决。

机制审计还显示 treatment 的 cross-view 候选 gate 最终仅选中约 17.3% 的 mixed discovery pool；训练日志中只用于审计、不会进入 loss/gate/阈值/优化器决策的池标签统计显示，入选集已知污染约 9.0%，但未知召回只有约 19.7%。`discovery_knn_*_selection_rate` 为零是因为 selection-conditional PU 开关关闭，不代表没有候选入选；本轮应看 `feature_candidate_*` 审计字段。该信号提示候选未知覆盖率可能限制训练损失效果。下一步优先研究预先固定的 label-free 候选规则如何提升未知边界覆盖，同时用训练已知样本控制误选，并严格隔离训练池诊断标签，不得使用测试标签或诊断标签进行筛选与调参。

**决策：**不改默认，不在测试集调 kNN 权重。该单 seed 结果仅为探索性混合信号；结合 kNN bank 使每 epoch 训练明显变慢、核心 overlap 完全未改善，暂不优先做同一 split 的权重扫描。后续转向训练/验证合法的局部边界样本诊断：审计候选选择稳定性、类条件 support 覆盖与 mixed-pool 污染代理，并预先固定 seed/split、校准规则及联合指标；只有排序/overlap 与 coverage-matched 拒识共同改善且 known utility 可接受，才考虑独立复核。完整条件、数值与限制见 [`analysis/semantic_hard_knn_support_train_s2029_e4.md`](analysis/semantic_hard_knn_support_train_s2029_e4.md)，机器可读结果见 [`analysis/semantic_hard_knn_support_train_s2029_e4_summary.json`](analysis/semantic_hard_knn_support_train_s2029_e4_summary.json)。

### 2026-10-19：取消候选 gate 的 full-pool soft-nnPU kNN 对照（seed=2030）

**实验问题与预设对比：**上节候选 gate 对未知样本的召回约为 19.7%，怀疑训练期 kNN 边界损失受候选覆盖不足限制。为隔离该因素，在 semantic-hard split、seed=2030、同一 teacher、4 epochs、同一 5,400 样本 mixed pool（known prior=0.2）、同一其余训练/评估配置下，比较 `alpha_discovery_knn_boundary=0` 与 `0.1`。两臂都关闭 feature/cross-view candidate gating，使用完整 mixed pool 上的 soft-nnPU kNN 边界损失（known prior=0.2、softness=0.05）；拒识器均为 `feature_rejector`，known validation-only 校准到 95% coverage。实验前固定联合观察 AUROC、FPR95、OSCR、校准工作点未知拒绝率、known utility、KS 与 overlap proxy。测试标签没有用于训练、校准或配置选择；结果仅作单 seed 机制筛查。

| full-pool kNN 边界损失 | AUROC | FPR95 | OSCR | 测试已知接受率 | 未知拒绝率（95% known coverage） | 接受后已知准确率 | score KS | unknown below known-q95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 关闭（baseline） | 0.78714 | 0.63800 | 0.58901 | 96.07% | 22.45% | 71.56% | 0.42575 | 73.75% |
| 开启（weight=0.1） | 0.77725 | 0.64617 | 0.58615 | 95.32% | 22.35% | 71.94% | 0.41792 | 76.18% |

Treatment 相对 baseline 的 AUROC `−0.00988`、FPR95 `+0.00817`（变差）、OSCR `−0.00285`，固定 coverage 的未知拒绝率 `−0.10 pp`；KS 虽提高 `0.00783`，但 overlap proxy 反而恶化 `+2.425 pp`。接受后已知准确率只增加 `0.37 pp`，无法抵消开集排序及 overlap 指标的退步。因此本次没有支持“低候选召回是唯一瓶颈”：取消 gate 并让损失接触完整 pool，没有改善核心区分能力。训练日志确认 kNN loss 在四个 epoch 都非零（约 0.057–0.112），梯度范数约 0.59–0.79，effective prior=0.2；处理确实执行，并非配置未接线。full-pool kNN 计算也显著增加时间（该 treatment 约 14 分钟），收益/成本不理想。

**关于梯度量级的补充核查：**日志中的逐损失梯度范数是未乘 loss coefficient 的 raw gradient，不能直接当作最终优化器更新强度。treatment 的 kNN raw encoder 梯度均值约 0.720，CE 均值约 12.372；乘实际 `alpha_discovery_knn_boundary=0.1` 后，kNN 梯度范数约为 CE 的 0.60%。raw kNN 梯度与 CE 的 cosine 均值约 −0.131（与 KD 约 +0.007、SupCon 约 −0.049）；正系数缩放不改变 cosine，但小的相对范数意味着本次失败还不能排除“有效梯度过弱”。这不是将结果解释为单纯调权重问题，而是一个尚待单次检验的机制假设。

**seed=2031 结果：**只改变 kNN loss 系数的配对比较如下。保存配置确认 baseline/treatment 的其余训练、pool、teacher、split 和评估协议相同；calibration 使用 known validation，测试标签未用于训练、校准或选择。

| full-pool kNN 边界损失 | AUROC | FPR95 | OSCR | 测试已知接受率 | 未知拒绝率 | 接受后已知准确率 | score KS | unknown below known-q95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 关闭（baseline） | 0.78630 | 0.63200 | 0.58618 | 93.48% | 29.50% | 71.99% | 0.43567 | 76.38% |
| 开启（weight=0.5） | 0.79365 | 0.60833 | 0.58930 | 94.77% | 26.45% | 71.44% | 0.44125 | 74.40% |

Treatment 的 AUROC `+0.00735`、FPR95 改善 `0.02367`、OSCR `+0.00312`；KS `+0.00558`，overlap proxy 下降 `1.975 pp`。但 validation-calibrated 工作点未知拒绝率下降 `3.05 pp`，已知接受率上升 `1.283 pp`，接受后已知准确率下降 `0.553 pp`。所以这是排序/分布略改善与工作点效用变差的明确权衡，不符合预设的联合成功判据。validation 校准目标是 95%，测试实际 known acceptance 并不保证恰为 95%；报告中必须保留这个泛化差异，不能把测试结果伪称为精确 coverage matching。

梯度记录 raw kNN encoder 范数约 `0.44–0.72`，raw cosine 与 CE 约 `−0.07 至 −0.14`。乘 alpha=0.5 后，加权 kNN/CE 梯度比例随 epoch 约 `1.49%–3.20%`，比 alpha=0.1 高，但仍明显小于主要监督梯度。单 seed 不能据此断言最佳权重，也不能用该测试集选择参数。按预设计划，以完全相同的 alpha=0.5 配方在 seed=2032 做一次固定复核；不扫权重。若工作点 tradeoff 或 overlap 方向不稳定，则停止此分支。结果明细见 [`analysis/semantic_hard_knn_pu_full_pool_s2031_e4_w05_summary.json`](analysis/semantic_hard_knn_pu_full_pool_s2031_e4_w05_summary.json)。

### 固定复核 seed=2032：full-pool kNN weight=0.5 未复制联合收益

严格复用 seed=2031 的 alpha=0.5 配方，只换训练 seed 为 2032，并在该 seed 内与 alpha=0 baseline 成对。split、teacher、4-epoch budget、完整 mixed pool、soft-nnPU 参数、candidate gate 关闭、score mode、known validation-only 95% coverage calibration 均保持不变。两臂训练和 discovery 均完成，保存配置相符，测试标签未进入训练、校准或选择。

| seed=2032 | AUROC | FPR95 | OSCR | 测试已知接受率 | 未知拒绝率 | 接受后已知准确率 | score KS | unknown below known-q95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.78935 | 0.62717 | 0.58806 | 95.00% | 25.95% | 71.46% | 0.43200 | 73.93% |
| full-pool kNN weight=0.5 | 0.78804 | 0.59633 | 0.58857 | 94.95% | 23.025% | 71.25% | 0.43867 | 77.15% |

相对 baseline，AUROC `−0.00131`，FPR95 改善 `0.03083`，OSCR `+0.00052`；但工作点未知拒绝率下降 `2.925 pp`，接受后已知准确率下降 `0.208 pp`，KS 提高 `0.00667`，而 overlap proxy 恶化 `3.225 pp`。保存训练历史确认 kNN 项持续产生非零梯度；乘 alpha=0.5 后，kNN/CE encoder 梯度比例约 `0.93%–3.25%`，但这仍没有转化为稳定的分布与工作点收益。

**两 seed 联合判断：**seed=2031 的 AUROC/OSCR/KS/overlap 点估计略正向，但未知拒绝率 `−3.05 pp`；seed=2032 的工作点未知拒绝率 `−2.925 pp`、overlap `+3.225 pp`，AUROC 也微降。只有 FPR95 两 seed 都改善，其他关键指标并未共同改善。因此“alpha=0.1 时梯度过弱”并不足以解释失败；增至 0.5 也没有达到预设联合成功标准。**停止 full-pool soft-nnPU kNN 权重路线，不改默认、不继续扫权重。**这是两个训练 seed 的探索性配对，不是跨 split 的确认性结论；其意义是该方法当前不值得投入更多算力，而不是证明 kNN support 方法在所有任务上都无效。

后续优先转向共享表征与拒识支路的机制审计：从保存日志核对“raw gradient × loss coefficient”的真实量级、不同目标梯度夹角，并区分梯度作用于 shared encoder、classifier 还是独立 rejection head；随后依据训练/已知验证数据可合法获得的 class-conditional support 风险，设计一个结构不同的新表征目标。先做梯度/数据流检查和便宜的 smoke test，再决定是否值得跑完整配对训练。评估仍联合报告 AUROC、FPR95、OSCR、score KS/overlap、validation-calibrated 工作点和 known utility，不以阈值或单一指标替代核心目标。

seed=2032 结果明细见 [`analysis/semantic_hard_knn_pu_full_pool_s2032_e4_w05_summary.json`](analysis/semantic_hard_knn_pu_full_pool_s2032_e4_w05_summary.json)；本轮不上传 GitHub。

结果与保存配置见 [`analysis/semantic_hard_knn_pu_full_pool_s2030_e4_summary.json`](analysis/semantic_hard_knn_pu_full_pool_s2030_e4_summary.json)；可复跑脚本为 [`scripts/run_semantic_hard_knn_pu_full_pool_compare.py`](scripts/run_semantic_hard_knn_pu_full_pool_compare.py)。首次运行在 treatment 开始前中断，只留下完整 baseline；随后脚本调整为仅复用两臂均完整的既有产物、遇到不完整目录拒绝覆盖，并安全完成 treatment，因此上表为同一 seed/配置的完整配对结果。本轮未上传 GitHub。

### 2026-10-19 semantic-hard rejection branch CE balance pilot

**实验前假设与对比：**此前 rejection 分支梯度审计显示 known rejection CE 的梯度可能显著强于若干未知侧目标；semantic-isolated 上降低该 CE 的结果有弱正向但不完整的信号。因此在 semantic-hard 上固定 seed=2029、同一 split/teacher、4 epoch、mixed pool、其余训练目标及 detector/discovery/calibration 配置，仅比较 `alpha_rejection_known_ce=1.0`（已有 baseline）与 `0.25`。拒识器为 `feature_rejector + support_augmented`，仅用 known validation 校准到 95% coverage；测试标签不参与训练、选模或校准。训练与评估均使用 RTX 4060 CUDA。依据来源及旧审计见 [`analysis/rejection_branch_gradient_audit_20261013.md`](analysis/rejection_branch_gradient_audit_20261013.md)。

| seed=2029 | AUROC | FPR95（低为好） | OSCR | 测试已知接受率 | 校准工作点未知拒绝率 | 接受后已知准确率 | 全部已知准确率 | score KS | unknown below known-q95 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CE=1.0 | 0.79551 | 0.60517 | 0.59105 | 94.7667% | 25.625% | 71.1748% | 69.6000% | 0.44683 | 74.900% |
| CE=0.25 | 0.80304 | 0.60917 | 0.59326 | 94.7500% | 29.475% | 70.9235% | 69.3833% | 0.46375 | 72.000% |

seed=2029 出现值得固定复核的混合信号后，按原计划在 seed=2030 完成一次 1.0 vs 0.25 配对复核，没有扫描中间权重。结果如下（每个 seed 都只与本 seed 内的 CE=1.0 对照比较）：

| Seed / CE | AUROC | FPR95（低为好） | OSCR | 测试已知接受率 | 工作点未知拒绝率 | 接受后已知准确率 | 全部已知准确率 | KS | overlap proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2029 / 1.0 | 0.79551 | 0.60517 | 0.59105 | 94.7667% | 25.625% | 71.1748% | 69.6000% | 0.44683 | 74.900% |
| 2029 / 0.25 | 0.80304 | 0.60917 | 0.59326 | 94.7500% | 29.475% | 70.9235% | 69.3833% | 0.46375 | 72.000% |
| 2030 / 1.0 | 0.77163 | 0.65100 | 0.58540 | 95.4833% | 20.900% | 72.2290% | 70.9333% | 0.40242 | 77.475% |
| 2030 / 0.25 | 0.78349 | 0.63850 | 0.58853 | 94.8667% | 26.250% | 72.0309% | 70.6833% | 0.42092 | 74.350% |

在最初两个训练 seed（2029/2030）上，AUROC、OSCR、KS、overlap proxy 和 validation-calibrated 工作点未知拒绝率方向一致改善；FPR95 在 seed=2030 改善、seed=2029 略退。测试已知接受率变化较小，known accuracy 下降约 `0.2–0.25 pp`。但代码审计后来发现 semantic-hard split 生成器当时固定选取每个 superclass 排序后的前三类，因此这两个 seed **只是训练初始化不同，类别 split 并不独立**，不能把它们当作跨 split 复现。完整条件、产物位置见 [`analysis/semantic_hard_rejection_ce_balance_s2029_e4_20261019.md`](analysis/semantic_hard_rejection_ce_balance_s2029_e4_20261019.md)；运行脚本 [`scripts/run_rejection_ce_balance_pilot.py`](scripts/run_rejection_ce_balance_pilot.py)。

### semantic-hard split 生成器修复与独立类别划分复核（2026-10-19）

**审计发现：** `scripts/make_cifar100_splits.py` 原先虽然接收 `seed`，semantic-hard 却固定取每个 coarse superclass 排序后的前三个 fine class。现已修正为由独立 seeded RNG 在每个 superclass 内抽取 3/5 类；每组仍为 3 known / 2 novel，random 与 semantic-isolated 生成逻辑不变，现有 split JSON 未覆盖。新增测试确认相同 seed 可复现、不同 seed 会改变 class assignment，且每组数量仍正确。当前旧 semantic-hard 文件保留作为历史协议；新生成器创建的 seed-specific 文件构成新的、明确标记的协议版本，不应与旧版混写。

使用 seed=2040 生成独立类别 split（60 known/40 novel），其 known 类与旧 hard split 仅重合 36/60；在该 split 上重新训练匹配的 5-epoch teacher，再按预注册方案只比较 CE=1.0 与 0.25，学生 4 epochs，其余训练/拒识/验证校准配置固定。测试标签不用于训练、选模、校准或权重选择。

| seed=2040 新类别 split | AUROC | FPR95（低为好） | OSCR | 测试已知接受率 | 工作点未知拒绝率 | 接受后已知准确率 | 全部已知准确率 | KS | overlap proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CE=1.0 | 0.80441 | 0.61100 | 0.60227 | 95.1833% | 25.250% | 71.9664% | 70.3167% | 0.46442 | 73.725% |
| CE=0.25 | 0.81219 | 0.57067 | 0.60907 | 96.1167% | 24.025% | 72.0999% | 70.8500% | 0.47183 | 71.700% |

独立 split 上 AUROC `+0.00778`、FPR95 改善 `0.04033`、OSCR `+0.00680`、KS `+0.00742`、overlap proxy 降低 `2.025 pp`，已知效用没有下降；但校准工作点未知拒绝率反而下降 `1.225 pp`。因此跨两个类别 split，降低 known CE 的排序/分布尾部信号有一定一致性，但**不能稳定提高固定 validation-known-coverage 下的未知拒绝率**。新 split 仍有 `71.7%` 未知分数低于 known q95，核心重叠仍突出。这条方法不能宣称有效解决核心问题，不改默认、不继续用测试指标调权重；下一步应先从训练与已知验证数据分析“ranking/overlap 改善为何未转成工作点收益”，再决定是否另设预注册训练目标。详细数值、seed=2029/2030/2040 条件与产物见 [`analysis/semantic_hard_rejection_ce_balance_s2029_e4_20261019.md`](analysis/semantic_hard_rejection_ce_balance_s2029_e4_20261019.md)。新 split 文件：`splits_cifar100_semantic_hard_seed2040/cifar100_60_40_semantic_hard.json`；复跑入口：[`scripts/run_semantic_hard_ce_balance_independent_split.py`](scripts/run_semantic_hard_ce_balance_independent_split.py)。本轮未上传 GitHub。

### semantic-hard 第二个独立类别 split 复核（seed=2041，2026-10-20）

针对 seed=2040 上“排序与 overlap 有改善，但工作点未知拒绝率下降”的混合结果，预先固定再做一次独立 split 配对复核：seed=2041、匹配 5-epoch teacher、学生 4 epochs、同一 mixed pool 与拒识/发现配置，唯一目标因素仍为 `alpha_rejection_known_ce=1.0 vs 0.25`，只用 known validation 校准至 95% coverage。最终测试标签只用于描述性评估。保存的训练配置核对确认除目标系数、输出路径及 validation 派生 checkpoint 字段外无其他配置差异。

| seed=2041 新类别 split | AUROC | FPR95（低为好） | OSCR | 测试已知接受率 | 工作点未知拒绝率 | 接受后已知准确率 | 全部已知准确率 | KS | overlap proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CE=1.0 | 0.79098 | 0.61367 | 0.58286 | 93.883% | 28.625% | 70.939% | 68.733% | 0.44192 | 75.075% |
| CE=0.25 | 0.79337 | 0.60417 | 0.58375 | 94.967% | 24.875% | 70.762% | 69.100% | 0.43983 | 75.275% |

**判断：** 排序指标仅小幅改善；KS 几乎不变，overlap proxy 反而恶化 `0.2 pp`。最重要的是工作点未知拒绝率下降 `3.75 pp`，尽管测试已知接受率增加 `1.083 pp`；两臂 validation calibration 均为 `94.977%`。因此降 CE 未通过预先设定的工作点复核标准，停止继续做该系数的测试集驱动试验，不改默认；约 `75%` 未知分数仍低于 known q95，核心重叠没有解决。后续转向训练期 class-conditional 局部几何与保守、label-free 候选选择，先审计训练/验证代理能否识别边界样本，再预注册单因素目标比较，联合看工作点拒识、known utility、排序和 overlap，并保留未触碰的最终测试协议。详细实验条件、报告路径见 [`analysis/semantic_hard_rejection_ce_balance_s2029_e4_20261019.md`](analysis/semantic_hard_rejection_ce_balance_s2029_e4_20261019.md)；split 为 `splits_cifar100_semantic_hard_seed2041/cifar100_60_40_semantic_hard.json`。本轮未上传 GitHub。

### semantic-hard training-pool 候选 gate 的 fine-class 覆盖审计（seed=2041，2026-10-20）

（旧版随机 MC-dropout 审计结果，已由下方“审计更正”取代；这里只留历史上下文。）为弄清候选 gate 是否真的覆盖未知边界，而不只看总体 purity，在同一冻结 CE=1.0 student、同一 5,400 样本训练期 mixed pool（known prior=0.2）上，对七种现有 label-free selector 做事后标签审计；pool label 仅用于计算污染与 per-class recall，不参与 selector、训练、校准或方法选择，最终 test 未被访问。25% 预算的随机选择期望 novel recall 为 25%；entropy+uncertainty 为 `28.43%`，distance consensus `28.75%`，prototype/support boundary 约 `28.3–28.4%`，也只有约 `3.2–3.8 pp` 绝对提升。总体 known contamination 为 `8.0–9.85%`。按 40 个 novel fine class 分解后更明显：entropy+uncertainty 的 median recall `29.03%`，最低 class 只有 `2.63%`；consensus 最低 `0.88%`；所有 selector 最低类 recall 都低于 `5%`。因此候选的总体纯度会掩盖严重的类别覆盖不均，当前 gate 并未可靠覆盖全体 novel 类边界；但这仍是单个冻结模型的机制审计，不等价于证明 class-balanced gate 一定能提升最终测试指标。

**后续决定：** 不依据这次使用隐藏 pool 标签做过的事后审计调 selector，也不直接把标签喂入训练。下一步可研究仅按已知模型预测类别/类条件 support 形成的 label-free 候选配额；先在同一冻结训练池上核对最低分位的 per-class coverage 与 known contamination，再决定是否进行预注册的单因素训练配对。由于选样会改变 PU prior，若继续用 PU loss 必须同时审计 selection-conditional 风险校正，不能默认原先验仍适用。完整指标和限制见 [`analysis/semantic_hard_candidate_classwise_audit_s2041_20261020.md`](analysis/semantic_hard_candidate_classwise_audit_s2041_20261020.md) 与机器可读结果 [`analysis/semantic_hard_candidate_pool_audit_s2041_20261020.json`](analysis/semantic_hard_candidate_pool_audit_s2041_20261020.json)。本轮只更新审计代码和文档，未上传 GitHub。

（旧版随机 MC-dropout 结果，已作废。）随后在同一冻结 seed=2041 checkpoint 上做了固定预算的 class-balanced selector 机制筛查：按预测 known class 分 bin，每个 bin 分配相等候选配额，bin 内按 entropy+uncertainty 排序，剩余名额由全局风险分补足。选择 pool 的 25% 后，最差 novel-class recall 从 `2.63%` 升至 `4.39%`，但 median recall 从 `29.03%` 降至 `28.08%`、总体 novel recall 从 `28.43%` 降至 `27.50%`，known contamination 从 `9.04%` 升至 `12.00%`。尾部只有有限改善且污染上升 2.96 pp，未达到预设共同标准；没有接入训练，也不值得直接跑完整训练。可继续探索的不是“均匀配额”本身，而是用独立 known validation 的各预测类风险分布校准每类选择阈值/配额，目标是在控制 known false selection 下改善长尾覆盖；先做纯 label-free 候选机制筛查，之后才考虑训练配对，并正确处理选择导致的 PU prior 变化。该方向仍是假设，不是有效性结论。

（旧版随机 MC-dropout 结果，已作废。）进一步测试了一个区别于均匀配额的规则：只用 known validation 中每个预测类的 entropy+uncertainty 95 分位作为阈值（该类 validation 预测少于 20 个时回退到全局阈值），再应用于同一个 mixed training pool。选中 `15.43%` pool，known contamination `7.44%`、novel recall `17.85%`，较同预算随机选择期望 `15.43%` 高 `2.42 pp`；但 per-class median recall 仅 `17.01%`，最差类仍为 `2.63%`。known validation 各预测类实际接受率为 `91.3–95.1%`，显示 finite/discrete bin 的校准误差；两个 bin 回退到全局阈值。该法通过减少选择量降低污染，却没有修复 class-tail coverage，不接入训练。两种候选分配改动均未达到机制门槛，因此当前停止候选 selector 探索，不跑完整训练；所有 mixed-pool 标签只用于事后审计，未访问最终 test。详情见 [`analysis/semantic_hard_candidate_classwise_audit_s2041_20261020.md`](analysis/semantic_hard_candidate_classwise_audit_s2041_20261020.md)。

**审计更正（2026-10-20）：** 上面首轮候选池数字及 class-balanced/q95 结论已被后续复核取代。审计调用 `extract_outputs(..., mc_samples=1)`，但该 helper 即使只采样一次也会启用随机 MC-dropout；先前将其称为确定性提取有误，导致类别统计复跑变化。现已改为普通 eval 前向，且加入重复提取一致性测试。按两个独立 semantic-hard split 的确定性结果，validation classwise empirical-CDF top-25% 在 seed 2041 将 contamination/novel recall/最差类 recall 从 `9.19%/28.38%/1.75%` 改为 `8.81%/28.50%/4.39%`；但 seed 2040 则从 `6.37%/29.26%/2.73%` 变为 `6.89%/29.10%/2.73%`，没有复现改善。因此该方法不接入训练、不跑完整实验；下一步先审计确定性训练池信号的跨 split 稳定性。更正后报告与 JSON：[`analysis/semantic_hard_candidate_classwise_audit_s2041_20261020.md`](analysis/semantic_hard_candidate_classwise_audit_s2041_20261020.md)、[`analysis/semantic_hard_candidate_pool_audit_s2041_deterministic_20261020.json`](analysis/semantic_hard_candidate_pool_audit_s2041_deterministic_20261020.json)、[`analysis/semantic_hard_candidate_pool_audit_s2040_deterministic_20261020.json`](analysis/semantic_hard_candidate_pool_audit_s2040_deterministic_20261020.json)。旧 JSON 保留供追溯，但不得再作为确定性结果引用。
