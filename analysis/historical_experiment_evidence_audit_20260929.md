# 历史实验方法与证据审计（2026-09-29）

## 审计范围与判定口径

本次核对本地保存的配置、检测报告、部分训练历史、数据划分文件、当前训练/评分调用链，以及 README 对以下实验的描述：

1. Standard KD 与不确定性加权 KD；
2. Outlier Exposure（OE）风格的 Energy / uniform-logit 实验；
3. 冻结特征 rejector 与 mixed-pool / PU 实验；
4. CIFAR-100 random、semantic-hard、semantic-isolated 类别协议。

“配置文件写了某参数”与“代码确实按该参数执行”分开核对；“代码运行了”与“实验足以支持因果或推广结论”也分开判断。当前仓库工作树已有其他未提交改动，本审计不覆盖或清理它们，也没有改动 README、训练代码或运行历史。

## 1. 不确定性加权 KD：主三 seed 对照基本配对，解释需精确

### 已核实

- `runs/revised_ms_s{42,123,3407}_{B_standard_kd,C_uncertainty_kd}/config.json` 均记录 CIFAR-100 60/40、15 epochs、ResNet-18 student、相同 seed 配对使用同一路径的 teacher、`alpha_kd=1`、温度 2、`alpha_unc=0`，且关闭 feature KD、SupCon、prototype 等额外项。
- 配对的主要训练差异确为 `kd_mode=standard` 对 `kd_mode=uncertainty`；保存的三 seed 检测配置使用相同 `normalized_entropy_mahalanobis`、MC=8、同一 class split 和 known-coverage 校准。完整多 seed 摘要见 `analysis/audit_kd_standard_full3/multiseed_summary.md` 与 `analysis/audit_kd_uncertainty_full3/multiseed_summary.md`。
- 当前调用链证实 `kd_mode=uncertainty` 会把**教师** uncertainty 变成每样本 KL 蒸馏权重：`pipeline.py` 将 `t_out["uncertainty"]` 传给 `distillation_loss`，`losses.py` 对每样本 KL 乘权后求均值。因此这不是“仅训练了 uncertainty head”的对照，而确实实现了 uncertainty-weighted KL KD。
- `alpha_unc=0` 只关闭学生辅助 uncertainty alignment loss；它不会关闭上面的教师不确定性加权。这一区分很重要，不能把这项实验误解为“完全没有用不确定性”。
- 三 seed 汇总中 Standard KD 的均值略占优：AUROC `0.5764±0.0233` vs `0.5737±0.0340`；FPR95 `0.8826±0.0174` vs `0.8833±0.0165`；known accuracy `0.4261±0.0055` vs `0.4143±0.0042`；unknown reject `0.0561±0.0064` vs `0.0564±0.0114`。结论应是“该实现已验证运行，但在当前协议下没有证明优于 Standard KD”，而非“蒸馏没有执行”或“已解决未知检测”。

### 限制

- seed 42/123 的这些训练目录保留配置和 checkpoint，但没有 `train_history.json`；seed 3407 才保留训练历史。因此现有材料足以核对主要参数和模型模式，但不能逐 epoch 重建三 seed 的训练过程。
- 历史 artifacts 没有记录源代码 commit/hash，也没有完整命令、环境锁文件或每轮样本顺序；不能声称从当前源代码对历史运行做到了逐位复现。
- README 已注明旧报告缺 AUPR/OSCR 的问题；摘要中这些字段仅 `n=1`，不得称为三 seed 统计量。

## 2. OE 风格损失：uniform 对照有单因素含义，但不是“有无 OE”对照

### 已核实

- `runs/oe_medium_baseline/config.json` 与 `runs/oe_medium_uniform/config.json` 使用相同 CIFAR-100 split、seed 42、1200 known train、1200 discovery、300 val、1000 test、5 epochs、同一 teacher 路径、相同 `alpha_discovery_energy=0.1`；唯一明确的损失差异是 `alpha_discovery_uniform: 0` 对 `0.1`。
- seed 43/44 的对应训练配置同样是 `alpha_discovery_energy=0.1` 固定，只切换 uniform 权重。检测配置使用 Energy 分数与 `open_val_ratio=0.2`。
- 因此这些配对实验检验的是“在 Energy 型纯未知暴露目标上，额外加 uniform-logit loss 是否有帮助”，并非“OE 对比不做 OE”，也不单独证明 Energy OE 本身有效。
- discovery pool 标成 `unknown`，当前数据流水线按类别划分从训练数据筛出纯 novel 样本；属于 oracle-filtered unknown pool，作为上限/受控实验，不等同于真实 mixed 无标签池。
- 这项 uniform 项借鉴 OE 的辅助异常暴露思路，但实现并非 Hendrycks et al. 论文完整复现。三 seed 的 AUROC 多数上升，但 unknown reject、known accuracy 并未一致改善；README 中“排序有初步信号、工作点未证实”的谨慎结论与现有记录一致。

### 需要避免的措辞

表格中的“Energy baseline”建议明确写成“Energy-loss baseline（`alpha_discovery_energy=0.1`, uniform=0）”，避免被读成“不含任何 OE/异常暴露目标的普通 KD baseline”。若要判断 Energy OE 本身是否有效，需增加同一协议的 `alpha_discovery_energy=0` 对照。

## 3. Feature rejector：纯未知池高分是上限，不是开放环境结果

### 已核实

- `runs/full_compare_standard_kd_rejector/config.json` 与 uncertainty KD rejector 配置均使用完整已知数据、5000 个 `discovery_pool_mode=unknown` 样本、Logistic `feature_rejector`、embedding 特征、known-coverage 95% 校准；其结果分别为 AUROC/FPR95/unknown reject `0.7103/0.7670/0.1670` 与 `0.7035/0.7608/0.1413`。
- 这些数值是从纯未知池训练 rejector 得到的受控上限式结果。它们证明在提供纯未知样本时 frozen-feature rejector 可学到一定边界；不证明可直接从未标注 mixed stream 学得同一边界。
- `runs/full_compare_ce_strict_mixed_pu/config.json` 是另一协议：CE checkpoint、mixed pool、5000 样本、留出 open validation（ratio 0.2）、strict mixed、soft-PU。结果 AUROC `0.4785`、FPR95 `0.9360`、unknown reject `0.0295`。这不是与上述 KD 结果只差一个训练方法的直接对照，因为模型 checkpoint、pool 语义与 rejector 训练规则不同。
- 当前代码调用链确认 `strict_mixed` 会从 open-validation 中提取 known 样本，并把 discovery pool 交给 PU rejector；这与把 mixed pool 全部当 unknown 的旧式 hard rejector 不是同一实现。

### 解释边界

README 将纯未知高分标为上限，并指出 mixed 情况失败，这一主结论正确。以后汇总表必须分开标注纯 unknown、mixed-hard、mixed-PU；不要把它们放在同一“模型优劣”排名中。所谓“mixed-PU 无效”也应限于当前 seed、当前 CE 表征和当前 PU 配置，不能泛化成 PU 方法整体无效。

## 4. 三套类别协议：划分构造有效，但已有数值不是严格的协议单变量比较

### 划分本身

`scripts/make_cifar100_splits.py` 使用 CIFAR-100 官方 fine/coarse taxonomy 生成三个互斥、覆盖全部 100 类且 known/novel 各 60/40 的固定协议：

- `random`：seeded 随机抽取 60 个 known fine classes；
- `semantic_hard`：每个 coarse superclass 固定取 3 个 known fine classes，另外 2 个成为 novel，已知与未知语义相近；
- `semantic_isolated`：选 12 个完整 coarse superclasses 作为 known，其余 8 个 coarse groups 为 novel。

这些定义本身与脚本文档一致，可用于评估类别划分敏感性。

### 历史运行中的协议混杂

- 三组训练大体采用相同 seed 42、1200 train、300 val、1000 test、3 epochs；但它们是单 seed、小样本探索实验，不是最终 benchmark。
- 检测配置并不一致：random 使用 `normalized_entropy_mahalanobis`，semantic-hard 与 semantic-isolated 使用 `entropy_proto`。因此其 AUROC `0.5577 / 0.4925 / 0.5973` 和拒绝率 `0.0709 / 0.0651 / 0.1178` 同时混合了“类别 split 变化”和“检测分数变化”，不能据此把差异归因于语义协议。
- 测试集仅 1000 张，known-coverage 校准集有限；报告中的测试 known acceptance 为约 92.7%–93.7%，并非精确 95%。这些工作点受有限校准样本影响。
- 训练配置显示三组当前都是 `kd_mode=uncertainty`；因而结果不是 Standard KD 对照，也没有办法从该批结果判断不确定性 KD 在不同语义划分下是否优于其他蒸馏方式。

### 纠正方向

保留现有 checkpoints 作探索记录，不删改历史数值；后续重新评估时，对同一个 split 使用固定 score mode、固定 checkpoint-selection 规则、相同 known-validation coverage，并将 seed 扩为至少 3 个。要声称“划分难度影响结果”，就只能改变类别 split，不能同时改变检测器；若重点是检测器对不同协议的适配，则应在每个 split 内配对比较多个检测器，并清晰区分问题。

## 5. 当前可以信任到什么程度

| 结论 | 审计判断 |
|---|---|
| uncertainty-weighted KL 的代码路径确实被启用 | 已由当前调用链与 `kd_mode` 配置核实 |
| 三 seed 下 uncertainty KD 优于 standard KD | 不支持；现有均值总体略差，unknown reject 几乎持平 |
| uniform loss 在 Energy exposure 之上可能提升排序 | 有单 seed / 三 seed 初步信号，但收益有限且工作点指标不稳 |
| pure-unknown feature rejector 结果代表真实混合开放环境 | 不成立；这是纯 unknown 池上限 |
| strict mixed PU 当前实现改善了检测 | 当前保存的单 seed结果不支持；但不可外推为 PU 方法普遍无效 |
| semantic-hard/isolated 的类别 split 构造符合定义 | 脚本与 JSON 元数据支持 |
| 现有三协议性能差异由 split 语义造成 | 不能判断；历史 detector mode 不一致，且只有一个 seed |
| 历史实验完全可复现 | 不能保证；缺少源码 hash、部分训练日志和执行环境快照 |

## 推荐的证据修补顺序

1. 先用已有 semantic split checkpoints 只重跑检测，统一 `normalized_entropy_mahalanobis` 或另一预注册 score，不重训；这能隔离当前最明显的检测器混杂。
2. 之后才对 random / hard / isolated 做至少 3-seed 的同协议比较，并保留已知类别覆盖、score mode、阈值校准与 checkpoint 选择完全一致。
3. 若需要重新验证 OE 本身，增加 `energy=0, uniform=0`，与 `energy>0, uniform=0`、`energy>0, uniform>0` 三组配对，区分 Energy 暴露效应和 uniform 的增量效应。
4. 对 rejector 后续主张，先固定同一个 student checkpoint 与特征，把 pure-unknown、mixed-hard、mixed-PU 作为不同训练协议分层报告；未知标签仍只能用于事后测试诊断。
5. 新运行保存 git commit、完整命令、依赖版本、训练 history、validation calibration 信息和全部指标；README 历史记录保留，但按“已复核 / 部分复核 / 探索性”标记。

## 本轮没有做的事

没有重训、没有重写历史指标、没有修改模型代码或 README，也没有推送 GitHub。上述新结论是对已存在本地 artifacts 和当前调用链的静态审计；第 4 节所建议的同 detector 重评是下一步实验，而非本轮已完成内容。

## 后续补充：三种类别协议统一评分器复评（2026-09-29）

### 实验问题与协议

问题：之前 random / semantic-hard / semantic-isolated 结果差异是否只是 detector 不一致造成？

- 固定：三个已保存 student checkpoint，CIFAR-100、seed 42、`limit_train=1200`、`limit_val=300`、`limit_test=1000`、ResNet-18、MC=4、`known_coverage=0.95`、跳过聚类。
- 唯一计划变量：类别 split JSON。统一 `score_mode=normalized_entropy_mahalanobis`。
- 仅重新执行检测，不重训；测试标签只由程序用于最终 AUROC/FPR95/OSCR 等事后指标，未用于选择 score 或阈值。
- 执行设备：CUDA，NVIDIA GeForce RTX 4060 Laptop GPU。

### 统一评分器结果

| split | AUROC | AUPR | FPR95 | OSCR | known accuracy | test known accept | unknown reject | calibration known accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.5577 | 0.4185 | 0.8826 | 0.1476 | 0.2083 | 0.9273 | 0.0709 | 0.9500 |
| semantic-hard | 0.4823 | 0.4013 | 0.9761 | 0.0773 | 0.1299 | 0.9333 | 0.0602 | 0.9500 |
| semantic-isolated | 0.6188 | 0.4802 | 0.8469 | 0.1549 | 0.2013 | 0.9434 | 0.0802 | 0.9500 |

复评输出分别保存在 `runs/protocol_audit_unified_random_s42`、`runs/protocol_audit_unified_hard_s42`、`runs/protocol_audit_unified_isolated_s42`，其中 `config.json` 保存实际 detector 与数据配置，`discovery_report.json` 保存指标。

### 解释与限制

- 统一 detector 后，semantic-hard 仍是三者中最难：AUROC 最低、FPR95 最高、OSCR 最低；semantic-isolated 反而优于 random。方向符合“语义相似未知更难”的假设，说明此前不同 detector 不是全部差异来源。
- 但不能把这批结果当作类别 split 的干净因果实验。训练配置并不完全一致：random 与 hard 使用 `known_split_mode=random`，isolated 使用 `stratified`；三模型均只有单 seed、3 epoch 和缩小训练预算，且训练随机性并未做配对控制。故结果是统一评估条件下的探索性 split 敏感性证据，不是统计结论。
- 测试 known acceptance 未精确等于 95%（92.7%–94.3%），这是有限验证校准集在独立测试集上的实际覆盖率偏差；不得再用测试标签重调阈值。比较未知拒绝率时应同时看该列，不能假设每组 operating point 完全相同。
- 统一 detector 也没有解决核心问题：三组 AUROC 仍低，未知拒绝率仅约 6.0%–8.0%，而且 known accuracy 很低。它说明协议难度会影响表现，但目前证据不足以把核心瓶颈归因于 split，而非表征/训练质量。

### 下一步

如继续追求可归因结论，需用相同训练配置重新训练三种 split（特别是统一 `known_split_mode`、teacher 初始化/训练、epoch、checkpoint 选择），至少 3 个 seed，再沿用本轮固定 detector 和 known-only validation 校准。当前这批 checkpoint 只适合定位风险、不能代替该正式实验。

## 后续补充：三划分统一训练的配对小规模复核（2026-09-29）

### 实验目的与协议

目的：判断 semantic-hard 相对于 random 是否在统一训练和检测设置下仍表现更差，并检查差异是否跨随机种子重复。

- 3 类别 split × 3 seeds（42、123、3407），每个 seed 在三种 split 间配对；总计 9 套 teacher/student 训练及检测。
- 所有模型从相同架构的预训练 ResNet-34 teacher / ResNet-18 student 起步，teacher 与 student 各训练 3 epochs；训练样本预算 1200，验证 300，测试 1000；batch 32，seed 对应的数据顺序与增强由程序统一控制。
- 所有 student 固定 `kd_mode=uncertainty`、`alpha_kd=1`、`alpha_unc=0.1`、`alpha_feat_kd=0.1`、`alpha_proto=0.1`、`alpha_supcon=0.1`；类别切分统一 `known_split_mode=random`。checkpoint 规则相同：按 known validation accuracy 选择。
- 检测全部固定 `normalized_entropy_mahalanobis`、MC=4、已知验证集 95% coverage 校准、跳过聚类。阈值不看未知测试标签。
- 预定的实验变量：类别 split 与 seed；按每个 seed 配对观察 split 差异。使用 GPU（RTX 4060 Laptop GPU）。
- 这是控制较好的**短程/小样本敏感性试验**，不是全数据、长训练的最终性能实验。

运行脚本：`scripts/run_controlled_protocol_audit.ps1`。每组完整配置、训练历史、权重及报告存于 `runs/protocol_controlled_{random,hard,isolated}_s{42,123,3407}_{teacher,student,detect}`。

### 三 seed 结果（均值 ± 样本标准差）

| 类别协议 | AUROC | AUPR | FPR95 | OSCR | known accuracy | test known accept | unknown reject |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 0.5499 ± 0.0335 | 0.4312 ± 0.0354 | 0.9016 ± 0.0187 | 0.1431 ± 0.0279 | 0.2029 ± 0.0317 | 0.9418 ± 0.0143 | 0.0602 ± 0.0174 |
| semantic-hard | 0.5048 ± 0.0355 | 0.4149 ± 0.0274 | 0.9437 ± 0.0095 | 0.1280 ± 0.0299 | 0.1932 ± 0.0347 | 0.9444 ± 0.0036 | 0.0535 ± 0.0141 |
| semantic-isolated | 0.5613 ± 0.0656 | 0.4175 ± 0.0596 | 0.8639 ± 0.0422 | 0.1148 ± 0.0290 | 0.1608 ± 0.0511 | 0.9294 ± 0.0305 | 0.0761 ± 0.0488 |

逐 seed AUROC 顺序（random / hard / isolated）：

| seed | random | hard | isolated | hard − random |
|---:|---:|---:|---:|---:|
| 42 | 0.5153 | 0.4659 | 0.6093 | -0.0494 |
| 123 | 0.5821 | 0.5129 | 0.5880 | -0.0692 |
| 3407 | 0.5522 | 0.5354 | 0.4866 | -0.0167 |

### 结果解释

- 三个 seed 中 semantic-hard 的 AUROC 都低于配对 random；均值差为 `-0.0451`。其平均 FPR95 高 `0.0421`、unknown reject 低约 `0.0067`。因此，在这套短程控制训练和固定 detector 下，“语义相似未知更难”得到一致方向的初步证据。
- 这**没有解决核心检测瓶颈**：random 的 unknown reject 均值只有约 6.0%，hard 约 5.4%，仍有超过九成未知样本被接收；AUROC 也仅 0.50–0.56。
- semantic-isolated 没有稳定地“更容易”：它的 AUROC seed 波动最大（SD 0.0656），且 known accuracy 均值更低、OSCR 也最低。不能把 semantic-isolated 仅凭 AUROC 均值称为全面更好。
- `known_split_mode`、模型结构、损失、数据预算和检测器在本次九组配置中一致，修复了上一轮旧 checkpoint 对照中的关键配置混杂。但 pretrained 权重训练、样本增强、优化器顺序仍有随机性；三个 seed 及每组 3 epoch 远不足以支持强统计结论或推广到完整训练。
- 测试 known acceptance 均值约 92.9%–94.4%，并非精确 95%。unknown reject 是在各自验证集已知覆盖校准阈值下的测试表现，需与 known acceptance 一并解释，不能用测试标签重校准。

### 审计更新与建议

本轮不再只是“统一评分器下旧模型差异”，而是统一训练协议的配对小实验。此前旧 checkpoint 的统一评分结果仍保留为历史探索，不用于因果结论。本次结果增强了“类别语义难度影响开放集表现”的证据，但核心问题仍是分离能力差，不能以换 split 代替改进算法。

下一步应优先扩大 random 与 semantic-hard 两协议的训练预算（完整已知训练数据、足够 epoch），先用 seed 42 做资源可行性/指标验证；若训练稳定，再至少补两个配对 seed。之后比较特征重叠、按 coarse superclass 分层误接收率和 known validation 校准差异，判断 hard split 的损失究竟来自特征相似性、known classifier 泛化不足，还是评分器失配。semantic-isolated 可作为补充泛化协议，不应替代 hard split。

## 后续补充：完整训练预算的 random / semantic-hard pilot（2026-09-29）

### 本次要回答的问题与控制条件

问题：此前 3-seed 对照使用 1200 个训练样本和 3 epochs，是否因训练预算太小而低估模型表现？在完整 CIFAR-100 已知训练子集、5 epochs 下，semantic-hard 与 random 的差异是否仍存在？

- 配对条件：seed 42；ResNet-34 预训练 teacher、ResNet-18 预训练 student；完整已知训练数据（未设置 limit）；batch size 64；teacher/student 均 5 epochs；相同 checkpoint 选择规则（known validation accuracy）；同一 `normalized_entropy_mahalanobis` 检测器、MC=4、known validation 95% coverage、跳过聚类；同一份代码与运行脚本。
- 预定变化：CIFAR-100 的 known/novel class split（random vs semantic-hard）。类别构成本身改变训练样本，故它是协议对照，不是同一批图像上的纯单因素模型 ablation。
- 训练配置澄清：脚本显式设置 uncertainty KD、feature KD、prototype 与 SupCon 权重；但教师/学生的其他辅助损失沿用 CLI 默认值。因此两组比较的是同一套**当前默认训练损失组合**，不能描述成只包含四项损失的最小基线。
- 测试集标签仅用于最终 AUROC/FPR95/OSCR、拒绝率等事后评估；阈值按各自 known validation 校准，未用 novel test labels 选模型或调阈值。

运行脚本：`scripts/run_full_controlled_protocol_pilot.ps1`。产物分别位于 `runs/protocol_fullpilot_{random,hard}_s42_{teacher,student,detect}`。两组 config 均记录全量数据、5 epochs、相同模型/损失参数和检测参数；teacher/student 的 best epoch 均为第 5 / 第 2（random）及第 5 / 第 3（hard），best student validation accuracy 分别为 0.430 / 0.356。

### 结果

| 协议 | AUROC | FPR95 | OSCR | 测试 known accuracy | 测试 known accept | 未知 reject | 校准集 known accept |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 0.6026 | 0.8683 | 0.2955 | 0.3987 | 0.9475 | 0.0690 | 0.9500 |
| semantic-hard | 0.5179 | 0.8958 | 0.2270 | 0.3358 | 0.9550 | 0.0400 | 0.9500 |

在固定 operating point 下，hard 相对 random 的 AUROC 低 0.0847、OSCR 低 0.0685、未知拒绝率低 0.029；known accuracy 也低 0.0628。检测 score 的测试分位数（顺序为 0/10/50/90/100%）显示两类仍明显重叠：random known `[-3.221,-1.546,0.086,1.494,6.483]`、unknown `[-2.813,-0.989,0.498,1.722,6.736]`；hard known `[-3.264,-1.602,0.010,1.455,7.908]`、unknown `[-3.022,-1.240,0.044,1.390,5.502]`。unknown 中位数虽偏高，但低分尾部落入 known 区间，单一分数阈值无法把两群分开。

### 解释、自检与边界

- 完整数据预算使 random 的 AUROC / known accuracy 高于此前小样本 pilot，但 unknown reject 仍只有 6.9%；hard 的未知拒绝率更低，为 4.0%。因此增加数据与 epoch 有助于分类/排序的一部分，却没有解决“未知大多被接收”的核心失败。
- 这是单 seed、5 epochs 的 pilot；它支持“semantic-hard 在此配置下更困难”的方向性观察，不足以作统计或普遍因果结论。训练轮数也仍短，且 best validation accuracy 仍较低。
- 两组在训练配置、数据预算、检测器及校准策略上相同。由于类别 split 会改变类别组合和对应训练图像，不能声称只有输入图像完全相同；此处结论限定为“配对 split 协议比较”。
- 报告里的 classwise test acceptance 是事后诊断，不可用于调阈值。两组中都出现几乎 100% 被接受的 novel 类别，说明失败并非少量阈值边缘样本造成；后续应分析这些类别的 feature neighborhood / 与最近 known superclass 的混淆，而非继续收紧阈值。
- 本轮没有修改模型算法或 README，也没有上传仓库。结果只记录在此审计文档。

### 下一步的算法判断

先停止同一 known-only Mahalanobis score 上的阈值微调。下一步应在**固定 split、seed、训练预算、student checkpoint、known-only 校准**下，做一个有明确训练机制差异的小对照：选择单一、可追溯的 unknown exposure 来源/目标（优先审查代码已有的 outlier exposure / mixed-pool PU 实现，明确是否使用 oracle novel labels），与当前损失组合比较；如果 exposure 的候选池含有 CIFAR-100 novel 标签筛出的样本，只能作为 oracle 上限，不能声称符合无标注开放场景。评估同时报告 AUROC、FPR95、OSCR、known accuracy、known acceptance 与 unknown rejection，并事后核对 score 分布及高误接收类别。任何提升若伴随大量 known false rejection，或只来自改变 operating point，都不算解决核心问题。

## 后续补充：独立 CIFAR-10 Outlier Exposure uniform loss pilot（2026-09-29）

### 目的与协议

目的：测试独立辅助图像源的均匀已知类输出约束，是否能让未知样本不再被已知分类头高置信接收。此尝试沿用 Hendrycks et al., *Deep Anomaly Detection with Outlier Exposure* (ICLR 2019) 的 auxiliary outlier exposure 方向；代码具体采用的是对辅助图像 logits 做 uniform-softmax loss，并非完整复现论文所有训练和 benchmark 设置。

- baseline 是上节已完成的 full-data random seed-42 student；OE 组复用**同一 teacher checkpoint**和完全相同的 CIFAR-100 random split、known train/val/test、架构、5 epochs、基础 KD / uncertainty / feature KD / prototype / SupCon 参数与优化配置。
- 唯一有意改变的训练项：增加无标签 CIFAR-10 batch，`alpha_outlier_uniform=0.05`。CIFAR-10 标签不进入 loss；OE 图像仅令 student 对 60 个 known logits 输出均匀分布。
- 检测两组均使用 `normalized_entropy_mahalanobis`、MC=4、各自 known validation 95% coverage threshold、跳过聚类。阈值没有使用未知测试标签。未知标签只用于事后指标。
- OE 使用额外前向和随机增强，可能改变全局 RNG 消耗及 CIFAR-100 batch augmentation 序列。因此这是匹配配置/目标的单 seed pilot，不是严格控制每个随机数流的配对因果实验。
- CIFAR-10 是与 CIFAR-100 不同数据集来源，但存在语义类重合可能（例如动物、车辆）；这是“不完美/含潜在近分布样本”的辅助 OE，不应称为保证语义完全 disjoint 的异常集。

运行脚本：`scripts/run_full_cifar10_oe_pilot.ps1`。OE 训练与检测产物位于 `runs/protocol_fullpilot_random_s42_oe_cifar10_u005_{student,detect}`。config 确认辅助数据集为 CIFAR-10、uniform loss 权重为 0.05；训练 history 中 `outlier_uniform` 非零，best epoch 4，best known validation accuracy 0.459。

### 结果：同一固定 score/calibration 的事后测试比较

| 训练方案 | AUROC | AUPR | FPR95 | OSCR | 测试 known accuracy | 测试 known accept | unknown reject | 校准 known accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 基线（无 CIFAR-10 OE） | 0.6026 | 0.4667 | 0.8683 | 0.2955 | 0.3987 | 0.9475 | 0.0690 | 0.9500 |
| CIFAR-10 OE uniform, weight 0.05 | 0.6352 | 0.4920 | 0.8262 | 0.3435 | 0.4448 | 0.9485 | 0.0818 | 0.9500 |
| OE − baseline | +0.0326 | +0.0252 | −0.0422 | +0.0480 | +0.0462 | +0.0010 | +0.0128 | 0.0000 |

### 自检解释

- 多个排序指标、unknown rejection 与 known accuracy 同向改善，known acceptance 几乎不变；故目前信号不像单纯“把阈值调严、用更多已知误拒换未知拒绝”。OE 训练日志也证实对应分支实际执行。此结果值得保留为较有希望的方向。
- 但未知拒绝率仍仅 8.18%，约 91.82% novel test 样本仍被接受。它只是小幅改进，核心已知/未知重叠依旧严重，不能写成“解决未知检测问题”。
- 这是 n=1、5 epochs、单一 loss weight、单一 detector 的试验；student 训练随机数流无法逐操作匹配，且 CIFAR-10 语义可能和 CIFAR-100 类别重合。不能据此声称稳健收益或归因于“语义完全未知的外部异常”这一机制。
- full-data Mahalanobis 统计阶段很慢（每个完整检测运行数分钟级）。该计算没有改变指标，但应作为实验效率问题记录；后续复核可保留统一 detector，考虑用同一特征上更快的 diagonal Mahalanobis 做探索性 ablation，不能把不同 score mode 的结果混在主对照表中。

### 后续验证

1. 固定 `alpha_outlier_uniform=0.05` 不动，先补至少两个 seeds，复用同 split、同 teacher/student 训练轮数、相同 known-only calibration 和指标；若收益方向不一致，不继续扩大投入。
2. 再视多 seed 结果比较 0 / 0.025 / 0.05 的小型 weight ablation；只能用 known validation 选择训练 checkpoint/权重，测试 novel labels 不得参与选择。
3. 同时查看 score 分布与分类型拒绝率、known false reject；如 AUROC 提升而 matched known coverage 下 unknown recall 不提升，则停止把该 OE loss 作为核心算法方案。
4. 若 CIFAR-10 OE 信号可复现，再使用语义类重合更少、可明确追踪来源的辅助数据协议验证迁移性；把它标作辅助 OOD exposure，不等同于无标注 mixed-pool 新类发现。

## 后续补充：CIFAR-10 OE uniform 的 seed-123 配对复核（2026-09-29）

### 实验问题与协议检查

问题：seed-42 pilot 中 `alpha_outlier_uniform=0.05` 的改善能否在另一个 seed 重复？本次只检验固定 0.05 的可重复方向，不搜索权重、不挑测试阈值。

- seed 123、固定 CIFAR-100 random 60/40 split、完整训练集、ResNet-34 teacher / ResNet-18 student、5 epochs、相同 uncertainty KD / feature KD / prototype / SupCon 参数与检测器。
- baseline 与 OE variant 复用同一个 seed-123 teacher checkpoint。student 两组使用相同训练命令参数；唯一计划差异为 OE 组启用 CIFAR-10 unlabeled auxiliary loader 以及 `alpha_outlier_uniform=0.05`。
- 固定 `normalized_entropy_mahalanobis`、MC=4、known validation 95% coverage calibration、skip clustering。最终测试标签仅用于事后报告。
- 验证配置和训练 history：baseline 的 OE dataset 空、uniform 权重 0、loss 为 0；OE 组 `outlier_dataset=cifar10`、uniform 权重 0.05、loss 非零。OE 组 best epoch 4 / validation acc 0.4710；baseline best epoch 5 / 0.4753。
- 局限：单 seed；辅助数据额外消耗随机状态、增强和前向，故不是逐样本 RNG 完全配对的严格因果干预。CIFAR-10 与 CIFAR-100 类语义可能重合。

### 配对结果

| seed 123 variant | AUROC | AUPR | FPR95 | OSCR | known accuracy | test known accept | unknown reject | validation known accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.6479 | 0.5190 | 0.8268 | 0.3699 | 0.4750 | 0.9472 | 0.1100 | 0.9500 |
| CIFAR-10 OE uniform 0.05 | 0.6237 | 0.4761 | 0.8203 | 0.3482 | 0.4600 | 0.9527 | 0.0623 | 0.9500 |
| OE − baseline | −0.0242 | −0.0429 | −0.0065 | −0.0217 | −0.0150 | +0.0055 | −0.0478 | 0.0000 |

### 结合 seed 42 的自我检查结论

- seed 42 的 unknown reject 为 6.90% → 8.18%（+1.28 个百分点），AUROC +0.0326；seed 123 则 unknown reject 为 11.00% → 6.23%（−4.78 个百分点），AUROC −0.0242。故 OE 0.05 的“提升”没有跨 seed 重复，不能把 seed-42 单次信号当作方法有效。
- seed 123 的 FPR95 有小幅改善，但 AUROC、OSCR、known accuracy 和 unknown rejection 均下降；这是重要反例，也再次说明不可只挑一个指标宣称有效。
- baseline 自身 seed 间差异明显：seed 42 AUROC/known accuracy/reject 为 0.6026/0.3987/0.0690，seed 123 为 0.6479/0.4750/0.1100。因此需要 paired multi-seed summary，而非跨 seed 把某一组拼起来比较。
- 目前合理结论：CIFAR-10 uniform OE 是**未证实有效、效果 seed-sensitive** 的候选；它没有解决核心 feature/score overlap。暂不继续用更多权重网格追逐最佳单 seed。

脚本：`scripts/run_oe_paired_seed_pilot.ps1`。seed-123 产物在 `runs/protocol_oe_pair_s123_{teacher,baseline_student,baseline_detect,oe_u005_student,oe_u005_detect}`。脚本语法和 `git diff --check` 通过；全套测试 `94 passed`（1 个 joblib CPU 核数提示，不影响测试）。

### 后续决策

先按预先固定的协议补 seed 3407（同一 teacher 复用、baseline/OE 配对、0.05 不变），然后汇总三 seed 的 paired delta 与方差。若第三 seed 同样无一致收益，停止此 uniform-OE 方向，不再扫权重。之后转去分析 unknown 被接收的主要原因：按 CIFAR-100 coarse superclass 统计误接收，比较 embedding/projection 的近邻类别组成及 known-vs-novel score 分布；再选一个有理论动机且只改单一机制的表示学习实验。不得依据测试集标签选择 score mode、loss 权重或阈值。
