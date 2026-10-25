# 全部程序与研究主题审计（2026-10-24）

## 审计结论

当前项目已经具备可追溯的开放集新类发现实验框架，但还不能把“uncertainty-KD”称为已经验证的新类发现主方法。当前最可靠的核心问题定位是：semantic-near novel 与 known 在共享表征、拒识表征和最终拒识分数上严重重叠，导致 unknown rejection 和自动新类聚类不能稳定同时改善。

因此当前主线应是：先保证实验协议和代码路径严格一致，再验证合法的 hard-unknown 信号和真正独立的 image-to-rejection 表征；uncertainty 训练期重加权只保留为显式 opt-in 对照，validation-calibrated uncertainty 只保留为候选拒识/gating 信号。

## 本次发现并修复的实现问题

### 1. calibration overlap-control 参数曾在 train_student 中丢失

`parse_args` 和配置文件可以保存 `calibration_overlap_control=true`，但 `fit_student` 构造 `DataBundle` 时没有传递该参数，实际仍使用 disjoint calibration split。这会使诊断实验的条件名与实际数据协议不一致。

已修复：`train_teacher`、`train_student`、`discover`、`inspect_data` 统一调用 `train.py::build_bundle_from_args`，由单一入口转发 split、open-validation、calibration、mixed-pool 和 transform 参数。新增回归测试确认 overlap-control 参数真实传入。

### 2. uncertainty-KD 曾是 train_student 的隐式默认值

研究审阅和 README 已经决定 uncertainty-KD 暂不作为默认方法，但 CLI 默认仍为 `kd_mode=uncertainty`。这会让未显式指定参数的新运行自动使用尚未被验证的主方法。

已修复：默认改为 `kd_mode=standard`；`--kd-mode uncertainty` 仍可显式启用并用于配对消融。新增测试锁定“uncertainty-KD 必须显式 opt-in”。已有保存的实验配置不被修改。

### 3. strict mixed rejector 的 hard 训练条件不合法

`--rejector-strict-mixed` 使用 reserved open-validation known 样本作为 known side，但 discovery pool 在 mixed 模式下仍含有 known 样本。若同时使用 `rejector-training=hard`，整个 mixed pool 会被赋予 hard unknown 标签，产生已知样本污染，不能称为严格的 mixed-pool 拒识实验。

已修复：该组合现在直接报错；mixed pool 必须使用 `soft_pu` 或 `nnpu`，或者关闭 strict mixed 后明确承认使用 supervised-known support 的协议。新增测试覆盖该非法组合。

## 已确认可信的程序路径

- class split 会检查覆盖、互斥和重复；checkpoint 会检查 known/novel class metadata。
- mixed discovery pool 从 supervised known train 中保留不重叠子集；validation/test 不作为训练样本。
- `extract_outputs` 只提取模型输出；test/mixed-pool 的 `labels`、`raw_labels` 和 `is_known` 仅用于最终回顾或 validation diagnostics，不进入训练损失、阈值校准或自动 K 选择。
- 默认阈值使用 known validation；95% coverage 校准和 classwise threshold 都不需要 unknown/test labels。
- automatic K 使用候选池几何和内部聚类指标；oracle-K 以及 test-label cluster diagnostics 单独记录，不能用于方法选择。
- independent rejection encoder 确实从原始图像走独立参数路径；其梯度隔离、初始化隔离和 shared-feature fallback 已有测试覆盖。
- CE-only 三臂已完成并通过 runner 审计；feature KD、rejection-branch KD 和 discovery/outlier/rejection 训练均按定义关闭。
- uncertainty-KD 的 teacher uncertainty 已 detach，standard-KD 与 uncertainty-KD 的 logits KD 路径可被训练日志和权重 spread 审计。

## 仍需在研究解释中明确的限制

- `score_mode=auto` 或 open-set checkpoint selection 会使用 reserved open-validation 的 known/unknown identity；代码已有 warning，但这类运行只能作为 diagnostic protocol，不能和 known-only deployment protocol 混写。
- test labels 会出现在最终 `discovery_report` 的 ACC/NMI/ARI、unknown rejection 等回顾指标中，这是评估所必需的，但不能反向选择 score mode、threshold、purification、fusion weight 或 checkpoint。
- `feature_rejector` 如果使用 discovery pool 的 hard unknown 标签，只能在 pure-unknown pool 中解释为 hard control；mixed pool 必须采用 PU/nnPU 或明确标注为污染控制。
- scratch-CNN 与 shared-backbone 的 rejection encoder 结果同时改变了架构和初始化，不能作为纯初始化因果证据；同架构的 shared-backbone vs random-backbone 才是有效初始化对照。
- 排序指标（AUROC、AUPR、KS、FPR95）改善不能直接等价于固定 known-coverage 工作点 unknown rejection 或 known utility 改善。

## 研究问题与修改优先级

### P0：协议真实性（已基本完成，继续保持）

每个正式实验必须有唯一变化因素、保存完整 command config、split metadata、checkpoint、calibration report 和结果摘要。任何配置参数若未真正传入数据或训练函数，必须先修复再解释结果。

### P1：hard-unknown 信号审计

只使用 known train/validation、CIFAR-100 coarse taxonomy 和模型预测，测量候选覆盖、known contamination、跨 seed 稳定性以及 semantic-near 类别的错误结构。不能使用 mixed-pool/test 真值选择候选规则。

### P2：单独验证 image-to-rejection 表征

若继续训练期方法，只引入一个独立机制：真正独立输入的 rejection encoder，或一个已知类条件 support/boundary 约束。不要同时叠加 VOS、OE、PU boundary、candidate gating 和 uncertainty-KD，否则无法解释增益来源。

### P3：validation-calibrated uncertainty 的 opt-in 复核

可以在冻结 checkpoint 上验证 uncertainty-to-error calibration 和 rejection fusion，但固定 fusion 规则，不能按 test 结果重新调权重。必须联合报告 rejection、known utility、automatic K、ACC/NMI/ARI 和 overlap。

## 当前停止标准

- 不再在旧 split 上继续扫描 `exp(-u)` 或其他 uncertainty-KD weighting function。
- 不把单 seed 正向结果写成核心方法收益。
- 不把自动聚类 oracle-K 或 test-label diagnostics 写成无监督方法性能。
- 若新的单因素机制在独立 fresh split 上不能稳定改善固定 coverage 的 unknown rejection，同时保持 known utility 和聚类质量，则停止该机制，回到 hard-unknown 表征/信号审计。

## 验证记录

- `pytest -q`：302 passed，2 个既有 warning。
- `python -m compileall -q train.py novel_discovery tests`：通过。
- 补充协议保护：普通 `mixed + rejector-training=hard + rejector-pool-selection=all`
  现在也会直接报错；只有显式 high-risk/weighted-boundary 候选伪未知实验可以
  保留该 hard 控制，并必须在结果解释中标明其不是全池无监督 hard-label 协议。
- 补充工具审计：`scripts/audit_rejection_representation.py` 已修复为按 case
  重建 independent rejection encoder；seed=2040 的复核显示独立分支确实降低
  shared-to-rejection 可预测性，但有效秩和拒识探针性能同步下降，暂不恢复该架构。
- known-only 复核进一步确认：seed=2040 independent scratch CNN 的 raw
  image-to-rejection train→validation known-class probe 仅为 `31.67%`，最终
  rejection feature 为 `20.59%`；seed=2051 的 scratch/random 两臂同样低于
  `30%`，而 shared-pretrained ResNet18 为 `70.96%`。因此当前首要问题是
  rejection branch 的容量/初始化与训练利用，而不是继续堆叠新的拒识损失。
- 本轮新增 deterministic train/validation split 审计、raw rejection-input 输出和
  配对 JSON/Markdown 记录；未启动新的 GPU 训练，也未用 test 标签选择方法。
- `git diff --check`：通过（仅 CRLF 提示）。
- 本次没有启动新的 GPU 训练，也没有修改或删除有效实验结果；已确认无效的运行仍保留在归档目录中。
