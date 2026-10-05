# CIFAR-100 全量 GPU 重训报告（2026-10-05）

## 协议

- 数据：仓库内 CIFAR-100 ImageFolder，train 50,000、test 10,000，每类分别 500/100。
- 固定随机 60/40 类划分：`splits_cifar100_60_40_imagefolder_imagefolder.json`（ImageFolder loader 会把传入 split 文件名转换为该文件名）。
- Seeds：42、123、3407；known 训练验证按类别分层；mixed discovery pool，保留 known 训练池比例 0.2，nnPU prior 自动按实际 pool 估计。
- 由于 RTX 5060 Laptop 8GB 与速度约束，本次采用 32×32 输入、batch 128、teacher/student 各 5 epochs、ImageNet pretrained ResNet-34/ResNet-18。全部训练图像和测试图像均参与，不使用 `limit-*`。因此这是完整数据、多 seed 的短轮次重训，不等同于仓库历史 64×64/更长训练协议。
- 每 seed 同一教师训练 student；loss 包括 uncertainty KD、feature KD 0.1、SupCon 0.1、mixed nnPU uncertainty 0.1。检测 MC=8、normalized entropy + min-class kNN（k=10）、known-only 95% coverage threshold。每 seed 的 auto-K 与 oracle-K 分开运行。
- 最佳 student 按 known validation accuracy 选择。seed 3407 训练过程只留下逐轮权重，故按 config/history 中 best epoch 3 选取 `student_epoch_3.pt` 作为其检测 checkpoint；已修复代码使以后启用逐轮存档时同步保存 best `student.pt`。

## 三 seed auto-K 结果

指标为 mean ± sample std；比例以 0–1 表示。

| 指标 | mean ± std |
|---|---:|
| AUROC | 0.5371 ± 0.0073 |
| AUPR | 0.4238 ± 0.0062 |
| FPR95 | 0.9058 ± 0.0117 |
| OSCR | 0.1495 ± 0.0083 |
| Known accept rate | 0.9508 ± 0.0017 |
| Unknown reject rate | 0.0602 ± 0.0047 |
| Candidate pool purity | 0.4489 ± 0.0142 |
| Candidate unknown NMI（各 seed 候选子集 oracle 评估） | 0.4888 ± 0.0025 |
| Candidate unknown ARI（各 seed 候选子集 oracle 评估） | 0.0426 ± 0.0074 |
| 全未知子集 NMI（oracle K） | 0.2530 ± 0.0056 |
| 全未知子集 ARI（oracle K） | 0.0670 ± 0.0044 |
| auto-K（真实 K=40） | 41.33 ± 8.96 |

各 seed auto-K 分别为 47、31、46；虽然均值接近 40，但单次波动较大。known coverage calibration 达到目标，不意味着 unknown rejection 好：实际未知拒识仅约 6%，FPR95 约 91%，检测能力仍不足。Oracle-K 并未消除聚类困难；候选池 NMI 约 0.49，但 ARI 约 0.04，真实未知全体聚类 NMI 约 0.25。

## 结果解释与代码修复

本轮重训中发现并修正 ImageFolder 的类别键类型问题：`ImageFolder.targets` 是整数，但已知类映射按类名字符串建立。原实现使 stratified 标签映射报错，并会将已知样本误当 unknown 子池，导致 mixed-pool known prior 错误。修复后 seed 42/123/3407 的实测 prior 均约为 0.2126。增加了针对标签映射、unknown 子池及 mixed prior 的回归测试。

结果表明此 5 epoch、32px 配置下分类/开放集表征没有充分收敛；这次收尾不宣称性能已改善。后续若要提升，应在固定 split 上增加 epoch、评估较高分辨率或调整预训练微调策略，并继续保持配对多 seed 验证，而不是继续调阈值。原 64px/10 epoch 尝试在完成首轮 batch 后因耗时过高而中止，没有纳入结果。

## 产物

- 每个 seed 的 teacher/student checkpoints、epoch checkpoints、train history 和 config 位于 `runs/full_imagefolder32_s{42,123,3407}_{teacher,student}/`。
- 每个 seed 的 auto-K 和 oracle-K 检测结果分别位于 `runs/full_imagefolder32_s{42,123,3407}_detect_{auto,oracle}/`。
- 机器可读跨 seed 指标：`analysis/cifar_full_retrain_20261005.json`。
- 训练入口与数据协议修复：`train.py`、`novel_discovery/data.py`；回归测试：`tests/test_core_behaviors.py`。
