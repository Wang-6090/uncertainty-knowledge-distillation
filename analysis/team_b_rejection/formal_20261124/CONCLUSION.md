# 李柯颖线正式结论（formal_20261124）

> 未知标签只用于事后评测，未参与训练、阈值校准或模型选择。

## 0. 一句话结论

未知检测的瓶颈主要在**风险分数本身的排序能力**（AUROC 0.53）和**阈值附近的样本级不稳定**
（约 13% 测试样本的候选标记会随种子翻转）；拒识器初始化方式（`scratch_cnn` vs
`shared_backbone`）在三个正式种子上**没有稳定收益**，按第 10 节停止规则保留 `scratch_cnn`。

## 1. 这次实际跑的是什么

| 项 | 值 |
| --- | --- |
| 数据 | CIFAR-100 imagefolder（`../uncertainty-knowledge-distillation1.0/CIFAR-100-dataset-main`） |
| 划分 | `splits/cifar100_60_40_imagefolder.json`，60 已知 / 40 未知 |
| 教师 | ResNet-34，30 epoch，每种子上限独立训练 |
| 学生 | ResNet-18，`--independent-rejection-encoder`，两种初始化各一遍 |
| 种子 | 42、123、3407 |
| 风险分数 | `normalized_entropy_mahalanobis` |
| 阈值 | 只用已知验证集（3000 张），`known_val_global` q95 |
| 评测 | 10000 张测试样本，`mc_samples=8` |
| 聚类（旁证） | `feature_pca` + `kmeans`，oracle K=40 |

### 1.1 协议偏离（必须写明）

学生训练没有跑满 30 epoch，只用了 **16 epoch 预算**。原因：这台机器在长时间满负荷 GPU
下三次触发 `BugCheck 0x116 VIDEO_TDR_FAILURE`（显卡驱动超时未恢复导致系统重启），
`train.py` 又只在训练结束才写 `student.pt`。补救方式是用
`scripts/promote_epoch_checkpoints.py` 把每个 epoch 的 `student_epoch_<n>.pt` 按训练
日志里的 `checkpoint_selection` 选出最佳 epoch 还原成 `student.pt`。

六个配置**统一使用 16 epoch 预算**、在同一预算内选最佳 epoch，因此彼此可比，但
**绝对值不能与 30 epoch 的历史结果并表**。

| 配置 | 实际使用 epoch | 来源 |
| --- | ---: | --- |
| s42 scratch_cnn | 16 / 16 | 快照还原（选择分 0.4527） |
| s42 shared_backbone | 16 / 16 | 训练完整结束，正常保存 |
| s123 scratch_cnn | 14 / 16 | 快照还原（0.4360） |
| s123 shared_backbone | 13 / 16 | 快照还原（0.4093） |
| s3407 scratch_cnn | 6 / 16 | 快照还原（0.4163） |
| s3407 shared_backbone | 16 / 16 | 快照还原（0.4167） |

`s3407 scratch_cnn` 的最佳 epoch 只有 6，说明该配置在 16 epoch 预算内就出现明显波动；
它的验证准确率（0.4227）也确实低于另两个种子的 scratch 配置（0.4567 / 0.4460）。

## 2. 主要结果

### 2.1 两种拒识器配置（均值 ± 标准差，n=3）

| 指标 | scratch_cnn | shared_backbone |
| --- | ---: | ---: |
| AUROC | 0.5370 ± 0.0061 | 0.5324 ± 0.0144 |
| AUPR | 0.4109 ± 0.0043 | 0.4088 ± 0.0050 |
| FPR95 | 0.9113 ± 0.0062 | 0.9141 ± 0.0060 |
| OSCR | 0.2853 ± 0.0134 | 0.2745 ± 0.0123 |
| 已知接受率 | 0.9432 ± 0.0011 | 0.9456 ± 0.0074 |
| 未知拒识率 | 0.0467 ± 0.0008 | 0.0439 ± 0.0015 |
| 候选池纯度 | 0.3540 ± 0.0025 | 0.3516 ± 0.0252 |
| 候选池聚类 ACC | 0.3272 ± 0.0403 | 0.3300 ± 0.0305 |
| 候选池聚类 NMI | 0.5968 ± 0.0304 | 0.6087 ± 0.0226 |
| 候选池聚类 ARI | 0.0980 ± 0.0620 | 0.0960 ± 0.0378 |

### 2.2 成对对照（shared − scratch，同 seed）

| seed | AUROC | OSCR | 池纯度 | 池召回 | 候选数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | +0.0126 | −0.0172 | −0.0136 | −0.0028 | −11 |
| 123 | −0.0270 | −0.0312 | +0.0239 | −0.0047 | −83 |
| 3407 | +0.0006 | +0.0160 | −0.0175 | −0.0008 | +18 |

方向一致性：AUROC 2 优 1 劣（局部信号）；AUPR / FPR95 / OSCR / 池纯度 各 1 优 2 劣；
**候选池未知召回 3/3 全部变差**。

## 3. 6.2 风险分数与阈值：结论

**分数分布高度重叠。** 已知/未知风险分数的 50 分箱重叠面积 0.894–0.935，
中位数间距只有 0.058–0.229。这正是 AUROC 只有 0.53 的直接原因：分数能把分布整体拉开一点，
但排不开绝大多数样本。

**阈值本身稳定，样本级标记不稳定。** 阈值落在 1.978–2.243（跨度 0.265），
已知验证集自助法 std 0.045–0.066（约为阈值的 2.5%–3.3%）。这个波动不大，
但代价很高：跨种子比较时，**12.7%–13.0% 的测试样本候选标记发生翻转**
（已知 826–842 个、未知 446–462 个），其中阈值邻域内的翻转率 53.8%–54.2%，
邻域外只有 8.1%–9.1%；翻转样本距阈值平均 1.20–1.25，稳定样本平均 2.18–2.28。

也就是说：**阈值数值可复现，但"哪些样本刚好过线"不可复现**。任何依赖候选池边界的下游结论
（新类数量、簇大小）都会继承这个不稳定性。

## 4. 6.3 两种拒识器配置：结论

`shared_backbone` 相比 `scratch_cnn` **未观察到稳定收益**：

- AUROC 在两个种子改善（+0.0126、+0.0006）、一个种子变差（−0.0270），属于局部信号；
- OSCR、FPR95、候选池纯度全部是 1 优 2 劣；
- **候选池未知召回在三个种子上全部变差**（−0.0028 / −0.0047 / −0.0008），
  即用学生主干初始化拒识编码器会让候选池更保守。

按第 10 节判据，这条只能写成"未观察到收益"，**默认流程继续用 `scratch_cnn`**
（它也更省事：不需要拷贝主干参数，参数量更小）。

## 5. 6.5 候选池误差分解：结论

| 误差项 | 量级 |
| --- | --- |
| 未知漏检（未知样本进不了候选池） | **95.3%–95.5%** |
| 已知污染（已知样本被误拒进池） | 占已知 5.6%–5.8%，占候选池 64.3%–66.4% |
| 跨种子标记翻转 | 占测试集 12.7%–13.0% |
| 候选池内真实未知的聚类 ACC / NMI / ARI | 0.29–0.37 / 0.57–0.63 / 0.04–0.17 |

三条可以直接落地的判断：

1. **候选池的"不纯"不是因为筛得不够，而是因为筛得太少**：候选池只有 444–537 个样本，
   其中只有 188–189 个真未知（占全部未知的 4.5%–4.7%），剩下 64% 是已知样本。
   这是"阈值卡在已知 q95、分数又排不开"的必然结果。
2. **聚类不是这一阶段的瓶颈**：候选池内真未知的 NMI 有 0.57–0.63，
   说明给定这个池子，KMeans 还能分出一些结构；问题在池子本身。
3. **误差集中在语义相近/外观易混的类别**：翻转率最高的已知类是 forest、motorcycle、
   bicycle、snake、maple_tree、caterpillar、willow_tree（树木/交通工具/细长动物），
   未知类里 tiger、lion、crocodile、dinosaur、bee 也高。这些类别正是已知/未知语义边界
   最模糊的地方。

## 6. 回答第 6 节的原始问题

> 当前未知检测不稳定，主要来自风险分数、阈值、拒识器，还是候选池污染？

按证据强度排序：

1. **风险分数的排序能力（主因）**：AUROC 0.537 / 重叠面积 ~0.92，任何阈值下都不可能有好的
   已知-未知分离。这是 6.2 第一步就暴露的问题。
2. **阈值附近的样本级不稳定（次因，但影响下游）**：自助 std 只有 0.05，可 13% 的样本会翻标记；
   下游新类数量/簇大小直接受影响。
3. **候选池污染（结果，不是原因）**：池纯度 35% 是前两条的必然结果，
   不是"净化算法没做好"。
4. **拒识器初始化（未观察到效应）**：`shared_backbone` 没有稳定收益，不构成解释。

## 7. 为什么没有做 NNPU（6.4）

NNPU 的前提是"发现池中同时存在已知和未知样本，且把整个池当未知类会造成错误监督"。
本次正式协议用的是默认 `--discovery-pool-mode unknown`（按训练类别标签过滤出的纯未知池），
**这个前提不成立**。要验证 NNPU 必须先切到 `--discovery-pool-mode mixed`，
那属于新协议（并且会改变候选池语义），需要先和王笑颜线对齐后再做。命令写在
`docs/team_b_rejection_runbook.md` 第 6 节。

## 8. 结果的置信度分层

| 结论 | 状态 |
| --- | --- |
| `scratch_cnn` 与 `shared_backbone` 在未知风险排序上无稳定差异 | **已验证**（三 seed 方向一致性检验） |
| 未知检测瓶颈在分数排序而非拒识器配置 | **已验证**（AUROC 0.53、重叠面积 0.92、漏检 95%） |
| 阈值数值可复现但样本级标记不可复现 | **已验证**（自助 CI + 跨 seed 翻转率） |
| 候选池污染主要来自已知样本误拒 | **已验证**（池纯度 0.35、已知占池 65%） |
| AUROC 的绝对值 | **有待复核**：学生只有 16 epoch，不能与 30 epoch 结果并表 |
| 更强调分方案（knn、相对 mahalanobis）能否改善 | **未验证**（本轮未做单因素对照） |
| NNPU | **未执行**（前提不成立） |

## 9. 下一步建议（按优先级）

1. **换分数，先不换模型**：在学生固定的前提下，用 `score_sweep.py` 把
   `normalized_entropy_mahalanobis` 换成 `normalized_entropy_min_class_relative_knn`、
   `normalized_entropy_relative_mahalanobis` 等，做单因素对照。当前 AUROC 0.53 的空间
   主要在这里。
2. **阈值策略对照**：`global` vs `class_conditional` vs `open_known_coverage`，
   看在已知接受率不变时邻域翻转率能否下降。
3. **补满 30 epoch**：等有稳定 GPU 的机器，按 runbook §9.4 的一条命令重跑，
   把绝对值替换掉。
4. **NNPU 暂缓**：等第 1、2 步确认分数/阈值改动空间后再决定是否引入混合池协议。

## 10. 复现入口

```powershell
# 训练（教师已存在会跳过；建议最多 3 并发，避免本机显卡 TDR 蓝屏）
.\scripts\launch_team_b_detached.ps1 -Tag formal_20261124 -Launch

# 聚合
python scripts/run_team_b_rejection.py --mode formal --tag formal_20261124 `
  --seeds 42,123,3407 --student-epochs 16 `
  --device cuda --threads 2 --num-workers 0 --image-size 64 --mc-samples 8 --skip-existing
```

产物：每个配置的 `s<seed>_<mode>_discover/{metrics.json,threshold_stability.json,summary.md}`，
以及本目录的 `summary.md`、`summary.json`、`candidate_pool_scratch_cnn.md`、
`candidate_pool_shared_backbone.md`。

## 11. 第 6.6 节交付清单

| 6.6 要求 | 状态 | 证据文件 |
| --- | --- | --- |
| 固定风险分数和阈值的基线结果 | 完成 | `summary.md` §2–3；`runs/.../s*_discover/metrics.json` |
| `scratch_cnn` 与 `shared_backbone` 的成对对照 | 完成 | `summary.md` §4（逐 seed 差值 + 方向一致性） |
| 三个正式种子的风险分数、阈值与候选池统计 | 完成 | `summary.md` §2；`candidate_pool_*.md` §1–2 |
| 阈值附近样本的稳定性分析 | 完成 | `candidate_pool_*.md` §3–4；`threshold_stability.json` |
| 有监督混合依据时的 NNPU 对照 | 未触发 | 前提不成立，理由见本文 §7 |
| 明确问题来源的结论 | 完成 | 本文 §6 |

第 6.2–6.5 节的分项对应：

| 文档条目 | 实现 |
| --- | --- |
| 6.2-1/2 分数提取、分布、范围、重叠 | `candidate_pool_*.md` §2（重叠面积、median gap、分位数） |
| 6.2-3/4 阈值由哪些统计量确定、各种子数值 | `calibration_report.json`（policy=known_val_global, percentile 95）；`summary.md` §2 |
| 6.2-5 已知接受率、未知拒识率、候选数量 | `summary.md` §2 |
| 6.2-6 候选池未知召回、纯度、已知污染率 | `candidate_pool_*.md` §1 |
| 6.2-7 阈值附近跨种子标记变化 | `candidate_pool_*.md` §4（翻转率、邻域内外对比、逐类） |
| 6.3 两种拒识器配置对照 | `summary.md` §4 |
| 6.5 候选池误差分解 + 逐类统计 | `candidate_pool_*.md` §5–6 |

## 12. 两个必须随结论一起交付的限定

1. **学生是 16 epoch 预算**（请求 30，实际训练 16）。见 §1.1；
   `s*_student/config.json` 里已写 `epochs=30`、`effective_epochs_trained=16`、
   `selected_epoch=N`、`interrupted=true`、`reconstructed=true`，不会与 30 epoch 结果混淆。
2. **NNPU 未执行**，因为默认发现池是纯未知池（见 §7），不是遗漏。

## 13. 记录格式补记

五个从快照还原的学生目录原本缺 `config.json`（`train_student` 只在训练结束写），
已用 `scripts/backfill_student_configs.py` 按 `command.txt` 补齐，并标注
`reconstructed`。`train_history.json` 不补：那是训练过程原始统计，伪造比缺失更糟；
每个 epoch 的选择分仍在 `promote_report.json` 与 `train.log` 中可查。