# MC-epistemic logits-KD 三 split 复核报告（2114–2116）

## 研究问题

在相同的 split-matched teacher MC-mean logits 下，把 teacher 的 MC-epistemic
不确定性用于逐样本 logits-KD 加权，是否能够改善 semantic-hard CIFAR-100
60/40 新类发现，而不是只改善某一个拒识指标。

## 预注册条件与真实性审计

- fresh semantic-hard CIFAR-100 60/40 split：seed `2114`、`2115`、`2116`。
- 每个 seed 的 standard 与 uncertainty 两臂共享同一个 teacher、同一 split、
  同一 student recipe、4 个 student epoch 和同一个 matched mixed discovery pool。
- 两臂都使用 4-pass MC-mean teacher logits；唯一训练因素是 logits-KD 的
  `kd_mode`：普通 KL（standard）或 detached、batch-mean-normalized
  `exp(-MC_epistemic)` 加权 KL（uncertainty）。
- feature KD 和 rejection-branch KD 在两臂均关闭；其他辅助损失保持一致。
- checkpoint 只按 known validation accuracy 选择；拒识阈值只由 known validation
  的 95% coverage 校准；automatic K 只使用无标签候选几何。
- mixed/test 真值标签未用于训练、checkpoint 选择、阈值校准或方法选择，只用于
  最终回顾性指标和污染审计。
- 三个 seed 的机器结果均完整生成，训练日志确认 uncertainty 权重实际应用，
  KD loss 非零，且配置审计确认了单因素差异。

## 配对结果（uncertainty − standard）

FPR95 的负值表示改善；known acceptance、unknown rejection 和 accepted-known
accuracy 均以百分点（pp）表示。

| seed | ΔAUROC | ΔAUPR | ΔFPR95 | ΔOSCR | Δ known acceptance | Δ unknown rejection | Δ accepted-known accuracy | Δ NMI | Δ ARI | Δ Auto-K 绝对误差 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2114 | +0.000639 | +0.003511 | +1.717 pp | +0.003247 | +0.083 pp | +0.650 pp | +0.291 pp | +0.003532 | +0.009277 | +1 |
| 2115 | +0.001159 | +0.000037 | −3.950 pp | +0.006023 | −0.700 pp | +2.925 pp | +0.716 pp | −0.018462 | −0.017720 | +2 |
| 2116 | +0.002836 | +0.002905 | −1.583 pp | −0.005829 | −2.000 pp | +6.400 pp | −0.477 pp | −0.010940 | +0.018976 | −2 |
| 三 seed 平均 | +0.001545 | +0.002151 | −1.272 pp | +0.001147 | −0.872 pp | +3.325 pp | +0.177 pp | −0.008623 | +0.003511 | +0.333 |

三组机器产物见：

- [`uncertainty_kd_mc_epistemic_s2114_2116_s2114.json`](uncertainty_kd_mc_epistemic_s2114_2116_s2114.json)
- [`uncertainty_kd_mc_epistemic_s2114_2116_s2115.json`](uncertainty_kd_mc_epistemic_s2114_2116_s2115.json)
- [`uncertainty_kd_mc_epistemic_s2114_2116_s2116.json`](uncertainty_kd_mc_epistemic_s2114_2116_s2116.json)

## 权重机制是否真实生效

每个 seed 的 uncertainty 臂权重统计如下，分别为 4 个 epoch 的平均 weight
standard deviation 和 ESS fraction：

| seed | weight std | ESS fraction |
|---:|---:|---:|
| 2114 | 0.010158 | 0.999852 |
| 2115 | 0.005457 | 0.999967 |
| 2116 | 0.027569 | 0.999088 |

因此不是 wiring 失败：权重被应用且不是严格常数。但 ESS 接近 1，说明实际样本
重分配很弱，不能把该机制解释为强烈的难例重加权。

## 是否有效

该方法没有通过预注册的整体新类发现门槛。

有利变化是：三 seed 平均 unknown rejection 增加 `3.325 pp`，AUROC/AUPR
小幅增加，平均 FPR95 改善 `1.272 pp`；2115 和 2116 的 unknown rejection
分别增加 `2.925/6.400 pp`。但不利变化同样明确：

- 2114 的 unknown rejection 只增加 `0.650 pp`，未达到 `+2 pp` 主门槛；
- 2116 的 known acceptance 下降 `2.000 pp`，超过 `1 pp` 允许损失，且 OSCR
  和 accepted-known accuracy 下降；
- overall NMI 平均下降 `0.008623`，2115 和 2116 均下降；
- Auto-K 绝对误差在 2114、2115 变差，三 seed 平均变差 `0.333` 类；
- candidate purity 在三 seed 平均下降 `0.797 pp`。

因此这不是完整新类发现改进，而是“部分 split 上拒识增强、但聚类/已知保持不
稳定”的 trade-off。不能仅依据 2115 或 2116 的拒识提升宣传方法有效。

## 决策与下一步

停止 MC-epistemic logits-KD 的继续扫描：不再在这些 split 上搜索 MC 次数、权重
系数或 weighting function，不改变默认配置。standard logits-KD 继续作为对照，
uncertainty-KD 仅保留为显式 opt-in 消融/诊断。

下一阶段回到当前真正瓶颈：semantic-near novel 与 known 的表征/拒识重叠，以及
Auto-K 低估。已有 `feature_pca` 复核显示它能改善聚类一致性但会牺牲 Auto-K，
8 epoch 能改善拒识和 K 估计但损害 NMI/ARI；因此后续仍应采用单因素 fresh-split
协议，同时报告拒识、known utility、Auto-K 和 NMI/ARI，不能只追求 unknown
rejection。

