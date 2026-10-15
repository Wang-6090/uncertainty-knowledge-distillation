# 修正后候选空间配对实验总结（2026-10-16）

## 实验目的

本实验验证：在真正读取候选 mask 的 rejection kNN boundary loss 中，把候选空间从共享分类特征改为独立 rejection-support 特征，是否能够减少已知/未知分数与特征分布的重叠。

## 实验协议

- 数据集：CIFAR-100 semantic-isolated 60/40；
- seed：44；训练：4 epochs；设备：CUDA；
- discovery pool：同一 matched mixed pool，known prior=0.2；
- 唯一算法变量：`discovery_feature_candidate_space`，分别为 `shared` 和 `rejection_support`；
- 两组均启用 `alpha_discovery_rejection_knn_boundary=0.1`、双视图候选交集门控；
- 独立 PU boundary 项关闭；检测器、open validation 和 95% known-coverage 校准规则保持一致。

训练历史核对通过：

| 候选空间 | 最后 epoch candidate ratio | 最后 epoch rejection-kNN boundary | 最佳 known val acc |
|---|---:|---:|---:|
| shared | 0.1247 | 0.01620 | 0.7089 |
| rejection_support | 0.1833 | 0.00429 | 0.7144 |

两组的候选门控和对应损失均非零，因此本次不是“参数写入但损失未运行”的 wiring 失效实验。旧版 `alpha_discovery_rejection_boundary` 实验仍不能作为候选空间结论，详见 [`candidate_space_wiring_audit_20261015.md`](candidate_space_wiring_audit_20261015.md)。

## 结果

| 指标 | shared | rejection_support | 变化（support - shared） |
|---|---:|---:|---:|
| AUROC | 0.89230 | 0.89266 | +0.00036 |
| AUPR | 0.83089 | 0.83111 | +0.00022 |
| FPR95 | 0.38083 | 0.39783 | +0.01700 |
| OSCR | 0.65442 | 0.65808 | +0.00366 |
| known accuracy after accept | 0.72574 | 0.73158 | +0.00584 |
| known accuracy on all known | 0.71383 | 0.71533 | +0.00150 |
| known accept rate | 0.95833 | 0.94567 | -0.01267 |
| unknown reject rate | 0.42375 | 0.47800 | +0.05425 |
| KS distance | 0.62450 | 0.63875 | +0.01425 |
| unknown below known q95 | 0.52750 | 0.55125 | +0.02375 |
| known above unknown q05 | 0.38083 | 0.39800 | +0.01717 |
| mean score gap | 8.44956 | 7.88763 | -0.56194 |
| median score gap | 8.55508 | 7.63356 | -0.92160 |

在报告阈值下，`rejection_support` 的未知拒绝率虽然提高了 5.43 个百分点，但已知拒绝数从 250 增至 326，已知接受率下降 1.27 个百分点。AUROC 几乎没有变化，FPR95 变差，两个 overlap 指标变差，均值和中位数分离间隔反而缩小。

## 判断

本实验不支持把 `rejection_support` 候选空间提升为默认方案，也不能说明核心的已知/未知表征重叠问题已经解决。它的主要效果是改变拒绝决策的工作点：牺牲一部分已知接受，换取更高的表面未知拒绝率；这与真正改善排序和分布分离不同。

这条方向仍有一个有限的可取之处：`rejection_support` 的 KS 略高、OSCR 略高，说明 rejection-support 可能包含少量互补信息。但信号不足以抵消 FPR95 和 overlap 的恶化，不应继续进行无约束的权重或阈值扫描。

## 后续处理

1. 保留候选空间开关和本次结果作为可复现实验/消融记录，默认仍使用 `shared`。
2. 暂停继续调候选空间本身，下一步应检查 rejection-support 的类条件 support 半径、候选 purity 和 semantic-near unknown 的局部误接收率。
3. 若继续研究该支路，应先采用独立 open-validation 规则预先固定 support 半径与边界权重，再用至少两个 seed 复核，并同时要求 AUROC、FPR95、overlap、known utility 和固定覆盖率拒绝率共同改善。
4. 下一阶段优先回到共享 encoder 的类条件几何：针对靠近已知类 support 的未知样本施加可靠的局部边界约束，避免只依赖全局分数或阈值。

## 产物

- shared：`runs/s44_e4_rejection_knn_candidate_space_control_shared_discover/discovery_report.json`
- rejection-support：`runs/s44_e4_rejection_knn_candidate_space_control_rejection_support_discover/discovery_report.json`
- 对照脚本：[`../scripts/run_rejection_boundary_candidate_space_control.py`](../scripts/run_rejection_boundary_candidate_space_control.py)
