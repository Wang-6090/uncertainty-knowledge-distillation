# kNN PU boundary：三臂结果与共享表征梯度审计（2026-10-17）

## 实验目的

本轮先完成 `pure`、`hard_nnpu`、`soft_nnpu` 的严格配对对照，再补做 1 epoch 的共享 encoder 梯度审计。目的有两个：

1. 判断 soft-NNPU 是否同时改善排序、分布分离和固定校准工作点，而不是只提高未知拒绝率；
2. 核对 kNN 边界项是否真的改变共享表征，以及它是否与 CE、KD、SupCon 发生梯度冲突。

三臂对照固定 seed=44、CIFAR-100 semantic-isolated 60/40、4 epochs、mixed discovery pool=5400、known prior=0.2、shared-boundary 候选空间、kNN 边界权重 0.1、同一检测器和 validation-only 95% known-coverage 校准；唯一算法变量是 `discovery_knn_risk`。测试标签没有参与训练、校准或模型选择。

## 三臂结果

| kNN risk | AUROC | FPR95 | OSCR | 已知接受率 | 未知拒绝率 | KS | unknown below known-q95 | 接受后已知准确率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pure | 0.900255 | 0.364833 | 0.658245 | 94.517% | 50.800% | 0.653000 | 51.775% | 72.827% |
| hard_nnpu | 0.903022 | 0.347667 | 0.665626 | 94.367% | 52.500% | 0.656583 | 50.625% | 73.384% |
| soft_nnpu | 0.898656 | 0.360833 | 0.659466 | 93.917% | 52.125% | 0.642917 | 52.025% | 73.203% |

`pure` 的已知接受率为 94.5167%。上表的 hard/soft 结果来自新运行目录 `runs/knn_pu_boundary_control_s44_e4_soft_added_*_discover/`；对应 `pure` 的报告字段为 0.945167。表中 pure 的展示值因此保留 3 位小数时写作 94.517%，不应与历史严格复核中的另一目录混用。

### 判断

- `soft_nnpu` 不是有效改进：相对 pure，AUROC、KS 和 overlap 均变差；未知拒绝率提高 1.325 个百分点，同时已知接受率下降 0.600 个百分点。
- 本次 `hard_nnpu` 比同一次运行的 pure 更好，但此前严格两臂复核中 hard-NNPU 相对 pure 的排序指标变差，说明该优势尚未跨实验稳定复现，不能把 hard 设为默认。
- 不能用未知拒绝率单项判定成功。当前仍没有证据证明 kNN PU 风险解决了已知/未知分数重叠。

## 共享 encoder 梯度审计

代码此前没有把 `discovery_knn_boundary` 放入共享 encoder 的梯度诊断。本轮补齐了该项，并增加了与 CE、KD、SupCon、uncertainty-PU 的余弦记录。1 epoch 审计沿用同一训练配置，仅比较 pure 与 soft-NNPU。

| risk | kNN boundary loss | kNN encoder 梯度范数 | CE 与 kNN 余弦 | KD 与 kNN 余弦 | SupCon 与 kNN 余弦 |
|---|---:|---:|---:|---:|---:|
| pure | 0.019866 | 1.557455 | -0.035698 | -0.004556 | -0.011396 |
| soft_nnpu | 0.039864 | 1.046466 | -0.047631 | -0.003683 | -0.015823 |

结论是：kNN 边界项确实有梯度，不能再说它“没有参与训练”；但其方向与主要已知分类目标略有冲突，soft-NNPU 还把有效梯度减弱了。梯度审计只能解释机制，不能替代多 epoch 开放集结果。

## 代码变更

- `novel_discovery/pipeline.py`：将共享 encoder 的 `discovery_knn_boundary`、`discovery_boundary` 纳入逐损失梯度范数和余弦冲突审计。
- `tests/test_core_behaviors.py`：增加 kNN 边界梯度诊断键的测试。
- `scripts/run_knn_gradient_audit.py`：新增可复现的 1 epoch pure/soft 梯度审计脚本；只做机制诊断，不作为正式性能结果。
- `scripts/run_knn_pu_boundary_control.py`：保留三臂严格对照，soft-NNPU 仍为显式消融，不改变默认配置。

## 后续方向

停止继续扫描 hard/soft NNPU 的 softness 或阈值。下一步先按未知细类、预测已知类和 kNN 支持距离统计误接收样本，确认重叠主要来自 semantic-near unknown、支持半径覆盖不足，还是共享 encoder 的类间几何。只有在该诊断确定了错误来源后，才设计新的类条件表征损失；新方法仍需与 pure 基线在相同 seed、数据协议、校准规则下比较 AUROC、FPR95、OSCR、KS、overlap 和已知效用。
