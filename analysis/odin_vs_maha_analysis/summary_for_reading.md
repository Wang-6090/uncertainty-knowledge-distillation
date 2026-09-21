# ODIN 与 Mahalanobis 检测数据对比小结

本次只使用已有的 quick_de_discovery_energy 模型做检测结果分析，没有重新训练模型。

## 对比设置

- 数据集：CIFAR-100，60 个已知类 / 40 个未知类
- 测试子集：1000 张
- 模型：runs/quick_de_discovery_energy/student.pt
- 对比方法：
  - normalized_entropy_mahalanobis
  - odin_msp，epsilon=0.0005，temperature=1000

## 整体指标

| 方法 | AUROC | FPR95 | known acc | unknown reject | 候选池纯度 |
|---|---:|---:|---:|---:|---:|
| normalized entropy + Mahalanobis | 0.5505 | 0.9276 | 0.3470 | 0.0561 | 0.4000 |
| ODIN eps=0.0005 | 0.5414 | 0.8964 | 0.3536 | 0.0791 | 0.4306 |

## 阈值和错误数量

| 方法 | 阈值 | 误拒已知样本 | 正确拒绝未知样本 | 误接收未知样本 |
|---|---:|---:|---:|---:|
| normalized entropy + Mahalanobis | 2.526315 | 33 | 22 | 370 |
| ODIN eps=0.0005 | 0.983246 | 41 | 31 | 361 |

ODIN 多拒绝了 9 个未知样本，但同时多误拒了 8 个已知样本。因此它改善了未知拒绝率和候选池纯度，但代价是候选池更大、已知误拒也更多。

## 分数分布

| 方法 | 已知分数中位数 | 未知分数中位数 | 已知 P75 | 未知 P25 |
|---|---:|---:|---:|---:|
| normalized entropy + Mahalanobis | 0.1662 | 0.4231 | 1.2649 | -0.4892 |
| ODIN eps=0.0005 | 0.983193 | 0.983200 | 0.983219 | 0.983166 |

Mahalanobis 组合分数的已知和未知中位数有差异，但四分位区间仍明显重叠。ODIN 的分数几乎全部挤在 0.983 附近，说明它没有真正把已知和未知拉开，只是在阈值附近改变了一小部分样本的排序。

## 主要错误类别

Mahalanobis 组合分数中，误拒较多的已知类包括 can、motorcycle、pickup_truck、bicycle、lizard。未知类中 orange、dinosaur、castle、butterfly、orchid、wolf、tiger、train 全部被误接收为已知。

ODIN 中，误拒较多的已知类变成 snake、oak_tree、otter、bowl、pine_tree、crab。未知类中 orange、dinosaur、butterfly、orchid、pear、wolf、girl、cup 仍全部被误接收。

## 当前结论

ODIN 可以作为检测基线保留，但还不能作为主方法。它对 FPR95、unknown reject rate 和候选池纯度有小幅改善，但 AUROC 没有提高，而且分数分布没有明显分开。当前最主要的问题仍然是：已知和未知样本在模型特征与开放集分数上高度重叠。

下一步更值得做的是改善特征空间，而不是继续只调检测阈值。优先方向是更强的已知分类训练、更稳定的 discovery pool 训练，以及更系统的多 seed 对比。
