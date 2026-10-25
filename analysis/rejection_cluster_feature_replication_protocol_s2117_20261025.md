# 预注册协议：聚类输入 `feature_pca` 复核（seed=2117）

## 研究问题

在同一个冻结 student checkpoint、同一个拒识候选池和同一个 known-validation-only
阈值下，使用 rejection feature 的 PCA 表征而不是 projection feature 的 PCA 表征，
能否稳定改善 semantic-hard 新类聚类质量，并且不恶化自动 K 估计。

## 配对条件

- fresh semantic-hard CIFAR-100 60/40 split，seed `2117`。
- 一次 split-matched teacher；一份 4-epoch student checkpoint；两次 discovery
  使用完全相同的 checkpoint 和 candidate mask。
- control：`cluster_feature=projection_pca`；treatment：`cluster_feature=feature_pca`。
- feature-rejector、candidate rule、silhouette Auto-K、PCA 维度、KMeans、known
  validation 95% coverage calibration 和其余参数全部固定。
- test/mixed labels 不参与训练、checkpoint 选择、阈值校准、候选选择或方法选择，
  只用于最终回顾性指标和污染审计。

## 预注册联合门槛

由于两臂共享冻结 checkpoint 和 candidate mask，AUROC、AUPR、FPR95、OSCR、known
acceptance、unknown rejection 及 accepted-known accuracy 应完全相同；若不相同，
先判定实现或审计失败。方法只有在以下条件同时满足时才作为聚类组件候选：

1. all-unknown 和 candidate-unknown 的 NMI、ARI 均不低于 control，且至少一项
   NMI 与一项 ARI 有实质改善；
2. Auto-K 绝对误差不增加；
3. candidate purity 不下降超过 1 pp；
4. 不根据 seed=2117 的结果继续调 PCA 维度、K 准则或候选池。

单个指标改善而 Auto-K 退化，只记录为 trade-off，不改默认配置。

## 结果状态

协议先于运行固定；运行结束后补充配对绝对值、差值、配置一致性和决策。机器结果
预计保存为 `analysis/rejection_cluster_feature_replication_s2117.json`，详细报告
为 `analysis/rejection_cluster_feature_replication_s2117_20261025.md`。

