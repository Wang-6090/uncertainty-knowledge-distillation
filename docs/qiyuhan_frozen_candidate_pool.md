# 齐誉涵：冻结候选池与聚类对照流程

本文对应 `parallel_work_division.md` 第 7 节。目标是在拒识配置固定后，只研究候选表征、PCA、聚类算法和类别数估计。聚类结果不得反向用于调整拒识阈值或候选成员。

## 1. 先冻结候选池

先由上游确认学生 checkpoint、类别划分、随机种子、风险分数和阈值配置。下面的命令只运行一次，`--save-candidate-pool` 会在发现目录中生成：

- `candidate_pool.npz`：样本索引、分类特征、投影特征、风险分数、候选标记和候选权重，不含任何真实标签；
- `candidate_pool_metadata.json`：schema v2 完整性元数据，包括 NPZ、各数组、checkpoint 和索引哈希，以及划分、种子、阈值和特征配置；
- `discovery_detail.json`：单独保存的评价明细，只允许在聚类完成后读取。

```powershell
python train.py discover `
  --dataset cifar100 `
  --data-root ./data `
  --split-path ./splits/cifar100_60_40.json `
  --student-ckpt ./runs/student/student.pt `
  --work-dir ./runs/qiyuhan_seed42 `
  --seed 42 `
  --threshold-policy global `
  --threshold-percentile 95 `
  --cluster-candidate-mode detector `
  --candidate-purify none `
  --cluster-k oracle `
  --num-novel 40 `
  --save-candidate-pool `
  --download
```

正式实验应直接复用上游已经确认的风险分数和阈值参数，不要为了改善 ACC、NMI 或 ARI 修改本命令。生成后记录 `candidate_pool_metadata.json` 中的 `artifact_sha256`、`array_sha256` 和 `source_index_sha256`；同一组对照的哈希必须一致，离线脚本会在分析前校验文件和数组完整性。

## 2. oracle K 单因素对照

以下命令全部读取同一个冻结文件。`--num-novel 40` 来自预先定义的 CIFAR-100 60/40 协议，不由脚本读取测试标签推导。

### A. 分类特征 + PCA + KMeans

```powershell
python scripts/analyze_frozen_candidate_pool.py `
  ./runs/qiyuhan_seed42/candidate_pool.npz `
  --output-dir ./runs/qiyuhan_seed42/cluster_feature_pca_kmeans `
  --feature feature_pca --pca-dim 32 `
  --method kmeans --cluster-k oracle --num-novel 40 `
  --seed 42 `
  --evaluation-detail ./runs/qiyuhan_seed42/discovery_detail.json
```

### B. 投影特征 + PCA + KMeans

```powershell
python scripts/analyze_frozen_candidate_pool.py `
  ./runs/qiyuhan_seed42/candidate_pool.npz `
  --output-dir ./runs/qiyuhan_seed42/cluster_projection_pca_kmeans `
  --feature projection_pca --pca-dim 32 `
  --method kmeans --cluster-k oracle --num-novel 40 `
  --seed 42 `
  --evaluation-detail ./runs/qiyuhan_seed42/discovery_detail.json
```

### C. 分类特征（无 PCA）+ KMeans

```powershell
python scripts/analyze_frozen_candidate_pool.py `
  ./runs/qiyuhan_seed42/candidate_pool.npz `
  --output-dir ./runs/qiyuhan_seed42/cluster_feature_kmeans `
  --feature feature `
  --method kmeans --cluster-k oracle --num-novel 40 `
  --seed 42 `
  --evaluation-detail ./runs/qiyuhan_seed42/discovery_detail.json
```

### C2. 分类特征（匹配 L2 归一化、无 PCA）+ KMeans

该对照与 `feature_pca` 使用相同的输入 L2 归一化，只移除 PCA/whitening，用于避免把归一化变化误归因于 PCA。

```powershell
python scripts/analyze_frozen_candidate_pool.py `
  ./runs/qiyuhan_seed42/candidate_pool.npz `
  --output-dir ./runs/qiyuhan_seed42/cluster_feature_normalized_kmeans `
  --feature feature --normalize `
  --method kmeans --cluster-k oracle --num-novel 40 `
  --seed 42 `
  --evaluation-detail ./runs/qiyuhan_seed42/discovery_detail.json
```

### D. 固定表征后比较聚类算法

`feature_pca` 在这里固定为诊断锚点，再保持表征、PCA 维数、K 和种子不变，仅替换算法。由于该锚点是在查看回顾性标签指标后保留的，方法比较只能作为探索性结果，不能作为确认性模型选择。下面以 `feature_pca` 为例：

```powershell
python scripts/analyze_frozen_candidate_pool.py `
  ./runs/qiyuhan_seed42/candidate_pool.npz `
  --output-dir ./runs/qiyuhan_seed42/cluster_feature_pca_agglomerative `
  --feature feature_pca --pca-dim 32 `
  --method agglomerative --cluster-k oracle --num-novel 40 `
  --seed 42 `
  --evaluation-detail ./runs/qiyuhan_seed42/discovery_detail.json

python scripts/analyze_frozen_candidate_pool.py `
  ./runs/qiyuhan_seed42/candidate_pool.npz `
  --output-dir ./runs/qiyuhan_seed42/cluster_feature_pca_spectral `
  --feature feature_pca --pca-dim 32 `
  --method spectral --cluster-k oracle --num-novel 40 `
  --seed 42 `
  --evaluation-detail ./runs/qiyuhan_seed42/discovery_detail.json
```

每个输出目录包含 `config.json`、`command.txt`、`metrics.json`、`summary.md`、`cluster_report.json` 和 `cluster_assignments.npz`。重点比较：

- `retrospective_evaluation.unknown_only_metrics` 中的 ACC、NMI、ARI；
- `candidate_purity`、`known_contamination` 和 `candidate_unknown_recall`；
- `cluster_sizes`；
- `unknown_classwise` 中的候选召回、簇分散程度和 Hungarian 对齐召回；
- `known_contamination_classwise` 中各已知类的污染数量和误拒率。

默认 oracle-K 基线还会生成 `all_unknown_oracle_assignments.npz`，并在报告的 `all_unknown_oracle_evaluation` 中记录全部真实未知样本的诊断指标。真实标签只用于定义该评价子集和 oracle K，不用于候选、表征或聚类方法选择。其他重复对照可以使用 `--skip-all-unknown-oracle` 避免重复计算。

其中，离线报告的 `candidate_unknown_recall` 与发现报告的
`cluster_candidate_unknown_recall` 都以评测池中的全部真实未知样本为分母。
发现报告另行提供 `cluster_candidate_unknown_recall_before_purification` 和
`cluster_candidate_unknown_retention_after_purification`，分别表示净化前的端到端召回率，
以及最终候选池对净化前真实未知候选的保留率；三者不能混作同一指标。

## 3. oracle K 结果可解释后再运行 auto K

auto K 只使用候选特征的几何指标选 K。不要根据测试标签挑选 `--selection` 或扩大搜索范围；选择规则应在正式运行前固定。

```powershell
python scripts/analyze_frozen_candidate_pool.py `
  ./runs/qiyuhan_seed42/candidate_pool.npz `
  --output-dir ./runs/qiyuhan_seed42/cluster_feature_pca_auto_k `
  --feature feature_pca --pca-dim 32 `
  --method kmeans --cluster-k auto `
  --selection silhouette --max-auto-clusters 50 `
  --seed 42 `
  --evaluation-detail ./runs/qiyuhan_seed42/discovery_detail.json
```

报告中的 `selected_k` 是标签不可见阶段的选择结果；`true_unknown_k` 和 `k_absolute_error` 只在聚类完成后由评价明细补充。若 oracle K 明显优于 auto K，瓶颈归因于类别数估计；若两者都差，应先检查候选污染、类别级漏检和特征可分性。

## 4. 三种子与停止规则

正式结论至少复现三个预先指定的种子。每个种子分别冻结候选池，不跨种子混合样本，并保持除被比较因素外的配置不变。

- 全部真实未知样本的 oracle K 仍差：优先报告表征或语义重叠问题；
- 全部未知样本可聚类，但冻结候选池差：优先报告候选漏检或已知污染；
- oracle K 稳定而 auto K 差：报告类别数估计问题；
- 改善只出现在单种子或单指标：保留为局部信号，不升级为默认方案。

当前脚本已完成协议、导出和离线分析闭环；本工作区已使用确认的 checkpoint 完成三个正式种子，结论由下一节的汇总命令统一生成。

## 5. 三种子汇总

完成三个正式种子的 A-C2 表征对照和固定诊断锚点上的三种聚类算法后，运行：

```powershell
python scripts/summarize_frozen_candidate_pool.py `
  ./runs/qiyuhan_cifar_seed42 `
  ./runs/qiyuhan_cifar_seed123 `
  ./runs/qiyuhan_cifar_seed3407 `
  --output-dir ./analysis/qiyuhan_frozen_candidate_pool_multiseed
```

汇总器只接受 oracle K 结果，并检查候选池 artifact/索引哈希、随机种子、候选数量、归一化设置和标签隔离声明。输出包含全部真实未知 oracle 诊断、逐种子原始值、均值/样本标准差、类别级候选漏检摘要和停止决定。当前证据把候选池质量定位为直接瓶颈，但不排除表征或 K 估计误差，因此 auto K 暂缓，而不是宣称 K 估计已被排除。
