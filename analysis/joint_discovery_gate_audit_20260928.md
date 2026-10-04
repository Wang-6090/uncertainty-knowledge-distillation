# Joint discovery / independent unknown gate audit

## 目的

本轮针对“已知/未知分数重叠、未知拒绝率低”的核心问题，先区分新类聚类头和未知检测门控，避免让一个 100 类联合 Softmax 同时承担两个任务。

## 诊断结论

对三个 CIFAR-100 60/40、seed=42、10 epoch checkpoint 的离线诊断显示：

- 联合已知+新类分配中，真实未知样本被分到新类的比例约为 2.5%–3.5%；
- 联合分配长期只有约 19–21 个新类原型被激活，其余原型空置；
- 直接使用联合分配的未知池 ARI 接近 0；
- 分开检查新类头后，原有模型的平均最大概率约 0.028–0.037，归一化熵接近 1，说明新类头接近均匀输出；
- 因此，问题不只是阈值或 logits 尺度，而是新类头没有形成稳定的类别结构，同时联合 Softmax 不能作为未知门控。

## 已做的代码修改

1. `scripts/diagnose_joint_assignments.py`
   - 现在同时报告独立新类头和联合已知/新类空间；
   - 防止把“联合门控失败”误判为“新类头本身失败”。

2. `novel_discovery/joint_discovery.py`
   - 新增 `prototype_pseudo_label_loss`；
   - 使用跨增强、平衡 Sinkhorn 伪标签进行硬交叉监督。

3. `novel_discovery/pipeline.py` / `train.py`
   - 新增可选 `--alpha-joint-pseudo`；
   - 新增独立未知门控损失 `--alpha-joint-gate` 和 `--joint-gate-margin`；
   - 该门控只约束 uncertainty head，不再把 known/unknown 放进同一个分类目标。

## 小规模结果

配置：CIFAR-100 60/40、seed=42、ResNet34 teacher / ResNet18 student、1200 train、300 val、1000 test、5 epoch、GPU。

| 配置 | AUROC | FPR95 | 已知接受率 | 未知拒绝率 | cluster ACC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原有联合头，简单候选 | 0.5093 | 0.9293 | 0.9507 | 0.0255 | 0.7500 |
| 硬伪标签 | 0.5271 | 0.9128 | 0.9260 | 0.0536 | 0.6061 |
| 独立 uncertainty gate，简单候选，Mahalanobis 检测 | **0.5395** | **0.8980** | 0.9128 | **0.0791** | 0.5000 |
| 独立 gate + 硬伪标签 | 0.5079 | 0.9095 | 0.9671 | 0.0281 | 0.8387 |
| EMA + kNN 邻域过滤 + gate，head uncertainty 检测 | 0.4555 | 0.9507 | 0.9605 | 0.0128 | 0.4828 |
| 仅 prototype distance 候选 + gate | 0.5233 | 0.9145 | 0.9046 | 0.0740 | 0.4943 |
| distance + uncertainty/entropy 融合候选 + gate | 0.5293 | 0.9128 | 0.9556 | 0.0561 | 0.7143 |

这些结果只是小样本筛选实验，不能作为最终论文结论。独立 gate 的正向变化需要至少 3 个 seed 和完整训练集复核。

## 当前采用和暂不采用的设置

- 暂保留：独立 uncertainty ranking 作为可选未知门控，检测时继续以 Mahalanobis / Energy 等成熟分数为主；
- 暂不默认启用：硬伪标签。它改善了新类头的局部结构，但与未知门控叠加后性能下降；
- 暂不采用：EMA + kNN 邻域过滤的当前参数。过滤过严，候选比例约降到 1.35%，训练信号不足；
- 暂不采用：prototype distance 或 distance-consensus 候选。它们没有超过简单 consensus，说明未知样本与已知原型的距离不是充分的未知信号；
- 暂不把联合 novel mass 当作未知检测分数，因为已知和未知的 novel mass 分布几乎重叠。

## 下一步

1. 在相同协议下对独立 gate 做 `alpha-joint-gate=0.1/0.5/1.0` 的小规模消融；
2. 保持简单候选筛选，先不要同时加入硬伪标签、EMA 和邻域过滤；
3. 使用至少 3 个 seed 复核 AUROC、FPR95、OSCR、已知接受率和未知拒绝率；
4. 新类发现单独评估候选池纯度、未知召回率、oracle-K 聚类和 auto-K 聚类，避免把检测漏检掩盖在聚类指标中；
5. 如果独立 gate 仍不稳定，再实现跨 batch memory bank / global kNN 伪标签，而不是继续增加批内损失项。

## 进一步方向判断

本轮已经比较了分类风险、联合 novel mass、已知原型距离及其融合。它们都只能从现有已知模型输出中间接推断未知性，提升有限。下一种更有区分度的方案应引入真正的未知分布暴露：

- 使用独立无标签外部图像作为 Outlier Exposure，训练已知分类 logits 接近均匀分布，并同步提高 uncertainty；
- 或者在纯未知 discovery pool 仅作为上限诊断，直接训练独立 known/unknown rejector；
- 外部未知数据不能直接混入最终测试集，且应与真实 novel 类别保持协议隔离。

参考思路包括 Hendrycks et al., *Deep Anomaly Detection with Outlier Exposure*（ICLR 2019）的 uniform-logit OE，Liu et al., *Energy-based Out-of-distribution Detection*（NeurIPS 2020）的能量间隔，以及 Lee et al., *A Simple Unified Framework for Detecting Out-of-Distribution Samples*（NeurIPS 2018）的特征生成/类条件距离。下一步优先实现可选外部 OE 数据接口，再与当前独立 uncertainty gate 做单因素对照。

## 外部 Outlier Exposure 接口（代码已实现，尚未下载数据做正式实验）

当前代码新增可选参数：

```powershell
--outlier-dataset cifar10
--outlier-data-root .\data
--outlier-download
--alpha-outlier-uniform 0.1
--alpha-outlier-energy 0.1
--alpha-outlier-uncertainty 0.1
```

也支持 `--outlier-dataset imagefolder`，可连接独立的 Tiny-ImageNet 或其他无标签图片目录。该数据只参与 uniform-logit、energy 和 uncertainty 训练，不参与已知类别分类，也不直接用于最终测试指标。

代码已通过 78 个测试；使用内存构造的已知 batch 和外部 batch 完成了 OE 训练循环 smoke test，三个损失均产生有限值。正式实验前仍需确认外部数据源与 CIFAR-100 的类别协议没有重叠，并单独比较 uniform、energy、uncertainty 及其组合。
