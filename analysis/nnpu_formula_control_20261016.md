# nnPU rejector risk normalization control（2026-10-16）

## 目的

本轮检查混合 discovery pool 的 nnPU feature rejector 是否存在历史风险缩放问题。rejector 的分数定义为“越大越未知”：已知样本是可靠负类，mixed pool 是未标注样本，已知先验为 `pi_K=0.2`。

对 mixed pool 中未知正类风险进行分解时：

\[
R = \pi_K E_K[\ell(f,0)] + \max\{0, E_U[\ell(f,1)]-\pi_K E_K[\ell(f,1)]\}.
\]

代码中的 `legacy` 还会把第二项除以 `1-pi_K`；`nu_corrected` 去掉这一额外缩放，但保留相同的类别项。两者都保留非负风险截断。这里的“corrected”只表示风险归一化方式修正，不表示已经证明检测性能稳定提升。

## 严格对照条件

- CIFAR-100 semantic-isolated 60/40；
- seed=44 和 seed=123 分别使用已有的匹配 student checkpoint；
- 同一 mixed discovery pool，大小 5,400，实际已知比例 0.2；
- 同一 `support_augmented` 特征、linear nnPU rejector、MC=4、验证集 95% known-coverage 校准；
- 不重新训练学生，不改变测试数据、阈值策略、聚类配置和检测器；
- 每个 seed 只改变 `rejector_nnpu_risk=legacy` 或 `nu_corrected`。

## 结果

| seed | risk | AUROC | AUPR | FPR95 | OSCR | 已知接受率 | 未知拒绝率 | KS | unknown below known q95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 44 | legacy | 0.89385 | 0.83148 | 0.37200 | 0.65765 | 95.18% | 46.43% | 0.6283 | 52.88% |
| 44 | nu_corrected | 0.89999 | 0.84038 | 0.35783 | 0.66143 | 94.98% | 47.48% | 0.6445 | 52.53% |
| 123 | legacy | 0.88511 | 0.82934 | 0.45433 | 0.63460 | 96.13% | 42.60% | 0.6145 | 52.18% |
| 123 | nu_corrected | 0.88326 | 0.82760 | 0.43617 | 0.62922 | 96.37% | 40.18% | 0.6038 | 52.45% |

## 判断

该修正通过了公式单元测试，且 seed=44 的排序和分离指标同时改善；但 seed=123 的 AUROC、OSCR、KS、未知拒绝率和重叠代理没有同步改善。FPR95 在两个 seed 都改善，但不足以抵消其他指标的方向不一致。

因此：

1. `nu_corrected` 是理论上更清晰、值得保留的实验选项；
2. 当前证据不足以把它单独宣称为稳定解决特征/分数重叠的方法；
3. 不应继续仅调风险缩放或阈值；后续应优先检查 mixed pool 的已知支持是否类均衡、不同已知类的 support 覆盖是否一致，并继续报告完整指标；
4. 历史 `legacy` 结果仍保留用于复现，新的实验应明确记录 risk mode，避免把两种目标混写为同一算法。

原始结果：

- `runs/nnpu_formula_control_s44_legacy/discovery_report.json`
- `runs/nnpu_formula_control_s44_nu_corrected/discovery_report.json`
- `runs/nnpu_formula_control_s123_legacy/discovery_report.json`
- `runs/nnpu_formula_control_s123_nu_corrected/discovery_report.json`

复现实验：`scripts/run_nnpu_formula_control.py`。
