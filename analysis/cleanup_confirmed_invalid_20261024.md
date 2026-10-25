# 已确认无效运行产物清理记录（2026-10-24）

本次只清理审计文档明确判定为无效、且不应作为算法结果使用的运行目录。为保留可追溯性，目录没有永久删除，而是移入：

`runs/_archive_confirmed_invalid_20261024/`

## 已归档目录

- `joint_pilot_s44_prioraware/`：第一次 prior-aware Sinkhorn 执行，调用链 wiring 缺失。
- `joint_pilot_s44_prioraware_discover/`：上述无效执行对应的 discovery 产物。
- `joint_pilot_s44_prioraware_fixed/`：第二次执行，Sinkhorn 未加权归一化错误。
- `joint_pilot_s44_prioraware_fixed_discover/`：上述无效执行对应的 discovery 产物。
- `joint_pilot_s44_prioraware_fixed_cluster_diag/`：上述无效执行对应的聚类诊断产物。
- `semantic_isolated_full_s123_disjointcal_lr01_gating_student/`：candidate-gating 首轮 treatment，配置错误，未形成有效受控比较。
- `semantic_isolated_full_s123_disjointcal_lr01_gating_discover/`：上述无效首轮 treatment 对应的 discovery 产物。

这些目录不参与任何主表、对照结论或后续模型选择。对应的分析报告、修正后的有效运行、源代码和测试均保留。

## 本次明确没有清理的内容

- `s2079` CE-only 重跑：CE-only 和 standard 两臂已有完整训练与 discovery 报告，但 uncertainty 臂尚未完成，因此保留，不能把“未完成”误判为“无效”。
- `analysis/` 中的失败审计报告：它们记录了配置错误和排除依据，仍具有研究追溯价值。
- 仅仅效果较差、但协议有效的实验：它们属于负结果，不等同于无效运行，暂不删除。
