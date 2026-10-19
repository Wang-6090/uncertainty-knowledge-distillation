# nnPU 已知支持分层抽样对照（2026-10-16）

## 动机与唯一变量

feature rejector 的已知支持最多使用 5,000 个样本。历史实现随机抽样，可能使 60 个已知类在支持集合中的数量不均，从而把某些类的 support 几何误当成总体已知分布。本轮使用已有 `rejector_stratified_known` 选项，按已知类别近似均匀抽样。

两组固定：student checkpoint、CIFAR-100 semantic-isolated 60/40、mixed pool=5,400、已知先验=0.2、`nu_corrected`、`support_augmented`、linear nnPU、MC=4、验证集 95% known-coverage 校准、测试集和阈值策略。唯一变量是已知支持抽样方式：random vs stratified。

## 结果

| seed | known support | AUROC | AUPR | FPR95 | OSCR | 已知接受率 | 未知拒绝率 | KS | unknown below known q95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 44 | random | 0.88225 | 0.81623 | 0.4070 | 0.64918 | 95.15% | 42.40% | 0.6046 | 56.73% |
| 44 | stratified | 0.88389 | 0.81580 | 0.4027 | 0.65048 | 93.28% | 50.15% | 0.6098 | 56.35% |
| 123 | random | 0.88326 | 0.82760 | 0.4362 | 0.62922 | 96.37% | 40.18% | 0.6038 | 52.45% |
| 123 | stratified | 0.88964 | 0.82854 | 0.4070 | 0.63537 | 96.65% | 39.40% | 0.6279 | 51.88% |

## 判断

两 seed 的 AUROC、FPR95、OSCR、KS 和 overlap 代理都同向改善，说明分层抽样有可能减少已知支持集合的类分布偏差，而不是简单地改变阈值。它比 nnPU 风险归一化更值得继续研究。

但工作点指标并不完全同向：seed=44 的已知接受率下降 1.87 个百分点，seed=123 的未知拒绝率下降 0.78 个百分点。因而当前不把它直接设为无条件默认；下一步应在不使用测试标签的 open validation 上预先固定已知覆盖保护条件，检查分层支持是否能在相同已知覆盖下保持分离收益。若无法保护已知接受率，则保留为 ablation，而不是作为主方法。

原始结果：

- `runs/nnpu_stratified_support_control_s44_random/discovery_report.json`
- `runs/nnpu_stratified_support_control_s44_stratified/discovery_report.json`
- `runs/nnpu_stratified_support_control_s123_random/discovery_report.json`
- `runs/nnpu_stratified_support_control_s123_stratified/discovery_report.json`

复现实验：`scripts/run_nnpu_stratified_support_control.py`。
