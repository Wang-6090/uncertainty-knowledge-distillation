# 2x2 full-pipeline factorial: seed 2118

## Protocol validity

This fresh semantic-hard CIFAR-100 60/40 experiment crossed two previously
studied components on one split-matched teacher and one shared-backbone student
checkpoint: `cluster_feature` (`projection_pca` versus `feature_pca`) and score
(`feature_rejector` versus fixed known-validation-calibrated uncertainty
fusion). The student was trained once; all four discovery arms used the same
frozen checkpoint and the same candidate/training configuration.

The fusion rule was fixed at the existing 0.5/0.5 score combination. Its
uncertainty-to-error mapping and 95% known-coverage threshold used known
validation only. No test or mixed-pool labels were used for training,
checkpoint selection, calibration, candidate selection or arm selection; they
were used only for retrospective metrics. The four configurations differed
from the control only in the two registered factors and output paths.

## Results

| Metric | rejector + projection | rejector + feature | fusion + projection | fusion + feature |
|---|---:|---:|---:|---:|
| AUROC | 0.811574 | 0.811574 | 0.813532 | 0.813532 |
| AUPR | 0.718728 | 0.718728 | 0.727585 | 0.727585 |
| FPR95 | 59.667% | 59.667% | 59.517% | 59.517% |
| OSCR | 0.582323 | 0.582323 | 0.583601 | 0.583601 |
| Known acceptance | 95.450% | 95.450% | 96.200% | 96.200% |
| Unknown rejection | 25.750% | 25.750% | 25.675% | 25.675% |
| Accepted-known accuracy | 69.233% | 69.233% | 69.006% | 69.006% |
| Candidate count / purity | 1303 / 79.048% | 1303 / 79.048% | 1255 / 81.833% | 1255 / 81.833% |
| Automatic K (true K=40) | 31 (error 9) | 28 (error 12) | 28 (error 12) | 24 (error 16) |
| All-unknown NMI / ARI | 0.4711479 / 0.2269193 | 0.5342233 / 0.3029029 | 0.4711479 / 0.2269193 | 0.5342233 / 0.3029029 |
| Candidate-unknown NMI / ARI | 0.5407413 / 0.2180289 | 0.6383560 / 0.3447323 | 0.5268004 / 0.2133343 | 0.6295752 / 0.3451445 |

## Changes relative to the control

The clustering-only arm (`rejector + feature_pca`) leaves all rejection and
known-class metrics exactly unchanged, while improving all-unknown NMI/ARI by
`+0.063075/+0.075984` and candidate-unknown NMI/ARI by
`+0.097615/+0.126703`. Its Auto-K error worsens from 9 to 12.

The score-only arm (`fusion + projection_pca`) improves AUROC by `+0.001957`,
AUPR by `+0.008857`, FPR95 by `-0.150 pp` and OSCR by `+0.001278`, but unknown
rejection decreases by `0.075 pp`; it also changes the candidate mask and
increases Auto-K error from 9 to 12. Because the candidate mask changes, its
candidate-clustering metrics are not a paired representation-only comparison.

The full treatment (`fusion + feature_pca`) has the same rejection changes as
the score-only arm and the same all-unknown NMI/ARI changes as the clustering-
only arm. Relative to control, unknown rejection changes by `-0.075 pp`,
accepted-known accuracy by `-0.228 pp`, and Auto-K error by `+7` classes
(`9→16`). It therefore fails the preregistered full-treatment gate, despite
improvements in ranking and all-unknown clustering agreement.

## Decision

This experiment does not validate the combined full pipeline. The two
components do not show a positive interaction on seed 2118: `feature_pca`
improves clustering but harms automatic K, while fixed uncertainty fusion
improves ranking slightly without improving the calibrated unknown-rejection
working point. Their combination inherits both limitations and does not
improve the complete novel-class discovery objective.

Do not change the project default, do not tune the fusion weight or Auto-K
using seed 2118, and do not claim that separately supported components are
additively effective. Keep `feature_pca` and fixed uncertainty fusion as
explicitly reported opt-in components/ablations. The next method should target
the upstream semantic-near representation overlap or label-free class-count
estimation, rather than further combining post-hoc components.

Machine-readable result:
[`rejection_full_pipeline_factorial_s2118.json`](rejection_full_pipeline_factorial_s2118.json).
Protocol:
[`rejection_full_pipeline_factorial_protocol_s2118_20261025.md`](rejection_full_pipeline_factorial_protocol_s2118_20261025.md).
Runner:
[`../scripts/run_rejection_full_pipeline_factorial.py`](../scripts/run_rejection_full_pipeline_factorial.py).
