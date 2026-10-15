# Uncertainty-score fusion weight recheck (2026-10-13)

## Question and pre-registered comparison

The current bottleneck is overlap between known and unknown scores/features, not a threshold-selection issue. This retrospective detector-only check asks whether the head-uncertainty correction in `normalized_entropy_uncertainty_min_class_knn` adds a repeatable ranking benefit.

- Fixed student checkpoint within each seed (`nnpu_encoder_gradient_s{44,123}_encoder_grad_student/student.pt`).
- CIFAR-100 semantic-isolated 60/40 split; full test set (6,000 known and 4,000 unknown); MC=4; same kNN feature bank and score normalization protocol.
- Same checkpoint-selection and threshold-calibration validation indices within each seed (selection/calibration hashes match across the three weights).
- Only the uncertainty correction weight changes: 0.25, 0.50, or 0.75. The 0.50 run is the baseline.
- Test labels are used only for retrospective metrics and paired bootstrap confidence intervals. These results must not be used to tune a production threshold or choose a weight.

The score form is the sum of z-normalized entropy and minimum-class kNN distance, plus `w` times z-normalized head uncertainty. This tests score fusion only; it does not change training or the learned representation.

## Full-test results

Metrics below come from each run's `discovery_report.json`. The run report's operating point is calibrated on known validation samples to target 95% known coverage; finite-sample test coverage can differ slightly.

| Seed | uncertainty weight | AUROC | AUPR | FPR95 | OSCR | test known accept | test unknown reject | KS | unknown below known q95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 123 | 0.25 | 0.8131 | 0.7179 | 0.6012 | 0.5638 | 94.62% | 28.18% | 0.4886 | 73.48% |
| 123 | 0.50 | 0.8125 | 0.7198 | 0.6030 | 0.5632 | 93.80% | 31.50% | 0.4869 | 72.83% |
| 123 | 0.75 | 0.8097 | 0.7187 | 0.6137 | 0.5611 | 93.73% | 32.53% | 0.4819 | 72.35% |
| 44 | 0.25 | 0.8177 | 0.7159 | 0.5808 | 0.5334 | 94.78% | 28.63% | 0.4947 | 72.08% |
| 44 | 0.50 | 0.8191 | 0.7183 | 0.5755 | 0.5336 | 94.72% | 28.88% | 0.4918 | 71.98% |
| 44 | 0.75 | 0.8184 | 0.7183 | 0.5832 | 0.5325 | 94.98% | 27.53% | 0.4882 | 72.50% |

Higher unknown rejection at the validation-calibrated threshold is not by itself evidence of better separation: for example, on seed 123, weight 0.75 raises rejection relative to 0.25 while AUROC, FPR95, OSCR and KS all worsen.

## Paired bootstrap (2,000 stratified resamples)

The script `scripts/paired_open_score_bootstrap.py` compares each treatment to weight 0.50 on the same ordered test examples. Its “exact 95% known threshold” is calculated from test-known scores for retrospective sensitivity analysis only; it is not a deployable calibration procedure.

| Seed | weight vs 0.50 | AUROC delta (95% CI) | unknown-rejection delta at exact 95% known coverage (95% CI) |
|---:|---:|---:|---:|
| 123 | 0.25 | +0.00054 [-0.00039, +0.00151] | -0.00518 [-0.01675, +0.00625] |
| 123 | 0.75 | -0.00282 [-0.00368, -0.00196] | +0.00414 [-0.00875, +0.01700] |
| 44 | 0.25 | -0.00141 [-0.00226, -0.00056] | +0.00044 [-0.01050, +0.01150] |
| 44 | 0.75 | -0.00069 [-0.00143, +0.00008] | -0.00200 [-0.01350, +0.01101] |

The small AUROC differences have opposite effects for weight 0.25 across seeds. Weight 0.75 reduces AUROC on both seeds; its seed-123 decrease is clearly outside that test's paired-bootstrap interval, while seed 44 is borderline. None of the four fixed-coverage unknown-rejection intervals excludes zero. The apparent weight-0.25 benefit on seed 123 is therefore not a reproducible multi-seed improvement.

## Conclusion and next step

- Do not change the default fusion weight (0.50) based on these data; do not describe this as solving known/unknown overlap.
- Treat head uncertainty as a weak, seed-sensitive score correction, not a robust new detector signal. Higher rejection under some settings partly trades against known acceptance and does not establish better ranking.
- The detector-only experiment cannot improve the representation. Next work should target train-time geometry/objectives and validate with a predeclared validation protocol on fresh seeds; keep test labels out of model/weight selection.
- Existing outputs and bootstrap JSON: `runs/nnpu_uncertainty_score_{s,w025,w075}_s{44,123}_encoder_grad_discover/` and `analysis/uncertainty_weight_bootstrap_s{44,123}_w{025,075}_vs_w05.json`.

