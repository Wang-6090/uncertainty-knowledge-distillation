# Rejection-feature KD weighting ablation (seed 2104)

## Question and protocol

Seed 2102 compared no rejection-feature KD with uncertainty-weighted rejection-feature KD. Its negative operating-point result did not identify whether the loss came from feature KD itself or from weighting examples by teacher uncertainty. This fresh-split experiment isolates that factor at a fixed nonzero coefficient.

- Fresh semantic-hard CIFAR-100 60/40 split, seed 2104.
- Uniform arm: `alpha_rejection_feat_kd=0.1`, `rejection_feature_kd_weighting=uniform`.
- Uncertainty arm: `alpha_rejection_feat_kd=0.1`, `rejection_feature_kd_weighting=teacher_uncertainty`.
- Both arms share one split-matched frozen teacher, the same initialization, student recipe, four-epoch budget, mixed-pool nnPU, known-validation checkpoint selection, feature-rejector score, and known-validation-only 95% coverage calibration.
- The weighting switch affects only rejection-feature KD. Main logits KD remains uncertainty-weighted in both arms as prescribed by the shared recipe; other losses and discovery rules are unchanged.
- No mixed-pool hard labels or test labels were used for training, checkpoint/score choice, or calibration.

## Execution and integrity

Both student histories contain four epochs, rejection-feature KD loss is nonzero in every epoch, discovery and automatic clustering completed, and the paired configuration audit passed. The runner initially failed after completing both training and discovery because its final audit hard-coded arm names `control` and `feature_kd`; the arm-name handling and completed-artifact summary path were fixed, then the runner was rerun in summary-only mode. The final JSON is [`rejection_feature_kd_weighting_s2104.json`](rejection_feature_kd_weighting_s2104.json).

Uniform / uncertainty-weighted rejection-feature KD losses by epoch were:

| Epoch | Uniform | Teacher-uncertainty weighted |
|---:|---:|---:|
| 1 | 0.437950 | 0.255388 |
| 2 | 0.300136 | 0.198873 |
| 3 | 0.258880 | 0.172720 |
| 4 | 0.235789 | 0.157698 |

The corresponding rejection-parameter gradient norms were nonzero in both arms: uniform `0.144496, 0.083576, 0.066985, 0.053922`; uncertainty-weighted `0.066709, 0.044791, 0.036510, 0.031286`. Thus the intervention was active; uncertainty weighting reduced the effective feature-KD loss/gradient on this run.

## Paired results

| Metric | Uniform | Teacher-uncertainty weighted | Difference |
|---|---:|---:|---:|
| AUROC | 0.792513 | 0.781977 | -0.010536 |
| AUPR | 0.693633 | 0.674539 | -0.019094 |
| FPR95 (lower is better) | 0.619667 | 0.633000 | +1.33 pp |
| OSCR | 0.586225 | 0.578877 | -0.007348 |
| Test known acceptance | 94.55% | 95.28% | +0.73 pp |
| Unknown rejection at calibrated threshold | 26.50% | 21.95% | -4.55 pp |
| Accepted-known accuracy | 71.13% | 70.67% | -0.46 pp |
| Correctly accepted fraction of all known | 67.25% | 67.33% | +0.08 pp |
| Automatic K / NMI / ARI | 34 / 0.47619 / 0.15111 | 32 / 0.47428 / 0.13029 | -2 / -0.00192 / -0.02082 |

The uncertainty-weighted arm failed the preregistered joint criterion: fixed-coverage unknown rejection decreased by 4.55 pp, AUROC/AUPR/OSCR declined, and FPR95 worsened by 1.33 pp. The equal-coefficient uniform arm is better on this split, but this single seed does not establish uniform feature KD as a generally useful improvement.

## Decision

This experiment attributes the seed-2102 failure pattern more specifically: uncertainty weighting itself is not beneficial relative to uniform rejection-feature KD on seed 2104, while feature KD's broader value remains unresolved because this pair does not include a no-feature-KD arm. Stop exploring teacher-uncertainty weighting for this rejection-feature KD recipe; do not promote either weighting to default based on this one split. A future claim that feature KD helps would require a separately preregistered fresh-split comparison against the no-feature-KD baseline, with the weight mode fixed in advance.

## Correction after uncertainty-source audit

The 2026-10-25 audit showed this teacher's head uncertainty is exactly `1.0`
for all checked known-validation examples; the per-epoch training weight spread
is around `4e-9`. Thus this seed-2104 pair does **not** establish that
sample-wise uncertainty weighting is worse than uniform weighting. The
reported loss/gradient differences and downstream metrics are real, but the
weighting intervention was effectively a constant `e^-1` multiplier. The
correct interpretation is that the head-uncertainty source was degenerate for
this mechanism test. Seed 2105 provides the follow-up using non-degenerate MC
epistemic uncertainty; see
[`rejection_feature_kd_mc_epistemic_s2105_20261025.md`](rejection_feature_kd_mc_epistemic_s2105_20261025.md).
