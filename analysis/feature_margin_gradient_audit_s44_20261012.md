# Uncertainty feature-margin gradient audit (seed 44, 1 epoch)

## Purpose

The earlier feature-margin comparison showed a small operating-point change,
but its gradient diagnostic was accidentally wired to the disabled historical
`discovery_feature_margin` field. This audit reruns one epoch with the actual
`discovery_uncertainty_feature_margin` objective connected to
`--audit-loss-gradients`. It checks whether the loss really reaches the shared
encoder; it is not a performance claim.

## Controlled setting

- CIFAR-100 semantic-isolated 60/40 split, seed 44, CIFAR transform.
- Full train/validation data, batch size 64, one student epoch, CUDA.
- Matched mixed discovery pool of 5,400 samples, known prior 0.2.
- Same teacher/student, optimizer, detector and training protocol as the
  feature-margin comparison.
- Uncertainty feature-margin coefficient `0.05`, cosine margin `0.2`,
  uncertainty soft weighting. Only the gradient-audit flag was added for this
  mechanism check.

## Result

The logged raw gradient norms averaged over 338 batches were:

| quantity | value |
|---|---:|
| CE gradient norm | 9.8114 |
| uncertainty feature-margin raw gradient norm | 0.1520 |
| effective feature-margin gradient norm (`0.05 * raw`) | 0.0076 |
| effective feature-margin / CE | about 0.08% |
| cosine(CE, raw feature-margin) | 0.0025 |
| logged feature-margin loss | 0.00834 |
| mean uncertainty weight | 0.7787 |

The loss and raw gradient are non-zero, so the previous zero in the
diagnostic was a wiring error, not evidence that the training loss was
inactive. After applying its training coefficient, however, the update is
very small relative to CE and nearly orthogonal to it.

## Interpretation

The five-epoch feature-margin treatment did improve the selected operating
point in one seed (unknown rejection 23.68% to 30.70%), but its overall
ranking and geometry changes were weak and not directionally consistent:

- AUROC: 0.80917 to 0.80981;
- AUPR: 0.70519 to 0.70466;
- FPR95: 0.59433 to 0.59233;
- OSCR: 0.52846 to 0.53135;
- classifier-prototype distance AUROC: 0.6596 to 0.6628;
- empirical-centroid distance AUROC: 0.7182 to 0.7137;
- nearest-known-sample distance AUROC: 0.7394 to 0.7383.

Thus the rejection-rate increase is mainly a work-point change, not evidence
that the shared feature distributions have been robustly separated. The
feature-margin implementation can remain as an explicitly enabled ablation,
but it should not become the default or be stacked with more unverified
boundary losses. A future follow-up would need a stronger, directly scaled
feature-geometry objective and matched multi-seed validation.

Artifact: `runs/gradient_audit_cifar_s44_20261012_feature_margin_1ep_audit/`.

