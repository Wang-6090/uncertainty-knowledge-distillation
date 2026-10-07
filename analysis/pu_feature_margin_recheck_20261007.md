# PU-corrected uncertainty feature-margin recheck (2026-10-07)

## Question

The current uncertainty-weighted feature margin uses detached uncertainty as a
soft weight for samples in a mixed discovery pool. Because the pool contains
known samples, this can still push contaminated known features away from known
class prototypes. This experiment tested a positive-unlabeled risk correction:
subtract the estimated known contribution from the mixed-pool margin risk, then
apply the remaining non-negative unknown risk to the feature/prototype pair.

The implementation is exposed as
`--discovery-uncertainty-feature-margin-mode pu_corrected`. The historical
behavior remains the default `soft_weighted` mode. The known correction is
detached, so the correction cannot create a gradient that repels labeled known
features.

## Controlled pilot

Both arms used the same semantic-hard CIFAR-100 60/40 split, seed 42,
pretrained ResNet-34 teacher and ResNet-18 student, 1200/300/1000 train/val/test
limits, 400-sample mixed discovery pool with observed known fraction 0.21,
two epochs, `alpha_discovery_uncertainty_pu=0.1`, feature-margin weight 0.05,
cosine margin 0.2, min-class kNN detection with `k=10`, MC=4, and a threshold
calibrated from known validation samples for 95% known coverage. The only
training variable was the feature-margin estimator.

| Metric | soft_weighted | pu_corrected | Change |
| --- | ---: | ---: | ---: |
| AUROC | 0.5222 | 0.4804 | -0.0418 |
| FPR95 | 0.9316 | 0.9573 | +0.0256 |
| OSCR | 0.1280 | 0.0915 | -0.0365 |
| Known acceptance | 95.38% | 93.33% | -2.05 pp |
| Unknown rejection | 6.51% | 6.02% | -0.48 pp |
| Candidate purity | 50.00% | 39.06% | -10.94 pp |

The training meter was non-zero in both epochs (`0.0046` and `0.0138`), so
the branch was active. The negative result is therefore not a disabled-loss or
argument-attribution issue.

## Decision

The PU-corrected feature margin is rejected as a current improvement. The
correction is too conservative and, under this pilot, does not produce a
better separation boundary. It remains available as an explicitly named
ablation, but it must not replace the original uncertainty-weighted margin or
be combined with more losses without a new predeclared hypothesis.

The current primary candidate remains full-data mixed-pool nnPU plus the
original uncertainty-weighted feature margin and min-class kNN detection. The
large residual feature overlap still requires a separately validated
representation or rejector objective; threshold tuning and further PU-margin
variants are not the next step.

## Artifacts

- `runs/pu_margin_pilot_soft_s42/`
- `runs/pu_margin_pilot_corrected_s42/`
- `novel_discovery/losses.py`: `pu_unknown_feature_margin_loss`
- `train.py`: `--discovery-uncertainty-feature-margin-mode`
