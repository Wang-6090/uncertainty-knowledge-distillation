# Rejection-branch PU feature-margin paired pilot (2026-10-10)

## Purpose

This pilot tests whether the newly added mixed-pool PU feature-margin loss on
the independent rejection embedding improves the known/unknown separation.
The comparison changes only
`alpha_discovery_rejection_pu_feature_margin`: `0.0` in the structural
baseline and `0.1` in the treatment.

## Fixed protocol

- CIFAR-100 semantic-isolated 60/40 split.
- Seed 42.
- Same pretrained ResNet-34 teacher and ResNet-18 student setup.
- Same full-data training split, batch size 64, and 3 student epochs.
- Same independent rejection embedding dimension: 128.
- Same mixed discovery pool: 5,400 samples with known fraction 0.2.
- Same existing mixed-pool uncertainty nnPU and uncertainty feature-margin
  losses in both arms.
- Detection: matched 5,400-sample pool, linear `nu_corrected` nnPU rejector,
  `support_augmented` features, MC=4, 95% known-coverage calibration, and
  `feature_rejector` score.
- Clustering was skipped because this experiment isolates unknown detection.

## Training-path check

The baseline history reports
`discovery_rejection_pu_feature_margin = 0.0` in every epoch. The treatment
reports approximately `0.002105`, `0.0000045`, and `0.0000050` in epochs 1–3.
Thus the new loss is connected to the training graph and is not merely a
configuration-only change. Its value rapidly approaches zero, which is an
important diagnostic.

## Results

| Metric | Baseline | PU feature-margin treatment | Difference |
|---|---:|---:|---:|
| AUROC | 0.7823 | 0.7762 | -0.0062 |
| FPR95 (lower is better) | 0.6318 | 0.6263 | -0.0055 |
| OSCR | 0.3723 | 0.3701 | -0.0022 |
| Known acceptance | 94.57% | 95.25% | +0.68 pp |
| Unknown rejection | 23.15% | 21.08% | -2.08 pp |

## Interpretation

This is not evidence that the rejection-branch PU feature margin solves the
feature-overlap problem. The small FPR95 improvement is accompanied by lower
AUROC, OSCR, and unknown rejection, so it is not a consistent improvement.
The loss being nearly zero after the first epoch suggests that the hard
`clamp_min(0)` non-negative PU correction is saturating or that the chosen
prototype margin does not provide a useful training signal. Increasing the
threshold or repeatedly tuning the detector would not establish a better
representation.

Decision: keep the implementation for a negative ablation record, but do not
promote it to the main method and do not spend more experiments on its weight
or detection threshold alone. The next code change should address the
non-saturating PU risk formulation or use a different source of unknown
separation, followed by the same paired protocol.

## Verification

- Both student trainings completed on CUDA.
- Both detection runs used the same command-level protocol.
- The two training histories confirm that only the new PU feature-margin term
  differs.
- Previous repository checks remain passing: 191 tests, compileall, and
  `git diff --check`.

