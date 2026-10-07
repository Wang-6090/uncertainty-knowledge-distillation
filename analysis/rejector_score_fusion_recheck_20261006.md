# Rejector score fusion recheck (2026-10-06)

## Hypothesis

The frozen support-augmented nnPU rejector and the student detector provide
partly different signals.  A validation-normalized score fusion may improve
ranking and OSCR while avoiding the loss of known classification quality seen
with the rejector alone.

## Implementation

The code adds the optional `feature_rejector_fusion` score mode.  It combines:

- the current student detector, by default
  `normalized_entropy_min_class_knn`;
- the support-augmented nnPU rejector score.

Both signals are standardized using known validation samples only.  The first
trial used equal weights.  A second trial used rejector weight `0.75` and base
detector weight `0.25`.  Test labels are never used for fitting, normalization,
or threshold selection.  Existing score modes remain unchanged and fusion is
not the default.

All CIFAR-100 runs used the same random 60/40 split, pretrained ResNet-34
teacher and ResNet-18 student checkpoints, 64px images, 10-epoch student
training, a matched 5,400-image mixed pool with known prior `0.2`, feature kNN
with `k=10`, MC samples `8`, and a validation-only threshold targeting 95%
known coverage.  Clustering was skipped because this experiment evaluates only
the unknown detector.

## Three-seed results

| Seed | Detector | AUROC | FPR95 | OSCR | Known accuracy | Known acceptance | Unknown rejection |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026 | Student base | 0.7205 | 0.7170 | 0.4764 | 56.25% | 95.70% | 13.15% |
| 2026 | Support-only nnPU | 0.7596 | 0.6855 | 0.4775 | 55.45% | 94.62% | 22.65% |
| 2026 | Fusion, rejector 0.75 | 0.7649 | 0.6318 | 0.4923 | 56.12% | 95.53% | 20.93% |
| 42 | Student base | 0.7250 | 0.7233 | 0.4802 | 56.82% | 95.03% | 16.33% |
| 42 | Support-only nnPU | 0.7529 | 0.6853 | 0.4796 | 56.00% | 94.85% | 19.73% |
| 42 | Fusion, rejector 0.75 | 0.7610 | 0.6748 | 0.4918 | 56.53% | 94.83% | 22.75% |
| 3407 | Student base | 0.7105 | 0.7295 | 0.4593 | 54.42% | 95.57% | 12.68% |
| 3407 | Support-only nnPU | 0.7467 | 0.6683 | 0.4607 | 53.78% | 94.75% | 19.18% |
| 3407 | Fusion, rejector 0.75 | 0.7526 | 0.6638 | 0.4749 | 54.30% | 95.28% | 17.45% |

Three-seed means:

| Detector | AUROC | FPR95 | OSCR | Known accuracy | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Student base | 0.7187 | 0.7233 | 0.4720 | 55.83% | 95.43% | 14.05% |
| Support-only nnPU | 0.7531 | 0.6797 | 0.4726 | 55.08% | 94.74% | 20.52% |
| Fusion, rejector 0.75 | 0.7595 | 0.6568 | 0.4863 | 55.65% | 95.22% | 20.38% |

## Interpretation

The weighted fusion is the strongest detector by mean AUROC, FPR95, OSCR,
known accuracy, and known acceptance in this three-seed comparison.  It keeps
most of the rejector's unknown rejection benefit while reducing its cost on
known samples.  The unknown rejection mean is slightly below support-only
(`20.38%` vs. `20.52%`) and is not monotonic across seeds, so the fusion does
not eliminate the known/unknown overlap problem.

The equal-weight fusion was weaker than the 0.75-rejector version on seed 3407
(AUROC `0.7387`, FPR95 `0.6708`, OSCR `0.4721`, unknown rejection `16.58%`).
This supports giving the support-boundary signal more weight, but does not
justify tuning the weight on test results.  The `0.75` value must be treated as
a candidate setting until it is checked on an independent class split or a
held-out validation protocol.

## Decision

- Keep `feature_rejector_fusion` as an optional candidate detector, not the
  default score and not a change to student training.
- Do not add more detector features or tune the fusion weight against the
  final test set.
- Next validate the fixed `0.75` setting on an independent class split and a
  separate calibration subset, then restore clustering and report candidate
  purity, NMI, and ARI.
- The main scientific bottleneck remains representation overlap.  Fusion
  improves the ranking layer, but it cannot prove that the learned feature
  space has separated known and unknown classes.

Artifacts:

- `runs/matched_mixed_nnpu_margin_s2026_e10_fusion075/`
- `runs/matched_mixed_nnpu_margin_s2026_e10_rejector/`
- `runs/matched_mixed_nnpu_margin_s42_e10_fusion075/`
- `runs/matched_mixed_nnpu_margin_s42_e10_rejector/`
- `runs/matched_mixed_nnpu_margin_s3407_e10_fusion075/`
- `runs/matched_mixed_nnpu_margin_s3407_e10_rejector/`
