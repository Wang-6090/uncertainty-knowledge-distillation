# Strict feature-KD / prototype-loss ablation (2026-10-11)

## Purpose

The current bottleneck is overlap between known and unknown feature/score
distributions. Earlier open-validation experiments suggested that disabling
feature KD and prototype loss might help, but those checkpoints also used a
different `open_val_ratio`, so the result could not be attributed to the two
losses. This is a strict paired audit: only `alpha_feat_kd` and
`alpha_proto` are changed.

## Fixed protocol

- CIFAR-100 semantic-isolated 60/40 split
- seed `123`
- pretrained ResNet-34 teacher and ResNet-18 student
- full training/validation/test data, 5 student epochs
- matched mixed discovery pool: 5,400 samples, known prior `0.2`
- uncertainty nnPU configuration and optimizer unchanged
- MC samples `4`
- detector: `normalized_entropy_min_class_knn`, feature kNN `k=10`
- threshold: known-validation-only 95% known coverage
- clustering skipped

The reference checkpoint is `runs/distance_consensus_full_s123_baseline` with
`alpha_feat_kd=0.1`, `alpha_proto=0.1`. The paired checkpoint is
`runs/semantic_isolated_full_s123_no_feat_proto` with both weights set to zero.
Test labels are used only for final reporting, never for training or threshold
selection.

## Detection results

| Variant | AUROC | FPR95 | OSCR | known acceptance | unknown rejection | known accuracy |
|---|---:|---:|---:|---:|---:|---:|
| feature KD + prototype loss | 0.7273 | 0.7455 | 0.3950 | 95.43% | 13.63% | 46.58% |
| both disabled | 0.7261 | 0.7327 | 0.3955 | 95.47% | 14.70% | 46.63% |

The no-loss variant improves FPR95 by 0.0128 and unknown rejection by 1.07
percentage points, but AUROC decreases by 0.0013 and OSCR is essentially
unchanged. This is a mixed, small effect, not evidence that either variant
solves the core overlap problem.

## Feature geometry results

Each entry is `unknownness AUROC / histogram overlap`; lower overlap is better.

| Distance reference | feature KD + prototype | both disabled |
|---|---:|---:|
| classifier prototype | 0.6894 / 0.7238 | 0.6952 / 0.7176 |
| empirical class centroid | 0.7048 / 0.6908 | 0.7100 / 0.6891 |
| nearest known training sample | 0.7203 / 0.6678 | 0.7250 / 0.6545 |

The disabled-loss model has a modestly better local-support geometry, but the
improvement does not consistently transfer to the final entropy-plus-kNN
detector. This indicates that the two losses are not the main cause of the
current failure, and that score calibration/detector coupling remains part of
the problem.

## Decision

1. Do not delete feature KD or prototype loss from the default pipeline based
   on this one seed.
2. Do not claim that these losses improve unknown separation; the strict result
   does not support that claim.
3. Keep the no-loss checkpoint as a reproducible ablation and report both
   variants in future tables.
4. Stop spending experiments on small weight changes until a representation
   diagnostic improves across seeds. The next method should explicitly improve
   known-support geometry or use an independent rejection representation, then
   be evaluated with the same fixed detector protocol.

Raw outputs:

- `analysis/feature_geometry_strict_pair_s123_20261011.json`
- `runs/distance_consensus_full_s123_baseline_recheck/`
- `runs/semantic_isolated_full_s123_no_feat_proto_discover/`

