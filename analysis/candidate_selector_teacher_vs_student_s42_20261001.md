# Teacher-guided candidate selection recheck (seed 42, 2026-10-01)

## Purpose

This is the second-seed replication of the teacher-versus-student candidate
selector comparison. The question is whether a frozen teacher reduces mixed
discovery-pool contamination in candidate-gated feature separation, and
whether that improvement transfers to open-set detection rather than only to
candidate-pool purity.

## Controlled protocol

Both runs use the same CIFAR-100 random 60/40 split, seed 42, complete known
training partition, matched pretrained ResNet-34 teacher, pretrained ResNet-18
student, 300 validation samples, 1200 mixed discovery samples, five student
epochs, uncertainty KD, mixed-pool nnPU, candidate-gated feature separation,
`entropy_uncertainty` selection, and identical detection settings:
`normalized_entropy_min_class_knn`, known-only validation threshold, oracle
`K=40`, and `feature_pca` clustering. The test subset contains 1000 images.
The only intended training change is the selector source:

- control: `--discovery-selection-model student`;
- treatment: `--discovery-selection-model teacher`.

## Detection and clustering results

| Metric | Student selector | Frozen teacher selector | Teacher - student |
| --- | ---: | ---: | ---: |
| AUROC | 0.6709 | 0.6753 | +0.0045 |
| AUPR | 0.5639 | 0.5536 | -0.0103 |
| FPR95 | 0.8132 | 0.8000 | -0.0132 |
| OSCR | 0.3702 | 0.3729 | +0.0027 |
| Known acceptance | 95.87% | 95.37% | -0.50 pp |
| Unknown rejection | 15.44% | 14.43% | -1.01 pp |
| Accepted-known accuracy | 47.24% | 48.01% | +0.08 pp |
| Candidate purity | 0.7093 | 0.6706 | -0.0387 |
| Candidate unknown NMI | 0.7837 | 0.8011 | +0.0173 |
| Unknown-only cluster NMI | 0.7730 | 0.8051 | +0.0321 |

The teacher selector slightly improves AUROC, FPR95, OSCR, and clustering NMI,
but lowers candidate purity and raw validation-threshold unknown rejection.
Therefore it does not dominate the student selector on this seed either.

## Matched known-coverage diagnostic

To remove the threshold-transfer effect, each test score distribution was
retrospectively recalibrated using the test-known scores to approximately 95%
known acceptance. This uses test labels and is diagnostic only, not a
deployable calibration rule.

| Metric | Student selector | Frozen teacher selector |
| --- | ---: | ---: |
| Test known acceptance | 94.88% | 94.88% |
| Unknown rejection | 16.71% | 14.68% |
| Accepted-known accuracy | 47.74% | 48.26% |

At the matched operating point, the teacher selector is slightly worse on
unknown rejection for seed 42. This contradicts the seed-123 advantage and
shows that the selector effect is seed-sensitive.

## Decision

The teacher selector remains a useful optional ablation because it can improve
candidate clustering quality and gives a stable-target comparison. It is not
promoted to the default and is not evidence that the known/unknown feature
overlap has been solved. Further selector tuning is lower priority than a
representation objective that learns a reliable known support boundary.

