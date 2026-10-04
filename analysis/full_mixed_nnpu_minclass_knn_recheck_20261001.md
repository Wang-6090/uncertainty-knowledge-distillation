# Full mixed nnPU with min-class kNN detection (2026-10-01)

## Purpose

The default Mahalanobis score uses uncertainty plus a class-conditional
Gaussian distance. The earlier predicted-class kNN score can miss an unknown
sample that is confidently assigned to the wrong known class. This evaluation
keeps the nnPU-trained student fixed and replaces only the detection score with
the minimum class-wise kNN distance over all 60 known classes.

This is a detection-side test, not a new training objective. The threshold is
calibrated from known validation samples at 95% known coverage; no test labels
are used to fit the threshold. Clustering is skipped in this audit.

## Per-seed results

| Seed | Score | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection | Accepted-known accuracy |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | nnPU + Mahalanobis | 0.6498 | 0.8283 | 0.3735 | 95.45% | 9.45% | 50.50% |
| 42 | nnPU + min-class kNN | 0.7009 | 0.7608 | 0.4134 | 95.83% | 12.78% | 51.32% |
| 123 | nnPU + Mahalanobis | 0.6533 | 0.8167 | 0.3728 | 94.90% | 12.23% | 50.61% |
| 123 | nnPU + min-class kNN | 0.6836 | 0.7710 | 0.4070 | 94.50% | 13.78% | 51.85% |
| 3407 | nnPU + Mahalanobis | 0.6441 | 0.8365 | 0.3692 | 95.38% | 9.65% | 50.43% |
| 3407 | nnPU + min-class kNN | 0.6866 | 0.7723 | 0.4105 | 95.38% | 12.53% | 51.42% |

## Mean change across seeds

| Metric | nnPU + Mahalanobis | nnPU + min-class kNN | Mean change |
| --- | ---: | ---: | ---: |
| AUROC | 0.6491 | 0.6903 | +0.0413 |
| FPR95 | 0.8272 | 0.7681 | -0.0591 |
| OSCR | 0.3718 | 0.4103 | +0.0385 |
| Known acceptance | 95.24% | 95.24% | -0.01 pp |
| Unknown rejection | 10.44% | 13.03% | +2.58 pp |
| Accepted-known accuracy | 50.51% | 51.53% | +1.02 pp |

The direction is consistent across all three seeds and the operating-point
improvement is not explained by rejecting more known samples: mean known
acceptance is unchanged and accepted-known accuracy improves. The result
supports the local-support diagnosis that predicted-class distance misses some
unknown samples.

## Automatic score selection caveat

The code now includes raw min-class kNN in the `--score-mode auto` candidate
list and has a unit test for its score computation. However, current auto
selection ranks candidates using an open validation set with known/unknown
labels. That is useful for an analysis upper bound, but it is not an unlabeled
deployment rule. In this check, auto selected `full` rather than min-class kNN.
Therefore the current reproducible method should explicitly pass
`--score-mode normalized_entropy_min_class_knn --knn-ood`; auto selection must
be reported as diagnostic until it is redesigned around label-free calibration.

## Decision

Promote `nnPU training + explicit min-class kNN detection` to the current best
detection pipeline candidate. It still does not prove that the embedding space
is fully separated, and clustering must be evaluated separately on the same
checkpoint and score protocol.
