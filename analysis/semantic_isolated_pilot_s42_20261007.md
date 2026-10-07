# Semantic-isolated pilot (2026-10-07)

## Purpose

This pilot checks whether the leading mixed-pool training treatment transfers
from the semantic-hard class split to an independent semantic-isolated split.
It also separates the contribution of training from the contribution of the
post-hoc support-only nnPU rejector.

The comparison is intentionally short and is not a final result: CIFAR-100
uses 1200/300/1000 train/validation/test samples and 3 epochs. Test labels are
used only for final descriptive metrics. No test label is used to fit a
checkpoint, rejector, or threshold.

## Matched protocol

Both student arms use the same pretrained ResNet-34 teacher, pretrained
ResNet-18 student, seed 42, semantic-isolated 60/40 class split, 5400-image
mixed discovery pool, measured/requested known fraction 0.2, and the same
classification, KD, SupCon, and prototype losses.

The treatment changes only:

```text
alpha_discovery_uncertainty_pu = 0.1
discovery_uncertainty_pu_risk = nu_corrected
alpha_discovery_uncertainty_feature_margin = 0.05
discovery_uncertainty_feature_margin = 0.2
```

Detection is identical for both checkpoints: a linear support-only nnPU
rejector, `support_augmented` features, known-validation 95% coverage
calibration, and no clustering. The support rejector is trained after student
training from frozen features; it is not evidence that the backbone alone
separates known and unknown samples.

## Results

### Basic detector and training effect

| Student | Detector | AUROC | FPR95 | Known acceptance | Unknown rejection |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | normalized entropy + min-class kNN | 0.5928 | 0.8785 | 96.67% | 6.02% |
| treatment | normalized entropy + min-class kNN | 0.6178 | 0.8386 | 95.17% | 10.53% |
| baseline | support-only nnPU rejector | 0.7004 | 0.7770 | 96.01% | 14.04% |
| treatment | support-only nnPU rejector | **0.7469** | **0.7105** | 92.51% | **23.31%** |

The treatment improves the basic detector and gives an additional gain when
the same support-only rejector is applied. In this pilot, treatment versus
baseline with the support rejector is +0.0465 AUROC and +9.27 percentage
points unknown rejection, but known acceptance drops by 3.50 points.

### Feature-overlap diagnostic

| Distance reference | Baseline AUROC / overlap | Treatment AUROC / overlap |
| --- | ---: | ---: |
| classifier prototype | 0.5997 / 0.7813 | 0.6053 / 0.7908 |
| empirical class centroid | 0.5796 / 0.7955 | 0.6040 / 0.7757 |
| nearest known training sample | 0.5816 / 0.7746 | **0.6287 / 0.7734** |

This is partial evidence that local support geometry improves, not a complete
separation result. Classifier-prototype overlap is slightly worse, while
centroid and nearest-sample unknownness improve. The remaining distribution
overlap explains why the unknown rejection rate is still far from complete.

### Rejector feature ablation

On the same treatment checkpoint, adding MC uncertainty to the rejector input
was worse than support-only features:

| Rejector input | AUROC | FPR95 | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: |
| support + MC uncertainty | 0.7334 | 0.7088 | 93.34% | 21.80% |
| support-only | **0.7469** | 0.7105 | 92.51% | **23.31%** |
| support-only + 0.75 score fusion | 0.7089 | 0.7321 | 94.34% | 18.80% |

The current evidence favors a simpler support-only rejector. Score fusion is
not promoted because it reduced ranking and unknown rejection in this matched
pilot.

## Decision and next check

Keep the training treatment and support-only nnPU rejector as the leading
candidate pair, but do not claim the core overlap problem is solved. The next
required check is a multi-seed semantic-isolated replication with the same
protocol, followed by a full-data run if the direction remains positive.
Report feature-overlap diagnostics together with AUROC, FPR95, OSCR, known
acceptance, unknown rejection, accepted-known accuracy, and candidate purity.

The exact short-pilot commands are captured in
`scripts/run_semantic_isolated_pilot.ps1`.

## Seed-43 replication

The same script was rerun with only the training seed changed to 43. The
support-only detector results were:

| Student | AUROC | FPR95 | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.6939 | 0.7634 | 93.79% | **17.57%** |
| treatment | **0.7048** | **0.7383** | 94.13% | 13.86% |

The treatment still improves ranking, but the selected 95%-coverage operating
point rejects fewer unknowns. The feature audit explains the discrepancy:

| Distance reference | Baseline AUROC / overlap | Treatment AUROC / overlap |
| --- | ---: | ---: |
| classifier prototype | 0.5156 / 0.8368 | 0.5702 / 0.8180 |
| empirical class centroid | 0.4665 / 0.8538 | 0.5944 / 0.7894 |
| nearest known training sample | 0.4984 / 0.8759 | **0.6170 / 0.7515** |

Thus the treatment consistently improves local feature geometry across seeds,
but the mapping from rejector score to a global threshold is seed-sensitive.
This is a calibration/operating-point limitation layered on top of the still
overlapping distributions, not evidence that the treatment solves open-set
detection.

As a diagnostic, class-conditional 95% known-coverage thresholds were tested
on the seed-43 treatment with the same rejector. AUROC remained `0.7048`, but
known acceptance fell to `88.93%` while unknown rejection rose to `22.03%`.
Because the known false-rejection cost is too high, this is not promoted as a
fix. The next useful work is a validation-only calibration study with more
known validation samples or a calibrated rejector score, followed by another
independent seed; do not continue tuning classwise thresholds on the test set.

## Seed-44 replication and three-seed summary

The same protocol was also run with seed 44:

| Student | AUROC | FPR95 | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.7081 | 0.7994 | 92.68% | 19.89% |
| treatment | **0.7417** | **0.7070** | 96.50% | 15.32% |

The three-seed isolated means are:

| Student | AUROC | FPR95 | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.7008 | 0.7799 | 94.16% | 17.17% |
| treatment | **0.7311** | **0.7186** | 94.38% | 17.50% |

The treatment's mean unknown rejection is only 0.33 percentage points higher,
despite a clear mean ranking improvement. This confirms that the main reliable
benefit is score ordering/local geometry, not a stable operating-point gain.
The seed-44 feature audit still moved in the expected direction:

| Distance reference | Baseline AUROC / overlap | Treatment AUROC / overlap |
| --- | ---: | ---: |
| classifier prototype | 0.5694 / 0.8202 | 0.5922 / 0.7909 |
| empirical class centroid | 0.5644 / 0.8365 | 0.5945 / 0.8029 |
| nearest known training sample | 0.5763 / 0.7925 | 0.5863 / 0.7693 |

The next experiment should therefore be a full-data isolated paired run using
the existing treatment and support-only rejector, with a larger validation set
and the same validation-only calibration. New loss stacking is paused until
that experiment establishes whether the small-pilot calibration instability is
caused by sample size or by the rejector objective itself.
