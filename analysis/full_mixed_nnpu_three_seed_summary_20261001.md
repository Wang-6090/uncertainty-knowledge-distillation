# Full mixed nnPU three-seed summary (2026-10-01)

## Controlled comparison

This summary combines the three independent full-data mixed-pool pairs from
seeds 42, 123, and 3407. Each seed used its own matching pretrained ResNet-34
teacher, the same random CIFAR-100 60/40 split protocol, a pretrained
ResNet-18 student, five epochs, the complete known training partition, and the
complete 10,000-image open test set. Detection used MC=4,
`normalized_entropy_mahalanobis`, and a threshold calibrated only on known
validation samples at 95% known coverage. Clustering was skipped to isolate
unknown detection.

The only treatment variable was `alpha_discovery_uncertainty_pu`:

- baseline: `0.0`;
- treatment: `0.1`, with `discovery_uncertainty_known_prior=auto` and the
  measured mixed-pool known proportion `0.212598`.

## Per-seed results

| Seed | Method | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection | Accepted-known accuracy |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | baseline | 0.5866 | 0.8667 | 0.2921 | 94.50% | 6.33% | 42.01% |
| 42 | nnPU | 0.6498 | 0.8283 | 0.3735 | 95.45% | 9.45% | 50.50% |
| 123 | baseline | 0.6077 | 0.8497 | 0.3137 | 94.22% | 7.88% | 44.47% |
| 123 | nnPU | 0.6533 | 0.8167 | 0.3728 | 94.90% | 12.23% | 50.61% |
| 3407 | baseline | 0.5988 | 0.8732 | 0.3098 | 95.12% | 7.13% | 44.42% |
| 3407 | nnPU | 0.6441 | 0.8365 | 0.3692 | 95.38% | 9.65% | 50.43% |

## Mean and standard deviation

| Metric | Baseline mean +/- std | nnPU mean +/- std | Mean change |
| --- | ---: | ---: | ---: |
| AUROC | 0.5977 +/- 0.0106 | 0.6491 +/- 0.0047 | +0.0514 |
| FPR95 | 0.8632 +/- 0.0121 | 0.8272 +/- 0.0100 | -0.0360 |
| OSCR | 0.3052 +/- 0.0115 | 0.3718 +/- 0.0023 | +0.0666 |
| Known acceptance | 94.61% +/- 0.46 pp | 95.24% +/- 0.30 pp | +0.63 pp |
| Unknown rejection | 7.11% +/- 0.78 pp | 10.44% +/- 1.55 pp | +3.33 pp |
| Accepted-known accuracy | 43.63% +/- 1.41 pp | 50.51% +/- 0.09 pp | +6.88 pp |

All three seeds move in the same direction on every reported metric. This is
stronger evidence than the earlier single-seed pilot, so nnPU is now the
current primary mixed-pool candidate. It still does not solve the core
overlap: unknown rejection remains only 10.44% on average, and the previous
diagnostics showed that feature and Mahalanobis distributions remain heavily
overlapped. The correct interpretation is improved uncertainty ranking and
operating point, not complete known/unknown embedding separation.

## Combination decision

The most defensible next combination is nnPU plus class-wise KNN
support-boundary warm-up/ramp. The two terms act on different objects:

- nnPU trains the uncertainty head using labeled known samples and an
  unlabeled mixed pool;
- KNN support-boundary constrains discovery features relative to known-class
  local support, with candidate gating and delayed activation to reduce early
  feature noise.

Objectosphere, feature repulsion, selective energy, EMA candidate gating, and
post-hoc rejectors are not included in this combination because their mixed
results were either negative, unstable, or prone to candidate contamination.
The combination must first be compared with the nnPU-only checkpoint under
the same seed and detector. A positive combination result would justify a
three-seed confirmation; a negative result would keep nnPU alone and stop
stacking losses.
