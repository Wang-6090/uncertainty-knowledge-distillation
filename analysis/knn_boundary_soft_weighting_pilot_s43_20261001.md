# Soft-weighted KNN support-boundary pilot (2026-10-01)

## Purpose and pre-registered comparison

The previous class-wise KNN support-boundary loss used a hard candidate gate:
every selected mixed-pool sample received the same boundary-loss weight. This
pilot tests whether continuous weighting can reduce damage from contaminated
candidates. The weighting follows the existing FixMatch/Mean Teacher/SCAN
motivations already used in the project: risk strength, local neighbor
agreement, and teacher/student agreement determine a continuous sample weight.

Only the KNN boundary loss changed. The baseline and hard-gated warm-up/ramp
results are the existing matched comparisons; the new checkpoint adds only
`--discovery-knn-soft-weighting`.

Fixed protocol:

- CIFAR-100 semantic-hard 60/40 split, seed 43;
- pretrained ResNet-34 teacher and ResNet-18 student;
- full known training partition, 5 epochs, batch size 32;
- mixed discovery pool, ratio 0.2, discovery limit 1200;
- KNN support `k=5`, radius quantile 0.95, margin 0.02;
- two warm-up epochs and two-epoch linear ramp;
- detection: normalized entropy + kNN distance, MC=4, disjoint calibration ratio 0.5, validation known coverage 95%;
- full 10,000-image open test set; clustering skipped for this detector audit.

## Implementation check

The new path is active, not just parsed:

| Epoch | KNN loss | Feature candidate ratio | Mean candidate weight | Schedule weight |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| 2 | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| 3 | 0.0236 | 0.2500 | 0.3086 | 0.5 |
| 4 | 0.0442 | 0.2500 | 0.3163 | 1.0 |
| 5 | 0.0535 | 0.2500 | 0.3123 | 1.0 |

The checkpoint selected by validation accuracy is
`runs/semantic_hard_candidate_knn_boundary_soft_s43/student.pt`, with
`best_epoch=4` and validation known accuracy `0.4367`.

## Detection result

| Metric | No KNN baseline | Hard KNN warm-up/ramp | Soft KNN warm-up/ramp |
| --- | ---: | ---: | ---: |
| AUROC | 0.5652 | 0.5988 | 0.5991 |
| FPR95 | 0.8910 | 0.8570 | 0.8582 |
| OSCR | 0.2947 | 0.3308 | 0.3351 |
| Known acceptance | 0.9212 | 0.9458 | 0.9385 |
| Unknown rejection | 0.0875 | 0.0918 | 0.0938 |
| Accepted-known accuracy | 0.4241 | 0.4664 | 0.4729 |
| All-known classification accuracy | 0.3907 | 0.4412 | 0.4438 |

Soft weighting gives a small operating-point trade-off over hard weighting:
unknown rejection, OSCR and accepted-known accuracy increase, while known
acceptance decreases by 0.73 percentage points. AUROC is effectively
unchanged and FPR95 is slightly worse. This is not a clean win.

## Feature-overlap result

The diagnostic uses test labels only to stratify descriptive distributions;
it does not fit a detector or choose a threshold.

| Distance statistic | Hard AUROC / overlap | Soft AUROC / overlap |
| --- | ---: | ---: |
| Classifier prototype | 0.5956 / 0.8555 | 0.6105 / 0.8187 |
| Empirical class centroid | 0.5714 / 0.8952 | 0.5688 / 0.8943 |
| Nearest known training sample | 0.5990 / 0.8482 | 0.5940 / 0.8490 |

The soft objective improves the global classifier-prototype diagnostic, but
does not improve the local nearest-support statistic. Since the final kNN
detector AUROC changes only from 0.5988 to 0.5991, the remaining overlap is
not explained by a simple hard-versus-soft candidate weighting choice.

## Decision

- Keep `--discovery-knn-soft-weighting` as an optional ablation and retain hard
  warm-up/ramp as the current reference candidate.
- Do not enable soft weighting by default and do not claim it solves unknown
  detection.
- Stop tuning this option on the same seed. A future replication should use a
  new fixed seed or split and report the same matched-coverage metrics.
- The next algorithmic direction should target class-conditional local
  support directly, or move to a unified known/novel training objective; more
  post-hoc score or candidate-weight adjustments are unlikely to remove the
  residual overlap.

Artifacts:

- Training: `runs/semantic_hard_candidate_knn_boundary_soft_s43/`
- Evaluation: `runs/semantic_hard_knn_fulltest_soft_s43/`
- Feature diagnostic: `analysis/knn_boundary_soft_feature_diagnostics_full_s43_20261001.json`
