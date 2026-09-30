# Candidate-gated feature separation: seed-3407 recheck

## Purpose

This is a paired recheck of the candidate-gated feature-separation loss. The
baseline and variant use CIFAR-100 semantic-hard 60/40, the same teacher
checkpoint, pretrained ResNet-18 student, full known training data, 5 epochs,
the same mixed discovery pool, and the same detector. The only training
difference is the candidate-gated feature-separation term:

```text
alpha_discovery_feature_separation=0.05
discovery_feature_separation_margin=0.0
discovery_feature_separation_temperature=0.1
discovery_feature_candidate_gating=true
discovery_select_ratio=0.25
discovery_select_mode=entropy_uncertainty
```

Detection uses `normalized_entropy_mahalanobis`, `mc_samples=4`, no clustering,
and a threshold calibrated from known validation samples to target 95% known
coverage. Test labels are used only for final evaluation.

## Detection result

| model | AUROC | FPR95 | OSCR | unknown rejection | known acceptance | accepted-known accuracy |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.5190 | 0.9138 | 0.2789 | 5.71% | 94.66% | 44.63% |
| candidate-gated | 0.5565 | 0.9000 | 0.3084 | 5.95% | 93.45% | 47.05% |
| delta | +0.0375 | -0.0138 | +0.0294 | +0.24 pp | -1.21 pp | +2.42 pp |

The variant improves all ranking metrics and the actual unknown rejection rate
slightly. It also improves the accuracy of accepted known samples. The small
decrease in known acceptance means the gain is not simply produced by a large
increase in known false rejection, but the operating point is still below the
95% target and the unknown rejection remains low.

## Feature diagnostic

The diagnostic uses test labels only to stratify known and unknown distances;
it does not fit a model or choose a threshold.

| distance score | baseline AUROC | candidate-gated AUROC | baseline overlap | candidate overlap |
|---|---:|---:|---:|---:|
| classifier prototype distance | 0.5826 | 0.6145 | 0.7850 | 0.7887 |
| empirical class-centroid distance | 0.4993 | 0.5645 | 0.8071 | 0.8378 |
| nearest known training sample distance | 0.5494 | 0.5885 | 0.8229 | 0.8017 |

The AUROC improvements suggest that the feature-separation term changes the
representation in the intended direction. However, histogram overlap does
not improve for every distance, so the representation is not yet uniformly
separated. The method remains a promising candidate rather than a solved
unknown detector.

## Next controlled test

The next run keeps the seed-3407 protocol and all weights fixed, and changes
only candidate selection from `entropy_uncertainty` to `consensus`. The aim is
to test whether multi-signal candidate purity is more important than a single
entropy-plus-uncertainty ranking. Success requires improvement in the same
metrics without a materially larger loss of known acceptance.

## Candidate selector follow-up

Purpose: test whether the result depends on entropy-plus-uncertainty selecting
noisy mixed-pool candidates. The follow-up keeps the seed, data, teacher,
student architecture, optimizer, loss weight, margin, temperature, ratio,
training duration, detector and threshold protocol fixed. The only change is
`discovery_select_mode=consensus` in place of `entropy_uncertainty`.

| variant | AUROC | FPR95 | OSCR | unknown rejection | known acceptance | accepted-known accuracy |
|---|---:|---:|---:|---:|---:|---:|
| entropy + uncertainty | 0.5565 | 0.9000 | 0.3084 | 5.95% | 93.45% | 47.05% |
| consensus | 0.5598 | 0.9103 | 0.2755 | 5.24% | 94.48% | 41.97% |

Consensus raises AUROC by only 0.0033, while FPR95 and OSCR worsen, unknown
rejection decreases by 0.71 percentage points, and accepted-known accuracy
decreases by 5.08 points. This is not a useful improvement at the operating
point and does not justify further consensus tuning.

The consensus feature diagnostic is mixed: prototype-distance AUROC changes
from 0.5826 to 0.6042 and nearest-training-sample AUROC from 0.5494 to 0.5878,
but empirical-centroid AUROC is 0.5596 (below the entropy variant's 0.5645).
Histogram overlap also improves for some distances and worsens for others.
Therefore the feature-separation objective has some representation-level
signal, but the selector choice does not consistently translate that signal
into better rejection.

## Decision

Keep candidate-gated feature separation as an experimental ablation, not as a
default or a claimed solution. Seed 3407 supports a small positive paired
effect over its baseline; seed 123 showed improved ranking/accepted-known
accuracy but slightly worse unknown rejection and excessive known acceptance;
the earlier seed-42 evidence was mixed. Across seeds, the actual unknown
rejection remains around 4–6%, so the core problem is not solved. Stop selector
search for this loss. The next direction should investigate the calibration
and detector model separately (including a held-out open-validation protocol
and frozen detector comparison), while preserving this loss as a fixed
representation candidate rather than combining more training terms.

## Frozen detector comparison: local kNN support

To check whether local support captures information missed by the Mahalanobis
detector, the seed-3407 baseline and entropy-gated checkpoints were evaluated
without retraining. All settings were held fixed except the score mode. kNN
used the complete known training set as its support bank, normalized student
`features`, `k=10`, and known-validation calibration targeting 95% known
coverage. No unknown labels entered the support bank or threshold.

| checkpoint | score | AUROC | FPR95 | OSCR | unknown rejection | known acceptance | accepted-known accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | normalized entropy + Mahalanobis | 0.5190 | 0.9138 | 0.2789 | 5.71% | 94.66% | 44.63% |
| baseline | kNN distance only | 0.5299 | 0.9190 | 0.2738 | 5.95% | 94.83% | 44.55% |
| baseline | normalized entropy + kNN | 0.5598 | 0.8966 | 0.3070 | 5.48% | 95.17% | 44.93% |
| entropy-gated | normalized entropy + Mahalanobis | 0.5565 | 0.9000 | 0.3084 | 5.95% | 93.45% | 47.05% |
| entropy-gated | kNN distance only | 0.5791 | 0.9052 | 0.3231 | 5.95% | 95.69% | 48.11% |
| entropy-gated | normalized entropy + kNN | **0.6069** | **0.8621** | **0.3496** | **7.38%** | 96.38% | **48.30%** |

The combined normalized-entropy + kNN score is the strongest of these frozen
detector comparisons. On the entropy-gated checkpoint versus baseline, it
improves AUROC by 0.0471, FPR95 by 0.0345, OSCR by 0.0427, and unknown rejection
by 1.90 percentage points; known acceptance and accepted-known accuracy also
rise. This is evidence that local support is more useful than a single
Gaussian/Mahalanobis support model on this split and seed. It remains one seed,
so the result is a promising detector ablation, not yet a general conclusion.

The next experiment should replicate the frozen-score comparison on at least
one independent seed using the same known-only support-bank protocol. Only if
the direction persists should kNN become the primary detector baseline; then
report its inference/storage cost and repeat the representation-training
ablation under the fixed kNN score.

## Independent seed-123 frozen-score replication

To test seed sensitivity, the same two score modes were evaluated on the
existing seed-123 baseline and candidate-gated checkpoints. The support bank,
feature space, `k=10`, known-only threshold calibration, and all detection
settings match the seed-3407 experiment. No training was repeated and no
threshold or method was selected from the seed-123 test labels.

| seed | checkpoint | score | AUROC | FPR95 | OSCR | unknown rejection | known acceptance | accepted-known accuracy |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 123 | baseline | normalized entropy + Mahalanobis | 0.5063 | 0.9146 | 0.2443 | 4.22% | 94.64% | 40.71% |
| 123 | baseline | normalized entropy + kNN | 0.5475 | 0.8878 | 0.2776 | 3.97% | 95.31% | 41.12% |
| 123 | candidate-gated | normalized entropy + Mahalanobis | 0.5611 | 0.9112 | 0.2896 | 3.97% | 96.65% | 44.54% |
| 123 | candidate-gated | normalized entropy + kNN | 0.5861 | 0.8275 | 0.3179 | 9.68% | 93.97% | 46.17% |

On both seed-123 checkpoints, the kNN composite improves AUROC, FPR95, and
OSCR over Mahalanobis. However, for the candidate-gated checkpoint its 9.68%
unknown rejection comes with a 93.97% known acceptance rate, below the target;
part of the increase can therefore reflect operating-point drift. On the
baseline checkpoint, kNN improves ranking metrics but does not increase
unknown rejection. Combined with seed 3407, this supports kNN as a promising
ranking/detector candidate, but does not establish a robust operating-point
gain across seeds. The representation loss itself also remains seed-sensitive.

Before calling this a solution, the next validation should use a separately
reserved known calibration set and an independent open-validation set to
choose/assess the threshold protocol, then report risk-coverage curves or
unknown rejection at matched *test-reported* known coverage as a descriptive
comparison (without choosing thresholds on test labels). Include at least a
third seed. Keep the kNN detector and candidate-gated training term as separate
factors in that comparison so their effects are not conflated.

## Independent seed-42 frozen-score replication

The same kNN score was also evaluated on existing seed-42 checkpoints. The
training configs confirm matching 5-epoch budgets and teacher checkpoint,
with candidate feature separation as the intended training difference.

| checkpoint | score | AUROC | FPR95 | OSCR | unknown rejection | known acceptance | accepted-known accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | normalized entropy + kNN | 0.5933 | 0.8701 | 0.3298 | 12.53% | 92.31% | 47.22% |
| candidate-gated | normalized entropy + kNN | 0.5826 | 0.8735 | 0.3352 | 7.95% | 95.21% | 47.40% |

The baseline's higher unknown rejection is accompanied by accepting substantially
fewer known samples, so it is not a fair gain at the intended 95% coverage.
The candidate model is close to the target known acceptance, but its AUROC is
lower and unknown rejection is also lower than that baseline operating point.
This seed does not support a consistent benefit from candidate-gated training.

## Cross-seed interpretation and protocol issue

The detector ranking trend is encouraging but not conclusive: normalized
entropy + kNN raises AUROC over normalized entropy + Mahalanobis for both
baseline and candidate checkpoints on seeds 123 and 3407, and for seed 42
candidate. The seed-42 baseline is also higher than its Mahalanobis score,
though its operating point is poorly transferred. Candidate-gated training
beats its same-seed baseline on the kNN AUROC for seeds 123 and 3407, but not
42. Thus neither kNN's operating-point gain nor the representation loss is yet
robustly established.

The test known-acceptance rates vary around a threshold calibrated to 95% on
known validation samples (for example, seed-42 baseline 92.31% and seed-3407
candidate 96.38%). This is expected finite-sample/shift behavior, but it means
unknown rejection must always be read alongside the *observed* known
acceptance; raw reject-rate comparisons can be misleading. A priority protocol
fix is to separate the validation samples used to select the training
checkpoint from the known-only samples used to calibrate the detector. Any
open-validation data used to select score/threshold policy must be a third,
separate partition. Keep test labels strictly for final reporting. Once this
separation is implemented, rerun the fixed kNN-vs-Mahalanobis comparison and
report coverage-risk curves plus unknown rejection near matched known coverage.
