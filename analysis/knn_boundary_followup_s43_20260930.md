# Candidate-gated KNN support-boundary loss: seed 43 follow-up

## Purpose

This experiment tests whether the newly enabled class-wise KNN support-boundary
loss changes the known/unknown feature overlap, rather than merely changing the
decision threshold. It is a paired training ablation against the existing
seed-43 disjoint-protocol baseline.

## Pre-registered comparison

Both models use CIFAR-100 semantic-hard 60/40, seed 43, the same teacher,
pretrained ResNet-18 student, full known training partition, 5 epochs, batch
size 32, the same mixed discovery pool, and the same optimizer and loss
weights. The only intentional algorithmic change is:

| Setting | Baseline | Candidate |
| --- | ---: | ---: |
| `alpha_discovery_knn_boundary` | 0 | 0.1 |
| `discovery_feature_candidate_gating` | false | true |

The candidate loss uses `k=5`, the 95th-percentile class-wise support radius,
and margin `0.02`. Training history confirms that the candidate term is active
(epoch means approximately 0.0072, 0.0514, 0.0584, 0.0600, and 0.0599).
Configuration audit found no other substantive training differences.

Detection is fixed for both checkpoints: independent calibration subset
(`calibration_ratio=0.5`), train-only known support bank, normalized entropy
plus kNN distance (`normalized_entropy_knn`, `k=10`, backbone features), target
known coverage 0.95, MC samples 4, the same test subset, and clustering skipped.
Test labels were used only for final descriptive metrics.

## Detection result

| Metric | Baseline | KNN-boundary candidate | Change |
| --- | ---: | ---: | ---: |
| AUROC | 0.5814 | 0.5780 | -0.0034 |
| FPR95 | 0.8780 | 0.8390 | -0.0390 (better) |
| OSCR | 0.2946 | 0.3452 | +0.0507 (better) |
| Known acceptance | 0.9366 | 0.9041 | -0.0325 |
| Unknown rejection | 0.0753 | 0.1195 | +0.0442 |
| Known classification accuracy after accept | 0.4219 | 0.5054 | +0.0835 |
| Known classification accuracy on all known | 0.3951 | 0.4569 | +0.0618 |

The candidate rejects more unknowns and improves accepted-known classification,
but it also rejects more known samples. Its overall ranking AUROC is slightly
worse, so this is not a clean win and must not be described as a solved
unknown-detection problem.

## Feature-overlap diagnostic

The post-hoc diagnostic uses only the fixed test labels to separate known and
unknown distributions; it does not fit a model or choose a threshold.

| Distance to known support | Baseline AUROC | Candidate AUROC | Histogram overlap |
| --- | ---: | ---: | ---: |
| Classifier prototype | 0.5782 | 0.5664 | 0.8376 -> 0.7790 |
| Empirical class centroid | 0.5503 | 0.5759 | 0.8053 -> 0.8053 |
| Nearest known training sample | 0.5893 | 0.5970 | 0.7951 -> 0.7649 |

The local-support statistic moves in the intended direction: unknown samples
are farther from the known training support and histogram overlap decreases.
However, classifier-prototype distance does not improve and residual overlap
is still substantial. This explains why the operating point improves while
the global AUROC does not clearly improve.

## Decision

Keep this loss as a research ablation, not as the default method yet. The
result is stronger than a pure threshold adjustment because the feature
diagnostic changes, but the known-coverage cost is too large and the result is
one seed. The next controlled step is an independent-seed replication with
the same fixed settings. If the same pattern appears, investigate a warm-up
or gated/ramped application of the boundary loss, with model selection based
on validation metrics and no test-label tuning. If it fails to replicate,
drop it as a general solution and retain only the diagnostic insight that
local support is more informative than classifier prototypes.

Artifacts:

- Training checkpoint: `runs/semantic_hard_candidate_knn_boundary_s43/student.pt`
- Evaluation: `runs/semantic_hard_candidate_knn_boundary_eval_s43/`
- Baseline evaluation: `runs/semantic_hard_disjoint_baseline_knn_s43/`
- Feature diagnostic: `analysis/knn_boundary_feature_diagnostics_s43_20260930.json`
- Classwise analysis: `analysis/classwise_knn_boundary_s43_20260930.json`

## Independent-seed replication: seed 44

The same full-weight ablation was repeated with seed 44. The baseline and
candidate configs again differ only in `alpha_discovery_knn_boundary` (0 vs
0.1) and the required feature candidate gate (false vs true). The candidate
loss was active in all epochs (means approximately 0.0095, 0.0548, 0.0473,
0.0530, 0.0635). Data split protocol, teacher checkpoint, architecture,
training set, schedule, detector, calibration and test-set size match the
seed-44 baseline.

| Metric | Baseline | KNN-boundary candidate | Change |
| --- | ---: | ---: | ---: |
| AUROC | 0.5690 | 0.5647 | -0.0043 |
| FPR95 | 0.8564 | 0.8513 | -0.0051 (better) |
| OSCR | 0.2688 | 0.3261 | +0.0573 (better) |
| Known acceptance | 0.9573 | 0.9197 | -0.0376 |
| Unknown rejection | 0.0361 | 0.0819 | +0.0458 |
| Known classification accuracy after accept | 0.3714 | 0.4814 | +0.1100 |
| Known classification accuracy on all known | 0.3556 | 0.4427 | +0.0872 |

Feature diagnostics are mixed rather than uniformly improved. Nearest-known
training-sample distance AUROC increases `0.5543 -> 0.5625`, but its histogram
overlap only changes `0.8026 -> 0.7937`; classifier-prototype distance AUROC
increases `0.5627 -> 0.5758`, with overlap `0.7917 -> 0.8045`; empirical
centroid overlap worsens `0.8389 -> 0.8578`. Thus the detection operating
point is repeatable in direction across two seeds, but the feature geometry
does not improve consistently across all support summaries.

## Warm-up/ramp follow-up: seed 43

To reduce early-training damage, the same seed-43 candidate was retrained with
the only additional change being a KNN-loss schedule: two warm-up epochs and a
two-epoch linear ramp. The effective multipliers were verified from history
as `[0, 0, 0.5, 1, 1]`; the corresponding raw loss was zero in the first two
epochs and active afterwards. All data, initialization, support parameters,
candidate gate, and evaluation settings match the full-weight seed-43 run.

| Metric | Baseline | Full weight | Warm-up/ramp |
| --- | ---: | ---: | ---: |
| AUROC | 0.5814 | 0.5780 | 0.6102 |
| FPR95 | 0.8780 | 0.8390 | 0.8537 |
| OSCR | 0.2946 | 0.3452 | 0.3454 |
| Known acceptance | 0.9366 | 0.9041 | 0.9382 |
| Unknown rejection | 0.0753 | 0.1195 | 0.1143 |
| Accepted-known accuracy | 0.4219 | 0.5054 | 0.4974 |

Warm-up/ramp improves AUROC and recovers almost all baseline known acceptance,
while retaining higher unknown rejection and OSCR than baseline. Relative to
the full-weight candidate, it raises known acceptance by 3.41 percentage
points while reducing rejection by only 0.53 points. This is promising but is
still a single-seed schedule comparison and should be replicated before it is
selected.

The matching feature diagnostic supports a real representation effect:
nearest-known training-sample distance AUROC improves `0.5893 -> 0.6110` and
histogram overlap changes only slightly `0.7951 -> 0.7949`; classifier
prototype-distance AUROC improves `0.5782 -> 0.6104`, overlap
`0.8376 -> 0.7659`. The remaining substantial overlap confirms that the core
problem is not solved.

### Follow-up decision

Keep the full KNN support-boundary loss as a candidate, but prefer testing the
warm-up/ramp variant in the next independent seed. Do not set it as the
default yet. The seed-43/44 full-weight runs consistently trade known
coverage for rejection; seed-43 warm-up/ramp appears to soften that trade-off.
Next compare warm-up/ramp against its matched baseline on seed 44 using the
same detector and validation protocol. If the benefit does not replicate,
drop the scheduling claim. If it does, run a later multi-seed summary and
optimize the current redundant epoch support-bank construction before
promoting the loss.

The implementation now exposes `--discovery-knn-warmup-epochs` and
`--discovery-knn-ramp-epochs`. A zero effective weight skips support-bank
collection; this is a semantics-preserving runtime optimization because no
KNN boundary gradient is used during those epochs.

Additional artifacts:

- Seed-44 checkpoint: `runs/semantic_hard_candidate_knn_boundary_s44/student.pt`
- Seed-44 evaluation: `runs/semantic_hard_candidate_knn_boundary_eval_s44/`
- Seed-44 feature diagnostic: `analysis/knn_boundary_feature_diagnostics_s44_20260930.json`
- Warm-up checkpoint: `runs/semantic_hard_candidate_knn_boundary_warmup_s43/student.pt`
- Warm-up evaluation: `runs/semantic_hard_candidate_knn_boundary_warmup_eval_s43/`
- Warm-up feature diagnostic: `analysis/knn_boundary_warmup_feature_diagnostics_s43_20260930.json`

### Independent warm-up/ramp replication: seed 44

The warm-up/ramp setting was repeated on seed 44 with the same schedule and
all other conditions matched to its full-weight candidate and baseline. The
verified effective loss multipliers are `[0, 0, 0.5, 1, 1]`; the KNN loss is
zero in the first two epochs and nonzero thereafter.

| Metric | Seed-44 baseline | Full weight | Warm-up/ramp |
| --- | ---: | ---: | ---: |
| AUROC | 0.5690 | 0.5647 | 0.5984 |
| FPR95 | 0.8564 | 0.8513 | 0.8188 |
| OSCR | 0.2688 | 0.3261 | 0.3351 |
| Known acceptance | 0.9573 | 0.9197 | 0.9470 |
| Unknown rejection | 0.0361 | 0.0819 | 0.0771 |
| Accepted-known accuracy | 0.3714 | 0.4814 | 0.4747 |

On this seed, warm-up/ramp improves AUROC, FPR95 and OSCR over baseline while
retaining nearly baseline known acceptance (down 1.03 percentage points) and
more than doubling unknown rejection. It also avoids the larger known
acceptance loss of the full-weight variant. Post-hoc diagnostics show
nearest-training-support AUROC `0.5543 -> 0.6087` and overlap
`0.8026 -> 0.7771`; centroid AUROC is `0.5410 -> 0.5868`, overlap
`0.8389 -> 0.7695`. These are consistent with a representation-space effect.

Across seeds 43 and 44, the warm-up/ramp setting has a consistent favorable
direction on AUROC, FPR95, OSCR and unknown rejection without the large known
coverage penalty seen with full-strength-from-epoch-1. This is promising
replication evidence, but still only two seeds on one semantic-hard CIFAR-100
split and a 1,000-image test subset; the method is not yet established as
general. Next, preserve these seeds as exploratory validation and run a
larger independent seed or a second split before promotion. Also measure the
full 10,000-image test protocol when compute permits. Do not tune the schedule
or score from these test labels.

Additional seed-44 artifacts:

- Warm-up checkpoint: `runs/semantic_hard_candidate_knn_boundary_warmup_s44/student.pt`
- Warm-up evaluation: `runs/semantic_hard_candidate_knn_boundary_warmup_eval_s44/`
- Warm-up feature diagnostic: `analysis/knn_boundary_warmup_feature_diagnostics_s44_20260930.json`

## Full 10,000-image test verification

The most promising comparison was rerun without a test-size limit. The
student checkpoints, detector, calibration split, kNN bank and target known
coverage remained fixed; only the test sample count changed from 1,000 to the
full 10,000 CIFAR-100 test images.

| Seed | Model | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 43 | Baseline | 0.5652 | 0.8910 | 0.2947 | 0.9212 | 0.0875 |
| 43 | Warm-up/ramp | 0.5988 | 0.8570 | 0.3308 | 0.9458 | 0.0918 |
| 44 | Baseline | 0.5584 | 0.8790 | 0.2585 | 0.9483 | 0.0598 |
| 44 | Warm-up/ramp | 0.6016 | 0.8357 | 0.3475 | 0.9510 | 0.0700 |

The full-test result preserves the direction found on the limited test subset:
both seeds improve AUROC, FPR95, OSCR and unknown rejection, while known
acceptance is maintained or improved. Accepted-known classification accuracy
also improves: seed 43 `0.4241 -> 0.4664`, seed 44 `0.3698 -> 0.4791`.

Full-test representation diagnostics provide an independent check of the
mechanism:

| Seed | Distance statistic | Baseline AUROC / overlap | Warm-up AUROC / overlap |
| --- | --- | ---: | ---: |
| 43 | Nearest known training sample | 0.5669 / 0.8833 | 0.5990 / 0.8482 |
| 43 | Classifier prototype | 0.5721 / 0.8885 | 0.5956 / 0.8555 |
| 44 | Nearest known training sample | 0.5560 / 0.8986 | 0.6033 / 0.8395 |
| 44 | Classifier prototype | 0.5642 / 0.8955 | 0.5985 / 0.8418 |

These results are the strongest evidence so far that the candidate improves
known/unknown representation separation rather than only changing a decision
threshold. Nevertheless, AUROC remains near 0.60 and histogram overlap remains
around 0.84--0.86, so the core overlap problem is reduced, not solved.

Full-test artifacts:

- `runs/semantic_hard_knn_fulltest_baseline_s43/`
- `runs/semantic_hard_knn_fulltest_warmup_s43/`
- `runs/semantic_hard_knn_fulltest_baseline_s44/`
- `runs/semantic_hard_knn_fulltest_warmup_s44/`
- `analysis/knn_boundary_warmup_feature_diagnostics_full_s43_20260930.json`
- `analysis/knn_boundary_warmup_feature_diagnostics_full_s44_20260930.json`

### Independent seed 45

Seed 45 was trained with a fresh baseline and warm-up/ramp candidate. The
candidate training config differs from the baseline only in the KNN loss
weight (0 vs 0.1) and its required feature candidate gate; the schedule was
verified as `[0, 0, 0.5, 1, 1]` and loss values were zero in the first two
epochs and positive thereafter.

On the 1,000-image pilot, baseline -> warm-up/ramp gave AUROC `0.5584 ->
0.5721`, FPR95 `0.8818 -> 0.8514`, OSCR `0.2797 -> 0.3054`, known acceptance
`0.9105 -> 0.9155`, and unknown rejection `0.1029 -> 0.0956`. Since unknown
rejection was slightly lower, this was treated as supportive but not
conclusive and promoted to full-test verification.

On the full 10,000-image test set:

| Metric | Baseline | Warm-up/ramp | Change |
| --- | ---: | ---: | ---: |
| AUROC | 0.5928 | 0.6036 | +0.0107 |
| FPR95 | 0.8672 | 0.8480 | -0.0192 (better) |
| OSCR | 0.2932 | 0.3303 | +0.0371 |
| Known acceptance | 0.9255 | 0.9327 | +0.0072 |
| Unknown rejection | 0.1058 | 0.1020 | -0.0038 |
| Accepted-known accuracy | 0.4093 | 0.4585 | +0.0492 |
| All-known classification accuracy | 0.3788 | 0.4277 | +0.0488 |

Seed 45 therefore supports better ranking/open-set classification and known
classification without sacrificing known coverage, but does not show higher
unknown rejection at the calibrated operating point. This nuance matters: the
candidate is improving AUROC/FPR95/OSCR and classification, not uniformly
raising every metric. Across three seeds and the full test set, warm-up/ramp
improves AUROC/FPR95/OSCR and accepted-known accuracy on all three, preserves
or improves known acceptance on all three, while unknown rejection improves on
seeds 43/44 but is slightly lower on seed 45.

Seed-45 artifacts:

- Baseline checkpoint/evaluation: `runs/semantic_hard_disjoint_baseline_s45/`, `runs/semantic_hard_knn_fulltest_baseline_s45/`
- Warm-up checkpoint/evaluation: `runs/semantic_hard_candidate_knn_boundary_warmup_s45/`, `runs/semantic_hard_knn_fulltest_warmup_s45/`

The seed-45 full-test feature diagnostic is also consistent with a real but
partial representation change: nearest-training-support AUROC/overlap changes
`0.5867 / 0.8619 -> 0.6078 / 0.8374`; classifier-prototype AUROC/overlap
changes `0.6003 / 0.8424 -> 0.6076 / 0.8313`. The centroid statistic improves
AUROC `0.5530 -> 0.5669`, while overlap remains high (`0.9155 -> 0.8879`).
Residual overlap is still the central limitation.

- Seed-45 feature diagnostic: `analysis/knn_boundary_warmup_feature_diagnostics_full_s45_20260930.json`

## Current conclusion and next controlled work

The current best candidate is the candidate-gated class-wise KNN
support-boundary loss with two warm-up epochs and a two-epoch linear ramp.
Full-strength application from epoch 1 is rejected as the preferred schedule
because it repeatedly loses known coverage. The warm-up/ramp implementation is
now the only version that has passed a two-seed, full-test verification.

The next algorithmic work should be controlled rather than a broad parameter
search: (1) validate the same schedule on a third fixed seed or a second
semantic split; (2) if retained, replace repeated full support-bank
construction with cached or low-frequency updates; and (3) only then evaluate
unknown-sample clustering using the same fixed rejected pool. Further threshold
adjustment alone is not justified, because the remaining issue is residual
feature/support overlap.
