# Pipeline audit and current full-data check (2026-10-04)

## Scope

This audit checks whether the current training, detection, and discovery
experiments are actually testing the intended variables. It separates code
execution evidence from algorithm-performance evidence. Test labels are used
only for retrospective evaluation; they are not used to fit the model or to
calibrate the default threshold.

## Code audit

### Verified

- The current test suite passes: `147 passed`.
- Python compilation and whitespace checks pass.
- `pipeline.py` passes detached teacher logits and teacher uncertainty through
  the public `losses.py` interface to `uncertainty_kd.py`.
- Standard KL and uncertainty-weighted KL are selected by `kd_mode`; teacher
  uncertainty is detached before it becomes a per-sample KD weight.
- The uncertainty calibration target is detached from the logits, so the
  uncertainty-head auxiliary loss cannot change the classification target
  through a hidden gradient path.
- Mixed-pool nnPU is guarded by `discovery_pool_mode=mixed` and receives the
  measured known prior from the actual split. It is not allowed to treat a
  pure unknown pool as an unlabeled mixture.
- Detection scores use the documented direction: larger score means more
  unknown. Known-only coverage calibration uses validation known samples;
  test labels are used only to calculate AUROC, FPR95, OSCR, and operating
  point metrics.
- The current end-to-end GPU path uses CUDA on the RTX 4060 Laptop GPU.

### Protocol issues to keep explicit

- `discovery_pool_mode=unknown` is an oracle-filtered pure-novel pool when
  class labels were used to construct it. It is an upper bound, not a mixed
  open-world deployment protocol.
- `discovery_pool_mode=mixed` is the main realistic protocol. Its actual
  known proportion must be read from `config.json`; it is not necessarily the
  requested proportion after concatenation and limiting.
- `cluster_k=oracle` uses the true novel-class count and is an evaluation
  upper bound for clustering. It must not be compared with `cluster_k=auto`
  as if they were the same task.
- Current full-data checks deliberately use `skip-clustering` to isolate
  unknown detection. They therefore provide no new claim about automatic
  novel-class clustering.
- Historical run directories without a source commit hash, complete command,
  environment snapshot, and training history are reproducibility records,
  not strong causal evidence.

## Current paired full-data experiment

### Fixed conditions

- CIFAR-100 random 60/40 split, seed 42.
- Full known training partition and full 10,000-image open test set.
- Same pretrained ResNet-34 teacher checkpoint and pretrained ResNet-18
  student architecture.
- Five student epochs, batch size 64, mixed unlabeled pool, known pool ratio
  0.2, uncertainty KD, supervised contrastive projection loss 0.1.
- The only training variable is `alpha_discovery_uncertainty_pu`:
  baseline `0.0`; treatment `0.1`.
- Detection uses the same MC=8 extraction, explicit
  `normalized_entropy_mahalanobis` score, 95% known-coverage threshold, and
  clustering disabled.

### Results: current code

| Method | AUROC | AUPR | FPR95 | OSCR | Known accept | Unknown reject | Accepted-known acc | All-known acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mixed baseline | 0.5440 | 0.4140 | 0.8887 | 0.2075 | 0.9423 | 0.0513 | 0.3164 | 0.2982 |
| Mixed nnPU | 0.5849 | 0.4511 | 0.8792 | 0.2338 | 0.9402 | 0.0710 | 0.3544 | 0.3332 |
| Change | +0.0410 | +0.0371 | -0.0095 | +0.0263 | -0.0022 | +0.0198 | +0.0380 | +0.0350 |

The treatment improves both ranking metrics and the matched operating-point
unknown rejection in this reproduction. It also improves known
classification, so the gain is not explained by simply rejecting more known
samples. However, `7.10%` unknown rejection is still far from a solved open
set detector. This is evidence for nnPU as a useful component, not evidence
that the core feature-overlap problem is solved.

### Detection-score follow-up on the same checkpoints

Replacing only the score with `normalized_entropy_min_class_knn` gives:

| Method | AUROC | FPR95 | OSCR | Known accept | Unknown reject | Accepted-known acc |
|---|---:|---:|---:|---:|---:|---:|
| Mixed baseline + min-class kNN | 0.5602 | 0.8882 | 0.2236 | 0.9455 | 0.0615 | 0.3229 |
| Mixed nnPU + min-class kNN | 0.5914 | 0.8905 | 0.2466 | 0.9532 | 0.0748 | 0.3597 |

The min-class kNN score improves AUROC and unknown rejection relative to the
Mahalanobis score for these current checkpoints, but its FPR95 is not better
for the treatment. It should remain an explicit detector candidate, not be
described as a complete representation solution.

### End-to-end candidate clustering check

The same checkpoints were then evaluated with oracle `K=40` and
`feature_pca` KMeans. This isolates clustering quality from automatic-K
estimation, but the candidate pool is still produced by the real detector.

| Method | Candidate count | Candidate purity | Candidate unknown NMI | Candidate unknown ARI | All-unknown NMI |
|---|---:|---:|---:|---:|---:|
| Mixed baseline + min-class kNN | 573 | 0.4293 | 0.5096 | 0.0757 | 0.3698 |
| Mixed nnPU + min-class kNN | 580 | 0.5155 | 0.4510 | 0.0516 | 0.3753 |

nnPU makes the candidate pool cleaner and recovers more true unknown samples,
but the clustering structure of the candidate subset becomes worse in this
single paired run. Therefore the current method has a split outcome: better
unknown selection, unresolved novel-class representation. Automatic-K and
non-oracle clustering remain unvalidated for this new checkpoint.

## Feature-overlap diagnosis

The fixed-checkpoint feature diagnostic compares known and unknown test
distributions without fitting anything on test labels.

| Distance statistic | Baseline AUROC | nnPU AUROC | Baseline overlap | nnPU overlap |
|---|---:|---:|---:|---:|
| Classifier prototype distance | 0.5815 | 0.5775 | 0.8743 | 0.8791 |
| Empirical known-class centroid distance | 0.5149 | 0.5560 | 0.9298 | 0.8925 |
| Nearest known training sample distance | 0.5521 | 0.6010 | 0.8984 | 0.8486 |

This separates two effects that had previously been conflated:

1. nnPU improves local known-support separation and the uncertainty score.
2. It does not improve the global classifier-prototype boundary. The main
   remaining error is therefore representation geometry and known-class
   support quality, not merely a bad scalar threshold.

## Evidence classification of existing experiments

### Suitable for the current main line

- Mixed-pool nnPU uncertainty training: supported by three historical seeds
  and reproduced once with the current code. Keep it as the primary training
  component, but report the residual low unknown rejection.
- Explicit min-class kNN detection: supported by a consistent three-seed
  historical comparison and a current-checkpoint recheck. Keep it as an
  explicit detector ablation and candidate best detector.

### Useful only as controlled ablations

- Uncertainty-weighted KL distillation: the code path is real and tested, but
  the existing three-seed comparison does not show a clear gain over standard
  KD.
- Feature KD, projection SupCon, prototype alignment, candidate selectors,
  memory banks, and prototype refresh: executable components with local
  evidence, but no stable full-data multi-seed improvement over the current
  main line.
- Semantic-hard and semantic-isolated splits: useful robustness protocols,
  but they are different tasks and must not be pooled with random 60/40.

### Upper bounds or results that must not support deployment claims

- Pure-unknown feature rejectors and oracle-filtered candidate pools.
- `cluster_k=oracle`, unknown-only clustering, and any score selected using
  labeled open-validation data.

### Directions currently not supported as defaults

- OpenMax and prototype-repulsion variants after matched full-data checks.
- Direct raw-feature SupCon: the existing pilot worsened both open-set
  metrics and feature diagnostics. Generic compactness alone is not enough.
- More threshold sweeps without a representation change. They can move the
  operating point but cannot remove the observed distribution overlap.

## Recommended next technical direction

The next change should target the global known representation while retaining
mixed nnPU as the uncertainty component:

1. Keep the current mixed nnPU objective and explicit min-class kNN detector
   fixed as the reference pipeline.
2. Test one training-time known-support objective at a time, with a separate
   report for classifier-prototype distance, empirical-centroid distance,
   nearest-support distance, and accepted-known accuracy.
3. Prefer a class-conditional margin or a correctly designed known/novel
   head over another post-hoc threshold. The objective must avoid treating
   every mixed-pool sample as unknown and must use detached or EMA pseudo
   targets when novel assignments are introduced.
4. Before promoting a new objective, repeat it on at least three seeds under
   the same split, score, threshold, and test protocol.
5. Only after detection representation quality improves, evaluate automatic
   novel clustering on the same checkpoint with both oracle-K and auto-K
   clearly separated.

The immediate next experiment is therefore a full-data paired test of one
known-support representation objective against the current mixed nnPU
reference, followed by the same feature-overlap diagnostic. It should not
combine multiple new losses in one run.

## Artifacts

- Current baseline: `runs/audit_current_baseline_s42`
- Current mixed nnPU treatment: `runs/audit_current_treatment_s42`
- Mahalanobis detection reports: corresponding `*_detect` directories
- Min-class kNN detection reports: corresponding `*_knn_detect` directories
- Feature-overlap diagnostic: `analysis/audit_current_feature_overlap_s42.json`
- Full oracle-K clustering reports: corresponding `*_cluster` directories
