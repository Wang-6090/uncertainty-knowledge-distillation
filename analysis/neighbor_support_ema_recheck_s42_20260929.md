# Mixed-pool neighborhood support and EMA recheck (2026-09-29)

> **Interpretation note:** The saved configurations confirm that the neighbor-support pair toggled batch-local kNN weighting, and the EMA run additionally toggled EMA-based weight generation. Their reported detector was `normalized_entropy_mahalanobis`, not novel-mass; these runs test whether training reweighting indirectly changes the Mahalanobis detector, not whether novel-mass itself is an effective detector. The later method-fidelity audit adds direct novel-mass evaluation and supersedes the proposed Sinkhorn follow-up here, because the historical Sinkhorn contrast was not independently controlled.

## Experiment question

Does local feature-neighborhood agreement improve novel-candidate weighting on
a mixed known/novel discovery pool, and does generating those weights from an
EMA student make them more stable?

## Controlled protocol

- CIFAR-100 random 60/40 class split, seed 42; stratified known train/validation.
- Same pretrained ResNet-34 teacher, ResNet-18 student, training sample limits
  1200/300/1000, disjoint mixed discovery pool, known pool ratio 0.2.
- Same KMeans initialization for 40 novel prototypes, same joint losses, 3
  epochs, and same student evaluation checkpoint policy.
- Detection: explicit `normalized_entropy_mahalanobis`, MC=4, validation-known
  95% coverage calibration, clustering skipped.
- Intended differences: add batch-local feature-neighbor support to
  cross-view novel-mass weights; then source those weights from an EMA student.

## Results

| Weight recipe | AUROC | AUPR | FPR95 | OSCR | Known acc (all known) | Unknown reject (reported) | Unknown reject at exact 95% known coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Novel mass (student weights) | 0.5570 | 0.4324 | 0.8893 | 0.1914 | 0.2628 | 3.04% | 6.33% |
| Novel mass + batch-local kNN support | **0.5887** | **0.4661** | 0.9058 | 0.1874 | 0.2512 | **10.89%** | **8.86%** |
| Novel mass + kNN support, EMA weights | 0.5475 | 0.4370 | 0.9273 | 0.1769 | **0.2645** | 6.33% | 6.08% |

The operating-point rejection rates in the saved reports do not always have
exactly the same test known coverage due to finite validation calibration. The
last column retrospectively recalibrates each saved score at exactly 95% test
known coverage for a paired diagnostic; test labels are not used for training
or calibration.

## Interpretation

- Batch-local kNN support is a promising single-seed direction: versus novel
  mass alone it raises AUROC by 0.0316 and unknown rejection at matched known
  coverage by 2.53 percentage points. However, FPR95 worsens by 0.0165,
  OSCR falls slightly, and known accuracy falls by about 1.16 points. It is
  therefore a tradeoff, not a demonstrated overall improvement.
- EMA-generated weights did not help this experiment: AUROC, FPR95, OSCR and
  matched-coverage rejection are worse than the student-weighted neighbor
  version. EMA should not be enabled by default for this joint objective.
- The experiment is one seed and uses a reduced sample/epoch budget. No
  statistical or generalization claim is warranted.

## Method basis and next diagnostic

The local-neighbor agreement is motivated by SCAN's neighbor-consistency
principle and AutoNovel's use of sample relations. EMA weights follow the
Mean Teacher/FixMatch teacher-student stabilization idea. The result suggests
local support contains useful signal, but the EMA teacher in this run is not a
better source of that signal.

The originally proposed follow-up (weighted Sinkhorn) was implemented, but a
subsequent call-chain audit found the historical comparator did not isolate
Sinkhorn weighting. That causal claim is withdrawn; see the method-fidelity
audit for the corrected paired experiment. Do not infer from these
Mahalanobis-detected checkpoints that novel-mass itself is a useful detector.

## Validation

- Added opt-in `--joint-novel-neighbor-support` and `--joint-novel-ema-weights`.
- Smoke training completed; diagnostics showed the neighborhood support
  weights were active and finite.
- Full test suite: `90 passed`; Python compilation passed before the EMA
  experiment.
