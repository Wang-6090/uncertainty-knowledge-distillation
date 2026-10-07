# Full-data seed-43 support-rejector replication (2026-10-07)

## Purpose

Seed 42 showed that class-stratified known support sampling improved the
support-only nnPU rejector, but one seed was not enough to decide whether the
effect was stable. This run keeps the full-data semantic-isolated protocol
fixed and changes only the random seed to 43.

The experiment is a paired comparison between the existing baseline student
and the uncertainty-margin treatment student. Both use the same teacher,
matched mixed discovery pool, rejector, support budget, and validation-only
threshold calibration.

## Fixed protocol

- CIFAR-100 semantic-isolated 60/40 split
- ResNet-34 teacher and ResNet-18 students
- Full train/validation/test data, five epochs
- Matched mixed discovery pool: 5400 samples, known prior 0.2
- Support-augmented `nu_corrected` nnPU rejector
- Known support sampled approximately equally across the 60 known classes
- Support budget: 5000 samples
- Threshold calibrated on known validation data for 95% known coverage
- Clustering skipped to isolate unknown detection
- Test labels used only for final descriptive metrics

## Results

| Student | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.7707 | 0.6055 | 0.3814 | 94.87% | 19.53% |
| treatment | **0.7836** | **0.5972** | **0.3977** | 94.80% | **21.35%** |

Treatment changes relative to the same-seed baseline:

- AUROC: `+0.0130`
- FPR95: `-0.0083`
- OSCR: `+0.0163`
- known acceptance: `-0.07` percentage points
- unknown rejection: `+1.83` percentage points

The treatment therefore improves ranking and the fixed-coverage operating
point on seed 43 without a meaningful loss of known acceptance.

## Comparison with seed 42

Seed 42 used the same protocol. The paired results were:

| Seed | Student | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | baseline | 0.7541 | 0.6722 | 0.3293 | 95.22% | 18.90% |
| 42 | treatment | 0.7642 | 0.6485 | 0.3796 | 94.90% | 19.08% |
| 43 | baseline | 0.7707 | 0.6055 | 0.3814 | 94.87% | 19.53% |
| 43 | treatment | 0.7836 | 0.5972 | 0.3977 | 94.80% | 21.35% |

Stratified support sampling improves both arms relative to random 5000
support on seed 42, and seed 43 shows the same direction for the treatment
comparison. This is evidence that the sampling policy reduces rejector
variance and improves ranking. It is not yet enough to make it the default;
one more independent full-data seed is needed.

## Interpretation and limits

The result supports three narrow conclusions:

1. Class-stratified support sampling is more promising than simply increasing
   the support limit. The earlier 20000-sample control was seed-dependent.
2. The uncertainty-margin treatment changes the representation/detector
   ranking in a favorable direction under the matched protocol.
3. The core overlap problem remains. Unknown rejection is still only about
   20%, so most unknown samples are accepted at the 95% known-coverage point.

This run does not validate the full end-to-end novel-class discovery claim:
clustering was deliberately skipped, and the joint novel-head losses were
not active in this support-only audit. The result should therefore be
reported as an unknown-detection and rejector audit, not as final new-class
discovery performance.

## Decision

Keep `--rejector-stratified-known` as an opt-in candidate and keep
`--rejector-max-samples 5000` for controlled comparisons. Do not promote
larger support budgets or conformal support scoring based on the current
evidence.

The next experiment should be one independent full-data seed with the same
protocol. If it confirms improved AUROC/FPR95 but unknown rejection remains
near 20%, further sampling and threshold sweeps should stop. The next code
change should then target either the nnPU rejector objective or an explicit
known-boundary representation loss, followed by a matched baseline/treatment
ablation.
