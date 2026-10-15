# CIFAR-10 uniform Outlier Exposure pilot (2026-10-13)

## Purpose and comparison

The uncertainty-margin recheck did not reliably reduce overlap. This pilot
tests a different idea: provide the student with an independent auxiliary
outlier stream and train its known-class logits toward a uniform distribution.
This is the standard Outlier Exposure direction, applied here as a boundary
signal rather than another known-prototype margin.

For each seed, the baseline is the already matched mixed-pool nnPU student and
the treatment changes only:

```text
outlier_dataset = CIFAR-10 train
alpha_outlier_uniform = 0.1
```

The teacher, CIFAR-100 semantic-isolated 60/40 split, pretrained student,
five epochs, optimizer, mixed-pool prior, detector, kNN bank, MC=4, and
validation-only 95%-known-coverage calibration are held fixed. CIFAR-10 is
only a feasibility pilot: it is not a perfectly disjoint semantic source from
CIFAR-100, so a positive result would require a later disjoint-source test.

## Implementation audit

The loss is active in both treatments (`outlier_uniform` is about `4.53` for
seed 44 and `4.42` for seed 123 in the final training epoch). During this
pilot a preprocessing issue was found and corrected: CIFAR-10 outliers had
been normalized with CIFAR-100 statistics when `transform_profile=cifar` was
used. `build_outlier_dataset` now selects CIFAR-10 mean/std explicitly, and
regression tests cover both CIFAR-10 and CIFAR-100 transforms.

## Results

| seed | arm | AUROC | FPR95 | OSCR | known acc | known accept | unknown reject | KS | unknown below known-q95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 44 | nnPU baseline | 0.8120 | 0.5965 | 0.5214 | 57.90% | 94.12% | 29.85% | 0.4843 | 73.93% |
| 44 | + uniform OE | 0.8144 | 0.5880 | 0.5498 | 61.50% | 94.00% | 30.05% | 0.4923 | 74.48% |
| 123 | nnPU baseline | 0.8183 | 0.5850 | 0.5626 | 62.33% | 95.50% | 25.60% | 0.4962 | 72.18% |
| 123 | + uniform OE | 0.8191 | 0.5908 | 0.5448 | 60.75% | 94.77% | 28.80% | 0.4976 | 72.15% |

At the report's validation-calibrated threshold, seed 44 changes little in
unknown rejection and seed 123 increases rejection but loses known accuracy
and OSCR. The overlap diagnostic is essentially unchanged or slightly worse.
At exact 95% known coverage, the paired post-hoc diagnostic gives treatment
minus baseline unknown-rejection changes of `-0.45` percentage points for
seed 44 and `+0.07` points for seed 123; both bootstrap intervals include zero.

The paired bootstrap files are:

- `analysis/nnnpu_oe_uniform_bootstrap_s44_20261013.json`;
- `analysis/nnpu_oe_uniform_bootstrap_s123_20261013.json`.

The first filename contains a historical typo (`nnnpu`) but is the actual
saved artifact; it is retained to avoid silently losing the recorded result.

## Decision

Uniform OE is correctly wired and gives a small, mixed ranking signal. It is
not evidence that the core known/unknown overlap has been solved: the overlap
metric does not decrease consistently, and known utility/OSCR is not stable
across seeds. Keep the option as an ablation, but do not make it the default
or combine it with more losses yet.

The result also exposes an important methodological point. An auxiliary source
can regularize the classifier and improve known recognition without producing
an unknown boundary for the actual novel CIFAR-100 classes. The next useful
experiment should therefore use a genuinely disjoint, visually diverse
outlier source or a class-held-out CIFAR-100 boundary protocol, and should
route the boundary signal to the independent rejection representation. This
must be compared against the same baseline with candidate purity and input
normalization recorded.
