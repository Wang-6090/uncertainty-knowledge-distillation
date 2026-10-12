# Rejector known-support source audit and matched-size experiment (seed 44)

## Question

The frozen-feature nnPU rejector had looked strong in the preceding uniform-OE
comparison. Before changing the student representation again, this audit asks
whether that detector result is sensitive to where its known support features
come from, or to the support/data-transform protocol.

## Protocol integrity audit

The experiment used CIFAR-100 semantic-isolated 60/40, seed 44, a matched
5,400-sample mixed discovery pool (known prior 0.2), and open validation
reserved from the training partition. Using the dataset's raw CIFAR train/test
indices, the sets were reconstructed and checked:

| Set | Samples | CIFAR train | CIFAR test |
|---|---:|---:|---:|
| Supervised train | 21,600 | 21,600 | 0 |
| Known discovery part | 1,080 | 1,080 | 0 |
| Novel discovery part | 4,320 | 4,320 | 0 |
| Checkpoint-selection known validation | 2,045 | 2,045 | 0 |
| Threshold-calibration known validation | 655 | 655 | 0 |
| Open validation | 2,300 | 2,300 | 0 |
| Official test | 10,000 | 0 | 10,000 |

Every pair of sets was disjoint. The known discovery proportion was exactly
0.2. The known checkpoint-selection and threshold-calibration subsets were
also disjoint according to the saved index audit. Test data was used only for
final retrospective metrics. Explicit score selection was used, not
open-label automatic score selection.

## Controlled comparison

No model was retrained. The two frozen seed-44 student checkpoints from the
previous experiment were evaluated with identical test samples, 5,400-sample
mixed unlabeled pool, rejector settings, detector, and validation-only 95%
known-coverage calibration. The only treatment was the *known support used to
fit the rejector*. To remove the earlier sample-count confound, every arm was
limited to 300 known support rows and 300 unlabeled rows.

The three support protocols were:

1. `train_known_support`: 300 known examples from the supervised training set,
   extracted with the historical random-train augmentation.
2. `train_known_eval_transform`: the same training-set source, but extracted
   with deterministic evaluation transforms, matching discovery and test.
3. `openval_known_support`: the 300 disjoint known examples reserved in
   open-validation, extracted with deterministic evaluation transforms.

The last comparison changes both source partition and independence from model
training, while the second isolates the transform difference. Thus the
train-vs-open-val contrast is not a pure source-only contrast; it is interpreted
alongside the train-transform control.

## Results

| Student checkpoint | Known support protocol | AUROC | FPR95 | OSCR | Test known accept | Test unknown reject | KS | Unknown below known q95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Margin only | Train, train augmentation | 0.7929 | 0.7422 | 0.4966 | 94.53% | 33.23% | 0.4513 | 68.58% |
| Margin only | Train, eval transform | 0.7959 | 0.6808 | 0.5040 | 94.93% | 30.48% | 0.4464 | 70.03% |
| Margin only | Open-val known | 0.7763 | 0.7462 | 0.4905 | 96.08% | 27.18% | 0.4133 | 70.05% |
| Margin + uniform OE | Train, train augmentation | 0.7840 | 0.7315 | 0.4968 | 94.65% | 29.10% | 0.4345 | 71.85% |
| Margin + uniform OE | Train, eval transform | 0.7778 | 0.7192 | 0.4959 | 93.83% | 30.43% | 0.4178 | 72.42% |
| Margin + uniform OE | Open-val known | 0.7770 | 0.7180 | 0.4920 | 94.72% | 29.83% | 0.4086 | 71.13% |

Paired test bootstrap (2,000 stratified replicates; retrospective only) for
train-known vs open-val-known support at matched 300/300 sample counts:

- Margin-only checkpoint: AUROC difference -0.0165, 95% CI
  [-0.0261, -0.0069]; unknown rejection at exact 95% test-known coverage
  difference -1.18 pp, 95% CI [-4.23, +1.68] pp.
- Margin + uniform checkpoint: AUROC difference -0.0069, 95% CI
  [-0.0177, +0.0044]; unknown rejection difference +0.91 pp, 95% CI
  [-1.73, +3.75] pp.

The paired bootstrap also compared train augmentation with deterministic eval
transforms while keeping the *training support source* and sample count fixed:

- Margin-only checkpoint: AUROC difference +0.0031, 95% CI
  [-0.0025, +0.0090]; unknown rejection at exact 95% test-known coverage
  difference -0.96 pp, 95% CI [-3.55, +1.48] pp.
- Margin + uniform checkpoint: AUROC difference -0.0062, 95% CI
  [-0.0120, -0.0001]; unknown rejection difference -0.71 pp, 95% CI
  [-2.78, +1.28] pp.

Transform choice can affect the fitted detector (especially for the
uniform-OE checkpoint), but deterministic extraction is not a consistent
improvement and does not explain the detector ranking. Treat this as protocol
sensitivity, not evidence that training augmentation is the main cause of the
known/unknown overlap.

All-known classification accuracy is unchanged within each checkpoint, as
expected: these comparisons only refit a post-hoc rejector on frozen features.
Test operating-point differences should be read with the test known-coverage
bootstrap, not as deployable test-calibrated thresholds.

## Interpretation and decision

- The dataset protocol is clean: no train/discovery/validation/test sample
  leakage was found, and the prior and disjoint calibration audit match the
  configured protocol.
- Switching the training support extraction to evaluation transforms did not
  consistently improve ranking or overlap. Augmentation mismatch alone does
  not explain the detector's strong earlier result.
- Replacing training-known support with only 300 open-validation known examples
  did not help either checkpoint. For the margin-only checkpoint AUROC fell
  significantly in the paired bootstrap. Uniform-OE's AUROC interval crossed
  zero, but its point estimate also fell and operating-point evidence was
  inconclusive.
- The 300-support experiment is a diagnostic, not a fair estimate of the
  original 5,000-support rejector's absolute performance. It specifically
  isolates support protocol at a common sample budget. Do not promote
  strict-mixed support or change defaults based on these results.
- This does not invalidate the earlier full-pool AUROC near 0.88; it shows that
  the result is sample-budget-sensitive and that small independent supports
  add variance. It also does not show the core known/unknown overlap is solved.

## Next direction

Stop spending cycles changing only the rejector support source. Retain the
known-support fit from training for now. The next useful experiment should
target the representation or the PU training objective, with the 0.1 encoder
learning-rate scale held fixed, and compare against a same-seed baseline using
the same student checkpoint protocol, test samples, fixed validation known
coverage, AUROC, FPR95, OSCR, score overlap, and known classification. Only
advance a candidate if ranking/overlap improves without materially harming
known coverage or classification; confirm promising candidates on a second
predeclared seed.

## Artifacts

- Runner: `scripts/run_rejector_support_source_s44.ps1`
- Matched-size runs: `runs/rejector_support_source_s44_matched300_*`
- Bootstrap JSON: `analysis/rejector_support_source_s44_matched300_*_bootstrap_20261012.json`
- Transform-control bootstrap JSON: `analysis/rejector_support_transform_s44_matched300_*_bootstrap_20261012.json`
- Prior full-pool uniform-OE comparison: `analysis/rejection_uniform_lr01_openval_s44_20261012.md`
