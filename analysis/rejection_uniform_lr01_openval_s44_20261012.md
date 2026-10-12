# Rejection-branch uniform OE on the LR-0.1 baseline (seed 44)

## Question and pre-registered comparison

Earlier limited-data experiments found a possible complementarity between an
external-outlier feature margin and uniform rejection logits. This experiment
asks whether the uniform-logit term adds value on the better-supported
`encoder_lr_scale=0.1` CIFAR-100 baseline. The only treatment factor is
`alpha_outlier_rejection_uniform`: 0.0 versus 0.05. Both arms use the same
external CIFAR-10 data, rejection-feature margin weight 0.05, and margin 0.2.
Because CIFAR-10 may be semantically related to CIFAR-100 classes, this is a
controlled auxiliary-outlier pilot, not a claim of novel-class generalization.

## Fixed conditions and protocol check

- CIFAR-100 semantic-isolated 60/40 split, seed 44; same pretrained ResNet-34
  teacher and pretrained ResNet-18 student initialization.
- Full training partition, five epochs, batch size 64, CUDA, same
  `encoder_lr_scale=0.1`.
- Same audited uncertainty KD, mixed nnPU loss, matched discovery pool (5,400;
  known prior 0.2), rejection branch (dimension 128), known rejection CE,
  external CIFAR-10 outlier stream, and feature-margin OE.
- `open_val_ratio=0.1` reserves known and novel examples from the training
  partition before discovery-pool construction. The official test set remains
  untouched for final evaluation.
- Same detector: linear `nu_corrected` nnPU rejection model on
  `support_augmented` rejection features, MC=4, validation-only 95%-known
  coverage threshold. Clustering disabled.
- Only the uniform-logit OE coefficient differs: 0.0 vs 0.05.

The first detection attempt failed before fitting the rejector because the
discover command omitted the required explicit mixed-pool mode. It produced no
usable metric. The script was corrected and both detector runs then completed;
the completed student checkpoints were reused, so training was not repeated.
The final configs and command are in the two run directories and
`scripts/run_rejection_uniform_lr01_openval_s44.ps1`.

## Results

| Metric | Margin only | Margin + uniform OE | Change |
|---|---:|---:|---:|
| AUROC | 0.8805 | 0.8800 | -0.0006 |
| AUPR | 0.8108 | 0.8049 | -0.0060 |
| FPR95 (lower is better) | 0.4162 | 0.4113 | -0.0048 |
| OSCR | 0.5592 | 0.5616 | +0.0024 |
| Known classification accuracy, all known | 60.57% | 61.13% | +0.57 pp |
| Test known acceptance | 95.80% | 93.65% | -2.15 pp |
| Unknown rejection | 39.05% | 46.75% | +7.70 pp |
| Score KS distance | 0.6008 | 0.6041 | +0.0033 |
| Unknown below known q95 | 57.15% | 59.35% | +2.20 pp (worse) |

Both calibration subsets accepted 94.96% of their known samples, but on the
official test set the treatment rejects more known samples. The large increase
in unknown rejection is therefore partly an operating-point tradeoff; ranking
metrics are essentially unchanged, and the unknown-below-known-q95 overlap is
slightly worse. Known classification rises modestly, but that does not offset
the lower test known acceptance. The score means and quantiles shift, so the
threshold itself shifts; this must not be interpreted as clean feature
separation.

### Retrospective paired test bootstrap at matched known coverage

To separate ranking from this threshold shift, a paired, stratified bootstrap
(2,000 replicates; the same sampled known and unknown test indices for both
arms) recomputed the threshold on each replicate's known test scores to target
95% known coverage. This uses test labels only as a post-hoc diagnostic and
was not used for model selection. Observed unknown rejection at matched test
coverage is 42.80% for margin-only versus 40.40% for margin + uniform OE. The
treatment-minus-baseline bootstrap mean difference is -1.84 percentage points,
with 95% percentile CI [-5.05, +1.38] pp; only 13.25% of replicates favor the
treatment. The AUROC difference is -0.00060 (95% CI [-0.00579, +0.00476]).
Thus the apparent +7.70 pp rejection increase at separately calibrated
thresholds does not reflect stronger ranking; it is consistent with a
threshold/score-scale shift and reduced known acceptance. This further argues
against promoting uniform OE on this baseline.

Implementation and output: `scripts/paired_open_score_bootstrap.py` and
`analysis/rejection_uniform_lr01_openval_s44_paired_bootstrap_20261012.json`.

## Decision

This is inconclusive, not a validated improvement. The uniform term changes
the validation-calibrated operating point, but does not meaningfully improve
AUROC/AUPR or score overlap. Do not promote it to the default and do not claim
the core known/unknown overlap is solved. Since the prior multi-seed pilot had
mixed results and this full-data seed largely preserves ranking while trading
away known coverage, the combination is lower priority than an independent
representation intervention. If retained, replicate on one pre-specified
second seed and compare both test operation and ranking at the same known
coverage; otherwise stop tuning this loss coefficient.

## Artifacts

- `runs/rej_uniform_lr01_openval_s44_margin_only_student/`
- `runs/rej_uniform_lr01_openval_s44_margin_only_discover/`
- `runs/rej_uniform_lr01_openval_s44_margin_plus_uniform_student/`
- `runs/rej_uniform_lr01_openval_s44_margin_plus_uniform_discover/`
- `scripts/run_rejection_uniform_lr01_openval_s44.ps1`
