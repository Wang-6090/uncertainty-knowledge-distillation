# Hard vs soft nnPU feature-margin: seed-44 paired experiment

## Question and controlled factor

The previous rejection-PU feature-margin experiment used a hard non-negative
PU-risk clamp. Its feature-margin loss became small over training, raising the
possibility that the clamp was suppressing useful gradients. This experiment
compares that implementation with a softplus non-negative PU risk, while
keeping the margin coefficient active and fixed.

The sole intended algorithmic change is
`discovery_rejection_pu_risk=hard_nnpu` versus `soft_nnpu`. Both use the
same CIFAR-100 semantic-isolated 60/40 split, seed 44, ResNet-34 teacher,
ResNet-18 student, pretrained initialization, complete training data, five
epochs, encoder LR scale 0.1, matched mixed discovery pool (5,400 examples,
known prior 0.2), and detector/calibration protocol. The PU-margin coefficient
is 0.1 and softness is 0.05 in both runs. Best checkpoints are selected by the
same validation-known-accuracy rule. Discovery uses the same test examples and
95% validation-known-coverage calibration; clustering is skipped to isolate
open-set detection.

## Results

| Metric | Hard nnPU | Soft nnPU | Soft - hard / interpretation |
|---|---:|---:|---:|
| AUROC | 0.8706 | 0.8742 | +0.0036; paired 95% CI [-0.0020, +0.0093] |
| AUPR | 0.7927 | 0.8013 | +0.0087 (descriptive) |
| FPR95 | 0.4213 | 0.4203 | -0.0010 (descriptive; lower is better) |
| OSCR | 0.5597 | 0.5635 | +0.0037 (descriptive) |
| Known classification accuracy (all known) | 0.6135 | 0.6142 | +0.0007 |
| Report threshold's test known acceptance | 0.9410 | 0.9595 | +0.0185; realized coverages differ |
| Unknown rejection at each report threshold | 0.4165 | 0.3540 | -0.0625; not a fair fixed-coverage comparison |
| Unknown rejection at exact 95% test-known coverage | 0.3750 | 0.4008 | +0.0258; paired 95% CI [-0.0063, +0.0580] |
| Score KS distance | 0.5882 | 0.5935 | +0.0053 |
| Unknown score fraction below known q95 | 0.6245 | 0.5958 | -0.0288; lower is better |

The paired bootstrap resamples the *same ordered official-test examples* and
uses labels only retrospectively for confidence intervals and the exact-95%
operating point. It is not used for training, threshold calibration, or model
selection. Neither AUROC nor fixed-coverage unknown-rejection confidence
interval excludes zero; the point estimates favor soft nnPU, but this single
seed is not evidence of a reliable gain. The report-threshold rejection rate
falls for soft nnPU because its realized test known coverage is higher, so that
comparison alone must not be read as a regression.

## Did the change activate as intended?

The recorded `discovery_rejection_pu_feature_margin` loss stays materially
larger under soft nnPU late in training: at epoch 5 it is 0.0536, versus 0.0181
for hard nnPU (epoch 1: 0.1234 vs 0.1063). Thus the soft risk does mitigate the
rapid loss collapse that motivated the experiment. However, that mechanism
change translates only into small, statistically inconclusive test-ranking
and score-overlap changes. Validation-known accuracy and training known CE are
nearly identical, reducing concern that the point-estimate difference merely
comes from a large known-class accuracy shift.

## Predeclared second-seed replication

The same hard-vs-soft comparison was repeated on seed 123. The seed, data
protocol, model recipe, matched pool, loss weights, calibration rule, detector,
and test examples within this seed were held common across the pair.

| Metric | Hard nnPU | Soft nnPU | Soft - hard |
|---|---:|---:|---:|
| AUROC | 0.8827 | 0.8855 | +0.0028; paired 95% CI [-0.0016, +0.0071] |
| AUPR | 0.8212 | 0.8170 | -0.0042 |
| FPR95 | 0.4273 | 0.4078 | -0.0195 |
| OSCR | 0.5751 | 0.5771 | +0.0020 |
| Known classification accuracy (all known) | 0.6213 | 0.6245 | +0.0032 |
| Unknown rejection at exact 95% test-known coverage | 0.4538 | 0.4350 | -0.0188; paired 95% CI [-0.0478, +0.0123] |
| KS distance | 0.6054 | 0.6139 | +0.0085 |
| Unknown score fraction below known q95 | 0.5453 | 0.5648 | +0.0195 (worse) |

AUROC's point estimate is positive in both seeds but neither seed's paired CI
excludes zero. FPR95 improves only in seed 123; AUPR improves only in seed 44.
At exact 95% known coverage, unknown rejection improves in seed 44 but declines
in seed 123, and the score-tail overlap also moves in opposite directions.
Thus the small AUROC trend is not accompanied by consistent fixed-operating-
point or overlap improvement. The soft formulation does consistently keep the
training margin loss larger: at epoch 5 it is 0.0536 vs 0.0190 on seed 123,
and 0.0536 vs 0.0181 on seed 44. Thus its intended optimization effect is
reproducible, but that alone does not establish better open-set separation.

Keep hard nnPU as the current default. Do not claim a reliable soft-nnPU gain,
and stop sweeping this loss shape. Return to representation geometry / feature
learning signals while holding the audited encoder LR and evaluation protocol
fixed. Continue to judge progress jointly by AUROC, FPR95, OSCR, fixed-coverage
rejection, score overlap, and known classification—not rejection rate alone.

## Artifacts

- Runner: `scripts/run_rejection_pu_margin_risk_s44.ps1`
- Seed-123 replication runner: `scripts/run_rejection_pu_margin_risk_s123.ps1`
- Hard run: `runs/rejection_pu_margin_s44_hard_nnpu_{student,discover}`
- Soft run: `runs/rejection_pu_margin_s44_soft_nnpu_{student,discover}`
- Retrospective paired bootstrap: `analysis/rejection_pu_margin_s44_soft_vs_hard_bootstrap_20261012.json`
- Seed-123 hard/soft runs: `runs/rejection_pu_margin_s123_{hard,soft}_nnpu_{student,discover}`
- Seed-123 retrospective paired bootstrap: `analysis/rejection_pu_margin_s123_soft_vs_hard_bootstrap_20261012.json`

This two-seed exploratory comparison does not resolve the central known/unknown
score overlap; a larger multi-seed study would be needed to estimate small
effects precisely.
