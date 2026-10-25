# Protocol: uniform rejection-feature KD vs no feature KD (seed 2106)

## Question

Does adding the existing cosine rejection-feature distillation objective, with
uniform sample weights, improve semantic-hard novel rejection at a fixed
known-validation coverage? The seed-2105 result compared two nonzero KD arms
and therefore did not answer whether feature KD itself helps over no feature
KD.

## Pre-registered comparison

- Fresh CIFAR-100 semantic-hard 60/40 split, seed 2106.
- Control: `alpha_rejection_feat_kd=0`; treatment: `0.1` with
  `rejection_feature_kd_weighting=uniform`.
- Same split-matched frozen teacher (128-d rejection feature), student recipe,
  standard logits KD, optimizer and four-epoch budget, mixed-pool nnPU,
  known-validation checkpoint selection, feature-rejector evaluation, and
  threshold calibrated to 95% known-validation coverage.
- No VOS/boundary/extra selector or score fusion changes between arms. No
  mixed-pool or test labels for training, selection, or calibration. Test labels
  are used only for final metrics.

## Decision rule

Report AUROC, AUPR, FPR95, OSCR, calibrated unknown rejection, known acceptance,
accepted-known accuracy, correctly accepted-known fraction, automatic K/NMI/ARI,
and per-epoch feature-KD loss/gradient. Treat this single split as exploratory;
do not promote on ranking metrics alone. The primary practical gate is at least
`+2 pp` unknown rejection with no material decline (`>1 pp`) in known acceptance
or accepted-known accuracy, and no OSCR decline beyond `0.005`. Failure stops
this exact uniform-feature-KD recipe; a positive result requires a new fresh
split replication before promotion.

## Result

The paired run completed after resuming the already complete teacher/control
artifacts. The first attempt stopped before treatment artifacts were written due
to a runner/runtime exception; no partial treatment result was interpreted.
The runner was then changed to resume only missing arms without overwriting
complete artifacts.

| Metric | No feature KD | Uniform feature KD | Treatment - control |
|---|---:|---:|---:|
| AUROC | 0.797788 | 0.789474 | -0.008313 |
| AUPR | 0.697152 | 0.681160 | -0.015992 |
| FPR95 | 59.867% | 60.933% | +1.067 pp |
| OSCR | 0.573681 | 0.568502 | -0.005179 |
| Known acceptance | 95.917% | 95.683% | -0.233 pp |
| Unknown rejection at calibrated threshold | 22.200% | 21.925% | -0.275 pp |
| Accepted-known accuracy | 68.341% | 68.333% | -0.008 pp |
| Correctly accepted fraction of all known | 65.550% | 65.383% | -0.167 pp |
| Automatic K | 32 | 33 | +1 |
| NMI / ARI | 0.495570 / 0.163382 | 0.475731 / 0.141455 | -0.019838 / -0.021927 |
| Candidate purity before purification | 78.376% | 77.201% | -1.175 pp |

The treatment mechanism was active: feature-KD loss was
`0.428973, 0.289000, 0.247891, 0.230172` across four epochs, and the recorded
rejection feature-KD gradient norms were nonzero. The control loss was zero.
Configuration and label-flow audits completed; the only substantive training
difference was `alpha_rejection_feat_kd: 0 -> 0.1`, with uniform weighting in
the treatment.

## Decision

This fresh split does not support uniform rejection-feature KD. It failed the
predeclared practical gate and worsened ranking, fixed-coverage rejection,
OSCR, candidate purity, and clustering. The exact uniform feature-KD recipe is
stopped and is not promoted or added to the default configuration. Together
with seed 2105, this means the prior uncertainty-weighting experiment was not
the missing explanation: the underlying uniform feature-KD contribution is
also not supported on this fresh split. No coefficient or weighting sweep is
justified from these results.
