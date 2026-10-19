# Inter-class virtual outlier pairing audit — 2026-10-17

## Purpose

The current bottleneck is overlap between known and unknown feature/score
distributions. The first virtual-outlier experiment used a different-class
partner sampled uniformly from the current batch. This audit tests whether
choosing the nearest different-class centroid makes the synthetic outliers
more boundary-focused.

## Controlled factors

- CIFAR-100 semantic-isolated 60/40 split;
- seed 44, four student epochs, the same teacher and student initialization;
- matched mixed discovery pool of 5400 samples, known prior 0.2;
- same detector, checkpoint rule, and validation-only 95% known-coverage
  calibration;
- `alpha_vos=0.01`, `vos_mode=interclass_mix` for both treatment arms;
- only the inter-class partner strategy changes: `random` versus
  `nearest_class`.

The control and treatment in this audit were trained in the same script and
must be compared with each other. Earlier random-pair results are reported
separately because GPU training is not bitwise deterministic in this
environment.

## Results

| arm | AUROC | FPR95 | OSCR | known acceptance | unknown rejection | KS | unknown below known q95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| random control | 0.90062 | 0.36033 | 0.66090 | 93.02% | 58.73% | 0.65250 | 50.73% |
| nearest different class | **0.90495** | **0.33167** | 0.65971 | **94.90%** | 48.78% | **0.66383** | 51.75% |

The nearest-class treatment improves ranking metrics: AUROC +0.00434,
FPR95 -0.02867, and KS +0.01133. However, at the validation-calibrated
test threshold its unknown rejection is 9.95 percentage points lower than
the matched control. OSCR is slightly lower and the final known-class
accuracy is 0.7067 versus 0.7155.

The earlier random-pair run, compared with its own control, increased the
reported unknown rejection from 44.43% to 50.23% but slightly reduced AUROC
and worsened FPR95. The two control runs are not identical in their absolute
metrics despite the same nominal seed, so they are not pooled as repeated
trials. This is an audit reason to report matched-arm results and not claim
that one virtual-outlier strategy has solved the core problem.

## Interpretation

Nearest-class pairing is a useful optional research direction because it
changes the score ordering in the desired direction and reduces the
known-above-unknown overlap measured by FPR95. It is not ready to replace
the random/default strategy: the fixed calibration work point is unstable,
and the unknown rejection trade-off is unfavorable in this matched run.

The likely issue is that every known sample is still forced to create a
synthetic unknown near a batch centroid boundary. This can reshape the
global energy scale and move the calibrated threshold without selectively
targeting the high-confidence unknown errors. The next focused modification
should therefore gate virtual-outlier loss to hard known boundary samples or
use a detached/frozen support bank, while preserving an explicit known
classification constraint. Do not select the gate or threshold using test
labels.

## Reproducibility

- random/inter-class control: `runs/interclass_vos_control_s44_e4_control`
- random/inter-class treatment: `runs/interclass_vos_control_s44_e4_interclass_mix`
- nearest-class control: `runs/interclass_vos_nearest_control_s44_e4_control`
- nearest-class treatment: `runs/interclass_vos_nearest_control_s44_e4_interclass_mix`
- implementation: `novel_discovery/pipeline.py`, `train.py`
- driver: `scripts/run_interclass_vos_control.py`
