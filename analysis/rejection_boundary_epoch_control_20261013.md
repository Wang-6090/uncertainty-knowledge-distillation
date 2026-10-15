# PU rejection-boundary loss: equal-epoch replication

## Purpose

The first boundary-loss pilot compared each arm's selected checkpoint. The
baseline selected epoch 4 while the treatment selected epoch 5, so that result
could contain a checkpoint-age confound. This audit fixes the training budget
to four epochs for both arms and keeps the following unchanged: seed-specific
data order, the audited CIFAR-100 semantic-isolated 60/40 split, teacher,
mixed discovery pool, known prior, rejection feature head, validation protocol,
MC samples, and final score/calibration protocol. The only treatment change is
`alpha_discovery_rejection_boundary=0.1` with PU mode.

Test labels are used only for retrospective metrics and paired bootstrap; they
are not used for training, threshold selection, or model selection.

## Results

| seed | arm | AUROC | FPR95 | OSCR | known accuracy | KS | unknown below known q95 | unknown rejection at exact 95% known coverage |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | baseline | 0.8833 | 0.4362 | 0.6292 | 0.6938 | 0.6038 | 0.5245 | 0.4753 |
| 123 | PU boundary 0.1 | **0.8910** | **0.4127** | **0.6392** | **0.6992** | **0.6263** | **0.5135** | **0.4865** |
| 44 | baseline | 0.8822 | 0.4070 | 0.6492 | 0.7150 | 0.6046 | 0.5673 | 0.4308 |
| 44 | PU boundary 0.1 | **0.8982** | **0.3797** | **0.6576** | 0.7112 | **0.6493** | **0.5320** | **0.4650** |

The exact-95% coverage numbers in this table are from the paired bootstrap
diagnostic, not the ordinary global threshold in the discovery report.

## Paired bootstrap

- Seed 123: AUROC difference `+0.00780`, 95% CI `[+0.00311,+0.01259]`;
  unknown-rejection difference `+1.09` percentage points, CI
  `[-2.05,+4.30]` percentage points.
- Seed 44: AUROC difference `+0.01600`, 95% CI `[+0.01114,+0.02084]`;
  unknown-rejection difference `+3.51` percentage points, CI
  `[+0.57,+6.45]` percentage points.

The AUROC gain is positive in both independent seeds and both intervals
exclude zero. The rejection-rate interval is positive for seed 44 but still
crosses zero for seed 123. Therefore this is promising evidence for improved
score ranking and reduced overlap, not yet a final claim of robust deployment
performance.

## Interpretation and next check

The mixed-pool PU boundary term uses the estimated known prior to correct the
boundary penalty, so known samples in the unlabeled pool are not directly
treated as unknown. It acts on the independent rejection representation and
the class-conditional known support boundary. The observed changes are
consistent with the intended mechanism: KS and the mean unknown-minus-known
score gap increase, while the overlap proxy decreases on both seeds.

The method should remain an experimental candidate until a small weight
sensitivity check (`0.05` and `0.2`, same four-epoch protocol) and, if the
effect remains, a longer run with predeclared checkpoint selection are complete.
Do not combine it with additional score fusion or tune the threshold on the
test set while this check is pending.

Artifacts:

- `scripts/run_rejection_boundary_epoch_control.py`
- `analysis/rejection_boundary_epoch_control_bootstrap_s123_e4_20261013.json`
- `analysis/rejection_boundary_epoch_control_bootstrap_s44_e4_20261013.json`

