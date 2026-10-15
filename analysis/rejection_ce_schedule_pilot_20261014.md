# Rejection CE schedule pilot (2026-10-14)

## Purpose

Static rejection-CE weights gave a small but conflicting signal: lower CE
improved AUROC/KS/overlap on two seeds, but FPR95 did not improve consistently.
This pilot tests a training-time alternative: keep the known rejection CE at
full strength while the representation stabilizes, then decay it so unknown
boundary losses have more relative influence.

## Implementation audit

The new optional schedule is controlled by:

- `--rejection-known-ce-decay-start-epoch`
- `--rejection-known-ce-decay-epochs`
- `--rejection-known-ce-final-scale`

For the pilot, the base coefficient was `1.0`, the decay started at one-based
epoch 3, lasted two epochs, and ended at scale `0.25`. The recorded scales were
exactly `[1.0, 1.0, 0.625, 0.25, 0.25]`. With all schedule arguments at their
defaults, the scale remains 1.0, so historical behavior is unchanged.

## Controlled result: seed 44

The semantic-isolated CIFAR-100 60/40 split, teacher, pretrained student,
rejection dimension, matched mixed pool, five epochs, detector, MC samples,
checkpoint criterion, and validation-only calibration were fixed. Only the
rejection CE schedule differed.

| arm | AUROC | FPR95 | OSCR | KS | unknown below known q95 | known accept | unknown reject |
|---|---:|---:|---:|---:|---:|---:|---:|
| static CE=1.0 | 0.8995 | 0.3832 | 0.6558 | 0.6453 | 50.45% | 93.90% | 54.50% |
| static CE=0.5 | 0.8966 | 0.3777 | 0.6599 | 0.6357 | 51.53% | 95.10% | 48.18% |
| static CE=0.25 | 0.9033 | 0.3923 | 0.6609 | 0.6528 | 46.48% | 93.78% | 58.38% |
| staged 1.0 -> 0.25 | 0.8969 | 0.3968 | 0.6687 | 0.6475 | 52.63% | 94.98% | 47.45% |

## Conclusion

The staged schedule has the best OSCR in this one seed, but it worsens FPR95,
score overlap, and unknown rejection relative to the static CE=1.0 control.
Therefore it does not solve the core known/unknown overlap problem and should
remain an optional ablation, not the default. The next priority is explicit
class-conditional boundary supervision on the independent rejection
representation, especially the PU rejection-boundary loss that previously
showed more direct overlap improvements under equal-epoch control. Its
checkpoint and loss-weight sensitivity must be audited together.

Artifacts:

- `scripts/run_rejection_ce_balance_pilot.py`
- `runs/rejection_ce_balance_s44_ce1p0_decay3x2to0p25/`
- `analysis/rejection_ce_balance_pilot_20261013.md`
