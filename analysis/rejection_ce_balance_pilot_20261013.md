# Rejection CE balance pilot (2026-10-13)

## Question

The rejection-branch gradient audit found that the known-class rejection CE
gradient was much larger than the mixed-pool and outlier unknown-side terms.
This pilot tests whether reducing only the rejection CE coefficient changes the
known/unknown geometry rather than merely changing the final threshold.

## Controlled protocol

The treatment changes only `alpha_rejection_known_ce` from `1.0` to `0.25`.
The following are fixed within each seed: semantic-isolated CIFAR-100 60/40
split, teacher checkpoint, pretrained student initialization, independent
128-dimensional rejection representation, matched mixed discovery pool,
five student epochs, checkpoint rule, detector, MC samples, and validation-only
calibration. Test labels are used only for retrospective metrics.

The seed-44 control is the previously audited
`rejection_boundary_epoch_control_s44_e5_baseline`; its treatment reuses the
same source configuration and teacher. Seed 123 uses the corresponding
`encoder_freeze_rejection_s123_finetune_lr01` control.

## Results

| seed | arm | AUROC | FPR95 | OSCR | KS | unknown below known q95 | known accept | unknown reject |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | CE=1.0 | 0.8865 | 0.4380 | 0.6389 | 0.6202 | 54.30% | 95.08% | 45.30% |
| 123 | CE=0.25 | 0.8895 | 0.4438 | 0.6463 | 0.6277 | 49.43% | 96.72% | 41.88% |
| 44 | CE=1.0 | 0.8995 | 0.3832 | 0.6558 | 0.6453 | 50.45% | 93.90% | 54.50% |
| 44 | CE=0.25 | 0.9033 | 0.3923 | 0.6609 | 0.6528 | 46.48% | 93.78% | 58.38% |

## Interpretation

Both seeds show a small improvement in AUROC, OSCR, KS, and the score-overlap
proxy. This supports the mechanism hypothesis that the rejection branch may be
over-constrained by known CE. However, FPR95 worsens on both seeds. Unknown
rejection also changes in opposite directions, so the result is not a robust
operating-point improvement. The gain is therefore a promising ablation signal,
not a solved core problem and not a reason to change the default coefficient.

The correct next check is an intermediate coefficient such as `0.5`, with the
same protocol. If it does not improve FPR95 and unknown rejection without
reducing known utility, stop tuning CE and return to explicit class-conditional
unknown-boundary supervision. Do not compensate for the FPR95 regression by
post-hoc threshold tuning on the test set.

Artifacts:

- `scripts/run_rejection_ce_balance_pilot.py`
- `runs/rejection_ce_balance_s123_ce0p25/`
- `runs/rejection_ce_balance_s44_ce0p25/`
- `analysis/rejection_branch_gradient_audit_20261013.md`
