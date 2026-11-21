# Late CE decay plus boundary uncertainty: frozen interaction audit

## Purpose and protocol

The seed-2160 late rejection-CE decay replication improved ranking, fixed-coverage
unknown rejection, candidate purity, and clustering, but worsened FPR95. This
evaluation-only stage applies the previously fixed boundary-uncertainty correction
to both already trained arms. It does not retrain or select a checkpoint.

The correction was fixed at `band_multiplier=1.0` and `alpha=0.5`. Both arms use
their own raw discovery report as the within-arm control. Calibration remains
known-validation-only at 95% known coverage, with a disjoint calibration subset;
test labels are used only for the final metrics below. The two corrected runs both
report `test_labels_used_for_training_or_calibration=false`.

## Results: corrected minus raw within each arm

| metric | static control | late decay |
|---|---:|---:|
| AUROC | +0.009318 | +0.009384 |
| AUPR | +0.022049 | +0.020483 |
| FPR95 | 0.000000 | 0.000000 |
| OSCR | +0.004229 | +0.005659 |
| test known acceptance | +0.033 pp | +1.317 pp |
| accepted-known accuracy | -0.134 pp | -0.385 pp |
| unknown rejection | +3.975 pp | +1.225 pp |
| candidate purity | +2.157 pp | +3.748 pp |
| ACC | +3.332 pp | +2.448 pp |
| NMI | +1.132 pp | +0.994 pp |
| ARI | +1.784 pp | +1.653 pp |
| Auto-K error | 9 -> 12 | 6 -> 7 |

Absolute values for the late-decay arm were raw/corrected: AUROC
`0.811941/0.821325`, AUPR `0.726373/0.746856`, FPR95
`0.634333/0.634333`, OSCR `0.594852/0.600511`, unknown rejection
`38.050%/39.275%`, candidate purity `77.693%/81.441%`, and ACC/NMI/ARI
`0.407351/0.431830`, `0.561216/0.571159`, `0.300579/0.317111`.

## Decision

Boundary uncertainty is partially complementary to late CE decay: it improves
ranking, OSCR, candidate purity, and clustering on this split. However, it leaves
FPR95 unchanged, increases known acceptance beyond the calibrated target on the
late arm, reduces accepted-known accuracy, and slightly worsens Auto-K error.
Therefore this combination does not solve the current tail-separation problem.
Keep it as a diagnostic composition, do not change the default configuration, and
do not scan `alpha` or band width on this test split. A future continuation should
target tail-aware score training or validation-only calibration on a fresh split,
with FPR95 and known utility as explicit success criteria.

Runner: `scripts/run_rejection_ce_decay_boundary_combo_s2160.py`.
