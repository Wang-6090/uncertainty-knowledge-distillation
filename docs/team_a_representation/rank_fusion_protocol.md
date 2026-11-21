# Validation-only empirical rank fusion protocol

## Scope

This document freezes the discovery-time candidate score
`feature_rejector_uncertainty_rank_fusion` as an opt-in diagnostic. It is a
post-hoc score audit only: teacher and student checkpoints remain frozen and
no training loss is changed.

## Fixed rule

For each component, compute an empirical mid-rank against the known-validation
reference array only:

`rank(x; R) = (left_R(x) + right_R(x) + 1) / (2 * |R|)`

where `left_R` and `right_R` are the counts strictly below and at-or-below
`x`. Values are clipped to `[0, 1]`; ties use the midpoint rank. The final
score is fixed as:

`0.5 * rank(feature_rejector_score; R_feature) +`
`0.5 * rank(head_uncertainty; R_uncertainty)`.

The operating threshold is calibrated from the resulting known-validation
scores with the existing known-coverage policy. No fusion weight, threshold,
rank reference, candidate ratio, or test result is tuned.

## Information boundary

- `R_feature` and `R_uncertainty` are fitted from known validation samples.
- Unknown labels are not used for training, score construction, or threshold
  calibration; they are used only for final retrospective evaluation.
- The raw feature-rejector score and the raw student uncertainty head remain
  separate signals. This mechanism does not use calibrated uncertainty error,
  classification entropy, teacher uncertainty, or KD weights.
- Calibration exports include both raw validation components and their rank
  components for auditability.

## Frozen pilot decision (2026-11-21)

The canonical implementation was evaluated on the existing frozen formal
conjunction A/B/C artifacts for seeds `42/123/3407`, without retraining. The
comparison changed only the discovery-time score and used the same known-
validation 95% coverage policy. The machine result is the ignored local file
`analysis/rank_fusion_pilot_canonical_20261121.json`; the runner is
[`scripts/run_rank_fusion_pilot.py`](../../scripts/run_rank_fusion_pilot.py).

Across all nine paired arm/seed comparisons against raw
`feature_rejector`, the mean changes were:

| Metric | Mean change | Direction count |
|---|---:|---:|
| AUROC | -0.069463 | 0/9 improved |
| AUPR | -0.071072 | 0/9 improved |
| FPR95 | +0.102204 | 9/9 worse |
| OSCR | -0.049257 | 0/9 improved |
| Unknown rejection | -4.103 pp | 0/9 improved |
| Candidate purity | -6.475 pp | 0/9 improved |

The candidate therefore fails the fixed ranking and operating-point gates. It
is frozen as an opt-in diagnostic and will not enter formal fresh-seed
training, default configuration, or further weight/threshold scans.
