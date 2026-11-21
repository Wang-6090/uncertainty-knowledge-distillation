# Late CE decay plus boundary-uncertainty frozen audit

## Purpose

The seed-2160 late rejection-CE decay replication improved ranking, fixed-coverage
unknown rejection, candidate purity, and clustering, but worsened FPR95. Earlier
frozen audits found a fixed boundary-uncertainty correction (`band=1.0`, `alpha=0.5`)
to improve ranking on several completed splits. This stage tests the interaction
without retraining either student.

## Fixed protocol

- Inputs are the already completed seed-2160 `static_control` and `late_decay`
  checkpoints and their matching discovery configurations.
- The only evaluation change is boundary uncertainty with `band_multiplier=1.0`
  and `alpha=0.5`, fixed before reading this stage's test metrics.
- Calibration remains known-validation-only at 95% known coverage. Test labels are
  read only for final detection and clustering reports.
- The same detector, mixed pool, candidate rule, PCA/KMeans settings, and
  checkpoint are used as in the raw discovery reports.
- The raw report for each arm is the within-arm control; no cross-arm parameter is
  selected from test results.

## Decision rule

The combination is a candidate only if it improves FPR95 or preserves it while
retaining the late-decay gains in AUROC, unknown rejection, and clustering. A
mixed result is recorded as a tradeoff and does not trigger an alpha or band scan.
This is an evaluation-only audit, not evidence that the boundary correction caused
the original late-decay training gains.

Runner: `scripts/run_rejection_ce_decay_boundary_combo_s2160.py`.
