# Ungated mixed-pool kNN PU control (2026-10-18)

## Purpose

This paired control tests whether the hard NNPU correction itself improves the
shared feature geometry when the candidate gate is removed. The earlier audit
found a possible prior/distribution mismatch after candidate selection. If
hard-NNPU were useful independently of that mismatch, it should improve the
same ungated mixed pool against a contamination-ignorant control.

The control is deliberately small and is not a full production experiment.
It uses the same four-epoch protocol at seeds 44 and 123.

## Fixed conditions and changed factor

- CIFAR-100 semantic-isolated 60/40 split;
- mixed discovery pool of 5,400 samples with configured known prior 0.2;
- same source train/discover configurations, architecture, data protocol,
  optimizer, checkpoint selection, detector and validation-only 95% known
  coverage calibration;
- `candidate_gating=false` and `cross_view_gating=false` for both arms;
- `alpha_discovery_knn_boundary=0.1`, all independent rejection-boundary
  terms disabled;
- test labels are used only for the final report;
- the only algorithmic variable is `discovery_knn_risk`:
  `naive_mixed` versus `hard_nnpu`.

`naive_mixed` is an explicit contaminated-pool control. It applies the
ordinary boundary penalty to the whole mixed pool and does not claim that the
pool is pure unknown data.

## Results

| seed | risk | AUROC | FPR95 | OSCR | known accept | unknown reject | AUPR |
|---:|---|---:|---:|---:|---:|---:|---:|
| 44 | naive_mixed | 0.902374 | 0.346500 | 0.659630 | 94.767% | 49.500% | 0.842958 |
| 44 | hard_nnpu | 0.901560 | 0.347667 | 0.663014 | 94.617% | 50.700% | 0.839660 |
| 123 | naive_mixed | 0.889590 | 0.420167 | 0.638726 | 95.500% | 45.475% | 0.833659 |
| 123 | hard_nnpu | 0.889536 | 0.412500 | 0.634528 | 96.867% | 40.450% | 0.837142 |

Relative to `naive_mixed`, hard-NNPU changes were:

- seed 44: AUROC -0.000814, FPR95 +0.001167, OSCR +0.003384,
  known acceptance -0.15 percentage points, unknown rejection +1.20 points;
- seed 123: AUROC -0.000053, FPR95 -0.007667, OSCR -0.004197,
  known acceptance +1.37 points, unknown rejection -5.03 points.

The direction is not consistent across seeds. The small changes in AUROC are
also much smaller than the changes in the operating-point rejection rate, so
the result does not support a claim that NNPU separated the known and unknown
score distributions.

The final kNN boundary loss was lower for hard-NNPU in both runs
(`0.07261` vs `0.09204` at seed 44; `0.06250` vs `0.08455` at seed 123),
but a lower training loss did not translate into stable detection gains.

## Decision

Do not make `hard_nnpu` the default and do not continue tuning its prior or
threshold based on this control. The experiment narrows the diagnosis:

1. candidate-gated PU prior mismatch is a valid wiring concern for the gated
   path, but it is not sufficient to explain all known/unknown overlap;
2. hard NNPU on an ungated mixed pool is not a stable solution by itself;
3. the core issue remains the learned representation and class-conditional
   boundary, not a single risk scale or threshold.

The gated-path correction should therefore be treated as a separate,
label-free methodological audit. Before using it for a performance claim,
validate the selection-conditional derivation on a synthetic mixture and log
selection rates, effective prior, raw/corrected risk and clamp rate. If that
passes, compare it against the current gated control with all other settings
fixed. If it fails, stop the PU branch and return to direct class-conditional
representation/rejector supervision.

## Artifacts

- `analysis/knn_pu_ungated_s44_e4_summary.json`
- `analysis/knn_pu_ungated_s123_e4_summary.json`
- `scripts/run_knn_pu_ungated_control.py`
- `runs/knn_pu_ungated_s44_e4_*`
- `runs/knn_pu_ungated_s123_e4_*`

## Follow-up formula audit

Before changing the gated training path, a standalone synthetic audit checked
the selection-conditional identity

`pi_selected = pi * selection_rate_known / selection_rate_mixed`.

With 500,000 simulated samples in two different selection-rate settings, the
corrected PU formula recovered the known synthetic unknown risk with maximum
absolute error `1.98e-7`. This validates the algebra and implementation of the
helper, not the real candidate gate. The real implementation still needs to
measure both rates with exactly the same label-free rule and log sampling
noise/invalid estimates before it can be used in training.

The helper and audit are in
`novel_discovery/losses.py` and `scripts/audit_selection_conditional_pu.py`;
the generated report is
`analysis/selection_conditional_pu_synthetic_audit_20261018.json`.
