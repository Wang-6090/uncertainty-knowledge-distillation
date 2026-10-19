# Candidate-gated PU kNN risk prior audit (2026-10-18)

## Scope

This is a code-and-log audit, not a new performance experiment. It checks
whether the `known_prior` passed to the PU kNN boundary loss still describes
the candidate distribution after the training candidate gate is applied.
No test labels are used here; the fractions below are training-time diagnostic
fields recorded for the mixed discovery pool and are not inputs to optimization.

## Wiring finding

In `novel_discovery/pipeline.py`, `feature_candidate_masks` first selects the
mixed discovery examples using the shared-boundary gate and cross-view
agreement. The selected tensors are then passed to
`pu_knn_support_boundary_loss`. The PU loss computes

`(R_selected_mixed - pi * R_all_known) / (1 - pi)`

using the configured `discovery_knn_known_prior` (0.2) and the full known
minibatch as its detached correction term. It does not apply the same
selection rule to the known correction examples, nor does it adjust `pi` to
the post-selection mixture. Candidate weights, where enabled, further alter
the selected mixture.

The ordinary PU identity only holds when the positive/known correction risk
and prior correspond to the same sampling distribution as the unlabeled risk.
After candidate selection, this condition is not established by the current
wiring. The selected pool should instead use a selection-conditional known
risk and a selection-conditional known prior, or retain the unfiltered pool
and perform a correctly weighted risk estimate.

## Evidence in saved training history

The two arms below use the same seed-44, four-epoch strict kNN risk control.
`feature_candidate_known_fraction` is a diagnostic estimate from known/unknown
labels in the training discovery pool; it is explicitly not used by the loss.

| arm | epoch | selected candidate fraction | known fraction among selected candidates | kNN boundary loss |
|---|---:|---:|---:|---:|
| pure | 1 | 12.58% | 16.67% | 0.01986 |
| pure | 2 | 15.11% | 8.84% | 0.02534 |
| pure | 3 | 15.65% | 6.84% | 0.02503 |
| pure | 4 | 14.94% | 6.19% | 0.02309 |
| hard-NNPU | 1 | 12.74% | 17.21% | 0.01095 |
| hard-NNPU | 2 | 14.90% | 8.75% | 0.00156 |
| hard-NNPU | 3 | 15.93% | 8.47% | 0.00125 |
| hard-NNPU | 4 | 15.26% | 7.11% | 0.00051 |

This is consistent with an over-subtraction/clamping explanation for the
near-zero hard-NNPU risk, but it is not by itself proof that this explains all
open-set score overlap. The logged selected-known fraction is retrospective
diagnostic evidence only and must not be fed into training, model selection,
or threshold calibration.

## Decision and next experiment

Do not promote hard/soft NNPU from these runs, and do not tune its prior using
the retrospective label diagnostics. Before another PU performance run,
implement a label-free selection-conditional correction: apply the identical
cross-view candidate selection rule to labeled known training views, estimate
the selection rates from known training and mixed unlabeled data, form the
implied selected prior, and compute the known-risk correction only on the
selected known examples. Keep the estimate detached from gradients and log
selection rates, effective prior, raw risk, corrected risk, and clamp rate.

Validate the derivation with a synthetic mixture where the true conditional
prior is known, including no-selection and different-known/unknown-selection
cases. Then run only a matched seed-44, four-epoch `pure` versus corrected-PU
comparison; keep data split, candidate gate, architecture/checkpoint seed,
loss weight, detector, and validation-only 95% known-coverage calibration
fixed. Compare AUROC, FPR95, OSCR, KS, score overlap, known acceptance, and
unknown rejection jointly. If the synthetic calibration or known utility
fails, do not spend a full run; revisit the correction derivation first.

## Sources

- Elkan & Noto, *Learning Classifiers from Only Positive and Unlabeled Data*,
  WWW 2008: PU mixture assumptions and the role of the positive prior.
- Kiryo et al., *原文: Positive-Unlabeled Learning with Non-Negative Risk
  Estimator*, NeurIPS 2017: non-negative correction addresses estimator risk
  behavior but does not remove the requirement that the prior/risk distributions
  match the sampled mixture.
- The proposed candidate-conditional correction is a project-specific
  derivation from selection-conditioned mixture proportions; it is not a
  direct reproduction of either paper and requires synthetic and empirical
  validation before use.

## Artifacts

- `runs/knn_pu_boundary_control_s44_e4_pure/train_history.json`
- `runs/knn_pu_boundary_control_s44_e4_hard_nnpu/train_history.json`
- `novel_discovery/pipeline.py`
- `novel_discovery/losses.py`
