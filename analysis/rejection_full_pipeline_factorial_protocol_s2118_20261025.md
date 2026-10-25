# Preregistered 2x2 full-pipeline factorial: seed 2118

## Question

The current evidence supports two components separately: `feature_pca` improves
clustering agreement, while fixed known-validation-calibrated uncertainty
fusion is an opt-in rejection candidate. This experiment tests whether their
combination improves the complete novel-class discovery pipeline on one fresh
semantic-hard split, without treating separate ablations as an end-to-end
claim.

## Design

Use one fresh semantic-hard CIFAR-100 60/40 split (`seed=2118`), one
split-matched teacher and one four-epoch shared-backbone student checkpoint.
Evaluate four discovery arms on the same frozen checkpoint:

| Arm | Score | Clustering input |
|---|---|---|
| control | `feature_rejector` | `projection_pca` |
| clustering component only | `feature_rejector` | `feature_pca` |
| rejection component only | `feature_rejector_uncertainty_fusion` | `projection_pca` |
| full treatment | `feature_rejector_uncertainty_fusion` | `feature_pca` |

The uncertainty fusion is fixed at the existing `0.5/0.5` rule, with the
uncertainty-to-error calibration and the 95% known-coverage threshold fit only
on known validation data. Candidate selection, automatic K, PCA dimension,
KMeans and all training hyperparameters are fixed. No weight, threshold,
candidate budget or K-rule search is allowed after seeing seed 2118 results.

Mixed-pool and test labels are excluded from training, checkpoint selection,
calibration, candidate selection and arm selection. They may be used only for
final retrospective rejection, clustering and label-flow audits.

## Joint success gate for the full treatment

Relative to the control arm, the full treatment must simultaneously satisfy:

1. calibrated unknown rejection increases by at least 2 percentage points;
2. test known acceptance decreases by no more than 1 percentage point;
3. FPR95 increases by no more than 1 percentage point;
4. OSCR decreases by no more than 0.005;
5. accepted-known accuracy decreases by no more than 0.5 percentage points;
6. Auto-K absolute error does not increase; and
7. at least one all-unknown NMI and one all-unknown ARI improve materially.

The two main effects and their interaction will be reported separately. A
partial improvement in rejection or clustering is recorded as a component
result, not as validation of the complete method. The project default will not
be changed from this single fresh split.

Machine output is expected at
`analysis/rejection_full_pipeline_factorial_s2118.json`; the post-run report
will be written after all four arms finish.
