# Independent clustering-input replication protocol (seed 2110)

## Question and preregistration

An exploratory frozen-checkpoint audit on seeds 2107--2109 found higher NMI and
ARI when clustering used the backbone embedding (`feature_pca`) instead of the
projection head (`projection_pca`). Because that audit inspected test-label
metrics, it is hypothesis-generating only. This new semantic-hard CIFAR-100
60/40 split tests the locked candidate `feature_pca` against the existing
`projection_pca` control.

## Paired conditions

- New split seed 2110; same split-matched teacher and one four-epoch student
  checkpoint for both arms.
- Control: `projection_pca`; treatment: `feature_pca`.
- Only the clustering representation may differ. The feature-rejector score,
  candidate mask, automatic-K rule, checkpoint, and known-validation-only 95%
  coverage calibration are fixed.
- No test or mixed-pool labels may be read for training, checkpoint selection,
  calibration, method selection, or tuning. Test labels may be used only for
  final retrospective metrics (including NMI/ARI).

## Decision rule

First verify all config differences are limited to `cluster_feature` and output
paths, and that rejection, coverage, and known-utility metrics are identical
between arms. Treat the candidate as independently replicated only if both
NMI and ARI improve on this split. A single split is still limited evidence and
does not warrant changing the default without further independent replication.
If either clustering metric fails to improve, stop this candidate branch rather
than tuning it against this split's test labels.

