# Independent clustering-input replication protocol (seed 2111)

## Preregistered comparison

Seed 2110 showed improved clustering metrics with `feature_pca` and no change
to rejection metrics. This second independently generated semantic-hard
CIFAR-100 60/40 split repeats the same single-factor comparison to test whether
that result persists. The hypothesis, metrics, and stop rule are fixed before
this split's test metrics are examined.

- One split-matched teacher and one four-epoch student checkpoint are shared
  between arms.
- Control uses `projection_pca`; treatment uses `feature_pca`.
- Score (`feature_rejector`), candidate mask, automatic-K rule, known-validation
  only 95% coverage calibration, training recipe and checkpoint rule are fixed.
- Test/mixed labels cannot be used for training, selection, calibration or
  tuning; test labels are permitted only for final retrospective metrics.
- Require NMI and ARI both to improve to count as a replication. If either
  fails, stop this representation candidate rather than tune to this split.
  Even if both improve, keep the default unchanged until more splits confirm.

