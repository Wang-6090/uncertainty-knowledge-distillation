# Auto-K selector comparison protocol (seed 2112)

## Motivation

Two independent frozen-checkpoint clustering comparisons support `feature_pca`
for NMI/ARI, but the estimated K moved farther below the known evaluation
count (40). Existing code provides four label-free selectors: silhouette,
composite (silhouette/Calinski-Harabasz/Davies-Bouldin), stability, and
diagonal-GMM BIC. Compare them without changing the representation or model.

## Locked conditions

- New semantic-hard CIFAR-100 60/40 split, seed 2112.
- Train one split-matched teacher and one four-epoch student using the existing
  registered recipe; reuse that student checkpoint for all discovery arms.
- All arms use `feature_pca`, identical feature-rejector score, candidate mask,
  threshold calibrated on known validation only to 95%, KMeans, PCA, and
  candidate-pool processing. Only `cluster_selection` changes.
- Compare all four pre-existing selectors; do not tune selector settings or
  choose a winner from test labels. Report rejection, known utility, selected
  K, NMI/ARI/ACC, and the per-K internal diagnostic curves. Test labels are
  only for retrospective metrics and never enter model fitting, K selection,
  calibration, or selector choice.
- Default changes are out of scope for this sweep, regardless of test metric
  ordering. A follow-up change requires an independently registered protocol.

