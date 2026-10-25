# Independent 4-vs-8 epoch and clustering replication protocol (seed 2113)

## Motivation

An exploratory frozen-checkpoint audit on seeds 2107--2109 suggested that
`feature_pca` can improve automatic-K error for eight-epoch models relative to
four-epoch models (absolute errors 8->1, 15->0 and 4->3). Those diagnostics
used test labels and are hypothesis-generating only. Seed 2113 is a fresh,
locked replication of the training-budget comparison with `feature_pca` fixed
for both arms.

## Paired conditions

- Fresh semantic-hard CIFAR-100 60/40 split, seed 2113; one split-matched
  teacher is shared by both students.
- Control: 4 student epochs. Treatment: 8 student epochs.
- Both use the same registered student recipe, independent shared-backbone
  rejection head, nnPU training, feature-rejector score, `feature_pca`,
  silhouette Auto-K, and known-validation-only 95% coverage calibration.
- No feature KD, VOS, new boundary loss, fusion, or selector change. Only the
  epoch budget differs. Test/mixed labels cannot enter training, checkpoint
  selection, threshold calibration, or method selection.
- Report ranking, calibrated unknown rejection, known coverage/utility,
  unknown-only NMI/ARI/ACC, automatic K and absolute K error. Test labels are
  used only for final retrospective metrics.

## Decision rule

Call the longer-budget recipe promising on this split only if unknown rejection
improves by at least 2 percentage points, automatic-K absolute error decreases,
and NMI and ARI do not both decline. Also report known acceptance and accepted
known accuracy; the result does not change defaults after one split. If this
joint gate fails, do not tune epochs on seed 2113; stop this interaction
candidate pending a separately registered study.

