# Preregistered prototype-boundary rejector replication: seed 2119

## Question

Does fixed prototype-boundary weighting improve the current feature-rejector
operating point by increasing the influence of label-free candidate samples
near known-support boundaries, without using mixed-pool or test labels?

## Paired protocol

- Fresh semantic-hard CIFAR-100 60/40 split, seed `2119`.
- One split-matched teacher and one four-epoch shared-backbone student.
- Same mixed pool, known prior, feature mode, NNPU model, score mode and
  known-validation-only 95% coverage calibration in both arms.
- Control: `rejector_pool_selection=all`.
- Treatment: `weighted_boundary`, `rejector_candidate_mode=prototype_boundary`,
  fixed candidate ratio `0.25`, fixed candidate weight `2.0`.
- Clustering is disabled so this experiment evaluates the detector component
  only; it is not an end-to-end clustering claim.

The treatment rule and values are fixed before running this fresh split. Test
and mixed-pool labels are excluded from training, checkpoint selection,
calibration, candidate selection and method selection. They may be used only
for final retrospective metrics and label-flow auditing.

## Joint success gate

The fixed treatment is considered directionally useful on this split only if it
simultaneously improves calibrated unknown rejection by at least 2 percentage
points, keeps test known acceptance loss within 1 percentage point, does not
increase FPR95 by more than 1 percentage point, does not reduce OSCR by more
than 0.005, and does not reduce accepted-known accuracy by more than 0.5
percentage points. A ranking-only improvement or a rejection-rate increase
accompanied by unacceptable known loss is recorded as a trade-off.

This single fresh split cannot change the default. If it fails, stop this
prototype-boundary branch; if it passes, record it as a candidate for a later
multi-split confirmation without tuning on seed 2119.

Machine output: `analysis/prototype_boundary_fresh_replication_s2119.json`.
