# Continuous risk-weighted mixed-pool rejector protocol

## Purpose

The seed-2160 late-CE audit showed that the main FPR95 failure is in the low
unknown-score tail. The next candidate keeps the entire mixed discovery pool
for nnPU fitting, but gives a continuous, label-free weight to samples that
look more unknown-like. This avoids hard pseudo-unknown selection, which can
remove difficult unknowns from the training pool.

## Paired comparison

- control: `rejector_pool_selection=all`, equal mixed-pool weights;
- treatment: `rejector_pool_selection=weighted_risk`, all mixed-pool samples
  retained with weights in `[1, 2]` from the existing ranked consensus of
  entropy, MSP risk, classifier-prototype distance, and student uncertainty;
- follow-up candidate: `rejector_pool_selection=weighted_neighbor_risk`, which
  keeps the same pointwise consensus and adds a fixed 10-neighbor feature-risk
  consensus with coefficient `0.25`; this is a separate opt-in mechanism, not
  a weight scan;
- fixed: student checkpoint, teacher, semantic-hard CIFAR-100 60/40 split,
  mixed pool, `known_prior=0.2`, `nu_corrected` linear nnPU, feature mode,
  validation-only 95% known-coverage calibration, clustering and all test
  evaluation code;
- unknown labels are not used for weighting, fitting, calibration, checkpoint
  selection, or clustering selection. They are used only for final metrics.

The candidate risk is detached and normalized before fitting. It is a weighting
signal, not a label or a second detector score. The final detector remains the
fitted feature rejector in both arms.

For the neighbor variant, the feature kNN is computed on detached normalized
student features from the mixed pool. Each sample's neighbor risk is the mean
pointwise risk of its 10 nearest samples, and the combined risk is
`0.75 * pointwise + 0.25 * neighbor`. No labels, threshold, or test score is
used in this calculation.

## Decision rule

Report AUROC, AUPR, FPR95, OSCR, known acceptance and accepted-known accuracy
separately from candidate purity, unknown candidate recall, and ACC/NMI/ARI.
The method is not promoted by an increase in unknown rejection alone. A useful
result must improve ranking/tail behavior without a material loss of known
utility, and must be confirmed on a later predeclared split before changing
defaults.
