# Semantic-hard rejection Outlier Exposure pair protocol (seed 2120)

## Question

Does adding a uniform-logit objective on external CIFAR-10 images improve the
current independent rejection branch when it already uses an external
feature-margin objective? The external images are used only as an auxiliary
unlabelled exposure source. Their CIFAR-10 class labels are not loaded into
the training decision; CIFAR-100 mixed-pool and test labels are not used for
training, checkpoint selection, calibration, or method selection.

## Fixed paired comparison

- Fresh semantic-hard CIFAR-100 60/40 split, seed `2120`.
- One split-matched teacher shared by both students.
- Current shared-backbone independent rejection encoder and the registered
  four-epoch student recipe.
- Same standard logits KD, mixed-pool NNPU discovery objective, candidate
  protocol, feature rejector, automatic-K procedure, and known-validation-only
  95% coverage calibration.
- Control: external CIFAR-10 rejection feature-margin loss only,
  `alpha_outlier_rejection_feature_margin=0.05`, margin `0.2`, and
  `alpha_outlier_rejection_uniform=0`.
- Treatment: the same configuration plus fixed rejection uniform-logit OE,
  `alpha_outlier_rejection_uniform=0.05`.
- No weight, margin, threshold, candidate ratio, checkpoint, or clustering
  parameter is selected from mixed-pool/test labels.

## Decision rule

The treatment must be judged jointly by AUROC, AUPR, FPR95, OSCR, score-tail
overlap, calibrated unknown rejection, known acceptance, accepted-known
accuracy, automatic K, and NMI/ARI. A single split is exploratory and cannot
change the default. The preregistered success screen is: unknown rejection
improves by at least 2 percentage points, AUROC does not decline, FPR95 does
not worsen by more than 1 percentage point, OSCR and accepted-known accuracy
do not decline by more than 0.005, and clustering is not materially harmed.
If the uniform term is active but the joint screen fails, stop this exact
combination rather than scanning its coefficient on the same split.

## Integrity checks

The runner must verify that the only algorithmic difference is
`alpha_outlier_rejection_uniform`, that the treatment history contains a
nonzero uniform rejection term, that the control term is zero, and that both
arms use the same teacher/split/evaluation protocol. The report records these
checks together with the absolute arm metrics and paired differences.
