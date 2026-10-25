# Descriptive four-split summary: 4 vs 8 epochs with `feature_pca`

This summary combines frozen-checkpoint diagnostics from seeds 2107--2109 with
the independently registered seed-2113 paired run. The first three splits were
used to formulate the seed-2113 hypothesis and are exploratory, not independent
confirmation. The mean is descriptive only; no inferential claim is made.

| Seed | Status | AUROC delta | Unknown rejection delta (pp) | Test known acceptance delta (pp) | Auto-K absolute-error delta | Unknown-only NMI delta | Unknown-only ARI delta | Unknown-only ACC delta (pp) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2107 | Exploratory | +0.028950 | +2.100 | +0.533 | -7 | +0.027510 | +0.033120 | -0.497 |
| 2108 | Exploratory | +0.031401 | +13.375 | -1.783 | -15 | -0.057850 | -0.108370 | -8.894 |
| 2109 | Exploratory | +0.018197 | -1.175 | +0.683 | -1 | -0.004975 | -0.007878 | -1.880 |
| 2113 | Independent replication | +0.026000 | +7.550 | -1.167 | -2 | -0.036504 | -0.043541 | -5.566 |
| Mean | Descriptive, n=4 | +0.026137 | +5.463 | -0.433 | -6.25 | -0.017955 | -0.031667 | -4.209 |

Across all four splits, AUROC, AUPR, OSCR, accepted-known accuracy and the
correctly accepted fraction of known samples improved on every split. Unknown
rejection improved on three of four, while test known acceptance declined on
two. Auto-K absolute error improved on all four. In contrast, unknown-only NMI
and ARI both improved only on seed 2107 and both declined on the other three;
their mean deltas were `-0.017955` and `-0.031667`. Thus longer training
reliably improves ranking and estimated K in this small set, but usually
degrades novel-cluster agreement. It does not pass the joint objective and is
not promoted to the default.

The seed-2113 decision is the only independent confirmation in this summary.
The prior three comparisons are retained as exploratory context and must not be
counted as three fresh replications of a pre-registered hypothesis.

Detailed independent result: [`rejection_epoch_feature_pca_replication_s2113_20261025.md`](rejection_epoch_feature_pca_replication_s2113_20261025.md).
