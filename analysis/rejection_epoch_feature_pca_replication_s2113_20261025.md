# Four-versus-eight epoch replication with `feature_pca`: seed 2113

## Protocol and implementation audit

This fresh semantic-hard CIFAR-100 60/40 split paired four- and eight-epoch
student budgets. Both students used one split-matched teacher, the same
registered student recipe, feature-rejector score, `feature_pca`, silhouette
Auto-K, and known-validation-only 95% calibration. Configuration audits found
only epoch budget, checkpoint metadata, and paths differed. The calibration
acceptance was `94.890%` in both arms. Test labels were not used for training or
calibration; they were used only for final retrospective metrics.

The initial runner attempt did not produce a student checkpoint: it loaded an
experiment metadata wrapper instead of the underlying registered student
configuration, leaving the rejection feature dimension unset. This was
corrected before training/evaluation by loading the source config directly and
checking student/teacher dimensions. The failed output directory was
preserved; results below come from the corrected `_v2` run.

## Results

| Metric | 4 epochs | 8 epochs | Change |
|---|---:|---:|---:|
| AUROC | 0.799805 | 0.825805 | +0.026000 |
| AUPR | 0.700038 | 0.731109 | +0.031071 |
| FPR95 | 59.000% | 54.933% | -4.067 pp |
| OSCR | 0.589808 | 0.625881 | +0.036073 |
| Test known acceptance | 95.367% | 94.200% | -1.167 pp |
| Unknown rejection at calibrated threshold | 24.825% | 32.375% | +7.550 pp |
| Accepted-known accuracy | 70.814% | 73.496% | +2.682 pp |
| Correctly accepted fraction of all known | 67.533% | 69.233% | +1.700 pp |
| Candidate purity | 78.127% | 78.819% | +0.692 pp |
| Automatic K (true K=40) | 35 (error 5) | 37 (error 3) | absolute error -2 |
| Unknown-only NMI | 0.566624 | 0.530119 | -0.036504 |
| Unknown-only ARI | 0.301486 | 0.257946 | -0.043541 |
| Unknown-only ACC | 45.720% | 40.154% | -5.566 pp |

## Decision

The eight-epoch arm improved ranking, fixed-threshold novel rejection, accepted
known accuracy, candidate purity, and automatic-K error. However, both
unknown-only NMI and ARI declined, and test known acceptance fell by `1.167
pp`. It therefore fails the preregistered joint gate requiring no simultaneous
NMI/ARI decline, despite satisfying the rejection and K-error conditions. This
is a tradeoff, not a general improvement to new-class discovery. Keep the
four-epoch default and do not tune further on seed 2113.

Compared with exploratory seeds 2107--2109, the direction is consistent for
K-error improvement (here `5→3`), and unknown rejection improves on this split;
however, those earlier data were test-label diagnostics and this independent
split exposes the clustering-quality cost clearly. More independent registered
splits would be required before considering a multi-objective recipe.

Machine output: [`rejection_epoch_feature_pca_s2113_v2.json`](rejection_epoch_feature_pca_s2113_v2.json).
Protocol: [`rejection_epoch_feature_pca_replication_protocol_s2113_20261025.md`](rejection_epoch_feature_pca_replication_protocol_s2113_20261025.md).
Runner: [`../scripts/run_rejection_epoch_feature_pca_pair.py`](../scripts/run_rejection_epoch_feature_pca_pair.py).
