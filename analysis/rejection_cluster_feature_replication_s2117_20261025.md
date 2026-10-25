# Third independent clustering-input replication: seed 2117

## Protocol validity

This is the preregistered independent semantic-hard CIFAR-100 60/40
replication. One split-matched teacher and one four-epoch student checkpoint
were shared by both discovery runs. The only substantive condition difference
was `cluster_feature` (`projection_pca` versus `feature_pca`); the configuration
audit found no other differences except output paths. The candidate pool,
feature-rejector score, automatic-K rule, PCA dimension, KMeans settings and
known-validation-only 95% coverage calibration were fixed before comparison.
Test and mixed labels were not used for training, checkpoint selection,
calibration, candidate selection or method selection; they were used only for
retrospective metrics and the audit.

The runner's identity audit found exact equality for AUROC, AUPR, FPR95, OSCR,
test known acceptance, calibrated-threshold unknown rejection,
accepted-known accuracy, correctly accepted-known fraction and validation
calibration acceptance. The candidate mask was also identical: 1,103 samples,
with retrospective candidate purity 77.335%.

## Results

| Metric | `projection_pca` | `feature_pca` | Change |
|---|---:|---:|---:|
| AUROC | 0.799849 | 0.799849 | 0 |
| AUPR | 0.692041 | 0.692041 | 0 |
| FPR95 | 58.933% | 58.933% | 0 pp |
| OSCR | 0.575753 | 0.575753 | 0 |
| Test known acceptance | 95.833% | 95.833% | 0 pp |
| Unknown rejection at calibrated threshold | 21.325% | 21.325% | 0 pp |
| Accepted-known accuracy | 68.904% | 68.904% | 0 pp |
| Correctly accepted fraction of all known | 66.033% | 66.033% | 0 pp |
| Validation calibration acceptance | 94.882% | 94.882% | 0 pp |
| Automatic K (true K=40) | 36 (error 4) | 40 (error 0) | error -4 |
| All-unknown clustering NMI / ARI | 0.434485 / 0.195720 | 0.490112 / 0.258722 | +0.055627 / +0.063003 |
| Candidate-unknown clustering NMI / ARI | 0.510516 / 0.188264 | 0.583006 / 0.310336 | +0.072489 / +0.122073 |
| Candidate-unknown clustering ACC | 33.411% | 44.080% | +10.668 pp |
| Unknown-only clustering NMI / ARI / ACC | 0.494127 / 0.187952 / 33.060% | 0.573623 / 0.303154 / 43.845% | +0.079496 / +0.115202 / +10.785 pp |
| Candidate purity | 77.335% | 77.335% | 0 pp |

## Decision

This split passes the preregistered joint gate: all reported NMI and ARI
values improve, automatic-K absolute error does not increase and in fact
decreases from 4 to 0, and candidate purity does not decrease. The unchanged
rejection and known-class metrics show that the gain is isolated to the
clustering representation rather than caused by a different candidate pool or
threshold.

Together with seeds 2110 and 2111, `feature_pca` improves all-unknown NMI/ARI
on all three independent replications. The corresponding mean gains across
the three replications are approximately `+0.06896/+0.07846` for all-unknown
NMI/ARI and `+0.08332/+0.12392` for candidate-unknown NMI/ARI. However,
automatic-K error worsened on 2110 (`4→6`) and 2111 (`9→10`) before improving
on 2117 (`4→0`); the three-split mean absolute error changes from `5.67` to
`5.33` classes. Thus this is stronger evidence for a clustering-quality
component, but not evidence that feature_pca alone solves the full novel-class
discovery problem or the cross-split Auto-K instability.

The project default is therefore not changed by this result. Keep
`feature_pca` as the explicitly recommended clustering component in the main
analysis path and retain `projection_pca` as the controlled ablation, while
reporting Auto-K error separately. Do not tune PCA dimensions, K rules or
candidate thresholds using this seed.

Machine-readable result:
[`rejection_cluster_feature_replication_s2117.json`](rejection_cluster_feature_replication_s2117.json).
Protocol:
[`rejection_cluster_feature_replication_protocol_s2117_20261025.md`](rejection_cluster_feature_replication_protocol_s2117_20261025.md).
Runner:
[`../scripts/run_rejection_cluster_feature_replication.py`](../scripts/run_rejection_cluster_feature_replication.py).
