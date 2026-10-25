# Independent clustering-input replication: seed 2110

## Protocol and validity

This is the preregistered independent semantic-hard CIFAR-100 60/40 replication
of the exploratory seeds 2107--2109 clustering-input audit. One split-matched
teacher and one four-epoch student checkpoint were used for both discovery
runs. The only condition difference was `cluster_feature` (`projection_pca`
versus `feature_pca`); the config audit found no other differences except output
paths. Both arms use the same feature-rejector score, automatic-K rule, and
known-validation-only 95% coverage calibration. Calibration acceptance was
`94.92%` in both. The rejector protocol report confirms test labels were not
used for training or calibration; test labels were used only for retrospective
metrics. No candidate tuning was performed.

## Results

| Metric | `projection_pca` | `feature_pca` | Change |
|---|---:|---:|---:|
| AUROC | 0.789447 | 0.789447 | 0 |
| AUPR | 0.685280 | 0.685280 | 0 |
| FPR95 | 62.817% | 62.817% | 0 pp |
| OSCR | 0.569943 | 0.569943 | 0 |
| Test known acceptance | 95.617% | 95.617% | 0 pp |
| Unknown rejection at calibrated threshold | 22.625% | 22.625% | 0 pp |
| Accepted-known accuracy | 69.078% | 69.078% | 0 pp |
| Correctly accepted fraction of all known | 66.050% | 66.050% | 0 pp |
| Automatic K (true K=40) | 36 (error 4) | 34 (error 6) | -2; absolute error +2 |
| All-unknown clustering NMI / ARI | 0.456326 / 0.213457 | 0.547689 / 0.319961 | +0.091362 / +0.106505 |
| Candidate-unknown clustering NMI / ARI | 0.520367 / 0.214286 | 0.603958 / 0.333923 | +0.083591 / +0.119637 |
| Candidate-unknown clustering ACC | 35.028% | 46.740% | +11.713 pp |

The fixed-seed clustering input change improved both NMI and ARI, as well as
candidate-unknown clustering ACC, while leaving rejection and known-class
utility exactly unchanged. However, automatic K moved farther below the true
40 novel classes (absolute error 4 to 6). This independently reproduces the
exploratory direction on clustering agreement but exposes a tradeoff in class
count estimation. Treat this as promising but incomplete evidence for the
clustering component, not a validated improvement to the complete novel-class
discovery method; keep the default unchanged pending further splits and
explicit evaluation of K error.

Machine-readable paired output:
[`rejection_cluster_feature_replication_s2110.json`](rejection_cluster_feature_replication_s2110.json).
Runner:
[`../scripts/run_rejection_cluster_feature_replication.py`](../scripts/run_rejection_cluster_feature_replication.py).
