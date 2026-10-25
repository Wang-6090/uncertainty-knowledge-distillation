# Second independent clustering-input replication: seed 2111

## Protocol validity

This follows the preregistered seed-2111 protocol. One split-matched teacher and
one four-epoch student checkpoint were shared by both arms. Config comparison
found only `cluster_feature` and output paths differed. The score, candidate
mask, automatic-K rule and known-validation-only 95% calibration were fixed.
The runner also verified exact equality of AUROC, AUPR, FPR95, OSCR, test known
acceptance, fixed-threshold unknown rejection, accepted-known accuracy,
correctly accepted-known fraction and validation calibration. Test/mixed labels
were not used for training, checkpoint selection or calibration; test labels
were used only for retrospective metrics.

## Results

| Metric | `projection_pca` | `feature_pca` | Change |
|---|---:|---:|---:|
| AUROC | 0.801594 | 0.801594 | 0 |
| AUPR | 0.700395 | 0.700395 | 0 |
| FPR95 | 59.517% | 59.517% | 0 pp |
| OSCR | 0.583340 | 0.583340 | 0 |
| Test known acceptance | 95.467% | 95.467% | 0 pp |
| Unknown rejection at calibrated threshold | 23.850% | 23.850% | 0 pp |
| Accepted-known accuracy | 70.059% | 70.059% | 0 pp |
| Correctly accepted fraction of all known | 66.883% | 66.883% | 0 pp |
| Automatic K (true K=40) | 31 (error 9) | 30 (error 10) | -1; absolute error +1 |
| All-novel NMI / ARI | 0.429551 / 0.195642 | 0.489437 / 0.261508 | +0.059886 / +0.065867 |
| Candidate-novel NMI / ARI | 0.491782 / 0.178374 | 0.585669 / 0.308426 | +0.093887 / +0.130051 |
| Candidate-novel clustering ACC | 31.866% | 43.082% | +11.216 pp |

Both NMI and ARI improve, reproducing seed 2110's direction on a second
independent split. Combined across 2110 and 2111 (descriptive only), the mean
all-novel gains are `+0.07562` NMI and `+0.08619` ARI; candidate-novel gains
are `+0.08874` NMI and `+0.12484` ARI. Rejection and known-class utility remain
unchanged by construction. Automatic K also moves one class farther below the
true 40 novel classes, so the NMI/ARI gains trade off against worse class-count
estimation on this split. This supports `feature_pca` as a clustering-quality
candidate, but is not evidence that the full uncertainty-KD/rejection method
is better. Keep the project default unchanged until broader split replication
and review of automatic-K error are complete.

Machine result: [`rejection_cluster_feature_replication_s2111.json`](rejection_cluster_feature_replication_s2111.json).
Protocol: [`rejection_cluster_feature_replication_protocol_s2111_20261025.md`](rejection_cluster_feature_replication_protocol_s2111_20261025.md).
