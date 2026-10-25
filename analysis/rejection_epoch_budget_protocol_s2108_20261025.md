# Replication protocol: shared-backbone rejection training budget (seed 2108)

This is an exact fresh-split replication of the registered seed-2107 comparison.

- Fresh CIFAR-100 semantic-hard 60/40 split, seed 2108.
- Control: 4 student epochs; treatment: 8 student epochs.
- Same split-matched teacher, shared-backbone independent rejection branch,
  optimizer/objectives, mixed-pool nnPU, checkpoint rule, feature-rejector score,
  automatic clustering, and known-validation-only 95% coverage calibration.
- No feature-KD, uncertainty-KD, VOS, extra boundary loss, selector or fusion
  change. Test and mixed-pool labels are excluded from training, selection and
  calibration.

The seed-2107 treatment improved fixed-coverage unknown rejection by `+2.10 pp`
and AUROC by `+0.02895`, while NMI/ARI declined. This replication is required
before changing the default epoch budget. The joint comparison reports ranking,
fixed-coverage rejection, known utility, OSCR, candidate purity and clustering;
no single favorable metric is sufficient for promotion.

## Result

The runner completed all four and eight training epochs. Both arms calibrated
to the same known-validation coverage; checkpoint selection chose epoch 3 and
epoch 8 respectively. The configuration audit found only the registered epoch
budget and output/checkpoint fields differed.

| Metric | 4 epochs | 8 epochs | Treatment - control |
|---|---:|---:|---:|
| AUROC | 0.785430 | 0.816831 | +0.031401 |
| AUPR | 0.683535 | 0.732591 | +0.049057 |
| FPR95 | 62.050% | 58.767% | -3.283 pp |
| OSCR | 0.572020 | 0.621009 | +0.048989 |
| Test known acceptance | 95.267% | 93.483% | -1.783 pp |
| Unknown rejection at calibrated threshold | 22.975% | 36.350% | +13.375 pp |
| Accepted-known accuracy | 69.629% | 73.792% | +4.163 pp |
| Correctly accepted fraction of all known | 66.333% | 68.983% | +2.650 pp |
| Automatic K | 28 | 39 | +11 |
| NMI / ARI | 0.489030 / 0.155328 | 0.453620 / 0.148356 | -0.035410 / -0.006973 |
| Candidate purity before purification | 76.392% | 78.808% | +2.415 pp |

## Interpretation

Ranking, OSCR, unknown rejection, accepted-known accuracy, correctly accepted
known fraction, and candidate purity improved. However, the test known
acceptance fell by `1.78 pp`, despite equal known-validation calibration
coverage, and clustering NMI fell by `0.0354`. This indicates both a useful
rejection gain and a validation-to-test coverage / clustering trade-off. Along
with seed 2107, the ranking/rejection direction replicates, but the known
acceptance and clustering costs are split-dependent. Keep eight epochs opt-in
and perform one more identical split before deciding whether the training
budget should become the mainline setting.
