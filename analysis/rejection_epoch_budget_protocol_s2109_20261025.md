# Replication protocol: shared-backbone rejection training budget (seed 2109)

Third and final registered split for the current 4-vs-8 epoch budget check.

- Fresh CIFAR-100 semantic-hard 60/40 split, seed 2109.
- Control: 4 student epochs; treatment: 8 student epochs.
- Same split-matched teacher, shared-backbone independent rejection branch,
  objectives/optimizer, mixed-pool nnPU, checkpoint selection, feature-rejector
  score, automatic clustering, and known-validation-only 95% coverage
  calibration.
- No feature-KD, uncertainty-KD, VOS, extra boundary loss, selector or fusion
  change. Mixed-pool/test labels do not enter training, checkpoint selection,
  calibration or method choice.

Seeds 2107/2108 both improved ranking and fixed-coverage unknown rejection, but
seed 2108 lost `1.78 pp` test known acceptance and `0.0354` NMI. This third split
will assess whether the gains and costs persist. No hyperparameter or score
adjustment is allowed after inspecting prior test metrics.

## Result

The paired run completed with history lengths 4 and 8. The same known-validation
checkpoint rule selected epoch 4 and epoch 8. Configuration audit found only
the epoch budget and output/checkpoint metadata differed.

| Metric | 4 epochs | 8 epochs | Treatment - control |
|---|---:|---:|---:|
| AUROC | 0.809740 | 0.827937 | +0.018197 |
| AUPR | 0.702909 | 0.726664 | +0.023756 |
| FPR95 | 56.117% | 51.283% | -4.833 pp |
| OSCR | 0.599373 | 0.629529 | +0.030156 |
| Test known acceptance | 94.883% | 95.567% | +0.683 pp |
| Unknown rejection at calibrated threshold | 26.700% | 25.525% | -1.175 pp |
| Accepted-known accuracy | 71.544% | 73.073% | +1.529 pp |
| Correctly accepted fraction of all known | 67.883% | 69.833% | +1.950 pp |
| Automatic K | 37 | 37 | 0 |
| NMI / ARI | 0.492595 / 0.159659 | 0.486271 / 0.145252 | -0.006324 / -0.014407 |
| Candidate purity before purification | 77.673% | 79.332% | +1.659 pp |

## Three-split decision

Across seeds 2107–2109, extending to eight epochs improved AUROC, AUPR, FPR95,
OSCR, accepted-known accuracy, and correctly accepted-known fraction on all
three splits. Candidate purity improved on all three. However, fixed-coverage
unknown rejection improved on only two splits (+2.10 pp, +13.375 pp, -1.175 pp),
while NMI and ARI declined on all three. Seed 2108 also had a `1.78 pp` drop in
test known acceptance despite matched known-validation calibration.

Therefore the longer budget is a reproducible ranking/known-class optimization
change but not a reliable new-class rejection improvement. It is not promoted
to the default; no further epoch-budget search is justified from these results.
The next bottleneck is the transfer from known-validation calibration and known
classification improvements to semantic-near unknown rejection and clustering.
