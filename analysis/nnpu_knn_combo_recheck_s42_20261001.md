# nnPU plus KNN support-boundary combination (seed 42, 2026-10-01)

## Purpose and controlled comparison

The three-seed nnPU recheck established a stable improvement in uncertainty
ranking. KNN support-boundary had shown partial local-support gains in earlier
experiments, so this test asked whether it adds complementary representation
separation on top of nnPU.

The control was the already completed full-data mixed nnPU student at seed 42.
The treatment used the same teacher, split, data, five epochs, optimizer,
mixed-pool prior, and nnPU weight, adding only:

- `alpha_discovery_knn_boundary=0.1`;
- mixed-pool candidate gating;
- two warm-up epochs and a two-epoch linear ramp;
- hard candidate weights, with soft weighting disabled.

Both checkpoints were evaluated with the same test set, MC=4,
`normalized_entropy_mahalanobis`, known-validation 95% coverage calibration,
and clustering disabled.

## Results

| Metric | nnPU only | nnPU + KNN boundary | Change |
| --- | ---: | ---: | ---: |
| AUROC | 0.6498 | 0.6349 | -0.0149 |
| FPR95 | 0.8283 | 0.8340 | +0.0057 |
| OSCR | 0.3735 | 0.3476 | -0.0259 |
| Known acceptance | 95.45% | 94.67% | -0.78 pp |
| Unknown rejection | 9.45% | 10.33% | +0.88 pp |
| Accepted-known accuracy | 50.50% | 47.83% | -2.67 pp |

The extra rejection is accompanied by worse ranking, lower known coverage, and
lower accepted-known accuracy. This is a false trade-off for the project goal,
not a useful complementary gain. The KNN boundary term remains an optional
ablation, but this combination should not be promoted or tuned further on the
same protocol.

## Decision

Keep nnPU as the training method and use the min-class kNN score as the next
detection candidate. Do not combine nnPU with this KNN training loss unless a
new protocol gives independent evidence.
