# Angular schedule and uncertainty-score follow-up (2026-10-01)

## Purpose

This round tested two hypotheses without changing the CIFAR-100 random 60/40
protocol:

1. Introducing the ArcFace-style angular-margin loss after a known-class
   warm-up might avoid the early representation damage seen when it is enabled
   from the first epoch.
2. The uncertainty head trained by mixed-pool nnPU might provide extra unknown
   ranking information when added to the current entropy plus min-class kNN
   score.

The first comparison retrained the student. The second comparison reused the
same `angular_nnpu_s42` checkpoint and changed only the detection score.

## Controlled conditions

- CIFAR-100 random 60/40 split, seed 42
- pretrained ResNet-34 teacher and ResNet-18 student
- 2 epochs, 1200 known train samples, 300 validation samples, 1200 mixed
  discovery samples, 1000 open-test samples
- mixed-pool known prior measured from the split: `0.210833`
- nnPU weight: `alpha_discovery_uncertainty_pu=0.1`
- Angular direct baseline: `alpha_angular=0.05`, margin `0.2`, scale `16`
- detector: normalized entropy plus min-class kNN, `k=10`
- clustering: oracle `K=40`, `feature_pca`

## Results

| Method | AUROC | FPR95 | Known acceptance | Unknown rejection | Candidate purity | Candidate NMI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Angular direct + nnPU | 0.6084 | 0.8694 | 96.36% | 8.61% | 0.6071 | 0.7971 |
| Angular warm-up/ramp + nnPU | 0.6093 | 0.8893 | 96.53% | 7.09% | 0.5714 | 0.8343 |
| Angular direct + nnPU + uncertainty score | 0.6099 | 0.8694 | 94.38% | 10.13% | 0.5405 | 0.8319 |

The warm-up schedule improves candidate NMI but worsens the operating-point
and candidate-purity metrics. The uncertainty correction increases unknown
rejection mainly by rejecting additional known samples; its AUROC gain is
negligible and candidate purity decreases. Neither change resolves the
known/unknown overlap.

## Code changes and decision

- Added optional `--angular-warmup-epochs` and `--angular-ramp-epochs`.
  Defaults are zero, so historical runs are unchanged. The current schedule
  remains an ablation candidate, not the default training recipe.
- Added the experimental score
  `normalized_entropy_uncertainty_min_class_knn`. It standardizes the entropy,
  min-class kNN distance, and head uncertainty on known validation data, then
  uses `z_entropy + z_knn + 0.5*z_uncertainty`.
- The uncertainty-corrected score remains disabled by default because the
  extra rejection is not sufficiently selective.

The current primary direction remains nnPU training plus explicit min-class
kNN detection. Angular `alpha=0.05` is worth a three-seed confirmation because
its earlier pilot improved FPR95 and candidate purity, but the warm-up variant
is not preferred on this protocol. The next useful work should focus on
improving the representation or obtaining cleaner mixed-pool candidates rather
than adding more unverified score terms.
