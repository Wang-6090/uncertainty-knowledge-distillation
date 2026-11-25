# Frozen candidate-pool three-seed summary

All clustering assignments were produced from label-free frozen pools. Ground-truth labels were loaded only for retrospective evaluation.

## Candidate pool

| Seed | Candidates | Purity | Known contamination | Unknown recall | Zero-recall unknown classes |
|---:|---:|---:|---:|---:|---:|
| 42 | 466 | 0.3562 | 0.6438 | 0.0415 | 5 |
| 123 | 432 | 0.3403 | 0.6597 | 0.0367 | 3 |
| 3407 | 494 | 0.3482 | 0.6518 | 0.0430 | 3 |

## Oracle-K feature comparison

These are retrospective descriptions. They are not used to select a deployable representation.

| Feature pipeline | ACC | NMI | ARI |
|---|---:|---:|---:|
| feature_pca (L2 + PCA/whitening + L2) | 0.3407 +/- 0.0186 | 0.6151 +/- 0.0280 | 0.1008 +/- 0.0346 |
| projection_pca (L2 + PCA/whitening + L2) | 0.3246 +/- 0.0198 | 0.6075 +/- 0.0126 | 0.0946 +/- 0.0252 |
| feature (raw) | 0.3371 +/- 0.0203 | 0.5796 +/- 0.0183 | 0.0986 +/- 0.0276 |
| feature (L2-normalized) | 0.3292 +/- 0.0295 | 0.6011 +/- 0.0137 | 0.0985 +/- 0.0208 |

## All-true-unknown oracle diagnostic

Ground-truth known/unknown identity and K are used only to define this evaluation-only diagnostic.

| Scope | ACC | NMI | ARI |
|---|---:|---:|---:|
| All true unknown samples | 0.2496 +/- 0.0026 | 0.3336 +/- 0.0056 | 0.1236 +/- 0.0014 |
| Rejected true-unknown candidates | 0.3407 +/- 0.0186 | 0.6151 +/- 0.0280 | 0.1008 +/- 0.0346 |

## Exploratory method comparison on feature_pca

`feature_pca` is a fixed diagnostic anchor here. Because the anchor was retained after retrospective label-based evaluation, this method comparison is exploratory rather than confirmatory.

| Method | ACC | NMI | ARI |
|---|---:|---:|---:|
| kmeans | 0.3407 +/- 0.0186 | 0.6151 +/- 0.0280 | 0.1008 +/- 0.0346 |
| agglomerative | 0.3368 +/- 0.0252 | 0.6115 +/- 0.0329 | 0.0967 +/- 0.0199 |
| spectral | 0.3464 +/- 0.0246 | 0.6203 +/- 0.0256 | 0.0987 +/- 0.0224 |

## Decision

- Mean candidate purity: 0.3482 +/- 0.0080.
- Mean unknown recall: 0.0404 +/- 0.0033.
- Changed factor: feature form, matched L2 normalization, or clustering method; candidate membership and oracle K remain fixed.
- Formal baseline: `feature_pca + KMeans` on the frozen candidate pool. Feature comparisons are retrospective descriptions only.
- The matched raw control isolates the added PCA+whitening stage from the L2-normalization change.
- No clustering method dominates ACC, NMI, and ARI across all seeds.
- The three seeds agree that candidate purity is below 0.36 and unknown recall is below 0.05; metric-level representation and method winners are not uniformly consistent.
- The evidence supports candidate-pool quality as the immediate bottleneck, but does not rule out representation or K-estimation error.
- Next step: improve candidate recall/purity under a separately validated detector protocol. Auto K remains deferred until that pool is usable.
