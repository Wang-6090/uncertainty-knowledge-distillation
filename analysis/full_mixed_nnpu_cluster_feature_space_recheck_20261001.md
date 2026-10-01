# Full mixed nnPU clustering-space recheck (2026-10-01)

## Purpose and protocol

Unknown detection improved after switching to the explicit min-class kNN
score, but this does not guarantee that the rejected samples form separable
novel classes. This recheck compares the two existing clustering
representations while keeping the nnPU checkpoint, min-class kNN candidate
pool, oracle `K=40`, KMeans settings, and threshold calibration fixed.

- `projection_pca`: the previous default clustering representation;
- `feature_pca`: the encoder feature representation before the projection
  head.

The oracle K is used only to evaluate cluster quality independently from the
auto-K problem. Test labels are used only for descriptive ACC/NMI/ARI and
candidate-purity reporting.

## Candidate clustering results

| Seed | Representation | ACC | NMI | ARI | Candidate purity |
| ---: | --- | ---: | ---: | ---: | ---: |
| 42 | projection_pca | 0.1781 | 0.3717 | 0.0287 | 0.6715 |
| 42 | feature_pca | 0.2172 | 0.4148 | 0.0645 | 0.6715 |
| 123 | projection_pca | 0.1760 | 0.3637 | 0.0316 | 0.6254 |
| 123 | feature_pca | 0.1869 | 0.3897 | 0.0518 | 0.6254 |
| 3407 | projection_pca | 0.1916 | 0.3738 | 0.0318 | 0.6440 |
| 3407 | feature_pca | 0.1976 | 0.3881 | 0.0399 | 0.6440 |

| Mean | projection_pca | 0.1819 | 0.3698 | 0.0307 | 0.6470 |
| Mean | feature_pca | 0.2006 | 0.3975 | 0.0521 | 0.6470 |

The feature representation improves all three cluster metrics in all three
seeds, while candidate purity is unchanged. This is evidence that the choice
of clustering space matters and that the projection head is not currently the
best space for novel-class discovery.

The improvement is modest. Even with oracle K, NMI remains below 0.40 on
average and ARI remains low. The dominant unresolved issue is therefore the
novel-class structure of the learned embedding, not only the unknown threshold
or clustering algorithm.

## Auto-K audit

On seed 42 with the same candidate pool, silhouette auto-K selected `K=48`
(true K=40), while composite auto-K selected `K=29`. Their cluster NMI/ARI
were `0.4351/0.0314` and `0.3630/0.0276`, respectively. The disagreement
shows that internal criteria are not yet reliable class-count estimators for
this embedding.

## Decision

Use `feature_pca` as the current explicit clustering representation in the
recommended experimental pipeline, but keep `projection_pca` as an ablation.
Do not claim automatic new-class discovery is complete. The next substantive
algorithmic change should target representation learning with a genuine
known-plus-novel GCD objective, periodic pseudo-label updates, and balanced
assignments, rather than more threshold or KMeans tuning.
