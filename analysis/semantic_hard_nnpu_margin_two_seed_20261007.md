# Semantic-hard nnPU + uncertainty-margin two-seed replication (2026-10-07)

## Purpose and protocol

This report extends the seed-42 paired test with seed 43. The hypothesis is
that mixed-pool nnPU uncertainty learning plus an uncertainty-weighted feature
margin changes the known/unknown representation geometry. The baseline and
treatment at each seed used the same semantic-hard CIFAR-100 60/40 split,
teacher checkpoint, pretrained ResNet-34/ResNet-18, full data, 10 epochs,
batch size 64, matched 5,400-image mixed pool with known prior 0.2, and the
same detector. The detector used normalized entropy plus feature kNN, MC=8,
and a threshold fitted only on known validation data for 95% known coverage.
Test labels were used only for final reporting and post-hoc diagnostics.

The treatment enabled `alpha_discovery_uncertainty_pu=0.1` and
`alpha_discovery_uncertainty_feature_margin=0.05`, with cosine margin 0.2.
The paired baseline set both coefficients to zero.

## Open-set results

| Seed | Arm | AUROC | FPR95 | OSCR | Known accuracy | Known acceptance | Unknown rejection |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | Baseline | 0.5735 | 0.8627 | 0.3170 | 42.23% | 95.35% | 6.43% |
| 42 | nnPU + uncertainty margin | **0.6319** | **0.8430** | **0.3955** | **51.48%** | 95.62% | **9.03%** |
| 43 | Baseline | 0.5837 | 0.8602 | 0.2955 | 39.20% | 94.93% | 6.40% |
| 43 | nnPU + uncertainty margin | **0.6260** | **0.8113** | **0.3999** | **51.88%** | 95.13% | **8.60%** |
| Mean | Baseline | 0.5786 | 0.8614 | 0.3062 | 40.72% | 95.14% | 6.41% |
| Mean | nnPU + uncertainty margin | **0.6290** | **0.8272** | **0.3977** | **51.68%** | 95.38% | **8.81%** |

Across the two seeds, the treatment changes are +0.0503 AUROC, -0.0343
FPR95, +0.0915 OSCR, +10.97 percentage points known accuracy, and +2.40
percentage points unknown rejection. Known acceptance changes by only +0.23
percentage points. This is meaningful replication evidence, but the absolute
unknown rejection remains low; the core problem is reduced, not solved.

## Feature-overlap results

| Distance reference | Seed | Baseline AUROC / overlap | Treatment AUROC / overlap |
| --- | ---: | ---: | ---: |
| Classifier prototype | 42 | 0.5694 / 0.8805 | 0.6367 / 0.7079 |
| Classifier prototype | 43 | 0.5807 / 0.8763 | 0.6141 / 0.8133 |
| Empirical class centroid | 42 | 0.5341 / 0.9293 | 0.6049 / 0.8395 |
| Empirical class centroid | 43 | 0.5471 / 0.9159 | 0.5970 / 0.8472 |
| Nearest known training sample | 42 | 0.5749 / 0.8765 | 0.6209 / 0.8208 |
| Nearest known training sample | 43 | 0.5739 / 0.8774 | 0.6136 / 0.8318 |

All six treatment comparisons improve unknownness AUROC and reduce histogram
overlap. This confirms that the open-set gain is not just threshold movement.
The centroid and local-support overlaps remain high, which explains why the
unknown operating-point rejection is still modest.

## Decision and next step

1. Retain nnPU plus uncertainty-weighted feature margin as the leading
   representation-training candidate, but keep it optional until a third seed
   or an independent split is checked.
2. Do not stack more score fusion or tune the threshold from these results;
   the remaining failure is representation/support overlap.
3. Run the same two-seed treatment with clustering enabled and fixed oracle K
   to determine whether the improved representation also improves unknown
   candidate purity, NMI, and ARI.
4. If clustering does not improve, inspect the candidate pool and move toward
   a stronger generalized-category-discovery objective or a class-conditional
   support model. If it improves, test auto-K separately without using test
   labels.

## Seed-43 consistency follow-up

Because the treatment improved candidate purity but reduced unknown-only NMI and
ARI, an opt-in backbone feature consistency term was tested. It aligned two
augmented discovery views without assigning pseudo labels. Detection slightly
regressed, while candidate purity and candidate unknown-only NMI/ARI improved:

| Seed-43 arm | AUROC | FPR95 | OSCR | Unknown rejection | Candidate purity | Unknown-only NMI | Unknown-only ARI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nnPU + uncertainty margin | 0.6260 | 0.8113 | 0.3999 | 8.60% | 52.86% | 0.4403 | 0.0346 |
| + feature consistency | 0.6241 | 0.8183 | 0.3928 | 8.13% | **54.17%** | **0.4854** | **0.0565** |

This is evidence that the consistency term partially repairs the clustering
side effect, but it does not improve the complete objective and remains below
the seed-43 baseline on NMI/ARI. It should remain an ablation, not the default.
Full details are in
`analysis/semantic_hard_feature_consistency_recheck_s43_20261007.md`.

Artifacts:

- `analysis/semantic_hard_nnpu_margin_paired_s42_20261006.md`
- `analysis/semantic_hard_nnpu_margin_overlap_s42_full_20261006.json`
- `analysis/semantic_hard_nnpu_margin_overlap_s43_full_20261007.json`
- `runs/semantic_hard_current_nnpu_margin_s43_detect/`
- `runs/semantic_hard_current_nnpu_baseline_s43_detect/`
