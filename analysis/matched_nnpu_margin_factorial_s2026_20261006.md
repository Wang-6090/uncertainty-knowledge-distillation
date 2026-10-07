# Matched PU x uncertainty-margin diagnostic (2026-10-06)

## Question

The previous matched 2x2 study disabled nnPU in every arm. This follow-up
keeps the mixed discovery pool fixed and asks whether the uncertainty-weighted
feature-margin remains useful when mixed-pool nnPU uncertainty learning is
enabled. The factors are nnPU uncertainty loss (off/on) and uncertainty-
weighted feature-margin (off/on).

All four runs use CIFAR-100 random 60/40, seed 2026, the same split,
pretrained ResNet-34 teacher, pretrained ResNet-18 student, full data, batch
size 64, 10 epochs, a matched 5400-image mixed discovery pool with known
prior 0.2, and the same optimizer and augmentations. The detector uses
normalized entropy plus feature kNN with k=10, MC=8, and a threshold fitted
only on known validation data for 95% known coverage. Clustering is skipped.

The PU-on arms use `alpha_discovery_uncertainty_pu=0.1`. The margin arms add
`alpha_discovery_uncertainty_feature_margin=0.05` with cosine margin 0.2.
Baseline arms set the corresponding coefficient to zero. Training logs confirm
that both treatment terms are nonzero only in the intended arms.

## Open-set results

| Pool | PU | Margin | AUROC | AUPR | FPR95 | OSCR | Known accuracy | Known acceptance | Unknown rejection |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed | off | off | 0.6718 | 0.5581 | 0.7783 | 0.3725 | 45.47% | 95.00% | 13.88% |
| Mixed | off | on | 0.7095 | 0.5723 | 0.7182 | 0.4658 | 54.97% | 95.25% | 12.88% |
| Mixed | on | off | 0.7103 | 0.5799 | 0.7293 | 0.4647 | 55.28% | 95.25% | 13.13% |
| Mixed | on | on | 0.7205 | 0.5927 | 0.7170 | 0.4764 | 56.25% | 95.70% | 13.15% |

Within the PU-on pair, adding the margin improves AUROC by 0.0102, reduces
FPR95 by 0.0123, improves OSCR by 0.0118, and improves all-known accuracy by
0.97 percentage points. Known acceptance rises by 0.45 percentage points.
Unknown rejection changes by only 0.02 percentage points, so the margin does
not solve the selected operating-point failure.

The PU-on baseline is also better than the PU-off baseline on ranking and
utility metrics in this seed. That comparison is informative but still
single-seed; it does not establish that nnPU and the margin are independently
robust.

## Feature geometry

The post-hoc diagnostic uses labels only to describe known and unknown test
distributions. It does not fit a model, score, or threshold.

| Distance reference | PU baseline AUROC / overlap | PU + margin AUROC / overlap |
| --- | ---: | ---: |
| Classifier prototype | 0.6860 / 0.7107 | 0.7047 / 0.6898 |
| Empirical class centroid | 0.6912 / 0.7232 | 0.7014 / 0.7004 |
| Nearest known training sample | 0.7138 / 0.6743 | 0.7277 / 0.6534 |

All three distance AUROCs improve and all three histogram overlaps decrease.
This supports a real representation change rather than a threshold-only
effect, but the remaining overlap is still large. The centroid improvement is
smaller than the other two references, so the feature-margin should not be
described as fully separating the known manifold.

## Decision

- Keep nnPU plus uncertainty-weighted feature-margin as the most promising
  current combined candidate, but keep both options default-off until they
  replicate across seeds.
- Do not report the nearly unchanged unknown rejection as a success.
- Do not add NT-Xent or more uniform consistency losses to this combination;
  previous matched tests were unstable and did not give a reliable direction
  away from the known manifold.
- Replicate this exact PU x margin comparison on at least two more seeds with
  paired initialization and controlled sampler/augmentation RNG streams.
- If ranking and geometry improve again but rejection remains flat, inspect
  score calibration and class-conditional operating points on validation data
  and consider a separately trained rejector. If geometry gains disappear,
  demote the margin and prioritize a stronger generalized-category-discovery
  representation objective.

## Implementation audit

Training logs now include
`discovery_uncertainty_feature_margin_weight_mean`, the mean detached soft
novelty weight used by the margin loss. This prevents a nonzero loss value
from being mistaken for effective uncertainty-based sample selection. A toy
smoke run reported a nonzero mean weight and the full unit suite passed.

Artifacts:

- `runs/matched_mixed_nnpu_base_s2026_e10/`
- `runs/matched_mixed_nnpu_margin_s2026_e10/`
- `runs/matched_mixed_nnpu_base_s2026_e10_detect/`
- `runs/matched_mixed_nnpu_margin_s2026_e10_detect/`
- `analysis/matched_nnpu_overlap_s2026.json`
- `runs/smoke_weight_logging/`
