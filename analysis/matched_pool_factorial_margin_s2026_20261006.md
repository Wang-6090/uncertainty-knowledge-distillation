# Matched pool-composition × uncertainty-margin diagnostic (2026-10-06)

## Question and predeclared factors

The previous pure-unknown-pool pilot suggested that uncertainty-weighted
feature-margin might help, but did not isolate the effect of pool composition:
the earlier pure and mixed runs differed in supervised known-data count and
nnPU. This 2×2 diagnostic asks (1) whether the loss effect remains within a
controlled mixed pool, and (2) whether its effect appears to depend on pool
composition.

| Factor | Level 1 | Level 2 |
| --- | --- | --- |
| Discovery pool | Pure unknown (oracle-filtered diagnostic) | Mixed, 20% known / 80% unknown |
| Student loss | Baseline | Uncertainty-weighted feature-margin |

All four arms use CIFAR-100 random 60/40 class split, training seed 2026,
full data, the same ResNet-34 teacher checkpoint, pretrained ResNet-18
student, 64px images, batch size 64, and 10 epochs. A matched protocol
reserves the same known supervised subset in both pool conditions and keeps
the discovery pool at 5,400 images; supervised train size is 25,650 in every
arm. The pure pool has 0% known samples; the mixed pool has 20% known samples.
The nnPU loss is disabled in all four arms. Treatment adds only
`alpha_discovery_uncertainty_feature_margin=0.05`, cosine margin `0.2`; the
baseline has coefficient 0. Training logs confirm the treatment loss is
nonzero in the treatment arms and zero in baseline arms.

Detection conditions are fixed across arms: full open test set,
`normalized_entropy_min_class_knn`, feature kNN `k=10`, MC=8, and a threshold
calibrated only from known validation examples for target 95% known coverage.
Clustering is skipped. Test labels are used only for final metrics and
post-hoc known/unknown geometry descriptions, never for model selection or
threshold calibration. Pure unknown is an oracle diagnostic and not a
deployable protocol.

## Open-set results

| Pool | Arm | AUROC | AUPR | FPR95 | OSCR | Known accuracy (all known) | Known acceptance | Unknown rejection |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure | Baseline | 0.6779 | 0.5526 | 0.7693 | 0.3865 | 46.32% | 95.55% | 10.98% |
| Pure | + uncertainty margin | 0.7063 | 0.5744 | 0.7373 | 0.4580 | 54.67% | 94.95% | 13.88% |
| Pure | Treatment − baseline | +0.0284 | +0.0218 | −0.0320 | +0.0716 | +8.35pp | −0.60pp | +2.90pp |
| Mixed | Baseline | 0.6718 | 0.5581 | 0.7783 | 0.3725 | 45.47% | 95.00% | 13.88% |
| Mixed | + uncertainty margin | 0.7095 | 0.5723 | 0.7182 | 0.4658 | 54.97% | 95.25% | 12.88% |
| Mixed | Treatment − baseline | +0.0377 | +0.0142 | −0.0602 | +0.0933 | +9.50pp | +0.25pp | −1.00pp |

The one-seed treatment improved AUROC, FPR95, OSCR, and all-known
classification accuracy in both pool conditions. But fixed-coverage unknown
rejection increased in the pure pool and decreased in the mixed pool. This is
important: improved ranking and classification do **not** guarantee higher
unknown rejection at one calibrated operating point. Do not claim that pool
contamination is the cause of this difference: this is one training seed,
and changing pool composition can also change the random-number trajectory
used by data sampling/augmentation. The pool×loss interaction needs replicated
seeds and a deterministic shared initialization/data-order protocol.

## Feature-distance overlap diagnostics

Reported pairs are distance AUROC / histogram overlap; higher AUROC and lower
overlap are favorable. These are descriptive diagnostics on test-known and
test-unknown examples, not fitting signals.

| Pool | Distance reference | Baseline | + uncertainty margin |
| --- | --- | ---: | ---: |
| Pure | Classifier prototype | 0.6627 / 0.7564 | 0.6834 / 0.7156 |
| Pure | Empirical class centroid | 0.6140 / 0.8339 | 0.6779 / 0.7358 |
| Pure | Nearest known training sample | 0.6698 / 0.7521 | 0.7026 / 0.6851 |
| Mixed | Classifier prototype | 0.6599 / 0.7628 | 0.6919 / 0.7031 |
| Mixed | Empirical class centroid | 0.6211 / 0.8319 | 0.6786 / 0.7333 |
| Mixed | Nearest known training sample | 0.6665 / 0.7638 | 0.7077 / 0.6793 |

All six geometry comparisons move favorably for this seed, which is consistent
with a representation effect rather than only a threshold shift. However,
substantial overlap remains (treatment histogram overlap is about 0.68–0.74),
and the pool-specific rejection discrepancy shows that geometry metrics alone
do not establish deployment utility or robust separation.

## Interpretation and next steps

This is a more informative, controlled diagnostic than the earlier
pure-vs-mixed comparison because supervised sample count, pool size, teacher,
training budget, and nnPU status are held constant. It supports keeping the
margin loss as a promising optional component, but does not justify enabling
it by default. It also does not prove that the mixed pool is harmful or that
the pure-pool benefit transfers to deployment.

Before another algorithm change, replicate this exact 2×2 experiment on at
least two more training seeds, reusing the same split per seed and sharing
teacher initialization between each baseline/treatment pair. Ensure the
sampler and augmentation RNG streams are explicitly controlled so the pool
factor does not silently alter training randomness. Report paired deltas and
the pool×loss interaction for AUROC, FPR95, OSCR, known accuracy, actual known
acceptance, unknown rejection, and feature overlap. If ranking/geometry gains
replicate but rejection does not, next inspect score calibration and
class-conditional operating points on a separate validation protocol; do not
change the test threshold. If geometry gains fail to replicate, demote the
margin term and prioritize a stronger GCD representation objective or a
separate learned rejector trained without test labels.

## Artifacts

- Training: `runs/matched_pure_base_s2026_e10`,
  `runs/matched_pure_margin_s2026_e10`,
  `runs/matched_mixed_base_s2026_e10`,
  `runs/matched_mixed_margin_s2026_e10`.
- Detection reports: corresponding `_detect/discovery_report.json` files.
- Full-data geometry: `analysis/matched_pool_factor_pure_overlap_s2026.json`
  and `analysis/matched_pool_factor_mixed_overlap_s2026.json`.
