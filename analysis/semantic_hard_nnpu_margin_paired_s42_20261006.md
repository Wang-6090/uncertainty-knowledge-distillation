# Semantic-hard nnPU + uncertainty-margin paired test (2026-10-06)

## Question

The central failure mode is overlap between known and unknown feature/score
distributions. This paired test asks whether mixed-pool nnPU uncertainty
learning plus an uncertainty-weighted feature margin changes the representation,
rather than only changing the detector threshold.

## Controlled factors

Both arms used the same CIFAR-100 semantic-hard 60/40 split, seed 42,
pretrained ResNet-34 teacher, pretrained ResNet-18 student, full known data,
batch size 64, 10 epochs, the same teacher checkpoint, and a matched 5,400
image mixed discovery pool with known prior 0.2. The detector used the same
feature kNN plus normalized entropy score, MC=8, and a threshold calibrated
only on known validation data for 95% known coverage. Test labels were used
only for final descriptive metrics.

The treatment enabled `alpha_discovery_uncertainty_pu=0.1` and
`alpha_discovery_uncertainty_feature_margin=0.05` with cosine margin 0.2.
The baseline set both coefficients to zero. No detector threshold or test
label was changed between arms.

## Open-set results

| Arm | AUROC | FPR95 | OSCR | Known accuracy | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.5735 | 0.8627 | 0.3170 | 42.23% | 95.35% | 6.43% |
| nnPU + uncertainty margin | **0.6319** | **0.8430** | **0.3955** | **51.48%** | 95.62% | **9.03%** |

The treatment improves ranking and utility metrics while keeping known
acceptance almost unchanged. Unknown rejection rises by 2.60 percentage
points. This is evidence of a useful single-seed improvement, but rejection
is still low and the result is not yet a multi-seed claim.

## Feature-overlap result

The post-hoc diagnostic uses test labels only to describe distributions; it
does not fit a model, threshold, or detector.

| Distance reference | Baseline AUROC / overlap | Treatment AUROC / overlap |
| --- | ---: | ---: |
| Classifier prototype | 0.5694 / 0.8805 | **0.6367 / 0.7079** |
| Empirical class centroid | 0.5341 / 0.9293 | **0.6049 / 0.8395** |
| Nearest known training sample | 0.5749 / 0.8765 | **0.6209 / 0.8208** |

All three references show lower histogram overlap and higher unknownness
ranking AUROC. Therefore the gain is not explained by threshold adjustment
alone. However, the remaining overlap is substantial, so this method has not
solved the core problem.

## Decision

- Keep nnPU plus uncertainty-weighted feature margin as the leading candidate,
  but keep it optional until an independent seed reproduces the direction.
- Do not increase the loss weights or stack more detector scores based on this
  single result.
- Complete the seed-43 paired replication with the same protocol.
- If the representation gain replicates, run at least one more seed and then
  restore clustering to compare candidate purity, NMI, and ARI.
- If only AUROC improves while the fixed operating point remains weak, focus
  next on the known-support model and candidate quality, not threshold search.

## Implementation interpretation

The nnPU term is an uncertainty-risk objective for a mixed unlabeled pool; it
does not assign per-example unknown labels. The feature-margin term uses the
detached uncertainty-derived novelty weight, so its effect depends on whether
the uncertainty head ranks mixed-pool samples meaningfully. The logged
nonzero loss and weight mean confirm that both terms were active, while the
overlap diagnostic provides the stronger evidence that they changed geometry.

Artifact: `analysis/semantic_hard_nnpu_margin_overlap_s42_full_20261006.json`.
