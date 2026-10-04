# Corrected disjoint mixed-pool NT-Xent comparison (2026-09-29)

## Purpose

This experiment tests whether the earlier mixed-pool result was distorted by
reusing supervised known samples as unlabeled discovery samples. The corrected
protocol reserves 20% of the known training partition as a disjoint unlabeled
known subset. The question is whether adding two-view NT-Xent on this corrected
mixed pool improves open-set detection over the same training protocol without
the discovery loss.

## Controlled conditions

- CIFAR-100, fixed 60/40 class split, seed 42.
- Same saved pretrained ResNet-34 teacher and pretrained ResNet-18 student.
- 3 epochs, batch size 64, 1200 train / 300 validation / 1000 test limits.
- `mixed_known_pool_ratio=0.2`, `limit_discovery=1200` for both runs.
- Same uncertainty KD, feature KD, SupCon and prototype weights:
  `alpha_unc=0.1`, `alpha_kd=1.0`, `alpha_feat_kd=0.1`,
  `alpha_supcon=0.1`, `alpha_proto=0.1`.
- Only intended training change: corrected mixed NT-Xent is disabled in the
  baseline and enabled with `alpha_discovery=0.05`, temperature 0.2 in the
  treatment.
- Detection: same test subset, MC=4, `normalized_entropy_mahalanobis`,
  known-only 95% coverage threshold, clustering skipped.

The baseline also uses the corrected disjoint data construction, but does not
create a discovery loader. This keeps the supervised sample removal identical
between the two groups.

## Results

| Method | AUROC | AUPR | FPR95 | OSCR | Known accuracy | Known accept rate | Unknown reject rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Corrected mixed, no NT-Xent | 0.4953 | 0.4029 | 0.9030 | 0.1453 | 0.2220 | 0.9490 | 0.0561 |
| Corrected mixed + NT-Xent | **0.5494** | **0.4205** | 0.9178 | 0.1827 | **0.2664** | 0.9605 | 0.0536 |
| Change | +0.0541 | +0.0176 | +0.0148 | +0.0374 | +0.0444 | +0.0115 | -0.0026 |

The reported operating point is calibrated on known validation samples at a
target of 95% known coverage; finite-sample quantiles produce small deviations
in the test known acceptance rate. The baseline and treatment use the same
policy, so the comparison remains paired.

## Interpretation

The corrected NT-Xent run is a promising representation signal: AUROC, AUPR,
OSCR and known classification accuracy improve in this one-seed smoke-scale
comparison. However, FPR95 becomes worse and unknown rejection at the matched
known-coverage operating point decreases slightly. Therefore NT-Xent does not
yet demonstrate that it separates known and unknown distributions; it may be
improving the general representation/classification ranking without moving the
operating boundary in the desired direction.

The result is not directly comparable to the earlier full-data pure-novel-pool
F results because this run uses a disjoint mixed pool, reduced training budget
and only one seed. It is also not enough to promote NT-Xent as the final method.

## Decision

Keep corrected mixed NT-Xent as a candidate representation baseline and do not
search its weight blindly. The next targeted method should use the mixed pool
to learn semantic known-versus-novel assignments: retain supervised known
anchors, allow high-confidence unlabeled samples to match known classes, and
apply balanced novel prototype consistency only to samples that are supported
by cross-view and neighborhood evidence. This directly addresses the failure
of instance-level consistency to create a known/unknown boundary.

Before a final claim, repeat the corrected comparison with at least two more
seeds and report matched-coverage rejection, FPR95, known accuracy and feature
overlap diagnostics together.
