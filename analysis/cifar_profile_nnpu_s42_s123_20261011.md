# CIFAR-profile mixed-pool nnPU replication and follow-up ablations (2026-10-11)

## Purpose

This record tests whether the positive seed-123 result for CIFAR-style
preprocessing plus audited mixed-pool uncertainty nnPU reproduces on seed 42.
It also records two follow-up changes: a larger nnPU coefficient and semantic
support-based soft weights. The goal is to reduce known/unknown score overlap,
not to optimize a test-set threshold.

All discover runs use the same semantic-isolated CIFAR-100 60/40 split,
CIFAR-style preprocessing, ResNet-34 teacher / ResNet-18 student, five epochs,
MC=4, `normalized_entropy_min_class_knn`, k=10, validation-only 95% known
coverage, and clustering skipped. Test labels are used only for final
descriptive metrics.

## Replication: alpha=0.1, unweighted mixed-pool nnPU

The only training-side change from the seed-42 CIFAR-profile baseline is the
audited mixed-pool nnPU objective with `alpha=0.1`, known prior 0.2, and
`weight-source=none`.

| Metric | CIFAR baseline | CIFAR + nnPU | Change |
|---|---:|---:|---:|
| Best validation known accuracy | 56.30% | 62.87% | +6.57 pp |
| Test known accuracy | 56.65% | 63.08% | +6.43 pp |
| AUROC | 0.7710 | 0.7906 | +0.0196 |
| AUPR | not recorded | 0.6757 | - |
| FPR95 | 0.6547 | 0.6285 | -0.0262 |
| OSCR | 0.4934 | 0.5536 | +0.0602 |
| Known acceptance | 95.08% | 94.80% | -0.28 pp |
| Unknown rejection | 19.35% | 23.00% | +3.65 pp |
| Score KS | 0.4173 | 0.4554 | +0.0381 |
| Mean score gap (unknown - known) | not recorded | 2.1761 | - |
| Unknown below known q95 | not recorded | 0.7770 | - |

The improvement reproduces the seed-123 direction: ranking, operating-point
rejection, known accuracy, and score separation all improve together. It is
therefore a credible training-side gain, but it does not solve the overlap:
77% of unknown test samples are still accepted at this operating point.

## Follow-up A: increase nnPU coefficient to 0.2

Only `alpha_discovery_uncertainty_pu` changes from the seed-42 nnPU treatment
(`0.1 -> 0.2`). The mixed-pool composition, prior, architecture, detector,
and calibration rule remain fixed.

| Metric | alpha=0.1 | alpha=0.2 | Change |
|---|---:|---:|---:|
| Best validation known accuracy | 62.87% | 47.57% | -15.30 pp |
| Test known accuracy | 63.08% | 48.75% | -14.33 pp |
| AUROC | 0.7906 | 0.7336 | -0.0570 |
| AUPR | 0.6757 | 0.6013 | -0.0744 |
| FPR95 | 0.6285 | 0.7302 | +0.1017 |
| OSCR | 0.5536 | 0.4165 | -0.1371 |
| Known acceptance | 94.80% | 93.95% | -0.85 pp |
| Unknown rejection | 23.00% | 18.55% | -4.45 pp |
| Score KS | 0.4554 | 0.3588 | -0.0966 |

Decision: increasing the coefficient is harmful. The uncertainty objective
overwhelms classification and representation learning at this strength; it is
not a valid way to force more rejection.

## Follow-up B: support-blended soft weights

Only the mixed-pool PU weighting source changes from `none` to
`support_blended`. This combines prototype distance, entropy/MSP and local
known-class kNN support when weighting unlabeled risk terms. It is a soft
weight, not a hard unknown pseudo-label.

| Metric | alpha=0.1, none | alpha=0.1, support-blended | Change |
|---|---:|---:|---:|
| Best validation known accuracy | 62.87% | 48.00% | -14.87 pp |
| Test known accuracy | 63.08% | 47.63% | -15.45 pp |
| AUROC | 0.7906 | 0.7389 | -0.0517 |
| AUPR | 0.6757 | 0.6125 | -0.0632 |
| FPR95 | 0.6285 | 0.7328 | +0.1043 |
| OSCR | 0.5536 | 0.4084 | -0.1452 |
| Known acceptance | 94.80% | 94.75% | -0.05 pp |
| Unknown rejection | 23.00% | 18.13% | -4.87 pp |
| Score KS | 0.4554 | 0.3566 | -0.0988 |

Decision: the support-blended weighting is also harmful in this implementation.
It does not provide a reliable class-conditional correction for the mixed-pool
risk and should remain an opt-in ablation, not part of the main recipe.

## Current decision and next direction

The strongest current CIFAR-100 recipe is CIFAR preprocessing plus unweighted
audited mixed-pool nnPU at `alpha=0.1`, supported by seed 123 and seed 42.
The evidence rejects two tempting shortcuts: simply increasing the PU weight
and adding local-support soft weights.

The remaining issue is representation-level overlap, not a missing threshold
trick. Existing detached-feature and longer-training audits also did not solve
it. The next useful experiment should therefore be a targeted representation
change with a validation/open-validation protocol, such as a class-conditional
boundary objective or a genuinely disjoint open-validation pool. Any new loss
must be compared against the exact alpha=0.1 CIFAR-profile nnPU baseline with
one changed factor and the full metric set above.

## Artifacts

- `runs/semantic_isolated_full_s42_cifar_profile_nnpu_student/`
- `runs/semantic_isolated_full_s42_cifar_profile_nnpu_discover/`
- `runs/semantic_isolated_full_s42_cifar_profile_nnpu_a02_student/`
- `runs/semantic_isolated_full_s42_cifar_profile_nnpu_a02_discover/`
- `runs/semantic_isolated_full_s42_cifar_profile_nnpu_support_blended_student/`
- `runs/semantic_isolated_full_s42_cifar_profile_nnpu_support_blended_discover/`
