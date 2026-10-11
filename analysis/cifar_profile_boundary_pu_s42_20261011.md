# CIFAR profile + nnPU + PU boundary loss (seed 42, 2026-10-11)

## Purpose

This controlled experiment tests whether the existing class-conditional
prototype/local-centroid boundary objective reduces the remaining known/unknown
feature overlap when corrected for a mixed discovery pool. The boundary term is
implemented by `pu_unknown_feature_boundary_loss`; it does not assign every
mixed-pool image a hard unknown label.

## Protocol

- CIFAR-100 semantic-isolated 60/40 split, seed 42, CIFAR transform profile.
- Pretrained ResNet-34 teacher and ResNet-18 student, five epochs, full data.
- Matched mixed discovery pool of 5,400 samples with known prior 0.2.
- Baseline and treatment both keep feature KD, prototype loss, and audited
  uncertainty nnPU (`alpha_discovery_uncertainty_pu=0.1`).
- Detector: `normalized_entropy_min_class_knn`, k=10, MC=4.
- Threshold: known-validation-only 95% known coverage.
- The only training change is `alpha_discovery_boundary=0.1` with
  `discovery_boundary_mode=pu`; test labels are used only for final metrics.

## Results

| Metric | nnPU baseline | + PU boundary | Change |
|---|---:|---:|---:|
| AUROC | 0.7906 | 0.7829 | -0.0077 |
| AUPR | 0.6757 | 0.6719 | -0.0038 |
| FPR95 | 0.6285 | 0.6570 | +0.0285 |
| OSCR | 0.5536 | 0.5387 | -0.0149 |
| Known accuracy | 63.08% | 61.65% | -1.43 pp |
| Known acceptance | 94.80% | 94.42% | -0.38 pp |
| Unknown rejection | 23.00% | 23.30% | +0.30 pp |
| Score KS | 0.4554 | 0.4412 | -0.0142 |
| Mean score gap | 2.1761 | 2.1046 | -0.0715 |

## Decision

The boundary loss is active, but it does not improve the core separation
problem. The tiny increase in operating-point unknown rejection is outweighed
by worse AUROC, FPR95, OSCR, known accuracy, score KS, and mean score gap. It
is therefore not promoted to the main recipe and should remain an ablation.

This supports the current diagnosis: generic repulsion from global
prototypes/local centroids is not enough for unknowns that are genuinely close
to known visual support. The next training-side test should use a
class-conditional angular margin on labelled known samples, with the exact
nnPU/CIFAR baseline otherwise unchanged.

Artifact: `runs/semantic_isolated_full_s42_cifar_profile_boundary_pu_student/`

