# Independent rejection branch plus uncertainty-PU recheck (2026-10-07)

## Purpose

This paired experiment tested whether adding mixed-pool uncertainty nnPU
training to an independently parameterized rejection representation improves
known/unknown separation. The purpose was to test a real training signal, not
to tune the final threshold.

## Controlled protocol

- CIFAR-100 semantic-isolated 60/40 split, seed 42
- full train/validation/test data
- pretrained ResNet-34 teacher and ResNet-18 student
- identical teacher checkpoint, batch size 64, three student epochs, and data order
- matched mixed discovery pool of 5,400 images with known prior 0.2
- student has `rejection_feature_dim=128` and the independent rejection branch
- same linear `nu_corrected` nnPU post-hoc rejector
- same `rejection_embedding` rejector input, MC=4, validation-only 95%
  known-coverage calibration, and no clustering

The only student-training difference was:

| Arm | `alpha_discovery_uncertainty_pu` | PU risk |
| --- | ---: | --- |
| independent-branch baseline | 0.0 | disabled |
| treatment | 0.1 | `nu_corrected` |

The training logs confirmed that the treatment was active: its
`discovery_uncertainty_pu` loss was non-zero, while the baseline value was
exactly zero.

## Results

| Metric | Baseline | Independent branch + PU | Difference |
| --- | ---: | ---: | ---: |
| AUROC | 0.7627 | 0.7178 | -0.0449 |
| AUPR | 0.6280 | 0.5803 | -0.0477 |
| FPR95 | 0.6327 | 0.7045 | +0.0718 |
| OSCR | 0.3485 | 0.3141 | -0.0344 |
| Known acceptance | 94.48% | 93.83% | -0.65 pp |
| Unknown rejection | 18.63% | 15.33% | -3.30 pp |
| Accepted-known accuracy | 42.90% | 41.10% | -1.80 pp |

## Decision

The treatment is a clear negative result under this protocol. It worsens both
threshold-free ranking metrics and the fixed 95% known-coverage operating
point. The result is not explained by rejecting more known samples: unknown
rejection also decreases, and accepted-known accuracy decreases.

The likely failure mode is that the mixed-pool uncertainty target is an
unreliable pseudo-label for the rejection embedding. Applying the PU loss to
that branch therefore distorts a representation that already has useful
ranking information. We stop tuning this PU term and keep it opt-in; it is not
promoted to the main recipe.

Artifacts:

- baseline student: `runs/semantic_isolated_rejection_branch_s42_base/`
- treatment student: `runs/semantic_isolated_rejection_branch_s42_nnpu/`
- baseline report: `runs/semantic_isolated_rejection_branch_s42_base_rejector/`
- treatment report: `runs/semantic_isolated_rejection_branch_s42_nnpu_rejector/`

The next controlled check reuses the same two checkpoints and compares
`rejection_embedding` with the already implemented uncertainty-augmented
rejector features. This separates the question “is the representation bad?”
from “are we failing to use the available uncertainty signals?”.
