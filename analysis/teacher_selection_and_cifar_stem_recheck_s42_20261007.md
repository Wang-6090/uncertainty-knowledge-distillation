# Teacher candidate weighting and CIFAR stem recheck (2026-10-07)

## Scope

This note records two controlled checks against the same core problem: known and
unknown representations remain overlapped, so the unknown rejection rate is low.
The checks were run after the student rejection-CE wiring fix. They are pilot
results, not multi-seed claims.

## 1. Frozen-teacher candidate weighting

The existing uncertainty-weighted feature-margin treatment uses the current
student uncertainty as a soft mixed-pool weight. This recheck asked whether a
frozen teacher would provide a cleaner target. The only code fix needed was to
allow `--discovery-selection-model teacher` together with `ema_*` feature-margin
weight sources; the previous validation incorrectly allowed only `ema`.

The protocol was CIFAR-100 semantic-isolated 60/40, seed 42, pretrained
ResNet-34/ResNet-18, 3 epochs, 1200/300/1000 train/validation/test limits,
matched mixed discovery pool of 5400, known prior 0.2, the existing uncertainty
KD + feature KD + SupCon + prototype losses, and the existing mixed-pool nnPU
term. Detection used the same support-augmented `nu_corrected` nnPU rejector
and validation-only 95% known-coverage calibration.

| Treatment | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: |
| Student uncertainty weight (existing) | 0.7469 | 0.7105 | 0.2372 | 96.01% | 23.31% |
| Frozen teacher uncertainty x MSP novelty | 0.7255 | 0.7155 | 0.2240 | 94.01% | 16.29% |
| Frozen teacher MSP novelty | 0.7331 | 0.7055 | 0.2411 | 94.68% | 13.78% |

The training terms were non-zero, so these are valid negative results rather
than disabled-path results. Replacing the student weight with frozen-teacher
weights did not improve the representation or operating point. The product
weight was especially conservative. This direction remains available as an
ablation but is not promoted.

## 2. CIFAR-style ResNet stem

The second check changed only the ResNet input stem: a `3x3, stride=1` first
convolution with no initial max-pooling, while keeping image size 64, split,
seed, training budget, losses, detector and threshold protocol fixed. This is a
small-resolution architecture check, not an OOD algorithm.

| Student arm | AUROC | FPR95 | OSCR | All-known accuracy | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default ImageNet stem pilot baseline | 0.7004 | 0.7770 | 0.1543 | 19.30% | 96.01% | 14.04% |
| CIFAR-style stem | 0.7199 | 0.7488 | 0.1151 | 14.64% | 94.18% | 18.55% |

The stem gives a small AUROC/FPR95 and unknown-rejection improvement, but the
known-class accuracy and OSCR are substantially worse. It does not provide a
reliable solution to the overlap problem and remains an opt-in structural
ablation. A longer or differently initialized stem experiment would be a new
protocol, not evidence that this pilot solved the issue.

## Code and evidence status

- `train.py` now accepts `teacher` as a valid frozen selection model for
  `ema_*` feature-margin weights.
- Regression tests increased to `182 passed`.
- The student rejection CE wiring fix is active; older rejection-branch
  checkpoints remain historical and must not be compared as strict baselines.
- Neither teacher weighting nor CIFAR stem is part of the default recipe.

## Decision

Stop tuning teacher candidate weights and stop treating the CIFAR stem as the
main direction. The next algorithmic work should improve the known-class
representation and mixed-pool novel supervision together, with a fixed detector
and matched known-coverage evaluation. In particular, a unified GCD-style
training objective needs a stable target/refresh protocol; adding another
post-hoc score or another candidate selector is unlikely to resolve the overlap.
