# Uncertainty-head nnPU formula audit and warmup pilot (2026-10-11)

## Why this audit was necessary

The project bottleneck remains overlap between known and unknown feature/score
distributions. Before adding another representation loss, the mixed-pool
uncertainty risk itself was checked against its label convention.

In `nnpu_known_uncertainty_loss`, `u` is the novelty probability and

```text
z = log((1-u)/u)
```

is therefore the knownness logit. The labeled training samples are the
positive (known) distribution, while the discovery pool is an unlabeled
mixture with known prior `pi`. For logistic loss, the negative-class term is
`softplus(z)`. The mixed-pool decomposition is consequently

```text
pi * E_P[softplus(-z)]
+ E_U[softplus(z)]
- pi * E_P[softplus(z)]
```

with the existing non-negative-PU correction.

The former `nu_corrected` implementation used
`E_U[softplus(-z)]`, which is the positive-class loss under this convention.
That was a sign error. The implementation now uses the standard expression
for both `legacy` and `nu_corrected`. The two names remain available only
for compatibility with historical configurations; old results that claimed
`nu_corrected` was a different uncertainty-head objective require rechecking.

This audit applies only to the uncertainty-head training loss. The separate
post-hoc feature rejector has the opposite operational label convention
(unknown is the positive score), so its nnPU formula was not changed by this
patch.

## Controlled pilot

Fixed:

- CIFAR-100 random 60/40 split, seed 42;
- pretrained ResNet-34 teacher and ResNet-18 student;
- matched mixed pool of 400 samples with known prior 0.2;
- train/validation/test limits 1200/300/1000;
- two student epochs, CUDA, same teacher checkpoint;
- same detector: normalized entropy + Mahalanobis, validation-only 95% known
  coverage, clustering disabled.

Only the uncertainty-head risk schedule was varied.

| arm | AUROC | FPR95 | OSCR | known acceptance | unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct nnPU, legacy name | 0.6003 | 0.8843 | 0.1781 | 93.72% | 8.86% |
| audited formula, nu_corrected name | 0.6119 | 0.8760 | 0.1637 | 94.88% | 6.33% |
| audited formula + warmup/ramp | 0.5753 | 0.8959 | 0.1652 | 95.70% | 7.09% |

The small difference between the first two rows is not evidence that the
formula names define different objectives: the two training runs were seeded
independently by the training pipeline's data-loader/augmentation sequence,
and after the code audit the objectives are identical. The row is retained as
a paired smoke result, not as a claimed algorithmic gain.

The warmup/ramp pilot did not improve ranking under this short two-epoch budget.
It slightly increased the operating-point unknown rejection relative to the
audited direct run, but worsened AUROC, FPR95, and known utility. It is not a
default configuration.

## Full three-seed follow-up

After the formula audit, the direct mixed-pool uncertainty nnPU arm was rerun
on the full CIFAR-100 random 60/40 protocol for seeds 42, 123, and 3407.

Fixed:

- pretrained ResNet-34 teacher and ResNet-18 student;
- five student epochs;
- matched mixed pool, `mixed_known_pool_ratio=0.2`;
- `alpha_discovery_uncertainty_pu=0.1`;
- `discovery_uncertainty_known_prior=auto`;
- audited `discovery_uncertainty_pu_risk=nu_corrected` alias;
- same `normalized_entropy_mahalanobis` detector;
- validation-only 95% known-coverage threshold;
- full 10,000-image CIFAR-100 test split, clustering disabled.

The old treatment rows below used the former sign convention and should be
read only as historical/partially audited references.

| seed | arm | AUROC | AUPR | FPR95 | OSCR | known acc | known accept | unknown reject |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | baseline | 0.5866 | 0.4486 | 0.8667 | 0.2921 | 39.70% | 94.50% | 6.33% |
| 42 | old treatment | 0.6498 | 0.5189 | 0.8283 | 0.3735 | 48.20% | 95.45% | 9.45% |
| 42 | audited direct nnPU | 0.6477 | 0.5248 | 0.8295 | 0.3493 | 46.77% | 94.53% | 11.83% |
| 123 | baseline | 0.6077 | 0.4695 | 0.8497 | 0.3137 | 41.90% | 94.22% | 7.88% |
| 123 | old treatment | 0.6533 | 0.5322 | 0.8167 | 0.3728 | 48.03% | 94.90% | 12.23% |
| 123 | audited direct nnPU | 0.6710 | 0.5400 | 0.7953 | 0.4063 | 53.15% | 94.67% | 12.23% |
| 3407 | baseline | 0.5988 | 0.4688 | 0.8732 | 0.3098 | 42.25% | 95.12% | 7.13% |
| 3407 | old treatment | 0.6441 | 0.5168 | 0.8365 | 0.3692 | 48.10% | 95.38% | 9.65% |
| 3407 | audited direct nnPU | 0.6367 | 0.4980 | 0.8232 | 0.3747 | 50.28% | 94.43% | 9.53% |

Three-seed means:

| arm | AUROC | AUPR | FPR95 | OSCR | known acc | known accept | unknown reject |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.5977 | 0.4623 | 0.8632 | 0.3052 | 41.28% | 94.61% | 7.11% |
| old treatment | 0.6491 | 0.5226 | 0.8272 | 0.3718 | 48.11% | 95.24% | 10.44% |
| audited direct nnPU | 0.6518 | 0.5209 | 0.8160 | 0.3767 | 50.07% | 94.54% | 11.19% |

Interpretation:

- The audited direct nnPU arm consistently improves over the baseline on
  AUROC, FPR95, OSCR, known accuracy, and unknown rejection.
- The improvement is not only a threshold artifact: known acceptance remains
  close to the 95% target, and known classification accuracy improves.
- Compared with the old treatment, the audited arm has slightly better mean
  FPR95, OSCR, known accuracy, and unknown rejection, but seed 3407 has lower
  AUROC/AUPR. The method remains seed-sensitive.
- The core problem is still open: mean unknown rejection is only 11.19%, so
  most unknown test images are still accepted as known.

## Prior sensitivity pilot

A short seed-42 pilot checked whether the global known prior in the matched
mixed pool was the main cause of saturation. The experiment kept CIFAR-100
random 60/40, train/validation/test limits 1200/300/1000, matched pool size
400, two student epochs, the same teacher checkpoint, and the same
`normalized_entropy_mahalanobis` detector. Only the matched known ratio and
the automatically resolved nnPU known prior were changed.

| matched known ratio | AUROC | AUPR | FPR95 | OSCR | known acc | known accept | unknown reject |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.6075 | 0.4757 | 0.8876 | 0.1657 | 22.98% | 96.69% | 3.80% |
| 0.2 smoke reference | 0.6119 | n/a | 0.8760 | 0.1637 | n/a | 94.88% | 6.33% |
| 0.3 | 0.5520 | 0.4136 | 0.9190 | 0.1691 | 23.97% | 95.70% | 2.78% |

Both 0.1 and 0.3 are worse than the 0.2 smoke reference at the fixed operating
point. This suggests that blind global-prior sweeping is not the right next
direction. The next variant should make the PU weighting class-conditional or
semantic-distance-aware instead of applying one global prior to every mixed
sample.

## What this changes

- The uncertainty-head risk formula is now mathematically aligned with its
  documented known-positive labels.
- Existing historical runs using the old `nu_corrected` branch must be
  labeled as historical/partially audited rather than used as evidence that
  uncertainty nnPU solved feature overlap.
- The post-hoc rejector `nu_corrected` results remain a separate line of
  evidence and should not be conflated with the uncertainty-head audit.
- The core problem is still not solved: full-protocol unknown rejection improves
  from 7.11% to 11.19% on average, but detector scores still overlap heavily.

## Next controlled experiment

Do not add another score fusion or threshold sweep as the next primary action.
The next controlled experiments should isolate why the mixed-pool nnPU signal
helps but saturates at low unknown rejection:

1. check sensitivity to `mixed_known_pool_ratio` and the resolved known prior;
2. test class-conditional or semantic-distance-aware priors/weights rather than
   one global prior;
3. report feature-overlap diagnostics together with AUROC/FPR95/OSCR;
4. keep warmup/ramp out of the default path unless a full-budget isolated
   experiment is requested.

Use the same checkpoint-selection rule, independent calibration subset, same
detector, and at least three seeds. Report AUROC, FPR95, OSCR, known
acceptance/accuracy, unknown rejection, and feature-overlap statistics.

