# Rejection geometry and score audit (2026-10-07)

## Question

The current bottleneck is overlap between known and unknown representations. This
round tested two known-only objectives on a separate rejection embedding, then
checked whether class-conditional Mahalanobis scoring could explain the remaining
error. No test labels were used for training, score fitting, or threshold fitting.

## Protocol for the rejection-branch pilot

- CIFAR-100 semantic-isolated 60/40 split;
- seeds 42 and 43;
- pretrained ResNet-34 teacher and ResNet-18 student;
- 1200/300/1000 train/validation/test limits;
- two student epochs, rejection embedding dimension 128;
- matched mixed discovery pool of 5400 samples with known prior 0.2;
- linear `nu_corrected` rejector on `rejection_embedding`;
- validation-only 95% known-coverage calibration;
- only the rejection-branch loss changed between paired arms.

## Known-only rejection geometry losses

`--alpha-rejection-supcon 0.1` applies supervised contrastive loss to the
independent rejection embedding. It was motivated by SupCon's labelled class
compactness, but does not use unknown pseudo-labels.

`--alpha-rejection-center 0.1` applies the existing batch center compactness
loss to that embedding. It is a lower-noise alternative to mixed-pool feature
repulsion, but it also has no unknown supervision.

### SupCon result

| Seed | Arm | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | no rejection SupCon | 0.6474 | 0.8220 | 0.0800 | 94.01% | 10.53% |
| 42 | rejection SupCon | 0.7317 | 0.6739 | 0.1626 | 93.18% | 18.30% |
| 43 | no rejection SupCon | 0.7287 | 0.7232 | 0.1367 | 95.30% | 15.59% |
| 43 | rejection SupCon | 0.7086 | 0.7299 | 0.0776 | 97.82% | 9.65% |

The improvement does not reproduce across seeds. The option remains an
ablation, not a default method.

### Center result

| Seed | Arm | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | no rejection center loss | 0.6474 | 0.8220 | 0.0800 | 94.01% | 10.53% |
| 42 | rejection center loss | 0.6757 | 0.8103 | 0.1026 | 94.34% | 12.53% |
| 43 | no rejection center loss | 0.7287 | 0.7232 | 0.1367 | 95.30% | 15.59% |
| 43 | rejection center loss | 0.6894 | 0.7668 | 0.1276 | 94.80% | 10.40% |

The center objective also fails to reproduce. It may compact known samples, but
known-only compactness is not sufficient to create a boundary around unknown
classes.

## Fixed-checkpoint score audit

The same ten-epoch semantic-isolated treatment checkpoint was evaluated with
the same known-coverage protocol while changing only the score:

| Score | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: |
| normalized entropy + shared Mahalanobis | 0.6699 | 0.8037 | 0.3789 | 94.01% | 9.52% |
| normalized entropy + diagonal Mahalanobis | 0.6924 | 0.7820 | 0.4051 | 96.01% | 9.77% |
| classwise Mahalanobis | 0.5997 | 0.9185 | 0.3234 | 95.17% | 5.76% |
| normalized entropy + classwise Mahalanobis | 0.6745 | 0.8153 | 0.3934 | 93.84% | 10.28% |

Diagonal covariance is the best score in this small audit, but its rejection
rate remains low. Class-conditional normalization does not solve the overlap.
This closes score-only tuning as the next main direction.

## Decision

- Keep `alpha_rejection_supcon` and `alpha_rejection_center` as opt-in
  ablations with default `0`.
- Do not claim either loss solved unknown detection.
- Do not promote classwise Mahalanobis to the default detector.
- The next algorithmic change must provide reliable unknown structure during
  representation learning, preferably with a stable target/refresh mechanism
  or a protocol-defined external OOD source, and must report feature overlap
  together with AUROC, FPR95, OSCR, known acceptance, and unknown rejection.
