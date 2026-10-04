# Full-data mixed nnPU recheck (seed 42, 2026-10-01)

## Purpose

This is the second independent full-data check of the mixed-pool nnPU
uncertainty objective. The goal is to test whether the positive result from
seed 123 repeats under an independently trained seed, rather than to tune a
threshold or select a favorable test subset.

## Controlled protocol

- CIFAR-100 random 60/40 split, seed 42;
- matching pretrained ResNet-34 teacher and pretrained ResNet-18 student;
- complete known training partition and complete 10,000-image open test set;
- five training epochs, image size 64, batch size 64, CUDA device;
- mixed discovery pool with requested known ratio 0.2 and measured prior
  `0.212598`;
- detection score `normalized_entropy_mahalanobis`, MC samples 4;
- threshold calibrated only from known validation samples at 95% target known
  coverage;
- clustering skipped so this comparison isolates unknown detection;
- control: `alpha_discovery_uncertainty_pu=0.0`;
- treatment: `alpha_discovery_uncertainty_pu=0.1`;
- all other training and detection options are identical.

The treatment log records a non-zero `discovery_uncertainty_pu` loss (for
example, approximately `0.2525` in epoch 3), so the tested objective really
participated in optimization.

## Results

| Metric | Full mixed baseline | Full mixed nnPU | Change |
| --- | ---: | ---: | ---: |
| Best validation known accuracy | 41.33% | 49.43% | +8.10 pp |
| AUROC | 0.5866 | 0.6498 | +0.0633 |
| AUPR | 0.4486 | 0.5189 | +0.0703 |
| FPR95 | 0.8667 | 0.8283 | -0.0383 |
| OSCR | 0.2921 | 0.3735 | +0.0814 |
| Known acceptance | 94.50% | 95.45% | +0.95 pp |
| Unknown rejection | 6.33% | 9.45% | +3.13 pp |
| Accepted-known accuracy | 42.01% | 50.50% | +8.49 pp |

The direction agrees with the seed 123 full-data recheck: ranking metrics,
OSCR, unknown rejection, and accepted-known accuracy all improve. Because the
two models use the same validation-calibrated operating rule, this is not
evidence obtained by moving the test threshold. It is still not a final
multi-seed claim: the sample of seeds is only two and the absolute unknown
rejection remains low.

## Decision

nnPU is currently the strongest mixed-pool uncertainty candidate and should be
kept as an optional treatment for the next confirmation. It should not yet be
made the only default or combined with Objectosphere, feature repulsion,
selective energy, or candidate gating, because that would lose one-factor
attribution and could hide the source of any change.

## Next checks

1. Run the same control/treatment pair with a third fixed seed and the
   matching teacher checkpoint.
2. Report mean and standard deviation across the three seeds for AUROC,
   FPR95, OSCR, known acceptance, unknown rejection, and accepted-known
   accuracy.
3. Re-enable clustering only after detection is confirmed, and report cluster
   purity/ACC, NMI, ARI, candidate purity, and estimated-K error separately.
4. Inspect score and feature overlap for all seeds. nnPU improves the
   uncertainty ranking, but it is not yet proof that the embedding space has
   separated.
