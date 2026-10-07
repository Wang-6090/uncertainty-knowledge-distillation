# Mixed-pool nnPU risk formula recheck (2026-10-05)

## Objective

The current bottleneck is still overlap between known and unknown feature or
score distributions. This experiment tests the rejector objective itself,
without changing the student checkpoint, feature representation, mixed pool,
threshold policy, or test protocol.

The existing `fit_nnpu_feature_rejector` used the historical risk term

```text
prior * known_as_unknown
  + relu((unlabeled_as_unknown - prior * known_as_known) / (1 - prior))
```

For a mixed pool in which known samples are reliable negatives and unknown
samples are the positive class, the direct negative-unlabeled decomposition is

```text
prior * known_as_unknown
  + relu(unlabeled_as_unknown - prior * known_as_known)
```

This is implemented as the opt-in `--rejector-nnpu-risk nu_corrected`; the
default `legacy` mode is unchanged for backward compatibility. The idea is
related to non-negative PU/NU risk estimation (du Plessis et al.; Kiryo et
al.), but this small linear rejector is not claimed to reproduce those papers'
full algorithms.

## Controlled protocol

- CIFAR-100 random 60/40 split, seeds 42, 123, and 3407;
- matching full-data mixed-pool nnPU student checkpoint for each seed;
- ResNet-18 frozen student features and class-wise known-support score;
- mixed pool with measured known prior `0.2125984252`;
- support quantile `0.95`, known-coverage threshold `0.95`;
- MC samples `4` for the detector; clustering skipped;
- only the rejector risk formula changes.

## Results

| Seed | Risk | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | legacy | 0.70315 | 0.74517 | 0.34015 | 94.77% | 15.83% |
| 42 | nu_corrected | 0.70697 | 0.74517 | 0.34184 | 94.77% | 16.40% |
| 123 | legacy | 0.71436 | 0.73250 | 0.35983 | 95.90% | 14.18% |
| 123 | nu_corrected | 0.71811 | 0.73167 | 0.36119 | 95.50% | 14.95% |
| 3407 | legacy | 0.71625 | 0.74100 | 0.36245 | 94.53% | 18.58% |
| 3407 | nu_corrected | 0.71727 | 0.73833 | 0.36133 | 94.80% | 18.35% |
| **mean** | **legacy** | **0.71125** | **0.73956** | **0.35414** | **95.07%** | **16.19%** |
| **mean** | **nu_corrected** | **0.71411** | **0.73839** | **0.35478** | **95.02%** | **16.57%** |

## Interpretation

The corrected risk improves mean AUROC by `+0.00286`, mean FPR95 by
`-0.00117`, mean OSCR by `+0.00064`, and mean unknown rejection by `+0.38`
percentage points. Known acceptance changes by only `-0.04` percentage points
on average. The AUROC and FPR95 direction is consistent in all three seeds;
OSCR and operating-point unknown rejection are mixed in seed 3407.

This is a useful implementation-level improvement, not a solution to the
feature-overlap problem. The absolute unknown rejection remains only `16.57%`
at the 95% known-coverage operating point. The corrected mode should therefore
be retained as an ablation/current candidate, while the next major effort must
still target representation learning or a better calibrated mixed-pool
rejector. Threshold-only tuning is not justified by this result.

## Interaction check: corrected risk plus MC uncertainty features

The previously weaker `support_uncertainty_augmented` representation was
retested with the corrected risk formula on seed 42:

| Features and risk | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: |
| support + corrected risk | 0.70697 | 0.74517 | 0.34184 | 94.77% | 16.40% |
| support + MC uncertainty + corrected risk | 0.70482 | 0.74917 | 0.34007 | 94.52% | 16.60% |

The extra MC uncertainty coordinates do not complement the corrected risk:
they worsen ranking and known utility while only increasing operating-point
unknown rejection by 0.20 percentage points. This combination is rejected.

## Outputs

- `runs/posthoc_rejector_nnpu_support_uncertainty_s42_recheck/`
- `runs/posthoc_rejector_nnpu_support_nu_corrected_s42/`
- `runs/posthoc_rejector_nnpu_support_nu_corrected_s123/`
- `runs/posthoc_rejector_nnpu_support_nu_corrected_s3407/`
- `runs/posthoc_rejector_nnpu_support_uncertainty_nu_corrected_s42/`
- `runs/posthoc_rejector_nnpu_support_nu_corrected_mlp_s42/`
- `runs/posthoc_rejector_nnpu_support_nu_corrected_mlp_s123/`
- `runs/posthoc_rejector_nnpu_support_nu_corrected_mlp_s3407/`

## Nonlinear MLP nnPU follow-up

A small two-hidden-layer MLP (64/32 units) was added as an opt-in rejector
family; the historical linear model remains the default. It uses the same
standardization and corrected risk as the linear comparison. Full-data runs
use matching seeds/checkpoints and otherwise identical mixed-pool protocol.

| Seed | Model | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection | Accepted-known accuracy |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | corrected linear | 0.70697 | 0.74517 | 0.34184 | 94.77% | 16.40% | 43.19% |
| 42 | corrected MLP | 0.72431 | 0.71133 | 0.34582 | 95.82% | 15.38% | 42.67% |
| 123 | corrected linear | 0.71811 | 0.73167 | 0.36119 | 95.50% | 14.95% | 45.17% |
| 123 | corrected MLP | 0.74024 | 0.71433 | 0.36665 | 94.55% | 21.53% | 45.36% |
| 3407 | corrected linear | 0.71727 | 0.73833 | 0.36133 | 94.80% | 18.35% | 45.31% |
| 3407 | corrected MLP | 0.50472 | 0.93600 | 0.22417 | 100.00% | 0.00% | 43.78% |

The MLP is not reliable: it improves ranking for seeds 42 and 123, but
collapses to near-chance AUROC and accepts every test sample for seed 3407.
Its three-seed means are worse than the corrected linear rejector on AUROC,
FPR95, OSCR, and operating-point unknown rejection. Therefore do not use the
single MLP as a default or claim that nonlinearity solved overlap. The
variation points to optimization/variance sensitivity in this small-sample
mixed-pool risk fit; a future ensemble or regularized model would need a
predeclared, validation-only selection protocol before any test comparison.

| Three-seed mean | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: |
| corrected linear | 0.71411 | 0.73839 | 0.35478 | 95.02% | 16.57% |
| corrected MLP | 0.65642 | 0.78722 | 0.31221 | 96.79% | 12.30% |
