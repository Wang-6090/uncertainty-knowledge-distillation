# Proxy-contrastive representation pilot (2026-10-13)

## Question

The detector score is not the main bottleneck: known and unknown examples still
overlap in the shared feature space. This pilot therefore changed the training
representation objective rather than the threshold or detector. The tested
term is `proxy_contrastive_loss`, a normalized-softmax/Proxy-NCA-style loss
that pulls a labeled feature toward its classifier-weight proxy and pushes it
away from the other known-class proxies. It does not require a same-class
positive in the mini-batch.

## Controlled protocol

For seeds 123 and 44, each treatment was retrained from the same teacher and
the same semantic-isolated CIFAR-100 60/40 protocol. The baseline used
`alpha_proxy=0`; the treatment used `alpha_proxy=0.05`. All other student
losses, optimizer settings, five epochs, discovery pool, kNN bank, MC=4,
score fusion, and validation-only 95%-known-coverage calibration were held
fixed. The official test labels were used only for retrospective reporting and
paired bootstrap diagnostics, never for training or selection.

## Results

| seed | arm | AUROC | AUPR | FPR95 | OSCR | known acc | known accept | unknown reject | KS | unknown below known-q95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 123 | baseline | 0.8167 | 0.7237 | 0.5865 | 0.5560 | 61.88% | 95.20% | 28.58% | 0.4939 | 70.98% |
| 123 | proxy 0.05 | 0.8200 | 0.7253 | 0.5907 | 0.5628 | 62.32% | 95.10% | 26.78% | 0.5046 | 72.78% |
| 44 | baseline | 0.8114 | 0.7073 | 0.5827 | 0.5276 | 58.87% | 94.70% | 27.40% | 0.4868 | 73.60% |
| 44 | proxy 0.05 | 0.8107 | 0.7091 | 0.5873 | 0.5260 | 58.72% | 95.22% | 25.18% | 0.4885 | 73.90% |

The seed-123 gain in AUROC is only `+0.0033`; its exact-95%-known-coverage
unknown-rejection difference is `-1.84` percentage points, with bootstrap CI
`[-3.43, -0.10]` pp. The seed-44 AUROC difference is `-0.0006` with CI
`[-0.0032, +0.0019]`, and unknown-rejection difference is `-0.15` pp with CI
`[-2.03, +1.78]` pp. The paired bootstrap files are
[`proxy_representation_bootstrap_s123.json`](proxy_representation_bootstrap_s123.json)
and [`proxy_representation_bootstrap_s44.json`](proxy_representation_bootstrap_s44.json).

Validation geometry gives a similar warning: seed 123 within-class mean
distance changes `0.1972 -> 0.1958`, but q95 changes `0.3565 -> 0.3601`; seed
44 changes are `0.2045 -> 0.2017` and `0.3672 -> 0.3644`. These small and
inconsistent changes do not demonstrate a reliable reduction of unknown
overlap.

## Decision

This is a valid negative/mixed result, not a solution to the core problem.
Proxy contrastive learning may slightly improve known classification or
ranking for one seed, but it does not consistently improve FPR95, OSCR,
unknown rejection, KS, or score overlap. Keep the implementation and the
`alpha_proxy` switch for controlled ablation, but keep the default at zero and
do not combine it into the main method yet.

The result also suggests that simply adding another known-class proxy loss is
not enough: the loss has no explicit unknown boundary and uses the same known
classifier proxies that already dominate the open-set decision. The next
training intervention should provide a separately controlled boundary signal
(for example, a disjoint open-validation/training outlier protocol or a
carefully routed energy/objectosphere objective), while preserving known-class
classification. It must be compared against the matched baseline on both
seeds and judged by ranking, FPR95, OSCR, KS/overlap, and known utility—not by
unknown rejection at a separately shifted threshold.

