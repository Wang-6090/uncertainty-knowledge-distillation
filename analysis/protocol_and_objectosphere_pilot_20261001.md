# Protocol audit and Objectosphere pilot (2026-10-01)

## 1. Protocol correction

The mixed discovery pool is built by concatenating a reserved known subset and
the withheld-class pool, then `limit_discovery` samples from the concatenation.
Therefore the requested `mixed_known_pool_ratio` is not necessarily the actual
known fraction seen by the PU objective. The code now exposes
`known_proportion(dataset)` and supports:

```text
--discovery-uncertainty-known-prior auto
```

`auto` measures the composition after all splits and limits, records it in
`config.json`, and passes that resolved value to the nnPU loss. An explicit
numeric value remains available for genuinely unlabeled deployment data.

The toy smoke test made the issue visible: with `limit_discovery=100`, the
actual known fraction was `0.01`, although the requested pool construction
ratio was `0.2`. The CIFAR-100 protocol with `limit_discovery=1200` had an
actual fraction of `0.210833`.

## 2. nnPU prior A/B

Only the prior source changed. Both runs used CIFAR-100 60/40, seed 42, the
same ResNet-34 teacher, ResNet-18 student, 1200/300/1000 known train/val/test
limits, 1200 mixed discovery samples, three epochs, GPU, PU weight `0.1`, and
the same `normalized_entropy_mahalanobis` detector at known coverage 95%.

| Metric | Explicit prior 0.2 | Auto prior 0.210833 | Change |
| --- | ---: | ---: | ---: |
| AUROC | 0.5337 | 0.5492 | +0.0154 |
| FPR95 | 0.9391 | 0.9112 | -0.0280 |
| OSCR | 0.1509 | 0.1499 | -0.0010 |
| Known acceptance | 0.9490 | 0.9474 | -0.0016 |
| Unknown rejection | 0.0536 | 0.0434 | -0.0102 |
| Known accuracy (all known) | 0.2220 | 0.2237 | +0.0016 |

The corrected prior has a small ranking signal but does not improve the
operating-point unknown rejection. It is retained as an experimental protocol
fix, not as a solution to feature overlap.

## 3. Objectosphere-style representation pilot

Following the Objectosphere idea, the optional pure-unknown training loss is

```text
L = ReLU(radius - ||f_known||)^2 / radius^2
    + lambda_unknown * ||f_unknown||^2 / radius^2
```

It is implemented as `--alpha-discovery-objectosphere` and is rejected for a
mixed pool. The new `feature_norm` and
`normalized_entropy_feature_norm` scores expose the radial signal during
evaluation. Defaults keep the historical behavior unchanged.

The paired CIFAR-100 pilot used seed 42, the same teacher/student and
1200/300/1000 limits, 1200 pure-unknown discovery samples, three epochs,
`known_radius=10`, `unknown_weight=1`, and compared `alpha=0` with `alpha=0.01`.

| Detector / metric | Baseline | Objectosphere | Change |
| --- | ---: | ---: | ---: |
| `feature_norm` AUROC | 0.4998 | 0.5101 | +0.0103 |
| `feature_norm` unknown rejection | 0.0689 | 0.0791 | +0.0102 |
| `normalized_entropy_feature_norm` AUROC | 0.4964 | 0.5249 | +0.0285 |
| `normalized_entropy_feature_norm` OSCR | 0.1132 | 0.1235 | +0.0104 |
| `normalized_entropy_feature_norm` unknown rejection | 0.0867 | 0.0561 | -0.0306 |
| Main `normalized_entropy_mahalanobis` AUROC | 0.4901 | 0.5870 | +0.0968 |
| Main `normalized_entropy_mahalanobis` FPR95 | 0.9589 | 0.8865 | -0.0724 |
| Main `normalized_entropy_mahalanobis` unknown rejection | 0.0510 | 0.0281 | -0.0229 |

This is promising for ranking and feature-space diagnostics, but the calibrated
unknown rejection point is still weak and the result is one seed with a small
training budget. The method should remain an optional ablation until at least
three seeds and full-data training are run. Do not raise its weight based only
on AUROC/FPR95.

## Decision

- Keep automatic mixed-pool prior resolution and its metadata logging.
- Keep Objectosphere as a pure-unknown, opt-in representation ablation.
- Do not change the default detector or default training losses.
- Next verification should use matched known coverage, full test data, and at
  least three seeds; report score-distribution overlap and unknown rejection in
  addition to AUROC/FPR95.
