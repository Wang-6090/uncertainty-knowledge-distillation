# All-class local-support kNN score pilot (2026-10-01)

## Motivation

The previous predicted-class kNN score measured distance only to the known
class selected by the classifier. An unknown sample can be confidently
misclassified into class A while its feature is close to class B. This pilot
adds a class-agnostic local-support statistic: compute the kNN distance inside
each known class and use the minimum distance across all known classes. The
implementation also supports per-class support-radius normalization, but the
raw distance is the version evaluated here.

This is a detection-side ablation. It does not change the checkpoint,
training loss, data split, or threshold calibration protocol.

## Protocol

- CIFAR-100 semantic-hard 60/40 split;
- ResNet-18 student checkpoints trained with the existing hard KNN
  warm-up/ramp configuration;
- complete known training partition and complete 10,000-image test set;
- MC dropout samples = 4;
- mixed discovery pool, limit 1200;
- disjoint validation calibration ratio = 0.5;
- threshold calibrated from known validation samples at 95% known coverage;
- score form: normalized predictive entropy + one kNN distance signal;
- clustering disabled so the comparison concerns only unknown detection.

The only score change is:

```text
old: predicted-class kNN distance
new: min over known classes of class-conditional kNN distance
```

## Results

| Seed | Score | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection | Accepted-known accuracy |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 43 | predicted-class kNN | 0.5991 | 0.8582 | 0.3351 | 0.9385 | 0.0938 | 0.4729 |
| 43 | all-class minimum raw kNN | 0.6042 | 0.8560 | 0.3405 | 0.9070 | 0.1468 | 0.4859 |
| 44 | predicted-class kNN | 0.6016 | 0.8357 | 0.3475 | 0.9510 | 0.0700 | 0.4791 |
| 44 | all-class minimum raw kNN | 0.6061 | 0.8315 | 0.3528 | 0.9418 | 0.0878 | 0.4843 |

The direction is consistent on two seeds: AUROC, FPR95, OSCR, and unknown
rejection improve. The gain is modest, and seed 43 loses more known coverage
than seed 44. The seed-43 operating-point increase in unknown rejection is
partly affected by a lower validation-calibrated threshold, so the method must
not be described as a large separation improvement.

At descriptive test-set quantiles, where the threshold is recomputed only for
diagnosis, the unknown-rejection improvement is much smaller. This confirms
that calibration drift and score scale still matter.

## Decision

- Keep `normalized_entropy_min_class_knn` as an optional detector and include
  it in automatic score candidates when `--knn-ood` is enabled.
- Do not replace the default detector yet.
- The result supports the hypothesis that conditioning kNN distance only on the
  predicted class misses some unknown samples.
- The residual overlap remains large; detection-side score engineering alone
  is unlikely to solve it. The next major direction should be training-time
  known/unknown representation separation or a dedicated open-set head.

