# OpenMax fixed-checkpoint recheck (seed 42, 2026-10-01)

## Purpose

OpenMax is a literature-based open-set detector that fits a Weibull tail to
known-class activation distances. This experiment asks whether it can improve
unknown ranking without changing the learned representation. It is therefore
a detector-only comparison, not a training ablation.

## Controlled protocol

The same `candidate_sep_student_s42_full/student.pt` checkpoint is evaluated
with the existing primary score and with `--score-mode openmax`. Both runs use
the CIFAR-100 random 60/40 split, the complete known training partition for
the support/statistics bank, 300 validation samples, 1000 test samples, MC=4,
and a threshold calibrated only on known validation data at the default global
95th percentile. No test labels are used for score or threshold selection.

## Results

| Metric | Normalized entropy + min-class kNN | OpenMax-Weibull | OpenMax - primary |
| --- | ---: | ---: | ---: |
| AUROC | 0.6709 | 0.5668 | -0.1040 |
| AUPR | 0.5639 | 0.4557 | -0.1081 |
| FPR95 | 0.8132 | 0.9273 | +0.1140 |
| OSCR | 0.3702 | 0.2823 | -0.0879 |
| Known acceptance | 95.87% | 96.03% | +0.16 pp |
| Raw unknown rejection | 15.44% | 4.81% | -10.63 pp |

At a retrospective approximately 95% test-known coverage, which is a
test-label diagnostic only, unknown rejection is `16.71%` for the primary
score and `8.86%` for OpenMax. Accepted-known classification accuracy is
`47.74%` and `45.47%`, respectively.

## Decision

OpenMax does not improve the current representation. The result is consistent
with the earlier observation that the bottleneck is known/unknown feature
overlap rather than the choice of a post-hoc distance or tail model. Keep
OpenMax as a literature-based detector baseline and stop tuning it for the
mainline. Future gains must come from representation learning or a correctly
isolated mixed-pool GCD objective.

