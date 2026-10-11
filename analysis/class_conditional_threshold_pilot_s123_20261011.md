# Class-conditional threshold pilot (2026-10-11)

## Purpose and protocol

The prior per-class error audit showed unknown false accepts are routed into
different known predictions (for example pickup truck to couch/tractor and
dolphin to shark). The code already had a validation-known-only
`class_conditional` threshold policy, but it had not been evaluated against
the same checkpoint and score with a fixed comparison protocol. This pilot
tests whether correcting per-predicted-class score-scale differences changes
the operating point usefully.

Only the threshold policy changes. The two fixed checkpoints are the audited
seed-123 CIFAR-100 semantic-isolated 60/40 unweighted nnPU 5-epoch and 10-epoch
students. Both runs use the complete data, `normalized_entropy_proto`, MC=4,
95th-percentile validation-known calibration, minimum 5 calibration samples
per predicted class, and no clustering. The class threshold for each predicted
class is estimated from known validation samples whose predicted class equals
that class; a global threshold is the fallback for undersupported classes.
Unknown test labels are used only for final metrics and classwise retrospective
diagnosis. No test labels or test-derived thresholds enter calibration.

The global reference is the existing 95%-known-coverage validation threshold
for each same checkpoint. Because the policies differ (one global coverage
quantile versus per-predicted-class 95th percentiles), actual test known
acceptance need not be equal. This is intentionally reported as a coverage vs.
unknown-rejection tradeoff, not as a matched-coverage superiority claim.

## Results

| Checkpoint and policy | AUROC | AUPR | FPR95 | OSCR | Test known acceptance | Test unknown rejection | Detection balanced accuracy* |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5 ep, global known-coverage | 0.7180 | 0.5938 | 0.7635 | 0.3914 | 95.92% | 13.53% | 54.72% |
| 5 ep, class-conditional | 0.7180 | 0.5938 | 0.7635 | 0.3914 | 94.30% | 16.45% | 55.38% |
| 10 ep, global known-coverage | 0.7156 | 0.5824 | 0.7685 | 0.4358 | 95.07% | 13.60% | 54.33% |
| 10 ep, class-conditional | 0.7156 | 0.5824 | 0.7685 | 0.4358 | 93.68% | 14.98% | 54.33% |

*Detection balanced accuracy is `0.5 * (test known acceptance + test unknown
rejection)`, a descriptive operating-point metric, not a replacement for
AUROC/AUPR/FPR95. It uses test labels only for reporting.

For the 5-epoch checkpoint, class-conditional thresholds trade 1.62 percentage
points of known acceptance for 2.92 points of unknown rejection, increasing
this balanced metric by 0.65 points. For the 10-epoch checkpoint, they trade
1.39 points of known acceptance for 1.38 points of rejection, leaving the
balanced metric unchanged. AUROC, AUPR, FPR95 and OSCR do not change: the score
vectors/model did not change, only the binary decision threshold did.

As an additional check, the class threshold was subtracted from each test
score using the model's predicted class, yielding a class-centered residual
score. This uses only the already calibrated known-validation thresholds and
does change the global ranking. It was worse on every checked seed/model:

| Run | Raw AUROC / AUPR / FPR95 | Class-centered residual AUROC / AUPR / FPR95 |
|---|---:|---:|
| seed 123, 5 ep | 0.7180 / 0.5938 / 0.7635 | 0.6986 / 0.5760 / 0.7960 |
| seed 123, 10 ep | 0.7156 / 0.5824 / 0.7685 | 0.7004 / 0.5685 / 0.7695 |
| seed 42, 5 ep | 0.7086 / 0.5811 / 0.7870 | 0.6939 / 0.5675 / 0.8158 |

Therefore, class-specific score centering is rejected: the per-class operating
threshold can slightly trade coverage for rejection, but using those offsets
to globally reorder examples damages open-set ranking.

Class-conditional thresholding does not eliminate difficult classes. Under the
5-epoch policy, television and pickup truck still have 97% and 96% false
acceptance; under the 10-epoch policy, pickup truck is 97%, bus 94%, and dolphin
94%. False accepts remain associated with plausible known predictions (e.g.
pickup truck to couch, dolphin to shark). This is consistent with residual
feature overlap, not merely a global score-scale problem.

## Second-seed check

The same 5-epoch threshold comparison was repeated on the existing seed-42
semantic-isolated checkpoint, without changing score mode or class percentile.

| Seed-42 policy | AUROC | AUPR | FPR95 | OSCR | Test known acceptance | Test unknown rejection | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Global known-coverage | 0.7086 | 0.5811 | 0.7870 | 0.3799 | 94.43% | 14.95% | 54.69% |
| Class-conditional | 0.7086 | 0.5811 | 0.7870 | 0.3799 | 93.48% | 17.28% | 55.38% |

Across seeds 42 and 123, the 5-epoch class-conditional policy increased
unknown rejection by 2.33 and 2.92 percentage points, respectively, while
reducing known acceptance by 0.95 and 1.62 points. The descriptive balanced
accuracy increased by 0.69 and 0.65 points. This is directionally consistent,
but still a small threshold-only effect; the 10-epoch seed-123 result had no
balanced-accuracy gain. The second seed strengthens the case for keeping this
as an optional operating-point baseline, not for treating it as the solution
to representation overlap.

## Decision and limitations

This is a small optional operating-point policy, not a core algorithmic fix.
The 5-epoch result is mildly encouraging for a balanced decision objective and
the operating-point direction replicated on seed 42. However, it remains a
small effect and the 10-epoch result does not replicate the balanced-accuracy
gain. Class-specific quantiles can be noisy and test coverage shifted from the
validation target. Do not claim improved ranking, learned representation, or
robust generalization. Keep the policy available as an ablation; do not make it
the default or tune per-class thresholds using test errors.

Next, compare it on another predeclared seed using the identical percentile,
minimum-count rule, checkpoint protocol, and score mode. Only retain it as a
secondary operating policy if it improves rejection at a tolerable known
coverage cost across seeds. Continue the core investigation through train/val
representation and confusion diagnostics rather than repeatedly tuning test
operating points.

## Artifacts

- 5-epoch class-conditional run: `runs/class_conditional_5epoch_s123/`
- 10-epoch class-conditional run: `runs/class_conditional_10epoch_s123/`
- Seed-42 5-epoch class-conditional run: `runs/class_conditional_5epoch_s42/`
- Per-class comparison: `analysis/classwise_open_errors_class_conditional_5vs10_s123_20261011.json`
- Seed-42 comparison: `analysis/classwise_open_errors_class_conditional_s42_20261011.json`
- Residual score metrics above are computed from the corresponding saved test scores, predicted classes, validation-calibrated threshold vectors, and known/unknown flags.
- The classwise audit script now handles scalar and per-predicted-class threshold arrays.
