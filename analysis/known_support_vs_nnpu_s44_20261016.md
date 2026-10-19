# Known-support score versus full-support nnPU rejector (2026-10-16)

## Purpose

The all-support experiment improved the nnPU rejector, but that improvement
could have been incorrectly attributed to the class-conditional support score
itself. This diagnostic kept the same student checkpoint, data protocol and
95% known-coverage calibration, and replaced the learned rejector score with
the known-only class-conditional support score.

## Result

| detector | AUROC | FPR95 | OSCR | known accept | unknown reject | KS | unknown below known q95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all-support `support_augmented` nnPU | 0.90342 | 0.3590 | 0.66667 | 95.37% | 49.48% | 0.6512 | 48.40% |
| known-only class support | 0.76023 | 0.6763 | 0.58271 | 93.82% | 23.23% | 0.3876 | 79.45% |

## Interpretation

The class-conditional support distance alone is not sufficient for this
checkpoint: it produces substantially more overlap. Therefore the current
gain is not just a threshold or prototype-distance effect. The mixed-pool
nnPU rejector is learning a useful direction from the support-augmented
features, while `known_support_score` should remain an input feature or
diagnostic rather than the final detector by itself.

The run also exposed a configuration issue: a `known_support` score does not
use rejector features, so it must be paired with `rejector_feature_mode`
`embedding` in the command/config. The validation check correctly rejects an
inconsistent combination instead of silently ignoring the feature mode.

The next controlled test is to normalize the high-dimensional class-logit
block inside `rejection_balanced_support_augmented`, keeping the all-support
nnPU protocol fixed. This tests whether the current rejector is dominated by
logit scale or feature count.
