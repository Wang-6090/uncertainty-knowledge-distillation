# CIFAR transform profile strict recheck, seed 123

## Purpose

This recheck tests whether the CIFAR-100 preprocessing protocol is a major
cause of the known/unknown feature and score overlap. Earlier historical
comparisons were partly confounded by discovery-pool and PU settings, so this
run uses a stricter paired setup.

## Controlled protocol

- Dataset: CIFAR-100 semantic-isolated 60/40 split.
- Seed: 123.
- Data budget: full train/validation/test.
- Teacher/student: pretrained ResNet-34 teacher and ResNet-18 student.
- Training budget: 5 epochs.
- Student losses: uncertainty KD, supervised contrastive, feature KD, and
  prototype loss kept identical.
- Discovery pool: matched pool, size 5400, known prior 0.2; PU discovery
  losses disabled for this comparison.
- Detector: `normalized_entropy_min_class_knn`, k=10, MC=4.
- Threshold: validation-only 95% known coverage.
- Clustering: skipped.
- Only intended model/data difference: `transform_profile` and the matching
  teacher/student checkpoints trained under that profile.

The config audit confirmed that the two student training configs differ only
by `transform_profile`:

```
PAIR_STATUS: non-strict (1 configuration differences)
- transform_profile: left="imagenet" | right="cifar"
```

The discovery config audit showed only output/checkpoint paths and
`transform_profile` differed.

## Results

| Profile | teacher val acc | student val acc | test known acc | AUROC | AUPR | FPR95 | OSCR | known accept | unknown reject | score KS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ImageNet profile | 42.00% | 42.00% | 42.37% | 0.6789 | 0.5412 | 0.7643 | 0.3511 | 95.35% | 10.73% | 0.2578 |
| CIFAR profile | 58.70% | 54.83% | 54.53% | 0.7550 | 0.6314 | 0.6793 | 0.4726 | 95.20% | 16.08% | 0.3972 |

Additional score-distribution evidence:

- Mean unknown-minus-known score gap improves from 1.1485 to 1.7302.
- Median gap improves from 1.3068 to 2.1207.
- Unknown-below-known-q95 overlap improves from 0.8838 to 0.8303.
- Known-above-unknown-q05 overlap improves from 0.7643 to 0.6795.

## Conclusion

CIFAR-style preprocessing is now a validated high-value direction. It improves
known-class representation and known/unknown score separation on both seed 42
and this stricter seed 123 comparison, while keeping known acceptance around
95%. It does not solve open-set detection by itself, because most unknown
samples are still accepted at the 95% known-coverage point, but it addresses a
real upstream representation problem.

Because of this, the CLI now defaults to `--transform-profile auto`: CIFAR-100
uses the CIFAR profile by default, while non-CIFAR datasets keep ImageNet-style
preprocessing. Historical runs can still be reproduced by explicitly passing
`--transform-profile imagenet`.

## Next steps

1. Treat CIFAR profile as the recommended CIFAR-100 protocol.
2. Re-run the most promising open-set training recipes under this protocol
   before judging them, because older ImageNet-profile results may underestimate
   representation quality.
3. Continue reporting AUROC, AUPR, FPR95, OSCR, known accuracy, known
   acceptance, unknown rejection, KS distance, and overlap diagnostics together.
4. Do not tune thresholds or fusion weights on test labels; test labels remain
   for final reporting and post-hoc error analysis only.
