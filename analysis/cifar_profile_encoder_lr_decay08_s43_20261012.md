# Layer-wise encoder learning-rate decay (seed 43, 2026-10-12)

## Purpose

The three-seed paired experiments supported a uniform `encoder_lr_scale=0.1`.
This experiment tested whether protecting shallow pretrained features more
strongly, while allowing deeper stages to adapt, would further reduce the
known/unknown overlap.

## Protocol

- CIFAR-100 semantic-isolated 60/40 split, seed 43, CIFAR transforms.
- ResNet-34 teacher and ResNet-18 student, full data, five epochs, batch size 64.
- Matched mixed discovery pool of 5,400 samples, known prior 0.2.
- Feature KD, prototype loss, and audited uncertainty nnPU at alpha 0.1.
- Detector: `normalized_entropy_min_class_knn`, k=10, MC=4.
- Threshold: validation-only 95% known coverage.
- The only training change is `encoder_lr_decay=0.8`; both arms use
  `encoder_lr_scale=0.1`. `decay=1.0` is the paired baseline.

## Results

| Metric | scale 0.1, decay 1.0 | scale 0.1, decay 0.8 | Change |
|---|---:|---:|---:|
| AUROC | 0.8228 | 0.8223 | -0.0006 |
| AUPR | 0.7241 | 0.7257 | +0.0016 |
| FPR95 | 0.5560 | 0.5823 | +0.0263 |
| OSCR | 0.5570 | 0.5440 | -0.0130 |
| Known accuracy | 62.12% | 60.78% | -1.34 pp |
| Known acceptance | 95.45% | 94.95% | -0.50 pp |
| Unknown rejection | 26.25% | 28.18% | +1.53 pp |
| Score KS | 0.5113 | 0.5075 | -0.0038 |
| Mean score gap | 2.3149 | 2.3120 | -0.0029 |
| Unknown below known q95 | 0.7173 | 0.7198 | +0.0025 |

## Decision

Layer-wise decay did not improve the continuous detector ranking or feature
distribution separation. The higher unknown rejection rate at the selected
operating point came with lower known acceptance and worse FPR95/OSCR, so it is
not evidence that the core overlap problem was solved. Keep
`--encoder-lr-decay` as an optional reproducibility/ablation switch, but keep
`encoder_lr_scale=0.1, encoder_lr_decay=1.0` as the current main training
candidate and stop scanning decay values.

This result also supports a broader conclusion: the remaining bottleneck is
not a small adjustment to the backbone learning-rate schedule. Future work
should target the training signal and representation geometry that distinguish
unknown samples, with the same paired protocol and continuous score metrics.

Artifacts:

- `runs/semantic_isolated_full_s43_cifar_profile_lr01_decay08_student/`
- `runs/semantic_isolated_full_s43_cifar_profile_lr01_decay08_discover/`
