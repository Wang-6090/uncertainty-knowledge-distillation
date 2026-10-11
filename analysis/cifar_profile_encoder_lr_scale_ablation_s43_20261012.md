# Encoder learning-rate scale ablation (seed 43, 2026-10-12)

## Purpose

After three paired seeds supported `encoder_lr_scale=0.1`, this experiment
checks whether the gain simply comes from making the pretrained encoder as
static as possible. Only the scale changes; all other training and detection
settings are fixed.

## Protocol

- CIFAR-100 semantic-isolated 60/40 split, seed 43, CIFAR transforms.
- Same pretrained ResNet-34 teacher, ResNet-18 student, five epochs, full data.
- Matched mixed discovery pool of 5,400 samples, known prior 0.2.
- Feature KD, prototype loss, audited uncertainty nnPU alpha 0.1.
- Detector: `normalized_entropy_min_class_knn`, k=10, MC=4.
- Threshold: validation-only 95% known coverage.
- Compared `encoder_lr_scale=0.1` and `0.01`.

## Results

| Metric | scale 0.1 | scale 0.01 | Change (0.01 - 0.1) |
|---|---:|---:|---:|
| AUROC | 0.8228 | 0.7971 | -0.0257 |
| AUPR | 0.7241 | 0.6872 | -0.0369 |
| FPR95 | 0.5560 | 0.6027 | +0.0467 |
| OSCR | 0.5570 | 0.4762 | -0.0808 |
| Known accuracy | 62.12% | 53.62% | -8.50 pp |
| Known acceptance | 95.45% | 94.68% | -0.77 pp |
| Unknown rejection | 26.25% | 23.98% | -2.28 pp |
| Score KS | 0.5113 | 0.4584 | -0.0529 |
| Mean score gap | 2.3149 | 1.9600 | -0.3549 |
| Unknown below known q95 | 0.7173 | 0.7695 | +0.0523 |

## Decision

The smaller scale is clearly worse. It reduces task adaptation and harms both
known classification and open-set ranking; the slight change in the operating
point is not a useful trade-off. Keep `encoder_lr_scale=0.1` as the current
candidate and stop scanning smaller uniform scales. A later extension can test
layer-wise or scheduled learning rates, but it must be compared against the
fixed 0.1 baseline as a new single-factor experiment.

Artifacts:

- `runs/semantic_isolated_full_s43_cifar_profile_lr001_student/`
- `runs/semantic_isolated_full_s43_cifar_profile_lr001_discover/`
