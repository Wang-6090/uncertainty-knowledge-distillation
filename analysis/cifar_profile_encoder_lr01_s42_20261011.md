# CIFAR profile + nnPU + encoder LR scale 0.1 (seed 42, 2026-10-11)

## Purpose

The CIFAR + nnPU baseline improves the representation, but the student still
updates the pretrained encoder with the full classification learning rate.
This experiment tests whether preserving more pretrained visual structure
reduces known/unknown feature overlap.

## Protocol

- CIFAR-100 semantic-isolated 60/40 split, seed 42, CIFAR transforms.
- Pretrained ResNet-34 teacher and ResNet-18 student, five epochs, full data.
- Matched mixed discovery pool of 5,400 samples, known prior 0.2.
- Feature KD, prototype loss, and audited uncertainty nnPU at alpha 0.1.
- Detector: `normalized_entropy_min_class_knn`, k=10, MC=4.
- Threshold: known-validation-only 95% known coverage.
- The only training change is `encoder_lr_scale: 1.0 -> 0.1` for the student.

## Results

| Metric | nnPU baseline | encoder LR 0.1 | Change |
|---|---:|---:|---:|
| AUROC | 0.7906 | 0.8128 | +0.0222 |
| AUPR | 0.6757 | 0.7121 | +0.0364 |
| FPR95 | 0.6285 | 0.5712 | -0.0573 |
| OSCR | 0.5536 | 0.5536 | ~0.0000 |
| Known accuracy | 63.08% | 62.08% | -1.00 pp |
| Known acceptance | 94.80% | 95.12% | +0.32 pp |
| Unknown rejection | 23.00% | 26.78% | +3.78 pp |
| Score KS | 0.4554 | 0.4837 | +0.0283 |
| Mean score gap | 2.1761 | 2.2529 | +0.0768 |
| Unknown below known q95 | 0.7770 | 0.7295 | -0.0475 |

## Decision

This is a promising result rather than a final conclusion. Ranking metrics,
score separation, and fixed known-coverage unknown rejection improve together;
known acceptance is preserved and the known-accuracy cost is about one point.
The next step is a strict seed-123 replication with no other changes. If it
repeats, use LR scale 0.1 as the new baseline and study only a small scale
ablation or multi-seed confirmation; do not immediately combine it with a new
loss.

Artifact: `runs/semantic_isolated_full_s42_cifar_profile_lr01_student/`

