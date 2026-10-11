# CIFAR profile + nnPU + encoder LR scale 0.1 (seed 43, 2026-10-12)

## Purpose

The seed-42 and seed-123 runs both suggested that updating the pretrained
encoder with a smaller learning rate reduces open-set score overlap. This run
is a strict paired seed-43 check. It uses a newly trained nnPU baseline with
the same protocol and changes only the encoder learning-rate scale.

## Protocol

- CIFAR-100 semantic-isolated 60/40 split, seed 43, CIFAR transforms.
- The same pretrained ResNet-34 teacher and ResNet-18 student architecture in
  both arms, full data, five epochs, batch size 64.
- Matched mixed discovery pool of 5,400 samples, known prior 0.2.
- Feature KD, prototype loss, and audited uncertainty nnPU at alpha 0.1.
- Detector: `normalized_entropy_min_class_knn`, k=10, MC=4.
- Threshold: known-validation-only 95% known coverage.
- Only `encoder_lr_scale` changes: `1.0` versus `0.1`.

## Results

| Metric | nnPU, scale 1.0 | nnPU, scale 0.1 | Change |
|---|---:|---:|---:|
| AUROC | 0.7901 | 0.8228 | +0.0327 |
| AUPR | 0.6773 | 0.7241 | +0.0468 |
| FPR95 | 0.6400 | 0.5560 | -0.0840 |
| OSCR | 0.5324 | 0.5570 | +0.0246 |
| Known accuracy | 60.55% | 62.12% | +1.57 pp |
| Known acceptance | 95.08% | 95.45% | +0.37 pp |
| Unknown rejection | 21.45% | 26.25% | +4.80 pp |
| Score KS | 0.4527 | 0.5113 | +0.0587 |
| Mean score gap | 2.1084 | 2.3149 | +0.2065 |
| Unknown below known q95 | 0.7850 | 0.7173 | -0.0678 |

## Decision

The paired result supports the encoder-LR direction: all major ranking and
distribution-separation metrics improve, and the known-coverage operating
point improves without sacrificing known acceptance. This is not a threshold-
only effect because both models are re-trained and their continuous score
distributions change.

The method still does not solve open-set detection. Most unknown samples remain
accepted even after the improvement, so `encoder_lr_scale=0.1` should be
treated as a stronger training baseline, not as the final algorithm. The old
seed-43 baseline/treatment directories were not used for attribution because
they changed nnPU and feature-margin together or used a different detector.

Artifacts:

- `runs/semantic_isolated_full_s43_cifar_profile_nnpu_student/`
- `runs/semantic_isolated_full_s43_cifar_profile_nnpu_discover/`
- `runs/semantic_isolated_full_s43_cifar_profile_lr01_student/`
- `runs/semantic_isolated_full_s43_cifar_profile_lr01_discover/`
