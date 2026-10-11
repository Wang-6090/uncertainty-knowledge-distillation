# CIFAR profile + nnPU + encoder LR scale 0.1 (seed 123, 2026-10-11)

## Purpose

The seed-42 result suggested that preserving more of the pretrained encoder
could reduce known/unknown overlap without changing the loss design. This is a
strict replication on a second seed, not a new loss combination.

## Protocol

- CIFAR-100 semantic-isolated 60/40 split, seed 123, CIFAR transforms.
- Pretrained ResNet-34 teacher and ResNet-18 student, five epochs, full data.
- Matched mixed discovery pool of 5,400 samples, known prior 0.2.
- Feature KD, prototype loss, and audited uncertainty nnPU at alpha 0.1.
- Detector: `normalized_entropy_min_class_knn`, k=10, MC=4.
- Threshold: known-validation-only 95% known coverage.
- The only training change versus the paired nnPU baseline is
  `encoder_lr_scale: 1.0 -> 0.1` for the student.

## Results

| Metric | nnPU baseline | encoder LR 0.1 | Change |
|---|---:|---:|---:|
| AUROC | 0.7933 | 0.8170 | +0.0237 |
| AUPR | 0.6701 | 0.7226 | +0.0525 |
| FPR95 | 0.6088 | 0.5910 | -0.0178 |
| OSCR | 0.5349 | 0.5603 | +0.0254 |
| Known accuracy | 60.73% | 62.10% | +1.37 pp |
| Known acceptance | 95.68% | 94.55% | -1.13 pp |
| Unknown rejection | 19.78% | 29.53% | +9.75 pp |
| Score KS | 0.4547 | 0.4974 | +0.0427 |
| Mean score gap | unavailable | 2.3052 | -- |
| Unknown below known q95 | unavailable | 0.7180 | -- |

## Decision

This replication supports the seed-42 direction: ranking metrics, score KS,
and the fixed 95% known-coverage operating point all improve, while known
accuracy also increases. The change is not only a threshold adjustment because
the checkpoint and continuous score distribution change. It is still not a
solution to the overlap problem: 29.53% unknown rejection means most unknown
samples remain accepted, and known acceptance decreases by about 1.1 points.

The next check should be a third-seed replication or a small strict scale
ablation (`1.0`, `0.1`, `0.01`) with all other settings fixed. Do not combine
this change with boundary loss, angular margin, or weighted-PU variants before
that check.

Artifact: `runs/semantic_isolated_full_s123_cifar_profile_lr01_student/` and
`runs/semantic_isolated_full_s123_cifar_profile_lr01_discover/`.
