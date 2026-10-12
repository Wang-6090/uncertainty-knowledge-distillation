# Encoder learning-rate scale with disjoint calibration (seed 44, 2026-10-12)

## Purpose

The earlier three-seed comparison suggested that protecting the pretrained
student encoder with `encoder_lr_scale=0.1` reduces known/unknown score
overlap. This run checks that conclusion under a stricter evaluation protocol:
the known validation set is split into a model-selection subset and a separate
threshold-calibration subset. The test labels are used only for the final
report.

## Fixed protocol

- CIFAR-100 semantic-isolated 60/40 split, seed 44.
- CIFAR preprocessing, pretrained ResNet-34 teacher and ResNet-18 student.
- Five epochs, full training/validation/test data, batch size 64, CUDA.
- Same teacher checkpoint for both students.
- Matched mixed discovery pool of 5,400 images with known prior 0.2.
- Same uncertainty KD, feature KD, prototype loss, and audited uncertainty
  nnPU settings in both students.
- Detector: `normalized_entropy_min_class_knn`, feature kNN `k=10`, MC=4.
- Threshold: validation-only 95% known coverage, with
  `calibration_ratio=0.25`; no test label is used for threshold selection.
- Clustering disabled to isolate unknown detection.
- Only changed training factor: `encoder_lr_scale` (1.0 vs 0.1).

## Results

| Metric | scale 1.0 | scale 0.1 | Change (0.1 - 1.0) |
|---|---:|---:|---:|
| AUROC | 0.7738 | 0.8089 | +0.0351 |
| AUPR | 0.6620 | 0.7037 | +0.0417 |
| FPR95 | 0.6870 | 0.5888 | -0.0982 |
| OSCR | 0.5087 | 0.5217 | +0.0130 |
| Known classification accuracy | 58.55% | 58.10% | -0.45 pp |
| Known acceptance | 95.18% | 93.88% | -1.30 pp |
| Unknown rejection | 21.40% | 30.50% | +9.10 pp |
| Score KS | 0.4147 | 0.4807 | +0.0660 |
| Mean unknown-known score gap | 1.9369 | 2.1712 | +0.2343 |
| Unknown below known q95 | 0.7793 | 0.7385 | -0.0408 |

The calibration subset achieved 94.92% known acceptance for both arms, so the
different test operating points are not caused by using different calibration
data. The treatment changes the continuous score distributions: its unknown
mean increases from 1.9385 to 2.2066 while the known mean changes only from
0.0015 to 0.0354, and KS/overlap metrics improve together.

## Decision

This is a positive replication of the encoder-learning-rate direction under a
disjoint calibration protocol. Keep `encoder_lr_scale=0.1` in the current
CIFAR candidate recipe and stop scanning smaller uniform scales or minor
layer-wise decay variants. The result is still not a solution to the core
problem: about 69.5% of unknown test samples remain accepted at the calibrated
operating point, and known acceptance is slightly below the nominal 95% on the
test distribution.

The next algorithmic experiment should target the training signal that creates
the remaining overlap, not another threshold or score fusion. It should use
the same seed-44 disjoint-calibration protocol, compare one factor at a time,
and report AUROC, FPR95, OSCR, KS, score overlap, known acceptance, and unknown
rejection together. Previously tested independent-rejection-branch losses and
post-hoc rejector variants remain opt-in/negative ablations and should not be
silently combined with this result.

## Artifacts

- `runs/semantic_isolated_full_s44_disjointcal_teacher/`
- `runs/semantic_isolated_full_s44_disjointcal_lr1_student/`
- `runs/semantic_isolated_full_s44_disjointcal_lr1_discover/`
- `runs/semantic_isolated_full_s44_disjointcal_lr01_student/`
- `runs/semantic_isolated_full_s44_disjointcal_lr01_discover/`
