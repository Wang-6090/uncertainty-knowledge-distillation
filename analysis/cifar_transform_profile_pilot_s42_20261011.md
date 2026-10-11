# CIFAR preprocessing paired pilot (2026-10-11)

## Purpose

Repeated algorithmic changes only modestly changed unknown rejection. One
possible upstream cause is that CIFAR-100 images are small, while the current
pipeline historically uses ImageNet-style preprocessing: RandomResizedCrop,
ColorJitter, ImageNet mean/std, and ImageNet-pretrained ResNet stem behavior.
This pilot tests whether a CIFAR-specific transform profile improves the
student representation and the open-set detector.

## Code change

Added an opt-in command-line option:

- --transform-profile imagenet: historical behavior, default
- --transform-profile cifar: CIFAR-100 mean/std, resize to target size,
  RandomCrop with padding and horizontal flip for training, deterministic
  resize for evaluation

The default remains imagenet, so old commands are unchanged. The transform
profile is now passed consistently through teacher training, student training,
discovery evaluation, outlier datasets, and deterministic kNN-bank clones.

## Important protocol correction

A first quick check used a CIFAR-profile student with an old ImageNet-profile
teacher checkpoint. That is not a valid KD comparison, because the teacher then
receives CIFAR-normalized inputs it was not trained on. The results from that
mismatched check are not used as evidence.

The stricter paired pilot below trains a matching teacher and student for each
profile.

## Paired pilot protocol

Fixed settings:

- CIFAR-100 semantic-isolated 60/40 split, seed 42
- 1,200 known training samples and 300 known validation samples for speed
- pretrained ResNet-34 teacher, 2 epochs
- pretrained ResNet-18 student, 2 epochs
- same CE + uncertainty KD + feature KD + SupCon + prototype losses
- full 10,000-image open test set for detection
- detector: normalized_entropy_min_class_knn, feature kNN k=10, MC=4
- threshold: validation-known-only 95% known coverage
- clustering skipped

Only the transform profile changes, and each student uses the teacher trained
with the same profile.

## Results

| Profile | teacher known acc | student known acc | AUROC | AUPR | FPR95 | OSCR | known accept | unknown reject | score KS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ImageNet profile | 13.67% | 20.70% | 0.6062 | 0.4851 | 0.8875 | 0.1521 | 95.27% | 7.98% | 0.1617 |
| CIFAR profile | 18.00% | 27.68% | 0.6502 | 0.5147 | 0.8528 | 0.2253 | 94.42% | 10.23% | 0.2389 |
| difference | +4.33 pp | +6.98 pp | +0.0440 | +0.0296 | -0.0347 | +0.0732 | -0.85 pp | +2.25 pp | +0.0773 |

## Interpretation

The CIFAR profile is the most promising new direction from this round. It
improves closed-set known accuracy, ranking metrics, FPR95, OSCR, unknown
rejection, and score-distribution KS in a matched short pilot. The known
acceptance point drops slightly, so this is not a solved detector, but the
trade-off is better than most recent loss-level tweaks.

Because this is only a short, limited-training pilot, it should not be promoted
directly to the final recipe. The next justified experiment is a full or
half-full paired recheck with matching teachers and students, using the same
semantic-isolated split and the same detector. If the gain persists, set
--transform-profile cifar as the recommended CIFAR-100 protocol while keeping
imagenet as the historical compatibility default.

Artifacts:

- runs/preprocess_e2_s42_imagenet_teacher/
- runs/preprocess_e2_s42_imagenet_student/
- runs/preprocess_e2_s42_imagenet_discover/discovery_report.json
- runs/preprocess_e2_s42_cifar_teacher/
- runs/preprocess_e2_s42_cifar_student/
- runs/preprocess_e2_s42_cifar_discover/discovery_report.json

