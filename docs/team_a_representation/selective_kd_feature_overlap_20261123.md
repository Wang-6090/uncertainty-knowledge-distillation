# Frozen feature-overlap and teacher/student support geometry

## Validity correction

The first four feature-overlap outputs used the default ImageNet transform,
although the saved CIFAR-100 experiment configs specify `transform_profile=cifar`.
Those raw files are retained for auditability, but **all metrics and interpretations
from them are invalid** and must not be cited:

- `analysis/team_a_feature_overlap_selective_20261123_s2163.json`
- `analysis/team_a_feature_overlap_selective_20261123_s2164.json`
- `analysis/team_a_feature_overlap_uncertainty_20261123_s2163.json`
- `analysis/team_a_feature_overlap_uncertainty_20261123_s2164.json`

The corrected script passes the explicit CIFAR profile to dataset construction
and deterministic known-support extraction. It also refuses to overwrite an
existing output. The corrected outputs have `_cifar` in their filenames.

## Protocol

This is a read-only audit of the already trained standard, selective, and
uncertainty student checkpoints on semantic-hard CIFAR-100 60/40 seeds
`2163/2164`. Each run used the same fixed 27,000 known training images and
10,000 test images (6,000 known, 4,000 unknown). Test labels were used only
for retrospective known/unknown distance summaries and AUROC. No weights,
thresholds, scores, or models were fitted or selected. Teacher/student support
geometry was extracted separately with the same CIFAR transform; the official
max-softmax metrics below are read from the completed discovery reports.

The main corrected artifacts are:

- `analysis/team_a_feature_overlap_selective_20261123_s2163_cifar.json`
- `analysis/team_a_feature_overlap_selective_20261123_s2164_cifar.json`
- `analysis/team_a_feature_overlap_uncertainty_20261123_s2163_cifar.json`
- `analysis/team_a_feature_overlap_uncertainty_20261123_s2164_cifar.json`
- `analysis/team_a_teacher_student_support_geometry_20261123_cifar.json`

## Student support-distance AUROC

Scores are cosine distance to the classifier weight prototype, empirical
known-class centroid, or nearest known training example. Higher AUROC means
better retrospective known/unknown ranking for that diagnostic score; these
are **not** the deployed detector.

| Seed | Arm | Classifier prototype | Empirical centroid | Nearest known train sample |
|---:|---|---:|---:|---:|
| 2163 | standard | 0.673140 | 0.662513 | 0.677714 |
| 2163 | selective | 0.674058 | 0.665770 | 0.680203 |
| 2163 | uncertainty | 0.673028 | 0.671199 | 0.688143 |
| 2164 | standard | 0.679708 | 0.676212 | 0.702700 |
| 2164 | selective | 0.698868 | 0.680614 | 0.708806 |
| 2164 | uncertainty | 0.686055 | 0.685469 | 0.711792 |

Paired changes from standard:

| Arm | Seed | Prototype | Centroid | Nearest support |
|---|---:|---:|---:|---:|
| selective | 2163 | +0.000918 | +0.003257 | +0.002489 |
| selective | 2164 | +0.019160 | +0.004402 | +0.006106 |
| uncertainty | 2163 | -0.000112 | +0.008686 | +0.010429 |
| uncertainty | 2164 | +0.006348 | +0.009257 | +0.009092 |

Both variants improve empirical-centroid and nearest-support AUROC on these
two splits. Uncertainty-KD's prototype-distance change is nearly zero on
2163 and positive on 2164. This is descriptive evidence of altered student
support geometry, not an independent-seed estimate: the class protocols
overlap, and the sample is only two seeds.

## Teacher/student comparison and downstream boundary

Teacher versus standard-student diagnostic AUROCs were:

| Seed | Model | Prototype distance | Centroid distance |
|---:|---|---:|---:|
| 2163 | teacher | 0.613330 | 0.626661 |
| 2163 | standard student | 0.673140 | 0.662513 |
| 2164 | teacher | 0.626217 | 0.609799 |
| 2164 | standard student | 0.679708 | 0.676212 |

The teacher has no advantage on either fixed support-distance diagnostic in
either seed. Therefore this result does **not** motivate a teacher-to-student
support-distance transfer loss; no such training pilot was launched.

The actual max-softmax AUROC for standard/selective/uncertainty was
`0.674532/0.671938/0.677807` on 2163 and
`0.679621/0.700087/0.690911` on 2164. These official values are taken from
the discovery reports, not recomputed by the geometry script. The existing
confirmation report shows that selective-KD lowers unknown rejection and
candidate purity on both seeds; uncertainty-KD's rejection, candidate, and
clustering changes remain mixed. Hence the corrected distance gains do not
establish stable transfer to unknown detection or new-class discovery and do
not justify changing the default `kd_mode=standard`.

## Decision

This stage corrects an analysis-pipeline error and establishes that both KD
variants change some student known-support geometry. It also rejects the
specific hypothesis that the teacher provides a superior support-distance
target. Do not repeat the two-seed checkpoint extraction or start a support-
distance transfer loss on this evidence. Continue evaluating the assigned KD
questions through formal multi-seed standard-versus-uncertainty evidence and
mechanism checks; do not change rejection, scoring, candidate selection, or
clustering components in this workstream.
