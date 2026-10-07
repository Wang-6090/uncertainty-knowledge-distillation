# Hard-proxy margin pilot (2026-10-05)

## Purpose

The current bottleneck is overlap between known and unknown feature/support
distributions. This pilot tests a known-only representation objective rather
than another post-hoc score. The objective is a hard proxy margin: each known
feature is pulled toward its labeled class proxy and required to be separated
from the most similar incorrect proxy.

This is related to proxy-based metric learning, but it is an explicit
hard-negative margin variant for this project; it is not a claim of reproducing
Proxy Anchor or ArcFace.

## Fixed protocol

- CIFAR-100 random 60/40 split: `splits_cifar100_60_40.json`.
- Seed `42`, pretrained ResNet-34 teacher and ResNet-18 student.
- Same freshly trained teacher: `runs/hard_proxy_teacher_s42/teacher.pt`.
- Full known training partition, mixed discovery pool, known pool ratio `0.2`.
- Five student epochs, batch size `64`, uncertainty nnPU weight `0.1`.
- Detection: `normalized_entropy_min_class_knn`, kNN `k=10`, backbone feature
  bank, MC `8`, known-only 95% coverage threshold.
- The only training change was `alpha_hard_proxy_margin=0.05`; margin was
  fixed at `0.2`.

## Open-set result

| Metric | nnPU baseline | + hard-proxy margin | Change |
| --- | ---: | ---: | ---: |
| AUROC | 0.5981 | 0.5966 | -0.0015 |
| FPR95 | 0.8835 | 0.8705 | -0.0130 |
| OSCR | 0.2474 | 0.2593 | +0.0119 |
| Known acceptance | 0.9505 | 0.9515 | +0.0010 |
| Unknown rejection | 0.0868 | 0.0795 | -0.0073 |
| Accepted-known accuracy | 0.3410 | 0.3600 | +0.0189 |

## Feature-overlap result

The diagnostic used the complete known training partition and complete open
test set. Test labels only stratified descriptive known/unknown statistics;
they were not used for training, threshold selection, or score fitting.

| Distance | Baseline AUROC | Margin AUROC | Baseline overlap | Margin overlap |
| --- | ---: | ---: | ---: | ---: |
| Classifier prototype | 0.5781 | 0.5766 | 0.8836 | 0.8757 |
| Empirical class centroid | 0.5618 | 0.5665 | 0.8978 | 0.8981 |
| Nearest known train sample | 0.6058 | 0.6051 | 0.8430 | 0.8397 |

## Decision

The method improves FPR95, OSCR, and accepted-known accuracy, but lowers
AUROC and the actual unknown rejection rate. Geometry changes are mixed and
small, with no consistent improvement across prototype, centroid, and local
support distances. It is therefore not promoted to the default pipeline and
will not receive further margin tuning in this round.

The failed first command using an incompatible historical teacher was rejected
by checkpoint split validation and is not part of this result. The valid
checkpoints and detector reports are in:

- `runs/hard_proxy_teacher_s42/`
- `runs/hard_proxy_base_s42/`
- `runs/hard_proxy_treatment_s42/`
- `runs/hard_proxy_base_s42_knn_detect/`
- `runs/hard_proxy_treatment_s42_knn_detect/`
- `analysis/hard_proxy_overlap_s42_full.json`

