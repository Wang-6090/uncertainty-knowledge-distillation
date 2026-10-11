# Mixed-pool PU boundary pilot (2026-10-11)

## Motivation

The historical hybrid boundary loss used the whole discovery pool as if it
were pure unknown. That is unsafe for a mixed pool: known samples can be
incorrectly pushed away from the known manifold. This pilot adds an explicit
PU-corrected boundary objective. It estimates a mixed-pool boundary penalty,
subtracts the observed known contribution using the configured known prior,
and backpropagates only the remaining non-negative unknown risk.

The new opt-in controls are:

- `--discovery-boundary-mode {pure,pu}`;
- `--discovery-boundary-pu-risk {hard_nnpu,soft_nnpu}`;
- `--discovery-boundary-pu-softness`.

The default remains `pure`, so historical configurations do not change.

## Controlled protocol

All arms used semantic-isolated CIFAR-100 60/40, the same teacher for each
seed, pretrained ResNet-18 students, 1200/300/1000 train/validation/test
limits, two student epochs, a 400-sample matched mixed pool with known prior
0.2, the same uncertainty nnPU loss, `normalized_entropy_proto` scoring,
MC=4, validation-only 95% known-coverage calibration, and clustering off.
The only training variable was the PU boundary term (`0` versus weight
`0.05`, cosine margin `0.2`, prototype/local-centroid weight `0.5`).

| Seed | Arm | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection | KS |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | baseline | 0.6420 | 0.8436 | 0.1722 | 93.51% | 10.53% | 0.2667 |
| 42 | PU boundary | 0.6581 | 0.8802 | 0.1963 | 95.67% | 9.27% | 0.2801 |
| 43 | baseline | 0.5886 | 0.9161 | 0.1861 | 96.14% | 5.20% | 0.1714 |
| 43 | PU boundary | 0.6251 | 0.8725 | 0.1904 | 92.11% | 10.89% | 0.2232 |
| 44 | baseline | 0.5975 | 0.8774 | 0.1501 | 92.04% | 13.98% | 0.1835 |
| 44 | PU boundary | 0.6000 | 0.8519 | 0.1484 | 94.27% | 10.22% | 0.1548 |
| 3-seed mean | baseline -> PU boundary | 0.6094 -> 0.6277 | 0.8790 -> 0.8682 | 0.1695 -> 0.1784 | 93.90% -> 94.02% | 9.90% -> 10.13% | 0.2072 -> 0.2194 |

## Decision

This is a useful but small positive signal. Mean AUROC, FPR95, OSCR,
known acceptance, unknown rejection, and KS all move in the desired direction,
but the unknown-rejection gain is only 0.23 percentage points and the seed
effects are visible. The method therefore remains an opt-in ablation and is
not the default configuration. It does not resolve the known/unknown feature
overlap.

### Hard versus soft PU risk

On the same seed-42 treatment protocol, replacing `hard_nnpu` with
`soft_nnpu` (softness `0.05`) produced AUROC `0.6457`, FPR95 `0.8552`, OSCR
`0.1961`, known acceptance `93.84%`, unknown rejection `14.54%`, and KS
`0.2235`. Hard-PU produced `0.6581/0.8802/0.1963/95.67%/9.27%/0.2801`.
The soft variant is more aggressive at the selected operating point but has
worse ranking, geometry separation, and known acceptance. Keep hard-PU as the
preferred boundary ablation and do not tune the soft surrogate further now.

The next validation should use the full-data protocol and more than two
epochs, with the same checkpoint-selection rule and an independent calibration
subset. If the ranking/KS improvement survives while fixed-coverage rejection
does not, the boundary objective is only improving score ordering and the
project should return to stronger representation learning or a better
semantic split. Do not stack it with another loss before that confirmation.

## Implementation note

The implementation is in `novel_discovery/losses.py` and is wired through
`novel_discovery/pipeline.py` and `train.py`. A validation guard now permits
`mode=pu` only for mixed pools and requires a valid mixed-pool prior. The
historical pure boundary mode is still rejected for full mixed-pool use.
## Full-data confirmation (seed 42)

The pilot was followed by a matched full-data run to check whether its small
three-seed signal survives a realistic training budget. Both arms used the
semantic-isolated CIFAR-100 60/40 split, the same pretrained ResNet-34
teacher, ResNet-18 student, seed 42, all training samples, five student
epochs, a matched mixed pool of 5400 samples, known prior 0.2, the same
uncertainty nnPU term, and the same validation-only 95% known-coverage
evaluation. The only intended training difference was the PU boundary term:
baseline weight 0 versus treatment weight 0.05 with `mode=pu`,
`hard_nnpu`, margin 0.2, and prototype weight 0.5. Configuration audit found
no other training or data differences apart from output paths and the
resulting best epoch/checkpoint statistics.

| Metric | Baseline | PU boundary | Difference (treatment - baseline) |
|---|---:|---:|---:|
| AUROC | 0.6993 | 0.7059 | +0.0066 |
| AUPR | 0.5666 | 0.5761 | +0.0095 |
| FPR95 | 0.8023 | 0.7715 | -0.0308 |
| OSCR | 0.3840 | 0.3808 | -0.0031 |
| Known acceptance | 94.43% | 94.95% | +0.52 pp |
| Unknown rejection | 13.58% | 14.43% | +0.85 pp |
| Score KS distance | 0.3079 | 0.3043 | -0.0036 |

The full-data result supports a small ranking/operating-point benefit, but it
does not show a clear reduction in global score-distribution overlap: KS is
slightly lower and OSCR is slightly worse. The unknown rejection rate is also
still low, with 3423 of 4000 unknown test samples accepted by the treatment.
Therefore PU boundary remains an optional ablation, not the main solution to
the core overlap problem. Further tuning of this loss is lower priority than
direct feature-space objectives, calibrated rejection modeling, and an audit
of semantic split difficulty. Full reports are in
`runs/pu_boundary_full_s42_baseline_discover/discovery_report.json` and
`runs/pu_boundary_full_s42_treatment_discover/discovery_report.json`.

## Detector audit on the same full-data checkpoints

To separate representation effects from score selection, both checkpoints
were evaluated again with the same class-wise kNN support score
(`normalized_entropy_min_class_knn`, feature bank from known training data,
`k=10`, MC=4, and the same 95% known-coverage policy). No model or data was
changed.

| Metric | Baseline + kNN | PU boundary + kNN | Difference |
|---|---:|---:|---:|
| AUROC | 0.7053 | 0.7185 | +0.0132 |
| FPR95 | 0.7743 | 0.7473 | -0.0270 |
| OSCR | 0.3857 | 0.3838 | -0.0019 |
| Known acceptance | 94.52% | 95.25% | +0.73 pp |
| Unknown rejection | 14.05% | 13.53% | -0.53 pp |
| Score KS distance | 0.3190 | 0.3339 | +0.0149 |

The kNN score itself is clearly more informative than the prototype score on
these checkpoints, and the PU boundary treatment improves ranking and KS under
that score. However, the fixed-coverage unknown-rejection point does not
improve. This supports keeping class-wise support scoring as a detector-side
component and treating PU boundary as a weak optional representation
ablation, rather than claiming that it solves the overlap problem.
