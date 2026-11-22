# Selective logits KD pilot result

This document records the frozen result of the seed `2138` pilot described in
[`selective_kd_protocol.md`](selective_kd_protocol.md). The scope is limited to
student classification representation, logits KD, and downstream detection
and discovery evaluation.

The three paired arms were standard logits KD, teacher-correct selective logits
KD, and existing uncertainty-KD. All used the same semantic-hard CIFAR-100
60/40 split, teacher, student initialization, four epoch budget,
known-validation checkpoint selection, max-softmax score, 95% known-coverage
calibration, automatic K, and clustering. Feature KD, rejection-branch KD and
discovery/PU objectives were disabled. Unknown labels were used only for final
evaluation.

## Mechanism evidence

Selective KD was active in every epoch. The teacher-correct fractions were
`0.578287`, `0.576435`, `0.579630`, and `0.578241`. Selective KD loss, its
audited gradient norm, and the student classifier update norm were nonzero in
all epochs. The standard arm had zero selective activity. The uncertainty arm
applied its separate uncertainty weighting in all epochs; at the final epoch
the weight standard deviation was `0.185878` and ESS fraction was `0.966473`.

These fields establish that the two mechanisms reached the student objective
through separate paths. The teacher output remained detached, and no student
uncertainty head, classification entropy, or final rejection score was used as
the selective gate.

## Paired metrics

| Metric | Standard | Selective | Δ selective | Uncertainty | Δ uncertainty |
|---|---:|---:|---:|---:|---:|
| AUROC | 0.661509 | 0.664273 | +0.002764 | 0.659632 | -0.001877 |
| AUPR | 0.527693 | 0.536717 | +0.009023 | 0.528384 | +0.000691 |
| FPR95 | 0.790833 | 0.809833 | +1.900 pp | 0.810000 | +1.917 pp |
| Known accuracy | 0.672000 | 0.688833 | +1.683 pp | 0.669000 | -0.300 pp |
| Unknown rejection | 0.098750 | 0.115750 | +1.700 pp | 0.128750 | +3.000 pp |
| Candidate purity | 0.574128 | 0.610013 | +3.589 pp | 0.601636 | +2.751 pp |
| Cluster ACC | 0.188953 | 0.193676 | +0.472 pp | 0.177570 | -1.138 pp |
| Cluster NMI | 0.467173 | 0.455422 | -1.175 pp | 0.430164 | -3.701 pp |
| Cluster ARI | 0.046169 | 0.062076 | +1.591 pp | 0.045411 | -0.076 pp |
| Auto-K absolute error | 5 | 0 | -5 | 4 | -1 |

On known validation, selective KD reduced feature within-class mean from
`0.227453` to `0.215512` and improved center margin from `-0.067853` to
`-0.056426`. This supports a representation-level signal, while the lower NMI
shows that the signal does not yet imply uniform clustering improvement.

## Decision

The pilot passes the fixed promotion rule because selective KD improves a
detection metric and candidate purity without known-accuracy loss, and its
mechanism audit passes. It remains opt-in: one seed and lower NMI are
insufficient to change the default `kd_mode=standard`, and uncertainty-KD has
no extra stable gain in this comparison.

The next stage is a fixed two-seed confirmation on new semantic-hard splits
`2163/2164`. It must use the same gate and protocol without scanning gate
thresholds, loss coefficients, temperatures, or score rules. If the
confirmation does not reproduce detection plus a candidate or clustering gain
without a material trade-off, selective KD will be frozen as a diagnostic
variant.
