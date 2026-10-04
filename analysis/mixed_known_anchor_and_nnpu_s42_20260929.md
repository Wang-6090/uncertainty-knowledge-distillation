# Mixed-pool known anchors and nnPU rejector (2026-09-29)

## Question and fixed protocol

The target problem is still known/unknown feature and score overlap, not merely
threshold selection. This round first tested whether low-risk, cross-view
consistent mixed-pool samples can be safely pulled toward known classes, then
tested a PU rejector without retraining the student representation.

The CIFAR-100 rejector comparison uses the same `corrected_mixed_ntxent_s42`
student checkpoint, 60/40 class split, seed 42, 1200/300/1000 sample limits,
disjoint mixed discovery pool, and known-only 95% coverage calibration. The
only intended change among the final three detector runs is the mixed-pool
rejector training rule (hard, heuristic soft-PU, or nnPU). Test labels are
used only for retrospective benchmark metrics.

## Small joint-training experiment

A new optional mixed-pool known-anchor loss was added. Candidate risk is now
computed once from the average of paired views rather than selecting top-k
separately in each view and intersecting the masks. The old intersection could
empty the candidate set on small batches. Low-risk samples are eligible for
known consistency only when both views agree and a detached target model
exceeds a confidence threshold. The target can be the EMA student or the
teacher.

Toy smoke confirmed the paired gate and known-anchor loss can both produce
nonzero training values. On CIFAR-100 1200/300/1000, 3 epochs, the EMA-target
run had AUROC `0.5049`, FPR95 `0.9408`, OSCR `0.1609`, known accuracy `0.2451`,
and reported unknown rejection `0.0842`. The teacher-target run had AUROC
`0.4955`, FPR95 `0.9227`, OSCR `0.1811`, known accuracy `0.2878`, and reported
unknown rejection `0.0612`. The corrected mixed NT-Xent reference had AUROC
`0.5494`, FPR95 `0.9178`, OSCR `0.1827`, known accuracy `0.2664`, and reported
unknown rejection `0.0536`.

The anchor losses execute, but neither treatment improves all target metrics.
The teacher-target variant recovers known accuracy, while its ranking and
unknown rejection remain weak. A retrospective audit of the teacher on the
mixed pool found that even samples with confidence at least `0.9` and
cross-view agreement were only `44.4%` known; therefore confidence agreement
is not a reliable known pseudo-label in this setting. Do not promote this loss
as a solution to overlap.

## nnPU implementation and controlled comparison

The optional `--rejector-training nnpu` uses normalized frozen embeddings,
known training examples as positive data, and the disjoint mixed pool as
unlabeled data. It optimizes the non-negative PU risk with a user-supplied
estimated known prior (`--rejector-known-prior`, default `0.2`). This is a
small linear-logistic nnPU-inspired rejector, not a reproduction of the full
deep nnPU method from Kiryo et al. (NeurIPS 2017). Its assumption about the
known prior is a material limitation and should be sensitivity-tested.

### Reported detector results

| Mixed-pool rejector | AUROC | AUPR | FPR95 | OSCR | Known accuracy (all known) | Reported unknown reject |
|---|---:|---:|---:|---:|---:|---:|
| nnPU, prior 0.20 | 0.6041 | 0.4848 | 0.8832 | 0.1649 | 0.2599 | 0.0893 |
| Heuristic soft-PU | 0.6155 | 0.5023 | 0.8717 | 0.1704 | 0.2566 | 0.1429 |
| Hard mixed-as-unknown | 0.6204 | 0.5022 | 0.8668 | 0.1720 | 0.2516 | 0.1709 |

The default reported operating point is calibrated from finite known
validation data, so test known coverage differs slightly between these runs.
At retrospective exact 95% test-known coverage, unknown rejection was `9.18%`
for nnPU, `11.73%` for soft-PU, and `11.99%` for hard labeling. These test-label
calibrations are diagnostic only and are not deployable thresholds. nnPU does
not beat the two simpler rejectors on unknown rejection or AUROC. It raises a
promising question about PU-risk estimation but is not an improvement yet.

The predeclared known-prior sensitivity check (same checkpoint and pool) gave:

| Assumed known prior | AUROC | FPR95 | OSCR | Reported unknown reject | Unknown reject at retrospective exact 95% coverage |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.5981 | 0.8865 | 0.1644 | 8.93% | 8.42% |
| 0.20 | 0.6041 | 0.8832 | 0.1649 | 8.93% | 9.18% |
| 0.30 | 0.5818 | 0.8783 | 0.1664 | 13.01% | 8.93% |

The reported operating-point rejection increases at prior `0.30`, but test
known acceptance also falls to `92.60%`; at exactly matched known coverage the
rejection is only `8.93%`. Thus this is primarily a threshold/score-scale
trade-off, not evidence of better separation. The ranking metrics are also
non-monotonic in the prior. Stop prior tuning here; nnPU has not shown a robust
gain over the simpler PU baselines.

## Implementation checks

- `python -m pytest -q tests --disable-warnings --maxfail=1`: **96 passed**.
- Python compilation and `git diff --check`: passed.
- The first malformed experiment command stopped before evaluation; the
  corrected command completed successfully.

## Decision and next step

Do not claim that either known pseudo-label anchoring or nnPU solves the
overlap. Stop adjusting their weights for now. The most informative next step
is a controlled `known_prior` sensitivity check (e.g. `0.10`, `0.20`, `0.30`)
on the same frozen checkpoint, paired with the fixed hard/soft-PU references,
then repeat only promising settings on another seed. Report AUROC, FPR95,
OSCR, known accuracy, and unknown rejection at the same validation-calibrated
known coverage; do not select settings using test labels. If prior changes do
not yield a stable gain, return to representation learning or a separately
held-out auxiliary unknown source rather than adding more threshold rules.
