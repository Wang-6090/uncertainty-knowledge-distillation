# 2026-10-06 direction audit: feature margin under a pure-unknown pool

## Why this experiment was run

Recent full-data, 10-epoch matched runs showed that extending the baseline
training budget improved both known classification and open-set metrics, while
the uncertainty-weighted feature-margin treatment regressed against the
10-epoch mixed-pool baseline. This makes it important to distinguish two
possibilities before proposing another loss:

1. feature-margin itself is not useful at a fair training budget; or
2. its mixed-pool soft weighting is harmed by known samples inside the
   unlabeled discovery pool.

The key project problem remains known/unknown representation and score
overlap. A higher rejection rate alone is not sufficient evidence: it can be
caused by rejecting more known examples.

## Controlled training runs

All runs used CIFAR-100 random 60/40 classes, seed 2026, full data, the same
ResNet-34 teacher checkpoint, ResNet-18 student, pretrained initialization,
batch size 64, 10 epochs, and the same optimizer/augmentation and known-only
checkpoint selection. The discovery pool was pure novel-class data, selected
using training class labels; this is an oracle-filtered diagnostic upper bound,
not a realistic mixed-unlabeled deployment protocol. Test labels were not used
for training, checkpoint selection, score selection, or threshold fitting.

Three arms were trained:

| Arm | Discovery loss change |
| --- | --- |
| Pure-unknown baseline | None |
| Fixed feature-margin | `alpha_discovery_feature_margin=0.05`, cosine margin `0.2` |
| Uncertainty-weighted feature-margin | `alpha_discovery_uncertainty_feature_margin=0.05`, margin `0.2` |

The pure-unknown baseline is shared between the two paired comparisons. The
fixed-margin arm is a distinct objective from the mixed-pool uncertainty-weighted
margin; do not treat them as identical algorithms.

All three checkpoints were evaluated on the complete 10,000-image open test
set using the same `normalized_entropy_min_class_knn` score, feature kNN
`k=10`, MC dropout `8`, and threshold calibrated on known validation data for
95% target known coverage. Clustering was skipped because this experiment asks
about detection and feature separation, not novel-class clustering.

## Open-set results

| Pure-unknown arm | AUROC | AUPR | FPR95 | OSCR | Known acc (all known) | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.6891 | 0.5561 | 0.7638 | 0.4182 | 50.48% | 95.65% | 11.18% |
| Fixed feature-margin | 0.6984 | 0.5701 | 0.7565 | 0.4425 | 52.83% | 94.87% | 14.40% |
| Uncertainty-weighted feature-margin | 0.7043 | 0.5748 | 0.7332 | 0.4593 | 54.80% | 95.62% | 12.75% |

Against the pure-unknown baseline, the uncertainty-weighted arm improves AUROC
by `0.0151`, FPR95 by `0.0307`, OSCR by `0.0411`, and all-known accuracy by
`4.32` percentage points. Known acceptance is essentially unchanged
(`-0.03pp`), while unknown rejection increases by `1.58pp`. This is a coherent
single-seed positive signal, not a solution: `87.25%` of unknowns are still
accepted at the chosen operating point.

The fixed-margin arm also improves several metrics, but is weaker than the
uncertainty-weighted arm on this seed. This supports examining soft weighting
further, but does not establish a general benefit.

## Feature-overlap diagnostics

The diagnostic extracts normalized embeddings from all 27,000 known training
images and all 10,000 test images. Test labels are used only to divide the
reported known/unknown distributions after inference. They do not fit any
prototype, model, score, or threshold.

| Distance source | Baseline AUROC / overlap | Fixed-margin AUROC / overlap | Uncertainty-margin AUROC / overlap |
| --- | ---: | ---: | ---: |
| Classifier prototypes | 0.6794 / 0.7268 | 0.6773 / 0.7255 | 0.6858 / 0.7167 |
| Empirical class centroids | 0.6459 / 0.7825 | 0.6709 / 0.7456 | 0.6759 / 0.7403 |
| Nearest known training sample | 0.6820 / 0.7297 | 0.6961 / 0.7093 | 0.6992 / 0.6962 |

For the uncertainty-weighted arm, all three diagnostic AUROCs rise and all
three histogram overlaps decrease. The centroid and nearest-sample distances
show the clearest effect. The classifier-prototype distance changes only
slightly. This is direct evidence that this training run changed the measured
embedding geometry, rather than merely shifting the final detector threshold;
however, overlap remains large (about `0.70–0.74`).

## Direction audit and limits

This result rehabilitates uncertainty-weighted feature-margin as a candidate
under a clean oracle pool, but it does **not** prove that known-sample
contamination caused the mixed-pool failure. The pure and mixed protocols also
differ in other relevant ways:

- mixed training uses an estimated/observed known prior and nnPU uncertainty
  training (`alpha_discovery_uncertainty_pu=0.1` in the 10-epoch baseline),
  while the pure-pool run correctly disables that mixed-PU objective;
- mixed data construction reserves part of known training data into the
  unlabeled pool, changing the number of supervised known examples;
- the pure pool itself is selected using class labels and therefore is an
  oracle upper-bound condition.

Thus the within-pure-pool comparison is a valid paired test of the added
uncertainty-weighted margin under that protocol, but the cross-protocol
comparison with mixed training is not a one-factor causal test of pool
contamination. This is exactly why past claims based only on a run name or a
single aggregate metric need scrutiny.

## Decision and next experiment

- Keep the margin disabled by default. Preserve it as a promising ablation,
  not as a promoted solution.
- Do not sweep its coefficient or combine it with more losses yet.
- Next run a small, explicitly matched pool-composition study: use the same
  known supervised subset and teacher/student initialization; create the
  unlabeled pool from the same number of samples, comparing a pure-unknown
  pool with a mixed pool containing a controlled known fraction; keep nnPU
  disabled in both arms for this first contamination-only comparison; then
  compare baseline and uncertainty-margin within each pool condition.
- After establishing that controlled contrast, re-enable the mixed nnPU
  objective as a separate factor. Report the complete factorial contrasts,
  more than one seed, known acceptance, unknown rejection, OSCR, and embedding
  overlap—not just AUROC.
- Any pure-unknown result remains diagnostic and must not be presented as the
  expected deployment result.

## Artifacts

- Training: `runs/audit_pure_unknown_baseline_s2026_e10/`,
  `runs/audit_pure_unknown_margin_s2026_e10/`,
  `runs/audit_pure_unknown_uncertainty_margin_s2026_e10/`.
- Detection: corresponding `_detect/` directories.
- Full embedding diagnostics:
  `analysis/pure_unknown_margin_overlap_s2026_e10_full.json` and
  `analysis/pure_unknown_uncertainty_margin_overlap_s2026_e10_full.json`.
- Prior mixed 10-epoch audit:
  `analysis/training_budget_treatment_overlap_s2026_e10_full.json` and
  `analysis/uncertainty_margin_replication_s2026_20261006.md`.
