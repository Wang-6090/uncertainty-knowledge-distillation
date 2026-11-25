# Known-label classification view-stability audit

## Question and scope

Several recent KD variants changed the student's known-class geometry, but
none produced stable downstream novel-class discovery gains. Before adding
another training objective, this audit asks whether the student classification
representation is unstable under the existing CIFAR train augmentation, and
whether that instability concentrates on known-validation samples that are
already hard to classify. This is within Team A's student representation/KD
scope. It does not change the rejector, discovery candidate selection, score,
calibration, Auto-K, or clustering.

## Procedure

For each existing frozen student checkpoint, the same known-validation image
was sampled twice using two independent draws of the repository's existing
CIFAR train transform (resize, random crop, horizontal flip). We measured
classification backbone feature cosine, symmetric Jensen-Shannon divergence of
the logits, and prediction flips. Labels were used only after inference to
form the groups `both_views_correct` and `at_least_one_view_wrong`.

No unknown/test labels were loaded. No threshold, checkpoint, training
parameter, or downstream method was selected from these results.

## Results

| Seed / arm | Feature cosine, all | Feature cosine, both correct | Feature cosine, at least one wrong | Logit JS, all | Prediction flips |
|---|---:|---:|---:|---:|---:|
| 2173 standard | 0.9475 | 0.9560 | 0.9325 | 0.0307 | 16.10% |
| 2173 uncertainty | 0.9488 | 0.9576 | 0.9339 | 0.0318 | 17.57% |
| 2176 standard | 0.9457 | 0.9567 | 0.9285 | 0.0310 | 18.27% |
| 2176 uncertainty | 0.9450 | 0.9570 | 0.9272 | 0.0325 | 18.10% |
| 2176 target-repair | 0.9412 | 0.9520 | 0.9222 | 0.0423 | 16.57% |

The error-associated stability gap repeats: feature cosine is lower and logit
JS is higher when at least one augmented view is wrong. Standard-to-uncertainty
changes are inconsistent: uncertainty-KD slightly raises cosine on seed 2173
but lowers it on 2176, while logit JS increases on both seeds. Target-repair
has fewer prediction flips on 2176, but its error-group feature cosine is
lower and JS is substantially higher than standard; its known-validation
correctness improvement therefore is not evidence of better augmentation
invariance.

## Decision and next stage

The audit justifies investigating a new, narrowly scoped hypothesis:
known-label classification-view consistency may improve the student feature
stability that standard/uncertainty KD do not reliably improve. It does not
justify a performance claim or a new training run yet.

The next implementation must be default-off and paired: standard and
treatment must consume the same two-view augmentation stream, while only the
treatment adds a feature-consistency loss on the shared student classification
backbone. CE, standard/uncertainty logits KD, teacher detach rules, checkpoint
selection, and all discovery evaluation must remain unchanged. The
implementation gate must verify finite loss, student-only gradients, paired
view alignment, and zero behavior when the coefficient is zero. Only after
that gate should one fresh semantic-hard pilot be considered.

Machine-readable diagnostics are locally retained under gitignored `analysis/`:
`team_a_classification_view_stability_20261125_s2173.json` and
`team_a_classification_view_stability_20261125_s2176.json`.
