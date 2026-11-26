# Frozen student classification impact audit (2026-11-26)

## Question and scope

This audit asks which known-validation samples account for the formal
CE-only/standard-KD/uncertainty-KD differences. It is limited to Team A's
student classification representation and distillation mechanism:

- whether standard-KD changes the student's known-label classification;
- whether uncertainty-KD changes that effect relative to standard-KD;
- whether the change is concentrated on samples for which the teacher is
  correct or incorrect, on high-uncertainty samples, or on particular known
  classes.

The audit does not modify or evaluate the rejector, candidate selection,
score fusion, Auto-K, or clustering algorithm. It does not retrain a model.
Unknown/test labels were not loaded.

## Protocol checks

The existing formal frozen checkpoints were used for semantic-hard CIFAR-100
60/40 splits and seeds `42/123/3407`:

- one split-matched teacher;
- CE-only, standard-KD, and uncertainty-KD student checkpoints;
- the same known validation construction, 60 known classes, image size,
  CIFAR evaluation transform, and known split protocol;
- deterministic validation order with 3,000 samples per seed.

The audit stops before evaluation if the split, seed, transform, protocol
fields, validation sample order, or labels do not match. The validation labels
are used only after inference to calculate known classification metrics and to
form teacher-correct/error and class groups. They are not used for training,
calibration, checkpoint selection, or parameter tuning.

The implementation is
[`scripts/audit_frozen_student_classification_impact.py`](../../scripts/audit_frozen_student_classification_impact.py).
Machine-readable outputs are local and ignored by repository policy:

- `analysis/team_a_formal_ce_kd_s42_classification_impact_20261126.json`
- `analysis/team_a_formal_ce_kd_s123_classification_impact_20261126.json`
- `analysis/team_a_formal_ce_kd_s3407_classification_impact_20261126.json`

## Overall known-validation result

The table reports accuracy and NLL on the same known validation samples. NLL
is cross-entropy from the student's logits. Deltas are paired against the
CE-only checkpoint within the same seed.

| Seed | Arm | Accuracy | NLL | Accuracy delta vs CE | NLL delta vs CE |
|---:|---|---:|---:|---:|---:|
| 42 | CE-only | 0.7157 | 0.9554 | — | — |
| 42 | standard-KD | 0.6820 | 1.0620 | -0.0337 | +0.1066 |
| 42 | uncertainty-KD | 0.6730 | 1.1026 | -0.0427 | +0.1472 |
| 123 | CE-only | 0.7253 | 0.9612 | — | — |
| 123 | standard-KD | 0.6900 | 1.0981 | -0.0353 | +0.1369 |
| 123 | uncertainty-KD | 0.6960 | 1.0406 | -0.0293 | +0.0794 |
| 3407 | CE-only | 0.7160 | 0.9677 | — | — |
| 3407 | standard-KD | 0.7073 | 1.0502 | -0.0087 | +0.0825 |
| 3407 | uncertainty-KD | 0.7193 | 1.0033 | +0.0033 | +0.0355 |

Across the three overlapping semantic-hard splits, standard-KD lowers known
validation accuracy in all three seeds. Uncertainty-KD changes accuracy
relative to standard by `+0.0060`, `+0.0120`, and `-0.0090`; it remains
below CE-only in two seeds and has higher NLL than CE-only in every seed. This
is a local classification diagnostic, not evidence of improved discovery.

## Reliability and uncertainty groups

The following values are mean paired deltas versus CE-only. The uncertainty
groups are teacher uncertainty quintiles from low (`Q1`) to high (`Q5`).

| Group | Standard accuracy delta | Standard NLL delta | Uncertainty-KD accuracy delta | Uncertainty-KD NLL delta | Seeds where UKD improves over standard |
|---|---:|---:|---:|---:|---:|
| Teacher-correct | +0.0585 | -0.1140 | +0.0551 | -0.1187 | 0/3 by accuracy |
| Teacher-error | -0.1192 | +0.3568 | -0.1088 | +0.3146 | 2/3 by accuracy |
| Q1, lowest uncertainty | -0.0100 | +0.0108 | -0.0122 | +0.0274 | 1/3 by accuracy |
| Q5, highest uncertainty | -0.0428 | +0.1817 | -0.0233 | +0.0971 | 2/3 by accuracy |

The stable pattern is that standard-KD helps samples where the teacher is
correct but hurts samples where the teacher is wrong, and the latter group is
large enough to dominate the overall result. Uncertainty-KD partly reduces
the teacher-error damage in seeds 123 and 3407 and reduces the Q5 damage on
those seeds. Seed 42 reverses this pattern: its teacher-error and Q5 NLL are
worse under uncertainty-KD than standard-KD. Therefore the uncertainty weight
has a plausible local effect, but it does not provide a stable correction rule
across the three splits.

## Class-wise result

The class-level deltas are included in each JSON report. Several classes show
large changes, but the affected classes differ by seed. For example, the
largest uncertainty-KD accuracy losses relative to CE-only occur in classes
29/55/27 for seed 123, 47/4/27 for seed 3407, and 44/22/40 for seed 42.
Some other classes improve strongly in each seed. This dispersion does not
support a single class-independent training adjustment, and the class sets
overlap across seeds, so their means are descriptive rather than independent
statistical evidence.

## Interpretation and decision

This audit refines the earlier mechanism conclusion:

1. Standard-KD is active, but its transfer improves teacher-consistent known
   examples while transferring teacher mistakes strongly enough to worsen the
   aggregate student classification result.
2. Uncertainty-KD does not merely turn KD on or off. It changes the balance of
   those effects and often reduces high-uncertainty damage, but the direction
   is seed-sensitive.
3. The result is consistent with the prior target-alignment audit: uncertainty
   reweights the existing teacher target; it does not supply a replacement
   target for teacher-error samples or information about the 40 unseen
   classes.
4. No new weighting scan or default change is justified. Standard-KD remains
   the clean baseline and uncertainty-KD remains a mechanism-verified but
   performance-unstable ablation.

The next Team A experiment should be considered only if it states a new target
for teacher-error samples in advance and includes a transfer gate covering
known classification, the KD mechanism, and the existing downstream report.
This audit alone does not justify another training run.

## Validation

- Python compilation of the new audit script passed.
- The three-seed runs passed protocol, sample-order, and label-alignment
  checks.
- Each report contains exactly 3,000 known-validation samples per arm.
- No training, calibration, checkpoint selection, unknown label loading, or
  downstream algorithm change was performed.
