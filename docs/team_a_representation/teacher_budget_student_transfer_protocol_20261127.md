# Teacher-budget KD transfer pilot (seed 2182)

## Question

Does extending the frozen teacher from 5 to 15 epochs improve what standard
logits KD transfers to the student, relative to the same student trained with
CE only? This is a Team A student/KD experiment. It does not change the teacher
recipe, introduce another KD loss, or modify the discovery pipeline.

The teacher-only pilot found a `+6.03 pp` known-validation top-1 gain for the
15-epoch teacher, but known-validation NLL worsened by `0.1229` on one
semantic-hard split. That mixed result makes student transfer the next useful
test; another teacher-epoch or KD-weight sweep is not justified.

## Arms and controls

All arms use semantic-hard CIFAR-100 60/40 seed `2182`, all 27,000 known
training samples, the same ResNet-18 student recipe, four student epochs,
identical initialization seed and data order, and known-validation-accuracy
checkpoint selection. Matched discovery-pool reservation is disabled so no
known training rows are withheld from this classification/KD-only gate.

| Arm | Classification objective | Frozen teacher |
|---|---|---|
| `ce_only` | CE (`alpha_kd=0`) | 5-epoch checkpoint loaded but contributes no KD loss |
| `standard_kd_e5` | CE + standard logits KD (`alpha_kd=1`) | 5-epoch checkpoint |
| `standard_kd_e15` | CE + standard logits KD (`alpha_kd=1`) | 15-epoch checkpoint |

All other numeric auxiliary-loss coefficients, feature KD, uncertainty-KD,
rejection KD, and discovery-pool training are disabled. Teacher configs are
checked to match in every field except epoch count and run/checkpoint metadata;
checkpoint class lists must match the registered split. The runner receives
both frozen checkpoint paths explicitly and does not call teacher training.

## Measurements and limits

The training history must show a nonzero KD loss/gradient and student update in
both KD arms. A deterministic known-validation audit reports accuracy, NLL,
within-class cosine distance, between/total scatter, and their paired deltas
from CE-only. Known labels are used only for the validation metrics and
geometry summaries; they do not select the method. Unknown/test labels are not
used for student objectives or checkpoint selection.

This is a one-split exploratory gate, not a confirmatory estimate. A promising
classification or representation result only justifies a separately recorded
downstream detection/discovery pilot under the fixed evaluation protocol; it
does not establish a general benefit. If the two KD arms do not show credible
known-class transfer or the mechanism audit fails, stop the teacher-budget KD
branch rather than tuning weights on seed `2182`.

## Reproduction

Runner: `scripts/run_teacher_budget_student_transfer.py` (tag
`team_a_teacher_budget_transfer_full_known_20261127`).

The runner refuses existing output paths, writes teacher checkpoint SHA-256
identifiers into its protocol record before training, and reports whether any
teacher-fit call occurred. Dry-run must show three arms and zero teacher-fit
calls before the experiment is started.

An initial run using the inherited matched-discovery setting exposed an
unexpected 21,600 training rows (338 batches) rather than the full known set.
It was interrupted before a student checkpoint was saved and is excluded from
analysis. The corrected runner disables matched discovery reservation and
reports the effective sample and batch counts during dry-run; the interrupted
protocol record is retained under the original tag and is not reused.

## Result and stage decision (2026-11-27)

The corrected full-known run completed four student epochs for all three arms.
On the same 3,000 known-validation rows:

| Student arm | Accuracy | NLL | Accuracy delta vs CE |
|---|---:|---:|---:|
| CE only | 72.47% | 0.9629 | reference |
| Standard KD, 5-epoch teacher | 68.77% | 1.1081 | -3.70 pp |
| Standard KD, 15-epoch teacher | 69.90% | 1.0960 | -2.57 pp |

Both KD arms reduce mean within-class cosine distance relative to CE
(`-0.0141` and `-0.0014`), but only the 5-epoch teacher increases the
between/total scatter ratio (`+0.0214`; the 15-epoch arm changes it by
`-0.0065`). These mixed geometry changes do not offset the consistent known
classification and NLL regressions. The 15-epoch teacher improves student
accuracy by 1.13 pp relative to the 5-epoch teacher, but remains 2.57 pp below
CE; this is one split and four training epochs, not evidence of a general
teacher-budget effect.

The mechanism audit confirms `alpha_kd=0` for CE and `alpha_kd=1` for both KD
arms. CE's separately measured raw KD component has nonzero gradients, but its
effective weighted KD loss and gradient are exactly zero. Both KD arms have
nonzero KD loss and component gradient in all four epochs, plus nonzero
classifier updates; thus KD is wired into the objective, but that does not
establish that its update is beneficial. The v2 audit-only rerun reproduced
all existing known-validation metrics and did not retrain a model or instantiate
test/unknown data. Its output is
`analysis/team_a_teacher_budget_transfer_full_known_20261127_s2182_v2.json`.

**Decision:** the predeclared classification/representation promotion gate is
not met. Do not proceed to unknown detection or clustering for this pilot, and
do not tune KD weights on seed `2182`. Preserve the result as evidence that
simply extending this weak teacher's training budget does not make standard KD
beat the matched CE student. This does not test uncertainty-KD and does not
justify changing the formal method or default configuration.
