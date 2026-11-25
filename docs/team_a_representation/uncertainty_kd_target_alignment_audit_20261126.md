# Uncertainty-KD target-alignment audit (2026-11-26)

## Question and scope

This frozen-checkpoint audit asks what uncertainty-weighted logits KD changes
in the student: does it make the student closer to the detached teacher target
on trusted samples, and does it selectively change high-uncertainty samples?
It is within Team A's scope because it separates the KD weight mechanism from
the student uncertainty head and from the deployed unknown score.

No model was retrained. The audit used the existing formal CE-only
`42/123/3407` checkpoints and the two fixed-coefficient view-consistency
confirmation checkpoints `2178/2179`. For each seed, the same 3,000 known
validation samples were evaluated with the CIFAR transform. The teacher's
known labels were used only for post-hoc correct/error grouping; unknown/test
labels were not loaded and no result was used for training, calibration,
checkpoint selection, or parameter tuning.

The measured quantity is the per-sample temperature-2 student-to-teacher
logits KL. The reported delta is uncertainty-KD minus standard-KD for the
same seed and validation sample order.

## Results

| Seed | Teacher error | All KL delta | Teacher-correct delta | Teacher-error delta | Q1 delta | Q2 delta | Q3 delta | Q4 delta | Q5 delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 49.10% | -0.00642 | -0.00229 | -0.01070 | -0.06412 | -0.02992 | -0.01708 | +0.01990 | +0.05911 |
| 123 | 44.07% | +0.16625 | +0.08718 | +0.26661 | +0.05482 | +0.09608 | +0.15482 | +0.25322 | +0.27231 |
| 3407 | 48.40% | +0.08350 | +0.01088 | +0.16092 | -0.03931 | -0.00190 | +0.08483 | +0.15311 | +0.22076 |
| 2178 | 46.40% | -0.00089 | +0.01235 | -0.01617 | -0.07382 | -0.02806 | -0.00882 | +0.02127 | +0.08501 |
| 2179 | 46.47% | +0.02832 | +0.00465 | +0.05559 | -0.02968 | +0.00347 | +0.01866 | +0.02470 | +0.12443 |

Q1 through Q5 are teacher-uncertainty quintiles from low to high. The high
uncertainty quintile has a positive KL delta in all five seeds, while Q1 is
negative in four of five seeds. The all-sample delta changes sign across
seeds, and the teacher-error delta also changes sign. Thus the stable effect
is reallocation of teacher-target fidelity by uncertainty, not a stable global
improvement in target alignment.

## Interpretation

The current `exp(-u)` mean-normalized uncertainty-KD does what its algebra
predicts: it gives less optimization mass to high-uncertainty teacher targets,
so the student is allowed to move farther from them; it gives relatively more
mass to low-uncertainty targets, so those rows usually remain closer. This is
mechanistically real, but it does not establish that the direction is useful
for unknown detection or novel-class discovery. In particular, the high-
uncertainty rows contain many teacher errors, but uncertainty alone does not
specify the replacement target or the unknown-class partition.

This explains why a nonzero KD gradient and nonuniform weights were not enough
to produce stable downstream gains. The uncertainty-KD arm changes which
known-class teacher targets are followed; it does not create a validated
representation objective for the 40 unknown classes. The result also argues
against another blind weighting-function scan: the primary limitation is
target definition and transfer, not whether the current weight is numerically
active.

## Decision

Do not change the default `standard-KD`, promote uncertainty-KD, or scan another
weight function from this audit. Target-repair and teacher-error-repair have
already tested label-anchored alternatives and improved candidate quality in
some seeds while repeatedly worsening Auto-K, so they remain opt-in ablations.
The next Team A work should begin only with a new, explicitly stated target
that can be audited using known labels and has a predeclared transfer gate;
otherwise the correct action is to keep standard-KD as the baseline and report
uncertainty-KD as a real but unstable reweighting ablation.

Machine-readable reports:

- `analysis/team_a_formal_ce_kd_s42_kd_alignment_20261126.json`
- `analysis/team_a_formal_ce_kd_s123_kd_alignment_20261126.json`
- `analysis/team_a_formal_ce_kd_s3407_kd_alignment_20261126.json`
- `analysis/team_a_viewconsistency_2178_kd_alignment.json`
- `analysis/team_a_viewconsistency_2179_kd_alignment.json`

The audit script is `scripts/audit_frozen_kd_target_alignment.py`.
