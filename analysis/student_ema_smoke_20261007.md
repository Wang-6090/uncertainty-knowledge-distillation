# Student EMA smoke test (2026-10-07)

## Purpose

The new `--student-ema-decay` option is intended to reduce representation
oscillation across epochs. This test checks only that the optional path can
train, evaluate, and save a checkpoint; it is not an efficacy experiment.

## Procedure

- Dataset: toy, 10 known classes, 32 train and 16 validation samples.
- Student: one epoch with `--student-ema-decay 0.9`.
- Teacher: a fresh one-epoch toy checkpoint created in
  `runs/audit_ema_smoke_teacher/`.
- Device: CUDA.

## Result

The student completed training and saved
`runs/audit_ema_smoke/student.pt`. The EMA path updated the evaluated model,
selected a checkpoint, and completed prototype/statistics saving without an
exception.

The first attempt used a missing historical `runs/demo_teacher/teacher.pt` and
failed before training; this was an environment path issue, not an EMA failure.

## Decision

Keep EMA opt-in with default decay `0`. A CIFAR-100 paired experiment is still
required before deciding whether it improves AUROC, FPR95, unknown rejection,
or feature-overlap diagnostics.
