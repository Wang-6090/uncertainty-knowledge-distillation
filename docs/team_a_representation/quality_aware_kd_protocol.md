# Quality-Aware Logits KD: staged protocol

## Why this stage exists

The existing uncertainty-KD branch uses a detached `exp(-u)` weight on the
whole per-sample KL. Audits show that its effective sample size stays close to
uniform and that the learned uncertainty head is not a reliable proxy for
teacher classification correctness. Repeating the same uncertainty source or
weight scan is therefore not justified.

This stage tests a different mechanism: use the known training label to measure
the frozen teacher's true-class probability, and use that probability only to
scale the soft-target KL. The hard-label CE term remains unchanged. The
teacher uncertainty head, student uncertainty head, rejection score and test
labels are separate signals and are not mixed into this weight.

The idea is a quality-aware distillation hypothesis, not an established result
for novel-class discovery. It is motivated by the general confidence-aware KD
pattern and must be evaluated against the existing standard-KD control.

## Objective

For a known training sample with label `y`, define

```text
q_i = p_teacher(y | x_i)
w_i = q_i^power, clipped below by quality_floor
L_QKD = mean_i[w_i * KL(p_student^T || p_teacher^T)]
```

With `mean_normalized`, `w_i` is divided by its batch mean so the expected KD
scale remains comparable to standard KD. The weight is detached. The default
configuration remains `kd_mode=standard`; this branch is opt-in as
`kd_mode=quality_aware`.

## Stage gates

1. Implementation gate: explicit formula, invalid-label, finite-loss,
   student-gradient and detached-teacher tests must pass.
2. Pipeline gate: a toy run must report `kd_quality_weight_applied=true`, a
   nonzero weight spread/ESS diagnostic, and a saved student checkpoint.
3. Research pilot: compare standard logits-KD, unweighted DKD and
   quality-aware logits-KD on one fresh semantic-hard seed. Keep teacher,
   student initialization, budget, auxiliary losses, checkpoint selection,
   detector, calibration, candidate pool and Auto-K fixed. Unknown labels are
   used only in final evaluation.
4. Promotion: only repeat on fresh seeds if quality-aware KD improves at least
   one ranking/fixed-coverage metric and one candidate/clustering metric while
   avoiding a material known-accuracy or candidate-purity loss. No parameter
   scan is allowed before this gate.

The pilot is exploratory. A toy smoke test is not evidence of improved
open-set detection or clustering.

## Audit fields

Each student epoch must retain `kd`, `kd_quality_weight_applied`,
`kd_quality_floor`, `kd_quality_power`, `kd_quality_weight_mode`, weight mean,
weight standard deviation and ESS fraction. The implementation must keep the
teacher logits detached and must not use the uncertainty head as a hidden
substitute for the quality weight.

