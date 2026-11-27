# Teacher quality gate audit (2026-11-26)

## Purpose and scope

Before introducing another Team A KD variant, this read-only audit checks
whether the teacher used by the formal CE-only/standard-KD/uncertainty-KD
comparison is accurate enough on known classes to be a useful distillation
source. It is limited to teacher/student classification and KD interpretation.
No model was retrained, no checkpoint was changed, and no unknown/test labels
or downstream method internals were used.

## Evidence

The frozen models were evaluated deterministically on the complete known-train
split and the same 3,000-sample known-validation split. CE-only and
standard-KD values therefore use the exact samples used by the paired frozen
student audit on validation, while the train evaluation is an additional
known-only diagnostic.

| Seed | Split | Teacher ResNet-34 | CE-only ResNet-18 | Standard-KD ResNet-18 | KD minus CE |
|---:|---|---:|---:|---:|---:|
| 42 | known-train | 0.586778 | 0.818111 | 0.762037 | -0.056074 |
| 42 | known-validation | 0.509000 | 0.715333 | 0.682000 | -0.033333 |
| 123 | known-train | 0.663148 | 0.833593 | 0.775148 | -0.058444 |
| 123 | known-validation | 0.559333 | 0.725333 | 0.690000 | -0.035333 |
| 3407 | known-train | 0.581852 | 0.803296 | 0.772000 | -0.031296 |
| 3407 | known-validation | 0.516000 | 0.716000 | 0.707333 | -0.008667 |
| Mean, train | — | 0.610593 | 0.818333 | 0.769728 | -0.048605 |
| Mean, validation | — | 0.528111 | 0.718889 | 0.693111 | -0.025778 |

The teacher validation error ranges from `44.07%` to `49.10%`, and its
known-train accuracy is only `58.19%`–`66.31%`. On average, the CE student
exceeds the teacher by `19.08` percentage points on validation and `20.77`
points on train, while standard KD is `2.58` points below CE-only on
validation. This is an underpowered or undertrained teacher for the current
protocol, rather than a teacher that merely has a small generalization gap.

The validation error-overlap audit gives the more direct KD evidence. The
standard student's error rate conditional on a teacher error is
`42.50%/58.40%/52.69%` for seeds `42/123/3407`; the corresponding CE-only
rates are `31.19%/46.82%/41.74%`. Thus standard KD raises the error rate on the
teacher-error subset by `11.31/11.57/10.95` percentage points. On the
teacher-correct subset, standard KD reduces the student error rate by
`6.16/9.52/6.16` points. KD is therefore doing exactly the expected transfer
of a noisy teacher target: it helps teacher-correct rows and harms rows where
the teacher is wrong.

## Interpretation limits

This evidence does **not** invalidate the paired comparison: within each seed,
the student arms share the same teacher checkpoint and differ by the declared
KD treatment. It does limit the claim that the result measures KD under a
competent teacher. Nor does low validation accuracy prove the teacher is
overfit or identify the cause: these are semantic-hard class splits, and the
audit has not compared teacher train accuracy, per-class validation accuracy,
teacher/CE error overlap, calibration, or alternative teacher training
protocols. The teacher quality gap is a high-priority confound for method
interpretation, not yet a causal diagnosis.

## Decision and next gate

Do not add or scan another KD loss based on the current evidence. Keep the
formal results unchanged as a valid comparison under the existing teacher.
Before any new KD treatment, run a read-only, known-only audit on these frozen
checkpoints to report:

1. teacher train versus validation accuracy under deterministic evaluation;
2. per-class teacher accuracy and its variation across the three splits;
3. teacher-correct / teacher-error overlap with CE-student correctness;
4. teacher-to-student per-sample target information, including whether the
   teacher provides useful non-ground-truth ranking on CE-student errors.

These are diagnostics only, not checkpoint-selection or tuning signals. If the
teacher provides little complementary information, stop KD-variant work and
retain CE-only as the representation control. If a complementary signal is
present, define a single treatment that uses it, then require a fresh paired
split and joint gates for student representation, detection, and discovery.
Any teacher retraining must be evaluated as a separate protocol factor and
must not be conflated with a KD-loss change.

The immediate next stage is therefore a teacher-protocol audit or a clean
teacher-quality replication, not another student KD loss. It should compare a
fixed teacher recipe against the existing teacher on a fresh split or a
predeclared subset, report known-train and known-validation accuracy, and keep
CE-only/standard-KD/uncertainty-KD student arms unchanged. A teacher recipe is
eligible for later KD experiments only if it materially exceeds the current
teacher on known validation and provides complementary correctness relative to
CE-only. No unknown-label metric may select the teacher.

## Follow-up: training-budget pilot (2026-11-27)

A single fresh semantic-hard split (seed `2182`) compared the matched 5- and
15-epoch teacher checkpoints. The only intended treatment was the training
budget. Deterministic evaluation used the same 27,000 known-train and 3,000
known-validation samples. The corrected audit constructs only known-filtered
training datasets and does not instantiate the test split or an unknown-pool
loader. CIFAR training targets are consulted only to exclude non-known rows
from the known loaders; no unknown metric is used for selection or tuning.

| Budget | Known-train accuracy | Known-validation accuracy | Known-validation NLL |
|---:|---:|---:|---:|
| 5 epochs | 0.599778 | 0.533000 | 1.837632 |
| 15 epochs | 0.797704 | 0.593333 | 1.960540 |
| Delta | +0.197926 | +0.060333 | +0.122908 (worse) |

The longer run improves validation top-1 by 6.03 percentage points, but the
much larger train gain and worse validation NLL show that it does not uniformly
repair teacher quality. This is one split and does not establish a generally
better teacher recipe. No student was trained, so nothing is yet known about
whether this gain transfers through standard KD, affects the student
representation, or reaches unknown detection/discovery.

The current paired-student runner is not yet safe for this comparison: it
derives the teacher checkpoint path from `--tag` and seed, reusing a file only
at that exact location; otherwise it calls `fit_teacher`. It has no explicit
external frozen-checkpoint argument. Add such an input together with source
config/checkpoint provenance validation before running students. The next
comparison should hold seed/split fixed and include CE-only, standard KD from
the 5-epoch teacher, and standard KD from the 15-epoch teacher. Evaluate the
known-only student classification/representation gate first; only proceed to
the existing downstream detection/discovery protocol if transfer is credible.
Do not use unknown labels for model selection and do not overwrite formal runs.

The first audit implementation used the generic data-bundle constructor, which
also instantiated unknown-pool and test dataset objects even though neither was
iterated. Its earlier `unknown_test_samples_loaded: false` field overstated the
isolation guarantee. The corrected result is
`analysis/team_a_teacher_budget_s2182_known_metrics_20261127_v2.json`; the
earlier JSON is superseded for provenance claims.

## Reproducibility

Sources: the frozen checkpoints under
`runs/team_a_formal_ce_kd_20261122_s{42,123,3407}_{teacher,ce_only,standard,uncertainty}`
and the machine-readable reports
`analysis/team_a_teacher_quality_gate_s{42,123,3407}_20261126.json`.
The audit runner is `scripts/audit_teacher_quality_gate.py`. This note records
a read-only comparison; no training artifacts were changed.
