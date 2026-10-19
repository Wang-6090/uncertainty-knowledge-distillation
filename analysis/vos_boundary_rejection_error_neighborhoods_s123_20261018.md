# Post-hoc error-neighborhood audit for matched rejection-BCE arms (seed 123)

## Protocol

This is a descriptive analysis of the already completed seed-123 control and
treatment checkpoints. It reuses each arm's existing test scores, its own
validation-calibrated threshold, and the same 10-nearest-known-training-feature
bank. Test labels are used only after inference to summarize accepted/rejected
cohorts; they were not used to train, calibrate, or choose either arm. No model
was retrained and no threshold was changed.

The neighborhood script now accepts `--first-name` and `--second-name`, while
retaining its historical `five_epoch`/`ten_epoch` defaults. This avoids
mislabeling arbitrary paired checkpoints as training durations.

## Observations

| Measure | BCE off | BCE 0.05 |
|---|---:|---:|
| Unknown false accepts | 2,072 / 4,000 (51.80%) | 2,290 / 4,000 (57.25%) |
| Mean nearest-known cosine distance among false accepts | 0.2090 | 0.2230 |
| Top-1 known neighbor agrees with predicted known class | 34.03% | 32.62% |
| Mean top-10 purity for predicted known class | 32.09% | 31.59% |

Highest false-accept rates in the control include forest (76%), lizard (75%),
lamp (72%), crocodile (70%), kangaroo (67%), beaver (64%), snake (64%), dolphin
(61%), and porcupine (61%). In the treatment, lamp (82%), snake (81%), clock
(78%), kangaroo (76%), fox (74%), lizard (74%), possum (71%), beaver (70%),
forest (69%), and porcupine (67%) are among the worst. Each class has 100 test
examples in this protocol; these are diagnostic ranks, not independent
generalization estimates.

Across false accepts, frequently nearest known classes overlap substantially
between arms: squirrel, shrew, rabbit, mouse, shark, worm, and ray are common
neighbors. The identities and ordering shift, but the broader picture is
consistent: failures concentrate in particular semantic/local neighborhoods,
not uniformly across unknown classes.

Notably, treatment false accepts have a *larger* mean nearest-known distance
than control, yet more unknowns cross the treatment's own acceptance threshold.
This reinforces that a simple global score/threshold does not consistently
translate local feature geometry into rejection. It does not prove that adding
a nearest-neighbor score would help: prior support-score comparisons were
negative, and this audit is post-hoc.

## Decision

The BCE treatment is not rescued by a classwise view: its overall unknown
false-accept count rises by 218 (5.45 percentage points), and several hard
classes worsen. Do not keep this loss as the next algorithm direction.

Next, analyze *validation-only* class-conditioned score calibration and local
support features on a frozen checkpoint before training another model. A
candidate should proceed only if the calibration relation is estimated without
unknown test labels and improves ranking/overlap or fixed-coverage rejection
without harming known-class utility. Avoid a classwise threshold fitted on test
labels; the historical classwise scoring ablation was also negative.

## Artifacts

- Raw report: `analysis/vos_boundary_rejection_error_neighborhoods_s123_20261018.json`
- Checkpoints/reports: `runs/vos_boundary_rejection_matched_s123_e4_{control,interclass_mix}[_discover]/`
- Command: `python scripts/analyze_error_neighborhoods.py --split-path splits_cifar100_protocols/cifar100_60_40_semantic_isolated.json --seed 123 --rejection-feature-dim 128 --five-epoch-ckpt runs/vos_boundary_rejection_matched_s123_e4_control/student.pt --five-epoch-detail-dir runs/vos_boundary_rejection_matched_s123_e4_control_discover --ten-epoch-ckpt runs/vos_boundary_rejection_matched_s123_e4_interclass_mix/student.pt --ten-epoch-detail-dir runs/vos_boundary_rejection_matched_s123_e4_interclass_mix_discover --first-name control_bce_off --second-name treatment_bce_0p05 --output analysis/vos_boundary_rejection_error_neighborhoods_s123_20261018.json`
