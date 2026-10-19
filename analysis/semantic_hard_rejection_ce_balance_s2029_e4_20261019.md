# Semantic-hard rejection-branch known-CE balance pilot (2026-10-19)

## Question and predeclared comparison

The independent rejection-branch gradient audit found that known rejection CE
could dominate several unknown-side rejection objectives in a small smoke run.
Earlier semantic-isolated paired runs gave a weak, mixed signal for lowering
`alpha_rejection_known_ce`; this pilot checks whether that signal transfers to
the harder semantic-hard protocol.

Compare the complete existing seed-2029 baseline (`alpha_rejection_known_ce =
1.0`) with one treatment (`0.25`). Both use the semantic-hard CIFAR-100 60/40
split, same seed and teacher, 4 student epochs, the same mixed-pool and all
other saved training settings, and the same discovery configuration. The
discovery uses `feature_rejector` with `support_augmented` features and
known-validation-only calibration targeting 95% known coverage. No other loss,
detector, threshold, pool, or calibration setting was changed. The treatment
was run with the available RTX 4060 CUDA device.

The treatment configuration confirms the intended CE change; the other
configuration differences are output paths and validation-derived checkpoint
selection values. The best checkpoint is selected using known validation
accuracy, not test labels. Test-set results below are descriptive final
evaluation for this preplanned comparison and are not used to train, calibrate,
or tune the treatment.

## Results

| Metric | CE=1.0 baseline | CE=0.25 | Change |
|---|---:|---:|---:|
| AUROC | 0.79551 | 0.80304 | +0.00753 |
| FPR95 (lower is better) | 0.60517 | 0.60917 | +0.00400 |
| OSCR | 0.59105 | 0.59326 | +0.00221 |
| Test known acceptance | 94.7667% | 94.7500% | -0.0167 pp |
| Unknown rejection at validation-calibrated operating point | 25.625% | 29.475% | +3.850 pp |
| Accepted-known classification accuracy | 71.1748% | 70.9235% | -0.2513 pp |
| All-known classification accuracy | 69.6000% | 69.3833% | -0.2167 pp |
| Score KS distance | 0.44683 | 0.46375 | +0.01692 |
| Unknown-below-known-q95 overlap proxy (lower is better) | 74.900% | 72.000% | -2.900 pp |

The realized validation known-acceptance rate was 94.9387% in both runs. Test
known acceptance was also nearly identical, so the higher unknown rejection
does not appear to come from a material increase in known false rejection.
Ranking and both reported distribution-separation diagnostics move in the
favorable direction, but FPR95 is slightly worse and known classification
utility falls slightly. This is a **promising but mixed one-seed signal**, not
a resolution of the core overlap problem: 72.0% of unknown scores remain below
the known q95, and 29.475% unknown rejection still means most unknown samples
are accepted at this operating point.

## Decision and next step

Do not change the default CE weight based on one seed. Unlike the previously
stopped full-pool kNN-weight route, this result is worth one fixed, independent
semantic-hard training-seed replication because the validation-calibrated
unknown rejection and overlap proxy both improved while test known acceptance
was stable. Predeclare the same `1.0 vs 0.25` comparison and all settings on a
new seed; do not scan intermediate weights. Retain AUROC, FPR95, OSCR, KS,
overlap, validation-calibrated unknown rejection, test known acceptance, and
known classification accuracy as the joint criteria. Stop this route if the
operating-point/overlap gain does not replicate or if known utility degrades
materially. Testing another seed is replication, not parameter selection; do
not inspect test outcomes to choose a coefficient.

## Artifacts

- Baseline: `runs/semantic_hard_knn_support_train_s2029_e4_budgeted_baseline/`
- Baseline discovery: `runs/semantic_hard_knn_support_train_s2029_e4_budgeted_baseline_discover/`
- Treatment: `runs/rejection_ce_balance_s2029_ce0p25_semantic_hard_s2029_e4/`
- Treatment discovery: `runs/rejection_ce_balance_s2029_ce0p25_semantic_hard_s2029_e4_discover/`
- Reproduction entry point: `scripts/run_rejection_ce_balance_pilot.py`
- Method motivation/gradient audit: `analysis/rejection_branch_gradient_audit_20261013.md`

This is a diagnostic four-epoch pilot on one semantic-hard class split and one
training seed. It is not an independent-split confirmation, and the test
metrics should not be reused for further tuning.

## Fixed training-seed replication (seed 2030)

Because the first seed showed a favorable operating-point/overlap signal with
stable known acceptance, one replication was run with the same fixed 1.0 vs
0.25 coefficients, same semantic-hard split, teacher, 4 epochs, pool and
discovery/calibration protocol. Only the runtime seed and resulting model
initialization/order changed between seed-paired experiments. No weight sweep
was performed.

| Seed=2030 | CE=1.0 | CE=0.25 | Change |
|---|---:|---:|---:|
| AUROC | 0.77163 | 0.78349 | +0.01186 |
| FPR95 | 0.65100 | 0.63850 | -0.01250 |
| OSCR | 0.58540 | 0.58853 | +0.00313 |
| Test known acceptance | 95.4833% | 94.8667% | -0.6167 pp |
| Unknown rejection at validation-calibrated operating point | 20.900% | 26.250% | +5.350 pp |
| Accepted-known classification accuracy | 72.2290% | 72.0309% | -0.1981 pp |
| All-known classification accuracy | 70.9333% | 70.6833% | -0.2500 pp |
| Score KS distance | 0.40242 | 0.42092 | +0.01850 |
| Unknown-below-known-q95 overlap proxy | 77.475% | 74.350% | -3.125 pp |

The two seed-paired directions agree on AUROC, OSCR, KS, overlap proxy and
validation-calibrated unknown rejection. FPR95 improves on seed 2030 but
slightly degrades on seed 2029. Test known acceptance changes by -0.0167 pp
and -0.6167 pp, and known classification utility changes by about -0.2 to
-0.25 pp in both seeds. This is a repeatable exploratory signal, but it is
still only two training seeds on the same semantic-hard class split; the
modest utility cost and remaining 74.35% overlap proxy in seed 2030 mean the
core issue is not solved and the default should not yet change. Do not run a
coefficient sweep on these test outcomes. The next useful step is an
independent class split with the same predeclared two-arm comparison, or a
third seed only if compute permits and all criteria remain predeclared.

Replication artifacts:

- CE=1.0: `runs/rejection_ce_balance_s2030_ce1p0_semantic_hard_replication/`
- CE=0.25: `runs/rejection_ce_balance_s2030_ce0p25_semantic_hard_replication/`

## Independent semantic-hard class split check (seed 2040)

### Protocol audit and correction

Before running the proposed independent-split check, code review found that
`build_protocols(..., seed=...)` always assigned the first three sorted fine
classes of every superclass to semantic-hard. Thus earlier seed 2029/2030
comparisons were independent training seeds but **not independent class
splits**. The generator now uses a dedicated seeded RNG to sample 3 of 5 fine
classes per superclass; random and semantic-isolated protocol generation is
unchanged. Existing split JSON files were not overwritten. A unit test checks
same-seed reproducibility, different-seed class assignments, and 3/2 counts in
every superclass.

The new seed-2040 split has 60 known and 40 novel classes, 3 known/2 novel in
each coarse superclass, and 36 known classes in common with the old hard split
(not the same split). The split JSON is
`splits_cifar100_semantic_hard_seed2040/cifar100_60_40_semantic_hard.json`.
A matching 5-epoch teacher was freshly trained from the audited teacher
configuration. On this split, a fixed pair compared CE=1.0 versus 0.25 with
the same seed-2040 teacher, 4 student epochs, mixed pool, all other student
objectives and detector/discovery configuration, and validation-only 95%
known-coverage calibration. Saved train configs differ, apart from output and
validation-derived checkpoint fields, only in `alpha_rejection_known_ce`.
Test labels were not used for training, checkpoint selection, calibration, or
coefficient choice.

### Results

| Seed=2040 independent split | CE=1.0 | CE=0.25 | Change |
|---|---:|---:|---:|
| AUROC | 0.80441 | 0.81219 | +0.00778 |
| FPR95 | 0.61100 | 0.57067 | -0.04033 |
| OSCR | 0.60227 | 0.60907 | +0.00680 |
| Test known acceptance | 95.1833% | 96.1167% | +0.9333 pp |
| Unknown rejection at validation-calibrated operating point | 25.250% | 24.025% | -1.225 pp |
| Accepted-known classification accuracy | 71.9664% | 72.0999% | +0.1335 pp |
| All-known classification accuracy | 70.3167% | 70.8500% | +0.5333 pp |
| Score KS distance | 0.46442 | 0.47183 | +0.00742 |
| Unknown-below-known-q95 overlap proxy | 73.725% | 71.700% | -2.025 pp |

This independent split contradicts the earlier two seeds on the operating
point: AUROC/FPR95/OSCR and overlap improve, and known utility does not fall,
but unknown rejection at the validation-calibrated point decreases by 1.225
pp. Consequently the lower-CE direction has a consistent ranking/score-tail
signal across the two class splits, but **does not consistently improve
coverage-calibrated unknown rejection**. It is not a complete solution to the
core issue: 71.7% of unknown scores still lie below the known q95. Do not
change the default or tune on these test metrics. Stop the claim that CE
reweighting reliably improves the operating point; retain it only as an
exploratory representation/ranking regularizer candidate. Further work should
first use train/known-validation diagnostics to explain why improved ranking
and overlap do not translate to the calibrated operating point; any additional
training comparison needs a predeclared split/seed and a separate untouched
final test protocol.

Artifacts:

- Split: `splits_cifar100_semantic_hard_seed2040/cifar100_60_40_semantic_hard.json`
- Teacher: `runs/semantic_hard_ce_balance_s2040_teacher/`
- CE=1.0: `runs/rejection_ce_balance_s2040_ce1p0_independent_hard_split/`
- CE=0.25: `runs/rejection_ce_balance_s2040_ce0p25_independent_hard_split/`
- Orchestration entry point: `scripts/run_semantic_hard_ce_balance_independent_split.py`

## Second independent class split check (seed 2041; completed 2026-10-20)

### Predeclared question and conditions

The seed-2040 independent split improved AUROC/FPR95/OSCR and overlap proxy
under CE=0.25, but its validation-calibrated unknown rejection fell by 1.225
pp. This fixed replication asks whether the same CE reduction gives a useful
overall result on another independently sampled semantic-hard class split.
The comparison remained CE=1.0 versus 0.25 only; matching 5-epoch teacher,
4-epoch students, seed=2041, mixed pool, detector/discovery setup and
known-validation-only 95% coverage calibration were held fixed. Test labels
were used only for final descriptive metrics. The saved student configs differ
in the intended CE coefficient, output paths, and validation-derived checkpoint
fields; no other training setting differs.

### Results

| Seed=2041 independent split | CE=1.0 | CE=0.25 | Change |
|---|---:|---:|---:|
| AUROC | 0.79098 | 0.79337 | +0.00240 |
| FPR95 | 0.61367 | 0.60417 | -0.00950 |
| OSCR | 0.58286 | 0.58375 | +0.00089 |
| Test known acceptance | 93.883% | 94.967% | +1.083 pp |
| Unknown rejection at validation-calibrated operating point | 28.625% | 24.875% | -3.750 pp |
| Accepted-known classification accuracy | 70.939% | 70.762% | -0.177 pp |
| All-known classification accuracy | 68.733% | 69.100% | +0.367 pp |
| Score KS distance | 0.44192 | 0.43983 | -0.00208 |
| Unknown-below-known-q95 overlap proxy | 75.075% | 75.275% | +0.200 pp |

The ranking gains are small, KS is effectively unchanged/slightly worse, and
the overlap proxy slightly worsens. Most importantly, calibrated unknown
rejection drops 3.75 pp despite test known acceptance increasing by 1.08 pp;
the treatment is not improving the intended operating point. Validation
calibration was 94.977% in both runs. Lowering this CE is inconsistent across
the independent splits for calibrated rejection and does not improve overlap
on seed 2041. Do not change the default or run another coefficient/split to
rescue this direction. Retain the modest ranking result as exploratory only;
CE reweighting failed its predeclared operating-point replication criterion
and is not a solution to the core problem. Unknown-below-known-q95 overlap
remains about 75%.

### Artifacts and next direction

- Split: `splits_cifar100_semantic_hard_seed2041/cifar100_60_40_semantic_hard.json`
- Matching teacher: `runs/semantic_hard_ce_balance_s2041_teacher/`
- CE=1.0: `runs/rejection_ce_balance_s2041_ce1p0_independent_hard_split/`
- CE=0.25: `runs/rejection_ce_balance_s2041_ce0p25_independent_hard_split/`
- Discovery reports are in the corresponding `_discover/` directories.

Next, move away from changing the rejection-branch CE coefficient. Prioritize
training-time class-conditional local geometry with conservative label-free
candidate selection, one objective component at a time. First audit whether
training/known-validation proxies identify semantic-hard known boundary
examples without test labels; then predeclare a paired intervention and judge
it jointly by calibrated rejection, known utility, ranking, and overlap while
preserving a final untouched test protocol.
