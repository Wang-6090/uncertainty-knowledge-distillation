# Feature-geometry and training-budget recheck (2026-10-11)

## Question and controlled comparisons

The remaining failure is substantial overlap between known and unknown examples.
This analysis checks whether (a) the PU gradient-routing variants improve simple
feature-space separation, and (b) training the unweighted model longer improves
open-set ranking rather than only closed-set classification.

All geometry runs use CIFAR-100 semantic-isolated 60/40, seed 123, 64-pixel
inputs, the full train/test sets, deterministic evaluation transforms, and the
same train/test examples. There are 27,000 known training samples and 6,000
known / 4,000 unknown test samples. Test labels are used only to stratify
descriptive diagnostics; they are not used to fit a model, choose a threshold,
or tune a method. `boundary_variant` in the source JSONs means direct weighted
PU, detached-feature weighted PU, or the 10-epoch model, as indicated in the
table below.

The training-budget comparison uses the same seed-123 teacher, split, student
configuration, unweighted audited nnPU, matched mixed pool (5,400; known prior
0.2), detector (`normalized_entropy_proto`, MC=4), and validation-only 95%
known-coverage calibration. Only the requested training duration changes from
5 to 10 epochs. In both cases, the checkpoint follows the same validation
known-accuracy selection rule; the selected checkpoint epoch is an outcome,
not a manually fixed epoch.

## Geometry results

Each entry is `unknownness AUROC / histogram overlap` (lower overlap is better).
These are three post-hoc scalar distance diagnostics, not the final detector
AUROC. Distances and formulas are implemented in
`scripts/analyze_feature_overlap.py`.

| Checkpoint | Classifier prototype distance | Empirical class-centroid distance | Nearest known training sample distance |
|---|---:|---:|---:|
| Unweighted nnPU, 5 epochs | 0.6894 / 0.7238 | 0.7048 / 0.6908 | 0.7203 / 0.6678 |
| Weighted PU, direct feature gradient | 0.6614 / 0.7524 | 0.6884 / 0.7200 | 0.7144 / 0.6818 |
| Weighted PU, detached features | 0.6920 / 0.7074 | 0.6662 / 0.7546 | 0.7059 / 0.6911 |
| Unweighted nnPU, 10 epochs | 0.6904 / 0.7178 | 0.7029 / 0.6946 | 0.7245 / 0.6622 |

No treatment wins across all three measures. Detached PU slightly reduces
prototype-distance overlap, but worsens centroid and nearest-sample overlap;
direct weighted PU worsens all three relative to the 5-epoch baseline. Ten
epochs modestly improves nearest-sample separation, while prototype and
centroid changes are negligible or mixed. This does not establish a general
representation improvement.

## Detector results: 5 vs 10 epochs

| Metric | Unweighted nnPU, 5 epochs | Unweighted nnPU, 10 epochs | Change |
|---|---:|---:|---:|
| AUROC | 0.7180 | 0.7156 | -0.0023 |
| AUPR | 0.5938 | 0.5824 | -0.0114 |
| FPR95 | 0.7635 | 0.7685 | +0.0050 (worse) |
| OSCR | 0.3914 | 0.4358 | +0.0444 |
| Known acceptance | 95.92% | 95.07% | -0.85 pp |
| Unknown rejection | 13.53% | 13.60% | +0.07 pp |
| Known accuracy, all known | 46.58% | 52.13% | +5.55 pp |
| Score KS distance | 0.3348 | 0.3251 | -0.0097 (worse) |

Ten epochs improved known classification and OSCR, but did not improve AUROC,
AUPR, FPR95, score-distribution separation, or unknown rejection in a
meaningful way. Longer training is therefore useful for the known-class task,
but is not a solution to known/unknown score overlap. This is a single-seed
comparison; it is not evidence of a statistically reliable effect across
seeds.

### Paired test-example bootstrap for AUROC

To quantify uncertainty in the tiny 5-to-10 epoch AUROC change, a stratified
paired bootstrap sampled the same known and unknown test indices for both
checkpoints (2,000 replicates; seed 20261011). The difference was
`AUROC(10ep)-AUROC(5ep) = -0.00233`, with percentile 95% interval
`[-0.01126, 0.00673]`; 30.25% of bootstrap replicates favored 10 epochs.
This interval includes zero, so this one test split does not distinguish the
ranking performance of the two budgets. It is conditional on the fixed split
and checkpoints; it is not a substitute for multi-seed training uncertainty.
The bootstrap is retrospective reporting only and was not used to select a
checkpoint or tune a method.

### Attempted 20-epoch extension

A 20-epoch extension was started as a follow-up, keeping the seed-123 protocol
and unweighted nnPU configuration. It was deliberately stopped early before
producing a checkpoint: each full-data epoch took roughly two minutes, while
the existing 5-vs-10 epoch detector comparison had already shown no useful
AUROC/AUPR/FPR95 or unknown-rejection trend. This interrupted run is **not an
experiment result** and must not be cited as evidence about 20-epoch training.
Given the low expected information gain, the next compute should go to
diagnostics or a targeted training change, not another longer-budget run.

## Per-class retrospective error audit

The existing `scripts/analyze_classwise_open_errors.py` was run on these same
two detector outputs, using each run's already calibrated validation threshold
(no threshold was refit on test data). Every test class has 100 examples under
this CIFAR-100 split. Unknown false-accept rate remains broad rather than being
limited to a tiny set: 3,459/4,000 unknowns were accepted at 5 epochs and
3,456/4,000 at 10 epochs. The most false-accepted unknown classes were:

| Rank | 5-epoch baseline | False-accept rate | 10-epoch model | False-accept rate |
|---:|---|---:|---|---:|
| 1 | pickup truck (58) | 98% | pickup truck (58) | 98% |
| 2 | television (87) | 96% | dolphin (30) | 96% |
| 3 | clock (22) | 95% | telephone (86) | 96% |
| 4 | bear (13) | 94% | bear (13) | 95% |
| 5 | cloud / mountain / telephone (23/49/86)* | 94% | lamp / whale (40/95)* | 94% |
| 6 | — | — | chimpanzee / lion (21/43)* | 93% |

*The tied classes are listed in the JSON, not fully expanded in this compact
table. They each have 100 test examples.

False accepts are also spread across predicted known classes. At 5 epochs, the
largest sinks were hamster (255), couch (240), skyscraper (218), mushroom
(207), and wardrobe (182); at 10 epochs, shark (193), couch (185), hamster
(151), mushroom (144), and squirrel (136). Thus this is not just one bad
unknown category or one known-class output. Class-to-prediction mappings show
plausible visual confusions: false-accepted pickup trucks most often map to
couch (49 at 5 epochs; 44 at 10), while false-accepted dolphins most often map
to shark (38 at 5 epochs; 52 at 10). Pickup truck remains highly confusable
under both models, which makes it a useful *post-hoc* case to inspect, but it
does not by itself justify training on its test identity. Per-class known
classification accuracy also remains very uneven, so representation quality
is a likely contributor alongside the reject score.

These findings motivate inspecting train/validation class support, predicted
known labels and example-level nearest neighbors for the consistently hard
classes; do not tune a class-specific threshold on this test audit. Artifact:
`analysis/classwise_open_errors_unweighted_5vs10_s123_20261011.json`.

## Decision and next diagnostic

Do not promote detached PU or longer training as an open-set improvement based
on these results. Keep the detach option as an ablation; retain the shorter
unweighted nnPU run as the detector reference until paired multi-seed evidence
changes the conclusion. The disagreement among prototype, centroid, and
nearest-neighbor distances suggests that a single global geometry statistic
is hiding class-conditional failure modes.

The per-class retrospective audit already identifies broad false-acceptance
across unknown classes and several known-class prediction sinks. Next, inspect
example-level neighborhoods for the consistently confused cases and determine
whether known-class mistakes and unknown false accepts share the same local
support. Test labels may be used only for this post-hoc descriptive report,
never for training, threshold selection, or hyperparameter tuning. Any
class-conditional or local-boundary training change must be motivated by that
diagnostic and evaluated on untouched seeds/splits.

## Artifacts

- `analysis/feature_geometry_unweighted_vs_weighted_direct_s123_20261011.json`
- `analysis/feature_geometry_unweighted_vs_detached_s123_20261011.json`
- `analysis/feature_geometry_unweighted_5vs10epoch_s123_20261011.json`
- `analysis/classwise_open_errors_unweighted_5vs10_s123_20261011.json`
- 5-epoch detector: `runs/distance_consensus_full_s123_baseline_discover/discovery_report.json`
- 10-epoch detector: `runs/unweighted_nnpu_10epoch_s123_discover/discovery_report.json`
- 10-epoch training config/history: `runs/unweighted_nnpu_10epoch_s123/`
