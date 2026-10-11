# Error-neighborhood audit (CIFAR-100 semantic-isolated, seed 123)

## Question and protocol

Do unknown samples falsely accepted as known occupy the same local feature
neighborhoods as correctly classified known samples, or are they more similar
to known classification errors? This is a post-hoc diagnostic, not a training
or threshold-selection experiment.

The audit compares the existing 5-epoch audited nnPU checkpoint and 10-epoch
unweighted nnPU checkpoint on the same semantic-isolated 60/40 split. The
feature bank exactly reconstructs the supervised known subset used by the
matched mixed-pool training protocol: 27,000 known train-partition samples
before reserving 1,080 known examples for the unlabeled discovery pool, leaving
21,600 labeled known training samples. A cosine kNN bank is formed only from
those labeled examples (deterministic eval transforms, normalized student
embeddings; `k=10`). The known/unknown decision
uses each run's existing validation-calibrated threshold; no threshold is
refit. Detector details are checked against the deterministic test loader for
sample order and raw labels. Test labels are used only to define descriptive
cohorts and unknown-class breakdowns. A first draft included the reserved
unlabeled-known subset in the labeled neighbor bank; protocol self-audit caught
this mismatch. That draft was discarded and the reported output recomputed
with the exact 21,600-example supervised support bank.

Implementation: `scripts/analyze_error_neighborhoods.py`.
Machine-readable output: `analysis/error_neighborhoods_s123_20261011.json`.

## Main cohort results

| Model / cohort | Count | Nearest train cosine distance | Top-1 neighbor matches predicted class | Mean top-10 purity of predicted class |
|---|---:|---:|---:|---:|
| 5ep known correct + accepted | 2,754 | 0.0465 | 80.0% | 76.5% |
| 5ep known wrong + accepted | 3,001 | 0.0699 | 27.2% | 28.8% |
| 5ep unknown false accept | 3,459 | 0.0846 | 32.0% | 33.3% |
| 5ep unknown rejected | 541 | 0.1267 | 12.2% | 11.7% |
| 10ep known correct + accepted | 3,085 | 0.0512 | 82.2% | 78.8% |
| 10ep known wrong + accepted | 2,619 | 0.0794 | 31.4% | 33.5% |
| 10ep unknown false accept | 3,456 | 0.0936 | 40.2% | 39.6% |
| 10ep unknown rejected | 544 | 0.1367 | 13.2% | 13.3% |

Correctly classified known samples have compact, prediction-consistent local
support. Unknown false accepts are farther from known training examples than
correct-known samples, but substantially closer and more prediction-consistent
than rejected unknowns. Their neighborhood metrics are similar to, though
slightly stronger than, accepted known-class errors. This supports a mixed
failure mode: some unknowns are genuinely embedded near known support, while
others have weak local support but still pass the scalar detector threshold.
The detector's false-accepted group remains very large (about 3,450 of 4,000
unknowns in both models).

From 5 to 10 epochs, unknown false accepts become more aligned with the model's
predicted known class (top-10 purity 33.3% to 39.6%) while the false-accept
count barely changes. Thus longer training can make the classifier's known
decision locally more self-consistent without creating better rejection.
This is consistent with the previous finding that known accuracy/OSCR improve
while AUROC and unknown rejection do not.

## Per-class observations

Pickup truck (raw CIFAR-100 class 58) remains 98% falsely accepted by both
models. In the 5-epoch run its most common nearest known training class is
orange (89), while its most common predicted class is couch (25); in 10 epochs
the same top nearest class / top prediction pair remains. The discrepancy
shows the classifier head and local nearest-sample geometry are not identical
signals. Do not infer a universal class pairing from this one class.

Other difficult unknown classes have different nearest-neighbor and predicted
class patterns. Therefore a single class-specific threshold or one targeted
known-class repulsion is not justified by this audit. Full class-level output
is in the JSON artifact.

## Decision and next experiment

This diagnostic does not establish that nearest-neighbor training loss will
help. It does identify an actionable direction: distinguish (a) low-support
unknown samples incorrectly accepted by an overconfident classifier from
(b) unknown samples genuinely close to a known-class support region. The
existing checkpoints have no reserved open-validation data: their training
configuration uses `open_val_ratio=0`, so their validation split is known-only.
Their discovery pool was sampled from the novel training pool, therefore an
offline split made after training cannot be assumed independent of training.
Do not use test scores to choose fusion weights.

The valid next step is a controlled retraining protocol with
`--open-val-ratio > 0`, which reserves known and novel examples from the CIFAR
training partition before the discovery pool is formed. Select any local
support / detector score fusion on that reserved open-validation partition,
then freeze its weight and evaluate once on the untouched official test set.
Compare against the same checkpoint's unfused detector at matched known
coverage and report AUROC/AUPR/FPR95/OSCR as well as acceptance/rejection.
Only if the validation-selected treatment improves ranking without trading
away known coverage should it be considered for a broader multi-seed study.

Do not train on test labels, identify classes by test errors for the training
objective, or tune per-class thresholds from the class breakdown. The audit is
one seed/split; repeat a promising result on an untouched seed before changing
the default pipeline.
