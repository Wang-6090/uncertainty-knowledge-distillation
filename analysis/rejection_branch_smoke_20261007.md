# Independent rejection branch: implementation and smoke audit (2026-10-07)

## Motivation

The main unresolved problem is overlap between known and unknown features and
scores. Existing detector improvements mostly operate after student training,
while the classifier feature is shared by classification, uncertainty, and
discovery objectives. This change tests the next architectural hypothesis:
classification and rejection need separate representations.

## Code change

`UKDNet` now has an opt-in independent rejection branch controlled by
`--rejection-feature-dim`:

- `classifier` continues to consume the original encoder feature;
- `rejection_projector` maps the encoder feature to a separate rejection
  embedding;
- `rejection_classifier` provides a known-class auxiliary CE signal;
- `rejection_uncertainty_head` predicts uncertainty from the rejection
  embedding;
- `extract_outputs` exposes `rejection_features`, and the post-hoc rejector
  accepts `--rejector-feature-mode rejection_embedding`;
- `--alpha-discovery-rejection-feature-margin` applies the unknown feature
  margin to the rejection embedding, using soft uncertainty weights on mixed
  pools rather than hard unknown labels.

All new options default to disabled. The historical shared branch therefore
remains the default and old experiments are not silently changed.

## Verification

- `python -m py_compile train.py novel_discovery/models.py novel_discovery/pipeline.py novel_discovery/losses.py`: passed.
- `python -m pytest -q`: `172 passed, 2 warnings`.
- Toy teacher/student smoke with `rejection_feature_dim=64`, known-branch CE,
  and rejection feature margin: training and checkpoint saving completed.
- Toy `discover` with `--rejector-feature-mode rejection_embedding` completed;
  the new embedding was extracted and consumed by the rejector. The toy result
  (`AUROC=0.6339`, unknown rejection `3.57%`) is only an interface smoke test,
  not evidence of algorithmic improvement.

## Important protocol constraint

Teacher and student checkpoints used in the same distillation run must be
created with the same `--rejection-feature-dim`. Loading a historical teacher
without the new branch into a branch-enabled student run correctly fails due
to missing parameters; a new matched teacher must be trained first.

## CIFAR pilot comparison

After the smoke test, a short CUDA pilot was run under one fixed protocol:
semantic-hard CIFAR-100 60/40, seed 42, pretrained ResNet-34/ResNet-18,
1,200 known training images, 300 validation images, 1,000 test images, a
400-image matched mixed pool with known prior 0.2, two epochs, linear
`nu_corrected` nnPU rejector, MC=4, and 95% known-validation calibration.
The pilot is intentionally too small for a final claim, but it checks whether
the new objectives move the detector in the intended direction.

| Arm | AUROC | FPR95 | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: |
| Baseline, embedding rejector | 0.5723 | 0.9094 | 96.07% | 6.51% |
| Independent rejection branch, margin 0.2 | 0.5319 | 0.9111 | 97.44% | 4.82% |
| Independent branch, margin 0.0 | 0.5776 | 0.9607 | 95.38% | 8.19% |
| OE: uniform + uncertainty + feature margin | 0.5344 | 0.9385 | 93.33% | 10.36% |
| OE: uniform only | **0.5788** | 0.9128 | 95.38% | **9.64%** |

The rejection branch is not yet a successful improvement. Its margin loss was
zero with margin 0.2 and 0.8 because the implemented hinge is
`ReLU(similarity - margin)`; increasing the margin reduces, rather than
increases, the active region. Margin 0.0 produced a non-zero first-epoch loss
but saturated by epoch 2, and the detector gains were too small and unstable.
The implementation is kept as an opt-in architecture experiment, not as the
default.

The three-loss OE combination increased unknown rejection mainly by rejecting
more known samples and reduced AUROC. It is rejected as a combined treatment.
Uniform-only OE gave the best pilot AUROC and rejection, but the gain is only
directional and FPR95 did not improve. It is worth a full matched multi-seed
check, preferably with a disjoint validation protocol, but it does not replace
the current leading nnPU plus uncertainty-augmented detector yet.

## Current decision

The implementation is valid, but its effectiveness is not established. Do
not replace the current matched three-seed detector result yet. The next
experiment should compare, under the existing semantic-hard CIFAR-100 matched
protocol, only these two training arms:

1. current `nnPU + uncertainty-margin` student;
2. the same student plus **uniform-only OE** from CIFAR-10, with the auxiliary
   data kept disjoint from CIFAR-100 evaluation;
3. the independent rejection branch only as a secondary ablation, using a
   non-saturating or support-based rejection objective rather than more hinge
   margin tuning.

Use the same data split, pool size and prior, epochs, pretrained backbones,
MC samples, calibration coverage, and test-only reporting rule. Compare AUROC,
FPR95, OSCR, known accuracy/acceptance, unknown rejection, and feature-overlap
diagnostics. A higher AUROC with unchanged known acceptance and lower overlap
would support the hypothesis; a detection gain with clustering degradation
would indicate that rejection and novel-class geometry still need separate
representations.

## Soft support-separation recheck

The next implementation replaced the saturating hinge-style rejection margin
with a smooth-maximum plus softplus support penalty on the independent
rejection embedding. Uncertainty is used only as a detached soft sample weight
on the mixed pool. The training log confirmed a non-zero objective (`0.5609`
in epoch 1 and `0.4331` in epoch 2), so this was a real gradient path rather
than a disabled option.

The detector was then run under exactly the same semantic-hard pilot protocol
and calibration rule as the table above:

| Arm | AUROC | FPR95 | Known acceptance | Unknown rejection | Candidate purity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent branch + soft support separation | 0.5686 | 0.9128 | 96.07% | 7.95% | 58.93% |

This is below the independent branch margin-0 result (`AUROC=0.5776`) and the
uniform-only OE result (`AUROC=0.5788`). The higher rejection rate is not
enough evidence of improvement because 23 of the 56 rejected candidates were
actually known samples; candidate purity was only `58.93%`. The method is
therefore not retained as a leading treatment. We stop tuning this loss and
move to a controlled combination of the two more promising signals: the
independent rejection representation and uniform-only OE.

## Independent rejection + uniform-only OE combination

The controlled combination was trained with the same semantic-hard CIFAR-100
pilot protocol. It retained the margin-0 independent rejection branch and
added only CIFAR-10 uniform-logit OE (`alpha_outlier_uniform=0.1`); soft
support separation, Energy OE, and uncertainty OE remained disabled.

| Arm | AUROC | FPR95 | Known acceptance | Unknown rejection | Candidate purity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent branch + uniform-only OE | 0.5445 | 0.9453 | 94.02% | 6.27% | 42.62% |

The combination is worse than both single-factor references: independent
branch margin-0 (`0.5776` AUROC, `8.19%` unknown rejection) and uniform-only
OE (`0.5788`, `9.64%`). This is evidence against simple loss stacking. The
next diagnostic removes the mixed-pool uncertainty-feature-margin and
uncertainty-PU terms while retaining the two factors, to test whether those
objectives are the source of the conflict. If that also fails, the combination
will be abandoned rather than tuned further.

The cleaned combination checkpoint removed both mixed-pool terms as intended:
the two corresponding loss meters were exactly zero in both epochs. Its
matched detector result was:

| Arm | AUROC | FPR95 | Known acceptance | Unknown rejection | Candidate purity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent branch + uniform-only OE, cleaned | 0.5664 | 0.9487 | 95.38% | 7.23% | 52.63% |

Removing the mixed-pool terms recovered part of the previous degradation
(`0.5445 -> 0.5664` AUROC and `42.62% -> 52.63%` purity), so those terms do
conflict with the combined objective. However, the cleaned combination still
underperforms both single-factor references and does not improve the core
known/unknown separation. The combination direction is therefore closed.

## Strict uniform-only attribution audit

The earlier row labelled "uniform-only OE" was not a pure ablation. Its saved
configuration still enabled `alpha_discovery_uncertainty_feature_margin=0.05`
and `alpha_discovery_uncertainty_pu=0.1`. The earlier baseline used the same
two mixed-pool terms. Those rows are retained as historical pilot results, but
they cannot establish the effect of uniform OE.

To correct this, a new paired experiment used semantic-hard CIFAR-100 60/40,
seeds 42/43/44, pretrained ResNet-34/18, 1,200/300/1,000 train/validation/test
images, a 400-image mixed pool with known prior 0.2, two epochs, MC=4, the
same linear `nu_corrected` nnPU rejector, and 95% known-validation coverage.
The only treatment difference was CIFAR-10 uniform OE with weight 0.1. Both
arms disabled the mixed-pool uncertainty-feature-margin, uncertainty-PU, and
independent rejection branch.

| Metric | Strict baseline mean | Strict uniform-only mean | Difference |
| --- | ---: | ---: | ---: |
| AUROC | 0.5715 | 0.5378 | -0.0337 |
| FPR95 | 0.9206 | 0.9327 | +0.0122 |
| Known acceptance | 95.83% | 94.91% | -0.91 pp |
| Unknown rejection | 7.44% | 5.65% | -1.78 pp |
| Candidate purity | 55.50% | 43.28% | -12.22 pp |
| Candidate unknown-only NMI | 0.8420 | 0.8772 | +0.0352 |

The strict paired result rejects uniform-only OE as a primary improvement for
the current short protocol. The previous positive interpretation was caused by
confounded treatment definitions, not a verified uniform-OE gain. Uniform OE
may remain as a documented negative ablation, but it should not be combined
with further rejection losses or used to motivate the main method.

## Strict angular-margin recheck

An ArcFace-style angular-margin term was tested as a representation-only
alternative to the rejected OE and rejection-branch combinations. The term
was enabled with `alpha_angular=0.05`, margin `0.2`, and scale `16`; all three
seeds used the same strict baseline protocol and disabled mixed-pool auxiliary
losses.

| Metric | Strict baseline mean | Angular mean | Difference |
| --- | ---: | ---: | ---: |
| AUROC | 0.5715 | 0.5468 | -0.0247 |
| FPR95 | 0.9206 | 0.9172 | -0.0034 |
| Known acceptance | 95.83% | 95.32% | -0.50 pp |
| Unknown rejection | 7.44% | 5.83% | -1.61 pp |
| Candidate purity | 55.50% | 48.47% | -7.03 pp |
| Candidate unknown-only NMI | 0.8420 | 0.6615 | -0.1805 |

Angular margin slightly reduced FPR95 but degraded AUROC, unknown rejection,
candidate purity, and clustering. It is retained only as a negative ablation;
known-class angular compactness alone does not resolve unknown feature overlap.

## Strict reciprocal-points recheck

The ARPL-inspired reciprocal-point option was then tested with eight reciprocal
points, margin `0.2`, weight `0.1`, and a pure-unknown discovery pool. This is
the first short experiment here that explicitly models an unknown region,
rather than only changing known-class geometry or logits.

| Metric | Strict baseline seed 42 | Reciprocal seed 42 | Difference |
| --- | ---: | ---: | ---: |
| AUROC | 0.5725 | 0.5164 | -0.0561 |
| FPR95 | 0.9077 | 0.9487 | +0.0410 |
| Known acceptance | 94.87% | 96.58% | +1.71 pp |
| Unknown rejection | 6.99% | 4.82% | -2.17 pp |
| Candidate purity | 49.15% | 50.00% | +0.85 pp |
| Candidate unknown-only NMI | 0.8593 | 0.8927 | +0.0334 |

The reciprocal representation has a small candidate-clustering signal, but its
unknown detector is clearly worse under this matched short protocol. It is not
promoted to the main method. The discrepancy with older positive reciprocal
results is likely protocol- and training-budget-sensitive, so those historical
results must remain separate rather than being merged with this strict audit.

## Feature-overlap diagnostic for the strict pair

To check whether the OE result was only a score-calibration effect, the strict
seed-42 baseline and strict uniform-only checkpoint were compared with the
same post-hoc feature diagnostic. Test labels were used only to stratify
descriptive statistics; no threshold or model was fitted from them.

| Distance | Baseline unknownness AUROC | Uniform-only unknownness AUROC | Baseline overlap | Uniform-only overlap |
| --- | ---: | ---: | ---: | ---: |
| Classifier prototype | 0.5636 | 0.5127 | 0.8197 | 0.8518 |
| Empirical class centroid | 0.5006 | 0.5075 | 0.8541 | 0.8627 |
| Nearest known train sample | 0.5093 | 0.5133 | 0.8464 | 0.8385 |

Uniform OE therefore did not improve the feature geometry. The prototype
distance became less discriminative, while centroid and nearest-neighbour
distances stayed close to random. This confirms that the remaining problem is
representation-level known/unknown overlap, not a threshold-only issue. It
also shows that short 2-epoch/1,200-sample pilots can produce weak geometry;
future algorithm claims must be separated from full-budget training results.
