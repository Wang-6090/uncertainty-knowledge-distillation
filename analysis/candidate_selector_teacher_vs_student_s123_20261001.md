# Teacher-guided candidate selection recheck (seed 123, 2026-10-01)

## Question

Candidate-gated feature separation currently selects high-risk samples with the
changing student model. This can contaminate the mixed-pool feature loss with
known samples that are merely hard to classify. A new optional selector uses the
frozen teacher checkpoint for candidate selection, following the stable-target
idea used by teacher-guided pseudo-labeling and Mean Teacher-style methods.

## Controlled protocol

Both runs use CIFAR-100 random 60/40, seed 123, the same matched pretrained
ResNet-34 teacher, pretrained ResNet-18 student, complete known training
partition, 300 validation samples, 1200 mixed discovery-pool samples, five
student epochs, uncertainty KD, mixed-pool nnPU, and candidate-gated feature
separation. Both use ratio 0.25 and `entropy_uncertainty` selection.

The only intended training change is the selector:

- control: `--discovery-selection-model student`;
- treatment: `--discovery-selection-model teacher`.

Detection is identical for both checkpoints: MC=4,
`normalized_entropy_min_class_knn`, a known-only validation threshold,
oracle K=40 KMeans, and `feature_pca`. The test subset has 1000 images. Test
labels are used only for final metrics and retrospective class diagnostics.

## Detection and clustering results

| Metric | Student selector | Frozen teacher selector | Change (teacher - student) |
| --- | ---: | ---: | ---: |
| AUROC | 0.6899 | 0.6696 | -0.0203 |
| AUPR | 0.5597 | 0.5586 | -0.0011 |
| FPR95 | 0.7806 | 0.7722 | -0.0084 |
| OSCR | 0.4160 | 0.4226 | +0.0066 |
| Known acceptance | 92.46% | 94.97% | +2.51 pp |
| Unknown rejection | 19.85% | 16.38% | -3.47 pp |
| Accepted-known accuracy | 53.44% | 54.32% | +0.88 pp |
| Candidate count | 125 | 96 | -29 |
| Candidate purity | 0.6400 | 0.6875 | +0.0475 |
| Candidate unknown NMI | 0.6941 | 0.7691 | +0.0750 |
| Unknown-only cluster NMI | 0.7468 | 0.7856 | +0.0388 |

The teacher selector produces a cleaner candidate pool and a slightly better
operating-point balance, but it rejects fewer unknown test samples and has
lower AUROC. The student selector is more aggressive and obtains higher
unknown rejection, partly by accepting fewer known samples. Neither selector
dominates the other across the required metrics.

Because the validation threshold transfers to the finite test subset
differently, the raw validation-threshold rejection rates are not a fair
standalone comparison. As a retrospective diagnostic only, each test score
distribution was recalibrated from its own test-known scores to approximately
95% known acceptance:

| Matched test-known coverage diagnostic | Student selector | Frozen teacher selector |
| --- | ---: | ---: |
| Test known acceptance | 94.97% | 94.97% |
| Unknown rejection | 12.16% | 16.38% |
| Accepted-known accuracy | 52.73% | 54.32% |

This diagnostic uses test-known labels and is not deployable or usable for
model selection. It does show that the teacher selector's lower raw rejection
was largely a threshold-transfer effect: at the same known coverage it rejects
more unknowns and retains a cleaner candidate pool.

## Per-novel-class diagnostic

Compared with the student selector, teacher selection reduces false-accept
rate for 9 of 40 novel classes, increases it for 16 classes, and leaves 15
unchanged. The mean per-class change in false-accept rate is `+0.0357`. This
confirms that higher candidate purity is mostly caused by removing known
false-reject samples, not by uniformly separating novel classes from known
classes.

## Decision

The new option is a promising conservative operating-point variant, but it is
not promoted to the default from one seed. Its AUROC is lower while its
matched-coverage rejection and candidate purity are better. Keep the default
student selector for the current primary nnPU + min-class-kNN comparison until
the teacher selector is repeated on another seed with a separately reserved
known calibration set and an independent open-validation set. Report both raw
validation-threshold metrics and matched-coverage diagnostics.

The next meaningful direction is a stronger mixed-pool representation target
that uses reliable known anchors and novel neighborhood structure together;
simply replacing the candidate selector is insufficient. Any follow-up must
compare both selectors at matched known coverage and retain per-class
false-accept diagnostics.
