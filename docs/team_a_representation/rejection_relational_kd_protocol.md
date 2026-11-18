# Relational KD for the rejection representation

## Why this is a new direction

The UA-DKD failure-mode audit showed that the final discovery score is the
independent rejection feature-rejector score. UA-DKD changes classification
logits, so its training signal is only indirectly connected to the downstream
candidate pool. Existing pointwise rejection-feature KD experiments already
tested cosine feature matching and MC-epistemic weighting; they did not show a
stable fixed-coverage or clustering gain. Repeating that loss or its weight
search is not justified.

This protocol tests a different target: preserve teacher sample-to-sample
geometry in the student's rejection representation without requiring the
student to copy teacher coordinates. The teacher's ordinary classification
features provide the geometry, while the student rejection features receive
the gradient. The relation loss L2-normalizes both feature sets, computes
within-batch pairwise distances, normalizes each distance vector by its own
mean, and applies smooth-L1 matching to the upper triangle. The teacher target
is detached.

## Hypothesis

Pointwise feature matching can overconstrain the independent rejection branch
and reduce the geometry useful for unknown discovery. Matching normalized
relations may transfer class and neighborhood structure while allowing the
student rejection representation to retain a detector-specific coordinate
system. This should improve candidate ranking or downstream clustering more
reliably than the stopped pointwise feature-KD recipe.

The hypothesis can fail: classification geometry may not be the right geometry
for rejection, and relation matching can preserve known-class structure while
making known and semantic-near unknown samples harder to separate.

## Implementation

The opt-in argument is `--alpha-rejection-relational-kd`, default `0`. When it
is positive, the student loss adds

```text
alpha * relational_distance_distillation_loss(
    student_rejection_features,
    teacher_classification_features,
)
```

The loss is computed on the ordinary known training batch. It does not use
unknown labels, the mixed discovery pool, test labels, thresholds, or
clustering. The new meter and rejection-branch gradient audit are saved in
`train_history.json` as `rejection_relational_kd` and
`gradient_audit_grad_norm_rejection_relational_kd_term`.

## Staged experiment

The control and treatment share one split-matched teacher, student
initialization, data order, optimizer, epoch budget, all existing losses,
known-validation checkpoint selection, feature-rejector, known-only 95%
coverage calibration, and Auto-K procedure. The only training difference is
`alpha_rejection_relational_kd=0` versus a fixed `1.0`. No coefficient scan is
planned; the fixed value is an implementation pilot, not a tuned optimum.

The first run is one fresh semantic-hard CIFAR-100 60/40 split. Before the
formal run, a smoke configuration must verify finite relation loss, a nonzero
student rejection gradient, zero teacher gradient, and matching arm configs.
The pilot can proceed to two additional fresh splits only if the treatment
improves at least one ranking metric (AUROC or OSCR) and one downstream metric
(candidate purity or Auto-K NMI/ARI), while known accuracy does not fall by
more than one percentage point, fixed-coverage unknown rejection does not fall
by more than two percentage points, and candidate purity does not materially
decline. These are screening conditions, not significance claims.

The default method remains unchanged until the staged evidence passes. Every
arm will report raw detection and clustering metrics separately, the relation
loss and gradient audit, and the complete configuration difference.
