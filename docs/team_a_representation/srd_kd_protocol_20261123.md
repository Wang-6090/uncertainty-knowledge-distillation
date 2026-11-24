# Semantic representational distillation (SRD) pilot

## Scope and motivation

This pilot remains within Team A's student representation and KD scope. Recent
CE-only A/B/C, centered-feature KD, class-center feature KD, and selective-KD
confirmation did not establish a stable end-to-end benefit from standard or
uncertainty-weighted logits KD. The next distinct hypothesis is that the
student may benefit from being evaluated in the teacher's semantic classifier
space, rather than matching logits produced by its own classifier.

For a student feature `h_s`, frozen teacher classifier `(W_t, b_t)`, teacher
logits `z_t`, and temperature `T`, the treatment is:

```text
L_SRD = T^2 * KL(softmax(z_t / T) || softmax((W_t h_s + b_t) / T))
```

The teacher classifier and teacher logits are detached. The gradient is
intended to update the student encoder only through `h_s`. This requires
matching feature dimensions; the configured ResNet-34 and ResNet-18 both
produce 512-dimensional pooled features in the current CIFAR profile.

## Fixed pilot protocol

Use semantic-hard CIFAR-100 60/40 split seed `2170` and three paired arms:

1. `standard`: CE + standard logits KL.
2. `srd`: CE + cross-network SRD KL through the frozen teacher classifier.
3. `uncertainty`: CE + existing teacher-uncertainty-weighted logits KL.

All arms share one split-matched teacher, student initialization, data order,
four-epoch student budget, `alpha_kd=1`, temperature, known-validation-only
checkpoint selection, `max_softmax` evaluation, known-validation 95% coverage
calibration, matched discovery pool, automatic K, and clustering. Other
auxiliary coefficients and discovery/PU training are zero. Test labels are
used only for final metrics. No KD coefficient, temperature, epoch, feature
normalization, or score rule is scanned. Existing files are not overwritten;
the run tag is `team_a_srd_20261123`.

## Preflight and mechanism gates

The loss unit tests must verify the formula, teacher detachment, nonzero
student-feature gradient, and dimension validation. The runner dry-run must
verify fresh output targets, the 60/40 split, arm definitions, known-only
selection, and gradient auditing. For a valid performance comparison, every
SRD epoch must have a positive KD loss and positive KD gradient norm measured
on student encoder parameters. Standard and uncertainty arms remain their
existing mechanisms; uncertainty weighting is not conflated with SRD.

## Interpretation and stopping rule

Seed `2170` is a single-seed exploratory pilot, not a promotion test. Its
mechanism gates passed and its known-validation normalized between/total
scatter plus most downstream paired metrics improved, while NMI decreased.
This mixed but promising result justifies the fixed confirmation below; it
does not justify a default change.

## Fixed confirmation

Run fresh semantic-hard seeds `2171/2172` with the exact same three arms and
configuration as the pilot. No coefficient, temperature, epoch, score,
normalization, or candidate-rule scan is allowed. The confirmation may proceed
only if all checks below pass:

1. The standard and SRD training configs differ only by `kd_mode`; teacher,
   data protocol, training budget, initialization, and evaluation are paired.
2. Every SRD epoch has positive KD loss, positive KD gradient norm on student
   encoder parameters, and a positive student update. Teacher logits and
   classifier remain detached. Uncertainty-KD has its separate, active
   nonuniform weights.
3. On known validation, SRD improves normalized between/total scatter on both
   seeds; report raw feature norms, normalized within-class distance, nearest
   center distance, and center margin as diagnostics, not checkpoint criteria.
4. Across the two seeds, mean AUROC, fixed-coverage unknown rejection, and
   candidate purity do not decrease versus standard-KD; at least two of
   Cluster ACC/NMI/ARI have nonnegative mean paired changes; known accuracy
   does not fall by more than 1 percentage point on average. Auto-K is
   secondary and reported separately.

This is a screening gate, not statistical proof. Passing only motivates a
larger formal replication; it does not change the default. If the mechanism
gate fails, repair implementation before interpretation. If the representation
or downstream criteria fail, freeze SRD without parameter sweeps. Unknown
labels remain reserved for final retrospective evaluation.
