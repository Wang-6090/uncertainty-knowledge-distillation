# Confidence-gated relational KD for the rejection representation

## Motivation

The three-seed relational-KD confirmation shows that transferring all teacher
classification geometry to the independent rejection representation is
mechanically valid but not a stable end-to-end improvement. The oracle-K audit
also shows that the problem is not only automatic K selection. A plausible
failure mode is that low-confidence known samples near class boundaries carry
unstable teacher geometry into the rejection branch and make semantic-near
unknown samples less separable.

## Hypothesis

Use teacher confidence on the known training batch as a gate for relational
KD. High-confidence teacher examples provide more reliable class geometry;
low-confidence examples remain trained by the existing classification,
rejection and uncertainty objectives but do not receive this additional
geometry constraint. The gate is calculated from detached teacher logits and
does not use unknown labels, the discovery pool, test labels, thresholds or
clustering outputs.

## Registered comparison

The control and treatment will share one fresh semantic-hard CIFAR-100 60/40
split, teacher checkpoint, student initialization, optimizer, epoch budget,
all existing loss terms, known-validation checkpoint selection,
feature-rejector, known-only 95% calibration and Auto-K procedure.

The control keeps the current default with no rejection relational KD. The
treatment enables the fixed relational coefficient `1.0` and a single fixed
confidence gate defined before the run. The gate will retain the top 75% of
teacher-confidence examples in each known training batch; no gate scan is
planned. If a batch has fewer than two retained examples, the implementation
must retain the two most confident examples so the loss remains finite.

The pilot will report detection and clustering separately: AUROC, AUPR, FPR95,
OSCR, known accuracy, known acceptance, unknown rejection, candidate count,
candidate purity, unknown recall, Cluster ACC/NMI/ARI, oracle-K metrics and
Auto-K with its absolute error. It must also report gate coverage, relation
loss, rejection-branch gradient norm, teacher gradient status and the complete
configuration difference.

## Decision rule

This is a mechanism pilot, not a coefficient search. It may proceed to fresh
confirmation seeds only if the treatment improves at least one ranking or
fixed-coverage rejection metric and at least one oracle-K or Auto-K clustering
metric, while known accuracy does not fall by more than one percentage point,
unknown rejection does not fall by more than two percentage points, and
candidate purity does not materially decline. A failure freezes this variant
and leaves `alpha_rejection_relational_kd=0` as the default.

