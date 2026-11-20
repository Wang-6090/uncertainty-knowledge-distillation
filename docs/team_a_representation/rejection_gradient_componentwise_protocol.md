# Componentwise rejection gradient routing protocol

## Why this is the next test

The first `project_auxiliary` screening showed a real trade-off: candidate
purity and clustering improved, while unknown rejection at the fixed known
coverage operating point fell. A read-only audit found a genuine score
distribution change and a higher known-only calibration threshold, so this is
not a threshold-reporting bug.

The first router projects the sum of all auxiliary rejection objectives as one
vector. Different auxiliary terms can point in opposite directions. Summing
them first can hide which term conflicts with the known-support objective and
can cause one helpful term to cancel one harmful term. The next hypothesis is
to route each weighted auxiliary loss separately, preserving non-conflicting
components before summing the routed gradients.

This is a new opt-in training rule. It does not change the teacher, student
uncertainty definition, rejection score formula, calibration policy or
clustering feature. The existing `project_auxiliary` mode remains frozen and
the default remains `off`.

## Proposed update

Let `g_known` be the same weighted known-support gradient used by the first
protocol. For each nonzero weighted auxiliary term `L_i`, calculate its
gradient `g_i` on the independent rejection parameters. Route that term
independently:

```text
if g_known · g_i < 0:
    g_i' = g_i - ((g_known · g_i) / (||g_known||² + eps)) g_known
else:
    g_i' = g_i
```

The rejection branch receives `g_known + sum_i(g_i')`. The known gradient is
not repeatedly subtracted for every auxiliary term. Missing or disabled terms
are zero. Only independent rejection parameters are routed; classification
and shared-backbone gradients keep their existing paths.

The audit must report, at minimum, total auxiliary conflict rate, total
projection fraction, the number of active auxiliary terms, and per-term
conflict/projection summaries. These diagnostics describe execution and do
not use test labels.

The CLI selector is `--rejection-gradient-conflict-mode
project_auxiliary_componentwise`. Per-term rates are averaged over batches
where that term has a nonzero configured coefficient and an autograd path;
the active-term count is averaged over all routed batches.

## Auxiliary terms and isolation

The initial implementation must use the existing auxiliary partition without
adding a new loss coefficient: episode pseudo-novel rejection, rejection KD,
feature/relational KD, rejection discovery margin/separation/kNN/PU/boundary,
uncertainty auxiliary terms, outlier rejection terms and VOS rejection. The
formal candidate keeps episode pseudo-novel disabled and keeps all existing
coefficients fixed at the current mainline values.

This preserves the exact objective partition used by the frozen aggregate
`project_auxiliary` baseline, so the planned contrast isolates sum-then-route
versus route-each-term-then-sum. Rejection consistency terms that were outside
that baseline partition remain on the ordinary residual-loss path in both
modes; they are deliberately not moved into the new router in this test.

The teacher remains frozen. Unknown/test labels, mixed-pool hard labels,
AUROC, threshold outcomes and clustering results must not enter routing or
checkpoint selection. Checkpoints use known-validation accuracy and detection
uses the disjoint known-validation 95% coverage calibration.

## Gates before a formal run

1. Synthetic gradients must show that one conflicting auxiliary term is
   projected, one non-conflicting term is unchanged, and two auxiliary terms
   are routed independently rather than after summation.
2. Disabled objectives without an autograd path must remain safe, and all
   routed tensors must be finite with the original parameter shapes.
3. A real one-batch independent-rejection update must change rejection
   parameters and record both aggregate and per-term diagnostics.
4. Mode `off` must preserve the historical path; the existing aggregate
   `project_auxiliary` mode must remain unchanged.
5. Full tests, Python compilation, diff checks and a low-budget paired smoke
   must pass before a formal run is created.

## Fresh paired screening

Use a fresh semantic-hard CIFAR-100 60/40 split, with seed `2153` unless the
current runner audit finds that seed unavailable. Control uses
`rejection_gradient_conflict_mode=off`; treatment uses the new componentwise
mode. Both arms share the split-matched teacher, student initialization,
optimizer, epoch budget, all existing coefficients, discovery pool,
known-validation checkpoint selection, feature-rejector, known-only 95%
calibration, Auto-K and clustering implementation. Episode pseudo-novel
rejection remains disabled.

Report detection and clustering separately: AUROC, AUPR, FPR95, OSCR, known
accuracy, known acceptance, unknown rejection, accepted-known accuracy,
candidate count/purity/unknown recall, Auto-K error and Cluster ACC/NMI/ARI.
Also report score-distribution and calibration diagnostics, aggregate and
per-term routing statistics, and the complete configuration/label-use audit.

The variant is promoted to at most two confirmation splits only if the fresh
screening improves at least one ranking or fixed-coverage rejection metric
and at least one clustering metric without the known-utility and candidate
purity trade-offs observed in seed 2152. Otherwise it is frozen without
coefficient, projection-strength, or conflict-threshold scans.
