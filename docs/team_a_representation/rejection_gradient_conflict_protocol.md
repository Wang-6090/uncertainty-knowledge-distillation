# Rejection gradient conflict routing protocol

## Motivation

The independent rejection branch currently receives several objectives with
different roles. Known-support objectives should preserve the representation
of known classes, while auxiliary open-set objectives try to separate or
downweight samples that are difficult to classify as known. A negative
gradient cosine between these groups is a concrete training-time conflict:
the auxiliary update would locally reduce the known-support objective.

This protocol tests whether routing only the conflicting auxiliary component
can improve the rejection representation without changing the classifier,
teacher uncertainty, calibration rule, or clustering procedure. It is a
gradient update rule, not a new uncertainty signal. The teacher's uncertainty,
the student's uncertainty head, classification entropy, and the final
rejection score remain separate quantities.

The mechanism is opt-in through
`--rejection-gradient-conflict-mode=project_auxiliary`. The default is
`off`, which preserves the historical optimizer path exactly. The mode is
valid only with `--independent-rejection-encoder`.

## Objective partition

For the parameters of the independent rejection branch, let the weighted
gradient of the known-support objective be `g_known` and the weighted
gradient of the auxiliary objective be `g_aux`.

`g_known` contains the terms that supervise or regularize known support:

- known uncertainty loss (`alpha_unc * loss_unc`);
- rejection known-class CE;
- rejection SupCon and superclass-aware rejection SupCon;
- rejection center loss;
- feature decorrelation;
- rejection feature variance floor.

`g_aux` contains the opt-in or open-set terms routed through the rejection
branch:

- episode pseudo-novel rejection;
- rejection-logit KD and rejection-feature KD;
- ordinary and confidence-gated rejection relational KD;
- rejection discovery margin, separation, kNN, PU-margin and boundary terms;
- uncertainty feature margin, separation and PU terms;
- outlier rejection margin, uncertainty and uniform terms;
- VOS rejection loss.

Each group is formed from the coefficient-scaled loss terms used by the total
training objective. Terms with a zero coefficient contribute a zero gradient.
Only the independent rejection parameters are included in this routing;
classifier, shared encoder and unrelated parameter gradients are not
projected.

## Update rule

The full vectors are formed by concatenating per-parameter gradients and
missing gradients are treated as zero. If

```text
dot = g_known · g_aux < 0
```

and `||g_known||^2` is nonzero, the auxiliary vector is projected onto the
half-space whose dot product with `g_known` is zero or positive:

```text
g_aux_projected = g_aux
                   - (dot / (||g_known||^2 + eps)) * g_known
```

The optimizer receives `g_known + g_aux_projected`, together with all other
gradients from the total loss. If there is no conflict, the auxiliary vector
is unchanged. If the known gradient is zero, no projection is applied. The
projection is calculated after the individual objectives are differentiated,
so it cannot silently alter the scalar loss values reported in the history.

The training history records the selected mode, the fraction of batches with
a negative dot product, and the mean fraction of auxiliary gradient norm
removed by projection. These diagnostics establish that the mechanism was
active; they are not performance metrics.

## Label and evaluation isolation

The routing decision uses only gradients of losses already constructed from
the training batch and the configured discovery inputs. It does not use
unknown test labels, candidate-pool labels, cluster labels, AUROC, threshold
outcomes, or post-hoc clustering results. The teacher remains frozen and its
uncertainty is still computed by the configured teacher path. Known-validation
data alone continues to select checkpoints and calibrate the 95% known
coverage threshold. Detection and clustering are reported separately.

## Correctness gates

Before a formal run, all of the following must pass:

1. The projection helper must leave a non-conflicting auxiliary gradient
   unchanged and make a deliberately conflicting gradient non-conflicting,
   while preserving parameter shapes and finite values.
2. Disabled objectives with no autograd path must be treated as zero rather
   than causing `autograd.grad` to fail.
3. The mode must reject configurations without an independent rejection
   encoder and must remain numerically equivalent to the historical path when
   set to `off`.
4. A real one-batch independent-rejection update must change rejection
   parameters, record the mode and conflict diagnostics, and leave the
   classification update path connected.
5. Core tests, the full test suite, Python compilation, and `git diff --check`
   must pass before creating formal run directories.

The unit and one-batch regression tests for these gates are in
`tests/test_core_behaviors.py`. A low-budget smoke run may verify checkpoint
serialization and history fields, but its detection or clustering values are
not evidence of algorithmic benefit.

## Fresh paired screening protocol

The first formal screening uses a new semantic-hard CIFAR-100 60/40 split,
preferably seed `2152`, and a split-matched teacher. Control and treatment
must share the split, teacher checkpoint, student initialization, optimizer,
training budget, all existing loss coefficients, known-validation checkpoint
selection, matched discovery pool, feature-rejector, known-only 95% coverage
calibration, Auto-K procedure, and clustering implementation.

The two arms differ only in:

```text
control:   rejection_gradient_conflict_mode=off
treatment: rejection_gradient_conflict_mode=project_auxiliary
```

Both arms keep episode pseudo-novel rejection disabled. The old relational KD
and confidence-gated relational KD coefficients also remain fixed at their
current protocol values, so this experiment measures gradient routing rather
than a new distillation coefficient. The runner must audit the complete
configuration diff and refuse to continue if any additional training setting
differs.

Every completed arm must report detection and clustering independently:
AUROC, AUPR, FPR95, OSCR, known accuracy, known acceptance, unknown rejection,
accepted-known accuracy, candidate count/purity/unknown recall, Auto-K error,
and Cluster ACC/NMI/ARI. The report must also include conflict rate,
projection fraction, loss/gradient audits, checkpoint-selection policy,
calibration policy, and the label-use audit.

## Decision rule

The treatment may proceed to confirmation seeds only if the screening result
shows a positive change in at least one ranking or fixed-coverage rejection
metric and a positive change in at least one clustering metric, without
crossing the existing known-utility or candidate-purity degradation limits.
A single favorable metric or a lower conflict rate is insufficient.

If either the detection or clustering gate fails, freeze this routing variant
and do not scan projection coefficients, loss weights, conflict thresholds,
or additional seeds on the same split. If the screening passes, use at most
two new semantic-hard paired seeds under the same frozen protocol; only a
consistent paired result can justify changing the default. The default mode
remains `off` until that evidence exists.
