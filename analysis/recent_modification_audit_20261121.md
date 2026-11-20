# Recent modification audit (2026-11-21)

## Scope

This audit reviews the changes recorded from 2026-11-18 through 2026-11-20,
their mechanism evidence, and their reported detection and discovery effects.
It does not retrain a model or select a method using test labels. The current
branch is `codex/wxy-3407-audit`, the worktree is clean, and the full test
suite passes with `378 passed, 1 warning`.

## Findings

| Direction | Mechanism check | Main observed effect | Decision |
|---|---|---|---|
| UA-DKD / decoupled uncertainty KD | Teacher uncertainty weights and the KD path are active; the uncertainty weighting has very low effective sample size | AUROC/AUPR changes are small and downstream OSCR, known accuracy or ARI are worse or unstable | Freeze; keep standard KD as default |
| Rejection relational KD | Nonzero relation loss and rejection-branch gradients are recorded | OSCR can improve, but NMI/ARI and Auto-K do not improve consistently; oracle-K geometry is also unstable | Freeze |
| Confidence-gated relational KD | Gate coverage is fixed at 75%; gated loss and rejection gradients are nonzero | Unknown rejection and Auto-K error sometimes improve, while OSCR, known utility, candidate purity and clustering decline across the three confirmation splits | Freeze |
| Episode pseudo-novel rejection | Query/support class separation, loss isolation and nonzero rejection gradients are verified | On seed 2151, AUROC/AUPR, unknown rejection and all three clustering metrics decline | Freeze |
| Aggregate gradient projection | Conflict rate, projection rate and score-distribution changes are recorded | Seed 2152 improves ranking and clustering but loses fixed-coverage unknown rejection and known accuracy | Freeze |
| Componentwise gradient projection | Per-component routing is active and separately audited | Seed 2153 gains unknown rejection but loses AUROC/AUPR, candidate purity, OSCR, known accuracy and all clustering metrics | Freeze |
| Fixed uncertainty conjunction | Frozen score computation is `min(z_feature_rejector, z_calibrated_uncertainty_error)`; both components and threshold use known validation only | On frozen checkpoints from seeds 2152/2153, ranking and several candidate/cluster metrics often improve, but the direction is not uniform across arms | Promote only to formal confirmation candidate |

## What the recent results actually establish

The training-side variants did not solve the main problem. They changed the
trade-off between unknown coverage and candidate contamination, but no recent
loss or gradient-routing mechanism improved detection and clustering together
under the fixed known-coverage operating point.

The conjunction result is more promising because it changes only discovery-time
scoring. Relative to the same frozen `feature_rejector` score:

- seed 2153 control gained `+7.275 pp` unknown rejection but lost `0.575 pp`
  candidate purity and `2.242 pp` NMI; its ACC and ARI improved;
- seed 2153 componentwise treatment gained `+3.725 pp` unknown rejection,
  `+4.664 pp` candidate purity, and `+0.921/+2.268 pp` NMI/ARI;
- seed 2152 control gained `+5.901 pp` candidate purity and
  `+2.969/+2.192 pp` NMI/ARI;
- seed 2152 treatment gained `+6.025 pp` unknown rejection,
  `+3.745 pp` candidate purity, and `+1.619/+1.212 pp` ACC/ARI, while NMI
  changed by `-0.159 pp`.

These are frozen-score comparisons within heterogeneous training checkpoints,
not a final algorithm claim. The checkpoints are from earlier gradient-routing
experiments, so the two-seed evidence supports a confirmation audit but does
not yet support changing the default score mode.

## Compatibility warning for interpretation

The earlier seed 2090 experiment compared conjunction against the fixed
`feature_rejector_uncertainty_fusion` score and found a negative result. The
recent 2152/2153 audits compare conjunction against raw `feature_rejector`.
Those are different controls, so the results are not contradictory, but they
must not be pooled as one treatment effect. The formal next comparison must
freeze the control as raw `feature_rejector`, freeze the treatment as the
existing conjunction rule, and reuse the same student checkpoint and discovery
pool within each seed.

## Code and protocol status

The conjunction implementation has regression coverage for the minimum of the
two validation-normalized components and missing-signal failures. The frozen
audit runner refuses to overwrite outputs, forces known-validation 95%
coverage calibration, keeps clustering enabled, and records the label audit
fields. The full suite and compilation checks pass.

Before formal 42/123/3407 confirmation, the runner still needs a preflight
comparison of split identity, student checkpoint provenance, training budget,
rejection configuration, discovery-pool size/prior, and clustering settings.
The current runner validates the discovery config and checkpoint existence but
does not itself prove all of those cross-artifact compatibility relations.

## Next stage

Run the fixed conjunction versus raw `feature_rejector` on fresh matched
students under seeds `42/123/3407`, after the compatibility preflight. Report
detection, fixed-coverage rejection, candidate count/purity/unknown recall,
Auto-K and ACC/NMI/ARI separately, with mean and standard deviation. Do not
scan conjunction thresholds, component weights or candidate ratios before that
confirmation.
