# Sample-weight gradient fidelity audit (2026-10-12)

## Why this follow-up was needed

The residual audit found that a value documented as a sample-selection weight
must be checked at the autograd boundary, not only by inspecting its formula.
The same audit was therefore applied to `novel_mass_weights` and its bounded
view-agreement transform.

## Finding and correction

`novel_mass_weights` computes the summed softmax probability of novel logits
in a unified known-plus-novel space. It is used as a soft sample weight, not as
a differentiable prediction target. Before this correction, gradients could
flow from a weighted auxiliary objective back through the weight into both
known and novel logits. The standard joint-discovery reduction already
detaches sample weights internally, but `unknown_feature_margin_loss` uses
its supplied sample weights directly. With a nonzero joint novel-margin
coefficient, the model could therefore lower the effective weight of samples
with large margin violations and partially evade the intended separation
penalty.

Both `novel_mass_weights` and `bounded_novel_mass_weights` now return detached
weights. `unknown_feature_margin_loss` also detaches its optional sample weight
at the loss boundary, guarding against other callers passing a differentiable
selection signal. Forward values and ranking are unchanged. Regression tests
assert that weights have no gradient path when input logits/masses require
gradients and that feature gradients still flow through the margin objective.
The weight remains usable in all existing weighted reductions.

## Scope of historical evidence

The stored `runs/joint_novel_mass_s42_e3/config.json` confirms that
`joint_novel_mass` was enabled, but does not record a nonzero
`alpha_joint_novel_margin`; the training history records zero
`discovery_feature_margin`, which is a different loss meter. Thus this audit
does **not** establish that the historical novel-mass result was numerically
wrong. Its observed forward weights and usual detached weighted joint losses
are unchanged. It does establish that any run combining novel-mass weights
with `unknown_feature_margin_loss` must be re-evaluated if that margin
coefficient was nonzero. Old experiments without that active path are not
retroactively attributed to this gradient issue.

This is a method-fidelity correction, not evidence of improved AUROC,
unknown rejection, OSCR, or clustering. A valid algorithmic test would need a
matched pair with the same seed, checkpoint initialization, data protocol,
training budget, detector, and calibration, changing only whether a
nonzero-weighted margin uses the selection signal. The selection signal is
now detached by design, so such a pair would require a temporary historical
implementation and is not needed to validate the code invariant. First
confirm the intended margin objective and run a matched detached novel-mass
versus equal-mass comparison before considering it for the default pipeline.

## Validation

- Targeted tests verify novel-mass and bounded weights are detached, retain
  their expected numeric values, and cannot receive gradients through the
  weighted feature-margin objective.
- Full test-suite and compile checks are recorded in the current work-session
  summary after execution.
