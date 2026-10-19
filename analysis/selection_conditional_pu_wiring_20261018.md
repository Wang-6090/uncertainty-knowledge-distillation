# Selection-conditional PU-kNN wiring audit (2026-10-18)

## Why this change was made

The previous PU-kNN path treated a candidate-filtered mixed pool as `R_mix`,
but subtracted the known risk from the unfiltered known batch and kept the
pre-selection prior.  That is only valid when the selection event is
independent of class.  The synthetic audit of
`pi_selected = pi * s_known / s_mixed` verified the formula, but did not verify
that the real candidate gate used the same event on both populations.

## Implementation boundary

`--discovery-knn-selection-conditional` is an explicit ablation and remains
off by default.  It is accepted only when all of the following hold:

- discovery pool is `mixed`;
- feature candidate gating is enabled;
- kNN risk is `hard_nnpu` or `soft_nnpu`;
- a shared or rejection kNN boundary loss is active;
- cross-view intersection is disabled.

With the switch enabled, the mixed pool uses the first view as the candidate
selection frame.  The same single-view selection helper is applied to the
known batch.  If an EMA/teacher selection model is configured, it is used for
both selection-rate estimates; the student features are still used for the
actual known-risk reference.  This separates the selection model from the
representation receiving the PU loss.

The training log now exposes:

- `discovery_knn_mixed_selection_rate`;
- `discovery_knn_known_selection_rate`;
- `discovery_knn_effective_prior`.

An invalid selected prior or a cross-view request fails explicitly instead of
silently applying an unmatched correction.

## Verification completed

- `python -m pytest -q tests/test_core_behaviors.py`: **221 passed**;
- `python -m compileall -q novel_discovery train.py`: passed;
- `git diff --check`: passed;
- synthetic selection-conditional audit: maximum recovered-risk absolute error
  `1.9841e-7`;
- CLI audit: default is `False`, and the explicit switch parses correctly.

These are implementation and formula checks, not evidence of improved open-set
performance.  No default configuration was changed and no real gated model
has yet been selected from this ablation.

## Real paired control completed

The paired control used seed 44, CIFAR-100 semantic-isolated 60/40, the same
student initialization, 4 epochs, a 5,400-sample mixed pool with known prior
0.2, shared-boundary candidate space, hard-NNPU kNN risk, and validation-only
95% known-coverage calibration.  The only algorithmic switch was
`discovery_knn_selection_conditional`.

| arm | AUROC | FPR95 | OSCR | known accept | unknown reject | accepted-known accuracy | KS | unknown below known-q95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| historical gate + raw prior | 0.90048 | 0.35550 | 0.65797 | 95.783% | 42.450% | 72.212% | 0.65717 | 53.600% |
| same gate + selection-conditional prior | 0.89906 | 0.36317 | 0.65904 | 95.350% | 45.750% | 72.610% | 0.64725 | 51.325% |

The conditional prior was recovered as 0.2 in the final training logs, with
mixed and known selection rates both 0.25.  This confirms that the correction
was wired and numerically active.  It did not, however, improve the ranking
separation: AUROC decreased by 0.00142, FPR95 worsened by 0.00767, and KS
decreased by 0.00992.  The higher fixed-threshold unknown rejection is
accompanied by lower known acceptance, so it is an operating-point trade-off,
not evidence that the known/unknown feature distributions separated.

The overlap proxy `unknown below known-q95` decreased from 0.5360 to 0.51325,
but the global ranking metrics moved in the opposite direction and the mean
score gap decreased from 9.0473 to 8.8249.  This mixed result is insufficient
to make the switch a default.  The generated summary is
`analysis/selection_conditional_pu_s44_e4_summary.json`; the final JSON writer
was also fixed to serialize Windows paths and to expose report fields whose
names differ from the generic metric names.

## Next controlled experiment

Because this matched real control changed the operating point without improving
ranking separation, stop tuning the selection prior.  The next method should
provide direct class-conditional rejection supervision or representation
geometry for semantic-near unknowns, with the candidate-gate/PU correction
kept only as an optional ablation.  That experiment must keep the same split,
detector, calibration protocol, and report both ranking and utility metrics.
