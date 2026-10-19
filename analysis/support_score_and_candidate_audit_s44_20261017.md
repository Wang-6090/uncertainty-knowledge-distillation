# Support-score and candidate-pool audit (2026-10-17)

## Purpose

This audit checks two possible explanations for the known/unknown overlap:

1. the class-conditional support radius may be a better final rejection score;
2. the mixed discovery candidate pool may be too contaminated by known samples.

The audits use the frozen seed-44 student checkpoint from the strict PU-kNN
recheck. No model is retrained. Test/open labels are used only for reporting,
not for score selection or calibration.

## Final-score control on the same checkpoint

All arms use the same CIFAR-100 semantic-isolated 60/40 split, mixed pool,
validation-only 95% known-coverage calibration, and checkpoint. Only the
final score mode changes.

| score mode | AUROC | FPR95 | known acceptance | unknown rejection | KS | unknown below known-q95 |
|---|---:|---:|---:|---:|---:|---:|
| feature rejector (current) | 0.9008 | 0.3623 | 95.63% | 47.08% | 0.6494 | 50.75% |
| known support | 0.7717 | 0.6602 | 95.27% | 21.75% | 0.4073 | 77.48% |
| known conformal support | 0.7161 | 0.7885 | 87.97% | 31.38% | 0.3478 | 88.75% |
| normalized entropy + prototype | 0.7874 | 0.6302 | 94.80% | 23.13% | 0.4435 | 77.68% |

As a separate calibration diagnostic, replacing the global threshold with
class-conditional 95th-percentile thresholds kept the same AUROC (0.9008) and
raised unknown rejection only from 47.08% to 48.53%, while the reported known
acceptance fell to 87.95%. This is not a reliable solution either: it changes
the operating point without improving ranking and does not explain the feature
overlap.

The direct support and conformal scores are clearly weaker than the current
mixed-pool nnPU feature rejector. Therefore changing only the support-radius
formula or replacing the rejector with a support score is not a justified fix.

## Mixed candidate-pool audit

The matched discovery pool contains 5,400 samples with 20% known and 80%
unknown. Candidate ratio is 25%. Deterministic evaluation views are used for
this diagnostic; the actual two-view training gate remains unchanged.

| selector | selected | known contamination | unknown purity | unknown recall |
|---|---:|---:|---:|---:|
| entropy + uncertainty | 1,350 | 4.96% | 95.04% | 29.70% |
| consensus | 1,029 | 4.57% | 95.43% | 22.73% |
| prototype distance | 1,350 | 5.48% | 94.52% | 29.54% |
| distance consensus | 1,350 | 3.41% | 96.59% | 30.19% |
| support boundary | 1,350 | 5.85% | 94.15% | 29.42% |
| support distance | 1,350 | 5.48% | 94.52% | 29.54% |

The candidate pool is not heavily contaminated. `distance_consensus` gives a
small purity improvement, but it does not increase unknown recall materially.
This means repeated candidate-gate changes are unlikely to remove the main
overlap: the selected unknowns are relatively clean, while many difficult
unknown samples remain outside the training candidate subset or remain close
to known features after selection.

## Hard-unknown concentration

Using the already calibrated global threshold only for post-hoc diagnosis, the
lowest unknown-class rejection rates included: `clock` 16%, `lamp` 23%,
`beaver` 23%, `porcupine` 24%, `possum` 25%, `turtle` 29%, and `television`
30% (100 test images per class). This is evidence that the failure is
concentrated in visually/semantically close unknown classes rather than being
uniform across all 40 novel classes. The next representation experiment
should therefore report per-class unknown rejection and accepted-unknown
nearest-known-class statistics, instead of relying only on one global rate.

## New code instrumentation

`train.py` now writes `support_radius_diagnostics` into
`calibration_report.json` whenever a classwise support model is fitted. It
records the number of classes, quantile, and min/q25/median/q75/max radius.
This is diagnostic only and does not change any loss, score, threshold, or
default configuration.

## Decision

- Do not replace the current feature rejector with direct support or conformal
  support scoring.
- Do not promote a new candidate gate based on this single frozen-checkpoint
  audit.
- The next algorithmic work should target the representation of the hard
  unknown classes and the low-recall part of the unknown distribution, with a
  paired validation protocol. The next audit should report per-unknown-class
  rejection rates and accepted-unknown feature similarity before adding a new
  loss.

Artifacts:

- `analysis/candidate_pool_audit_strict_s44.json`
- `scripts/audit_discovery_candidate_pool.py`
- `runs/strict_support_score_audit_s44_e4_*/discovery_report.json`
