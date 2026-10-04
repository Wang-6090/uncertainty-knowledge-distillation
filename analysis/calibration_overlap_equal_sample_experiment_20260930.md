# Equal-size calibration-overlap control (2026-09-30)

## Question

Does using calibration examples that also participated in checkpoint selection
change the open-set operating point, independently of calibration-set size?
This is an evaluation-protocol audit, not a proposed fix for known/unknown
feature overlap.

## Protocol and comparison

Both runs use the same CIFAR-100 semantic-hard 60/40 split, seed 3407,
five-epoch full-data student checkpoint, deterministic 27,000-example known
training feature bank, 1,000-example open test subset, kNN distance (`k=10`,
backbone features), and a threshold calibrated to 95% known coverage. Both
reserve half of the limited 300-example known validation set, producing the
same 166 checkpoint-selection examples and 134 threshold-calibration examples.
Clustering is disabled. The only difference is whether the 134 calibration
examples are drawn from the held-out half (`disjoint`, ordinary protocol) or
sampled from within the selection half (`overlap control`).

The `open_scores.npy` files have identical SHA-256 hashes, confirming that the
checkpoint, test examples, feature bank, and test-score ordering are unchanged.
Only the calibration subset and resulting threshold/acceptance decisions vary.

| Metric | Disjoint calibration | Same-size overlapping calibration |
| --- | ---: | ---: |
| Checkpoint-selection examples | 166 | 166 |
| Calibration examples | 134 | 134 |
| Calibration known acceptance | 94.78% | 94.78% |
| Calibrated threshold | 0.098021 | 0.100916 |
| Test AUROC | 0.5313 | 0.5313 |
| Test FPR95 | 0.8983 | 0.8983 |
| Test known acceptance | 92.59% | 93.62% |
| Test unknown rejection | 6.19% | 5.00% |
| Test known-class accuracy | 33.97% | 34.14% |

Artifacts:

- `runs/calibration_equaln_disjoint_s3407/`
- `runs/calibration_equaln_overlap_s3407/`
- Checkpoint: `runs/semantic_hard_calibration_split_student5_s3407/student.pt`

## Interpretation

- As expected, AUROC and FPR95 are unchanged: calibration does not change the
  model or the ordering of test scores, only the threshold.
- The approximately one-percentage-point acceptance/rejection differences
  are small and based on one seed and only 134 calibration examples. They do
  not establish that overlap is beneficial or harmful.
- A preceding exploratory comparison used 300 calibration examples in the
  overlapping condition and 134 in the disjoint condition. That comparison is
  confounded by sample size and should not be used to attribute effects solely
  to overlap.
- Independent calibration remains the cleaner reporting protocol because it
  prevents the threshold set from reusing checkpoint-selection examples. It
  does not address the underlying overlap of known and novel feature scores.

## Code change and verification

Added the diagnostic-only `--calibration-overlap-control` flag. It preserves
the selection subset and calibration-set size while deliberately sampling the
calibration examples from the selection subset. The calibration report marks
this condition as non-disjoint. The normal default remains unchanged.

Validation: `python -m pytest tests/test_core_behaviors.py -q` — 102 passed.
The paired CIFAR detection runs completed on CUDA. The overlap-control option
is for controlled analysis only and must not be used for final evaluation.

## Next step

Use the disjoint protocol consistently in the next algorithm comparison. The
existing candidate-gated feature-separation loss has a positive seed-3407
ranking/representation signal but inconsistent operating-point behavior across
seeds. Retest that fixed candidate against its matched baseline with disjoint
calibration, reporting AUROC, FPR95, known acceptance, unknown rejection,
OSCR, and the feature-distance diagnostics. Do not tune the score threshold on
test labels or describe a higher rejection rate as an improvement if it comes
from lower known acceptance.

## Candidate-gated feature-separation recheck under the disjoint protocol

The earlier candidate-gated loss had a promising but inconsistent signal. A
paired recheck now uses the calibration-split baseline checkpoint and a
candidate checkpoint trained with the same split and protocol. Config audit
confirmed the only substantive training differences are enabling the
candidate-gated feature-separation term (`0 -> 0.05`) and its required gate;
teacher, pretrained initialization, data, schedule, optimizer, and all other
loss weights match. The separation term was nonzero in the training history.

Detection uses the same `normalized_entropy_knn` score, deterministic full
known-training feature bank (`k=10`), 134-example independent known-only
calibration set, 1,000-example test subset, target 95% known coverage, MC=4,
and no clustering. Only the student checkpoint changes.

| Metric | Matched baseline | Candidate separation | Change |
| --- | ---: | ---: | ---: |
| AUROC | 0.5449 | 0.5731 | +0.0282 |
| AUPR | 0.4334 | 0.4667 | +0.0333 |
| FPR95 | 0.8707 | 0.8810 | +0.0103 (worse) |
| OSCR | 0.2651 | 0.3073 | +0.0423 |
| Test known acceptance | 92.93% | 91.55% | -1.38 pp |
| Test unknown rejection | 4.05% | 9.29% | +5.24 pp |
| Known-class accuracy (all known) | 34.48% | 41.03% | +6.55 pp |
| Accepted-known accuracy | 37.11% | 44.82% | +7.72 pp |

The candidate has a real one-seed ranking/OSCR and representation diagnostic
improvement, but FPR95 worsens and test-known acceptance is below the target
for both models. Thus the higher rejection is not explained only by a large
increase in known rejection (the candidate improves known classification),
but the operating point remains poor. It is a promising candidate, not a
validated fix.

On the exact same test set, descriptive unknownness AUROC from nearest-known
training-feature distance changed `0.5415 -> 0.5758`; empirical-centroid
distance changed `0.5157 -> 0.5320`; classifier-prototype distance changed
`0.5550 -> 0.5706`. Nearest-training-feature histogram overlap decreased
`0.8269 -> 0.8095`, while centroid overlap slightly worsened. Diagnostics are
at `analysis/calibration_split_candidate_feature_sep_diagnostics_s3407_20260930.json`.
They support a representation shift in the intended direction, but residual
overlap remains very high (about 0.80).

An initial attempted candidate run was invalid because it omitted
`--pretrained`; its config showed `pretrained=false` against a pretrained
baseline. That run was not evaluated or used in the result above. The valid
candidate was retrained with matching pretrained initialization, and its
configuration and active loss were checked before evaluation.

### Decision

Keep candidate-gated feature separation as an optional research ablation. The
new isolated-protocol result gives reason to replicate it on another fixed
seed, but do not promote it based on one seed: the earlier seed-42/123/3407
records are mixed, FPR95 still worsens here, and known/unknown feature
distributions continue to overlap substantially. Next, run one predeclared
independent-seed paired replication with the same disjoint protocol and frozen
score; report feature diagnostics and all operating-point metrics together.
