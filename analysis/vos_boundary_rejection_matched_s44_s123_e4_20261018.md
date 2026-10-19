# Matched boundary-VOS rejection-BCE experiment (2026-10-18)

## Question and controlled comparison

Does adding a rejection-head BCE term on boundary-focused virtual outliers
improve known/unknown score separation beyond the same boundary-VOS training?

For each seed (44 and 123), both arms use CIFAR-100 semantic-isolated 60/40,
the same teacher and training recipe, 4 epochs, nearest-class inter-class VOS,
`alpha_vos=0.01`, a 25% known-class boundary selection, the same matched mixed
discovery pool, checkpoint rule, feature rejector, and validation-only
95%-known-coverage calibration. The only planned training-factor change is
`alpha_vos_rejection=0` versus `0.05`. Configuration audit confirms all other
training options match apart from output paths and the expected
`alpha_vos_rejection` / resulting best-validation-accuracy metadata. The
treatment logs show nonzero `vos_rejection` loss, confirming the term is active.

This tests the implemented objective only: BCE pushes synthetic virtual
outliers toward the unknown target in the rejection head. Synthetic outliers
are not guaranteed to be real unknowns, so this test cannot establish that the
signal transfers to every unseen semantic class.

## Results

All metrics are from the same test protocol; thresholding is calibrated on a
separate known validation set to target 95% known acceptance.

| Seed | Rejection BCE | AUROC | FPR95 | OSCR | KS | Unknown below known-q95 | Known above unknown-q05 | Known accept | Unknown reject | Accepted-known accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 44 | off | 0.89524 | 0.38033 | 0.65273 | 0.64292 | 0.54375 | 0.38033 | 93.92% | 51.45% | 72.64% |
| 44 | 0.05 | 0.90118 | 0.34583 | 0.65685 | 0.64700 | 0.51500 | 0.34583 | 94.92% | 49.08% | 72.36% |
| 123 | off | 0.89518 | 0.40750 | 0.64017 | 0.63950 | 0.50900 | 0.40750 | 95.18% | 48.20% | 71.00% |
| 123 | 0.05 | 0.89078 | 0.41417 | 0.63212 | 0.63025 | 0.53700 | 0.41417 | 95.73% | 42.75% | 70.40% |

Paired treatment-minus-control changes:

| Seed | ΔAUROC | ΔFPR95 | ΔOSCR | ΔKS | Δunknown-below-q95 overlap | Δknown-above-q05 overlap | Δknown accept | Δunknown reject |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 44 | +0.00594 | -0.03450 | +0.00412 | +0.00408 | -0.02875 | -0.03450 | +1.00 pp | -2.37 pp |
| 123 | -0.00440 | +0.00667 | -0.00806 | -0.00925 | +0.02800 | +0.00667 | +0.55 pp | -5.45 pp |

## Interpretation and next direction

- The single-seed positive ranking/overlap signal at seed 44 did not reproduce
  at seed 123. Seed 123 moves AUROC, FPR95, OSCR, KS, and both overlap
  diagnostics in the wrong direction.
- At the fixed validation-calibrated operating point, unknown rejection drops
  in both seeds (+/- signs above): by 2.37 pp at seed 44 and 5.45 pp at seed
  123. This is not an operational improvement; higher known acceptance alone
  is not a win.
- Accepted-known accuracy changes only slightly (−0.28 pp and −0.60 pp), while
  the detector's ranking and unknown rejection are inconsistent/worse in the
  replication. This argues against treating the BCE as a useful robust signal.
- Decision: do not enable `alpha_vos_rejection=0.05` by default and do not
  continue weight sweeps for this objective. Keep the boundary-VOS idea
  separate from its optional rejection-head BCE; the former previously had a
  more promising matched result, but still needs replication.
- Next, diagnose the false-accepted semantic-near unknowns by unknown class and
  nearest known class using the existing error-neighborhood audit. Then test a
  class-conditional training signal only if it can use non-test validation
  information without treating synthetic or selected candidates as certain
  unknown labels. Maintain AUROC/FPR95/OSCR/KS/overlap and known utility as
  co-primary checks; never select methods on test-set rejection rate.

The earlier high-risk pseudo-unknown screening result remains a separate,
already rejected direction: its rejection-rate gain was mainly known-sample
false rejection, not evidence of improved representation separation. :codex-annotation{index="1"}

## Artifacts and checks

- Seed 44 reports: `runs/vos_boundary_rejection_matched_s44_e4_control_discover/`
  and `runs/vos_boundary_rejection_matched_s44_e4_interclass_mix_discover/`.
- Seed 123 reports: `runs/vos_boundary_rejection_matched_s123_e4_control_discover/`
  and `runs/vos_boundary_rejection_matched_s123_e4_interclass_mix_discover/`.
- Runner: `scripts/run_interclass_vos_control.py`.
- Test and syntax checks should be rerun after any subsequent source edits.
- No GitHub upload was performed.
