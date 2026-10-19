# Strict PU-kNN boundary wiring recheck (2026-10-17)

## Purpose

The earlier pure-vs-hard-NNPU runs were not attributable to the kNN risk
form alone: their saved configurations contained
`alpha_discovery_rejection_boundary=0.1`. This recheck disables that
independent rejection-boundary term and the rejection-kNN term in both arms.
The only algorithmic variable is `discovery_knn_risk`.

## Protocol

- seed 44; CIFAR-100 semantic-isolated 60/40 split;
- 4 epochs, same source train/discover recipes, student initialization,
  mixed discovery pool (5,400, known prior 0.2), candidate space and kNN
  boundary weight 0.1;
- `alpha_discovery_rejection_boundary=0`,
  `alpha_discovery_rejection_knn_boundary=0`,
  `alpha_discovery_boundary=0`, `alpha_discovery_knn_boundary=0.1`;
- same detector and validation-only 95% known-coverage calibration;
- test labels are used only for reporting, not training or calibration.

## Wiring audit

| arm | alpha rejection | alpha rejection-kNN | alpha kNN | last `discovery_rejection_boundary` | last `discovery_rejection_knn_boundary` | last `discovery_knn_boundary` |
|---|---:|---:|---:|---:|---:|---:|
| pure | 0 | 0 | 0.1 | 0 | 0 | 0.023561 |
| hard_nnpu | 0 | 0 | 0.1 | 0 | 0 | 0.000405 |

The candidate path is active in both arms, while the independent rejection
boundary path is inactive. The hard-NNPU kNN risk is nearly clamped to zero in
the final epoch, which is evidence of weak gradient contribution rather than
evidence of stronger boundary learning.

## Results

| risk | AUROC | FPR95 | OSCR | known accept | unknown reject | KS | unknown below known-q95 | known accuracy after accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pure | 0.900783 | 0.362333 | 0.658302 | 95.633% | 47.075% | 0.649417 | 50.750% | 72.342% |
| hard_nnpu | 0.893804 | 0.386333 | 0.663041 | 94.333% | 51.650% | 0.633917 | 51.050% | 73.958% |

Hard-NNPU changes the operating point toward more rejection, but it worsens
AUROC by 0.00698, worsens FPR95 by 0.02400, reduces KS by 0.01550, and
increases the overlap proxy by 0.30 percentage points. OSCR improves by only
0.00474 and accepted-known accuracy improves by 1.62 points, while known
acceptance falls by 1.30 points. This is not a reliable solution to the
known/unknown feature-score overlap.

## Decision

Do not promote `hard_nnpu` to the default and do not interpret the historical
non-strict runs as a pure PU-kNN ablation. Keep the strict result as an audit
artifact. The next experiments should inspect support-radius calibration,
candidate known contamination/unknown recall, and semantic-near unknowns.
Further PU-kNN work should first address the near-zero hard-NNPU gradient and
use a predeclared validation criterion; threshold scans and additional gates
are not justified by this result.

Artifacts:

- `runs/knn_pu_boundary_strict_s44_e4_pure/`
- `runs/knn_pu_boundary_strict_s44_e4_pure_discover/`
- `runs/knn_pu_boundary_strict_s44_e4_hard_nnpu/`
- `runs/knn_pu_boundary_strict_s44_e4_hard_nnpu_discover/`
- `scripts/run_knn_pu_boundary_control.py`
