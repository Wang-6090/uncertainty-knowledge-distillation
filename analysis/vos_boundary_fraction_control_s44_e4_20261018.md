# Boundary-focused virtual outlier control (2026-10-18)

## Purpose

The inter-class VOS audit suggested that synthesizing an unknown-like feature
from every known sample can reshape the whole energy scale and hurt the
calibrated operating point.  The existing implementation already supports
selecting the lowest known classification-margin samples through
`--vos-boundary-fraction`.  This control tests whether restricting VOS to the
hardest known boundary samples is more aligned with the overlap problem.

## Controlled factors

- CIFAR-100 semantic-isolated 60/40;
- seed 44, the same teacher/student initialization, four student epochs;
- mixed discovery pool of 5,400 with known prior 0.2;
- nearest-different-class inter-class VOS, `alpha_vos=0.01`;
- `alpha_vos_rejection=0` in both arms;
- same detector, checkpoint rule, and validation-only 95% known-coverage
  calibration;
- only the VOS boundary fraction changed relative to the no-VOS control.

## Results

| arm | AUROC | FPR95 | OSCR | known accept | unknown reject | accepted-known accuracy | KS | unknown below known-q95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no VOS control | 0.89218 | 0.38883 | 0.65732 | 95.833% | 43.600% | 72.748% | 0.63058 | 53.325% |
| nearest-class VOS, boundary 25% | **0.89919** | **0.36667** | **0.65976** | 94.133% | **52.850%** | **73.194%** | **0.64617** | **52.800%** |

The treatment improves AUROC by 0.00700, FPR95 by 0.02217, OSCR by
0.00244, and KS by 0.01558.  The unknown rejection rate rises by 9.25
percentage points, but known acceptance falls by 1.70 points.  Therefore the
method shows a useful ranking signal and a non-negligible operating-point
trade-off; it is not yet evidence of a universally better detector.

## Implementation audit

The optional `alpha_vos_rejection` branch previously applied unknown BCE to
all virtual samples even when `vos_boundary_fraction < 1`.  The new
`weighted_unknown_bce` helper applies the same boundary selection weights to
both the Energy VOS term and the rejection-head term.  The default remains
unchanged because `alpha_vos_rejection=0` by default.  A regression test checks
that zero-weighted virtual samples receive no gradient.

## Decision and next step

The seed-44 pilot initially appeared more promising than the
selection-conditional PU prior correction, but its seed-123 replication moved
ranking, overlap, and unknown rejection in the wrong direction. The two-seed
evidence does not support treating boundary-focused VOS as a robust gain.
Keep it as an opt-in historical ablation, do not combine it with rejection
BCE, and do not expand weight/fraction sweeps. The next step is a mechanism
audit of which known examples are selected by the low-margin rule and how
their VOS gradients affect known-class and rejector geometry. Any new
intervention must be predeclared and judged jointly by ranking, overlap,
known utility, and fixed-coverage rejection.

Replication results: `analysis/vos_boundary_vos_replication_s123_20261018.md`.

Artifacts:

- control: `runs/vos_boundary_fraction_s44_e4_control_discover/`;
- treatment: `runs/vos_boundary_fraction_s44_e4_interclass_mix_discover/`;
- driver: `scripts/run_interclass_vos_control.py`.
