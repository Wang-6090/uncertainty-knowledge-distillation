# Soft-NNPU rejection-boundary audit (seed 44, 2026-10-15)

## Purpose and protocol

This audit completed the previously interrupted soft-NNPU treatment. It compares
the same four-epoch protocol under the same seed, semantic-isolated CIFAR-100
60/40 split, teacher/student initialization, mixed discovery pool, known prior,
data transforms, rejection representation, detector, and validation calibration.
The treatment-only run reused the already trained baseline and changed only the
PU boundary risk form from `hard_nnpu` to `soft_nnpu`, with boundary weight
`0.1`. Test labels were used only for retrospective reporting and paired
bootstrap, never for training or selection.

The two saved training configurations differ in the intended treatment fields
(`alpha_discovery_rejection_boundary`, `discovery_boundary_mode`, and
`discovery_boundary_pu_risk`) plus the resulting validation checkpoint score.
Both selected epoch 4, so the equal-epoch comparison is preserved.

## Results

| metric | baseline | soft-NNPU | treatment - baseline |
|---|---:|---:|---:|
| AUROC | 0.8959 | 0.9000 | +0.0041 |
| FPR95 | 0.3667 | 0.3578 | -0.0088 |
| OSCR | 0.6534 | 0.6614 | +0.0081 |
| known accuracy (all known) | 0.7088 | 0.7155 | +0.0067 |
| known acceptance | 0.9412 | 0.9498 | +0.0087 |
| unknown rejection at report threshold | 0.5100 | 0.4748 | -0.0353 |
| KS distance | 0.6433 | 0.6445 | +0.0012 |
| unknown below known q95 | 0.5233 | 0.5253 | +0.0020 |

The paired exact-95%-known-coverage diagnostic gives AUROC `+0.00412`, with
95% bootstrap CI `[-0.00043, +0.00857]`. The unknown-rejection difference is
`+0.00438` in that diagnostic, with CI `[-3.15, +4.35]` percentage points.
Thus the positive ranking signal is small and its interval crosses zero; the
operating-point rejection gain is not established. The ordinary report
threshold shows lower unknown rejection for soft-NNPU, so the two threshold
summaries must not be conflated.

## Decision

Soft-NNPU is a valid implementation and a useful diagnostic candidate, but it
does not provide stable evidence that the core known/unknown overlap problem is
solved. It slightly improves AUROC, FPR95, OSCR, and known utility on this seed,
while the rejection operating point is mixed and the paired AUROC interval
crosses zero. Keep `hard_nnpu` as the default and do not stack more risk-form or
threshold sweeps on this result.

The next algorithmic direction should return to the shared representation:
measure and then train class-conditional support geometry using only reliable
known labels and high-purity unknown candidates. Any new loss must be tested as
a single change under the same protocol, and must improve ranking metrics,
distribution diagnostics, and known utility together rather than only increasing
unknown rejection.

Artifacts:

- `runs/rejection_boundary_epoch_control_s44_e4_soft_nnpu_audit_baseline/`
- `runs/rejection_boundary_epoch_control_s44_e4_soft_nnpu_audit_w0p1/`
- `runs/rejection_boundary_epoch_control_s44_e4_soft_nnpu_audit_baseline_discover/`
- `runs/rejection_boundary_epoch_control_s44_e4_soft_nnpu_audit_w0p1_discover/`
- `analysis/rejection_boundary_soft_nnpu_paired_s44_20261015.json`
- `scripts/run_rejection_boundary_epoch_control.py`
