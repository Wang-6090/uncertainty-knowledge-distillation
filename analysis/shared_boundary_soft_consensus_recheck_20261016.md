# Shared-boundary soft-consensus recheck (2026-10-16)

## Motivation and controlled comparison

The preceding hard intersection (`shared_boundary_consensus`) reduced known
candidate contamination but also removed most unknown candidates. This treatment
keeps the cross-view `shared_boundary` candidate mask unchanged and uses the
classifier/uncertainty consensus risk only as a bounded sample weight in the
shared-feature kNN boundary loss. The weight is label-free and lies in
`[0.25, 1.0]`; the candidate is never hard-deleted.

Each seed compares `shared_boundary` with `shared_boundary_soft_consensus`.
The data protocol, four epochs, mixed pool (5400; known prior 0.2), support
bank, loss coefficient, detector, and validation-only 95% known-coverage
calibration are fixed. The seed-123 run uses the same audited soft-NNPU config
template as seed 44; older hard-NNPU seed-123 results are not used here.

## Results

| seed | arm | AUROC | FPR95 | OSCR | known accept | unknown reject | KS | unknown below known-q95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 44 | boundary-only | 0.89964 | 0.35500 | 0.66006 | 95.48% | 46.50% | 0.64492 | 51.20% |
| 44 | soft consensus | **0.90447** | **0.34467** | **0.66855** | 94.70% | **50.95%** | **0.66783** | **50.30%** |
| 123 | boundary-only | 0.89788 | 0.39217 | 0.64421 | 95.98% | 44.95% | 0.64058 | 50.43% |
| 123 | soft consensus | **0.89914** | 0.39967 | 0.64257 | **96.68%** | 42.58% | 0.63775 | **48.00%** |

The treatment is active: candidate weight means are approximately 0.61--0.75,
while candidate ratios and unknown recall remain close to boundary-only. This
is materially different from the hard intersection, whose unknown candidate
recall collapsed to roughly 1--5%.

## Decision

This is a useful research direction but not a reliable default improvement.
Seed 44 shows a coherent multi-metric gain with a 0.78-point known-acceptance
cost. Seed 123 improves AUROC and the overlap proxy but worsens FPR95, OSCR,
KS, and fixed-working-point unknown rejection. The cross-seed disagreement
means the method should remain an explicit ablation, not be stacked with other
changes or selected using test labels.

The result supports one narrower hypothesis: soft risk weighting preserves the
boundary candidate recall that hard consensus destroys, but the current kNN
boundary objective may still treat contaminated boundary-inside known samples
and genuine outside-support unknowns too similarly. Before more gate tuning,
audit that loss and consider a signed, class-conditional treatment of inside
versus outside support samples. Any follow-up must retain the same paired
multi-seed protocol and validation selection rule.

Artifacts:

- Runner: `scripts/run_shared_boundary_soft_consensus_control.py`
- Seed 44: `runs/shared_boundary_soft_consensus_control_s44_e4_*`
- Seed 123: `runs/shared_boundary_soft_consensus_control_s123_e4_softnnpu_*`
