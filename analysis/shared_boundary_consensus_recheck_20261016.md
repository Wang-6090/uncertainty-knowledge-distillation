# Shared-boundary consensus recheck (2026-10-16)

## Question and matched protocol

The first seed-44 result suggested that intersecting the shared-feature
support-boundary candidates with the existing classifier/uncertainty consensus
gate might reduce known-sample contamination. The concern was that this hard
intersection could also discard too many unknown candidates.

The recheck compared three candidate spaces: `shared`, `shared_boundary`, and
`shared_boundary_consensus`. Each arm used the same four-epoch CIFAR-100
semantic-isolated protocol, mixed discovery pool (5,400 samples; known prior
0.2), shared-feature kNN boundary loss (weight 0.1), cross-view gating,
detector, MC count, and validation-only calibration to 95% known coverage.
The only treatment factor was the candidate space. Seed 44 was inspected first;
seed 123 was then run from the same audited soft-NNPU configuration template.
Existing seed-123 runs using hard-NNPU were deliberately not used as matched
controls.

## Detection results

| seed | candidate space | AUROC | FPR95 | OSCR | known accept | unknown reject | KS | unknown below known-q95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 44 | shared | 0.89753 | 0.35867 | 0.66378 | 95.67% | 43.78% | 0.64692 | 52.20% |
| 44 | shared_boundary | 0.89467 | 0.39017 | 0.65287 | 94.18% | 51.73% | 0.63750 | 52.85% |
| 44 | shared_boundary_consensus | 0.90097 | 0.35417 | 0.65818 | 96.02% | 44.28% | 0.64617 | 50.13% |
| 123 | shared | 0.89468 | 0.40833 | 0.64609 | 95.97% | 45.03% | 0.63333 | 48.78% |
| 123 | shared_boundary | 0.89328 | 0.40050 | 0.64483 | 94.88% | 48.45% | 0.63208 | 52.80% |
| 123 | shared_boundary_consensus | 0.89436 | 0.40983 | 0.64446 | 95.80% | 46.53% | 0.63817 | 50.70% |

## Mechanism audit

The new gate is active, not just configured: the kNN boundary loss is non-zero
in all epochs. However, compared with `shared`, candidate unknown recall falls
from 10.9–15.0% to 1.1–4.9% on seed 44 and from 11.2–14.7% to 1.4–3.7% on
seed 123. Compared with `shared_boundary`, candidate known contamination is
lower (roughly 2.6–7.7% for consensus versus 7.2–19.3% for boundary-only), but
the much lower candidate coverage also means far fewer unknowns contribute to
the boundary loss. Cross-view agreement is also lower for consensus.

## Decision

The hard consensus intersection is not promoted to the default. Seed 44 has a
small ranking/overlap improvement versus `shared`, but OSCR declines and the
candidate unknown recall collapses. Seed 123 does not reproduce a detection
improvement: AUROC, FPR95, and OSCR are slightly worse than `shared`, and the
overlap proxy is worse. The consistent finding is a precision-recall tradeoff,
not a reliable solution to known/unknown representation overlap.

Do not continue searching the hard intersection or tune it against test
metrics. Keep this mode only as an explicit ablation for reproducibility. A
possible next research hypothesis is to preserve boundary candidates and use
classifier consensus as a bounded *soft* loss weight, but it should first be
implemented as a distinct treatment and compared against boundary-only with a
predeclared validation rule. If that does not improve ranking and overlap
without reducing known utility, stop adding candidate gates and investigate
support-radius calibration, the open-set split protocol, and direct rejection
head supervision instead.

## Artifacts

- Seed 44: `runs/shared_boundary_aligned_control_s44_e4_*`
- Seed 44 consensus checkpoint/evaluation: `runs/shared_boundary_consensus_control_s44_e4*`
- Seed 123: `runs/shared_boundary_aligned_control_s123_e4_softnnpu_*`
- Runner: `scripts/run_shared_boundary_aligned_control.py`
- Consensus runner: `scripts/run_shared_boundary_consensus_control.py`
