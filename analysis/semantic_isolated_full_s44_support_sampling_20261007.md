# Full-data seed-44 support-rejector replication (2026-10-07)

## Purpose

This is the third independent full-data seed for the class-stratified known
support sampling change. The protocol keeps the student treatment, mixed
discovery pool, support budget, rejector objective, threshold policy, and
semantic-isolated CIFAR-100 split fixed. Test labels are used only for final
descriptive metrics.

## Results

| Student | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.7738 | 0.6397 | 0.3575 | 95.83% | 16.70% |
| treatment | **0.7901** | **0.6118** | **0.4196** | 95.25% | **21.83%** |

Treatment changes relative to the same-seed baseline:

- AUROC: `+0.0163`
- FPR95: `-0.0278`
- OSCR: `+0.0621`
- known acceptance: `-0.58` percentage points
- unknown rejection: `+5.13` percentage points

## Three-seed paired summary

| Seed | AUROC delta | FPR95 delta | OSCR delta | Known acceptance delta | Unknown rejection delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | +0.0101 | -0.0237 | +0.0503 | -0.32pp | +0.18pp |
| 43 | +0.0130 | -0.0083 | +0.0163 | -0.07pp | +1.83pp |
| 44 | +0.0163 | -0.0278 | +0.0621 | -0.58pp | +5.13pp |
| mean | **+0.0131** | **-0.0199** | **+0.0429** | **-0.32pp** | **+2.38pp** |

The direction is consistent for AUROC, FPR95, and OSCR across all three
seeds. Unknown rejection also improves in every pair, but remains only
`16.7%` to `21.8%` at the fixed 95% known-coverage operating point. This is a
real but limited rejector-stability improvement, not a solution to the
known/unknown representation overlap.

## Decision

Keep `--rejector-stratified-known` and `--rejector-max-samples 5000` as an
opt-in, reproducible detector configuration. Stop spending compute on more
support-size or global-threshold sweeps. The next experiment should change
the representation training budget or an explicitly separated boundary
objective while keeping this support protocol fixed.

The test was a support-only audit: clustering and joint novel-head losses
were disabled. Therefore it does not validate the complete end-to-end new
class discovery claim.
