# nnPU plus uncertainty-margin replication (seeds 42 and 2026)

## Purpose

The previous seed-2026 factorial showed that mixed-pool nnPU and the
uncertainty-weighted feature-margin may be complementary. This replication
checks that claim under a second training seed and separates the margin's
incremental effect from the effect of nnPU alone.

All runs use the same CIFAR-100 random 60/40 split, full data, matched 5400-
image mixed discovery pool, known prior 0.2, pretrained ResNet-34 teacher,
pretrained ResNet-18 student, 64px images, batch size 64, 10 epochs, and the
same normalized-entropy plus feature-kNN detector (`k=10`, MC=8). Thresholds
are fitted only on known validation data for 95% known coverage. Clustering is
skipped. The seed-2026 runs are the complete 2x2 study; seed 42 contains the
off/off, on/off, and on/on arms needed to measure the margin increment.

## Open-set results

| Seed | PU | Margin | AUROC | FPR95 | OSCR | Known accuracy | Known acceptance | Unknown rejection |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026 | off | off | 0.6718 | 0.7783 | 0.3725 | 45.47% | 95.00% | 13.88% |
| 2026 | off | on | 0.7095 | 0.7182 | 0.4658 | 54.97% | 95.25% | 12.88% |
| 2026 | on | off | 0.7103 | 0.7293 | 0.4647 | 55.28% | 95.25% | 13.13% |
| 2026 | on | on | 0.7205 | 0.7170 | 0.4764 | 56.25% | 95.70% | 13.15% |
| 42 | off | off | 0.6872 | 0.7478 | 0.4319 | 51.85% | 93.73% | 13.10% |
| 42 | on | off | 0.7152 | 0.7575 | 0.4637 | 55.40% | 95.20% | 15.75% |
| 42 | on | on | 0.7250 | 0.7233 | 0.4802 | 56.82% | 95.03% | 16.33% |

The margin increment within the PU-on condition is positive on both seeds:

| Seed | AUROC change | FPR95 change | OSCR change | Known accuracy change | Known acceptance change | Unknown rejection change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026 | +0.0102 | -0.0123 | +0.0118 | +0.97pp | +0.45pp | +0.02pp |
| 42 | +0.0098 | -0.0342 | +0.0165 | +1.42pp | -0.17pp | +0.58pp |
| Mean | +0.0100 | -0.0233 | +0.0141 | +1.19pp | +0.14pp | +0.30pp |

This is stronger evidence than the earlier single-seed result: the margin
increment is consistent for ranking, FPR95, OSCR, and known classification.
The operating-point unknown-rejection gain is positive but small on average,
so the core overlap problem is improved rather than solved. PU itself also
helps AUROC and OSCR on both seeds, but its FPR95 effect is seed-dependent;
the stable part of the combined result is the additional margin improvement
within the PU-on comparison.

## Feature-overlap check on seed 42

The following comparison is PU-only versus PU+margin. Higher distance AUROC
and lower histogram overlap are favorable.

| Distance reference | PU-only | PU + margin |
| --- | ---: | ---: |
| Classifier prototype | 0.6946 / 0.7016 | 0.7002 / 0.6938 |
| Empirical class centroid | 0.6918 / 0.7169 | 0.7090 / 0.6930 |
| Nearest known training sample | 0.7161 / 0.6820 | 0.7270 / 0.6626 |

All three distance AUROCs improve and all three overlaps decrease. Together
with the fixed detector comparison, this supports a real representation
effect rather than a threshold-only artifact. Substantial overlap remains,
and the diagnostic still uses test labels only for post-hoc description.

## Decision

- Retain mixed-pool nnPU plus uncertainty-weighted feature-margin as the
  current leading combined candidate.
- Keep the margin coefficient and cosine margin fixed at the tested values
  (`0.05` and `0.2`) until more seeds or a new class split are evaluated;
  do not start a broad hyperparameter sweep.
- Do not add NT-Xent or other uniform consistency losses to this candidate;
  previous matched experiments were unstable and did not provide a reliable
  separation direction.
- The next useful extension is a separately trained rejector or calibrated
  score head on validation data, using the improved embedding and without
  test-label threshold tuning. It should be compared against the current
  fixed detector, not silently replace it.
- A third seed or an independently generated class split is still required
  before enabling this combination as the default experimental method.

Artifacts:

- `runs/matched_mixed_base_s42_e10/`
- `runs/matched_mixed_nnpu_base_s42_e10/`
- `runs/matched_mixed_nnpu_margin_s42_e10/`
- corresponding `_detect/` directories
- `analysis/matched_nnpu_margin_overlap_s42.json`
- seed-2026 factorial report:
  `analysis/matched_nnpu_margin_factorial_s2026_20261006.md`
