# Feature-boundary diagnosis and local-centroid ablation (2026-09-30)

## Why this follow-up was run

The hybrid-boundary pilot showed small gains in AUROC/FPR95 and candidate
purity, but lower unknown rejection at its validation-calibrated operating
point. Before changing the objective again, the paired checkpoints were
re-extracted to check which representation reference actually moved. A second
single-factor pilot then tested whether removing classifier-weight prototypes
and relying only on empirical batch class centroids would help.

## Diagnostic protocol

- Checkpoints: the fixed seed-42 baseline and hybrid-boundary variant from
  `boundary_pair_baseline_s42` and `boundary_pair_variant_s42`; then hybrid
  variant versus the local-centroid-only checkpoint.
- Same CIFAR-100 random 60/40 split, seed 42, deterministic evaluation transform,
  1,200 known training samples and the same 1,000-sample open test subset
  (605 known / 395 unknown).
- For each model, compared cosine distance to: (a) normalized classifier
  weights, (b) empirical known-class means, and (c) the nearest known training
  feature.
- Open-test labels were used only after inference to stratify descriptive
  known/unknown statistics and calculate diagnostic AUROC. No fitting,
  training, threshold selection, or parameter tuning used test labels.
- Histogram overlap is the intersection of two normalized 50-bin histograms
  on a shared distance range; lower is more separated. It is a descriptive
  statistic, not a formal distribution-distance estimator.

## Baseline vs hybrid-boundary representation results

| Reference distance (unknownness = distance) | Baseline AUROC / overlap | Hybrid AUROC / overlap |
| --- | ---: | ---: |
| Classifier-weight prototype | 0.5819 / 0.8124 | 0.5522 / 0.8685 |
| Empirical class centroid | 0.5498 / 0.8369 | 0.5759 / 0.8265 |
| Nearest known training feature | 0.5663 / 0.8262 | 0.5832 / 0.8106 |

The hybrid objective improved separation against empirical centroids and
nearest training examples, but worsened the classifier-weight reference. This
supports the concern that classifier weights and the feature support are not
well aligned in this run. Even the best diagnostic AUROC is only about 0.58,
with high histogram overlap (about 0.81), so the features remain strongly
overlapping.

## Single-factor training ablation: prototype weight 0.5 vs 0.0

The only intended training change was
`--discovery-boundary-prototype-weight=0.5` versus `0.0`. Both used
`alpha_discovery_boundary=0.1`, margin `0.2`, temperature `0.1`, the same
teacher, split, sample limits, 3 epochs, batch size, and pure-unknown pool.
Both were evaluated with the same `normalized_entropy_mahalanobis` score and
95% known-coverage calibration from known validation data.

| Metric | Hybrid weight 0.5 | Local centroid only 0.0 |
| --- | ---: | ---: |
| AUROC | 0.5618 | 0.5527 |
| AUPR | 0.4534 | 0.4310 |
| FPR95 | 0.8512 | 0.9240 |
| OSCR | 0.1660 | 0.1272 |
| Known test accuracy | 0.2397 | 0.1851 |
| Test known acceptance (target 0.95 on validation) | 0.9719 | 0.9207 |
| Test unknown rejection | 0.0658 | 0.0987 |
| Candidate purity | 0.6047 | 0.4483 |
| Oracle all-unknown ARI | 0.0665 | 0.0582 |
| Oracle candidate-unknown ARI | -0.0418 | 0.0318 |

The higher rejection rate for local-only is not a clean improvement: it rejects
more known samples too, with test known acceptance falling to 92.1% rather than
the 95% validation target. AUROC, FPR95, OSCR, known accuracy, and candidate
purity all worsen. Candidate-conditional ARI improves from a negative value but
is still very low; all-unknown ARI also falls slightly. This is a tradeoff,
not a win.

Feature diagnostics comparing hybrid with local-only likewise do not justify
switching: local-only's classifier-prototype diagnostic AUROC was 0.5627, but
empirical-centroid AUROC fell from 0.5759 to 0.5606, nearest-training-feature
AUROC fell from 0.5832 to 0.5716, and overlap increased for these two empirical
references. The classifier metric alone would give the wrong impression.

## Decision and next direction

- Do not set either boundary variant as the default; keep the option available
  only as an ablation.
- Stop tuning the prototype/local-centroid mixture based on this single seed.
- The relevant signal is that empirical training support gives more useful
  unknown ranking than classifier weights in this checkpoint, but its AUROC is
  still weak. A next mechanistic candidate should model *local support/density*
  (for example, a class-conditional nearest-neighbor support margin with a
  radius estimated from known training features), rather than repelling unknown
  samples from one batch centroid or globally changing a score threshold.
- Before that training experiment, implement the support estimate using only
  known training features and verify it does not use test labels. Then compare
  one loss change against the same baseline; require known coverage, unknown
  rejection, AUROC/FPR95/OSCR, and clustering to be reported together.
- Results are one-seed, 3-epoch, limited-data pilots. Repeat on at least two
  additional fixed seeds before claiming effectiveness.

The reusable post-hoc script is `scripts/analyze_feature_overlap.py`; its
summary artifacts are `hybrid_unknown_boundary_feature_diagnostics_s42.json`
and `hybrid_vs_local_boundary_feature_diagnostics_s42.json` in this directory.

## Reproducibility fix

The CLI previously wrote both training and discovery arguments to
`config.json`, so running `discover` in a training directory overwrote the
training configuration. Training commands now additionally preserve
`train_teacher_config.json` / `train_student_config.json`, and discovery writes
`discover_config.json`; legacy `config.json` remains for compatibility. A unit
test verifies the command-specific snapshots survive a later legacy-config
write. This fixes future traceability; it cannot restore the overwritten
training arguments for past runs.
