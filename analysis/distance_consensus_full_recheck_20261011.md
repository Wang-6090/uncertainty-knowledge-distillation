# Distance-consensus weighted nnPU full-data recheck (2026-10-11)

## Question

The previous seed-42 full-data run suggested that weighting the mixed-pool
uncertainty nnPU unlabeled term with `distance_consensus` might improve unknown
detection. That result was not sufficient to change the default: the method
had only one full-data seed and the gain could have been seed-specific.

This recheck therefore uses a paired seed-123 run. The comparison changes only
`discovery_uncertainty_pu_weight_source` (`none` versus `distance_consensus`).
The teacher, semantic-isolated CIFAR-100 60/40 split, student architecture,
training budget, mixed pool, known prior, checkpoint rule, and detector are
held fixed.

## Protocol

- CIFAR-100 semantic-isolated 60/40 split
- seed 123 for data splitting and training
- full known training/validation/test data
- pretrained ResNet-34 teacher and ResNet-18 student
- five student epochs, batch size 64, CUDA
- matched mixed discovery pool: 5400 samples, known prior 0.2
- audited uncertainty nnPU risk (`nu_corrected` convention)
- detector: `normalized_entropy_proto`, MC samples 4
- threshold: validation-only 95% known coverage
- clustering skipped; test set remains 6000 known + 4000 unknown

The two student configurations were generated from the same saved seed-42
configuration and differ only in the PU weight source, seed, checkpoint paths,
and output directories. The baseline reached best epoch 4 with validation
known accuracy 0.4467; the treatment reached best epoch 5 with 0.4407.

## Results

| Metric | none baseline | distance-consensus | treatment - baseline |
|---|---:|---:|---:|
| AUROC | 0.7180 | 0.7012 | -0.0167 |
| AUPR | 0.5938 | 0.5716 | -0.0222 |
| FPR95 | 0.7635 | 0.7800 | +0.0165 |
| OSCR | 0.3914 | 0.3720 | -0.0193 |
| Known acceptance | 95.92% | 94.75% | -1.17 pp |
| Unknown rejection | 13.53% | 13.18% | -0.35 pp |
| Score KS distance | 0.3348 | 0.3041 | -0.0308 |
| Unknown below known q95 | 83.75% | 87.60% | +3.85 pp |

The saved reports are:

- `runs/distance_consensus_full_s123_baseline_discover/discovery_report.json`
- `runs/distance_consensus_full_s123_treatment_discover/discovery_report.json`

## Decision

The seed-42 positive result does not replicate. Across the two full-data
seeds, `distance_consensus` is therefore seed-sensitive rather than a stable
improvement. It must not become the default training objective, and further
temperature or fusion-ratio sweeps are not justified at this stage.

## Component-level audit

The degradation is not only a threshold artifact. On the same seed-123 test
predictions, the treatment changed the individual signals as follows:

| Signal | Baseline AUROC / KS | Treatment AUROC / KS |
|---|---:|---:|
| Final `normalized_entropy_proto` score | 0.7180 / 0.3348 | 0.7012 / 0.3041 |
| Prototype distance | 0.7042 / 0.3053 | 0.6851 / 0.2812 |
| Predictive entropy | 0.6994 / 0.2993 | 0.6863 / 0.2785 |
| Head uncertainty | 0.6087 / 0.1741 | 0.6415 / 0.2072 |

Thus the weighted PU term makes the uncertainty head itself more predictive,
but simultaneously weakens the prototype-distance and entropy geometry that
the final detector relies on. The combined score consequently gets worse.
This is evidence of objective conflict, not evidence that the weight merely
needs a different threshold. It also explains why continuing to tune the same
weight source is unlikely to solve the feature overlap.

The implementation remains useful as an opt-in ablation because it is tested,
reproducible, and can diagnose whether local semantic distance is helpful on a
particular split. The current mainline should use the unweighted audited nnPU
configuration until a different representation-level method provides a
multi-seed gain.

## What this rules out and what comes next

This result rules out the claim that mixed-pool distance weighting alone fixes
the known/unknown overlap. The next experiments should first audit the source
of the overlap rather than add another score fusion:

1. Compare known and unknown features before and after student training using
   class-conditional distances, nearest-known support, and per-class overlap.
2. Verify that the semantic-isolated split is not substantially harder or
   differently sampled across seeds; report class counts and split hashes.
3. Test one direct feature-space objective at a time, with the same fixed
   detector and independent calibration subset. Any candidate must improve
   AUROC/OSCR or KS without trading the gain for known rejection.
4. Keep the final claim conservative: a method that only changes a selected
   operating point is not evidence that the feature distributions have been
   separated.
