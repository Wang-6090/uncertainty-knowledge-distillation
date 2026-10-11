# Strict feature-KD / prototype-loss ablation, seed 42 (2026-10-11)

## Purpose

The core failure is still overlap between known and unknown feature/score
distributions. A seed-123 strict ablation gave mixed evidence about feature KD
and prototype loss. This seed-42 recheck asks whether that conclusion was
stable, while also auditing that the paired runs really differ only in the two
intended loss weights.

## Protocol

Fixed settings:

- CIFAR-100 semantic-isolated 60/40 split, seed 42
- pretrained ResNet-34 teacher and ResNet-18 student
- full training/validation/test data, 5 student epochs
- matched mixed discovery pool: 5,400 samples, known prior 0.2
- discovery uncertainty PU and feature-margin losses disabled in both arms
- detector: normalized_entropy_min_class_knn, feature kNN k=10, MC=4
- threshold: validation-known-only 95% known coverage
- clustering skipped

The only non-ignored configuration differences reported by
scripts/audit_paired_runs.py are:

- alpha_feat_kd: 0.1 vs 0.0
- alpha_proto: 0.1 vs 0.0

A previous seed-42 pair was rejected as non-strict because it had 39
configuration differences, including different discovery uncertainty losses.

## Results

| Variant | AUROC | AUPR | FPR95 | OSCR | known acc | known accept | unknown reject | score KS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| feature KD + prototype loss | 0.6939 | 0.5596 | 0.7622 | 0.3447 | 41.37% | 95.22% | 11.75% | 0.2893 |
| both disabled | 0.6829 | 0.5436 | 0.7680 | 0.3214 | 38.43% | 94.92% | 10.83% | 0.2840 |
| difference | +0.0110 | +0.0160 | -0.0058 | +0.0233 | +2.93 pp | +0.30 pp | +0.93 pp | +0.0053 |

## Interpretation

This strict seed-42 pair supports keeping feature KD and prototype loss: they
improve ranking, OSCR, known accuracy, and the validation-calibrated unknown
rejection point. However, the absolute improvement is small and unknown
rejection remains only about 12%. These losses are useful supporting terms, not
the main solution to known/unknown overlap.

Together with the seed-123 result, the decision is:

1. keep feature KD + prototype loss in the current main recipe;
2. stop spending experiments on small weight tweaks for these two terms;
3. use scripts/audit_paired_runs.py before claiming a paired ablation;
4. shift algorithmic effort toward representation/data-protocol changes that
   directly reduce known/unknown overlap.

Artifacts:

- runs/semantic_isolated_full_s42_strict_feat_proto/
- runs/semantic_isolated_full_s42_strict_no_feat_proto/
- runs/semantic_isolated_full_s42_strict_feat_proto_discover/discovery_report.json
- runs/semantic_isolated_full_s42_strict_no_feat_proto_discover/discovery_report.json

