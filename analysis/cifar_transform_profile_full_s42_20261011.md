# Full CIFAR transform-profile recheck, seed 42 (2026-10-11)

## Purpose

The short paired pilot suggested that CIFAR-specific preprocessing may improve
known/unknown separation. This full-data recheck tests whether that signal
survives under the same training scale used by the current strict seed-42
mainline.

## Protocol

Reference ImageNet-profile run:

- runs/semantic_isolated_full_s42_strict_feat_proto/
- historical transform behavior, equivalent to transform_profile=imagenet
- pretrained ResNet-34 teacher and ResNet-18 student
- full CIFAR-100 semantic-isolated 60/40, seed 42
- 5 student epochs
- matched mixed discovery pool: 5,400 samples, known prior 0.2
- feature KD and prototype loss enabled, no discovery uncertainty PU/margin
- detector: normalized_entropy_min_class_knn, feature kNN k=10, MC=4
- threshold: validation-known-only 95% known coverage
- clustering skipped

CIFAR-profile treatment:

- newly trained teacher: runs/semantic_isolated_full_s42_cifar_profile_teacher/
- newly trained student: runs/semantic_isolated_full_s42_cifar_profile_student/
- same architecture, seed, split, data scale, losses, matched pool and detector
- transform_profile=cifar for both teacher and student

The teacher was retrained for the CIFAR profile to avoid KD input-distribution
mismatch.

## Results

| Profile | teacher val acc | student val acc | test known acc | AUROC | AUPR | FPR95 | OSCR | known accept | unknown reject | score KS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ImageNet profile | 43.87% | 40.80% | 41.37% | 0.6939 | 0.5596 | 0.7622 | 0.3447 | 95.22% | 11.75% | 0.2893 |
| CIFAR profile | 57.33% | 56.30% | 56.65% | 0.7710 | 0.6503 | 0.6547 | 0.4934 | 95.08% | 19.35% | 0.4173 |
| difference | +13.46 pp | +15.50 pp | +15.28 pp | +0.0771 | +0.0907 | -0.1075 | +0.1487 | -0.13 pp | +7.60 pp | +0.1280 |

## Interpretation

This is the strongest positive result from the recent debugging cycle. The
CIFAR profile improves closed-set known accuracy, open-set ranking, FPR95,
OSCR, unknown rejection and score-distribution separation, while preserving
roughly the same known acceptance rate at the validation-calibrated operating
point.

The current evidence shifts the next priority away from adding more losses.
The immediate next step should be protocol consolidation:

1. repeat the CIFAR-profile run on at least seed 123, and preferably seed 3407;
2. keep the detector, threshold policy and matched pool fixed;
3. use scripts/audit_paired_runs.py to verify paired comparisons;
4. if the improvement repeats, set transform_profile=cifar as the recommended
   CIFAR-100 protocol while keeping imagenet available for historical runs.

This still does not fully solve unknown detection: unknown rejection is 19.35%,
so most unknown samples are still accepted. But it changes the feature/score
distribution in the right direction and is more impactful than the recent
loss-level tweaks.

Artifacts:

- runs/semantic_isolated_full_s42_cifar_profile_teacher/
- runs/semantic_isolated_full_s42_cifar_profile_student/
- runs/semantic_isolated_full_s42_cifar_profile_discover/discovery_report.json

