# CIFAR profile + audited mixed-pool nnPU, seed 123

## Purpose

After confirming that CIFAR-style preprocessing improves the upstream
representation, this experiment checks whether the audited mixed-pool
uncertainty nnPU objective still adds value on top of that stronger baseline.

The goal is directly tied to the core problem: known/unknown features and
scores still overlap heavily, so the unknown rejection rate remains low.

## Controlled protocol

- Dataset: CIFAR-100 semantic-isolated 60/40.
- Seed: 123.
- Transform profile: CIFAR.
- Teacher: the same CIFAR-profile ResNet-34 teacher.
- Student: pretrained ResNet-18, 5 epochs.
- Detector: `normalized_entropy_min_class_knn`, k=10, MC=4.
- Threshold: validation-only 95% known coverage.
- Clustering: skipped.
- Discovery evaluation config: unchanged except for the student checkpoint.

The baseline student has no discovery nnPU objective and uses the matched
novel-only discovery pool. The treatment enables:

- `alpha_discovery_uncertainty_pu=0.1`
- `discovery_pool_mode=mixed`
- matched discovery pool size 5400
- known prior / mixed known ratio 0.2
- `discovery_uncertainty_pu_risk=nu_corrected`
- no additional semantic sample weighting

The training config audit shows the expected differences:

```
- alpha_discovery_uncertainty_pu: 0.0 -> 0.1
- discovery_pool_known_proportion: 0.0 -> 0.2
- discovery_pool_mode: unknown -> mixed
- discovery_pool_semantics: novel-only -> matched mixed known/novel pool
- discovery_uncertainty_known_prior_resolved: null -> 0.2
```

The discover config audit shows only the student checkpoint/output path and
the newly added requested-profile bookkeeping differ; detector settings are
unchanged.

## Results

| Student | val known acc | test known acc | AUROC | AUPR | FPR95 | OSCR | known accept | unknown reject | score KS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CIFAR profile baseline | 54.83% | 54.53% | 0.7550 | 0.6314 | 0.6793 | 0.4726 | 95.20% | 16.08% | 0.3972 |
| CIFAR profile + nnPU | 59.03% | 60.73% | 0.7933 | 0.6701 | 0.6088 | 0.5349 | 95.68% | 19.78% | 0.4547 |

Score-distribution changes:

- Mean unknown-minus-known score gap: 1.7302 -> 2.1143.
- Median gap: 2.1207 -> 2.6894.
- Unknown below known q95 overlap: 0.8303 -> 0.7805.
- Known above unknown q05 overlap / FPR95 proxy: 0.6795 -> 0.6088.

## Conclusion

This is a useful positive result. Under the stronger CIFAR preprocessing
protocol, audited mixed-pool uncertainty nnPU improves both known accuracy and
open-set separation metrics. The gain is not merely a threshold artifact:
AUROC, AUPR, FPR95, OSCR, unknown rejection, score KS, and overlap diagnostics
all move in the right direction.

The core problem is still not solved. At 95% known-coverage calibration, about
80% of unknown samples are still accepted. However, this combination is now a
more credible mainline than earlier threshold-only or score-fusion tweaks.

## Next steps

1. Repeat this exact comparison on at least one more seed before calling it
   stable.
2. If the gain repeats, make CIFAR profile + audited mixed-pool nnPU the
   recommended CIFAR-100 recipe.
3. Then test semantic weighting or feature-margin variants only as single
   changes on top of this recipe.
4. Continue using fixed detector/threshold protocols and never select
   hyperparameters from test labels.
