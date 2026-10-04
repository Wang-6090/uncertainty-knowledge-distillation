# Candidate-gated feature separation recheck (seed 123, 2026-09-30)

## Purpose and control

The seed-42 pilot of candidate-gated feature separation was mixed: it slightly improved the exact operating point but did not improve AUROC and gave inconsistent feature diagnostics. This recheck uses a new seed under a strict paired protocol.

Both runs use CIFAR-100 semantic-hard 60/40, seed 123, full known training data, batch 32, 5 epochs, pretrained ResNet-34 teacher / ResNet-18 student, the same teacher checkpoint, the same mixed discovery pool, and the same normalized-entropy + Mahalanobis detector with a 95% known-coverage threshold. The only training difference is the candidate-gated feature separation term:

```text
alpha_discovery_feature_separation=0.05
discovery_feature_separation_margin=0.0
discovery_feature_separation_temperature=0.1
discovery_feature_candidate_gating=true
discovery_select_ratio=0.25
discovery_select_mode=entropy_uncertainty
```

## Detection result

| Method | AUROC | FPR95 | OSCR | Unknown rejection | Accepted-known accuracy | Test known acceptance |
|---|---:|---:|---:|---:|---:|---:|
| seed-123 baseline | 0.5063 | 0.9146 | 0.2443 | 4.22% | 40.71% | 94.64% |
| candidate gating | 0.5611 | 0.9112 | 0.2896 | 3.97% | 44.54% | 96.65% |

The candidate model improves AUROC, FPR95, OSCR, and accepted-known accuracy, but its validation threshold transfers to a higher-than-target known acceptance and a slightly lower unknown rejection. Therefore it is not a complete solution to the operating-point failure. Its best validation known accuracy also improves (`43.67% -> 49.00%`).

## Feature diagnostic result

Full-data test feature diagnostics support a real representation change:

- classifier-prototype distance AUROC: `0.5731 -> 0.5776`;
- empirical class-centroid distance AUROC: `0.5212 -> 0.5707`;
- nearest-known-training-sample distance AUROC: `0.5335 -> 0.5847`;
- centroid-distance histogram overlap: `0.8160 -> 0.8026`.

This is stronger evidence than the seed-42 pilot that candidate gating can improve backbone support separation. However, the final detector's threshold transfer remains unstable. The method should remain an experimental candidate, not the default, until it is checked on at least one more seed and with an independent open-validation split or a strict matched-coverage operating-point report.

## Interpretation

The result separates two questions that were previously conflated:

1. Does the loss change the feature geometry? In this seed, yes, with small-to-moderate positive diagnostic changes.
2. Does the current validation-threshold deployment reject more unknowns at the intended known coverage? Not yet.

The next work should focus on calibration/evaluation protocol and then a third-seed confirmation, rather than adding another generic feature loss.

