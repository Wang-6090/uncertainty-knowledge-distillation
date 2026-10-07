# Uncertainty-weighted discovery feature margin recheck (2026-10-05)

## Question and hypothesis

The persistent issue is overlap between known and novel feature/support
distributions. This experiment tests whether an unlabeled discovery image can
be pushed away from the nearest known classifier prototype in proportion to
the student's detached uncertainty estimate, without assigning it a hard
unknown label.

For normalized feature `z`, normalized known classifier prototypes `w_c`,
margin `m`, and detached uncertainty `u`, the implemented per-view term is

```text
d(z) = max_c cosine(z, w_c)
L = sum_i u_i * max(0, d(z_i) - m) / sum_i u_i
```

It is averaged over the two discovery views and added with weight
`alpha_discovery_uncertainty_feature_margin`. This is a project hypothesis
inspired by uncertainty-guided weighting and margin-based representation
learning; it is not a reproduction of a named paper. It uses the existing
nnPU-trained uncertainty head as a soft gate, so calibration errors in that
head remain a potential failure mode.

## Controlled protocol

- CIFAR-100, random 60/40 class split, full data, seeds 42, 123, and 3407.
- Matched pretrained ResNet-34 teacher and pretrained ResNet-18 student;
  five epochs, batch size 64, mixed discovery pool, known-pool ratio 0.2.
- Both arms use immediate mixed-pool uncertainty nnPU (`alpha=0.1`, automatic
  known prior resolved to 0.212598). Treatment adds only
  `alpha_discovery_uncertainty_feature_margin=0.05`, margin `0.2`.
- Detection was rerun for both arms with the same
  `normalized_entropy_min_class_knn` score, feature kNN `k=10`, MC=8, and a
  known-validation threshold targeting 95% known coverage. Clustering was
  skipped to isolate detection.
- The paired confirmation was completed for seed 3407 using its matching
  pretrained ResNet-34 teacher and otherwise identical training/evaluation
  configuration.
- Feature overlap was recomputed over the complete 27,000 known training
  images and 10,000 open test images. Test labels were used only to report
  known/unknown descriptive distributions and AUROC; they did not enter
  training, threshold fitting, or model selection.

## Matched open-set results

| Seed | Arm | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection | Accepted-known accuracy |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | nnPU baseline | 0.7012 | 0.7590 | 0.4110 | 95.55% | 13.40% | 51.19% |
| 42 | + uncertainty feature margin | 0.7108 | 0.7348 | 0.4132 | 95.30% | 13.90% | 50.87% |
| 123 | nnPU baseline | 0.6830 | 0.7693 | 0.4071 | 94.48% | 14.08% | 51.91% |
| 123 | + uncertainty feature margin | 0.7091 | 0.7282 | 0.4449 | 94.62% | 14.88% | 55.33% |
| 3407 | nnPU baseline | 0.6868 | 0.7685 | 0.4115 | 95.42% | 12.70% | 51.56% |
| 3407 | + uncertainty feature margin | 0.6969 | 0.7595 | 0.4066 | 94.95% | 13.35% | 50.80% |

Across three seeds, mean AUROC improved from 0.6903 to 0.7056 (+0.0153),
mean FPR95 decreased from 0.7656 to 0.7408 (-0.0248), mean OSCR improved from
0.4099 to 0.4216 (+0.0117), and mean unknown rejection increased from 13.39%
to 14.04% (+0.65 percentage points). Known acceptance changed from 95.15% to
94.96% (-0.19 points), and accepted-known accuracy from 51.56% to 52.33%
(+0.78 points). AUROC, FPR95, and unknown rejection moved in the favorable
direction on all three seeds; OSCR and accepted-known accuracy did not (seed
3407 regressed). This is a promising, reasonably consistent ranking signal,
but the effect on task utility is mixed and three seeds are not enough to
claim a robust final method.

## Full-data feature-overlap diagnostics

Overlap is the histogram overlap in `[0,1]` (lower is better); distance AUROC
uses larger distance as the unknown score. All three diagnostics moved in the
desired direction on both seeds:

| Seed | Distance | Baseline AUROC / overlap | Treatment AUROC / overlap |
| ---: | --- | ---: | ---: |
| 42 | Classifier prototype | 0.6639 / 0.7496 | 0.6794 / 0.7312 |
| 42 | Empirical class centroid | 0.6704 / 0.7537 | 0.6759 / 0.7413 |
| 42 | Nearest known training sample | 0.7074 / 0.6937 | 0.7142 / 0.6782 |
| 123 | Classifier prototype | 0.6657 / 0.7528 | 0.6824 / 0.7241 |
| 123 | Empirical class centroid | 0.6574 / 0.7714 | 0.6847 / 0.7301 |
| 123 | Nearest known training sample | 0.6885 / 0.7243 | 0.7109 / 0.6830 |
| 3407 | Classifier prototype | 0.6513 / 0.7667 | 0.6740 / 0.7374 |
| 3407 | Empirical class centroid | 0.6695 / 0.7535 | 0.6730 / 0.7474 |
| 3407 | Nearest known training sample | 0.6967 / 0.7116 | 0.7030 / 0.7037 |

All three distance-based unknownness AUROCs improved for each seed, and all
nine overlap estimates decreased. This supports a representation-geometry
effect beyond a threshold shift. The effects are not uniform in size (for
seed 3407, centroid AUROC gain is small), and substantial overlap remains
(roughly 0.68--0.77). The reported histogram overlap is bin-dependent and
should be treated as a descriptive diagnostic, not a formal hypothesis test.

## Validity checks and excluded attempts

- The first feature-diagnostic invocation omitted `--limit-train 0`,
  `--limit-val 0`, and `--limit-test 0`; the script therefore used its small
  defaults (1200/300/1000). Those two outputs are pilot-only and are not used
  in the tables above. The full-data diagnostics were rerun explicitly.
- One seed-123 train command omitted `--discovery-pool` and was rejected before
  training. A second command used the wrong default teacher architecture and
  failed checkpoint loading before training. Neither produced a checkpoint
  nor contributes to the results. The valid run explicitly used the matching
  ResNet-34 teacher and `--discovery-pool`.
- The hard-proxy margin result in
  `analysis/hard_proxy_margin_pilot_s42_20261005.md` is a separate older
  experiment with a different split/checkpoint protocol; it is not pooled
  with this paired comparison.

## Decision and next experiment

Keep this loss optional and default-off; do not replace the current baseline
or claim the core problem is solved. The three-seed confirmation justified a
small, predeclared weight ablation (`alpha=0.025` versus `0.05`) with matching
seed 42. Its negative result is reported below. Do not tune thresholds to
rescue a negative result.

### Follow-up: lower-weight ablation (`alpha=0.025`, seed 42)

The predeclared lower-weight run used the same seed-42 teacher, data split,
training protocol, and detector; only the feature-margin coefficient changed
from 0.05 to 0.025. Results were:

| Arm | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection | Accepted-known accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nnPU baseline | 0.7012 | 0.7590 | 0.4110 | 95.55% | 13.40% | 51.19% |
| + margin, alpha=0.05 | 0.7108 | 0.7348 | 0.4132 | 95.30% | 13.90% | 50.87% |
| + margin, alpha=0.025 | 0.6934 | 0.7275 | 0.3978 | 94.97% | 12.28% | 49.25% |

At alpha=0.025, all three full-data geometry AUROCs were slightly lower than
baseline; prototype and centroid histogram overlap worsened (0.7496 to
0.7633 and 0.7537 to 0.7598), while nearest-sample overlap improved only
slightly (0.6937 to 0.6905). Overall AUROC, OSCR, unknown rejection, known
acceptance, and accepted-known accuracy all regressed. FPR95 alone improved,
which is not sufficient evidence of better separation. Therefore do not
continue blind coefficient sweeps: keep alpha=0.05 as a promising but
unconfirmed candidate and alpha=0.025 as a negative ablation. The next
diagnostic should inspect the uncertainty gate itself (weight distribution,
whether high weights have high unknown purity, and known contamination),
before considering a different smooth gate or a staged/ramped version.

The complete checkpoints and reports are under `runs/uncertainty_feature_margin_*`;
full geometry outputs are `analysis/uncertainty_feature_margin_overlap_s42_full.json`,
`analysis/uncertainty_feature_margin_overlap_s123_full.json`, and
`analysis/uncertainty_feature_margin_overlap_s3407_full.json`.
