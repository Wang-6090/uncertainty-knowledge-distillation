# Strict candidate-gating comparison (seed 123, 2026-10-13)

## Question and controlled factor

This experiment tests whether enabling `discovery_feature_candidate_gating`
improves known/unknown score separation. It is compared with the seed-123
encoder-LR `0.1` baseline at
`runs/semantic_isolated_full_s123_disjointcal_lr01_student/` and its matching
detection report. Both arms use the same CIFAR-100 semantic-isolated 60/40
split, seed 123, pretrained ResNet-18 student / ResNet-34 teacher, 5 epochs,
encoder LR scale 0.1, uncertainty KD, feature KD 0.1, prototype loss 0.1,
SupCon 0.1, uncertainty PU loss 0.1 with `nu_corrected`, matched mixed pool
size 5,400 and known prior 0.2. The treatment changes only
`discovery_feature_candidate_gating` from false to true.

Detection settings are held fixed: `normalized_entropy_min_class_knn`, kNN
OOD enabled with k=10, MC=4, disjoint calibration, validation-only target
known coverage 95%, and clustering skipped. The official test set is used
only for final evaluation.

## Configuration audit

The first attempted gating run was invalid as a controlled comparison. Its
saved config showed `pretrained=false`, `alpha_feat_kd=0`, `alpha_proto=0`,
and legacy PU risk, while the baseline uses pretrained weights, feature KD,
prototype loss, and `nu_corrected`. Its validation known accuracy was only
39.17%, versus 62.02% at the baseline checkpoint. The run is retained as an
invalid historical comparison and is not evidence for or against gating.

The corrected treatment is saved under
`runs/semantic_isolated_full_s123_disjointcal_lr01_gating_strict_v2_student/`
and `..._gating_strict_v2_discover/`. The saved configs were checked: all
listed training and detection settings match the baseline except output paths,
checkpoint metadata, and the intended gating flag. Treatment selected epoch 4
with validation known accuracy 59.42%, below the baseline's 62.02%.

## Results

| Metric | Baseline | Gating | Difference (gating - baseline) |
|---|---:|---:|---:|
| AUROC | 0.8191 | 0.7985 | -0.0207 |
| AUPR | 0.7255 | 0.6879 | -0.0376 |
| FPR95 | 0.5797 | 0.6328 | +0.0532 (worse) |
| OSCR | 0.5602 | 0.5368 | -0.0234 |
| Known accuracy (all known) | 62.20% | 60.62% | -1.58 pp |
| Known acceptance | 95.85% | 94.97% | -0.88 pp |
| Unknown rejection | 25.10% | 25.70% | +0.60 pp |
| KS distance | 0.5000 | 0.4642 | -0.0358 |
| Unknown below known-score q95 | 70.30% | 74.93% | +4.63 pp (more overlap) |

The 0.60 pp increase in unknown rejection is small and comes with worse
ranking, OSCR, known accuracy, KS separation, and overlap. It is not a useful
tradeoff. Under this tested configuration, candidate gating does not solve
the overlap problem and should remain off by default. This is one seed, so it
does not establish that gating is universally harmful; it only fails the
prespecified single-seed screening criterion and does not justify a multi-seed
follow-up unless the gating mechanism itself is redesigned.

## Artifacts and next action

- Baseline: `runs/semantic_isolated_full_s123_disjointcal_lr01_student/` and
  `runs/semantic_isolated_full_s123_disjointcal_lr01_discover/`.
- Corrected treatment: `runs/semantic_isolated_full_s123_disjointcal_lr01_gating_strict_v2_student/`
  and `runs/semantic_isolated_full_s123_disjointcal_lr01_gating_strict_v2_discover/`.
- Invalid first treatment: `runs/semantic_isolated_full_s123_disjointcal_lr01_gating_student/`;
  retain only to document the configuration error, never use as a causal result.

Stop tuning this gating switch. The next useful work should target whether
the training pool and uncertainty objective provide reliable, class-balanced
unknown supervision, while keeping a disjoint open-validation protocol and
reporting score overlap, OSCR, and known-class utility—not only a thresholded
unknown-rejection percentage.
