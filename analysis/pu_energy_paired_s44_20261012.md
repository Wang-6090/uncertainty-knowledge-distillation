# PU-energy one-factor replication (seeds 44 and 123, 2026-10-12)

## Question

Does adding a PU-corrected energy loss to the current CIFAR-100 candidate
training recipe improve known/unknown separation, beyond the existing
uncertainty nnPU loss?

## Controlled comparison

- CIFAR-100 semantic-isolated 60/40 split, seed 44, full train/validation/test.
- Same pretrained ResNet-34 teacher and ResNet-18 student recipe, CIFAR
  transform profile, encoder LR scale 0.1, five epochs, and matched mixed
  discovery pool (5,400 samples; known prior 0.2).
- Both runs retain `alpha_discovery_uncertainty_pu=0.1` with
  `nu_corrected`; treatment adds only `alpha_discovery_energy_pu=0.1`, using
  hard nnPU and known prior 0.2. A config comparison found this as the sole
  effective training-factor difference; output/checkpoint paths and derived
  best-validation-accuracy fields naturally differ.
- Same detector (`normalized_entropy_min_class_knn`, MC=4), clustering
  disabled, validation-only 95% known-coverage threshold, and disjoint
  calibration ratio 0.25. Both calibration subsets accepted 94.9246% known
  samples.
- Full test set: 6,000 known and 4,000 unknown. Test labels were used only for
  final evaluation.

The treatment's `train_history.json` logs non-zero `discovery_energy_pu`
before weighting (epoch 1: 0.3431; epoch 5: 0.0144), confirming the loss was
connected and active. This verifies execution, not by itself the quality of
the learned signal.

## Results: seed 44

| Metric | baseline | PU-energy | Change (treatment - baseline) |
|---|---:|---:|---:|
| AUROC | 0.8089 | 0.8152 | +0.0062 |
| AUPR | 0.7037 | 0.7169 | +0.0132 |
| FPR95 | 0.5888 | 0.5942 | +0.0053 (worse) |
| OSCR | 0.5217 | 0.5204 | -0.0013 (worse) |
| Known classification accuracy | 58.10% | 57.77% | -0.33 pp |
| Test known acceptance | 93.88% | 94.82% | +0.93 pp |
| Unknown rejection | 30.50% | 27.85% | -2.65 pp |
| Score KS distance | 0.4807 | 0.4899 | +0.0093 |
| Unknown below known q95 (lower is better) | 0.7385 | 0.7273 | -0.0113 |
| Mean unknown-known score gap | 2.1712 | 2.1789 | +0.0077 |

## Independent seed-123 replication

The same single-factor treatment was then run on seed 123. Its baseline is
`runs/semantic_isolated_full_s123_disjointcal_lr01_student/` with the matching
`..._discover/`; treatment artifacts are in
`runs/semantic_isolated_full_s123_disjointcal_lr01_energypu_student/` and
`..._energypu_discover/`. Config comparison again found the only active
training-factor change to be PU-energy weight `0 -> 0.1` (the
`rejector_known_support_transform` field differs in serialized configs, but
the rejection-feature branch and its losses are disabled in both recipes).

| Metric | baseline | PU-energy | Change |
|---|---:|---:|---:|
| AUROC | 0.8191 | 0.8265 | +0.0074 |
| AUPR | 0.7255 | 0.7383 | +0.0128 |
| FPR95 | 0.5797 | 0.5845 | +0.0048 (worse) |
| OSCR | 0.5602 | 0.5603 | +0.0001 (effectively unchanged) |
| Known classification accuracy | 62.20% | 61.82% | -0.38 pp |
| Test known acceptance | 95.85% | 95.28% | -0.57 pp |
| Unknown rejection | 25.10% | 29.58% | +4.48 pp |
| Score KS distance | 0.5000 | 0.5108 | +0.0108 |
| Unknown below known q95 | 70.30% | 69.15% | -1.15 pp |
| Mean unknown-known score gap | 2.3490 | 2.3590 | +0.0101 |

Both arms' validation calibration acceptance is approximately 94.92%.

## Interpretation and decision

This is **mixed, not a demonstrated improvement**. Across both seeds,
AUROC/AUPR, KS, and one tail-overlap statistic move slightly in the favorable
direction, but FPR95 worsens in both, OSCR is flat or worse, and known
classification slips. Most importantly, calibrated unknown rejection moves
in opposite directions (-2.65 pp at seed 44, +4.48 pp at seed 123), while
mean unknown-known score gaps barely change (+0.0077 and +0.0101). Thus the
loss did not robustly move unknown examples away from known support; the
small ranking gains alone are not enough to retain it in the main recipe.

There was an earlier seed-42, three-epoch PU-energy pilot recorded in the
README that was negative on AUROC/FPR95/OSCR. This two-seed replication does
not reverse that conclusion: gains are small and the operating-point effect
is seed-sensitive. Keep the option for research ablations, but do not enable
it by default or continue weight scanning on this evidence.

Next priority: stop adding another global scalar loss until its PU risk
assumptions and gradients are checked against the actual mixed-pool sampling
process. Prefer diagnostics/experiments that reveal whether the fixed prior
0.2 and the energy distributions are calibrated per batch and whether the
energy gradient moves unknown candidates away without degrading known-class
geometry. Any follow-up should use at least two matched seeds and judge
AUROC, FPR95, OSCR, KS/overlap, known accuracy/acceptance, and unknown
rejection together.

## Artifacts

- Training: `runs/semantic_isolated_full_s44_disjointcal_lr01_energypu_student/`
- Detection: `runs/semantic_isolated_full_s44_disjointcal_lr01_energypu_discover/`
- Reproduction script: `scripts/run_energy_pu_on_lr01_s44.ps1`
- Second-seed script: `scripts/run_energy_pu_on_lr01_s123.ps1`
- Matched baseline: `runs/semantic_isolated_full_s44_disjointcal_lr01_student/`
  and `runs/semantic_isolated_full_s44_disjointcal_lr01_discover/`
- Seed-123 baseline: `runs/semantic_isolated_full_s123_disjointcal_lr01_student/`
  and `runs/semantic_isolated_full_s123_disjointcal_lr01_discover/`
