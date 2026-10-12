# Encoder LR-scale strict replication, seed 123 (2026-10-12)

## Question and controlled comparison

Does reducing the student encoder LR scale from 1.0 to 0.1 improve the
known/unknown feature-score separation, beyond a threshold change? This
replicates the positive seed-44 disjoint-calibration pilot on a second seed.

Both arms used CIFAR-100 semantic-isolated 60/40, seed 123, the same pretrained
ResNet-34 teacher, ResNet-18 student, full train/validation/test splits, five
epochs, batch size 64, mixed discovery training pool of size 5,400 with known
prior 0.2, uncertainty KD, nnPU, optimizer/loss/augmentation settings, and the
same min-class kNN detector (`k=10`, MC=4). Validation checkpoint selection
used 2,272 known examples; a disjoint 728-example known subset calibrated the
95% known-coverage threshold. The sample-index audit reports zero overlap.
Only `encoder_lr_scale` changed (1.0 vs 0.1). Test labels were used only for
retrospective evaluation. Discovery itself was run in detector-only mode with
clustering skipped; its logged default unknown-only discovery pool is not used
by this score/calibration path.

An earlier pair of seed-123 checkpoints was explicitly excluded: their configs
differed in `discovery_pool_mode` (unknown vs mixed), nnPU loss coefficient
(0 vs 0.1), and known prior (absent vs 0.2), in addition to LR scale. They
cannot establish a causal LR effect.

## Results

| Metric | LR scale 1.0 | LR scale 0.1 | Change |
|---|---:|---:|---:|
| AUROC | 0.7945 | **0.8191** | +0.0246 |
| AUPR | 0.6871 | **0.7255** | +0.0384 |
| FPR95 | 0.6403 | **0.5797** | -0.0607 |
| OSCR | 0.5331 | **0.5602** | +0.0271 |
| Known accuracy, all known | 60.35% | **62.20%** | +1.85 pp |
| Test known acceptance | 94.58% | 95.85% | +1.27 pp |
| Unknown rejection | 25.18% | 25.10% | -0.08 pp |
| Score KS distance | 0.4590 | **0.5000** | +0.0410 |
| Unknown below known q95 overlap | 76.35% | **70.30%** | -6.05 pp |

Both calibration subsets accepted 94.92% of known samples, exactly the same
calibration data/procedure. Test operating-point deviations are independent
generalization variation, not post-hoc threshold tuning.

The second seed replicates the direction for continuous ranking and score
distribution separation, and known classification also improves. The
95%-coverage operating-point benefit is not replicated: unknown rejection is
essentially unchanged at this seed, unlike the seed-44 pilot where it improved
by 9.1 percentage points. Thus scale 0.1 is a promising representation
training setting, not a reliable unknown-rejection solution. About 75% of
unknown test samples are still accepted, and 70.3% of unknown scores remain
below the known q95 score. The core overlap remains substantial.

## Decision and next experiment

Keep LR scale 0.1 as the better-supported working baseline, not a proven
universal optimum. The two-seed evidence is directionally consistent for
AUROC/FPR95/OSCR/score separation but seed-sensitive at the calibrated
operating point. Do not claim the core problem is solved. Next, inspect the
per-class and score-tail diagnostics under this fixed setting, then test one
training-time geometry intervention against LR-0.1 with the same disjoint
calibration protocol. Any claimed rejection improvement must preserve known
acceptance and known classification; AUROC alone is insufficient.

## Artifacts and validation

- Script: `scripts/run_encoder_lr_disjoint_calibration_s123.ps1`.
- LR 1.0: `runs/semantic_isolated_full_s123_disjointcal_lr10_student/` and
  `runs/semantic_isolated_full_s123_disjointcal_lr10_discover/`.
- LR 0.1: `runs/semantic_isolated_full_s123_disjointcal_lr01_student/` and
  `runs/semantic_isolated_full_s123_disjointcal_lr01_discover/`.
- Both final training configs were compared; apart from LR, output paths,
  checkpoint-selection outcomes, and derived prototype-init stats, no
  algorithm/protocol configuration differed.
- Both calibration reports have `disjoint=true`, overlap count 0, and 728
  calibration examples.
- Full test suite after sample-weight autograd fixes: 217 passed, 2 warnings;
  `compileall` and `git diff --check` passed.
