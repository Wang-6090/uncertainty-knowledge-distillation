# Rejection-feature KD with MC epistemic weighting (seed 2105)

## Why this follow-up was needed

The seed-2102 and seed-2104 feature-KD runs used the teacher's learned uncertainty head. A post-run audit found that the seed-2104 teacher output was exactly `1.0` for every checked known-validation example; the logged raw `exp(-u)` weights therefore had standard deviation around `4e-9`, i.e. effectively a constant `e^-1`. Those runs measured adding a constant-scaled rejection-feature KD term, not sample-wise uncertainty redistribution. Their endpoint comparisons remain valid for those exact implementations, but they cannot support a causal claim about uncertainty weighting.

This fresh split therefore compares uniform rejection-feature KD with a non-degenerate MC epistemic source, while holding main logits KD at standard/unweighted in both arms.

## Protocol

- Fresh CIFAR-100 semantic-hard 60/40 split, seed 2105.
- Uniform arm: `alpha_rejection_feat_kd=0.1`, `rejection_feature_kd_weighting=uniform`.
- Treatment: same coefficient with `rejection_feature_kd_weighting=teacher_uncertainty` and `kd_uncertainty_source=mc_epistemic`.
- Both arms share one frozen split-matched teacher, student initialization/recipe, four-epoch budget, mixed-pool nnPU, known-validation checkpoint selection, feature-rejector scoring, and known-validation-only 95% coverage calibration.
- `kd_mode=standard` for both arms; all other training and discovery settings are fixed. No mixed-pool or test labels enter training, checkpoint/score selection, or calibration.
- The runner requires non-degenerate uncertainty-weight spread in each epoch and records it. The treatment passed: batch/epoch-aggregated standard deviation was `0.01448, 0.01474, 0.01456, 0.01467`.

## Mechanism audit

Both arms had active rejection-feature KD and nonzero rejection-parameter gradients. Losses by epoch:

| Epoch | Uniform loss | MC-weighted loss | Uniform gradient norm | MC-weighted gradient norm |
|---:|---:|---:|---:|---:|
| 1 | 0.425725 | 0.424166 | 0.131706 | 0.130714 |
| 2 | 0.285026 | 0.280082 | 0.075774 | 0.073966 |
| 3 | 0.253863 | 0.249159 | 0.058420 | 0.057136 |
| 4 | 0.235436 | 0.229657 | 0.049900 | 0.050108 |

The weighted branch had actual sample-level uncertainty variation; the loss and gradient audits confirm the term reached the rejection branch.

## Paired outcomes

| Metric | Uniform | MC epistemic weighted | Difference |
|---|---:|---:|---:|
| AUROC | 0.777539 | 0.779926 | +0.002388 |
| AUPR | 0.682091 | 0.684291 | +0.002201 |
| FPR95 (lower is better) | 0.659667 | 0.658167 | -0.15 pp |
| OSCR | 0.567364 | 0.567612 | +0.000248 |
| Test known acceptance | 93.50% | 93.68% | +0.18 pp |
| Unknown rejection at calibrated threshold | 29.75% | 28.50% | -1.25 pp |
| Accepted-known accuracy | 70.27% | 70.31% | +0.04 pp |
| Correctly accepted fraction of all known | 65.70% | 65.87% | +0.17 pp |
| Automatic K / NMI / ARI | 35 / 0.46594 / 0.13532 | 36 / 0.46380 / 0.12767 | +1 / -0.00214 / -0.00765 |

The treatment modestly improved AUROC/AUPR/FPR95, with essentially unchanged OSCR and known utility, but reduced fixed-coverage unknown rejection by `1.25 pp` and reduced NMI/ARI. It fails the predeclared joint criterion requiring at least `+2 pp` unknown rejection. This is an interpretable negative result for the tested MC-epistemic rejection-feature weighting recipe, not proof that every uncertainty signal or uncertainty-KD formulation is ineffective.

## Decision and correction to earlier interpretation

Do not promote MC-epistemic weighting or uniform rejection-feature KD from this single split; stop this exact weighting variant without tuning on seed 2105. Rejection-feature KD itself remains unresolved because both arms use it. The earlier seed-2102/2104 results must be described as head-uncertainty weighting with a saturated/near-constant teacher signal, not as successful tests of informative sample-wise uncertainty weighting. The machine result is [`rejection_feature_kd_mc_epistemic_s2105.json`](rejection_feature_kd_mc_epistemic_s2105.json).
