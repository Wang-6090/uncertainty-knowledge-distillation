# Quality-aware logits-KD fresh confirmation summary

This confirmation stage tested the fixed quality-aware teacher true-class
confidence weighting on fresh semantic-hard CIFAR-100 60/40 seeds `2155` and
`2156`. Each seed had standard logits-KD, unweighted decoupled KD and
quality-aware logits-KD arms. Teacher, split, initialization, training budget,
auxiliary objectives, checkpoint selection, detector, calibration, candidate
pool and automatic-K clustering were matched within seed. Feature KD was off.
Unknown labels were isolated from training, checkpoint selection and
calibration.

## Quality-aware versus standard KD

Positive direction is higher for every metric except FPR95 and Auto-K error,
where lower is better. Values are paired seed differences; `sd` is the sample
standard deviation across the two fresh seeds.

| metric | seed 2155 | seed 2156 | mean change | sd |
|---|---:|---:|---:|---:|
| AUROC | -0.008288 | +0.020908 | +0.006310 | 0.020645 |
| AUPR | -0.026399 | +0.035464 | +0.004533 | 0.043744 |
| FPR95 | -0.016667 | -0.009667 | -0.013167 | 0.004950 |
| OSCR | +0.024564 | +0.015777 | +0.020170 | 0.006213 |
| known accuracy | +2.683 pp | +1.117 pp | +1.900 pp | 1.108 pp |
| known acceptance | +1.267 pp | -0.867 pp | +0.200 pp | 1.509 pp |
| unknown rejection | -7.200 pp | +8.750 pp | +0.775 pp | 11.278 pp |
| candidate purity | -1.792 pp | +3.681 pp | +0.945 pp | 3.870 pp |
| Auto-K absolute error | +7 | -7 | 0 | 9.899 |
| cluster ACC | -0.040531 | +0.044436 | +0.001952 | 0.060080 |
| cluster NMI | -0.044604 | +0.037641 | -0.003481 | 0.058156 |
| cluster ARI | -0.042205 | +0.071881 | +0.014838 | 0.080671 |

## Gates and mechanism audit

- Both seeds completed all three arms, discovery, automatic-K clustering and
  summary validation.
- Known-validation calibration realized acceptance was `0.949176` on 2155 and
  `0.949384` on 2156 for all arms.
- Both result reports state `test_labels_used_for_training_or_calibration=false`.
- Quality-aware weights were applied in all four epochs on both seeds, with
  final standard deviation / ESS fraction `0.541676 / 0.773068` on 2155 and
  `0.512310 / 0.791916` on 2156.
- KD gradients and student classifier updates were nonzero; teacher weights
  remained detached/frozen.

## Decision

The implementation and protocol gates pass, but the end-to-end effect is
seed-sensitive. Seed 2155 shows a substantial loss in unknown rejection,
candidate purity and clustering, while seed 2156 shows the opposite pattern.
The mean gains are small relative to their across-seed variation, and NMI,
unknown rejection and Auto-K do not provide stable joint improvement.

Under the pre-registered promotion rule, quality-aware KD is therefore
**frozen as an opt-in diagnostic**. The default remains `kd_mode=standard`.
No quality-floor, power, temperature or loss-coefficient scan is justified on
these results. A future formal A/B/C aggregate may include it as a declared
ablation, but it should not be presented as a confirmed algorithmic gain.

Machine results: `analysis/team_a_quality_kd_confirm_s2155.json` and
`analysis/team_a_quality_kd_confirm_s2156.json`.
