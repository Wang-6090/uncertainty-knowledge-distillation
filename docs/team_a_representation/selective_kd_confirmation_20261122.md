# Teacher-correct selective logits KD confirmation

## Scope and fixed protocol

This confirmation belongs only to the student classification representation
and uncertainty-KD audit. It compares standard logits KD, teacher-correct
selective logits KD, and the existing uncertainty-KD on fresh semantic-hard
CIFAR-100 60/40 splits with seeds `2163` and `2164`.

The three arms for each seed shared the split-matched teacher, student
initialization, four-epoch budget, known-validation checkpoint selection,
max-softmax score, known-validation 95% coverage calibration, matched mixed
discovery pool, automatic K, and clustering procedure. Feature KD,
rejection-branch KD, discovery/PU training objectives, and rejector training
were disabled. Test and unknown labels were used only for final retrospective
metrics. No gate ratio, loss coefficient, temperature, or score rule was
scanned.

The machine-readable results are `analysis/team_a_selective_kd_confirm_20261122_s2163.json`,
`analysis/team_a_selective_kd_confirm_20261122_s2164.json`, and
`analysis/team_a_selective_kd_confirm_20261122_summary.json`.

## Mechanism audit

The mechanism gates passed for both confirmation seeds:

| Check | Seed 2163 | Seed 2164 |
|---|---:|---:|
| Selective active fraction by epoch | 0.6494, 0.6475, 0.6509, 0.6475 | 0.6386, 0.6344, 0.6403, 0.6358 |
| Selective KD applied every epoch | yes | yes |
| Selective KD gradient nonzero every epoch | yes | yes |
| Student classifier update nonzero every epoch | yes | yes |
| Standard arm uncertainty weighting | never applied | never applied |
| Uncertainty arm weighting | applied every epoch | applied every epoch |

The selective arm's KD gradient norms ranged from `11.7466` to `19.9858`
on seed 2163 and from `12.6224` to `21.3102` on seed 2164. Its student
classifier updates were nonzero in every epoch. The uncertainty arm had
nonzero weight spread in every epoch, with standard deviations ranging from
`0.1814` to `0.2024` and effective sample size fractions from `0.9605` to
`0.9680`. The teacher output was detached and the optimizer updated student
parameters only. The gate did not use the student uncertainty head,
classification entropy, or final rejection score.

Both seeds used the same `max_softmax` evaluation and known-validation
calibration. Calibration acceptance was `0.949176` for seed 2163 and
`0.949106` for seed 2164. The effective score protocol reported no test
labels during training or calibration, and the rejector was inactive.

## Raw paired results

Values are reported per seed. Deltas below compare each arm with the standard
arm from the same seed; positive values are improvements except for FPR95 and
Auto-K absolute error, where lower is better.

| Seed | Arm | AUROC | AUPR | FPR95 | Known acc. | Unknown rejection | Candidate purity | Cluster ACC | NMI | ARI | Auto-K error |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2163 | standard | 0.674532 | 0.541303 | 0.765667 | 0.675667 | 0.124750 | 0.607795 | 0.166870 | 0.407765 | 0.051190 | 11 |
| 2163 | selective | 0.671938 | 0.534975 | 0.761333 | 0.682833 | 0.112000 | 0.591810 | 0.204756 | 0.473798 | 0.057908 | 5 |
| 2163 | uncertainty | 0.677807 | 0.542671 | 0.764000 | 0.684333 | 0.111000 | 0.597577 | 0.150740 | 0.409672 | 0.041039 | 9 |
| 2164 | standard | 0.679621 | 0.556987 | 0.783333 | 0.662000 | 0.143250 | 0.637375 | 0.171301 | 0.429409 | 0.043617 | 9 |
| 2164 | selective | 0.700087 | 0.566160 | 0.742667 | 0.698667 | 0.141000 | 0.631579 | 0.181411 | 0.439523 | 0.050516 | 3 |
| 2164 | uncertainty | 0.690911 | 0.561653 | 0.769333 | 0.679167 | 0.153500 | 0.638254 | 0.193347 | 0.436786 | 0.063296 | 3 |

Selective-KD deltas relative to standard were:

| Metric | Seed 2163 | Seed 2164 | Mean ± sample SD |
|---|---:|---:|---:|
| AUROC | -0.002593 | +0.020465 | +0.008936 ± 0.016305 |
| AUPR | -0.006328 | +0.009173 | +0.001423 ± 0.010961 |
| FPR95 | -0.433 pp | -4.067 pp | -2.250 ± 2.569 pp |
| Known accuracy | +0.717 pp | +3.667 pp | +2.192 ± 2.086 pp |
| Unknown rejection | -1.275 pp | -0.225 pp | -0.750 ± 0.742 pp |
| Candidate purity | -1.599 pp | -0.580 pp | -1.089 ± 0.721 pp |
| Cluster ACC | +3.789 pp | +1.011 pp | +2.400 ± 1.964 pp |
| Cluster NMI | +6.603 pp | +1.011 pp | +3.807 ± 3.954 pp |
| Cluster ARI | +0.672 pp | +0.690 pp | +0.681 ± 0.013 pp |
| Auto-K absolute error | -6 | -6 | -6 ± 0 |

Uncertainty-KD relative to standard had AUROC `+0.003275` and `+0.011290`
for the two seeds, but unknown rejection changed by `-1.375` and `+1.025`
percentage points, and candidate purity changed by `-1.022` and `+0.088`
percentage points. Its clustering changes were mixed across seeds. This does
not establish an extra stable benefit over standard-KD.

## Decision

The confirmation establishes that teacher-correct selective KD is a real,
separate student-training mechanism and can improve the clustering geometry
and Auto-K estimate on these two splits. It does not establish the required
end-to-end improvement: both seeds lose unknown rejection and candidate
purity, and the detection ranking metrics are not consistent across seeds.
The candidate therefore fails the fixed promotion rule for changing the
default. It is frozen as an opt-in diagnostic variant; `kd_mode=standard`
remains the default, uncertainty-KD remains an explicitly audited comparison,
and no further gate or coefficient scan is justified on this evidence.

The next work should return to the four required questions using existing
formal A/B/C evidence and avoid expanding into rejector training, candidate
selection, score fusion, or clustering algorithm changes.
