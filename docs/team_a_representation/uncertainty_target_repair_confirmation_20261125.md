# Uncertainty target-repair KD fresh-seed confirmation

## Protocol

This is the one fixed fresh-seed confirmation motivated by the seed-2175
screening pilot. It uses semantic-hard CIFAR-100 60/40 seed `2176` and paired
`standard`, `target_repair`, and `uncertainty` arms. The split-matched teacher,
student initialization, CE/KD-only objective, four-epoch budget, known-only
checkpoint selection, known-validation 95% coverage calibration,
`max_softmax`, matched 5,400-item discovery pool, Auto-K and clustering are
held fixed. Only `kd_mode` differs. No parameter scan or retraining of seed
2175 was performed. Unknown/test labels were not used for training, selection,
calibration, or parameter choice.

The runner dry-run confirmed all three arms, four epochs, and gradient audit;
the new split and result directories were absent before launch. Full tests
passed (`402 passed`, one pre-existing warning), and the full run completed
without errors. Raw machine output is
`analysis/team_a_uncertainty_target_repair_confirm_20261125_s2176.json`.

## Mechanism gate

Target-repair was active in all four epochs. Mean repair coefficients were
`0.345782, 0.345296, 0.344003, 0.344693`; KD gradient norms were
`19.5956, 17.0138, 15.6579, 14.6644`; student classifier update norms were
`0.03975, 0.03026, 0.02949, 0.02869`. The standard and uncertainty arms
reported no target repair. Thus uncertainty enters the repaired target, the
KD loss, and effective student updates as intended.

## Results

| Metric | standard-KD | target-repair | uncertainty-KD |
|---|---:|---:|---:|
| AUROC | 0.677624 | 0.698409 | 0.673489 |
| AUPR | 0.533323 | 0.564702 | 0.529899 |
| FPR95 | 0.754833 | 0.737333 | 0.751500 |
| Known accuracy | 0.686500 | 0.707000 | 0.676500 |
| Unknown rejection | 0.101000 | 0.125750 | 0.099500 |
| Candidate count | 684 | 770 | 714 |
| Candidate purity | 0.590643 | 0.653247 | 0.557423 |
| Candidate-only oracle ACC/NMI/ARI | 0.254950 / 0.436359 / 0.080199 | 0.280318 / 0.482262 / 0.118930 | 0.248744 / 0.443189 / 0.076418 |
| All-unknown oracle ACC/NMI/ARI | 0.435250 / 0.503920 / 0.269534 | 0.421750 / 0.504103 / 0.268665 | 0.403000 / 0.490533 / 0.253747 |
| Auto-K (absolute error; true K=40) | 43 (3) | 47 (7) | 44 (4) |
| Auto-K candidate clustering ACC/NMI/ARI | 0.198830 / 0.461381 / 0.052822 | 0.235065 / 0.505926 / 0.104448 | 0.161064 / 0.438003 / 0.029702 |

Relative to standard-KD, target-repair improves AUROC by `0.020785`, AUPR by
`0.031379`, FPR95 by `-1.750 pp`, known accuracy by `2.050 pp`, unknown
rejection by `2.475 pp`, candidate purity by `6.260 pp`, and Auto-K candidate
clustering ACC/NMI/ARI by `3.623/4.455/5.163 pp`. Candidate-only oracle
clustering also improves, while all-unknown oracle clustering is essentially
flat/slightly lower. Auto-K moves from 43 to 47, increasing absolute K error
from 3 to 7.

The seed-2175 corrected pilot showed the same pattern: target-repair improved
the detection, candidate, and Auto-K candidate-clustering measures, but Auto-K
increased from 43 to 49 and absolute K error from 3 to 9. Across these two
semantic-hard seeds, gains therefore appear concentrated in selecting a
cleaner and more clusterable candidate subset, not in complete unknown-class
coverage or estimating the number of unknown classes. Two overlapping
semantic-hard splits are still a small, descriptive sample and do not support
statistical generalization.

## Decision

The mechanism gate passes and the positive candidate/discovery signal
replicates. The end-to-end promotion gate fails because Auto-K error worsens
on both seeds and all-unknown oracle clustering does not improve. Keep
target-repair as an opt-in ablation; do not change the default, scan repair
strength, or launch more seeds for this unchanged hypothesis. The useful
research finding is narrower: uncertainty-guided target repair consistently
produces a higher-purity, more clusterable selected candidate pool, while
increasing the number of selected candidates without recovering a more
accurate global unknown-class count. The Auto-K mechanism belongs to the
downstream discovery component and is not modified in this Team A experiment.

## Two-seed paired deltas versus standard-KD

| Metric | seed 2175 | seed 2176 |
|---|---:|---:|
| AUROC | +0.010968 | +0.020785 |
| AUPR | +0.018895 | +0.031379 |
| Unknown rejection | +3.650 pp | +2.475 pp |
| Candidate purity | +4.173 pp | +6.260 pp |
| Candidate clustering ACC | +1.514 pp | +3.623 pp |
| Candidate clustering NMI | +1.928 pp | +4.455 pp |
| Candidate clustering ARI | +1.752 pp | +5.163 pp |
| Auto-K absolute error change | +6 | +4 |
