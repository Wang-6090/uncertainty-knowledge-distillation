# Uncertainty target-repair KD pilot

## Question and scope

This pilot tests whether a teacher uncertainty estimate is more useful for
repairing a detached KD target than for merely down-weighting the whole KD
term. It is within Team A's student representation/KD scope: standard-KD
representation, uncertainty-KD's additional value, the uncertainty-to-KD
gradient path, and transfer to unknown detection and novel-class discovery.
It does not modify the rejector, candidate-selection algorithm, Auto-K
procedure, or clustering implementation.

For a known training label `y`, detached teacher soft target `p_t`, and
detached uncertainty `u`, the target-repair arm uses

```text
q = (1 - u) p_t + u one_hot(y)
L = T^2 KL(q || p_student)
```

The existing uncertainty arm keeps the previous detached `exp(-u)` sample
weighting. The standard arm uses ordinary logits KD. All three arms retain the
same CE/KD-only training objective and the same student initialization,
split-matched teacher, four-epoch budget, known-validation checkpoint
selection, known-validation 95% coverage calibration, `max_softmax` score,
matched discovery pool, automatic-K clustering, and evaluation protocol.
Unknown/test labels were not used for training, checkpoint selection,
calibration, or parameter selection.

## Implementation and mechanism audit

The target-repair loss is opt-in as `kd_mode=uncertainty_target_repair` and
does not change the default `kd_mode=standard`. Teacher logits, uncertainty,
and the one-hot repair target are detached. The repair coefficient is recorded
separately from the existing uncertainty weight statistics, so these signals
are not conflated.

The seed-2175 target-repair arm had repair means
`0.343359, 0.343883, 0.343901, 0.343539` in epochs 1--4. Its KD gradient
norms were `19.8126, 16.5873, 15.3496, 14.3499`, and its student classifier
update norms were `0.04019, 0.02966, 0.02898, 0.02842`. Thus the repair was
active in every epoch and the KD path produced a nonzero student gradient and
parameter update.

The preceding teacher audit found validation error-ranking AUROC `0.7506` for
the teacher uncertainty head. In contrast, the existing `exp(-u)` weighting
had effective sample size about `0.95`, so repeating weight scans was not
justified. Target repair was therefore tested as a distinct target-alignment
hypothesis.

## Corrected seed-2175 results

The first frozen discovery pass accidentally inherited the `feature_rejector`
score. That result is retained as an audit artifact, but its final metrics
must not be used. The three completed checkpoints were re-evaluated without
retraining using the intended `max_softmax` score, known-validation 95%
coverage calibration, matched pool, and Auto-K. The corrected machine-readable
result is
`analysis/team_a_uncertainty_target_repair_20261125_s2175_corrected.json`.

| Metric | standard-KD | target-repair | uncertainty-KD |
|---|---:|---:|---:|
| AUROC | 0.688044 | 0.699012 | 0.692642 |
| AUPR | 0.555815 | 0.574710 | 0.562961 |
| FPR95 | 0.752833 | 0.753833 | 0.755333 |
| Known accuracy | 0.679167 | 0.716333 | 0.685167 |
| Unknown rejection | 0.145000 | 0.181500 | 0.126500 |
| Candidate purity | 0.624327 | 0.666055 | 0.630923 |
| Cluster ACC | 0.189451 | 0.204587 | 0.189526 |
| Cluster NMI | 0.438201 | 0.457479 | 0.467942 |
| Cluster ARI | 0.059762 | 0.077286 | 0.060575 |
| Auto-K | 43 | 49 | 44 |

Relative to standard-KD, target repair changes AUROC by `+0.010968`, AUPR by
`+0.018895`, unknown rejection by `+3.650 pp`, candidate purity by
`+4.173 pp`, Cluster ACC/NMI/ARI by `+1.514/+1.928/+1.752 pp`, and known
accuracy by `+3.717 pp`. These are encouraging single-seed results. However,
the estimated number of novel clusters changes from 43 to 49 while the true
novel-class count is 40; absolute Auto-K error therefore increases from 3 to 9.
The pilot cannot establish stable end-to-end improvement from this one split.

## Decision

The fixed fresh-seed confirmation on seed `2176` retained standard-KD and
uncertainty-KD as paired controls and used the same score, calibration,
candidate-pool, Auto-K, and clustering protocol. Target-repair again improved
detection, candidate purity, and candidate-clustering metrics, but Auto-K
absolute error worsened from 3 to 7. The same direction occurred on seed
`2175` (error 3 to 9). All-unknown oracle clustering was flat/slightly lower,
while candidate-only oracle clustering improved. Thus the effect is
concentrated in the selected candidate subset rather than global unknown
coverage or class-count estimation. The end-to-end promotion gate fails;
target-repair remains an opt-in ablation, with no coefficient scan or further
same-hypothesis seeds. Full confirmation details are in
[`uncertainty_target_repair_confirmation_20261125.md`](uncertainty_target_repair_confirmation_20261125.md).

The original incorrect discovery output remains at
`analysis/team_a_uncertainty_target_repair_20261125_s2175.json` for audit
traceability only and is excluded from conclusions.
