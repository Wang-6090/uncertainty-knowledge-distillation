# Teacher-error-only target repair pilot

## Motivation

The target-repair pilot on seeds `2175` and `2176` improved ranking,
candidate purity, and candidate-only clustering, but increased Auto-K error on
both seeds. A frozen teacher audit on seed `2176` found that teacher
uncertainty ranks teacher errors (validation AUROC `0.7561`), while the
teacher's true-class confidence is stronger (`0.8112`). This suggests that
repairing every uncertain sample may over-correct teacher-correct rows.

This pilot tests a narrower target-alignment hypothesis: use the known training
label to identify teacher-wrong rows, preserve the detached teacher soft target
on teacher-correct rows, and repair only the wrong rows. The uncertainty-scaled
arm uses

```text
q = p_teacher                                  if argmax(p_teacher) == y
q = (1-u) p_teacher + u one_hot(y)              otherwise
```

The hard arm uses the same row mask with repair coefficient `1` on teacher
errors. The hard-versus-soft pair tests whether uncertainty strength adds
value after the error set is identified. This is a Team A KD/representation
experiment; it does not modify rejector scoring, candidate selection, Auto-K,
or clustering.

## Fixed screening protocol

Use fresh semantic-hard CIFAR-100 60/40 seed `2177` with four paired arms:

| Arm | KD target | Purpose |
|---|---|---|
| `standard` | original teacher soft target | clean baseline |
| `hard_error_repair` | one-hot known label only on teacher-wrong rows | isolate error-set correction |
| `error_repair` | uncertainty-scaled repair only on teacher-wrong rows | test uncertainty strength after error gating |
| `uncertainty` | existing detached `exp(-u)` weighted KD | existing comparison |

All arms share the split-matched teacher, student initialization, CE/KD-only
objective, four-epoch budget, known-validation checkpoint selection,
known-validation 95% coverage calibration, `max_softmax`, matched discovery
pool, Auto-K, clustering, and data transforms. Unknown/test labels are used
only for final retrospective metrics. No repair-strength, score, temperature,
or seed scan is allowed in this screening stage.

## Mechanism and promotion gates

The runner must show all four arms differ only by the declared `kd_mode`.
Each repair arm must record a nonzero teacher-error active fraction, repair
mean, KD gradient, and student classifier update in every epoch. Teacher
logits, uncertainty, and repaired targets must remain detached.

The hypothesis is worth a later fixed confirmation only if an error-repair arm
improves at least one ranking or fixed-coverage metric and one candidate or
clustering metric versus standard without a material known-accuracy or
candidate-purity loss, and without repeating the two-seed Auto-K degradation
pattern. Otherwise freeze it without scanning or repeating the seed.

