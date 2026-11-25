# Uncertainty target repair KD protocol

## Question

Does using a teacher's calibrated error-risk proxy to repair its detached
soft target improve the student representation and downstream novel-class
discovery, compared with standard logits-KD and the existing uncertainty-KD?

## Why this is a new test

The frozen seed-2173 audit found that the teacher uncertainty head ranks known
classification errors reasonably well (validation error AUROC `0.7506`), but
the existing `exp(-u)` rule only reduces the mass of the full KL term and
leaves the erroneous teacher distribution unchanged. Its effective sample
size is about `0.95`, so repeating weight scans is not justified.

This experiment changes the target rather than the KD mass. For a known
training label `y`, teacher soft target `p_t` and detached uncertainty `u`,

```text
q = (1 - u) p_t + u one_hot(y)
L_repair = T^2 KL(q || p_s)
```

The hard-label CE loss remains active. The uncertainty value and target are
detached; only student parameters receive gradients. This is an opt-in
mechanism and does not alter the default `standard` KD path.

## Fixed pilot

Use fresh semantic-hard CIFAR-100 60/40 seed `2175` with three paired arms:

| Arm | Main KD objective | Purpose |
|---|---|---|
| `standard` | ordinary logits KL | clean baseline |
| `target_repair` | uncertainty-repaired target KL | test target alignment |
| `uncertainty` | existing detached `exp(-u)` weighted KL | existing comparison |

All arms share the split-matched teacher, student initialization, four-epoch
budget, CE/KD-only objective, known-validation checkpoint selection,
known-validation 95% coverage calibration, max-softmax score, matched
discovery configuration, automatic-K clustering, and data protocol. Feature
KD, rejection-branch KD, discovery training and parameter scans are disabled.
Unknown/test labels are used only for final retrospective metrics.

## Gates and stopping rule

Before training, require the explicit formula/gradient tests, compile check,
fresh output paths and a dry-run showing only the declared KD treatment differs.
During training, record repair strength, KD loss/gradient and student update for
each epoch. The target-repair arm must show a nonzero repair strength and
nonzero KD gradient/update. The pilot may motivate fresh confirmation only if
it improves at least one ranking or fixed-coverage detection metric and one
candidate/clustering metric versus standard without a material known-accuracy
or candidate-purity loss. Otherwise stop this hypothesis without scanning
coefficients or repeating the seed.

## Scope boundary

This remains a Team A student representation/KD experiment. It does not change
the rejector, unknown score definition, candidate selection, Auto-K procedure,
or clustering method. Teacher uncertainty, student uncertainty, classification
entropy and final rejection scores remain separate signals.
