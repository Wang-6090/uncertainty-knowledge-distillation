# Late rejection-CE decay fresh-split replication: seed 2160

## Question and protocol

This experiment tests whether reducing the rejection head's known-class CE pressure
in the last two student epochs improves semantic-hard discovery. It is a fresh,
single-factor paired replication of the fixed schedule used in the earlier
semantic-hard runs:

- static control: rejection known-class CE scale `1.0` in all four epochs;
- late decay: scales `1.0`, `1.0`, `0.625`, `0.25` in epochs 1--4;
- both arms use the same seed `2160`, CIFAR-100 60/40 semantic-hard split, teacher,
  student architecture and initialization, mixed discovery pool of 5,400 samples
  with known prior `0.2`, four-epoch budget, checkpoint rule, detector, and
  known-validation-only 95% coverage calibration;
- test labels are not used for training, checkpoint selection, candidate selection,
  or calibration. The calibration subset is disjoint from checkpoint-selection
  validation samples.

Configuration comparison found no unexpected difference beyond the pre-registered
schedule fields and output/checkpoint paths. The generated split is
`splits/semantic_hard/seed2160/cifar100_60_40_semantic_hard.json`.

## Detection and rejection results

| metric | static control | late decay | treatment - control |
|---|---:|---:|---:|
| AUROC | 0.798998 | 0.811941 | +0.012943 |
| AUPR | 0.699335 | 0.726373 | +0.027038 |
| FPR95 | 0.608167 | 0.634333 | +0.026167 |
| OSCR | 0.594214 | 0.594852 | +0.000639 |
| test known acceptance | 92.400% | 92.717% | +0.317 pp |
| accepted-known accuracy | 72.276% | 71.796% | -0.481 pp |
| unknown rejection | 33.350% | 38.050% | +4.700 pp |

Both arms had the same known-validation calibration acceptance, `94.939%`, and
used the same known-validation-only threshold policy. The ranking and fixed-point
unknown rejection signals improved, but FPR95 and accepted-known accuracy did not
improve jointly. The result is therefore a tradeoff rather than an unconditional
detection success.

## Candidate and clustering results

| metric | static control | late decay | treatment - control |
|---|---:|---:|---:|
| candidate count | 1,790 | 1,959 | +169 |
| candidate purity | 74.525% | 77.693% | +3.168 pp |
| candidate unknown recall | 100.000% | 100.000% | 0 pp |
| Auto-K | 31 | 34 | +3 |
| absolute Auto-K error | 9 | 6 | -3 |
| ACC | 0.344693 | 0.407351 | +0.062658 |
| NMI | 0.529303 | 0.561216 | +0.031913 |
| ARI | 0.259345 | 0.300579 | +0.041233 |
| unknown-only ACC | 0.457271 | 0.521682 | +0.064410 |
| unknown-only NMI | 0.577611 | 0.605587 | +0.027976 |
| unknown-only ARI | 0.360580 | 0.395051 | +0.034471 |

On this split, the additional candidates were not merely false rejects: true
unknown candidates increased from `1,334` to `1,522`, while false rejects fell
from `456` to `437`. The candidate and clustering gains are encouraging, but they
are still one fresh split and must not be presented as a stable multi-seed result.

## Mechanism audit

The training history confirms that the schedule changed the actual loss path:

| epoch | CE scale | raw rejection CE | known-CE gradient norm | uncertainty-term gradient norm | KD gradient norm |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.000 | 1.894912 | 13.1313 | 0.1011 | 12.5353 |
| 2 | 1.000 | 1.199282 | 13.5239 | 0.1536 | 10.2494 |
| 3 | 0.625 | 0.853309 | 7.9617 | 0.1870 | 9.0361 |
| 4 | 0.250 | 0.583899 | 2.8416 | 0.1942 | 8.4922 |

All reported values are finite. The known-CE gradient decreases as scheduled,
while KD and uncertainty gradients remain active; this supports the intended
mechanistic interpretation that the late schedule rebalances training pressure.
It does not, by itself, establish causality for every final metric.

## Decision

This fresh split strengthens the case for late rejection-CE decay as a research
candidate because it improves detection ranking, fixed-point unknown rejection,
candidate purity, and all reported clustering metrics on the same paired split.
However, FPR95 remains worse and accepted-known accuracy falls slightly. Combined
with the earlier split-dependent tradeoffs, this is not sufficient to change the
default configuration or to search more schedules using existing test results.

Keep this result as a confirmatory single-split record. The next responsible step,
if this direction is resumed, is a pre-registered multi-split confirmation with a
validation-only joint criterion covering ranking, FPR95/tail behavior, 95%
known-coverage rejection, and known utility. Do not tune the schedule from this
test result.

Machine-readable result: `analysis/rejection_ce_decay_s2160_20261121.json`.
