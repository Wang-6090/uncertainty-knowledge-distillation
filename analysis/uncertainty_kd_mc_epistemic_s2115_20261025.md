# MC-epistemic logits-KD paired result (seed 2115)

This fresh split uses the pre-registered comparison: both arms share the same
four-pass MC-mean teacher logits, while only the treatment applies detached,
batch-mean-normalized MC-epistemic weights to the per-example logits KL.

| Metric | Standard KL | MC-epistemic weighted KL | Difference |
|---|---:|---:|---:|
| AUROC | 0.811164 | 0.812323 | +0.001159 |
| AUPR | 0.718851 | 0.718888 | +0.000037 |
| FPR95 (lower is better) | 60.467% | 56.517% | -3.950 pp |
| OSCR | 0.604013 | 0.610036 | +0.006023 |
| Test known acceptance | 96.167% | 95.467% | -0.700 pp |
| Calibrated unknown rejection | 23.650% | 26.575% | +2.925 pp |
| Accepted-known accuracy | 71.473% | 72.189% | +0.716 pp |
| Automatic K / absolute error | 36 / 4 | 34 / 6 | error +2 |
| Overall clustering NMI / ARI | 0.510296 / 0.173829 | 0.491834 / 0.156109 | -0.018462 / -0.017720 |

The treatment passes the local rejection/known-utility gate on this split:
unknown rejection improves by 2.925 pp, known acceptance declines by 0.7 pp,
and accepted-known accuracy improves. However, automatic K becomes worse and
both clustering metrics decline materially. Therefore this is mixed evidence,
not a complete new-class discovery improvement. The MC weighting path was
active, but redistribution was weak: per-epoch weight standard deviations were
`0.005490, 0.005609, 0.005320, 0.005408`, with ESS fractions about `0.99997`.

Decision for this seed: do not promote or tune the method on seed 2115. The
third pre-registered split, seed 2116, remains necessary for the joint decision.
