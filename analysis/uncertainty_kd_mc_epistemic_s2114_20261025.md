# MC-epistemic logits-KD paired result (seed 2114)

This is the first fresh split in the pre-registered 2114–2116 mechanism
replication. Both arms used the same split-matched teacher and the same
four-pass MC-mean teacher logits. The treatment only applied detached,
batch-mean-normalized `exp(-MC_epistemic)` weights to per-example logits KL.
Feature KD and rejection-branch KD were disabled in both arms.

| Metric | Standard KL | MC-epistemic weighted KL | Difference |
|---|---:|---:|---:|
| AUROC | 0.792826 | 0.793465 | +0.000639 |
| AUPR | 0.682692 | 0.686203 | +0.003511 |
| FPR95 (lower is better) | 59.867% | 61.583% | +1.717 pp |
| OSCR | 0.601508 | 0.604755 | +0.003247 |
| Test known acceptance | 93.650% | 93.733% | +0.083 pp |
| Calibrated unknown rejection | 27.675% | 28.325% | +0.650 pp |
| Accepted-known accuracy | 73.074% | 73.364% | +0.291 pp |
| All-known accuracy | 71.217% | 71.050% | -0.167 pp |
| Candidate purity | 74.395% | 75.083% | +0.688 pp |
| Automatic K / absolute error | 33 / 7 | 32 / 8 | error +1 |
| Overall clustering NMI / ARI | 0.474691 / 0.154064 | 0.478223 / 0.163340 | +0.003532 / +0.009277 |
| Unknown-only NMI / ARI | 0.475543 / 0.200222 | 0.484928 / 0.216511 | +0.009385 / +0.016289 |

The treatment did not reach the pre-registered primary gate of at least +2 pp
unknown rejection. It also worsened FPR95 and automatic-K error, although
ranking, fixed-coverage rejection, accepted-known accuracy, candidate purity,
and clustering improved slightly. The mechanism was active: KD loss was
nonzero in all four epochs and weight standard deviations were
`0.010261, 0.010412, 0.010197, 0.009763`, with ESS fractions about `0.99985`.
Thus the MC weights were technically non-degenerate but produced only weak
sample redistribution in this split.

Decision for this seed: negative/mixed evidence; do not tune on seed 2114 or
change defaults. The preregistered fresh-split replication continues on seeds
2115 and 2116. Full machine output is
[`uncertainty_kd_mc_epistemic_s2114_2116_s2114.json`](uncertainty_kd_mc_epistemic_s2114_2116_s2114.json).
