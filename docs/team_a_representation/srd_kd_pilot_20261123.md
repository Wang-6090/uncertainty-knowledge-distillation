# SRD cross-network KD pilot (seed 2170)

## Protocol and validity

This single-seed exploratory pilot compares `standard`, `srd`, and
`uncertainty` on one semantic-hard CIFAR-100 60/40 split. The arms share a
split-matched ResNet-34 teacher, ResNet-18 student initialization, four-epoch
budget, `alpha_kd=1`, temperature, CE-only objective, known-validation
checkpoint selection, max-softmax, known-validation 95% coverage calibration,
matched discovery pool, Auto-K, and clustering. Feature KD and all auxiliary,
discovery, PU, and rejection-training losses are disabled. Saved configs
differ only in `kd_mode`. The runner reports that pool/test labels were not
used for training or selection. The machine-readable result is the local,
gitignored artifact `analysis/team_a_srd_20261123_s2170.json`.

The run completed all training, discovery, and clustering gates. Calibration
acceptance was `0.949176` in all three arms, and the rejector was inactive.
The SRD loss unit tests and runtime gates passed. Its teacher logits and
teacher classifier parameters were detached; `alpha_kd * loss_kd` entered the
student objective, and audited KD gradients were measured on student encoder
parameters. SRD loss by epoch was `2.059221/0.937482/0.760278/0.674810`, with
nonzero encoder KD gradient norms `14.7992/10.6942/9.2230/8.2236`. The student
classifier update norm was positive in every epoch. The uncertainty arm
applied nonuniform weights in all epochs (weight SD `0.22545-0.22596`, ESS
fraction `0.95126-0.95149`).

## Raw results

| Metric | Standard | SRD | SRD minus standard | Uncertainty | Uncertainty minus standard |
|---|---:|---:|---:|---:|---:|
| AUROC | 0.679782 | 0.682111 | +0.002328 | 0.681804 | +0.002022 |
| AUPR | 0.545150 | 0.547238 | +0.002088 | 0.545856 | +0.000706 |
| FPR95 | 0.785333 | 0.750667 | -0.034667 | 0.775833 | -0.009500 |
| OSCR | 0.542916 | 0.552520 | +0.009604 | 0.545188 | +0.002272 |
| Known accuracy | 0.691833 | 0.700000 | +0.008167 | 0.691667 | -0.000167 |
| Unknown rejection | 0.112250 | 0.116000 | +0.003750 | 0.084500 | -0.027750 |
| Candidate purity | 0.590013 | 0.609724 | +0.019711 | 0.602496 | +0.012482 |
| Cluster ACC | 0.190539 | 0.193167 | +0.002628 | 0.203209 | +0.012670 |
| Cluster NMI | 0.463166 | 0.447856 | -0.015310 | 0.510343 | +0.047177 |
| Cluster ARI | 0.053730 | 0.061964 | +0.008233 | 0.054282 | +0.000552 |
| Auto-K absolute error | 4 | 2 | -2 | 4 | 0 |

All entries are one-seed observations, not estimates of a population effect.
SRD improved several detection and candidate metrics modestly, but its
unknown-rejection gain was only `0.375` percentage points and Cluster NMI fell
`1.531` points. Uncertainty-KD improved ranking metrics slightly and several
clustering metrics, but unknown rejection fell `2.775` points. Neither arm
shows a joint, stable end-to-end result on one seed.

## Representation reading

An additional read-only audit used the same 3,000 known-validation samples and
the exact CIFAR transform. Raw feature norm mean increased
`30.0581 -> 40.2028` (`+33.7%`); q05/median/q95 all increased. After L2
normalization, within-class cosine distance changed `0.214449 -> 0.174222`,
nearest-center cosine distance `0.176697 -> 0.136102`, and the between/total
scatter ratio `0.469358 -> 0.492839`. Center margin changed only
`-0.037752 -> -0.038120`, slightly in the wrong direction. Known-validation
accuracy changed `0.705333 -> 0.704667`. Thus SRD changed feature magnitude and
made normalized features more compact with a higher global between/total
ratio, but did not improve the local nearest-center margin. This is a
plausible representation signal, not proof of better novel-class geometry;
the lower NMI remains a real counter-signal.

## Decision

SRD is correctly implemented and mechanistically active. The representation
audit plus mostly positive seed-2170 detection/candidate/clustering deltas
justify a fixed fresh-seed confirmation, but not a promotion or a default
change. Cluster NMI decreased, so the result remains mixed. The confirmation
uses seeds `2171/2172`, the same three arms and full fixed protocol, with no
coefficient, temperature, score, or epoch scan. Do not repeat seed 2170. The
confirmation decision rules are in
[`srd_kd_protocol_20261123.md`](srd_kd_protocol_20261123.md).
