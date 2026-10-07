# Known center-margin pilot (seed 42, 2026-10-07)

## Purpose

This pilot tests a class-boundary representation objective using known labels
only. For each known sample, `supervised_center_margin_loss` compares the
sample's similarity to its own batch class center with the nearest wrong class
center. It does not assign pseudo-unknown labels to the mixed discovery pool.

The pilot compares three arms on the same semantic-isolated CIFAR-100 60/40
protocol: baseline, the existing uncertainty-margin treatment, and a
center-margin-only arm. All use the same teacher, three epochs, mixed pool,
stratified 5000-sample support, support-only `nu_corrected` nnPU rejector, and
validation-only 95% known-coverage calibration. Clustering was skipped.

## Results

| Arm | AUROC | FPR95 | OSCR | Known accuracy | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.6691 | 0.8286 | 0.1571 | 20.13% | 97.17% | 7.27% |
| uncertainty treatment | **0.7206** | **0.6889** | **0.2153** | **26.62%** | 92.01% | **19.05%** |
| center-margin-only | 0.6983 | 0.8170 | 0.1673 | 20.63% | 95.34% | 12.03% |

Center-margin-only versus baseline changes are AUROC `+0.0292`, FPR95
`-0.0116`, OSCR `+0.0102`, known acceptance `-1.83pp`, and unknown rejection
`+4.76pp`. It therefore has a small positive signal, but it is weaker than
the uncertainty treatment and does not materially improve known classification
in this short budget.

## Decision

Keep center-margin as an opt-in boundary ablation. One ten-epoch confirmation
is justified because the five-epoch/short pilot may understate supervised
geometry losses. Do not tune its weight based on this pilot. If the ten-epoch
center-margin arm remains below the uncertainty treatment, stop this direction
and focus on a decoupled rejector or a stronger class-conditional boundary
representation objective.
