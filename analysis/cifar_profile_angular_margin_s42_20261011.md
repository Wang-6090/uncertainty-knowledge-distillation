# CIFAR profile + nnPU + angular margin (seed 42, 2026-10-11)

## Purpose

This experiment tests an ArcFace-style additive angular margin on labelled
known samples. The motivation is to tighten known-class angular boundaries so
unknown samples are less likely to receive a confident known prediction.

## Protocol

The strict seed-42 CIFAR-profile + audited mixed-pool nnPU recipe is the
baseline: semantic-isolated CIFAR-100 60/40, pretrained ResNet-34/ResNet-18,
five epochs, matched mixed pool of 5,400 with known prior 0.2, feature KD,
prototype loss, `alpha_discovery_uncertainty_pu=0.1`,
`normalized_entropy_min_class_knn`, k=10, MC=4, and validation-only 95% known
coverage. The only training change is `alpha_angular=0.05`, angular margin
0.2, scale 16. Test labels are used only for final reporting.

## Results

| Metric | nnPU baseline | + angular margin | Change |
|---|---:|---:|---:|
| AUROC | 0.7906 | 0.7806 | -0.0099 |
| AUPR | 0.6757 | 0.6647 | -0.0110 |
| FPR95 | 0.6285 | 0.6518 | +0.0233 |
| OSCR | 0.5536 | 0.5424 | -0.0112 |
| Known accuracy | 63.08% | 61.95% | -1.13 pp |
| Known acceptance | 94.80% | 93.92% | -0.88 pp |
| Unknown rejection | 23.00% | 25.13% | +2.13 pp |
| Score KS | 0.4554 | 0.4347 | -0.0207 |
| Mean score gap | 2.1761 | 2.1181 | -0.0580 |

## Decision

The angular margin increases operating-point rejection, but the score
distributions separate less according to AUROC, FPR95, KS, and mean gap, while
known acceptance and accuracy decline. This is not a valid solution to the
core overlap problem; it is retained only as an ablation.

Artifact: `runs/semantic_isolated_full_s42_cifar_profile_angular_student/`

