# Semantic-hard backbone feature consistency recheck (2026-10-07)

## Question

The nnPU plus uncertainty-weighted feature margin improves unknown-candidate
purity, but seed-43 clustering showed lower unknown-only NMI/ARI than the
baseline. This recheck tests whether a two-view cosine consistency term on the
backbone feature can preserve within-novel-class structure without assigning
pseudo labels in the mixed pool.

## Code change

Added the opt-in argument `--alpha-discovery-feature-consistency`.
When nonzero, the training loss includes

`1 - cosine(features_view_1, features_view_2)`

for the two augmented views from the discovery pool. The term uses backbone
features, is independent of the unknown-vs-known margin, and does not use
novel labels. Its default is zero, so historical commands are unchanged.

## Controlled protocol

The new treatment kept the seed-43 semantic-hard CIFAR-100 60/40 split,
pretrained ResNet-34 teacher, pretrained ResNet-18 student, full known data,
batch size 64, 10 epochs, matched 5,400-image mixed pool, known prior 0.2,
and the previous nnPU plus uncertainty-margin coefficients:

- `alpha_discovery_uncertainty_pu=0.1`
- `alpha_discovery_uncertainty_feature_margin=0.05`
- `alpha_discovery_feature_consistency=0.1`

Detection was unchanged: normalized entropy plus feature kNN, MC=8, and a
threshold calibrated on known validation data for 95% known coverage. The
clustering comparison used oracle `K=40`, normalized `projection_pca` with 32
dimensions, and KMeans. Test labels were used only for final descriptive
metrics.

## Results

### Detection

| Arm | AUROC | FPR95 | OSCR | Known accuracy | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seed-43 nnPU + uncertainty margin | 0.6260 | 0.8113 | 0.3999 | 51.88% | 95.13% | 8.60% |
| + backbone feature consistency | 0.6241 | 0.8183 | 0.3928 | 51.07% | 95.42% | 8.13% |

The consistency term does not improve unknown detection and causes a small
regression at this weight.

### Clustering

| Arm | Candidate purity | Candidate unknown-only NMI | Candidate unknown-only ARI |
| --- | ---: | ---: | ---: |
| Seed-43 nnPU + uncertainty margin | 52.86% | 0.4403 | 0.0346 |
| + backbone feature consistency | **54.17%** | **0.4854** | **0.0565** |
| Seed-43 baseline | 46.46% | 0.5143 | 0.0681 |

The new term improves candidate purity and improves the nnPU treatment's own
NMI/ARI, but it still does not exceed the baseline on NMI/ARI. The candidate
pool remains heavily contaminated: 325 of 600 candidates are truly unknown.

## Decision

This is a partial, task-dependent improvement, not a solution to the core
problem. Keep the option in code for ablation, but do not make it part of the
default method or claim that it improves the complete open-world objective.
The result suggests a real trade-off: stronger known/unknown separation can
improve candidate purity while damaging novel-class geometry, and consistency
can recover part of that geometry at a small detection cost.

The next method should avoid adding another global loss blindly. Prefer a
decoupled representation design: use one branch for known/unknown rejection
and a separate projection or prototype branch for generalized category
discovery, with candidate selection shared only through a detached gate. This
directly addresses the observed conflict between detection and novel-class
clustering.

Artifacts:

- `runs/semantic_hard_current_nnpu_margin_consistency_s43/`
- `runs/semantic_hard_current_nnpu_margin_consistency_s43_detect/`
- `runs/semantic_hard_current_nnpu_margin_consistency_s43_cluster/`
