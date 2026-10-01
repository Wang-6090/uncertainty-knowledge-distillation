# Angular strict recheck (seed 123, 2026-10-01)

## Purpose

This was a strict single-factor replication of the direct Angular-margin
student loss. The goal was to determine whether the earlier seed-42 pilot was
reproducible with an independently trained, protocol-matched teacher.

Both students used the same CIFAR-100 random 60/40 split, pretrained
ResNet-34 teacher, pretrained ResNet-18 student, 1200 labeled training
examples, 300 known validation examples, 1200 mixed discovery-pool examples,
two student epochs, uncertainty KD, and mixed-pool nnPU uncertainty training.
Detection and clustering were identical: MC=4,
`normalized_entropy_min_class_knn`, class-wise kNN support, oracle K=40,
KMeans, and `feature_pca`.

The only treatment variable was `alpha_angular`:

- control: `alpha_angular=0`;
- treatment: `alpha_angular=0.05`, margin `0.2`, scale `16`.

## Results

| Metric | nnPU control | nnPU + Angular | Change |
| --- | ---: | ---: | ---: |
| AUROC | 0.5672 | 0.5470 | -0.0202 |
| AUPR | 0.4607 | 0.4379 | -0.0228 |
| FPR95 | 0.8945 | 0.8928 | -0.0017 |
| OSCR | 0.1523 | 0.1420 | -0.0103 |
| Known acceptance | 94.64% | 92.46% | -2.18 pp |
| Unknown rejection | 7.94% | 8.93% | +0.99 pp |
| Accepted-known accuracy | 22.48% | 22.28% | -0.20 pp |
| Candidate purity | 0.5000 | 0.4444 | -0.0556 |
| Candidate NMI | 0.8494 | 0.8184 | -0.0310 |

## Decision

Angular does not provide a reliable improvement. The small increase in
unknown rejection is accompanied by lower known acceptance, lower AUROC,
lower candidate purity, and lower clustering NMI. It is therefore retained as
an ablation option only and is not enabled in the main pipeline.

The next investigation should address training sufficiency and representation
quality: the current rapid comparison trains the student on only 1200 samples
for two epochs, while the core failure is feature overlap. A fixed nnPU
baseline should first be trained longer and/or with a larger labeled subset;
the detector and all evaluation settings must remain unchanged. Only if the
feature space improves should another representation loss be evaluated.
