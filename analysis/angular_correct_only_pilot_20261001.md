# Angular correct-only pilot (2026-10-01)

## Purpose

This pilot tested a conservative combination of the two most promising recent
directions: mixed-pool nnPU uncertainty training and the direct Angular
margin loss. The Angular term was restricted to samples that the current
student classified correctly. The motivation was to avoid imposing an angular
constraint on a sample whose current class direction is still wrong.

## Controlled protocol

- CIFAR-100 random 60/40 split, seed 42
- pretrained ResNet-34 teacher and ResNet-18 student
- 2 student epochs, 1200 known train samples, 300 validation samples, 1200
  mixed discovery samples, 1000 open-test samples
- mixed known prior measured by the program: `0.210833`
- nnPU weight: `0.1`
- Angular margin: `0.2`, scale `16`, weight `0.05`
- detection: normalized entropy plus min-class kNN, `k=10`
- clustering: oracle `K=40`, `feature_pca`

Only the Angular sample mask differed from the direct Angular run.

## Results

| Method | AUROC | FPR95 | Known acceptance | Unknown rejection | Candidate purity | Candidate NMI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct Angular + nnPU | 0.6084 | 0.8694 | 96.36% | 8.61% | 0.6071 | 0.7971 |
| Correct-only Angular + nnPU | 0.5872 | 0.8959 | 95.87% | 5.06% | 0.4444 | 0.8927 |

The correct-only mask substantially reduced the candidate pool quality and
unknown rejection. Its high candidate NMI is not sufficient evidence of an
improvement because the candidate set contained only 20 true unknown samples
and 25 incorrectly rejected known samples. This is exactly the failure mode
that the project must avoid: a clustering score can improve while detection
selectivity gets worse.

## Decision

`--angular-correct-only` is retained as an explicit ablation option, but stays
disabled by default. The direct Angular + nnPU combination remains a candidate
for a properly matched multi-seed confirmation. A seed-123 confirmation cannot
be claimed yet because the local seed-123 teacher available for reuse was a
different 3-epoch/1200-sample checkpoint, while the seed-42 comparison used a
5-epoch full-data teacher. Mixing those teachers would invalidate the causal
comparison.

The main pipeline remains nnPU training, min-class kNN detection, and
feature-space PCA clustering. Further work should use matched teacher
checkpoints and full protocol comparisons rather than adding another Angular
gate.
