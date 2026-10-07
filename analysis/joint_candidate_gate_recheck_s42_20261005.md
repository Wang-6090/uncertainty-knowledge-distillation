# Joint Candidate Gating Recheck (2026-10-05)

## Purpose

The current joint objective can accidentally train novel prototypes on known-like
samples. This recheck compares hard gating, soft gating, and teacher-based gating under
one fixed protocol. The question is whether a more reliable candidate gate reduces
known/unknown overlap without simply increasing false rejection.

## Fixed protocol

- CIFAR-100 random 60/40 split, seed 42.
- Same `resnet34` teacher checkpoint and `resnet18` student architecture.
- 1200/300/1000 train/validation/test and 1200 mixed discovery samples.
- 3 epochs, batch size 64, CUDA, K=40, random novel prototype initialization.
- Same `mc_samples=4`, known-only validation threshold, and KMeans evaluation.
- Baseline: joint discovery without candidate gating.

## Results

| Method | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection | Candidate purity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Joint random baseline | 0.5051 | 0.9359 | 0.0648 | 0.9704 | 0.0561 | 0.5500 |
| Student hard gate | 0.4882 | 0.9326 | 0.0477 | 0.9342 | 0.0383 | 0.2727 |
| Student soft gate, floor 0.05 | 0.5031 | 0.9490 | 0.0758 | 0.9391 | 0.0638 | 0.4032 |
| Frozen teacher hard gate | 0.5046 | 0.9276 | 0.0547 | 0.9655 | 0.0434 | 0.4474 |

## Interpretation

Student hard gating starves the joint objective and performs worse. Soft weighting keeps
more gradient signal and improves OSCR in this run, but FPR95 and known acceptance become
worse. Frozen teacher gating raises candidate purity relative to student gating, but does
not improve AUROC or unknown rejection over the baseline. The candidate gate is therefore
not a solution to the core feature-overlap problem.

The remaining issue is upstream: neither the student nor teacher risk score provides a
clean enough mixed-pool partition. The next method should learn a representation with an
explicit known-support/novel-separation objective, preferably using a frozen or EMA target
and a global memory bank, rather than adding more hard gates to the current score.
