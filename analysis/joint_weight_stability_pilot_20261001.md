# Joint novel-weight stability pilots (2026-10-01)

## Question

The mixed-pool joint discovery objective uses a sample weight to decide how
strongly each unlabeled sample contributes to novel prototype learning. The
original novel-mass weight is self-referential: the still-unstable novel head
helps determine the weight used to train that same head. This pilot tested
three ways to reduce that failure mode.

## Controlled protocol

- CIFAR-100 semantic-hard 60/40, seed 43;
- pretrained ResNet-34 teacher and ResNet-18 student;
- 1,200 known training samples, 300 validation samples, 1,200 mixed discovery
  samples, 1,000 test samples;
- 2 student epochs, batch size 32, CUDA;
- uncertainty KD, supervised contrastive loss, and unified project settings
  otherwise fixed;
- `unified_novel_mass` used for all three detection reports;
- disjoint validation calibration ratio 0.5 and 95% known coverage;
- clustering skipped; this is an unknown-detection/training-signal pilot.

## Methods

1. **Novel mass, floor=0**: the existing baseline. Novel probability mass from
   known+novel logits is multiplied by two-view agreement.
2. **Novel mass, floor=0.05**: applies `w = 0.05 + 0.95 * raw_weight` so every
   mixed sample keeps a small novel-learning path.
3. **EMA novel mass**: computes the weight from a Mean-Teacher-style EMA
   student instead of the rapidly changing student.
4. **Known residual**: uses `1 - max softmax(known logits)` and excludes known
   logits from the novel Sinkhorn space. This removes the novel-head
   self-weighting loop.

## Training-signal diagnostics

| Method | Epoch-2 mean weight | Epoch-2 minimum weight | Best validation known accuracy |
| --- | ---: | ---: | ---: |
| Novel mass, floor=0 | 0.0489 | 0.0032 | 0.2400 |
| Novel mass, floor=0.05 | 0.0954 | 0.0532 | 0.2367 |
| EMA novel mass | 0.0616 | 0.0574 | 0.2333 |
| Known residual | 0.6351 | not applicable | 0.2233 |

The floor and EMA variants did change the intended weight path. The residual
variant gives much broader novel supervision, but also includes more uncertain
known samples from the mixed pool.

## Detection results

| Method | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection | Accepted-known accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Novel mass, floor=0 | 0.5503 | 0.8976 | 0.1733 | 0.9480 | 0.0727 | 0.2521 |
| Novel mass, floor=0.05 | 0.5258 | 0.8927 | 0.1729 | 0.9642 | 0.0156 | 0.2513 |
| EMA novel mass | 0.5439 | 0.9138 | 0.1658 | 0.9528 | 0.0857 | 0.2423 |
| Known residual | 0.5700 | 0.9122 | 0.1642 | 0.9659 | 0.0416 | 0.2306 |

## Decision

- The weight floor is rejected: it increases contamination from mixed known
  samples and sharply reduces unknown rejection.
- EMA weighting is retained only as an optional ablation. Its higher unknown
  rejection is accompanied by worse AUROC, FPR95, OSCR, and known accuracy.
- Known residual has the best AUROC in this pilot, but its operating-point
  behavior is worse and known accuracy drops. It is not a main method.
- These results do not support further tuning of joint sample-weight formulas
  on this small protocol. The remaining bottleneck is not merely a missing
  weight floor or an unstable EMA; it is the lack of a reliable training signal
  that separates known support from unknown support.
- The next meaningful direction is an explicit known-vs-unknown representation
  or rejector objective with a clearly defined unlabeled/mixed-pool protocol,
  evaluated separately from novel clustering.

