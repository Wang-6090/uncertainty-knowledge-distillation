# Joint Assignment and Soft Pseudo-Label Recheck (2026-10-05)

## Purpose

The joint novel head showed nearly uniform batch assignments and inactive
prototypes. This recheck tested two explanations without changing the
evaluation protocol: (1) batch-only Sinkhorn lacks cross-batch context, and
(2) hard `argmax` pseudo-labels amplify early random differences.

## Fixed protocol

- CIFAR-100 random 60/40 split, seed 42.
- Matching ResNet-34 teacher checkpoint: `runs/hard_proxy_teacher_s42/teacher.pt`.
- ResNet-18 student; 1200 train / 300 validation / 1000 test / 1200 discovery.
- Three epochs, batch size 64, CUDA, mixed discovery pool, `K=40`.
- Same threshold, detector, KMeans, PCA and oracle-independent clustering
  evaluation for every treatment.
- Baseline: `runs/refresh_joint_random_s42`.

## Results

| Method | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection | Candidate purity | Active prototypes (last epoch) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Joint random baseline | 0.5051 | 0.9359 | 0.0648 | 97.04% | 5.61% | 55.00% | about 5 |
| Global assignment, memory weight 1.0 | 0.5001 | 0.9293 | 0.0658 | 93.91% | 7.14% | 43.08% | about 5 |
| Global assignment, memory weight 0.1 | 0.5120 | 0.9359 | 0.0681 | 96.22% | 5.61% | 48.89% | about 17 |
| Hard pseudo, balance weight 0.1 | 0.4971 | 0.9293 | 0.0683 | 88.65% | 10.46% | 37.27% | about 9 |
| Hard pseudo, balance weight 1.0 | 0.4958 | 0.9095 | 0.0659 | 93.09% | 8.16% | 43.24% | about 8 |
| Soft pseudo, balance weight 1.0 | 0.5014 | 0.9457 | 0.0632 | 91.94% | 6.12% | 32.88% | about 25 |

## Interpretation

Global assignment is active, but a memory weight of 1.0 lets the large FIFO
queue dominate the current batch and produces prototype concentration. Reducing
the memory weight avoids that collapse, but gives only a weak AUROC/OSCR signal
and does not improve unknown rejection or candidate purity. The mechanism is
therefore retained as an opt-in diagnostic, not as the main method.

Hard pseudo-labeling makes the assignment more confident, but this confidence
comes from prototype collapse and false rejection. Increasing the balance loss
does not restore all 40 prototypes. Soft pseudo-labeling keeps many more active
prototypes and confirms that `argmax` is a source of collapse, but its detector
and candidate-purity results are worse. Better occupancy alone is not enough:
the novel head is not aligned with true unknown semantics.

## Decision

Do not enable global assignment, hard pseudo-labeling or soft pseudo-labeling
by default. Do not continue tuning their scalar weights on this low-budget
mixed-pool protocol. The more reliable main candidate remains full-data
mixed-pool nnPU plus min-class kNN detection; its three-seed results are
recorded earlier in `README.md`.

The next joint-discovery redesign should use an EMA/frozen target, explicit
prototype occupancy and cross-epoch assignment stability, and a training
objective that separates the known support from the novel partition. A memory
bank without a reliable target only propagates contaminated predictions.

