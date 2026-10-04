# Joint memory-bank pilot (2026-10-01)

## Purpose

The current joint discovery objective uses neighbors from the current mini-batch.
This pilot tested whether a detached FIFO memory bank can provide more stable
cross-batch novel assignments. The change was implemented as an optional
`memory_neighbor_consistency_loss`; it does not change the default pipeline.

The first implementation reset the bank at every epoch and activated it as
soon as one previous batch existed. A second implementation carried the bank
across epochs, filtered mixed-pool entries by residual novel weight, and used a
256-entry warm-up before enabling the loss.

## Controlled protocol

Both pilots used CIFAR-100 random 60/40, seed 42, pretrained ResNet-34/18,
2 epochs, 1200 known training samples, 300 validation samples, 1200 mixed
discovery samples, actual mixed known prior `0.210833`, `alpha_discovery_uncertainty_pu=0.1`,
`joint_mixed_residual`, residual floor `0.05`, KMeans candidate prototype
initialization, oracle `K=40`, `feature_pca`, and explicit min-class kNN
detection. Only the memory term changed within each pair.

## Results

| Method | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection | Candidate purity | Candidate NMI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Joint residual baseline | 0.59118 | 0.89091 | 0.15992 | 95.21% | 7.09% | 49.12% | 0.86994 |
| Memory, alpha=0.1, no warm-up | 0.58799 | 0.88760 | 0.15454 | 95.04% | 6.08% | 44.44% | 0.86889 |
| Memory, alpha=0.05, 256-entry warm-up | 0.59261 | 0.88595 | 0.16142 | 95.04% | 6.84% | 47.37% | 0.84345 |

## Interpretation

The memory term was active and technically correct (`joint_memory_neighbor`
was nonzero), but it did not improve the core problem. The warm-up variant
gave only a negligible ranking change and reduced candidate purity and NMI.
The first variant also reduced unknown rejection. This suggests that stale
novel assignments from a mixed pool are not reliable enough to serve as
global targets, even when the bank is filtered by residual novelty.

The implementation remains available for ablation with default-off options:
`--alpha-joint-memory-neighbor`, `--joint-memory-size`,
`--joint-memory-k`, `--joint-memory-temperature`, and
`--joint-memory-warmup-size`. It is not part of the recommended pipeline.

## Decision

Stop tuning this memory-bank objective on the current protocol. The next
substantive direction should use a momentum/teacher representation or a
published GCD objective with class-balanced global assignments and periodic
pseudo-label refresh, rather than treating stale mixed-pool predictions as
fixed targets. The current primary pipeline remains nnPU training plus
explicit min-class kNN detection and `feature_pca` clustering.
