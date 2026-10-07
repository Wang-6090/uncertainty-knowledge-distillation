# Feature rejector and uncertainty-augmented recheck (2026-10-07)

## Purpose

The main detector used entropy plus known-class kNN distance, while the
backbone already contained information that was not fully used by that score.
This experiment tests a frozen post-hoc rejector trained on known features and
a mixed unlabeled discovery pool. It does not change the student checkpoint,
does not use test labels for fitting or threshold selection, and keeps the
discovery representation unchanged.

The first arm uses normalized backbone embeddings only (`embedding`). The
second arm adds logits, entropy, MSP, classifier margin, learned uncertainty,
MC epistemic uncertainty, expected entropy, and aleatoric uncertainty
(`uncertainty_augmented`). The rejector uses the corrected mixed-pool nnPU
risk with known prior 0.2.

## Fixed protocol

- CIFAR-100 semantic-hard 60/40 split;
- pretrained ResNet-34 teacher and ResNet-18 student checkpoints trained with
  the existing nnPU plus uncertainty-margin treatment;
- full data, batch size 64, MC samples 8;
- mixed discovery pool with known prior 0.2;
- threshold fitted from known validation data at 95% known coverage;
- clustering uses oracle `K=40`, normalized `projection_pca`, PCA dimension 32,
  and KMeans;
- test labels are used only for final descriptive metrics.

## Detection results

| Seed | Detector | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | Existing entropy + min-class kNN | 0.5735 | 0.8627 | 0.3170 | 95.35% | 6.43% |
| 42 | nnPU feature rejector, embedding | 0.6847 | 0.7860 | 0.3871 | 94.50% | 13.60% |
| 42 | nnPU feature rejector, uncertainty augmented | **0.6910** | **0.7743** | **0.3968** | 94.85% | **15.60%** |
| 43 | Existing entropy + min-class kNN | 0.6260 | 0.8113 | 0.3999 | 95.13% | 8.60% |
| 43 | nnPU feature rejector, embedding | 0.6799 | 0.7825 | 0.3880 | 94.83% | 13.05% |
| 43 | nnPU feature rejector, uncertainty augmented | **0.6937** | **0.7670** | **0.4014** | 95.20% | 13.63% |

The rejector improves ranking and rejection on both seeds. The augmented
version is better than embedding-only on AUROC and FPR95, while the absolute
unknown rejection is still far from complete open-set recognition.

## Clustering results

| Seed | Detector | Candidate purity | Unknown-only NMI | Unknown-only ARI |
| ---: | --- | ---: | ---: | ---: |
| 42 | Existing treatment |  -- | 0.5143 | 0.0681 |
| 42 | nnPU rejector, embedding | 62.24% | 0.4653 | 0.0971 |
| 42 | nnPU rejector, uncertainty augmented | **66.88%** | 0.4702 | **0.1216** |
| 43 | Existing treatment | 52.86% | 0.4403 | 0.0346 |
| 43 | nnPU rejector, embedding | 62.74% | 0.5070 | 0.1479 |
| 43 | nnPU rejector, uncertainty augmented | **65.43%** | **0.5218** | **0.1745** |

Candidate purity improves consistently. NMI is seed-dependent, but ARI improves
on both seeds. This supports the interpretation that the rejector provides a
cleaner candidate gate without changing the feature space used by KMeans.

## Decision and limitations

This is currently the most promising evaluation route, but it is not yet the
final end-to-end method. The rejector is fitted after student training, so it
should be described as a decoupled open-set detection module or an ablation,
not as proof that the whole student was jointly optimized for unknown
rejection. It also assumes a mixed-pool known prior of 0.2 and uses an oracle
cluster count in this report.

Keep the following as the current controlled comparison:

```powershell
python train.py discover `
  --dataset cifar100 --data-root .\data `
  --split-path .\splits_cifar100_protocols\cifar100_60_40_semantic_hard.json `
  --num-known 60 --seed 43 --num-novel 40 `
  --discovery-pool-mode mixed --mixed-known-pool-ratio 0.2 `
  --mc-samples 8 --target-known-coverage 0.95 `
  --score-mode feature_rejector --rejector-training nnpu `
  --rejector-nnpu-risk nu_corrected --rejector-known-prior 0.2 `
  --rejector-feature-mode uncertainty_augmented --rejector-mc-samples 8 `
  --cluster-k oracle --cluster-feature projection_pca --cluster-pca-dim 32 `
  --cluster-normalize --student-ckpt <student.pt> `
  --work-dir <evaluation-dir> --device cuda
```

The next implementation priority is to make this decoupled rejector protocol
more principled: reserve a known calibration split, estimate or validate the
mixed-pool prior without test labels, compare against a third seed, and then
consider an independently trained rejection branch. Do not replace the
default score or claim final improvement until those checks are complete.
