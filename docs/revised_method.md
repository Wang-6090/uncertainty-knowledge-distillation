# Revised Method

## Research question

The main question is whether uncertainty-aware knowledge distillation improves
open-set detection and novel-class clustering over ordinary knowledge
distillation under the same training and evaluation protocol.

## Model

- Teacher: pretrained ResNet-34.
- Student: pretrained ResNet-18 for the main ablation; MobileNetV3-Small is a
  separate compression experiment.
- Both models expose class logits, encoder features, and a projection vector.
- Novel samples are never used as labels during training.

## Losses

The student objective is staged through ablations:

```text
L = L_CE + a_kd L_KD + a_feat L_featKD
        + a_unc L_unc + a_supcon L_SupCon + a_proto L_proto
```

The standard KD term is temperature-scaled KL divergence. The uncertainty KD
term multiplies each sample's KL loss by a clipped, mean-normalized
`exp(-u_teacher)` weight so the average distillation strength stays comparable
to standard KD while unreliable teacher predictions are down-weighted. Feature
KD aligns normalized teacher and student projections with cosine distance and
remains valid for different encoder sizes.

An optional discovery stage adds an unlabeled discovery pool. Each unlabeled
image is turned into two augmented views. The student is updated with cosine
consistency and NT-Xent on those views, and optionally with periodic K-Means
pseudo labels kept only for high-confidence assignments.

## Uncertainty

For `M` stochastic MC-Dropout predictions `p_m`, the implementation reports:

```text
predictive entropy = H(mean(p_m))
aleatoric proxy    = mean(H(p_m))
epistemic         = H(mean(p_m)) - mean(H(p_m))
```

The existing auxiliary head is retained as a separate classification-
uncertainty signal; it is not conflated with the MC-Dropout aleatoric value.

## Open-set scores

The fixed, label-independent comparison includes MSP, Energy, predictive
entropy, prototype distance, diagonal Mahalanobis distance, and normalized
entropy plus Mahalanobis distance. Thresholds are calibrated only on known
validation samples at a fixed percentile. Unknown labels are not used to
choose a score or threshold.

## Discovery

Unknown samples are filtered by the calibrated score and clustered in the
normalized projection space. `--cluster-k oracle` uses the true novel-class
count and must be labelled as an upper-bound experiment. `--cluster-k auto`
estimates K from silhouette, Calinski-Harabasz and Davies-Bouldin scores over a
sample-size range that does not use the true novel-class count. `--cluster-k
both` writes the two reports separately. Optional `--cluster-confidence-percentile`
keeps only high-score unknown candidates for clustering.

The two-stage detector-plus-clustering pipeline remains the default baseline.
The discovery-pool stage is the first end-to-end NCD option and is enabled
explicitly with `train_discovery` or `--discovery-epochs`.
