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
term multiplies each sample's KL loss by `exp(-u_teacher)`, so reliable teacher
predictions are transferred more strongly. The implementation supports the
original `raw` weighting and a `mean_normalized` variant, which divides the
batch weights by their mean to keep the average KD strength approximately
fixed. Feature KD aligns normalized teacher and student projections with
cosine distance and remains valid for different encoder sizes.

An optional discovery-pool objective is now available for the NCD/GCD stage:

```text
L = ... + a_dis L_NT-Xent + a_du L_unknown
```

`L_NT-Xent` applies SimCLR-style two-view contrastive learning to unlabeled
discovery images, so different augmentations of the same image remain close
while other images in the batch act as negatives. `L_unknown` can be enabled
only for an unknown-only discovery pool; it reduces maximum known-class
confidence, increases prediction entropy, and raises the uncertainty head on
unlabeled novel images. This term must not be used for a mixed known/unknown
pool.

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
normalized projection space. `--cluster-k oracle` is the known-class-count
baseline and must be labelled as such. `--cluster-k auto` estimates the number
of clusters using silhouette score without unknown labels; it is reported as a
separate experiment.

This version is still a two-stage discovery pipeline. A later NCD/GCD phase
should add pseudo-label updates or clustering-assignment consistency on top of
the current discovery-pool contrastive baseline.
