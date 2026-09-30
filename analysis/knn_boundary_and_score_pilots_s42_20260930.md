# kNN boundary and score-combination pilots (2026-09-30)

## Goal and comparison discipline

These pilots target the persistent overlap between known and novel feature
support. In both comparisons, test labels were used only for final descriptive
metrics, not training, score normalization, threshold calibration, or parameter
selection. Results are a single-seed, limited-data pilot and are not a claim of
general effectiveness.

## Training-loss ablation: class-wise kNN support boundary

The candidate loss uses a class-wise known-feature bank, estimates a support
radius from leave-one-out known kNN distances, and penalizes pure-unknown pool
features that remain inside any known class support radius. Both runs used the
same seed-42 CIFAR-100 random 60/40 split, teacher, student architecture,
sample limits (1,200 train / 300 validation / 1,000 test / 1,200 discovery),
three epochs, batch size 32, and `alpha_discovery=0.1`. The only intentional
training variable was `alpha_discovery_knn_boundary` (0 vs 0.1); the variant
used k=5, radius quantile 0.95, margin 0.02. Both used the same
`normalized_entropy_mahalanobis` detector and known-validation 95%-coverage
threshold policy.

The loss was active in the variant (epoch means 0.0575, 0.2030, 0.2034), so
this was not an unwired-option failure.

| Metric | Baseline | kNN-boundary loss | Change |
| --- | ---: | ---: | ---: |
| AUROC | 0.5726 | 0.5619 | -0.0107 |
| AUPR | 0.4367 | 0.4468 | +0.0102 |
| FPR95 | 0.8727 | 0.8926 | +0.0198 (worse) |
| OSCR | 0.1633 | 0.1511 | -0.0121 |
| Known test acceptance | 0.9322 | 0.9587 | +0.0264 |
| Unknown test rejection | 0.0734 (29/395) | 0.0785 (31/395) | +2 samples |
| Known-class accuracy after accept | 0.2394 | 0.2241 | -0.0152 |
| Candidate purity | 0.4143 | 0.5536 | +0.1393 |
| All-unknown clustering ARI | 0.0691 | 0.0691 | unchanged |
| Candidate clustering ARI | -0.0239 | 0.0218 | +0.0457 |

The result is mixed: the candidate pool became purer, but AUROC/FPR95/OSCR and
accepted-known classification did not improve. The boundary-loss training
variant should remain an ablation, not the preferred checkpoint. Do not tune
its margin/weight on this test split.

## Score ablation: adding a global kNN distance signal

First, with the same baseline checkpoint and split, `normalized_entropy_knn`
was compared to `normalized_entropy_mahalanobis`. The run conditions were the
same and the only score change was adding known-training-bank kNN distance;
normalization statistics came from known validation and the threshold targeted
95% known coverage.

| Metric | Entropy + Mahalanobis | Entropy + kNN | Change |
| --- | ---: | ---: | ---: |
| AUROC | 0.5726 | 0.5838 | +0.0111 |
| AUPR | 0.4367 | 0.4669 | +0.0302 |
| FPR95 | 0.8727 | 0.8992 | +0.0264 (worse) |
| OSCR | 0.1633 | 0.1720 | +0.0087 |
| Known test acceptance | 0.9322 | 0.9455 | +0.0132 |
| Unknown test rejection | 0.0734 (29/395) | 0.1114 (44/395) | +15 samples |
| Candidate purity | 0.4143 | 0.5714 | +0.1571 |
| Candidate clustering ARI | -0.0239 | 0.0140 | +0.0379 |

This is a promising ranking/purity signal but not a clean operating-point win:
FPR95 worsened and test known coverage remained below the 95% validation target.
It is one seed and should be replicated before adoption.

Then an equal-weight, known-validation-z-scored sum of entropy, Mahalanobis,
and kNN distance (`normalized_entropy_mahalanobis_knn`) was compared against
entropy + Mahalanobis on the exact same baseline checkpoint, feature bank,
split, and clustering configuration. Both built the same kNN bank; only score
composition changed.

| Metric | Entropy + Mahalanobis | Entropy + Mahalanobis + kNN | Change |
| --- | ---: | ---: | ---: |
| AUROC | 0.5726 | 0.5748 | +0.0022 |
| AUPR | 0.4367 | 0.4517 | +0.0150 |
| FPR95 | 0.8727 | 0.8744 | +0.0017 (worse) |
| OSCR | 0.1633 | 0.1655 | +0.0022 |
| Known test acceptance | 0.9322 | 0.9587 | +0.0264 |
| Unknown test rejection | 0.0734 (29/395) | 0.0759 (30/395) | +1 sample |
| Candidate purity | 0.4143 | 0.5455 | +0.1312 |
| Candidate clustering ARI | -0.0239 | 0.0455 | +0.0694 |
| Overall clustering ARI | 0.0049 | -0.0166 | -0.0216 |

The extra kNN term adds little to the AUROC/OSCR operating behavior and does
not improve FPR95; candidate purity rises, while overall clustering degrades.
Do not prefer this score over the current one based on this pilot.

## Follow-up: predicted-class conditional kNN support

To test whether class-conditional local support is more useful than distance
to the global known bank, the scorer now also measures each sample's mean
cosine distance to its `k` nearest known-training features from the classifier's
predicted class. If the predicted class has no bank entries, the global kNN
distance is used as a documented fallback. The candidate score is the sum of
known-validation-standardized entropy and this predicted-class distance.

This was compared on the same baseline checkpoint, seed, split, sample limits,
kNN bank (k=10, backbone features), threshold policy, and cluster settings.
Only the score formula changed from entropy + Mahalanobis to entropy +
predicted-class kNN distance.

| Metric | Entropy + Mahalanobis | Entropy + predicted-class kNN | Change |
| --- | ---: | ---: | ---: |
| AUROC | 0.5726 | 0.5907 | +0.0180 |
| AUPR | 0.4367 | 0.4694 | +0.0328 |
| FPR95 | 0.8727 | 0.8893 | +0.0165 (worse) |
| OSCR | 0.1633 | 0.1766 | +0.0134 |
| Known test acceptance | 0.9322 | 0.9669 | +0.0347 |
| Unknown test rejection | 0.0734 (29/395) | 0.0633 (25/395) | -4 samples |
| Known-class accuracy after accept | 0.2394 | 0.2410 | +0.0017 |
| Candidate purity | 0.4143 | 0.5556 | +0.1413 |
| All-unknown clustering ARI | 0.0691 | 0.0691 | unchanged |
| Candidate clustering ARI | -0.0239 | -0.0239 | unchanged |

This score modestly improves AUROC/AUPR/OSCR and increases candidate purity,
but at the calibrated operating point it rejects fewer unknowns, while FPR95
worsens. It is therefore a ranking/candidate-selection signal, not a validated
solution to low unknown rejection. Keep it experimental and do not claim the
core overlap is solved. The result warrants multi-seed validation only if the
goal is improved ranking/purity; it does not justify replacing the primary
detector for rejection.

### Follow-up: classwise calibration of predicted-class kNN distance

Hypothesis: classes may have different local feature-density scales, so one
global mean/std may over-reject naturally sparse classes. We changed only the
kNN distance normalization to per-predicted-class mean/std fitted on correctly
classified known validation samples (classes with fewer than two samples fall
back to global statistics). Entropy, checkpoint, bank, split, threshold
policy, and clustering were unchanged. The global-normalization control was
rerun to verify the scorer after refactoring.

| Metric | Global predicted-class kNN | Classwise predicted-class kNN | Change |
| --- | ---: | ---: | ---: |
| AUROC | 0.5907 | 0.5674 | -0.0232 |
| AUPR | 0.4694 | 0.4270 | -0.0424 |
| FPR95 | 0.8893 | 0.8711 | -0.0182 (better) |
| OSCR | 0.1766 | 0.1784 | +0.0017 |
| Known test acceptance | 0.9669 | 0.9471 | -0.0198 |
| Unknown test rejection | 0.0633 (25/395) | 0.0456 (18/395) | -7 samples |
| Candidate purity | 0.5556 | 0.3600 | -0.1956 |
| Overall clustering ARI | 0.0173 | 0.0327 | +0.0154 |

This tradeoff does not meet the goal: unknown rejection, AUROC/AUPR, and
candidate purity all decline. The validation set has only 300 examples across
60 known classes, so per-class estimates are especially noisy. Do not use this
classwise version as the preferred score; keep it only as an experimental
ablation.

## Implementation and audit notes

- Added default-off `--alpha-discovery-knn-boundary` training loss and its
  class-wise support-radius construction.
- Fixed an implicit coupling where enabling any discovery regularizer also
  activated the two-view discovery contrastive loss irrespective of its own
  weight. The ablation runs explicitly set `alpha_discovery=0.1`; older hybrid
  comparisons affected by the implicit coupling must not be interpreted as
  isolated boundary-loss comparisons.
- Added `normalized_entropy_mahalanobis_knn`, an explicit equal-weight sum of
  three z-scored signals. Its normalization uses known validation only. It is
  not a new training objective and remains an experimental option.
- Existing kNN scoring uses a global bank of known training embeddings and
  aggregate neighbor distance. It does not model class-conditional local
  density or neighborhood label conflict; this limits its ability to separate
  nearby semantic classes from genuinely out-of-support samples.
- KNN loss and score options remain disabled unless explicitly requested.
- The predicted-class support-radius option was evaluated on three paired
  seeds; its multi-seed result below supersedes the initial single-seed
  interpretation.

## Decision and next experiment

The training boundary-loss direction is not supported as an improvement in
this seed. The global kNN score has a weak positive ranking/purity signal but
its FPR95 tradeoff is unfavorable; the three-signal combination is effectively
neutral. Keep these as ablations and do not tune against this test set.

This initial score pilot motivated a predeclared multi-seed replication of the
global predicted-class kNN score; the subsequent deterministic-bank correction
and class-radius experiment are documented below.

## Deterministic-bank correction and class-support-radius pilot

### Audit finding and implementation correction

The CIFAR `bundle.train` dataset carries random resized crop, horizontal flip,
and color-jitter transforms. The discovery path previously passed it directly
to `collect_knn_feature_bank`, so repeated detector runs could construct
different known-reference features. This adds avoidable noise to kNN scores
and invalidates exact reproducibility of older kNN detector comparisons.

The discovery path now shallow-clones the known training subset and switches
the base dataset to the deterministic evaluation transform before extracting
the kNN bank. `--knn-support-quantile` controls the training-only leave-one-out
quantile for each class support radius (default 0.95). A sample's predicted-
class kNN distance can be divided by that predicted class's radius. Radii are
estimated from same-class bank neighbors with self excluded; classes with too
few bank examples fall back to the pooled radius. No validation or test labels
are used to construct these radii.

Older kNN results that used the random training transform should be considered
exploratory and should not be pooled with results from the corrected bank.
The non-kNN scores are unaffected by this transform correction.

### Single-factor paired comparison

Question: does scaling predicted-class distance by that class's known-training
support radius improve detection relative to using raw predicted-class
distance? Both evaluations used the same seed-42 baseline checkpoint, CIFAR-100
random 60/40 split, deterministic known train bank (1,200 samples), 300 known
validation samples, 1,000 open test samples (605 known / 395 unknown), k=10,
support quantile=0.95, four MC samples, validation-known 95% coverage
calibration, and identical clustering settings. Only the distance expression
changed: raw predicted-class distance versus distance divided by its
training-derived class radius. The two discovery-detail files have identical
entropy, predicted class, truth labels, and raw/relative kNN distances, so the
paired detector comparison used identical per-sample inputs.

| Metric | Raw predicted-class kNN | Distance / class support radius | Change |
| --- | ---: | ---: | ---: |
| AUROC | 0.5949 | 0.5929 | -0.0020 |
| AUPR | 0.4833 | 0.4652 | -0.0181 |
| FPR95 | 0.9025 | 0.8959 | -0.0066 (better) |
| OSCR | 0.1761 | 0.1782 | +0.0021 |
| Test known acceptance | 0.9620 | 0.9455 | -0.0165 |
| Test unknown rejection | 0.0810 (32/395) | 0.1139 (45/395) | +13 samples |
| Known-class accuracy after accept | 0.2405 | 0.2465 | +0.0060 |
| Candidate purity | 0.5818 | 0.5769 | -0.0049 |
| Overall clustering ARI | 0.0334 | -0.0154 | -0.0488 |
| Candidate clustering ARI | -0.0310 | -0.0286 | +0.0025 |

This is a tradeoff, not an unambiguous win. It rejects 13 additional novel
examples but also falsely rejects 10 additional known examples at the
validation-calibrated threshold; AUROC/AUPR do not improve and clustering is
mixed. Keep the radius-normalized score as an experimental option only. The
increase in unknown rejection should not be reported without its accompanying
known false-rejection increase.

### Decision

The deterministic reference-bank fix should remain because it improves
reproducibility and evaluation validity. The radius-normalized score showed a
promising single-seed signal, so it was repeated on seed 43 and 44 using
separately trained students and the fixed seed-42 teacher. Each seed retained
the same within-seed paired protocol.

| Seed | Unknown rejection: raw → radius | Known false rejects: raw → radius | AUROC delta | AUPR delta | FPR95 delta | OSCR delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 32 → 45 (+13) | 23 → 33 (+10) | -0.0020 | -0.0181 | -0.0066 | +0.0021 |
| 43 | 36 → 31 (-5) | 38 → 37 (-1) | -0.0103 | -0.0172 | -0.0033 | +0.0070 |
| 44 | 31 → 34 (+3) | 26 → 36 (+10) | +0.0113 | +0.0101 | +0.0050 | +0.0028 |
| Mean change | +3.7 samples | +6.3 samples | -0.0003 | -0.0084 | -0.0016 | +0.0040 |

The unknown-rejection gain is inconsistent and small on average, while known
false rejection rises by more samples on average. AUROC/AUPR do not show a
consistent gain and candidate purity declines on all three seeds. The
class-radius normalization therefore does **not** pass replication; keep the
code as an ablation but do not recommend or further tune this branch.

## Neighborhood-vote consistency pilot (seed 42)

After the distance/radius approaches failed to show consistent operating-point
gains, we tested a different signal: whether the model's predicted known class
is supported by labels of the nearest known-training neighbors. For each test
sample, support is the similarity-weighted fraction of its global top-k
neighbors whose class matches the model's predicted class. Unknownness uses
one minus that support, added to known-validation-standardized predictive
entropy. This is distinct from measuring distance to known feature support.

The paired comparison used the same seed-42 checkpoint, deterministic known
training bank, split, sample counts, k=10, MC count, threshold policy, and
clustering configuration. The only score change was raw predicted-class kNN
distance versus neighborhood-vote conflict.

| Metric | Predicted-class kNN distance | Neighborhood-vote conflict | Change |
| --- | ---: | ---: | ---: |
| AUROC | 0.5949 | 0.5946 | -0.0003 |
| AUPR | 0.4833 | 0.4629 | -0.0204 |
| FPR95 | 0.9025 | 0.8942 | -0.0083 (better) |
| OSCR | 0.1761 | 0.1846 | +0.0085 |
| Known test acceptance | 0.9620 | 0.9306 | -0.0314 |
| Unknown test rejection | 0.0810 (32/395) | 0.1063 (42/395) | +10 samples |
| Candidate purity | 0.5818 | 0.5000 | -0.0818 |
| Clustering ARI | 0.0334 | 0.0011 | -0.0323 |

This is not a successful improvement: the ten extra rejected unknowns came
with nineteen additional known false rejections, while AUROC was unchanged,
AUPR/purity/ARI worsened. Treat the slightly better FPR95/OSCR as a mixed
tradeoff, not evidence that neighbor voting solves feature overlap. Do not
continue tuning this score on the same test set.

## Current decision

- Deterministic known-bank extraction is retained as an evaluation-correctness
  fix; prior kNN runs built from random augmentation should be labeled
  exploratory.
- Training-time kNN support boundary, global kNN mixture, classwise radius
  normalization, and neighborhood-vote conflict are not promoted to defaults.
- The raw predicted-class kNN score showed some ranking signal but failed to
  increase unknown rejection consistently across seeds. The measured core
  failure therefore remains unresolved.
- The next algorithm change should not be another post-hoc kNN score. First
  inspect the training objective actually active in each run and use a
  predeclared paired training ablation only if it directly changes the
  known/unknown representation or uncertainty objective. Otherwise prioritize
  a larger, protocol-valid representation audit and data split analysis.
