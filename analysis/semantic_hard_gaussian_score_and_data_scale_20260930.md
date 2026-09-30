# Semantic-hard score check and training-data scale follow-up (2026-09-30)

## Question and decision rule

The core issue is weak separation between known and held-out fine classes. This
note first checks whether the Gaussian class-conditional NLL score helps on an
independent semantic-hard class split, then starts a data-scale check because
the current smoke-trained classifier has very low known-class accuracy.
Raising the unknown rejection rate alone is not success if it is obtained by
rejecting more known examples.

## Paired score evaluation

Only the score mode changes. Both evaluations use the same seed-42
ResNet-18 student checkpoint, CIFAR-100 semantic-hard 60/40 split, 64-pixel
images, 1,200/300/1,000/1,200 train/validation/test/discovery limits, MC=4,
known-validation 95% coverage calibration, and skipped clustering. Test labels
are used only to calculate final metrics. Outputs are in
`runs/semantic_hard_gaussian_s42_{mahal,nll}_eval/`.

| Metric | Normalized entropy + Mahalanobis | Normalized entropy + Gaussian NLL |
|---|---:|---:|
| AUROC | 0.4785 | 0.4911 |
| AUPR | 0.3929 | 0.4128 |
| FPR95 | 0.9436 | 0.9419 |
| OSCR | 0.1257 | 0.1418 |
| Test known acceptance at validation-calibrated threshold | 93.85% | 93.16% |
| Unknown rejection at validation-calibrated threshold | 4.34% (18/415) | 5.78% (24/415) |
| Known classification accuracy, all known test samples | 20.51% | 20.51% |

A diagnostic-only threshold recomputed from test-known scores to give exactly
95% test-known acceptance yields 15/415 (3.61%) unknown rejection for
Mahalanobis and 18/415 (4.34%) for Gaussian NLL. This uses test-known labels,
is not deployable, and is not a threshold-selection result. Thus NLL improves
some ranking metrics and rejects only three more unknown test samples at that
retrospective matched-coverage point. One seed does not establish a robust
benefit; the independent split reproduces only a small/mixed signal. Keep the
score optional and do not change the default based on this result.

The 40 held-out classes are also highly uneven at the per-class operating
point: median class rejection is 0 for both scores. Many novel categories are
never rejected, so this is not just a global calibration issue. The worst
classes include cockroach, elephant, man, pear, road, shark, skunk,
skyscraper, snake, and squirrel; several are mapped to semantically unrelated
known predictions as well as to same-coarse-group classes. This supports
checking representation quality and training adequacy rather than treating
the problem as a threshold-only issue. See the raw predictions and labels in
the two `discovery_detail.json` files.

## Training-data scale check

The current limited-data student has 22.67% validation known accuracy and
20.51% test-known accuracy. Before changing the loss again, compare it with a
full-known-training-data run using the same semantic-hard split, seed, model
architectures, teacher/student epoch counts, optimizer/loss defaults, image
size, batch size, validation/test limits, mixed-pool ratio, and discovery-pool
limit. The intended changed factor is the available labeled known training
data (1,200 vs 21,600 student training examples; the teacher likewise changes
from 1,200 to its full known training partition). Both constructed student
discovery pools were checked to contain the same 1,200 examples: 947 known and
253 novel. Because epoch count is held fixed, the full-data run naturally has
more optimizer steps; this measures the practical full-data training protocol,
not an isolated optimizer-step ablation.

Planned evaluation is the same checkpoint-paired detector protocol: same
1,000 test samples, 300 known validation samples, score mode, MC count,
95%-known validation calibration, and no clustering. Primary checks are
validation/test known accuracy, test known acceptance, unknown rejection,
AUROC/FPR95/OSCR, and per-class rejection. Do not infer that data scale alone
caused a change if any other training configuration differs; record all
observed deviations before interpreting the result.

## Full-data schedule follow-up

The full-data student was then trained for five epochs instead of three. The
teacher, all loss weights, mixed discovery-pool construction, batch size,
optimizer, split, and detection protocol were held fixed. This is a schedule
control, not a new loss proposal.

| Metric | Full data, 3 epochs | Full data, 5 epochs |
|---|---:|---:|
| Best known validation accuracy | 41.00% | 46.67% |
| Test known classification accuracy | 39.15% | 45.30% |
| AUROC | 0.5107 | 0.5669 |
| FPR95 | 0.9265 | 0.9009 |
| OSCR | 0.2383 | 0.3093 |
| Validation-threshold known acceptance | 92.31% | 95.04% |
| Validation-threshold unknown rejection | 6.75% | 6.27% |
| Unknown rejection at exact 95% test-known coverage | 5.54% (23/415) | 6.27% (26/415) |
| Accepted-known accuracy at exact 95% coverage | 40.00% | 46.13% |

The schedule extension improves the main ranking and joint open-set metrics;
the exact-coverage unknown rejection also rises while accepted-known accuracy
improves. The validation-threshold rejection rate decreases slightly, which is
why the matched-coverage and OSCR rows are necessary. Five epochs is therefore
the better current development baseline, although 6.27% unknown rejection
remains far from satisfactory.

The paired representation diagnostic is saved at
`analysis/semantic_hard_epoch_scale_feature_diagnostics_s42.json`. From three
to five epochs, classifier-prototype distance AUROC changes 0.5457 to 0.5972,
empirical class-centroid distance changes 0.5284 to 0.5530, and nearest-known
sample distance changes 0.5486 to 0.5717. Histogram overlap remains high, but
the direction is consistent with real representation improvement. Future
algorithm comparisons should use this five-epoch full-data student as the
baseline.

## Current status

- Gaussian NLL: not promoted; small one-seed signal only.
- Full-data teacher, student, paired detection, and feature diagnostics are
  complete in `runs/semantic_hard_full_data_*_s42/` and
  `analysis/semantic_hard_data_scale_feature_diagnostics_s42.json`.
- A strict 1,200-example matched control was also completed in
  `runs/semantic_hard_matched_1200_*_s42/`.
- The five-epoch full-data student and evaluation are in
  `runs/semantic_hard_full_data_student5_s42/` and
  `runs/semantic_hard_full_data_student5_mahal_eval_s42/`.
- No source-code changes or GitHub push in this follow-up so far.

## Completed data-scale comparison

The strict control group was then completed. Both student runs use the same
semantic-hard split, seed, ResNet-18/ResNet-34 architectures, batch size 32,
three student epochs, uncertainty KD, mixed discovery pool, 20% known-pool
ratio, 1,200 discovery samples, and the same loss weights. The only intended
change is labeled known training size: 1,200 versus the full 21,600 examples.
The corresponding teachers were trained with the same data-size change.

| Metric | Matched 1,200 | Full known training data |
|---|---:|---:|
| Student best known validation accuracy | 22.33% | 41.00% |
| Test known classification accuracy | 17.26% | 37.09% |
| AUROC, normalized entropy + Mahalanobis | 0.4867 | 0.5107 |
| FPR95 | 0.9402 | 0.9265 |
| OSCR | 0.1134 | 0.2383 |
| Test known acceptance at validation threshold | 95.04% | 92.31% |
| Unknown rejection at validation threshold | 3.13% | 6.75% |
| Unknown rejection at retrospective exact 95% test-known coverage | 3.13% (13/415) | 5.54% (23/415) |

The exact-coverage row is diagnostic-only: its threshold uses test-known
labels and was not used for selection. The full-data run rejects more unknowns
while rejecting the same fraction of known samples in this diagnostic (5.13%),
and its known classification accuracy is much higher. This is the first clear
evidence in the current protocol that insufficient labeled known data is a
real contributor to the bad open-set result. It does not solve the problem:
the absolute unknown rejection remains low and the per-novel-class median
rejection is still 0 in both runs.

The post-hoc feature diagnostic is saved at
`analysis/semantic_hard_data_scale_feature_diagnostics_s42.json`. For the full
data student versus the matched 1,200 student, unknownness AUROC based on
classifier-prototype distance changes 0.5164→0.5457, empirical class-centroid
distance changes 0.4716→0.5284, and nearest-known-training-sample distance
changes 0.4884→0.5486. Thus the improvement is present in the representation,
not only in threshold calibration, although histogram overlap remains high.

## Negative score follow-up

To test whether class-conditional covariance would exploit that improvement,
the same two checkpoints were evaluated with the existing
`normalized_entropy_classwise_mahalanobis` score. No training or threshold
selection on test labels was changed.

| Metric | Matched 1,200 | Full data |
|---|---:|---:|
| AUROC | 0.4878 | 0.4979 |
| FPR95 | 0.9470 | 0.9231 |
| OSCR | 0.1088 | 0.2379 |
| Unknown rejection at validation threshold | 2.17% | 8.43% |
| Test known acceptance at validation threshold | 93.85% | 90.94% |
| Unknown rejection at exact 95% test-known coverage | 1.69% | 4.34% |

Although some ranking metrics and the validation-threshold rejection rate look
better, exact matched-coverage rejection is worse than ordinary Mahalanobis in
both data regimes. This is a score-scale/false-rejection trade-off, not a
solution to known/unknown overlap. Keep the classwise score as an optional
diagnostic only and do not make it the project default.

## Revised interpretation and next technical direction

1. Use the full known training partition for the main development protocol;
   the 1,200-example runs are smoke/control runs only.
2. Keep ordinary Mahalanobis (or a predeclared score comparison) for the next
   training experiment; do not select scores using test labels.
3. The next algorithm change should target representation learning with the
   full-data protocol and be compared against the full-data student, not
   against the under-trained 1,200 baseline. The change must be one explicit
   training objective at a time, with known accuracy and matched-coverage
   unknown rejection as co-primary checks.
4. Continue reporting per-novel-class rejection and known/unknown distance
   overlap. A global AUROC increase alone is insufficient.

## Strict mixed nnPU rejector follow-up

Because a pure-unknown feature rejector is not a fair open-world protocol, the
existing strict-mixed nnPU option was evaluated post hoc on both student
checkpoints. It used a 20% open-validation split, 116 held-out known samples
as the labeled positive side, the disjoint mixed discovery pool as unlabeled
data, known prior 0.2, logistic frozen embeddings, and the same validation
95%-coverage threshold. The student checkpoints and test protocol were not
changed.

| Metric | Matched 1,200 Mahalanobis | Matched 1,200 strict nnPU | Full Mahalanobis | Full strict nnPU |
|---|---:|---:|---:|---:|
| AUROC | 0.4867 | 0.5772 | 0.5107 | 0.5523 |
| FPR95 | 0.9402 | 0.9060 | 0.9265 | 0.8855 |
| OSCR | 0.1134 | 0.0922 | 0.2383 | 0.1936 |
| Validation-threshold unknown rejection | 3.13% | 5.78% | 6.75% | 4.82% |
| Validation-threshold known acceptance | 95.04% | 95.90% | 92.31% | 95.73% |
| Unknown rejection at exact 95% test-known coverage | 3.13% (13/415) | 7.71% (32/415) | 5.54% (23/415) | 5.78% (24/415) |
| Accepted-known classification accuracy at exact 95% coverage | 18.20% | 18.02% | 40.00% | 37.66% |

The control-model gain is not enough: OSCR decreases and accepted-known
accuracy is unchanged. On the stronger full-data representation, matched
unknown rejection improves by only one sample and accepted-known accuracy
decreases. The rejector is therefore useful as an optional ranking diagnostic,
but it is not a stable solution to the feature-overlap problem. Its output is
saved in `runs/semantic_hard_{matched_1200,full_data}_nnpu_rejector_s42/`.

The same strict-mixed nnPU protocol was finally applied to the five-epoch
full-data student. Relative to its ordinary Mahalanobis baseline, AUROC changed
0.5669→0.5796, but FPR95 stayed 0.9009, validation-threshold unknown rejection
fell 6.27%→4.10%, exact-95% unknown rejection fell 6.27%→5.30%, OSCR fell
0.3093→0.2431, and accepted-known accuracy fell 46.13%→43.42%. This confirms
that the apparent AUROC gain is not the desired operating-point improvement.
Stop treating strict mixed nnPU as the main solution; retain it only as an
optional diagnostic/rejector ablation.

The current best comparison is therefore the five-epoch full-data Mahalanobis
student. The central problem remains: even after adequate training, only about
6.27% of unknown test samples are rejected at matched 95% known coverage.
The next algorithm change must modify the representation/training objective,
not merely replace the score or threshold.

## Candidate-gated feature separation (code change and pilot)

The prior feature-margin/separation/boundary objectives were restricted to a
pure-unknown discovery pool, because applying them to every sample in a mixed
pool would incorrectly push known samples away from the known support. The
code now provides `--discovery-feature-candidate-gating`: in a mixed pool it
selects the top-risk fraction using the existing uncertainty/entropy candidate
rule, using the average risk of the two augmented views, and applies the
feature loss only to those candidates. The default remains off. A toy mixed
pool smoke test confirmed that the path trains and the full regression suite
still passes.

The first full-data pilot changed only this training objective relative to the
five-epoch baseline: `alpha_discovery_feature_separation=0.05`, margin 0,
temperature 0.1, candidate ratio 0.25, `entropy_uncertainty` mode. All other
training and detection settings were fixed.

| Metric | Full-data 5-epoch baseline | Candidate-gated feature separation |
|---|---:|---:|
| Best known validation accuracy | 46.67% | 48.00% |
| Test known classification accuracy | 45.30% | 46.15% |
| AUROC | 0.5669 | 0.5550 |
| FPR95 | 0.9009 | 0.9060 |
| OSCR | 0.3093 | 0.3119 |
| Validation-threshold known acceptance | 95.04% | 95.90% |
| Validation-threshold unknown rejection | 6.27% | 5.54% |
| Unknown rejection at exact 95% test-known coverage | 6.27% (26/415) | 6.99% (29/415) |
| Accepted-known accuracy at exact 95% coverage | 46.13% | 47.39% |

The matched operating point improves by three rejected unknown samples and
accepted-known accuracy also rises, but AUROC/FPR95 worsen. The independent
feature diagnostic in
`analysis/semantic_hard_candidate_feature_sep_diagnostics_s42.json` is mixed:
classifier-prototype distance AUROC changes 0.5972 to 0.5984, while empirical
centroid distance changes 0.5530 to 0.5437 and nearest-known-sample distance
changes 0.5717 to 0.5617. Therefore this is a promising but unconfirmed
candidate, not a promoted method. It should be repeated on fixed additional
seeds or selected on a held-out open-validation set before any coefficient or
candidate-ratio tuning. Do not claim it solves the overlap problem from this
single result.
