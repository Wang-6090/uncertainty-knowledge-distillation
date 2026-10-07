# Mixed-pool nnPU + uncertainty-margin rejector recheck (2026-10-06)

## Purpose

The current leading student is the mixed-pool nnPU plus uncertainty-weighted
feature-margin model. This diagnostic asks whether a separately trained
feature rejector can improve the detector without changing representation
training. It is a detector-only comparison: the student checkpoint, discovery
pool, known validation set, split, and threshold policy are fixed.

The rejector is trained after student training from frozen features. The known
side is held-out known open-validation data and the unlabeled side is the
matched mixed discovery pool. Test labels are not used for fitting the
rejector or threshold.

## Results

| Detector | AUROC | FPR95 | OSCR | Known accuracy | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Student normalized entropy + feature kNN | 0.7250 | 0.7233 | 0.4802 | 56.82% | 95.03% | 16.33% |
| Support-only nnPU rejector | 0.7529 | 0.6853 | 0.4796 | 56.00% | 94.85% | 19.73% |
| Support + MC uncertainty nnPU rejector | 0.7470 | 0.6943 | 0.4739 | 55.98% | 94.93% | 20.75% |

The support-only rejector improves AUROC by 0.0279, FPR95 by 0.0380, and
unknown rejection by 3.40 percentage points relative to the fixed student
detector. OSCR decreases by 0.0005 and known accuracy by 0.82 points, so it
is a useful detector candidate rather than an unqualified improvement.

Adding MC uncertainty to the rejector input increases unknown rejection by
another 1.02 points, but worsens AUROC, FPR95, OSCR, and known accuracy. The
extra uncertainty feature is therefore not retained as the preferred input.

## Decision

- Keep the student training method as mixed-pool nnPU plus uncertainty-margin.
- Keep support-only nnPU rejector as an optional detector ablation; do not
  silently replace the main detector in historical comparisons.
- Do not continue stacking MC uncertainty, more rejector inputs, or more
  rejector losses without a new hypothesis. The current result suggests the
  support boundary carries useful information, while redundant uncertainty
  features add noise.
- Report AUROC, FPR95, OSCR, known accuracy, known acceptance, and unknown
  rejection together. Unknown rejection alone is not sufficient because a
  detector can improve it by rejecting more known samples.

Artifacts:

- `runs/matched_mixed_nnpu_margin_s42_e10_rejector/`
- `runs/matched_mixed_nnpu_margin_s42_e10_rejector_uncertainty/`

