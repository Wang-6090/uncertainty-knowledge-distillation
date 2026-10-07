# Detector operating-point audit (2026-10-06)

## Purpose and protocol

This audit checks whether the recent `feature_rejector_fusion` improvement is
stable across operating points or only appears at the single threshold saved
by an earlier run. It compares the saved baseline student detector,
support-only nnPU rejector, and fusion detector on the same CIFAR-100 random
60/40 test sets for seeds `2026`, `42`, and `3407`.

The model checkpoints, split, test samples, and detector implementations are
fixed. Only the saved detector scores differ. The 90%, 95%, and 97% known
coverage points below are post-hoc diagnostics using test labels to obtain a
common comparison threshold; they are not deployable thresholds and were not
used to fit or select a method.

## Results at 95% known coverage

| Seed | Detector | AUROC | FPR95 | Unknown rejection | Known accept | Histogram overlap |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2026 | Student base | 0.7205 | 0.7170 | 14.63% | 95.00% | 0.674 |
| 2026 | Support-only nnPU | 0.7596 | 0.6855 | 21.90% | 95.00% | 0.623 |
| 2026 | Fusion 0.75 | 0.7649 | 0.6318 | 22.48% | 95.00% | 0.609 |
| 42 | Student base | 0.7250 | 0.7233 | 16.38% | 95.00% | 0.665 |
| 42 | Support-only nnPU | 0.7529 | 0.6853 | 19.35% | 95.00% | 0.624 |
| 42 | Fusion 0.75 | 0.7610 | 0.6748 | 21.90% | 95.00% | 0.608 |
| 3407 | Student base | 0.7105 | 0.7295 | 14.03% | 95.00% | 0.693 |
| 3407 | Support-only nnPU | 0.7467 | 0.6683 | 18.48% | 95.00% | 0.642 |
| 3407 | Fusion 0.75 | 0.7526 | 0.6638 | 18.20% | 95.00% | 0.633 |

The fusion improves AUROC, FPR95, and score-distribution overlap on all three
seeds. At the fixed 95% coverage point it improves over the base detector on
all seeds, but is slightly below support-only nnPU on seed `3407`. Therefore
fusion is a useful ranking candidate, not evidence that the representation
overlap has been solved.

At 90% coverage, fusion unknown rejection is `33.78%`, `34.80%`, and `30.35%`
for seeds `42`, `2026`, and `3407`; support-only nnPU gives `33.35%`, `34.75%`,
and `32.03%`. At 97% coverage, fusion gives `15.23%`, `16.00%`, and `13.50%`,
while support-only gives `13.73%`, `16.08%`, and `13.15%`. The seed-dependent
crossing confirms that further fusion-weight tuning on the same test set is
not justified.

## Metric audit finding

The audit also found a reporting bug in `run_discovery`: the old
`known_class_accuracy_all_known` value used accepted-and-correct known samples
as its numerator while dividing by all known samples. That quantity is useful
only as `accepted_correct_fraction_of_all_known`, not as ordinary known-class
accuracy. The code now reports:

- `known_class_accuracy_all_known`: classifier accuracy over every known sample;
- `known_class_accuracy_after_accept`: classifier accuracy among accepted known samples;
- `accepted_correct_fraction_of_all_known`: accepted-and-correct known samples divided by all known samples.

Existing JSON reports are historical artifacts and were not rewritten. New
experiments must be rerun after this fix before their corrected classification
metrics are used in a final table.

## Decision

1. Keep support-only nnPU and fusion as optional detector candidates. Fusion is
   currently the strongest ranking candidate, but it is not the default and
   does not replace representation learning.
2. Stop tuning fusion weights against the final test set.
3. Rerun the fixed candidate on an independent semantic-hard or
   semantic-isolated class split with a disjoint calibration subset. This is
   the next high-value check because the current gains may depend on the
   random class split and the validation/test score distribution.
4. If the independent split preserves the ranking/overlap gain, restore
   clustering and evaluate candidate purity, NMI, and ARI. If it fails, stop
   expanding the detector and return to the representation/data protocol.

Artifact: `analysis/detector_operating_points_audit_20261006.json`.
