# Uncertainty-score correction weight comparison (2026-10-12)

## Question and controlled factor

Does adding the learned uncertainty-head signal to the normalized-entropy +
minimum-class-kNN detector reduce known/unknown score overlap? This is a
detector-only comparison: four previously trained nnPU checkpoints (seeds 44
and 123, encoder-gradient and head-only arms) were held fixed. The split,
known-validation normalization, feature bank, MC samples, calibration policy
(95% known coverage), and test set were unchanged within each checkpoint.
Only `--uncertainty-score-weight` changed: 0.25 versus 0.75. The existing 0.5
run is included as a reference; it was run before this weight sweep. The
underlying base score is entropy-z + minimum-class-kNN-distance-z.

The score formula is

```text
open_score = entropy_z + min_class_knn_distance_z
             + weight * uncertainty_head_z
```

All values below are from the saved `discovery_report.json` files. Unknown
rejection and known acceptance are measured at the validation-derived
95%-known-coverage threshold; small deviations on test are expected.

| seed | nnPU arm | weight | AUROC | AUPR | FPR95 | OSCR | known accept | unknown reject | KS | unknown below known-q95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 44 | encoder gradient | 0.25 | 0.8177 | 0.7159 | 0.5808 | 0.5334 | 94.78% | 28.63% | 0.4947 | 72.08% |
| 44 | encoder gradient | 0.50 | 0.8191 | 0.7183 | 0.5755 | 0.5336 | 94.72% | 28.88% | 0.4918 | 71.98% |
| 44 | encoder gradient | 0.75 | 0.8184 | 0.7183 | 0.5832 | 0.5325 | 94.98% | 27.53% | 0.4882 | 72.50% |
| 44 | head only | 0.25 | 0.8144 | 0.7106 | 0.5765 | 0.5279 | 94.97% | 26.20% | 0.4904 | 74.03% |
| 44 | head only | 0.50 | 0.8178 | 0.7156 | 0.5698 | 0.5288 | 94.33% | 29.63% | 0.4972 | 73.10% |
| 44 | head only | 0.75 | 0.8190 | 0.7181 | 0.5753 | 0.5284 | 94.32% | 31.63% | 0.4974 | 71.63% |
| 123 | encoder gradient | 0.25 | 0.8131 | 0.7179 | 0.6012 | 0.5638 | 94.62% | 28.18% | 0.4886 | 73.48% |
| 123 | encoder gradient | 0.50 | 0.8125 | 0.7198 | 0.6030 | 0.5632 | 93.80% | 31.50% | 0.4869 | 72.83% |
| 123 | encoder gradient | 0.75 | 0.8097 | 0.7187 | 0.6137 | 0.5611 | 93.73% | 32.53% | 0.4819 | 72.35% |
| 123 | head only | 0.25 | 0.8189 | 0.7197 | 0.5757 | 0.5645 | 95.42% | 25.18% | 0.5094 | 73.08% |
| 123 | head only | 0.50 | 0.8200 | 0.7226 | 0.5853 | 0.5642 | 95.65% | 25.38% | 0.5094 | 71.48% |
| 123 | head only | 0.75 | 0.8193 | 0.7232 | 0.5908 | 0.5627 | 95.22% | 27.78% | 0.5063 | 71.58% |

## Interpretation

- Raising the weight from 0.25 to 0.75 is not a robust improvement. It often
  raises the operating-point unknown rejection in the head-only arms, but this
  does not consistently improve ranking metrics or overlap. For example,
  seed-123 encoder-gradient rejection rises from 28.18% to 32.53%, while
  AUROC falls from 0.8131 to 0.8097, FPR95 worsens from 0.6012 to 0.6137,
  and KS falls from 0.4886 to 0.4819.
- At seed 44, the encoder-gradient arm has its best AUROC/FPR95 at weight 0.5;
  at seed 123, the head-only arm has its best AUROC/AUPR at 0.5/0.75 but best
  FPR95 at 0.25. The preferred weight depends on seed and metric.
- The overlap diagnostic `unknown below known-q95` remains about 71.5%-74.0%
  across the sweep. KS also changes only modestly and inconsistently. This
  means the uncertainty score correction has not separated the distributions
  in a stable way. A higher rejection rate alone is not evidence of reduced
  overlap.
- Do not change the default weight based on this small two-seed detector-only
  study. Keep the existing setting as an experimental option pending a
  prespecified selection rule and additional seeds.

## Validation and artifacts

The CLI parameter is checked by a unit test: weight 0 must exactly reduce to
the original entropy+kNN score, and nonzero weights must add the stated
standardized uncertainty component linearly. Full test suite: 218 passed,
2 warnings; `compileall` and `git diff --check` passed.

Reproduction: [`scripts/run_uncertainty_score_recheck_s44_s123.ps1`](../scripts/run_uncertainty_score_recheck_s44_s123.ps1).
Reports are in `runs/nnpu_uncertainty_score_w*_s*_discover/discovery_report.json`
and the weight-0.5 reference runs are in
`runs/nnpu_uncertainty_score_s*_discover/discovery_report.json`.

The next useful experiment should change a training-time representation
objective rather than continue searching detector weights. Keep disjoint
validation, fixed checkpoints/seeds, and compare score-distribution overlap,
AUROC/FPR95/OSCR, and known accuracy together.
