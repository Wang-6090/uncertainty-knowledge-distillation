# Class-holdout support proxy pilot result

## Protocol

This is the single-seed screening run registered in
`class_holdout_proxy_protocol_20261124.md`: semantic-hard CIFAR-100 60/40,
seed `2173`, four training epochs, one split-matched teacher, and three paired
student arms. The arms use the same known-validation checkpoint selection,
95% known-coverage calibration, max-softmax score, matched discovery pool,
Auto-K, and clustering procedure.

- `standard`: CE + standard logits KD;
- `class_holdout`: standard plus `alpha_class_holdout_proxy=0.1`;
- `uncertainty`: CE + existing teacher-uncertainty-weighted logits KD.

Feature KD, rejection training, discovery training, score fusion, and all
parameter scans were disabled. Unknown/test labels were used only for the
final retrospective metrics.

## Mechanism audit

The treatment was isolated correctly. The proxy loss, active fraction, and
audited proxy gradient were zero in the `standard` and `uncertainty` arms.
For `class_holdout`, the four epoch values were:

| quantity | epoch 1 | epoch 2 | epoch 3 | epoch 4 |
|---|---:|---:|---:|---:|
| proxy loss | 0.497184 | 0.430094 | 0.401969 | 0.378516 |
| active fraction | 0.2620 | 0.2579 | 0.2570 | 0.2568 |
| proxy gradient norm | 0.173408 | 0.168570 | 0.169200 | 0.168451 |
| student update norm | 0.038391 | 0.027614 | 0.027003 | 0.026655 |

All mechanism gates passed. The proxy therefore entered the shared student
feature objective and produced a nonzero audited gradient; this is an
implementation result, not a performance result.

## Final metrics

Changes below are relative to `standard` on the same seed. Percentage-point
values are shown as percentage points.

| metric | standard | class-holdout | delta | uncertainty | delta |
|---|---:|---:|---:|---:|---:|
| AUROC | 0.664918 | 0.665120 | +0.000202 | 0.666416 | +0.001498 |
| AUPR | 0.524237 | 0.523440 | -0.000797 | 0.527838 | +0.003601 |
| FPR95 | 0.775167 | 0.776333 | +0.117 pp | 0.772167 | -0.300 pp |
| known accuracy | 0.680000 | 0.672667 | -0.733 pp | 0.678667 | -0.133 pp |
| unknown rejection | 0.088750 | 0.093750 | +0.500 pp | 0.099000 | +1.025 pp |
| candidate purity | 0.565287 | 0.562219 | -0.307 pp | 0.575581 | +1.029 pp |
| cluster ACC | 0.187898 | 0.203898 | +1.600 pp | 0.183140 | -0.476 pp |
| cluster NMI | 0.485871 | 0.490217 | +0.435 pp | 0.463750 | -2.212 pp |
| cluster ARI | 0.057472 | 0.066844 | +0.937 pp | 0.060311 | +0.284 pp |
| Auto-K absolute error | 2 | 7 | +5 | 1 | -1 |

## Decision

The class-holdout proxy passes the mechanism gate but fails the promotion
gate. Its unknown rejection increase is accompanied by lower known accuracy
and candidate purity, and Auto-K becomes substantially worse. The small
AUROC/AUPR changes do not establish a stable transfer from representation
training to unknown detection or novel-class discovery. This is one seed, so
it is not evidence that the hypothesis is universally false, but it is enough
to stop parameter scanning under the registered screening rule.

The treatment remains an opt-in diagnostic/ablation. It does not replace
`standard-KD`, and it does not change the default uncertainty or discovery
pipeline. The raw result remains at
`analysis/team_a_class_holdout_proxy_20261124_s2173.json`.

## Interpretation and next stage

The experiment narrows the problem: a known-only proxy can push shared
features away from sampled known support prototypes, but that local pressure
does not specify the unknown class partition and can reduce closed-set
classification quality. The next experiment should address target alignment
between the training signal and the deployed unknown score, while keeping the
standard-KD arm and the uncertainty-KD arm as clean comparisons. It should
first be a minimal, known-only mechanism audit and use a fresh split only
after the audit passes; no further class-holdout coefficient sweep is
justified by this pilot.
