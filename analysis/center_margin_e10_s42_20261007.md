# Ten-epoch center-margin confirmation (seed 42, 2026-10-07)

## Purpose and controlled factors

The three-epoch center-margin pilot showed a small positive signal, but it was
weaker than the uncertainty treatment. This run tests whether that result was
only caused by the short training budget.

The semantic-isolated CIFAR-100 60/40 split, pretrained ResNet-34 teacher,
pretrained ResNet-18 student, mixed discovery pool with known prior 0.2,
ten-epoch budget, class-stratified 5000-sample support, `nu_corrected` linear
nnPU rejector, MC setting, and validation-only 95% known-coverage calibration
were fixed. The center-margin-only arm changed only the student loss by adding
`alpha_center_margin=0.02` and `center_margin=0.1`; it did not enable the
uncertainty treatment terms.

## Results

| Arm | AUROC | FPR95 | OSCR | Known accuracy | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.7866 | 0.6027 | 0.3795 | 43.63% | 95.35% | 21.85% |
| uncertainty treatment | **0.8008** | **0.5518** | **0.4629** | **52.45%** | 94.18% | **23.65%** |
| center-margin-only | 0.7751 | 0.6375 | 0.3716 | 43.15% | 94.88% | 21.25% |

Relative to the matched baseline, center-margin-only changes AUROC by
`-0.0115`, FPR95 by `+0.0348`, OSCR by `-0.0079`, and unknown rejection by
`-0.60pp`. It is therefore not a useful main method at this budget. The
three-epoch positive signal did not survive the longer confirmation.

## Decision

Stop tuning center-margin. Keep the option as a reproducible negative
ablation. The uncertainty treatment remains the stronger training direction,
while the remaining unknown-rejection gap must be addressed by a more direct
rejector objective or representation learning target.

Artifacts:

- `runs/semantic_isolated_full_e10_s42_center_margin/`
- `runs/semantic_isolated_full_e10_s42_center_margin_rejector_support/`
