# Rejection-logit detector ablation

## Question

Does exposing the separately trained rejection head's logits to the post-hoc
rejector add stable unknown-detection information beyond the rejection
embedding, known classifier summaries, and known-support score?

This is a detector-only ablation. It does not test whether the uniform outlier
exposure training objective is itself effective.

## Code changes

- Added `rejection_logit_augmented` and
  `rejection_support_logit_augmented` feature modes. They append rejection
  logits, softmax entropy, maximum softmax probability, and the top-1/top-2
  probability gap.
- Added `rejection_support_augmented` as a matched control. It uses the same
  normalized rejection embedding, known-class logits/probability summaries,
  learned uncertainty summary, and known-support score, but excludes rejection
  logits and their summaries.
- Kept all modes opt-in; the default detector is unchanged.
- Added tests for parser choices, feature shape/content, shared features
  between matched modes, and clear errors when required logits/support scores
  are unavailable.

## Protocol

For each seed, the exact same saved student checkpoint is evaluated twice;
the only intended difference is `rejector_feature_mode`:

- control: `rejection_support_augmented`
- treatment: `rejection_support_logit_augmented`

Shared settings: CIFAR-100 semantic-isolated 60/40 split; 1200/300/1000
train/validation/test limits; 5400-sample mixed unlabeled pool with known prior
0.2; linear `nu_corrected` nnPU rejector; MC=4 model extraction; validation-only
95% known-coverage threshold calibration; clustering disabled. Seed 42 and 43
checkpoints use feature-margin OE; seed 44 uses feature-margin plus uniform
rejection-logit OE. Therefore, comparisons are paired within each checkpoint;
cross-seed differences must not be attributed to a single training treatment.
No test labels were used to tune the detector or threshold.

## Results

| Seed | Rejector features | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | rejection embedding + support | 0.7312 | 0.7571 | 0.1121 | 96.51% | 14.29% |
| 42 | same + rejection logits | 0.7332 | 0.7621 | 0.1114 | 95.34% | 15.54% |
| 43 | rejection embedding + support | 0.7101 | 0.7651 | 0.1193 | 95.30% | 10.89% |
| 43 | same + rejection logits | 0.7047 | 0.7869 | 0.1186 | 96.48% | 9.16% |
| 44 | rejection embedding + support | 0.7150 | 0.7484 | 0.1737 | 95.70% | 11.83% |
| 44 | same + rejection logits | 0.7132 | 0.7691 | 0.1729 | 94.43% | 14.25% |

Run directories:

- Seed 42: `runs/posthoc_rejection_support_base_s42/` and
  `runs/posthoc_rejection_support_logit_s42/`
- Seed 43: `runs/posthoc_rejection_support_base_s43/` and
  `runs/posthoc_rejection_support_logit_s43/`
- Seed 44: `runs/posthoc_rejection_support_base_s44/` and
  `runs/posthoc_rejection_support_logit_s44/`

The report values are in each directory's `discovery_report.json`; full
commands/configuration are in `discover_config.json`.

## Interpretation

The treatment does not consistently improve ranking or the calibrated
operating point. Unknown rejection rises on seeds 42 and 44 but falls on seed
43. The corresponding changes in known acceptance, FPR95, and OSCR show a
tradeoff rather than a robust separation improvement. Do not make this mode
the default or claim that it resolves known/unknown feature overlap.

An earlier exploratory comparison used the legacy `support_augmented` mode as
control. That comparison changed both the base embedding (main classifier
embedding versus rejection embedding) and the appended logits, so it was
confounded and is not evidence for the independent effect of rejection logits.
The matched comparison above supersedes it for that question.

## Next check

Do not add more correlated post-hoc scores yet. First inspect (on validation
data, with test labels reserved for final reporting) the per-class and
semantic-neighbor-group distributions and correlations of rejection embedding,
rejection-logit entropy/margin, and the final rejector score. If the logits are
redundant or unstable across seeds, stop feature stacking and redirect effort
to the training objective and reliable open-validation/mixed-pool supervision.

## Verification

- `python -m compileall -q train.py novel_discovery`: passed
- `python -m pytest -q`: 187 passed
- `git diff --check`: passed
