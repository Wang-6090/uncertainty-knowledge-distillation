# Full-data rejection uniform-OE multi-seed recheck (2026-10-07)

## Question

Does adding uniform-logit Outlier Exposure to the independent rejection head
help beyond the existing rejection-feature-margin objective when the model is
trained on the complete semantic-isolated CIFAR-100 training split? This
checks whether the earlier short, limited-data experiments understated the
effect. The initial seed-42 comparison was subsequently repeated with training
RNG seeds 43 and 44 while keeping the class/data split fixed. The three-seed
evidence does not support a stable efficacy claim.

## Controlled comparison

All students use the same saved ResNet-34 teacher, semantic-isolated CIFAR-100
60/40 class/data split (`--seed 42`), complete train/validation/test sets,
pretrained ResNet-18 student, batch size 64, five epochs, rejection feature
dimension 128, and all other saved training options. Student training RNG
seeds are 42, 43, and 44 (`--model-seed`), so each pair shares the data split
but varies initialization and training randomness. Within each pair, a
structured comparison of the training configs found one algorithmic setting
changed:

- control: `alpha_outlier_rejection_uniform=0.0`
- treatment: `alpha_outlier_rejection_uniform=0.05`

Both retain `alpha_outlier_rejection_feature_margin=0.05`. The uniform loss is
computed on external CIFAR-10 outliers and the independent rejection logits:

`L_uniform = -mean_i mean_c log softmax(rejection_logits_i)_c`.

For the seed-42 treatment, the loss was active in every epoch (logged values
5.57, 5.77, 5.45, 4.94, 4.53); the control value was zero. This matches the uniform-target
Outlier Exposure idea of Hendrycks et al. (ICLR 2019), applied here to the
rejection head rather than the known-class classifier.

Detection was also paired: complete CIFAR-100 data, the same 5,400-item mixed
pool (known fraction 0.2), linear `nu_corrected` nnPU rejector,
`rejection_support_augmented` features, MC=4 extraction, validation-only 95%
known-coverage calibration, and clustering disabled. The only detector input
that changed was the student checkpoint. Run directories:

- seed-42 control/treatment: `runs/rejection_full_s42_margin/` and
  `runs/rejection_full_s42_margin_uniform/`
- seed-43 control/treatment: `runs/rejection_full_s42_model43_margin/` and
  `runs/rejection_full_s42_model43_margin_uniform/`
- seed-44 control/treatment: `runs/rejection_full_s42_model44_margin/` and
  `runs/rejection_full_s42_model44_margin_uniform/`
- matching detection reports use the same directory names with `_detect`
  appended
- shared teacher: `runs/rejection_full_s42_teacher/`

The final test labels were used only for reporting metrics and descriptive
per-unknown-class error analysis, never for threshold or model selection.

## Results

| Model seed | Arm | Test known accuracy | AUROC | AUPR | FPR95 | OSCR | Known acceptance | Unknown rejection |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | Margin only | 0.3965 | 0.7485 | 0.6176 | 0.6780 | 0.3392 | 95.00% | 17.15% |
| 42 | Margin + uniform OE | 0.4433 | 0.7623 | 0.6333 | 0.6323 | 0.3790 | 94.42% | 20.10% |
| 43 | Margin only | 0.3905 | 0.7703 | 0.6425 | 0.6158 | 0.3402 | 94.92% | 19.50% |
| 43 | Margin + uniform OE | 0.4303 | 0.7698 | 0.6342 | 0.6010 | 0.3742 | 94.58% | 19.13% |
| 44 | Margin only | 0.4855 | 0.7808 | 0.6483 | 0.5968 | 0.4216 | 94.38% | 19.93% |
| 44 | Margin + uniform OE | 0.4307 | 0.7743 | 0.6476 | 0.6277 | 0.3720 | 94.72% | 20.33% |

Paired changes (treatment minus control):

| Model seed | Test known accuracy | AUROC | AUPR | FPR95 | OSCR | Known acceptance | Unknown rejection |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | +0.0468 | +0.0138 | +0.0157 | -0.0457 | +0.0398 | -0.58 pp | +2.95 pp |
| 43 | +0.0398 | -0.0004 | -0.0083 | -0.0148 | +0.0340 | -0.33 pp | -0.38 pp |
| 44 | -0.0548 | -0.0065 | -0.0007 | +0.0308 | -0.0495 | +0.33 pp | +0.40 pp |
| Mean change | +0.0106 | +0.0023 | +0.0022 | -0.0099 | +0.0081 | -0.19 pp | +0.99 pp |

At the validation-calibrated operating point, unknown false accepts decreased
from 3,314/4,000 to 3,196/4,000. In a descriptive per-raw-class breakdown,
unknown acceptance decreased for 27 of 40 novel classes, was unchanged for 2,
and increased for 11. Mean novel-class acceptance changed from 85.7% to 79.9%;
the median changed from 85.5% to 82.5%. Some classes regressed notably (for
example raw labels 4, 23, and 34), so this does not eliminate semantic
overlap. This test-set breakdown is explanatory only and must not drive
hyperparameter selection.

## Interpretation and limitations

The seed-42 gain does not generalize consistently. Seed 43 has near-zero AUROC
change and a small decrease in unknown rejection; seed 44 loses AUROC, FPR95,
OSCR, and known accuracy, while unknown rejection increases by only 0.40 pp.
Across the three paired seeds, mean AUROC gain is only 0.0023 and mean unknown
rejection gain is 0.99 pp. The direction varies by seed, so this treatment is
not established as effective. Unknown rejection remains about 19%--20%, with
roughly four fifths of unknown test images accepted at the calibrated
operating point.

The threshold was calibrated using validation data only, with no test-label
tuning. However, that same validation split selected the best student
checkpoint, so checkpoint selection and threshold calibration were not
independent. The multi-seed comparison is now complete, but it used this same
overlapping validation protocol. The next useful check is to reserve a
disjoint calibration subset and repeat with fixed training and selection
rules; do not select a loss weight based on these test results.

Keep this loss opt-in as an ablation, not a promoted method. First inspect
whether it changes known/unknown feature and score distributions consistently
across seeds under a disjoint calibration protocol. Only if a non-test-set
diagnostic supports a common trend should a predeclared weight comparison be
run. Otherwise, shift effort to semantic-near unknown representation learning
rather than tuning a threshold or OE weight.

## Verification

- Full-data paired training completed on CUDA; each student ran five epochs.
- Both full-data detection runs completed with identical detector/calibration
  arguments and clustering disabled.
- Configuration audit found the intended single algorithmic training delta.
- `python -m compileall -q train.py novel_discovery tests`: passed.
- `python -m pytest -q`: 189 passed, 2 non-failing warnings.
- `git diff --check`: passed.
