# Uncertainty-KD score-alignment audit result

## Scope

This is a frozen, discovery-time audit on semantic-hard CIFAR-100 60/40
seed `2173`. It reuses the completed standard-KD and uncertainty-KD student
checkpoints from the class-holdout pilot. No model was retrained. For each
checkpoint, the same source discovery configuration was evaluated with four
existing score modes: `max_softmax`, `entropy_only`, `energy`, and
`head_uncertainty`.

Every mode used the same CIFAR transform, matched mixed discovery pool,
known-validation threshold calibration targeting 95% known coverage, and
`open_val_ratio=0`. The audit skipped clustering because it tests score
alignment only; the official `max_softmax` pilot already contains the full
candidate and clustering results. All reports record
`test_labels_used_for_training_or_calibration=false`.

## Results

The table reports uncertainty-KD minus standard-KD for the same score mode.
Percentage-point values are percentage points.

| score mode | AUROC | AUPR | FPR95 | known accuracy | unknown rejection |
|---|---:|---:|---:|---:|---:|
| `max_softmax` | +0.001498 | +0.003601 | -0.300 pp | -0.133 pp | +1.025 pp |
| `entropy_only` | +0.001744 | +0.005193 | -0.017 pp | -0.133 pp | +1.425 pp |
| `energy` | +0.001212 | +0.003375 | +0.833 pp | -0.133 pp | -0.025 pp |
| `head_uncertainty` | +0.006369 | +0.001191 | -0.317 pp | -0.133 pp | +0.025 pp |

Absolute `head_uncertainty` performance was poor for both students: standard
AUROC `0.476618` and unknown rejection `2.925%`; uncertainty-KD AUROC
`0.482987` and unknown rejection `2.950%`. This is a diagnostic result about
the current uncertainty head and score direction, not a claim that uncertainty
heads are generally invalid.

## Decision

The audit does not support the hypothesis that previous uncertainty-KD results
were hidden solely because the official evaluation used `max_softmax`.
Uncertainty-KD's small positive ranking change appears under both
`max_softmax` and `entropy_only`, while `energy` and `head_uncertainty` do not
provide a consistent downstream gain. The uncertainty head itself is not a
useful standalone unknown score on this checkpoint.

No score channel, weight, or training coefficient is promoted from this
single-seed diagnostic. The default score and `standard-KD` remain unchanged;
uncertainty-KD remains an opt-in comparison. A future training experiment
should test target alignment only if it can define a known-only, validation
auditable relation between the training uncertainty target and the deployed
score. It should retain standard-KD and uncertainty-KD as paired controls and
must pass the existing gradient and downstream gates before any fresh-seed
confirmation.

Raw reports are stored in the ignored run directories under
`runs/team_a_score_alignment_20261125_standard_*` and
`runs/team_a_score_alignment_20261125_uncertainty_*`.
