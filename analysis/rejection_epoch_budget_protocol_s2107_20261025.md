# Protocol: shared-backbone rejection training budget (seed 2107)

## Question

Does the current shared-backbone independent rejection branch remain
under-optimized after four student epochs? This tests training budget directly,
without changing the rejection representation, loss family, uncertainty source,
candidate selector, or score rule.

## Pre-registered comparison

- Fresh CIFAR-100 semantic-hard 60/40 split, seed 2107.
- Control: four student epochs; treatment: eight student epochs.
- One split-matched frozen teacher and the same pretrained/shared-backbone
  independent rejection branch in both arms.
- Same optimizer, learning rate, batch size, mixed-pool nnPU discovery setup,
  checkpoint selection rule, feature-rejector score, automatic K, and
  known-validation-only 95% coverage calibration.
- No feature-KD, uncertainty-KD, VOS, extra boundary loss, selector or fusion
  change. No mixed-pool/test labels enter training, checkpoint selection,
  calibration, or method choice.

## Decision rule

Compare AUROC, AUPR, FPR95, OSCR, fixed-coverage unknown rejection, known
acceptance, accepted-known accuracy, candidate purity/recall, automatic K/NMI/ARI,
score overlap, and known-only rejection representation diagnostics. The budget
increase is useful only if it improves fixed-coverage unknown rejection by at
least `+2 pp` without materially worsening known utility, OSCR, or clustering.
A single positive split is exploratory and requires replication; a negative
result stops this exact 8-epoch budget comparison.

## Result

The paired run completed with four and eight history entries respectively.
The checkpoint rule selected epoch 3 for the four-epoch arm and epoch 8 for the
eight-epoch arm. Both arms reached the same known-validation calibration
coverage (`0.9490`), and the configuration audit found no training difference
other than the registered epoch budget and output/checkpoint fields.

| Metric | 4 epochs | 8 epochs | Treatment - control |
|---|---:|---:|---:|
| AUROC | 0.782367 | 0.811315 | +0.028947 |
| AUPR | 0.677964 | 0.714516 | +0.036552 |
| FPR95 | 64.933% | 58.783% | -6.150 pp |
| OSCR | 0.570402 | 0.606348 | +0.035946 |
| Known acceptance | 95.133% | 95.667% | +0.533 pp |
| Unknown rejection at calibrated threshold | 23.500% | 25.600% | +2.100 pp |
| Accepted-known accuracy | 69.516% | 71.446% | +1.930 pp |
| Correctly accepted fraction of all known | 66.133% | 68.350% | +2.217 pp |
| Automatic K | 35 | 38 | +3 |
| NMI / ARI | 0.491531 / 0.153857 | 0.486975 / 0.140670 | -0.004556 / -0.013187 |
| Candidate purity before purification | 76.299% | 79.751% | +3.452 pp |

## Interpretation

On this fresh split, extending the training budget produced a joint gain in
ranking, fixed-coverage rejection, OSCR, and known utility. The candidate pool
purity also improved. Automatic clustering NMI/ARI declined, so the result does
not establish a complete new-class discovery improvement. It is a useful
mechanism result suggesting that the shared-backbone rejection branch is still
under-optimized at four epochs, but it requires an unchanged fresh-split
replication before changing the default budget.

## Decision

Keep the registered eight-epoch budget as an explicit candidate only. Do not
add other losses, uncertainty weighting, selectors, or fusion to this result.
Run one exact replication on a new semantic-hard split; promote only if the
fixed-coverage and known-utility gains remain jointly positive and the clustering
degradation is not materially worse.
