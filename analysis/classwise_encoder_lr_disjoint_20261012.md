# Class-wise open-error audit for the disjoint-calibration LR comparison

## Purpose and protocol

This is a retrospective diagnostic of the already completed seed-44 and
seed-123 encoder-LR comparisons. It does not train a model, choose a threshold,
or select a method. For each seed, it compares the existing LR=1.0 and LR=0.1
detector outputs using the same semantic-isolated CIFAR-100 split and the
already frozen validation-calibrated threshold. Official test labels are used
only to group errors by raw class after inference.

The run artifacts' `split.pt` files were checked against
`splits_cifar100_protocols/cifar100_60_40_semantic_isolated.json`; their class
lists match. A separate hidden split JSON in the workspace has different class
lists and no class-name mapping, so it was not used. The LR arms within each
seed share the same student training protocol, including the matched mixed
pool (5,400 samples; known prior 0.2), five epochs, teacher, detector, and
disjoint known calibration subset. Cross-seed class recurrence is descriptive,
not a new causal comparison.

Implementation: `scripts/analyze_classwise_open_errors.py`.
Machine-readable results:

- `analysis/classwise_encoder_lr_disjoint_s44_20261012.json`
- `analysis/classwise_encoder_lr_disjoint_s123_20261012.json`

## Findings at LR scale 0.1

The worst ten novel classes by false-accept rate share seven classes across
the two seeds. Several difficult classes repeat with very high false-accept
rates:

| Novel class | Seed 44 false-accept rate | Seed 123 false-accept rate |
|---|---:|---:|
| television | 97% | 94% |
| telephone | 94% | 94% |
| lamp | 92% | 93% |
| clock | 90% | 95% |
| bus | 89% | 92% |
| pickup truck | 89% | 93% |
| mountain | 85% | 92% |

The classes most often predicted for these false accepts are not a single
universal sink: television is most often predicted as wardrobe, bus as
streetcar, pickup truck as lawn mower/couch, while telephone and lamp often
map to different object classes. The unstable target mapping argues against
adding a class-specific repulsion loss based on these test errors.

The overall LR=0.1 unknown false-accept rates are 69.50% (seed 44) and 74.90%
(seed 123). Thus the repeated class pattern coexists with substantial seed
variation in aggregate behavior. This diagnostic supports the claim that the
remaining overlap is structured, but does not identify a single class pair or
loss that will solve it. It is not evidence that per-class thresholds or
class-targeted training will generalize.

## Interpretation and next step

Do not train against the test-derived class list. The model was trained with
`open_val_ratio=0`, so it has no independent known-and-unknown validation
subset for selecting an unknown-boundary objective or score. A post-hoc split
of the existing discovery pool would be invalid because its novel samples
were exposed during training.

The next controlled experiment should reserve genuinely disjoint known and
novel open-validation examples from the CIFAR training partition before
constructing the unlabeled discovery pool, then compare the LR=0.1 baseline
with one training-time boundary intervention. Fix the validation protocol and
all training factors; use the open-validation data only for the prespecified
selection/calibration step, and evaluate once on the untouched official test
set. Report AUROC, FPR95, OSCR, score overlap, known acceptance, unknown
rejection, and candidate quality. If a treatment only changes the operating
threshold or sacrifices known performance, do not count it as a representation
improvement.

The open-validation protocol can test whether current failure is partly a
score-selection/calibration mismatch, but it cannot itself be described as
improving representation geometry. Any claim about feature separation must be
supported by the frozen test ranking and feature-overlap measurements, not by
the validation-selected threshold alone.
