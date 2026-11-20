# Episode-conditioned pseudo-novel rejection protocol

## Purpose

The current shared-backbone independent rejection branch is trained mostly on
known-class rejection labels. Existing VOS, Outlier Exposure, relational KD,
SupCon, view consistency and candidate-gating variants have not produced a
stable end-to-end improvement. This protocol tests a different hypothesis:
whether a rejection representation can learn to treat a class as novel when
that class is absent from the episode support, while preserving ordinary
classification supervision.

This is a known-only meta-training hypothesis. It does not claim that a CIFAR
fine class is intrinsically unknown. A class is pseudo-novel only relative to
the current episode support.

## Mechanism

For each known-labelled training batch, sample a subset of the classes present
in that batch as pseudo-novel query classes. The remaining batch classes are
the episode support. The main student classification loss, logits KD, and all
other ordinary classification objectives keep the original labels for every
image.

Only the rejection branch changes:

- support samples retain the ordinary rejection known-class CE and normal
  uncertainty calibration;
- pseudo-novel query samples are excluded from rejection known-class CE and
  receive a uniform rejection-logit target plus uncertainty target one;
- no unknown/test/mixed-pool labels enter episode construction or loss;
- the episode loss is disabled by default with
  `--alpha-rejection-episode-pseudonovel 0`.

The initial fixed protocol uses a `0.25` holdout fraction of batch-present
classes, uncertainty-loss weight `1.0`, and treatment coefficient
`alpha_rejection_episode_pseudonovel=0.1`. These values are fixed before the
first semantic-hard screening run; the fraction and coefficient are not to be
scanned on that split.

Episode class sampling uses a separate CPU RNG stream. In the training entry,
the stream is seeded as `seed + 1000003 * epoch` so treatment sampling does not
consume the data augmentation/dropout RNG stream and paired control/treatment
runs retain the same stochastic data path apart from the enabled loss.

## Correctness gates before formal training

The implementation must satisfy the following checks:

1. the held-out mask contains complete known classes and leaves at least one
   support class whenever a batch contains at least two classes;
2. the main classification loss still receives every original known label;
3. pseudo-novel rows receive no rejection known-class CE gradient;
4. pseudo-novel rows receive nonzero uniform-logit/uncertainty gradients;
5. the episode loss is finite, appears in the total loss and is reported in
   training history;
6. the default parsed configuration is numerically identical in behavior to
   the historical path.

The explicit-generator reproducibility check is covered in
`tests/test_core_behaviors.py`; the one-epoch training test also checks the
nonzero episode gradient audit field.

The first five checks are covered by the loss and one-epoch tests in
`tests/test_core_behaviors.py`. The training history fields are
`rejection_episode_pseudonovel` and
`rejection_episode_pseudonovel_fraction`.

## Screening comparison

The first screening run must use a new semantic-hard CIFAR-100 60/40 split
and a split-matched teacher. Control and treatment share the student seed,
teacher, initialization, optimizer, epoch budget, standard logits KD, the
shared-backbone independent rejection encoder, mixed-pool NNPU, checkpoint
selection, detector, known-validation-only 95% coverage calibration and
Auto-K/clustering procedure. The only training change is the episode loss
coefficient `0` versus the fixed treatment value chosen before the run.

The treatment may proceed to confirmation splits only if it shows a positive
change in at least one ranking or fixed-coverage rejection metric and at least
one clustering metric, while known utility and candidate purity do not cross
the existing screening degradation limits. A single positive split is not
enough to change the default.

Every completed arm must report AUROC, AUPR, FPR95, OSCR, known accuracy,
known acceptance, unknown rejection, accepted-known accuracy, candidate
count/purity/unknown recall, Auto-K error, and Cluster ACC/NMI/ARI. The
episode loss, held-out fraction, configuration diff, label-use audit and
rejection-branch gradient audit must be included.

No formal run is authorized by this document until the smoke gates pass and a
fresh output directory and experiment-log entry are prepared. Test labels may
be used only after completion for final retrospective metrics.
