# Uncertainty-KD score-alignment audit

## Question

The existing uncertainty-KD training changes detached teacher weights, while
the completed paired pilots evaluate every arm with `max_softmax`. This audit
checks whether the uncertainty-KD student behaves differently under existing
deployable score channels, without changing the trained representation.

## Fixed protocol

Use the completed semantic-hard CIFAR-100 60/40 seed `2173` checkpoints from
the class-holdout pilot:

- standard-KD student as the clean reference;
- uncertainty-KD student as the existing comparison.

For each frozen checkpoint, evaluate exactly these existing score modes:

1. `max_softmax`;
2. `entropy_only` (the repository's explicit entropy-only mode);
3. `energy`;
4. `head_uncertainty`.

All source discovery settings, split, CIFAR transform, matched mixed pool,
known-validation calibration target (95%), MC sample count, and student
checkpoint remain fixed. Clustering is skipped in this diagnostic because the
question is score-channel alignment; the official paired pilot already
contains the full clustering result under `max_softmax`. No test labels are
used for score, threshold, or channel selection. The audit changes neither
teacher/student weights nor the training objective.

## Interpretation rule

Report every channel for both students. Do not select a channel using test
labels. A channel is interesting only if it improves the known-validation
calibration-compatible detection behavior without a material known-accuracy
loss, and if the result is interpreted as a frozen score diagnostic rather
than evidence that uncertainty-KD itself is improved.

If `head_uncertainty` is not competitive, the next training hypothesis should
not add another uncertainty signal. If it is competitive, the next stage may
test a pre-registered training objective that aligns uncertainty-KD with that
existing score, with standard-KD and uncertainty-KD retained as clean arms.

## Output

The runner writes separate discovery reports and an analysis JSON under the
`team_a_score_alignment_20261125_*` prefix. Existing runs are never
overwritten.
