# Known-label classification view-consistency protocol

## Scope

This candidate belongs to Team A's student classification representation and
KD work. It addresses the frozen-checkpoint finding that augmentation
instability is concentrated on known-validation samples that are hard for the
student. It does not change the teacher, uncertainty head target, rejection
encoder, rejection score, candidate selection, calibration, Auto-K, or
clustering algorithm.

## Objective

For two independently augmented views of the same known labelled training
sample, let `f_s^a` and `f_s^b` be the shared student classification backbone
features. The opt-in term is

```text
L_view = mean(1 - cosine(f_s^a, f_s^b))
L_total = L_existing + alpha_view * L_view
```

The term uses no teacher output and no unknown/test label. CE and the selected
logits-KD mode are computed on the first view exactly as before. Teacher
parameters remain frozen. The default coefficient is zero.

## Pairing requirement

`--paired-classification-views` makes the known training loader return two
independent existing CIFAR train augmentations and is required when
`--alpha-classification-view-consistency > 0`. It should be enabled for both
control and treatment arms in a paired pilot so both consume the same two-view
augmentation stream. With coefficient zero, the second forward is retained
only to keep the paired control path aligned and the consistency term contributes
zero gradient. Without the pairing flag, historical one-view training remains
unchanged.

## Implementation gate

The implementation must keep the default off and pass:

1. finite loss, shape validation, and zero-batch behavior;
2. student-feature gradient with no teacher dependency;
3. paired loader metadata/label alignment;
4. nonzero coefficient-scaled consistency gradient in the student epoch audit;
5. coefficient-zero inactive statistics and unchanged one-view path;
6. full repository tests and compilation checks.

These checks are implemented in
`tests/test_classification_view_consistency.py`. The current focused result is
`5 passed`.

## Research pilot gate

Only after the implementation gate, run one fresh semantic-hard CIFAR-100
60/40 pilot comparing `standard` and `classification_view_consistency`, with
an `uncertainty` arm retained as the existing Team A comparison. All arms must
share teacher, student initialization, paired augmentation stream, optimizer,
four-epoch budget, known-validation checkpoint selection, max-softmax score,
known-validation calibration, candidate pool, Auto-K, and clustering. The
coefficient is fixed before the run and is not scanned. Continue to a fresh
confirmation only if a known-only stability/geometry diagnostic and at least
one ranking or candidate/clustering metric improve without material known
utility or candidate-purity loss.
