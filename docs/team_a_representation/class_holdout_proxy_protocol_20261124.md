# Class-holdout support proxy pilot

## Why this stage exists

The prior KD variants produced real student gradients and sometimes improved
known-class geometry, but those changes did not reliably transfer to unknown
rejection and novel-class clustering. This pilot adds a known-only proxy for
the missing support-boundary objective. It is deliberately separate from the
rejector, score, candidate-selection, and clustering implementations.

## Candidate mechanism

For each labelled student batch, randomly hold out a fixed fraction of the
classes as pseudo-novel classes. Build detached cosine prototypes from the
remaining known classes. For every held-out-class feature, penalize similarity
to the nearest remaining support prototype above a fixed margin:

```text
L_holdout = mean ReLU(max_c cosine(h_i, center_c) - m)
```

The held-out classes are still supervised by the ordinary hard-label CE and
standard logits KD terms. The proxy only adds a shared student-feature loss;
it does not create rejector labels, use discovery-pool identities, or read
unknown/test labels. Teacher parameters and teacher logits remain detached.

The pilot fixes `holdout_fraction=0.25`, `margin=0.2`, and
`alpha_class_holdout_proxy=0.1`. No coefficient, fraction, margin, score,
threshold, or clustering scan is allowed in this stage.

## Fixed pilot protocol

Use semantic-hard CIFAR-100 60/40 seed `2173` with three paired arms:

1. `standard`: CE + standard logits KD;
2. `class_holdout`: the same objective plus the class-holdout support proxy;
3. `uncertainty`: CE + existing teacher-uncertainty-weighted logits KD.

The arms share the split-matched teacher, student initialization, data
protocol, four-epoch budget, known-validation checkpoint selection,
max-softmax evaluation, known-validation 95% coverage calibration, matched
discovery pool, Auto-K, and clustering. Auxiliary feature KD, rejection
training, and discovery training remain disabled. Unknown/test labels are used
only for final retrospective metrics.

## Gates before interpretation

The implementation gate requires shape and parameter validation, finite loss,
nonzero student-feature gradient, and a zero loss when fewer than three batch
classes are available. The runner must verify that the treatment coefficient
and fixed proxy geometry are recorded and that only the treatment arm differs.

The pilot is valid only if every class-holdout epoch has a positive proxy loss,
positive held-out fraction, positive audited encoder gradient, and positive
student update. If these gates fail, repair the implementation before
interpreting performance.

## Promotion and stopping rule

This is a screening pilot, not evidence for a default change. Continue to a
fresh confirmation only if the proxy improves a known-only representation
diagnostic and at least one downstream detection or clustering metric without
a material known-accuracy or candidate-purity loss. A failure to transfer the
representation change to unknown rejection and clustering freezes this
hypothesis without scanning its parameters. Standard-KD remains the clean
baseline and uncertainty-KD remains a separate comparison.
