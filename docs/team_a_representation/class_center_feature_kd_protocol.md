# Class center feature KD protocol

## Scope

This candidate belongs to the student classification representation and
uncertainty KD work. It transfers labelled class-center directions from the
frozen teacher to the student. It does not change the rejection encoder,
rejection score, candidate selection, threshold rule, or clustering method.

## Candidate mechanism

For every known class represented in a training batch, compute the mean of the
student and detached teacher backbone features and minimize the cosine distance
between the two class means:

```text
c_s(y) = mean(f_s[x] : label(x) = y)
c_t(y) = mean(f_t[x] : label(x) = y)
L_class_center = mean_y(1 - cosine(c_s(y), c_t(y)))
```

Unlike ordinary feature KD, this does not force sample-level teacher
within-class variation onto the student. Unlike the previously frozen
class-centered feature KD, it targets between-class center directions. It uses
known training labels only; teacher features and teacher centers are detached.
The pilot coefficient is fixed at `alpha_class_center_feat_kd=0.1` with no
weight scan.

## Staged execution

1. Pass shape, finite-loss, teacher-detachment, student-gradient, default-off,
   and configuration tests.
2. Run one fresh semantic-hard seed with three paired arms:
   `standard` (CE + standard logits KD), `class_center_feature` (the same plus
   class-center feature KD), and `uncertainty` (CE + uncertainty-KD).
3. Keep CE/KD-only training, max-softmax evaluation, known-validation
   calibration, automatic K, and all training budgets fixed. Do not use unknown
   labels, discovery-pool identity, student uncertainty, or final rejection
   scores in the new loss.
4. Require nonzero class-center loss, active labelled rows, and nonzero audited
   encoder gradient in every epoch. Continue to a fresh confirmation only if
   known-only representation diagnostics and at least one downstream metric
   improve without material known utility loss.

The candidate is exploratory and opt-in. It must not alter the default KD
configuration without multi-seed evidence.
