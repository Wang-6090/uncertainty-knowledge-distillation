# Rejection-branch gradient audit (2026-10-13)

## Purpose

The independent rejection branch had already been shown to receive non-zero
losses, but a non-zero scalar loss does not prove that the intended branch is
receiving a useful amount of gradient. This audit therefore records
loss-specific gradient norms on the rejection projector/classifier/uncertainty
head, separately from the shared encoder.

This is a mechanism and wiring audit, not an AUROC experiment. It does not
select a new default weight and it does not use test labels for training or
selection.

## Code correction

The previous diagnostic recorded only shared-encoder gradients for CE, KD,
SupCon, and selected discovery losses. It omitted several losses that reach
the independent rejection branch through its uncertainty head, including:

- known uncertainty alignment;
- mixed-pool uncertainty nnPU;
- external-outlier uncertainty supervision;
- uncertainty separation and uncertainty feature-margin terms.

`--audit-loss-gradients` now reports these terms under
`gradient_audit_grad_norm_rejection_*`, together with selected cosine
similarities between rejection-side objectives. The default training behavior
is unchanged when the flag is disabled.

## Verification

The implementation passed syntax checking and the core test suite (`195
passed, 2 warnings` for the targeted suite). A toy one-epoch run confirmed
that a non-zero rejection CE coefficient produces a non-zero
`gradient_audit_grad_norm_rejection_known_ce_term`.

A small CIFAR-100 smoke run used:

- semantic-isolated 60/40 split, seed 44;
- the saved seed-44 teacher and pretrained ResNet-18 student;
- 64 train samples, 32 validation samples, one epoch;
- rejection dimension 128;
- rejection known CE `1.0`;
- mixed-pool uncertainty nnPU `0.1`;
- external CIFAR-10 uncertainty `0.1`;
- external rejection feature margin `0.05`;
- `--audit-loss-gradients` enabled.

The smoke run was intentionally too small for a performance claim. Its
rejection-side gradient diagnostics were:

| Objective term | Rejection gradient norm |
| --- | ---: |
| known uncertainty alignment | 0.0588 |
| known rejection CE | 1.9179 |
| mixed-pool uncertainty PU | 0.0770 |
| external outlier uncertainty | 0.1227 |
| external rejection feature margin | 0.0346 |

The uncertainty-related terms had pairwise gradient cosine approximately
`0.904`, `0.953`, and `0.933`, so this smoke did not show a dominant direction
conflict among them. Instead, the known rejection CE update was roughly 25x
the mixed-pool PU update, 16x the external uncertainty update, and 55x the
feature-margin update.

## Interpretation and next experiment

This does not prove that the current weights are wrong: gradient norms vary
with batch, model stage, and loss scale. It does establish a concrete
hypothesis that was not previously measured: the rejection branch may be
over-optimized for known classification while its unknown supervision is too
weak to reshape the boundary.

The next controlled experiment should compare a predeclared rejection-loss
balance, for example:

1. the current `alpha_rejection_known_ce=1.0` control;
2. a lower known-CE coefficient with the same unknown-loss coefficients; and
3. only if the first comparison is promising, a gradient-normalized variant
   that keeps the unknown-side rejection gradient in a fixed range.

The training protocol, data split, teacher, seed, epoch budget, detector,
validation-only calibration, and checkpoint rule must remain fixed. Promotion
requires improvement in ranking/OSCR or score-overlap diagnostics together
with known utility; a larger unknown rejection rate caused by more known
false rejections is not sufficient.

