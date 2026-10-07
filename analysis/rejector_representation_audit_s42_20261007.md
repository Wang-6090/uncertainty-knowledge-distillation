# Rejector representation audit (seed 42, 2026-10-07)

## Purpose and protocol

The current core problem is overlap between known and unknown representations.
This audit tests two post-hoc rejector changes without retraining either
student: (1) append cosine similarity to every known classifier prototype;
and (2) replace the linear nnPU rejector with the existing small MLP nnPU
rejector. The baseline and uncertainty-treatment student checkpoints are the
same ten-epoch checkpoints used by the support-sampling audit.

All runs use the semantic-isolated CIFAR-100 60/40 split, the same mixed pool
with known prior 0.2, stratified 5000-sample known support, `nu_corrected`,
MC=4 extraction, validation-only 95% known-coverage calibration, and no
clustering. Only the rejector representation or rejector model family changes.

## Prototype-similarity representation

`prototype_support_augmented` appends the normalized feature-to-class-proxy
cosine similarities to the existing support-augmented rejector input.

| Student | Rejector | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | linear support-augmented | 0.7866 | 0.6027 | 0.3795 | 95.35% | 21.85% |
| baseline | prototype-support-augmented | 0.7832 | 0.6130 | 0.3771 | 95.68% | 20.00% |
| treatment | linear support-augmented | 0.8008 | 0.5518 | 0.4629 | 94.18% | 23.65% |
| treatment | prototype-support-augmented | 0.8000 | 0.5490 | 0.4633 | 94.57% | 22.33% |

The treatment has nearly unchanged ranking, but lower unknown rejection. The
baseline becomes worse on all ranking metrics. This is not a stable solution;
the new feature mode remains opt-in and is not used by the main protocol.

## Nonlinear nnPU rejector

The `mlp` option changes only the rejector function family. It is intended to
test whether a linear boundary is too restrictive, but it does not change the
student representation.

| Student | Rejector | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | linear support-augmented | 0.7866 | 0.6027 | 0.3795 | 95.35% | 21.85% |
| baseline | nnPU MLP | 0.5157 | 0.9157 | 0.2309 | 100.00% | 0.00% |
| treatment | linear support-augmented | 0.8008 | 0.5518 | 0.4629 | 94.18% | 23.65% |
| treatment | nnPU MLP | 0.5156 | 0.9160 | 0.2735 | 100.00% | 0.00% |

The nonlinear rejector collapses under the current nnPU risk, likely because
the mixed pool is only weakly constrained and the small unlabeled-positive
risk does not prevent a near-constant solution. Do not promote it or tune its
architecture before changing the risk objective and adding validation-side
diagnostics for score variance and gradient behavior.

## Code and verification

The new optional feature modes are `prototype_augmented` and
`prototype_support_augmented`. Prototype similarities are extracted with
detached tensors, and the support branch is explicitly initialized for the
new mode. The first smoke run caught both integration issues before any result
was recorded. After the fixes, compileall passed and the full test suite passed
with `177 passed, 2 warnings`.

## Decision

Keep the main detector unchanged: ten-epoch uncertainty treatment plus the
linear, class-stratified, support-augmented `nu_corrected` rejector. Stop
adding post-hoc feature columns or increasing rejector nonlinearity. The next
algorithmic experiment should change the training target for a dedicated
rejection representation or use a validated PU risk with explicit score
regularization, with one controlled factor at a time.
