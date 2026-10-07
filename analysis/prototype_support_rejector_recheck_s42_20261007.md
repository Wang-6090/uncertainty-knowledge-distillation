# Prototype similarity rejector recheck (2026-10-07)

## Purpose

The current best post-hoc detector uses a support-only nnPU rejector. The code
also exposes classifier-prototype cosine similarities as optional rejector
features. This matched test asks whether those similarities add information
beyond support geometry, without changing the student representation.

## Fixed protocol

- CIFAR-100 semantic-isolated 60/40 split, seed 42.
- Same pretrained ResNet-18 student checkpoint:
  `runs/semantic_isolated_full_e10_s42_treatment/student.pt`.
- Same 5,400-sample mixed discovery pool, known prior 0.2.
- Same linear `nu_corrected` nnPU rejector, stratified 5,000 known support
  samples, MC=4, and validation-only 95% known-coverage calibration.
- The only changed factor is the rejector feature input.

## Results

| Rejector input | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: |
| `support_augmented` | 0.80078 | 0.55183 | 0.46295 | 94.18% | 23.65% |
| `prototype_support_augmented` | 0.80002 | 0.54900 | 0.46332 | 94.57% | 22.33% |

Prototype features improve FPR95 by only 0.00283 and OSCR by 0.00037,
while AUROC decreases and unknown rejection decreases by 1.32 percentage
points. This is a small score/operating-point fluctuation, not evidence that
classifier-prototype similarities reduce the known/unknown representation
overlap.

## Decision

Keep `prototype_augmented` and `prototype_support_augmented` as diagnostic
options, but do not use them in the main detector and do not tune their feature
weights. The next work should modify training-time representation learning or
the mixed-pool positive-unlabeled assumption; adding more post-hoc score
features is unlikely to solve the core problem.

The implementation and test suite remain valid. Both runs completed on CUDA
under identical data and calibration conditions.
