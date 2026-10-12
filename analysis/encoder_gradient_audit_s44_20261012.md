# Shared-encoder gradient audit for mixed-pool nnPU (seed 44, 2026-10-12)

## Question

The mixed-pool uncertainty nnPU term had shown a small, seed-sensitive change
in the final operating point. Before changing its coefficient or routing, this
audit checked whether the term actually supplies a substantial and coherent
gradient to the shared student encoder.

This is a mechanism audit, not a detection-performance comparison.

## Controlled protocol

- CIFAR-100 semantic-isolated 60/40 split, seed 44, CIFAR transform.
- Pretrained ResNet-34 teacher and pretrained ResNet-18 student.
- Full training data, batch size 64, five student epochs, CUDA.
- Matched mixed discovery pool of 5,400 samples; known prior 0.2.
- Audited `nu_corrected` mixed-pool uncertainty nnPU, coefficient 0.1.
- Existing CE, uncertainty KD, feature KD, prototype and SupCon losses kept
  unchanged.
- Only `--audit-loss-gradients` was added. It calls `autograd.grad` for the
  raw per-loss objectives before the ordinary combined backward pass, so it
  does not change the optimizer update.

## Results

Values below are epoch averages over 338 batches. Norms are for the raw loss;
the effective nnPU norm is `0.1 * raw nnPU norm` because the training
coefficient is 0.1.

| epoch | CE norm | KD norm | SupCon norm | nnPU raw norm | nnPU effective norm | effective nnPU / CE | cos(CE, nnPU) | cos(KD, nnPU) | cos(SupCon, nnPU) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.8289 | 8.5832 | 10.7239 | 1.7659 | 0.1766 | 1.80% | -0.0068 | 0.0025 | 0.0106 |
| 2 | 8.1097 | 6.2273 | 10.5547 | 3.8440 | 0.3844 | 4.74% | -0.0046 | 0.0023 | 0.0062 |
| 3 | 7.2380 | 5.4682 | 10.5484 | 3.8091 | 0.3809 | 5.26% | -0.0017 | -0.0014 | 0.0029 |
| 4 | 6.6409 | 5.0226 | 10.8721 | 3.7258 | 0.3726 | 5.61% | 0.0020 | -0.0034 | 0.0063 |
| 5 | 6.2604 | 4.8000 | 11.2437 | 3.6593 | 0.3659 | 5.85% | 0.0028 | -0.0051 | 0.0015 |

The training run completed normally and selected epoch 4 by known validation
accuracy (0.5839). That classification value is only a run-health signal; no
test detection claim is made from this audit.

## Interpretation

1. The nnPU term is connected to the shared encoder, but its effective
   gradient is small compared with CE/KD/SupCon under the current coefficient.
2. Its batch-averaged direction is nearly orthogonal to the main objectives,
   not consistently strongly opposed to them. Thus the evidence does not
   support “nnPU is failing because of severe gradient conflict.”
3. The result is consistent with a different bottleneck: the uncertainty head
   is learning a weak or task-misaligned signal, while the main detector uses
   entropy plus feature-neighborhood geometry. Updating the head alone cannot
   force those detector features to separate.
4. This does **not** prove that a larger nnPU coefficient is useless. It only
   says that blindly increasing it is not justified, especially because old
   weight-sweep runs used different data limits, teachers and rejectors and are
   not a matched comparison.

## Code validation

- Added regression tests for normal gradients, unconnected parameters, zero
  gradients and empty parameter lists.
- Fixed the diagnostic implementation to insert zero blocks for unused
  parameters; otherwise different losses could produce different vector
  lengths and cosine computation could fail.
- Targeted tests: `216 passed`.
- Full tests: `225 passed, 2 warnings`.
- `python -m compileall -q train.py novel_discovery tests`: passed.
- `git diff --check`: passed.

## Next controlled direction

Do not make the diagnostic flag or encoder-gradient routing the default yet.
The next algorithmic experiment should act directly on the shared feature
geometry with one clearly isolated objective, while keeping the same split,
teacher, student initialization, pool, epochs, detector and validation-only
calibration. It must compare AUROC, AUPR, FPR95, OSCR, known accuracy,
known acceptance, unknown rejection, KS/overlap and feature-distance
distributions together. A rise in unknown rejection alone is insufficient.

Relevant existing options should be tested one at a time and only if their
loss is nonzero in the logs; previously ineffective feature-margin, OE,
boundary and rejector variants must not be silently combined into a new
“improvement.”

Artifact: `runs/gradient_audit_cifar_s44_20261012_5ep/`.
