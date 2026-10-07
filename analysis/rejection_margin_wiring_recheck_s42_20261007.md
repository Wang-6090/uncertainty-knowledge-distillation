# Rejection-feature margin wiring recheck (seed 42, 2026-10-07)

## Why this audit was necessary

The independent rejection branch had two separate options:

- `alpha_discovery_rejection_feature_margin`: push candidate discovery
  embeddings away from rejection-branch class prototypes;
- `alpha_discovery_rejection_feature_separation`: push them away from the
  observed known rejection embeddings.

Code inspection found that the first option only initialized its tensors. Its
actual loss calculation was incorrectly nested inside the second option. Thus
an experiment that enabled only the rejection-feature margin silently trained
without that margin. Earlier margin-specific rows must therefore be treated as
historical, not as valid evidence about the margin objective.

The implementation was corrected so that the two losses are independent. For
mixed pools, both use detached uncertainty as a soft candidate weight; for a
pure unknown pool, all discovery samples receive unit weight. A regression
test now verifies that the margin meter is non-zero when separation is off.

## Verification before the pilot

- `python -m compileall -q train.py novel_discovery tests`: passed.
- `python -m pytest -q`: `179 passed, 2 warnings`.
- `git diff --check`: passed.
- Control training log: rejection margin meter exactly `0`.
- Treatment training log: rejection margin meter non-zero (`0.001285` in
  epoch 1), confirming that the corrected path is active.

## Controlled pilot

Only the rejection-feature margin target changed. Both arms used semantic-hard
CIFAR-100 60/40, seed 42, pretrained ResNet-34/18, 1,200/300/1,000 train,
validation, and test images, a 400-image mixed pool with known prior 0.2,
two epochs, rejection embedding dimension 128, the same teacher checkpoint,
the same uncertainty-PU and uncertainty-feature-margin terms, a linear
`nu_corrected` nnPU rejector on `rejection_embedding`, and validation-only
95% known-coverage calibration. Clustering was skipped.

| Arm | Rejection target cosine | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | disabled | 0.5893 | 0.8923 | 0.1388 | 96.07% | 4.34% |
| Corrected margin | 0.2 | 0.5610 | 0.9590 | 0.1286 | 93.33% | 7.71% |
| Corrected margin | 0.0 | 0.5208 | 0.9521 | 0.1062 | 97.61% | 6.27% |

The margin groups reject more unknown samples than the control at the selected
operating point, but they also reject known samples and lose ranking quality.
The `0.0` target is especially poor in AUROC and OSCR. This is not a valid
solution to the representation-overlap problem.

## Decision

The code fix is kept because it makes the option truthful and testable. The
rejection-feature margin is not promoted to the main pipeline and should not
be tuned further under the current mixed-pool soft-weighting setup. The
earlier rejection-branch margin claims are downgraded to unverified historical
explorations; the corrected pilot is the first valid attribution.

The main protocol remains the ten-epoch uncertainty treatment with a linear
support-augmented `nu_corrected` rejector. The next algorithmic direction
should change the training target or PU formulation in a way that is not a
global prototype hinge. In particular, audit the class-conditional known prior
and candidate contamination before adding another rejection head.

Artifacts:

- `runs/rejection_pilot_margin_fix_control/`
- `runs/rejection_pilot_margin_fix_treatment/`
- `runs/rejection_pilot_margin_fix_zero/`
- `runs/rejection_pilot_margin_fix_control_detect/`
- `runs/rejection_pilot_margin_fix_treatment_detect/`
- `runs/rejection_pilot_margin_fix_zero_detect/`
