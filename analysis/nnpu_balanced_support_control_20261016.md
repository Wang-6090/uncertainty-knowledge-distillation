# Balanced logit block control for the full-support nnPU rejector (2026-10-16)

## Motivation and matched factors

The default full-support rejector concatenates the embedding, a 60-class logit
block, uncertainty summaries and `known_support_score`. Because the logit block
has many coordinates, its scale or dimensionality could dominate the linear
nnPU fit. The treatment used the existing
`rejection_balanced_support_augmented` mode, which normalizes the logit block
per sample before the common standardization step.

Only that feature construction changed. Both seeds used all known support
(`rejector_max_samples=0`), stratified-known sampling, `nu_corrected` linear
nnPU, the same student checkpoint, mixed pool, MC=4, and validation-only 95%
known-coverage calibration.

## Results

| seed | feature mode | AUROC | FPR95 | OSCR | known accept | unknown reject | KS | unknown below known q95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 44 | support_augmented | 0.90342 | 0.3590 | 0.66667 | 95.37% | 49.48% | 0.6512 | 48.40% |
| 44 | balanced support | **0.90863** | **0.3318** | **0.67036** | **96.37%** | 42.40% | **0.6685** | 49.75% |
| 123 | support_augmented | 0.90315 | 0.3812 | 0.64405 | 95.72% | 50.05% | 0.6571 | 45.95% |
| 123 | balanced support | **0.90582** | **0.3488** | **0.64791** | 94.90% | **52.03%** | **0.6636** | 48.13% |

## Decision

AUROC, FPR95, OSCR and KS improve on both seeds, which makes this a useful
candidate for further study. However, the overlap proxy worsens on both seeds,
seed 44 loses fixed-working-point unknown rejection, and seed 123 loses about
0.8 percentage points of known acceptance. Therefore it is not promoted to the
default feature mode yet. It remains an explicit ablation and a possible
component for a later validation-first combination with the full-support
configuration.

The result is evidence that feature geometry and scale matter, but it does not
show that the shared encoder has become fully separable. Further work should
use a pre-declared validation selection rule and at least one additional seed,
not choose this mode from the test metrics above.
