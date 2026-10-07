# Full-data support-rejector audit (2026-10-07)

## Purpose

The semantic-isolated full-data paired run showed that the treatment changes
the feature geometry, but the fixed 95% known-coverage operating point only
improves unknown rejection slightly. This audit checks whether the post-hoc
nnPU rejector is being limited by the random support subsample rather than by
the learned representation.

All runs use CIFAR-100 semantic-isolated 60/40 classes, seed 42, image size
64, ResNet-18 student checkpoints, the same matched mixed discovery pool of
5400 samples with known prior 0.2, `nu_corrected` nnPU, support-augmented
features, full train/validation/test data, and a threshold calibrated only on
known validation data for 95% known coverage. Clustering is skipped because
this experiment isolates detection.

## Paired results

| checkpoint | known support sampling | max samples | AUROC | FPR95 | OSCR | known acceptance | unknown rejection |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | random | 5000 | 0.7541 | 0.6722 | 0.3293 | 95.22% | 18.90% |
| baseline | random | 20000 | 0.7464 | 0.6413 | 0.3224 | 94.25% | 16.95% |
| baseline | stratified by known class | 5000 | **0.7677** | 0.6547 | **0.3364** | **95.17%** | **19.70%** |
| treatment | random | 5000 | 0.7642 | 0.6485 | 0.3796 | 94.90% | 19.08% |
| treatment | random | 20000 | 0.7697 | **0.6120** | 0.3854 | 94.73% | **20.05%** |
| treatment | stratified by known class | 5000 | **0.7705** | **0.6070** | 0.3804 | 94.65% | 19.83% |

## Interpretation

Increasing the support limit alone is not a stable solution: it improves the
treatment run but makes the baseline unknown rejection worse. The better
controlled result is class-stratified support sampling. It improves both
baseline and treatment relative to their random-5000 counterparts, so the
effect is not attributable only to the treatment loss. The improvement is
still modest, and unknown rejection remains about 20%; this is evidence of a
more stable rejector, not a solution to known/unknown feature overlap.

The current default should remain unchanged until this sampling choice is
replicated on another seed. The option is exposed as
`--rejector-stratified-known`, with `--rejector-max-samples` controlling the
support budget. For a controlled follow-up, use stratified sampling at 5000
and keep every other setting fixed.

## Related negative controls

The class-conditional empirical conformal support score was also tested on the
same pilot protocol. It produced AUROC `0.5632` for baseline and `0.5295` for
treatment, far below the support-augmented nnPU rejector. It is retained only
as an optional calibration baseline; further tuning of that score is not a
priority.
