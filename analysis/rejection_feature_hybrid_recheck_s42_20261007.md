# Rejection and uncertainty feature-mode recheck (2026-10-07)

## Purpose

The independent rejection branch plus uncertainty-PU experiment showed that
the choice of rejector input mattered: `rejection_embedding` was harmed by
the PU treatment, while the existing `uncertainty_augmented` features partly
recovered the ranking. This follow-up tested whether explicitly concatenating
the independent rejection embedding with the uncertainty summaries would
produce a better detector.

## Controlled protocol

- CIFAR-100 semantic-isolated 60/40 split, seed 42
- same full data, teacher, student checkpoints, matched 5,400-image pool, and
  known prior `0.2`
- same linear `nu_corrected` rejector, MC=4, validation-only 95%
  known-coverage calibration, and no clustering
- only `--rejector-feature-mode` changed

The tested modes were:

- `rejection_embedding`
- `uncertainty_augmented`
- new `rejection_uncertainty_augmented`
- new `rejection_support_uncertainty_augmented`

The new modes use the normalized independent rejection embedding as the base
and append classifier summaries, learned uncertainty, MC epistemic signals,
expected entropy, and aleatoric uncertainty. The support variant appends the
known-support score as well.

## Results

| Checkpoint | Feature mode | AUROC | AUPR | FPR95 | OSCR | Known acceptance | Unknown rejection |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| independent branch | `rejection_embedding` | 0.7627 | 0.6280 | 0.6327 | 0.3485 | 94.48% | 18.63% |
| independent branch | `uncertainty_augmented` | 0.7666 | 0.6375 | 0.6475 | 0.3545 | 95.20% | 17.63% |
| independent branch | `rejection_uncertainty_augmented` | 0.7658 | 0.6267 | 0.6150 | 0.3563 | 94.73% | 17.78% |
| independent branch | `rejection_support_uncertainty_augmented` | 0.7654 | 0.6262 | 0.6168 | 0.3582 | 94.28% | 19.25% |
| branch + uncertainty-PU | `rejection_embedding` | 0.7178 | 0.5803 | 0.7045 | 0.3141 | 93.83% | 15.33% |
| branch + uncertainty-PU | `uncertainty_augmented` | 0.7528 | 0.6314 | 0.6673 | 0.3361 | 93.98% | 20.63% |
| branch + uncertainty-PU | `rejection_uncertainty_augmented` | 0.7607 | 0.6323 | 0.6442 | 0.3405 | 93.92% | 21.48% |
| branch + uncertainty-PU | `rejection_support_uncertainty_augmented` | 0.7636 | 0.6342 | 0.6315 | 0.3441 | 93.73% | 21.95% |

## Decision

The new hybrid modes are valid and provide a small, reproducible post-hoc
improvement over `rejection_embedding`, especially for the PU checkpoint.
The support hybrid gives the highest unknown rejection in this pair, while
the non-support hybrid gives the lowest FPR95 for the baseline checkpoint.
However, no mode dominates all metrics, and the unknown rejection remains
below 22% at the fixed operating point. These modes therefore remain opt-in
detector ablations; they do not demonstrate that the student feature space
has been separated.

The feature-combination exploration is closed for now. Further gains should
come from a cleaner training target or a better open-set protocol, not from
adding more correlated post-hoc columns. The next audit should compare
feature-overlap statistics and class-conditional score distributions under
the best modes, then revisit training-side objectives only if they change the
representation rather than just the threshold score.

Artifacts:

- baseline hybrid: `runs/semantic_isolated_rejection_branch_s42_base_rej_unc_aug/`
- treatment hybrid: `runs/semantic_isolated_rejection_branch_s42_nnpu_rej_unc_aug/`
- baseline support hybrid: `runs/semantic_isolated_rejection_branch_s42_base_rej_support_unc_aug/`
- treatment support hybrid: `runs/semantic_isolated_rejection_branch_s42_nnpu_rej_support_unc_aug/`
