# nnPU score-margin audit (seed 42, 2026-10-07)

## Purpose and controlled factors

The linear support-only nnPU rejector remains the strongest current detector,
but the mixed-pool objective can in principle have a weakly separated or
near-constant score solution. A small regularizer was added to penalize a
violation of

`mean(unlabeled_score) - mean(known_score) >= margin`.

The regularizer is opt-in and has no effect at its default weight of zero. The
test used `score_margin_weight=0.05` and `score_margin=0.5`. Student
checkpoints, semantic-isolated split, mixed pool, support sampling, linear
`nu_corrected`, detector, and validation-only 95% known-coverage calibration
were fixed. Only the rejector objective changed.

## Results

| Student | Rejector | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | original nnPU | 0.7866 | 0.6027 | 0.3795 | 95.35% | 21.85% |
| baseline | + score margin | 0.7816 | 0.6165 | 0.3780 | 95.27% | 21.03% |
| treatment | original nnPU | 0.8008 | 0.5518 | 0.4629 | 94.18% | 23.65% |
| treatment | + score margin | 0.7988 | 0.5553 | 0.4634 | 94.20% | 23.00% |

The regularizer does not improve ranking or unknown rejection. The small OSCR
increase for the treatment is not sufficient because the main unknown-rejection
metric decreases and AUROC/FPR95 also worsen. This is not evidence that the
student representation improved.

## Decision

Keep the option for ablation and diagnostics, but leave its default disabled
and stop tuning this scalar separation term. The mixed-pool risk needs a more
principled class-conditional or representation-level treatment; a global mean
score gap does not solve the known/unknown overlap.

The code and focused tests remain valid. Final regression verification passed:
`178 passed, 2 warnings`, together with compileall and `git diff --check`.
