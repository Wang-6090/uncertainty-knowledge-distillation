# Training-budget audit: 5 versus 10 epochs (seed 42, 2026-10-07)

## Purpose and controlled factors

The recent full-data support-rejector experiments used five epochs, while
known classification accuracy was still low. This audit tests whether the
apparent known/unknown overlap is partly caused by an under-trained student.

Only the training budget changed from the existing seed-42 protocol: five
student epochs versus ten student epochs. The teacher was also trained with
the same selected epoch budget within each run. Architecture, pretrained
initialization, semantic-isolated CIFAR-100 60/40 split, mixed discovery pool
(known prior 0.2), uncertainty-margin treatment, stratified 5000-sample known
support, `nu_corrected` support-only nnPU rejector, MC setting, test set, and
validation-only 95% known-coverage calibration were fixed. Clustering was
skipped.

## Results

| Budget / arm | AUROC | FPR95 | OSCR | Known accuracy (all) | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 ep baseline | 0.7541 | 0.6722 | 0.3293 | 39.07% | 95.22% | 18.90% |
| 5 ep treatment | 0.7642 | 0.6485 | 0.3796 | 45.18% | 94.90% | 19.08% |
| 10 ep baseline | 0.7866 | 0.6027 | 0.3795 | 43.63% | 95.35% | 21.85% |
| 10 ep treatment | **0.8008** | **0.5518** | **0.4629** | **52.45%** | 94.18% | **23.65%** |

## Interpretation

Increasing the budget from five to ten epochs improves both arms. For the
baseline, AUROC increases by `+0.0325`, FPR95 falls by `-0.0695`, OSCR rises
by `+0.0502`, and unknown rejection rises by `+2.95pp`. For the treatment,
the corresponding changes are `+0.0365`, `-0.0967`, `+0.0834`, and
`+4.57pp`. This confirms that the earlier five-epoch protocol was partly
under-trained and should not be the only basis for judging the method.

At the matched ten-epoch budget, the treatment still improves AUROC by
`+0.0142`, FPR95 by `-0.0508`, OSCR by `+0.0834`, and unknown rejection by
`+1.80pp` over its baseline. The treatment also raises all-known accuracy
from `43.63%` to `52.45%`, at the cost of `1.17pp` known acceptance.

This is meaningful progress, but not a solution to the core problem:
`76.35%` of unknown test samples are still accepted by the ten-epoch
treatment at the fixed operating point. More epochs alone should not be
presented as resolving feature overlap. The next representation experiment
should use ten epochs and the stratified support protocol as the fixed
baseline, and should change one explicit boundary objective at a time.

Artifacts:

- `runs/semantic_isolated_full_e10_s42_baseline/`
- `runs/semantic_isolated_full_e10_s42_treatment/`
- `runs/semantic_isolated_full_e10_s42_baseline_rejector_support/`
- `runs/semantic_isolated_full_e10_s42_treatment_rejector_support/`
