# Candidate feature-separation recheck on semantic-isolated CIFAR-100

## Purpose

This paired test evaluated a candidate-gated feature-separation loss. The
question was whether pushing selected mixed-pool candidate features away from
the known representation would improve the frozen-feature unknown detector.

## Controlled protocol

- CIFAR-100 semantic-isolated 60/40 split, seed 42
- pretrained ResNet-34 teacher and ResNet-18 student
- full train/validation/test data, 3 student epochs
- identical teacher, optimizer, data order, mixed pool, and known prior 0.2
- identical support-augmented linear `nu_corrected` nnPU rejector
- validation-only 95% known-coverage threshold calibration
- clustering skipped

The only student-training change was enabling:

```text
alpha_discovery_feature_separation = 0.05
discovery_feature_separation_margin = 0.0
discovery_feature_separation_temperature = 0.1
discovery_feature_candidate_gating = true
```

The treatment log had a non-zero `discovery_feature_separation` term while
the baseline had zero, confirming that the intended branch was active.

## Results

| Student arm | AUROC | AUPR | FPR95 | OSCR | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.7665 | 0.6434 | 0.6630 | 0.3693 | 94.92% | 18.55% |
| candidate separation | 0.7617 | 0.6431 | 0.6497 | 0.3660 | 95.15% | 19.08% |

Changes relative to baseline were AUROC `-0.0048`, AUPR `-0.0004`, FPR95
`-0.0133`, OSCR `-0.0033`, known acceptance `+0.23pp`, and unknown rejection
`+0.53pp`.

## Decision

This is not sufficient evidence that candidate feature separation improves the
core problem. The small FPR95 and unknown-rejection gains are accompanied by
lower AUROC and OSCR, and the operating-point gain is only `0.53pp`. The
candidate gate remains an opt-in ablation and is not promoted to the main
training recipe.

The result supports the earlier audit conclusion: the limitation is not a
missing scalar threshold or one extra rejector feature. Candidate selection
from a mixed pool is still noisy, so a margin applied to those candidates does
not reliably create a separated representation. The next credible direction
is to change the representation target with cleaner positive/negative
structure or a dedicated open-set objective, while keeping the current nnPU
plus support-rejector protocol as the fixed reference.

Artifacts:

- baseline checkpoint: `runs/semantic_isolated_pilot_candidate_sep_s42_baseline/`
- treatment checkpoint: `runs/semantic_isolated_pilot_candidate_sep_s42_candidate_sep/`
- baseline report: `runs/semantic_isolated_pilot_candidate_sep_s42_baseline_rejector_support/discovery_report.json`
- treatment report: `runs/semantic_isolated_pilot_candidate_sep_s42_candidate_sep_rejector_support/discovery_report.json`
