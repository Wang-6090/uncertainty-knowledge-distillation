# Strict seed-44 and nonlinear nnPU audit (2026-10-07)

## Purpose

This audit checks whether the recent uncertainty-augmented feature rejector
result is reproducible and whether a nonlinear nnPU boundary improves the
known/unknown overlap problem. Test labels were used only for final metrics.
Thresholds were calibrated on known validation data with 95% known coverage.

## Protocol correction

An initial seed-44 student command omitted `--pretrained` and did not
explicitly match the teacher backbone. That run is not comparable with seeds
42/43 and is retained only as a configuration-audit artifact.

The valid rerun used CIFAR-100 semantic-hard 60/40, pretrained ResNet-34 and
ResNet-18, full known training data, 10 student epochs, a matched 5,400-image
mixed pool with known prior 0.2, MC=8, and known-validation coverage 95%.

## Strict seed-44 detection

| Seed | AUROC | FPR95 | Known acceptance | Unknown rejection |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.6910 | 0.7743 | 94.85% | 15.60% |
| 43 | 0.6937 | 0.7670 | 95.20% | 13.63% |
| 44, valid pretrained rerun | 0.6593 | 0.7937 | 94.97% | 10.18% |

Seed 44 is directionally better than the historical baseline, but weaker than
seeds 42/43 and has lower best validation accuracy (0.3867 versus
0.5153/0.5183). The method is therefore promising but seed-sensitive, not a
stable final claim.

Artifacts:

- `runs/semantic_hard_current_nnpu_margin_s44_pretrained/`
- `runs/semantic_hard_current_nnpu_margin_s44_pretrained_feature_rejector_uncertainty_augmented/`

## Oracle and automatic K

On the valid seed-44 checkpoint, the same candidate pool gave:

| K policy | Estimated K | Candidate purity | Unknown-only NMI | Unknown-only ARI |
| --- | ---: | ---: | ---: | ---: |
| Oracle K | 40 | 57.40% | 0.5074 | 0.1267 |
| Automatic silhouette K | 36 | 57.40% | 0.5034 | 0.1279 |

Automatic K uses only candidate-pool geometry. The true K is written only as a
post-hoc diagnostic (`cluster_k_abs_error`), so it is not used for selection.
The similar oracle/automatic results show that K selection is not the main
source of the current failure; candidate contamination and weak representation
are more important.

## Nonlinear nnPU rejector pilot

Only the rejector model changed from the valid seed-42 linear nnPU baseline.
The checkpoint, mixed pool, prior, feature mode, MC count, calibration rule,
and test protocol were fixed.

| Rejector | AUROC | FPR95 | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: |
| Linear nnPU | 0.6910 | 0.7743 | 94.85% | 15.60% |
| MLP nnPU | 0.5268 | 0.9010 | 100.00% | 0.00% |

The MLP nnPU collapsed to an almost non-rejecting score range under the
current non-negative risk optimization. This failed variant is not made the
default and will not be rescued by test-set threshold tuning. A future
nonlinear version would need a separately validated objective, calibration, and
collapse diagnostics.

## Decision

Keep the linear uncertainty-augmented nnPU rejector as an optional candidate,
but do not claim that it solves the representation-overlap problem. The next
high-value work is to improve and stabilize the student representation, record
validation accuracy and seed variance, and then retest the fixed linear
rejector. Automatic clustering can remain enabled for deployment-style
results, while oracle K is reported only as a clustering upper bound.

## Verification

- `python -m pytest -q`: 170 passed, 2 warnings.
- The invalid non-pretrained seed-44 run was not used in any comparison.
