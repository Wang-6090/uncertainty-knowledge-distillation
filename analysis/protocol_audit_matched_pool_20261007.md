# Matched discovery-pool protocol audit (2026-10-07)

## Purpose

This audit checks whether the recent detector comparison actually used the
same unlabeled discovery-pool protocol across seeds. It does not change the
model or detector. Test labels are used only for final metrics.

## Finding before the rerun

The saved detection configurations were not identical:

| Run | `matched_discovery_pool` | Test pool used to fit rejector | Actual known fraction |
| --- | ---: | ---: | ---: |
| seed 42 | false | 25,400 samples | about 0.2126 |
| seed 43 | false | 25,400 samples | about 0.2126 |
| seed 44 | true | 5,400 samples | 0.2000 |

The earlier report described all three as a matched 5,400-sample protocol.
That description was inaccurate. The old seed-44 comparison is therefore
kept as a historical audit result, but not treated as a strict replication of
seeds 42/43.

The seed-44 student-training history also has
`alpha_discovery_uncertainty_pu=0` and
`alpha_discovery_uncertainty_feature_margin=0`; the corresponding loss meters
are zero. It was not a valid training-time treatment run. Its post-hoc
rejector result must not be described as evidence that those two losses were
trained successfully.

## Corrected detection-only comparison

The same saved seed-42 and seed-43 treatment checkpoints were evaluated with
the same matched pool construction: semantic-hard CIFAR-100 60/40, pool size
5,400, known fraction 0.2, linear `nu_corrected` nnPU rejector on uncertainty-
augmented frozen features, MC=8, and a 95% known-validation threshold.

| Seed | AUROC | FPR95 | Known acceptance | Unknown rejection |
| ---: | ---: | ---: | ---: | ---: |
| 42, matched | 0.6899 | 0.7558 | 95.10% | 14.58% |
| 43, matched | 0.6985 | 0.7587 | 95.38% | 14.45% |
| Earlier seed 42, unmatched | 0.6910 | 0.7743 | 94.85% | 15.60% |
| Earlier seed 43, unmatched | 0.6937 | 0.7670 | 95.20% | 13.63% |

The corrected pool changes the ranking only slightly and does not remove the
known/unknown overlap. Therefore pool composition is an important reporting
control, but it is not the main cause of the low unknown rejection rate.

## Code change

`discover` now records the actual discovery-pool size and known fraction in
`calibration_report.json` and the saved run configuration. For nnPU runs it
also warns when the requested prior differs from the measured pool fraction by
more than 0.02. `inspect_data` already prints the same protocol statistics.

## Decision before the corrected third seed

Keep the linear uncertainty-augmented nnPU rejector as a candidate, but do not
claim that it solves the representation-overlap problem. At that point, the
required next check was a genuinely matched training-time treatment on seed
44, followed by the same detector evaluation. The result of that check is
recorded below.

## Verification

- `python -m pytest -q`: 170 passed, 2 warnings.
- `python -m py_compile train.py novel_discovery/data.py novel_discovery/pipeline.py novel_discovery/losses.py`: passed.
- Matched pool inspection: 5,400 samples and known fraction 0.2.
- Detection runs: `runs/audit_protocol_matched_s42/` and
  `runs/audit_protocol_matched_s43/`.

## Corrected seed-44 training-time treatment

The seed-44 student was then retrained with the treatment actually enabled:
pretrained ResNet-34/ResNet-18, semantic-hard 60/40, full known data, 10
epochs, matched 5,400-sample pool, `alpha_discovery_uncertainty_pu=0.1`,
`alpha_discovery_uncertainty_feature_margin=0.05`, `nu_corrected` risk, and the
same distillation settings as the earlier treatment. The training loss meters
were non-zero and the best checkpoint was epoch 8 with known validation
accuracy 0.4990.

The fixed matched detector evaluation produced:

| Seed | AUROC | FPR95 | Known acceptance | Unknown rejection |
| ---: | ---: | ---: | ---: | ---: |
| 42, treatment | 0.6899 | 0.7558 | 95.10% | 14.58% |
| 43, treatment | 0.6985 | 0.7587 | 95.38% | 14.45% |
| 44, corrected treatment | 0.6822 | 0.7880 | 94.98% | 13.08% |
| Mean | 0.6902 | 0.7675 | 95.15% | 14.03% |

The corrected third seed is directionally consistent but weaker. This is the
first valid three-seed treatment table; it supports retaining the method as a
promising candidate, not claiming that it solves unknown detection.

Artifacts: `runs/audit_protocol_matched_s44_treatment/` and
`runs/semantic_hard_current_nnpu_margin_s44_matched/`.
