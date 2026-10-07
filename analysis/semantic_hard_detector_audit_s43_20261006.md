# Semantic-hard detector audit (2026-10-06)

## Protocol

This is an independent-protocol audit of the detector layer. It reuses the
existing `semantic_hard_disjoint_baseline_s43/student.pt` checkpoint and keeps
the semantic-hard class split, 300 known calibration samples, 1,000 open test
samples, 27,000-vector kNN bank, MC=4, and the 95% known-coverage calibration
settings fixed. The only changed factor is the detector: Student baseline,
support-only rejector, or fixed 0.75-weight fusion. No test labels were used
to fit or select a detector.

This checkpoint is an older 5-epoch baseline with a limited test protocol, so
the result is a generalization audit, not a final comparison with the current
full-data random-split nnPU+margin runs.

## Results

| Detector | AUROC | FPR95 | Saved known accept | Saved unknown rejection | Known-class accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Student baseline | 0.5877 | 0.8618 | 93.33% | 8.31% | 40.65% |
| Support-only rejector | 0.6510 | 0.8179 | 92.52% | 14.29% | 40.65% |
| Fusion, rejector weight 0.75 | 0.6536 | 0.8065 | 93.01% | 11.69% | 40.65% |

The fixed fusion improves ranking slightly over support-only (`+0.0026`
AUROC, `-0.0114` FPR95), but loses `2.60` percentage points of unknown
rejection at the saved operating point. The support-only rejector is the
better working-point detector in this protocol. The three detectors share the
same classifier predictions, so ordinary known-class accuracy is unchanged;
the difference is only which samples are accepted.

## Interpretation

The semantic-hard baseline is substantially weaker than the recent random
60/40 runs. This confirms that the class split and validation/test protocol
have a large effect on the apparent unknown-detection result. It also confirms
that the rejector can transfer some ranking improvement across a split, but
neither rejector nor fusion removes the feature-overlap problem. Fusion must
remain optional and must not be selected by this test result.

The next representation experiment should use a current full-data student
trained on the semantic-hard split with the same nnPU + uncertainty-margin
protocol, followed by a disjoint calibration subset. Only then can we decide
whether the current training method or the detector is responsible for the
cross-split drop.

Artifacts:

- `runs/audit_semhard_base_s43/`
- `runs/audit_semhard_rejector_s43/`
- `runs/audit_semhard_fusion_s43/`
- `analysis/semantic_hard_detector_audit_s43_20261006.json`
