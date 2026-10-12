# Matched open-validation checkpoint-selection experiment (seed 44)

## Purpose

This experiment tests whether selecting the student checkpoint by open-
validation AUROC improves the final unknown-detection result over the
historical known-validation-accuracy rule. It is an evaluation-protocol test,
not a new representation-learning loss.

## Controlled protocol

- CIFAR-100 semantic-hard 60/40 split, seed 44
- ResNet-34 teacher and ResNet-18 student
- 64px images, full known training partition, validation limit 300,
  discovery limit 1200, test limit 1000
- two student epochs, `open_val_ratio=0.2`
- same initialization, data protocol, losses, optimizer and detector
- `max_softmax` was used for both checkpoint selection and final discovery
- threshold was calibrated from known validation samples for 95% known
  coverage; clustering was skipped
- final test labels were not used for checkpoint selection

The open-AUROC run selected epoch 1. The known-accuracy run selected epoch 2.
The first comparison initially used different final detectors (`max_softmax`
versus kNN); that comparison was discarded. The table below is the corrected
matched comparison.

## Independent test results

| selection rule | selected epoch | AUROC | AUPR | FPR95 | OSCR | known accuracy (all known) | known accept | unknown reject | KS | unknown below known q95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| open-val AUROC | 1 | 0.5351 | 0.4515 | 0.9350 | 0.1267 | 0.1829 | 0.9504 | 0.0530 | 0.0847 | 93.98% |
| known-val accuracy | 2 | 0.5447 | 0.4426 | 0.8752 | 0.2344 | 0.3128 | 0.9368 | 0.0723 | 0.1003 | 93.73% |

The known-accuracy checkpoint is better on AUROC, FPR95, OSCR, known
classification accuracy, unknown rejection, KS and the score-overlap
diagnostic. Open-AUROC selection is therefore not supported by this matched
test. The difference is based on one seed and two epochs, so it is not a
general theorem; it is sufficient to stop treating open-AUROC selection as a
core-algorithm fix.

## Interpretation

Checkpoint selection changes which already-learned representation is used; it
does not directly make known and unknown embeddings separable. The matched
result provides no evidence that model-selection changes solve the central
overlap problem. Keep `known_acc` as the default and retain open-set selection
only for controlled diagnostics after freezing the final detector and guardrail
metrics.

Artifacts:

- `runs/checkpoint_openauroc_s44_msp_discover/discovery_report.json`
- `runs/checkpoint_knownacc_s44_msp_discover/discovery_report.json`
- `runs/checkpoint_openauroc_s44/student.pt`
- `runs/checkpoint_knownacc_s44_fix/student.pt`

