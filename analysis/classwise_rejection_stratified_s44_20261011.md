# Classwise rejection audit for stratified support (seed 44)

This is a retrospective error analysis of two detector runs using the same
student checkpoint and the same data protocol. Test labels are used only to
group errors after inference; they are not used for training, threshold
calibration, model selection, or feature design.

## Controlled comparison

- Dataset: CIFAR-100 semantic-isolated 60/40.
- Student checkpoint: `runs/audit_rejection_branch_s44/student.pt`.
- Discovery pool: matched mixed pool, 5,400 samples, known prior 0.2.
- Rejector: linear `nu_corrected` nnPU, MC=4.
- Detector: `score_mode=feature_rejector`.
- Threshold: validation-only 95% known coverage.
- Only changed factor: `rejector_stratified_known`.

| Setting | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
|---|---:|---:|---:|---:|---:|
| uniform known support | 0.7681 | 0.6137 | 0.3859 | 94.22% | 22.03% |
| stratified known support | 0.7700 | 0.6115 | 0.3837 | 94.58% | 20.25% |

## Observed error pattern

The uniform-support run has high unknown false-acceptance across many novel
classes, not just one semantic-near class. The ten highest false-acceptance
rates are 0.97, 0.92, 0.91, 0.91, 0.91, 0.89, 0.89, 0.88, 0.88, and 0.87.
The class-stratified variant changes classes inconsistently: some classes
improve, while others become easier to accept. The largest improvements are
raw classes 31 (-10 pp), 60 (-8 pp), and 49 (-7 pp); the largest regressions
are raw classes 48 (+14 pp), 75 (+12 pp), and 86/44 (+8 pp).

## Interpretation

Known-support balancing does not create a stable boundary between known and
unknown samples. It slightly changes the rejector's sampling bias, but the
feature distributions remain substantially overlapped. This supports the
decision to keep the option only as an ablation and to move the next effort
back into training-time representation learning and mixed-pool supervision.

The generated machine-readable report is
`analysis/classwise_rejection_stratified_s44_20261011.json`.
