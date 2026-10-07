# Original uncertainty feature-margin: new-seed paired replication

## Question and predeclared comparison

Does the original uncertainty-weighted feature-margin retain its effect on a
new training random seed, beyond seeds 42, 123, and 3407? The hypothesis was
that adding the loss improves unknown ranking/rejection while reducing
known/unknown feature overlap.

This is a *training-seed replication on the existing random CIFAR-100 60/40
class split*, not a new class split. Seed 2026 was used for the teacher and
both student arms. All full data were used, with 64px input, batch size 64,
ResNet-34 teacher, pretrained ResNet-18 student, five epochs, the same
optimizer/loss settings, and a mixed unlabeled discovery pool. The measured
known proportion was `0.212598`; both arms used immediate nnPU with weight
`0.1`. Treatment changed one factor only: uncertainty feature-margin weight
`0.05`, cosine margin `0.2`, uncertainty source, power `1`.

Detection for both used normalized entropy + min-class kNN (`k=10`, feature
bank from all known training data), MC=8, the full 10,000-image open test set,
and a threshold calibrated from known validation samples only for 95% target
known coverage. Clustering was skipped. Test labels were used only for final
metrics, not training, model selection, or threshold fitting.

## Results

| Seed 2026 | AUROC | AUPR | FPR95 | OSCR | Known accuracy (all known) | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nnPU baseline | 0.700089 | 0.573936 | 0.740500 | 0.437740 | 0.521667 | 95.55% | 12.70% |
| + uncertainty feature-margin | 0.703333 | 0.582211 | 0.731167 | 0.429724 | 0.510333 | 95.75% | 13.875% |
| Treatment minus baseline | +0.003243 | +0.008275 | -0.009333 | -0.008016 | -0.011333 | +0.20pp | +1.175pp |

The treatment weakly improves ranking and fixed-coverage unknown rejection,
but worsens OSCR and known classification accuracy. Since the ranking gain is
small and utility metrics disagree, this seed is mixed evidence—not a robust
win. It does not show that the overlap problem is solved. Full-data geometry
is reported below and is mixed rather than consistently favorable.

### Full-data feature geometry

Geometry was evaluated on all 27,000 known training samples and 10,000
open-test samples; test labels only stratified descriptive diagnostics.

| Distance reference | Baseline distance AUROC / histogram overlap | Treatment distance AUROC / histogram overlap |
| --- | ---: | ---: |
| Classifier prototype | 0.6807 / 0.7211 | 0.6723 / 0.7402 |
| Empirical class centroid | 0.6731 / 0.7464 | 0.6740 / 0.7544 |
| Nearest known training sample | 0.6974 / 0.6974 | 0.7071 / 0.6979 |

This seed does **not** support a general representation-separation effect:
prototype geometry worsened, centroid AUROC improved trivially while its
overlap worsened, and nearest-sample AUROC improved with essentially
unchanged overlap. This matches the mixed detection metrics.

Appending this paired result to the previous three-seed results gives rough
four-seed mean deltas of AUROC `+0.0123`, FPR95 `-0.0209`, OSCR `+0.0068`,
unknown rejection `+0.78pp`, and known classification accuracy `+0.30pp`.
These summaries are descriptive only: four training seeds on one class split
do not establish generalization to new class splits.

## Execution validity

An initial teacher invocation omitted `--backbone resnet34` and thus trained
the default ResNet-18. Loading it into the required ResNet-34 teacher slot
failed strictly before the student optimizer started. That attempt is
excluded. A correctly configured ResNet-34 teacher was trained and shared by
both valid student arms. Configs and checkpoints are under
`runs/uncertainty_margin_replication_s2026_*`.

## Decision

- Retain original uncertainty feature-margin as a promising but unconfirmed
  optional candidate; do not make it the default based on this single new
  seed.
- Do not pursue the tested power-sharpening or cross-view-min gate variants;
  both were negative in `analysis/uncertainty_margin_gate_pilots_20261005.md`.
- Geometry is now measured for seed 2026 and is mixed, not a consistent
  improvement. Pause further gate/exponent tuning and test a separate class
  split with a predeclared protocol before promoting this loss.
- Keep test labels out of any threshold calibration or model selection.
