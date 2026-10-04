# Hybrid unknown-feature boundary pilot (2026-09-30)

## Question

Does an explicit feature-space boundary objective help with the persistent
known/unknown overlap, beyond the current student objective and detector?

The working diagnosis is not that *all* feature-separation objectives have
failed. Earlier experiments tested different mechanisms (pseudo-unknown Energy,
VOS, reciprocal points, and prototype/batch-feature margins); they do not test
the same objective. The existing `unknown_feature_separation_loss` uses only
the current known batch as reference, while the existing prototype margin uses
classifier weights only.

## Change

Added the optional `--alpha-discovery-boundary` loss. For a protocol-defined
pure-unknown discovery pool, it penalizes unknown feature cosine similarity to
a hybrid known support reference:

1. global normalized classifier prototypes;
2. normalized centroids of known classes present in the current labeled batch.

For each unknown feature, a smooth maximum is calculated over the local
centroids; the global prototype maximum and local-centroid maximum are combined
with `--discovery-boundary-prototype-weight`, then penalized above
`--discovery-boundary-margin`. Known references are detached; gradients act on
unknown features. The two augmented views are scored separately and averaged.
The term is default-off and is rejected for `discovery-pool-mode=mixed` by the
same protocol guard as other full-pool unknown losses.

This is a project-specific hybrid objective motivated by the known-support
overlap diagnosis and by prototype/metric-learning and open-set feature
separation ideas. It is **not** a faithful reproduction of ARPL, VOS, or a
class-conditional density detector, and the pilot alone does not establish a
novel method.

## Code validation

- `python -m compileall -q novel_discovery train.py`: passed.
- `python -m pytest -q tests --disable-warnings --maxfail=1`: **98 passed**.
- `git diff --check`: passed.
- Toy student smoke: completed and saved a checkpoint. The recorded
  `discovery_boundary` loss was `0.25764` (nonzero); the existing two-view
  discovery loss was also nonzero. The new objective therefore reached the
  training loop and backpropagated. This is a functionality check, not an
  effectiveness result.

## Paired pilot protocol

- Dataset/split: CIFAR-100, random 60/40 split, seed 42.
- Same pretrained ResNet-34 teacher checkpoint and ResNet-18 student setup.
- Same 1,200 known training, 300 known validation, 1,000 open-test, and 1,200
  discovery samples; same 3 epochs, batch size 32, image size 64, optimizer and
  all other losses.
- Both runs explicitly enabled the same pure-unknown discovery pool. The
  baseline set `alpha_discovery_boundary=0`; the variant set it to `0.1`.
- Same `normalized_entropy_mahalanobis` score and known-validation-only
  threshold calibration targeting 95% known coverage. No test labels were used
  to choose the threshold or any hyperparameter.
- The first baseline invocation omitted the discovery-pool flag. It was **not**
  used for comparison; baseline was rerun with the exact same pool/configuration
  as the variant except for the loss weight.
- The `discover` command writes its own arguments to the same `config.json` in
  the work directory, replacing the training arguments. The comparison above
  therefore relies on the explicit commands used during the run plus
  `train_history.json` (which confirms zero vs nonzero boundary loss). Preserve
  a separate train configuration snapshot in future runs; this is a traceability
  weakness in the current CLI workflow, not a metric issue.

## Results

| Metric | Baseline | Hybrid boundary | Change |
| --- | ---: | ---: | ---: |
| AUROC | 0.5552 | 0.5618 | +0.0066 |
| AUPR | 0.4240 | 0.4534 | +0.0294 |
| FPR95 | 0.9041 | 0.8512 | -0.0529 |
| OSCR | 0.1533 | 0.1660 | +0.0127 |
| Known accuracy (open test) | 0.2149 | 0.2397 | +0.0248 |
| Test known acceptance (validation target 0.95) | 0.9438 | 0.9719 | +0.0281 |
| Test unknown rejection | 0.0734 | 0.0658 | -0.0076 |
| Candidate-pool purity | 0.4603 | 0.6047 | +0.1444 |
| ARI, all true unknowns (oracle diagnostic) | 0.0492 | 0.0665 | +0.0174 |
| ARI, true unknowns among detected candidates (oracle diagnostic) | 0.0713 | -0.0418 | -0.1131 |

Detection-score ranking and candidate purity show preliminary positive signals,
but the actual unknown rejection at the calibrated operating point went down,
and candidate-conditional clustering ARI worsened. Test known acceptance also
deviated from the 95% validation target, so the raw unknown-rejection figures
are not a perfectly equal-coverage test comparison. This one-seed, limited-data,
3-epoch result is mixed and is **not evidence that the core problem is solved**.

## Interpretation and next check

The loss was active, so this result is not due to an unwired option. It suggests
the hybrid support target may improve global ranking/candidate purity slightly,
but it did not consistently improve the rejection/clustering operating point.
Likely limitations include noisy minibatch centroids, a single global cosine
margin, and mismatch between the raw backbone feature geometry and the
Mahalanobis detector. Do not tune the margin on this test set.

Next, before extending the method, use the saved baseline/variant checkpoints
for score-distribution diagnostics on validation and test separately, and
compare known/unknown cosine-to-prototype and cosine-to-local-centroid
distributions. If the representation statistics do not move in the intended
direction, stop this branch. If they do, repeat the exact fixed configuration
on at least two more seeds; report validation-calibrated coverage and test
coverage separately. Keep the feature-boundary loss default-off until that
replication.

Artifacts are in `runs/boundary_pair_baseline_s42` and
`runs/boundary_pair_variant_s42`. No GitHub upload was performed.
