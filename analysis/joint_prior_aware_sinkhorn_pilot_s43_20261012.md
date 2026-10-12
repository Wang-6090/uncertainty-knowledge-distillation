# Prior-aware unified Sinkhorn replication: seed 43 (2026-10-12)

## Question and pre-registered comparison

The seed-44 pilot suggested a modest score-ranking improvement from assigning
20% total Sinkhorn mass to the 60 known logits and 80% to the 40 novel
prototypes, but candidate purity and candidate-only ARI declined. This paired
seed-43 experiment asks whether that detector signal replicates, and whether
candidate quality changes in the same direction.

The sole treatment factor is `--joint-prior-aware-assignments`. Both arms use
the same seed-43 pretrained ResNet-34 teacher, semantic-isolated CIFAR-100
60/40 split, seed, student settings, mixed discovery pool (size 1200, known
fraction 0.2), 3-epoch budget, loss coefficients, optimizer, pretrained
ResNet-18, detection score, kNN setting, MC samples, and validation-only
95%-known-coverage calibration. Clustering is an oracle-K retrospective
diagnostic (K=40) after the same known-coverage threshold; test labels do not
select the detector or training pseudo-labels.

This is a second-seed replication, not a full-scale benchmark: student runs
use 1200/300/1000 train/validation/test limits and the shared seed-43 teacher
was trained for 5 epochs on the full known training set. CUDA was available
and selected (`device: cuda`).

## Results

| Metric | Equal mass | Prior-aware 20/80 | Difference (treatment - baseline) |
|---|---:|---:|---:|
| AUROC | 0.7491 | 0.7519 | +0.0028 |
| AUPR | 0.6363 | 0.6408 | +0.0045 |
| FPR95 | 0.6963 | 0.6795 | -0.0168 |
| OSCR | 0.3061 | 0.3045 | -0.0015 |
| Known accuracy (all known) | 34.73% | 34.23% | -0.50 pp |
| Test known accept rate | 92.11% | 92.45% | +0.34 pp |
| Unknown reject rate | 27.97% | 24.50% | -3.47 pp |
| Score KS distance | 0.3789 | 0.3737 | -0.0053 |
| Unknown below known score q95 | 81.93% | 81.44% | -0.49 pp |
| Candidate count | 160 | 144 | -16 |
| Candidate purity | 70.63% | 68.75% | -1.88 pp |
| Candidate unknown-only ARI | 0.0455 | 0.0450 | -0.0005 |
| All-unknown ARI | 0.0848 | 0.0966 | +0.0118 |

At the chosen threshold, known false rejections were 47/596 and 45/596;
unknown correct rejections were 113/404 and 99/404 respectively. The
calibration subset reached 94.03% known acceptance in both arms, while test
known acceptance was lower than the nominal 95% target; this is calibration
generalization error, not evidence that the threshold was tuned on test data.

## Interpretation

This does **not** replicate the useful part of seed 44. AUROC/AUPR move only
slightly upward, while OSCR, KS separation, unknown rejection, candidate
purity, and candidate-only ARI are flat or worse. FPR95 improves modestly but
is not supported by consistent movement in the other separation metrics.
The data therefore do not support enabling prior-aware Sinkhorn by default
or describing it as a robust improvement. Its higher candidate-all-unknown
ARI is a secondary retrospective metric and does not offset lower candidate
purity/recall at the operating point.

Across the two pilots the direction is inconsistent: seed 44 improved several
detector metrics but harmed candidate purity/ARI; seed 43 barely changed
ranking, reduced unknown rejection, and again reduced candidate purity. The
strong persistent signal is still score/feature overlap: about 81% of unknown
scores lie below the known q95 in both seed-43 arms. A different allocation
prior is not the core fix.

## Audit and artifacts

- Student checkpoints/configs/history:
  `runs/joint_pilot_s43_equal_repro/` and
  `runs/joint_pilot_s43_prioraware_repro/`.
- Detection-only outputs (used for operating-point comparisons):
  `runs/joint_pilot_s43_equal_repro_detect/` and
  `runs/joint_pilot_s43_prioraware_repro_detect/`.
- Oracle-K cluster diagnostics:
  `runs/joint_pilot_s43_equal_repro_cluster/` and
  `runs/joint_pilot_s43_prioraware_repro_cluster/`.
- Reproduction script: `scripts/run_joint_prior_aware_pilot_s43.ps1`.
- Regression suite before the experiment: `215 passed, 2 warnings`.

Both student configs confirm the same seed, data protocol, and controlled
training inputs; among algorithm settings only
`joint_prior_aware_assignments` differs. Checkpoint paths/work directories and
outcome-derived fields (`best_val_known_acc`, prototype inertia/refresh
history) differ as expected because training trajectories differ. The
detector configs are paired and the evaluated test split is identical. A prior command omitted
clustering to isolate the detector; the subsequent oracle-K runs used the
same checkpoints and detector settings and produced the candidate diagnostics
reported above.

## Decision / next step

Retain prior-aware Sinkhorn as an opt-in research ablation, not the default
recipe. Do not stack it with another loss based on these results. Prioritize
the stronger training-time direction already supported by a separate
disjoint-calibration replication: preserving the pretrained encoder with
`encoder_lr_scale=0.1`, then investigate why even that leaves substantial
known/unknown overlap. For the joint-discovery branch, the next useful study
should separate known rejection from novel assignment explicitly and evaluate
candidate contamination and clustering independently rather than changing
group priors again.
