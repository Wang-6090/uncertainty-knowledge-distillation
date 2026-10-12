# Seed-43 encoder LR scale 0.05 pilot (2026-10-12)

## Why this comparison

Per-score diagnostics from the prior-aware Sinkhorn replication showed that
the final detector score (AUROC 0.7491) was close to kNN feature distance
(0.7463) and minimum-class kNN distance (0.7471), while the uncertainty head
was much weaker (AUROC 0.5745) and energy/ViM had no usable separation in the
saved outputs. This suggests the dominant limitation is representation
geometry, not a missing score-fusion formula.

Encoder LR scale 0.1 had been a stronger direction in separate matched
experiments. This pilot probes the nearby value 0.05, changing only that scale
relative to the seed-43 equal-mass `encoder_lr_scale=0.1` arm. Teacher, class
split, seed, 3-epoch student budget, known/discovery sample limits, all losses,
optimizer, equal-mass joint Sinkhorn, detector, and calibration protocol are
fixed. Test labels are used only for retrospective metrics. CUDA was used.

## Results

| Metric | LR scale 0.1 | LR scale 0.05 | Change |
|---|---:|---:|---:|
| AUROC | 0.7491 | 0.7303 | -0.0187 |
| AUPR | 0.6363 | 0.6195 | -0.0168 |
| FPR95 | 0.6963 | 0.6980 | +0.0017 |
| Known accuracy (all known) | 34.73% | 30.70% | -4.03 pp |
| Test known accept | 92.11% | 96.64% | +4.53 pp |
| Unknown reject | 27.97% | 16.09% | -11.88 pp |
| Score KS distance | 0.3789 | 0.3579 | -0.0210 |
| Unknown below known score q95 | 81.93% | 81.44% | -0.49 pp |
| Candidate purity | 70.63% | 76.47% | +5.84 pp |
| Candidate unknown-only ARI | 0.0455 | 0.0227 | -0.0228 |

At scale 0.05, the detector rejects fewer unknown samples despite accepting
more knowns. The threshold was calibrated using the same validation-only
95%-known-coverage policy, but the test operating point generalized
differently; do not interpret the increase in known acceptance as a
calibration success. Score ranking and distribution separation both decline,
and known classification accuracy falls. Candidate purity increases because
the rejection set shrinks, but candidate-only ARI also falls and the smaller
candidate set does not indicate better unknown discovery overall.

## Decision

The finer step from encoder LR scale 0.1 to 0.05 is not promising under this
pilot. It does not fix score overlap and harms multiple metrics. Keep 0.1 as
the better-supported working value, not a final optimum; do not continue a
blind LR-scale sweep. The residual ~81% unknown-below-known-q95 overlap and
near-equivalence of kNN score and fused score reinforce that future changes
should improve known/unknown feature geometry during learning, and should be
tested with multiple seeds and adequate training budgets.

## Artifacts

- Treatment: `runs/joint_encoder_lr_s43_lr005/` and
  `runs/joint_encoder_lr_s43_lr005_detect/`.
- Paired baseline: `runs/joint_pilot_s43_equal_repro/` and its matched,
  normalized-clustering diagnostic `runs/joint_pilot_s43_equal_repro_cluster_norm/`.
- Script: `scripts/run_joint_encoder_lr_s43_lr005.ps1`.
