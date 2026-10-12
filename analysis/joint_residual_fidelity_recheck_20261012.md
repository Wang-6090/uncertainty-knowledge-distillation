# Joint mixed-residual fidelity and replication audit (2026-10-12)

## Motivation and implementation correction

The joint residual idea was worth a controlled recheck because earlier
negative runs used a weaker/mismatched recipe (2 epochs, encoder LR scale
1.0), while the newer seed-43 baseline used 3 epochs and scale 0.1. The
proposal uses known-class confidence only to down-weight novel assignment on
mixed-pool samples; Sinkhorn itself then operates on novel prototypes rather
than making known and novel outputs compete in one balanced assignment.

Before the matched rerun, code inspection found a method-fidelity defect:
`known_residual_weights` documented a detached sample weight, but returned a
differentiable tensor. Consequently, loss gradients could manipulate known
logits through the weight itself. The helper now detaches its output and a
regression test asserts that it has no gradient path. Historical residual
results produced before this correction are not evidence for the documented
detached-weight algorithm and must be treated as legacy exploratory results.

## Controlled protocol

- CIFAR-100 semantic-isolated 60/40; seed 43 and 44 paired replications.
- Per seed, same pretrained ResNet-34 teacher, ResNet-18 student, 3 epochs,
  1200/300/1000 train/validation/test limits, mixed pool size 1200 and known
  fraction 0.2, encoder LR scale 0.1, loss weights, optimizer, augmentations,
  and detector.
- Treatment: enable `--joint-mixed-residual` with temperature 1 and floor 0.
- Detection: normalized entropy + minimum-class kNN, k=10, MC=4; validation
  calibration targets 95% known coverage; test labels are retrospective only.
- Candidate metrics use the same oracle-K=40 diagnostic in both arms. Oracle
  K is not used to train or select the detector.

For seed 43, the equal-mass baseline is the paired equal arm from the prior-aware
experiment; both configs were checked and only the residual switch differs
among algorithm settings. For seed 44, the baseline is the matched equal-mass
arm from the prior-aware pilot. A separate residual-floor follow-up changes
only the floor from 0 to 0.1 relative to the corrected seed-43 residual run.

## Results

### Detached residual, equal-mass baseline

| Seed | Variant | AUROC | FPR95 | OSCR | Known acc (all) | Test known accept | Unknown reject | KS | Unknown below known q95 | Candidate purity | Candidate unknown ARI |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 43 | Equal mass | 0.7491 | 0.6963 | 0.3061 | 34.73% | 92.11% | 27.97% | 0.3789 | 81.93% | 70.63%* | 0.0455* |
| 43 | Residual, **detached** | 0.7527 | 0.6913 | 0.3080 | 34.90% | 93.12% | 25.00% | 0.3775 | 81.93% | 71.13% | 0.0313 |
| 44 | Equal mass | 0.6953 | 0.8105 | 0.2434 | 28.34% | 98.41% | 6.45% | 0.3249 | 85.75% | 70.59% | 0.1825 |
| 44 | Residual, **legacy gradient path** | 0.7188 | 0.7468 | 0.2748 | 31.85% | 95.86% | 14.25% | 0.3310 | 82.53% | 67.09% | -0.0006 |
| 44 | Residual, **detached** | 0.7208 | 0.7420 | 0.2693 | 31.05% | 95.38% | 15.32% | 0.3322 | 82.80% | 66.28% | -0.0290 |
| 44 | Residual + view agreement | 0.7203 | 0.7452 | 0.2764 | 32.01% | 96.18% | 13.98% | 0.3370 | 82.80% | 68.42% | 0.0001 |

`*` Seed-43 equal-arm clustering diagnostics were run with the same checkpoint,
detector, and oracle-K protocol after the detector-only run.

The seed-43 corrected result is modest: AUROC/FPR95/OSCR improve slightly,
known accuracy/acceptance move up slightly, but fixed-threshold unknown
rejection falls by 2.97 percentage points and candidate unknown ARI falls.
The residual floor=0.1 variant yields AUROC 0.7502, FPR95 0.6980, OSCR
0.3006, known accuracy 34.06%, test known acceptance 93.62%, unknown rejection
22.28%, candidate purity 70.31%, and candidate unknown ARI 0.0480. The small
KS increase to 0.3844 does not compensate for the worse ranking, OSCR, and
unknown rejection.

The corrected seed-44 run has a stronger AUROC/AUPR and FPR95 than its equal
baseline, but lower OSCR, known accuracy, known acceptance, candidate purity,
and candidate ARI. This suggests residual weighting may improve ranking while
making the selected candidate set less trustworthy. Its view-agreement
variant raises purity from 66.28% to 68.42% and OSCR from 0.2693 to 0.2764
relative to detached residual alone, but lowers AUROC slightly, raises FPR95,
and reduces unknown rejection from 15.32% to 13.98%. It does not solve the
tradeoff. The corrected method now has two seeds, but their operating-point
and clustering tradeoffs are substantial; evidence remains mixed, not a
robust overall win. The legacy pre-detach seed-44 result is still not
validation of the corrected method.

## Decision

Keep `--joint-mixed-residual` experimental and off by default. It has not
shown a robust joint gain across ranking, fixed-coverage operation, and
candidate clustering. The floor=0.1 tweak does not fix the tradeoff. Retire
the hypothesis that a confidence-derived scalar weight alone solves the
overlap problem unless a corrected second-seed run validates it.

The key self-audit lesson is to verify the actual gradient graph, not only
the formula and configuration. A weight described as a detached selection
signal must not receive gradients. Existing README statements about the
mechanism now refer to the corrected implementation; earlier run artifacts
remain as historical records and should be clearly labelled.

## Artifacts and checks

- Corrected seed 43: `runs/joint_residual_s43_detached/` and
  `runs/joint_residual_s43_detached_detect/`.
- Corrected seed 44: `runs/joint_residual_s44_detached/` and
  `runs/joint_residual_s44_detached_detect/`.
- Seed-44 view-agreement variant: `runs/joint_residual_s44_viewagree/` and
  `runs/joint_residual_s44_viewagree_detect/`.
- Legacy, pre-detach seed 43: `runs/joint_residual_s43_matched/` and
  `runs/joint_residual_s43_matched_detect/`.
- Floor 0.1 follow-up: `runs/joint_residual_s43_floor01/` and
  `runs/joint_residual_s43_floor01_detect/`.
- Legacy, pre-detach seed 44: `runs/joint_residual_s44_repro/` and
  `runs/joint_residual_s44_repro_detect/`.
- Seed-43 equal baseline: `runs/joint_pilot_s43_equal_repro/` and its
  clustering diagnostic `runs/joint_pilot_s43_equal_repro_cluster_norm/`.
- Seed-44 equal baseline: `runs/joint_pilot_s44_baseline/` and
  `runs/joint_pilot_s44_baseline_discover/`.
- Scripts: `scripts/run_joint_residual_s43_detached_recheck.ps1`,
  `scripts/run_joint_residual_s43_matched.ps1`,
  `scripts/run_joint_residual_floor_s43_01.ps1`, and
  `scripts/run_joint_residual_s44_replication.ps1`,
  `scripts/run_joint_residual_s44_detached.ps1`, and
  `scripts/run_joint_residual_view_agreement_s44.ps1`.
- Regression suite before the experiments: `215 passed, 2 warnings`; focused
  joint-discovery tests after the detach correction: `29 passed`.
