# Semantic-hard prototype-boundary rejector: seed-2028 replication

## Predeclared question and comparison

Does the exploratory prototype-boundary weighted rejector produce a practically
useful effect on a fresh training seed, beyond the earlier seed-2026/2027
replications? The training and detector recipe was fixed before this run:

- CIFAR-100 semantic-hard 60/40 split; teacher 5 epochs, student 4 epochs;
- one teacher/student pair and one frozen student checkpoint shared by both arms;
- mixed discovery pool, 5,400 examples, known prior 0.2;
- feature rejector with NNPU and support-augmented features;
- known-validation-only calibration targeting 95% known coverage;
- control: all-pool rejector fitting; treatment: prototype-boundary weighted
  pool, candidate ratio 0.25 and weight 2.0.

The sole intended detector factor is rejector pool treatment. Test labels were
not used for training, calibration, or choosing parameters. This remains
exploratory evidence: the method was prioritized after earlier exploratory
test diagnostics, and this is the same class split rather than an independent
split.

## Configuration audit

The saved `discover_config.json` files were compared. Both arms use seed 2028,
the identical student checkpoint, mixed-pool mode/size/prior, score mode,
NNPU training, rejector feature mode, and threshold policy/target. The intended
differences are exactly `rejector_pool_selection=all` versus
`weighted_boundary`, and the treatment's candidate rule `prototype_boundary`
(the all-pool control records `entropy_uncertainty`, which is inactive when the
whole pool is selected); ratio and weight are 0.25 and 2.0. Calibration-set
known acceptance is 0.9493865 in both arms.

## Results

| Metric | All-pool | Prototype-boundary weighted | Difference |
|---|---:|---:|---:|
| AUROC | 0.775236 | 0.783284 | +0.008048 |
| AUPR | 0.681005 | 0.678510 | -0.002495 |
| FPR95 | 0.658833 | 0.631000 | -0.027833 |
| OSCR | 0.566233 | 0.578813 | +0.012581 |
| Known acceptance | 95.917% | 95.533% | -0.383 pp |
| Unknown rejection | 22.800% | 22.375% | -0.425 pp |
| Accepted-known accuracy | 70.078% | 70.307% | +0.229 pp |
| KS distance | 0.412417 | 0.425000 | +0.012583 |
| Unknown below known score q95 | 74.150% | 76.225% | +2.075 pp (worse) |

The fixed-coverage unknown-rejection operating point did not improve. The
ranking metrics moved favorably, but the overlap diagnostic moved unfavorably;
therefore this is not evidence of reduced known/unknown overlap. In particular,
do not translate ranking gains into a claim that more unknowns are correctly
rejected at the deployment operating point.

## Three-seed decision

Across seeds 2026, 2027, and 2028, AUROC/FPR95/OSCR differences all have the
same favorable direction, but validation-calibrated unknown-rejection changes
are `+2.60`, `+0.625`, and `-0.425` percentage points. The third seed reverses
the operating-point effect, and its `unknown below known q95` overlap worsens.
The aggregate evidence supports at most a small ranking signal for this
post-hoc rejector pool treatment, not a reliable operating-point or
representation improvement. Keep the default unchanged and stop parameter
sweeps on this test set. The three seeds share one semantic-hard class split;
the direction also has a historical test-diagnostic selection path, so they
are not independent confirmatory evidence.

## Next direction

Return to training-time class-conditional representation support, not more
post-hoc rejector tuning. A next experiment should be one-factor and paired
against the current baseline. Define support regions from known training
labels only; choose boundary examples with a predeclared train/validation-only
proxy; use unlabeled mixed-pool samples conservatively and label-free, with
selection-rate/contamination sensitivity explicitly audited. Do not use test
labels or retrospective classwise results to select examples, loss weights,
or thresholds. Evaluate ranking, score overlap, unknown rejection at
validation-only 95% known coverage, and accepted-known accuracy together.
Keep representation training and detector post-processing as separate factors.

## Artifacts

- Runner: `scripts/run_semantic_hard_rejector_seed_recheck.py`
- Summary: `analysis/semantic_hard_rejector_seed2028_summary.json`
- Runs: `runs/semantic_hard_rejector_seed2028_{teacher,student,all_pool,prototype_boundary_weighted}/`
