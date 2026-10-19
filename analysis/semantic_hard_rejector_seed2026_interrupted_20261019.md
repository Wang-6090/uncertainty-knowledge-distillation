# semantic-hard prototype-boundary rejector replications (seeds 2026 and 2027)

## Planned question and comparison

The experiment was intended to check whether the exploratory prototype-boundary
weighted rejector result persists on a newly trained random seed. One semantic-
hard CIFAR-100 teacher/student pair (seed 2026; teacher 5 epochs, student 4
epochs) would be trained once, then reused for two discovery arms:

1. all-pool feature rejector;
2. weighted-boundary feature rejector using prototype-boundary candidates,
   candidate ratio 0.25, candidate weight 2.0.

The split, student checkpoint, mixed discovery pool, known prior, score family,
and validation-only 95% known-coverage calibration were to be held fixed. The
test set was reserved for final reporting, not training, calibration, or method
selection. Metrics were to include AUROC, FPR95, OSCR, validation-calibrated
unknown rejection, known acceptance/classification utility, and score overlap.

This was explicitly exploratory, not confirmatory: prototype-boundary was
prioritized after earlier exploratory candidate-pool analysis that included
retrospective test-label diagnostics. Seed 2026 also retains the same fixed
semantic-hard class split, so it could test stochastic-seed repeatability but
not generalization to a new class split.

## Execution and configuration audit

An earlier run was interrupted during teacher training and produced no
metrics. The experiment was subsequently restarted from scratch and completed:
5 teacher epochs, 4 student epochs, then both discovery arms using the same
student checkpoint. The saved configs confirm identical seed (2026), split,
mixed pool size (5400), known prior (0.2), score mode, NNPU rejector, support-
augmented features, threshold policy, and target known coverage. Only the
rejector pool treatment differs. Test labels were not used for training,
calibration, or method selection within this run. The candidate rule had,
however, been prioritized using earlier exploratory diagnostics, so this
remains an exploratory replication rather than an untouched confirmatory test.

## Results

| Metric | all-pool | prototype-boundary weighted | Difference (weighted - all) |
|---|---:|---:|---:|
| AUROC | 0.7843265 | 0.7940588 | +0.0097323 |
| AUPR | 0.6801916 | 0.6995362 | +0.0193447 |
| FPR95 | 0.6378333 | 0.5998333 | -0.0380000 |
| OSCR | 0.5874302 | 0.5931198 | +0.0056895 |
| Known acceptance rate | 0.9528333 | 0.9518333 | -0.0010000 |
| Unknown rejection rate | 0.2360000 | 0.2620000 | +0.0260000 |
| Known class accuracy after acceptance | 0.7157600 | 0.7166871 | +0.0009271 |
| KS distance | 0.4313333 | 0.4353333 | +0.0040000 |
| Unknown below known score q95 | 0.7535000 | 0.7305000 | -0.0230000 |
| Unknown above calibrated threshold | 0.2360000 | 0.2620000 | +0.0260000 |

Calibration-set known acceptance was 0.9493865 in both arms (the target is
95%). On the held-out test set, known acceptance was approximately 95% in both
arms. The unknown-rejection increase therefore did not come with a large
increase in known rejection; accepted-known classification accuracy was also
nearly unchanged. Ranking metrics moved in the same favorable direction, but
the score-distribution separation change was small (KS +0.004), and 73.05% of
unknown scores still lie below the known q95 in the weighted arm. The core
known/unknown overlap remains substantial and unresolved.

These are point estimates from one seed and one fixed semantic-hard split.
They do not establish robustness across seeds/splits. No default change is
justified. Because the candidate rule was chosen after historical exploratory
test diagnostics, do not treat this as a clean confirmatory result or tune
candidate ratio/weight on this test set. A next evaluation should pre-register
the fixed recipe and a fresh training seed (preferably another independent
class split), comparing only against all-pool under the same protocol. If the
gain does not replicate, stop tuning this rejector-pool direction and move to
training-time class-conditional support/outlier representation objectives.

## Fixed-recipe replication on seed 2027

To test whether the seed-2026 effect was seed-specific, the same runner and
the exact same saved teacher/student/discovery configuration templates were
used with training seed 2027 (5 teacher epochs, 4 student epochs). As before,
each pair shares one student and differs only in rejector pool treatment.
Both arms again used the same 5400-sample mixed pool, known prior 0.2, and
validation-only 95% known-coverage calibration. The configurations were
audited before interpreting the test summary.

| Metric | seed-2027 all-pool | seed-2027 prototype-boundary weighted | Difference |
|---|---:|---:|---:|
| AUROC | 0.7938422 | 0.8008922 | +0.0070500 |
| AUPR | 0.6932565 | 0.7003600 | +0.0071035 |
| FPR95 | 0.6076667 | 0.5761667 | -0.0315000 |
| OSCR | 0.5915682 | 0.5979052 | +0.0063370 |
| Known acceptance rate | 0.9495000 | 0.9486667 | -0.0008333 |
| Unknown rejection rate | 0.2545000 | 0.2607500 | +0.0062500 |
| Known class accuracy after acceptance | 0.7105494 | 0.7118763 | +0.0013269 |
| KS distance | 0.4378333 | 0.4478333 | +0.0100000 |
| Unknown below known score q95 | 0.7475000 | 0.7422500 | -0.0052500 |

Calibration-set known acceptance was 0.9495413 in both arms. Across seeds
2026 and 2027, AUROC/FPR95/OSCR all move in the favorable direction and
known acceptance/classification utility is essentially preserved. However,
unknown rejection gain contracts from +2.60 pp to +0.625 pp, while 74.225% of
unknown scores remain below known q95 on seed 2027. This is a consistent but
small post-hoc rejector signal, not a resolution of the representation/score
overlap. Both runs use the same semantic-hard class split, and the method was
prioritized after historical test diagnostics; the runs are therefore not
independent confirmation. Keep defaults unchanged, do not sweep weights on
these tests, and only continue with a pre-registered fresh seed/independent
split. Any separate representation-learning change should be evaluated as a
new factor rather than combined immediately with this post-hoc treatment.

Seed-2027 machine-readable output is
[`semantic_hard_rejector_seed2027_summary.json`](semantic_hard_rejector_seed2027_summary.json).

The runner is [`scripts/run_semantic_hard_rejector_seed_recheck.py`](../scripts/run_semantic_hard_rejector_seed_recheck.py).
It refuses to overwrite pre-existing outputs. If resumed later, restart the
entire run from the beginning (or deliberately add checkpoint-resume support
and verify teacher checkpoint integrity first); do not silently shorten the
student run or treat a partial teacher as a completed matched setup.

## Artifacts

Machine-readable metrics and per-arm saved configs are in
[`semantic_hard_rejector_seed2026_summary.json`](semantic_hard_rejector_seed2026_summary.json).
The run outputs are under `runs/semantic_hard_rejector_seed2026_{teacher,student,all_pool,prototype_boundary_weighted}/`.
The runner is [`../scripts/run_semantic_hard_rejector_seed_recheck.py`](../scripts/run_semantic_hard_rejector_seed_recheck.py).
