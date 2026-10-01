# Objectosphere full-data strict recheck (2026-10-01)

## Why this recheck was necessary

The first full-data Objectosphere run was not a valid one-factor comparison:
its command omitted the baseline `alpha_feat_kd=0.1` and `alpha_proto=0.1`
settings. That run is retained as an audit failure and is not used as evidence.

The second treatment was also not initially paired with the available baseline,
because the old baseline did not enable a discovery pool. A new control was
therefore trained with the same pure-unknown discovery pool and every training
setting held fixed except the Objectosphere coefficient.

## Controlled protocol

- CIFAR-100 random 60/40 split, seed 42;
- full known training partition, 5 student epochs, batch size 64, CUDA;
- ImageNet-pretrained ResNet-34 teacher and ResNet-18 student;
- identical teacher checkpoint, KD, feature KD, SupCon and prototype settings;
- identical pure-unknown discovery pool;
- `alpha_discovery_objectosphere=0` versus `0.01`;
- complete 10,000-image open test set;
- `normalized_entropy_mahalanobis`, MC=4;
- threshold calibrated only from known validation samples at 95% known coverage;
- clustering disabled, so this is an unknown-detection/representation test.

## Strict paired result

| Metric | Control, alpha=0 | Objectosphere, alpha=0.01 | Change |
| --- | ---: | ---: | ---: |
| Best validation known accuracy | 40.70% | 45.77% | +5.07 pp |
| AUROC | 0.6108 | 0.6283 | +0.0175 |
| FPR95 | 0.8540 | 0.8343 | -0.0197 |
| OSCR | 0.2845 | 0.3306 | +0.0461 |
| Known acceptance | 95.12% | 94.65% | -0.47 pp |
| Unknown rejection | 7.475% | 7.975% | +0.50 pp |
| Accepted-known accuracy | 40.00% | 45.89% | +5.89 pp |

The treatment has a small positive signal across ranking, OSCR and the
validation-calibrated operating point. It is not only a threshold movement:
the score histogram overlap decreases from `0.8202` to `0.7940`, and the
feature-norm histogram overlap decreases from `0.8202` to `0.7853`.

However, the absolute unknown rejection remains very low. This is a one-seed
result under a pure-unknown discovery-pool protocol, where training labels
select the withheld novel classes. It is therefore an encouraging ablation,
not a solution and not a result for a realistic mixed unlabeled pool.

## Implementation audit correction

The detector computed `feature_norm`, but `train.py` previously serialized a
zero-filled fallback because `run_discovery` did not put that field into its
detail dictionary. The two strict evaluations were rerun after fixing this
path. The toy smoke output now contains nonzero feature norms, and the test
suite passes with 130 tests.

## Decision

- Keep Objectosphere as an optional pure-unknown representation ablation.
- Do not enable it by default or claim that it solves known/unknown overlap.
- Do not tune its coefficient on seed 42 again.
- Next, test whether the signal survives a second fixed seed and a mixed-pool
  protocol. If it disappears in mixed data, stop treating Objectosphere as the
  main direction and prioritize a reliable mixed-pool known/novel objective.
- All historical numbers from the two unmatched runs remain exploratory and
  must not be used in the final comparison table.

## Second-seed replication

The same one-factor protocol was repeated with seed 123. The teacher
checkpoint, split, optimizer settings, pure-unknown pool, detector, and
threshold calibration were kept fixed; only the Objectosphere coefficient
changed from 0 to 0.01.

| Metric | Control, alpha=0 | Objectosphere, alpha=0.01 | Change |
| --- | ---: | ---: | ---: |
| Best validation known accuracy | 41.13% | 49.53% | +8.40 pp |
| AUROC | 0.6065 | 0.6245 | +0.0180 |
| FPR95 | 0.8598 | 0.8268 | -0.0330 |
| OSCR | 0.2950 | 0.3563 | +0.0613 |
| Known acceptance | 94.23% | 94.03% | -0.20 pp |
| Unknown rejection | 8.43% | 9.18% | +0.75 pp |
| Accepted-known accuracy | 41.93% | 50.35% | +8.42 pp |

The feature-norm histogram overlap also decreased from `0.8148` to `0.7714`
on this seed. The direction therefore agrees with seed 42, but the absolute
unknown rejection is still low and the protocol is still oracle-filtered
pure unknown. This is evidence for a useful auxiliary representation loss,
not evidence that the open-set boundary has been solved.

## Mixed-pool implementation check

The training code now allows Objectosphere with
`--discovery-pool-mode mixed` only when
`--discovery-feature-candidate-gating` is enabled. The known term uses the
labeled known batch; the unknown term uses only the paired high-risk
candidate mask from the mixed pool. Without that gate, the code still raises
an error rather than treating the entire mixed pool as unknown.

A toy smoke test completed and recorded a nonzero
`discovery_objectosphere` loss and a candidate ratio of `0.5`. This verifies
the execution path and gradient flow only. A CIFAR-100 mixed-pool performance
comparison is still required before deciding whether this extension improves
the core known/unknown overlap problem.

## Updated decision

- Keep pure-unknown Objectosphere as an optional representation ablation.
- Do not make it the default or describe it as a complete solution.
- Treat the mixed-pool candidate-gated version as a new, unvalidated
  experimental branch.
- The next fair performance test should compare mixed-pool baseline versus
  mixed-pool candidate-gated Objectosphere with the same seed and all other
  settings fixed.

## Mixed-pool pilot result

The planned CIFAR-100 pilot used seed 123, the same random 60/40 split and
teacher, pretrained ResNet-34/18, 1,200 training samples, 300 validation
samples, 1,000 test samples, 3 epochs, MC=4, and a 95% known-coverage
threshold. The mixed discovery pool contained a measured 20% known fraction.
The only training change was candidate-gated Objectosphere with coefficient
`0.01`; both runs kept candidate gating enabled in the command, but it is
inactive in the baseline because no discovery loss is enabled.

| Metric | Mixed baseline | Mixed + gated Objectosphere | Change |
| --- | ---: | ---: | ---: |
| Best validation known accuracy | 29.67% | 26.67% | -3.00 pp |
| AUROC | 0.5726 | 0.5357 | -0.0369 |
| FPR95 | 0.8693 | 0.9313 | +0.0620 |
| OSCR | 0.2360 | 0.2038 | -0.0323 |
| Known acceptance | 95.31% | 93.47% | -1.84 pp |
| Unknown rejection | 9.43% | 4.22% | -5.21 pp |
| Accepted-known accuracy | 34.09% | 34.05% | -0.04 pp |

The feature-norm histogram overlap decreased from `0.8281` to `0.7722`, but
the open-set ranking metrics became worse. This is an important negative
result: reducing one marginal feature statistic does not guarantee a useful
known/unknown decision boundary. Candidate purity and the selected unknown
subset were not reliable enough for Objectosphere to learn from in this
small mixed-pool setup.

## Decision after the pilot

- Do not tune the mixed-pool Objectosphere coefficient further based on this
  run; the main open-set metrics and unknown rejection worsened.
- Keep the implementation only as a guarded ablation and retain the pure
  unknown two-seed result as a limited positive signal.
- Future mixed-pool work should first improve candidate reliability or use a
  positive-unlabeled/rejector objective that models the mixed pool directly;
  applying a feature-norm target to selected candidates is not sufficient.
