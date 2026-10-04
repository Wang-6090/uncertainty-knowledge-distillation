# Mixed-pool candidate and representation recheck (2026-10-01)

## Question

The core failure is overlap between known and unknown features/scores. This
round tested whether safer candidate selection and local feature repulsion can
make mixed-pool discovery training useful without labeling the whole pool as
unknown.

All CIFAR-100 pilot runs used the same random 60/40 split, seed 123, the same
per-seed ResNet-34 teacher, pretrained ResNet-18 student, 1,200 known training
images, 300 validation images, 1,000 test images, 3 epochs, mixed pool with
20% known proportion, MC=4 detection, normalized entropy plus Mahalanobis
score, 95% known-coverage calibration, and no clustering. The main baseline is
the mixed-pool student with no discovery regularizer.

## Code changes

- Added `--discovery-cross-view-gating`. A mixed-pool candidate must be selected
  independently by both augmented views before feature/Objectosphere losses are
  applied.
- Added `prototype_distance` and `distance_consensus` candidate modes. They use
  distance from the nearest normalized known classifier prototype, optionally
  combined with entropy, max-softmax risk, and uncertainty ranks.
- These options are disabled by default; historical entropy-based behavior is
  unchanged.
- Unit tests and toy smoke tests cover argument parsing, prototype selection,
  mixed candidate gating, and nonzero Objectosphere dispatch.

## Results

| Variant | AUROC | FPR95 | OSCR | Unknown rejection | Accepted-known accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mixed baseline | 0.5726 | 0.8693 | 0.2360 | 9.43% | 34.09% |
| Entropy candidate + Objectosphere | 0.5357 | 0.9313 | 0.2038 | 4.22% | 34.05% |
| Entropy candidate + cross-view + Objectosphere | 0.5579 | 0.9229 | 0.2267 | 5.46% | 33.45% |
| Distance-consensus candidate + cross-view + Objectosphere | 0.5528 | 0.9045 | 0.2285 | 7.20% | 34.95% |
| Distance-consensus + cross-view + feature separation | 0.5702 | 0.8710 | 0.2311 | 2.23% | 34.55% |

The cross-view intersection and prototype-distance ranking both reduce the
damage caused by entropy-only candidates, but no treatment exceeds the mixed
baseline. The feature-norm statistic can improve while open-set metrics get
worse, so marginal norm separation is not a sufficient success criterion.

## Invalid or inactive attempt

The first feature-margin run used `alpha_discovery_feature_margin=0.1` with
margin `0.2`; its logged loss was zero in every epoch. A corrected run with
margin `0.8` was also zero. These runs are not performance evidence because
the loss did not contribute gradients. The likely reason is that the selected
distance-consensus candidates were already below the prototype-similarity
margin.

## Decision

- Keep cross-view and prototype-distance selection as optional diagnostics;
  they improve candidate handling locally but do not solve the core boundary.
- Do not enable Objectosphere or local feature repulsion by default for mixed
  pools.
- Do not keep tuning their weights on the 1,200-image pilot.
- The next reliable check should use a full-data matched protocol. The recent
  pilots are useful for eliminating directions, but their low training budget
  can hide representation-learning effects.
- If a full-data run still shows overlap, prioritize a direct mixed-pool
  positive-unlabeled/rejector objective or stronger pretrained/metric
  representation, rather than adding another candidate-only repulsion loss.
