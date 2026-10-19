# VOS source-gradient classwise false-accept audit (2026-10-18)

## Question and scope

Check whether false acceptance is concentrated in the same novel semantic
classes across the matched source-gradient experiment's two seeds and two
arms. This is a post-hoc description of the existing test outputs only. Test
labels were not used for training, model selection, calibration, threshold
fitting, or any code/configuration decision. No model was retrained.

The four `discovery_detail.json` files each contain 10,000 test examples:
6,000 known and 4,000 novel. The split file
`splits_cifar100_protocols/cifar100_60_40_semantic_isolated.json` identifies
the 40 novel CIFAR-100 classes. `true_known == 0` denotes novel examples in
these details; using `raw_labels` alone without the split would be ambiguous.
There are 100 test examples per novel class in each run.

## Aggregate false acceptance

| Seed | VOS arm | Novel false accepts / 4,000 | Novel false-accept rate |
|---:|---|---:|---:|
| 123 | detached | 2,332 | 58.33% |
| 123 | source-gradient | 2,128 | 53.20% |
| 44 | detached | 2,021 | 50.53% |
| 44 | source-gradient | 2,152 | 53.80% |

The treatment reduces false accepts at seed 123 but increases them at seed 44,
consistent with the mixed aggregate comparison in the paired experiment.

## Classwise pattern

Highest false-accept-rate novel classes, by arm (class names from the split
metadata):

| Seed/arm | Top five novel classes (false-accept rate) |
|---|---|
| 123 detached | forest 85%, kangaroo 77%, bus 75%, crocodile 74%, lizard 74% |
| 123 source-gradient | lizard 80%, lamp 74%, beaver 70%, forest 70%, snake 70% |
| 44 detached | crocodile 79%, beaver 72%, lizard 71%, train 70%, bus 68% |
| 44 source-gradient | television 88%, bus 86%, lamp 83%, clock 82%, motorcycle 80% |

Only `bus` appears in the top ten in all four runs. Other classes recur in
some arms, but per-class rankings and treatment effects vary substantially:
for example, source-gradient changes the false-accept rate for `telephone`
by +43 percentage points at seed 44, while its largest improvements there
are `whale` (-32 pp) and `dolphin` (-30 pp). At seed 123, the largest
decreases are `mountain` (-26 pp) and `plain`/`train` (-18 pp), while
`bicycle` increases by 14 pp. This heterogeneous response is not evidence of
a stable class-conditional benefit.

Across the four runs, the largest mean false-accept rates are bus 74.5%,
lizard 71.5%, forest 70.2%, lamp 70.0%, train 70.0%, crocodile 69.0%,
television 68.8%, and beaver 68.2%. These means are descriptive only: each
seed/arm has just 100 test examples per novel class, the same fixed semantic
split is used in both seeds, and the classwise treatment direction is not
stable. They must not be interpreted as estimates of universal class
difficulty or used to create class-specific test-informed training weights.

## Decision

This audit does **not** support the hypothesis that a small, stable set of
semantic-neighbor classes alone explains the aggregate overlap, nor does it
justify class-conditioned loss weights derived from test errors. It supports
stopping the source-gradient VOS variant: the overall two-seed results are
mixed, and per-class changes are highly seed/arm dependent.

Next work should use a training-legal source of structure: audit whether
available known training/validation data can provide stable class-conditional
boundary-risk signals, or revisit the mixed-pool protocol/representation
objective. Before a new training run, define a label-free or training-label-
only proxy, verify its wiring and known-sample contamination, and predeclare a
single-factor paired comparison. Keep AUROC, FPR95, OSCR, score overlap, the
validation-calibrated operating point, and known utility together; do not
select a method using test-class error rates.

## Artifacts

- Split: `splits_cifar100_protocols/cifar100_60_40_semantic_isolated.json`.
- Four detail files: matching `*_discover/discovery_detail.json` under
  `runs/vos_source_gradient_compare_s{44,123}_e4_20261018*`.
- Aggregate paired results and gradient mechanism audit:
  `analysis/vos_source_gradient_compare_s123_e4_20261018.md`.
