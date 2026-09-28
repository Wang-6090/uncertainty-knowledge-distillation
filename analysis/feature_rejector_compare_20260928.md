# Frozen-feature rejector comparison

## Purpose

The main failure mode is overlap between known and unknown feature/score
distributions. Outlier Exposure improved AUROC slightly but reduced the actual
unknown rejection rate, so this ablation tests a separate binary boundary.

`feature_rejector` fits a balanced Logistic Regression on normalized frozen
student features:

- known samples: the known training split;
- unknown samples: the protocol's unknown discovery pool;
- final test labels are not used for fitting or threshold selection;
- the threshold is still calibrated from known validation samples at the
  existing 95% coverage rule.

This is a controlled upper-bound style experiment because the current
`discovery_pool_mode=unknown` protocol supplies a pure unknown pool. It does
not claim that an unlabeled deployment stream can be separated for free.

## Paired result

Protocol: CIFAR-100 60/40, seed 42, pretrained ResNet-34 teacher and ResNet-18
student, 5 epochs, 1200 train / 300 validation / 1000 test samples, CUDA,
`normalized_entropy_mahalanobis` for the existing gate comparison, oracle-K
clustering.

| Method | AUROC | FPR95 | Known accept | Unknown reject | OSCR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent uncertainty gate | 0.5395 | 0.8980 | 0.9128 | 0.0791 | 0.2091 |
| CIFAR-10 OE, uniform + energy + uncertainty | 0.5591 | 0.9227 | 0.9523 | 0.0459 | 0.2191 |
| Frozen-feature binary rejector | **0.6666** | **0.8076** | 0.9391 | **0.1454** | **0.2269** |

### Cross-seed and mixed-pool checks

The same detector was also evaluated on two existing standard-KD
checkpoints. These checkpoints use a smaller 512/128/512 protocol, so the
numbers are a stability check rather than a direct replacement for the main
5-epoch result.

| Seed | Baseline AUROC | Rejector AUROC | Baseline FPR95 | Rejector FPR95 | Baseline unknown reject | Rejector unknown reject |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 123 | 0.4958 | **0.5799** | 0.9274 | **0.8614** | 0.0287 | **0.0813** |
| 3407 | 0.5051 | **0.5964** | 0.9144 | 0.9075 | 0.0727 | **0.1273** |

With seed 42, replacing the pure unknown discovery pool by a mixed pool
(`discovery_pool_mode=mixed`) gave AUROC `0.6316`, FPR95 `0.8487`, and unknown
rejection `0.0791`. The drop from the pure-pool result shows that treating all
mixed-pool samples as negative is not appropriate, but the result remains
better than the original independent gate.

### Mixed-pool pseudo-negative ablation

Two label-free candidate-selection variants were tested on the same mixed
pool. They did not improve the all-pool rejector:

| Mixed-pool training rule | AUROC | FPR95 | Unknown reject | Candidate purity |
| --- | ---: | ---: | ---: | ---: |
| All mixed samples as rejector negatives | **0.6316** | **0.8487** | 0.0791 | 0.5439 |
| Top 25% consensus pseudo-unknowns | 0.5428 | 0.9079 | **0.0944** | 0.4512 |
| Top 50% entropy + uncertainty pseudo-unknowns | 0.5744 | 0.8766 | **0.0969** | 0.4935 |

The higher rejection rate in the two pseudo-negative variants is caused by
more false rejection of known samples, not by a reliable improvement in the
ranking. The current decision is therefore to disable high-risk pseudo-
negative selection by default and investigate PU learning or a separately
trained rejector with a genuinely disjoint auxiliary source.

The rejector also raised candidate-pool purity from the OE run's 0.3830 to
0.6064. This supports the diagnosis that the missing component is an explicit
known/unknown boundary, not another threshold adjustment.

## Interpretation and limits

The result is promising, but it is not yet a final method result. The rejector
uses a pure unknown discovery pool, so it must next be tested with a mixed
unlabeled pool and with a held-out discovery split. The next implementation
should compare:

1. frozen-feature Logistic Regression;
2. a small MLP rejector trained jointly or after warm-up;
3. mixed-pool training with pseudo-negative selection;
4. at least three seeds and the full training split.

The detector is exposed as `--score-mode feature_rejector`. It is opt-in and
does not alter historical score modes.

## Follow-up audit: representation quality and virtual outliers

The first follow-up test used a known-only virtual-outlier rejector. It mixed
features from different known classes and trained a logistic boundary on those
synthetic points, following the general motivation of VOS/NPOS-style
low-density or virtual outlier methods. On the joint-discovery checkpoint it
failed badly: AUROC was `0.4789`, FPR95 `0.9836`, known acceptance `0.9178`,
and unknown rejection only `0.0536`. The number of rejected known samples
rose from 17 to 50. This synthetic feature geometry does not match the real
CIFAR-100 unknown distribution, so this variant is disabled and is not a
candidate final method.

The more important control test changed the checkpoint, not the detector. A
15-epoch CE-only student trained on the complete known training split was
evaluated with the same frozen-feature logistic rejector:

| Checkpoint / rejector | AUROC | FPR95 | Known accept | Unknown reject |
| --- | ---: | ---: | ---: | ---: |
| 5-epoch joint model, feature rejector | 0.6666 | 0.8076 | 0.9391 | 0.1454 |
| 15-epoch CE-only student, feature rejector, 5k discovery pool | **0.6961** | **0.7598** | 0.9390 | **0.1558** |
| 15-epoch CE-only student, feature rejector, full discovery pool | 0.6928 | 0.7650 | 0.9397 | 0.1578 |

This is evidence that the previous low rejection rate was partly caused by weak
student representations and the smoke-scale training protocol (`1200`
training images, about 20 per known class). It does not prove that the
rejector alone solves open-set recognition. The full training protocol and a
strictly held-out discovery split are still required, and the known rejection
rate remains substantial. The next experiments should therefore improve the
student representation first, then compare the rejector under a disjoint
mixed-pool protocol.

## Full-data CE/KD comparison

To separate the effect of representation learning from the rejector, the same
test protocol was applied to three existing 15-epoch students trained on the
complete known training split. All used the same frozen-feature Logistic
rejector, the same 5,000-sample pure-unknown discovery pool, the same global
95th-percentile known-only threshold, and the same 10,000-image test set.

| Student representation | Historical gate AUROC | Rejector AUROC | FPR95 | Known accept | Unknown reject |
| --- | ---: | ---: | ---: | ---: | ---: |
| CE only | 0.5958 | 0.6961 | 0.7598 | 0.9390 | 0.1558 |
| Standard KD | 0.5781 | **0.7103** | 0.7670 | 0.9470 | **0.1670** |
| Uncertainty-weighted KD | 0.5989 | 0.7035 | **0.7608** | **0.9518** | 0.1413 |
| Uncertainty KD, mean-normalized weights | 0.5985 | 0.7080 | **0.7430** | 0.9513 | 0.1290 |

The independent rejector improves all three historical gates substantially.
Standard KD currently provides the best ranking and highest unknown rejection,
while uncertainty-weighted KD provides the highest known acceptance and a
similar FPR95 to CE. Therefore the present evidence supports retaining KD as
the representation-learning component, but it does not yet demonstrate that
the current uncertainty weighting is superior to standard KD. The uncertainty
component needs a controlled ablation with the same loss scale, calibration,
and seeds before it can be claimed as an improvement.

Mean-normalizing the uncertainty weights improves the FPR95 from `0.7608` to
`0.7430` and keeps AUROC close to standard KD, but its unknown rejection rate
falls to `0.1290`. Thus it improves ranking at the selected operating point
without solving the main recall problem. The current evidence favors keeping
mean normalization as the uncertainty-KD implementation candidate, while
changing the uncertainty target or adding an explicit unknown-aware objective
still requires a new controlled training experiment.

These numbers still use a pure-unknown discovery pool selected by the dataset
protocol. The next fair test is a disjoint mixed-pool/PU protocol, followed by
three-seed full-data training. The rejector should be evaluated both with and
without unknown-pool supervision so that the final method does not rely on an
oracle novel-only pool.
