# Lightweight project audit (2026-10-01)

## Audit scope

This audit checks the current training, detection, and discovery pipeline
against the actual experimental evidence. It does not treat an implemented
command-line option as a validated method.

## What is currently supported by evidence

1. The mixed-pool nnPU uncertainty objective has the strongest evidence: in
   three full-data seeds it improved AUROC, FPR95, OSCR, known acceptance, and
   accepted-known accuracy in the same direction. It improves uncertainty
   ranking and the operating point, but the mean unknown rejection is still
   only 10.44%.
2. `normalized_entropy_min_class_knn` improves the detection score relative to
   the earlier Mahalanobis score in the existing three-seed comparison.
3. `feature_pca` improves clustering representation metrics without changing
   candidate purity. It is a clustering-side improvement, not a detector-side
   solution.
4. Frozen-teacher candidate selection is executable and has local value for
   candidate clustering, but its unknown-rejection and AUROC gains are not
   consistent across two seeds.

## Main unresolved problems

### 1. The learned known representation is still weak at the operating point

Accepted-known accuracy is only about 50% in the strongest full-data nnPU
comparison. This means many samples accepted as known are classified into the
wrong known class. A detector cannot reliably separate unknowns if the known
support itself is poorly modeled. Closed-set accuracy and per-class support
diagnostics must therefore be treated as a gate before interpreting unknown
rejection.

### 2. Mixed-pool novel supervision is still contaminated

The current joint discovery code has novel-only, unified, residual, novel-mass,
candidate-gating, and memory-bank options, but these are not a complete UNO or
SimGCD reproduction. In particular, novel weights and balanced assignments can
still depend on unstable predictions from the same evolving student/head.
The candidate pool can also contain hard known samples. This is the most
important algorithmic gap.

### 3. The evidence is unevenly distributed across methods

Many post-hoc scores and boundary losses changed individual metrics but did not
produce a stable matched-coverage improvement. OpenMax is a clear negative
fixed-checkpoint result. Selector changes and prototype/memory variants should
remain ablations until they survive paired multi-seed tests.

### 4. The data protocol still needs a robustness check

The primary protocol is random CIFAR-100 60/40. Semantic-hard and
semantic-isolated splits exist, but they are not interchangeable with the
random protocol. A method should not be promoted from one split without a
fixed split-specific report and at least two seeds.

## Highest-priority direction

The next main experiment should be a controlled representation-learning
experiment, not another detector. Keep the current nnPU baseline fixed and
compare one mixed-pool GCD variant that has:

- a known-class head trained only with known labels;
- an independent novel prototype head rather than one forced known+novel
  balanced Softmax;
- an EMA or frozen teacher for pseudo-label targets;
- periodic prototype/pseudo-label refresh from the current unlabeled pool;
- explicit tracking of candidate purity, prototype occupancy, assignment
  stability, closed-set known accuracy, and unknown detection metrics.

The newly added `--joint-prototype-refresh-epochs` is only the first minimal
piece of this direction. Its toy smoke test confirms execution, but no CIFAR
performance claim is allowed yet.

## Decision gates for the next iteration

1. First compare one-time KMeans initialization against periodic refresh under
   the same CIFAR split, teacher, data budget, epochs, detector, and threshold.
2. Require the treatment to improve at least one ranking metric and one
   matched-coverage metric without reducing accepted-known accuracy materially.
3. Inspect prototype occupancy and pseudo-label stability. If refresh only
   changes loss values or candidate counts, reject it as ineffective.
4. If it fails, stop adding prototype/memory variants and redesign the full
   mixed-pool GCD objective with a documented UNO/SimGCD-style protocol.
5. Only after representation quality improves should OpenMax, rejectors, or
   further score fusion be reconsidered.

