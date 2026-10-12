# Prior-aware Sinkhorn marginals for unified known/novel discovery (2026-10-12)

## Question and one-factor hypothesis

The unified joint head concatenates 60 known logits and 40 novel-prototype
logits. The earlier implementation used equal per-class Sinkhorn mass across
all 100 classes. That imposes 60% total mass on the known group and 40% on the
novel group, although this experiment's unlabeled discovery pool has known
prior 0.2 and novel prior 0.8. This mismatch could bias cross-view pseudo-labels
and entangle known rejection with novel discovery.

The treatment changes only the target class marginal in the unified Sinkhorn
assignment: each known class receives `0.2 / 60`, each novel prototype receives
`0.8 / 40`. The pool prior comes from the configured/measured training discovery
pool, not test labels. The legacy equal-class behavior remains the default; the
new behavior is opt-in through `--joint-prior-aware-assignments`.

This is motivated by the balanced-assignment family used in self-labeling and
novel discovery (e.g. SwAV; Asano et al., 2020; UNO; Fini et al., 2021), while
correcting the assumption that every output class—including known and novel
groups together—has equal population mass. It is not a claim that the CIFAR
novel classes are necessarily uniform in real deployment; that is a protocol
assumption to validate.

## Protocol and comparison

- CIFAR-100 semantic-isolated 60/40 split, seed 44.
- Same pretrained ResNet-34 teacher, ResNet-18 student, 3 epochs, train/val/test
  limits 1200/300/1000, discovery pool 1200 with measured known fraction 0.2.
- Same optimizer, `encoder_lr_scale=0.1`, KD, feature KD, SupCon, prototype loss,
  uncertainty nnPU, KMeans initialization/refresh, unified joint objective,
  detector (`normalized_entropy_min_class_knn`, kNN k=10, MC=4), and disjoint
  validation calibration (`calibration_ratio=0.25`, target known coverage 95%).
- Treatment factor: prior-aware 20%/80% group mass vs the existing equal-mass
  assignment. Detection and clustering use the same test examples and score
  family. Oracle K is used only in the retrospective cluster diagnostic, never
  to select the detector or training labels.

## Results

The first two pilot executions are invalid and excluded: the first did not
pass the prior marginal through the standard cross-view call chain; the second
fixed that wiring but still had incorrect unweighted Sinkhorn normalization.
The results below use a third run started after both fixes and after tests
verified row sums and weighted/unweighted class marginals.

### Detection and representation

| Variant | AUROC | FPR95 | OSCR | Known accept | Unknown reject | KS | Unknown below known q95 | Known acc (all known) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Equal per-class mass | 0.6977 | 0.7962 | 0.2434 | 98.41% | 6.45% | 0.3249 | 0.8575 | 28.34% |
| Prior-aware 20/80 mass | 0.7159 | 0.7627 | 0.2748 | 95.70% | 13.44% | 0.3356 | 0.8414 | 31.69% |

This same-command comparison suggests a modest ranking gain, but the fixed
known-coverage calibration produced lower test known acceptance in the
treatment. AUROC improves by 0.0183 and OSCR by 0.0314, while FPR95 drops by
0.0334 and unknown rejection rises by 6.99 points. The gain is not merely
threshold tuning because AUROC, OSCR, and KS move in the same direction, but
the test operating point includes about 2.71 percentage points more known
rejections. One seed is insufficient to establish robustness.

### Candidate quality and clustering (same detector, oracle K diagnostic)

| Variant | Candidate count | Candidate purity | Unknown recall | Candidate unknown-only NMI | Candidate unknown-only ARI | All-unknown ARI |
|---|---:|---:|---:|---:|---:|---:|
| Equal per-class mass | 34 | 70.59% | 100% | 0.8587 | 0.1825 | 0.0580 |
| Prior-aware 20/80 mass | 77 | 64.94% | 100% | 0.7631 | 0.0294 | 0.0722 |

The treatment raises candidate count and lowers purity and candidate-only
unknown ARI. All unknown test samples are included in the candidate set in
both arms because this diagnostic uses the permissive oracle-K clustering
candidate protocol; this does not mean the operational detection threshold
rejects all unknowns. These cluster values must be read as end-to-end behavior,
not a pure clustering-feature ablation.

## Implementation audit and invalid first run

Two executions were invalid: `runs/joint_pilot_s44_prioraware/` had a call-chain
wiring omission; `runs/joint_pilot_s44_prioraware_fixed/` had a faulty
unweighted Sinkhorn normalization. Both are audit artifacts and excluded from
the tables. The valid treatment is `runs/joint_pilot_s44_prioraware_valid/`;
the paired diagnostics are in `runs/joint_pilot_s44_prioraware_valid_cluster_diag/`
and `runs/joint_pilot_s44_baseline_cluster_diag_matched/`.

Regression tests assert both the ordinary cross-view propagation of
`[0.2/60]*60 + [0.8/40]*40` and the resulting sample/class marginals. After the
final implementation correction, all 29 tests in `tests/test_joint_discovery.py`
passed and Python compile checks passed.

## Decision

Keep prior-aware assignment opt-in, not default. It gives a one-seed pilot
detection improvement but harms candidate purity and candidate-only ARI. The
planned soft known-residual combination is structurally incompatible with
this unified assignment path as implemented: residual mode deliberately
removes known logits from the Sinkhorn space. Its attempted run failed with a
clear validation error before producing a checkpoint, so it is not an
algorithm result. Do not combine the two by silently bypassing that design
choice. Next replicate the corrected prior-aware-vs-equal-mass pair on another
seed; only if the detection direction holds should a separately designed
soft-gating method be considered, with an explicitly specified assignment
space and calibrated known/novel marginals.

## Artifacts

- Baseline: `runs/joint_pilot_s44_baseline/`,
  `runs/joint_pilot_s44_baseline_discover/`,
  `runs/joint_pilot_s44_baseline_cluster_diag/`.
- Corrected treatment: `runs/joint_pilot_s44_prioraware_valid/`,
  `runs/joint_pilot_s44_prioraware_valid_discover/`,
  `runs/joint_pilot_s44_prioraware_valid_cluster_diag/`.
- Reproduction script: `scripts/run_joint_prior_aware_pilot_s44.ps1`.
- Invalid first execution (audit only): `runs/joint_pilot_s44_prioraware/`.
- Invalid second execution (audit only): `runs/joint_pilot_s44_prioraware_fixed/`.
