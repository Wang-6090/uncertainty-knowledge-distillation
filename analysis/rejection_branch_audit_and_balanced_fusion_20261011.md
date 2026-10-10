# Rejection-branch audit and balanced detector fusion (2026-10-11)

## Why this round was needed

The central failure remains overlap between known and unknown representations.
The previous comparison of the rejection branch was not fully causal: the
baseline and treatment used different student checkpoints. This round first
repeated the comparison with controlled checkpoints, then tested the detector
representation without retraining.

## Controlled training comparison

Fixed protocol: CIFAR-100 semantic-isolated 60/40, seed 42, pretrained
ResNet-34/18, full data, three student epochs, matched mixed discovery pool of
5,400 samples with known prior 0.2, linear `nu_corrected` nnPU rejector,
`rejection_support_augmented`, MC=4, clustering disabled, and validation-only
95% known-coverage calibration.

The baseline used no loss on the independent rejection branch. The first
treatment changed only `alpha_rejection_known_ce` from 0 to 1.0. A second
treatment additionally set `alpha_rejection_supcon=0.1`.

| checkpoint / detector input | AUROC | FPR95 | OSCR | known acceptance | unknown rejection |
|---|---:|---:|---:|---:|---:|
| baseline, untrained rejection branch | 0.7652 | 0.6472 | 0.3552 | 95.27% | 17.70% |
| rejection CE only | 0.7626 | 0.6417 | 0.3351 | 95.48% | 17.23% |
| rejection CE + SupCon | 0.7496 | 0.6235 | 0.3253 | 94.80% | 16.15% |

Both new losses were non-zero in the training logs, so this is not a wiring
failure. They nevertheless reduce the useful open-set metrics overall. The
independent rejection branch should not receive more supervised losses by
default; it remains an ablation direction until a target that represents
unknown structure is available.

## Invalid first comparison and corrected detector-only audit

The first representation comparison was invalid: its command changed
`rejector_feature_mode` but kept `score_mode=entropy_proto`. The rejector was
fitted and attached, but its score was not consumed by the final detector.
Therefore the earlier table reporting `0.7700/0.6332/0.3570` must not be used
as evidence for balanced fusion.

The code now rejects non-default rejector feature modes when the selected score
mode cannot consume them. It also preserves `feature_rejector_score` and
`known_support_score` in the no-clustering detail output; previously those
fields were silently serialized as zeros.

The corrected comparison uses the same seed-43 student checkpoint, CIFAR-100
semantic-isolated 60/40 protocol, matched mixed pool (5,400 samples, known
prior 0.2), linear `nu_corrected` nnPU, MC=4, and validation-only 95%
known-coverage calibration. The only algorithmic change is the rejector input:

| rejector input | AUROC | FPR95 | OSCR | known acceptance | unknown rejection |
|---|---:|---:|---:|---:|---:|
| `rejection_support_augmented` | 0.7509 | 0.6557 | 0.3151 | 94.58% | 16.43% |
| `rejection_balanced_support_augmented` | 0.7519 | 0.6512 | 0.3161 | 95.30% | 15.03% |
| balanced - support | +0.0010 | -0.0045 | +0.0011 | +0.72pp | -1.40pp |

The balanced representation gives a very small ranking improvement but worse
unknown rejection at the selected operating point. It is therefore retained
as an opt-in ablation, not promoted as a solution to the overlap problem.

## Code and validity checks

- Added opt-in `rejection_balanced_support_augmented` mode in
  `novel_discovery/pipeline.py` and the CLI validation in `train.py`.
- Added a unit test for the class-logit block normalization.
- Added a configuration guard preventing unused rejector feature ablations.
- Added a regression test ensuring no-clustering reports preserve fitted
  rejector/support scores.
- `python -m pytest -q tests/test_core_behaviors.py`: 159 passed.
- The balanced run used `--threshold-policy known_coverage`; an earlier pilot
  omitted this flag and therefore used the default global policy. That pilot
  is not used for the causal conclusion.

## Decision

Keep balanced fusion as an opt-in detector ablation. Do not make it the
default and do not claim that it fixes known/unknown overlap. The next useful
experiment is a multi-seed recheck of this detector-only change, followed by
an audit of score distributions and semantic-near unknown classes. Further
stacking CE/SupCon on the rejection branch is currently low priority.

## Seed-44 stratified-support recheck

The seed-44 student checkpoint was held fixed. The only change was whether
the known examples used to fit the mixed-pool rejector were sampled uniformly
or class-stratified. The detector consumed the rejector score directly via
`score_mode=feature_rejector`; earlier seed-44 runs using `entropy_proto` were
not used for this comparison.

Protocol: semantic-isolated CIFAR-100 60/40, 5,400-sample mixed pool,
known prior 0.2, linear `nu_corrected` nnPU, MC=4, validation-only 95%
known-coverage calibration, clustering disabled.

| Rejector fitting | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
|---|---:|---:|---:|---:|---:|
| uniform known sampling | 0.7681 | 0.6137 | 0.3859 | 94.22% | 22.03% |
| class-stratified known sampling | 0.7700 | 0.6115 | 0.3837 | 94.58% | 20.25% |
| stratified - uniform | +0.0019 | -0.0022 | -0.0022 | +0.36 pp | -1.78 pp |

This reproduces the earlier seed-43 pattern: a negligible ranking gain, but
lower unknown rejection at the calibrated operating point and no OSCR gain.
Class-stratified support therefore remains an ablation, not a default change.

The retrospective classwise audit also shows that the failure is broad rather
than caused by one class. Under uniform support, the worst unknown false
acceptance rates include raw classes 43 (0.97), 22 (0.92), 15/34/40 (0.91),
and 38/87 (0.89). Stratification changes individual classes in both
directions: raw class 31 improves by 10 percentage points, while classes 48
and 75 worsen by 14 and 12 points. This is not a stable semantic separation
pattern. Test labels were used only for this retrospective grouping.

The next algorithmic work should target training-time separation of the mixed
pool and improved closed-set representation quality. More rejector feature
concatenation and threshold sweeps are not the next priority.
