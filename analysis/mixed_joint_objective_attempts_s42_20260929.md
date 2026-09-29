# Mixed-pool joint objective attempts (2026-09-29)

> **Audit limitation:** These single-seed model comparisons were exploratory, and the Sinkhorn target weighting was not independently controlled in the historical code path. Interpret only the recorded checkpoint outcomes; do not attribute them causally to equal/weighted Sinkhorn. See `analysis/method_fidelity_and_sinkhorn_audit_20260929.md` for the corrected experiment.

## Experiment question and protocol

The goal was to test whether the current joint generalized-category-discovery
objective harms unknown detection by forcing every mixed unlabeled batch into
a balanced known-plus-novel Sinkhorn assignment. CIFAR-100 used a fixed random
60/40 split (seed 42), a shared pretrained ResNet-34 teacher, ResNet-18
students, a disjoint mixed discovery pool with known-pool ratio 0.2, and
1200/300/1000 train/validation/test limits. Both students trained for 3 epochs
with the same remaining losses and KMeans novel-prototype initialization.
Detection used the same test subset, explicit
`normalized_entropy_mahalanobis`, MC=4, known-only 95% coverage calibration,
and skipped clustering.

## Results

The model-report operating points are shown first. Because finite validation
quantiles and the small test subset do not yield exactly 95% test known
coverage, an additional retrospective computation from saved per-sample
scores reports unknown rejection at exactly 95% test known coverage. Test
labels are used only for this retrospective metric, not for threshold fitting.

| Method | AUROC | AUPR | FPR95 | OSCR | Known acc (all known) | Reported known accept | Reported unknown reject | Unknown reject at exact 95% known coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unified known+novel Sinkhorn baseline | 0.5676 | 0.4310 | 0.8909 | 0.1891 | 0.2579 | 0.9421 | 0.0658 | 0.0557 |
| Novel-mass weighted novel assignments | 0.5570 | 0.4324 | 0.8893 | 0.1914 | 0.2628 | 0.9686 | 0.0304 | 0.0633 |
| Novel mass + soft feature repulsion | 0.5721 | 0.4329 | 0.8843 | 0.1738 | 0.2397 | 0.9504 | 0.0506 | 0.0506 |

The first proposed weight, `1 - max softmax(known logits)`, was also smoke-tested
and trained once. Compared with unified Sinkhorn, it yielded AUROC 0.5793 vs
0.5976, FPR95 0.8612 vs 0.8595, OSCR 0.1951 vs 0.2049 and unknown rejection
8.10% vs 8.35% under its original auto-selected detector. This weight was
rejected: it measures uncertainty *within* the known classifier and is not a
reliable known-versus-novel comparison.

## Interpretation

- Avoiding one balanced 100-class Sinkhorn space is a reasonable design change,
  but novel-mass weighting alone did not consistently improve ranking. AUROC
  decreased by 0.0106 relative to the unified baseline, while FPR95 changed by
  only 0.0017 and exact-coverage unknown rejection improved by 0.0076.
- Adding soft feature repulsion produced small AUROC/FPR95 changes relative to
  the unified baseline, but worsened OSCR, known accuracy, and exact-coverage
  unknown rejection. It is not a successful solution to feature overlap.
- The first pair of detections initially used auto-selected `entropy_proto`.
  Those results are not comparable with the project detector baseline. Both
  groups were re-evaluated with explicit Mahalanobis scoring before the
  conclusions above were made.
- All new comparisons use one seed and a reduced training budget. Treat the
  deltas as diagnostic, not statistically significant. Neither new objective
  should become the default or be claimed as a solved open-set boundary.

## Code and checks

- Added `known_residual_weights` and `novel_mass_weights` helpers.
- Added opt-in `--joint-mixed-residual` and `--joint-novel-mass` flags, with
  metrics logging their mean weights.
- Added optional `--alpha-joint-novel-margin` feature repulsion, gated on
  novel-mass mode and weighted by the soft novel evidence.
- Unit tests: `python -m pytest -q tests --disable-warnings --maxfail=1` ->
  `88 passed`; Python compilation passed. Smoke training for both new modes
  completed and showed finite nonzero diagnostics.

## Decision and next experiment

Keep all new behavior opt-in. The next algorithmic test should not add another
global repulsion term. Instead, check whether the pseudo-novel candidates have
real neighborhood support and cluster coherence before updating prototypes:
use a cross-view teacher/EMA prediction plus a global feature memory bank or
nearest-neighbor agreement, and measure candidate precision/recall against
held-out labels strictly as evaluation diagnostics. Compare against the
current mixed-pool baseline at matched known coverage, with the same score,
and do not promote the method unless unknown rejection, FPR95, and OSCR
improve without a meaningful known-accuracy loss. Follow a positive smoke
result with at least three seeds.
