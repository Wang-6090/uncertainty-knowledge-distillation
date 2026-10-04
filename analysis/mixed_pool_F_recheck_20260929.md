# Mixed discovery-pool recheck for F (2026-09-29)

## Question

Does the previously promising F recipe (uncertainty KD + feature KD + SupCon +
prototype alignment + unlabeled-pool NT-Xent) still improve open-set detection
when the discovery pool contains both known and novel samples, rather than being
filtered to novel classes using training labels?

## Experimental conditions

- Dataset/protocol: CIFAR-100, 60 known / 40 novel, split seed 42, full known
  training and test data, 64px inputs.
- Student/teacher: pretrained ResNet-18 / ResNet-34; same saved seed-42 teacher.
- Training: 15 epochs, batch size 64, same optimizer and loss weights as F
  (`alpha_unc=0.1`, `alpha_kd=1`, `alpha_feat_kd=0.1`, `alpha_supcon=0.1`,
  `alpha_proto=0.1`, discovery NT-Xent `alpha=0.05`, temperature 0.2).
- The F reference uses all novel training samples (oracle-filtered, 20,000
  samples); this recheck uses a deterministic 20,000-sample subset of the
  concatenated known+novel pool. Thus pool size and epochs match, while pool
  semantics/composition is the intended change. The mixed pool is not balanced
  by class; this is a limitation of this first recheck.
- Detection uses the same held-out CIFAR-100 test split, `normalized_entropy_mahalanobis`,
  8 MC samples, and known-validation global thresholding. The raw ranking
  metrics are independent of the threshold. For operating-point comparison,
  test scores were additionally post-hoc evaluated at exactly 95% known
  coverage (labels used only for retrospective metric calculation, not model
  selection).
- Detection used `--skip-clustering`: clustering is not needed to assess this
  detection hypothesis and auto-K took disproportionately long. This report
  therefore makes no new clustering claim.

## Results

| Pool recipe | AUROC | AUPR | FPR95 | Unknown reject at matched 95% known coverage | Known classification accuracy (all known) |
|---|---:|---:|---:|---:|---:|
| F, oracle-filtered novel-only pool | 0.6419 | 0.5066 | 0.8425 | 9.15% | 52.37% |
| F, unlabeled mixed pool | 0.5779 | 0.4458 | 0.9055 | 6.38% | 53.18% |

At 95% known coverage, accepted-known classification accuracy was 54.14% for
novel-only and 54.09% for mixed. Mixed-pool detection loses 0.0640 AUROC and
2.77 percentage points of unknown rejection versus the novel-only reference;
FPR95 worsens by 0.0630. The small known-class accuracy change does not offset
the detection regression.

## Interpretation and limitations

This is evidence that F's apparent detection gain is not robust to realistic
pool contamination: the pure-novel pool is an oracle-filtered upper-bound
setting, not a practical unlabeled discovery pool. The experiment has one seed,
so it establishes a direction for this protocol, not statistical significance.
The mixed pool's class composition is naturally imbalanced and differs from
the oracle novel-only pool by design. A three-seed confirmation would be
needed before making a general claim.

The result does not support simply increasing NT-Xent weight. Instance
augmentation consistency can improve representation invariance without
separating known from novel semantic classes. The most justified next step is
not another threshold adjustment: evaluate a genuine generalized-category-
discovery objective that jointly forms known and novel semantic clusters on a
mixed unlabeled pool, with known-class supervision/retention and conservative
candidate filtering. Compare it first against the current F mixed baseline at
matched 95% known coverage; only promote it if AUROC/FPR95 and unknown rejection
improve without reducing known accuracy, then repeat over at least three seeds.

## Runtime implementation note

Full-data Mahalanobis evaluation was constructing an `N x C x D` temporary at
once. It now computes the exact same formula in sample chunks (default 512),
reducing peak memory while preserving outputs. The new unit test checks both
diagonal and shared-covariance results against one-chunk computation.

Validation: `python -m pytest -q tests --disable-warnings --maxfail=1` ->
84 passed; `py_compile` passed.
