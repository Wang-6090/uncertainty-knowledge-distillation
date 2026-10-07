# EMA pseudo-label target and KMeans prototype synchronization recheck

Date: 2026-10-05  
Dataset: CIFAR-100 random 60/40 split, seed 42  
Protocol: mixed discovery pool, 1200 train / 300 validation / 1000 test / 1200 discovery, ResNet-34 teacher and ResNet-18 student, 3 epochs, CUDA.

## Why this check was necessary

The optional EMA pseudo-label target contains both a student encoder and a novel
prototype head. Before this fix, `--joint-prototype-init kmeans` refreshed only
the online novel head. If the EMA target was created before that refresh, it kept
random prototypes while the online head used KMeans prototypes. The resulting
experiment did not faithfully test KMeans initialization with an EMA target.

The training code now copies the refreshed online novel head into the EMA target
immediately after every initial or periodic KMeans refresh. The EMA target remains
detached and is still updated by EMA after optimizer steps.

## Matched result

The detector, threshold policy, clustering method, feature, split, checkpoint
selection and test protocol were kept the same. The only intended change was
`joint_prototype_init: random -> kmeans`; both runs used soft EMA pseudo-labels.

| Variant | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection | Candidate purity | Auto-K | K error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random + EMA soft target | 0.5071 | 0.9260 | 0.0682 | 94.57% | 5.10% | 0.3774 | 20 | 20 |
| KMeans + EMA soft target | 0.5247 | 0.9260 | 0.0621 | 95.72% | 5.10% | 0.4348 | 29 | 11 |

## Interpretation

- KMeans initialization improves the ranking score slightly and produces a
  cleaner candidate pool, so it has value for novel-cluster initialization.
- It does not increase unknown rejection at the fixed 95% known-coverage
  operating point, and OSCR becomes worse. Therefore it does not solve the
  known/unknown score overlap.
- The result is evidence for retaining KMeans as an optional clustering aid,
  not evidence for making it the main unknown detector or adding more KMeans
  refresh hyperparameters.
- The earlier random-prototype EMA result is now considered valid for its own
  protocol. Any future KMeans+EMA result must use the synchronized implementation.

## Next comparison

Keep the corrected implementation fixed and return to the strongest supported
mainline: mixed-pool nnPU training with explicit min-class kNN detection and
feature-space clustering. KMeans+EMA should be evaluated only as an isolated
initialization ablation or as a controlled auxiliary component; it should not be
combined with several new losses before its effect is understood.
