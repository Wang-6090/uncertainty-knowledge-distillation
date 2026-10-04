# Mixed-pool nnPU uncertainty pilot (2026-09-30)

## Question and predeclared comparison

Can non-negative positive-unlabeled (nnPU) risk training of the uncertainty
head improve known/unknown ranking when the discovery pool contains both
known and withheld-class images, without labeling every pool image as unknown?

The single intended treatment change is
`alpha_discovery_uncertainty_pu: 0.0 -> 0.1`. Both runs use the same seed 42,
CIFAR-100 random 60/40 class split, CIFAR-100 60-class ResNet-34 teacher,
pretrained ResNet-18 student, teacher checkpoint, sample limits (1200/300/1000
known train/validation/test and 1200 discovery), batch size 32, three epochs,
optimizer, remaining loss weights, mixed-pool construction, and detector.
The PU known prior is fixed at 0.2. Saved training configs were compared:
after excluding output paths and the selected best validation accuracy, the
only difference is the PU loss weight.

Detector evaluation is fixed to `normalized_entropy_mahalanobis`, four MC
samples, and a threshold calibrated to 95% known coverage using the known
validation split. Test labels are used only for the final metrics. Clustering
is skipped because it is not part of this detector hypothesis.

## Method and protocol

The uncertainty head outputs novelty probability `u`; the PU objective uses
knownness logit `g = log((1-u)/u)`. Labeled known samples form positive set P,
and the mixed discovery pool is unlabeled set U. The objective is

`R_nnPU = pi * E_P[softplus(-g)] + max(0, E_U[softplus(g)] - pi * E_P[softplus(g)])`,

where `pi=0.2` is the assumed known proportion in U. This is the non-negative
PU risk idea of Kiryo et al., NeurIPS 2017; it does not assign per-example
unknown targets to U. The two augmented views' uncertainty scores are averaged
before computing U risk.

The benchmark builder uses the known/novel class split to assemble a controlled
mixed pool, but `TwoViewDataset` passes only images to the training loop. A
post-run protocol audit (not used for training or model selection) found 253
known and 947 novel images among the 1200 pool samples, giving known fraction
0.2108, close to the fixed 0.2 prior. Thus this is a label-hidden training
objective under a benchmark-defined mixture, not a fully natural online data
stream.

## Results

| Metric | Mixed baseline (PU weight 0) | nnPU (weight 0.1) | Change |
|---|---:|---:|---:|
| AUROC | 0.5445 | 0.5819 | +0.0373 |
| AUPR | 0.4226 | 0.4693 | +0.0467 |
| FPR95 | 0.9289 | 0.8992 | -0.0298 |
| OSCR | 0.1543 | 0.1483 | -0.0060 |
| Test known acceptance | 0.9653 | 0.9537 | -0.0116 |
| Test unknown rejection | 0.0608 | 0.0886 | +0.0278 |
| Known classification accuracy (all known) | 0.2314 | 0.2083 | -0.0231 |
| Accepted-known classification accuracy | 0.2397 | 0.2184 | -0.0214 |

The PU loss was active (final-epoch unweighted value 0.5053; baseline 0). The
ranking metrics improve in this one paired seed, and the calibrated operating
point rejects more unknowns. However, known acceptance and known classification
accuracy decline, and OSCR also declines slightly. This is a promising but
mixed single-seed signal, not evidence that the core overlap is solved.

## Initial pilot and protocol correction

The first 0/0.1 pair used a previously trained seed-42 teacher and appeared
promising, but it was only a one-seed pilot. A later attempted multi-seed
extension reused that teacher for seeds 43/44, which risks teacher exposure to
examples outside those seeds' training partitions. Those initial cross-seed
student runs are exploratory and are **not** included in the clean summary
below. To resolve the issue, all three seeds were rerun with a teacher trained
on that same seed's known train/validation split, using the same 5-epoch
ResNet-34 teacher recipe and 1200/300 sample limits. Within each seed, the
baseline and PU student share the exact same teacher checkpoint.

## Clean paired three-seed results: gradient-corrected nnPU

The initial implementation used a plain clamp for negative risk. A method-
fidelity audit caught that this removes the gradient in the negative-risk
region, unlike the gradient correction in Kiryo et al. The clamp results above
are therefore only exploratory and are superseded by the results in this
section. The corrected objective was retrained for all three seeds. All rows
use per-seed teachers; PU loss was active in every treatment, and config diffs
were checked: the only substantive pairwise difference is
`alpha_discovery_uncertainty_pu` (0 vs 0.1).

| Seed | AUROC base → PU | AUPR base → PU | FPR95 base → PU | OSCR base → PU | Known acceptance base → PU | Unknown rejection base → PU | Known accuracy base → PU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.5581 → 0.5775 | 0.4236 → 0.4391 | 0.9041 → 0.9074 | 0.1557 → 0.1822 | 0.9521 → 0.9570 | 0.0506 → 0.0304 | 0.2231 → 0.2479 |
| 43 | 0.5271 → 0.5417 | 0.3994 → 0.4160 | 0.9193 → 0.9209 | 0.1740 → 0.1664 | 0.9588 → 0.9325 | 0.0382 → 0.0585 | 0.2488 → 0.2455 |
| 44 | 0.5691 → 0.5810 | 0.4543 → 0.4418 | 0.9064 → 0.8729 | 0.1431 → 0.1296 | 0.9716 → 0.9632 | 0.0522 → 0.0348 | 0.2090 → 0.1856 |
| Mean paired change | **+0.0153** | **+0.0065** | **-0.0095** | **+0.0018** | **-0.0099** | **-0.0058** | **-0.0006** |

Known accuracy is all-known test classification accuracy; accepted-known
accuracy changes were +0.0247, +0.0038, and -0.0224 for seeds 42, 43, and 44,
respectively (mean +0.0020). AUROC improves on all three seeds and mean FPR95
slightly improves, but AUPR and OSCR are mixed. Unknown rejection falls on two
seeds and on average; known acceptance is also lower on average. The resulting
evidence is **mixed**: ranking improves, but the desired operating-point
rejection does not. It does not solve the core problem.

As a retrospective diagnostic only, test labels were used to recompute
unknown rejection at exactly 95% test-known coverage. This is not a deployable
threshold and was not used to select the method. At matched test coverage:

| Seed | Baseline unknown reject | Corrected nnPU unknown reject |
|---:|---:|---:|
| 42 | 0.0582 | 0.0354 |
| 43 | 0.0560 | 0.0483 |
| 44 | 0.0597 | 0.0373 |
| Mean | **0.0580** | **0.0403** |

Thus the modest AUROC gain did not translate into more reliable rejection at
the same known coverage. Do not increase the loss weight based on ranking
metrics alone.

## Decision and next test

- Keep the corrected PU uncertainty objective as an experimental option, but
  do not claim it improves the primary open-set outcome.
- Do not tune its weight based on these test metrics. If pursued, run a
  separately predeclared coefficient comparison on development/validation
  data, then evaluate once on untouched test data.
- The low unknown rejection remains the major limitation (mean 4.70% baseline
  vs 4.12% PU with validation-calibrated thresholds; at retrospective exact
  95% test-known coverage, 5.80% vs 4.03%). Further work should prioritize
  representation-level diagnostics or a stronger open-world category-discovery
  objective, while retaining PU as an auxiliary uncertainty objective.
- The 0.2 prior is an assumption. Any prior sensitivity study must use a
  documented pool-composition protocol and must not select settings with test
  labels.

## Implementation checks

- Added `nnpu_known_uncertainty_loss` with the negative-risk gradient
  correction, the `--alpha-discovery-uncertainty-pu`
  and `--discovery-uncertainty-known-prior` options, and enforcement that this
  loss is used only with `--discovery-pool-mode mixed`.
- Unit tests cover non-negative finite risk, gradients, prior validation, and
  nonzero training dispatch on an unlabeled two-view pool.
- Clean teacher/student pilot commands completed on CUDA. Full-suite and
  static checks passed: `python -m pytest -q` -> 114 passed;
  `compileall` and `git diff --check` passed.
