# Prototype Refresh Recheck (2026-10-05)

## Purpose

The previous prototype-refresh pilot stopped because of a CPU resource problem. After
limiting KMeans to one BLAS/OpenMP worker, this recheck asks whether periodic KMeans
prototype updates improve the mixed-pool joint discovery objective. It is a controlled
comparison, not a final paper result.

## Protocol

- Dataset: CIFAR-100, random known/novel split 60/40, seed 42.
- Teacher: `runs/hard_proxy_teacher_s42/teacher.pt`, `resnet34`.
- Student: `resnet18`, 1200 train / 300 validation / 1000 test / 1200 discovery samples.
- Training: 3 epochs, batch size 64, CUDA, `num_novel=40`, mixed discovery pool.
- Evaluation: same known-only validation threshold, `mc_samples=4`, KMeans clustering
  with `n_init=10`.
- Baseline: joint discovery with random novel prototypes and no refresh.
- Treatment A: KMeans initialization from the full mixed pool plus refresh every epoch.
- Treatment B: KMeans initialization/refresh from the top-risk 25% candidate subset.

## Results

| Method | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection | Candidate purity | Candidate NMI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Joint random baseline | 0.5051 | 0.9359 | 0.0648 | 0.9704 | 0.0561 | 0.5500 | 0.9235 |
| Full-pool KMeans refresh | 0.5013 | 0.9326 | 0.0463 | 0.9359 | 0.0689 | 0.4091 | 0.8103 |
| Candidate-only KMeans refresh | 0.4993 | 0.9326 | 0.0516 | 0.9243 | 0.0944 | 0.4458 | 0.7245 |

## Interpretation

The implementation now completes and logs three refresh events (`initial`, epoch 2,
and epoch 3). The full-pool treatment slightly improves FPR95 and unknown rejection,
but worsens AUROC, OSCR, known acceptance, and candidate purity. Candidate-only refresh
reduces some contamination relative to full-pool refresh, but it still does not beat the
random baseline on the main ranking metrics and lowers known acceptance further.

These results do not solve the core known/unknown overlap problem. They indicate that
periodically moving prototypes in the current representation is insufficient: the
candidate selector is still contaminated, and KMeans is following a mixed feature
distribution rather than discovering stable novel semantics. Do not enable refresh as
the default method yet.

## Engineering findings

- `initialize_novel_head_kmeans` now uses `threadpool_limits(limits=1)` during fitting.
- `n_init` is explicit; the smoke test uses `n_init=1`, while the formal recheck uses 10.
- A previous run with an unmatched teacher backbone and a previous run with `num_novel=4`
  against a K=40 protocol were rejected by checkpoint/shape checks and are not results.
- The next useful direction is a soft mixed-pool objective that downweights known-like
  samples without assigning hard novel labels, followed by candidate-purity and geometry
  diagnostics. If that also fails, stop tuning KMeans and redesign the representation
  objective rather than changing thresholds.

## Follow-up: residual weighting

To test the soft mixed-pool alternative, a third treatment kept the random novel head
but multiplied joint-loss sample weights by `known_residual_weights`, with floor `0.05`.
Everything else matched the joint-random baseline. The mean residual weight was logged as
`0.886`, `0.854`, and `0.797` over the three epochs, confirming that the mechanism was active.

| Method | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection | Candidate purity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Joint random baseline | 0.5051 | 0.9359 | 0.0648 | 0.9704 | 0.0561 | 0.5500 |
| Random head + residual weighting | 0.4952 | 0.9375 | 0.0629 | 0.9424 | 0.0791 | 0.4697 |

Residual weighting raises rejection mainly by rejecting more samples, while ranking quality,
known acceptance, and candidate purity decline. It is therefore not retained as the main
method and should not be rescued by threshold tuning.
