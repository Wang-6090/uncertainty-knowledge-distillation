# Discovery Pool Multi-seed Comparison

## Protocol

- Dataset: CIFAR-100, 60 known classes / 40 novel classes
- Seeds: 42 and 123
- Teacher: pretrained ResNet-34
- Student: pretrained ResNet-18
- Training epochs: 15
- Fixed open-set score: normalized entropy + diagonal Mahalanobis distance
- F: existing full representation model plus an unlabeled novel-only
  discovery pool and NT-Xent two-view contrastive learning
- All clustering metrics below use the oracle-K protocol unless marked
  auto-K

## Main result

| Method | AUROC | AUPR | FPR95 | OSCR | Known acc | Unknown reject | Cluster ACC | NMI | ARI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E: full representation | 0.6027 ± 0.0169 | unavailable | 0.8655 ± 0.0255 | unavailable | 0.4218 ± 0.0040 | 0.0660 ± 0.0141 | 0.1895 ± 0.0101 | 0.5073 ± 0.0238 | 0.0514 ± 0.0037 |
| F: E + discovery pool NT-Xent | 0.6397 ± 0.0031 | 0.5021 ± 0.0063 | 0.8413 ± 0.0016 | 0.4084 ± 0.0069 | 0.5221 ± 0.0107 | 0.0864 ± 0.0062 | 0.2058 ± 0.0070 | 0.5053 ± 0.0101 | 0.0679 ± 0.0093 |
| Change F - E | +0.0370 | new metric | -0.0242 | new metric | +0.1003 | +0.0204 | +0.0163 | -0.0020 | +0.0165 |

The improvement is consistent across both seeds. The discovery-pool
objective improves the known representation and open-set detection, while
the clustering improvement is modest. AUPR and OSCR were not recorded by the
old E reports, so they should be added when rerunning the E baseline under
the new evaluator.

## Auto-K result

| Method | AUROC | FPR95 | Known acc | Cluster ACC | NMI | ARI | Estimated K |
|---|---:|---:|---:|---:|---:|---:|---:|
| E: full representation | 0.6027 ± 0.0169 | 0.8655 ± 0.0255 | 0.4218 ± 0.0040 | 0.1343 ± 0.0112 | 0.3700 ± 0.0076 | 0.0363 ± 0.0007 | 15.0 ± 1.4 |
| F: E + discovery pool NT-Xent | 0.6397 ± 0.0031 | 0.8413 ± 0.0016 | 0.5221 ± 0.0107 | 0.1258 ± 0.0008 | 0.3389 ± 0.0103 | 0.0426 ± 0.0067 | 11.5 ± 0.7 |

The detection metrics remain improved under auto-K, but automatic cluster
number estimation becomes worse: the true number of novel classes is 40,
whereas F estimates about 12. This means the current silhouette-based
auto-K procedure is not ready to support a claim of automatic novel-class
number discovery.

## Candidate-pool diagnosis

For the F oracle-K runs, the predicted novel candidate pool contains many
known samples rejected by the detector:

- Seed 42: 657 candidates, 363 true unknown samples, 294 false rejects
- Seed 123: 631 candidates, 328 true unknown samples, 303 false rejects

Thus only about half of the clustering candidates are genuinely novel. The
main remaining bottleneck is not only the clustering algorithm; it is the
quality of the unknown filtering stage. The next improvement should reduce
known-sample contamination before adding more clustering losses.

## Reproducibility

The full runner is:

    powershell -ExecutionPolicy Bypass -File .\scripts\run_discovery_pool_multiseed.ps1

The generated run directories are:

- runs/revised_ms_s42_F_discovery_pool
- runs/revised_ms_s123_F_discovery_pool
- runs/revised_ms_s42_F_discovery_pool_detect_oracle
- runs/revised_ms_s123_F_discovery_pool_detect_oracle

The aggregate summary is also written to
analysis/revised_multiseed/multiseed_summary.md.
