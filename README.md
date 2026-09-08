# Uncertainty-Aware Knowledge Distillation for Novel Class Discovery

This repository contains the code and experiment summaries for the project
"Uncertainty-Aware Knowledge Distillation for Novel Class Discovery".

## What is included

- CIFAR-100 and ImageFolder open-set dataset loaders
- teacher/student training with standard and uncertainty-weighted distillation
- uncertainty estimation with MC Dropout and an auxiliary uncertainty head
- prototype, entropy, and Mahalanobis open-set scoring
- projection-space clustering for unknown samples
- controlled ablation scripts and result summaries

## Current status

The current implementation is a reproducible open-set recognition and unknown
clustering pipeline. It is not yet a full end-to-end novel class discovery
system, but it already supports controlled comparisons of the main ideas.

The main ablation was run on CIFAR-100 with a fixed 60/40 known-novel split,
one seed, one teacher, and one score policy.

| Run | AUROC | FPR95 | Known acc | Unknown reject | Cluster ACC | NMI | ARI |
|---|---:|---:|---:|---:|---:|---:|---:|
| CE | 0.5939 | 0.8812 | 0.4527 | 0.0673 | 0.1820 | 0.5127 | 0.0535 |
| Standard KD | 0.5955 | 0.8632 | 0.4093 | 0.0628 | 0.2022 | 0.5190 | 0.0571 |
| Uncertainty KD | 0.5898 | 0.8613 | 0.4290 | 0.0508 | 0.2117 | 0.5387 | 0.0628 |
| Feature KD | 0.5846 | 0.8790 | 0.4113 | 0.0653 | 0.1865 | 0.5083 | 0.0483 |
| Full model | 0.5977 | 0.8713 | 0.4375 | 0.0747 | 0.1768 | 0.4759 | 0.0398 |

## Interpretation

- Standard KD slightly improves open-set detection over CE.
- Uncertainty-weighted KD helps clustering more than detection in this run.
- Feature KD did not help under the current setting.
- The full model gave the best AUROC and unknown rejection in this seed, but
  the worst clustering metrics, so detection and clustering still conflict.

## Main files

- `train.py` - entry script for training and discovery
- `novel_discovery/` - models, losses, pipeline, metrics, and data helpers
- `scripts/` - runnable experiment scripts
- `analysis/revised_ablation/` - result summaries for the main controlled ablation
- `docs/revised_method.md` - current method description
- `docs/revised_experiment_plan.md` - revised experiment plan

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Inspect the data split:

```bash
python train.py inspect_data --dataset cifar100 --data-root ./data --download
```

Run the revised controlled ablation:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_revised_ablation.ps1
```

Run the lightweight MobileNet compression test:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_mobilenet_compression.ps1
```

## Notes

- The repository intentionally excludes raw datasets, cached weights, and
  experiment outputs under `runs/`.
- The analysis folder keeps compact JSON and Markdown summaries that are safe
  to version-control.
- The current results are single-seed results and should be treated as
  preliminary until multi-seed runs are finished.
