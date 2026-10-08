# Uncertainty 2x2 pilot (seed 42)

## Purpose

This pilot separates the two newly combined student-training terms:

1. mixed-pool uncertainty nnPU loss;
2. uncertainty-weighted feature margin.

The goal is to identify whether either term reduces known/unknown score
overlap. This is a component screen, not a final result.

## Controlled protocol

- Dataset: CIFAR-100 semantic-isolated 60/40 split.
- Seed: 42 for the data split and model initialization.
- Shared teacher: ResNet-34, ImageNet initialization, 3-epoch pilot run.
- Student: ResNet-18, ImageNet initialization.
- Training data: 1200 known samples; validation: 300; test: 1000.
- Discovery pool: matched mixed pool of 5400 samples, known prior 0.2.
- Detection: frozen student representation, `support_augmented` nnPU rejector,
  `nu_corrected` risk, and known-only 95% coverage calibration.
- The test labels are used only for reporting, never for fitting or threshold
  selection. Clustering is skipped in this screen.

## Completed 3-epoch pilot

| Arm | AUROC | FPR95 | OSCR | Known accept | Unknown reject |
|---|---:|---:|---:|---:|---:|
| baseline | 0.6758 | 0.8020 | 0.1189 | 95.51% | 8.27% |
| nnPU only | 0.7191 | 0.7072 | 0.2170 | 94.68% | 17.79% |
| feature margin only | 0.7232 | 0.7488 | 0.1821 | 97.50% | 9.52% |
| combined | 0.7126 | 0.7404 | 0.1811 | 96.01% | 13.28% |

Interpretation: nnPU is the most promising single component for the operating
point: it adds 9.52 percentage points of unknown rejection over baseline while
slightly lowering known acceptance. Feature margin improves ranking metrics,
but adds only 1.25 points of unknown rejection. The combined arm does not show
synergy at 3 epochs; this may be optimization interference rather than proof
that the combination is invalid.

## Status and limits

The 3-epoch result is not sufficient to select a final method. It uses a
limited training/test sample and one seed. In particular, it cannot establish
that nnPU solves the representation-overlap problem. A 10-epoch paired run is
being used to test whether the ordering survives a longer optimization budget.
If nnPU remains better and combined remains worse, nnPU should become the main
ablation while feature margin remains optional. If combined catches up, the
next test should vary loss ramp-up or weights rather than changing the
detector threshold.
