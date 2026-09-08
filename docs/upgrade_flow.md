# Upgraded Experimental Flow

## Current baseline

- Dataset: CIFAR-100
- Known / novel split: 60 / 40
- Backbone: ResNet-18
- Score: entropy + prototype distance
- Prototype loss weight: alpha_proto = 0.1

## Upgrade order

### 1. Pretrained backbone

Run:

    powershell -ExecutionPolicy Bypass -File .\scripts\run_pretrained_compare.ps1

This keeps the 5000 / 1000 / 2000 sample setting and only enables ImageNet initialization.

### 2. Full available data

Run:

    powershell -ExecutionPolicy Bypass -File .\scripts\run_full_data_compare.ps1

This uses the pretrained ResNet-18 and removes the train, validation, and test limits.

### 3. Pseudo-unknown training

Quick toy/fake verification:

    powershell -ExecutionPolicy Bypass -File .\scripts\run_pseudo_toy.ps1

Formal CIFAR-100 experiment:

    powershell -ExecutionPolicy Bypass -File .\scripts\run_pseudo_cifar.ps1

The pseudo-unknown batch is formed by mixing two known images and adding mild noise.
The model is trained to assign high uncertainty and low class confidence to those
synthetic samples. The pseudo loss is disabled by default and is enabled with:

    --alpha-pseudo 0.1

## How to compare

For every experiment, record:

- known classification accuracy
- AUROC
- FPR95
- unknown reject rate
- cluster ACC / NMI / ARI

The most important comparison is between the same settings with and without
pseudo-unknown training.
