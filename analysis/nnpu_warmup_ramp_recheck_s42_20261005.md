# Mixed-pool nnPU warm-up/ramp recheck

Date: 2026-10-05  
Dataset: CIFAR-100 random 60/40 protocol split, seed 42  
Protocol: full known training partition, mixed discovery pool, pretrained
ResNet-34 teacher, pretrained ResNet-18 student, 5 epochs, batch size 64,
MC=4, known-only 95% coverage threshold, explicit min-class kNN detector.

## Motivation

The mixed-pool nnPU objective has the strongest existing evidence in this
repository, but applying it at full weight from the first student epoch may
interact badly with an unstable early representation. This pilot adds a
linear schedule without changing the final nnPU weight:

- immediate nnPU: `alpha_discovery_uncertainty_pu=0.1` from epoch 1;
- scheduled nnPU: weight 0 for epoch 1, 0.5 for epoch 2, and 1.0 from epoch 3,
  using `--discovery-uncertainty-pu-warmup-epochs 1`
  and `--discovery-uncertainty-pu-ramp-epochs 2`.

The new implementation records `uncertainty_pu_weight` in each epoch history.
The schedule only multiplies the nnPU term; it does not alter the known prior,
discovery pool, detector, threshold, or test labels.

## Result

| Variant | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection | Accepted-known accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Immediate nnPU | 0.7009 | 0.7608 | 0.4134 | 95.83% | 12.78% | 51.32% |
| nnPU warm-up/ramp | 0.6970 | 0.7452 | 0.3987 | 94.12% | 14.40% | 50.15% |

## Interpretation

The schedule improves FPR95 and the fixed operating-point unknown rejection,
but lowers AUROC, OSCR, known acceptance, and accepted-known accuracy. It is
therefore not a clearly better training method. The appropriate conclusion is
that delaying nnPU may improve one operating point while changing the ranking
and coverage trade-off; it does not demonstrate improved representation
separation.

Keep the feature in the code as a default-off ablation and do not promote the
ramp schedule to the main pipeline without a multi-seed confirmation. The
current main reference remains immediate mixed-pool nnPU plus explicit
min-class kNN detection. The failed attempt using a teacher from a different
split is excluded from the result; checkpoint split validation correctly
rejected it.

Artifacts:

- `runs/full_mixed_nnpu_ramp_s42/`
- `runs/full_mixed_nnpu_ramp_s42_knn_detect/`
