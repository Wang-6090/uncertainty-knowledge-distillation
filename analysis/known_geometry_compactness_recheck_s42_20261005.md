# Known Geometry Compactness Recheck (2026-10-05)

## Objective

The core failure remains overlap between known and unknown feature/score
distributions. This round tested labelled-known representation objectives,
without assigning pseudo-labels to the mixed discovery pool.

All runs used CIFAR-100 random 60/40, seed 42, the same ResNet-34 teacher,
pretrained ResNet-18 student, full data, 5 epochs, mixed discovery pool,
automatic known prior, immediate nnPU weight 0.1, and the same detection
protocol: normalized entropy plus explicit min-class kNN, thresholded at 95%
known validation coverage. Only the auxiliary known geometry loss changed.

## Results

| method | AUROC | FPR95 | OSCR | known acceptance | unknown rejection | accepted-known accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| immediate nnPU baseline | 0.7009 | 0.7608 | 0.4134 | 95.83% | 12.78% | 51.32% |
| nnPU + batch center loss, alpha 0.05 | 0.6962 | 0.7757 | 0.4014 | 94.78% | 13.40% | 50.11% |
| nnPU + radius hinge, alpha 0.05, radius 0.2 | 0.6875 | 0.7777 | 0.3902 | 94.72% | 14.73% | 49.39% |
| nnPU + center margin, alpha 0.05, margin 0.1 | 0.6941 | 0.7500 | 0.4187 | 94.00% | 14.98% | 52.84% |

The exact run outputs are in `runs/center_nnpu_s42_detect`,
`runs/radius_nnpu_s42_detect`, and `runs/center_margin_nnpu_s42_detect`.

## Interpretation

The batch center loss reduced within-class spread but did not improve open-set
ranking. The radius hinge behaved similarly and traded known acceptance for
unknown rejection. These objectives are therefore disabled by default and are
retained only as ablations.

The center-margin objective is more promising because it directly compares a
sample with its own and nearest wrong class centers. It improved FPR95,
OSCR, and accepted-known accuracy, but reduced AUROC and known acceptance.
This is mixed evidence, not a solution: the larger rejection rate still has a
known-sample cost.

## Code changes

`novel_discovery/losses.py` now contains optional `supervised_center_loss`,
`supervised_radius_loss`, and `supervised_center_margin_loss`. `train.py`
exposes them through `--alpha-center`, `--alpha-radius`, and
`--alpha-center-margin`; all defaults are zero, preserving the established
main pipeline. Unit tests cover finite values, gradients, and compactness
behavior.

## Next test

Run only a lower-weight center-margin treatment, alpha 0.02 and margin 0.1,
under the same protocol. Keep it only if it improves at least one ranking
metric without reducing known acceptance by more than the baseline noise. If
the trade-off remains, stop adding known-only geometry losses and return to
the more fundamental mixed-pool representation design.
