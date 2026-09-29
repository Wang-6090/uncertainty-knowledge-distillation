# Prototype Repulsion Recheck (2026-09-29)

## Protocol

CIFAR-100, fixed 60/40 class split, pretrained ResNet-34 teacher and ResNet-18
student, standard KD, seed-specific teacher checkpoints, 1,200 known training
images, 300 known validation images, 1,000 open-test images, 3 student epochs.
The paired conditions differ only in `alpha_proto_repulsion` (`0` vs `0.01`).
Both use the same `normalized_entropy_mahalanobis` detector and global
known-validation threshold policy. These are small diagnostic runs, not the
full-data final benchmark.

## Results

| Seed | Repulsion | AUROC | FPR95 | Known accept | Unknown reject | Known accuracy |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | off | 0.4698 | 0.9408 | 0.9457 | 0.0485 | 0.1464 |
| 42 | 0.01 | 0.4690 | 0.9474 | 0.9572 | 0.0281 | 0.1612 |
| 123 | off | 0.5210 | 0.9224 | 0.9439 | 0.0457 | 0.1766 |
| 123 | 0.01 | 0.5438 | 0.8944 | 0.9538 | 0.0330 | 0.2228 |
| 3407* | off | 0.5335 | 0.9289 | 0.9074 | 0.0608 | 0.2264 |
| 3407* | 0.01 | 0.5536 | 0.9008 | 0.9455 | 0.0582 | 0.2529 |

`*` The seed-3407 pair is the already-existing run from the previous day, with
the same sample budget and epochs. It is included for directional context; its
actual validation coverage differs from the other two seeds, so aggregate
work-point comparisons are approximate.

Across the three runs, mean AUROC is `0.5081` without repulsion and `0.5221`
with repulsion. Mean FPR95 is `0.9307` vs `0.9142`. However, unknown rejection
falls for every paired seed: the three-run mean changes from `0.0516` to
`0.0398`. The seed-42 AUROC is effectively unchanged and its FPR95 worsens;
the positive ranking signal is driven mainly by seeds 123 and 3407.

## Interpretation

The repulsion loss may improve known-class representation geometry and ranking
in some seeds, but this evidence does not show a reliable improvement at the
operating point. The lower unknown rejection rate is partly coupled to higher
known acceptance, so the method does not solve the project's main failure
mode. Do not enable it by default or present it as a proven fix. Retain it as
an ablation candidate only.

This run also reinforces that training on only 1,200 images and 3 epochs is
too small for a final claim. Next steps should be:

1. Run the paired baseline/repulsion comparison on the full known training
   split and at least three seeds, with stratified known train/validation
   splits held fixed across the pair.
2. Report AUROC/AUPR and FPR95 alongside unknown rejection at a *matched known
   coverage* operating point, plus known accuracy and candidate purity.
3. Inspect nearest-known-class prototype distance distributions for known and
   unknown samples. If the distributions still overlap, stop tuning the
   repulsion coefficient and move to stronger pretrained/metric-learning
   representations or a correctly formulated GCD objective for mixed pools.

Run directories: `runs/proto_compare_s42_*`, `runs/proto_compare_s123_*`, and
`runs/proto_compare_s3407_*`.
