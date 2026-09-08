# Classification-Error-Aware Uncertainty Comparison

The experiment changed the uncertainty-head target from `1 - true-class confidence` to a binary target indicating whether the current argmax prediction was wrong. The CIFAR-100 split, pretrained ResNet-18, 15 epochs, `alpha_proto=0.1`, no pseudo-unknown loss, and evaluation protocol were otherwise unchanged.

| Score mode | AUROC | FPR95 | Known accuracy | Unknown reject | Cluster ACC | NMI | ARI |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 0.5996 | 0.8435 | 0.4640 | 0.0681 | 0.1812 | 0.5003 | 0.0301 |
| entropy_aleatoric | 0.5981 | 0.8446 | 0.4638 | 0.0734 | 0.1610 | 0.4782 | 0.0176 |
| entropy_proto | 0.5987 | 0.8529 | 0.4638 | 0.0696 | 0.1805 | 0.4962 | 0.0260 |

## Conclusion

The error-aware uncertainty target does not improve unknown detection over the original fair baseline (`AUROC=0.6149`, `FPR95=0.8302`). The small gain in known classification accuracy is not enough to justify replacing the baseline. The likely reason is that the target uses the current discrete argmax error, which is noisy early in training and does not represent unknownness; it also cannot be optimized through the classifier decision.

The implementation remains available through `--uncertainty-target-mode classification_error`, while the default remains `confidence`.

## Next step

Use a continuous, calibration-oriented target for the uncertainty head, such as normalized cross-entropy or a detached prediction margin, and evaluate it with the same fixed score and threshold protocol. Do not combine this change with a new pseudo-unknown generator in the next experiment.
