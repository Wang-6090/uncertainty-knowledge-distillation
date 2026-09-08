# Margin-Based Uncertainty Comparison

The uncertainty head was trained with a continuous target derived from the difference between the strongest competing logit and the true-class logit. The experiment used CIFAR-100 60/40, pretrained ResNet-18, 15 epochs, `alpha_proto=0.1`, no pseudo-unknown loss, and the same open-set evaluation protocol as the baseline.

| Score mode | AUROC | FPR95 | Known accuracy | Unknown reject | Cluster ACC | NMI | ARI |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 0.5908 | 0.8537 | 0.4449 | 0.0549 | 0.1959 | 0.5346 | 0.0291 |
| entropy_aleatoric | 0.5902 | 0.8547 | 0.4451 | 0.0605 | 0.1876 | 0.5223 | 0.0269 |
| entropy_proto | 0.5894 | 0.8560 | 0.4449 | 0.0524 | 0.1968 | 0.5383 | 0.0286 |

## Conclusion

The margin target does not improve the original fair baseline (`AUROC=0.6149`, `FPR95=0.8302`). It also lowers known-class accuracy to about `0.445`. Therefore it should remain an experimental negative result, not the main method.

The results suggest that changing the uncertainty-head target alone is insufficient. The current open-set detector is limited primarily by weak separation in the learned representation and by the mismatch between known-only training and real novel classes.

## Next step

Stop adding uncertainty targets for now. Use the validated baseline (`confidence` target, `legacy` pseudo mode) and run a representation-focused experiment: train with CIFAR-specific normalization and a longer schedule, then compare the feature distance and open-set metrics. Keep the margin implementation for reproducibility.
