# Strong Pseudo-Unknown Generator Comparison

The strong generator was evaluated against the legacy generator using the same CIFAR-100 split, pretrained ResNet-18, 15 epochs, `alpha_pseudo=0.1`, `alpha_proto=0.1`, seed 42, `mc_samples=8`, and fixed `entropy_proto` scoring.

| Version | AUROC | FPR95 | Known accuracy | Unknown reject | Cluster ACC | NMI | ARI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy | 0.6008 | 0.8566 | 0.4609 | 0.0627 | 0.2086 | 0.5599 | 0.0549 |
| Strong | 0.5812 | 0.8655 | 0.4183 | 0.0637 | 0.1982 | 0.5407 | 0.0455 |

## Conclusion

The strong image perturbation did not improve the open-set detector. The unknown rejection rate changed by only `+0.0010`, while AUROC, FPR95, known classification accuracy, and clustering metrics became worse. The added erasing, smoothing, noise, and feature perturbation therefore should not be used as the main method.

The default command-line mode is restored to `legacy`. The strong implementation remains available through the explicit `--pseudo-mode strong` option for reproducibility and negative-result reporting.

## Next step

Keep the legacy generator as the baseline and improve the uncertainty branch itself: calibrate the uncertainty score on known validation data, compare entropy, epistemic uncertainty, aleatoric uncertainty, and prototype distance under the same threshold policy, then test a small learned or validation-fitted score fusion. This directly targets the current bottleneck—unknown detection—without further damaging the feature representation.
