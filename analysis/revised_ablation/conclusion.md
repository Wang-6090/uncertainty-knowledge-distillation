# Revised Ablation Conclusion

The five runs use the same CIFAR-100 60/40 split, seed, pretrained teacher,
training length, fixed normalized entropy plus Mahalanobis score, and oracle
K=40 clustering. Therefore this is a controlled one-seed ablation, not yet a
final statistical claim.

| Run | AUROC | FPR95 | Known acc | Unknown reject | Cluster ACC | NMI | ARI |
|---|---:|---:|---:|---:|---:|---:|---:|
| A CE | 0.5939 | 0.8812 | 0.4527 | 0.0673 | 0.1820 | 0.5127 | 0.0535 |
| B standard KL | 0.5955 | 0.8632 | 0.4093 | 0.0628 | 0.2022 | 0.5190 | 0.0571 |
| C uncertainty KL | 0.5898 | 0.8613 | 0.4290 | 0.0508 | 0.2117 | 0.5387 | 0.0628 |
| D feature KD | 0.5846 | 0.8790 | 0.4113 | 0.0653 | 0.1865 | 0.5083 | 0.0483 |
| E full | 0.5977 | 0.8713 | 0.4375 | 0.0747 | 0.1768 | 0.4759 | 0.0398 |

## Interpretation

- B improves AUROC and FPR95 slightly over CE, but lowers known accuracy.
- C does not validate uncertainty-weighted KD for AUROC or unknown rejection.
  It gives the best clustering metrics in this one seed.
- D does not validate the current feature-KD weight or feature target.
- E gives the best AUROC and unknown rejection in this run, but it gives the
  worst clustering metrics. Detection and clustering objectives conflict.

The project should not claim that every added module helps. The defensible
current claim is that uncertainty-weighted KD is implemented and shows a
representation/clustering tendency, but its open-set detection benefit is not
yet established. Three-seed repetition is required before selecting the final
method.
