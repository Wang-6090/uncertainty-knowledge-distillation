# Raw-feature supervised contrastive pilot (2026-09-30)

## Motivation

The detector's Mahalanobis/prototype distances are computed from the backbone `features`, while the historical supervised contrastive loss was applied only to the projection head `proj`. This pilot added an optional supervised contrastive term directly on normalized backbone features, following the supervised contrastive learning objective of Khosla et al. (NeurIPS 2020).

New option:

```text
--alpha-raw-supcon 0.1
```

The default is zero, so historical runs are unchanged.

## Controlled result

Fixed CIFAR-100 semantic-hard 60/40, seed 42, full known training data, pretrained ResNet-34/ResNet-18 teacher/student, batch 32, 5 epochs, the same teacher and mixed pool, and the same Mahalanobis/entropy detector with 95% validation known coverage.

| Method | AUROC | FPR95 | OSCR | Unknown rejection | Accepted-known accuracy |
|---|---:|---:|---:|---:|---:|
| baseline | 0.5669 | 0.9009 | 0.3093 | 6.27% | 46.13% |
| + raw-feature SupCon | 0.5011 | 0.9265 | 0.2223 | 4.82% | 36.91% |

Full-data feature diagnostics also worsened:

- empirical centroid-distance AUROC: `0.5648 -> 0.5173`;
- nearest-known-sample-distance AUROC: `0.5928 -> 0.5627`;
- classifier-prototype-distance AUROC: `0.5972 -> 0.5654`.

## Judgment

Directly applying the same SupCon objective to the detector's backbone feature did not improve known/unknown separation under this protocol. It harmed classification and open-set operating-point metrics, so it is retained only as an optional ablation and is not a default component. This negative result suggests that generic intra-class compactness is insufficient; the representation objective must model the known support boundary or reliable unknown evidence more explicitly.

