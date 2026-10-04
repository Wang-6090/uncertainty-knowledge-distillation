# Prototype Repulsion Comparison

Protocol: CIFAR-100 60/40, seed `3407`, pretrained ResNet-34 teacher and ResNet-18 student, 3 epochs, 1200 train / 300 validation / 1000 test samples, standard KL KD, identical `normalized_entropy_mahalanobis` detector, oracle-K clustering.

| Method | AUROC | FPR95 | Known accuracy | Known accept rate | Unknown reject rate | Candidate purity | Candidate cluster ACC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline, `alpha_proto_repulsion=0` | 0.5335 | 0.9289 | 0.2264 | 0.9074 | 0.0608 | 0.3000 | 0.5500 |
| Prototype repulsion, `alpha=0.01` | **0.5536** | **0.9008** | **0.2529** | 0.9455 | 0.0582 | **0.4107** | 0.6250 |
| Prototype repulsion, `alpha=0.05` | **0.5459** | **0.9041** | 0.2116 | **0.9504** | 0.0481 | **0.3878** | **0.7143** |

The auxiliary known-validation accuracy during training was `0.2167` for the baseline, `0.2733` for `alpha=0.01`, and `0.2467` for `alpha=0.05`. The repulsion term was nonzero throughout training, so it was active rather than a no-op.

## Interpretation

- AUROC increased by `0.0124` and FPR95 decreased by `0.0248`.
- Candidate purity and clustering improved, consistent with better separation among known prototypes.
- The 95th-percentile operating point changed: the repulsion model accepted more samples as known, so its unknown reject rate decreased from `0.0608` to `0.0481`.
- The lighter `alpha=0.01` setting gives a better trade-off in this run: it improves AUROC, FPR95, known accuracy, and candidate purity while keeping unknown reject rate close to baseline (`0.0582` vs `0.0608`).
- This is a partial improvement, not a solved unknown-detection result. Unknown rejection must also be compared at matched known coverage; otherwise score-scale changes can obscure the trade-off.
- This is one seed and only three epochs. It is evidence for a follow-up experiment, not a final method claim.

Runs:

- `runs/proto_compare_s3407_baseline`
- `runs/proto_compare_s3407_baseline_detect`
- `runs/proto_compare_s3407_repulsion`
- `runs/proto_compare_s3407_repulsion_detect`
- `runs/proto_compare_s3407_repulsion001`
- `runs/proto_compare_s3407_repulsion001_detect`
