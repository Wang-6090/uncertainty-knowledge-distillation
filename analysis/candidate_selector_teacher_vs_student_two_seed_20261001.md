# Teacher versus student candidate selector: two-seed synthesis

The two full-data replications use the same training and detection protocol;
only the candidate selector source changes. The fair comparison is the
retrospective matched-known-coverage diagnostic, while raw validation-threshold
metrics are reported separately because finite test subsets can have different
known acceptance rates.

## Matched-coverage results

| Metric | Student mean | Teacher mean | Teacher - student |
| --- | ---: | ---: | ---: |
| Unknown rejection at about 95% known acceptance | 14.44% | 15.53% | +1.09 pp |
| Accepted-known accuracy | 50.26% | 51.29% | +1.03 pp |
| Candidate purity | 0.6747 | 0.6790 | +0.0043 |
| Candidate unknown NMI | 0.7389 | 0.7851 | +0.0462 |

The teacher selector has a small average advantage at the matched operating
point and a clearer average advantage in candidate-cluster NMI. However, the
unknown-rejection change is not consistent: teacher is better for seed 123
(16.38% vs 12.16%) but worse for seed 42 (14.68% vs 16.71%). AUROC is also not
consistently better (teacher is lower for seed 123 and slightly higher for
seed 42).

## Interpretation

The frozen teacher is useful as a stable candidate-selection ablation and may
be preferable when the downstream objective is candidate-pool purity or
clustering. It does not provide reliable evidence of a better global
known/unknown separator. The per-class seed-123 audit also showed that the
teacher selector reduced false acceptance for only 9 of 40 novel classes,
while increasing it for 16 classes.

The most defensible conclusion is therefore:

1. Keep `--discovery-selection-model teacher` as an optional conservative
   selector and report it separately.
2. Keep `student` as the current primary selector for the main nnPU baseline,
   so the main line remains consistent with the established three-seed nnPU
   comparison.
3. Stop spending the next iteration on selector-source or threshold tuning.
   The remaining bottleneck is representation overlap, so the next code change
   should be a controlled representation-learning objective or a correctly
   isolated mixed-pool GCD objective.

