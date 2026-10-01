# Full-data mixed nnPU recheck (2026-10-01)

## Purpose and protocol

The previous mixed-pool candidate/Objectosphere attempts failed or only
partially recovered the baseline. This experiment tested the existing
positive-unlabeled uncertainty objective with the full training budget.

Fixed conditions:

- CIFAR-100 random 60/40 split, seed 123;
- same per-seed ResNet-34 teacher checkpoint;
- pretrained ResNet-18 student, image size 64, batch size 64, 5 epochs;
- complete known training partition and complete open test set;
- mixed discovery pool with measured known fraction `0.212598`;
- normalized entropy plus Mahalanobis score, MC=4;
- threshold calibrated from known validation samples at 95% known coverage;
- clustering skipped to isolate detection.

The only substantive training change was
`alpha_discovery_uncertainty_pu: 0.0 -> 0.1`. The treatment used the actual
pool prior automatically recorded by the code. Configuration diff audit
confirmed that no other training hyperparameter changed.

## Results

| Metric | Full mixed baseline | Full mixed nnPU | Change |
| --- | ---: | ---: | ---: |
| Best validation known accuracy | 42.63% | 49.77% | +7.14 pp |
| AUROC | 0.6077 | 0.6533 | +0.0456 |
| AUPR | 0.4695 | 0.5322 | +0.0627 |
| FPR95 | 0.8497 | 0.8167 | -0.0330 |
| OSCR | 0.3137 | 0.3728 | +0.0591 |
| Known acceptance | 94.22% | 94.90% | +0.68 pp |
| Unknown rejection | 7.88% | 12.23% | +4.35 pp |
| Accepted-known accuracy | 44.47% | 50.61% | +6.14 pp |

This is the strongest mixed-pool result in the current full-data audit. It is
still one seed and is not a final paper claim.

## Distribution diagnostics

The improvement is not uniform across every feature statistic:

| Diagnostic | Baseline overlap | nnPU overlap | Interpretation |
| --- | ---: | ---: | --- |
| Main detection score | 0.8303 | 0.7796 | Better ranking separation |
| Entropy | 0.7873 | 0.7618 | Better uncertainty separation |
| Epistemic uncertainty | 0.8282 | 0.7969 | Better MC uncertainty separation |
| Feature norm | 0.7809 | 0.8093 | Worse; no norm-space solution |
| Mahalanobis | 0.9367 | 0.9165 | Slightly better but still highly overlapped |
| Head uncertainty | 0.8270 | 0.8398 | Slightly worse |

The accurate conclusion is therefore that nnPU improves the learned
uncertainty/entropy ranking and the calibrated operating point under this
protocol. It does not prove that the entire embedding space has separated.

## Decision

- Promote full-data nnPU uncertainty to the most promising current mixed-pool
  direction, but keep it optional until a second independently trained seed is
  checked.
- Do not combine it with Objectosphere or candidate-only feature repulsion in
  the next test; those additions have failed in mixed pilots and would destroy
  one-factor attribution.
- Next test: repeat the exact full-data baseline/nnPU pair with seed 42 and its
  matching teacher. If the AUROC and unknown-rejection direction repeats, run a
  three-seed confirmation and then evaluate clustering separately.
