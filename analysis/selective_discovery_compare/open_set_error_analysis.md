# Open-set Error Analysis

## Cross-run comparison

| run | AUROC | AUPR | FPR95 | OSCR | known acc all known | unknown reject rate | candidate purity | estimated K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| selective_compare_mixed_base_e3_detect [normalized_entropy_mahalanobis_diag] | 0.5693 | 0.4497 | 0.9276 | 0.2374 | 0.3158 | 0.0714 | 0.4590 | 19 |
| selective_compare_mixed_selective_e3_detect [normalized_entropy_mahalanobis_diag] | 0.5545 | 0.4379 | 0.9095 | 0.2120 | 0.2878 | 0.0663 | 0.4727 | 19 |

### selective_compare_mixed_base_e3_detect [normalized_entropy_mahalanobis_diag]

- AUROC: 0.5693
- AUPR: 0.4497
- FPR95: 0.9276
- OSCR: 0.2374
- known accept rate: 0.9457
- unknown reject rate: 0.0714
- known class acc after accept: 0.3339
- known class acc all known: 0.3158
- ECE: 0.2621
- temperature: 1.0000
- uncertainty/error correlation: -0.1197
- all-unknown oracle NMI: 0.5191
- candidate-unknown oracle NMI: 0.8241
- cluster NMI: 0.7593
- cluster ARI: 0.0195
- candidate pool: 61 samples, true unknown 28, false rejects 33, purity 0.4590
- cluster K: estimated 19, true 40, absolute error 21

- Score distribution:
  - known mean 0.0773, median 0.1938, p75 1.0776
  - unknown mean 0.4238, median 0.5728, p75 1.4035

- Biggest known-class rejection rates:
  - class 16 (can): reject_rate=0.300, accuracy=0.000, count=10
  - class 37 (house): reject_rate=0.286, accuracy=0.143, count=7
  - class 33 (forest): reject_rate=0.286, accuracy=0.429, count=7
  - class 39 (keyboard): reject_rate=0.222, accuracy=0.444, count=9
  - class 61 (plate): reject_rate=0.182, accuracy=0.182, count=11

- Biggest novel-class false-accept rates:
  - class 29 (dinosaur): false_accept_rate=1.000, correct_reject_rate=0.000, count=5
  - class 17 (castle): false_accept_rate=1.000, correct_reject_rate=0.000, count=5
  - class 14 (butterfly): false_accept_rate=1.000, correct_reject_rate=0.000, count=6
  - class 54 (orchid): false_accept_rate=1.000, correct_reject_rate=0.000, count=6
  - class 57 (pear): false_accept_rate=1.000, correct_reject_rate=0.000, count=7

### selective_compare_mixed_selective_e3_detect [normalized_entropy_mahalanobis_diag]

- AUROC: 0.5545
- AUPR: 0.4379
- FPR95: 0.9095
- OSCR: 0.2120
- known accept rate: 0.9523
- unknown reject rate: 0.0663
- known class acc after accept: 0.3022
- known class acc all known: 0.2878
- ECE: 0.2881
- temperature: 1.0000
- uncertainty/error correlation: -0.1058
- all-unknown oracle NMI: 0.5300
- candidate-unknown oracle NMI: 0.7909
- cluster NMI: 0.7422
- cluster ARI: -0.0064
- candidate pool: 55 samples, true unknown 26, false rejects 29, purity 0.4727
- cluster K: estimated 19, true 40, absolute error 21

- Score distribution:
  - known mean 0.0088, median 0.0894, p75 0.9898
  - unknown mean 0.2791, median 0.3347, p75 1.1451

- Biggest known-class rejection rates:
  - class 37 (house): reject_rate=0.286, accuracy=0.000, count=7
  - class 61 (plate): reject_rate=0.273, accuracy=0.000, count=11
  - class 58 (pickup_truck): reject_rate=0.250, accuracy=0.500, count=12
  - class 8 (bicycle): reject_rate=0.222, accuracy=0.111, count=9
  - class 39 (keyboard): reject_rate=0.222, accuracy=0.444, count=9

- Biggest novel-class false-accept rates:
  - class 17 (castle): false_accept_rate=1.000, correct_reject_rate=0.000, count=5
  - class 14 (butterfly): false_accept_rate=1.000, correct_reject_rate=0.000, count=6
  - class 54 (orchid): false_accept_rate=1.000, correct_reject_rate=0.000, count=6
  - class 79 (spider): false_accept_rate=1.000, correct_reject_rate=0.000, count=7
  - class 35 (girl): false_accept_rate=1.000, correct_reject_rate=0.000, count=8
