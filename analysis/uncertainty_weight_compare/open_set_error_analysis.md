# Open-set Error Analysis

## Cross-run comparison

| run | AUROC | FPR95 | known acc all known | unknown reject rate | known accept rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| uncertainty_weight_raw_detect [normalized_entropy_mahalanobis] | 0.5926 | 0.8620 | 0.4048 | 0.0530 | 0.9582 |
| uncertainty_weight_mean_normalized_detect [normalized_entropy_mahalanobis] | 0.5985 | 0.8763 | 0.4198 | 0.0717 | 0.9483 |

### uncertainty_weight_raw_detect [normalized_entropy_mahalanobis]

- AUROC: 0.5926
- FPR95: 0.8620
- known accept rate: 0.9582
- unknown reject rate: 0.0530
- known class acc after accept: 0.4225
- known class acc all known: 0.4048
- cluster NMI: 0.5259
- cluster ARI: 0.0417

- Score distribution:
  - known mean -0.0072, median 0.0619, p75 1.1474
  - unknown mean 0.4727, median 0.5915, p75 1.4280

- Biggest known-class rejection rates:
  - class 48 (motorcycle): reject_rate=0.290, accuracy=0.000, count=100
  - class 8 (bicycle): reject_rate=0.170, accuracy=0.000, count=100
  - class 16 (can): reject_rate=0.110, accuracy=0.000, count=100
  - class 1 (aquarium_fish): reject_rate=0.080, accuracy=0.000, count=100
  - class 58 (pickup_truck): reject_rate=0.070, accuracy=0.000, count=100

- Biggest novel-class false-accept rates:
  - class 20 (chair): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 67 (ray): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 64 (possum): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 88 (tiger): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 71 (sea): false_accept_rate=0.980, correct_reject_rate=0.020, count=100

### uncertainty_weight_mean_normalized_detect [normalized_entropy_mahalanobis]

- AUROC: 0.5985
- FPR95: 0.8763
- known accept rate: 0.9483
- unknown reject rate: 0.0717
- known class acc after accept: 0.4427
- known class acc all known: 0.4198
- cluster NMI: 0.4897
- cluster ARI: 0.0485

- Score distribution:
  - known mean -0.0244, median 0.0021, p75 1.1754
  - unknown mean 0.4920, median 0.6201, p75 1.4597

- Biggest known-class rejection rates:
  - class 48 (motorcycle): reject_rate=0.290, accuracy=0.000, count=100
  - class 8 (bicycle): reject_rate=0.220, accuracy=0.000, count=100
  - class 45 (lobster): reject_rate=0.100, accuracy=0.000, count=100
  - class 51 (mushroom): reject_rate=0.090, accuracy=0.000, count=100
  - class 61 (plate): reject_rate=0.090, accuracy=0.000, count=100

- Biggest novel-class false-accept rates:
  - class 0 (apple): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 31 (elephant): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 71 (sea): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
  - class 11 (boy): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
  - class 35 (girl): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
