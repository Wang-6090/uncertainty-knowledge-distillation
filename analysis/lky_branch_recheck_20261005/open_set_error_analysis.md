# Open-set Error Analysis

## Cross-run comparison

| run | AUROC | AUPR | FPR95 | OSCR | known acc all known | unknown reject rate | candidate purity | estimated K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| revised_ms_s42_E_full_representation_detect_oracle [normalized_entropy_mahalanobis] | 0.5907 | 0.4575 | 0.8835 | 0.3206 | 0.4190 | 0.0560 | - | - |
| revised_ms_s123_E_full_representation_detect_oracle [normalized_entropy_mahalanobis] | 0.5739 | 0.4442 | 0.8823 | 0.3031 | 0.3947 | 0.0617 | - | - |
| revised_ms_s42_F_discovery_pool_detect_oracle [normalized_entropy_mahalanobis] | 0.6419 | 0.5066 | 0.8425 | 0.4035 | 0.5145 | 0.0907 | - | - |
| revised_ms_s123_F_discovery_pool_detect_oracle [normalized_entropy_mahalanobis] | 0.6375 | 0.4976 | 0.8402 | 0.4133 | 0.5297 | 0.0820 | - | - |

### revised_ms_s42_E_full_representation_detect_oracle [normalized_entropy_mahalanobis]

- AUROC: 0.5907
- AUPR: 0.4575
- FPR95: 0.8835
- OSCR: 0.3206
- known accept rate: 0.9577
- unknown reject rate: 0.0560
- known class acc after accept: 0.4375
- known class acc all known: 0.4190
- ECE: -
- temperature: -
- uncertainty/error correlation: -
- all-unknown oracle NMI: -
- candidate-unknown oracle NMI: -
- cluster NMI: 0.5241
- cluster ARI: 0.0488
- candidate pool: 478 samples, true unknown 224, false rejects 254, purity -
- cluster K: estimated None, true 40, absolute error 0

- Score distribution:
  - known mean -0.0498, median -0.0344, p75 1.1304
  - unknown mean 0.4267, median 0.5267, p75 1.4291

- Biggest known-class rejection rates:
  - class 48 (motorcycle): reject_rate=0.180, accuracy=0.240, count=100
  - class 1 (aquarium_fish): reject_rate=0.160, accuracy=0.490, count=100
  - class 85 (tank): reject_rate=0.140, accuracy=0.420, count=100
  - class 8 (bicycle): reject_rate=0.130, accuracy=0.190, count=100
  - class 7 (beetle): reject_rate=0.100, accuracy=0.330, count=100

- Biggest novel-class false-accept rates:
  - class 4 (beaver): false_accept_rate=1.000, correct_reject_rate=0.000, count=100
  - class 35 (girl): false_accept_rate=1.000, correct_reject_rate=0.000, count=100
  - class 95 (whale): false_accept_rate=1.000, correct_reject_rate=0.000, count=100
  - class 0 (apple): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 28 (cup): false_accept_rate=0.990, correct_reject_rate=0.010, count=100

### revised_ms_s123_E_full_representation_detect_oracle [normalized_entropy_mahalanobis]

- AUROC: 0.5739
- AUPR: 0.4442
- FPR95: 0.8823
- OSCR: 0.3031
- known accept rate: 0.9493
- unknown reject rate: 0.0617
- known class acc after accept: 0.4157
- known class acc all known: 0.3947
- ECE: -
- temperature: -
- uncertainty/error correlation: -
- all-unknown oracle NMI: -
- candidate-unknown oracle NMI: -
- cluster NMI: 0.4850
- cluster ARI: 0.0308
- candidate pool: 551 samples, true unknown 247, false rejects 304, purity -
- cluster K: estimated None, true 40, absolute error 0

- Score distribution:
  - known mean -0.0133, median 0.0819, p75 1.2506
  - unknown mean 0.4155, median 0.5341, p75 1.4676

- Biggest known-class rejection rates:
  - class 8 (bicycle): reject_rate=0.190, accuracy=0.240, count=100
  - class 48 (motorcycle): reject_rate=0.190, accuracy=0.430, count=100
  - class 78 (snake): reject_rate=0.130, accuracy=0.220, count=100
  - class 10 (bowl): reject_rate=0.110, accuracy=0.190, count=100
  - class 47 (maple_tree): reject_rate=0.110, accuracy=0.160, count=100

- Biggest novel-class false-accept rates:
  - class 67 (ray): false_accept_rate=1.000, correct_reject_rate=0.000, count=100
  - class 0 (apple): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 28 (cup): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 94 (wardrobe): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 31 (elephant): false_accept_rate=0.990, correct_reject_rate=0.010, count=100

### revised_ms_s42_F_discovery_pool_detect_oracle [normalized_entropy_mahalanobis]

- AUROC: 0.6419
- AUPR: 0.5066
- FPR95: 0.8425
- OSCR: 0.4035
- known accept rate: 0.9510
- unknown reject rate: 0.0907
- known class acc after accept: 0.5410
- known class acc all known: 0.5145
- ECE: -
- temperature: -
- uncertainty/error correlation: -
- all-unknown oracle NMI: -
- candidate-unknown oracle NMI: -
- cluster NMI: 0.4982
- cluster ARI: 0.0613
- candidate pool: 657 samples, true unknown 363, false rejects 294, purity -
- cluster K: estimated None, true 40, absolute error 0

- Score distribution:
  - known mean -0.0349, median -0.1077, p75 1.1771
  - unknown mean 0.7575, median 0.8081, p75 1.8202

- Biggest known-class rejection rates:
  - class 1 (aquarium_fish): reject_rate=0.140, accuracy=0.600, count=100
  - class 44 (lizard): reject_rate=0.140, accuracy=0.290, count=100
  - class 51 (mushroom): reject_rate=0.110, accuracy=0.530, count=100
  - class 8 (bicycle): reject_rate=0.110, accuracy=0.320, count=100
  - class 7 (beetle): reject_rate=0.110, accuracy=0.570, count=100

- Biggest novel-class false-accept rates:
  - class 95 (whale): false_accept_rate=1.000, correct_reject_rate=0.000, count=100
  - class 71 (sea): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
  - class 17 (castle): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
  - class 35 (girl): false_accept_rate=0.970, correct_reject_rate=0.030, count=100
  - class 64 (possum): false_accept_rate=0.970, correct_reject_rate=0.030, count=100

### revised_ms_s123_F_discovery_pool_detect_oracle [normalized_entropy_mahalanobis]

- AUROC: 0.6375
- AUPR: 0.4976
- FPR95: 0.8402
- OSCR: 0.4133
- known accept rate: 0.9495
- unknown reject rate: 0.0820
- known class acc after accept: 0.5578
- known class acc all known: 0.5297
- ECE: -
- temperature: -
- uncertainty/error correlation: -
- all-unknown oracle NMI: -
- candidate-unknown oracle NMI: -
- cluster NMI: 0.5124
- cluster ARI: 0.0744
- candidate pool: 631 samples, true unknown 328, false rejects 303, purity -
- cluster K: estimated None, true 40, absolute error 0

- Score distribution:
  - known mean -0.0246, median -0.1663, p75 1.2869
  - unknown mean 0.7688, median 0.8890, p75 1.8872

- Biggest known-class rejection rates:
  - class 48 (motorcycle): reject_rate=0.180, accuracy=0.540, count=100
  - class 8 (bicycle): reject_rate=0.140, accuracy=0.330, count=100
  - class 85 (tank): reject_rate=0.140, accuracy=0.470, count=100
  - class 45 (lobster): reject_rate=0.140, accuracy=0.180, count=100
  - class 41 (lawn_mower): reject_rate=0.120, accuracy=0.560, count=100

- Biggest novel-class false-accept rates:
  - class 64 (possum): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
  - class 71 (sea): false_accept_rate=0.970, correct_reject_rate=0.030, count=100
  - class 53 (orange): false_accept_rate=0.970, correct_reject_rate=0.030, count=100
  - class 4 (beaver): false_accept_rate=0.970, correct_reject_rate=0.030, count=100
  - class 67 (ray): false_accept_rate=0.970, correct_reject_rate=0.030, count=100
