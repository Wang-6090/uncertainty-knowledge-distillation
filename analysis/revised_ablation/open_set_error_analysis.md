# Open-set Error Analysis

## Cross-run comparison

| run | AUROC | FPR95 | known acc all known | unknown reject rate | known accept rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| revised_A_ce_detect [normalized_entropy_mahalanobis] | 0.5939 | 0.8812 | 0.4527 | 0.0673 | 0.9505 |
| revised_B_standard_kd_detect [normalized_entropy_mahalanobis] | 0.5955 | 0.8632 | 0.4093 | 0.0628 | 0.9503 |
| revised_C_uncertainty_kd_detect [normalized_entropy_mahalanobis] | 0.5898 | 0.8613 | 0.4290 | 0.0508 | 0.9543 |
| revised_D_uncertainty_feature_kd_detect [normalized_entropy_mahalanobis] | 0.5846 | 0.8790 | 0.4113 | 0.0653 | 0.9497 |
| revised_E_full_representation_detect [normalized_entropy_mahalanobis] | 0.5977 | 0.8713 | 0.4375 | 0.0747 | 0.9452 |

### revised_A_ce_detect [normalized_entropy_mahalanobis]

- AUROC: 0.5939
- FPR95: 0.8812
- known accept rate: 0.9505
- unknown reject rate: 0.0673
- known class acc after accept: 0.4762
- known class acc all known: 0.4527
- cluster NMI: 0.5127
- cluster ARI: 0.0535

- Score distribution:
  - known mean -0.0068, median -0.0692, p75 1.1630
  - unknown mean 0.4924, median 0.5805, p75 1.5634

- Biggest known-class rejection rates:
  - class 48 (motorcycle): reject_rate=0.150, accuracy=0.000, count=100
  - class 8 (bicycle): reject_rate=0.140, accuracy=0.000, count=100
  - class 85 (tank): reject_rate=0.110, accuracy=0.000, count=100
  - class 1 (aquarium_fish): reject_rate=0.110, accuracy=0.000, count=100
  - class 44 (lizard): reject_rate=0.100, accuracy=0.000, count=100

- Biggest novel-class false-accept rates:
  - class 95 (whale): false_accept_rate=1.000, correct_reject_rate=0.000, count=100
  - class 67 (ray): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 77 (snail): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
  - class 64 (possum): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
  - class 88 (tiger): false_accept_rate=0.980, correct_reject_rate=0.020, count=100

### revised_B_standard_kd_detect [normalized_entropy_mahalanobis]

- AUROC: 0.5955
- FPR95: 0.8632
- known accept rate: 0.9503
- unknown reject rate: 0.0628
- known class acc after accept: 0.4307
- known class acc all known: 0.4093
- cluster NMI: 0.5190
- cluster ARI: 0.0571

- Score distribution:
  - known mean -0.0100, median 0.1028, p75 1.1597
  - unknown mean 0.5068, median 0.6575, p75 1.4724

- Biggest known-class rejection rates:
  - class 48 (motorcycle): reject_rate=0.190, accuracy=0.000, count=100
  - class 8 (bicycle): reject_rate=0.170, accuracy=0.000, count=100
  - class 85 (tank): reject_rate=0.120, accuracy=0.000, count=100
  - class 63 (porcupine): reject_rate=0.100, accuracy=0.000, count=100
  - class 41 (lawn_mower): reject_rate=0.090, accuracy=0.000, count=100

- Biggest novel-class false-accept rates:
  - class 0 (apple): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 28 (cup): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 71 (sea): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
  - class 77 (snail): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
  - class 14 (butterfly): false_accept_rate=0.970, correct_reject_rate=0.030, count=100

### revised_C_uncertainty_kd_detect [normalized_entropy_mahalanobis]

- AUROC: 0.5898
- FPR95: 0.8613
- known accept rate: 0.9543
- unknown reject rate: 0.0508
- known class acc after accept: 0.4495
- known class acc all known: 0.4290
- cluster NMI: 0.5387
- cluster ARI: 0.0628

- Score distribution:
  - known mean -0.0181, median 0.0725, p75 1.1964
  - unknown mean 0.4606, median 0.5757, p75 1.4034

- Biggest known-class rejection rates:
  - class 48 (motorcycle): reject_rate=0.230, accuracy=0.000, count=100
  - class 8 (bicycle): reject_rate=0.190, accuracy=0.000, count=100
  - class 44 (lizard): reject_rate=0.120, accuracy=0.000, count=100
  - class 33 (forest): reject_rate=0.090, accuracy=0.000, count=100
  - class 78 (snake): reject_rate=0.090, accuracy=0.000, count=100

- Biggest novel-class false-accept rates:
  - class 0 (apple): false_accept_rate=1.000, correct_reject_rate=0.000, count=100
  - class 35 (girl): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 95 (whale): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 71 (sea): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
  - class 20 (chair): false_accept_rate=0.980, correct_reject_rate=0.020, count=100

### revised_D_uncertainty_feature_kd_detect [normalized_entropy_mahalanobis]

- AUROC: 0.5846
- FPR95: 0.8790
- known accept rate: 0.9497
- unknown reject rate: 0.0653
- known class acc after accept: 0.4331
- known class acc all known: 0.4113
- cluster NMI: 0.5083
- cluster ARI: 0.0483

- Score distribution:
  - known mean 0.0303, median 0.1528, p75 1.1942
  - unknown mean 0.4779, median 0.6082, p75 1.4299

- Biggest known-class rejection rates:
  - class 48 (motorcycle): reject_rate=0.240, accuracy=0.000, count=100
  - class 8 (bicycle): reject_rate=0.220, accuracy=0.000, count=100
  - class 58 (pickup_truck): reject_rate=0.110, accuracy=0.000, count=100
  - class 85 (tank): reject_rate=0.110, accuracy=0.000, count=100
  - class 1 (aquarium_fish): reject_rate=0.100, accuracy=0.000, count=100

- Biggest novel-class false-accept rates:
  - class 0 (apple): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 6 (bee): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 28 (cup): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 94 (wardrobe): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 31 (elephant): false_accept_rate=0.990, correct_reject_rate=0.010, count=100

### revised_E_full_representation_detect [normalized_entropy_mahalanobis]

- AUROC: 0.5977
- FPR95: 0.8713
- known accept rate: 0.9452
- unknown reject rate: 0.0747
- known class acc after accept: 0.4629
- known class acc all known: 0.4375
- cluster NMI: 0.4759
- cluster ARI: 0.0398

- Score distribution:
  - known mean -0.0083, median 0.0719, p75 1.2446
  - unknown mean 0.5375, median 0.6996, p75 1.5827

- Biggest known-class rejection rates:
  - class 8 (bicycle): reject_rate=0.230, accuracy=0.000, count=100
  - class 45 (lobster): reject_rate=0.130, accuracy=0.000, count=100
  - class 63 (porcupine): reject_rate=0.120, accuracy=0.000, count=100
  - class 44 (lizard): reject_rate=0.120, accuracy=0.000, count=100
  - class 58 (pickup_truck): reject_rate=0.100, accuracy=0.000, count=100

- Biggest novel-class false-accept rates:
  - class 71 (sea): false_accept_rate=1.000, correct_reject_rate=0.000, count=100
  - class 53 (orange): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 4 (beaver): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 0 (apple): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
  - class 35 (girl): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
