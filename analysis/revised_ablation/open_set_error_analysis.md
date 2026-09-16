# Open-set Error Analysis

## Cross-run comparison

| run | AUROC | AUPR | FPR95 | OSCR | known acc all known | unknown reject rate | candidate purity | estimated K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| revised_A_ce_detect [normalized_entropy_mahalanobis] | 0.5811 | 0.4463 | 0.8882 | 0.3180 | 0.4278 | 0.0633 | 0.4325 | 2 |
| revised_B_standard_kd_detect [normalized_entropy_mahalanobis] | 0.5923 | 0.4535 | 0.8645 | 0.3216 | 0.4238 | 0.0548 | 0.4761 | 2 |
| revised_C_uncertainty_kd_detect [normalized_entropy_mahalanobis] | 0.6085 | 0.4732 | 0.8632 | 0.3448 | 0.4427 | 0.0720 | 0.4940 | 2 |
| revised_D_uncertainty_feature_kd_detect [normalized_entropy_mahalanobis] | 0.6021 | 0.4701 | 0.8562 | 0.3402 | 0.4378 | 0.0800 | 0.5039 | 2 |
| revised_E_full_representation_detect [normalized_entropy_mahalanobis] | 0.5935 | 0.4615 | 0.8802 | 0.3307 | 0.4323 | 0.0760 | 0.4735 | 10 |

### revised_A_ce_detect [normalized_entropy_mahalanobis]

- AUROC: 0.5811
- AUPR: 0.4463
- FPR95: 0.8882
- OSCR: 0.3180
- known accept rate: 0.9447
- unknown reject rate: 0.0633
- known class acc after accept: 0.4529
- known class acc all known: 0.4278
- cluster NMI: 0.0593
- cluster ARI: 0.0032
- candidate pool: 585 samples, true unknown 253, false rejects 332, purity 0.4325
- cluster K: estimated 2, true 40, absolute error 38

- Score distribution:
  - known mean 0.0091, median -0.0131, p75 1.1761
  - unknown mean 0.4229, median 0.5051, p75 1.4808

- Biggest known-class rejection rates:
  - class 48 (motorcycle): reject_rate=0.160, accuracy=0.550, count=100
  - class 45 (lobster): reject_rate=0.150, accuracy=0.140, count=100
  - class 59 (pine_tree): reject_rate=0.140, accuracy=0.390, count=100
  - class 96 (willow_tree): reject_rate=0.110, accuracy=0.320, count=100
  - class 51 (mushroom): reject_rate=0.100, accuracy=0.440, count=100

- Biggest novel-class false-accept rates:
  - class 97 (wolf): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 93 (turtle): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 35 (girl): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
  - class 64 (possum): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
  - class 94 (wardrobe): false_accept_rate=0.980, correct_reject_rate=0.020, count=100

### revised_B_standard_kd_detect [normalized_entropy_mahalanobis]

- AUROC: 0.5923
- AUPR: 0.4535
- FPR95: 0.8645
- OSCR: 0.3216
- known accept rate: 0.9598
- unknown reject rate: 0.0548
- known class acc after accept: 0.4416
- known class acc all known: 0.4238
- cluster NMI: 0.1171
- cluster ARI: 0.0145
- candidate pool: 460 samples, true unknown 219, false rejects 241, purity 0.4761
- cluster K: estimated 2, true 40, absolute error 38

- Score distribution:
  - known mean -0.0397, median 0.0678, p75 1.1015
  - unknown mean 0.4377, median 0.5906, p75 1.3040

- Biggest known-class rejection rates:
  - class 48 (motorcycle): reject_rate=0.270, accuracy=0.310, count=100
  - class 8 (bicycle): reject_rate=0.160, accuracy=0.250, count=100
  - class 1 (aquarium_fish): reject_rate=0.100, accuracy=0.410, count=100
  - class 61 (plate): reject_rate=0.080, accuracy=0.450, count=100
  - class 56 (palm_tree): reject_rate=0.080, accuracy=0.620, count=100

- Biggest novel-class false-accept rates:
  - class 64 (possum): false_accept_rate=1.000, correct_reject_rate=0.000, count=100
  - class 43 (lion): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 4 (beaver): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 35 (girl): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 24 (cockroach): false_accept_rate=0.980, correct_reject_rate=0.020, count=100

### revised_C_uncertainty_kd_detect [normalized_entropy_mahalanobis]

- AUROC: 0.6085
- AUPR: 0.4732
- FPR95: 0.8632
- OSCR: 0.3448
- known accept rate: 0.9508
- unknown reject rate: 0.0720
- known class acc after accept: 0.4656
- known class acc all known: 0.4427
- cluster NMI: 0.0852
- cluster ARI: 0.0055
- candidate pool: 583 samples, true unknown 288, false rejects 295, purity 0.4940
- cluster K: estimated 2, true 40, absolute error 38

- Score distribution:
  - known mean -0.0590, median 0.0420, p75 1.1778
  - unknown mean 0.5423, median 0.7404, p75 1.5735

- Biggest known-class rejection rates:
  - class 8 (bicycle): reject_rate=0.190, accuracy=0.300, count=100
  - class 1 (aquarium_fish): reject_rate=0.110, accuracy=0.480, count=100
  - class 45 (lobster): reject_rate=0.110, accuracy=0.150, count=100
  - class 63 (porcupine): reject_rate=0.100, accuracy=0.120, count=100
  - class 61 (plate): reject_rate=0.090, accuracy=0.330, count=100

- Biggest novel-class false-accept rates:
  - class 4 (beaver): false_accept_rate=1.000, correct_reject_rate=0.000, count=100
  - class 64 (possum): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 67 (ray): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
  - class 86 (telephone): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
  - class 75 (skunk): false_accept_rate=0.970, correct_reject_rate=0.030, count=100

### revised_D_uncertainty_feature_kd_detect [normalized_entropy_mahalanobis]

- AUROC: 0.6021
- AUPR: 0.4701
- FPR95: 0.8562
- OSCR: 0.3402
- known accept rate: 0.9475
- unknown reject rate: 0.0800
- known class acc after accept: 0.4621
- known class acc all known: 0.4378
- cluster NMI: 0.0896
- cluster ARI: 0.0082
- candidate pool: 635 samples, true unknown 320, false rejects 315, purity 0.5039
- cluster K: estimated 2, true 40, absolute error 38

- Score distribution:
  - known mean -0.0002, median 0.0921, p75 1.2841
  - unknown mean 0.5851, median 0.7388, p75 1.6469

- Biggest known-class rejection rates:
  - class 8 (bicycle): reject_rate=0.160, accuracy=0.320, count=100
  - class 48 (motorcycle): reject_rate=0.140, accuracy=0.590, count=100
  - class 85 (tank): reject_rate=0.110, accuracy=0.390, count=100
  - class 7 (beetle): reject_rate=0.100, accuracy=0.470, count=100
  - class 47 (maple_tree): reject_rate=0.100, accuracy=0.210, count=100

- Biggest novel-class false-accept rates:
  - class 71 (sea): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
  - class 0 (apple): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
  - class 75 (skunk): false_accept_rate=0.970, correct_reject_rate=0.030, count=100
  - class 93 (turtle): false_accept_rate=0.970, correct_reject_rate=0.030, count=100
  - class 11 (boy): false_accept_rate=0.970, correct_reject_rate=0.030, count=100

### revised_E_full_representation_detect [normalized_entropy_mahalanobis]

- AUROC: 0.5935
- AUPR: 0.4615
- FPR95: 0.8802
- OSCR: 0.3307
- known accept rate: 0.9437
- unknown reject rate: 0.0760
- known class acc after accept: 0.4581
- known class acc all known: 0.4323
- cluster NMI: 0.3310
- cluster ARI: 0.0451
- candidate pool: 642 samples, true unknown 304, false rejects 338, purity 0.4735
- cluster K: estimated 10, true 40, absolute error 30

- Score distribution:
  - known mean -0.0056, median 0.0048, p75 1.1726
  - unknown mean 0.4959, median 0.5713, p75 1.5056

- Biggest known-class rejection rates:
  - class 8 (bicycle): reject_rate=0.200, accuracy=0.280, count=100
  - class 48 (motorcycle): reject_rate=0.200, accuracy=0.400, count=100
  - class 1 (aquarium_fish): reject_rate=0.130, accuracy=0.530, count=100
  - class 61 (plate): reject_rate=0.120, accuracy=0.480, count=100
  - class 58 (pickup_truck): reject_rate=0.120, accuracy=0.680, count=100

- Biggest novel-class false-accept rates:
  - class 97 (wolf): false_accept_rate=1.000, correct_reject_rate=0.000, count=100
  - class 0 (apple): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 95 (whale): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 64 (possum): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 28 (cup): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
