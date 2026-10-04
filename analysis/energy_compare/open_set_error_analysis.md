# Open-set Error Analysis

## Cross-run comparison

| run | AUROC | AUPR | FPR95 | OSCR | known acc all known | unknown reject rate | candidate purity | estimated K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| energy_compare_base_detect [normalized_entropy_mahalanobis] | 0.5975 | 0.4595 | 0.8607 | 0.3327 | 0.4270 | 0.0673 | 0.4590 | 18 |
| energy_compare_energy_detect [normalized_entropy_mahalanobis] | 0.6036 | 0.4677 | 0.8623 | 0.3268 | 0.4165 | 0.0590 | 0.4758 | 19 |

### energy_compare_base_detect [normalized_entropy_mahalanobis]

- AUROC: 0.5975
- AUPR: 0.4595
- FPR95: 0.8607
- OSCR: 0.3327
- known accept rate: 0.9472
- unknown reject rate: 0.0673
- known class acc after accept: 0.4508
- known class acc all known: 0.4270
- cluster NMI: 0.3886
- cluster ARI: 0.0344
- candidate pool: 586 samples, true unknown 269, false rejects 317, purity 0.4590
- cluster K: estimated 18, true 40, absolute error 22

- Score distribution:
  - known mean -0.0113, median 0.0506, p75 1.2006
  - unknown mean 0.5168, median 0.6528, p75 1.5066

- Biggest known-class rejection rates:
  - class 48 (motorcycle): reject_rate=0.200, accuracy=0.400, count=100
  - class 8 (bicycle): reject_rate=0.150, accuracy=0.250, count=100
  - class 22 (clock): reject_rate=0.120, accuracy=0.330, count=100
  - class 10 (bowl): reject_rate=0.120, accuracy=0.170, count=100
  - class 47 (maple_tree): reject_rate=0.110, accuracy=0.190, count=100

- Biggest novel-class false-accept rates:
  - class 95 (whale): false_accept_rate=1.000, correct_reject_rate=0.000, count=100
  - class 0 (apple): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 28 (cup): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 27 (crocodile): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
  - class 67 (ray): false_accept_rate=0.980, correct_reject_rate=0.020, count=100

### energy_compare_energy_detect [normalized_entropy_mahalanobis]

- AUROC: 0.6036
- AUPR: 0.4677
- FPR95: 0.8623
- OSCR: 0.3268
- known accept rate: 0.9567
- unknown reject rate: 0.0590
- known class acc after accept: 0.4354
- known class acc all known: 0.4165
- cluster NMI: 0.4279
- cluster ARI: 0.0408
- candidate pool: 496 samples, true unknown 236, false rejects 260, purity 0.4758
- cluster K: estimated 19, true 40, absolute error 21

- Score distribution:
  - known mean -0.0165, median 0.2206, p75 1.2517
  - unknown mean 0.5753, median 0.7776, p75 1.5352

- Biggest known-class rejection rates:
  - class 8 (bicycle): reject_rate=0.170, accuracy=0.200, count=100
  - class 48 (motorcycle): reject_rate=0.170, accuracy=0.500, count=100
  - class 85 (tank): reject_rate=0.110, accuracy=0.290, count=100
  - class 44 (lizard): reject_rate=0.100, accuracy=0.200, count=100
  - class 61 (plate): reject_rate=0.090, accuracy=0.350, count=100

- Biggest novel-class false-accept rates:
  - class 11 (boy): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 67 (ray): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 81 (streetcar): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
  - class 69 (rocket): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
  - class 4 (beaver): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
