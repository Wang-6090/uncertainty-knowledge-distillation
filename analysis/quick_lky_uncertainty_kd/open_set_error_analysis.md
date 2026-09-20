# Open-set Error Analysis

## Cross-run comparison

| run | AUROC | AUPR | FPR95 | OSCR | known acc all known | unknown reject rate | candidate purity | estimated K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| quick_lky_A_ce_detect [normalized_entropy_mahalanobis] | 0.4927 | 0.3768 | 0.9172 | 0.0564 | 0.0828 | 0.0316 | 0.2830 | 40 |
| quick_lky_B_standard_kd_detect [normalized_entropy_mahalanobis] | 0.4822 | 0.3900 | 0.9490 | 0.1202 | 0.1903 | 0.0442 | 0.3962 | 40 |
| quick_lky_C_uncertainty_kd_clipped_detect [normalized_entropy_mahalanobis] | 0.5211 | 0.4118 | 0.9324 | 0.0907 | 0.1283 | 0.0379 | 0.3462 | 40 |

### quick_lky_A_ce_detect [normalized_entropy_mahalanobis]

- AUROC: 0.4927
- AUPR: 0.3768
- FPR95: 0.9172
- OSCR: 0.0564
- known accept rate: 0.9476
- unknown reject rate: 0.0316
- known class acc after accept: 0.0873
- known class acc all known: 0.0828
- cluster NMI: 0.8724
- cluster ARI: 0.0330
- candidate pool: 53 samples, true unknown 15, false rejects 38, purity 0.2830
- cluster K: estimated 40, true 40, absolute error 0

- Score distribution:
  - known mean 0.0226, median 0.1771, p75 0.6603
  - unknown mean 0.0120, median 0.1347, p75 0.5431

- Biggest known-class rejection rates:
  - class 68 (road): reject_rate=0.333, accuracy=0.167, count=6
  - class 96 (willow_tree): reject_rate=0.333, accuracy=0.000, count=9
  - class 47 (maple_tree): reject_rate=0.273, accuracy=0.000, count=11
  - class 32 (flatfish): reject_rate=0.250, accuracy=0.000, count=8
  - class 22 (clock): reject_rate=0.200, accuracy=0.200, count=10

- Biggest novel-class false-accept rates:
  - class 14 (butterfly): false_accept_rate=1.000, correct_reject_rate=0.000, count=6
  - class 53 (orange): false_accept_rate=1.000, correct_reject_rate=0.000, count=6
  - class 29 (dinosaur): false_accept_rate=1.000, correct_reject_rate=0.000, count=8
  - class 28 (cup): false_accept_rate=1.000, correct_reject_rate=0.000, count=8
  - class 17 (castle): false_accept_rate=1.000, correct_reject_rate=0.000, count=8

### quick_lky_B_standard_kd_detect [normalized_entropy_mahalanobis]

- AUROC: 0.4822
- AUPR: 0.3900
- FPR95: 0.9490
- OSCR: 0.1202
- known accept rate: 0.9559
- unknown reject rate: 0.0442
- known class acc after accept: 0.1991
- known class acc all known: 0.1903
- cluster NMI: 0.8842
- cluster ARI: -0.0147
- candidate pool: 53 samples, true unknown 21, false rejects 32, purity 0.3962
- cluster K: estimated 40, true 40, absolute error 0

- Score distribution:
  - known mean -0.0191, median 0.1228, p75 0.7737
  - unknown mean -0.0669, median -0.0048, p75 0.7080

- Biggest known-class rejection rates:
  - class 65 (rabbit): reject_rate=0.250, accuracy=0.000, count=16
  - class 59 (pine_tree): reject_rate=0.200, accuracy=0.200, count=10
  - class 22 (clock): reject_rate=0.200, accuracy=0.200, count=10
  - class 44 (lizard): reject_rate=0.188, accuracy=0.000, count=16
  - class 8 (bicycle): reject_rate=0.143, accuracy=0.000, count=14

- Biggest novel-class false-accept rates:
  - class 14 (butterfly): false_accept_rate=1.000, correct_reject_rate=0.000, count=6
  - class 53 (orange): false_accept_rate=1.000, correct_reject_rate=0.000, count=6
  - class 88 (tiger): false_accept_rate=1.000, correct_reject_rate=0.000, count=8
  - class 29 (dinosaur): false_accept_rate=1.000, correct_reject_rate=0.000, count=8
  - class 17 (castle): false_accept_rate=1.000, correct_reject_rate=0.000, count=8

### quick_lky_C_uncertainty_kd_clipped_detect [normalized_entropy_mahalanobis]

- AUROC: 0.5211
- AUPR: 0.4118
- FPR95: 0.9324
- OSCR: 0.0907
- known accept rate: 0.9531
- unknown reject rate: 0.0379
- known class acc after accept: 0.1346
- known class acc all known: 0.1283
- cluster NMI: 0.8773
- cluster ARI: 0.0280
- candidate pool: 52 samples, true unknown 18, false rejects 34, purity 0.3462
- cluster K: estimated 40, true 40, absolute error 0

- Score distribution:
  - known mean 0.0623, median 0.1019, p75 0.8073
  - unknown mean 0.1603, median 0.1718, p75 0.8800

- Biggest known-class rejection rates:
  - class 47 (maple_tree): reject_rate=0.364, accuracy=0.000, count=11
  - class 42 (leopard): reject_rate=0.286, accuracy=0.000, count=14
  - class 59 (pine_tree): reject_rate=0.200, accuracy=0.000, count=10
  - class 39 (keyboard): reject_rate=0.182, accuracy=0.727, count=11
  - class 33 (forest): reject_rate=0.182, accuracy=0.091, count=11

- Biggest novel-class false-accept rates:
  - class 53 (orange): false_accept_rate=1.000, correct_reject_rate=0.000, count=6
  - class 29 (dinosaur): false_accept_rate=1.000, correct_reject_rate=0.000, count=8
  - class 28 (cup): false_accept_rate=1.000, correct_reject_rate=0.000, count=8
  - class 17 (castle): false_accept_rate=1.000, correct_reject_rate=0.000, count=8
  - class 54 (orchid): false_accept_rate=1.000, correct_reject_rate=0.000, count=9
