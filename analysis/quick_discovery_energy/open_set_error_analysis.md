# Open-set Error Analysis

## Cross-run comparison

| run | AUROC | AUPR | FPR95 | OSCR | known acc all known | unknown reject rate | candidate purity | estimated K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| quick_de_base_detect [normalized_entropy_mahalanobis] | 0.5259 | 0.4138 | 0.9125 | 0.2029 | 0.2719 | 0.0456 | 0.3426 | 17 |
| quick_de_discovery_energy_detect [normalized_entropy_mahalanobis] | 0.5800 | 0.4558 | 0.8897 | 0.2782 | 0.3746 | 0.0714 | 0.5000 | 13 |

### quick_de_base_detect [normalized_entropy_mahalanobis]

- AUROC: 0.5259
- AUPR: 0.4138
- FPR95: 0.9125
- OSCR: 0.2029
- known accept rate: 0.9402
- unknown reject rate: 0.0456
- known class acc after accept: 0.2892
- known class acc all known: 0.2719
- cluster NMI: 0.6438
- cluster ARI: 0.0067
- candidate pool: 108 samples, true unknown 37, false rejects 71, purity 0.3426
- cluster K: estimated 17, true 40, absolute error 23

- Score distribution:
  - known mean 0.0520, median 0.1651, p75 1.0054
  - unknown mean 0.1844, median 0.2590, p75 1.0722

- Biggest known-class rejection rates:
  - class 8 (bicycle): reject_rate=0.238, accuracy=0.095, count=21
  - class 47 (maple_tree): reject_rate=0.200, accuracy=0.150, count=20
  - class 33 (forest): reject_rate=0.200, accuracy=0.100, count=20
  - class 96 (willow_tree): reject_rate=0.176, accuracy=0.000, count=17
  - class 16 (can): reject_rate=0.150, accuracy=0.200, count=20

- Biggest novel-class false-accept rates:
  - class 53 (orange): false_accept_rate=1.000, correct_reject_rate=0.000, count=11
  - class 14 (butterfly): false_accept_rate=1.000, correct_reject_rate=0.000, count=13
  - class 25 (couch): false_accept_rate=1.000, correct_reject_rate=0.000, count=14
  - class 28 (cup): false_accept_rate=1.000, correct_reject_rate=0.000, count=15
  - class 97 (wolf): false_accept_rate=1.000, correct_reject_rate=0.000, count=18

### quick_de_discovery_energy_detect [normalized_entropy_mahalanobis]

- AUROC: 0.5800
- AUPR: 0.4558
- FPR95: 0.8897
- OSCR: 0.2782
- known accept rate: 0.9512
- unknown reject rate: 0.0714
- known class acc after accept: 0.3938
- known class acc all known: 0.3746
- cluster NMI: 0.5996
- cluster ARI: 0.0247
- candidate pool: 116 samples, true unknown 58, false rejects 58, purity 0.5000
- cluster K: estimated 13, true 40, absolute error 27

- Score distribution:
  - known mean 0.0150, median 0.0194, p75 1.1654
  - unknown mean 0.4401, median 0.4896, p75 1.4368

- Biggest known-class rejection rates:
  - class 8 (bicycle): reject_rate=0.238, accuracy=0.143, count=21
  - class 16 (can): reject_rate=0.200, accuracy=0.350, count=20
  - class 58 (pickup_truck): reject_rate=0.158, accuracy=0.579, count=19
  - class 51 (mushroom): reject_rate=0.130, accuracy=0.217, count=23
  - class 1 (aquarium_fish): reject_rate=0.125, accuracy=0.375, count=16

- Biggest novel-class false-accept rates:
  - class 53 (orange): false_accept_rate=1.000, correct_reject_rate=0.000, count=11
  - class 88 (tiger): false_accept_rate=1.000, correct_reject_rate=0.000, count=15
  - class 97 (wolf): false_accept_rate=1.000, correct_reject_rate=0.000, count=18
  - class 0 (apple): false_accept_rate=1.000, correct_reject_rate=0.000, count=18
  - class 64 (possum): false_accept_rate=1.000, correct_reject_rate=0.000, count=19
