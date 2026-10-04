# Open-set Error Analysis

## Cross-run comparison

| run | AUROC | AUPR | FPR95 | OSCR | known acc all known | unknown reject rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| discovery_compare_base_detect [normalized_entropy_mahalanobis] | 0.5044 | 0.3863 | 0.9375 | 0.1340 | 0.1941 | 0.0255 |
| discovery_compare_ntxent_detect [normalized_entropy_mahalanobis] | 0.5629 | 0.4426 | 0.8816 | 0.1909 | 0.2714 | 0.0561 |
| discovery_compare_ntxent_unknown_detect [normalized_entropy_mahalanobis] | 0.5706 | 0.4657 | 0.9194 | 0.1799 | 0.2467 | 0.0969 |
| discovery_compare_ntxent_unknown002_detect [normalized_entropy_mahalanobis] | 0.5482 | 0.4202 | 0.8947 | 0.1600 | 0.2204 | 0.0587 |

### discovery_compare_base_detect [normalized_entropy_mahalanobis]

- AUROC: 0.5044
- AUPR: 0.3863
- FPR95: 0.9375
- OSCR: 0.1340
- known accept rate: 0.9605
- unknown reject rate: 0.0255
- known class acc after accept: 0.2021
- known class acc all known: 0.1941
- cluster NMI: 0.9491
- cluster ARI: 0.0000

- Score distribution:
  - known mean 0.0487, median 0.1215, p75 0.9543
  - unknown mean 0.0699, median 0.1993, p75 0.9128

- Biggest known-class rejection rates:
  - class 47 (maple_tree): reject_rate=0.222, accuracy=0.111, count=9
  - class 52 (oak_tree): reject_rate=0.200, accuracy=0.100, count=10
  - class 62 (poppy): reject_rate=0.154, accuracy=0.154, count=13
  - class 37 (house): reject_rate=0.143, accuracy=0.000, count=7
  - class 33 (forest): reject_rate=0.143, accuracy=0.000, count=7

- Biggest novel-class false-accept rates:
  - class 29 (dinosaur): false_accept_rate=1.000, correct_reject_rate=0.000, count=5
  - class 17 (castle): false_accept_rate=1.000, correct_reject_rate=0.000, count=5
  - class 14 (butterfly): false_accept_rate=1.000, correct_reject_rate=0.000, count=6
  - class 54 (orchid): false_accept_rate=1.000, correct_reject_rate=0.000, count=6
  - class 57 (pear): false_accept_rate=1.000, correct_reject_rate=0.000, count=7

### discovery_compare_ntxent_detect [normalized_entropy_mahalanobis]

- AUROC: 0.5629
- AUPR: 0.4426
- FPR95: 0.8816
- OSCR: 0.1909
- known accept rate: 0.9523
- unknown reject rate: 0.0561
- known class acc after accept: 0.2850
- known class acc all known: 0.2714
- cluster NMI: 0.8821
- cluster ARI: -0.0130

- Score distribution:
  - known mean 0.0254, median 0.1339, p75 0.9584
  - unknown mean 0.3551, median 0.3727, p75 1.1412

- Biggest known-class rejection rates:
  - class 83 (sweet_pepper): reject_rate=0.273, accuracy=0.636, count=11
  - class 9 (bottle): reject_rate=0.250, accuracy=0.250, count=8
  - class 96 (willow_tree): reject_rate=0.250, accuracy=0.000, count=8
  - class 8 (bicycle): reject_rate=0.222, accuracy=0.333, count=9
  - class 59 (pine_tree): reject_rate=0.200, accuracy=0.000, count=10

- Biggest novel-class false-accept rates:
  - class 53 (orange): false_accept_rate=1.000, correct_reject_rate=0.000, count=4
  - class 29 (dinosaur): false_accept_rate=1.000, correct_reject_rate=0.000, count=5
  - class 14 (butterfly): false_accept_rate=1.000, correct_reject_rate=0.000, count=6
  - class 54 (orchid): false_accept_rate=1.000, correct_reject_rate=0.000, count=6
  - class 97 (wolf): false_accept_rate=1.000, correct_reject_rate=0.000, count=7

### discovery_compare_ntxent_unknown_detect [normalized_entropy_mahalanobis]

- AUROC: 0.5706
- AUPR: 0.4657
- FPR95: 0.9194
- OSCR: 0.1799
- known accept rate: 0.9342
- unknown reject rate: 0.0969
- known class acc after accept: 0.2641
- known class acc all known: 0.2467
- cluster NMI: 0.8158
- cluster ARI: 0.0377

- Score distribution:
  - known mean 0.0928, median 0.2342, p75 0.9634
  - unknown mean 0.4369, median 0.5593, p75 1.1897

- Biggest known-class rejection rates:
  - class 9 (bottle): reject_rate=0.500, accuracy=0.375, count=8
  - class 59 (pine_tree): reject_rate=0.300, accuracy=0.100, count=10
  - class 70 (rose): reject_rate=0.250, accuracy=0.000, count=8
  - class 18 (caterpillar): reject_rate=0.222, accuracy=0.111, count=9
  - class 30 (dolphin): reject_rate=0.200, accuracy=0.200, count=5

- Biggest novel-class false-accept rates:
  - class 53 (orange): false_accept_rate=1.000, correct_reject_rate=0.000, count=4
  - class 29 (dinosaur): false_accept_rate=1.000, correct_reject_rate=0.000, count=5
  - class 17 (castle): false_accept_rate=1.000, correct_reject_rate=0.000, count=5
  - class 14 (butterfly): false_accept_rate=1.000, correct_reject_rate=0.000, count=6
  - class 54 (orchid): false_accept_rate=1.000, correct_reject_rate=0.000, count=6

### discovery_compare_ntxent_unknown002_detect [normalized_entropy_mahalanobis]

- AUROC: 0.5482
- AUPR: 0.4202
- FPR95: 0.8947
- OSCR: 0.1600
- known accept rate: 0.9572
- unknown reject rate: 0.0587
- known class acc after accept: 0.2302
- known class acc all known: 0.2204
- cluster NMI: 0.9012
- cluster ARI: 0.0474

- Score distribution:
  - known mean -0.0261, median 0.0584, p75 0.9520
  - unknown mean 0.2144, median 0.3337, p75 1.1082

- Biggest known-class rejection rates:
  - class 59 (pine_tree): reject_rate=0.200, accuracy=0.100, count=10
  - class 52 (oak_tree): reject_rate=0.200, accuracy=0.100, count=10
  - class 83 (sweet_pepper): reject_rate=0.182, accuracy=0.636, count=11
  - class 68 (road): reject_rate=0.167, accuracy=0.500, count=6
  - class 51 (mushroom): reject_rate=0.154, accuracy=0.231, count=13

- Biggest novel-class false-accept rates:
  - class 53 (orange): false_accept_rate=1.000, correct_reject_rate=0.000, count=4
  - class 29 (dinosaur): false_accept_rate=1.000, correct_reject_rate=0.000, count=5
  - class 17 (castle): false_accept_rate=1.000, correct_reject_rate=0.000, count=5
  - class 14 (butterfly): false_accept_rate=1.000, correct_reject_rate=0.000, count=6
  - class 54 (orchid): false_accept_rate=1.000, correct_reject_rate=0.000, count=6
