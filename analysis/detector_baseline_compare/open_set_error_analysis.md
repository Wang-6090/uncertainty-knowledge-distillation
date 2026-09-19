# Open-set Error Analysis

## Cross-run comparison

| run | AUROC | AUPR | FPR95 | OSCR | known acc all known | unknown reject rate | candidate purity | estimated K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| quick_de_discovery_energy_norm_maha_detect1000 [normalized_entropy_mahalanobis] | 0.5505 | 0.4157 | 0.9276 | 0.2496 | 0.3470 | 0.0561 | 0.4000 | 19 |
| quick_de_discovery_energy_odin_eps_0p0005 [odin_msp] | 0.5414 | 0.4209 | 0.8964 | 0.2501 | 0.3536 | 0.0791 | 0.4306 | 19 |
| quick_de_discovery_energy_shared_maha_fast_detect1000 [normalized_entropy_mahalanobis_shared] | 0.5207 | 0.3883 | 0.9178 | 0.2327 | 0.3372 | 0.0536 | 0.3621 | 19 |

### quick_de_discovery_energy_norm_maha_detect1000 [normalized_entropy_mahalanobis]

- AUROC: 0.5505
- AUPR: 0.4157
- FPR95: 0.9276
- OSCR: 0.2496
- known accept rate: 0.9457
- unknown reject rate: 0.0561
- known class acc after accept: 0.3670
- known class acc all known: 0.3470
- ECE: 0.2860
- temperature: 1.0000
- uncertainty/error correlation: -0.1173
- all-unknown oracle NMI: 0.5176
- candidate-unknown oracle NMI: 0.9066
- cluster NMI: 0.7829
- cluster ARI: 0.0079
- candidate pool: 55 samples, true unknown 22, false rejects 33, purity 0.4000
- cluster K: estimated 19, true 40, absolute error 21

- Score distribution:
  - known mean 0.1217, median 0.1662, p75 1.2649
  - unknown mean 0.3862, median 0.4231, p75 1.3684

- Biggest known-class rejection rates:
  - class 16 (can): reject_rate=0.300, accuracy=0.300, count=10
  - class 48 (motorcycle): reject_rate=0.286, accuracy=0.143, count=7
  - class 58 (pickup_truck): reject_rate=0.250, accuracy=0.417, count=12
  - class 8 (bicycle): reject_rate=0.222, accuracy=0.222, count=9
  - class 44 (lizard): reject_rate=0.167, accuracy=0.083, count=12

- Biggest novel-class false-accept rates:
  - class 53 (orange): false_accept_rate=1.000, correct_reject_rate=0.000, count=4
  - class 29 (dinosaur): false_accept_rate=1.000, correct_reject_rate=0.000, count=5
  - class 17 (castle): false_accept_rate=1.000, correct_reject_rate=0.000, count=5
  - class 14 (butterfly): false_accept_rate=1.000, correct_reject_rate=0.000, count=6
  - class 54 (orchid): false_accept_rate=1.000, correct_reject_rate=0.000, count=6

### quick_de_discovery_energy_odin_eps_0p0005 [odin_msp]

- AUROC: 0.5414
- AUPR: 0.4209
- FPR95: 0.8964
- OSCR: 0.2501
- known accept rate: 0.9326
- unknown reject rate: 0.0791
- known class acc after accept: 0.3792
- known class acc all known: 0.3536
- ECE: 0.2860
- temperature: 1.0000
- uncertainty/error correlation: -0.1173
- ODIN: epsilon 0.0005, temperature 1000.0000
- all-unknown oracle NMI: 0.5176
- candidate-unknown oracle NMI: 0.8229
- cluster NMI: 0.7120
- cluster ARI: -0.0068
- candidate pool: 72 samples, true unknown 31, false rejects 41, purity 0.4306
- cluster K: estimated 19, true 40, absolute error 21

- Score distribution:
  - known mean 0.9832, median 0.9832, p75 0.9832
  - unknown mean 0.9832, median 0.9832, p75 0.9832

- Biggest known-class rejection rates:
  - class 78 (snake): reject_rate=0.400, accuracy=0.200, count=10
  - class 52 (oak_tree): reject_rate=0.300, accuracy=0.000, count=10
  - class 55 (otter): reject_rate=0.273, accuracy=0.000, count=11
  - class 10 (bowl): reject_rate=0.222, accuracy=0.111, count=9
  - class 59 (pine_tree): reject_rate=0.200, accuracy=0.000, count=10

- Biggest novel-class false-accept rates:
  - class 53 (orange): false_accept_rate=1.000, correct_reject_rate=0.000, count=4
  - class 29 (dinosaur): false_accept_rate=1.000, correct_reject_rate=0.000, count=5
  - class 14 (butterfly): false_accept_rate=1.000, correct_reject_rate=0.000, count=6
  - class 54 (orchid): false_accept_rate=1.000, correct_reject_rate=0.000, count=6
  - class 57 (pear): false_accept_rate=1.000, correct_reject_rate=0.000, count=7

### quick_de_discovery_energy_shared_maha_fast_detect1000 [normalized_entropy_mahalanobis_shared]

- AUROC: 0.5207
- AUPR: 0.3883
- FPR95: 0.9178
- OSCR: 0.2327
- known accept rate: 0.9391
- unknown reject rate: 0.0536
- known class acc after accept: 0.3590
- known class acc all known: 0.3372
- ECE: 0.2860
- temperature: 1.0000
- uncertainty/error correlation: -0.1173
- all-unknown oracle NMI: 0.5176
- candidate-unknown oracle NMI: 0.7999
- cluster NMI: 0.7525
- cluster ARI: -0.0214
- candidate pool: 58 samples, true unknown 21, false rejects 37, purity 0.3621
- cluster K: estimated 19, true 40, absolute error 21

- Score distribution:
  - known mean 0.0929, median 0.0545, p75 1.0564
  - unknown mean 0.1708, median 0.2077, p75 0.9282

- Biggest known-class rejection rates:
  - class 18 (caterpillar): reject_rate=0.444, accuracy=0.333, count=9
  - class 44 (lizard): reject_rate=0.250, accuracy=0.083, count=12
  - class 39 (keyboard): reject_rate=0.222, accuracy=0.333, count=9
  - class 78 (snake): reject_rate=0.200, accuracy=0.200, count=10
  - class 12 (bridge): reject_rate=0.182, accuracy=0.091, count=11

- Biggest novel-class false-accept rates:
  - class 53 (orange): false_accept_rate=1.000, correct_reject_rate=0.000, count=4
  - class 29 (dinosaur): false_accept_rate=1.000, correct_reject_rate=0.000, count=5
  - class 14 (butterfly): false_accept_rate=1.000, correct_reject_rate=0.000, count=6
  - class 54 (orchid): false_accept_rate=1.000, correct_reject_rate=0.000, count=6
  - class 97 (wolf): false_accept_rate=1.000, correct_reject_rate=0.000, count=7
