# Open-set Error Analysis

## Cross-run comparison

| run | AUROC | AUPR | FPR95 | OSCR | known acc all known | unknown reject rate | candidate purity | estimated K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| quick_de_discovery_energy_odin_eps_0p0005 [odin_msp] | 0.5414 | 0.4209 | 0.8964 | 0.2501 | 0.3536 | 0.0791 | 0.4306 | 19 |

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
