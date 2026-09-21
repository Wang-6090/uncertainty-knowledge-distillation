# Open-set Error Analysis

## Cross-run comparison

| run | AUROC | FPR95 | known acc all known | unknown reject rate | known accept rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| cifar_full_data_entropy_proto [entropy_proto] | 0.6148 | 0.8323 | 0.4590 | 0.0653 | 0.9482 |

### cifar_full_data_entropy_proto [entropy_proto]

- AUROC: 0.6148
- FPR95: 0.8323
- known accept rate: 0.9482
- unknown reject rate: 0.0653
- known class acc after accept: 0.4841
- known class acc all known: 0.4590
- cluster NMI: 0.4860
- cluster ARI: 0.0358

- Score distribution:
  - known mean 1.3481, median 1.2850, p75 2.0969
  - unknown mean 1.6998, median 1.7846, p75 2.3486

- Biggest known-class rejection rates:
  - class 42 (leopard): reject_rate=0.170, accuracy=0.000, count=100
  - class 44 (lizard): reject_rate=0.150, accuracy=0.000, count=100
  - class 65 (rabbit): reject_rate=0.140, accuracy=0.000, count=100
  - class 33 (forest): reject_rate=0.120, accuracy=0.000, count=100
  - class 38 (kangaroo): reject_rate=0.120, accuracy=0.000, count=100

- Biggest novel-class false-accept rates:
  - class 17 (castle): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 86 (telephone): false_accept_rate=0.990, correct_reject_rate=0.010, count=100
  - class 71 (sea): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
  - class 53 (orange): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
  - class 35 (girl): false_accept_rate=0.980, correct_reject_rate=0.020, count=100
