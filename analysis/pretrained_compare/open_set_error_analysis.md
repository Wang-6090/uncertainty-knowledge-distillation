# Open-set Error Analysis

## Cross-run comparison

| run | AUROC | FPR95 | known acc all known | unknown reject rate | known accept rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| cifar_pretrained_entropy_proto [entropy_proto] | 0.5307 | 0.8779 | 0.3443 | 0.0456 | 0.9512 |
| cifar_pretrained_full [proto_only] | 0.5718 | 0.8788 | 0.3587 | 0.0424 | 0.9550 |

### cifar_pretrained_entropy_proto [entropy_proto]

- AUROC: 0.5307
- FPR95: 0.8779
- known accept rate: 0.9512
- unknown reject rate: 0.0456
- known class acc after accept: 0.3619
- known class acc all known: 0.3443
- cluster NMI: 0.7648
- cluster ARI: -0.0072

- Score distribution:
  - known mean 1.4539, median 1.4867, p75 2.1568
  - unknown mean 1.5510, median 1.5577, p75 2.1686

- Biggest known-class rejection rates:
  - class 74 (shrew): reject_rate=0.182, accuracy=0.000, count=22
  - class 21 (chimpanzee): reject_rate=0.182, accuracy=0.000, count=22
  - class 2 (baby): reject_rate=0.150, accuracy=0.000, count=20
  - class 65 (rabbit): reject_rate=0.136, accuracy=0.000, count=22
  - class 38 (kangaroo): reject_rate=0.133, accuracy=0.000, count=15

- Biggest novel-class false-accept rates:
  - class 53 (orange): false_accept_rate=1.000, correct_reject_rate=0.000, count=11
  - class 14 (butterfly): false_accept_rate=1.000, correct_reject_rate=0.000, count=13
  - class 25 (couch): false_accept_rate=1.000, correct_reject_rate=0.000, count=14
  - class 57 (pear): false_accept_rate=1.000, correct_reject_rate=0.000, count=17
  - class 79 (spider): false_accept_rate=1.000, correct_reject_rate=0.000, count=17

### cifar_pretrained_full [proto_only]

- AUROC: 0.5718
- FPR95: 0.8788
- known accept rate: 0.9550
- unknown reject rate: 0.0424
- known class acc after accept: 0.3756
- known class acc all known: 0.3587
- cluster NMI: 0.7782
- cluster ARI: 0.0333

- Score distribution:
  - known mean 0.1604, median 0.1655, p75 0.2106
  - unknown mean 0.1765, median 0.1806, p75 0.2177

- Biggest known-class rejection rates:
  - class 48 (motorcycle): reject_rate=0.238, accuracy=0.000, count=21
  - class 33 (forest): reject_rate=0.174, accuracy=0.000, count=23
  - class 65 (rabbit): reject_rate=0.167, accuracy=0.000, count=12
  - class 96 (willow_tree): reject_rate=0.167, accuracy=0.000, count=18
  - class 56 (palm_tree): reject_rate=0.130, accuracy=0.000, count=23

- Biggest novel-class false-accept rates:
  - class 69 (rocket): false_accept_rate=1.000, correct_reject_rate=0.000, count=11
  - class 35 (girl): false_accept_rate=1.000, correct_reject_rate=0.000, count=14
  - class 31 (elephant): false_accept_rate=1.000, correct_reject_rate=0.000, count=16
  - class 82 (sunflower): false_accept_rate=1.000, correct_reject_rate=0.000, count=17
  - class 28 (cup): false_accept_rate=1.000, correct_reject_rate=0.000, count=17
