# Open-set Error Analysis

## Cross-run comparison

| run | AUROC | FPR95 | known acc all known | unknown reject rate | known accept rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| cifar_mid_full [full] | 0.5749 | 0.8746 | 0.2559 | 0.0776 | 0.9402 |
| cifar_mid_entropy_proto [entropy_proto] | 0.5745 | 0.8729 | 0.2559 | 0.0862 | 0.9369 |
| cifar_proto_full [full] | 0.5877 | 0.8796 | 0.2685 | 0.0751 | 0.9394 |
| cifar_proto_entropy_proto [entropy_proto] | 0.5881 | 0.8695 | 0.2685 | 0.0727 | 0.9402 |

### cifar_mid_full [full]

- AUROC: 0.5749
- FPR95: 0.8746
- known accept rate: 0.9402
- unknown reject rate: 0.0776
- known class acc after accept: 0.2722
- known class acc all known: 0.2559
- cluster NMI: 0.7351
- cluster ARI: 0.0363

- Score distribution:
  - known mean 1.4833, median 1.5987, p75 1.8140
  - unknown mean 1.6050, median 1.6900, p75 1.8455

- Biggest known-class rejection rates:
  - class 44 (lizard): reject_rate=0.231, accuracy=0.000, count=26
  - class 45 (lobster): reject_rate=0.188, accuracy=0.000, count=16
  - class 37 (house): reject_rate=0.167, accuracy=0.000, count=18
  - class 99 (worm): reject_rate=0.167, accuracy=0.000, count=18
  - class 58 (pickup_truck): reject_rate=0.158, accuracy=0.000, count=19

- Biggest novel-class false-accept rates:
  - class 53 (orange): false_accept_rate=1.000, correct_reject_rate=0.000, count=11
  - class 29 (dinosaur): false_accept_rate=1.000, correct_reject_rate=0.000, count=15
  - class 28 (cup): false_accept_rate=1.000, correct_reject_rate=0.000, count=15
  - class 93 (turtle): false_accept_rate=1.000, correct_reject_rate=0.000, count=20
  - class 4 (beaver): false_accept_rate=1.000, correct_reject_rate=0.000, count=24

### cifar_mid_entropy_proto [entropy_proto]

- AUROC: 0.5745
- FPR95: 0.8729
- known accept rate: 0.9369
- unknown reject rate: 0.0862
- known class acc after accept: 0.2731
- known class acc all known: 0.2559
- cluster NMI: 0.7223
- cluster ARI: 0.0491

- Score distribution:
  - known mean 2.4908, median 2.6946, p75 3.1042
  - unknown mean 2.7154, median 2.8695, p75 3.1650

- Biggest known-class rejection rates:
  - class 44 (lizard): reject_rate=0.231, accuracy=0.000, count=26
  - class 8 (bicycle): reject_rate=0.190, accuracy=0.000, count=21
  - class 45 (lobster): reject_rate=0.188, accuracy=0.000, count=16
  - class 51 (mushroom): reject_rate=0.174, accuracy=0.000, count=23
  - class 37 (house): reject_rate=0.167, accuracy=0.000, count=18

- Biggest novel-class false-accept rates:
  - class 53 (orange): false_accept_rate=1.000, correct_reject_rate=0.000, count=11
  - class 28 (cup): false_accept_rate=1.000, correct_reject_rate=0.000, count=15
  - class 4 (beaver): false_accept_rate=1.000, correct_reject_rate=0.000, count=24
  - class 67 (ray): false_accept_rate=1.000, correct_reject_rate=0.000, count=27
  - class 69 (rocket): false_accept_rate=1.000, correct_reject_rate=0.000, count=29

### cifar_proto_full [full]

- AUROC: 0.5877
- FPR95: 0.8796
- known accept rate: 0.9394
- unknown reject rate: 0.0751
- known class acc after accept: 0.2858
- known class acc all known: 0.2685
- cluster NMI: 0.7276
- cluster ARI: 0.0079

- Score distribution:
  - known mean 1.3831, median 1.4887, p75 1.7432
  - unknown mean 1.5306, median 1.6079, p75 1.7942

- Biggest known-class rejection rates:
  - class 15 (camel): reject_rate=0.333, accuracy=0.000, count=18
  - class 38 (kangaroo): reject_rate=0.267, accuracy=0.000, count=15
  - class 66 (raccoon): reject_rate=0.192, accuracy=0.000, count=26
  - class 47 (maple_tree): reject_rate=0.150, accuracy=0.000, count=20
  - class 63 (porcupine): reject_rate=0.150, accuracy=0.000, count=20

- Biggest novel-class false-accept rates:
  - class 53 (orange): false_accept_rate=1.000, correct_reject_rate=0.000, count=11
  - class 14 (butterfly): false_accept_rate=1.000, correct_reject_rate=0.000, count=13
  - class 28 (cup): false_accept_rate=1.000, correct_reject_rate=0.000, count=15
  - class 57 (pear): false_accept_rate=1.000, correct_reject_rate=0.000, count=17
  - class 0 (apple): false_accept_rate=1.000, correct_reject_rate=0.000, count=18

### cifar_proto_entropy_proto [entropy_proto]

- AUROC: 0.5881
- FPR95: 0.8695
- known accept rate: 0.9402
- unknown reject rate: 0.0727
- known class acc after accept: 0.2856
- known class acc all known: 0.2685
- cluster NMI: 0.7218
- cluster ARI: 0.0175

- Score distribution:
  - known mean 2.3377, median 2.5234, p75 2.9986
  - unknown mean 2.6090, median 2.7480, p75 3.0917

- Biggest known-class rejection rates:
  - class 15 (camel): reject_rate=0.333, accuracy=0.000, count=18
  - class 38 (kangaroo): reject_rate=0.267, accuracy=0.000, count=15
  - class 66 (raccoon): reject_rate=0.192, accuracy=0.000, count=26
  - class 47 (maple_tree): reject_rate=0.150, accuracy=0.000, count=20
  - class 63 (porcupine): reject_rate=0.150, accuracy=0.000, count=20

- Biggest novel-class false-accept rates:
  - class 14 (butterfly): false_accept_rate=1.000, correct_reject_rate=0.000, count=13
  - class 25 (couch): false_accept_rate=1.000, correct_reject_rate=0.000, count=14
  - class 28 (cup): false_accept_rate=1.000, correct_reject_rate=0.000, count=15
  - class 57 (pear): false_accept_rate=1.000, correct_reject_rate=0.000, count=17
  - class 79 (spider): false_accept_rate=1.000, correct_reject_rate=0.000, count=17
