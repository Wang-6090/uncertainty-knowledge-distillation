# Class-center feature KD pilot (seed 2137)

## Question and scope

This pilot tests whether transferring labelled between-class feature-center
directions improves the student representation over standard logits KD. It is
within Team A's classification representation and uncertainty-KD scope. It
does not modify the rejection encoder, detector, threshold rule, candidate
selection, or clustering algorithm.

The three paired arms are:

- `standard`: CE + standard logits KD;
- `class_center_feature`: the same objective plus
  `alpha_class_center_feat_kd=0.1`;
- `uncertainty`: CE + uncertainty-weighted logits KD.

All arms use the same semantic-hard CIFAR-100 60/40 split (seed 2137),
teacher, student initialization, four-epoch budget, known-validation
checkpoint selection, max-softmax evaluation, known-validation 95% coverage
calibration, and automatic K. Unknown labels are used only for final
evaluation.

## Mechanism audit

The class-center loss was nonzero in all four epochs:

```text
0.655439, 0.650050, 0.648794, 0.645664
```

All batches contained valid known labels for the loss (`active fraction=1.0`)
and the audited encoder gradient norm was positive:

```text
0.082992, 0.064009, 0.062849, 0.062762
```

The teacher feature and teacher class center are detached. The loss is added
to the total student objective only in the treatment arm. Standard and
uncertainty arms recorded zero class-center loss and zero class-center
gradient. The unit test also verifies that the student receives gradients
while the teacher does not.

## Results

| Metric | standard | class-center feature | uncertainty |
|---|---:|---:|---:|
| AUROC | 0.652335 | 0.646764 | 0.645364 |
| AUPR | 0.516147 | 0.509018 | 0.506896 |
| FPR95 | 0.811167 | 0.810333 | 0.813667 |
| Known acceptance | 96.000% | 93.983% | 93.917% |
| Accepted-known accuracy | 71.215% | 72.566% | 72.138% |
| Unknown rejection | 8.100% | 11.350% | 11.575% |
| Candidate purity | 57.447% | 55.706% | 55.918% |
| Auto-K absolute error | 4 | 8 | 5 |
| Cluster ACC | 0.163121 | 0.159509 | 0.170290 |
| Cluster NMI | 0.454358 | 0.418556 | 0.428289 |
| Cluster ARI | 0.032267 | 0.048119 | 0.048368 |

Relative to standard KD, the class-center arm changes AUROC/AUPR/FPR95 by
`-0.005571/-0.007130/-0.000833`, unknown rejection by `+3.25` percentage
points, candidate purity by `-1.741` percentage points, and cluster
ACC/NMI/ARI by `-0.361/-3.580/+1.585` percentage points. Auto-K absolute
error increases from `4` to `8`.

Known-validation representation diagnostics do not show a coherent geometric
improvement. At the final epoch, feature within-class mean changes from
`0.226133` to `0.219553` and nearest-center distance from `0.171792` to
`0.162457`, while center margin changes from `-0.054341` to `-0.057095`.
Known accuracy changes only from `0.705667` to `0.706000`.

## Decision

The mechanism is real and updates the student, but this one fresh split does
not show a joint improvement in known representation geometry, detection,
candidate purity, and clustering. Freeze the candidate as an opt-in method;
do not scan its coefficient or promote it to the default. The existing
standard-KD and uncertainty-KD comparison remains the primary result for this
scope. The raw machine result is
`analysis/team_a_class_center_feature_kd_20261122_s2137.json`.
