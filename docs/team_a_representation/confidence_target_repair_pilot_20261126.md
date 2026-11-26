# Teacher-confidence target-repair KD screening (2026-11-26)

## Question and fixed treatment

This Team A screening asks whether a known-label target correction can improve
student representations and transfer to unknown detection and novel-class
discovery, beyond standard KD and uncertainty-KD. The opt-in treatment uses
only labeled known training examples:

`q = p_teacher(y|x) * p_teacher_T + (1 - p_teacher(y|x)) * one_hot(y)`

Here `p_teacher(y|x)` is the teacher's detached true-class probability at
temperature 1, while `p_teacher_T` is the detached teacher distribution at the
KD temperature. The loss is `T^2 KL(q || p_student_T)`. It changes the target,
not the per-example KD mass. CE, the standard-KD control, uncertainty-KD,
student uncertainty head, unknown score, calibration, candidate selection,
Auto-K, and clustering implementation were not changed. `standard` remains
the default.

## Protocol and validity

One fresh semantic-hard CIFAR-100 60/40 seed (`2180`) was run as a screening
pilot with `standard`, `confidence_repair`, and `uncertainty` arms. Each arm
used the same newly trained split-matched teacher, four student epochs,
CE/KD-only objective, known-validation checkpoint selection, max-softmax,
known-validation 95% coverage calibration, matched 5,400-item discovery pool,
Auto-K and clustering. The automated configuration comparison found only the
declared `kd_mode` treatment difference, plus expected run/checkpoint paths
and result-derived checkpoint scores. The discovery evaluator and its
configuration were matched across arms.

Known training labels are used by CE and the treatment formula, as intended.
Unknown/test labels were not used for training, checkpoint selection,
calibration, or treatment selection; they are used only for final reported
evaluation. No hyperparameter scan was run. Raw machine output is local and
gitignored at `analysis/team_a_confidence_target_repair_20261126_s2180.json`;
run artifacts are under the matching `runs/` prefix.

## Mechanism gate

The confidence-repair branch was applied in all four epochs. Its mean
hard-label mixture coefficient (`1 - p_teacher(y|x)`) was
`0.423451/0.419369/0.421737/0.420826`; the mean across epochs is about
`0.421`. KD gradient norms were
`27.0108/26.1421/24.5654/23.4550`, and student classifier update norms were
`0.040738/0.030968/0.030424/0.029520`. In the standard and uncertainty arms,
the repair was inactive and its reported coefficient was zero. This confirms
that the treatment changed the detached KD target, generated KD gradients,
and coincided with nonzero student classifier updates. It does not by itself
establish that the update was useful.

## Results

| Metric | standard-KD | confidence repair | uncertainty-KD |
|---|---:|---:|---:|
| AUROC | 0.698365 | 0.689899 | 0.699508 |
| AUPR | 0.564529 | 0.551951 | 0.564904 |
| FPR95 | 0.732333 | 0.743833 | 0.721333 |
| Known accuracy | 0.706500 | 0.713000 | 0.708500 |
| Unknown rejection at calibrated threshold | 0.116500 | 0.144000 | 0.142000 |
| Candidate count | 740 | 944 | 913 |
| Candidate purity | 0.629730 | 0.610169 | 0.622125 |
| Auto-K (absolute error; true K=40) | 43 (3) | 42 (2) | 37 (3) |
| Auto-K candidate clustering ACC/NMI/ARI | 0.168919 / 0.442622 / 0.036050 | 0.189619 / 0.459267 / 0.065851 | 0.176342 / 0.409864 / 0.057552 |
| All-unknown oracle clustering ACC/NMI/ARI | 0.392500 / 0.469141 / 0.230276 | 0.399750 / 0.481168 / 0.255584 | 0.400000 / 0.469370 / 0.239093 |

Relative to standard-KD, confidence repair changes AUROC/AUPR by
`-0.008467/-0.012578`, FPR95 by `+1.150 pp`, known accuracy by `+0.650 pp`,
unknown rejection by `+2.750 pp`, candidate purity by `-1.956 pp`, and
candidate count by `+204`. Auto-K absolute error falls from 3 to 2. The
candidate-pool Auto-K clustering ACC/NMI/ARI rise by
`+2.070/+1.665/+2.980 pp`; all-unknown oracle ACC/NMI/ARI rise by
`+0.725/+1.203/+2.531 pp`. Relative to uncertainty-KD, confidence repair has
lower AUROC/AUPR by `0.000961/0.012953`, similar unknown rejection
(`+0.200 pp`), and higher candidate clustering ACC/NMI/ARI by
`+1.328/+4.940/+0.830 pp`.

## Interpretation and decision

This is mixed, single-seed evidence, not a demonstrated general improvement.
The repair improves known accuracy, unknown rejection, Auto-K error, and
multiple clustering measures, but worsens ranking-based detection, increases
the selected candidate count substantially, and reduces candidate purity.
Thus target correction may help the selected representation's clusterability
without improving the full detection/candidate pipeline consistently. This is
not a replacement for uncertainty-KD: uncertainty-KD retains slightly better
AUROC/AUPR and the confidence repair's candidate pool is less pure than both
controls. Neither arm is promoted based on this one split.

## Fixed fresh-seed confirmation

Following the screening rule, one new semantic-hard seed (`2181`) was run with
the same three arms, formula, and evaluator. The config gate again found only
the declared `kd_mode` treatment (besides checkpoint paths and
known-validation-derived checkpoint scores); test labels were not used for
training, selection, or calibration. Confidence repair was active in all four
epochs, with mean hard-label coefficients
`0.427448/0.429332/0.428607/0.427599`, KD gradient norms
`28.6109/26.1303/24.9846/23.9036`, and classifier update norms
`0.040138/0.030589/0.029762/0.029922`.

| Metric | standard-KD | confidence repair | uncertainty-KD |
|---|---:|---:|---:|
| AUROC | 0.688144 | 0.685428 | 0.689557 |
| AUPR | 0.539988 | 0.545384 | 0.543572 |
| FPR95 | 0.716833 | 0.752167 | 0.723333 |
| Known accuracy | 0.705833 | 0.715500 | 0.701833 |
| Unknown rejection at calibrated threshold | 0.142750 | 0.136750 | 0.125250 |
| Candidate count | 965 | 928 | 862 |
| Candidate purity | 0.591710 | 0.589440 | 0.581206 |
| Auto-K (absolute error; true K=40) | 46 (6) | 38 (2) | 30 (10) |
| Auto-K candidate clustering ACC/NMI/ARI | 0.186528 / 0.441073 / 0.059752 | 0.177802 / 0.436419 / 0.054460 | 0.163573 / 0.402208 / 0.053174 |
| All-unknown oracle clustering ACC/NMI/ARI | 0.410500 / 0.481250 / 0.244995 | 0.438000 / 0.500714 / 0.266167 | 0.375500 / 0.472796 / 0.227651 |

Relative to standard, confidence repair has AUROC `-0.002716`, AUPR
`+0.005396`, FPR95 `+3.533 pp`, known accuracy `+0.967 pp`, unknown rejection
`-0.600 pp`, candidate purity `-0.227 pp`, and candidate Auto-K
ACC/NMI/ARI `-0.873/-0.465/-0.529 pp`. Auto-K error again falls, `6→2`.
All-unknown oracle ACC/NMI/ARI increase by `+2.750/+1.946/+2.117 pp`.
The latter is a diagnostic using final unknown labels and is not a selectable
training or checkpoint criterion.

## Interpretation and decision

Across the two seeds, all-unknown oracle clustering improves on all three
metrics in both splits, suggesting a repeatable global representation signal
under this diagnostic. However, candidate-pool clustering improves on all
three metrics only in seed 2180 and declines on all three in seed 2181. AUROC
is lower than standard in both seeds; candidate purity is also lower in both.
Unknown rejection changes sign, and AUPR changes sign. Auto-K absolute error
improves in both seeds, but that alone does not establish reliable end-to-end
discovery. The splits are a small descriptive sample, not independent
statistical evidence.

The fixed confirmation therefore fails the intended transfer gate. Freeze
confidence target repair as an opt-in ablation; do not scan the mixture
strength, repeat this hypothesis on more seeds, or change the default. Keep
standard-KD as the baseline and uncertainty-KD as a separate comparator. The
useful bounded finding is that the target repair may improve global feature
clusterability under oracle full-unknown analysis, but the gain is not
reliably preserved by detection and candidate selection. Team A should next
target the known-only training objective's relationship to global class
structure and audit it on known validation before another full downstream
experiment; downstream rejector, candidate, Auto-K, and clustering behavior
remain outside this implementation scope.

Implementation tests and the full suite passed (`419 passed`, one existing
warning). Confirmation machine output is local and gitignored at
`analysis/team_a_confidence_target_repair_confirm_20261126_s2181.json`; its
run artifacts are under the matching `runs/` prefix.
