# Teacher-correct selective logits KD protocol

## Scope and motivation

This candidate remains within the student classification representation and
uncertainty-KD assignment. The recent CE-only A/B/C audit found no stable
representation or downstream gain from standard or uncertainty-KD. Quality
aware KD already tested continuous weighting by the teacher's true-class
probability; this experiment tests a distinct hard gate: transfer the teacher
soft target only when the teacher's top-1 prediction matches the known
training label.

The hypothesis is that teacher-wrong soft targets can distort the student
representation. Known labels are available during training, so the gate is
label-verified. The gate is separate from the teacher uncertainty head,
student uncertainty head, classification entropy, and final rejection score.
Samples excluded from KD still receive the ordinary classification CE loss.

## Objective

For sample `i`, define

```text
g_i = 1[argmax(teacher_logits_i) = y_i]
L_selective = mean_{i:g_i=1} KL(student_i || teacher_i)
```

If a batch has no teacher-correct sample, its selective KD contribution is a
finite zero; the batch remains trained by CE and the other fixed objectives.
The loss returns and records the selected fraction. Teacher logits are
detached and only student parameters are optimized.

## Pilot design

Use one fresh semantic-hard CIFAR-100 60/40 split, seed `2138`, with three
paired arms:

1. `standard`: CE + standard logits KD;
2. `selective`: CE + teacher-correct selective logits KD;
3. `uncertainty`: CE + existing uncertainty-KD.

All arms share the split-matched teacher, student initialization, data order,
four-epoch budget, known-validation checkpoint selection, max-softmax
discovery score, known-validation 95% coverage calibration, automatic K, and
clustering. Feature KD, class-center feature KD, rejection-branch KD,
discovery and PU training objectives are disabled. Unknown labels are used
only for final evaluation.

## Mechanism gates and decision rule

Before accepting any performance comparison, require the selective arm to
show a positive teacher-correct fraction in every epoch, an active selective
KD loss, nonzero audited KD gradient and nonzero student classifier update.
The standard arm must have zero selective activity; the uncertainty arm must
retain its separate nonuniform teacher-uncertainty weighting.

The pilot is promoted only if selective KD improves a detection metric and a
downstream candidate or clustering metric without material known-accuracy or
candidate-purity loss. Otherwise it is frozen as an opt-in diagnostic and no
gate threshold, coefficient, or additional seed is scanned.

## Pilot decision (seed 2138)

The mechanism gates passed and the pilot improved AUROC, AUPR, unknown
rejection, candidate purity, known accuracy, cluster ACC and ARI relative to
standard KD. Cluster NMI decreased, so this is evidence for promotion to a
fixed confirmation stage rather than evidence for changing the default. The
existing uncertainty-KD arm did not show an additional stable gain in this
pilot. The detailed result is recorded in
[`selective_kd_pilot_20261122.md`](selective_kd_pilot_20261122.md).

The fixed confirmation seeds are `2163` and `2164`. No gate threshold, loss
coefficient, temperature, or score rule may be scanned in that stage.

## Confirmation decision (seeds 2163 and 2164)

The mechanism gates passed on both fresh splits: teacher-correct selective KD
was active in every epoch, produced nonzero KD gradients, and updated the
student classifier. Standard-KD never applied uncertainty weights, while the
uncertainty arm applied its separate nonuniform weights in every epoch.

The performance gate did not pass. Selective KD improved known accuracy,
Cluster ACC/NMI/ARI, and Auto-K error on both seeds, but both seeds lost
unknown rejection and candidate purity. AUROC and AUPR changed in opposite
directions across seeds. The method is therefore frozen as an opt-in
diagnostic and does not change the default `kd_mode=standard`. The detailed
per-seed report is in
[`selective_kd_confirmation_20261122.md`](selective_kd_confirmation_20261122.md).
