# Class centered feature KD protocol

## Scope

This stage belongs to Team A's student classification representation and
uncertainty KD work. It changes one training objective on the known labelled
student batches. It does not change the rejection encoder, rejection score,
candidate selection, threshold rule, clustering representation, or clustering
algorithm.

The completed formal CE/KD audit already showed that standard logits KD has at
most a weak seed sensitive representation effect and that uncertainty KD has
no stable additional downstream gain. Ordinary projection feature KD and
batch relational KD are historical directions and are not repeated here.

## Candidate mechanism

For each class represented by at least two samples in a known training batch,
let `f_s` and `f_t` be the student and frozen teacher classification backbone
features. The candidate removes the batch class mean and transfers the
within class direction:

```text
f'_s = normalize(f_s - mean_class(f_s))
f'_t = normalize(f_t - mean_class(f_t))
L_centered = mean(1 - cosine(f'_s, f'_t))
```

Singleton classes are skipped because they contain no within class direction.
Teacher features and class centers are detached. The loss is multiplied by a
fixed `alpha_centered_feat_kd=0.1`; this is a predeclared value matching the
scale used by the repository's existing feature KD control, with no pilot
weight scan.

The ordinary hard label CE and standard/uncertainty logits KD remain separate.
The candidate uses known training labels only. It does not use unknown labels,
discovery-pool identities, test labels, student uncertainty, or final
rejection scores.

## Staged execution

1. Implementation gate: formula, shape checks, singleton handling, finite
   loss, student-only gradient, teacher detachment, and default-off behavior
   must pass unit tests and compilation checks.
2. Pipeline gate: one fresh semantic-hard split must complete a three-arm
   paired run with CE-only auxiliary objectives:

   - `standard`: CE + standard logits KD;
   - `centered_feature`: CE + standard logits KD + centered feature KD;
   - `uncertainty`: CE + uncertainty weighted logits KD.

   The three arms share the split matched teacher, initialization, data order,
   optimizer, four epoch budget, checkpoint selection, known only calibration,
   max softmax evaluation, and automatic K procedure. Feature KD and all
   rejection/discovery losses are disabled. The pilot uses fresh semantic-hard
   seed `2135`, independent output paths, and no reuse of seeds 42/123/3407.

3. The centered arm must report a nonzero centered KD loss, repeated class
   fraction, centered KD encoder gradient, and student classifier update. If
   the path is inactive or the paired configuration differs unexpectedly, the
   run is invalid and no performance conclusion is made.
4. The pilot is exploratory. Continue to fresh confirmation seeds only if the
   centered arm improves a known-only representation diagnostic and at least
   one held out ranking or clustering metric without a material known utility
   loss. Otherwise stop this mechanism without scanning its coefficient or
   reusing the split.

## Reporting

Report the raw per-arm detection and clustering metrics, paired deltas against
standard KD, and the existing uncertainty-KD comparison separately. Report
known-only feature geometry and probe diagnostics separately from unknown-test
metrics. Unknown labels are read only once for final retrospective evaluation;
they do not determine training, checkpoint selection, calibration, automatic K,
or promotion.
