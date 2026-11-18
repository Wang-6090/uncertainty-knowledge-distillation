# Uncertainty-Gated Decoupled KD: Research Protocol

## Motivation and prior evidence

Team A's existing uncertainty-KD experiments repeatedly show that detached
`exp(-u)` weighting of the full per-example KL has near-unit effective sample
size and mixed downstream effects. This means the weighting is active but
barely redistributes training signal. Repeating a search over the same weight
function, source, or coefficient is not justified.

Decoupled Knowledge Distillation (DKD) separates target-class versus
non-target probability transfer from the conditional distribution among
non-target classes. This suggests a narrower and testable use of uncertainty:
keep target supervision intact, but gate the teacher's non-target relational
knowledge when teacher epistemic uncertainty is high. The reference is Zhao
et al., *Decoupled Knowledge Distillation*, CVPR 2022,
[DOI 10.1109/CVPR.2022.01165](https://doi.org/10.1109/CVPR.2022.01165).
The paper motivates the decomposition; it does not validate this
uncertainty-gated variant or its use for novel-class discovery.

## Hypothesis

On semantic-hard CIFAR-100 60/40 splits, uncertainty-gating only the
non-target component of decoupled KD will improve the transfer of useful
inter-class relations compared with both standard logits-KD and unweighted
DKD, without reducing known-class utility or downstream discovery quality.
This hypothesis can fail: teacher uncertainty may not identify unreliable
non-target relations, and even useful non-target relations may be important on
hard samples.

## Definition

For sample `i`, known label `y`, temperature `T`, teacher uncertainty `u_i`,
and batch-mean-normalized detached weight
`w_i = exp(-u_i) / mean_j(exp(-u_j))`:

```text
L_UA-DKD(i) = T^2 * (KL_target-vs-rest(i) + w_i * KL_conditional-non-target(i))
L_total = L_CE + alpha_kd * mean_i(L_UA-DKD(i)) + unchanged auxiliary losses
```

The target-vs-rest term compares the teacher and student probability mass on
the ground-truth class versus all other classes. The conditional non-target
term compares distributions after excluding the ground-truth class. The
uncertainty signal is detached and weights only the latter. CE remains active
for the ground-truth target. No unknown labels or test-derived values enter
the loss. The initial implementation uses `mc_epistemic`, four stochastic
teacher passes, and mean-normalized weights to match the prior Team A MC-KD
protocol; no weight-function or coefficient scan is planned.

## Comparison and staged execution

Stage 1 is implementation validation: compare helper outputs to explicit
formula calculations, verify finite loss, student-only gradients, detached
teacher/uncertainty, and invalid-label rejection. This is completed in
`tests/test_uncertainty_kd.py`; it is not performance evidence.

Stage 2 is one fresh-split pilot, proposed seed `2132`, only after a runner
preflight confirms that no split, result, or run directory already exists.
Three arms share one split-matched teacher, student initialization, data order,
optimizer, training budget, all auxiliary losses, checkpoint selection,
known-only calibration, detector, and Auto-K procedure:

| Arm | Main logits objective | Purpose |
|---|---|---|
| B | standard KL on shared MC-mean teacher logits | established control |
| D | unweighted DKD | isolate the effect of target/non-target decomposition |
| U | uncertainty-gated DKD | test the proposed uncertainty mechanism |

Only the main logits objective differs. Feature KD and rejection-branch KD
remain disabled. The pilot is exploratory and cannot promote the method.

Stage 3, only if the pilot passes its pre-registered screening rule, repeats
the same three arms on two additional fresh semantic-hard seeds before any
promotion decision. Do not tune on the pilot or reuse the formal 42/123/3407
class sets as if they were independent replications.

## Promotion screen and reporting

The primary comparison is U versus B in the complete pipeline, with D used to
attribute any change to decomposition versus uncertainty gating. A pilot only
continues if U improves at least one ranking metric (AUROC or OSCR) and one
downstream metric (candidate purity or Auto-K clustering NMI/ARI), while known
accuracy and known acceptance do not fall by more than 1 percentage point and
no other primary metric shows a material decline. This is a screening gate,
not a significance claim. Report every arm and all seeds, including failures.

Mechanism audit per epoch must include the MC uncertainty source, weight mean,
standard deviation and ESS fraction, separate target/non-target KD losses,
the total KD gradient norm, and the student parameter update check. Unknown
labels remain excluded from training, checkpoint selection, calibration,
Auto-K, and promotion decisions. They are used only for final metrics.

## Implementation status

The `uncertainty_decoupled` opt-in loss helper and CLI mode are implemented.
The existing default remains `standard`. The seed-2132 pilot completed with
the three planned arms and passed the exploratory screening rule, so the next
action is confirmation on two fresh semantic-hard seeds; this is not a final
promotion decision. The paired runner supports `--objective-profile ua_dkd`,
strict no-overwrite checks, a `--dry-run` preflight, configuration audits,
per-epoch component extraction, gradient checks, and optimizer-update checks.
The pilot's full metrics and audit interpretation are recorded in
`EXPERIMENT_LOG.md`.
