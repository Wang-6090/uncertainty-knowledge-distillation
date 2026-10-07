# Uncertainty-gate alternatives for discovery feature margin (2026-10-05)

## Why these trials were run

The original feature-margin treatment uses the student's detached uncertainty
as a soft weight on mixed discovery samples. A post-hoc audit showed that this
head is weaker than `1-MSP` as an unknownness ranking signal on the same pool,
so this round tests alternative weight sources while keeping the feature
margin, nnPU loss, teacher, split, optimizer, detector, and threshold fixed.

The experiments are diagnostic alternatives, not final claims. The core
question remains whether the training intervention reduces known/unknown
feature overlap without simply increasing known rejection.

## Gate audit before training

On the full mixed pool (known prior 0.212598), the learned uncertainty gate
had unknownness AUROC 0.6444/0.6623/0.5876 for seeds 42/123/3407, while `1-MSP`
had 0.6822/0.7046/0.6891. The top 25% uncertainty-selected samples had unknown
precision 0.862/0.874/0.822, meaning known contamination remained
0.138/0.126/0.178. The uncertainty weights had ESS fractions 0.957/0.965/0.975,
so the original weighted loss was close to an almost-uniform penalty rather
than a sharply purified unknown subset. These labels were used only for this
post-hoc audit, never for fitting or calibration.

## Controlled alternatives

All three runs use CIFAR-100 random 60/40, seed 42, full data, matched
pretrained ResNet-34 teacher and ResNet-18 student, 5 epochs, mixed pool with
known ratio 0.2, nnPU weight 0.1, feature-margin weight 0.05 and margin 0.2.
Detection uses normalized entropy + min-class kNN, k=10, MC=8, and a known
validation threshold targeting 95% known coverage. Only the gate source is
changed relative to the original treatment.

| Method | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection | Accepted-known accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nnPU baseline | 0.7012 | 0.7590 | 0.4110 | 95.55% | 13.40% | 51.19% |
| Uncertainty gate | 0.7108 | 0.7348 | 0.4132 | 95.30% | 13.90% | 50.87% |
| `1-MSP` gate | 0.6874 | 0.7603 | 0.3856 | 94.72% | 13.35% | 48.64% |
| Uncertainty × `1-MSP` gate | 0.7010 | 0.7477 | 0.4083 | 94.92% | 13.08% | 51.01% |

## Interpretation

Replacing uncertainty with `1-MSP` did not improve the representation: all
main utility metrics were below the original uncertainty-gate treatment, and
AUROC fell below the nnPU baseline. The conservative product gate recovered
some of the damage from `1-MSP`, but still did not match the original gate:
AUROC and unknown rejection were lower, with a small known-utility reduction.
This rejects “use MSP directly as the feature-margin weight” and provides no
reason to promote the product gate.

The result also suggests that the benefit of alpha=0.05 is not explained only
by selecting the most OOD-looking samples. The uncertainty head may be a noisy
but task-coupled training signal, or the margin's effect may depend on a broad
soft pressure rather than a purified candidate subset.

## Code and validation

`train.py` now exposes the optional
`--discovery-feature-margin-weight-source` values `uncertainty`, `msp`,
`mean`, `max`, and `product`; default remains `uncertainty`. The product gate
is `uncertainty * (1 - MSP)` and requires no unknown labels. Compilation,
toy msp smoke (with a valid teacher checkpoint), and the full test suite pass;
the invalid toy nnPU command with zero observed known pool was correctly
rejected and produced no result.

## Temporal alternatives

Two follow-ups were run without changing the detector or data protocol.

| Method | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection | Accepted-known accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original uncertainty gate | 0.7108 | 0.7348 | 0.4132 | 95.30% | 13.90% | 50.87% |
| EMA uncertainty gate, decay 0.99 | 0.6943 | 0.7563 | 0.3888 | 95.10% | 13.73% | 48.76% |
| Current-student gate, warmup=1/ramp=2 | 0.7007 | 0.7550 | 0.4276 | 94.78% | 14.30% | 53.65% |

The EMA gate is negative: it does not support the hypothesis that current
student self-feedback is the dominant failure cause. The warmup/ramp schedule
improves OSCR, accepted-known accuracy, and the selected operating-point
unknown rejection, but lowers AUROC and worsens FPR95. Full-data geometry for
the ramp variant is mixed: classifier-prototype AUROC/overlap changes
0.6639/0.7496 to 0.6597/0.7554, centroid changes 0.6704/0.7537 to
0.6925/0.7203, and nearest-sample changes 0.7074/0.6937 to 0.7105/0.6879.
Therefore this is a selective-utility tradeoff, not a reliable solution to
feature overlap. Keep scheduling optional and stop tuning warmup/EMA in this
family for now.

## Current decision

The original uncertainty gate remains the strongest candidate in this family,
because it is the only version with consistent three-seed ranking and geometry
improvement. `msp`, product, EMA, and warmup/ramp variants are retained as
diagnostic code/records but are not promoted to defaults. The next work should
move to a decoupled representation/rejector objective or a better calibrated
uncertainty target, rather than adding more static gates to the same margin.
