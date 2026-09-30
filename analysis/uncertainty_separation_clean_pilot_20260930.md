# Clean uncertainty-head separation ablation (2026-09-30)

## Question and predeclared comparison

The uncertainty output in prior baseline checkpoints was effectively saturated
near one. This pilot asks whether the existing BCE known/unknown separation
loss makes the uncertainty head usable, and whether that training change also
helps the project's primary detector.

Within each seed, the only student-training change is
`alpha_discovery_uncertainty_separation: 0.0 -> 0.1` with
`discovery_uncertainty_loss=bce`, zero warmup, and zero ramp. The same CIFAR-100
random 60/40 class split, known train/validation/test/discovery sample limits
(1,200/300/1,000/1,200), ResNet-18 student, ResNet-34 teacher, fixed teacher
checkpoint, initialization protocol, three epochs, optimizer, and other loss
weights are used. Config snapshots for each baseline/variant training pair
were checked; the separation coefficient is the only substantive difference.
Training history confirms the BCE term is active (around 0.70 per epoch).

Each checkpoint is evaluated twice:

1. `head_uncertainty`, to directly test the branch that the loss supervises;
2. `normalized_entropy_mahalanobis`, the project's established composite
   detector, to test whether the training change helps the broader detector.

Within each seed and score mode, detector configuration, known-validation-only
95% coverage threshold policy, MC sample count (4), and clustering procedure
are fixed. Test labels are used only for final metrics. Per-seed detector
configs match except for work directory and checkpoint path.

## Critical protocol limitation

The student training pool is configured as `discovery_pool_mode=unknown` and
the loss treats every discovery-pool item as novel. In this repository, that
pool is selected using the underlying dataset labels (`oracle_filtered_novel_only`),
as recorded in the training config. Thus this ablation is an **oracle-pool
upper-bound experiment**, not evidence that the loss can identify unknowns in
a genuinely mixed, unlabeled stream. Do not describe it as unsupervised
unknown-aware training or as a deployed open-world solution. A follow-up must
either use a legitimate unlabeled mixed pool with a reliable pseudo-label or
positive-unlabeled mechanism, or explicitly retain this oracle assumption as a
controlled research protocol.

## Direct uncertainty-score results

| Seed | Metric | Separation off | BCE separation on | Change |
| --- | --- | ---: | ---: | ---: |
| 42 | AUROC | 0.4363 | 0.5685 | +0.1322 |
| 42 | AUPR | 0.3572 | 0.4556 | +0.0984 |
| 42 | FPR95 | 0.9521 | 0.8760 | -0.0760 |
| 42 | OSCR | 0.0807 | 0.1687 | +0.0880 |
| 42 | Test known acceptance | 1.0000 | 0.9752 | -0.0248 |
| 42 | Test unknown rejection | 0.0000 | 0.0456 | +18/395 |
| 43 | AUROC | 0.4344 | 0.5712 | +0.1368 |
| 43 | AUPR | 0.3591 | 0.4629 | +0.1037 |
| 43 | FPR95 | 0.9819 | 0.9143 | -0.0675 |
| 43 | OSCR | 0.0612 | 0.1364 | +0.0751 |
| 43 | Test known acceptance | 1.0000 | 0.9506 | -0.0494 |
| 43 | Test unknown rejection | 0.0000 | 0.0891 | +35/393 |
| 44 | AUROC | 0.4513 | 0.5674 | +0.1161 |
| 44 | AUPR | 0.3750 | 0.4619 | +0.0869 |
| 44 | FPR95 | 0.9465 | 0.9013 | -0.0452 |
| 44 | OSCR | 0.0823 | 0.1703 | +0.0880 |
| 44 | Test known acceptance | 1.0000 | 0.9699 | -0.0301 |
| 44 | Test unknown rejection | 0.0000 | 0.0572 | +23/402 |

All three baseline uncertainty heads were nearly constant at one; known-only
95% calibration therefore produced a degenerate threshold and accepted every
test example. With BCE separation, the head became non-degenerate and ranked
known/unknown examples better on all three seeds. The operating-point
comparison is not symmetric because of the baseline ties: this is strong
evidence the loss repairs a collapsed uncertainty output under this protocol,
but not a fair claim of a net rejection-rate gain over a functioning baseline.

## Main-detector results

| Seed | Metric | Separation off | BCE separation on | Change |
| --- | --- | ---: | ---: | ---: |
| 42 | AUROC / AUPR | 0.5726 / 0.4367 | 0.5666 / 0.4277 | -0.0060 / -0.0089 |
| 42 | FPR95 / OSCR | 0.8727 / 0.1633 | 0.8810 / 0.1708 | +0.0083 / +0.0075 |
| 42 | Known acceptance / unknown rejection | 0.9322 / 0.0734 | 0.9603 / 0.0354 | +0.0281 / -0.0380 |
| 43 | AUROC / AUPR | 0.5442 / 0.4376 | 0.5733 / 0.4634 | +0.0292 / +0.0257 |
| 43 | FPR95 / OSCR | 0.9407 / 0.1125 | 0.9077 / 0.1428 | -0.0329 / +0.0303 |
| 43 | Known acceptance / unknown rejection | 0.9621 / 0.0662 | 0.9572 / 0.0865 | -0.0049 / +0.0204 |
| 44 | AUROC / AUPR | 0.5483 / 0.4304 | 0.5766 / 0.4495 | +0.0284 / +0.0191 |
| 44 | FPR95 / OSCR | 0.9197 / 0.1603 | 0.8796 / 0.1870 | -0.0401 / +0.0267 |
| 44 | Known acceptance / unknown rejection | 0.9415 / 0.0746 | 0.9482 / 0.0398 | +0.0067 / -0.0348 |

Across the three seeds, mean paired changes were AUROC +0.0172, AUPR +0.0120,
FPR95 -0.0216, OSCR +0.0215, test known acceptance +0.0099, and unknown
rejection -0.0175. AUROC, FPR95, and OSCR improved on two seeds and slightly
worsened on seed 42; unknown rejection improved only on seed 43 and declined
on seeds 42 and 44. Candidate-pool purity also fell on average (-0.0436), and
overall clustering ARI was essentially unchanged (-0.0031).

## Decision

- **Keep the BCE separation loss as an experimental option.** It consistently
  prevents the dedicated uncertainty output from collapsing and improves its
  ranking metrics in this three-seed pilot.
- **Do not claim the core low-rejection problem is solved.** The established
  composite detector does not reject more unknowns consistently, and its
  benefits are mixed at the operating point.
- **Do not tune alpha on these test sets.** A next test, if needed, should
  predeclare one alternative weight or a ranking-loss variant and compare it
  on new fixed seeds. However, first address the oracle-filtered discovery
  pool: a practically valid test must not use novel ground-truth labels to
  select training examples for the unknown target.
- Report both the dedicated uncertainty score and the primary composite
  detector, with known acceptance and unknown rejection next to AUROC/FPR95/
  OSCR. A ranking improvement alone is insufficient.

## Follow-up implementation audit

A later code-path audit found that the discovery-batch dispatch condition did
not include `alpha_discovery_uncertainty_separation`. Consequently, enabling
this loss **by itself** previously created a discovery loader but did not
consume its batches, so the logged loss stayed zero. This does not invalidate
the three-seed comparison above: its saved configs also enabled
`alpha_discovery=0.1` and joint discovery, either of which activated batch
processing, and the saved training histories show a non-zero separation term.
The dispatch condition is now fixed, and a regression test verifies that
uncertainty separation alone consumes the pool and produces a non-zero loss.
This implementation fix is not evidence of improved open-set performance; a
new controlled experiment is still required before making that claim.
