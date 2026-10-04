# MC-epistemic knowledge distillation pilot (2026-09-30)

## Purpose

The main failure mode remains severe overlap between known and unknown feature/score distributions. The existing uncertainty-aware KD used the teacher's learned uncertainty head as a KD weight. That head is trained on known samples to approximate classification confidence/error; it is not automatically epistemic uncertainty in the open-world sense.

This pilot added an optional MC-Dropout KD source, following the predictive-uncertainty decomposition commonly associated with Gal and Ghahramani. For each training batch, the teacher performs multiple stochastic-dropout passes. The student receives the mean teacher logits, and the KD weight is derived from normalized mutual information (`mc_epistemic`).

Implementation options:

```text
--kd-uncertainty-source head                 # historical default
--kd-uncertainty-source mc_epistemic        # this pilot
--kd-uncertainty-source mc_predictive_entropy
--kd-mc-samples 4
```

The default remains `head`, so historical runs are unchanged. The MC signal is normalized by `log(num_known_classes)` before being used as a weight.

## Controlled comparison

Fixed conditions: CIFAR-100 semantic-hard 60/40, seed 42, full known training set, ResNet-34 teacher / pretrained ResNet-18 student, batch size 32, 5 epochs, same teacher checkpoint, same mixed discovery pool, same validation/test limits, GPU, and the same Mahalanobis + entropy detector with a 95% known-coverage threshold.

Only the KD uncertainty source changed.

| Method | AUROC | FPR95 | OSCR | Unknown rejection | Accepted-known accuracy | Test known acceptance |
|---|---:|---:|---:|---:|---:|---:|
| learned-head uncertainty KD | 0.5669 | 0.9009 | 0.3093 | 6.27% | 46.13% | not exactly 95% under validation transfer |
| MC-epistemic KD, 4 passes | 0.5109 | 0.9299 | 0.2426 | 3.13% | 37.64% | 94.02% |

The MC variant also selected a weaker validation checkpoint (`best_val_known_acc=0.3333`) than the head baseline (`0.4667`).

## Judgment

This pilot is a negative result. MC-Dropout epistemic weighting did not reduce the operating-point overlap and harmed ranking, OSCR, and accepted-known accuracy. It should remain an optional ablation, not the default method. We should not tune the MC sample count or claim that epistemic KD is effective based on this single run.

The negative result is still useful: the current bottleneck is not solved merely by replacing the uncertainty-head weight with a theoretically motivated MC uncertainty. The next direction should explicitly improve or model the known support in representation space, with a fixed known-class accuracy constraint and exact matched-coverage reporting.

