# PU rejection-boundary stop schedule: five-epoch control

## Question

The constant PU rejection-boundary loss showed a positive four-epoch pilot but
did not reproduce an AUROC gain when both arms were extended to five epochs.
This control tested whether keeping the loss for epochs 1--4 and disabling it
after epoch 4 would preserve the short-budget benefit.

The treatment used `alpha_discovery_rejection_boundary=0.05` and
`discovery_rejection_boundary_stop_epoch=4`; the baseline used boundary weight
0. Both arms used runtime seed 44, the same teacher/configuration,
semantic-isolated CIFAR-100 60/40 protocol, matched mixed discovery pool,
validation protocol, and discovery evaluation. Test labels were used only for
retrospective metrics.

## Results

| arm | AUROC | FPR95 | OSCR | known accuracy | unknown rejection |
|---|---:|---:|---:|---:|---:|
| baseline | 0.9051 | 0.3395 | 0.6649 | 0.7152 | 0.5408 |
| boundary 0.05, stop after epoch 4 | 0.9024 | 0.3565 | 0.6604 | 0.7105 | 0.5495 |

Distribution diagnostics:

- mean unknown-minus-known score gap: `9.4131 -> 9.4811`;
- KS distance: `0.6608 -> 0.6547`;
- unknown-below-known-q95 overlap: `0.4995 -> 0.5078`;
- known-above-unknown-q05 overlap/FPR95: `0.3395 -> 0.3565`.

The schedule slightly increased the operating-point unknown rejection rate,
but reduced AUROC, OSCR, known accuracy, KS, and the overlap diagnostics. It
does not solve the representation-overlap problem and is not a candidate for
the default configuration.

## Implementation audit

The schedule is implemented in `train.py` through `scheduled_weight` and is
recorded in `train_history.json` as
`discovery_rejection_boundary_weight`. The treatment history confirmed weights
`1, 1, 1, 1, 0` for epochs 1--5 and a zero boundary loss in epoch 5. The
feature/loss branch was active, so this is a valid negative result rather than
a wiring failure.

## Consequence

The next controlled direction should be validation-based checkpoint selection,
not more stop-epoch tuning. The current protocol can establish open validation
but historically selects by known accuracy. We should compare `known_acc`
against `open_auroc` or `open_ks` using only the reserved open-validation set,
while freezing the threshold and test evaluation protocol.
