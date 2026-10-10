# PU energy-boundary paired experiment (2026-10-10)

## Hypothesis

The ordinary discovery energy-margin loss treats every sample in a mixed
discovery pool as unknown. This is incompatible with the matched mixed-pool
protocol because 20% of that pool is known. This experiment added a
PU-corrected energy boundary: labelled known samples provide a detached energy
reference, and the mixed-pool violation is corrected by the known prior
`pi=0.2`.

The treatment was enabled with `alpha_discovery_energy_pu=0.1`,
`risk=hard_nnpu`, `margin=1.0`, and `temperature=1.0`.

## Protocol

- CIFAR-100 semantic-isolated 60/40 split, seed 42
- pretrained ResNet-34 teacher and ResNet-18 student
- full data, 3 student epochs
- matched mixed discovery pool: 5,400 samples, known prior 0.2
- same teacher, optimizer, detector, support-augmented nnPU rejector, and
  95% known-coverage calibration
- only training variable: PU-energy weight `0 -> 0.1`

The new loss was non-zero in training (`0.5184` in treatment epoch 1), so the
parameter was connected to the training graph. An initial toy smoke exposed a
missing training-loop condition; that wiring bug was fixed before this formal
run.

## Results

| metric | baseline | PU-energy | treatment - baseline |
|---|---:|---:|---:|
| AUROC | 0.7703 | 0.7637 | -0.0066 |
| FPR95 | 0.6423 | 0.6483 | +0.0060 |
| OSCR | 0.3544 | 0.3508 | -0.0036 |
| known acceptance | 94.35% | 95.23% | +0.88 pp |
| unknown rejection | 20.25% | 20.43% | +0.18 pp |

The score distribution audit also does not show a useful separation. The
baseline score quantiles (q10/q50/q90/q95) were `known=
[-6.991,-0.363,3.685,4.795]` and `unknown=[-0.581,2.677,5.773,6.603]`.
The treatment values were `known=[-5.644,-0.093,3.740,4.565]` and
`unknown=[-0.512,2.768,5.647,6.441]`. The known distribution moved upward
slightly, while the unknown distribution did not move away; this is consistent
with the worse AUROC/FPR95 rather than evidence of a genuine feature gap.

## Decision

This single-seed paired experiment does not support the PU-energy loss as a
mainline improvement. The tiny gain at one operating point is outweighed by
worse AUROC, FPR95, and OSCR. The implementation is retained as an opt-in
ablation, but its weight should not be tuned further before checking the energy
definition and the known-support geometry.

The result also reinforces the central diagnosis: the problem is not solved by
adding another scalar boundary loss when the learned known and unknown
representations remain substantially overlapping.
