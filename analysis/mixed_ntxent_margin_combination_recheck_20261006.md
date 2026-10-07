# Mixed-pool NT-Xent and uncertainty feature-margin combination recheck

## Question

The original uncertainty feature-margin had a positive but mixed multi-seed
signal. An older branch reported a promising unknown-only discovery NT-Xent
experiment, but that result used a different split, teacher, epoch budget and
small test subset. This recheck asks whether the two ideas can be combined
under the current full-data mixed-pool protocol, and whether NT-Xent has value
in that protocol by itself.

## Valid protocol

All new runs use CIFAR-100 random 60/40 with the existing fixed class split,
full data, pretrained ResNet-34 teacher and ResNet-18 student, batch size 64,
five epochs, mixed discovery pool with measured known prior `0.212598`, and
immediate mixed-pool nnPU weight `0.1`. Detection is identical across runs:
normalized entropy + min-class kNN (`k=10`), MC=8, full 10,000-image open test,
and a threshold fitted from known validation data only for 95% known coverage.
Clustering was skipped. Test labels are used only for final metrics.

The seed-42 treatment is the existing single-view uncertainty feature-margin
(`alpha=0.05`, margin `0.2`). The new arms are:

- NT-Xent only: `alpha_discovery=0.05`, temperature `0.2`, no feature-margin.
- Combined: NT-Xent plus the original uncertainty feature-margin, with the
  same two coefficients and temperature.

## Seed 42 results

| Arm | AUROC | AUPR | FPR95 | OSCR | Known accuracy (all known) | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original uncertainty feature-margin | 0.7108 | 0.5799 | 0.7348 | 0.4132 | 0.4848 | 95.30% | 13.90% |
| NT-Xent only | 0.6988 | 0.5764 | 0.7513 | 0.4139 | 0.4903 | 94.98% | 14.60% |
| Combined | 0.7025 | 0.5776 | 0.7500 | 0.4117 | 0.4913 | 96.03% | 12.83% |

NT-Xent alone increases selected operating-point unknown rejection, but lowers
AUROC, worsens FPR95, and lowers known acceptance relative to the original
feature-margin. The combined arm also fails to reproduce the seed-2026 result:
on seed 2026 it reached AUROC `0.7075`, FPR95 `0.7442`, OSCR `0.4308`, and
unknown rejection `14.75%`, whereas on seed 42 it reached `0.7025`, `0.7500`,
`0.4117`, and `12.83%`. Therefore the apparent seed-2026 gain is not stable
enough to promote.

## Interpretation and decision

The direct combination does not solve the known/unknown feature overlap. The
mixed-pool NT-Xent objective treats known and novel discovery samples alike;
it provides view consistency but no reliable direction away from the known
manifold. The older unknown-only result cannot justify enabling it here because
its protocol was not matched.

Decision:

- Keep NT-Xent-only and NT-Xent+feature-margin as documented ablations, not
  defaults.
- Stop adding these two losses together or sweeping their coefficients on the
  same class split.
- Prefer the original uncertainty feature-margin only as an unconfirmed
  candidate, and evaluate it on a separately generated class split with a
  predeclared protocol.
- Any next algorithmic change should decouple the representation objective
  from the unknown rejector or use a validated mixed-pool positive-unlabeled
  formulation, rather than add another uniform consistency loss.

Artifacts:

- `runs/uncertainty_margin_combo_ntxent_s2026/`
- `runs/uncertainty_margin_combo_ntxent_s2026_detect/`
- `runs/uncertainty_margin_combo_ntxent_s42/`
- `runs/uncertainty_margin_combo_ntxent_s42_detect/`
- `runs/ntxent_only_mixed_s42/`
- `runs/ntxent_only_mixed_s42_detect/`

