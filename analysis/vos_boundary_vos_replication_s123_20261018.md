# Boundary-focused VOS replication (seed=123, 2026-10-18)

## Question and protocol

Does nearest-class virtual outlier synthesis restricted to the lowest-margin
25% of known training examples improve open-set score separation beyond the
same training recipe without VOS? This is a replication of the seed-44 pilot,
not a threshold sweep.

Both arms use CIFAR-100 semantic-isolated 60/40, seed 123, identical
teacher/student initialization and four epochs, the same 5,400-sample mixed
discovery pool (known prior 0.2), the same feature rejector and checkpoint
selection, and validation-only calibration targeting 95% known coverage.
The sole planned training change is VOS: off in control versus
nearest-class inter-class VOS with `alpha_vos=0.01` and boundary fraction
0.25. `alpha_vos_rejection=0` in both arms. The run configs and training
histories confirm four epochs completed; the treatment's `vos` loss is
nonzero. Test labels are used only for final evaluation, never for training or
calibration.

## Results

| Seed | Condition | AUROC | FPR95 | OSCR | Known accept | Unknown reject | Accepted-known accuracy | KS | Unknown below known-q95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 123 | no-VOS control | 0.90069 | 0.37983 | 0.64926 | 96.20% | 44.18% | 71.40% | 0.64883 | 49.65% |
| 123 | boundary VOS 25% | 0.89193 | 0.40950 | 0.63662 | 96.08% | 43.08% | 70.49% | 0.63350 | 50.85% |

Treatment-minus-control: AUROC `-0.00876`, FPR95 `+0.02967`, OSCR
`-0.01264`, KS `-0.01533`, overlap proxy `+0.01200`, known acceptance
`-0.12 pp`, unknown rejection `-1.10 pp`.

## Interpretation and decision

The seed-44 pilot looked favorable on ranking and fixed-coverage rejection,
with a known-acceptance cost. Seed 123 moves AUROC, FPR95, OSCR, KS, overlap,
and unknown rejection in the wrong direction. Thus the earlier positive result
does not replicate across these two seeds. Do not enable this VOS recipe by
default, do not expand its weight/fraction sweep, and do not combine it with
the rejection BCE that was also unstable across seeds.

This is evidence of seed sensitivity, not proof that all virtual-outlier
methods are ineffective. The next useful step is a mechanism audit: test
whether low known-classification margin actually selects regions relevant to
semantic-near unknown errors, and measure how VOS gradients alter known
classification boundaries and rejector scores before designing another
training intervention. Any follow-up should be predeclared, paired, and judged
jointly by ranking, overlap, fixed-coverage rejection, and known utility.

## Artifacts

- Control: `runs/vos_boundary_replicate_s123_e4_control/` and
  `runs/vos_boundary_replicate_s123_e4_control_discover/`.
- Treatment: `runs/vos_boundary_replicate_s123_e4_interclass_mix/` and
  `runs/vos_boundary_replicate_s123_e4_interclass_mix_discover/`.
- Driver: `scripts/run_interclass_vos_control.py`.
- Command:

```powershell
python scripts/run_interclass_vos_control.py `
  --source-train runs/vos_boundary_rejection_matched_s123_e4_control/train_student_config.json `
  --source-discover runs/vos_boundary_rejection_matched_s123_e4_control_discover/discover_config.json `
  --seed 123 --epochs 4 --tag vos_boundary_replicate_s123_e4 `
  --pairing-strategy nearest_class --boundary-fraction 0.25 `
  --control-alpha-vos 0 --alpha-vos-rejection 0
```
