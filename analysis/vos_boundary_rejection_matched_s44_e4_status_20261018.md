# Boundary VOS plus rejection-head BCE: matched experiment status (2026-10-18)

## Why this comparison is needed

The earlier exploratory command used a no-VOS control and a treatment with
nearest-class, 25%-boundary VOS plus `alpha_vos_rejection=0.05`. That changes
more than one training factor, so its metrics cannot isolate the effect of
rejection-head BCE. Those results are retained as exploratory observations,
not evidence for or against adding BCE to boundary VOS.

## Corrected one-factor design

Both arms use the same seed (44), CIFAR-100 semantic-isolated 60/40 protocol,
source teacher/student configuration, four epochs, nearest-class inter-class
VOS with `alpha_vos=0.01`, `vos_boundary_fraction=0.25`, checkpoint rule,
detector, and validation-only 95% known-coverage calibration. The sole planned
difference is `alpha_vos_rejection`: 0 for control versus 0.05 for treatment.

Command:

```powershell
python scripts/run_interclass_vos_control.py --tag vos_boundary_rejection_matched_s44_e4 --pairing-strategy nearest_class --boundary-fraction 0.25 --alpha-vos-rejection 0.05 --control-alpha-vos 0.01 --control-pairing-strategy nearest_class --control-boundary-fraction 0.25 --control-alpha-vos-rejection 0.0
```

The runner was updated with explicit control-arm VOS settings. The control arm
now uses `interclass_mix` when its VOS weight is nonzero, rather than silently
falling back to Gaussian VOS.

## Status

The seed=44 paired run completed after the status note was first written. It was
then replicated with the same protocol at seed=123. The results do not show a
stable benefit: seed 44 improved AUROC/FPR95/OSCR and overlap, while seed 123
regressed on those measures; validation-calibrated unknown rejection fell in
both seeds. Do not enable the rejection BCE by default or continue its weight
sweep. Full paired results and the next diagnostic direction are recorded in
[`vos_boundary_rejection_matched_s44_s123_e4_20261018.md`](vos_boundary_rejection_matched_s44_s123_e4_20261018.md).

The earlier no-VOS versus boundary-VOS-plus-BCE comparison remains confounded
and is not evidence for the BCE; this corrected pair is the relevant result.

## Code checks

- `python -m pytest -q tests/test_core_behaviors.py`: 222 passed, 2 warnings (run before the seed=44/123 experiments; no source changes were made for those runs).
- `git diff --check`: passed.
- No GitHub upload was performed.
