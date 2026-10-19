# nnPU rejector support-size control (2026-10-16)

## Purpose

The feature rejector historically capped its known and mixed-pool support at
5,000 samples. This control tested whether the cap itself was contributing to
known/unknown score overlap. The treatment used all available support samples;
the baseline used the same stratified support protocol with a 5,000-sample cap.

## Matched protocol

Both seeds used the same student checkpoint, CIFAR-100 semantic-isolated 60/40
split, mixed discovery pool of 5,400 samples, `nu_corrected` linear nnPU,
`support_augmented` features, MC=4, and a threshold calibrated to 95% known
coverage on validation data. The test labels were used only for final metrics.
The only changed factor was `rejector_max_samples`: `5000` versus `0` (all).

## Results

| seed | support | AUROC | FPR95 | OSCR | known accept | unknown reject | KS | unknown below known q95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 44 | stratified 5000 | 0.88389 | 0.4027 | 0.65048 | 93.28% | 50.15% | 0.6098 | 56.35% |
| 44 | all support | **0.90342** | **0.3590** | 0.66667 | **95.37%** | **49.48%** | **0.6512** | **48.40%** |
| 123 | stratified 5000 | 0.88964 | 0.4070 | 0.63537 | 96.65% | 39.40% | 0.6279 | 51.88% |
| 123 | all support | **0.90315** | **0.3812** | 0.64405 | 95.72% | **50.05%** | **0.6571** | **45.95%** |

The all-support treatment improved AUROC, FPR95, KS, score-overlap proxy and
unknown rejection on both seeds. OSCR improved as well, although the change is
smaller on seed 123 and the accepted-known classification utility still needs
to be monitored. This is evidence that the old cap caused a real support
coverage problem; it is not evidence that the representation overlap is fully
solved.

## Code decision

`train.py` now defaults `--rejector-max-samples` to `0`, meaning that the
rejector uses all available known and mixed-pool support. The positive cap is
still available for historical reproduction and ablations. This default change
does not alter the nnPU formula, threshold calibration, test protocol, or
student training objective. The lower-level `fit_nnpu_feature_rejector` API now
uses the same all-support default (`max_samples=None`), preventing direct API
experiments from silently using a different support-size protocol. The same
default is now used by the hard and soft-PU rejector APIs, and the hard branch
now receives the CLI cap explicitly; previously its CLI value was silently
ignored.

Reproduce the paired control with:

```powershell
python scripts/run_nnpu_support_size_control.py --seed 44
python scripts/run_nnpu_support_size_control.py --seed 123 `
  --source-config runs/rejection_boundary_epoch_control_s123_e4_baseline_discover/discover_config.json `
  --student-ckpt runs/rejection_boundary_epoch_control_s123_e4_baseline/student.pt `
  --tag nnpu_support_size_control_s123
```
