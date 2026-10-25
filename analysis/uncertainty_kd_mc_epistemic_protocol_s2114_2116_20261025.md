# Protocol: MC-epistemic weighting for logits KD (seeds 2114-2116)

Date fixed: 2026-10-25

## Question

Does non-degenerate MC-dropout epistemic uncertainty improve semantic-hard
novel-class discovery when used to redistribute the main logits-KD loss, over
unweighted KL computed from the same MC-mean teacher logits?

This tests a specific uncertainty-KD mechanism. It does not claim that MC
uncertainty is generally superior, and it does not change the default KD mode.

## Paired conditions

- Fresh CIFAR-100 semantic-hard 60/40 class splits: seeds 2114, 2115, and 2116.
- One split-matched frozen teacher is shared by both arms within each seed.
- Both arms use `kd_uncertainty_source=mc_epistemic` and four stochastic teacher
  passes. Therefore both receive the same MC-mean teacher logits; only the
  uncertainty arm multiplies per-example KL by detached `exp(-u)` weights,
  normalized to mean one per batch.
- Control: standard unweighted logits KL. Treatment: MC-epistemic-weighted
  logits KL. Feature KD and rejection-branch KD are zero in both arms.
- All other student losses, model initialization, optimizer, data, four-epoch
  budget, split-matched teacher, known-validation checkpoint selection,
  feature-rejector scoring, automatic-K procedure, and known-validation-only
  95% coverage calibration are held fixed within each pair.
- Mixed-pool labels are unavailable to training, checkpoint selection,
  candidate selection, calibration, or method choice. Test labels are used
  only for final retrospective metrics.

## Outcomes and decision rule

Report paired per-seed and mean changes in AUROC, AUPR, FPR95, OSCR, test known
acceptance, calibrated unknown rejection, accepted-known accuracy, correctly
accepted-known fraction, candidate purity/recall, automatic-K absolute error,
and automatic clustering ACC/NMI/ARI. Also report per-epoch KD loss, weight
standard deviation, effective sample-size fraction, selected epoch, and
realized known-validation coverage.

Each pair must pass configuration-difference and label-flow audits. The
treatment must have nonzero KD loss and non-degenerate weights in every epoch.
Promotion requires a repeatable improvement in the primary operating-point
outcome (at least +2 percentage points calibrated unknown rejection), with no
more than 1 percentage point loss in known acceptance, no material decline in
accepted-known accuracy or OSCR, and no material degradation in automatic-K or
clustering outcomes. These three fresh seeds are still exploratory; even a
joint pass requires a separately pre-registered confirmation before changing
defaults. A mixed/negative result stops this exact MC-epistemic logits-KD
variant without tuning on these seeds.

## Reproduction

```powershell
python scripts/run_uncertainty_kd_core_pair.py `
  --source-teacher-config runs/uncertainty_kd_triad_s2074_teacher/config.json `
  --source-student-config analysis/uncertainty_kd_triad_s2074_student_config.json `
  --source-discover-config runs/uncertainty_kd_triad_s2074_no_kd_discover/discover_config.json `
  --tag uncertainty_kd_mc_epistemic_s2114_2116 `
  --seeds 2114 2115 2116 `
  --kd-uncertainty-source mc_epistemic --kd-mc-samples 4
```

The runner writes one result JSON and student-config snapshot per seed plus a
summary JSON. Existing incompatible artifacts must not be overwritten.
