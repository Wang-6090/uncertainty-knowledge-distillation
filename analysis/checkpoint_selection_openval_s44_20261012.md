# Open-validation checkpoint-selection audit (seed 44)

## Question and controlled comparison

Does selecting a checkpoint only by known-class validation accuracy miss an
epoch with better known/unknown ranking? This was an audit of the existing
five-epoch CIFAR-100 semantic-isolated 60/40 run, not a training intervention.
Training data, initialization, losses, optimizer, and detector were held fixed.
For every checkpoint, the kNN bank was rebuilt from that checkpoint's own
embedding; validation normalization was fit on the known calibration subset;
open-validation labels were used only to calculate the audit metrics.

The run used seed 44, 64px images, ResNet-18, four MC samples,
`normalized_entropy_min_class_knn`, kNN k=10, `open_val_ratio=0.1`, and a
disjoint known threshold-calibration subset (`calibration_ratio=0.25`). The
data audit established that calibration and checkpoint-selection indices are
disjoint. The official test set was not used to select the checkpoint.

## Open-validation results

| Epoch | Train known-val accuracy | Open-val AUROC | AUPR | FPR95 | Open-val known accuracy | KS | Unknown below known q95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5311 | 0.7677 | 0.9493 | 0.6700 | 0.5767 | 0.4108 | 0.7485 |
| 2 | **0.5741** | 0.7882 | 0.9522 | **0.6000** | **0.5767** | 0.4687 | 0.7860 |
| 3 | 0.5511 | 0.7872 | 0.9524 | 0.5933 | 0.5333 | 0.4460 | 0.7425 |
| 4 | 0.5711 | 0.7926 | 0.9524 | 0.6067 | 0.5733 | 0.4665 | 0.8205 |
| 5 | 0.5413 | **0.7984** | **0.9564** | 0.6133 | 0.5500 | **0.4813** | **0.7385** |

The validation-known-accuracy rule selects epoch 2; open-validation AUROC
selects epoch 5. Relative to epoch 2, epoch 5 improves open-validation AUROC
by 0.0102, AUPR by 0.0042, KS by 0.0127, and reduces the fraction of unknown
scores below the known q95 by 4.75 percentage points. However, FPR95 worsens by
0.0133 and open-validation known accuracy drops by 2.67 percentage points.
Thus the result is a real ranking/operating-point tradeoff, not a uniform
improvement and not a threshold-only effect. It supports evaluating
checkpoint selection against a prespecified open-validation objective, while
retaining known-class performance as a guardrail; one seed is insufficient to
change the default selection rule.

## Audit correction and limitations

The first draft of this audit script reused epoch 1's kNN feature bank for
later checkpoints. That is invalid because each epoch has a different feature
space. The script was corrected to rebuild the bank separately per checkpoint
and rerun. An initial execution also exposed that `attach_knn_distances`
expects the labeled bank dictionary, not only its feature matrix; this caller
was corrected. Only the corrected run above is valid.

The five epochs' official-test discovery reports had already been generated
and inspected during exploration before this selection audit. Although no test
metric selected an epoch, those repeated test inspections mean this run cannot
be presented as a pristine independent test comparison. Do not use the
per-epoch test values to choose a checkpoint or claim generalization. Confirm
the selection rule on a fresh, prespecified seed (and ideally repeat across
several seeds), select using open validation only, then evaluate only the
selected checkpoint once on the official test set.

## Second-seed follow-up (seed 123)

To check whether the selection difference replicates, the same 5-epoch student
configuration and audit procedure were run with seed 123. Known-validation
accuracy and open-validation AUROC both select epoch 4. Its open-val AUROC is
0.7969, FPR95 0.6033, known accuracy 0.5967, KS 0.4807, and unknown-below-known
q95 overlap 0.7855. Epoch 5 is worse on AUROC (0.7872) and FPR95 (0.6233),
while its overlap is slightly lower (0.7705); the diagnostics do not all agree.
The ranking-based selection rule therefore did
not choose a different epoch on this seed. Seed 44's epoch-2 vs epoch-5 split
is not replicated; do not change the default checkpoint selection rule.

The seed-123 run reused the available `semantic_isolated_full_s123_cifar_profile_teacher`
checkpoint, whose teacher config used `calibration_ratio=0`; seed 44 used the
`semantic_isolated_full_s44_disjointcal_teacher` checkpoint with
`calibration_ratio=0.25`. Student training/evaluation settings were matched,
but teacher checkpoint-selection protocols were not identical. Treat this as
a useful second-seed diagnostic, not a fully controlled seed replication.

This audit does **not** change the training algorithm and does not establish
that open-validation selection solves the core representation-overlap
problem. Unknown-below-known-q95 overlap remains about 74–79% on open validation.
The next step should be to freeze one selection objective and teacher protocol,
then assess the core training representation objective with disjoint open
validation and multi-seed evaluation. Do not tune thresholds on test or
repeatedly scan the same open-validation set without accounting for selection.

Machine-readable corrected results:
[`checkpoint_selection_openval_s44_20261012.json`](checkpoint_selection_openval_s44_20261012.json).
Seed-123 diagnostic:
[`checkpoint_selection_openval_s123_20261012.json`](checkpoint_selection_openval_s123_20261012.json).
Reproduction/audit script:
[`../scripts/audit_epoch_selection_openval.py`](../scripts/audit_epoch_selection_openval.py).
