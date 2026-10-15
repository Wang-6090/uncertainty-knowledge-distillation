# Open-validation checkpoint audit (2026-10-13)

## Purpose

The earlier checkpoint-selection audit used positional indices to compare
calibration and open-validation datasets. CIFAR-100 creates separate dataset
objects for train and test views, so positional indices alone can collide even
when the underlying samples are different. This audit first fixed that audit
tooling issue, then replayed the saved epoch checkpoints with the detector
used by the current discovery protocol.

## Protocol and integrity checks

- CIFAR-100 semantic-isolated 60/40 split, seed 44;
- saved student checkpoints from the same five-epoch run;
- `normalized_entropy_min_class_knn`, MC=4, and a checkpoint-specific kNN bank;
- calibration ratio 0.25 and open-validation ratio 0.1;
- no final test samples used for selection;
- source identity includes dataset source, train/test view, and underlying
  sample index rather than only a local positional index;
- calibration/open-validation source-key intersection: **0**.

The audit now uses `dataset_source_keys()` in `novel_discovery/data.py`.
This is an audit-only correction; it does not change model training or test
evaluation.

## Open-validation replay

| epoch | open AUROC | FPR95 | known accuracy | KS | unknown below known q95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.7677 | 0.6700 | 0.5767 | 0.4108 | 0.7485 |
| 2 | 0.7882 | 0.6000 | 0.5767 | 0.4687 | 0.7860 |
| 3 | 0.7872 | 0.5933 | 0.5333 | 0.4460 | 0.7425 |
| 4 | 0.7926 | 0.6067 | 0.5733 | 0.4665 | 0.8205 |
| 5 | 0.7984 | 0.6133 | 0.5500 | 0.4813 | 0.7385 |

The current detector's open-validation AUROC selects epoch 5; the historical
known-accuracy rule selects epoch 2. This is evidence that checkpoint choice
can change the reported result, but it is not evidence that checkpoint choice
solves the representation-overlap problem. The open validation contains
known/unknown labels, so it is appropriate for a supervised diagnostic or a
clearly declared validation protocol, not for claiming a fully unsupervised
selection rule.

For retrospective diagnosis only, the already-generated independent test
reports for the same five checkpoints were also compared. These labels were
not used for training, checkpoint selection, or threshold selection:

| test checkpoint | AUROC | FPR95 | OSCR | known accuracy | unknown rejection |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.7842 | 0.6330 | 0.4751 | 0.5355 | 0.2163 |
| 2 | 0.7955 | 0.6178 | 0.5210 | 0.5873 | 0.2443 |
| 3 | 0.8003 | 0.5907 | 0.5036 | 0.5627 | 0.2410 |
| 4 | 0.8071 | 0.6095 | 0.5153 | 0.5773 | 0.2218 |
| 5 | 0.8044 | 0.6020 | 0.4905 | 0.5498 | 0.2278 |

The open-validation AUROC rule selected epoch 5, while the strongest test
AUROC was epoch 4 and the strongest test OSCR/known-utility combination was
not epoch 5. This single-run mismatch shows why open-validation selection
cannot yet be promoted as a solution to the core overlap problem.

## Decision

Do not change the default `known_acc` rule or promote `open_auroc` from this
single-seed replay. The current priority remains training-time separation of
known and unknown representations. Future checkpoint experiments must:

1. use the exact final detector and calibration protocol;
2. use source-key audits before comparing splits;
3. pre-register the selection criterion;
4. confirm the result on a fresh seed and an untouched test set;
5. report AUROC, FPR95, OSCR, KS, overlap, known accuracy, and unknown
   rejection together.

Artifact: `analysis/checkpoint_selection_openval_reaudit_s44_20261013.json`.
