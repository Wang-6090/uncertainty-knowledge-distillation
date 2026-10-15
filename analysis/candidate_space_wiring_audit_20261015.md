# Candidate-space control wiring audit (2026-10-15)

## Why this audit was required

The previous paired pilot was intended to compare the mixed-pool candidate gate in
the shared classifier space with the independent rejection-support space. Before
using its numbers, the training history and loss wiring were checked directly.

## Finding

The old script set `alpha_discovery_rejection_boundary=0.1`. That is the PU
rejection boundary loss. In the current implementation it consumes the whole
mixed pool and does not consume `feature_candidate_masks`.

The candidate gate is wired to the kNN boundary paths instead:

- `alpha_discovery_knn_boundary` for the shared feature support;
- `alpha_discovery_rejection_knn_boundary` for the rejection support.

The old runs recorded `feature_candidate_ratio=0.0` in every epoch for both
candidate-space arms, while `discovery_rejection_boundary` was non-zero. This
confirms that the candidate-space switch was not used by the loss under test.

Therefore the old directories
`runs/s44_e4_hard_nnpu_boundary_candidate_space_retry_shared*` and
`runs/s44_e4_hard_nnpu_boundary_candidate_space_retry_rejection_support*`
are retained as an implementation-audit record, but their metric differences
must not be reported as evidence for or against `shared` versus
`rejection_support`. The different checkpoint hashes do not repair this
protocol error; the relevant treatment was not active.

## Code correction

`scripts/run_rejection_boundary_candidate_space_control.py` now sets only
`alpha_discovery_rejection_knn_boundary=0.1` and disables the separate PU
boundary term. This is the rejection-boundary loss that actually applies the
candidate masks. A new output tag must be used so the invalid runs cannot be
silently reused.

## Corrected experiment protocol

Only `discovery_feature_candidate_space` changes:

| fixed condition | value |
|---|---|
| dataset / split | CIFAR-100 semantic-isolated 60/40 |
| seed / epochs | 44 / 4 |
| discovery pool | matched mixed pool, known prior 0.2 |
| boundary | rejection-support kNN boundary, weight 0.1 |
| gate | enabled, cross-view intersection enabled |
| teacher, optimizer, transforms | identical |
| detector and calibration | identical, open validation, 95% known coverage |

The corrected comparison is valid only if both histories show a non-zero
`feature_candidate_ratio` and non-zero `discovery_rejection_knn_boundary`.
It will be judged jointly by AUROC, FPR95, OSCR, known accuracy/acceptance,
fixed-coverage unknown rejection, KS, and score-overlap diagnostics.

## Verification

- `python -m py_compile scripts/run_rejection_boundary_candidate_space_control.py train.py novel_discovery/pipeline.py`: passed
- `python -m pytest -q`: `236 passed`
- `git diff --check`: passed (only line-ending warnings)

