# Discovery-loader wiring audit (2026-10-12)

## Why this audit was needed

The recent PU-energy paired test showed that a loss can look ineffective if its
command-line switch is not connected to discovery batches. The student-training
preflight therefore needs to agree with every discovery-dependent branch in
`train_one_epoch_student`, rather than maintaining a partial duplicate list.

## Findings and code change

Compared the objective parameters accepted by the student epoch function with
the preflight in `train.py::discovery_training_requirements`. The preflight had
omitted the following objectives, so `--discovery-pool` was not consistently
required even though the epoch function consumes discovery batches for them:

- `--alpha-discovery`;
- feature and projection consistency;
- rejection-branch feature margin and feature separation;
- `--alpha-reciprocal`;
- PU-corrected uncertainty feature margin's mixed-pool requirement.

Centralized the full discovery-objective switch list in
`discovery_training_requirements`; added reciprocal to the requirement; and
added the `pu_corrected` feature-margin mode to mixed-pool validation. Added
parameterized regression tests covering every currently exposed discovery
objective and each PU-specific mode. This is a correctness/wiring fix, not a
new representation-learning method.

## Verification

- Command-line requirement tests: `83 passed` in `tests/test_core_behaviors.py::CommandLineTest`.
- Full test suite before the last reciprocal test addition: `222 passed, 2 warnings`.
- `python -m compileall -q train.py novel_discovery tests`: passed.
- `git diff --check`: passed.
- Toy functional smoke test: one epoch with 32 known train samples, 16 validation
  samples, 24 discovery samples and `--alpha-discovery-feature-consistency
  0.01`. The first invocation correctly failed because it omitted
  `--discovery-pool`; rerunning with that flag consumed discovery batches,
  reported non-zero `discovery_feature_consistency=0.5708`, and saved
  `runs/discovery_wiring_smoke_20261012/student.pt`.
- A second toy smoke test enabled `--reciprocal-points 2 --alpha-reciprocal
  0.01` with a pure-unknown discovery pool. It reported non-zero
  `reciprocal=0.9266` and saved
  `runs/discovery_reciprocal_wiring_smoke_20261012/student.pt`.

This smoke test only establishes that the option reaches an active loss and
checkpoint saving works. It says nothing about AUROC or real-data efficacy.

## Impact on the core question

This audit does not fix known/unknown feature or score overlap and provides no
evidence of higher AUROC, lower FPR95, or better OSCR. It reduces the risk that
future controlled ablations silently fail to run the requested objective. The
existing evidence for PU-energy remains mixed across seeds, so it stays
optional rather than entering the default recipe.

## Next step

Finish the full suite after the reciprocal coverage addition. For any new
algorithm experiment, first verify (a) the corresponding loss meter is
non-zero, (b) config captures the intended active factor, and (c) data split,
sampling, calibration, and detector protocol match the baseline. Evaluate
AUROC, FPR95, OSCR, overlap/KS, known accuracy/acceptance, and unknown rejection
together; do not treat a threshold-driven rejection-rate increase as a fix.
