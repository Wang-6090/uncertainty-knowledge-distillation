# CIFAR-100 split-protocol sensitivity pilot: blocked by split-conditioned teacher (2026-10-18)

## Question and planned comparison

The goal was to determine whether the poor known/unknown separation is strongly
dependent on the CIFAR-100 class split. The proposed matched comparison was
semantic-isolated versus semantic-hard, both 60/40, seed 44, four epochs,
the same detached-VOS student recipe, matched mixed discovery pool, detector,
and validation-only 95%-known-coverage calibration. Only the class split was
intended to change. Test labels were not intended for training, model
selection, or calibration.

The split designs differ materially:

- `semantic_isolated`: 12 complete coarse superclasses are known and 8
  complete coarse superclasses are novel; no superclass is shared.
- `semantic_hard`: each of the 20 coarse superclasses contributes 3 known and
  2 novel fine classes; all superclasses are shared.

This is a useful protocol stress test, not a comparison of algorithms. It
would require protocol-compatible initialization checkpoints for both arms.

## What happened

The isolated arm completed training and discovery in the new isolated output
directory. Its test metrics are not reported here as a cross-protocol result.
When the hard arm started, `load_checkpoint` rejected the supplied teacher:
the teacher checkpoint
`runs/semantic_isolated_full_s123_disjointcal_teacher/teacher.pt` stores the
semantic-isolated `known_classes`/`novel_classes`, which do not match the
semantic-hard split. This checkpoint contains a split-dependent 60-way class
head and prototypes, so bypassing the validation or reusing it would be
methodologically invalid. No semantic-hard training epoch completed and no
valid paired protocol comparison exists.

The first runner version performed one isolated arm before encountering this
dependency. The runner was then hardened to preflight the teacher metadata for
**all** requested splits before beginning either arm. Re-running the preflight
now fails before training with the explicit split mismatch. This safety check
prevents another GPU run from being wasted and prevents partial outputs from
being mistaken for a completed comparison.

There is an existing `runs/semantic_hard_teacher_s44/teacher.pt`, so a
protocol-compatible hard teacher does exist. However, its saved recipe is not
matched to the isolated teacher used for the previous student runs: among
other differences, teacher batch size is 32 vs 64, validation/test limits are
300/1,000 vs full, and calibration ratio is 0.5 vs 0.25. Reusing it would
replace the split-metadata confound with a teacher-recipe confound. It was
therefore not substituted into the experiment.

## Decision and next step

Do not infer that either protocol is easier from this incomplete pilot. Keep
the isolated run as an isolated artifact; do not delete or overwrite it, but
do not use it as evidence for protocol sensitivity.

To perform a valid protocol comparison, train new isolated and hard teachers
from the same teacher configuration template (seed, epoch budget, batch size,
validation/calibration protocol, and selection rule), then run both student
arms from scratch using the same student template. Because teacher weights
necessarily depend on the protocol, the result estimates the practical
end-to-end protocol effect (split plus matching teacher training), not an
isolated split-only causal effect. If isolating the split effect is required,
use a split-agnostic initialization without class-index-dependent head or
prototype state and specify that design in advance.

Do not adjust loss functions using these test results. For now, prioritize
verifying the evaluation protocol on both splits and comparing validation
signals/score overlap under a declared task definition. Any performance claim
requires complete runs and preferably more than one seed.

## Artifacts and checks

- Guarded driver: `scripts/run_split_protocol_sensitivity.py`.
- Intended outputs: `runs/split_protocol_sensitivity_s44_e4_20261018_*`.
- The isolated arm completed; the hard arm was rejected before training after
  the guard was added. No paired summary was written.
- No GitHub upload was performed.
