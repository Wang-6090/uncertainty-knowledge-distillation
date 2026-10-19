# Frozen-checkpoint score audit on semantic-hard (seed 123)

## Question and controlled factors

After the matched split-protocol replication showed substantial semantic-hard
overlap, this audit asks whether an already implemented detector score is a
better readout on that hard split. The trained student checkpoint, 60/40
semantic-hard split, CIFAR-100 test set, 5,400-example mixed discovery pool,
seed, clustering-disabled setting, and validation-only approximately 95%
known-coverage calibration target were held fixed. The score/detector family
was varied among the current feature rejector, normalized entropy plus
prototype, normalized entropy plus diagonal Mahalanobis, and known support.
No student was retrained, and test labels were not used for calibration or
choice of score. This is a diagnostic score-family audit, not an algorithm
training comparison. The feature rejector is fitted from the same configured
discovery pool; the other score modes use their corresponding saved known
statistics.

Command:

```powershell
python scripts/run_score_mode_audit.py `
  --source-discover runs/split_protocol_matched_teacher_s123_e4_20261018_semantic_hard_discover/discover_config.json `
  --student-ckpt runs/split_protocol_matched_teacher_s123_e4_20261018_semantic_hard_student/student.pt `
  --output-prefix split_hard_score_audit_s123_20261018 `
  --score-modes feature_rejector normalized_entropy_proto normalized_entropy_mahalanobis known_support
```

## Results

| Score family | AUROC | FPR95 | OSCR | Test known accept | Test unknown reject | Accepted-known accuracy |
|---|---:|---:|---:|---:|---:|---:|
| feature rejector | **0.78111** | **0.64683** | **0.59025** | 95.00% | **24.53%** | 72.68% |
| normalized entropy + prototype | 0.69923 | 0.73950 | 0.55937 | 95.53% | 9.80% | 73.20% |
| normalized entropy + diagonal Mahalanobis | 0.67227 | 0.82433 | 0.52084 | 94.72% | 12.25% | 71.88% |
| known support | 0.69601 | 0.75100 | 0.54051 | 94.65% | 12.90% | 72.76% |

The feature rejector is better on ranking metrics and unknown rejection under
the matched operating point. None of the alternative implemented scores
reduces overlap enough to be a promising replacement. Small differences in
known acceptance are calibration/test sampling deviations around the target,
not evidence of better ranking. This agrees with older same-checkpoint audits
on other protocols: support, prototype, and Mahalanobis scores have not
consistently beaten the mixed-pool rejector.

## Decision and next work

- Keep the existing feature rejector as the score baseline; do not fuse or
  promote these weaker scores and do not tune a threshold from test outcomes.
- This audit does not solve the semantic-hard problem. The best score still
  rejects only 24.53% of unknown test images at approximately 95% known
  coverage, while AUROC is 0.78111 and FPR95 is 0.64683.
- Stop spending iterations on score stacking/calibration. Next training
  experiments should target the representation and use only known training
  labels plus unlabeled mixed-pool signals. First inspect class-conditional
  within-known geometry and training/validation neighbor structure, then test
  one representation intervention at a time (e.g. a conservative local
  boundary objective using unlabeled pool features with explicit known
  contamination control). Require matched semantic-hard control and report
  AUROC, FPR95, OSCR, score overlap, unknown rejection at fixed known
  coverage, and known classification utility. Any test-set class breakdown is
  descriptive only and must not drive training or method selection.

Artifacts are in `runs/split_hard_score_audit_s123_20261018_<score>/`.

## Follow-up: boundary-weighted rejector candidate modes (exploratory)

The audit script now permits explicit overrides for
`rejector_candidate_mode` and `rejector_pool_selection`. A first run changed
only the candidate mode while leaving `rejector_pool_selection=all`; code
inspection confirmed that the selector is not consulted in this mode. Its
metrics exactly matched the baseline, so this was an **inert override**, not a
valid negative result. It is retained as a wiring check only.

A valid paired check then fixed the student checkpoint, mixed pool, known
prior, candidate ratio (0.25), candidate weight (2.0), rejector representation,
threshold protocol, and all other detector settings. Both arms used
`weighted_boundary`, which retains the full unlabeled pool and reweights its
candidate subset; only the candidate mode (`prototype_boundary` vs
`confident_boundary`) changed. This was run for seeds 44 and 123 using the
existing semantic-hard student checkpoints; students were not retrained.

| Seed | Selector | AUROC | FPR95 | OSCR | Validation-calibrated test known accept | Unknown reject | Accepted-known accuracy |
|---:|---|---:|---:|---:|---:|---:|---:|
| 44 | prototype boundary | **0.79592** | **0.59100** | **0.59881** | 93.37% | **30.58%** | 72.47% |
| 44 | confident boundary | 0.77364 | 0.67717 | 0.57129 | 94.07% | 27.65% | 71.55% |
| 123 | prototype boundary | **0.79401** | **0.61217** | **0.60257** | 95.28% | 24.20% | 72.78% |
| 123 | confident boundary | 0.76185 | 0.71383 | 0.56718 | 95.07% | 21.28% | 72.34% |

Prototype-boundary weighting beats confident-boundary weighting on ranking
metrics in both seeds. Against the original all-pool rejector, AUROC/FPR95/
OSCR also move modestly in the favorable direction in both seeds; however,
validation-calibrated unknown rejection is not consistent (seed 44: +2.35 pp;
seed 123: -0.33 pp). This is not a robust solution to the core rejection
problem. It is at most a promising detector-training recipe requiring an
independent confirmation protocol before consideration for defaults.

Paired bootstrap files were generated for point-estimate uncertainty. Their
AUROC difference intervals exclude zero in both seeds, but this is uncertainty
over test examples for these fixed checkpoints, not variation over training
seeds. The existing bootstrap script recomputes an exact-95%-known test
threshold inside each resample; consequently its unknown-rejection intervals
are **not validation-only operating-point estimates** and must not be used to
claim deployment-calibrated rejection improvement.

Important exploratory-selection caveat: the initial candidate-pool audit
reported pool-label contamination/recall after candidate selection. Although
those labels never entered model code or the selector, that audit influenced
which rule was prioritized for this test-set follow-up. This creates a
human-in-the-loop exploratory selection path. Therefore the above test-set
results are not an untouched confirmatory comparison and must not justify
choosing a final selector. Any confirmation must be pre-registered on a fresh
seed/split using only unlabeled candidate geometry to specify the comparison;
test results remain final reporting only.

Candidate-pool audit: `analysis/semantic_hard_candidate_pool_audit_s123_20261018.json`.
Paired-bootstrap diagnostics:
`analysis/semantic_hard_weighted_boundary_bootstrap_s44_20261019.json` and
`analysis/semantic_hard_weighted_boundary_bootstrap_s123_20261019.json`.
