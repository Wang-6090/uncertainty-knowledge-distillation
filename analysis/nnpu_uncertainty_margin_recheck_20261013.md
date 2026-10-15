# nnPU + uncertainty-weighted feature margin recheck (2026-10-13)

## Question

The previous uncertainty-weighted feature-margin idea was rechecked under a
strict matched protocol. The purpose was to determine whether this loss really
reduces known/unknown score overlap, rather than merely moving a threshold or
rejecting more known samples.

The baseline used the audited mixed-pool nnPU uncertainty objective with
`alpha_discovery_uncertainty_pu=0.1` and uncertainty feature-margin weight
`0`. The treatment used the same configuration plus uncertainty-weighted
feature margin weight `0.05` and cosine margin `0.2`.

## Protocol and audit

- CIFAR-100 semantic-isolated 60/40 split;
- seeds `44` and `123`, each with a newly trained baseline/treatment pair;
- same teacher, pretrained student initialization, optimizer, five epochs,
  batch size, and matched mixed discovery pool;
- corrected `nu_corrected` mixed-pool risk and known prior `0.2`;
- same MC count `4`, kNN feature bank, detector score, and disjoint
  validation-only calibration at 95% known coverage;
- official test labels were used only for post-hoc reporting and paired
  bootstrap diagnostics.

The treatment loss was confirmed to be active. For example, in seed 44 the
final training history reports non-zero
`discovery_uncertainty_feature_margin=0.0311` and mean uncertainty weight
`0.7403`; the matched baseline reports zero for both. Thus the negative result
below is not explained by an inactive command-line option.

## Fixed detector reports

| seed | arm | AUROC | FPR95 | OSCR | known acc | known accept | unknown reject | KS | unknown below known-q95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 44 | baseline | 0.8120 | 0.5965 | 0.5214 | 57.90% | 94.12% | 29.85% | 0.4843 | 73.93% |
| 44 | margin 0.05 | 0.8083 | 0.5943 | 0.5235 | 58.40% | 94.28% | 27.53% | 0.4869 | 74.68% |
| 123 | baseline | 0.8183 | 0.5850 | 0.5626 | 62.33% | 95.50% | 25.60% | 0.4962 | 72.18% |
| 123 | margin 0.05 | 0.8179 | 0.5850 | 0.5620 | 61.75% | 95.70% | 26.20% | 0.4971 | 70.45% |

The validation-calibrated work point gives the same warning: seed 44 rejects
2.32 percentage points fewer unknowns with the treatment, while seed 123
rejects 0.60 points more. This is not a stable improvement. The treatment
changes the score distribution only slightly and does not consistently reduce
the large overlap region.

## Paired bootstrap diagnostic

The bootstrap uses the same ordered test examples in each pair and is strictly
retrospective; it is not used for model selection.

| seed | metric | treatment - baseline | 95% percentile CI | treatment better |
|---:|---|---:|---:|---:|
| 44 | AUROC | -0.00373 | [-0.00620, -0.00124] | 0.1% |
| 44 | unknown rejection at exact 95% known coverage | -0.00918 | [-0.02725, 0.00900] | 16.0% |
| 123 | AUROC | -0.00045 | [-0.00270, 0.00177] | 34.8% |
| 123 | unknown rejection at exact 95% known coverage | +0.01474 | [-0.00351, 0.03276] | 94.1% |

Artifacts:

- `runs/nnpu_margin_pair_s44_baseline_discover/`
- `runs/nnpu_margin_pair_s44_margin005_discover/`
- `runs/nnpu_margin_pair_s123_baseline_discover/`
- `runs/nnpu_margin_pair_s123_margin005_discover/`
- `analysis/nnpu_uncertainty_margin_bootstrap_s44_20261013.json`
- `analysis/nnpu_uncertainty_margin_bootstrap_s123_20261013.json`

## Decision

The loss is implemented and active, but it is not a reliable solution to the
core problem. It produces a small, seed-dependent change in representation
geometry and detector scores; it does not consistently improve AUROC, FPR95,
OSCR, unknown rejection, or the overlap diagnostic. It remains available as an
ablation, but its default coefficient should stay zero and it should not be
stacked into the main recipe.

This result narrows the diagnosis. The problem is not simply that the model
needs another local margin around known prototypes. Unknown images still map
into regions that are plausible under the known classifier, so a loss applied
to known-class geometry alone cannot reliably create an unknown boundary.

## Next direction

The next controlled intervention should provide a separately routed boundary
signal: use a disjoint, class-balanced open training pool or explicit
outlier-exposure/objectosphere supervision, while keeping known CE/KD and the
current detector fixed. The experiment must verify candidate purity, active
loss meters, and matched data protocol before interpreting metrics. A method is
promoted only if it improves ranking/overlap or OSCR across seeds without
meaningfully reducing known accuracy; a higher rejection rate alone is not
enough.
