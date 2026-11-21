# Quality-aware logits-KD confirmation: semantic-hard seed 2156

This is one fresh confirmation seed for the fixed quality-aware KD protocol.
The three arms share the same split-matched teacher, student initialization,
data order, four-epoch budget, auxiliary objectives, known-validation
checkpoint selection, matched mixed discovery pool, rejector, calibration and
automatic-K clustering. The only objective change is `kd_mode`.

## Protocol and mechanism gates

- Split: CIFAR-100 semantic-hard 60/40, seed `2156`.
- Arms: standard logits-KD, unweighted decoupled KD, quality-aware logits-KD.
- Feature KD and rejection-branch KD: disabled.
- Checkpoint selection: known-validation accuracy only.
- Calibration: known validation only, target coverage `0.95`; realized
  acceptance was `0.949384` for all arms.
- Test labels: not used for training or calibration; they appear only in final
  evaluation fields.
- Quality-aware formula: detached teacher true-class quality weight,
  `floor=0.25`, `power=1.0`, batch mean normalization.

The mechanism gate passed. Quality-aware weighting was applied in all four
epochs; the final weight standard deviation was `0.512310` and ESS fraction
was `0.791916`. Its final KD gradient norm was `11.091833` and student
classifier update norm was `0.026086`. The teacher was frozen. The standard
and decoupled arms reported `kd_quality_weight_applied=false`.

## Results

| arm | val known acc | AUROC | AUPR | FPR95 | OSCR | known acc | known accept | unknown reject | candidate purity | candidate recall | Auto-K error | ACC | NMI | ARI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| standard | 0.702667 | 0.782884 | 0.668654 | 0.595667 | 0.585708 | 0.692167 | 0.955167 | 0.190750 | 0.739341 | 1.000000 | 12 | 0.251938 | 0.477163 | 0.128035 |
| decoupled | 0.707000 | 0.796763 | 0.697505 | 0.600500 | 0.592820 | 0.700333 | 0.945000 | 0.271250 | 0.766784 | 1.000000 | 6 | 0.306007 | 0.522125 | 0.207483 |
| quality-aware | 0.706667 | 0.803792 | 0.704118 | 0.586000 | 0.601485 | 0.703333 | 0.946500 | 0.278250 | 0.776151 | 1.000000 | 5 | 0.296374 | 0.514804 | 0.199916 |

Relative to standard KD, quality-aware changes were:

| metric | change |
|---|---:|
| AUROC | +0.020908 |
| AUPR | +0.035464 |
| FPR95 | -0.009667 |
| OSCR | +0.015777 |
| known accuracy | +1.117 pp |
| known acceptance | -0.867 pp |
| unknown rejection | +8.750 pp |
| candidate purity | +3.681 pp |
| Auto-K absolute error | -7 |
| cluster ACC / NMI / ARI | +0.044436 / +0.037641 / +0.071881 |

## Decision

This seed provides a positive confirmation signal: ranking, fixed-coverage
unknown rejection, candidate purity, Auto-K error and clustering metrics all
move in a useful direction, with no known-accuracy loss. The direction is
opposite to seed 2155 on most end-to-end metrics, so the two-seed evidence is
not stable enough to change the default or support a general improvement
claim. Quality-aware KD remains an opt-in diagnostic pending a larger,
pre-specified A/B/C aggregation.

Machine result: `analysis/team_a_quality_kd_confirm_s2156.json`.
