# Quality-aware logits-KD confirmation: semantic-hard seed 2155

This is one fresh confirmation seed for the fixed quality-aware KD protocol.
The three arms share the same split-matched teacher, student initialization,
data order, four-epoch budget, auxiliary objectives, known-validation
checkpoint selection, matched mixed discovery pool, rejector, calibration and
automatic-K clustering. The only objective change is `kd_mode`.

## Protocol and mechanism gates

- Split: CIFAR-100 semantic-hard 60/40, seed `2155`.
- Arms: standard logits-KD, unweighted decoupled KD, quality-aware logits-KD.
- Feature KD and rejection-branch KD: disabled.
- Checkpoint selection: known-validation accuracy only.
- Calibration: known validation only, target coverage `0.95`; realized
  acceptance was `0.949176` for all arms.
- Test labels: not used for training or calibration; they appear only in final
  evaluation fields.
- Quality-aware formula: detached teacher true-class quality weight,
  `floor=0.25`, `power=1.0`, batch mean normalization.

The mechanism gate passed. Quality-aware weighting was applied in all four
epochs; the final weight standard deviation was `0.541676` and ESS fraction
was `0.773068`. Its final KD gradient norm was `11.884905` and student
classifier update norm was `0.025770`. The teacher was frozen. The standard
and decoupled arms reported `kd_quality_weight_applied=false`.

## Results

| arm | val known acc | AUROC | AUPR | FPR95 | OSCR | known acc | known accept | unknown reject | candidate purity | candidate recall | Auto-K error | ACC | NMI | ARI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| standard | 0.703000 | 0.768588 | 0.663637 | 0.656833 | 0.558264 | 0.682000 | 0.939000 | 0.260250 | 0.739872 | 1.000000 | 9 | 0.304193 | 0.550093 | 0.225527 |
| decoupled | 0.694667 | 0.779352 | 0.672365 | 0.634167 | 0.568377 | 0.679167 | 0.955167 | 0.212500 | 0.759607 | 1.000000 | 7 | 0.302949 | 0.551413 | 0.207498 |
| quality-aware | 0.719333 | 0.760300 | 0.637238 | 0.640167 | 0.582827 | 0.708833 | 0.951667 | 0.188250 | 0.721956 | 1.000000 | 16 | 0.263663 | 0.505490 | 0.183322 |

Relative to standard KD, quality-aware changes were:

| metric | change |
|---|---:|
| AUROC | -0.008288 |
| AUPR | -0.026399 |
| FPR95 | -0.016667 |
| OSCR | +0.024564 |
| known accuracy | +2.683 pp |
| known acceptance | +1.267 pp |
| unknown rejection | -7.200 pp |
| candidate purity | -1.792 pp |
| Auto-K absolute error | +7 |
| cluster ACC / NMI / ARI | -0.040531 / -0.044604 / -0.042205 |

## Decision

This seed does not repeat the pilot's joint candidate-quality signal. The
known accuracy and OSCR improvements are accompanied by worse unknown
coverage, candidate purity, Auto-K error and clustering quality. It is a
valid negative confirmation result, not evidence of an implementation
failure. Together with seed 2156, it prevents promotion to the default
configuration. Quality-aware KD remains an opt-in diagnostic; no floor,
power, temperature or loss-weight scan is justified.

Machine result: `analysis/team_a_quality_kd_confirm_s2155.json`.
