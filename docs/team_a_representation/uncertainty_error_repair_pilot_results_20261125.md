# Teacher-error-only target-repair KD pilot results

## Protocol and mechanism audit

The fresh semantic-hard CIFAR-100 60/40 seed `2177` completed all four paired
arms: `standard`, `hard_error_repair`, `error_repair`, and `uncertainty`.
The split-matched teacher and student initialization, CE/KD-only objective,
four-epoch budget, known-validation checkpoint selection, known-validation
95% coverage calibration, `max_softmax`, 5,400-item matched discovery pool,
Auto-K, and clustering protocol were held fixed. Unknown/test labels were
reserved for final retrospective metrics. The runner exited successfully.

The repair mechanism was active in every epoch. The teacher-error fraction was
`0.334630/0.334259/0.332222/0.331806` in both repair arms. Mean repair
coefficient was the same for hard repair, and
`0.110960/0.110797/0.110110/0.110527` for uncertainty-scaled repair. KD
gradient norms and student classifier update norms remained nonzero in every
epoch:

| Arm | KD gradient norm by epoch | Student classifier update norm by epoch |
|---|---|---|
| standard | 16.4731 / 13.3636 / 12.1972 / 11.1548 | 0.04008 / 0.02794 / 0.02702 / 0.02641 |
| hard error repair | 25.7850 / 24.5278 / 23.6069 / 22.4330 | 0.04210 / 0.03085 / 0.03045 / 0.03012 |
| uncertainty error repair | 17.5879 / 14.6114 / 13.2670 / 11.9968 | 0.04048 / 0.02896 / 0.02846 / 0.02801 |
| uncertainty-KD | 17.2319 / 13.6761 / 12.2457 / 11.2449 | 0.03985 / 0.02787 / 0.02698 / 0.02624 |

The existing uncertainty-KD weight was applied in all four epochs. Thus the
repair coefficient entered the target and KD objective and produced student
updates; the negative end-to-end outcome is not explained by an inactive loss
path.

## Results

| Metric | standard | hard error repair | uncertainty error repair | uncertainty-KD |
|---|---:|---:|---:|---:|
| AUROC | 0.692088 | 0.695575 | 0.694980 | 0.691185 |
| AUPR | 0.552125 | 0.554388 | 0.560444 | 0.553665 |
| FPR95 | 0.746667 | 0.743000 | 0.754833 | 0.756500 |
| Known accuracy | 0.706333 | 0.707000 | 0.724833 | 0.712167 |
| Known acceptance | 0.951833 | 0.934833 | 0.930167 | 0.951333 |
| Unknown rejection | 0.116500 | 0.144250 | 0.159500 | 0.117500 |
| Candidate purity | 0.617219 | 0.596074 | 0.603595 | 0.616798 |
| Auto-K (absolute error; true K=40) | 41 (1) | 47 (7) | 36 (4) | 46 (6) |
| All-unknown oracle ACC / NMI / ARI | 0.340000 / 0.455938 / 0.204795 | 0.406750 / 0.484123 / 0.254280 | 0.376750 / 0.473343 / 0.230012 | 0.371750 / 0.467720 / 0.228046 |
| Candidate-only oracle ACC / NMI / ARI | 0.218884 / 0.434744 / 0.053930 | 0.258232 / 0.443051 / 0.086451 | 0.258621 / 0.445156 / 0.103183 | 0.248936 / 0.448427 / 0.068535 |

Relative to standard-KD, uncertainty error-repair improves AUROC by `0.002892`,
AUPR by `0.008319`, and fixed-coverage unknown rejection by `4.300 pp`. Its
candidate-only oracle ACC/NMI/ARI improve by `3.974/1.041/4.925 pp`. However,
known acceptance falls by `2.167 pp`, candidate purity falls by `1.362 pp`,
Auto-K absolute error increases from `1` to `4`, and all-unknown oracle
clustering improves only modestly. Hard repair gives a stronger all-unknown
oracle result on this seed, but reduces candidate purity by `2.114 pp` and
increases Auto-K error by `6`.

Uncertainty scaling therefore did not establish an advantage over hard repair:
it modestly improves AUROC/AUPR and candidate ARI but has lower unknown
rejection and lower all-unknown oracle ACC/NMI/ARI. The single-seed screen does
not support a robust ranking between the repair variants.

## Decision

The mechanism gate passes, but neither repair arm passes the pre-registered
end-to-end gate. Both lose candidate purity and substantially reduce known
acceptance; both repeat the Auto-K degradation seen in seeds `2175` and
`2176`. Keep both KD modes opt-in for ablation only. Do not change the default,
scan repair coefficients, or repeat this hypothesis on another seed.

The useful finding is narrower: teacher-error gating can strengthen unknown
rejection and candidate-subset clustering on a screening split, but the
additional candidate recall does not translate into reliable candidate
purity or unknown-class-count estimation. Uncertainty-scaled correction does
not yet show a stable advantage over direct hard correction. Further work
should return to representation objectives that can be validated on known-only
geometry before paying for another full discovery run; rejector, candidate
selection, Auto-K, and clustering remain outside this workstream.

The machine-readable result is locally retained at
`analysis/team_a_uncertainty_error_repair_pilot_20261125_s2177.json` (the
`analysis/` directory is gitignored). The protocol is documented in
[`uncertainty_error_repair_pilot_20261125.md`](uncertainty_error_repair_pilot_20261125.md).
