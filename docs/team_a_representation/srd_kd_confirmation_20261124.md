# SRD cross-network KD confirmation

## Scope and protocol

This is the fixed two-seed confirmation of the SRD pilot. It remains within
Team A's student representation and KD scope: standard logits KD, SRD through
the frozen teacher classifier, and the existing uncertainty-KD arm were run on
semantic-hard CIFAR-100 60/40 splits with seeds `2171` and `2172`.

All arms used the same split-matched teacher, student initialization, four
student epochs, CE/KD-only objective, known-validation checkpoint selection,
max-softmax detection, known-validation 95% coverage calibration, matched
discovery protocol, Auto-K, and clustering. Unknown/test labels were reserved
for final evaluation. No coefficient, temperature, epoch, feature
normalization, score, or candidate-rule scan was performed. The machine
results are the local, gitignored artifacts
`analysis/team_a_srd_confirm_20261123_s2171.json` and
`analysis/team_a_srd_confirm_20261123_s2172.json`.

## Mechanism audit

The SRD implementation passed the formula, teacher-detachment, feature-gradient,
and dimension unit tests. The focused suite passed with `25 passed`, and
compileall passed. In both confirmation seeds, SRD had a positive KD loss,
positive student-encoder KD gradient, and positive student update in every
epoch. The teacher logits and teacher classifier are detached. Uncertainty-KD
remained a separate arm with nonuniform teacher-uncertainty weights.

This confirms that the treatment was active and updated the student. It does
not by itself establish a discovery benefit.

## Raw final metrics

The table reports the actual final evaluation fields. `Unknown rejection` is
the fixed-coverage unknown rejection rate, and `candidate purity` is the
purity of the automatically selected candidate pool.

| Seed | Arm | AUROC | AUPR | FPR95 | OSCR | Known acc. | Unknown rejection | Candidate purity | Cluster ACC | Cluster NMI | Cluster ARI | Auto-K error |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2171 | standard | 0.680526 | 0.553619 | 0.795500 | 0.529128 | 0.681667 | 0.129250 | 0.625151 | 0.178960 | 0.418438 | 0.049199 | 3 |
| 2171 | SRD | 0.703296 | 0.561952 | 0.711833 | 0.573252 | 0.713333 | 0.127500 | 0.619684 | 0.159174 | 0.440407 | 0.033039 | 6 |
| 2171 | uncertainty | 0.690219 | 0.558459 | 0.779667 | 0.540511 | 0.683000 | 0.122000 | 0.649800 | 0.202397 | 0.471955 | 0.055111 | 6 |
| 2172 | standard | 0.706335 | 0.556692 | 0.700667 | 0.546750 | 0.671167 | 0.091750 | 0.589085 | 0.176565 | 0.454729 | 0.033391 | 1 |
| 2172 | SRD | 0.706612 | 0.558784 | 0.690667 | 0.563417 | 0.693833 | 0.096250 | 0.595054 | 0.177743 | 0.467145 | 0.036420 | 5 |
| 2172 | uncertainty | 0.712128 | 0.563081 | 0.700833 | 0.551631 | 0.677500 | 0.108250 | 0.598893 | 0.157676 | 0.413029 | 0.032559 | 6 |

## Paired changes versus standard-KD

Values below are treatment minus standard. The uncertainty values are sample
standard deviations across the two fresh seeds and are descriptive only.

| Arm | AUROC | AUPR | FPR95 | OSCR | Known acc. | Unknown rejection | Candidate purity | Cluster ACC | Cluster NMI | Cluster ARI | Auto-K error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SRD, seed 2171 | +0.022769 | +0.008333 | -0.083667 | +0.044124 | +0.031667 | -0.001750 | -0.005467 | -0.019786 | +0.021968 | -0.016160 | +3 |
| SRD, seed 2172 | +0.000277 | +0.002092 | -0.010000 | +0.016667 | +0.022667 | +0.004500 | +0.005969 | +0.001178 | +0.012417 | +0.003030 | +4 |
| SRD mean +/- SD | +0.011523 +/- 0.015905 | +0.005213 +/- 0.004413 | -0.046833 +/- 0.052090 | +0.030396 +/- 0.019415 | +0.027167 +/- 0.006364 | +0.001375 +/- 0.004419 | +0.000251 +/- 0.008087 | -0.009304 +/- 0.014824 | +0.017193 +/- 0.006754 | -0.006565 +/- 0.013569 | +3.5 +/- 0.707 |
| uncertainty mean +/- SD | +0.007743 +/- 0.002758 | +0.005614 +/- 0.001095 | -0.007833 +/- 0.011314 | +0.008132 +/- 0.004598 | +0.003833 +/- 0.003536 | +0.004625 +/- 0.016794 | +0.017229 +/- 0.010494 | +0.002274 +/- 0.029929 | +0.005909 +/- 0.067329 | +0.002540 +/- 0.004768 | +4.0 +/- 1.414 |

SRD met the non-decrease requirements for mean AUROC, unknown rejection, and
candidate purity, and known accuracy did not fall. However, only one of the
three clustering changes was nonnegative on average: NMI improved, while ACC
and ARI decreased. The predeclared confirmation gate required at least two of
ACC/NMI/ARI to be nonnegative. Therefore SRD fails the downstream promotion
gate despite passing the mechanism and representation gates.

## Representation audit

The same 3,000 known-validation samples and explicit CIFAR transform were used
for both arms in each seed. Unknown labels were not accessed.

| Seed | Raw norm delta | Within cosine delta | Nearest-center delta | Center-margin delta | Between/total scatter delta |
|---:|---:|---:|---:|---:|---:|
| 2171 | +4.138207 | -0.044369 | -0.040948 | +0.003420 | +0.016440 |
| 2172 | +5.261857 | -0.045275 | -0.033329 | +0.011946 | +0.015459 |

The representation signal reproduced: SRD increased raw feature magnitude,
reduced normalized within-class and nearest-center distances, and improved
the between/total scatter ratio on both seeds. The downstream clustering gate
did not reproduce as a joint improvement, so the representation change is
diagnostic rather than evidence of a better novel-class partition.

## Decision and next stage

SRD is correctly implemented and mechanically active, but it is not a stable
end-to-end improvement for this novel-class discovery protocol. Keep it as an
opt-in ablation and keep `kd_mode=standard` as the default. Do not repeat
seeds `2170/2171/2172`, scan SRD coefficients, or continue uncertainty-weight
scans for this hypothesis.

The next Team A stage should target the missing objective directly: a
known-only, training-time constraint that improves the support boundary or
partition structure without using unknown labels, while retaining standard-KD
as the clean baseline and preserving the separate uncertainty-KD audit. Any
new proposal must first pass a mechanism test and a paired fresh-split pilot;
it should be stopped if a representation gain again fails to transfer to
unknown rejection and clustering.
