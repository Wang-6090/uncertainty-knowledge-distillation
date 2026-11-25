# Classification View-Consistency Screening Pilot (2026-11-26)

## Question and scope

Does a fixed known-label, paired-view cosine consistency loss improve student
classification representation stability and transfer that improvement to
unknown detection and novel-class discovery? This is Team A work: standard-KD,
uncertainty-KD, student representation, KD/loss gradients, and downstream
transfer. The experiment does not modify the teacher recipe, unknown score,
rejector, candidate selection, Auto-K, or clustering algorithms.

## Protocol

- Dataset/protocol: CIFAR-100 semantic-hard, 60 known / 40 novel, fresh seed 2178.
- Arms: `standard`, `view_consistency`, `uncertainty`.
- All arms consumed two independently augmented known training views; the
  consistency coefficient was 0 for standard and uncertainty, and fixed at
  0.1 for view-consistency. No coefficient scan was performed.
- CE/KD-only student objective, four epochs, split-matched teacher, known-only
  checkpoint selection, frozen max-softmax discovery score, known-validation
  calibration to 95% coverage, matched 5400-item pool, Auto-K, and clustering.
- Unknown/test labels were not used for training, checkpoint selection,
  calibration, or treatment selection. The frozen stability diagnostic used
  known validation only; unknown/test labels were loaded only for final metrics.
- All three arms completed the four epochs and discovery. Calibration
  acceptance was 0.949036 for each arm.

## Mechanism audit

The treatment's per-epoch consistency loss was
`0.064517/0.055497/0.054377/0.053666`; its audited student-feature gradient
norm was `0.058804/0.047559/0.045826/0.044378`. The term was applied in every
epoch and paired views were active in every epoch. Standard and uncertainty
arms kept the coefficient at zero. Thus the objective was active and
differentiable; this is an implementation/mechanism result, not a performance
claim.

## Known-validation representation diagnostic

The frozen paired-augmentation audit used the same 3000 known-validation
samples in each arm (identical source-index hash), with no unknown/test labels.

| Measure | Standard | View consistency | Difference |
| --- | ---: | ---: | ---: |
| All-sample feature cosine | 0.942850 | 0.946353 | +0.003503 |
| Feature cosine q05 | 0.860547 | 0.869643 | +0.009096 |
| All-sample logit JS | 0.034074 | 0.033460 | -0.000614 |
| Prediction flip rate | 17.833% | 18.067% | +0.233 pp |
| At-least-one-view-wrong feature cosine | 0.927091 | 0.932400 | +0.005310 |
| At-least-one-view-wrong logit JS | 0.060350 | 0.058903 | -0.001447 |

Representation similarity improved modestly, but prediction stability did not:
flip rate was slightly worse. The diagnostic is descriptive for one seed and
does not establish statistical significance.

## Downstream results

| Metric | Standard | View consistency | Difference |
| --- | ---: | ---: | ---: |
| AUROC | 0.699255 | 0.696464 | -0.002791 |
| AUPR | 0.564926 | 0.559715 | -0.005211 |
| FPR95 | 0.730833 | 0.733667 | +0.002833 |
| Known evaluation acceptance | 95.667% | 95.417% | -0.250 pp |
| Unknown rejection | 11.700% | 11.500% | -0.200 pp |
| Candidate count | 728 | 735 | +7 |
| Candidate purity | 64.286% | 62.585% | -1.701 pp |
| Candidate unknown recall | 100% | 100% | 0 pp |
| Auto-K (true K=40) | 29 (error 11) | 42 (error 2) | error -9 |
| Clustering ACC | 0.173077 | 0.168707 | -0.004369 |
| Clustering NMI | 0.412822 | 0.441088 | +0.028266 |
| Clustering ARI | 0.056186 | 0.036633 | -0.019554 |
| Best known-validation accuracy | 69.600% | 68.667% | -0.933 pp |

The Auto-K estimate and NMI improved, but ACC/ARI and candidate purity fell;
ranking detection also regressed. Therefore the pilot is mixed and does not
support default promotion. Because representation similarity and two discovery
signals (Auto-K error/NMI) improved while known utility fell by under one
percentage point, the predeclared screening gate allows one fixed, fresh-seed
confirmation. This is a replication check, not evidence that the method works.

## Seed 2179 confirmation

The fixed-coefficient confirmation completed successfully on the same
semantic-hard protocol. No alpha scan or post-hoc selection was performed.

| Metric | Standard | View consistency | Difference |
| --- | ---: | ---: | ---: |
| AUROC | 0.661677 | 0.664383 | +0.002706 |
| AUPR | 0.519103 | 0.520658 | +0.001555 |
| FPR95 | 0.769167 | 0.763167 | -0.600 pp |
| Known evaluation acceptance | 94.300% | 94.233% | -0.067 pp |
| Unknown rejection | 10.625% | 10.700% | +0.075 pp |
| Candidate count | 767 | 774 | +7 |
| Candidate purity | 55.411% | 55.297% | -0.114 pp |
| Auto-K (true K=40) | 42 (error 2) | 40 (error 0) | error -2 |
| Clustering ACC | 0.178618 | 0.171835 | -0.678 pp |
| Clustering NMI | 0.452035 | 0.435913 | -1.612 pp |
| Clustering ARI | 0.047461 | 0.044183 | -0.328 pp |
| Best known-validation accuracy | 67.467% | 66.950% | -0.517 pp |

The representation signal repeated only partially: feature cosine improved
from `0.943799` to `0.946012`, q05 from `0.859641` to `0.867986`, and
prediction flip rate fell from `18.333%` to `18.000%`; logit JS was slightly
worse (`0.033808` to `0.033930`). However, all three clustering metrics fell,
and candidate purity and best known accuracy also fell slightly. Together with
the 2178 pilot, this is mechanism success without stable end-to-end gain.

The uncertainty arm remains a required Team A comparator. It was better than
standard on several 2179 detection metrics, but this did not repeat the mixed
2178 direction; it is not evidence for an additional stable uncertainty-KD
benefit on this candidate.

## Decision

Keep classification view-consistency default-off and do not scan its
coefficient further. Preserve the implementation as an opt-in mechanism
audit and record the two-seed result as a stopped candidate. The experiment
does not justify changing the teacher, rejector, candidate selection, Auto-K,
or clustering components.

Machine-readable outputs:

- `analysis/team_a_viewconsistency_20261126_s2178.json`
- `analysis/team_a_viewconsistency_20261126_s2178_stability.json`
- `analysis/team_a_viewconsistency_confirm_20261126_s2179.json`
- `analysis/team_a_viewconsistency_confirm_20261126_s2179_stability.json`
- `runs/team_a_viewconsistency_20261126_s2178_{standard,view_consistency,uncertainty}/`

## Reproducibility and limitations

Runner: `scripts/run_uncertainty_kd_core_pair.py`, profile
`ce_classification_view_consistency`. Stability script:
`scripts/audit_classification_view_stability.py`. Full suite before the pilot:
416 passed, one pre-existing PyTorch scalar-conversion warning. This is one
semantic-hard split and one set of stochastic training runs; split identities
are not independent repeated measurements. No significance claim is made.
