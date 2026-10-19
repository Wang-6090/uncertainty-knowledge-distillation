# Semantic-hard candidate coverage audit by fine class (seeds 2041/2040)

## Question and protocol

After the independent-split CE comparison failed to improve the calibrated
operating point, this diagnostic asked whether a label-free candidate gate
covers novel training-pool classes, and whether known-validation classwise
percentile calibration improves that coverage. Frozen CE=1.0 student
checkpoints were evaluated on each split's matched 5,400-sample mixed
discovery pool (known prior 0.2). Both methods select exactly 25% of the pool:
(1) the existing entropy + uncertainty ranking and (2) ranking risk by the
empirical known-validation CDF within predicted class, then selecting the top
25% percentiles. A class with fewer than 20 validation examples uses the
global known-validation reference. Pool labels are inspected only after masks
are formed to measure contamination and per-novel-class recall; they do not
affect selection or calibration. Final test data are not used.

The experiment was a frozen-checkpoint mechanism audit, not a training result.
The split, checkpoint, pool budget and comparison were matched within each
seed. The cross-seed comparison tests whether a candidate-selection effect
generalizes across independent semantic-hard class splits.

## Corrected deterministic results

| Split | Selector | Selected | Known contamination | Novel recall | Median per-novel-class recall | Minimum per-class recall |
|---|---|---:|---:|---:|---:|---:|
| 2041 | Entropy + uncertainty | 25.00% | 9.19% | 28.38% | 26.77% | 1.75% |
| 2041 | Validation classwise empirical-CDF percentile, top 25% | 25.00% | 8.81% | 28.50% | 28.70% | 4.39% |
| 2040 | Entropy + uncertainty | 25.00% | 6.37% | 29.26% | 32.22% | 2.73% |
| 2040 | Validation classwise empirical-CDF percentile, top 25% | 25.00% | 6.89% | 29.10% | 31.32% | 2.73% |

On split 2041 the CDF method modestly improves minimum class recall
(`+2.63 pp`), median recall (`+1.93 pp`), contamination (`-0.37 pp`) and
aggregate novel recall (`+0.12 pp`). The independent split 2040 does not
replicate the effect: minimum recall is unchanged, median recall falls
`0.90 pp`, contamination rises `0.52 pp`, and aggregate recall falls `0.16
pp`. This fails the predeclared cross-split replication criterion; the CDF
selector remains diagnostic-only and is not wired into training. Even with
the CDF rule, worst-class recall remains below 5% on both splits, so the
candidate coverage tail is still a real bottleneck, not a solved problem.

## Implementation correction

The initial audit used `extract_outputs(..., mc_samples=1)` and described it
as deterministic. That was incorrect: the shared helper still invokes one
stochastic MC-dropout pass. Re-running it changed classwise statistics, so the
initial table and earlier class-balanced/q95 conclusions are superseded and
must not be treated as reproducible evidence. The audit now uses a dedicated
ordinary eval-mode forward pass for logits, uncertainty and features, with a
regression test that verifies repeatability. Corrected reports are saved under
new `*_deterministic_20261020.json` names; original JSON files are retained
for provenance, not as final deterministic measurements.

## Decision and next direction

Do not run full training for the CDF selector: its small 2041 improvement did
not replicate on split 2040. Do not use novel pool labels or final test labels
to tune selection. The next useful work is to audit training-pool signals
under deterministic extraction and ask whether they identify hard novel
examples consistently across splits while preserving low known contamination.
Only a mechanism that passes a predeclared cross-split lower-tail coverage and
contamination criterion should justify a paired training experiment. Any PU
loss must account for selection-induced class-prior changes rather than
assuming the original pool prior remains valid.

## Artifacts and validation

- Corrected audit JSONs:
  - `analysis/semantic_hard_candidate_pool_audit_s2041_deterministic_20261020.json`
  - `analysis/semantic_hard_candidate_pool_audit_s2040_deterministic_20261020.json`
- Frozen models: `runs/rejection_ce_balance_s2041_ce1p0_independent_hard_split/student.pt` and the corresponding seed-2040 checkpoint.
- Splits: `splits_cifar100_semantic_hard_seed2041/` and `splits_cifar100_semantic_hard_seed2040/`.
- Audit implementation: `scripts/audit_discovery_candidate_pool.py`.
- Validation: `231 passed, 2 warnings` in `tests/test_core_behaviors.py`.

This is a training-pool mechanism audit, not a final test-set performance
result. No full training run or GitHub upload was performed.
