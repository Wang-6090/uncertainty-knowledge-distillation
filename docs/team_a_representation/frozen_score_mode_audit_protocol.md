# Frozen rejection score mode audit protocol

## Motivation

The two recent gradient routing screens exposed a trade-off between unknown
coverage, candidate purity and clustering quality. The student checkpoints are
already trained, and the repository contains two fixed score combinations that
combine the NNPU feature rejector with the student's validation-calibrated
uncertainty error. This stage tests those combinations without retraining the
representation, so any change is attributable to discovery-time scoring.

## Fixed comparison

The reference is the existing `feature_rejector` score. The two frozen
alternatives are:

- `feature_rejector_uncertainty_fusion`: standardized rejector and uncertainty
  error averaged with fixed 0.5/0.5 weights;
- `feature_rejector_uncertainty_conjunction`: the minimum of the two
  standardized components, requiring both signals to be high.

The score formulas, weights and modes are selected from existing code. There is
no test-label tuning or weight scan. Each mode fits the same NNPU rejector from
the matched mixed discovery pool, uses the same frozen student checkpoint, and
calibrates its threshold on the independent known-validation subset at 95%
known coverage. Clustering remains enabled with the existing Auto-K pipeline.

## Label and protocol gates

The discovery pool is mixed and treated as unlabeled by NNPU; no hard mixed-pool
labels are used. Test labels are read only for final report metrics. The runner
refuses to overwrite output directories and validates the threshold and label
audit fields in every report. The audit changes no teacher, student, optimizer,
checkpoint-selection or representation setting.

## Seed 2153 frozen audit

On the control checkpoint, conjunction changed unknown rejection from `23.125%`
to `30.400%`, candidate purity from `81.426%` to `80.851%`, Cluster ACC from
`0.285211` to `0.289894`, and ARI from `0.173408` to `0.185803`. On the
componentwise treatment checkpoint, it changed unknown rejection from `26.650%`
to `30.375%`, candidate purity from `78.498%` to `83.162%`, Cluster ACC from
`0.268778` to `0.286105`, NMI from `0.492774` to `0.501986`, and ARI from
`0.154268` to `0.176946`. AUROC, AUPR, FPR95 and OSCR also improved within
each frozen checkpoint comparison.

The fusion mode improved ranking and fixed-coverage rejection as well, but its
clustering changes were mixed. Conjunction is therefore the more targeted
candidate for a confirmation audit because it improved candidate purity and
all three clustering measures on the componentwise checkpoint while retaining
the unknown-rejection gain.

This is one seed and reuses trained checkpoints, so it is exploratory evidence.
It does not change the default score or establish a final algorithmic gain.
The next stage, if pursued, is a fresh-seed frozen-score confirmation under the
same predeclared conjunction rule; no conjunction threshold or component
weight scan is allowed before that confirmation.
