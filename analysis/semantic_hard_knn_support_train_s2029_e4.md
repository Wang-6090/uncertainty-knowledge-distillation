# Semantic-hard training-time kNN support-boundary check (2026-10-19)

## Question and pre-registered comparison

Does adding the existing class-conditional kNN support-boundary objective to the shared feature during student training improve the semantic-hard known/unknown separation, rather than only changing the operating threshold?

- Dataset protocol: CIFAR-100 semantic-hard, 60 known / 40 novel fine classes, same split in both arms.
- Seed / budget: student seed 2029, 4 epochs; both arms share the same split-compatible teacher checkpoint (seed-2028 semantic-hard teacher).
- Single intended factor: `alpha_discovery_knn_boundary=0` versus `0.1`.
- Treatment implementation: `pure` risk; `shared_boundary` candidate space; candidate and cross-view gating enabled. Independent rejection-boundary and rejection-kNN losses are explicitly zero.
- Evaluation held fixed: `feature_rejector`, all-pool rejector, matched mixed pool (5,400, known prior 0.2), clustering skipped, threshold calibrated only on known validation data to target 95% known coverage.
- Test labels were not used for fitting, checkpoint selection, calibration, or method selection. The test metrics below are final held-out reporting only.

The experiment was intended as one-factor screening. It is exploratory (one seed and one semantic-hard class split), not a confirmatory result.

## Results

Treatment minus baseline; positive/negative directions are shown as raw differences. For FPR95, a negative difference is favorable. Rate deltas are percentage points.

| Metric | Baseline | kNN support-boundary | Difference |
|---|---:|---:|---:|
| AUROC | 0.79636 | 0.79578 | -0.00058 |
| AUPR | 0.69689 | 0.69875 | +0.00186 |
| FPR95 | 0.61367 | 0.60883 | -0.00483 |
| OSCR | 0.58346 | 0.59401 | +0.01055 |
| Known accept rate | 95.23% | 94.88% | -0.35 pp |
| Unknown reject rate at calibrated threshold | 26.50% | 27.68% | +1.18 pp |
| Accepted-known classification accuracy | 70.13% | 71.74% | +1.61 pp |
| Score KS distance | 0.44483 | 0.44425 | -0.00058 |
| Unknown score below known q95 | 72.50% | 72.525% | +0.025 pp |
| Mean unknown-minus-known score gap | 4.66232 | 4.74484 | +0.08252 |

The loss was actually active, not merely configured: treatment training history records nonzero `discovery_knn_boundary` (0.0195–0.0328 across epochs) and nonzero gradient norm (1.35–1.58). The other arm has weight zero. The treatment's best known-validation accuracy was 70.70% versus 70.04% in baseline; both selected epoch 4 under the same known-accuracy checkpoint rule.

The training log's candidate-gate audit adds an important mechanism check. In the treatment, the cross-view gate selected 12.9%, 17.0%, 17.4%, and 17.3% of the mixed discovery batches over the four epochs. Its diagnostic-only pool labels (not passed to the loss, gate, threshold, or optimizer) indicate the selected set's known fraction fell from 19.6% to 9.0%, while unknown recall rose only from 12.9% to 19.7%; view agreement was about 0.54 by the end. The `discovery_knn_*_selection_rate` fields are zero because selection-conditional PU was disabled; they must not be interpreted as zero candidate selection. The separate `feature_candidate_*` fields are the relevant cross-view gate diagnostics. This is training-pool audit information, not test-set selection.

## Interpretation and decision

This is a mixed, small signal. At the known-validation-calibrated operating point, unknown rejection rose by 1.18 pp and OSCR improved by 0.01055, while FPR95 improved slightly. But AUROC fell slightly, score KS was effectively unchanged, and the key overlap proxy (`unknown below known q95`) did not improve at all. Known coverage is 0.35 pp lower than target on this finite test sample; the comparison still uses the same validation-only calibration rule in both arms. The gate audit suggests why the training loss may have limited impact: it has low unknown recall, so most novel examples never contribute to the selected candidate loss even though the loss itself has a nonzero gradient. The apparent operating-point increase therefore does not establish genuine distribution separation.

Decision: do not enable this objective by default and do not tune its weight on this test set. Preserve it as a weak candidate only if an independent, pre-fixed seed/split replication is affordable; given the negligible overlap change and roughly 25–35% slower epochs from constructing the kNN support bank, it is not currently a priority. The main conclusion remains that semantic-hard known/unknown support regions overlap broadly.

## Next direction

Do not keep changing post-hoc thresholds or add more score fusion. Prioritize the candidate-recall bottleneck: study a pre-fixed, label-free candidate rule that increases coverage of plausible novel boundary points without using test labels, and validate false-positive/known-contamination behavior only through known-train-derived controls and held-out validation. Keep diagnostic discovery-pool labels strictly out of all training decisions. Explicitly log candidate selection/contamination proxies and per-class support coverage, and evaluate the representation objective separately from rejector fitting. Pre-fix the split, seed, model-selection rule, calibration, and metrics; require improvement in ranking/overlap plus coverage-matched rejection without materially harming known utility before considering a replication or default change.

## Artifacts

- Runner: [`scripts/run_semantic_hard_knn_support_train_compare.py`](../scripts/run_semantic_hard_knn_support_train_compare.py)
- Machine-readable paired report: [`semantic_hard_knn_support_train_s2029_e4_summary.json`](semantic_hard_knn_support_train_s2029_e4_summary.json)
- Baseline run: `runs/semantic_hard_knn_support_train_s2029_e4_baseline/`
- Treatment run: `runs/semantic_hard_knn_support_train_s2029_e4_knn_support_boundary/`
