# Full-data prototype-repulsion check (2026-09-29)

## Protocol

CIFAR-100 60/40, seed 42, stratified known train/validation split, complete
training data, pretrained ResNet-34 teacher and ResNet-18 student, 15 epochs,
Standard KD, and the complete 10,000-image open test set. Both students use
the same teacher checkpoint and differ only in `alpha_proto_repulsion` (0 vs
0.01). Detection uses the same `normalized_entropy_mahalanobis` score and
`known_coverage=0.95` calibration. Clustering is skipped to isolate detection.

## Results

| Method | AUROC | AUPR | FPR95 | Test known accept | Unknown reject | Known accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard KD baseline | **0.5309** | **0.4119** | **0.8987** | 0.9537 | **0.0433** | 0.4017 |
| Prototype repulsion, alpha=0.01 | 0.5167 | 0.3999 | 0.9047 | 0.9493 | 0.0380 | **0.4105** |

The baseline trained to 44.00% best known validation accuracy; repulsion trained
to 45.03%. Despite the slightly higher validation classification accuracy,
repulsion reduced AUROC by `0.0142`, increased FPR95 by `0.0060`, and reduced
unknown rejection by `0.0053` at a similar known coverage.

Known and unknown score distributions remain highly overlapping. Baseline
median scores are `0.059` for known and `0.165` for unknown; with repulsion the
medians are `0.024` and `0.097`. The known 90th percentile is close to the
unknown 90th percentile in both models. The loss therefore did not create a
useful detection margin.

## Conclusion and next step

The complete-data matched-protocol check does not support prototype repulsion
as a detection improvement. Keep it as an optional ablation, not the default.
The previous small-sample three-seed ranking gain did not transfer to this
full-data stratified comparison. Do not continue tuning its weight yet.

Next, inspect known and unknown nearest-class prototype-distance distributions
and per-class open-set errors on these fixed checkpoints. Then compare a
stronger pretrained representation or a metric-learning objective under the
same split. For the actual mixed-pool setting, the longer-term direction is a
GCD-style unified known+novel objective rather than treating every unlabeled
sample as unknown. Any new objective should be checked on at least three
full-data seeds before being claimed effective.

Run directories: `runs/full_baseline_s42_stratified` and
`runs/full_repulsion_s42_stratified`, with corresponding `*_detect` folders.
