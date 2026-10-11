# Detached-feature uncertainty nnPU recheck (2026-10-11)

## Motivation and hypothesis

The seed-123 distance-consensus recheck found that the weighted PU loss improved
the uncertainty head AUROC/KS but worsened prototype distance, entropy, and the
final detector. This motivated a gradient-routing hypothesis: perhaps the PU
objective should train the uncertainty head without changing the shared image
representation.

## Implementation and controlled comparison

Added the opt-in flag `--discovery-uncertainty-pu-detach-features`. With the flag,
the known and mixed-pool embeddings are detached and passed through the existing
uncertainty/rejection head to compute nnPU. Thus the PU loss updates that head,
while all other objectives retain their original gradients. The default remains
off. A unit test confirms the PU loss gives gradients to the head and none to
the input features.

The full treatment reused the seed-123 teacher, CIFAR-100 semantic-isolated
60/40 split, full data, pretrained ResNet-18 student, batch size 64, five
epochs, matched mixed pool of 5400 (known prior 0.2), weighted nnPU
(`distance_consensus`, weight 0.1), and the same validation checkpoint rule.
The only intentional training configuration difference from the direct weighted
PU treatment is the new detach flag. The selected best epoch is an outcome
(direct: 5, detached: 3), not a manually fixed checkpoint choice.

Both student checkpoints were evaluated with the same detector protocol:
`normalized_entropy_proto`, MC=4, validation-only 95% known coverage, no
clustering. The unweighted seed-123 model is included as a contextual reference.

## Results

| Metric | Unweighted nnPU | Weighted, direct gradient | Weighted, detached features |
|---|---:|---:|---:|
| AUROC | 0.7180 | 0.7012 | 0.6997 |
| AUPR | 0.5938 | 0.5716 | 0.5615 |
| FPR95 | 0.7635 | 0.7800 | 0.7832 |
| OSCR | 0.3914 | 0.3720 | 0.3639 |
| Known acceptance | 95.92% | 94.75% | 95.47% |
| Unknown rejection | 13.53% | 13.18% | 11.73% |
| Known accuracy, all known | 46.58% | 44.78% | 43.68% |
| Score KS | 0.3348 | 0.3041 | 0.3103 |

Reports/configs:

- Direct weighted: `runs/distance_consensus_full_s123_treatment_discover/discovery_report.json`
- Detached weighted: `runs/pu_detached_features_full_s123_discover/discovery_report.json`
- Training config: `runs/pu_detached_features_full_s123/train_student_config.json`

## Decision

The detached-feature treatment does not improve the overall detector and has
lower AUROC/AUPR/OSCR and unknown rejection than the unweighted reference. It
slightly raises known acceptance relative to direct weighted PU, but that is
not a sufficient gain: the model accepts more known examples while still
rejecting fewer unknowns than the baseline. This does not support adopting
detached PU as a solution to representation overlap. Keep the flag as an opt-in
gradient-routing ablation, not as the default. Do not continue sweeping this
single flag/weight combination from this one seed.

The broader hypothesis that PU gradients conflict with representation learning
is not disproved: only one gradient-routing intervention and one seed were
tested. However, this specific head-only route failed to improve open-set
ranking. Next, prioritize a data/protocol and representation audit: compare
class-conditional feature overlap for the same examples across teacher, student
baseline, and PU variants; verify whether the dominant errors come from semantic
similarity, low image resolution, or class-conditional multimodality. Any next
loss change should be justified by that diagnosis and tested with a paired seed
and the same detector/calibration split.
