# VOS source-gradient paired probe (seed=123, 2026-10-18)

## Question and predeclared comparison

Does allowing nearest-class inter-class virtual outliers to backpropagate
through their source features materially change VOS gradients and improve
known/unknown score separation? Both full-recipe arms used the same seed 123,
semantic-isolated CIFAR-100 60/40 split, teacher initialization, four epochs,
nearest-class pairing, boundary fraction 0.25, `alpha_vos=0.01`,
`alpha_vos_rejection=0`, mixed discovery pool, feature rejector, and validation
calibration targeting 95% known coverage. The only intended factor was
`vos_source_gradient=False` (detached) vs `True` (gradient through synthetic
feature construction). No test labels were used for training, checkpoint
selection, or calibration. The preceding 512-example run was only a gradient
mechanism smoke test and is not used for performance claims.

## Gradient mechanism

Gradient norms are averaged across the four epochs; each epoch audited 338
training batches. `vos_total_term` is the actual weighted VOS gradient,
including its real-known-logit branch. `vos_virtual_source_path` isolates the
synthetic-source branch by detaching the known logits.

| Arm | Encoder norm: total VOS | Encoder norm: virtual-source path | Classifier norm: VOS | Rejection-head norm | Cosine(total, source path) |
|---|---:|---:|---:|---:|---:|
| detached | 0.2173 | 0.0000 | 0.0623 | 0.0000 | 0.0000 |
| source-gradient | 0.2087 | 0.2868 | 0.0596 | 0.0000 | 0.6621 |

This confirms detach removes only the synthetic-feature construction path.
The total encoder VOS gradient remains nonzero because the energy margin also
depends on real known logits. Under source-gradient, the isolated source-path
gradient is larger than the total VOS gradient, and their positive cosine
shows partial alignment plus substantial cancellation in the combined VOS
gradient. This is a real mechanism difference, not evidence by itself of a
better detector.

## Detection results

| Arm | AUROC | FPR95 | OSCR | KS | Unknown below known-q95 | Known above unknown-q05 | Known acceptance | Unknown rejection | Accepted-known accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| detached | 0.88736 | 0.39533 | 0.63657 | 0.61783 | 56.40% | 39.53% | 95.42% | 41.70% | 71.11% |
| source-gradient | 0.89541 | 0.41783 | 0.64061 | 0.63600 | 47.80% | 41.78% | 95.77% | 46.80% | 71.02% |

Treatment-minus-control: AUROC `+0.00805`, FPR95 `+0.02250` (worse), OSCR
`+0.00403`, KS `+0.01817`, unknown-below-known-q95 overlap `-8.60 pp`,
known-above-unknown-q05 overlap `+2.25 pp` (worse), known acceptance
`+0.35 pp`, unknown rejection `+5.10 pp`, accepted-known accuracy `-0.09 pp`.
The source-gradient arm therefore improves several ranking/overlap views but
worsens FPR95, so the evidence is mixed. One seed cannot establish a robust
improvement, and the score distribution still overlaps substantially.

The same paired recipe was then replicated at seed 44 (again four epochs,
with only the source-gradient switch changing):

| Seed | Arm | AUROC | FPR95 | OSCR | KS | Unknown below known-q95 | Known above unknown-q05 | Known acceptance | Unknown rejection | Accepted-known accuracy |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 44 | detached | 0.89494 | 0.38033 | 0.65610 | 0.63742 | 52.43% | 38.03% | 94.37% | 49.48% | 72.73% |
| 44 | source-gradient | 0.89261 | 0.37400 | 0.65129 | 0.63783 | 53.58% | 37.40% | 95.07% | 46.20% | 72.25% |

Seed-44 treatment-minus-control: AUROC `-0.00233`, FPR95 `-0.00633`, OSCR
`-0.00481`, KS `+0.00042`, unknown-below-known-q95 overlap `+1.15 pp`,
known-above-unknown-q05 overlap `-0.63 pp`, known acceptance `+0.70 pp`,
unknown rejection `-3.28 pp`, accepted-known accuracy `-0.48 pp`. The seed-123
pattern does not replicate: source-gradient does not consistently improve
ranking, overlap, or the fixed-coverage rejection operating point. Across
both seeds the encoder gradient mechanism is consistent (source path around
0.286, absent when detached; cosine around 0.66), but the detector outcome is
seed-sensitive. Stop this source-gradient VOS direction; retain detach as the
default and do not run a weight sweep or combine this with rejection BCE.

## Decision and limitations

Do not change the default detach behavior. The planned second-seed replication
is complete and did not reproduce a robust benefit, so stop this source-gradient
VOS direction; do not run a weight sweep or combine it with rejection BCE.
The post-hoc classwise audit also shows seed/arm-dependent false-accept patterns
rather than a stable small set of classes with a consistent treatment response;
it must not be turned into test-informed class weights. See
`analysis/vos_source_gradient_classwise_audit_s44_s123_20261018.md`.

During the initial short mechanism probe, a driver bug inherited the source
config's absolute `student_ckpt` output path and overwrote the archived
seed-123 VOS treatment checkpoint with a 512-example, one-epoch model. The
four-epoch pair reported above was subsequently run into isolated output
directories with the path fixed. The overwritten archived checkpoint is not
recovered; previously recorded discovery metrics remain historical results,
but that exact checkpoint artifact must be regenerated before any later
checkpoint-level reanalysis. The incident is disclosed here rather than
presenting the initial run as a valid performance experiment.

## Artifacts

- Mechanism smoke test: `runs/vos_gradient_path_audit_s123_512_e1_20261018_*/`
- Full pair: `runs/vos_source_gradient_compare_s123_e4_20261018_safe_{detached,source_gradient}/`
- Discovery/calibration: matching `*_discover/` directories.
- Driver: `scripts/run_vos_source_gradient_compare.py`.
- Seed-44 replication: `runs/vos_source_gradient_compare_s44_e4_20261018_{detached,source_gradient}/` and matching `*_discover/` directories.
