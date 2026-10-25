# Frozen clustering-representation audit: seeds 2107–2109

## Question

The eight-epoch rejection branch improved known classification and open-set
ranking, but its automatic clustering NMI/ARI fell on all three seeds. Test
whether the current clustering input (`projection_pca`) is the limiting factor
by comparing it with the standard backbone representation (`feature_pca`) on
the exact same frozen four- and eight-epoch checkpoints.

## Paired conditions

- Seeds 2107, 2108 and 2109; semantic-hard CIFAR-100 60/40.
- Frozen checkpoints: the already completed four- and eight-epoch arms.
- Control: `cluster_feature=projection_pca` (existing score/report).
- Treatment: `cluster_feature=feature_pca`.
- Only the clustering feature changes. Rejector model, feature-rejector score,
  known-validation-only 95% threshold, checkpoint, candidate mask, automatic-K
  rule and all other discovery settings remain fixed.
- No training. Test labels are used only for retrospective ACC/NMI/ARI reporting;
  no label-driven choice, tuning or rerun is permitted. Mixed-pool labels are
  not read for this audit.

## Decision rule

Assess automatic K, NMI and ARI per seed and by epoch budget. Since the score and
candidate mask must remain unchanged, AUROC/AUPR/FPR95/OSCR/known utility and
unknown rejection should be identical; any difference is an implementation
error. A potential clustering improvement requires NMI and ARI to improve in at
least two of three splits without increasing automatic-K error systematically.
This is an evaluation-only mechanism audit, not a training-method claim.

## Exploratory outcome

Across the six frozen checkpoint comparisons, changing from `projection_pca` to
`feature_pca` left all rejection/coverage metrics exactly unchanged, as required,
while NMI and ARI improved in all six comparisons. For the 8-epoch checkpoints:

| Seed | projection NMI / ARI | backbone-feature NMI / ARI | Δ NMI / ARI | Auto-K: projection → feature |
|---:|---:|---:|---:|---:|
| 2107 | 0.486975 / 0.140670 | 0.562563 / 0.252266 | +0.075588 / +0.111595 | 38 → 41 |
| 2108 | 0.453620 / 0.148356 | 0.500277 / 0.199578 | +0.046657 / +0.051222 | 39 → 40 |
| 2109 | 0.486271 / 0.145252 | 0.568565 / 0.298631 | +0.082294 / +0.153379 | 37 → 37 |

The same direction appeared on the 4-epoch checkpoints (mean Δ NMI `+0.05895`,
mean Δ ARI `+0.10049`). Since this diagnostic evaluates clustering metrics
against test labels, it is exploratory evidence only and cannot justify
promoting `feature_pca` from these seeds. It motivates one locked independent
semantic-hard split with `feature_pca` fixed before examining that split's test
labels. The existing projection result remains the control; no score or
threshold changes are allowed.

Machine results: [`rejection_epoch_cluster_feature_audit_s2107_s2109.json`](rejection_epoch_cluster_feature_audit_s2107_s2109.json).
