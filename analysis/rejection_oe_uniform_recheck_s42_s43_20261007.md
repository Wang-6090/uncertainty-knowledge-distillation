# Rejection-branch uniform OE recheck (2026-10-07)

## Question

The previous rejection-branch OE margin pilot improved ranking on seed 42, but
did not improve validation-calibrated unknown rejection. This recheck tests a
more direct Outlier Exposure target: in addition to pushing auxiliary CIFAR-10
features away from known rejection prototypes, make their rejection-branch
logits close to the uniform distribution over known classes.

The added loss is

`L_rej-OE = - mean_i mean_c log softmax(z_rej(x_ood))_c`.

It is applied only to external outlier images and only to `rejection_logits`.
The default weight is zero. The paired treatment uses
`alpha_outlier_rejection_feature_margin=0.05` and
`alpha_outlier_rejection_uniform=0.05`; the comparison arm uses the same
margin but uniform weight zero.

## Fixed protocol

- CIFAR-100 semantic-isolated 60/40 split;
- seeds 42, 43, and 44;
- pretrained ResNet-34 teacher and ResNet-18 student;
- 1200/300/1000 train/validation/test limits;
- two student epochs, rejection embedding dimension 128;
- matched 5,400-sample mixed discovery pool with known prior 0.2;
- linear `nu_corrected` rejector on `support_augmented` rejection features;
- validation-only 95% known-coverage threshold calibration;
- the paired treatment changes only the new rejection-uniform OE weight.

The first seed-42 treatment command accidentally omitted the existing
`--alpha-rejection-known-ce 1.0` argument. Its checkpoint was discarded as an
invalid smoke run and is not included below.

## Results

| Seed | Arm | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | rejection margin OE | 0.7030 | 0.7720 | 0.1090 | 94.51% | 11.53% |
| 42 | margin OE + rejection uniform OE | 0.7144 | 0.7255 | 0.1203 | 95.34% | 16.29% |
| 43 | rejection margin OE | 0.6619 | 0.8188 | 0.1090 | 94.97% | 7.92% |
| 43 | margin OE + rejection uniform OE | 0.6742 | 0.8356 | 0.0969 | 95.97% | 12.62% |
| 44 | rejection margin OE | 0.6916 | 0.8073 | 0.1505 | 89.97% | 24.73% |
| 44 | margin OE + rejection uniform OE | 0.7375 | 0.6768 | 0.1743 | 93.95% | 16.67% |

Across all three seeds, AUROC improves and known acceptance improves. The
validation-calibrated unknown rejection increases on seeds 42 and 43 but falls
on seed 44; FPR95 and OSCR are also mixed. This is not enough to promote the
option to the default recipe when using the current validation operating point.

As a separate post-hoc diagnostic, the threshold was recomputed from the test
known scores only to enforce exactly 95% test-known coverage. This diagnostic
uses test labels and is not a valid model-selection result, but it isolates
ranking/coverage from validation score-scale shift:

| Seed | Margin-only exact-95 unknown rejection | Combined exact-95 unknown rejection |
| ---: | ---: | ---: |
| 42 | 9.02% | 16.54% |
| 43 | 8.42% | 15.35% |
| 44 | 12.10% | 15.05% |

The combined loss improves this matched-coverage diagnostic on all three
seeds. Therefore the method is a credible candidate for a larger, properly
held-out calibration experiment, but it has not solved the representation
overlap or the validation-calibrated operating point.

## Interpretation

The uniform target provides a useful complementary signal to the geometric
margin: it prevents auxiliary outliers from concentrating on a known rejection
class, while the margin discourages proximity to known rejection prototypes.
The consistent unknown-rejection gain suggests that the rejection branch is
learning a more useful novelty score. The mixed FPR95/OSCR result indicates
that the gain is not yet a uniformly better open-set operating curve.

### Uniform-only ablation

An additional seed-42 ablation disabled the feature margin and kept only the
new rejection-uniform OE weight at `0.05`. Compared with the original
no-OE rejection branch, it produced:

| Arm | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: |
| no OE | 0.6474 | 0.8220 | 0.0800 | 94.01% | 10.53% |
| uniform-only OE | 0.7180 | 0.7438 | 0.1285 | 96.67% | 11.03% |

Uniform-only improves ranking and known acceptance, but produces almost no
operating-point rejection gain. The larger rejection gain in the two-component
treatment therefore appears to require the interaction between uniform logits
and geometric feature margin. This interaction should be tested with a third
seed in the matched-coverage diagnostic, but still requires a larger budget
and disjoint calibration before being considered for the main recipe.

Keep both options opt-in. The next check is a larger training budget or a
disjoint open-validation calibration split for the combined loss. Do not tune
the threshold on test unknown labels; the exact-95 table above is diagnostic
only.

## Code and verification

- New flag: `--alpha-outlier-rejection-uniform`, default `0.0`.
- Training history records `outlier_rejection_uniform`.
- The loss requires `--rejection-feature-dim > 0` and is applied to
  `rejection_logits` only.
- `183 passed`, `compileall`, and `git diff --check` passed after the change.
