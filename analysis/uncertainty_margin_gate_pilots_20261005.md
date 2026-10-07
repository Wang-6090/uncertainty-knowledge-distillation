# Uncertainty-gated feature-margin pilots (2026-10-05)

## Question

The mixed-pool uncertainty feature-margin showed a positive three-seed signal,
but a separate audit found its uncertainty weights nearly uniform (ESS
fraction about 0.96--0.97). These two seed-42 pilots tested whether (1) a
sharper transform, or (2) agreement across the two augmented views, improves
the loss's ability to focus on true novelty. They are follow-ups to that
candidate, not replacements for the nnPU + min-class kNN baseline.

## Fixed protocol and controls

- CIFAR-100 random 60/40 class split, seed 42, full train/validation/test and
  discovery pools, 64px images, batch size 64, five student epochs.
- Same pretrained ResNet-34 teacher checkpoint and ResNet-18 student setup;
  mixed discovery pool with measured known prior 0.212598.
- Both treatments retain immediate mixed-pool nnPU (weight 0.1), uncertainty
  feature-margin (weight 0.05, cosine margin 0.2), and identical optimizer,
  augmentation, and validation model-selection protocol.
- Detection for each checkpoint used the same full known training feature
  bank, normalized entropy + min-class kNN (`k=10`), MC=8, full 10,000-image
  open test set, and a known-validation-only 95% coverage threshold. No
  clustering was run. Test labels were used only for final metrics.

## Pilot A: sharpen uncertainty weights

The only intended algorithmic variable was the uncertainty exponent,
`u -> u^4` versus the existing `u` (exponent 1 remains the default).

| Seed 42 | AUROC | AUPR | FPR95 | OSCR | Known accuracy (all known) | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing single-view, power 1 | 0.7108 | 0.5799 | 0.7348 | 0.4132 | 0.4848 | 95.30% | 13.90% |
| Sharpened, power 4 | 0.6891 | 0.5622 | 0.7770 | 0.3983 | 0.4767 | 94.30% | 15.10% |
| Change | -0.0217 | -0.0177 | +0.0422 | -0.0149 | -0.0082 | -1.00pp | +1.20pp |

The higher unknown rejection was purchased with lower known acceptance; the
ranking, FPR95, OSCR, and known accuracy all worsened. This is a negative
result, not evidence of better separation. Do not sweep more exponents.

## Pilot B: cross-view minimum uncertainty

The only intended gate change was from single-view uncertainty to the
per-sample minimum of the two detached view uncertainties; exponent stayed at
1.

| Seed 42 | AUROC | AUPR | FPR95 | OSCR | Known accuracy (all known) | Known acceptance | Unknown rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing single-view gate | 0.7108 | 0.5799 | 0.7348 | 0.4132 | 0.4848 | 95.30% | 13.90% |
| Cross-view minimum | 0.6894 | 0.5621 | 0.7552 | 0.4012 | 0.4768 | 95.37% | 13.03% |
| Change | -0.0214 | -0.0177 | +0.0203 | -0.0120 | -0.0080 | +0.07pp | -0.87pp |

The hypothesis is not supported. The slight increase in known acceptance does
not compensate for worse unknown rejection, AUROC, AUPR, FPR95, OSCR, and
known accuracy. Stop this family of cross-view/weight-sharpened margin
variants; retain only the original gate as a research ablation.

## Implementation and verification

- Added `--discovery-feature-margin-weight-power`; default 1.0 is identity.
- Added `cross_view_min_uncertainty` as an explicit gate choice.
- Unit tests cover identity/sharpening, bounds, shape mismatch, and CLI parse.
- Full test suite: `164 passed`, with two pre-existing warning-only notices.
- Runs: `runs/uncertainty_margin_power4_s42/`,
  `runs/uncertainty_margin_power4_s42_detect/`,
  `runs/uncertainty_margin_crossview_s42/`, and
  `runs/uncertainty_margin_crossview_s42_detect/`.

Both pilots are single-seed follow-ups and do not overturn the original
three-seed candidate. Next, prioritize independent-split/additional-seed
confirmation of the original candidate and full feature-overlap diagnostics;
do not keep tuning on the same test set or append more post-hoc kNN variants.
