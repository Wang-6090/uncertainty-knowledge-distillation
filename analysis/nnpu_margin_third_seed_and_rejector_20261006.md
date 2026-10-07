# Three-seed nnPU + margin and support-rejector recheck (2026-10-06)

## Protocol

This report extends the matched full-data study to seed 3407. Seeds 42, 2026,
and 3407 use the same CIFAR-100 random 60/40 split, pretrained ResNet-34
teacher, pretrained ResNet-18 student, 64px images, batch size 64, 10 epochs,
matched 5400-image mixed discovery pool, resolved known prior 0.2, and the
normalized entropy plus min-class feature kNN detector (`k=10`, MC=8). The
threshold is fitted only on known validation data at 95% known coverage;
clustering is skipped.

Within each seed, the control is nnPU-only and the treatment adds only
uncertainty-weighted feature-margin (`alpha=0.05`, cosine margin `0.2`).

## Three-seed open-set results

| Seed | Arm | AUROC | FPR95 | OSCR | Known accuracy | Known acceptance | Unknown rejection |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026 | nnPU | 0.7103 | 0.7293 | 0.4647 | 55.28% | 95.25% | 13.13% |
| 2026 | nnPU + margin | 0.7205 | 0.7170 | 0.4764 | 56.25% | 95.70% | 13.15% |
| 42 | nnPU | 0.7152 | 0.7575 | 0.4637 | 55.40% | 95.20% | 15.75% |
| 42 | nnPU + margin | 0.7250 | 0.7233 | 0.4802 | 56.82% | 95.03% | 16.33% |
| 3407 | nnPU | 0.7122 | 0.7357 | 0.4331 | 50.48% | 95.35% | 14.45% |
| 3407 | nnPU + margin | 0.7105 | 0.7295 | 0.4593 | 54.42% | 95.57% | 12.68% |

The margin improves FPR95, OSCR, and known accuracy on all three seeds. AUROC
improves on two seeds and its mean change is still positive (`+0.0061`), but
the operating-point unknown rejection decreases on seed 3407; the three-seed
mean change is `-0.39pp`. Therefore the margin is a promising utility and
representation candidate, not a complete solution and not evidence that
unknown rejection is solved.

## Full-data feature geometry on seed 3407

The first invocation of the overlap script used its default limited sample
protocol and was discarded. The corrected run explicitly set all limits to
zero and used 6000 known plus 4000 unknown test samples:

| Distance reference | nnPU AUROC / overlap | nnPU + margin AUROC / overlap |
| --- | ---: | ---: |
| Classifier prototype | 0.6782 / 0.7224 | 0.6857 / 0.7072 |
| Empirical class centroid | 0.7051 / 0.6965 | 0.7024 / 0.7056 |
| Nearest known training sample | 0.7212 / 0.6712 | 0.7190 / 0.6734 |

Only the classifier-prototype distance improves. The centroid and nearest-
support diagnostics slightly worsen, so this seed does not support a claim of
uniform feature separation. The corrected artifact is
`analysis/matched_nnpu_margin_overlap_s3407_full.json`; the limited-sample
artifact is not used as evidence.

## Support-only rejector on seeds 42 and 3407

The rejector is trained after student training from frozen features. It uses
the held-out known open-validation side, the mixed discovery pool, and a
support-augmented nnPU feature model. The student checkpoint, split, test set,
and validation-only 95%-coverage policy remain fixed.

| Seed | Detector | AUROC | FPR95 | OSCR | Known accuracy | Known acceptance | Unknown rejection |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | Original student detector | 0.7250 | 0.7233 | 0.4802 | 56.82% | 95.03% | 16.33% |
| 42 | Support-only nnPU rejector | 0.7529 | 0.6853 | 0.4796 | 56.00% | 94.85% | 19.73% |
| 3407 | Original student detector | 0.7105 | 0.7295 | 0.4593 | 54.42% | 95.57% | 12.68% |
| 3407 | Support-only nnPU rejector | 0.7467 | 0.6683 | 0.4607 | 53.78% | 94.75% | 19.18% |

The support-only rejector improves AUROC and FPR95 on both seeds, gives a
small positive OSCR change on seed 3407, and raises unknown rejection by
3.40pp and 6.50pp respectively. It consistently costs less than one point of
known accuracy and known acceptance. Keep it as the preferred optional
detector, but report it separately from the student training contribution.

The earlier support-plus-MC-uncertainty rejector was worse than support-only
on seed 42 for AUROC, FPR95, OSCR, and known accuracy. It is not extended to
seed 3407.

## Configuration audit

One rejector invocation initially omitted `--discovery-pool-mode mixed` and
was rejected by the program before producing metrics. This is a configuration
failure, not a negative experiment result. The corrected run included the
missing flag. Future reports must include the resolved config and explicit
zero limits for full-data diagnostics.

The overlap script now defaults all four limit parameters to zero. Positive
limits remain available only when a smoke test is explicitly requested.
