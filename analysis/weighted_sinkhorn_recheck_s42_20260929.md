# Weighted Sinkhorn recheck for mixed novel discovery (2026-09-29)

> **Superseded method attribution:** A later call-chain audit found that before an independent Sinkhorn switch existed, `joint_novel_mass` sample weights were passed to both pseudo-target transport and loss reduction. The runs below therefore do not establish a valid weighted-vs-unweighted Sinkhorn contrast. Do not use this report's causal comparison or conclusion. See `analysis/method_fidelity_and_sinkhorn_audit_20260929.md` for the corrected implementation and paired rerun.

## Why this experiment

The previous student-weighted novel-mass + neighborhood experiment had a
positive AUROC / matched-coverage rejection signal but worse FPR95 and slightly
worse OSCR / known accuracy. Code inspection found that sample weights affected
the loss reduction, but did not affect the Sinkhorn assignments used as
pseudo-targets. As a result, low-weight samples could still influence the
batch class-balance constraints and thereby alter the targets of candidate
samples.

This change adds weighted Sinkhorn transport: each sample's transport marginal
is proportional to its detached soft weight. Zero-weight examples are omitted
from the transport and cannot change active examples' assignments. The default
unweighted assignment path is unchanged.

## Controlled comparison

- CIFAR-100 random 60/40 class split, seed 42, stratified known train/val.
- Same pretrained ResNet-34 teacher, ResNet-18 student, mixed pool with 0.2
  reserved known fraction, 1200/300/1000 sample limits and 3 epochs.
- Both use joint discovery, novel-mass weights and batch-local kNN support
  (`k=5`); the new run changes only how Sinkhorn assignment targets account for
  these weights.
- Detection uses explicit `normalized_entropy_mahalanobis`, MC=4, known-only
  95% coverage calibration and no clustering.

| Sinkhorn target recipe | AUROC | AUPR | FPR95 | OSCR | Known acc (all known) | Unknown reject (reported) | Unknown reject at exact 95% known coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original equal-sample Sinkhorn | 0.5887 | 0.4661 | 0.9058 | 0.1874 | 0.2512 | 10.89% | 8.86% |
| Weighted sample marginals | 0.5518 | 0.4184 | 0.9140 | 0.1704 | 0.2479 | 7.09% | 4.56% |

At matched test known coverage, weighted Sinkhorn reduces unknown rejection by
4.30 percentage points. It does not improve FPR95, AUROC, AUPR or OSCR. This
single seed does not support the method as a detection improvement.

## Validation

- Added tests that zero-weight samples cannot affect active Sinkhorn targets,
  output probabilities remain normalized, and invalid weight shapes fail fast.
- `python -m py_compile train.py novel_discovery/joint_discovery.py
  novel_discovery/pipeline.py` passed.
- `python -m pytest -q tests --disable-warnings --maxfail=1` -> `92 passed`.

## Decision

Keep weighted Sinkhorn as an opt-in-capable corrected primitive, but do not
claim it improves the project or enable it as a default method. Batch-local
neighborhood support remains a mixed tradeoff; EMA-generated weights were
negative in the same protocol. Do not continue tuning Sinkhorn iterations or
neighbor k on this one seed. The next useful step is error attribution: inspect
per-class false accepts, known/novel feature overlap, candidate precision and
novel-cluster coherence under held-out labels used only for retrospective
analysis. If the candidate set is contaminated, revise candidate generation;
if candidates are clean but feature overlap remains, focus on representation
learning. Confirm any promising change over multiple seeds and at larger data
budgets before promotion.
