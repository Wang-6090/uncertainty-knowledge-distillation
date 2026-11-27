# Frozen teacher/CE rank complementarity audit

## Question and rationale

Prior audits establish that the formal teachers are less accurate than the CE
students and that standard KD transfers both correct and incorrect teacher
decisions. They do not establish whether the teacher's *full class ranking*
contains useful information on examples the CE student classifies incorrectly.
This read-only audit measures that remaining Team A question before proposing
another target-repair or uncertainty-KD training treatment. It does not search
for a score, threshold, checkpoint, or hyperparameter.

## Fixed scope and data

Use the frozen formal teacher and CE-only student checkpoints for semantic-hard
CIFAR-100 60/40 seeds `42`, `123`, and `3407`. For each seed, evaluate the same
3,000 known-validation images with deterministic CIFAR transforms. Construct
only the CIFAR-100 training dataset filtered to the 60 registered known
classes, then select its registered known-validation indices; do not construct
the test split or an unknown-pool dataset. Teacher and student sample indices
and mapped labels must match exactly.

Known labels are used after inference to define the CE-error and CE-correct
subgroups and to compute true-class rank metrics. They are not used for
training, calibration, checkpoint selection, or model/method selection.

## Measurements

For each model, rank the true class by descending logit (`1` is best; ties use
the count of strictly larger logits plus one). Report true-class rank, mean
reciprocal rank, top-1/top-5/top-10 rates, and true-class probability for all
known-validation examples and separately for CE-error and CE-correct subsets.
On CE-error examples also report the paired fraction where teacher rank is
better/equal/worse than CE rank and the mean rank advantage (`CE rank - teacher
rank`). This distinguishes complementary soft-target ordering from top-1
agreement alone.

## Predeclared interpretation gate

Treat the result as descriptive because the three semantic-hard class sets
overlap. A teacher-target complementarity signal is worth considering for a
separate future treatment only if, on the CE-error subgroup:

1. teacher mean reciprocal rank exceeds CE mean reciprocal rank in at least
   two of the three seeds; and
2. teacher top-5 true-class rate is no lower than CE in at least two seeds.

If either condition fails, do not propose another teacher-target correction
from this audit. If both pass, this only justifies a fresh, pre-registered
paired training pilot; it does not establish a performance benefit. Any such
pilot must compare CE, standard KD, and the single declared treatment, then
pass known-class representation and fixed downstream detection/discovery gates.

## Reproduction

Runner: `scripts/audit_teacher_ce_rank_complementarity.py`; focused metric tests:
`tests/test_teacher_ce_rank_complementarity.py`. Outputs are written to the
git-ignored `analysis/` directory. The runner is read-only and refuses to
overwrite an existing report.

## Results and decision (2026-11-27)

The known-validation rows and top-1 rates match the frozen formal audit for
all three seeds. Among examples misclassified by CE, teacher MRR is higher in
all three splits, but teacher top-5 is lower in all three. The paired rank
comparison also favors CE on average within every split:

| Seed | CE-error rows | Teacher / CE MRR | Teacher / CE top-5 | Teacher rank better / equal / worse |
|---:|---:|---:|---:|---:|
| 42 | 854 | 0.4354 / 0.3521 | 64.17% / 79.16% | 42.97% / 12.41% / 44.61% |
| 123 | 824 | 0.4209 / 0.3440 | 62.50% / 77.79% | 40.78% / 13.71% / 45.51% |
| 3407 | 852 | 0.4441 / 0.3460 | 62.21% / 77.82% | 43.43% / 11.74% / 44.84% |

The higher MRR is driven by the teacher sometimes placing the true class first
where CE is necessarily wrong (teacher top-1 on these rows is `24.9%–28.9%`).
It does not generalize to the broader top-5 ordering: teacher true-class rank
is worse than CE on `44.6%–45.5%` of paired rows, better on only
`40.8%–43.4%`, and mean rank advantage is negative in all seeds. The
predeclared gate fails because top-5 is no better in `0/3` seeds, despite MRR
being higher in `3/3`.

These are paired descriptive diagnostics on overlapping semantic-hard class
splits, not independent statistical replications. Labels were used only after
inference for grouping and rank calculation; no training or unknown/test
evaluation occurred. **Decision:** do not build another teacher-error or
uncertainty target-repair arm from these checkpoints. The teacher has some
correct top-1 information on CE-error samples, but the complete ranking is not
reliably better than CE. This reinforces the prior diagnosis that the current
teacher is an unreliable source for broad soft-target transfer; it does not
invalidate the formal paired KD results or test a stronger teacher recipe.

Machine reports: `analysis/team_a_teacher_ce_rank_complementarity_s{42,123,3407}_20261127.json`.
