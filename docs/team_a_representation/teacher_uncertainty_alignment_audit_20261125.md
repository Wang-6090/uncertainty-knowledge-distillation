# Teacher uncertainty alignment audit

## Question

Uncertainty-KD uses detached weights `exp(-teacher_uncertainty)` for the
logits KD term. This audit asks whether the teacher's uncertainty is aligned
with teacher error on known data. If it is not, the weighting signal is not a
direct proxy for “samples whose teacher target needs special handling”.

## Fixed procedure

Use the frozen split-matched teacher from the seed `2173` class-holdout pilot.
Run the teacher in evaluation mode on the complete known training and known
validation subsets using deterministic CIFAR transforms. For each sample,
record teacher uncertainty, maximum-softmax confidence, and whether the
teacher prediction matches the known label.

Report uncertainty-vs-error AUROC, confidence-vs-error AUROC, error rates in
uncertainty quintiles, and the effective distribution of `exp(-u)` weights.
Known labels are read only after inference for this post-hoc diagnostic; they
are never used to change the teacher, student, threshold, or training config.

## Decision rule

This report does not promote a new weighting rule. It only determines whether
teacher uncertainty is a plausible quality signal. A later weighting method is
justified only if the alignment is positive on both train and validation and
the weighting spread is meaningful. Otherwise, uncertainty-KD should remain an
opt-in ablation and further weight scans should stop.

The machine-readable output is written to
`analysis/team_a_teacher_uncertainty_alignment_20261125_s2173.json`.
