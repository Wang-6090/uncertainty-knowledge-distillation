# Revised Experiment Plan

## Fixed protocol

- CIFAR-100, 60 known classes and 40 novel classes.
- One saved class split: `splits_cifar100_60_40.json`.
- Same seed, image size, epochs, optimizer, and validation threshold policy.
- Pretrained ResNet-34 teacher and pretrained ResNet-18 student.
- No pseudo-unknown loss in the main table.
- Use `entropy_mahalanobis` or its normalized version as a fixed score; do not
  use `score-mode auto` for the final result.

## Ablation order

| ID | Student objective | Purpose |
|---|---|---|
| A | CE | Student baseline |
| B | CE + standard KL | Effect of ordinary distillation |
| C | CE + uncertainty-weighted KL | Effect of the proposed weighting |
| D | C + feature KD | Effect of representation transfer |
| E | D + uncertainty head + SupCon + prototype | Full representation model |

Run the table with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_revised_ablation.ps1
```

The output should be summarized with mean and standard deviation over at least
three seeds before making a final claim.

## Required reports

For every row report known accuracy, AUROC, FPR95, known acceptance rate,
unknown rejection rate, cluster ACC, NMI, and ARI. Also report parameter
count, FLOPs or an equivalent model-size measure, and inference time for the
ResNet-18/MobileNetV3-Small compression comparison.

## Interpretation rule

The method is supported only if C improves over B on a fixed protocol, and
the improvement is stable across seeds. If C does not improve, report that
the uncertainty weighting is not validated and keep it as a negative result.
