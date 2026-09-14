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
| F | E + discovery-pool NT-Xent | Effect of unlabeled novel representation learning |
| G | F + discovery unknown loss | Effect of explicitly pushing novel-pool samples away from known classes |

Run the table with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_revised_ablation.ps1
```

The output should be summarized with mean and standard deviation over at least
three seeds before making a final claim.

The reproducible multi-seed runner is:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_revised_multiseed.ps1
```

It repeats the A-E ablation for seeds 42, 123, and 3407 and evaluates both
`oracle K` and `auto K`. The summary is written to
`analysis/revised_multiseed/multiseed_summary.md` and
`analysis/revised_multiseed/multiseed_summary.json`.

The current quick discovery-pool comparison is:

```powershell
powershell -ExecutionPolicy Bypass -File .\\scripts\\run_discovery_pool_compare.ps1
```

It is intentionally small (`limit-train=1200`, `epochs=2`) and should be used
only as a sanity check. If F or G is promising, rerun the same comparison with
full training size and multiple seeds before writing a conclusion.

## Required reports

For every row report known accuracy, AUROC, AUPR, FPR95, OSCR, known acceptance
rate, unknown rejection rate, cluster ACC, NMI, and ARI. Also report parameter
count, FLOPs or an equivalent model-size measure, and inference time for the
ResNet-18/MobileNetV3-Small compression comparison.

## Interpretation rule

The method is supported only if C improves over B on a fixed protocol, and
the improvement is stable across seeds. If C does not improve, report that
the uncertainty weighting is not validated and keep it as a negative result.
