# Next Steps for the Project

## What the current experiments say

- The pipeline is now runnable end-to-end on CIFAR-100 with teacher training, student training, and open-set discovery.
- The current bottleneck is not the threshold rule alone; the known/unknown score distributions are still very close.
- Among the tested score modes, all are near random on AUROC:
  - full: 0.5195
  - entropy_only: 0.5201
  - entropy_proto: 0.5213
  - proto_only: 0.5225
- proto_only gives the highest unknown reject rate, but it hurts clustering stability.
- entropy_proto is the best balanced option right now.

## What you should do next

1. Keep entropy_proto as the default open-set score for now.
2. Improve the feature separation during training instead of only changing the post-processing score.
3. Run a larger teacher/student training schedule before changing the loss design again.
4. Collect a small set of hard known/unknown samples and inspect whether the failure comes from confusing semantics or weak features.
5. If results are still weak, try a stronger backbone or a better uncertainty score before adding more complicated modules.

## Concrete short-term plan

- First experiment:
  - train teacher for more epochs
  - train student for more epochs
  - evaluate with score mode entropy_proto
- Second experiment:
  - compare entropy_proto against full
  - keep analysis/open_set_error_analysis.md as the report base
- Third experiment:
  - inspect the top rejected known classes and top false-accepted novel classes
  - decide whether the issue is class confusion or score calibration

## Recommendation

If you only have time for one next move, do this:

- train longer on CIFAR-100
- use entropy_proto
- then inspect the hardest classes in analysis/open_set_error_analysis.md
