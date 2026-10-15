# PU rejection-boundary loss: five-epoch continuation check

This continuation keeps seed 44, the same audited data protocol, teacher,
mixed pool, known prior, rejection representation, and discovery evaluation.
Both baseline and treatment use five epochs. The only treatment change is
`alpha_discovery_rejection_boundary=0.05` in PU mode.

| arm | AUROC | FPR95 | OSCR | known accuracy | unknown rejection at exact 95% known coverage |
|---|---:|---:|---:|---:|---:|
| baseline | 0.8995 | 0.3832 | 0.6558 | 0.7095 | 0.4950 |
| PU boundary 0.05 | 0.8970 | 0.3740 | 0.6628 | 0.7175 | 0.4850 |

Paired bootstrap over the same test examples gives an AUROC difference of
`-0.00251`, 95% CI `[-0.00653,+0.00142]`; the exact-95%-coverage rejection
difference is `-0.94` percentage points, CI `[-3.73,+1.80]` percentage
points. Thus the five-epoch continuation does not confirm a robust AUROC or
rejection gain, although FPR95, OSCR, and known accuracy improve.

Conclusion: the boundary loss is promising but training-budget sensitive. The
four-epoch results remain a useful pilot signal, not a final default-method
claim. More epochs should not be used to imply improvement without a fixed
validation-based checkpoint rule and additional seeds.

