# Mahalanobis versus max-softmax on the same checkpoint (seed 44)

This is a detector-only check after the matched checkpoint-selection audit.
The student checkpoint, split, test subset, threshold policy, and known
coverage target were fixed. Only the open-set score changed. Clustering was
skipped.

| score | AUROC | AUPR | FPR95 | OSCR | known accept | unknown reject | KS | unknown below known q95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| max_softmax | 0.5447 | 0.4426 | 0.8752 | 0.2344 | 0.9368 | 0.0723 | 0.1003 | 93.73% |
| normalized_entropy_mahalanobis_diag | 0.5424 | 0.4471 | 0.9214 | 0.2149 | 0.9556 | 0.0434 | 0.0753 | 94.46% |

The diagonal class-conditional Gaussian/Mahalanobis score is not better on
this representation. Its slightly higher AUPR does not compensate for worse
AUROC, FPR95, OSCR, KS, and unknown rejection. This is a negative result for
the current checkpoint, not a claim that every Mahalanobis implementation is
invalid. The earlier small positive result was checkpoint-dependent and must
not be generalized.

Conclusion: freeze the detector comparison for now and return to objectives
that directly improve shared feature geometry. Further threshold or score
fusion sweeps are unlikely to solve the observed overlap.

