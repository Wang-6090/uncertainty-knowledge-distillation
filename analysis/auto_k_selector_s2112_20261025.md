# Auto-K selector sweep: seed 2112

## Protocol validity

On a new semantic-hard CIFAR-100 60/40 split, one split-matched teacher and one
four-epoch student were trained. The same frozen student, `feature_pca`
representation, feature-rejector score, candidate pool, KMeans settings and
known-validation-only 95% coverage calibration were reused for all four arms.
Only the existing label-free `cluster_selection` option changed. Configuration
and label-flow audits passed; test/mixed labels were not used for training,
checkpoint selection, calibration, K selection or selector choice. They were
used only to calculate final retrospective metrics. The known-validation
calibration acceptance was `94.938%` in every arm.

## Results

| Selector | Auto K (true K=40) | Absolute K error | Unknown-only NMI | Unknown-only ARI | Unknown-only ACC | Global NMI / ARI |
|---|---:|---:|---:|---:|---:|---:|
| Silhouette | 26 | 14 | 0.589488 | 0.354387 | 45.837% | 0.544849 / 0.259988 |
| Composite | 20 | 20 | 0.569246 | 0.321009 | 44.153% | 0.525103 / 0.227699 |
| Stability | 20 | 20 | 0.569246 | 0.321009 | 44.153% | 0.525103 / 0.227699 |
| Diagonal-GMM BIC | 20 | 20 | 0.569246 | 0.321009 | 44.153% | 0.525103 / 0.227699 |

All rejection and known-utility metrics were exactly identical by construction:
AUROC `0.807539`, AUPR `0.709452`, FPR95 `57.983%`, OSCR `0.605835`, test known
acceptance `95.117%`, calibrated unknown rejection `26.725%`, accepted-known
accuracy `71.964%`, and correctly accepted fraction of all known `68.450%`.

Silhouette was the best of these four on actual unknown-only NMI/ARI/ACC, but
still underestimated K by 14. Composite, stability and diagonal-GMM BIC all
selected K=20 and had 20-class absolute error. Thus none of the tested existing
criteria resolves the K bottleneck. No winner is promoted and no default or
hyperparameter is changed. The earlier feature-space NMI/ARI gains remain a
clustering-representation result, not a reliable solution to class-count
estimation or to rejection.

## Interpretation and next step

The fixed rejection threshold produced a candidate pool of 1,362 samples
(78.49% novel purity; 1,069 true novel examples), while the split has 40 novel
classes. The large K error shows
that internal compactness/stability on this representation does not recover
semantic class count. Any next selector needs a predeclared label-free signal
that reflects density/semantic granularity and must be evaluated across fresh
splits; it must not be tuned to this split's test-label scores. A modest next
step is a frozen-checkpoint audit of the current K diagnostic curves (already
saved per K) to determine whether selection is boundary-limited or whether
scores genuinely peak at low K; do not adapt the rule from seed 2112.

Machine output: [`auto_k_selector_s2112.json`](auto_k_selector_s2112.json).
Runner: [`../scripts/run_auto_k_selector_sweep.py`](../scripts/run_auto_k_selector_sweep.py).
